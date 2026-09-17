"""
NebulonDB Unified Search Orchestrator
=====================================

Top-level orchestration layer integrating vector, graph, and ranking subsystems, containing:
    NebulonOrbit          – unified API for insert/delete/search with transaction safety
    WAL Recovery          – write-ahead log replay with LSN tracking for crash consistency
    Hybrid Search         – Nova + Mesh BFS expansion with configurable boost/depth
    Ranked Search         – multi-signal fusion (BM25 + metadata + freshness) with optional RRF
    Cross-Encoder Rerank  – lazy-loaded re-ranking on top of ranked candidates
    Compaction            – automatic rebuild when deleted ratio exceeds threshold
    Consistency Check     – DB-vs-index reconciliation with automatic rebuild on mismatch
"""

import os
import json
import re
import threading
import time

import contextlib
import numpy as np

from pathlib import Path
from collections import deque
from typing import Any
from collections.abc import Sequence

from db.engine import NebulonCosmos
from db.engine.utils import DatabaseConfig
from utils.logger import NebulonDBLogger

# ── submodules ──────────────────────────────────────────────
from .nova_store import NovaStore
from .nova_engine import NovaEngine
from .mesh_engine import MeshEngine
from .document_store import DocumentStore

from .ranking import BM25Scorer, RRFMerger, QueryIntent, RankEngine, CrossEncoderReranker, RankConfig

from db.engine.utils import (
    FIELD_METADATA,
    FIELD_VECTOR,
    FIELD_TEXT,
    FIELD_LABEL,
)
from ndb_host.utils.time_utils import utc_now_iso


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()


def _cosine_similarity(a, b) -> float:
    va = np.asarray(a, dtype=np.float32).reshape(-1)
    vb = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    if denom == 0:
        return 0.0
    return float(float(np.dot(va, vb)) / float(denom))


# =============================================================================
# NovaStore – persistent storage (thread-safe)
# =============================================================================
class NebulonOrbit:
    def __init__(
            self,
            db_dir: str | Path,
            segment_name: str = "default",
            ef_search: int = 50,
            reset: bool = False,
            rank_config: RankConfig | None = None,
            flush_interval: float | None = None,
            wal_fsync_interval: int | None = None,
        ):
        config = DatabaseConfig(db_dir, is_vector=True, is_graph=True)
        self._store = NebulonCosmos(db_dir, reset=reset)

        # NOVA Config
        # Record-store segments are derived from `segment_name` so that
        # different orbit segments of the same corpus keep their vector /
        # document / mesh rows isolated in separate COSMOS segments. Without
        # this, every segment shares the fixed `nebulon_nova` etc. record
        # stores and rebuilds leak records across users.
        nova_segment = f"{config.NOVA_SEGMENT_NAME}_{segment_name}"
        docs_segment = f"{config.DOCUMENTS_SEGMENT_NAME}_{segment_name}"
        dim = config.VECTOR_DIM
        space = config.VECTOR_SPACE.lower()
        M = config.VECTOR_M
        ef_construction = config.VECTOR_EF_CONSTRUCTION
        ef_search = ef_search or config.VECTOR_EF_SEARCH

        nova_dir = config.NEBULON_NOVA_DIR / f"segment_{segment_name}"
        nova_config_path = nova_dir / config.NOVA_CONFIG_JSON.name
        nova_manifest_dir = nova_dir / config.NOVA_MANIFEST_FILE_JSON.name
        self.nova_wal_path = nova_dir / config.NOVA_WAL.name
        self._auto_id_watermark_path: Path = nova_dir / "id_watermark.json"
        self.compaction_deleted_ratio = config.COMPACTION_DELETED_RATIO
        self.nova_store = NovaStore(self._store, nova_segment)
        self.doc_store = DocumentStore(self._store, docs_segment)

        self.nova_engine = NovaEngine(
            dim=dim, space=space, M=M,
            ef_construction=ef_construction,
            ef_search=ef_search,
            nova_dir=nova_dir,
            nova_manifest_dir=nova_manifest_dir,
            nova_config_path=nova_config_path,
        )

        # MESH Config (record segments derived from segment_name, see NOVA)
        mesh_segment = f"{config.MESH_SEGMENT_NAME}_{segment_name}"
        node_segment = f"{config.MESH_NODE_SEGMENT_NAME}_{segment_name}"
        edge_segment = f"{config.MESH_EDGE_SEGMENT_NAME}_{segment_name}"

        # Per-segment visualization file so different segments of the same
        # corpus never overwrite each other's HTML graph output.
        safe_segment = re.sub(r"[^A-Za-z0-9._-]", "_", segment_name) or "default"
        self.mesh_graph_viz_html = (
            config.NEBULON_MESH_DIR / f"mesh_graph_visualization_{safe_segment}.html"
        )
        self.mesh_engine = MeshEngine(
            store=self._store,
            mesh_segment=mesh_segment,
            node_segment=node_segment,
            edge_segment=edge_segment,
            mesh_graph_viz_html=self.mesh_graph_viz_html
        )
        if not self.mesh_engine.load():
            logger.info("No saved Mesh found; starting empty Mesh engine.")

        self._dirty = False
        self._save_lock = threading.Lock()
        self._op_lock = threading.RLock()
        self._dirty_lock = threading.Lock()
        self._wal_lock = threading.Lock()
        self._wal_bytes_since_fsync = 0
        self._wal_fsync_interval = config.WAL_FSYNC_INTERVAL
        if wal_fsync_interval is not None and wal_fsync_interval > 0:
            self._wal_fsync_interval = int(wal_fsync_interval)
        self._last_flush_t: float = 0.0
        self._flush_interval: float = max(0.0, config.FLUSH_INTERVAL)
        if flush_interval is not None:
            self._flush_interval = max(0.0, float(flush_interval))

        self.rank_topk = config.RANK_TOPK
        self.top_k = config.TOP_MATCHES

        # --- LSN support ---
        self._lsn_counter = 0
        self._lsn_lock = threading.Lock()

        # --- Record ID auto-generation support ---
        self._auto_id: int | None = None

        self.weight = config.WEIGHT
        # Correct initialisation order
        nova_engine_load = self.nova_engine.load()
        if nova_engine_load:
            self._replay_wal()
        else:
            self._load_or_build()

        self._check_consistency()

        # Restore the persisted monotonic record-ID watermark so auto-ID
        # generation is O(1) once loaded. On reset, drop any stale watermark
        # so the rebuilt segment starts IDs fresh.
        if reset:
            with contextlib.suppress(OSError, FileNotFoundError):
                self._auto_id_watermark_path.unlink(missing_ok=True)
        self._load_id_watermark()

        # Ensure the LSN counter is always ahead of the last LSN already
        # applied to disk. Without this, a fresh per-request NebulonOrbit
        # starts its counter at 0 while the persisted last_applied_lsn may be
        # higher, so new WAL entries (e.g. add_relation) would be skipped on
        # replay and graph mutations would be silently lost.
        with self._lsn_lock:
            self._lsn_counter = max(self._lsn_counter, self.nova_engine.last_applied_lsn)

        # --- Ranking support (lazy) ---
        self.rank_config: RankConfig = rank_config or RankConfig()
        self._rank_lock = threading.RLock()
        self._corpus: list[dict[str, Any]] | None = None
        self._corpus_dirty: bool = True
        self._bm25: BM25Scorer | None = None
        self._rank_engine: RankEngine | None = None
        self._intent_cls: type = QueryIntent
        self._rrf_merger: RRFMerger = RRFMerger(k=60)
        self._reranker: CrossEncoderReranker | None = None
        self._reranker_available: bool | None = None

    # ------------------------------------------------------------------ #
    # Initialisation & Rebuild                                           #
    # ------------------------------------------------------------------ #

    def _load_or_build(self) -> None:
        logger.info("Rebuilding Nova from stored vectors...")
        records = self.nova_store.read_all()
        if not records:
            return
        items = []
        seen_ids = set()
        for record in records:
            rid = record.get("id")
            vec = record.get(FIELD_VECTOR)
            if rid is None or vec is None:
                continue
            if rid in seen_ids:
                logger.warning("Skipping duplicate record ID %s during rebuild", rid)
                continue
            try:
                self.nova_engine.validate_vector(vec, self.nova_engine.dim)
                items.append((rid, vec))
                seen_ids.add(rid)
                self.mesh_engine.add_node(rid)
            except ValueError as e:
                logger.warning("Skipping invalid vector for %s: %s", rid, e)
        if items:
            self.nova_engine.add_items(items)
            self.nova_engine.save()
            self.mesh_engine.save()
            self._set_dirty(False)
            logger.info("Rebuilt Nova with %d vectors", len(items))

    def _check_consistency(self) -> None:
        db_records = self.nova_store.read_all()
        db_ids = {rec.get("id") for rec in db_records if rec.get("id") is not None}
        graph_ids = set(self.nova_engine.id_map.keys())
        if db_ids != graph_ids:
            logger.warning(
                "Consistency check failed. db_ids=%s graph_ids=%s db_only=%s graph_only=%s",
                sorted(db_ids), sorted(graph_ids),
                sorted(db_ids - graph_ids), sorted(graph_ids - db_ids),
            )
            self._rebuild_nova_engine_from_db(force_save=False)

    def _rebuild_nova_engine_from_db(self, force_save: bool = True) -> None:
        logger.info("Rebuilding Nova from DB...")
        records = self.nova_store.read_all()
        new_graph = NovaEngine(
            dim=self.nova_engine.dim, space=self.nova_engine.space,
            M=self.nova_engine.M, ef_construction=self.nova_engine.ef_construction,
            ef_search=self.nova_engine.ef_search,
            nova_dir=self.nova_engine.nova_dir,
            nova_manifest_dir=self.nova_engine.manifest.path,
            nova_config_path=self.nova_engine.nova_config_path,
        )
        items = []
        seen = set()
        for rec in records:
            rid = rec.get("id")
            vec = rec.get(FIELD_VECTOR)
            if rid is None or vec is None or rid in seen:
                continue
            try:
                new_graph.validate_vector(vec, new_graph.dim)
                items.append((rid, vec))
                seen.add(rid)
            except ValueError:
                logger.warning("Skipping invalid vector for %s during rebuild", rid)
        if items:
            new_graph.add_items(items)
        self.nova_engine = new_graph
        self._set_dirty(True)
        if force_save:
            self.nova_engine.save()
            self.mesh_engine.save()
            self._set_dirty(False)

    # ------------------------------------------------------------------ #
    # Dirty flag helpers (thread‑safe)                                   #
    # ------------------------------------------------------------------ #

    def _set_dirty(self, value: bool) -> None:
        with self._dirty_lock:
            self._dirty = value

    def _is_dirty(self) -> bool:
        with self._dirty_lock:
            return self._dirty

    # ------------------------------------------------------------------ #
    # Write‑Ahead Log (WAL)                                              #
    # ------------------------------------------------------------------ #

    def _next_lsn(self) -> int:
        """Atomically increment and return the next LSN."""
        with self._lsn_lock:
            self._lsn_counter += 1
            return self._lsn_counter

    def _generate_record_id(self) -> int:
        """Generate the next monotonic record ID in O(1).

        The watermark (``_auto_id``) is seeded lazily from the in-memory
        id_map and persisted across saves, so no full-store scan is ever
        needed to allocate a fresh ID.
        """
        if self._auto_id is None:
            self._seed_auto_id()
        self._auto_id += 1
        return self._auto_id

    def _seed_auto_id(self) -> None:
        """Seed the ID watermark from the live in-memory graph (once)."""
        ids = self.nova_engine.id_map.keys()
        self._auto_id = max((i for i in ids if isinstance(i, int)), default=0)

    def _bump_auto_id(self, record_id: int) -> None:
        """Keep the ID watermark monotonic over explicit record IDs."""
        if self._auto_id is None or record_id > self._auto_id:
            self._auto_id = record_id

    def _load_id_watermark(self) -> None:
        """Restore the persisted ID watermark (max of watermark, live ids)."""
        wm: int | None = None
        with contextlib.suppress(OSError, ValueError):
            path = self._auto_id_watermark_path
            if path.exists():
                with open(path) as f:
                    wm = int(json.load(f).get("watermark", 0))
        live_max = max(
            (i for i in self.nova_engine.id_map if isinstance(i, int)), default=0
        )
        if wm is not None and wm > 0:
            self._auto_id = max(wm, live_max)
        elif live_max > 0:
            self._auto_id = live_max
        else:
            self._auto_id = None

    def _persist_id_watermark(self) -> None:
        """Atomically write the monotonic ID watermark after a save."""
        if self._auto_id is None:
            return
        path = self._auto_id_watermark_path
        tmp = path.with_suffix(".tmp")
        with contextlib.suppress(OSError):
            with open(tmp, "w") as f:
                json.dump({"watermark": self._auto_id}, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, path)
            with contextlib.suppress(FileNotFoundError):
                tmp.unlink(missing_ok=True)

    def _wal_log(self, action: str, record_id: int,
                 vector: list[float] | None = None,
                 metadata: dict | None = None,
                 target: int | None = None,
                 relation: str | None = None,
                 weight: float | None = None,
                 created_at: str | None = None) -> None:
        """Append an LSN‑tagged operation to the WAL."""
        lsn = self._next_lsn()
        entry = {"lsn": lsn, "action": action, "id": record_id}
        if vector is not None:
            entry["vector"] = list(vector)
        if metadata is not None:
            entry["metadata"] = metadata
        if target is not None:
            entry["target"] = target
        if relation is not None:
            entry["relation"] = relation
        if weight is not None:
            entry["weight"] = weight
        if created_at is not None:
            entry["created_at"] = created_at
        payload = json.dumps(entry) + "\n"
        with self._wal_lock:
            with open(self.nova_wal_path, "a") as f:
                f.write(payload)
                f.flush()
                self._wal_bytes_since_fsync += len(payload)
                # Group commit: fsync only once per accumulated threshold,
                # mirroring the Cosmos WAL (config WAL_FSYNC_INTERVAL).
                if self._wal_bytes_since_fsync >= self._wal_fsync_interval:
                    os.fsync(f.fileno())
                    self._wal_bytes_since_fsync = 0

    def _wal_fsync(self) -> None:
        """Persist any pending group-committed WAL bytes to disk."""
        with self._wal_lock:
            if self._wal_bytes_since_fsync <= 0:
                return
            with contextlib.suppress(OSError):
                with open(self.nova_wal_path, "a") as f:
                    os.fsync(f.fileno())
            self._wal_bytes_since_fsync = 0

    def _wal_clear(self) -> None:
        """Clear the WAL after a successful save."""
        with contextlib.suppress(FileNotFoundError):
            os.remove(self.nova_wal_path)

    def _replay_wal(self) -> None:
        """
        Replay operations from the WAL that occurred after the last save.
        Skips entries with LSN <= last_applied_lsn stored in the loaded nova.
        """
        if not self.nova_wal_path.exists():
            return

        # Retrieve the last applied LSN from the currently loaded nova
        last_applied = getattr(self.nova_engine, 'last_applied_lsn', 0)

        with self._wal_lock, open(self.nova_wal_path) as f:
            lines = f.readlines()

        max_replayed = last_applied
        for line in lines:
            try:
                entry = json.loads(line)
                lsn = entry.get("lsn", 0)

                # Skip entries already persisted in the saved nova
                if lsn <= last_applied:
                    continue

                action = entry["action"]
                rid = entry["id"]
                if action == "insert":
                    vec = entry["vector"]
                    meta = entry.get("metadata") or {}
                    text = meta.get(FIELD_TEXT, "")
                    label = meta.get(FIELD_LABEL)
                    created_at = utc_now_iso()
                    self.nova_store.insert(rid, vec, created_at=created_at)
                    self.doc_store.insert(rid, text, metadata=meta,
                                          label=label, created_at=created_at)
                    if rid in self.nova_engine.id_map:
                        self.nova_engine.update_item(rid, vec)
                    else:
                        self.nova_engine.add_item(rid, vec)
                    self.mesh_engine.add_node(rid, label, created_at)
                    self._bump_auto_id(rid)
                elif action == "delete":
                    self.nova_store.delete(rid)
                    self.doc_store.delete(rid)
                    self.nova_engine.delete(rid)
                    self.mesh_engine.remove_node(rid)
                elif action == "add_relation":
                    target = entry.get("target")
                    relation = entry.get("relation")
                    if target is None or relation is None:
                        logger.error("Invalid add_relation WAL entry: %s", entry)
                        continue
                    self.mesh_engine.add_edge(
                        rid, target, relation,
                        weight=entry.get("weight", 1.0),
                        created_at=entry.get("created_at"),
                    )
                elif action == "remove_relation":
                    target = entry.get("target")
                    relation = entry.get("relation")
                    if target is None:
                        logger.error("Invalid remove_relation WAL entry: %s", entry)
                        continue
                    self.mesh_engine.remove_edge(rid, target, relation)
                elif action == "add_node":
                    label = (entry.get("metadata") or {}).get(FIELD_LABEL)
                    self.mesh_engine.add_node(rid, label)
                elif action == "remove_node":
                    self.mesh_engine.remove_node(rid)
                else:
                    logger.warning("Unknown WAL action %r, skipping", action)
                    continue

                max_replayed = max(max_replayed, lsn)
            except Exception as e:
                logger.error("Failed to replay WAL entry: %s", e)

        # Update the LSN counter to at least the highest replayed LSN
        with self._lsn_lock:
            if max_replayed > self._lsn_counter:
                self._lsn_counter = max_replayed

        # Clear the WAL file – all entries up to the current LSN are now applied
        self._wal_clear()
        self._set_dirty(True)

    # ------------------------------------------------------------------ #
    # Transaction‑safe Insert (WAL log carries the LSN) #
    # ------------------------------------------------------------------ #

    def insert(self, record_id: int | None = None, vector: Sequence[float] | None = None,
               metadata: dict | None = None, text: str | None = None) -> tuple[int, str | None]:
        """
        Insert a vector into the store.

        Parameters
        ----------
        record_id : int, optional
            Explicit record ID. When omitted, an ID is auto-generated.
        vector : Sequence[float], optional
            The vector to insert. Required.
        metadata : Dict, optional
            Optional metadata associated with the record.
        text : str, optional
            Optional text content; stored under metadata["text"] when provided.

        Returns
        -------
        Tuple[int, Optional[str]]
            (record_id, None) on success, or (None, error_message) on failure.
        """
        try:
            if vector is None:
                return None, "vector is required"
            metadata = dict(metadata or {})
            if text is not None:
                metadata.setdefault(FIELD_TEXT, text)
            if record_id is None:
                record_id = self._generate_record_id()
            else:
                self._bump_auto_id(record_id)
            created_at = utc_now_iso()
            label = metadata.get(FIELD_LABEL)
            with self._op_lock:
                self.nova_engine.validate_vector(vector, self.nova_engine.dim)
                self._wal_log("insert", record_id, vector, metadata)

                old_result = self.nova_store.get(record_id)
                self.nova_store.insert(record_id, vector, created_at=created_at)
                self.doc_store.insert(record_id, text or metadata.get(FIELD_TEXT, ""),
                                      metadata=metadata, label=label,
                                      created_at=created_at)
                try:
                    if old_result is None:
                        self.nova_engine.add_item(record_id, vector)
                    else:
                        self.nova_engine.update_item(record_id, vector)

                    self.mesh_engine.add_node(record_id, label, created_at)
                except Exception:
                    if old_result is None:
                        self.nova_store.delete(record_id)
                        self.doc_store.delete(record_id)
                        self.mesh_engine.remove_node(record_id)
                    else:
                        self.nova_store.insert(record_id,
                                                 old_result.get(FIELD_VECTOR),
                                                 created_at=created_at)
                    self._rebuild_nova_engine_from_db(force_save=False)
                    raise
                self._set_dirty(True)
                self._invalidate_corpus()
                return record_id, None
        except Exception as e:
            return None, str(e)

    # ------------------------------------------------------------------ #
    # Transaction‑safe Batch Insert #
    # ------------------------------------------------------------------ #

    def add_items(self, items: list[tuple[int, Sequence[float]]],
                  metadatas: list[dict] | None = None) -> None:
        if not items:
            return
        with self._op_lock:
            ids = [rid for rid, _ in items]
            if len(ids) != len(set(ids)):
                logger.error("Duplicate record IDs in batch")
            for rid in ids:
                self._bump_auto_id(rid)
            for _, vec in items:
                self.nova_engine.validate_vector(vec, self.nova_engine.dim)
            # Log each item with LSN
            for i, (rid, vec) in enumerate(items):
                md = metadatas[i] if metadatas and i < len(metadatas) else None
                self._wal_log("insert", rid, vec, md)

            old_docs = {}
            for rid, _ in items:
                doc = self.nova_store.get(rid)
                if doc is not None:
                    old_docs[rid] = doc
            written = []
            created_at = utc_now_iso()
            try:
                for i, (rid, vec) in enumerate(items):
                    md = metadatas[i] if metadatas and i < len(metadatas) else None
                    self.nova_store.insert(rid, vec, created_at=created_at)
                    if md:
                        md = dict(md)
                        text = md.get(FIELD_TEXT, "")
                        label = md.get(FIELD_LABEL)
                        self.doc_store.insert(rid, text, metadata=md,
                                              label=label, created_at=created_at)
                    written.append(rid)
            except Exception:
                for rid in written:
                    if rid in old_docs:
                        old = old_docs[rid]
                        self.nova_store.insert(rid, old.get(FIELD_VECTOR), created_at=created_at)
                    else:
                        self.nova_store.delete(rid)
                        self.doc_store.delete(rid)
                raise
            try:
                self.nova_engine.batch_upsert(items)
                for i, (rid, _) in enumerate(items):
                    md = metadatas[i] if metadatas and i < len(metadatas) else None
                    label = (md or {}).get(FIELD_LABEL)
                    self.mesh_engine.add_node(rid, label, created_at)
            except Exception:
                for rid, _ in items:
                    if rid in old_docs:
                        old = old_docs[rid]
                        self.nova_store.insert(rid, old.get(FIELD_VECTOR), created_at=created_at)
                    else:
                        self.nova_store.delete(rid)
                        self.doc_store.delete(rid)
                self._rebuild_nova_engine_from_db(force_save=False)
                raise
            self._set_dirty(True)
            self._invalidate_corpus()
            logger.debug("Batch added %d items", len(items))

    def add_items_auto(self, vectors: Sequence[Sequence[float]],
                       metadatas: list[dict] | None = None) -> list[int]:
        """Insert vectors with auto-generated contiguous record IDs.

        ID allocation uses the persisted watermark, so it stays O(1)
        (no store scan) while the batch is written as a single WAL group
        with one ``batch_upsert`` into the HNSW engine.
        """
        n = len(vectors)
        if n == 0:
            return []
        with self._op_lock:
            if self._auto_id is None:
                self._seed_auto_id()
            start = self._auto_id + 1
            ids = list(range(start, start + n))
            self._auto_id = start + n - 1
            self.add_items(list(zip(ids, vectors)), metadatas)
            return ids

    # ------------------------------------------------------------------ #
    # Transaction‑safe Delete #
    # ------------------------------------------------------------------ #

    def delete(self, record_id: int) -> None:
        with self._op_lock:
            old_doc = self.nova_store.get(record_id)
            if old_doc is None:
                return
            self._wal_log("delete", record_id)
            self.nova_store.delete(record_id)
            self.doc_store.delete(record_id)
            try:
                self.nova_engine.delete(record_id)
                self.mesh_engine.remove_node(record_id)
            except Exception:
                self.nova_store.insert(record_id, old_doc.get(FIELD_VECTOR))
                self.doc_store.insert(record_id, old_doc.get(FIELD_TEXT, ""),
                                      metadata=old_doc.get(FIELD_METADATA) or {})
                self._rebuild_nova_engine_from_db(force_save=False)
                raise
            self._set_dirty(True)
            self._invalidate_corpus()

    def delete_records(self, record_ids: list[int]) -> int:
        """Bulk delete a list of record IDs in one WAL group + memtable pass.

        Existence is checked once up front (mirroring the per-row ``delete``
        guard). Store tombstones are written via ``delete_many`` (single WAL
        write + lock acquisition); HNSW and mesh removals still run per row
        in memory. Returns the number of records actually removed.
        """
        if not record_ids:
            return 0
        with self._op_lock:
            old_docs: dict[int, dict[str, Any]] = {}
            for rid in record_ids:
                doc = self.nova_store.get(rid)
                if doc is not None:
                    old_docs[rid] = doc
            if not old_docs:
                return 0
            surviving = list(old_docs)
            for rid in surviving:
                self._wal_log("delete", rid)
            try:
                self.nova_store.delete_many(surviving)
                self.doc_store.delete_many(surviving)
                for rid in surviving:
                    self.nova_engine.delete(rid)
                    self.mesh_engine.remove_node(rid)
            except Exception:
                for rid, old in old_docs.items():
                    self.nova_store.insert(rid, old.get(FIELD_VECTOR))
                    self.doc_store.insert(
                        rid,
                        old.get(FIELD_TEXT, ""),
                        metadata=old.get(FIELD_METADATA) or {},
                    )
                self._rebuild_nova_engine_from_db(force_save=False)
                raise
            self._set_dirty(True)
            self._invalidate_corpus()
            return len(surviving)

    def update(self, record_id: int,
               vector: Sequence[float] | None = None,
               metadata: dict | None = None) -> tuple[int, str | None]:
        """
        Update an existing record's vector and/or metadata (upsert semantics).

        Parameters
        ----------
        record_id : int
            The record to update. Must already exist.
        vector : Sequence[float], optional
            New vector. When omitted, the existing vector is kept.
        metadata : Dict, optional
            Merged into the existing metadata (key-wise override).

        Returns
        -------
        Tuple[int, Optional[str]]
            (record_id, None) on success, or (None, error_message) on failure.
        """
        try:
            old_doc = self.nova_store.get(record_id)
            if old_doc is None:
                return None, f"record {record_id} does not exist"
            with self._op_lock:
                new_vec = old_doc.get(FIELD_VECTOR)
                if vector is not None:
                    self.nova_engine.validate_vector(vector, self.nova_engine.dim)
                    new_vec = vector
                old_doc_doc = self.doc_store.get(record_id) or {}
                merged_meta = dict(old_doc_doc.get(FIELD_METADATA) or {})
                if metadata is not None:
                    merged_meta.update(metadata)
                text = merged_meta.get(FIELD_TEXT, old_doc_doc.get(FIELD_TEXT, ""))
                label = merged_meta.get(FIELD_LABEL)
                created_at = utc_now_iso()

                self._wal_log("insert", record_id, new_vec, merged_meta)
                self.nova_store.insert(record_id, new_vec, created_at=created_at)
                self.doc_store.insert(record_id, text, metadata=merged_meta,
                                      label=label, created_at=created_at)
                try:
                    if vector is not None:
                        self.nova_engine.update_item(record_id, vector)
                    self.mesh_engine.add_node(record_id, label, created_at)
                except Exception:
                    self.nova_store.insert(record_id, old_doc.get(FIELD_VECTOR),
                                           created_at=created_at)
                    self._rebuild_nova_engine_from_db(force_save=False)
                    raise
                self._set_dirty(True)
                self._invalidate_corpus()
                return record_id, None
        except Exception as e:
            return None, str(e)

    # ------------------------------------------------------------------ #
    # Graph relation management                                          #
    # ------------------------------------------------------------------ #

    def add_relation(self, source, target: int, relation: str,
                     weight: float | None = None) -> int:
        """Add a directed relationship between two nodes.

        ``source``/``target`` may be a node id (int) or a node label (string);
        unknown labels auto-create a node. When ``weight`` is omitted it is
        computed automatically from the two endpoint vectors (or 1.0).
        """
        with self._op_lock:
            sid = self.mesh_engine.resolve_node(source)
            tid = self.mesh_engine.resolve_node(target)
            if weight is None:
                weight = self._auto_weight(sid, tid)
            created_at = utc_now_iso()
            self._wal_log("add_relation", sid, target=tid, relation=relation,
                          weight=weight, created_at=created_at)
            edge_id = self.mesh_engine.add_edge(
                sid, tid, relation, weight=weight, created_at=created_at)
            self._set_dirty(True)
            return edge_id

    def remove_relation(self, source: int, target: int, relation: str | None = None) -> None:
        with self._op_lock:
            self._wal_log("remove_relation", source, target=target, relation=relation)
            self.mesh_engine.remove_edge(source, target, relation)
            self._set_dirty(True)

    def resolve_node(self, ref) -> int:
        """Resolve an int id (or string label) to a node id."""
        return self.mesh_engine.resolve_node(ref)

    def _auto_weight(self, source: int, target: int) -> float:
        """Auto edge weight: cosine similarity of endpoint vectors, else 1.0."""
        va = (self.nova_store.get(source) or {}).get(FIELD_VECTOR)
        vb = (self.nova_store.get(target) or {}).get(FIELD_VECTOR)
        if va is None or vb is None:
            return 1.0
        return _cosine_similarity(va, vb)

    def load_graph(self, nodes: list[dict] | None = None,
                   edges: list[dict] | None = None) -> dict[str, Any]:
        """Bulk graph load: create nodes and edges (Option A flow).

        nodes: [{"id":..., "label":..., "metadata":{...}}, ...]
        edges: [{"from":<ref>, "to":<ref>, "relation":..., "weight":...}, ...]
        """
        nodes = nodes or []
        edges = edges or []
        created_at = utc_now_iso()
        node_count = 0
        edge_count = 0
        with self._op_lock:
            for node in nodes:
                nid = node.get("id")
                label = node.get("label")
                if nid is not None:
                    nid = int(nid)
                    if not self.mesh_engine.has_node(nid):
                        self.mesh_engine.add_node(nid, label, created_at)
                else:
                    nid = self.mesh_engine.resolve_node(label or "")
                self._wal_log("add_node", nid, metadata={FIELD_LABEL: label})
                node_count += 1
            for edge in edges:
                sid = self.mesh_engine.resolve_node(edge.get("from"))
                tid = self.mesh_engine.resolve_node(edge.get("to"))
                relation = edge.get("relation", "related")
                weight = edge.get("weight")
                if weight is None:
                    weight = self._auto_weight(sid, tid)
                self._wal_log("add_relation", sid, target=tid, relation=relation,
                              weight=weight, created_at=created_at)
                self.mesh_engine.add_edge(
                    sid, tid, relation, weight=weight, created_at=created_at)
                edge_count += 1
            self._set_dirty(True)
            self._invalidate_corpus()
        return {"nodes_added": node_count, "edges_added": edge_count}

    # ------------------------------------------------------------------ #
    # Compaction #
    # ------------------------------------------------------------------ #

    def compact(self) -> None:
        with self._op_lock:
            logger.info("Starting Nova compaction...")
            self._rebuild_nova_engine_from_db(force_save=False)
            logger.info("Compaction complete.")

    def _maybe_compact(self) -> None:
        if self.nova_engine.deleted_ratio() > self.compaction_deleted_ratio:
            self.compact()

    # ------------------------------------------------------------------ #
    # Search, Get, Count #
    # ------------------------------------------------------------------ #

    def _record_payload(self, rid) -> dict[str, Any]:
        """Merge document + nova + node state for a record id."""
        nova = self.nova_store.get(rid)
        doc = self.doc_store.get(rid) or {}
        meta = dict(doc.get(FIELD_METADATA) or {})
        node = self.mesh_engine.get_node(rid) or {}
        return {
            "id": rid,
            "vector": (nova or {}).get(FIELD_VECTOR),
            "text": doc.get(FIELD_TEXT, ""),
            "metadata": meta,
            "label": meta.get(FIELD_LABEL) or node.get(FIELD_LABEL),
        }

    def _nova_search(self, vector: Sequence[float], top_k: int) -> list[dict[str, Any]]:
        """Pure vector similarity search."""
        with self._op_lock:
            results = self.nova_engine.search(vector, top_k)
            for result in results:
                result.update(self._record_payload(result["id"]))
            return results

    def _mesh_search(self, max_depth: int, top_k: int, start_node: int | None = None) -> list[dict[str, Any]]:
        """
        Graph‑only search: BFS from a start node, returning discovered nodes.
        Scores are based on distance (closer = higher).
        If no start_node is provided, returns empty list.
        """
        if start_node is None:
            return []

        with self._op_lock:
            visited = {}
            queue = deque([(start_node, 0)])
            while queue:
                node, depth = queue.popleft()
                if depth > max_depth:
                    continue
                if node in visited:
                    continue
                visited[node] = depth
                for neighbor, _ in self.mesh_engine.get_neighbors(node, "both"):
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

            # Convert to results
            results = []
            for node_id, depth in visited.items():
                # Score: closer nodes get higher scores (1 / (1+depth))
                score = 1.0 / (1.0 + depth)
                payload = self._record_payload(node_id)
                if payload.get("vector") is not None:
                    results.append({"id": node_id, "score": score, **payload})
                else:
                    # Pure graph node (no vector record)
                    node = self.mesh_engine.get_node(node_id) or {}
                    results.append({
                        "id": node_id,
                        "score": score,
                        "vector": None,
                        "text": "",
                        "metadata": {},
                        "label": node.get(FIELD_LABEL),
                    })

            # Sort by score descending, trim to top_k
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def _hybrid_search(self, vector: Sequence[float] | None, top_k: int, expand_depth: int,
                       graph_boost: float, graph_start_node: int | None = None) -> list[dict[str, Any]]:
        """Hybrid Nova + Mesh expansion (existing hybrid_search logic)."""
        if vector is None:
            logger.error("vector is required for hybrid mode; returning empty results")
            return []

        with self._op_lock:
            candidates = self.nova_engine.search(vector, top_k=top_k * 2)
            if not candidates:
                return []

            score_map = {c["id"]: c["score"] for c in candidates}
            seeds = [c["id"] for c in candidates]
            if graph_start_node is not None:
                seeds.append(graph_start_node)
                if graph_start_node not in score_map:
                    score_map[graph_start_node] = graph_boost
            visited = set(score_map.keys())
            for seed in seeds:
                queue = deque([(seed, 0)])
                while queue:
                    node, depth = queue.popleft()
                    if depth >= expand_depth:
                        continue
                    for neighbor, _ in self.mesh_engine.get_neighbors(node, "both"):
                        if neighbor not in visited:
                            visited.add(neighbor)
                            score_map[neighbor] = graph_boost
                            queue.append((neighbor, depth + 1))

            results = []
            for rid, score in score_map.items():
                payload = self._record_payload(rid)
                if payload.get("vector") is not None:
                    results.append({"id": rid, "score": score, **payload})
                else:
                    node = self.mesh_engine.get_node(rid) or {}
                    results.append({
                        "id": rid,
                        "score": score,
                        "vector": None,
                        "text": "",
                        "metadata": {},
                        "label": node.get(FIELD_LABEL),
                    })
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:top_k]

    def search(self,
        mode: str = "auto",
        expand_depth: int = 1,
        graph_boost: float = 0.1,
        rank: bool = True,
        top_k: int = None,
        vector: Sequence[float] | None = None,
        graph_start_node: int | None = None,
        query: str | None = None,
        **rank_kwargs: Any) -> list[dict[str, Any]]:
        """
        Unified search interface.

        Parameters
        ----------
        vector : Sequence[float], optional
            Query vector. Required for 'vector' and 'hybrid' modes.
        top_k : int
            Number of results to return.
        mode : str
            One of 'auto', 'nova', 'mesh', or 'hybrid'.
            - 'auto': picks 'hybrid' if a graph is present, else 'nova'.
            - 'nova': pure vector similarity search.
            - 'mesh': pure graph traversal from graph_start_node.
            - 'hybrid': Nova search + Mesh neighbour expansion.
        graph_start_node : int, optional
            Seed node for graph traversal. Required in 'mesh' mode; in 'hybrid'
            mode it is added as an additional expansion seed even when it is not
            a top vector hit.
        expand_depth : int
            Max BFS depth for Mesh expansion.
        graph_boost : float
            Score assigned to nodes discovered via Mesh expansion in 'hybrid' mode.
        query : str, optional
            Raw text query. When provided with ``rank=True``, enables BM25
            scoring and query-intent weight selection.
        rank : bool
            When True, applies multi-signal ranking (vector + BM25 + metadata +
            importance + freshness) with optional cross-encoder re-ranking
            instead of returning the raw retrieval order.

        Ranking behaviour (RRF fusion, re-ranking, weights, half-life,
        metadata rules, cross-encoder model) is controlled via the
        :class:`RankConfig` supplied at construction time.

        Returns
        -------
        List[Dict] with keys: id, score, vector (if present), metadata (if present).
        """
        top_k = top_k or self.top_k
        if rank:
            return self.ranked_search(
                query=query,
                query_vector=vector,
                top_k=top_k,
                mode=mode,
                graph_start_node=graph_start_node,
                expand_depth=expand_depth,
                graph_boost=graph_boost,
                **rank_kwargs,
            )
        mode = mode.lower()
        if mode == "auto":
            mode = "hybrid" if self.mesh_engine.has_edges() else "nova"
        if mode == "mesh":
            return self._mesh_search(
                start_node=graph_start_node,
                max_depth=expand_depth,
                top_k=top_k,
            )
        if mode == "hybrid":
            return self._hybrid_search(
                vector=vector,
                top_k=top_k,
                expand_depth=expand_depth,
                graph_boost=graph_boost,
                graph_start_node=graph_start_node,
            )
        # Default: vector-only search
        if vector is None:
            logger.error("vector is required for mode='nova'; returning empty results")
            return []
        return self._nova_search(vector, top_k)

    # ------------------------------------------------------------------ #
    # Ranking support (BM25 + multi-signal + optional cross-encoder)     #
    # ------------------------------------------------------------------ #
    def _invalidate_corpus(self) -> None:
        """Invalidate the cached BM25 corpus + rank engine after any write."""
        with self._rank_lock:
            self._corpus = None
            self._corpus_dirty = True
            self._bm25 = None
            self._rank_engine = None

    def _build_corpus(self) -> list[dict[str, Any]]:
        """Lazily build the document corpus used for BM25 scoring."""
        with self._rank_lock:
            if self._corpus is not None and not self._corpus_dirty:
                return self._corpus
            corpus: list[dict[str, Any]] = []
            for rec in self.doc_store.read_all():
                rid = rec.get("id")
                if rid is None:
                    continue
                doc = rec.get(FIELD_METADATA) or {}
                text = doc.get(FIELD_TEXT, rec.get(FIELD_TEXT, ""))
                corpus.append({"id": rid, "text": text})
            self._corpus = corpus
            self._corpus_dirty = False
            return corpus

    def _ensure_rank_engine(
        self,
        weights: dict[str, float] | None = None,
        half_life: float = 30.0,
    ) -> RankEngine:
        """Ensure the multi-signal RankEngine (with BM25) is built."""
        with self._rank_lock:
            cfg = self.rank_config
            if weights is None:
                weights = self.weight
            if self._rank_engine is None:
                corpus = self._build_corpus()
                self._bm25 = BM25Scorer(corpus)
                self._rank_engine = RankEngine(
                    documents=corpus,
                    weights=weights,
                    half_life=half_life,
                    mode="linear",
                )
            elif weights is not None:
                self._rank_engine.weights = weights
            if cfg.metadata_rules is not None:
                self._rank_engine.metadata_rules = cfg.metadata_rules
            return self._rank_engine

    def _ensure_reranker(self) -> CrossEncoderReranker | None:
        """Lazily load the cross-encoder; returns None if unavailable."""
        if self._reranker_available is False:
            return None
        with self._rank_lock:
            if self._reranker is None:
                self._reranker = CrossEncoderReranker()
            self._reranker_available = self._reranker.available()
            if not self._reranker_available:
                logger.warning("Cross-encoder re-ranker unavailable; skipping re-rank step.")
                return None
            return self._reranker

    def ranked_search(
        self,
        query: str | None = None,
        query_vector: Sequence[float] | None = None,
        top_k: int = 10,
        mode: str = "nova",
        **search_kwargs: Any,
    ) -> list[dict[str, Any]]:
        """
        Hybrid ranked search: vector retrieval + (optional) BM25, fused with
        multi-signal ranking and optional cross-encoder re-ranking.

        Parameters
        ----------
        query : str, optional
            Raw text query; used for BM25 scoring and intent detection.
        query_vector : Sequence[float], optional
            Query vector; required for 'nova'/'hybrid' modes.
        top_k : int
            Number of final results to return.
        mode : str
            Passed through to :meth:`search` ('nova', 'mesh', 'hybrid', 'auto').

        Ranking behaviour (RRF fusion, re-ranking, weights, half-life,
        metadata rules, cross-encoder model) is controlled via the
        :class:`RankConfig` supplied at construction time.

        Returns
        -------
        List[Dict] of final ranked results.
        """
        cfg = self.rank_config
        # candidate_topk = self.candidate_topk

        # The raw retrieval pass must never re-enter ranked_search.
        search_kwargs["rank"] = False

        if mode == "mesh" and query_vector is None:
            candidates = self._mesh_search(
                start_node=search_kwargs.get("graph_start_node"),
                max_depth=search_kwargs.get("expand_depth", 2),
                top_k=top_k,
            )
        elif query_vector is None:
            candidates = self.search(
                vector=None, top_k=top_k, mode=mode, **search_kwargs
            )
        else:
            candidates = self.search(
                vector=query_vector, top_k=top_k, mode=mode, **search_kwargs
            )
        if not candidates:
            return []

        # Optional RRF fusion of vector + BM25 lists
        if cfg.use_rrf and query and self._corpus and not self._corpus_dirty:
            bm25_results = self._bm25.search(query, top_k=top_k)
            if bm25_results:
                candidates = self._rrf_merger.merge(candidates, bm25_results, max_unique=self.rank_topk)

        # Multi-signal ranking
        intent = self._intent_cls(weights=self.weight)
        dyn_weights = intent.get_weights(query or "")
        if cfg.weights is not None:
            dyn_weights = cfg.weights
        engine = self._ensure_rank_engine(weights=dyn_weights, half_life=cfg.half_life,)
        ranked = engine.rank(query or "", candidates, return_top_n=self.rank_topk)

        # Optional cross-encoder re-ranking
        if cfg.rerank:
            reranker = self._ensure_reranker()
            if reranker is not None:
                ranked = reranker.rerank(query or "", ranked, text_key="text", top_k=top_k)
                final = ranked
            else:
                final = ranked[:top_k]
        else:
            final = ranked[:top_k]

        for doc in final:
            doc.pop("_rank_debug", None)
        return final

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """
        Re-rank an existing list of candidate results (no new retrieval).

        Signals: BM25 (text relevance) + metadata + importance + freshness.
        Optionally applies cross-encoder re-ranking on the top candidates.

        Parameters
        ----------
        query : str
            Raw text query used for BM25 scoring and intent detection.
        candidates : List[Dict]
            Candidate results, typically from a prior search() call.
            Each dict should contain 'id' and optionally 'metadata'.
        top_k : int, optional
            Number of re-ranked results to return (default: all candidates).

        Re-ranking behaviour (weights, half-life, cross-encoder model,
        metadata rules) is controlled via the :class:`RankConfig` supplied at
        construction time.

        Returns
        -------
        List[Dict] of re-ranked results, each with 'rank_score'.
        """
        if not candidates:
            return []
        cfg = self.rank_config
        engine = self._ensure_rank_engine(weights=cfg.weights, half_life=cfg.half_life)
        ranked = engine.rank(query, candidates)

        if cfg.rerank:
            reranker = self._ensure_reranker()
            if reranker is not None:
                ranked = reranker.rerank(query, ranked, text_key="text",
                                         top_k=top_k or len(ranked))

        if top_k is not None:
            ranked = ranked[:top_k]
        for doc in ranked:
            doc.pop("_rank_debug", None)
        return ranked

    # ------------------------------------------------------------------ #
    # Graph traversal methods exposed                                     #
    # ------------------------------------------------------------------ #
    def add_node(self, node_id: int, label: str | None = None) -> None:
        """Create a graph node explicitly (no vector required)."""
        with self._op_lock:
            self._wal_log("add_node", node_id, metadata={FIELD_LABEL: label})
            self.mesh_engine.add_node(node_id, label)
            self._set_dirty(True)

    def remove_node(self, node_id: int) -> None:
        """Remove a graph node and all edges connected to it."""
        with self._op_lock:
            self._wal_log("remove_node", node_id)
            self.mesh_engine.remove_node(node_id)
            self._set_dirty(True)

    def get_node(self, node_id: int) -> dict | None:
        """Return the graph node's metadata, or None if it does not exist."""
        return self.mesh_engine.get_node(node_id)

    def has_node(self, node_id: int) -> bool:
        """Return True if the node exists in the graph."""
        return self.mesh_engine.has_node(node_id)

    def count_nodes(self) -> int:
        """Number of nodes currently in the graph."""
        return self.mesh_engine.count_nodes()

    def count_edges(self) -> int:
        """Number of directed edges currently in the graph."""
        return self.mesh_engine.count_edges()

    def has_edges(self) -> bool:
        """Return True if the graph contains at least one edge."""
        return self.mesh_engine.has_edges()

    def get_edges(self) -> list[dict]:
        """Return all edges as dict rows (edge_id, source, target, ...)."""
        return self.mesh_engine.get_all_edges()

    def get_all_nodes(self) -> list[dict[str, Any]]:
        """Return every graph node as {"id": ..., "label": ...}."""
        return [
            {"id": node_id, "label": (self.mesh_engine.get_node(node_id) or {}).get(FIELD_LABEL)}
            for node_id in self.mesh_engine.get_all_nodes()
        ]

    def edges_by_relation(self, relation: str) -> list[dict]:
        """Return all edges that carry the given relation label."""
        return self.mesh_engine.edges_by_relation(relation)

    def get_neighbors(self, node_id: int, direction: str = "both") -> list[tuple[int, str]]:
        return self.mesh_engine.get_neighbors(node_id, direction)

    def get_in_neighbors(self, node_id: int) -> list[tuple[int, str]]:
        """Neighbours pointing at node_id (incoming edges)."""
        return self.mesh_engine.get_neighbors(node_id, "in")

    def get_out_neighbors(self, node_id: int) -> list[tuple[int, str]]:
        """Neighbours node_id points at (outgoing edges)."""
        return self.mesh_engine.get_neighbors(node_id, "out")

    def bfs(self, start: int, max_depth: int = 3) -> list[int]:
        return self.mesh_engine.bfs(start, max_depth)

    def dfs(self, start: int, max_depth: int = 3) -> list[int]:
        return self.mesh_engine.dfs(start, max_depth)

    def shortest_path(self, source: int, target: int) -> list[int] | None:
        return self.mesh_engine.shortest_path(source, target)

    def connected_components(self) -> list[set[int]]:
        return self.mesh_engine.connected_components()

    def get_visualization_html(self) ->  tuple[str | None, Path | None]:
        """Return an HTML string for visualizing the graph."""
        if not self.mesh_graph_viz_html.exists():
            return "Mesh graph visualization HTML template not configured.", None
        return None, self.mesh_graph_viz_html

    # ------------------------------------------------------------------ #
    # Get, Count, Flush                                                   #
    # ------------------------------------------------------------------ #
    def get(self, record_id: int) -> dict[str, Any] | None:
        with self._op_lock:
            return self._record_payload(record_id)

    def get_metadata(self, record_id: int) -> dict | None:
        """Return only the metadata of a record, or None."""
        doc = self.get(record_id)
        return doc.get(FIELD_METADATA) if doc else None

    def get_vector(self, record_id: int) -> list[float] | None:
        """Return only the vector of a record, or None."""
        return (self.nova_store.get(record_id) or {}).get(FIELD_VECTOR)

    def exists(self, record_id: int) -> bool:
        """Return True if a record with this ID exists."""
        return self.nova_store.get(record_id) is not None

    def list_ids(self) -> list[int]:
        """Return all record IDs currently stored."""
        with self._op_lock:
            return [rec.get("id") for rec in self.nova_store.read_all() if rec.get("id") is not None]

    def get_all(self) -> list[dict[str, Any]]:
        """Return all records (id, vector, text, metadata, label)."""
        with self._op_lock:
            return [self._record_payload(rec.get("id")) for rec in self.nova_store.read_all()]

    def count(self) -> int:
        with self._op_lock:
            return self.nova_engine.get_current_count()

    def stats(self) -> dict[str, Any]:
        """Return a summary snapshot of Nova + Mesh state."""
        with self._op_lock:
            try:
                from core.model_hub import resolve_device, ModelType
                embedding_device = resolve_device(ModelType.EMBEDDING)
            except Exception:
                embedding_device = "unknown"
            return {
                "vector_count": self.nova_engine.get_current_count(),
                "dimension": self.nova_engine.dim,
                "space": self.nova_engine.space,
                "node_count": self.mesh_engine.count_nodes(),
                "edge_count": self.mesh_engine.count_edges(),
                "deleted_ratio": round(self.nova_engine.deleted_ratio(), 6),
                "lsn": self._lsn_counter,
                "embedding_device": embedding_device,
                "flush_interval_s": self._flush_interval,
                "wal_fsync_interval_b": self._wal_fsync_interval,
            }

    def flush(self) -> None:
        self.save(force=False)

    def set_durability(self,
                       flush_interval: float | None = None,
                       wal_fsync_interval: int | None = None) -> None:
        """Tune the durability/latency trade-off for this segment at runtime.

        ``flush_interval`` : seconds between full index saves (0 = save on
                             every flush). Higher = fewer index rewrites, wider
                             crash-recovery window.
        ``wal_fsync_interval`` : accumulated WAL bytes before an ``fsync``
                                 (0 = fsync every write). Higher = faster
                                 writes, weaker durability on power loss.
        Either value may be None to leave it unchanged.
        """
        if flush_interval is not None:
            self._flush_interval = max(0.0, float(flush_interval))
        if wal_fsync_interval is not None and wal_fsync_interval >= 0:
            self._wal_fsync_interval = int(wal_fsync_interval)

    def _maybe_debounce(self) -> bool:
        """Return True when a full save should be skipped (debounced).

        Durability is preserved between saves by the group-committed WAL and
        the live in-memory engine, so a burst of single writes only forces
        one full index save once per ``FLUSH_INTERVAL`` seconds.
        """
        if self._flush_interval <= 0 or self._last_flush_t <= 0:
            return False
        return (time.monotonic() - self._last_flush_t) < self._flush_interval

    # ------------------------------------------------------------------ #
    # Save with LSN persistence                                          #
    # ------------------------------------------------------------------ #
    def save(self, force: bool = False) -> None:
        """
        Save the current state, including the last applied LSN.
        After a successful save, the WAL is cleared.

        Repeated saves are debounced to ``FLUSH_INTERVAL`` seconds unless
        ``force`` is True (used on close).
        """
        with self._op_lock, self._save_lock:
            if not self._is_dirty():
                return
            if not force and self._maybe_debounce():
                return
            self._maybe_compact()

            # Tell the graph which LSN to write into its mapping
            current_lsn = self._lsn_counter   # thread‑safe read (atomic int)
            self.nova_engine.set_last_applied_lsn(current_lsn)
            self.mesh_engine.save()
            self.nova_engine.save()
            self._persist_id_watermark()
            self._set_dirty(False)
            self._wal_fsync()
            self._wal_clear()
            # Rewrite the Cosmos WAL from the live memtable so a fresh
            # engine init never replays stale entries. No segment file
            # is written below the flush threshold.
            self._store.checkpoint_wal()
            self._last_flush_t = time.monotonic()

    def close(self) -> None:
        try:
            self.save(force=True)
        finally:
            self._store.close()
