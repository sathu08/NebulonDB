"""
NebulonDB Nova Vector Engine
============================

Core HNSW vector index engine with atomic operations, containing:
    NovaEngine            – thread-safe HNSW index with add/update/delete/search
    Manifest              – atomic generation tracking with fallback recovery
    Config Persistence    – dim/space/M parameter durability via atomic JSON writes
    Checksum Verification – SHA-256 integrity validation on save/load
    Generation Management – multi-generation save/load with automatic old-gen cleanup
"""


import os
import hnswlib

import hashlib
import threading

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Set, Tuple, Sequence

from .manifest import Manifest
from utils.logger import NebulonDBLogger

from ndb_host.utils.models import load_data, save_data


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()


# =============================================================================
# NovaEngine – with atomic operations, compaction, checksums, manifest recovery
# =============================================================================

class NovaEngine:
    def __init__(
        self,
        dim: int,
        space: str,
        M: int,
        ef_construction: int,
        ef_search: int,
        nova_dir: Path,
        nova_manifest_dir: Path,
        nova_config_path: Path,
    ):
        self.dim = dim
        self.space = space.lower()
        self.M = M
        self.ef_construction = ef_construction
        self.ef_search = ef_search

        self.nova_dir = nova_dir 
        self.nova_dir.mkdir(parents=True, exist_ok=True)

        self.manifest = Manifest(nova_manifest_dir)

        self.nova_config_path = nova_config_path
        self._load_config()

        # In‑memory state
        self.index: Optional[hnswlib.Index] = None
        self.id_map: Dict[int, int] = {}
        self.reverse_map: Dict[int, int] = {}
        self.deleted_ids: Set[int] = set()
        self.next_id: int = 0
        self._active_count: int = 0
        self.last_applied_lsn: int = 0
        self._current_gen: Optional[int] = None

        # Lock for all mutable operations (save, add, update, delete)
        self._lock = threading.RLock()

    # ---------- Public validation ----------
    @staticmethod
    def validate_vector(vector: Sequence[float], dim: int) -> None:
        if len(vector) != dim:
            logger.error(f"Vector dimension {len(vector)} != {dim}")
        arr = np.asarray(vector, dtype=np.float32)
        if not np.isfinite(arr).all():
            logger.error("Vector contains NaN or Inf")

    # ---------- Config persistence (dim, space, M) ----------
    def _load_config(self) -> bool:
        """Load index parameters (dim, space, M) from the persisted config JSON (if present)."""
        cfg = load_data(self.nova_config_path, default={}) or {}
        if not cfg:
            return False
        try:
            self.dim = int(cfg["dim"])
            self.space = str(cfg["space"]).lower()
            self.M = int(cfg["M"])
            self.ef_construction = int(cfg["ef_construction"])
            return True
        except (KeyError, ValueError, TypeError):
            logger.warning("Could not read index config from %s", self.nova_config_path)
            return False

    def _write_config(self) -> None:
        """Persist index parameters (dim, space, M) to the config JSON."""
        cfg = {
            "dim": self.dim,
            "space": self.space,
            "M": self.M,
            "ef_construction": self.ef_construction,
        }
        tmp = self.nova_config_path.with_suffix(".tmp")
        try:
            save_data(cfg, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp, self.nova_config_path)
        except Exception:
            logger.exception("Failed to persist index config to %s", self.nova_config_path)
            if tmp.exists():
                tmp.unlink(missing_ok=True)
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)

    def _ensure_config(self) -> None:
        """Create the config JSON on first insert if it does not exist yet."""
        if not self.nova_config_path.exists():
            self._write_config()

    # ---------- Internal helpers ----------
    def _init_index(self, max_elements: int = 1000) -> None:
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=self.ef_construction,
            M=self.M,
        )
        self.index.set_ef(self.ef_search)

    def _ensure_capacity(self, needed: int = 1) -> None:
        if self.index is None:
            self._init_index(max(needed, 1000))
        elif self.next_id + needed > self.index.max_elements:
            new_max = max(self.index.max_elements * 2, self.next_id + needed)
            self.index.resize_index(new_max)

    def _mark_deleted_by_internal(self, internal_id: int) -> None:
        """Mark an internal ID as deleted (used for update and delete)."""
        if internal_id in self.deleted_ids:
            return
        try:
            self.index.mark_deleted(internal_id)
        except RuntimeError:
            logger.warning("Could not mark internal id %d as deleted", internal_id)
            return
        self.deleted_ids.add(internal_id)
        self.reverse_map.pop(internal_id, None)
        self._active_count = max(0, self._active_count - 1)

    def _add_new_internal(self, record_id: int, vector: np.ndarray) -> int:
        """Allocate new internal ID and add vector to index (no duplicate check)."""
        self._ensure_capacity()
        int_id = self.next_id
        self.next_id += 1
        try:
            self.index.add_items(vector.reshape(1, -1), np.array([int_id], dtype=np.int32))
        except Exception:
            # rollback next_id
            self.next_id -= 1
            raise
        self.id_map[record_id] = int_id
        self.reverse_map[int_id] = record_id
        self._active_count += 1
        return int_id

    # ---------- Public API (all locked) ----------
    def add_item(self, record_id: int, vector: Sequence[float]) -> int:
        """Add new record. Raises ValueError if record_id already exists."""
        with self._lock:
            self._load_config()
            self._ensure_config()
            if record_id in self.id_map:
                logger.error(f"Record ID {record_id} already exists.")
            self.validate_vector(vector, self.dim)
            vec = np.asarray(vector, dtype=np.float32)
            return self._add_new_internal(record_id, vec)

    def add_items(self, items: List[Tuple[int, Sequence[float]]]) -> None:
        """Batch add new records. All must be new."""
        with self._lock:
            if not items:
                return
            self._load_config()
            self._ensure_config()
            for rid, vec in items:
                self.validate_vector(vec, self.dim)
                if rid in self.id_map:
                    logger.error(f"Record ID {rid} already exists.")
            self._ensure_capacity(len(items))
            n = self.next_id
            ids = list(range(n, n + len(items)))
            vectors = np.asarray([v for _, v in items], dtype=np.float32)
            self.index.add_items(vectors, np.array(ids, dtype=np.int32))
            for (rid, _), int_id in zip(items, ids):
                self.id_map[rid] = int_id
                self.reverse_map[int_id] = rid
            self.next_id += len(items)
            self._active_count += len(items)

    def update_item(self, record_id: int, vector: Sequence[float]) -> int:
        """
        Atomically update an existing record.
        If marking the old ID fails, the new ID is rolled back.
        """
        with self._lock:
            self._load_config()
            old_internal = self.id_map.get(record_id)
            if old_internal is None:
                logger.error(f"Record ID {record_id} not found for update.")
            self.validate_vector(vector, self.dim)

            vec = np.asarray(vector, dtype=np.float32)
            # ----- Phase 1: add new vector -----
            new_internal = self._add_new_internal(record_id, vec)   # already modifies mappings

            # ----- Phase 2: mark old as deleted, with rollback on failure -----
            try:
                self._mark_deleted_by_internal(old_internal)
            except Exception:
                # Rollback: mark the newly added vector as deleted and restore old mapping
                logger.exception("Marking old ID %d failed, rolling back new ID %d", old_internal, new_internal)
                # mark new as deleted (best effort)
                try:
                    self.index.mark_deleted(new_internal)
                except Exception:
                    logger.warning("Could not mark new ID %d as deleted during rollback", new_internal)
                self.deleted_ids.add(new_internal)
                # restore id_map
                self.id_map[record_id] = old_internal
                # clean up reverse_map
                self.reverse_map.pop(new_internal, None)
                self._active_count = max(0, self._active_count - 1)
                raise   # re-raise original failure

            return new_internal

    def batch_upsert(self, items: List[Tuple[int, Sequence[float]]]) -> None:
        """
        Atomic batch upsert: adds new records and updates existing ones.
        If any step fails, all changes in this call are rolled back.
        """
        with self._lock:
            if not items:
                return
            self._load_config()

            # Validate all first
            for rid, vec in items:
                self.validate_vector(vec, self.dim)

            # Snapshot state for rollback
            saved_id_map = self.id_map.copy()
            saved_reverse_map = self.reverse_map.copy()
            saved_active = self._active_count
            saved_next_id = self.next_id
            saved_deleted = self.deleted_ids.copy()

            # Classify into new and update
            new_items = []
            update_pairs = []   # (record_id, vector, old_internal)
            for rid, vec in items:
                if rid in saved_id_map:
                    old_int = saved_id_map[rid]
                    update_pairs.append((rid, vec, old_int))
                else:
                    new_items.append((rid, vec))

            added_internals = []   # list of new internal ids that were added during this batch
            old_to_restore = []    # list of (record_id, old_internal) that need to be restored on failure

            # Convert all vectors to numpy up front
            new_items_np = [(rid, np.asarray(vec, dtype=np.float32)) for rid, vec in new_items]
            update_pairs_np = [(rid, np.asarray(vec, dtype=np.float32), old_int) for rid, vec, old_int in update_pairs]

            try:
                # Phase 1: add new records (they are new, so just use add_items logic)
                if new_items_np:
                    self._ensure_capacity(len(new_items_np))
                    start = self.next_id
                    ids = list(range(start, start + len(new_items_np)))
                    vectors = np.asarray([v for _, v in new_items_np], dtype=np.float32)
                    self.index.add_items(vectors, np.array(ids, dtype=np.int32))
                    for (rid, _), int_id in zip(new_items_np, ids):
                        self.id_map[rid] = int_id
                        self.reverse_map[int_id] = rid
                        added_internals.append(int_id)
                    self.next_id += len(new_items_np)
                    self._active_count += len(new_items_np)

                # Phase 2: update existing records
                for rid, vec, old_internal in update_pairs_np:
                    # add new vector
                    new_int = self._add_new_internal(rid, vec)   # modifies id_map, reverse_map, etc.
                    added_internals.append(new_int)
                    old_to_restore.append((rid, old_internal))
                    # mark old as deleted
                    self._mark_deleted_by_internal(old_internal)

            except Exception:
                # Rollback everything: remove all added internals, restore old mappings
                logger.exception("Batch upsert failed, rolling back %d changes", len(added_internals))
                # Mark all newly added internals as deleted
                for int_id in added_internals:
                    try:
                        self.index.mark_deleted(int_id)
                    except Exception:
                        pass
                    self.deleted_ids.add(int_id)
                    self.reverse_map.pop(int_id, None)
                # Restore original mappings for updates
                for rid, old_int in old_to_restore:
                    self.id_map[rid] = old_int
                    # reverse_map[old_int] should still be there unless something weird happened
                    if old_int not in self.reverse_map:
                        self.reverse_map[old_int] = rid
                # Restore for new records: remove their id_map entries
                for (rid, _), int_id in zip(new_items_np, added_internals[:len(new_items_np)]):
                    if self.id_map.get(rid) == int_id:
                        del self.id_map[rid]
                # Restore state completely
                self.id_map = saved_id_map
                self.reverse_map = saved_reverse_map
                self.deleted_ids = saved_deleted
                self.next_id = saved_next_id
                self._active_count = saved_active
                raise

    def delete(self, record_id: int) -> None:
        """Mark a record as deleted."""
        with self._lock:
            self._load_config()
            int_id = self.id_map.pop(record_id, None)
            if int_id is None:
                return
            self._mark_deleted_by_internal(int_id)

    def search(self, vector: Sequence[float], top_k: int = 10) -> List[Dict[str, Any]]:
        """Search. Lock is only needed for reads of mutable structures? We'll use lock for consistency."""
        with self._lock:
            self._load_config()
            self.validate_vector(vector, self.dim)
            if self.index is None or self._active_count == 0:
                return []

            query_vector = np.asarray(vector, dtype=np.float32).reshape(1, -1)
            k = min(top_k, self._active_count)
            if k <= 0:
                return []

            try:
                labels, distances = self.index.knn_query(query_vector, k=k)
            except RuntimeError:
                logger.exception("HNSW knn_query failed; returning empty results")
                return []

            def score_fn(dist: float) -> float:
                return 1.0 - dist if self.space == "cosine" else 1.0 / (1.0 + float(dist))

            results = []
            seen = set()
            for label, dist in zip(labels[0], distances[0]):
                if label in self.deleted_ids:
                    continue
                rid = self.reverse_map.get(label)
                if rid is not None and rid not in seen:
                    seen.add(rid)
                    results.append({"id": rid, "score": score_fn(dist)})
            return results[:top_k]

    # ---------- Save with Manifest & checksum ----------
    def save(self) -> None:
        """Atomic save using generations, manifest, and index checksum."""
        with self._lock:
            if self.index is None:
                logger.debug("No index to save.")
                return

            current_gen = self.manifest.read_latest() or 0
            new_gen = current_gen + 1

            idx_file = self.nova_dir / f"index_{new_gen}.bin"
            map_file = self.nova_dir / f"mapping_{new_gen}.json"

            # Write index
            tmp_idx = idx_file.with_suffix(".tmp")
            try:
                self.index.save_index(str(tmp_idx))
                # Compute checksum
                sha256 = hashlib.sha256()
                with open(tmp_idx, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        sha256.update(chunk)
                index_checksum = sha256.hexdigest()
                os.replace(tmp_idx, idx_file)
            except Exception:
                if tmp_idx.exists():
                    tmp_idx.unlink(missing_ok=True)
                raise
            finally:
                if tmp_idx.exists():
                    tmp_idx.unlink(missing_ok=True)

            # Write mapping (include checksum)
            mapping_data = {
                "id_map": {str(k): v for k, v in self.id_map.items()},
                "reverse_map": {str(k): v for k, v in self.reverse_map.items()},
                "deleted_ids": list(self.deleted_ids),
                "next_id": self.next_id,
                "active_count": self._active_count,
                "dim": self.dim,
                "space": self.space,
                "M": self.M,
                "ef_construction": self.ef_construction,
                "ef_search": self.ef_search,
                "index_checksum": index_checksum,
                "last_applied_lsn": self.last_applied_lsn,
            }
            tmp_map = map_file.with_suffix(".tmp")
            try:
                save_data(mapping_data, tmp_map)
                with open(tmp_map, "rb") as f:
                    os.fsync(f.fileno())
                os.replace(tmp_map, map_file)
            except Exception:
                if tmp_map.exists():
                    tmp_map.unlink(missing_ok=True)
                raise
            finally:
                if tmp_map.exists():
                    tmp_map.unlink(missing_ok=True)

            # Update manifest
            self.manifest.write(new_gen)
            self._current_gen = new_gen

            # Clean old generations only after successful save
            self._clean_old_generations(new_gen, keep=5)
            logger.info("Saved generation %d", new_gen)

    def _clean_old_generations(self, current_gen: int, keep: int = 5) -> None:
        """Remove generations older than keep, but verify files exist."""
        for gen in range(1, current_gen - keep + 1):
            idx = self.nova_dir / f"index_{gen}.bin"
            mp = self.nova_dir / f"mapping_{gen}.json"
            if idx.exists() and mp.exists():
                idx.unlink()
                mp.unlink()

    # ---------- Load with recovery, integrity, and checksum ----------
    def load(self) -> bool:
        """Try to load the latest valid generation, falling back to older ones."""
        with self._lock:
            latest_gen = self.manifest.read_latest()
            if latest_gen is None:
                return False

            # Try from latest down to 1
            for gen in range(latest_gen, 0, -1):
                if self._load_generation(gen):
                    self._current_gen = gen
                    logger.info("Loaded generation %d", gen)
                    return True
            return False

    def _load_generation(self, gen: int) -> bool:
        """Attempt to load a specific generation. Returns True on success."""
        idx_file = self.nova_dir / f"index_{gen}.bin"
        map_file = self.nova_dir / f"mapping_{gen}.json"
        if not idx_file.exists() or not map_file.exists():
            return False

        # Load mapping
        mapping_data = load_data(map_file, default={}) or {}
        if not mapping_data:
            logger.warning("Failed to read mapping for gen %d", gen)
            return False

        # Verify configuration
        if (mapping_data.get("dim") != self.dim or
            mapping_data.get("space") != self.space or
            mapping_data.get("M") != self.M or
            mapping_data.get("ef_construction") != self.ef_construction or
            mapping_data.get("ef_search") != self.ef_search):
            logger.warning("Config mismatch for gen %d", gen)
            return False

        # Verify index checksum if present
        expected_checksum = mapping_data.get("index_checksum")
        if expected_checksum:
            sha256 = hashlib.sha256()
            try:
                with open(idx_file, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        sha256.update(chunk)
                actual = sha256.hexdigest()
                if actual != expected_checksum:
                    logger.warning("Checksum mismatch for index gen %d", gen)
                    return False
            except Exception:
                logger.warning("Failed to read index for checksum gen %d", gen)
                return False

        # Load index
        idx = hnswlib.Index(space=self.space, dim=self.dim)
        try:
            idx.load_index(str(idx_file))
        except Exception:
            logger.warning("Failed to load index for gen %d", gen)
            return False
        idx.set_ef(self.ef_search)

        # Populate structures
        id_map = {int(k): v for k, v in mapping_data.get("id_map", {}).items()}
        reverse_map = {int(k): v for k, v in mapping_data.get("reverse_map", {}).items()}
        deleted_ids = set(mapping_data.get("deleted_ids", []))
        next_id = mapping_data.get("next_id", 0)
        active_count = mapping_data.get("active_count", len(id_map))
        self.last_applied_lsn = mapping_data.get("last_applied_lsn", 0)

        # Integrity checks
        if not self._check_integrity(id_map, reverse_map, deleted_ids, next_id, active_count):
            logger.warning("Integrity check failed for gen %d", gen)
            return False

        # Apply to self
        self.index = idx
        self.id_map = id_map
        self.reverse_map = reverse_map
        self.deleted_ids = deleted_ids
        self.next_id = next_id
        self._active_count = active_count

        # Re‑mark deleted
        for int_id in self.deleted_ids:
            try:
                self.index.mark_deleted(int_id)
            except RuntimeError:
                logger.warning("Could not re‑mark deleted id %d", int_id)

        return True

    def _check_integrity(self, id_map, reverse_map, deleted_ids, next_id, active_count) -> bool:
        """Verify mapping consistency and deleted_ids range."""
        if len(id_map) != len(reverse_map):
            return False
        if set(id_map.values()) != set(reverse_map.keys()):
            return False
        if active_count != len(id_map):
            return False
        for int_id in deleted_ids:
            if int_id < 0 or int_id >= next_id:
                return False
            if int_id in reverse_map or int_id in id_map.values():
                return False
        return True

    def get_current_count(self) -> int:
        with self._lock:
            return self._active_count

    def set_last_applied_lsn(self, lsn: int) -> None:
        """Store the LSN to be persisted with the next save."""
        self.last_applied_lsn = lsn

    def clear(self) -> None:
        with self._lock:
            self.index = None
            self.id_map.clear()
            self.reverse_map.clear()
            self.deleted_ids.clear()
            self.next_id = 0
            self._active_count = 0
            self.last_applied_lsn = 0
            self._current_gen = None

    def deleted_ratio(self) -> float:
        """Return fraction of internal IDs that are deleted."""
        with self._lock:
            if self.next_id == 0:
                return 0.0
            return len(self.deleted_ids) / self.next_id


