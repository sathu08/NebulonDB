"""
cosmos/store.py
===============
NebulonCosmos – the public class that wires all submodules together.

The class itself is thin: it owns state (memtable, locks, caches …)
and delegates all logic to the purpose-built submodules:

    metadata       – meta.ndb / manifest.ndb persistence
    wal            – write-ahead log write & recovery
    index          – binary flat-file index management
    segment_reader – mmap-based record reading
    segment_writer – segment file creation & memtable flush
    compactor      – LSM-tree compaction & background daemon
"""

import time
import shutil

import threading

from pathlib import Path
from collections import OrderedDict
from typing import Any

from db.engine.utils import DatabaseConfig, BloomFilter, IndexEntry
from utils.logger import NebulonDBLogger

# ── submodules ──────────────────────────────────────────────
from . import metadata as _meta_mod
from . import wal as _wal_mod
from . import index as _idx_mod

from . import segment_reader as _reader_mod
from . import segment_writer as _writer_mod
from . import compactor as _compact_mod

from db.engine.utils import encode_object, decode_object


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()


class NebulonCosmos:
    """
    LSM-tree inspired storage engine with memtable, WAL, and immutable segments.

    Public API
    ----------
    insert(segment, doc)        -> int
    update(segment, doc)        -> int
    delete(segment, record_id)  -> int
    get(record_id, ...)       -> dict | None
    get_by_id(segment, id)      -> dict | None
    read_all(segment, ...)      -> list[dict]
    flush()
    close()
    """

    # ==========================================================
    def __init__(self, db_dir: str | Path, reset: bool = False) -> None:
        self.config = DatabaseConfig(db_dir=db_dir)

        # ── paths ─────────────────────────────────────────────
        self.db_dir          = self.config.DB_DIR
        self.seg_dir         = self.config.SEG_DIR
        self.wal_file        = self.config.WAL_FILE
        self.index_file      = self.config.INDEX_FILE
        self.meta_file       = self.config.META_FILE
        self.manifest_file   = self.config.MANIFEST_FILE

        # ── tunables ──────────────────────────────────────────
        self.flush_threshold              = self.config.FLUSH_RECORD_THRESHOLD
        self.wal_auto_flush               = self.config.WAL_AUTO_FLUSH
        self.compress_segments            = self.config.COMPRESS_SEGMENTS
        self.bloom_filter_enabled         = self.config.BLOOM_FILTER_ENABLED
        self.max_open_segments            = self.config.MAX_OPEN_SEGMENTS
        self.compaction_interval          = self.config.COMPACTION_INTERVAL
        self.max_segments_before_compact  = self.config.MAX_SEGMENTS_BEFORE_COMPACT
        self.flush_interval               = self.config.FLUSH_INTERVAL
        self.flush_size_threshold         = self.config.FLUSH_SIZE_THRESHOLD
        self.wal_fsync_interval           = self.config.WAL_FSYNC_INTERVAL

        # ── format strings / constants ────────────────────────
        self.segment_magic         = self.config.MAGIC
        self.segment_version       = self.config.VERSION
        self.segment_header_format = self.config.HEADER_FORMAT
        self.segment_header_size   = self.config.HEADER_SIZE
        self.record_header_format  = self.config.RECORD_HEADER_FORMAT
        self.record_header_size    = self.config.RECORD_HEADER_SIZE
        self.index_entry_format    = self.config.ENTRY_FORMAT
        self.index_entry_size      = self.config.ENTRY_SIZE

        # ── in-memory state ───────────────────────────────────
        self._lock                                          = threading.RLock()
        self.memtable:             dict[int, bytes]         = {}
        self._deleted:             set                      = set()
        self.latest:               dict[int, IndexEntry]    = {}
        self.meta:                 dict[str, Any]           = _meta_mod.default_meta()
        self.wal_count:            int                      = 0
        self._wal_bytes_since_fsync: int                    = 0
        self.memtable_oldest_ts:   float                    = 0.0
        self.wal_handle:           Any | None            = None
        self.segment_cache:        OrderedDict              = OrderedDict()
        self.segment_size_cache:   dict[int, int]           = {}
        self.bloom_filter_cache:   dict[int, BloomFilter]   = {}
        self.segment_info:         dict[int, dict]          = {}
        self.manifest:             list[str]                = []
        self.memtable_bytes:       int                      = 0
        self.max_memtable_bytes = 256 * 1024 * 1024

        # ── background threads ─────────────────────────────────
        self._compaction_thread:     threading.Thread | None = None
        self._compaction_stop_event: threading.Event           = threading.Event()
        self._flush_thread:          threading.Thread | None = None
        self._flush_stop_event:      threading.Event           = threading.Event()

        self._init_db(reset)

    # =========================================================
    # Internal helpers – thin wrappers that call submodules
    # =========================================================

    # ── metadata ─────────────────────────────────────────────
    def _atomic_write(self, file_path: Path, data_bytes: bytes) -> None:
        _meta_mod.atomic_write(file_path, data_bytes)

    def _save_meta(self) -> None:
        _meta_mod.save_meta(self.meta_file, self.meta)

    def _load_meta(self) -> None:
        self.meta = _meta_mod.load_meta(self.meta_file)

    def _default_meta(self) -> dict[str, Any]:
        return _meta_mod.default_meta()

    def _save_manifest(self) -> None:
        _meta_mod.save_manifest(self.manifest_file, self.manifest)


    def _load_manifest(self) -> None:
        self.manifest = _meta_mod.load_manifest(self.manifest_file, self.seg_dir)

    # ── id / version generators ───────────────────────────────
    def _ensure_global_record_id(self) -> None:
        """Seed the global record-id counter from existing per-segment counters."""
        if self.meta.get("global_record_id"):
            return
        self.meta["global_record_id"] = max(
            self.meta.get("tables", {}).values(), default=0
        )

    def _next_id(self, segment: str) -> int:
        self._ensure_global_record_id()
        self.meta["global_record_id"] += 1
        return self.meta["global_record_id"]

    def _next_version(self) -> int:
        self.meta["global_version"] += 1
        return self.meta["global_version"]

    def _get_next_segment_id(self) -> int:
        self.meta["last_segment_id"] += 1
        return self.meta["last_segment_id"]

    # ── segment validation / offsets ─────────────────────────
    def _validate_segment(self, seg_path: Path):
        return _reader_mod.validate_segment(
            seg_path,
            self.segment_header_format,
            self.segment_header_size,
            self.segment_magic,
            self.segment_version,
        )

    def _get_segment_data_offset(self, seg_path: Path, bf_size: int) -> int:
        return _reader_mod.get_segment_data_offset(
            seg_path, bf_size,
            self.segment_header_size,
            self.record_header_size,
            self.record_header_format,
        )

    # ── index ─────────────────────────────────────────────────
    def _append_index_entries(self, entries) -> None:
        _idx_mod.append_index_entries(self.index_file, entries, self.index_entry_format)

    def _rewrite_index_from_latest(self) -> None:
        with self._lock:
            _idx_mod.rewrite_index_from_latest(
                self.index_file, self.latest, self.index_entry_format
            )

    def _rebuild_index_from_all_segments(self) -> None:
        self.latest = _idx_mod.rebuild_index_from_all_segments(
            manifest=self.manifest,
            seg_dir=self.seg_dir,
            index_file=self.index_file,
            validate_segment_fn=self._validate_segment,
            get_segment_data_offset_fn=self._get_segment_data_offset,
            segment_size_cache=self.segment_size_cache,
            bloom_filter_cache=self.bloom_filter_cache,
            segment_info=self.segment_info,
            bloom_filter_enabled=self.bloom_filter_enabled,
            compress_segments=self.compress_segments,
            record_header_size=self.record_header_size,
            record_header_format=self.record_header_format,
            index_entry_format=self.index_entry_format,
        )

    def _load_index(self) -> None:
        self.latest = _idx_mod.load_index(
            index_file=self.index_file,
            manifest=self.manifest,
            seg_dir=self.seg_dir,
            validate_segment_fn=self._validate_segment,
            segment_size_cache=self.segment_size_cache,
            bloom_filter_cache=self.bloom_filter_cache,
            segment_info=self.segment_info,
            bloom_filter_enabled=self.bloom_filter_enabled,
            segment_header_size=self.segment_header_size,
            index_entry_size=self.index_entry_size,
            index_entry_format=self.index_entry_format,
            record_header_size=self.record_header_size,
            rebuild_fn=self._rebuild_index_from_all_segments,
        )

    # ── WAL ───────────────────────────────────────────────────
    def _write_wal_record(self, record_bytes: bytes) -> None:
        self._wal_bytes_since_fsync = _wal_mod.write_wal_record(
            self.wal_handle,
            record_bytes,
            self.wal_auto_flush,
            self._wal_bytes_since_fsync,
            self.wal_fsync_interval,
        )

    def _write_wal_records_batch(self, record_bytes_list: list[bytes]) -> None:
        self._wal_bytes_since_fsync = _wal_mod.write_wal_records_batch(
            self.wal_handle,
            record_bytes_list,
            self.wal_auto_flush,
            self._wal_bytes_since_fsync,
            self.wal_fsync_interval,
        )

    def _recover_wal(self) -> None:
        _wal_mod.recover_wal(
            wal_file=self.wal_file,
            memtable=self.memtable,
            deleted=self._deleted,
            meta=self.meta,
            save_meta_fn=self._save_meta,
        )
        self.wal_count = len(self.memtable)

    # ── segment writer ────────────────────────────────────────
    def _write_segment(self, seg_path: Path, record_list: list[bytes], seg_id: int) -> None:
        _writer_mod.write_segment(
            seg_path=seg_path,
            record_list=record_list,
            seg_id=seg_id,
            segment_header_format=self.segment_header_format,
            segment_magic=self.segment_magic,
            segment_version=self.segment_version,
            record_header_format=self.record_header_format,
            bloom_filter_enabled=self.bloom_filter_enabled,
            compress_segments=self.compress_segments,
            latest=self.latest,
            segment_size_cache=self.segment_size_cache,
            bloom_filter_cache=self.bloom_filter_cache,
            segment_info=self.segment_info,
            manifest=self.manifest,
            append_index_entries_fn=self._append_index_entries
        )

    def _write_segment_streaming(
        self,
        seg_path: Path,
        sources: dict[int, tuple],
        seg_id: int,
        append_index: bool = True,
    ) -> None:
        _writer_mod.write_segment_streaming(
            seg_path=seg_path,
            sources=sources,
            seg_id=seg_id,
            seg_dir=self.seg_dir,
            segment_header_format=self.segment_header_format,
            segment_magic=self.segment_magic,
            segment_version=self.segment_version,
            record_header_format=self.record_header_format,
            bloom_filter_enabled=self.bloom_filter_enabled,
            compress_segments=self.compress_segments,
            latest=self.latest,
            segment_size_cache=self.segment_size_cache,
            bloom_filter_cache=self.bloom_filter_cache,
            segment_info=self.segment_info,
            read_payload_fn=self._read_payload_from_file,
            append_index_entries_fn=self._append_index_entries,
            append_index=append_index,
        )

    def _flush(self, force: bool = False) -> None:
        with self._lock:
            if not self.seg_dir.exists():
                self._flush_stop_event.set()
                return
            wal_count_ref = [self.wal_count]
            _writer_mod.flush(
                memtable=self.memtable,
                deleted=self._deleted,
                wal_handle=self.wal_handle,
                wal_count_ref=wal_count_ref,
                flush_threshold=self.flush_threshold,
                get_next_segment_id_fn=self._get_next_segment_id,
                seg_dir=self.seg_dir,
                write_segment_fn=self._write_segment,
                # pass a callback that atomically saves manifest+meta
                save_manifest_and_meta_fn=lambda: _meta_mod.save_manifest_and_meta(
                    self.manifest_file, self.meta_file,
                    self.manifest, self.meta
                ),
                force=force,
            )
            self.wal_count = wal_count_ref[0]
            self._wal_bytes_since_fsync = 0
            self.memtable_bytes = 0
            self.memtable_oldest_ts = 0.0

    # ── compaction ────────────────────────────────────────────
    def _cleanup_segments(self, fnames: list[str], delete_files: bool = True) -> None:
        _compact_mod.cleanup_segments(
            fnames=fnames,
            seg_dir=self.seg_dir,
            segment_cache=self.segment_cache,
            segment_size_cache=self.segment_size_cache,
            bloom_filter_cache=self.bloom_filter_cache,
            segment_info=self.segment_info,
            delete_files=delete_files,
        )

    def _remove_segments_and_rebuild(self, fnames: list[str]) -> None:
        _compact_mod.remove_segments_and_rebuild(
            fnames=fnames,
            seg_dir=self.seg_dir,
            manifest=self.manifest,
            segment_cache=self.segment_cache,
            segment_size_cache=self.segment_size_cache,
            bloom_filter_cache=self.bloom_filter_cache,
            segment_info=self.segment_info,
            save_manifest_and_meta_fn=lambda: _meta_mod.save_manifest_and_meta(
                self.manifest_file, self.meta_file,
                self.manifest, self.meta
            ),
            rebuild_index_fn=self._rebuild_index_from_all_segments,
        )

    def _compact(self, segments_to_merge: list[str] | None = None) -> None:
        with self._lock:
            _compact_mod.compact(
            segments_to_merge=segments_to_merge,
            manifest=self.manifest,
            seg_dir=self.seg_dir,
            max_segments_before_compact=self.max_segments_before_compact,
            validate_segment_fn=self._validate_segment,
            get_segment_data_offset_fn=self._get_segment_data_offset,
            get_next_segment_id_fn=self._get_next_segment_id,
            write_segment_streaming_fn=self._write_segment_streaming,
            rewrite_index_fn=self._rebuild_index_from_all_segments,
            segment_cache=self.segment_cache,
            segment_size_cache=self.segment_size_cache,
            bloom_filter_cache=self.bloom_filter_cache,
            segment_info=self.segment_info,
            record_header_size=self.record_header_size,
            record_header_format=self.record_header_format,
            save_manifest_and_meta_fn=lambda: _meta_mod.save_manifest_and_meta(
                self.manifest_file, self.meta_file,
                self.manifest, self.meta
            ),
        )

    def _background_compaction_loop(self) -> None:
        def _do_compact():
            to_merge = self.manifest[:len(self.manifest) - 3]
            if len(to_merge) >= 2:
                self._compact(to_merge)

        _compact_mod.background_compaction_loop(
            stop_event=self._compaction_stop_event,
            compaction_interval=self.compaction_interval,
            manifest=self.manifest,
            max_segments_before_compact=self.max_segments_before_compact,
            compact_fn=_do_compact,
            db_dir=self.db_dir,
        )

    # ── background flush loop ─────────────────────────────────
    def _background_flush_loop(self) -> None:
        while not self._flush_stop_event.wait(timeout=self.flush_interval):
            with self._lock:
                if not self.seg_dir.exists():
                    self._flush_stop_event.set()
                    return
                if not self.memtable:
                    continue

                if (self.wal_count >= self.flush_threshold or
                    self.memtable_bytes >= self.flush_size_threshold):
                    self._flush(force=True)

                if self.memtable_oldest_ts > 0:
                    oldest_age = time.time() - self.memtable_oldest_ts
                    if oldest_age >= 30:
                        self._flush(force=True)

    # ── segment reader ────────────────────────────────────────
    def _read_payload_at_offset(self, seg_id: int, offset: int) -> bytes | None:
        return _reader_mod.read_payload_at_offset(
            seg_id=seg_id,
            offset=offset,
            seg_dir=self.seg_dir,
            segment_cache=self.segment_cache,
            max_open_segments=self.max_open_segments,
            segment_info=self.segment_info,
            compress_segments=self.compress_segments,
            record_header_size=self.record_header_size,
            record_header_format=self.record_header_format,
        )

    def _read_payload_from_file(self, seg_path: Path, offset: int) -> bytes | None:
        seg_id = int(seg_path.stem.split('_')[1])
        return self._read_payload_at_offset(seg_id, offset)

    # ── record building ───────────────────────────────────────
    def _build_record(
        self, segment: str, doc: dict[str, Any], is_delete: bool = False
    ) -> "tuple[bytes, int, int]":
        record = doc.copy()
        if "id" in record:
            record["_id"] = record.pop("id")
        elif "_id" not in record or record["_id"] is None:
            record["_id"] = self._next_id(segment)
        record["_segment"]   = segment
        version = self._next_version()
        record["_version"] = version
        if is_delete:
            record["_deleted"] = True
        return encode_object(record), record["_id"], version

    # ── write record (memtable + WAL) ─────────────────────────
    def _write_record(self, segment: str, doc: dict[str, Any], is_delete: bool = False) -> int:
        with self._lock:
            return self._write_record_unsafe(segment, doc, is_delete)

    def _write_record_unsafe(self, segment: str, doc: dict[str, Any], is_delete: bool = False) -> int:
        rec_bytes, rec_id, version = self._build_record(segment, doc, is_delete)
        key       = (segment, rec_id)

        self._write_wal_record(rec_bytes)

        was_empty = not self.memtable
        if is_delete:
            self._deleted.add(key)
            self._deleted.add(rec_id)
        else:
            self._deleted.discard(key)
            self._deleted.discard(rec_id)

        old_size = len(self.memtable.get(key, b''))
        self.memtable[key] = rec_bytes
        self.memtable_bytes += len(rec_bytes) - old_size

        self.wal_count += 1
        if was_empty:
            self.memtable_oldest_ts = time.time()

        if self.memtable_bytes >= self.max_memtable_bytes:
            self._flush(force=True)

        logger.debug(f"{'DELETE' if is_delete else 'INSERT/UPDATE'} segment={segment} id={rec_id} version={version}")

        return rec_id

    # ── existence check ───────────────────────────────────────
    def _exists_unsafe(self, rec_id: int, segment: str = "_main") -> bool:
        key = (segment, rec_id)
        if key in self.memtable:
            try:
                rec = decode_object(self.memtable[key])
                return not rec.get("_deleted", False)
            except Exception:
                return False
        if key in self._deleted or rec_id in self._deleted:
            return False
        return self.get(rec_id, segment=segment) is not None

    # =========================================================
    # Public API
    # =========================================================

    def insert(self, segment: str, doc: dict[str, Any]) -> int:
        doc.pop("id", None)
        return self._write_record(segment, doc, is_delete=False)

    def insert_many(self, segment: str, docs: list[dict[str, Any]]) -> list[int]:
        """
        Bulk insert: encode all docs, append them to the WAL in a single
        write, and update the memtable under one lock acquisition.
        """
        with self._lock:
            built = []
            for doc in docs:
                doc2 = doc.copy()
                doc2.pop("id", None)
                rec_bytes, rec_id, _ = self._build_record(segment, doc2, False)
                built.append(((segment, rec_id), rec_bytes))

            self._write_wal_records_batch([rb for _, rb in built])

            was_empty = not self.memtable
            rec_ids = []
            for key, rec_bytes in built:
                rec_id = key[1]
                self._deleted.discard(key)
                self._deleted.discard(rec_id)
                self.memtable[key] = rec_bytes
                self.memtable_bytes += len(rec_bytes)
                rec_ids.append(rec_id)
            self.wal_count += len(built)
            if was_empty:
                self.memtable_oldest_ts = time.time()

            if self.memtable_bytes >= self.max_memtable_bytes:
                self._flush(force=True)

            logger.debug(f"BULK INSERT {len(built)} records segment={segment}")
            return rec_ids

    def update(self, segment: str, doc: dict[str, Any]) -> int:
        rec_id = doc.get("id", doc.get("_id"))
        if rec_id is None:
            logger.error("Document must contain 'id' or '_id'")
        with self._lock:
            return self._write_record_unsafe(segment, doc, is_delete=False)

    def delete(self, segment: str, record_id: Any) -> int:
        with self._lock:
            if not self._exists_unsafe(record_id, segment=segment):
                return 0
            return self._write_record_unsafe(segment, {"id": record_id}, is_delete=True)

    # ── reading ───────────────────────────────────────────────
    def get(self, record_id: int, segment: str = "_main", include_internal: bool = False) -> dict[str, Any] | None:
        """
        Retrieve a document by its record ID and segment.
        """
        with self._lock:
            key = (segment, record_id)
            if key in self._deleted or record_id in self._deleted:
                return None

            payload = self.memtable.get(key)
            if payload is None and segment == "_main":
                payload = self.memtable.get(record_id)

            if payload is not None:
                rec_dict = decode_object(payload)
                if rec_dict.get("_deleted", False):
                    return None
                if include_internal:
                    return rec_dict
                res = {k: v for k, v in rec_dict.items() if not k.startswith("_")}
                if "_id" in rec_dict and "_id" not in res:
                    res["_id"] = rec_dict["_id"]
                return res

            entry = self.latest.get(key) or self.latest.get(record_id)
            if not entry:
                return None

            if self.bloom_filter_enabled and entry.segment_id in self.bloom_filter_cache:
                bf = self.bloom_filter_cache[entry.segment_id]
                if not bf.might_contain(record_id):
                    return None

            payload = self._read_payload_at_offset(entry.segment_id, entry.offset)
            if payload is None:
                return None

            try:
                rec_dict = decode_object(payload)
            except Exception:
                return None

            if rec_dict.get("_deleted", False):
                return None

            # The persisted index is keyed only by bare record_id while the
            # id space is global across tables, so a fallback hit may belong
            # to another segment; reject it to avoid cross-table leakage.
            if segment and rec_dict.get("_segment") != segment:
                return None

            if include_internal:
                return rec_dict
            res = {k: v for k, v in rec_dict.items() if not k.startswith("_")}
            if "_id" in rec_dict and "_id" not in res:
                res["_id"] = rec_dict["_id"]
            return res

    def read_all(
        self,
        segment: str | None = None,
        include_internal: bool = False,
    ) -> list[dict[str, Any]]:
        target_segment = segment or "_main"
        docs = []

        with self._lock:
            memtable = self.memtable

            # ── fast path 1: memtable (in-memory, zero disk I/O) ──
            for key, payload in memtable.items():
                if isinstance(key, tuple):
                    t, rec_id = key
                    if segment is not None and t != segment:
                        continue
                else:
                    rec_id = key
                    t = target_segment
                    if target_segment != "_main":
                        continue
                    if (t, rec_id) in memtable:
                        continue
                rec_dict = decode_object(payload)
                if rec_dict.get("_deleted", False):
                    continue
                if rec_dict.get("_segment", t) != target_segment:
                    continue
                docs.append(rec_dict)

            # ── fast path 2: one sequential scan per segment ────
            # `latest` is the authority on which (segment_id, offset) holds the
            # newest version; memtable always wins over disk.
            live_locations = {
                (e.segment_id, e.offset)
                for e in self.latest.values()
                if e.segment_id in self.segment_info
            }
            mem_ids = set()
            for key in memtable:
                mem_ids.add(key[1] if isinstance(key, tuple) else key)

            for fname in self.manifest:
                seg_path = self.seg_dir / fname
                try:
                    seg_id = int(fname.split("_")[1].split(".")[0])
                except (IndexError, ValueError):
                    continue
                info = self.segment_info.get(seg_id)
                if not info:
                    continue
                count = info.get("count", 0)
                compressed = info.get("compressed", self.compress_segments)
                bf = info.get("bf")
                bf_size = len(bf.to_bytes()) if bf else 0
                data_offset = self._get_segment_data_offset(seg_path, bf_size)

                for offset, payload in _reader_mod.scan_segment_payloads(
                    seg_path=seg_path,
                    data_offset=data_offset,
                    count=count,
                    compressed=compressed,
                    record_header_size=self.record_header_size,
                    record_header_format=self.record_header_format,
                ):
                    if (seg_id, offset) not in live_locations:
                        continue
                    try:
                        rec_dict = decode_object(payload)
                    except Exception:
                        continue
                    if rec_dict.get("_deleted", False):
                        continue
                    if rec_dict.get("_segment", target_segment) != target_segment:
                        continue
                    if rec_dict.get("_id") in mem_ids:
                        continue
                    docs.append(rec_dict)

        if include_internal:
            return docs
        results = []
        for doc in docs:
            res = {k: v for k, v in doc.items() if not k.startswith("_")}
            if "_id" in doc and "_id" not in res:
                res["_id"] = doc["_id"]
            results.append(res)
        return results

    def get_by_id(self, segment: str, record_id: Any) -> dict[str, Any] | None:
        return self.get(record_id, segment=segment)

    # =========================================================
    # Lifecycle
    # =========================================================

    def _init_db(self, reset: bool = False) -> None:
        with self._lock:
            if reset and self.db_dir.exists():
                shutil.rmtree(self.db_dir)

            self.db_dir.mkdir(parents=True, exist_ok=True)
            self.seg_dir.mkdir(parents=True, exist_ok=True)

            if not self.wal_file.exists():
                self.wal_file.touch()
            if self.wal_handle is None:
                self.wal_handle = self.wal_file.open("a+b")

            if not self.index_file.exists():
                self.index_file.touch()

            self._load_meta()
            self._ensure_global_record_id()
            self._load_manifest()
            self._load_index()
            self._recover_wal()
            self.wal_count = len(self.memtable)
            self.memtable_bytes = sum(len(v) for v in self.memtable.values())
            if self.wal_count > 0:
                logger.info(f"Recovered {self.wal_count} records – flushing immediately")
            self.segment_cache.clear()

            if self._compaction_thread is None or not self._compaction_thread.is_alive():
                self._compaction_stop_event.clear()
                self._compaction_thread = threading.Thread(
                    target=self._background_compaction_loop,
                    daemon=True,
                )
                self._compaction_thread.start()

            if self._flush_thread is None or not self._flush_thread.is_alive():
                self._flush_stop_event.clear()
                self._flush_thread = threading.Thread(
                    target=self._background_flush_loop,
                    daemon=True,
                )
                self._flush_thread.start()

    def flush(self) -> None:
        """Flush the current memtable to a new segment immediately."""
        with self._lock:
            self._flush(force=True)

    def checkpoint_wal(self) -> None:
        """
        Rewrite the WAL from the live memtable without writing a segment.

        Keeps the WAL as the sole durable store below the flush threshold
        while dropping superseded versions, so a fresh engine init no longer
        replays stale rows.
        """
        with self._lock:
            self.wal_handle = _wal_mod.checkpoint_wal(
                self.wal_file, self.wal_handle, self.memtable
            )
            self.wal_count = len(self.memtable)
            self._wal_bytes_since_fsync = 0
            self.memtable_bytes = sum(len(v) for v in self.memtable.values())

    def close(self) -> None:
        # Signal threads to stop
        self._flush_stop_event.set()
        self._compaction_stop_event.set()

        # Wait for them to finish
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=5)
        if self._compaction_thread and self._compaction_thread.is_alive():
            self._compaction_thread.join(timeout=5)

        # Close handles safely
        with self._lock:
            if self.wal_handle:
                self.wal_handle.close()
                self.wal_handle = None
            for f, mm in self.segment_cache.values():
                mm.close()
                f.close()
            self.segment_cache.clear()

        # Atomic final metadata write (no flush)
        _meta_mod.save_manifest_and_meta(
            self.manifest_file, self.meta_file,
            self.manifest, self.meta)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
