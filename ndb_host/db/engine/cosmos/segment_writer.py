"""
cosmos/segment_writer.py
========================
Handles writing immutable segment files for NebulonCosmos.

Responsibilities
----------------
- `write_segment`          – create a new segment from an in-memory list of payloads.
- `write_segment_streaming`– create a new segment by streaming records from existing
                             segment files (used during compaction).
- `flush`                  – move the current memtable to a segment and reset WAL.
"""

import os
import struct
import zlib

from pathlib import Path
from typing import Any

from db.engine.utils import BloomFilter, IndexEntry
from utils.logger import NebulonDBLogger

from db.engine.utils import decode_object


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#  Write segment from payload list
# ==========================================================

def write_segment(
    seg_path: Path,
    record_list: list[bytes],
    seg_id: int,
    # format strings / constants
    segment_header_format: str,
    segment_magic: int,
    segment_version: int,
    record_header_format: str,
    bloom_filter_enabled: bool,
    compress_segments: bool,
    # mutable state dicts (mutated in-place)
    latest: dict[int, IndexEntry],
    segment_size_cache: dict[int, int],
    bloom_filter_cache: dict[int, BloomFilter],
    segment_info: dict[int, dict],
    manifest: list[str],
    # callbacks
    append_index_entries_fn,
) -> None:
    """
    Write *record_list* to a new segment file atomically.

    Steps
    -----
    1. Build bloom filter from the record IDs.
    2. Write header + bloom-filter blob + compressed+CRC records to a temp file.
    3. Atomically rename temp -> seg_path.
    4. Update index, caches, manifest.
    """
    temp_path = seg_path.with_suffix(".tmp")

    # Build bloom filter
    bf = BloomFilter(len(record_list)) if bloom_filter_enabled else None
    if bf:
        for rec_payload in record_list:
            try:
                rec_dict = decode_object(rec_payload)
                rec_id = rec_dict.get("_id")
                if rec_id is not None:
                    bf.add(rec_id)
            except Exception:
                pass

    bf_bytes = bf.to_bytes() if bf else b""

    with temp_path.open("wb") as f:
        header = struct.pack(
            segment_header_format,
            segment_magic, segment_version,
            len(record_list), len(bf_bytes), compress_segments,
        )
        f.write(header)
        if bf_bytes:
            f.write(bf_bytes)

        offsets = []
        for rec_payload in record_list:
            offset = f.tell()
            try:
                rec_dict = decode_object(rec_payload)
                rec_id = rec_dict.get("_id")
                version = rec_dict.get("_version", 0)
            except Exception:
                rec_id = None
                version = 0

            # Guard: _id must be a plain int for struct.pack("<QIQQ", ...).
            # Non-int values (list, tuple, None) come from corrupt or legacy
            # records and must be skipped to avoid "required argument is not
            # an integer" errors.
            if not isinstance(rec_id, int):
                continue
            version = int(version) if isinstance(version, (int, float)) else 0

            comp_data = zlib.compress(rec_payload) if compress_segments else rec_payload
            crc = zlib.crc32(rec_payload) & 0xFFFFFFFF
            f.write(struct.pack(record_header_format, crc, len(comp_data), len(rec_payload)))
            f.write(comp_data)
            if rec_id is not None:
                offsets.append((rec_id, offset, version))

        f.flush()
        os.fsync(f.fileno())

    os.replace(temp_path, seg_path)
    fd = os.open(str(seg_path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

    # ── Update index and caches ───────────────────────────────
    index_entries = []
    for rec_id, offset, version in offsets:
        if rec_id not in latest or latest[rec_id].version < version:
            latest[rec_id] = IndexEntry(segment_id=seg_id, offset=offset, version=version)
        index_entries.append((rec_id, seg_id, offset, version))

    append_index_entries_fn(index_entries)

    segment_size_cache[seg_id] = seg_path.stat().st_size
    if bf:
        bloom_filter_cache[seg_id] = bf

    segment_info[seg_id] = {
        "count": len(record_list),
        "compressed": compress_segments,
    }
    if bf:
        segment_info[seg_id]["bf"] = bf

    manifest.append(seg_path.name)
    manifest.sort()


# ==========================================================
#  Write segment streaming (for compaction)
# ==========================================================

def write_segment_streaming(
    seg_path: Path,
    sources: dict[tuple[str, int], tuple[str, int, int]],  # (segment, rec_id) -> (fname, src_offset, version)
    seg_id: int,
    seg_dir: Path,
    # format strings / constants
    segment_header_format: str,
    segment_magic: int,
    segment_version: int,
    record_header_format: str,
    bloom_filter_enabled: bool,
    compress_segments: bool,
    # mutable state dicts
    latest: dict[int, IndexEntry],
    segment_size_cache: dict[int, int],
    bloom_filter_cache: dict[int, BloomFilter],
    segment_info: dict[int, dict],
    # callbacks
    read_payload_fn,           # (seg_path, offset) -> Optional[bytes]
    append_index_entries_fn,   # (entries) -> None
    append_index: bool = True,
) -> None:
    """
    Write a new segment by streaming payloads from their original files.

    Used exclusively during compaction so that we avoid loading everything
    into memory at once.
    If *append_index* is False the index file is not touched (caller will
    rewrite it afterwards).
    """
    temp_path = seg_path.with_suffix(".tmp")

    bf = None
    if bloom_filter_enabled:
        bf = BloomFilter(len(sources))
        for tkey in sources:
            bf.add(tkey[1])
    bf_bytes = bf.to_bytes() if bf else b""

    with temp_path.open("wb") as out:
        header = struct.pack(
            segment_header_format,
            segment_magic, segment_version,
            len(sources), len(bf_bytes), compress_segments,
        )
        out.write(header)
        if bf_bytes:
            out.write(bf_bytes)

        index_entries = []
        for tkey, (fname, src_offset, version) in sources.items():
            rec_id = tkey[1]
            src_path = seg_dir / fname
            payload = read_payload_fn(src_path, src_offset)
            if payload is None:
                logger.warning(f"Skipping missing record {rec_id} from {fname}")
                continue

            offset = out.tell()
            if compress_segments:
                try:
                    comp_data = zlib.compress(payload)
                except zlib.error:
                    continue
            else:
                comp_data = payload
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            out.write(struct.pack(record_header_format, crc, len(comp_data), len(payload)))
            out.write(comp_data)

            if rec_id not in latest or latest[rec_id].version <= version:
                latest[rec_id] = IndexEntry(segment_id=seg_id, offset=offset, version=version)
            index_entries.append((rec_id, seg_id, offset, version))

        out.flush()
        os.fsync(out.fileno())

    os.replace(temp_path, seg_path)
    fd = os.open(str(seg_path.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

    # ── Cache updates ─────────────────────────────────────────
    segment_size_cache[seg_id] = seg_path.stat().st_size
    if bf:
        bloom_filter_cache[seg_id] = bf
    segment_info[seg_id] = {
        "count": len(sources),
        "compressed": compress_segments,
    }
    if bf:
        segment_info[seg_id]["bf"] = bf

    if append_index:
        append_index_entries_fn(index_entries)


# ==========================================================
#  Flush memtable to a segment
# ==========================================================

def flush(
    memtable: dict[int, bytes],
    deleted: set[int],
    wal_handle: Any | None,
    wal_count_ref: list[int],
    flush_threshold: int,
    get_next_segment_id_fn,
    seg_dir: Path,
    write_segment_fn,
    save_manifest_and_meta_fn,
    force: bool = False,
) -> None:
    """
    If the memtable is large enough (or *force* is True), flush it to a new segment.

    Clears the memtable and resets the WAL on success.
    """
    if not force and wal_count_ref[0] < flush_threshold:
        return
    if not memtable:
        if force and wal_handle:
            wal_handle.seek(0)
            wal_handle.truncate(0)
            wal_handle.flush()
            os.fsync(wal_handle.fileno())
        return

    records = list(memtable.values())
    seg_id = get_next_segment_id_fn()
    seg_name = f"seg_{seg_id}.ndb"
    seg_path = seg_dir / seg_name

    try:
        write_segment_fn(seg_path, records, seg_id)
    except Exception as e:
        logger.error(f"Flush failed writing segment {seg_name}: {e}")
        return  # keep memtable and WAL intact

    # ── Success: clear memory and truncate WAL ────────────────
    memtable.clear()
    deleted.clear()
    wal_count_ref[0] = 0

    if wal_handle:
        wal_handle.seek(0)
        wal_handle.truncate(0)
        wal_handle.flush()
        os.fsync(wal_handle.fileno())

    save_manifest_and_meta_fn()
    logger.info(f"Flushed {len(records)} records to {seg_name}")
