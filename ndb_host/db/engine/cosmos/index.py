"""
cosmos/index.py
===============
Manages the binary flat-file index for NebulonCosmos.

The index maps record_id -> (segment_id, offset, version) on disk.
At load-time the whole file is read with mmap for speed.
When a record is written we append a new entry; compaction rewrites
the whole file from the in-memory `latest` dict.
"""

import mmap
import os
import struct

from typing import Dict, List, Tuple
from pathlib import Path

from db.engine.utils import BloomFilter, IndexEntry
from utils.logger import NebulonDBLogger

from db.engine.utils import decode_object


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()


# ==========================================================
#  Index load
# ==========================================================

def load_index(
    index_file: Path,
    manifest: List[str],
    seg_dir: Path,
    validate_segment_fn,          # (seg_path) -> (valid, count, bf_data, compressed)
    segment_size_cache: Dict[int, int],
    bloom_filter_cache: Dict[int, BloomFilter],
    segment_info: Dict[int, dict],
    bloom_filter_enabled: bool,
    segment_header_size: int,
    index_entry_size: int,
    index_entry_format: str,
    record_header_size: int,
    rebuild_fn,                    # callable: _rebuild_index_from_all_segments()
) -> Dict[int, IndexEntry]:
    """Load (or rebuild) the in-memory latest-entry dict from disk."""
    try:
        if not index_file.exists() or index_file.stat().st_size == 0:
            return {}

        segment_size_cache.clear()
        bloom_filter_cache.clear()
        segment_info.clear()

        for fname in manifest:
            seg_path = seg_dir / fname
            valid, count, bf_data, compressed = validate_segment_fn(seg_path)
            if not valid:
                logger.warning(f"Segment {fname} corrupt, skipping")
                continue
            seg_id = int(fname.split('_')[1].split('.')[0])
            segment_size_cache[seg_id] = seg_path.stat().st_size
            info = {"count": count, "compressed": compressed}
            if bloom_filter_enabled and bf_data:
                bf = BloomFilter.from_bytes(bf_data, count)
                bloom_filter_cache[seg_id] = bf
                info["bf"] = bf
            segment_info[seg_id] = info

        latest: Dict[int, IndexEntry] = {}
        with index_file.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                size = mm.size()
                for pos in range(0, size, index_entry_size):
                    if pos + index_entry_size > size:
                        break
                    data = mm[pos:pos + index_entry_size]
                    rec_id, seg_id, offset, version = struct.unpack(index_entry_format, data)
                    if seg_id not in segment_size_cache:
                        continue
                    if offset + record_header_size > segment_size_cache[seg_id]:
                        continue
                    if rec_id not in latest or latest[rec_id].version < version:
                        latest[rec_id] = IndexEntry(
                            segment_id=seg_id, offset=offset, version=version
                        )
            finally:
                mm.close()

        logger.info(f"Loaded {len(latest)} latest entries")
        return latest

    except Exception:
        logger.warning("Index corrupted. Rebuilding index.")
        rebuild_fn()
        return {}


# ==========================================================
#  Index append
# ==========================================================

def append_index_entries(
    index_file: Path,
    entries: List[Tuple[int, int, int, int]],
    index_entry_format: str,
) -> None:
    """Append a batch of (rec_id, seg_id, offset, version) entries to the index file."""
    with index_file.open("ab") as f:
        for rec_id, seg_id, offset, version in entries:
            f.write(struct.pack(index_entry_format, rec_id, seg_id, offset, version))
        f.flush()
        os.fsync(f.fileno())


# ==========================================================
#  Index rewrite (post-compaction)
# ==========================================================

def rewrite_index_from_latest(
    index_file: Path,
    latest: Dict[int, IndexEntry],
    index_entry_format: str,
) -> None:
    """Overwrite the index file with exactly the entries in *latest*."""
    with index_file.open("wb") as f:
        for rec_id, entry in latest.items():
            f.write(struct.pack(
                index_entry_format,
                rec_id, entry.segment_id, entry.offset, entry.version,
            ))
        f.flush()
        os.fsync(f.fileno())
    logger.debug(f"Index rewritten with {len(latest)} entries.")


# ==========================================================
#  Full index rebuild from all segment files
# ==========================================================

def rebuild_index_from_all_segments(
    manifest: List[str],
    seg_dir: Path,
    index_file: Path,
    validate_segment_fn,
    get_segment_data_offset_fn,
    segment_size_cache: Dict[int, int],
    bloom_filter_cache: Dict[int, BloomFilter],
    segment_info: Dict[int, dict],
    bloom_filter_enabled: bool,
    compress_segments: bool,
    record_header_size: int,
    record_header_format: str,
    index_entry_format: str,
) -> Dict[int, IndexEntry]:
    import zlib

    latest_per_id: Dict[int, IndexEntry] = {}

    segment_size_cache.clear()
    bloom_filter_cache.clear()
    segment_info.clear()

    for fname in manifest:
        seg_path = seg_dir / fname
        valid, count, bf_data, compressed = validate_segment_fn(seg_path)
        if not valid:
            continue
        seg_id = int(fname.split('_')[1].split('.')[0])
        segment_size_cache[seg_id] = seg_path.stat().st_size

        info = {"count": count, "compressed": compressed}
        if bloom_filter_enabled and bf_data:
            bf = BloomFilter.from_bytes(bf_data, count)
            bloom_filter_cache[seg_id] = bf
            info["bf"] = bf
        segment_info[seg_id] = info

        data_offset = get_segment_data_offset_fn(seg_path, len(bf_data) if bf_data else 0)
        with seg_path.open("rb") as f:
            f.seek(data_offset)
            for _ in range(count):
                offset = f.tell()
                header_data = f.read(record_header_size)
                if len(header_data) != record_header_size:
                    break
                stored_crc, comp_len, _ = struct.unpack(record_header_format, header_data)
                comp_data = f.read(comp_len)
                if len(comp_data) != comp_len:
                    break
                if compressed:
                    try:
                        payload = zlib.decompress(comp_data)
                    except Exception:
                        continue
                else:
                    payload = comp_data
                if (zlib.crc32(payload) & 0xFFFFFFFF) != stored_crc:
                    continue
                try:
                    rec_dict = decode_object(payload)
                except Exception:
                    continue
                rec_id = rec_dict.get("_id")
                if rec_id is None:
                    continue
                version = rec_dict.get("_version", 0)
                if rec_id not in latest_per_id or latest_per_id[rec_id].version < version:
                    latest_per_id[rec_id] = IndexEntry(
                        segment_id=seg_id, offset=offset, version=version
                    )

    with index_file.open("wb") as idx_f:
        for rec_id, entry in latest_per_id.items():
            idx_f.write(struct.pack(
                index_entry_format,
                rec_id, entry.segment_id, entry.offset, entry.version,
            ))
        idx_f.flush()
        os.fsync(idx_f.fileno())

    logger.info(f"Index rebuilt with {len(latest_per_id)} live records")
    return latest_per_id