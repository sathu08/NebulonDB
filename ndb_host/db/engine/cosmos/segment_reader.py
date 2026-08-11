"""
cosmos/segment_reader.py
========================
Low-level segment reading utilities for NebulonCosmos.

Responsibilities
----------------
- Validate a segment file header (magic, version, count).
- Compute the data start offset inside a segment (accounting for the
  bloom-filter blob that follows the header).
- Read a single record payload at a known byte offset using mmap,
  with an LRU open-file cache for performance.
"""

import zlib
import mmap
import struct

from pathlib import Path
from collections import OrderedDict

from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#  Segment header validation
# ==========================================================

def validate_segment(
    seg_path: Path,
    segment_header_format: str,
    segment_header_size: int,
    segment_magic: int,
    segment_version: int,
) -> tuple[bool, int, bytes | None, bool]:
    """
    Read and validate the fixed-size header of *seg_path*.

    Returns
    -------
    (valid, record_count, bloom_filter_bytes_or_None, is_compressed)
    """
    try:
        with seg_path.open("rb") as f:
            header = f.read(segment_header_size)
            if len(header) != segment_header_size:
                return False, 0, None, False
            magic, version, count, bf_size, compressed = struct.unpack(
                segment_header_format, header
            )
            if magic != segment_magic or version != segment_version:
                return False, 0, None, False
            bf_data = f.read(bf_size) if bf_size > 0 else None
            return True, count, bf_data, bool(compressed)
    except Exception:
        return False, 0, None, False


# ==========================================================
#  Data-start offset inside a segment
# ==========================================================

def get_segment_data_offset(
    seg_path: Path,
    bf_size: int,
    segment_header_size: int,
    record_header_size: int,
    record_header_format: str,
) -> int:
    """
    Detect the byte offset at which record data begins inside *seg_path*.

    The layout is:  [header][bloom-filter blob][records…]
    We probe two candidate offsets and pick the one that appears to hold
    a valid record header (positive compressed length within file bounds).
    """
    candidates = [segment_header_size + bf_size]
    if bf_size < 1024:
        candidates.append(segment_header_size + 1024)

    file_size = seg_path.stat().st_size
    for offset in candidates:
        if offset >= file_size:
            continue
        with seg_path.open("rb") as f:
            f.seek(offset)
            header_data = f.read(record_header_size)
            if len(header_data) != record_header_size:
                continue
            _, comp_len, _ = struct.unpack(record_header_format, header_data)
            if comp_len <= 0:
                continue
            if offset + record_header_size + comp_len > file_size:
                continue
        return offset
    return segment_header_size + bf_size


# ==========================================================
#  Single-record reader (LRU-cached mmap)
# ==========================================================

def read_payload_at_offset(
    seg_id: int,
    offset: int,
    seg_dir: Path,
    segment_cache: "OrderedDict[str, tuple]",
    max_open_segments: int,
    segment_info: dict[int, dict],
    compress_segments: bool,
    record_header_size: int,
    record_header_format: str,
) -> bytes | None:
    """
    Return the raw (decompressed, CRC-verified) payload bytes of one record.

    Uses an LRU open-file + mmap cache (*segment_cache*) for efficiency.
    Returns *None* if the record cannot be read or fails CRC.
    """
    seg_name = f"seg_{seg_id}.ndb"
    seg_path = seg_dir / seg_name

    # ── LRU cache management ──────────────────────────────────
    if seg_name in segment_cache:
        segment_cache.move_to_end(seg_name)
        f, mm = segment_cache[seg_name]
    else:
        if len(segment_cache) >= max_open_segments:
            _, (f_old, mm_old) = segment_cache.popitem(last=False)
            mm_old.close()
            f_old.close()
        f = seg_path.open("rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        segment_cache[seg_name] = (f, mm)

    # ── Read header ───────────────────────────────────────────
    header_end = offset + record_header_size
    if header_end > len(mm):
        return None
    header_data = mm[offset:header_end]
    if len(header_data) != record_header_size:
        return None
    stored_crc, comp_len, _ = struct.unpack(record_header_format, header_data)

    # ── Read payload ──────────────────────────────────────────
    payload_end = header_end + comp_len
    if payload_end > len(mm):
        return None
    comp_data = mm[header_end:payload_end]
    if len(comp_data) != comp_len:
        return None

    # ── Decompress ────────────────────────────────────────────
    seg_info = segment_info.get(seg_id)
    compressed = seg_info["compressed"] if seg_info else compress_segments
    if compressed:
        try:
            payload = zlib.decompress(comp_data)
        except Exception:
            return None
    else:
        payload = comp_data

    # ── CRC check ─────────────────────────────────────────────
    if (zlib.crc32(payload) & 0xFFFFFFFF) != stored_crc:
        return None
    return payload


# ==========================================================
#  Whole-segment scan (for read_all-style batch reads)
# ==========================================================

def scan_segment_payloads(
    seg_path: Path,
    data_offset: int,
    count: int,
    compressed: bool,
    record_header_size: int,
    record_header_format: str,
) -> "list[tuple[int, bytes]]":
    """
    Sequentially walk one segment and return (offset, payload) for every
    valid record. A single mmap + one pass replaces per-record lookups, so
    batch reads avoid the LRU-cache/bloom/index overhead of read_payload_at_offset.
    """
    results: list[tuple[int, bytes]] = []
    try:
        with seg_path.open("rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                pos = data_offset
                for _ in range(count):
                    offset = pos
                    header_end = pos + record_header_size
                    if header_end > len(mm):
                        break
                    stored_crc, comp_len, _ = struct.unpack(
                        record_header_format, mm[pos:header_end]
                    )
                    pos = header_end
                    payload_end = pos + comp_len
                    if payload_end > len(mm):
                        break
                    comp_data = mm[pos:payload_end]
                    pos = payload_end
                    if compressed:
                        try:
                            payload = zlib.decompress(comp_data)
                        except Exception:
                            continue
                    else:
                        payload = comp_data
                    if (zlib.crc32(payload) & 0xFFFFFFFF) != stored_crc:
                        continue
                    results.append((offset, payload))
            finally:
                mm.close()
    except Exception:
        return []
    return results
