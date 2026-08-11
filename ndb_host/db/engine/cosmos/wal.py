"""
cosmos/wal.py
=============
Write-Ahead Log (WAL) helpers for NebulonCosmos.

Responsibilities:
  - Writing a single record into the open WAL file handle.
  - Recovering the WAL on startup (rebuild memtable, flush to segment).
"""

import os
import zlib
import struct

from pathlib import Path
from typing import Any, Dict, Optional, Set

from utils.logger import NebulonDBLogger

from db.engine.utils import decode_object


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#  WAL writer
# ==========================================================

def write_wal_record(
    wal_handle: Optional[Any],
    record_bytes: bytes,
    auto_flush: bool = True,
    bytes_since_fsync: int = 0,
    fsync_interval: int = 65536,
) -> int:
    """Append a length-prefixed, CRC-checked record to the open WAL handle.

    The userspace buffer is always flushed, but ``os.fsync`` is only issued
    once *fsync_interval* bytes have accumulated (group commit). Returns the
    updated byte counter so the caller can track it across calls.
    """
    if wal_handle is None:
        return bytes_since_fsync
    crc = zlib.crc32(record_bytes) & 0xFFFFFFFF
    wal_handle.write(struct.pack("<II", crc, len(record_bytes)))
    wal_handle.write(record_bytes)
    if auto_flush:
        wal_handle.flush()
        bytes_since_fsync += 8 + len(record_bytes)
        if bytes_since_fsync >= fsync_interval:
            os.fsync(wal_handle.fileno())
            bytes_since_fsync = 0
    return bytes_since_fsync


# ==========================================================
#  WAL recovery
# ==========================================================

def recover_wal(
    wal_file: Path,
    memtable: Dict[int, bytes],
    deleted: Set[int],
    meta: Dict[str, Any],
    save_meta_fn,  # callable: _save_meta()
) -> None:
    """
    Replay the WAL into *memtable*.

    Parameters
    ----------
    wal_file      : path to the WAL file on disk
    memtable      : the store's in-memory dict {rec_id -> bytes}
    deleted       : the store's deleted-ID set
    meta          : the store's meta dict (tables, global_version, …)
    save_meta_fn  : callable that persists *meta* to disk
    """
    if not wal_file.exists() or wal_file.stat().st_size == 0:
        return

    with wal_file.open("rb") as f:
        while True:
            crc_bytes = f.read(4)
            if not crc_bytes:
                break

            stored_crc = struct.unpack("<I", crc_bytes)[0]

            len_bytes = f.read(4)
            if len(len_bytes) != 4:
                logger.warning("Truncated WAL entry (length missing)")
                break

            length = struct.unpack("<I", len_bytes)[0]
            rec_data = f.read(length)
            if len(rec_data) != length:
                logger.warning("Truncated WAL entry (payload missing)")
                break

            if (zlib.crc32(rec_data) & 0xFFFFFFFF) != stored_crc:
                logger.warning("WAL entry corrupted (CRC mismatch), skipping")
                continue

            try:
                rec_dict = decode_object(rec_data)
            except Exception:
                logger.warning("WAL entry decode failed, skipping")
                continue

            rec_id = rec_dict.get("_id")
            if rec_id is None:
                continue

            if not isinstance(rec_id, int):
                logger.warning(f"WAL entry skipped: _id is {type(rec_id).__name__!r} not int (value={rec_id!r})")
                continue

            # Restore segment counters
            segment = rec_dict.get("_segment") or "_main"
            if isinstance(rec_id, int):
                if (
                    segment not in meta["tables"]
                    or meta["tables"][segment] < rec_id
                ):
                    meta["tables"][segment] = rec_id
            # Restore global version / record id
            version = rec_dict.get("_version", 0)
            if isinstance(version, int) and version > meta["global_version"]:
                meta["global_version"] = version
            if isinstance(rec_id, int) and rec_id > meta.get("global_record_id", 0):
                meta["global_record_id"] = rec_id

            # Restore into memtable
            key = (segment, rec_id)
            memtable[key] = rec_data

            # Restore delete state
            if rec_dict.get("_deleted", False):
                deleted.add(key)
                deleted.add(rec_id)
            else:
                deleted.discard(key)
                deleted.discard(rec_id)

    # Flush recovered records to a new segment
    if memtable:
        logger.info(
            f"Recovered {len(memtable)} records into memtable "
            f"(wal_count={len(memtable)} – no flush)"
        )
        save_meta_fn()

    logger.info(
        f"WAL recovery complete. "
        f"Memtable now has {len(memtable)} live records, "
        f"{len(deleted)} deleted markers"
    )


def checkpoint_wal(
    wal_file: Path,
    wal_handle: Optional[Any],
    memtable: Dict,
) -> Optional[Any]:
    """
    Rewrite the WAL file with only the live memtable rows.

    Drops superseded versions without creating a segment file, so the WAL
    stays the sole durable store below the flush threshold. Returns the
    (possibly re-opened) WAL file handle.
    """
    if not memtable:
        if wal_handle:
            wal_handle.seek(0)
            wal_handle.truncate(0)
            wal_handle.flush()
            os.fsync(wal_handle.fileno())
        return wal_handle

    tmp_path = wal_file.with_name(wal_file.name + ".tmp")
    with tmp_path.open("wb") as f:
        for rec_bytes in memtable.values():
            crc = zlib.crc32(rec_bytes) & 0xFFFFFFFF
            f.write(struct.pack("<I", crc))
            f.write(struct.pack("<I", len(rec_bytes)))
            f.write(rec_bytes)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_path, wal_file)
    if wal_handle:
        try:
            wal_handle.close()
        except Exception:
            pass
    return wal_file.open("a+b")