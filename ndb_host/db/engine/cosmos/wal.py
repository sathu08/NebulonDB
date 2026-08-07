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
) -> None:
    """Append a length-prefixed, CRC-checked record to the open WAL handle."""
    if wal_handle is None:
        return
    crc = zlib.crc32(record_bytes) & 0xFFFFFFFF
    wal_handle.write(struct.pack("<I", crc))
    wal_handle.write(struct.pack("<I", len(record_bytes)))
    wal_handle.write(record_bytes)
    if auto_flush:
        wal_handle.flush()
        os.fsync(wal_handle.fileno())


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
    Replay the WAL into *memtable*, then flush the memtable to a new segment.

    Parameters
    ----------
    wal_file      : path to the WAL file on disk
    wal_handle    : open file handle (a+b) – truncated after recovery
    memtable      : the store's in-memory dict {rec_id -> bytes}
    deleted       : the store's deleted-ID set
    meta          : the store's meta dict (tables, global_version, …)
    flush_fn      : callable that performs the actual flush to a segment
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
            segment = rec_dict.get("_table") or "_main"
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