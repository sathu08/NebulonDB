"""
cosmos/metadata.py
==================
Handles atomic file writes, database metadata (meta.ndb), and
segment manifest persistence for NebulonCosmos.
"""

import os

from pathlib import Path
from typing import Any, Dict, List


from utils.logger import NebulonDBLogger

from db.engine.utils import encode_object, decode_object


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#  Atomic file helper
# ==========================================================

def atomic_write(file_path: Path, data_bytes: bytes) -> None:
    """Write *data_bytes* to *file_path* atomically via a temp file."""
    temp_path = file_path.parent / (file_path.name + ".tmp")
    with temp_path.open("wb") as f:
        f.write(data_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp_path, file_path)


# ==========================================================
#  Metadata helpers
# ==========================================================

def default_meta() -> Dict[str, Any]:
    return {
        "tables": {},
        "global_version": 0,
        "last_segment_id": 0,
    }


def load_meta(meta_file: Path) -> Dict[str, Any]:
    if meta_file.exists() and meta_file.stat().st_size > 0:
        with meta_file.open("rb") as f:
            data = f.read()
        try:
            meta = decode_object(data)
        except Exception as e:
            logger.warning(f"Failed to load metadata: {e}")
            meta = default_meta()
    else:
        meta = default_meta()

    # Ensure all required keys exist
    if "tables" not in meta:
        meta["tables"] = {}
    for key in ("global_version", "last_segment_id"):
        if key not in meta:
            meta[key] = 0
    return meta


def save_meta(meta_file: Path, meta: Dict[str, Any]) -> None:
    data = encode_object(meta)
    atomic_write(meta_file, data)

    fd = os.open(str(meta_file.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)


# ==========================================================
#  Manifest helpers
# ==========================================================

def _discover_segments(seg_dir: Path) -> List[str]:
    """Fallback: sort all seg_*.ndb files in seg_dir."""
    return sorted([
        f.name for f in seg_dir.iterdir()
        if f.name.startswith("seg_") and f.name.endswith(".ndb")
    ])


def load_manifest(manifest_file: Path, seg_dir: Path) -> List[str]:
    if manifest_file.exists() and manifest_file.stat().st_size > 0:
        with manifest_file.open("rb") as f:
            data = f.read()
        try:
            return decode_object(data)
        except Exception as e:
            logger.warning(f"Failed to load manifest: {e}")
            return _discover_segments(seg_dir)
    return _discover_segments(seg_dir)


def save_manifest(manifest_file: Path, manifest: List[str]) -> None:
    data = encode_object(manifest)
    atomic_write(manifest_file, data)
    fd = os.open(str(manifest_file.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

def save_manifest_and_meta(manifest_file: Path, meta_file: Path,
                           manifest: List[str], meta: Dict[str, Any]) -> None:
    """Atomically write manifest + meta using a temp directory entry."""
    # Write temp files
    manifest_tmp = manifest_file.parent / (manifest_file.name + ".tmp")
    meta_tmp = meta_file.parent / (meta_file.name + ".tmp")

    with open(manifest_tmp, 'wb') as fm, open(meta_tmp, 'wb') as ft:
        fm.write(encode_object(manifest))
        ft.write(encode_object(meta))
        fm.flush()
        ft.flush()
        os.fsync(fm.fileno())
        os.fsync(ft.fileno())

    # Rename both (not fully atomic, but if one fails, recovery will detect inconsistency)
    os.replace(manifest_tmp, manifest_file)
    os.replace(meta_tmp, meta_file)

    # Durability: fsync the parent directory
    fd = os.open(str(manifest_file.parent), os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)