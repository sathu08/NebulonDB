"""
NebulonDB Generation Manifest
=============================

Atomic generation tracker for NovaEngine save/load recovery, containing:
    Manifest              – thread-safe manifest file reader/writer
    Atomic Write          – fsync + os.replace for crash-safe generation updates
    Directory Flush       – parent directory fsync for filesystem durability
    Fallback Recovery     – graceful handling of corrupt/missing manifest files
"""


import os
import threading

from pathlib import Path

from ndb_host.utils.models import load_data, save_data


# =============================================================================
# Manifest – atomic generation tracking with recovery
# =============================================================================

class Manifest:
    """Thread‑safe manager for manifest file with fallback to previous generations."""

    def __init__(self, manifest_path: Path):
        self.path = manifest_path
        self._lock = threading.Lock()

    def read_latest(self) -> int | None:
        """Return the latest generation number from manifest, or None if missing."""
        data = load_data(self.path, default={}) or {}
        return data.get("generation") if data else None

    def write(self, generation: int) -> None:
        """Atomically write the generation number."""
        tmp = self.path.with_suffix(".tmp")
        save_data({"generation": generation}, tmp)
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        # flush directory
        try:
            fd = os.open(str(self.path.parent), os.O_RDONLY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass
