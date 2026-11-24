"""Manifest-vs-disk checker (F-10 remainder).

Read-only audit: for every Cosmos store under the storage root, compare the
manifest against the segment files on disk and report:
  * manifest entries whose file is missing  (read_all skips these; F-10)
  * segment files on disk not listed in the manifest (orphans)

Exit status: 0 = consistent, 1 = inconsistencies found, 2 = usage/config error.

Usage:
    NEBULONDB_HOME=/path/to/NebulonDB .venv/bin/python -m ndb_host.tools.check_manifest
    NEBULONDB_HOME=/path/to/NebulonDB .venv/bin/python ndb_host/tools/check_manifest.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _storage_root() -> Path:
    home = os.environ.get("NEBULONDB_HOME")
    if not home:
        print("NEBULONDB_HOME is not set.", file=sys.stderr)
        raise SystemExit(2)
    # Import here so the env var is honoured by the config loader.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from db.ndb_settings import NDBConfig  # noqa: E402

    return NDBConfig().NEBULONDB_STORAGE_PATH


def _iter_cosmos_stores(storage: Path):
    """Yield (corpus, seg_dir, manifest_file) for Cosmos-layout stores."""
    from db.engine.cosmos import metadata as meta_mod  # noqa: E402

    found = []
    if not storage.exists():
        return found
    for corpus_dir in sorted(storage.iterdir()):
        if not corpus_dir.is_dir():
            continue
        for cosmos_dir in sorted(corpus_dir.glob("**/NebulonCosmos")):
            seg_dir = cosmos_dir / "segments"
            manifest_file = cosmos_dir / "manifest.bin"
            if seg_dir.is_dir():
                found.append((corpus_dir.name, seg_dir, manifest_file, meta_mod))
    return found


def main() -> int:
    storage = _storage_root()
    stores = _iter_cosmos_stores(storage)
    print(f"Storage root: {storage} ({len(stores)} Cosmos stores)")
    problems = 0
    for corpus, seg_dir, manifest_file, meta_mod in stores:
        manifest = meta_mod.load_manifest(manifest_file, seg_dir)
        on_disk = sorted(
            f.name for f in seg_dir.iterdir()
            if f.name.startswith("seg_") and f.name.endswith(".ndb")
        )
        missing = [f for f in manifest if f not in on_disk]
        orphans = [f for f in on_disk if f not in manifest]
        status = "OK" if not missing and not orphans else "INCONSISTENT"
        print(f"[{status}] corpus={corpus} manifest={len(manifest)} disk={len(on_disk)}")
        for f in missing:
            print(f"  MISSING ON DISK: {f}")
        for f in orphans:
            print(f"  ORPHAN ON DISK:  {f}")
        if missing or orphans:
            problems += 1
    if problems:
        print(f"{problems} store(s) inconsistent (see F-10/F-08).")
        return 1
    print("All manifests match disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
