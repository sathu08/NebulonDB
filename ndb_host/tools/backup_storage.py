"""Automated Storage backup (follow-up to the manual 2026-11-24 backup).

Creates a timestamped ``Storage.backup_<UTC-ts>.tgz`` containing ``Storage/``
+ ``nebulondb.cfg`` and prunes older backups beyond ``--keep`` (default 7).

Safe to run while the server is live: Cosmos is crash-consistent by design,
so a torn tar replays cleanly through the WAL on restore.

Restore:
    1. ``nebulondb stop``
    2. ``mv Storage Storage.damaged_<date>``
    3. ``tar -xzf backups/Storage.backup_<ts>.tgz``   (from NEBULONDB_HOME)
    4. ``nebulondb start`` and verify login

Schedule (macOS cron, daily 03:00):
    0 3 * * * NEBULONDB_HOME=/Users/x344414/Documents/Codebase/NebulonDB /Users/x344414/Documents/Codebase/NebulonDB/.venv/bin/python -m ndb_host.tools.backup_storage --keep 7 >> /Users/x344414/Documents/Codebase/NebulonDB/logs/backup_cron.log 2>&1

Usage:
    NEBULONDB_HOME=/path/to/NebulonDB .venv/bin/python -m ndb_host.tools.backup_storage [--keep 7]
"""

from __future__ import annotations

import argparse
import os
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path


def _home() -> Path:
    home = os.environ.get("NEBULONDB_HOME")
    if not home:
        print("NEBULONDB_HOME is not set.", file=sys.stderr)
        raise SystemExit(2)
    return Path(home)


def backup(home: Path, keep: int = 7) -> Path:
    backup_dir = home / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = backup_dir / f"Storage.backup_{stamp}.tgz"
    with tarfile.open(dest, "w:gz") as tar:
        for name in ("Storage", "nebulondb.cfg"):
            src = home / name
            if src.exists():
                tar.add(src, arcname=name)
            else:
                print(f"warning: {src} missing, skipped")
    # Retention: keep newest `keep`, delete older.
    existing = sorted(backup_dir.glob("Storage.backup_*.tgz"))
    for old in existing[: max(0, len(existing) - keep)]:
        try:
            old.unlink()
            print(f"pruned {old.name}")
        except OSError as e:
            print(f"warning: could not prune {old.name}: {e}")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Back up NebulonDB Storage + config.")
    parser.add_argument("--keep", type=int, default=7, help="Backups to retain (default 7)")
    args = parser.parse_args(argv)
    home = _home()
    dest = backup(home, keep=args.keep)
    size_kb = dest.stat().st_size // 1024
    print(f"backup written: {dest} ({size_kb} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
