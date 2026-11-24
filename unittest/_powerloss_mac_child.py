"""macOS crash-recovery worker for test_powerloss_mac.py.

Usage:
    python _powerloss_mac_child.py <home_dir> <db_dir> <mode> <total>

Opens a NebulonCosmos store under <db_dir> (durability <mode> is baked into
<home_dir>/nebulondb.cfg by the parent), inserts <total> records one by one,
and prints ``ACK <seq>`` (flushed) after every insert. The parent SIGKILLs
this process mid-run and then asserts every ACKed seq is recoverable.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def main() -> int:
    home_dir, db_dir, mode, total = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
    os.environ["NEBULONDB_HOME"] = home_dir

    project = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project / "ndb_host"))

    from db.engine import NebulonCosmos

    store = NebulonCosmos(Path(db_dir))
    assert store.durability_mode == mode, f"{store.durability_mode} != {mode}"
    # Keep background threads quiet; recovery must work without their help.
    store.flush_threshold = 10_000_000
    store.flush_interval = 3600.0
    store.compaction_interval = 3600.0
    try:
        for seq in range(total):
            store.insert("_main", {"text": f"mackill-{seq}", "seq": seq})
            print(f"ACK {seq}", flush=True)
            time.sleep(0.005)
    finally:
        # No graceful close: the parent kills us with SIGKILL. If we ever
        # finish all inserts (parent too slow), exit 0 — the parent treats
        # that as "killed too late" and fails the trial.
        os._exit(0)
    return 0  # unreachable


if __name__ == "__main__":
    main()
