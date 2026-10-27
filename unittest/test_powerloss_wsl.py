"""
Opt-in WSL power-loss integration test.

Default test runs skip this.  To execute it, run as root inside WSL with:

    NDB_POWERLOSS=1 .venv/bin/python -m pytest unittest/test_powerloss_wsl.py -q

It drives one SIGKILL/snapshot trial per durability mode on a small matrix.
The heavier 20-trials-per-mode run is the standalone CLI:

    ndb_host/tools/powerloss/run.sh --trials 20
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

pytestmark = pytest.mark.powerloss


def _skip_reason() -> str | None:
    if os.environ.get("NDB_POWERLOSS") != "1":
        return "set NDB_POWERLOSS=1 to run the WSL power-loss integration test"
    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        return "the WSL power-loss harness must run as root"
    for tool in ("dmsetup", "losetup", "mkfs.ext4", "e2fsck", "blockdev"):
        if shutil.which(tool) is None:
            return f"missing required tool: {tool}"
    return None


def test_wsl_powerloss_smoke(tmp_path):
    reason = _skip_reason()
    if reason:
        pytest.skip(reason)

    from ndb_host.tools.powerloss import harness

    rc = harness.main([
        "--modes", "process,batch,per_ack",
        "--workloads", "insert",
        "--trials", "1",
        "--ops", "20",
        "--setup-count", "5",
        "--delay", "0.001",
        "--min-delay", "0.01",
        "--max-delay", "0.05",
        "--run-dir", str(tmp_path / "powerloss"),
        "--no-sysctl",
    ])
    assert rc == 0
