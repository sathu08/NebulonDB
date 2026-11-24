"""macOS crash-recovery (SIGKILL) tests — the Mac port of test_powerloss_wsl.py.

The WSL harness needs root + device-mapper/loop/ext4 to simulate a real
power cut at the block layer, none of which exists on macOS. This module
covers what *can* be tested on a Mac: OS-level crash consistency.

Method: a child process performs real inserts (real ``write``/``fsync``
syscalls through the real durability path) while the parent SIGKILLs it
mid-run, then the parent reopens the store and asserts every ACKed record
survived. ``kill -9`` does not drop the page cache, so this validates
crash recovery — it cannot validate true plug-pull loss (unflushed
``batch``/``process`` bytes), which still needs the WSL harness.

Modes covered: ``per_ack`` (every ACK durable) and ``process`` (WAL replay
recovers everything after a kill). ``batch`` is excluded — its guarantee
is prefix-up-to-last-fsync, which only a block-layer power cut can test.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "ndb_host"))

pytestmark = [pytest.mark.powerloss, pytest.mark.isolated]

HELPER = Path(__file__).resolve().parent / "_powerloss_mac_child.py"

requires_posix = pytest.mark.skipif(
    os.name != "posix", reason="SIGKILL crash test needs a POSIX platform"
)


def _make_home(base: Path, mode: str) -> Path:
    home = base / f"home_{mode}"
    home.mkdir(parents=True, exist_ok=True)
    text = (PROJECT / "nebulondb.cfg").read_text()
    text = re.sub(r"(?m)^durability_mode\s*=.*$", f"durability_mode = {mode}", text)
    (home / "nebulondb.cfg").write_text(text)
    (home / "ndb_host").mkdir(exist_ok=True)
    shutil.copytree(PROJECT / "ndb_host" / "web_dir", home / "ndb_host" / "web_dir",
                    dirs_exist_ok=True)
    (home / "logs").mkdir(exist_ok=True)
    (home / "Storage").mkdir(exist_ok=True)
    return home


def _run_kill_trial(tmp_path: Path, mode: str, kill_after: int = 25,
                    total: int = 60, timeout: float = 120.0) -> None:
    home = _make_home(tmp_path, mode)
    db_dir = tmp_path / f"db_{mode}"
    db_dir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.Popen(
        [sys.executable, str(HELPER), str(home), str(db_dir), mode, str(total)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1,
    )
    acked: list[int] = []
    try:
        deadline = time.monotonic() + timeout
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if line.startswith("ACK "):
                try:
                    acked.append(int(line.split()[1]))
                except ValueError:
                    continue
                if len(acked) >= kill_after:
                    break
            if time.monotonic() > deadline:
                raise TimeoutError(f"child produced only {len(acked)} ACKs in {timeout}s")
        assert len(acked) >= kill_after, f"child exited early with {len(acked)} ACKs"
    finally:
        # The actual power-cut simulation: unconditional SIGKILL, no cleanup.
        try:
            proc.kill()
        except OSError:
            pass
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=30)

    assert proc.returncode is not None and proc.returncode != 0, \
        "child should have been SIGKILLed, not exited cleanly"

    # Reopen in-process (WAL replay) and check every ACKed write survived.
    os.environ["NEBULONDB_HOME"] = str(home)
    from db.engine import NebulonCosmos

    store = NebulonCosmos(db_dir)
    store.flush_threshold = 10_000_000
    store.flush_interval = 3600.0
    store.compaction_interval = 3600.0
    try:
        got: dict[int, str] = {}
        for doc in store.read_all(segment="_main"):
            seq = doc.get("seq")
            if isinstance(seq, int):
                got[seq] = doc.get("text", "")
        missing = [s for s in acked if s not in got]
        assert not missing, f"mode={mode}: lost ACKed writes after SIGKILL: {missing[:10]}"
        wrong = [s for s in acked if got.get(s) != f"mackill-{s}"]
        assert not wrong, f"mode={mode}: corrupted records: {wrong[:10]}"
    finally:
        store._flush_stop_event.set()
        store._compaction_stop_event.set()
        for thread in (store._flush_thread, store._compaction_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=5)
        if store.wal_handle is not None:
            store.wal_handle.close()
            store.wal_handle = None
        for handle, mm in store.segment_cache.values():
            mm.close()
            handle.close()
        store.segment_cache.clear()


@requires_posix
def test_mac_sigkill_recovery_per_ack(tmp_path):
    """SIGKILL mid-write, per_ack mode: every ACKed record must survive."""
    _run_kill_trial(tmp_path, "per_ack")


@requires_posix
def test_mac_sigkill_recovery_process(tmp_path):
    """SIGKILL mid-write, process mode: WAL replay must recover all ACKs."""
    _run_kill_trial(tmp_path, "process")
