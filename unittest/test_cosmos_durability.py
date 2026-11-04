"""Durability / crash-recovery regression tests for NebulonCosmos.

These tests exercise the engine directly (no HTTP server) and target the
write-ahead-log / segment / manifest commit protocol:

* flush must persist the manifest *before* clearing the WAL
* a crash after the manifest commit but before the WAL truncate must replay
  safely without duplicating records
* orphan segments left by the historic ordering bug must be adopted on restart
* a stale ``meta`` file must never reuse a segment id
* a compaction crash before cleanup must not lose records or revive old ones
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent

# Isolate storage from any live NEBULONDB_HOME before importing config/engine.
_TMP_HOME = Path(tempfile.mkdtemp(prefix="ndb_durability_"))
os.environ["NEBULONDB_HOME"] = str(_TMP_HOME)
shutil.copy(PROJECT / "nebulondb.cfg", _TMP_HOME / "nebulondb.cfg")
(_TMP_HOME / "ndb_host").mkdir(exist_ok=True)
shutil.copytree(PROJECT / "ndb_host" / "web_dir", _TMP_HOME / "ndb_host" / "web_dir")
(_TMP_HOME / "logs").mkdir(exist_ok=True)
(_TMP_HOME / "Storage").mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT / "ndb_host"))

from db.engine import NebulonCosmos  # noqa: E402
from db.engine.cosmos import compactor as compactor_mod  # noqa: E402
from db.engine.cosmos import metadata as meta_mod  # noqa: E402
from db.engine.cosmos.store import (  # noqa: E402
    WalDurabilityError,
    resolve_durability_mode,
)

pytestmark = pytest.mark.isolated


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _open(db_dir: Path) -> NebulonCosmos:
    store = NebulonCosmos(db_dir)
    # Keep background daemons from racing the deterministic test steps.
    store.flush_threshold = 10_000_000
    store.flush_interval = 3600.0
    store.compaction_interval = 3600.0
    return store


def _hard_close(store: NebulonCosmos) -> None:
    """Simulate SIGKILL: stop threads and close handles with no final commit."""
    store._flush_stop_event.set()
    store._compaction_stop_event.set()
    for thread in (store._flush_thread, store._compaction_thread):
        if thread is not None and thread.is_alive():
            thread.join(timeout=2)
    if store.wal_handle is not None:
        store.wal_handle.close()
        store.wal_handle = None
    for handle, mm in store.segment_cache.values():
        mm.close()
        handle.close()
    store.segment_cache.clear()


def _insert(store: NebulonCosmos, start: int, count: int, segment: str = "_main") -> None:
    for i in range(start, start + count):
        store.insert(segment, {"text": f"record-{i}", "seq": i})


def _texts(store: NebulonCosmos, segment: str = "_main") -> list[str]:
    return sorted(
        (doc["text"] for doc in store.read_all(segment=segment)),
        key=lambda t: int(t.rsplit("-", 1)[1]),
    )


class _NoTruncateHandle:
    """WAL handle whose truncate() always fails (crash before WAL reset)."""

    def __init__(self, real):
        self._real = real

    def truncate(self, *args, **kwargs):
        raise OSError("simulated crash before WAL truncate")

    def __getattr__(self, name):
        return getattr(self._real, name)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_flush_failure_before_manifest_keeps_wal_and_recovers(tmp_path, monkeypatch):
    """If the manifest commit fails, the WAL must NOT be truncated."""
    store = _open(tmp_path)
    _insert(store, 0, 50)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated manifest write failure")

    monkeypatch.setattr(meta_mod, "save_manifest_and_meta", _boom)

    store.flush()

    assert len(store.memtable) == 50, "memtable must survive a failed commit"
    assert store.wal_file.stat().st_size > 0, "WAL must survive a failed commit"
    _hard_close(store)

    reopened = _open(tmp_path)
    try:
        assert _texts(reopened) == [f"record-{i}" for i in range(50)]
    finally:
        _hard_close(reopened)


def test_crash_after_manifest_before_wal_truncate_is_duplicate_safe(tmp_path):
    """Records present in both the segment and the WAL must not duplicate."""
    store = _open(tmp_path)
    _insert(store, 0, 50)
    store.wal_handle = _NoTruncateHandle(store.wal_handle)

    store.flush()  # segment + manifest succeed; the WAL reset is the failing step
    assert store.wal_file.stat().st_size > 0, "WAL still holds the rows"
    _hard_close(store)

    reopened = _open(tmp_path)
    try:
        docs = reopened.read_all(segment="_main")
        assert len(docs) == 50, "replayed duplicates must be pruned"
        assert _texts(reopened) == [f"record-{i}" for i in range(50)]
        assert reopened.wal_file.stat().st_size == 0, "persisted WAL must be reset"
    finally:
        _hard_close(reopened)


def test_orphan_segment_adopted_after_legacy_crash(tmp_path):
    """Reproduce the original bug state: segment on disk, empty WAL, stale manifest."""
    store = _open(tmp_path)
    _insert(store, 0, 25)
    store.flush()
    _hard_close(store)

    # Legacy crash fingerprint: manifest doesn't list the segment, WAL is empty.
    meta_mod.save_manifest(store.manifest_file, [])
    store.wal_file.write_bytes(b"")

    reopened = _open(tmp_path)
    try:
        assert _texts(reopened) == [f"record-{i}" for i in range(25)]
        assert len(reopened.manifest) == 1
    finally:
        _hard_close(reopened)


def test_stale_meta_never_reuses_segment_id(tmp_path):
    """A stale last_segment_id must be reconciled, not reused/overwritten."""
    store = _open(tmp_path)
    _insert(store, 0, 10)
    store.flush()
    _hard_close(store)

    meta = meta_mod.load_meta(store.meta_file)
    meta["last_segment_id"] = 0
    meta_mod.save_meta(store.meta_file, meta)

    reopened = _open(tmp_path)
    try:
        _insert(reopened, 10, 5)
        reopened.flush()
        segments = sorted(p.name for p in reopened.seg_dir.glob("seg_*.ndb"))
        assert len(segments) == 2, f"expected two segments, got {segments}"
    finally:
        _hard_close(reopened)

    final = _open(tmp_path)
    try:
        assert _texts(final) == [f"record-{i}" for i in range(15)]
    finally:
        _hard_close(final)


def test_compaction_crash_before_cleanup_keeps_all_records(tmp_path, monkeypatch):
    """Manifest + index are committed before old segments are deleted."""
    store = _open(tmp_path)
    for batch in range(4):
        _insert(store, batch * 10, 10)
        store.flush()

    def _crash_cleanup(*args, **kwargs):
        raise RuntimeError("simulated crash before old-segment cleanup")

    monkeypatch.setattr(compactor_mod, "cleanup_segments", _crash_cleanup)

    with pytest.raises(RuntimeError):
        store._compact(["seg_1.ndb", "seg_2.ndb"])

    _hard_close(store)

    reopened = _open(tmp_path)
    try:
        assert _texts(reopened) == [f"record-{i}" for i in range(40)]
        # Old files may remain on disk but must not be adopted/resurrected.
        on_disk = sorted(p.name for p in reopened.seg_dir.glob("seg_*.ndb"))
        assert "seg_1.ndb" in on_disk and "seg_2.ndb" in on_disk
        assert len(reopened.read_all(segment="_main")) == 40
    finally:
        _hard_close(reopened)


# ---------------------------------------------------------------------------
# WAL durability modes (process / batch / per_ack)
# ---------------------------------------------------------------------------

def _count_fsyncs(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = os.fsync

    def _counting(fd):
        calls.append(fd)
        return real(fd)

    monkeypatch.setattr(os, "fsync", _counting)
    return calls


def _set_mode(store: NebulonCosmos, mode: str) -> None:
    store.durability_mode = mode
    # Prevent the size-based group commit from firing during these checks.
    store.wal_fsync_interval = 10**9
    store.wal_auto_flush = True


def test_resolve_durability_mode_validates():
    assert resolve_durability_mode(None) == "process"
    assert resolve_durability_mode(" BATCH ") == "batch"
    with pytest.raises(ValueError):
        resolve_durability_mode("eventual")


def test_process_mode_does_not_fsync_each_write(tmp_path, monkeypatch):
    store = _open(tmp_path)
    _set_mode(store, "process")
    calls = _count_fsyncs(monkeypatch)

    _insert(store, 0, 3)

    assert len(calls) == 0
    assert store.wal_fsync_count == 0
    store.close()


def test_per_ack_mode_fsyncs_every_record(tmp_path, monkeypatch):
    store = _open(tmp_path)
    _set_mode(store, "per_ack")
    calls = _count_fsyncs(monkeypatch)

    _insert(store, 0, 3)

    assert len(calls) == 3
    assert store.wal_fsync_count == 3
    store.close()


def test_batch_mode_fsyncs_once_per_bulk_request(tmp_path, monkeypatch):
    store = _open(tmp_path)
    _set_mode(store, "batch")
    calls = _count_fsyncs(monkeypatch)

    store.insert_many("_main", [{"text": f"bulk-{i}"} for i in range(5)])

    assert len(calls) == 1
    assert store.wal_fsync_count == 1
    assert len(store.read_all(segment="_main")) == 5
    store.close()


def test_batch_mode_flushes_even_when_auto_flush_disabled(tmp_path, monkeypatch):
    store = _open(tmp_path)
    _set_mode(store, "batch")
    store.wal_auto_flush = False
    calls = _count_fsyncs(monkeypatch)

    store.insert("_main", {"text": "durable"})

    assert len(calls) == 1
    store.close()


def test_batch_mode_large_write_fsyncs_once(tmp_path, monkeypatch):
    store = _open(tmp_path)
    _set_mode(store, "batch")
    # Would normally trigger an implicit fsync; batch mode must suppress it.
    store.wal_fsync_interval = 1
    calls = _count_fsyncs(monkeypatch)

    store.insert_many("_main", [{"text": "x" * 1000} for _ in range(10)])

    assert len(calls) == 1
    store.close()


def test_per_ack_fsync_failure_does_not_acknowledge(tmp_path, monkeypatch):
    store = _open(tmp_path)
    _set_mode(store, "per_ack")
    real_fsync = os.fsync

    def _fail(fd):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "fsync", _fail)

    with pytest.raises(WalDurabilityError):
        store.insert("_main", {"text": "unacked"})

    # Not visible in-process: the record was written to the WAL but the
    # acknowledgement failed before the memtable was updated.
    assert store.read_all(segment="_main") == []
    assert store.wal_stats()["failed"] is True

    monkeypatch.setattr(os, "fsync", real_fsync)
    # The store must stay read-only until restart after a WAL failure.
    with pytest.raises(WalDurabilityError):
        store.insert("_main", {"text": "after-failure"})
    store.close()


def test_batch_fsync_failure_does_not_acknowledge_bulk(tmp_path, monkeypatch):
    store = _open(tmp_path)
    _set_mode(store, "batch")
    real_fsync = os.fsync

    def _fail(fd):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "fsync", _fail)

    with pytest.raises(WalDurabilityError):
        store.insert_many("_main", [{"text": f"bulk-{i}"} for i in range(5)])

    assert store.read_all(segment="_main") == []

    monkeypatch.setattr(os, "fsync", real_fsync)
    store.close()


@pytest.mark.parametrize("mode", ["process", "batch", "per_ack"])
def test_acknowledged_writes_recover_after_sigkill_in_all_modes(tmp_path, mode):
    store = _open(tmp_path)
    _set_mode(store, mode)
    _insert(store, 0, 20)
    _hard_close(store)

    reopened = _open(tmp_path)
    try:
        assert _texts(reopened) == [f"record-{i}" for i in range(20)]
    finally:
        _hard_close(reopened)


def _manifest(store) -> list:
    return list(store.manifest)


def test_dead_prefix_merge_converges_and_preserves_live(tmp_path):
    """A merge set with no live records must shrink the manifest.

    Regression test for the compaction hot-loop: the old no-live path kept
    every file listed, so the trigger condition never cleared.
    """
    store = _open(tmp_path)
    try:
        keep_ids = [store.insert("_main", {"text": f"keep-{i}"})
                    for i in range(3)]
        store.flush()
        live = _manifest(store)
        tmp_ids = []
        for _ in range(3):
            batch = [store.insert("_main", {"text": "tmp"}) for _ in range(3)]
            store.flush()
            for rid in batch:
                assert store.delete("_main", rid) == rid
            store.flush()
            tmp_ids.extend(batch)
        assert len(_manifest(store)) > len(live)
        dead = [f for f in _manifest(store) if f not in live]
        assert len(dead) >= 2
        before = len(_manifest(store))
        store._compact(dead)
        assert len(_manifest(store)) < before
        for fname in dead:
            assert not (store.seg_dir / fname).exists()
        for i, rid in enumerate(keep_ids):
            doc = store.get(rid, "_main")
            assert doc is not None and doc["text"] == f"keep-{i}"
        for rid in tmp_ids:
            assert store.get(rid, "_main") is None
        # A second identical merge is a safe no-op (files already gone).
        store._compact(dead)
        assert len(_manifest(store)) < before
    finally:
        _hard_close(store)


def test_merge_carries_tombstones_without_resurrection(tmp_path):
    """Deleting a tombstone's file must never unmask an older version."""
    store = _open(tmp_path)
    try:
        rid = store.insert("_main", {"text": "v1"})
        store.flush()
        assert store.delete("_main", rid) == rid
        store.flush()
        assert store.get(rid, "_main") is None
        store._compact(_manifest(store)[-1:])
        assert store.get(rid, "_main") is None
    finally:
        _hard_close(store)


def test_merge_with_missing_files_converges(tmp_path):
    """Lost segment files must drop out of the manifest, not crash merges."""
    store = _open(tmp_path)
    try:
        store.insert("_main", {"text": "x"})
        store.flush()
        store.insert("_main", {"text": "y"})
        store.flush()
        doomed = _manifest(store)[-2:]
        assert len(doomed) == 2
        for fname in doomed:
            (store.seg_dir / fname).unlink()
        store._compact(doomed)
        for fname in doomed:
            assert fname not in _manifest(store)
    finally:
        _hard_close(store)


def test_read_payload_missing_file_returns_none(tmp_path):
    """A vanished segment file reads as a miss, never raises (stale-tolerant)."""
    from db.engine.cosmos import segment_reader as reader_mod
    from db.engine.utils.constants import (
        RECORD_HEADER_FORMAT,
        RECORD_HEADER_SIZE,
    )

    assert reader_mod.read_payload_at_offset(
        4242, 0, tmp_path, {}, 16, {}, False,
        RECORD_HEADER_SIZE, RECORD_HEADER_FORMAT) is None


def test_get_after_segment_loss_returns_none(tmp_path):
    """Reads racing a merge cleanup get None, not a 500."""
    store = _open(tmp_path)
    try:
        rid = store.insert("_main", {"text": "x"})
        store.flush()
        assert store.get(rid, "_main") is not None
        seg_dir = store.seg_dir
    finally:
        _hard_close(store)
    for seg_file in seg_dir.glob("seg_*.ndb"):
        seg_file.unlink()
    reopened = _open(tmp_path)
    try:
        assert reopened.get(rid, "_main") is None
    finally:
        _hard_close(reopened)


def test_background_daemon_survives_merge_failure():
    """One bad merge must not kill the compaction daemon silently."""
    import threading

    from db.engine.cosmos import compactor as compactor_mod

    calls = []

    def _boom():
        calls.append(1)
        raise RuntimeError("simulated merge failure")

    stop = threading.Event()
    worker = threading.Thread(
        target=compactor_mod.background_compaction_loop,
        args=(stop, 0.01, ["seg_1.ndb"] * 11, 10, _boom), daemon=True)
    worker.start()
    stop.wait(0.2)
    stop.set()
    worker.join(timeout=5)
    assert calls, "daemon never fired"
    assert not worker.is_alive()


def test_corpus_name_for_layouts(tmp_path):
    from db.engine.cosmos.compactor import corpus_name_for

    assert corpus_name_for(tmp_path / "mind_semantic" / "NebulonCosmos"
                           / "segments") == "mind_semantic"
    assert corpus_name_for(tmp_path / "mind_semantic") == "mind_semantic"
    assert corpus_name_for(tmp_path / "nak_tasks") == "nak_tasks"
