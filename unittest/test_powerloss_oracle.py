"""Unit tests for the power-loss prefix oracle (no root / no devices needed)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

from ndb_host.tools.powerloss.oracle import (  # noqa: E402
    JournalError,
    evaluate,
    load_journal,
)

pytestmark = pytest.mark.isolated


def _write_journal(path: Path, events: list[dict], tail: bytes = b"") -> Path:
    data = b"".join(
        (json.dumps(ev, separators=(",", ":")) + "\n").encode() for ev in events
    )
    path.write_bytes(data + tail)
    return path


def _begin(op: int, kind: str, seqs: list[int], texts: list[str] | None = None):
    ev = {"ev": "begin", "op": op, "kind": kind, "seqs": seqs}
    if texts is not None:
        ev["texts"] = texts
    return ev


def _ack(op: int, kind: str, seqs: list[int], texts: list[str] | None = None):
    ev = {"ev": "ack", "op": op, "kind": kind, "seqs": seqs}
    if texts is not None:
        ev["texts"] = texts
    return ev


def _docs(mapping: dict[int, str]) -> list[dict]:
    return [{"seq": seq, "text": text} for seq, text in mapping.items()]


def _journal(tmp_path: Path, events: list[dict], tail: bytes = b""):
    return load_journal(_write_journal(tmp_path / "journal.jsonl", events, tail))


def test_process_allows_losing_acknowledged_tail(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert", [1], ["a"]), _ack(0, "insert", [1], ["a"]),
        _begin(1, "insert", [2], ["b"]), _ack(1, "insert", [2], ["b"]),
    ])
    verdict = evaluate(journal, _docs({1: "a"}), "process")
    assert verdict.ok
    assert verdict.prefix_records == 1
    assert verdict.acked_records == 2
    assert verdict.lost_acked_records == 1


def test_batch_rejects_losing_acknowledged_tail(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert", [1], ["a"]), _ack(0, "insert", [1], ["a"]),
        _begin(1, "insert", [2], ["b"]), _ack(1, "insert", [2], ["b"]),
    ])
    verdict = evaluate(journal, _docs({1: "a"}), "batch")
    assert not verdict.ok
    assert "acknowledged durability violated" in verdict.detail


def test_batch_accepts_full_acknowledged_set(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert", [1], ["a"]), _ack(0, "insert", [1], ["a"]),
        _begin(1, "insert", [2], ["b"]), _ack(1, "insert", [2], ["b"]),
    ])
    verdict = evaluate(journal, _docs({1: "a", 2: "b"}), "batch")
    assert verdict.ok
    assert verdict.prefix_records == 2
    assert verdict.lost_acked_records == 0


def test_partial_unacked_bulk_is_allowed(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert_many", [1, 2, 3], ["a", "b", "c"]),
        _ack(0, "insert_many", [1, 2, 3], ["a", "b", "c"]),
        _begin(1, "insert_many", [4, 5], ["d", "e"]),
    ])
    # Torn write: records 1-4 made it, record 5 did not.
    verdict = evaluate(journal, _docs({1: "a", 2: "b", 3: "c", 4: "d"}), "batch")
    assert verdict.ok
    assert verdict.prefix_records == 4
    assert verdict.acked_records == 3


def test_partial_acked_bulk_is_rejected(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert_many", [1, 2, 3], ["a", "b", "c"]),
        _ack(0, "insert_many", [1, 2, 3], ["a", "b", "c"]),
    ])
    verdict = evaluate(journal, _docs({1: "a", 2: "b"}), "per_ack")
    assert not verdict.ok
    assert "acknowledged durability violated" in verdict.detail


def test_non_prefix_survivors_are_rejected(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert", [1], ["a"]), _ack(0, "insert", [1], ["a"]),
        _begin(1, "insert", [2], ["b"]), _ack(1, "insert", [2], ["b"]),
        _begin(2, "insert", [3], ["c"]), _ack(2, "insert", [3], ["c"]),
    ])
    # A gap (record 2 lost, record 3 present) cannot be explained by a WAL
    # suffix loss and therefore indicates a real ordering/corruption bug.
    verdict = evaluate(journal, _docs({1: "a", 3: "c"}), "process")
    assert not verdict.ok
    assert "not a prefix" in verdict.detail


def test_update_and_delete_replay(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert", [1], ["v1"]), _ack(0, "insert", [1], ["v1"]),
        _begin(1, "update", [1], ["v2"]), _ack(1, "update", [1], ["v2"]),
        _begin(2, "delete_many", [1]), _ack(2, "delete_many", [1]),
    ])
    verdict = evaluate(journal, [], "per_ack")
    assert verdict.ok
    assert verdict.prefix_records == 3


def test_unacked_delete_partial_is_allowed(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert_many", [1, 2, 3], ["a", "b", "c"]),
        _ack(0, "insert_many", [1, 2, 3], ["a", "b", "c"]),
        _begin(1, "delete_many", [1, 2, 3]),
    ])
    # Only one tombstone survived; the bulk delete was never acknowledged.
    verdict = evaluate(journal, _docs({2: "b", 3: "c"}), "batch")
    assert verdict.ok
    assert verdict.prefix_records == 4


def test_incomplete_trailing_journal_line_is_ignored(tmp_path):
    journal = _journal(
        tmp_path,
        [_begin(0, "insert", [1], ["a"]), _ack(0, "insert", [1], ["a"])],
        tail=b'{"ev":"begin","op":1,"kind":"insert","seqs":[2]',
    )
    assert len(journal.ops) == 1
    assert evaluate(journal, _docs({1: "a"}), "per_ack").ok


def test_ack_hole_is_rejected(tmp_path):
    with pytest.raises(JournalError, match="not a prefix"):
        _journal(tmp_path, [
            _begin(0, "insert", [1], ["a"]), _ack(0, "insert", [1], ["a"]),
            _begin(1, "insert", [2], ["b"]),
            _begin(2, "insert", [3], ["c"]), _ack(2, "insert", [3], ["c"]),
        ])


def test_non_contiguous_ops_are_rejected(tmp_path):
    with pytest.raises(JournalError, match="not contiguous"):
        _journal(tmp_path, [_begin(1, "insert", [1], ["a"])])


def test_duplicate_recovered_seq_is_rejected(tmp_path):
    journal = _journal(tmp_path, [
        _begin(0, "insert", [1], ["a"]), _ack(0, "insert", [1], ["a"]),
    ])
    verdict = evaluate(journal, _docs({1: "a"}) + _docs({1: "a"}), "process")
    assert not verdict.ok
    assert "duplicate recovered seq" in verdict.detail


def test_worker_error_fails_trial(tmp_path):
    journal = _journal(tmp_path, [
        {"ev": "error", "op": 0, "error": "WalDurabilityError: boom"},
    ])
    verdict = evaluate(journal, [], "per_ack")
    assert not verdict.ok
    assert "worker reported an error" in verdict.detail
