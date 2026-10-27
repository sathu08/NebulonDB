"""
Power-loss oracle
=================

Pure, dependency-free logic used by the WSL crash harness.

The worker appends two kinds of records to a journal that lives *outside* the
device under test:

    {"ev": "begin", "op": 7, "kind": "insert_many",
     "seqs": [100, 101], "texts": ["b100", "b101"]}
    {"ev": "ack",   "op": 7, "seqs": [100, 101], "ids": [52, 53]}

``begin`` is fsynced *before* the engine call; ``ack`` is fsynced after it
returns.  A SIGKILL can therefore leave at most the last ``begin`` without an
``ack``.  Each op is expanded to its individual WAL records (a bulk op is a
run of records), because a torn write is allowed to persist only a prefix of
an unacknowledged bulk.

The recovered database is represented as ``seq -> text``.  A power loss may
only lose a *suffix* of the WAL record stream, so the recovered state must
equal the state after some prefix of the flattened record list.  For
``batch``/``per_ack`` every acknowledged record must be inside that prefix;
``process`` mode only guarantees that the recovered state is a valid prefix
(acknowledged records may be lost).
"""

from __future__ import annotations

import json

from dataclasses import dataclass, field

from pathlib import Path
from typing import Any, Iterable


# ==========================================================
#        Data Models
# ==========================================================

class JournalError(ValueError):
    """The crash journal is malformed (not a durability failure by itself)."""


@dataclass(frozen=True)
class Record:
    kind: str          # "upsert" | "delete"
    seq: int
    text: str | None = None


@dataclass
class Op:
    index: int
    kind: str
    seqs: list[int]
    texts: list[str | None]
    records: list[Record]
    acked: bool = False
    cumulative_records: int = 0


@dataclass
class Journal:
    ops: list[Op] = field(default_factory=list)
    records: list[Record] = field(default_factory=list)
    ready: bool = False
    done: bool = False
    errors: list[str] = field(default_factory=list)

    @property
    def acked_records(self) -> int:
        """Number of records covered by the last acknowledged operation.

        The worker acknowledges strictly in order, so acknowledged ops form a
        prefix.  If that invariant is broken the journal is rejected.
        """
        last = -1
        for op in self.ops:
            if op.acked:
                last = op.index
        if last < 0:
            return 0
        return self.ops[last].cumulative_records


# ==========================================================
#        Journal Parsing
# ==========================================================

def _parse_line(raw: bytes) -> dict[str, Any] | None:
    if not raw.strip():
        return None
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        raise JournalError(f"journal line is not an object: {raw!r}")
    return obj


def load_journal(path: str | Path) -> Journal:
    """Parse a JSONL crash journal, ignoring an incomplete trailing line."""
    data = Path(path).read_bytes()
    lines = data.split(b"\n")
    # The writer always terminates lines with "\n".  Anything after the last
    # newline is a torn tail and must be ignored, never guessed at.
    complete, tail = lines[:-1], lines[-1]
    journal = Journal()

    events: list[dict[str, Any]] = []
    for raw in complete:
        obj = _parse_line(raw)
        if obj is not None:
            events.append(obj)

    begins: dict[int, dict[str, Any]] = {}
    acks: set[int] = set()

    for ev in events:
        kind = ev.get("ev")
        if kind == "ready":
            journal.ready = True
        elif kind == "done":
            journal.done = True
        elif kind == "error":
            journal.errors.append(str(ev.get("error", "worker error")))
        elif kind == "begin":
            op = ev.get("op")
            if not isinstance(op, int):
                raise JournalError(f"begin without integer op: {ev!r}")
            if op in begins:
                raise JournalError(f"duplicate begin for op {op}")
            begins[op] = ev
        elif kind == "ack":
            op = ev.get("op")
            if not isinstance(op, int):
                raise JournalError(f"ack without integer op: {ev!r}")
            acks.add(op)
        else:
            raise JournalError(f"unknown journal event {kind!r}")

    if sorted(begins) != list(range(len(begins))):
        raise JournalError(
            f"journal op indexes are not contiguous: {sorted(begins)[:10]}"
        )

    cumulative = 0
    for index in range(len(begins)):
        ev = begins[index]
        kind = ev.get("kind")
        seqs = ev.get("seqs")
        texts = ev.get("texts")
        if not isinstance(seqs, list) or not all(isinstance(s, int) for s in seqs):
            raise JournalError(f"op {index} has invalid seqs: {seqs!r}")
        if kind in ("insert", "insert_many", "update"):
            if not isinstance(texts, list) or len(texts) != len(seqs):
                raise JournalError(f"op {index} texts do not match seqs")
            if not all(isinstance(t, str) for t in texts):
                raise JournalError(f"op {index} texts are not strings")
            records = [
                Record("upsert", seq, text) for seq, text in zip(seqs, texts)
            ]
        elif kind == "delete_many":
            records = [Record("delete", seq, None) for seq in seqs]
        else:
            raise JournalError(f"op {index} has unknown kind {kind!r}")

        cumulative += len(records)
        op = Op(
            index=index,
            kind=kind,
            seqs=list(seqs),
            texts=list(texts) if texts is not None else [None] * len(seqs),
            records=records,
            acked=index in acks,
            cumulative_records=cumulative,
        )
        journal.ops.append(op)
        journal.records.extend(records)

    # Acknowledged ops must be a prefix; a hole means the journal itself lost
    # an ack, which makes the "all acked are durable" check unsound.
    acked_indexes = [op.index for op in journal.ops if op.acked]
    if acked_indexes and acked_indexes != list(range(len(acked_indexes))):
        raise JournalError(f"acknowledged ops are not a prefix: {acked_indexes}")

    return journal


# ==========================================================
#        State Evaluation
# ==========================================================

def _apply(state: dict[int, str], record: Record) -> None:
    if record.kind == "upsert":
        state[record.seq] = record.text if record.text is not None else ""
    elif record.kind == "delete":
        state.pop(record.seq, None)
    else:  # pragma: no cover - load_journal validates this
        raise ValueError(f"unknown record kind {record.kind!r}")


def state_after(records: Iterable[Record], count: int) -> dict[int, str]:
    state: dict[int, str] = {}
    for record in list(records)[:count]:
        _apply(state, record)
    return state


def recovered_state(records: Iterable[dict[str, Any]]) -> dict[int, str]:
    """Normalise the verifier's recovered docs into ``seq -> text``."""
    state: dict[int, str] = {}
    for doc in records:
        if not isinstance(doc, dict):
            raise ValueError(f"recovered record is not an object: {doc!r}")
        seq = doc.get("seq")
        text = doc.get("text")
        if not isinstance(seq, int):
            raise ValueError(f"recovered record has invalid seq: {doc!r}")
        if not isinstance(text, str):
            raise ValueError(f"recovered record has invalid text: {doc!r}")
        if seq in state:
            raise ValueError(f"duplicate recovered seq {seq}")
        state[seq] = text
    return state


def _diff(expected: dict[int, str], actual: dict[int, str]) -> dict[str, Any]:
    missing = sorted(set(expected) - set(actual))[:10]
    extra = sorted(set(actual) - set(expected))[:10]
    changed = sorted(
        seq for seq in set(expected) & set(actual)
        if expected[seq] != actual[seq]
    )[:10]
    return {"missing": missing, "extra": extra, "changed": changed}


# ==========================================================
#        Verdict
# ==========================================================

@dataclass
class Verdict:
    ok: bool
    mode: str
    prefix_records: int | None
    attempted_records: int
    acked_records: int
    lost_acked_records: int
    recovered_records: int
    candidate_prefixes: int
    detail: str
    diff: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "mode": self.mode,
            "prefix_records": self.prefix_records,
            "attempted_records": self.attempted_records,
            "acked_records": self.acked_records,
            "lost_acked_records": self.lost_acked_records,
            "recovered_records": self.recovered_records,
            "candidate_prefixes": self.candidate_prefixes,
            "detail": self.detail,
            "diff": self.diff,
        }


# ==========================================================
#        Evaluation
# ==========================================================

def evaluate(
    journal: Journal,
    recovered: Iterable[dict[str, Any]],
    mode: str,
) -> Verdict:
    """Check the recovered state against the crash journal.

    ``mode`` is ``process``, ``batch`` or ``per_ack``.  The prefix property is
    required for every mode.  ``batch``/``per_ack`` additionally require the
    prefix to cover every acknowledged record.
    """
    mode = (mode or "").strip().lower()
    if mode not in ("process", "batch", "per_ack"):
        raise ValueError(f"unknown durability mode {mode!r}")

    if journal.errors:
        return Verdict(
            ok=False, mode=mode, prefix_records=None,
            attempted_records=len(journal.records),
            acked_records=journal.acked_records,
            lost_acked_records=journal.acked_records,
            recovered_records=0, candidate_prefixes=0,
            detail="worker reported an error: " + "; ".join(journal.errors),
        )

    try:
        actual = recovered_state(recovered)
    except ValueError as exc:
        return Verdict(
            ok=False, mode=mode, prefix_records=None,
            attempted_records=len(journal.records),
            acked_records=journal.acked_records,
            lost_acked_records=journal.acked_records,
            recovered_records=0, candidate_prefixes=0,
            detail=f"invalid recovered output: {exc}",
        )

    candidates: list[int] = []
    for k in range(len(journal.records) + 1):
        if state_after(journal.records, k) == actual:
            candidates.append(k)

    if not candidates:
        # Diagnostics: compare against the longest prefix that shares the most
        # records, purely to make failures actionable.
        best_k = 0
        best_score = -1
        for k in range(len(journal.records) + 1):
            expected = state_after(journal.records, k)
            score = len(set(expected) & set(actual))
            score -= len(set(expected) ^ set(actual))
            if score > best_score:
                best_k, best_score = k, score
        expected = state_after(journal.records, best_k)
        return Verdict(
            ok=False, mode=mode, prefix_records=None,
            attempted_records=len(journal.records),
            acked_records=journal.acked_records,
            lost_acked_records=journal.acked_records,
            recovered_records=len(actual), candidate_prefixes=0,
            detail=(
                "recovered state is not a prefix of the acknowledged WAL "
                "stream (non-durable records were lost but the surviving "
                "state is not ordered as a WAL prefix)"
            ),
            diff=_diff(expected, actual),
        )

    k = max(candidates)
    lost = max(0, journal.acked_records - k)
    if mode in ("batch", "per_ack") and k < journal.acked_records:
        return Verdict(
            ok=False, mode=mode, prefix_records=k,
            attempted_records=len(journal.records),
            acked_records=journal.acked_records,
            lost_acked_records=lost,
            recovered_records=len(actual),
            candidate_prefixes=len(candidates),
            detail=(
                f"acknowledged durability violated: {journal.acked_records} "
                f"records were acknowledged but only {k} are recoverable"
            ),
            diff=_diff(state_after(journal.records, journal.acked_records), actual),
        )

    detail = (
        f"recovered state matches WAL prefix {k}/{len(journal.records)}; "
        f"acked={journal.acked_records}, lost_acked={lost}"
    )
    return Verdict(
        ok=True, mode=mode, prefix_records=k,
        attempted_records=len(journal.records),
        acked_records=journal.acked_records,
        lost_acked_records=lost,
        recovered_records=len(actual),
        candidate_prefixes=len(candidates),
        detail=detail,
    )
