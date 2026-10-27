"""
Crash worker for the WSL power-loss harness.

Runs a deterministic workload against ``NebulonCosmos`` on the filesystem
under test and appends an fsynced begin/ack journal to a file *outside* that
filesystem.  The harness SIGKILLs this process and then snapshots the block
device; the journal tells the oracle exactly which operations were attempted
and acknowledged.

This process never calls ``close()``: doing so would flush metadata and hide
the crash window we are trying to exercise.
"""

from __future__ import annotations

import os
import time
import json

import random
import argparse

from typing import Any
from pathlib import Path

from ndb_host.db.engine import NebulonCosmos
from ndb_host.db.ndb_settings import NDBConfig

from ndb_host.db.engine.cosmos.store import resolve_durability_mode


# ==========================================================
#        Configuration
# ==========================================================

CONFIG = NDBConfig()

MODES = CONFIG.POWERLOSS_MODES
WORKLOADS = CONFIG.POWERLOSS_WORKLOADS
SEGMENT = CONFIG.POWERLOSS_SEGMENT
SENTINEL_SEQ = CONFIG.POWERLOSS_SENTINEL_SEQ
DISABLE_BACKGROUND = CONFIG.POWERLOSS_DISABLE_BACKGROUND

DEFAULT_SEED = CONFIG.POWERLOSS_WORKER_DEFAULT_SEED
DEFAULT_OPS = CONFIG.POWERLOSS_WORKER_DEFAULT_OPS
DEFAULT_SETUP_COUNT = CONFIG.POWERLOSS_WORKER_DEFAULT_SETUP_COUNT
DEFAULT_DELAY = CONFIG.POWERLOSS_WORKER_DEFAULT_DELAY
DEFAULT_WAL_FSYNC_INTERVAL = CONFIG.POWERLOSS_WORKER_DEFAULT_WAL_FSYNC_INTERVAL
IDLE_SLEEP = CONFIG.POWERLOSS_WORKER_IDLE_SLEEP

BULK_INSERT_MIN = CONFIG.POWERLOSS_WORKER_BULK_INSERT_MIN
BULK_INSERT_MAX = CONFIG.POWERLOSS_WORKER_BULK_INSERT_MAX
MIXED_BULK_MIN = CONFIG.POWERLOSS_WORKER_MIXED_BULK_MIN
MIXED_BULK_MAX = CONFIG.POWERLOSS_WORKER_MIXED_BULK_MAX
DELETE_MIN = CONFIG.POWERLOSS_WORKER_DELETE_MIN
DELETE_MAX = CONFIG.POWERLOSS_WORKER_DELETE_MAX

INSERT_ROLL = CONFIG.POWERLOSS_WORKER_INSERT_ROLL
BULK_ROLL = CONFIG.POWERLOSS_WORKER_BULK_ROLL
UPDATE_ROLL = CONFIG.POWERLOSS_WORKER_UPDATE_ROLL


# ==========================================================
#        Journal
# ==========================================================

class Journal:
    def __init__(self, path: Path) -> None:
        self.fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)

    def append(self, event: dict[str, Any]) -> None:
        payload = (json.dumps(event, separators=(",", ":")) + "\n").encode()
        view = memoryview(payload)
        while view:
            written = os.write(self.fd, view)
            view = view[written:]
        os.fsync(self.fd)


# ==========================================================
#        CLI Arguments
# ==========================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-dir", required=True)
    p.add_argument("--journal", required=True)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--workload", default="mixed", choices=WORKLOADS)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--ops", type=int, default=DEFAULT_OPS)
    p.add_argument("--setup-count", type=int,
                   default=DEFAULT_SETUP_COUNT)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    p.add_argument("--wal-fsync-interval", type=int,
                   default=DEFAULT_WAL_FSYNC_INTERVAL,
                   help="process-mode group-commit threshold in WAL bytes")
    return p.parse_args(argv)


# ==========================================================
#        Store Setup
# ==========================================================

def _open_store(db_dir: Path, wal_fsync_interval: int):
    store = NebulonCosmos(db_dir)
    # Neutralise every background path that could persist the memtable before
    # the harness gets a chance to kill us.
    store.flush_threshold = DISABLE_BACKGROUND
    store.flush_interval = DISABLE_BACKGROUND
    store.flush_size_threshold = DISABLE_BACKGROUND
    store.compaction_interval = DISABLE_BACKGROUND
    store.max_segments_before_compact = DISABLE_BACKGROUND
    store.max_memtable_bytes = DISABLE_BACKGROUND
    store.wal_fsync_interval = wal_fsync_interval
    store.wal_auto_flush = True
    return store


# ==========================================================
#        Journal Helpers
# ==========================================================

def _journal_begin(journal: Journal, op: int, kind: str, seqs: list[int],
                   texts: list[str] | None = None,
                   ids: list[int] | None = None) -> None:
    event: dict[str, Any] = {"ev": "begin", "op": op, "kind": kind, "seqs": seqs}
    if texts is not None:
        event["texts"] = texts
    if ids is not None:
        event["ids"] = ids
    journal.append(event)


def _journal_ack(journal: Journal, op: int, kind: str, seqs: list[int],
                 texts: list[str] | None = None,
                 ids: list[int] | None = None) -> None:
    event: dict[str, Any] = {"ev": "ack", "op": op, "kind": kind, "seqs": seqs}
    if texts is not None:
        event["texts"] = texts
    if ids is not None:
        event["ids"] = ids
    journal.append(event)


# ==========================================================
#        Workload
# ==========================================================

def run(args: argparse.Namespace) -> None:
    mode = resolve_durability_mode(args.mode)
    db_dir = Path(args.db_dir)
    journal = Journal(Path(args.journal))
    rng = random.Random(args.seed)

    store = _open_store(db_dir, args.wal_fsync_interval)
    store.durability_mode = mode

    segment = SEGMENT
    live: dict[int, int] = {}
    next_seq = 0
    op_index = 0

    def do_insert_many(count: int, prefix: str) -> None:
        nonlocal next_seq, op_index
        seqs = list(range(next_seq, next_seq + count))
        next_seq += count
        texts = [f"{prefix}{seq}" for seq in seqs]
        docs = [{"seq": seq, "text": text} for seq, text in zip(seqs, texts)]
        _journal_begin(journal, op_index, "insert_many", seqs, texts)
        ids = store.insert_many(segment, docs)
        for seq, rid in zip(seqs, ids):
            live[seq] = rid
        _journal_ack(journal, op_index, "insert_many", seqs, texts, ids)
        op_index += 1

    def do_insert() -> None:
        nonlocal next_seq, op_index
        seq = next_seq
        next_seq += 1
        text = f"i{seq}"
        _journal_begin(journal, op_index, "insert", [seq], [text])
        rid = store.insert(segment, {"seq": seq, "text": text})
        live[seq] = rid
        _journal_ack(journal, op_index, "insert", [seq], [text], [rid])
        op_index += 1

    def mutable_seqs() -> list[int]:
        # The sentinel seq is inserted first and never updated or deleted, so
        # every replay prefix has a distinguishable state.
        return [seq for seq in live if seq != SENTINEL_SEQ]

    def do_update() -> None:
        nonlocal op_index
        seq = rng.choice(mutable_seqs())
        rid = live[seq]
        text = f"u{seq}-{op_index}"
        _journal_begin(journal, op_index, "update", [seq], [text], [rid])
        store.update(segment, {"id": rid, "seq": seq, "text": text})
        _journal_ack(journal, op_index, "update", [seq], [text], [rid])
        op_index += 1

    def do_delete_many() -> None:
        nonlocal op_index
        candidates = mutable_seqs()
        count = min(
            len(candidates),
            rng.randint(DELETE_MIN, DELETE_MAX),
        )
        seqs = rng.sample(candidates, count)
        ids = [live[seq] for seq in seqs]
        _journal_begin(journal, op_index, "delete_many", seqs, None, ids)
        store.delete_many(segment, ids)
        for seq in seqs:
            live.pop(seq, None)
        _journal_ack(journal, op_index, "delete_many", seqs, None, ids)
        op_index += 1

    try:
        if args.setup_count:
            do_insert_many(args.setup_count, "s")
        journal.append({"ev": "ready", "op": op_index})

        for _ in range(args.ops):
            if args.workload == "insert":
                do_insert()
            elif args.workload == "bulk":
                do_insert_many(
                    rng.randint(
                        BULK_INSERT_MIN,
                        BULK_INSERT_MAX,
                    ),
                    "b",
                )
            else:
                roll = rng.random()
                can_mutate = len(mutable_seqs()) > 0
                if not can_mutate or roll < INSERT_ROLL:
                    do_insert()
                elif roll < BULK_ROLL:
                    do_insert_many(
                        rng.randint(
                            MIXED_BULK_MIN,
                            MIXED_BULK_MAX,
                        ),
                        "b",
                    )
                elif roll < UPDATE_ROLL:
                    do_update()
                else:
                    do_delete_many()
            if args.delay:
                time.sleep(args.delay)

        journal.append({"ev": "done", "op": op_index})
    except BaseException as exc:  # noqa: BLE001 - reported to the harness
        journal.append({
            "ev": "error",
            "op": op_index,
            "error": f"{type(exc).__name__}: {exc}",
        })
        raise

    # Wait for SIGKILL.  Never close the store: a clean close would flush.
    while True:
        time.sleep(IDLE_SLEEP)


# ==========================================================
#        Main
# ==========================================================

def main(argv: list[str] | None = None) -> int:
    run(_parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
