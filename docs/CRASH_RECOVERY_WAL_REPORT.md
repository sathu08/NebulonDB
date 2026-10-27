# NebulonDB Crash Recovery & WAL Test Report

- **Date:** 2026-10-26
- **Result:** PASS
- **Engine:** NebulonCosmos (append-only segments + WAL)
- **Environment:** WSL2 Ubuntu, Python 3.10, gunicorn + uvicorn worker, single worker
- **Isolation:** dedicated homes and ports; live services on `6969` / `9696` were not used
  - NebulonDB crash harness: `127.0.0.1:6970`
  - NebulonMind crash harness: DB `127.0.0.1:6971`, Mind `127.0.0.1:9697`

---

## 1. Executive Summary

| Check | Result |
|---|---|
| Acknowledged writes lost after SIGKILL | **0** |
| Duplicate rows after recovery | **0** |
| Missing acknowledged rows | **0** |
| Record ID reuse after recovery | **0** |
| Baseline 1,000-row corpus intact | **Yes (1000/1000)** |
| WAL CRC / corruption warnings in logs | **0** |
| Crash iterations completed | **8** |
| Final crash-corpus rows | **29,825** |
| Automated durability tests | **5 passed** |
| Automated storage/security tests | **6 passed** |

Final independent verification after the crash matrix:

```text
VERIFY rows=29825 unique=29825 dups=0 missing=0 extra=0
        baseline=1000 baseline_unique=1000
IDs contiguous 1..N: True max: 29825
VERIFY_OK
```

---

## 2. WAL Durability Model

WAL record framing:

```text
[ CRC32 (4 bytes) ][ payload length (4 bytes) ][ payload ]
```

Write path (`ndb_host/db/engine/cosmos/wal.py`):

- Every record is appended with a CRC32 checksum and length prefix.
- The userspace buffer is flushed on every write.
- `os.fsync()` is group-committed once 64 KiB of WAL data accumulates
  (`fsync_interval = 65536`); batch writes use at most one fsync per batch.
- WAL size is bounded by checkpointing live memtable rows back to the WAL and by
  flushing recovered/new rows to segment files.

Recovery path (`recover_wal`):

- Replays the WAL record by record.
- Verifies CRC32; corrupted/truncated entries are skipped with a warning.
- Restores segment counters, `global_version`, and `global_record_id`.
- Rebuilds the memtable and deleted-ID markers, then persists metadata.
- Recovered rows are flushed to a new segment.

Crash-safe commit ordering:

1. Segment data written and fsynced.
2. Manifest + index committed.
3. WAL truncated/checkpointed only after the manifest commit succeeds.

Related guarantees exercised by `unittest/test_cosmos_durability.py`:

- Manifest failure before commit keeps the WAL (no data loss).
- Crash after manifest commit but before WAL truncate is duplicate-safe.
- Orphan segment from a legacy crash is adopted on restart.
- Stale `last_segment_id` is reconciled; segment IDs are never reused.
- Compaction commits manifest + index before deleting old segments.

> **Power-loss note:** the tested failure mode is process `SIGKILL`. A process
> kill preserves data already in the OS page cache. A true power cut guarantees
> only the WAL groups that reached `fsync()` (up to the 64 KiB group boundary).
> Use per-ack fsync / a smaller interval if strict power-loss durability of every
> acknowledged write is required.

---

## 3. Automated Test Results

| Suite | Command | Result |
|---|---|---|
| Cosmos durability | `pytest unittest/test_cosmos_durability.py` | **5 passed** (0.30s) |
| Auth cache / security | `pytest unittest/test_security_cache.py` | **6 passed** (5.58s) |
| Bulk client | `pytest unittest/test_bulk_client.py` | **4 passed** (22.90s) |
| Live API regression | `test_segment.py test_corpus.py test_auth.py test_system.py` | **75 passed** (11.20s) |
| NebulonMind chat + API | `test_chat_history.py test_api.py` | **54 passed** (4.28s) |
| NebulonMind storage stores | `test_stores.py` | **9 passed** in isolation |
| NebulonMind remaining modules | full suite excluding LLM/env-sensitive modules | **401 passed** (4 `test_stores` failures only under cross-module ordering; 9/9 alone) |
| WAL/storage lint | `py_compile` on changed modules | **OK** |

The five durability regressions explicitly cover:

1. Flush failure before manifest commit keeps the WAL.
2. Crash after manifest but before WAL truncate is duplicate-safe.
3. Orphan segment is adopted after a legacy crash.
4. Stale metadata never reuses a segment ID.
5. Compaction crash before cleanup keeps all records.

Full NebulonMind suite note: the remaining failures are environment/config
dependent (the repo `.env` contains a live LLM key and `nebulonmd.cfg` selects an
LLM provider), plus one pre-existing agent fallback-turn assertion. `test_stores`
also shows 4 ordering-dependent failures when run after the live-API modules, but
passes 9/9 in isolation. None are storage or crash-recovery regressions.

---

## 4. Crash / Recovery Matrix (Phase 3)

Eight `SIGKILL` scenarios were run against the isolated server. After every
restart the harness verified:

- every acknowledged record is readable,
- rows == unique IDs (no duplicates),
- the 1,000-row baseline corpus is intact,
- new writes use IDs greater than the recovered maximum (no ID reuse).

| # | Scenario | Kill point |
|---|---|---|
| C1 | `mid_write` | 0.5s into a write batch |
| C2 | `ack_immediate` | immediately after an acknowledged batch |
| C3 | `flush_window` | during flush-to-segment |
| C4 | `ack_plus_flush_tick` | acknowledged writes + 2.5s settle |
| C5 | `compaction_window` | during background compaction |
| C6 | `ack_plus_steady` | acknowledged writes + 6s steady state |
| C7 | `mid_write_repeat` | repeat of C1 |
| C8 | `flush_window_repeat` | repeat of C3 |

Final state:

```text
crash_cosmos rows=29825 unique=29825 dups=0 missing=0 baseline=1000
PHASE3_OK iterations=8 acked_total=29825
```

`state.json` evidence:

```text
iterations: 8
acked_total: 29825
unique: 29825
duplicates: 0
min_id: 1
max_id: 29825
contiguous 1..max: True
```

WAL recovery evidence from the isolated DB log:

```text
recoveries: 18
records replayed across restarts: 27,728
largest single WAL replay: 7,500 records
CRC / corruption / traceback lines: 0
```

Compaction during the matrix:

```text
Compaction merged 7 segments, retained  9,575 active records
Compaction merged 4 segments, retained 10,325 active records
Compaction merged 5 segments, retained 18,900 active records
```

After the final clean shutdown all recovered rows were flushed and the crash
corpus WAL was `0` bytes; the live data remained in the segment files and index.

---

## 5. Data Integrity and Restart Results

### Phase 1-2: baseline load and read-back

```text
baseline.json: count=1000, unique=1000
PHASE12_OK
```

### Phase 2b: server restart

```text
restart readback rows=1000 unique=1000
missing=0 extra=0
PHASE2B_OK
```

### Phase 4: NebulonMind + NebulonDB crash test

- 100 chats acknowledged before the crash.
- NebulonDB was `SIGKILL`ed while the Mind API was still writing.
- Mind liveness stayed `200` while the DB was down.
- DB restarted; acknowledged chats were verified **without restarting Mind**.
- 300 bulk chat documents written, DB killed again, recovery verified.
- Remaining 700 bulk documents written.
- Mind restarted and verified against the DB as source of truth.

Result:

```text
total chat docs expected=1120
PHASE4_OK expected=1120
```

DB recovery events during this phase replayed 1, 8, 3, 120, 1, and 300 records.
The final `mind_chats` partition held 1,120 unique chat documents with no
duplicates.

---

## 6. Evidence

| Artifact | Path |
|---|---|
| Crash-run state | `/tmp/ndb_crashtest/state.json` |
| Baseline IDs | `/tmp/ndb_crashtest/home/baseline.json` |
| Isolated DB log | `/tmp/ndb_crashtest/home/logs/nebulondb_2026-10-26.log` |
| Crash corpus storage | `/tmp/ndb_crashtest/home/Storage/crash_cosmos/` |
| Baseline corpus storage | `/tmp/ndb_crashtest/home/Storage/cra_cosmos/` |
| Mind crash DB log | `/tmp/nmd_crashtest/dbhome/logs/nebulondb_2026-10-26.log` |
| Mind crash API log | `/tmp/nmd_crashtest/mindhome/logs/mind.log` |
| Harness: baseline | `/tmp/ndb_crashtest/phase12.py` |
| Harness: restart verify | `/tmp/ndb_crashtest/phase2b_verify.py` |
| Harness: crash matrix | `/tmp/ndb_crashtest/phase3_crash.py` |
| Harness: Mind crash | `/tmp/ndb_crashtest/mind_crash.py` |

---

## 7. Notes and Recommendations

1. **Single worker only.** Nova/Mesh in-memory state assumes `--workers 1`. The
   test harness and live deployment both use one worker.
2. **Power-loss hardening.** For strict per-ack durability under power loss,
   lower `fsync_interval` or fsync per acknowledged batch.
3. **Isolated test runs.** Storage suites must run with `env -u NEBULONDB_HOME`
   in a separate pytest process so they do not pick up the live config.
4. **Mind suite environment.** LLM-related Mind tests assume no live provider
   key; the repository `.env` currently contains one. Run those with a clean
   `NEBULONMD_HOME` or unset LLM variables.
