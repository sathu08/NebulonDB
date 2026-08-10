# NebulonDB API Test Report

**Date:** 2026-08-06 (fourth run — post compaction-fix verification)
**Test target:** `ndb_host/api/routes/` — `auth.py`, `system.py`, `corpus.py`, `segment.py`
**Live API:** `http://127.0.0.1:6969/api/NebulonDB`
**Authentication:** HTTP Basic — `sathya` / `sathya`
**Runner:** `pytest` (venv: `/home/sathyaprakash/venv`), `requests`
**Report location:** `/home/sathyaprakash/CodeBase/tests/unittest/`

---

## 1. Summary

| Suite | Total | Passed | Failed | Pass % |
|-------|------:|-------:|-------:|-------:|
| test_auth.py (basic)    | 8  | 8  | 0 | 100% |
| test_system.py (basic)  | 4  | 4  | 0 | 100% |
| test_corpus.py (30: small/medium/large) | 30 | 30 | 0 | 100% |
| test_segment.py (30: small/medium/large) | 30 | 30 | 0 | 100% |
| **TOTAL** | **72** | **72** | **0** | **100%** |

**Result: 72 passed, 0 failed.** (Previous runs: 68/4 → 94.4%; 72/0 after first fix round.)

After the initial 4 fixes, follow-up robustness issues were investigated and
**resolved**: flush/Nova consistency warnings are gone, user persistence is
immediate, and the shared-metadata-store compaction is now data-safe. See
**Section 6**.

All four previously-documented failures in `test_segment.py` are now **FIXED**:

| # | Test | Before | After |
|---|------|--------|-------|
| 4.1 | `test_medium_add_relation_and_neighbors` | 0 neighbors, `edge_count: 0` | 1 neighbor, edge persisted |
| 4.2 | `test_large_shortest_path_with_path` | "No path between 1 and 3" | path `[1, 2, 3]`, `path_length: 3` |
| 4.3 | `test_large_search_rank_enabled` | `'NoneType' object has no attribute 'lower'` | `success: true` with ranked results |
| 4.4 | `test_large_remove_relation` | `'OrbitDBManager' object has no attribute 'remove_relation'` | `success: true` |

---

## 2. Credential finding

The stored bcrypt hash for user `sathya` verifies against password `sathya`. All tests
use `sathya` / `sathya` (role `super_user`). The password `sathay` mentioned in the
original notes is invalid.

---

## 3. Root causes & fixes

All fixes were applied to the deployed code that the server actually runs
(`CodeBase/tests/ndb_host/`) and mirrored to the main source tree
(`CodeBase/NebulonDB/ndb_host/`). The server (gunicorn + uvicorn, 1 worker) was
restarted to load the changes.

### 3.1 Edges never persisted → fixed (`orchestrator.py`, `segment.py`)
- **Root cause A (LSN reset):** Each API request constructs a fresh `NebulonOrbit`
  whose `_lsn_counter` started at `0`, while the persisted Nova mapping stored a
  higher `last_applied_lsn` (e.g. `3`). New WAL entries written by that process
  (e.g. `add_relation` with `lsn=1`) were therefore treated as *already applied*
  during `_replay_wal()` (`lsn <= last_applied` → skipped) and silently dropped.
  - **Fix:** after the nova load / WAL replay phase in `NebulonOrbit.__init__`,
    the LSN counter is now raised to `max(_lsn_counter, nova_engine.last_applied_lsn)`
    so newly written entries always have `lsn > last_applied` and replay correctly.
- **Root cause B (no flush):** the graph-mutating endpoints (`add_node`,
  `add_relation`, `remove_relation`, `delete_record`) created a per-request orbit,
  mutated the in-memory mesh, set `dirty`, and then discarded the object — the edge
  was never persisted (WAL is cleared on the next request, losing the change).
  `load_segment` already flushed; the single-op endpoints did not.
  - **Fix:** each of these route handlers now calls
    `orbit.initialize_or_flush()` after the mutation, persisting the graph.

### 3.2 `rank=True` crash → fixed (`ranking.py`)
- **Root cause:** `RankEngine._default_metadata_rules()` called
  `meta.get("lang", "").lower()`; segment metadata stores `"lang": None` when no
  language is provided, so `None.lower()` raised
  `'NoneType' object has no attribute 'lower'`.
  - **Fix:** coerce to string first — `str(meta.get("lang") or "").lower()` (and the
    same for `type`). `BM25Scorer._tokenize` was also hardened against `None` text.

### 3.3 `remove_relation` unimplemented → fixed (`index_manager.py`)
- **Root cause:** the running copy of `OrbitDBManager` (`CodeBase/tests/ndb_host/`)
  had no `remove_relation` wrapper, so the route 500'd with
  `'OrbitDBManager' object has no attribute 'remove_relation'`.
  - **Fix:** added `OrbitDBManager.remove_relation()` delegating to
    `NebulonOrbit.remove_relation()` (which already existed, along with
    `MeshEngine.remove_edge`).

---

## 4. Endpoint coverage (live HTTP) — all passing

- **Auth (`/api/NebulonDB/auth`) — 8:** register (new/duplicate/short-password 422/
  no-auth 401), verify (valid/wrong-password/unknown-user/no-creds 401).
- **System (`/api/NebulonDB/system`) — 4:** config host/port/url, no-auth access,
  url consistency, host non-empty.
- **Corpus (`/api/NebulonDB/corpus`) — 30:** create/list/activate/deactivate/delete,
  duplicate & system-corpus protection, bulk lifecycle, unicode/long/special names,
  recreate-after-delete.
- **Segment (`/api/NebulonDB/segment`) — 30:** load (dict/list/first-column/all/
  precomputed/doc+lang/append/empty-rejected), list, search (nova/mesh/hybrid/top-k/
  ranked), stats, get_record, add_node, add_relation + neighbors, bfs, shortest_path
  (found & no-path), remove_relation, delete_record, mesh_visualization.

---

## 5. Residual observations

- All previously-observed non-blocking warnings (`Flush failed writing segment …`,
  `Consistency check failed`) are **gone** in the post-follow-up run — see Section 6.
- All negative/security cases (401 unauth, wrong password, 422 validation, duplicate
  corpus, system-corpus protection) behave as expected.
- Dev-only unit tests under `ndb_host/tests/` (not part of this 72-test suite):
  `test_orbit_ranking.py` (12) and `test_orbit_recovery.py` (3) pass;
  `test_segment_manager.py` has a pre-existing search-result-format assertion failure
  that is unrelated to the fixes (it is untracked in git and not part of the API suite).

---

## 6. Follow-up issues (post 4-fix round) — root causes & fixes

Three robustness issues were found and fixed after the initial 72/72 pass. All changes
were applied to `CodeBase/tests/ndb_host/` (the tree the server runs) and mirrored to
`CodeBase/NebulonDB/ndb_host/`; both trees are byte-identical (`diff -rq` clean).

### 6.1 Source-tree reconciliation (`tests/` vs `NebulonDB/`)
- **Issue:** the two trees had drifted — `NebulonDB/` held `rerank` methods in
  `db/index_manager.py` and a `GraphQueryRequest` model; `tests/` held newer routes
  (`system.py`), `web_dir/`, PID-file constants and inline query fields that
  `NebulonDB/` lacked.
- **Fix:** ported `rerank` into the `tests/` copy (both `OrbitDBManager` wrapper and
  `SegmentManager` level), rsynced `tests/ndb_host/` → `NebulonDB/ndb_host/`, and
  copied `NebulonDB/ndb_host/tests/` dev unit tests into `tests/ndb_host/tests/`.
  Both trees now reconcile cleanly.

### 6.2 User-persistence fragility (`services/user_service.py`)
- **Issue:** `db.flush()` was **commented out** in `create_user` and `delete_user`.
  The cosmos background flush only runs on thresholds or after ~30s, so a hard process
  kill could lose a just-created user (this caused the earlier lost `sathya` account).
- **Fix:** restored the uncommented `db.flush()` calls after insert and after delete.
  Users now hit disk immediately (verified on disk under `tests/Storage/…/segments/`).

### 6.3 Flush-vs-delete race on corpus teardown (`db/engine/cosmos/`)
- **Issue:** per-request `NebulonCosmos` stores spawn daemon flush/compaction threads
  that outlive the request. `delete_corpus` does `shutil.rmtree(corpus_path)` while
  those threads still reference the store, so a wake-up flush raised
  `Flush failed writing segment seg_1.ndb: No such file or directory …/seg_1.tmp`.
- **Fix:** `store._background_flush_loop`, `store._flush`, and
  `compactor.background_compaction_loop` now bail out (setting the stop event) when
  `seg_dir`/`db_dir` no longer exists. Flush errors after restart: **0**.

### 6.4 Nova consistency false-positive on multi-segment corpora (`db/engine/orbit/`)
- **Issue:** the Nova HNSW engine files (`NebulonNova/`) were **shared per-corpus**
  while the cosmos vector records and mesh engine are **per-segment**
  (`nebulon_nova_<seg>` / `nebulon_mesh_<seg>`). Loading a second segment into a
  corpus made `_check_consistency` compare the new segment's (empty) records against
  the shared engine's id-map (`db_ids=[] graph_ids=[1, 2, 3]`), then
  `_rebuild_nova_engine_from_db` **rebuilt the shared index from just the new
  segment**, wiping the other segments' vectors from the index.
- **Fix:** `NebulonOrbit.__init__` now derives per-segment Nova paths
  (`NebulonNova/segment_<name>/` for the index, mapping, config, and `nova.wal`),
  matching the per-segment isolation of the cosmos records and mesh engine.
  Reproduced pre-fix with a 2-segment script (exact same warning), verified clean
  post-fix; segment A's index survives opening segment B.
- **Result (post-restart run):** `Consistency check failed` → **0** (was 4),
  `Rebuilding Nova from DB` → **0** (was 4), `Flush failed` → **0**.

### 6.5 Concurrent multi-user injection — verified + fixed (`db/engine/cosmos/`)
- **Verification:** 6 simultaneous users (distinct usernames) each completing
  register → create_corpus → load_segment → list_segment succeed (all `201/200`,
  `success: true`), both pre- and post-restart; no server crash. 1-worker async
  serialization + `RLock` + per-request stores make multi-user injection safe.
- **Issue found:** log inspection of those runs surfaced a background **compaction**
  crash on the shared metadata store (`tests/Storage/Secrets/NebulonCosmos`):
  `_write_segment_streaming` was called with a non-existent `index_entry_format`
  kwarg (`TypeError`), so compaction never completed and stale in-memory index
  entries kept pointing at segments that had been deleted.
- **Fix 1 (`segment_writer.py:227`):** compaction's in-memory `latest` map now
  repoints on equal version (`latest[rec_id].version <= version`, was `<`).
- **Fix 2 (`store.py`):** `_compact` now passes `_rebuild_index_from_all_segments`
  as the post-merge index rewrite callback (was `_rewrite_index_from_latest`,
  which only persisted the stale map). After a merge the in-memory `latest` is
  rebuilt from all remaining segment files, so no entry can reference a removed
  segment (previously → `FileNotFoundError: …/segments/seg_1.ndb` on read,
  making **every** metadata read return empty → `Corpus not found` on all
  segment/corpus ops).

### 6.6 Record-ID collision across tables destroyed user data (`db/engine/cosmos/`)
- **Issue:** `_next_id` allocated per-table ids (`tables[segment] += 1`) while
  `latest`/`latest_source` are keyed by bare `rec_id`. The `nebulon_userinfo` and
  `nebulon_metadata` tables in the shared Secrets store therefore collided on ids
  `1..15`; compaction keyed `latest_source` by bare `rec_id`, so for each colliding
  id only one table's record survived the merge and the other was silently dropped
  when the source segments were deleted. This destroyed all user records (users
  `1..15`) during the two live compactions and broke `sathya` login.
- **Fix 1 (`compactor.py`):** compaction now keys `latest_source` by
  `(table, rec_id)` so records from different tables sharing an id are never
  dropped during a merge (verified: 6 users + 55 corpus records all survive a full
  store compaction on a copy).
- **Fix 2 (`store.py`, `wal.py`):** `_next_id` now returns a **globally unique**
  id via a single `meta["global_record_id"]` counter (seeded from the max existing
  per-table counter on load and restored from the WAL), so new records across all
  tables can never collide again.
- **Data repair:** the pre-fix compaction had already destroyed the stored users
  and the `nebulon_origin` system-corpus metadata record; these were re-seeded
  (`sathya` → `super_user`, id 650; `nebulon_origin` → status `system`, id 717)
  and verified via the API. Recovery relied on the fact that `load_index`/rebuild
  scans actual segment files, so no further repair tooling was needed.

### 6.7 Final verification
- Full 72-test suite: **72 passed, 0 failed** (run after all fixes + data repair).
- Concurrent 6-user injection after the fixes: all `OK`, `5.2s`.
- Server log since latest boot: **0** `ERROR`/`Traceback`/`FileNotFoundError`;
  a live compaction ran during the suite (`Compaction merged 10 segments, retained
  61 active records`) with no errors.

---

## 7. Environment

- Server: gunicorn + uvicorn worker, 1 worker, `127.0.0.1:6969`, cwd
  `/home/sathyaprakash/CodeBase/tests`, code under `tests/ndb_host/` (mirrored to
  `NebulonDB/ndb_host/`).
- Python 3.10.12, fastapi, polars, numpy, passlib+bcrypt, hnswlib 0.8.0.
- Log: `CodeBase/tests/logs/server_test.log`
