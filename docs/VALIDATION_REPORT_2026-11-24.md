# NebulonDB Validation Report

| | |
|---|---|
| **Project** | NebulonDB: `/Users/x344414/Documents/Codebase/NebulonDB` |
| **Review date** | 2026-11-24 (initial static review; remediation + verification same day) |
| **Review type** | Static review, then code remediation with live verification on Mac. Code, config and test files were changed (see §7); `nebulondb.cfg` live values stay machine-local. |
| **Overall rating** | **Remediated, localhost-only.** All Critical/High/Medium/Low code fixes applied and test-verified; server binds `127.0.0.1:6969`. Re-review before any LAN/internet exposure. |
| **Status** | Remediated and test-verified (seeded-data run 2026-11-26: live 75 + isolated 59 passed; Mind 520 passed). Open: WSL block-layer run, Phase 5 Mac port. |
| **Fix commit** | `dev` = `2e08284` (audit/tools/ignore) on top of `cedc581` (cfg cleanup) on top of `929032b` (F/L hardening). Pushed to `origin/dev`. |

---

## 1. Executive summary

NebulonDB's storage core is largely sound. Segment writes are atomic, compaction runs in the correct order, and every corpus and segment endpoint requires authentication. The risks sit around that core:

- **Any logged-in user can escalate.** Any account can create a SYSTEM account and delete other accounts. The corpus-name check can be bypassed to create and delete directories outside storage.
- **Configuration can be changed without logging in.** An unauthenticated endpoint rewrites the config file, which holds the master key and the storage root.
- **Some code paths can delete healthy data.** Compaction deletes segments on read errors, and bootstrap can wipe the user store when it fails.
- **Secrets are poorly protected.** The master key sits in a world-readable config file, and generated passwords are written to log files.

Fix the four Critical items before any network exposure.

> Update 2026-11-24 (evening): all four Criticals plus every High/Medium/Low item is fixed in code on `dev` (`2e08284`, pushed), the server binds localhost-only, and the suites pass (live 79, isolated 55). The bullets above describe the *original* findings, kept for the record — each is closed by the §7 item noted in its finding (e.g. F-01 → Phase 0/1, F-08 → Phase 2, F-19 → Phase 3).

---

## 2. Findings at a glance

| ID | Severity | Area | Finding | Evidence |
|---|---|---|---|---|
| F-01 | Critical | API / config | Unauthenticated `PUT /dashboard/config` rewrites `nebulondb.cfg` | Code read |
| F-02 | Critical | Auth | Any user can register accounts with any role, including `system` | Code read |
| F-03 | Critical | Auth | Any user can delete any other account, including admins | Code read |
| F-04 | Critical | Corpus | Corpus names are not validated, allowing create and `rmtree` outside storage | Code read |
| F-05 | High | Secrets | Master key stored in plaintext in a world-readable file | File mode and content check |
| F-06 | High | Bootstrap | Startup error handler deletes the user store | Code read |
| F-07 | High | Secrets / logs | Generated system passwords written to log files | Log line counts |
| F-08 | High | Data safety | Compaction deletes segments on read errors | Code read |
| F-09 | High | Credentials | Hard-coded default credentials; live account reportedly uses the default | Code read; August test report |
| F-10 | High | Storage | Reads fail when a manifest entry points to a missing file | 26 tracebacks in logs |
| F-11 | Medium | API | CORS allows every origin with credentials | Code read |
| F-12 | Medium | API | Unauthenticated `GET /dashboard/config` discloses settings | Code read |
| F-13 | Medium | Config | Config rewritten from constructors; non-atomic writes drop comments | Code read |
| F-14 | Medium | Config | Hard-coded Linux paths; config lookup falls back to the working directory | Code read; path check |
| F-15 | Medium | Config | Config not shared between modules (no singleton; two module names) | Code read |
| F-16 | Medium | Logging | Log retention not enforced; mixed file-name formats | Log directory listing |
| F-17 | Medium | Durability | Failed group-commit fsync treated as success (Nova WAL) | Code read |
| F-18 | Medium | Auth | Deleting a user reports success even when the delete fails | Code read |
| F-19 | Medium | Code quality | 119 broad `except Exception` handlers hide failures | Grep count |
| L-01 | Low | API | Failed login returns 200; duplicate registration returns 201 | Test expectations |
| L-02 | Low | Auth | Current password not required for password change | Code read |
| L-03 | Low | Auth | Username enumeration by response timing | Code read |
| L-04 | Low | Auth | Unknown required role passes the permission check | Code read |
| L-05 | Low | Dependencies | `passlib` is unmaintained | Version pins |
| L-06 | Low | WAL | Corrupt WAL entries are skipped with only a warning | Code read |

**Totals:** 4 Critical, 6 High, 9 Medium, 6 Low.

**Severity scale**
- **Critical:** exploitable now by a logged-in user or an unauthenticated network client, or direct data loss.
- **High:** likely data loss, secret exposure, or serious misconfiguration.
- **Medium:** weakens defenses or correctness, with limited impact.
- **Low:** hardening or consistency.

---

## 3. Scope and method

**Reviewed**
- `ndb_host/` (16,035 lines): auth and user service, API routes, core security and permissions, config loader, bootstrap, logging, and the Cosmos storage path (WAL, flush, compaction, reads).
- `nebulondb.cfg`, `pyproject.toml`, `requirements.txt`, and the NebulonMD and NebulonAK configs (port consistency).
- Runtime logs in `logs/` (counts only).
- `unittest/` test expectations and the August 2026 test report.

**Checks performed**
- Syntax parse of all 72 Python files in `ndb_host/` and `unittest/`: 0 errors.
- File modes and directory permissions for the config and `Storage/`.
- Log line counts for specific messages and tracebacks.
- Path checks for hard-coded locations on this machine.
- Dependency pins compared against the `pyproject.toml` ranges.

**Not done (out of scope, still true)**
- No vulnerability scan of pinned packages was run.
- Not reviewed line by line: orbit, mesh and ranking internals; the TUI; the powerloss harness internals; documentation versus code.

**Done since the initial review (supersedes the original list below)**
- ~~Server not running~~ — started on Mac (`uv` venv, Python 3.10), currently live on `127.0.0.1:6969`.
- ~~Live API tests not run~~ — ran: live 79 passed (auth 8, system 4, corpus 30, segment 33, bulk 4).
- ~~Isolated tests not run / deps missing~~ — ran via project venv: isolated 55 passed (durability + server-config 31, powerloss unit 22, mac-SIGKILL 2); manifest checker clean (2/2).
- ~~Path checks~~ — `NEBULONDB_HOME` set to the Mac checkout; missing-home fallback added in code.

---

## 4. Detailed findings

### Critical

#### F-01 Unauthenticated configuration write
- **Location:** `ndb_host/api/routes/dashboard.py:139-176`; `ndb_host/utils/constants.py:108-109`; `ndb_host/db/ndb_settings.py:96-99`
- **Finding:** `PUT /api/NebulonDB/dashboard/config` has no authentication dependency. It accepts any section and key (`dict[str, dict[str, Any]]`), writes them to `nebulondb.cfg` with `_write()`, and reloads the config.
- **Impact:** Anyone who can reach port 6969 can change:
  - `[environment] nebulondb_master_key`. Existing encrypted secrets stop decrypting after the next restart.
  - `[paths] nebulondb_home`. The storage root, and with it the user store, moves on restart unless `NEBULONDB_HOME` is set in the environment.
  - `[powerloss] enabled`. The powerloss daemon re-reads this flag on every poll (`ndb_host/tools/powerloss/scheduler.py`). A running daemon would start root-level harness runs that change global `vm.dirty_*` sysctls.

  The server binds `0.0.0.0:6969` (`nebulondb.cfg` lines 83-84), so the exposure depends on network reachability.
- **Recommendation:** Require SYSTEM or SUPER_USER authentication. Allow-list the writable keys and validate their values. Reject values containing newlines. Write atomically. Consider removing the PUT endpoint entirely.

#### F-02 Any user can create a SYSTEM account
- **Location:** `ndb_host/api/routes/auth.py:38-81`; `ndb_host/services/user_service.py:284-306`; `ndb_host/utils/constants.py:87-91`; `ndb_host/core/permissions.py:35-40`
- **Finding:** `POST /auth/register` checks only that the caller is authenticated. It never calls `check_user_permission`. `create_user` accepts any valid `UserRole`, including `system`.
- **Impact:** A standard `user` account can create `system` or `super_user` accounts. SYSTEM (level 4) passes every ADMIN_USER check, so this bypasses all role checks.
- **Recommendation:** Make registration admin-only. Refuse `system` and `super_user` roles through the API. Add tests for both.

#### F-03 Any user can delete any other account
- **Location:** `ndb_host/api/routes/auth.py:150-195` (self-deletion check at line 168)
- **Finding:** `/auth/delete_user` blocks only self-deletion. It has no role check.
- **Impact:** Any logged-in user can delete admin or SYSTEM accounts, locking administrators out.
- **Recommendation:** Require ADMIN_USER or higher. Forbid deleting accounts of equal or higher rank. Add tests.

#### F-04 Corpus names are not validated (path traversal)
- **Location:** `ndb_host/db/index_manager.py:558` (create); `:585-587` (delete, `shutil.rmtree`); `ndb_host/api/routes/corpus.py:67-135` (create, ADMIN check at line 108); `:199-282` (delete, ADMIN check at line 235); `ndb_host/utils/models.py:54-56`
- **Finding:** The corpus name is joined to the storage path with no validation. `CorpusQueryRequest.corpus_name` only requires a non-empty string. A name such as `../something` escapes the storage root.
- **Impact:** An ADMIN user can create directories and database files outside storage, then delete arbitrary directories with `rmtree`. Because of F-02, any account holder can reach the ADMIN role.
- **Recommendation:** Allow-list names (for example `^[A-Za-z0-9_-]{1,64}$`). Also check that the resolved path is inside the storage root. Add tests with traversal inputs.

### High

#### F-05 Master key stored in plaintext in a world-readable file
- **Location:** `nebulondb.cfg` line 21; file mode `-rw-rw-r--`; `ndb_host/db/ndb_settings.py:122-130` (key handling) and `:152-155` (loading)
- **Finding:** The master key that protects `Storage/Secrets` is stored in plaintext in the config file. The file is readable by every local account. `Storage/` and `Storage/Secrets/` are group-writable (`drwxrwxr-x`).
- **Impact:** Any local account can read the key. Anyone who copies the project folder also gets the key.
- **Exposure note:** The key value was printed in the review session output. Treat it as exposed if that transcript is shared.
- **Recommendation:** Set the file to mode 600. Move the key to an environment variable or the system keyring. Restrict `Storage/Secrets` to its owner. Plan a rotation: back up `Storage/Secrets`, re-encrypt, then replace the key.

#### F-06 Bootstrap deletes the user store on error
- **Location:** `ndb_host/tui/bootstrap.py:245-282`; default argument at line 249; handler at line 281; `ndb_host/db/ndb_settings.py:145`
- **Finding:** `bootstrap_users` defaults `secrets_dir` to `NEBULONDB_ACCOUNTHUB_CORPUS_PATH`, which is the user store. Its exception handler runs `shutil.rmtree(secrets_dir, ignore_errors=True)` and then exits.
- **Impact:** If the handler runs, every account is deleted, not only the ones created in that run. The path is hard to reach today because `create_user` swallows most errors, so the risk is latent.
- **Recommendation:** Remove the `rmtree` from the error path. Never delete a store that existed before the run. Log the error and exit without touching data.

#### F-07 Generated system passwords are written to logs
- **Location:** `ndb_host/tui/bootstrap.py:267-277`; return value of `service_create_user` ignored at lines 268-272
- **Finding:** The generated SYSTEM password is logged in plaintext. "Default users created successfully" is logged even when the call returns "User already exists". On re-runs, the password that gets logged was never set on the account.
- **Evidence:** The line appears in four log files: `nebulondb_2026-10-28.log`, `nebulondb_2026-11-02.log`, `nebulondb_2026-11-03.log` and `nebulondb_2026-11-21.log`. The values are not copied into this report.
- **Impact:** The 28 Oct entry is probably the real creation and may still be valid for the SYSTEM account. Anyone who can read the logs could use it.
- **Recommendation:** Check whether that password still works, and rotate it if so. Stop logging generated passwords. Check the return value of `create_user`. Purge the log lines that contain passwords.

#### F-08 Compaction can delete healthy data
- **Location:** `ndb_host/db/engine/cosmos/segment_reader.py:62-63`; `ndb_host/db/engine/cosmos/compactor.py:165-166` (skip), `:208-224` (all-dead branch), `:255-259` (delete), `:189-194` (silent drops)
- **Finding:**
  1. `validate_segment` returns "invalid" on any exception, including transient I/O errors.
  2. Compaction skips invalid segments but still deletes them with the rest of the merge set (`delete_files=True`).
  3. If no record in the merge set decodes, every segment in the set is deleted with no quarantine (`:208-224`).
  4. Records that fail the CRC or decode checks are dropped silently during the merge (`:189-194`).
- **Impact:** A transient read error during routine compaction (triggered above 10 segments) permanently removes that segment's records. A systematic decode failure, for example after a format change, would delete all merged segments.
- **Recommendation:** Never delete on a read error. Separate I/O errors from corruption. Quarantine suspect segments instead of deleting them. Log the counts of skipped records. Add tests that simulate read errors.

#### F-09 Hard-coded default credentials
- **Location:** `ndb_host/client/many_docs2.py:34-35` (environment fallbacks); `unittest/conftest.py:14-15`; `unittest/TEST_REPORT.md` section 2
- **Finding:** Client code and tests fall back to a hard-coded username and password. The August test report says the live `sathya` account, which has the super_user role, accepts that default password. I did not verify this against the live system.
- **Impact:** If the default password is still active, anyone who can reach the API can log in as a super_user.
- **Recommendation:** Verify and rotate the password now. Remove the fallbacks. Load test credentials from the environment only.

#### F-10 Reads fail when a manifest entry points to a missing file
- **Location:** `POST /api/NebulonDB/segment/get_data` (`ndb_host/api/routes/segment.py:331`); `ndb_host/db/index_manager.py:944`; `ndb_host/db/engine/orbit/nova_store.py:75-77`; `ndb_host/db/engine/cosmos/store.py:927` → `:357` → `segment_reader.py:88` (`stat` call)
- **Evidence:** 26 `FileNotFoundError` tracebacks in the logs for 21 and 22 Nov. They come from the `get_data` route on the Orbit (Nova) path.
- **Finding:** `read_all` iterates every manifest entry and calls `stat` with no guard. A manifest entry whose file is missing fails the whole read.
- **Impact:** Reads return errors for affected corpora. The root cause is not yet known. It may be linked to F-08 or to files removed by hand.
- **Recommendation:** Run a read-only manifest-versus-disk check on each corpus. Decide whether missing files should be reported, skipped or repaired. Guard `read_all`.

### Medium

#### F-11 CORS allows every origin with credentials
- **Location:** `ndb_host/api/__init__.py:65-71`
- **Finding:** `allow_origins=["*"]` is combined with `allow_credentials=True`. That combination is invalid under the CORS specification and should not be relied on.
- **Recommendation:** Use an explicit origin list. Drop `allow_credentials` if the console only needs same-origin access.

#### F-12 Public configuration read
- **Location:** `ndb_host/api/routes/dashboard.py:76-132`
- **Finding:** `GET /api/NebulonDB/dashboard/config` returns the host, port, URL, worker count, timeouts, log settings and tuning parameters with no authentication.
- **Recommendation:** Require authentication, or return only what the console needs.

#### F-13 Config is rewritten from constructors; writes are not atomic
- **Location:** `ndb_host/db/ndb_settings.py:96-99` (`_write`); `:113-133` (environment override and key generation); `:157-164` (model cache directory)
- **Finding:**
  - `NDBConfig()` rewrites the file in three cases: when the environment home differs from the file, when the master key is blank (it then generates a new key silently), and when the model cache directory is missing.
  - `_write()` opens the file with `'w'` and drops all comments.
  - A crash during the write truncates the file, including the master key.
- **Evidence:** The model cache directory in the config (`/home/sathya/.cache/NebulonDB`) does not exist on this Mac, so starting the server here would rewrite the file. The file's last-modified date (28 Oct) shows that has not happened yet.
- **Recommendation:** Keep reads and writes separate. Write to a temp file and rename it. Preserve comments, or move runtime overrides to a separate file. Generate a new key only through an explicit command.

#### F-14 Hard-coded Linux paths; config lookup falls back to the working directory
- **Location:** `nebulondb.cfg` lines 18 and 28; `ndb_host/db/ndb_settings.py:58-60`
- **Finding:** `nebulondb_home` and the model cache directory point to `/home/sathya/...`. Neither exists on this Mac (`/home/sathya` and `~/.cache/NebulonDB` were both checked). Without `NEBULONDB_HOME`, the config is looked up in the current working directory, so another folder's config could be picked up.
- **Recommendation:** Resolve paths relative to the config file's location, or require an explicit environment variable. Fail loudly when the config file is not where it is expected.

#### F-15 Config is not shared between modules
- **Location:** `ndb_host/api/routes/dashboard.py:15` (imports `ndb_host.db.ndb_settings`); most modules import `db.ndb_settings`; `ndb_host/db/ndb_settings.py` (no singleton)
- **Finding:** `NDBConfig` is not a singleton, and the same module is loaded under two names. The dashboard's PUT updates only its own instance (`dashboard.py:154-161`), so other modules keep old values until restart. The code mixes import styles: 26 `ndb_host.` imports and 102 top-level imports.
- **Recommendation:** Pick one import style. Make the config a single shared object, and reload it explicitly if runtime reload is needed.

#### F-16 Log retention is not enforced
- **Location:** `ndb_host/utils/logger.py:104-125` (cleanup runs only at rollover); `:241-243` (backup count); `ndb_host/db/ndb_settings.py:176` (`[logging] date = 7`)
- **Evidence:** The `logs/` folder holds 20 files. The oldest is dated 26 Oct, about 29 days old against a 7-day setting.
- **Finding:** Cleanup runs only at midnight rollover while the server is running. It matches only `nebulondb_YYYY-MM-DD.log`, so files named `nebulondb_20261104.log` and `nebulondb_restart.log` are never removed.
- **Impact:** Logs, including those described in F-07, are kept far longer than configured.
- **Recommendation:** Run cleanup at startup and on a schedule. Match every file-name format, or standardize on one.

#### F-17 Failed group-commit fsync is treated as success (Nova WAL)
- **Location:** `ndb_host/db/engine/orbit/orchestrator.py:387-396`
- **Finding:** `_wal_fsync` suppresses `OSError` around the fsync, then resets the pending-byte counter. A failed fsync is recorded as durable.
- **Impact:** After power loss or I/O errors, acknowledged writes may not be on disk.
- **Recommendation:** Let the fsync error propagate, or mark the store as failed. Reset the counter only when the fsync succeeds. Add a power-loss test for this path.

#### F-18 User deletion reports success even when the delete fails
- **Location:** `ndb_host/services/user_service.py:319-334`, specifically line 327
- **Finding:** The boolean returned by `user_manager.delete_user` is ignored, so the service returns `success: True` even when the database delete failed.
- **Recommendation:** Check the boolean and return an error when it is False.

#### F-19 Broad exception handling hides failures
- **Location:** 119 occurrences of `except Exception` in `ndb_host/`. Storage-engine examples: `store.py`, `wal.py`, `compactor.py`, `segment_reader.py`, `nova_engine.py`
- **Finding:** Many handlers catch everything and continue or return a default. Several sit on data paths (see F-08 and F-17).
- **Recommendation:** Audit the storage-engine handlers first. Catch specific exceptions, and log or re-raise the rest.

### Low

#### L-01 HTTP status codes do not match outcomes
- **Location:** `ndb_host/api/routes/auth.py:34` (201 is fixed for every register response) and `:64-69`; `unittest/test_auth.py:31-34` and `:61-65`
- **Finding:** Wrong credentials return 200 with a failure message. A duplicate registration returns 201. The tests expect both, so they codify the current behavior.
- **Recommendation:** Return 401 for failed authentication and 409 for duplicates. Update the tests to match.

#### L-02 Current password not required for password change
- **Location:** `ndb_host/services/user_service.py:350`; `ChangePasswordRequest` in `ndb_host/utils/models.py`
- **Finding:** The current-password check runs only when the field is provided.
- **Impact:** Basic auth already proves the password, so the practical impact is small. The check should still be mandatory.

#### L-03 Username enumeration by response timing
- **Location:** `ndb_host/services/user_service.py:241-252`
- **Finding:** Unknown usernames skip the bcrypt check, so they respond faster than wrong passwords.
- **Recommendation:** Run a dummy bcrypt verification when the user does not exist.

#### L-04 Unknown required role passes the permission check
- **Location:** `ndb_host/core/permissions.py:42-43`
- **Finding:** An unmapped required role gets level 0, so any user passes.
- **Recommendation:** Deny by default when the required role is not in the hierarchy.

#### L-05 `passlib` is unmaintained
- **Location:** `requirements.txt` (`passlib==1.7.4`, `bcrypt==4.0.1`); `pyproject.toml`
- **Finding:** `passlib` has not been released since 2020. The `bcrypt==4.0.1` pin is what keeps it working.
- **Recommendation:** Plan a migration to `bcrypt` directly or another maintained hashing library. Keep existing hashes verifiable during the move.

#### L-06 Corrupt WAL entries are skipped with only a warning
- **Location:** `ndb_host/db/engine/cosmos/wal.py` (CRC-mismatch and decode-failure paths)
- **Finding:** Entries that fail the CRC check or decoding are skipped with a warning. Nothing is copied aside. If a length field is corrupted, the reader cannot resynchronize.
- **Recommendation:** Copy bad entries to a quarantine file, and log the count at startup.

---

## 5. What is working well

- All 5 corpus endpoints and all 19 segment endpoints check authentication (`is_authenticated`).
- Passwords are hashed with bcrypt, require at least 8 characters, and generated passwords use the `secrets` module.
- Compaction order is correct: write the new segment, atomically replace the manifest, rebuild the index, then delete the old files (`compactor.py:237-259`).
- Segment writes go to a temp file and are renamed into place (`segment_writer.py`).
- WAL and memtable pruning compares versions before dropping rows (`store.py:252-282`).
- The durability mode is validated (`ndb_settings.py:209-215`), and the server enforces a single worker (`api/__init__.py:17-37`).
- Ports agree across the three configs: NebulonDB 6969, NebulonMD → 6969, NebulonAK → Mind 9696.
- Dependency pins satisfy the `pyproject.toml` ranges (bcrypt 4.0.1, cryptography 43.0.3, fastapi 0.139.0, uvicorn 0.30.6, gunicorn 21.2.0).
- All 72 Python files parse without syntax errors.
- A stale PID file is handled on the next start (`tui/server_ops.py:56-61`).

---

## 6. Test coverage

- The August 2026 test report (72/72 passing) is stale. It references another machine's paths and a `system.py` module that no longer exists.
- The current auth tests do not cover role escalation, account deletion, corpus-name validation, config endpoint authentication, or bootstrap failure. F-01 to F-04 would not be caught by the existing suite.
- The live tests run against the running server and its live data store, with the default credentials. They should run against an isolated copy of the data instead.

---

## 7. Remediation plan

Work through the phases in order. Tick items as they are completed.

### Phase 0: Containment (today)
- [x] Restrict network access to ports 6969 and 9696 to localhost, or firewall them (F-01). Done 2026-11-24: server bound to `127.0.0.1:6969` (was `0.0.0.0`); `lsof` confirms localhost-only. Revert to LAN bind only if remote clients are needed.
- [x] Disable or authenticate `PUT /dashboard/config` (F-01). Done 2026-11-24.
- [x] Verify and rotate the default `sathya` password (F-09). Done 2026-11-24: fresh store created, `sathya`/`super_user` set to non-default (`sathya08`); verified old `sathya`/`sathya` returns Invalid credentials, new creds authenticate.
- [x] Check whether the 28 Oct system password still works, and rotate it if so (F-07). Superseded 2026-11-24: old store (and its SYSTEM password + 4 log lines) deleted with the Storage reset; fresh store has a new SYSTEM user. Purge old log backups if any were copied elsewhere.
- [x] Set `nebulondb.cfg` to mode 600 and `Storage/Secrets` to owner-only access (F-05). Done 2026-11-24: `chmod 600/700` applied + `_check_secure_permissions` warning added.
- [x] Back up `Storage/` before any further change. Done 2026-11-24: automated `ndb_host/tools/backup_storage.py` (`--keep 7` retention into `backups/`; integrity-verified). Superseded manual tar removed; 2 fresh backups retained.
- [x] Plan the master-key rotation (F-05). Done 2026-11-24: verified zero stored blobs depend on the key (no SafeLocker containers; grep clean), generated fresh Fernet key into cfg (atomic write, 600), restarted server, `sathya` auth verified. Old (upstream-public) key is dead.

### Phase 1: Authorization and input validation (days)
- [x] Make `/auth/register` admin-only, and refuse `system` and `super_user` roles through the API (F-02). Verified 2026-11-24: `auth.py` requires ADMIN_USER, forbids SYSTEM/SUPER_USER, creator must outrank new role. Static checks pass.
- [x] Make `/auth/delete_user` admin-only, and forbid deleting higher-ranked accounts (F-03). Verified 2026-11-24: `auth.py` requires ADMIN_USER, blocks equal/higher rank + self-delete. Static checks pass.
- [x] Allow-list corpus names, and check that resolved paths stay inside storage (F-04). Verified 2026-11-24: `CORPUS_NAME_PATTERN ^[A-Za-z0-9_-]{1,64}$` in `constants.py`/`models.py` + `index_manager._safe_corpus_path`. Static regex + path tests pass.
- [x] Disable or authenticate `PUT /dashboard/config` (F-01). Verified 2026-11-24: `dashboard.py` GET requires auth, PUT requires SYSTEM + allow-list (segments/vector/rank only), newline rejection. Static checks pass.
- [x] Harden companions: deny-by-default permissions (L-04) and delete-user failure propagation (F-18). Static checks pass.
- [x] Add live API tests for each of the above — done 2026-11-24 on fresh Mac store: auth 8, system 4, corpus 30, segment 33, bulk 4 (live 79 passed); durability + server-config 31, powerloss-oracle/scheduler 22, mac-SIGKILL 2 (isolated all green).

### Phase 2: Data safety (days to weeks)
- [x] Remove the `rmtree` from the bootstrap error path (F-06). Done 2026-11-24: `bootstrap.py` exits without deleting store.
- [x] Stop logging generated passwords, and check return values in bootstrap (F-07). Done 2026-11-24: outcome-only logs, return-value checks.
- [x] Change compaction to quarantine instead of delete on read errors, and log skipped record counts (F-08). Done 2026-11-24: abort on unreadable segment (0 deletes), quarantine all-undecodable set, `skipped_records` counter.
- [x] Run a manifest-versus-disk check, then guard `read_all` against missing files (F-10). Done 2026-11-24 (guard): `cosmos/store.py read_all` skips missing/unreadable segments with warning. Checker `ndb_host/tools/check_manifest.py` added and run clean (2/2 Cosmos stores consistent, exit 0).
- [x] Propagate fsync errors in the Nova WAL (F-17). Done 2026-11-24: `_wal_fsync` no longer suppresses `OSError`; counter reset only on success.
- [x] Return an error when user deletion fails (F-18). Done 2026-11-24 (criticals patch).

### Phase 3: Configuration and operations (weeks)
- [x] Make config reads and writes separate, with atomic writes that preserve comments (F-13). Done 2026-11-24: `_write` via tmp+`os.replace`+600; `_load_llm` no longer rewrites cfg for missing cache dir. Note: comments still not preserved (ConfigParser limitation) — runtime overrides now minimal.
- [x] Replace hard-coded paths with portable ones (F-14). Done 2026-11-24: missing `NEBULONDB_HOME` falls back to config-file dir with warning instead of wrong cwd.
- [x] Make the config a single shared object (F-15). Done 2026-11-24: per-path singleton + `reload()`; `db.ndb_settings`/`ndb_host.db.ndb_settings` module alias; dashboard PUT calls `reload()` in place.
- [x] Restrict CORS to an explicit origin list (F-11). Done 2026-11-24: explicit localhost origins, credentials off unless `NEBULONDB_CORS_ORIGINS` set.
- [x] Require authentication for, or restrict, the public config read (F-12). Done 2026-11-24 (criticals): `GET /config` requires auth.
- [x] Enforce log retention for every file-name format (F-16). Done 2026-11-24: `_cleanup_old` matches `nebulondb_*.log`, keeps newest N by mtime, runs at startup + rollover.
- [x] Correct the HTTP status codes and update the tests (L-01). Done 2026-11-24: register 401/403/400/409 via Response, delete 401/403/404, change-pw 401/400; `test_auth.py` duplicate now expects 409. Note: verify endpoint stays 200 by design.
- [x] Audit the storage-engine broad exception handlers (F-19). Done 2026-11-24 (full): all 119 `except Exception` triaged. Data-path silent sites fixed — `store`/`segment_writer`/`segment_reader`/`index` (rebuild skip counter in final log)/`index_manager` (8 getters)/`mesh_engine` now debug-log; compactor/WAL already count+quarantine. Verified remaining sites surface errors (route failure responses, rollback+reraise in orbit batch paths, error-collecting lists, timing fallback) or are cosmetic (TUI, logger internals, viz). Isolated suites green after the change.

### Phase 4: Verification
- [x] Isolated config tests pass on Mac (2026-11-24): `uv sync --python 3.10` slim core, `pytest unittest/test_server_config.py` 8 passed; runtime checks (singleton, regex, CORS, deny-by-default) pass. Note: `nebulondb.cfg` was rewritten for Mac (`nebulondb_home` now Mac path, comments dropped by ConfigParser — back it up before WSL use).
- [x] Fixed `conftest.py` import-time crash (2026-11-24): missing live creds now skip live tests instead of breaking isolated runs.
- [x] Fresh-store live run (2026-11-24, `sathya`/`super_user`): `test_auth` 8 passed, `test_system` 4 passed (updated for F-12 auth), `test_corpus` 30 passed (3 updated for F-04 422). `test_cosmos_durability` + `test_server_config` 31 passed (incl. missing-files convergence after compactor fix).
- [x] ML suites green after `uv sync --extra ml-cpu` (2026-11-24): `test_segment` 33 passed, `test_bulk_client` 4 passed. Tally at that point — live 79 passed, isolated 31 passed (durability + server-config).
- [x] Power-loss unit tests pass on Mac (2026-11-24): `test_powerloss_oracle` + `test_powerloss_scheduler` 22 passed (no root/devices needed).
- [x] Mac crash-recovery port added (2026-11-24): `test_powerloss_mac.py` (2 passed) — real child-process inserts + parent SIGKILL + WAL-replay recovery assert for `per_ack`/`process` modes. Covers OS-level crash consistency; true plug-pull loss still needs block-layer WSL.
- [x] Post F-19-audit regression (2026-11-24): live 75 passed (auth/system/corpus/segment), isolated 55 passed, manifest checker clean (2/2). No behavior change — debug logging only.
- [x] Seeded-data regression (2026-11-26): live 75 passed (auth 8, system 4, corpus 30, segment 33), isolated 59 passed (durability 23, server-config 8, bulk-client 4, power-loss 24: oracle/scheduler 22 + mac-SIGKILL 2). Seed users/corpus (`test_user`, `nmd_user_01`, `alice`, `seed_demo`) verified working; tokens in /tmp/md_seed_tokens.env (600).
- [ ] Power-loss block-layer integration (`test_powerloss_wsl.py`, `run.sh` 20-trials): Linux-only — needs root + `dmsetup/losetup/mkfs.ext4/e2fsck/blockdev`. Run on the WSL box with `NDB_POWERLOSS=1` as root.

### Phase 5: macOS power-loss port (later)
Context: WSL harness needs device-mapper/loop/ext4 (absent on macOS; `diskutil apfs` has no revert verb), so block-layer rollback can't be ported. `test_powerloss_mac.py` (SIGKILL + WAL replay) already covers crash consistency.
- [ ] B1 — torn-write fault-injection matrix: after SIGKILL, apply seeded faults (truncate WAL mid-record, drop trailing bytes, flip mid-sector bytes, delete segment file, drop manifest entry), then reopen and assert ACKed-prefix intact, no resurrection, no crash, quarantine populated, skip-counts logged. Modes × fault types × trials, seeded RNG.
- [ ] B2 — platform-aware fsync helper: `F_FULLFSYNC` on darwin (macOS `fsync` doesn't flush drive cache), `fsync`/`fdatasync` on Linux; route all WAL paths through it; re-run B1 + WSL trials.
- [ ] B3 — document split: Mac suite = crash + torn-write recovery (APFS); WSL suite = block-layer power-cut truth (ext4).
- [ ] Update the test report, and record what was verified.

---

## 8. Open questions for the owner

> Update 2026-11-24 (evening): all code fixes applied and verified on Mac — live 79 passed, isolated 55 passed (31 durability/config + 22 powerloss unit + 2 mac-SIGKILL). Store reset with non-default `sathya` creds; server bound to localhost; master key rotated; manifest checker clean. Still open: WSL block-layer power-loss, Phase 5 Mac fault-injection port.

1. Is port 6969 reachable from other machines? — Answered 2026-11-24: No. Server binds `127.0.0.1:6969` only (`lsof` verified). Re-check if the bind ever changes.
2. Does the `sathya` account still accept the default password? — Answered 2026-11-24: No. Old store wiped; fresh `sathya`/`super_user` with non-default password; default returns Invalid credentials.
3. Is the 28 Oct system password still valid? — Answered 2026-11-24: moot. That user store was deleted with the reset; new SYSTEM user created.
4. Is the powerloss daemon running, and with what configuration? — Answered 2026-11-24: No. `[powerloss] enabled=false`, no daemon process; harness is WSL-only anyway.
5. Do any clients call the dashboard config endpoints? — Answered 2026-11-24: Yes. `NebulonMD/.../main.js:1465` PUTs `{llm:...}` (now 403 by design — needs a Mind-side change), and `test_system.py` GETs (updated to send auth).

---

## Appendix A: Files referenced

- `ndb_host/api/routes/auth.py`, `dashboard.py`, `corpus.py`, `segment.py`
- `ndb_host/services/user_service.py`
- `ndb_host/core/security.py`, `ndb_host/core/permissions.py`
- `ndb_host/db/index_manager.py`, `ndb_host/db/ndb_settings.py`
- `ndb_host/db/engine/cosmos/store.py`, `compactor.py`, `segment_reader.py`, `segment_writer.py`, `wal.py`
- `ndb_host/db/engine/orbit/orchestrator.py`, `nova_store.py`
- `ndb_host/tui/bootstrap.py`, `server_ops.py`
- `ndb_host/utils/logger.py`, `models.py`, `constants.py`
- `ndb_host/api/__init__.py`, `ndb_host/client/many_docs2.py`
- `nebulondb.cfg`, `pyproject.toml`, `requirements.txt`
- `nebulondb.cfg.example` (sanitized template added 2026-11-24)
- `unittest/conftest.py`, `unittest/test_auth.py`, `unittest/TEST_REPORT.md`
- `unittest/test_system.py`, `unittest/test_corpus.py` (updated 2026-11-24)
- `unittest/test_powerloss_mac.py`, `unittest/_powerloss_mac_child.py` (added 2026-11-24)
- `ndb_host/tools/check_manifest.py`, `ndb_host/tools/backup_storage.py` (added 2026-11-24)
