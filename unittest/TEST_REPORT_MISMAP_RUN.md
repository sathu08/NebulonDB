# NebulonDB API Test Report — Mismap Fix Verification Run

**Date:** 2026-08-06
**Purpose:** Verification run after fixing the mismapped system route and response-shape mismatch.
**Test target:** `ndb_host/api/routes/` — `auth.py`, `system.py`, `corpus.py`, `segment.py`
**Live API:** `http://127.0.0.1:6969/api/NebulonDB`
**Authentication:** HTTP Basic — `sathya` / `sathya` (from `conftest.py`)
**Runner:** `pytest` (venv `/home/sathyaprakash/venv`), `requests`
**Changes applied:** server `system.py` prefix fix (by user) + `test_system.py` updated to read `body["server"]`

---

## 1. Summary

| Suite | Total | Passed | Failed | Pass % |
|-------|------:|-------:|-------:|-------:|
| test_auth.py    | 8  | 8  | 0 | 100% |
| test_corpus.py  | 30 | 30 | 0 | 100% |
| test_segment.py | 30 | 30 | 0 | 100% |
| test_system.py  | 4  | 4  | 0 | 100% |
| **TOTAL**       | **72** | **72** | **0** | **100%** |

**Result: 72 passed, 0 failed.**

---

## 2. Issues found & fixed

### 2.1 Mismapped router prefix → 404 (fixed on server)

- `ndb_host/api/routes/system.py` had `router = APIRouter(prefix="/system", ...)`
  while `ndb_host/main.py` also mounted it with `prefix="/api/NebulonDB/system"`.
- Only `/api/NebulonDB/system/system/config` existed; `/api/NebulonDB/system/config`
  returned `404`. Other routers had no own prefix, which is why only `system` broke.
- **Fix:** `router = APIRouter()` (prefix removed). Server restarted (gunicorn was
  still running the old code) → `/api/NebulonDB/system/config` now returns `200`.

### 2.2 Response-shape mismatch → KeyError (fixed in test)

- API nests config under sections: `host`/`port`/`url` live under `body["server"]`.
- Old tests read top-level `body["host"]`/`body["url"]` → `KeyError` /
  `assert "host" in body` failed.
- **Fix:** `test_system.py` now asserts against `body["server"]["host"]`,
  `body["server"]["port"]`, `body["server"]["url"]`.

---

## 3. Test file changes (`tests/unittest/test_system.py`)

- `test_config_returns_host_port_url`: checks `"server" in body` and
  `body["server"]["port"] == 6969`.
- `test_config_url_consistent_with_host_port`:
  `body["server"]["url"] == f"http://{body['server']['host']}:{body['server']['port']}"`.
- `test_config_host_is_nonempty`: checks `body["server"]["host"]`.

---

## 4. Passed coverage

- **Auth (8/8):** verify valid/wrong-password/unknown-user/no-credentials, register
  new/duplicate/short-password(422)/no-auth(401).
- **Corpus (30/30):** create/list/activate/deactivate/delete lifecycle, duplicate &
  system-corpus protection, bulk, unicode/long/special names, recreate-after-delete.
- **Segment (30/30):** load variants, list, search (nova/mesh/hybrid/top-k), add node,
  add/remove relation, BFS, shortest path, rank-enabled search, delete record,
  mesh visualization, empty-dataset rejection.
- **System (4/4):** config returns host/port/url, works without auth, url consistent
  with host/port, host non-empty.

---

## 5. Environment notes

- Server restarted on `127.0.0.1:6969` to load the prefix fix.
- Warnings (non-fatal): unknown pytest marks `auth`, `corpus`, `segment`, `system` —
  not registered in any `pytest.ini`/`pyproject.toml`.
- Runtime: 80.3s (72 tests, live API).
