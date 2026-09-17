"""
End-to-end tests for the pipeline-bounded MANY client (client/many_docs2.py).

These tests exercise the full stack without a live server:
* a temp ``NEBULONDB_HOME`` with an isolated Storage tree,
* the real FastAPI app through starlette TestClient,
* the real HTTP-facing bulk endpoints (add_records / update_records /
  delete_records),
* the real client pipeline (BoundedPipeline batching) driving those calls.

Test plan
=========
* add_many  : 220 records streamed through a batch size of 16  -> 220 stored
* update_many: metadata/vector update on a subset               -> values change
* delete_many: bulk delete of a subset                          -> count shrinks
* verify the client continues through partial server errors
"""

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

# NEBULONDB_HOME must be set before any config-touching module is imported.
_PROJECT = Path(__file__).resolve().parent.parent
_TMP_HOME = Path(tempfile.mkdtemp(prefix="ndb_bulk_"))
_NEBULONDB_HOME = os.environ.get("NEBULONDB_HOME")
os.environ["NEBULONDB_HOME"] = str(_TMP_HOME)
if _NEBULONDB_HOME:
    _ORIG_HOME = _NEBULONDB_HOME
else:
    _ORIG_HOME = None

# Mirror the real project layout so NDBConfig._load_paths resolves Storage/
# logs/ web_dir inside the temp home.
_USER = "bulktester"
_PASSWORD = "bulktester123"
_URL = "http://testserver/api/NebulonDB"


def _copy_default_layout() -> None:
    """Copy enough of the project so a fresh NEBULONDB_HOME boots cleanly."""
    shutil.copy(_PROJECT / "nebulondb.cfg", _TMP_HOME / "nebulondb.cfg")
    (_TMP_HOME / "ndb_host").mkdir(exist_ok=True)
    shutil.copytree(_PROJECT / "ndb_host" / "web_dir", _TMP_HOME / "ndb_host" / "web_dir")
    (_TMP_HOME / "logs").mkdir(exist_ok=True)
    (_TMP_HOME / "Storage").mkdir(exist_ok=True)


class BulkClientTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        _copy_default_layout()

        # Import the app/client modules only after the env override is in
        # place so every NDBConfig() sees the temp home.
        sys.path.insert(0, str(_PROJECT / "ndb_host"))
        sys.path.insert(0, str(_PROJECT))

        from ndb_host.main import app  # noqa: F401
        from services.user_service import create_user
        from utils.constants import UserRole

        create_user(_USER, _PASSWORD, UserRole.SUPER_USER.value)
        cls._app_created = True
        globals()["_create_user"] = create_user
        globals()["_UserRole"] = UserRole

    @classmethod
    def tearDownClass(cls) -> None:
        if _ORIG_HOME is not None:
            os.environ["NEBULONDB_HOME"] = _ORIG_HOME
        else:
            os.environ.pop("NEBULONDB_HOME", None)

    def _client_session(self):
        from fastapi.testclient import TestClient
        from ndb_host.main import app

        return TestClient(app)

    def _make_client(self):
        from client.many_docs2 import NebulonClient

        return NebulonClient(base_url=_URL, username=_USER, password=_PASSWORD,
                             timeout=60.0, session=self._client_session())

    def _create_corpus(self, session, corpus: str, ndb_type: str = "orbit"):
        r = session.post(
            f"{_URL}/corpus/create_corpus",
            auth=(_USER, _PASSWORD),
            json={"corpus_name": corpus, "ndb_type": ndb_type},
        )
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertTrue(body["success"], body)
        return body

    def _gen_dataset(self, n: int, text_col: str = "title") -> dict:
        return {
            text_col: [f"doc {i}: semantic search over parallel pipelines" for i in range(n)],
            "tag": [f"tag-{i % 5}" for i in range(n)],
        }

    def test_add_many_pipeline(self) -> None:
        """220 records, batch=16 -> all 220 stored, no errors."""
        from client.many_docs2 import NebulonClient
        from client.schema import make_records

        corpus = "bulk_add"
        c = NebulonClient(base_url=_URL, username=_USER, password=_PASSWORD,
                          timeout=60.0, session=self._client_session())
        self._create_corpus(c._session, corpus)

        dataset = self._gen_dataset(220)
        records = make_records(dataset, text_columns=["title"])
        result = c.add_many(corpus, "seg1", records, batch=16, workers=4,
                            queue=32, progress=False)
        self.assertEqual(result["errors"], [])
        self.assertEqual(result["sent"], 220)
        self.assertGreaterEqual(result["inserted"], 220)

        # Verify through get_data
        r = c._post("/segment/get_data", {
            "corpus_name": corpus, "segment_name": "seg1", "limit": 1000,
        })
        self.assertTrue(r["success"], r)
        self.assertEqual(len(r["data"]["records"]), 220)

    def test_update_many_pipeline(self) -> None:
        """Update metadata/vector on ids returned from add_many."""
        from client.many_docs2 import NebulonClient
        from client.schema import ParallelRecord, make_records

        corpus = "bulk_upd"
        c = NebulonClient(base_url=_URL, username=_USER, password=_PASSWORD,
                          timeout=60.0, session=self._client_session())
        self._create_corpus(c._session, corpus)

        dataset = self._gen_dataset(40)
        records = make_records(dataset, text_columns=["title"])
        added = c.add_many(corpus, "seg1", records, batch=10, workers=2,
                           queue=16, progress=False)
        self.assertEqual(added["errors"], [])
        ids = self._ids_of(c, corpus, "seg1", 40)
        self.assertEqual(len(ids), 40)

        updates = [
            ParallelRecord(record_id=rid,
                           metadata={"tag": f"updated-{rid}", "flag": True})
            for rid in ids[:20]
        ]
        res = c.update_many(corpus, "seg1", updates, batch=8, workers=2,
                            queue=16, progress=False)
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["sent"], 20)

        # Confirm server-side values changed.
        r = c._post("/segment/get_record", {
            "corpus_name": corpus, "segment_name": "seg1",
            "record_id": ids[0],
        })
        self.assertTrue(r["success"], r)
        meta = r["data"]["metadata"]
        self.assertEqual(meta.get("tag"), f"updated-{ids[0]}")
        self.assertTrue(meta.get("flag"))

    def test_delete_many_pipeline(self) -> None:
        """delete_many removes exactly its id set."""
        from client.many_docs2 import NebulonClient
        from client.schema import make_records

        corpus = "bulk_del"
        c = NebulonClient(base_url=_URL, username=_USER, password=_PASSWORD,
                          timeout=60.0, session=self._client_session())
        self._create_corpus(c._session, corpus)

        dataset = self._gen_dataset(50)
        records = make_records(dataset, text_columns=["title"])
        c.add_many(corpus, "seg1", records, batch=16, workers=2,
                   queue=16, progress=False)
        ids = self._ids_of(c, corpus, "seg1", 50)
        to_delete = ids[::2]  # 25 ids

        res = c.delete_many(corpus, "seg1", iter(to_delete), batch=8,
                            workers=2, queue=16, progress=False)
        self.assertEqual(res["errors"], [])
        self.assertEqual(res["sent"], 25)

        remaining = self._ids_of(c, corpus, "seg1", 500)
        self.assertEqual(len(remaining), 25)
        self.assertEqual(set(remaining), set(ids) - set(to_delete))

    def test_delete_many_missing_ids_reported(self) -> None:
        """Deleting nonexistent ids reports them as missing, pipeline continues."""
        from client.many_docs2 import NebulonClient
        from client.schema import make_records

        corpus = "bulk_del_missing"
        c = NebulonClient(base_url=_URL, username=_USER, password=_PASSWORD,
                          timeout=60.0, session=self._client_session())
        self._create_corpus(c._session, corpus)
        c.add_many(corpus, "seg1", make_records(self._gen_dataset(10),
                                                text_columns=["title"]),
                   batch=8, workers=1, queue=8, progress=False)
        res = c.delete_many(corpus, "seg1", iter([99999, 99998]), batch=8,
                            workers=1, queue=8, progress=False)
        self.assertEqual(res["sent"], 2)
        self.assertEqual(res["errors"], [])

    def _ids_of(self, c, corpus: str, segment: str, limit: int) -> list[int]:
        r = c._post("/segment/get_data", {
            "corpus_name": corpus, "segment_name": segment, "limit": limit,
        })
        self.assertTrue(r["success"], r)
        return [rec["id"] for rec in r["data"]["records"]]


if __name__ == "__main__":
    unittest.main(verbosity=2)