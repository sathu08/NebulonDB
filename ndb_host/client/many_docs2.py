"""
many_docs2.py – pipeline‑bounded MANY operations for NebulonDB.

Uploads (add_many), updates (update_many) and deletes (delete_many) are all
piped through :class:`client.pipeline.BoundedPipeline` so the server is hit
with batches of ``--batch`` records while memory stays bounded by ``--queue``.

Usage
-----
    python -m client.many_docs2 upload   corpus segment docs.json [--text title] [--batch 64]
    python -m client.many_docs2 update   corpus segment docs.json --include-ids
    python -m client.many_docs2 delete   corpus segment --ids 1,2,3,4,5
    python -m client.many_docs2 delete   corpus segment docs.json --id-field id

Credentials default to the ``NEBULONDB_USER`` / ``NEBULONDB_PASSWORD``
environment variables (fallback ``sathya`` / ``sathya``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import requests

from .pipeline import BoundedPipeline
from .schema import ParallelRecord, make_records

BASE_URL = os.environ.get("NEBULONDB_URL", "http://127.0.0.1:6969/api/NebulonDB")
DEFAULT_USER = os.environ.get("NEBULONDB_USER", "sathya")
DEFAULT_PASSWORD = os.environ.get("NEBULONDB_PASSWORD", "sathya")


class NebulonClient:
    def __init__(self, base_url: str = BASE_URL,
                 username: str = DEFAULT_USER,
                 password: str = DEFAULT_PASSWORD,
                 timeout: float = 300.0,
                 session: Any | None = None) -> None:
        self.base = base_url.rstrip("/")
        self.auth = (username, password)
        self.timeout = timeout
        # ``session`` may be a requests.Session (network) or a Starlette
        # TestClient (in-process) for tests. Both accept ``post(url, json=...)``.
        self._session = session if session is not None else requests.Session()
        self._is_test_client = not hasattr(self._session, "close")

    def _post(self, path: str, payload: dict) -> dict:
        kwargs: dict[str, Any] = {"json": payload, "auth": self.auth}
        if not self._is_test_client:
            kwargs["timeout"] = self.timeout
        r = self._session.post(f"{self.base}{path}", **kwargs)
        try:
            body = r.json()
        except ValueError:
            body = {"message": getattr(r, "text", str(r))[:500]}
        return {"status": getattr(r, "status_code", 200), **body}

    # -- single endpoints ------------------------------------------------
    def verify(self) -> bool:
        kwargs: dict[str, Any] = {"auth": self.auth}
        if not self._is_test_client:
            kwargs["timeout"] = self.timeout
        r = self._session.get(f"{self.base}/auth/verify", **kwargs)
        ok_status = bool(getattr(r, "json", None) and r.json().get("user", {}).get("is_authenticated", False))
        return ok_status

    # -- batch endpoints -------------------------------------------------
    def add_many(self, corpus: str, segment: str, records: Iterable[ParallelRecord],
                 batch: int = 64, workers: int = 4, queue: int = 256,
                 progress: bool = True) -> dict:
        return self._run("add_records", corpus, segment, records, batch, workers, queue, progress)

    def update_many(self, corpus: str, segment: str, records: Iterable[ParallelRecord],
                    batch: int = 64, workers: int = 4, queue: int = 256,
                    progress: bool = True) -> dict:
        return self._run("update_records", corpus, segment, records, batch, workers, queue, progress)

    def delete_many(self, corpus: str, segment: str, ids: Iterable[int],
                    batch: int = 64, workers: int = 4, queue: int = 256,
                    progress: bool = True) -> dict:
        totals = {"deleted": 0, "missing": 0}

        def transform(rid: object) -> int:
            return int(rid)

        def sink(batch: list[int]) -> None:
            resp = self._post(
                "/segment/delete_records",
                {"corpus_name": corpus, "segment_name": segment,
                 "record_ids": batch},
            )
            if resp.get("success") is not True:
                raise RuntimeError(f"delete_batch failed: {resp.get('message')}")
            data = resp.get("data") or {}
            if isinstance(data, dict):
                for key in totals:
                    val = data.get(key)
                    if isinstance(val, int):
                        totals[key] += val
                    elif isinstance(val, (list, tuple)):
                        totals[key] += len(val)

        pipe = BoundedPipeline(
            sink=sink, transform=transform,
            batch_size=batch, workers=workers, max_queue=queue,
            progress=(lambda n: print(f"\rdelete: {n}", end="", flush=True)) if progress else None,
        )
        pipe.feed(ids)
        if progress:
            print()
        result = self._summary("delete_records", pipe)
        result.update(totals)
        return result

    # -- shared runner ---------------------------------------------------
    def _run(self, endpoint: str, corpus: str, segment: str,
             records: Iterable[ParallelRecord], batch: int, workers: int,
             queue: int, progress: bool) -> dict:
        totals = {"inserted": 0, "updated": 0, "deleted": 0, "skipped": 0}
        sink_errors: list[str] = []

        def transform(rec: ParallelRecord) -> dict:
            p = rec.payload()
            p.setdefault("metadata", {})
            return p

        def sink(batch: list[dict]) -> None:
            resp = self._post(
                f"/segment/{endpoint}",
                {"corpus_name": corpus, "segment_name": segment, "records": batch},
            )
            if resp.get("success") is not True:
                raise RuntimeError(f"{endpoint} failed: {resp.get('message')}")
            data = resp.get("data") or {}
            if isinstance(data, dict):
                for key in totals:
                    val = data.get(key)
                    if isinstance(val, int):
                        totals[key] += val
                    elif isinstance(val, (list, tuple)):
                        totals[key] += len(val)
            errs = resp.get("errors")
            if isinstance(errs, list):
                sink_errors.extend(str(e) for e in errs)

        pipe = BoundedPipeline(
            sink=sink, transform=transform,
            batch_size=batch, workers=workers, max_queue=queue,
            progress=(lambda n: print(f"\r{endpoint}: {n}", end="", flush=True)) if progress else None,
        )
        pipe.feed(records)
        if progress:
            print()
        result = self._summary(endpoint, pipe)
        result.update(totals)
        result["errors"] = pipe.errors + sink_errors
        return result

    @staticmethod
    def _summary(op: str, pipe: BoundedPipeline) -> dict:
        return {"operation": op, "sent": pipe.sent, "errors": pipe.errors}


def _load_docs(path: Path) -> dict[str, list[Any]] | list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and all(isinstance(v, list) for v in data.values()):
        return data
    if isinstance(data, list) and all(isinstance(i, dict) for i in data):
        return data
    raise ValueError("docs.json must be a dict-of-lists or a list-of-dicts")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="many_docs2", description=__doc__)
    p.add_argument("action", choices=["upload", "update", "delete"])
    p.add_argument("corpus")
    p.add_argument("segment")
    p.add_argument("docs", nargs="?", help="JSON doc path (upload/update) or delete-by-id file")
    p.add_argument("--text", action="append", default=[], help="column(s) to treat as text")
    p.add_argument("--id-field", default="record_id", help="field holding record ids in docs (delete)")
    p.add_argument("--ids", default=None, help="comma-separated ids for delete")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--queue", type=int, default=256)
    p.add_argument("--url", default=BASE_URL)
    p.add_argument("--user", default=DEFAULT_USER)
    p.add_argument("--password", default=DEFAULT_PASSWORD)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    client = NebulonClient(base_url=args.url, username=args.user, password=args.password)
    if not client.verify():
        print(f"ERROR: auth failed for '{args.user}' at {args.url}", file=sys.stderr)
        return 2

    progress = not args.quiet

    if args.action == "delete":
        if args.ids:
            ids: Iterable[int] = (int(x) for x in args.ids.split(",") if x)
        elif args.docs:
            docs = _load_docs(Path(args.docs))
            if isinstance(docs, dict):
                col = args.id_field if args.id_field in docs else next(iter(docs))
                ids = (int(x) for x in docs[col] if x is not None)
            else:
                ids = (int(v) for d in docs
                       if (v := d.get(args.id_field)) is not None)
        else:
            print("ERROR: delete requires --ids or a docs file", file=sys.stderr)
            return 2
        result = client.delete_many(args.corpus, args.segment, ids,
                                    batch=args.batch, workers=args.workers, queue=args.queue,
                                    progress=progress)
    else:
        if not args.docs:
            print("ERROR: upload/update require a docs file", file=sys.stderr)
            return 2
        docs = _load_docs(Path(args.docs))
        records = make_records(docs, text_columns=args.text) if args.action == "upload" \
            else make_records(docs, text_columns=[])
        if args.action == "upload":
            result = client.add_many(args.corpus, args.segment, records,
                                     batch=args.batch, workers=args.workers, queue=args.queue,
                                     progress=progress)
        else:
            records = [r for r in records if r.record_id is not None]
            result = client.update_many(args.corpus, args.segment, records,
                                        batch=args.batch, workers=args.workers, queue=args.queue,
                                        progress=progress)

    print(json.dumps(result, indent=2))
    return 0 if not result.get("errors") else 1


if __name__ == "__main__":
    raise SystemExit(main())