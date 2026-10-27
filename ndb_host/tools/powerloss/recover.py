"""
Recovery reader for the WSL power-loss harness.

Opens the engine on a snapshot of the crashed filesystem, lets normal WAL
recovery run, and writes the recovered ``seq -> text`` state as JSON.  The
process exits with ``os._exit`` so it cannot hide a crash by flushing metadata
on interpreter shutdown.
"""

from __future__ import annotations

import os
import json
import argparse

import traceback

from pathlib import Path

from ndb_host.db.engine import NebulonCosmos
from ndb_host.db.ndb_settings import NDBConfig

# ==========================================================
#        Configuration
# ==========================================================

SEGMENT = NDBConfig().POWERLOSS_SEGMENT


# ==========================================================
#        Helpers
# ==========================================================

def _write_json(path: Path, payload: dict) -> None:
    data = json.dumps(payload, separators=(",", ":")).encode()
    with path.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


# ==========================================================
#        Main
# ==========================================================

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    out = Path(args.out)
    try:
        store = NebulonCosmos(Path(args.db_dir))
        docs = store.read_all(segment=SEGMENT)
        records = [
            {"seq": doc.get("seq"), "text": doc.get("text")}
            for doc in docs
        ]
        _write_json(out, {"ok": True, "records": records, "count": len(records)})
        os._exit(0)
    except BaseException as exc:  # noqa: BLE001 - the harness reports this
        _write_json(out, {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "records": [],
        })
        os._exit(1)


if __name__ == "__main__":
    raise SystemExit(main())
