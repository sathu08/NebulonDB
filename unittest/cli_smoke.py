"""End-to-end CLI smoke: upload -> update -> delete against a live isolated server."""
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

PROJECT = Path("/home/sathi/Codebase/NebulonDB")
HOME = Path(tempfile.mkdtemp(prefix="ndb_cli_"))
PORT = 6968

os.environ["NEBULONDB_HOME"] = str(HOME)

shutil.copy(PROJECT / "nebulondb.cfg", HOME / "nebulondb.cfg")
(HOME / "ndb_host").mkdir(exist_ok=True)
shutil.copytree(PROJECT / "ndb_host" / "web_dir", HOME / "ndb_host" / "web_dir")
(HOME / "logs").mkdir(exist_ok=True)
(HOME / "Storage").mkdir(exist_ok=True)

sys.path.insert(0, str(PROJECT / "ndb_host"))
sys.path.insert(0, str(PROJECT))

from services.user_service import create_user
from utils.constants import UserRole

create_user("cli", "clipass123", UserRole.SUPER_USER.value)

# Boot the app on 127.0.0.1:PORT with the isolated home.
env = {**os.environ, "PYTHONPATH": str(PROJECT), "NEBULONDB_HOME": str(HOME)}
server = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "ndb_host.main:app",
     "--host", "127.0.0.1", "--port", str(PORT), "--log-level", "warning"],
    env=env, cwd=str(PROJECT),
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
)

BASE = f"http://127.0.0.1:{PORT}/api/NebulonDB"
URL = f"http://127.0.0.1:{PORT}/api/NebulonDB"


def wait_up(timeout=60):
    import base64
    auth = base64.b64encode(b"cli:clipass123").decode()
    req = urllib.request.Request(f"{URL}/auth/verify", headers={"Authorization": f"Basic {auth}"})
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(req, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError("server did not start in time")


try:
    wait_up()

    # Create the corpus (admin op) so add_records has a target.
    import base64
    auth = base64.b64encode(b"cli:clipass123").decode()
    req = urllib.request.Request(
        f"{URL}/corpus/create_corpus",
        data=json.dumps({"corpus_name": "cli_corpus", "ndb_type": "orbit"}).encode(),
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = json.loads(resp.read())
    assert body["success"], body

    docs = {
        "title": [f"cli doc {i}: pipeline bounded ingest for NebulonDB" for i in range(30)],
        "tag": [f"c{i % 3}" for i in range(30)],
    }
    docs_path = HOME / "docs.json"
    docs_path.write_text(json.dumps(docs))

    cli_env = {**os.environ, "PYTHONPATH": str(PROJECT) + ":ndb_host"}

    def run(*args):
        r = subprocess.run(
            [sys.executable, "-m", "client.many_docs2", *args],
            capture_output=True, text=True, env=cli_env, cwd=str(PROJECT), timeout=300,
        )
        print("CMD", args)
        print("RC", r.returncode)
        print("OUT", r.stdout[-600:])
        print("ERR", r.stderr[-300:])
        return r.returncode, r.stdout

    code, out = run("upload", "cli_corpus", "seg1", str(docs_path), "--text", "title",
                    "--batch", "8", "--workers", "2", "--queue", "16",
                    "--user", "cli", "--password", "clipass123", "--quiet",
                    "--url", URL)
    assert code == 0, "upload failed"
    loaded = json.loads(out)
    assert loaded["sent"] == 30, loaded
    assert loaded["inserted"] == 30, loaded

    upd_rows = [{"record_id": i, "tag": f"upd{i}"} for i in range(1, 31)]
    upd_path = HOME / "upd.json"
    upd_path.write_text(json.dumps(upd_rows))

    code, out = run("update", "cli_corpus", "seg1", str(upd_path),
                    "--batch", "8", "--workers", "2", "--queue", "16",
                    "--user", "cli", "--password", "clipass123", "--quiet",
                    "--url", URL)
    assert code == 0, "update failed"
    loaded = json.loads(out)
    assert loaded["sent"] == 30, loaded
    assert loaded["updated"] == 30, loaded

    code, out = run("delete", "cli_corpus", "seg1", str(upd_path),
                    "--id-field", "record_id",
                    "--batch", "8", "--workers", "2", "--queue", "16",
                    "--user", "cli", "--password", "clipass123", "--quiet",
                    "--url", URL)
    assert code == 0, "delete failed"
    loaded = json.loads(out)
    assert loaded["sent"] == 30, loaded
    assert loaded["deleted"] == 30, loaded

    print("CLI_SMOKE_OK")
finally:
    server.terminate()
    try:
        server.wait(timeout=10)
    except Exception:
        server.kill()