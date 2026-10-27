"""Single-worker enforcement and durability config loading."""

from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT / "ndb_host"))

pytestmark = pytest.mark.isolated


def _temp_home(tmp_path: Path, *, workers: int = 1, mode: str = "process") -> Path:
    home = tmp_path / "home"
    home.mkdir()
    text = (PROJECT / "nebulondb.cfg").read_text()
    text = re.sub(r"(?m)^WORKERS\s*=.*$", f"WORKERS = {workers}", text)
    text = re.sub(r"(?m)^durability_mode\s*=.*$", f"durability_mode = {mode}", text)
    (home / "nebulondb.cfg").write_text(text)
    (home / "ndb_host").mkdir()
    shutil.copytree(PROJECT / "ndb_host" / "web_dir", home / "ndb_host" / "web_dir")
    (home / "logs").mkdir()
    (home / "Storage").mkdir()
    return home


def test_workers_are_forced_to_one(tmp_path, monkeypatch):
    home = _temp_home(tmp_path, workers=4)
    monkeypatch.setenv("NEBULONDB_HOME", str(home))

    from db.ndb_settings import NDBConfig

    assert NDBConfig().WORKERS == 1


def test_durability_mode_loaded_from_config(tmp_path, monkeypatch):
    home = _temp_home(tmp_path, mode="per_ack")
    monkeypatch.setenv("NEBULONDB_HOME", str(home))

    from db.ndb_settings import NDBConfig

    assert NDBConfig().DURABILITY_MODE == "per_ack"


def test_powerloss_settings_loaded_from_config(tmp_path, monkeypatch):
    home = _temp_home(tmp_path)
    cfg_path = home / "nebulondb.cfg"
    text = cfg_path.read_text()
    text = text.replace("enabled = false", "enabled = true", 1)
    text = text.replace("interval_seconds = 86400", "interval_seconds = 120", 1)
    text = text.replace("modes = process,batch,per_ack", "modes = batch", 1)
    text = text.replace("run_root = /var/tmp/ndb_powerloss",
                        f"run_root = {tmp_path / 'pl'}", 1)
    cfg_path.write_text(text)
    monkeypatch.setenv("NEBULONDB_HOME", str(home))

    from db.ndb_settings import NDBConfig
    from utils.constants import NDBMeta

    config = NDBConfig()
    assert config.POWERLOSS_ENABLED is True
    assert config.POWERLOSS_INTERVAL_SECONDS == 120
    assert config.POWERLOSS_MODES == ("batch",)
    assert config.POWERLOSS_HARNESS_RUN_ROOT == str(tmp_path / "pl")
    assert config.POWERLOSS_HARNESS_RUN_DIR_FORMAT == "run_%Y%m%d_%H%M%S"
    assert config.POWERLOSS_HARNESS_SYSCTLS["dirty_ratio"] == 95
    assert config.POWERLOSS_WORKER_DEFAULT_WAL_FSYNC_INTERVAL == 65536
    assert config.POWERLOSS_SENTINEL_SEQ == NDBMeta.PowerLoss.SENTINEL_SEQ
    assert (config.POWERLOSS_HARNESS_DM_REAL
            == NDBMeta.PowerLoss.Harness.DM_REAL)
    assert (config.POWERLOSS_WORKER_IDLE_SLEEP
            == NDBMeta.PowerLoss.Worker.IDLE_SLEEP)


def test_invalid_durability_mode_is_rejected(tmp_path, monkeypatch):
    home = _temp_home(tmp_path, mode="eventual")
    monkeypatch.setenv("NEBULONDB_HOME", str(home))

    from db.ndb_settings import NDBConfig

    with pytest.raises(ValueError):
        NDBConfig()


def test_create_app_rejects_multiple_workers(monkeypatch):
    from api import create_app

    monkeypatch.setenv("WEB_CONCURRENCY", "2")
    with pytest.raises(RuntimeError):
        create_app()


def test_create_app_rejects_cli_multiple_workers(monkeypatch):
    import sys

    from api import create_app

    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.delenv("GUNICORN_CMD_ARGS", raising=False)
    monkeypatch.setattr(sys, "argv", ["gunicorn", "ndb_host.main:app", "--workers", "2"])
    with pytest.raises(RuntimeError):
        create_app()


def test_create_app_allows_one_worker(monkeypatch):
    from api import create_app

    monkeypatch.delenv("WEB_CONCURRENCY", raising=False)
    monkeypatch.setenv("GUNICORN_CMD_ARGS", "--workers 1")
    assert create_app() is not None


def test_wal_failure_maps_to_503(monkeypatch):
    monkeypatch.setenv("NEBULONDB_HOME", str(PROJECT))

    from fastapi import HTTPException

    from api.routes.segment import _raise_durability
    from db.engine import WalDurabilityError

    with pytest.raises(HTTPException) as exc:
        _raise_durability(WalDurabilityError("simulated disk failure"))
    assert exc.value.status_code == 503

    # Ordinary errors are left to the route's existing handler.
    assert _raise_durability(ValueError("not a WAL error")) is None
