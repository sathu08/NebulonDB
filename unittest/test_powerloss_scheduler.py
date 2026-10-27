"""Unit tests for the power-loss scheduler (no root/devices required)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from ndb_host.tools.powerloss import scheduler

pytestmark = pytest.mark.isolated


def _config(tmp_path: Path, **overrides) -> SimpleNamespace:
    values = {
        "POWERLOSS_ENABLED": True,
        "POWERLOSS_INTERVAL_SECONDS": 100,
        "POWERLOSS_MODES": ("process", "batch", "per_ack"),
        "POWERLOSS_WORKLOADS": ("insert", "bulk", "mixed"),
        "POWERLOSS_NO_SYSCTL": False,
        "POWERLOSS_HARNESS_RUN_ROOT": str(tmp_path),
        "POWERLOSS_STATE_FILE": ".powerloss_state.json",
        "POWERLOSS_LOCK_FILE": ".powerloss.lock",
        "POWERLOSS_KEEP_RUNS": 7,
        "POWERLOSS_POLL_SECONDS": 60,
        "NEBULONDB_HOME": tmp_path,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_not_due_when_disabled():
    config = _config(Path("/tmp"), POWERLOSS_ENABLED=False)
    assert scheduler.is_due(config, {"last_started": 0}, now=10_000) is False


def test_due_when_never_run():
    config = _config(Path("/tmp"))
    assert scheduler.is_due(config, {}, now=10_000) is True


def test_not_due_before_interval(tmp_path):
    config = _config(tmp_path)
    assert scheduler.is_due(config, {"last_started": 9_950}, now=10_000) is False


def test_due_after_interval(tmp_path):
    config = _config(tmp_path)
    assert scheduler.is_due(config, {"last_started": 9_900}, now=10_000) is True


def test_build_command_includes_config_values(tmp_path):
    config = _config(
        tmp_path,
        POWERLOSS_MODES=("batch",),
        POWERLOSS_WORKLOADS=("insert",),
        POWERLOSS_NO_SYSCTL=True,
    )
    command = scheduler.build_command(config)
    assert command[-5:] == ["--modes", "batch", "--workloads", "insert",
                            "--no-sysctl"]


def test_state_round_trip(tmp_path):
    path = tmp_path / "state.json"
    scheduler.write_state(path, {"last_started": 123, "last_exit_code": 0})
    assert scheduler.read_state(path) == {
        "last_started": 123,
        "last_exit_code": 0,
    }
    assert scheduler.read_state(tmp_path / "missing.json") == {}


def test_prune_runs_keeps_newest(tmp_path):
    for name in ("run_20260101_000000", "run_20260102_000000",
                 "run_20260103_000000"):
        (tmp_path / name).mkdir()
    (tmp_path / "not_a_run").mkdir()

    scheduler.prune_runs(tmp_path, keep_runs=2)

    assert not (tmp_path / "run_20260101_000000").exists()
    assert (tmp_path / "run_20260102_000000").exists()
    assert (tmp_path / "run_20260103_000000").exists()
    assert (tmp_path / "not_a_run").exists()


def test_disabled_run_does_not_touch_state(tmp_path, monkeypatch):
    config = _config(tmp_path, POWERLOSS_ENABLED=False)
    monkeypatch.setattr(scheduler, "load_config", lambda *_: config)

    assert scheduler.run_once() == 0
    assert not (tmp_path / config.POWERLOSS_STATE_FILE).exists()


def test_dry_run_does_not_require_root(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(scheduler, "load_config", lambda *_: config)

    assert scheduler.run_once(dry_run=True) == 0
    assert not (tmp_path / config.POWERLOSS_STATE_FILE).exists()
