"""
Scheduler for the WSL power-loss validation harness.

Reads ``[powerloss]`` from ``nebulondb.cfg`` and runs the harness when the
configured interval has elapsed.  Intended to be driven by a systemd timer
(``--once``) or run as a small daemon (``--daemon``).

The harness itself must run as root and changes global ``vm.dirty_*`` sysctls,
so do not enable this on a host serving live traffic.
"""

from __future__ import annotations

import os
import sys
import time

import json
import fcntl
import shutil

import argparse
import subprocess

from pathlib import Path
from typing import Any

from ndb_host.db.ndb_settings import NDBConfig


# ==========================================================
#        Configuration
# ==========================================================

def load_config(config_path: str | Path | None = None) -> NDBConfig:
    return NDBConfig(config_path) if config_path else NDBConfig()


# ==========================================================
#        State
# ==========================================================

def read_state(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def write_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


# ==========================================================
#        Scheduling
# ==========================================================

def is_due(config: NDBConfig, state: dict[str, Any], now: float | None = None) -> bool:
    if not config.POWERLOSS_ENABLED:
        return False
    now = time.time() if now is None else now
    last = state.get("last_started")
    if not isinstance(last, (int, float)):
        return True
    return now - last >= config.POWERLOSS_INTERVAL_SECONDS


def build_command(config: NDBConfig) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "ndb_host.tools.powerloss.harness",
        "--modes",
        ",".join(config.POWERLOSS_MODES),
        "--workloads",
        ",".join(config.POWERLOSS_WORKLOADS),
    ]
    if config.POWERLOSS_NO_SYSCTL:
        command.append("--no-sysctl")
    return command


# ==========================================================
#        Run Management
# ==========================================================

def prune_runs(run_root: Path, keep_runs: int) -> None:
    if keep_runs < 0 or not run_root.is_dir():
        return
    runs = sorted(
        (p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("run_")),
        key=lambda p: p.name,
    )
    doomed = runs[:-keep_runs] if keep_runs else runs
    for path in doomed:
        shutil.rmtree(path, ignore_errors=True)


# ==========================================================
#        Locking
# ==========================================================

def _acquire_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(path), os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    return fd


# ==========================================================
#        Child Process Environment
# ==========================================================

def _child_env(config: NDBConfig) -> dict[str, str]:
    env = os.environ.copy()
    project = str(config.NEBULONDB_HOME)
    env["NEBULONDB_HOME"] = project
    old_path = env.get("PYTHONPATH")
    env["PYTHONPATH"] = project + (os.pathsep + old_path if old_path else "")
    return env


# ==========================================================
#        Execution
# ==========================================================

def run_once(
    config_path: str | Path | None = None,
    *,
    force: bool = False,
    dry_run: bool = False,
) -> int:
    config = load_config(config_path)
    run_root = Path(config.POWERLOSS_HARNESS_RUN_ROOT)
    state_path = run_root / config.POWERLOSS_STATE_FILE
    state = read_state(state_path)

    if not config.POWERLOSS_ENABLED:
        print("power-loss scheduler: disabled in config")
        return 0
    if not force and not is_due(config, state):
        print("power-loss scheduler: not due")
        return 0

    command = build_command(config)
    if dry_run:
        print("power-loss scheduler: would run")
        print("  " + " ".join(command))
        return 0

    if not hasattr(os, "geteuid") or os.geteuid() != 0:
        print("ERROR: power-loss scheduler must run as root", file=sys.stderr)
        return 2

    lock = _acquire_lock(run_root / config.POWERLOSS_LOCK_FILE)
    if lock is None:
        print("power-loss scheduler: another run is active")
        return 0

    started = time.time()
    try:
        state["last_started"] = started
        state["last_exit_code"] = None
        write_state(state_path, state)

        print("power-loss scheduler: starting")
        print("  " + " ".join(command))
        rc = subprocess.run(
            command,
            cwd=str(config.NEBULONDB_HOME),
            env=_child_env(config),
            check=False,
        ).returncode

        state["last_finished"] = time.time()
        state["last_exit_code"] = rc
        write_state(state_path, state)
        prune_runs(run_root, config.POWERLOSS_KEEP_RUNS)
        print(f"power-loss scheduler: finished rc={rc}")
        return rc
    finally:
        os.close(lock)


# ==========================================================
#        Daemon
# ==========================================================

def _daemon(config_path: str | Path | None) -> int:
    try:
        while True:
            config = load_config(config_path)
            run_once(config_path)
            sleep_for = max(1, min(config.POWERLOSS_POLL_SECONDS,
                                   config.POWERLOSS_INTERVAL_SECONDS))
            time.sleep(sleep_for)
    except KeyboardInterrupt:
        return 0


# ==========================================================
#        Main
# ==========================================================

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=None,
                   help="path to nebulondb.cfg (default: NEBULONDB_HOME/cwd)")
    p.add_argument("--once", action="store_true",
                   help="run at most once and exit (default)")
    p.add_argument("--daemon", action="store_true",
                   help="keep running and start a new run when due")
    p.add_argument("--force", action="store_true",
                   help="ignore the interval and run now")
    p.add_argument("--dry-run", action="store_true",
                   help="print the command without running it")
    args = p.parse_args(argv)

    if args.daemon:
        return _daemon(args.config)
    return run_once(
        args.config,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
