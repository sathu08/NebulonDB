"""
WSL power-loss harness for NebulonDB's WAL durability modes.

The harness creates a real ext4 filesystem on a loop device, puts a
``snapshot-origin`` device on top of the underlying linear device (the same
layering LVM uses), runs the engine, SIGKILLs it, and creates a live
device-mapper snapshot of the *underlying* device.  The snapshot therefore
contains exactly the blocks the kernel had written at crash time: fsynced data
is present, dirty page cache is not.

Why not snapshot the mounted ``snapshot-origin`` directly?  Device mapper
refuses to create a snapshot whose origin device is mounted ("Device or
resource busy").  LVM avoids this by pointing the snapshot at the underlying
linear device; that is what this harness does too.

Run as root inside WSL (see docs/POWER_LOSS_VALIDATION_WSL.md), from the
project root that contains ``nebulondb.cfg``:

    .venv/bin/python -m ndb_host.tools.powerloss.harness --trials 20

Requires: dmsetup, losetup, mkfs.ext4/e2fsck, mount/umount, blockdev.
"""

from __future__ import annotations

import os
import sys
import time

import json
import shutil
import random

import argparse
import traceback
import subprocess

from typing import Any
from pathlib import Path
from datetime import datetime

from ndb_host.db.ndb_settings import NDBConfig
from ndb_host.tools.powerloss.oracle import JournalError

from ndb_host.tools.powerloss.oracle import evaluate, load_journal


# ==========================================================
#        Configuration
# ==========================================================

CONFIG = NDBConfig()
PROJECT = CONFIG.NEBULONDB_HOME

TOOLS = Path(__file__).resolve().parent
RUN_ROOT = Path(CONFIG.POWERLOSS_HARNESS_RUN_ROOT)

DM_REAL = CONFIG.POWERLOSS_HARNESS_DM_REAL
DM_ORIGIN = CONFIG.POWERLOSS_HARNESS_DM_ORIGIN
DM_SNAP = CONFIG.POWERLOSS_HARNESS_DM_SNAP
DM_PREFIX = CONFIG.POWERLOSS_HARNESS_DM_PREFIX
FS_LABEL = CONFIG.POWERLOSS_HARNESS_FS_LABEL
SNAPSHOT_CHUNK_SIZE = CONFIG.POWERLOSS_HARNESS_SNAPSHOT_CHUNK_SIZE

MODE_CHOICES = CONFIG.POWERLOSS_MODES
WORKLOAD_CHOICES = CONFIG.POWERLOSS_WORKLOADS
DISABLE_BACKGROUND = CONFIG.POWERLOSS_DISABLE_BACKGROUND

SYSCTLS = CONFIG.POWERLOSS_HARNESS_SYSCTLS

RUN_DIR_FORMAT = CONFIG.POWERLOSS_HARNESS_RUN_DIR_FORMAT
DEFAULT_TRIALS = CONFIG.POWERLOSS_HARNESS_DEFAULT_TRIALS
DEFAULT_OPS = CONFIG.POWERLOSS_HARNESS_DEFAULT_OPS
DEFAULT_SETUP_COUNT = CONFIG.POWERLOSS_HARNESS_DEFAULT_SETUP_COUNT
DEFAULT_DELAY = CONFIG.POWERLOSS_HARNESS_DEFAULT_DELAY
DEFAULT_PROCESS_FSYNC_INTERVAL = (
    CONFIG.POWERLOSS_HARNESS_DEFAULT_PROCESS_FSYNC_INTERVAL
)
DEFAULT_MIN_DELAY = CONFIG.POWERLOSS_HARNESS_DEFAULT_MIN_DELAY
DEFAULT_MAX_DELAY = CONFIG.POWERLOSS_HARNESS_DEFAULT_MAX_DELAY
DEFAULT_READY_TIMEOUT = CONFIG.POWERLOSS_HARNESS_DEFAULT_READY_TIMEOUT
DEFAULT_SEED = CONFIG.POWERLOSS_HARNESS_DEFAULT_SEED
DEFAULT_IMAGE_MB = CONFIG.POWERLOSS_HARNESS_DEFAULT_IMAGE_MB
DEFAULT_COW_MB = CONFIG.POWERLOSS_HARNESS_DEFAULT_COW_MB

CMD_TIMEOUT = CONFIG.POWERLOSS_HARNESS_CMD_TIMEOUT
UMOUNT_TIMEOUT = CONFIG.POWERLOSS_HARNESS_UMOUNT_TIMEOUT
CLEANUP_TIMEOUT = CONFIG.POWERLOSS_HARNESS_CLEANUP_TIMEOUT
DM_REMOVE_ATTEMPTS = CONFIG.POWERLOSS_HARNESS_DM_REMOVE_ATTEMPTS
DM_REMOVE_TIMEOUT = CONFIG.POWERLOSS_HARNESS_DM_REMOVE_TIMEOUT
LOOP_TIMEOUT = CONFIG.POWERLOSS_HARNESS_LOOP_TIMEOUT
FSCK_TIMEOUT = CONFIG.POWERLOSS_HARNESS_FSCK_TIMEOUT
FSCK_REPAIR_TIMEOUT = CONFIG.POWERLOSS_HARNESS_FSCK_REPAIR_TIMEOUT
MOUNT_TIMEOUT = CONFIG.POWERLOSS_HARNESS_MOUNT_TIMEOUT
RECOVERY_TIMEOUT = CONFIG.POWERLOSS_HARNESS_RECOVERY_TIMEOUT
KILL_WAIT_TIMEOUT = CONFIG.POWERLOSS_HARNESS_KILL_WAIT_TIMEOUT
READY_POLL_INTERVAL = CONFIG.POWERLOSS_HARNESS_READY_POLL_INTERVAL
RETRY_DELAY = CONFIG.POWERLOSS_HARNESS_RETRY_DELAY
DM_RETRY_DELAY = CONFIG.POWERLOSS_HARNESS_DM_RETRY_DELAY

KILL_DELAY_SEED_SALT = CONFIG.POWERLOSS_HARNESS_KILL_DELAY_SEED_SALT
SEED_MODE_STRIDE = CONFIG.POWERLOSS_HARNESS_SEED_MODE_STRIDE
SEED_WORKLOAD_STRIDE = CONFIG.POWERLOSS_HARNESS_SEED_WORKLOAD_STRIDE
SEED_TRIAL_STRIDE = CONFIG.POWERLOSS_HARNESS_SEED_TRIAL_STRIDE
SEED_MASK = CONFIG.POWERLOSS_HARNESS_SEED_MASK


class HarnessError(RuntimeError):
    pass


# ==========================================================
#        Shell Helpers
# ==========================================================

def run(
    args: list[str],
    *,
    check: bool = True,
    timeout: float = CMD_TIMEOUT,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            args, capture_output=True, text=True, timeout=timeout, env=env
        )
    except subprocess.TimeoutExpired as exc:
        if check:
            raise HarnessError(f"timed out after {timeout}s: {' '.join(args)}") from exc
        return subprocess.CompletedProcess(args, 124, exc.stdout or "", exc.stderr or "")
    if check and proc.returncode != 0:
        raise HarnessError(
            f"command failed ({proc.returncode}): {' '.join(args)}\n"
            f"stdout:\n{proc.stdout[-4000:]}\nstderr:\n{proc.stderr[-4000:]}"
        )
    return proc


def _umount(
    path: Path, *, timeout: float = UMOUNT_TIMEOUT,
) -> bool:
    for flags in ([], ["-l"]):
        proc = run(["umount", *flags, str(path)], check=False, timeout=timeout)
        if proc.returncode == 0:
            return True
        time.sleep(RETRY_DELAY)
    return False


def _dm_remove(name: str) -> bool:
    for _ in range(DM_REMOVE_ATTEMPTS):
        proc = run(
            ["dmsetup", "remove", "-f", name],
            check=False,
            timeout=DM_REMOVE_TIMEOUT,
        )
        if proc.returncode == 0:
            return True
        time.sleep(DM_RETRY_DELAY)
    return False


def _loop_detach(loop: str) -> None:
    run(
        ["losetup", "-d", loop],
        check=False,
        timeout=LOOP_TIMEOUT,
    )


def ensure_tools() -> None:
    os.environ["PATH"] = "/usr/sbin:/sbin:" + os.environ.get("PATH", "")
    needed = ["dmsetup", "losetup", "mkfs.ext4", "e2fsck", "mount", "umount",
              "blockdev", "truncate"]
    missing = [tool for tool in needed if shutil.which(tool) is None]
    if missing:
        raise HarnessError(f"missing required tools: {', '.join(missing)}")


def cleanup_stale(root: Path = RUN_ROOT) -> None:
    """Remove devices/mounts/loops left by an interrupted previous run."""
    timeout = CLEANUP_TIMEOUT
    mounts = run(["mount"], check=False, timeout=timeout).stdout
    for line in mounts.splitlines():
        if " on " not in line or " type " not in line:
            continue
        mountpoint = line.split(" on ", 1)[1].split(" type ", 1)[0]
        if mountpoint.startswith(str(root)):
            _umount(Path(mountpoint))

    devices = run(["dmsetup", "ls"], check=False, timeout=timeout).stdout
    for line in devices.splitlines():
        name = line.split("\t", 1)[0].strip()
        if name.startswith(DM_PREFIX):
            _dm_remove(name)

    loops = run(["losetup", "-a"], check=False, timeout=timeout).stdout
    for line in loops.splitlines():
        if str(root) in line and "(" in line:
            _loop_detach(line.split(":", 1)[0])


# ==========================================================
#        Sysctl Control
# ==========================================================

def apply_sysctls(enable: bool) -> dict[str, int]:
    saved: dict[str, int] = {}
    if not enable:
        print("WARNING: --no-sysctl: background writeback may make process mode "
              "look more durable than it is", file=sys.stderr)
        return saved
    for name in SYSCTLS:
        path = Path("/proc/sys/vm") / name
        try:
            saved[name] = int(path.read_text().strip())
        except OSError:
            continue
    for name, value in SYSCTLS.items():
        path = Path("/proc/sys/vm") / name
        try:
            path.write_text(str(value))
        except OSError as exc:
            print(f"WARNING: could not set vm.{name}={value}: {exc}", file=sys.stderr)
    return saved


def restore_sysctls(saved: dict[str, int]) -> None:
    for name, value in saved.items():
        try:
            (Path("/proc/sys/vm") / name).write_text(str(value))
        except OSError:
            pass


# ==========================================================
#        Child Process Environment
# ==========================================================

def child_env() -> dict[str, str]:
    env = os.environ.copy()
    env["NEBULONDB_HOME"] = str(PROJECT)
    env["PYTHONUNBUFFERED"] = "1"
    old = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(PROJECT) + (os.pathsep + old if old else "")
    return env


# ==========================================================
#        Device Setup
# ==========================================================

def setup_trial(trial_dir: Path, image_mb: int, cow_mb: int) -> dict[str, Any]:
    mnt = trial_dir / "mnt"
    smnt = trial_dir / "snap"
    mnt.mkdir(parents=True, exist_ok=True)
    smnt.mkdir(parents=True, exist_ok=True)
    origin_img = trial_dir / "origin.img"
    cow_img = trial_dir / "cow.img"
    ctx: dict[str, Any] = {
        "dir": trial_dir,
        "mnt": mnt,
        "smnt": smnt,
        "origin_img": origin_img,
        "cow_img": cow_img,
        "origin_loop": None,
        "cow_loop": None,
        "size": 0,
        "db_dir": mnt / "db",
    }

    try:
        run(["truncate", "-s", f"{image_mb}M", str(origin_img)])
        run(["truncate", "-s", f"{cow_mb}M", str(cow_img)])

        origin_loop = run(
            ["losetup", "--find", "--show", "--direct-io=on", str(origin_img)]
        ).stdout.strip()
        ctx["origin_loop"] = origin_loop
        sz = int(run(["blockdev", "--getsz", origin_loop]).stdout.strip())
        ctx["size"] = sz

        run(["dmsetup", "create", DM_REAL, "--table",
             f"0 {sz} linear {origin_loop} 0"])
        run(["dmsetup", "create", DM_ORIGIN, "--table",
             f"0 {sz} snapshot-origin /dev/mapper/{DM_REAL}"])
        run(["mkfs.ext4", "-q", "-F", "-L", FS_LABEL,
             f"/dev/mapper/{DM_ORIGIN}"])
        run(["mount", "-t", "ext4", "-o", "rw,noatime,data=ordered",
             f"/dev/mapper/{DM_ORIGIN}", str(mnt)])
    except BaseException:
        cleanup_trial(ctx, keep_images=False)
        raise

    return ctx


def create_snapshot(ctx: dict[str, Any]) -> str:
    cow_loop = run(
        ["losetup", "--find", "--show", str(ctx["cow_img"])]
    ).stdout.strip()
    ctx["cow_loop"] = cow_loop
    chunk = SNAPSHOT_CHUNK_SIZE
    table_n = f"0 {ctx['size']} snapshot /dev/mapper/{DM_REAL} {cow_loop} N {chunk}"
    proc = run(["dmsetup", "create", DM_SNAP, "--table", table_n], check=False)
    if proc.returncode != 0:
        _dm_remove(DM_SNAP)
        table_p = f"0 {ctx['size']} snapshot /dev/mapper/{DM_REAL} {cow_loop} P {chunk}"
        run(["dmsetup", "create", DM_SNAP, "--table", table_p])
    status = run(["dmsetup", "status", DM_SNAP], check=False).stdout.strip()
    return status


def cleanup_trial(ctx: dict[str, Any] | None, keep_images: bool) -> None:
    if not ctx:
        return
    _umount(ctx["smnt"])
    _umount(ctx["mnt"])
    _dm_remove(DM_SNAP)
    _dm_remove(DM_ORIGIN)
    _dm_remove(DM_REAL)
    if ctx.get("cow_loop"):
        _loop_detach(ctx["cow_loop"])
    if ctx.get("origin_loop"):
        _loop_detach(ctx["origin_loop"])
    if not keep_images:
        for image in (ctx.get("origin_img"), ctx.get("cow_img")):
            if image:
                try:
                    Path(image).unlink()
                except OSError:
                    pass


# ==========================================================
#        Worker And Recovery
# ==========================================================

def start_worker(
    ctx: dict[str, Any], args: argparse.Namespace, seed: int,
) -> tuple[subprocess.Popen[bytes], Any, Any]:
    journal = ctx["dir"] / "journal.jsonl"
    out = (ctx["dir"] / "worker.out").open("wb")
    err = (ctx["dir"] / "worker.err").open("wb")
    cmd = [
        sys.executable, "-u", str(TOOLS / "worker.py"),
        "--db-dir", str(ctx["db_dir"]),
        "--journal", str(journal),
        "--mode", args.mode,
        "--workload", args.workload,
        "--seed", str(seed),
        "--ops", str(args.ops),
        "--setup-count", str(args.setup_count),
        "--delay", str(args.delay),
        "--wal-fsync-interval",
        str(
            args.process_fsync_interval
            if args.mode == "process"
            else DISABLE_BACKGROUND
        ),
    ]
    proc = subprocess.Popen(
        cmd, cwd=str(PROJECT), env=child_env(),
        stdout=out, stderr=err, start_new_session=True,
    )
    return proc, out, err


def wait_ready(journal: Path, proc: subprocess.Popen[bytes], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if journal.exists():
            data = journal.read_bytes()
            if b'"ev":"ready"' in data or b'"ev": "ready"' in data:
                return
        rc = proc.poll()
        if rc is not None:
            tail = journal.read_text(errors="replace")[-2000:] if journal.exists() else ""
            raise HarnessError(f"worker exited rc={rc} before READY\n{tail}")
        time.sleep(READY_POLL_INTERVAL)
    raise HarnessError(f"worker did not reach READY within {timeout}s")


def _fsck_snapshot(device: str) -> dict[str, Any]:
    info: dict[str, Any] = {"first_rc": None, "second_rc": None}
    proc = run(
        ["e2fsck", "-p", device],
        check=False,
        timeout=FSCK_TIMEOUT,
    )
    info["first_rc"] = proc.returncode
    if proc.returncode in (0, 1, 2):
        return info
    proc = run(
        ["e2fsck", "-y", device],
        check=False,
        timeout=FSCK_REPAIR_TIMEOUT,
    )
    info["second_rc"] = proc.returncode
    return info


def _mount_snapshot(ctx: dict[str, Any]) -> tuple[bool, bool, dict[str, Any]]:
    """Return (mounted_rw, used_copy, details)."""
    device = f"/dev/mapper/{DM_SNAP}"
    smnt = ctx["smnt"]
    details: dict[str, Any] = {}
    details["fsck"] = _fsck_snapshot(device)

    proc = run(
        ["mount", "-t", "ext4", "-o", "rw,noatime", device, str(smnt)],
        check=False,
        timeout=MOUNT_TIMEOUT,
    )
    if proc.returncode == 0:
        return True, False, details

    details["rw_mount_error"] = proc.stderr.strip()[-1000:]
    # Last resort: mount read-only and copy the database to the host for
    # recovery.  This still preserves the WAL bytes visible at crash time.
    ro_mnt = ctx["dir"] / "snap_ro"
    ro_mnt.mkdir(exist_ok=True)
    ro = run(
        ["mount", "-t", "ext4", "-o", "ro,noload", device, str(ro_mnt)],
        check=False,
        timeout=MOUNT_TIMEOUT,
    )
    if ro.returncode == 0:
        src = ro_mnt / "db"
        dst = ctx["dir"] / "recovered_db"
        try:
            if not src.exists():
                details["copy_error"] = "no db directory in snapshot"
            else:
                shutil.copytree(src, dst, dirs_exist_ok=True)
                details["used_copy"] = True
        except OSError as exc:
            details["copy_error"] = str(exc)
        _umount(ro_mnt)
        return False, dst.exists(), details

    details["ro_mount_error"] = ro.stderr.strip()[-1000:]
    return False, False, details


def run_recovery(db_dir: Path, out_file: Path) -> tuple[int, dict[str, Any]]:
    proc = run(
        [sys.executable, "-u", str(TOOLS / "recover.py"),
         "--db-dir", str(db_dir), "--out", str(out_file)],
        check=False, timeout=RECOVERY_TIMEOUT, env=child_env(),
    )
    if not out_file.exists():
        return proc.returncode, {
            "ok": False,
            "error": f"recovery produced no output (rc={proc.returncode})",
            "stderr": proc.stderr[-2000:],
        }
    try:
        payload = json.loads(out_file.read_text())
    except json.JSONDecodeError as exc:
        return proc.returncode, {"ok": False, "error": f"bad recovery JSON: {exc}"}
    return proc.returncode, payload


# ==========================================================
#        One Trial
# ==========================================================

def run_trial(
    args: argparse.Namespace,
    run_dir: Path,
    mode: str,
    workload: str,
    trial_index: int,
    seed: int,
) -> dict[str, Any]:
    trial_dir = run_dir / "trials" / f"{mode}_{workload}_{trial_index:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "mode": mode,
        "workload": workload,
        "trial": trial_index,
        "seed": seed,
        "ok": False,
        "errors": [],
    }
    ctx: dict[str, Any] | None = None
    proc: subprocess.Popen[bytes] | None = None
    logs: tuple[Any, Any] | None = None

    try:
        ctx = setup_trial(trial_dir, args.image_mb, args.cow_mb)
        args.mode = mode
        args.workload = workload
        proc, out, err = start_worker(ctx, args, seed)
        logs = (out, err)

        try:
            wait_ready(trial_dir / "journal.jsonl", proc, args.ready_timeout)
            rng = random.Random(
                seed ^ KILL_DELAY_SEED_SALT
            )
            delay = rng.uniform(args.min_delay, args.max_delay)
            result["kill_delay"] = round(delay, 4)
            time.sleep(delay)
            if proc.poll() is not None:
                raise HarnessError(
                    f"worker exited rc={proc.returncode} before the harness could kill it"
                )
            proc.kill()
            proc.wait(timeout=KILL_WAIT_TIMEOUT)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=KILL_WAIT_TIMEOUT)
            for handle in logs:
                handle.close()
            logs = None

        result["worker_rc"] = proc.returncode
        result["cow_status"] = create_snapshot(ctx)

        origin_unmounted = _umount(ctx["mnt"])
        result["origin_unmounted"] = origin_unmounted
        if not origin_unmounted:
            result["errors"].append("could not unmount origin (continuing)")

        mounted_rw, used_copy, details = _mount_snapshot(ctx)
        result.update(details)

        if used_copy:
            db_dir = ctx["dir"] / "recovered_db"
        elif mounted_rw:
            db_dir = ctx["smnt"] / "db"
        else:
            raise HarnessError("could not mount the crash snapshot")

        out_file = trial_dir / "recovered.json"
        recover_rc, payload = run_recovery(db_dir, out_file)
        result["recover_rc"] = recover_rc
        if recover_rc != 0 or not payload.get("ok", False):
            raise HarnessError(
                f"recovery failed: {payload.get('error') or payload.get('stderr')}"
            )

        journal = load_journal(trial_dir / "journal.jsonl")
        if not journal.ready:
            raise HarnessError("journal has no READY marker")
        verdict = evaluate(journal, payload.get("records", []), mode)
        result.update(verdict.as_dict())
        result["journal_ops"] = len(journal.ops)
        result["recovered_count"] = int(payload.get("count", 0))
        result["ok"] = bool(verdict.ok) and not result["errors"]
    except (HarnessError, JournalError, OSError, ValueError) as exc:
        result["errors"].append(str(exc))
        result["ok"] = False
    except BaseException as exc:  # noqa: BLE001
        result["errors"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
        result["ok"] = False
    finally:
        if logs:
            for handle in logs:
                handle.close()
        cleanup_trial(ctx, keep_images=args.keep_images or not result["ok"])
    return result


# ==========================================================
#        Main
# ==========================================================

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--modes", default=",".join(MODE_CHOICES))
    p.add_argument("--workloads", default=",".join(WORKLOAD_CHOICES))
    p.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                   help="trials per durability mode (workloads rotate)")
    p.add_argument("--ops", type=int, default=DEFAULT_OPS)
    p.add_argument("--setup-count", type=int,
                   default=DEFAULT_SETUP_COUNT)
    p.add_argument("--delay", type=float, default=DEFAULT_DELAY,
                   help="sleep between workload ops in the child")
    p.add_argument("--process-fsync-interval", type=int,
                   default=DEFAULT_PROCESS_FSYNC_INTERVAL,
                   help="process-mode WAL group-commit threshold used during the "
                        "run (default config is 65536; reduced here to force "
                        "several fsync boundaries in a short trial)")
    p.add_argument("--min-delay", type=float,
                   default=DEFAULT_MIN_DELAY)
    p.add_argument("--max-delay", type=float,
                   default=DEFAULT_MAX_DELAY)
    p.add_argument("--ready-timeout", type=float,
                   default=DEFAULT_READY_TIMEOUT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--image-mb", type=int,
                   default=DEFAULT_IMAGE_MB)
    p.add_argument("--cow-mb", type=int, default=DEFAULT_COW_MB)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--keep-images", action="store_true",
                   help="keep loop images for every trial (large)")
    p.add_argument("--no-sysctl", action="store_true")
    p.add_argument("--cleanup-only", action="store_true")
    return p.parse_args(argv)


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_mode: dict[str, Any] = {}
    for result in results:
        mode = result["mode"]
        entry = by_mode.setdefault(mode, {
            "trials": 0, "passed": 0, "failed": 0,
            "acked_lost_trials": 0, "total_lost_acked": 0,
            "recovered_prefix_min": None, "recovered_prefix_max": None,
            "workloads": {},
        })
        entry["trials"] += 1
        if result.get("ok"):
            entry["passed"] += 1
        else:
            entry["failed"] += 1
        lost = int(result.get("lost_acked_records") or 0)
        if lost:
            entry["acked_lost_trials"] += 1
            entry["total_lost_acked"] += lost
        prefix = result.get("prefix_records")
        if isinstance(prefix, int):
            if entry["recovered_prefix_min"] is None:
                entry["recovered_prefix_min"] = prefix
            entry["recovered_prefix_min"] = min(entry["recovered_prefix_min"], prefix)
            if entry["recovered_prefix_max"] is None:
                entry["recovered_prefix_max"] = prefix
            entry["recovered_prefix_max"] = max(entry["recovered_prefix_max"], prefix)
        wl = entry["workloads"].setdefault(
            result["workload"], {"trials": 0, "passed": 0, "acked_lost": 0}
        )
        wl["trials"] += 1
        if result.get("ok"):
            wl["passed"] += 1
        wl["acked_lost"] += lost
    return {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "trials": len(results),
        "by_mode": by_mode,
    }


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    ensure_tools()

    if os.geteuid() != 0:
        print("ERROR: run the power-loss harness as root inside WSL", file=sys.stderr)
        return 2

    if args.cleanup_only:
        cleanup_stale()
        print("power-loss devices/mounts cleaned")
        return 0

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]
    bad = [m for m in modes if m not in MODE_CHOICES]
    bad += [w for w in workloads if w not in WORKLOAD_CHOICES]
    if bad or not modes or not workloads:
        print(f"ERROR: bad modes/workloads: {bad}", file=sys.stderr)
        return 2
    if args.trials < 1:
        print("ERROR: --trials must be >= 1", file=sys.stderr)
        return 2

    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if args.run_dir:
        run_dir = Path(args.run_dir).resolve()
    else:
        run_dir = RUN_ROOT / datetime.now().strftime(
            RUN_DIR_FORMAT
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"run dir: {run_dir}")
    print(f"modes: {modes}  workloads: {workloads}  trials/mode: {args.trials}")
    cleanup_stale()
    saved_sysctls = apply_sysctls(not args.no_sysctl)
    results: list[dict[str, Any]] = []
    results_file = run_dir / "results.json"

    try:
        for mode_index, mode in enumerate(modes):
            for trial_index in range(args.trials):
                workload = workloads[trial_index % len(workloads)]
                wl_index = workloads.index(workload)
                seed = (
                    args.seed
                    + mode_index * SEED_MODE_STRIDE
                    + wl_index * SEED_WORKLOAD_STRIDE
                    + trial_index * SEED_TRIAL_STRIDE
                ) & SEED_MASK
                result = run_trial(
                    args, run_dir, mode, workload, trial_index, seed
                )
                results.append(result)
                state = "PASS" if result.get("ok") else "FAIL"
                print(
                    f"[{state}] {mode:8s} {workload:6s} trial={trial_index:02d} "
                    f"prefix={result.get('prefix_records')} "
                    f"acked={result.get('acked_records')} "
                    f"lost_acked={result.get('lost_acked_records')} "
                    f"recovered={result.get('recovered_records')} "
                    f"delay={result.get('kill_delay')}"
                )
                if not result.get("ok"):
                    for err in result.get("errors", []):
                        print(f"        error: {err.splitlines()[0][:300]}")
                results_file.write_text(json.dumps(
                    {"summary": _summary(results), "results": results}, indent=2
                ))
    finally:
        restore_sysctls(saved_sysctls)

    summary = _summary(results)
    results_file.write_text(json.dumps(
        {"summary": summary, "results": results}, indent=2
    ))

    print("\n===== summary =====")
    for mode, entry in summary["by_mode"].items():
        print(
            f"{mode:8s} trials={entry['trials']} passed={entry['passed']} "
            f"failed={entry['failed']} acked_lost_trials={entry['acked_lost_trials']} "
            f"total_lost_acked={entry['total_lost_acked']} "
            f"prefix_range=[{entry['recovered_prefix_min']},"
            f"{entry['recovered_prefix_max']}]"
        )
    print(f"results: {results_file}")

    failures = [r for r in results if not r.get("ok")]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
