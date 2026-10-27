# Power-loss validation in WSL

Status: **implemented and executed** (2026-10-26).

This document describes the real power-loss harness in
`ndb_host/tools/powerloss/`.
It complements, and does not replace, `docs/CRASH_RECOVERY_WAL_REPORT.md`
(which covers process-crash / SIGKILL recovery). The crash report is left
untouched.

## What it validates

The three `[segments] durability_mode` contracts:

| Mode | Acknowledgement guarantee | Expected after power loss |
|---|---|---|
| `process` | `flush()`; `fsync()` every `wal_fsync_interval` bytes | Only the fsynced WAL prefix; the un-fsynced tail may be lost |
| `batch` | one `fsync()` per engine write call, before ack | **No acknowledged record may be lost** |
| `per_ack` | `fsync()` per acknowledged record, before ack | **No acknowledged record may be lost** |

The oracle checks a stronger property than "count of records": the recovered
state must equal the state after some **prefix of the flattened WAL record
stream**. A missing record in the middle (with later records surviving) is a
failure, because a WAL can only lose a suffix.

## How the harness simulates a power cut

```
       loop0 (direct I/O)              loop1
            |                            |
        ndbpw_real  (linear)         ndbpw_cow
            |                            |
    ndbpw_origin (snapshot-origin)       |
            |                            |
      ext4 mount -> engine writes        |
                                         |
    crash: SIGKILL, then create snapshot |
            \____________________________/
                         |
                 ndbpw_snap (snapshot of ndbpw_real)
                         |
              e2fsck -> mount -> WAL recovery -> oracle
```

1. A fresh 256 MiB ext4 filesystem is created on a loop device backed by a
   file in `/var/tmp`.
2. `ndbpw_real` is a `linear` device over that loop; `ndbpw_origin` is a
   `snapshot-origin` on top of it. The engine writes to the ext4 mount.
3. A worker process runs a deterministic workload. It appends an **fsynced**
   `begin` event before each engine call and an `ack` event after it returns.
   The journal lives outside the filesystem under test.
4. The harness SIGKILLs the worker at a random point in the workload.
5. It creates a live device-mapper snapshot of `ndbpw_real` (the *underlying*
   linear device), then unmounts the origin and mounts the snapshot.
6. `e2fsck -p` replays/repairs the snapshot's ext4 journal, normal
   `NebulonCosmos` WAL recovery runs, and the recovered `seq -> text` state is
   written to JSON.
7. The oracle replays the journal record-by-record and classifies the trial as
   `PASS`/`FAIL`.

### Why not snapshot the mounted `snapshot-origin`?

Device mapper refuses to create a snapshot whose origin device is mounted:

```
device-mapper: reload ioctl on ndbpw_snap failed: Device or resource busy
```

This is true even with `dmsetup suspend --noflush`. LVM works around it by
pointing the snapshot at the **underlying** volume (`Root-real`) while
`Root` is the mounted `snapshot-origin`; the harness uses the same layering.
With that arrangement the snapshot device can be created while the filesystem
is mounted.

### Dirty page cache is excluded

`vm.dirty_writeback_centisecs=0` (and a high dirty ratio) prevents background
writeback during a trial. The block snapshot therefore contains only data the
kernel actually wrote to the test device: `fsync()`ed data is present, dirty
page cache is not. Sysctls are saved and restored around the run. Use
`--no-sysctl` only for debugging; it can make `process` mode look more durable
than it really is.

## Workloads and oracle

`ndb_host/tools/powerloss/worker.py` runs three workloads:

* `insert` — single-record inserts.
* `bulk` — `insert_many` batches.
* `mixed` — inserts, bulk inserts, updates and `delete_many`.

Seq `0` is a sentinel: it is inserted first and never updated/deleted. This
makes every replay prefix state distinguishable, so the oracle does not
confuse "all records lost" with "all records applied and then deleted".

`ndb_host/tools/powerloss/oracle.py` expands every operation into individual WAL
records. For example, an unacknowledged `insert_many` may survive partially if
the write was torn; that is represented as a prefix ending inside the batch.
For `batch`/`per_ack`, the prefix must cover every acknowledged record. For
`process`, any prefix is allowed, and the report records how many acknowledged
records were lost.

The oracle is unit-tested without root or devices in
`unittest/test_powerloss_oracle.py` (13 tests).

## Running it

### Linux prerequisites

Run as root with these packages installed. `dm-log-writes`, LVM and btrfs are
**not** required.

Ubuntu / Debian / WSL2:

```bash
sudo apt update
sudo apt install -y dmsetup e2fsprogs util-linux mount coreutils kmod
```

Fedora / RHEL:

```bash
sudo dnf install -y device-mapper e2fsprogs util-linux coreutils kmod
```

Arch:

```bash
sudo pacman -S --needed device-mapper e2fsprogs util-linux coreutils kmod
```

openSUSE:

```bash
sudo zypper install -y device-mapper e2fsprogs util-linux coreutils kmod
```

Verify the required commands:

```bash
for t in dmsetup losetup mkfs.ext4 e2fsck mount umount blockdev truncate; do
  command -v "$t" >/dev/null || echo "MISSING: $t"
done
```

Load the kernel modules:

```bash
sudo modprobe dm_snapshot
sudo modprobe loop
lsmod | grep -E 'dm_snapshot|loop'
sudo dmsetup version
```

The default trial uses a 256 MB image and a 512 MB COW file, so keep at least
about 1 GB free in `powerloss_harness.run_root` (default `/var/tmp`).

On WSL2, systemd is optional for manual runs. To use the systemd timer, enable
systemd in `/etc/wsl.conf` and restart WSL:

```ini
[boot]
systemd=true
```

```powershell
wsl --shutdown
```

Without systemd, run the scheduler with `--daemon` under `nohup` or use cron.

From the project root (the directory containing `nebulondb.cfg`):

```bash
# full matrix: 20 trials per mode, workloads rotate
sudo ndb_host/tools/powerloss/run.sh --trials 20

# fast smoke
sudo ndb_host/tools/powerloss/run.sh \
  --modes process,batch,per_ack --workloads mixed \
  --trials 2 --ops 80 --setup-count 10

# clean up devices/mounts left by an interrupted run
sudo ndb_host/tools/powerloss/run.sh --cleanup-only
```

The wrapper derives the project root from its own location and uses
`$ROOT/.venv/bin/python`; there is no hard-coded checkout path. Override the
interpreter with `NEBULONDB_PYTHON=/path/to/python` if needed. You can also
skip the wrapper entirely:

```bash
sudo .venv/bin/python -m ndb_host.tools.powerloss.harness --help
```

Useful options:

* `--ops`, `--setup-count`, `--delay`, `--min-delay`, `--max-delay` — workload
  size and crash timing.
* `--image-mb` / `--cow-mb` — loop image and snapshot COW sizes.
* `--process-fsync-interval` — group-commit threshold for `process` trials
  (default in the harness: `8192`, deliberately lower than the production
  `65536` so several fsync boundaries occur in a short trial; pass `65536` to
  mirror the default config).
* `--keep-images` — keep loop images for every trial (large).
* `--no-sysctl` — do not disable background writeback.

Each trial writes `journal.jsonl`, `worker.out`, `worker.err`,
`recovered.json` and a result entry under the run directory printed at
startup (default `/var/tmp/ndb_powerloss/run_<timestamp>/`). The full machine
readable result is `results.json`.

### Manual usage from a shell

Start in the project root (the directory that contains `nebulondb.cfg`):

```bash
cd /path/to/NebulonDB

# 1. see the options
.venv/bin/python -m ndb_host.tools.powerloss.harness --help

# 2. if a previous run was interrupted, clean stale dm/loop/mount state
sudo .venv/bin/python -m ndb_host.tools.powerloss.harness --cleanup-only

# 3. quick smoke (safe, ~10 seconds)
sudo .venv/bin/python -m ndb_host.tools.powerloss.harness \
  --modes process,batch,per_ack --workloads mixed \
  --trials 2 --ops 80 --setup-count 10

# 4. full validation (20 trials per mode, ~3-5 minutes)
sudo .venv/bin/python -m ndb_host.tools.powerloss.harness --trials 20
```

The same thing through the optional wrapper (it finds the project root from
its own location):

```bash
sudo ndb_host/tools/powerloss/run.sh --trials 20
```

The harness prints a `PASS`/`FAIL` line per trial and a summary. Open
`results.json` in the printed run directory for per-trial details. It does not
touch the live NebulonDB service; it creates its own loop devices, filesystem
and database under `/var/tmp/ndb_powerloss/`.

The opt-in pytest wrapper is:

```bash
sudo env NDB_POWERLOSS=1 .venv/bin/python -m pytest \
  unittest/test_powerloss_wsl.py -q
```

It is skipped unless `NDB_POWERLOSS=1` and the process is root. It is not part
of the default suites, so a normal `pytest` run does not touch device mapper.

### Scheduled runs

User-tunable settings live in `nebulondb.cfg` under `[powerloss]`,
`[powerloss_harness]` and `[powerloss_worker]`. Internal values that should not
be changed per deployment (device-mapper names, seed derivation, sentinel,
worker idle sleep) live in `NDBMeta.PowerLoss` in
`ndb_host/utils/constants.py`. The scheduler reads the `[powerloss]` section
and runs the harness only when the interval has elapsed.

```ini
[powerloss]
enabled = false   # set true to let the scheduler run
interval_seconds = 86400
modes = process,batch,per_ack
workloads = insert,bulk,mixed
keep_runs = 7
no_sysctl = false
poll_seconds = 60
```

Useful commands:

```bash
# check config and show what would run
.venv/bin/python -m ndb_host.tools.powerloss.scheduler --dry-run

# run now, ignoring the interval
sudo .venv/bin/python -m ndb_host.tools.powerloss.scheduler --once --force

# run as a small daemon
sudo .venv/bin/python -m ndb_host.tools.powerloss.scheduler --daemon
```

For systemd, use a timer that calls `--once` every few minutes; the scheduler
itself checks `interval_seconds` and exits immediately when not due:

```ini
# /etc/systemd/system/nebulondb-powerloss.service
[Unit]
Description=NebulonDB power-loss validation
After=local-fs.target

[Service]
Type=oneshot
User=root
WorkingDirectory=<PROJECT>
Environment=NEBULONDB_HOME=<PROJECT>
ExecStart=<PROJECT>/.venv/bin/python -m ndb_host.tools.powerloss.scheduler --once
```

```ini
# /etc/systemd/system/nebulondb-powerloss.timer
[Unit]
Description=Run NebulonDB power-loss validation periodically

[Timer]
OnBootSec=10min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
```

Replace `<PROJECT>` with the checkout path, then enable the timer:

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now nebulondb-powerloss.timer
systemctl list-timers | grep powerloss
```

Check the service output with:

```bash
journalctl -u nebulondb-powerloss.service -n 100 --no-pager
```

The scheduler uses a lock file under `powerloss_harness.run_root` so overlapping
runs cannot happen. It stores `last_started`, `last_finished` and
`last_exit_code` in `powerloss.state_file` and keeps the newest
`powerloss.keep_runs` run directories.

Do not enable this on a host serving live traffic: the harness changes global
`vm.dirty_*` settings and creates loop/device-mapper devices. Use a dedicated
WSL/test VM, or set `no_sysctl = true` if you accept the weaker `process`-mode
signal.

## Executed result (2026-10-26)

Command:

```bash
sudo .venv/bin/python -m ndb_host.tools.powerloss.harness \
  --modes process,batch,per_ack \
  --workloads insert,bulk,mixed \
  --trials 20 --ops 300 --setup-count 40 \
  --delay 0.005 --min-delay 0.02 --max-delay 1.2
```

Environment: WSL2 kernel `6.6.87.2-microsoft-standard-WSL2`, ext4 on a
`--direct-io=on` loop, `/var/tmp` on `/dev/sdd`.

| Mode | Trials | Passed | Failed | Trials with acked loss | Total acked lost | Recovered prefix range |
|---|---:|---:|---:|---:|---:|---:|
| `process` | 20 | 20 | 0 | 20 | 1462 | 0–505 |
| `batch` | 20 | 20 | 0 | **0** | **0** | 46–392 |
| `per_ack` | 20 | 20 | 0 | **0** | **0** | 44–406 |

Interpretation:

* Every `batch`/`per_ack` trial recovered every acknowledged record, while
  still allowing unacknowledged WAL records from the crash window (prefix
  extended past the last `ack` in several trials).
* Every `process` trial lost a suffix of acknowledged records and none lost a
  middle segment. That is the documented contract: `process` does not
  guarantee acked-record durability on power loss. Most trials lost the whole
  WAL because the crash happened before the next 8 KiB group fsync; trials
  killed later show a surviving prefix (up to 505 records).
* No trial produced a recovery error, non-prefix state, duplicate seq, or
  snapshot/fsck/mount failure. COW usage was 0 sectors because the snapshot is
  created after the kill and only the origin's later unmount writes are
  COW'd; the 512 MiB COW was never close to overflow.

Raw results: `/var/tmp/ndb_powerloss/run_20261026_131136/results.json` (run
directories are host-local and not checked into git).

## Limitations

* This is a **media/power-cut simulation at the block layer**, not a host
  power cut. The host ext4 holding the loop files does not lose power; loop
  direct I/O makes the device view consistent with the test filesystem's
  writeback state.
* No torn-sector / bad-sector injection and no `dm-log-writes` replay. The
  prefix oracle catches WAL ordering loss, not arbitrary media corruption.
* ext4 only. Other filesystems may need different fsck/mount handling.
* `e2fsck -p` may repair the snapshot's filesystem metadata before the engine
  recovers. That models a reboot with fsck, not a totally unrepaired crash.
* Root and device-mapper are required; the harness should not be run against
  a production database. The sysctl change is global to the WSL instance
  (restored in `finally`); avoid running it while other write-heavy workloads
  depend on prompt writeback.
* The `process` matrix uses an accelerated 8 KiB fsync interval. Pass
  `--process-fsync-interval 65536` to mirror `nebulondb.cfg`.

## Phase checklist

1. Prove the device-mapper layout/snapshot semantics — done (linear +
   `snapshot-origin`, snapshot the underlying linear while mounted).
2. Implement and unit-test the prefix oracle — done (13 tests).
3. Implement the fsynced begin/ack worker and workloads — done.
4. Implement the loop/dm/ext4 orchestrator — done.
5. Implement snapshot fsck/mount + WAL recovery reader — done.
6. Run the mode × workload matrix — done (20 trials/mode, 60 total).
7. Document the harness and re-run the existing suites — done; see the
   repository README test section.
