# ndb_host/tui/processes.py
import os
import time

import socket
import platform
import subprocess

from pathlib import Path

from .context import cfg, logger

# ==========================================================
#       Helper Functions
# ==========================================================

def _probe_host(host: str) -> str:
    """0.0.0.0 / :: are bind targets, not valid connect targets."""
    if host in ("0.0.0.0", "::", "", None):
        return "127.0.0.1"
    return host


def _is_server_running(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((_probe_host(host), port)) == 0


def _pid_file_pid(pid_file: Path) -> int | None:
    """Read the PID recorded in the PID file, or None."""
    try:
        return int(pid_file.read_text().strip())
    except (OSError, ValueError, TypeError):
        return None


def is_server_starting_up(pid_file: Path) -> bool:
    """True when the PID file exists and its process is alive, yet the port
    is not reachable yet (server still booting)."""
    pid = _pid_file_pid(pid_file)
    return pid is not None and _is_pid_alive(pid)


def _is_pid_alive(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        if platform.system() != "Windows":
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False
        else:
            try:
                output = subprocess.check_output(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    stderr=subprocess.DEVNULL
                ).decode()
                return str(pid) in output
            except subprocess.CalledProcessError:
                return False


def _find_process_tree_root(pid: int) -> int:
    """Walk up the parent chain to find the top-most gunicorn process."""
    try:
        import psutil
    except ImportError:
        return pid
    try:
        proc = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return pid
    root = proc
    while True:
        try:
            parent = root.parent()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if parent is None:
            break
        try:
            cmd = " ".join(parent.cmdline()).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            break
        if "gunicorn" in cmd:
            root = parent
        else:
            break
    return root.pid


def _kill_process_tree(pid: int):
    """Kill a process and its whole gunicorn tree (master + workers)."""
    root_pid = _find_process_tree_root(pid)
    try:
        import psutil
        root = psutil.Process(root_pid)
        for child in reversed(root.children(recursive=True)):
            try:
                child.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        root.kill()
        root.wait(timeout=5)
    except (ImportError, psutil.NoSuchProcess, psutil.TimeoutExpired):
        _kill_process(root_pid, force=True)


def _kill_process(pid: int, force: bool = False):
    try:
        import psutil
        proc = psutil.Process(pid)
        if force:
            proc.kill()
        else:
            proc.terminate()
        proc.wait(timeout=5)
    except ImportError:
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F", "/T"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
        else:
            try:
                if force:
                    os.kill(pid, 9)
                else:
                    os.kill(pid, 15)
                    time.sleep(2)
                    if _is_pid_alive(pid):
                        os.kill(pid, 9)
            except ProcessLookupError:
                pass
    except (psutil.NoSuchProcess, ProcessLookupError):
        pass
    except psutil.TimeoutExpired:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass


def _find_pid_on_port(port: int) -> int | None:
    """Find the PID of the process listening on the given port (or None)."""
    try:
        import psutil
        for conn in psutil.net_connections(kind="inet"):
            laddr = conn.laddr
            if laddr and laddr.port == port and conn.pid:
                return conn.pid
    except (ImportError, psutil.AccessDenied, psutil.Error):
        pass

    try:
        if platform.system() == "Windows":
            output = subprocess.check_output(
                ["netstat", "-ano"], stderr=subprocess.DEVNULL
            ).decode(errors="ignore")
            for line in output.splitlines():
                parts = line.split()
                if len(parts) >= 5 and f":{port}" in parts[1] and "LISTENING" in line:
                    pid = parts[-1]
                    if pid.isdigit():
                        return int(pid)
        else:
            output = subprocess.check_output(
                ["lsof", "-ti", f"tcp:{port}"], stderr=subprocess.DEVNULL
            ).decode(errors="ignore").strip().splitlines()
            if output:
                pid = output[0].strip()
                if pid.isdigit():
                    return int(pid)
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return None


# ==========================================================
#         Clear Bytecode Cache
# ==========================================================

def clear_pycache(root: Path = None):
    """
    Recursively remove __pycache__ dirs, .pyc/.pyo bytecode files and
    .pytest_cache/.cache folders under root (defaults to NDB home).

    Only bytecode/tool caches are removed - Storage, logs and model caches
    are left untouched to avoid data loss.
    """
    import shutil
    root = root or cfg.NEBULONDB_HOME
    if not root.is_dir():
        logger.warning("Cache root %s not found. Skipping cache cleanup.", root)
        return 0

    removed = 0
    for dirpath, dirnames, filenames in os.walk(str(root)):
        if "__pycache__" in dirnames:
            target = os.path.join(dirpath, "__pycache__")
            shutil.rmtree(target, ignore_errors=True)
            dirnames.remove("__pycache__")
            removed += 1
        for dirname in list(dirnames):
            if dirname in (".pytest_cache", ".cache"):
                target = os.path.join(dirpath, dirname)
                shutil.rmtree(target, ignore_errors=True)
                dirnames.remove(dirname)
                removed += 1
        for name in filenames:
            if name.endswith((".pyc", ".pyo")):
                try:
                    os.remove(os.path.join(dirpath, name))
                    removed += 1
                except OSError:
                    pass

    logger.info("Cache cleanup: removed %s bytecode/cache entries under %s.", removed, root)
    return removed