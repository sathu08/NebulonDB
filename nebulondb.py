import os
import sys
import time

import socket
import platform
import subprocess

import pyfiglet

from pathlib import Path

from getpass import getpass
from colorama import init

from ndb_host.utils.logger import Fore, Style

from ndb_host.db.ndb_settings import NDBConfig
from ndb_host.utils.logger import NebulonDBLogger
from ndb_host.utils.bootstrap import NebulonInitializer


# ==========================================================
#         NebulonDB Runner
# ==========================================================

def _load_config() -> NDBConfig:
    """Load the NebulonDB configuration file."""
    ndb_home = os.environ.get('NEBULONDB_HOME')
    if not ndb_home:
        print("NEBULONDB_HOME environment variable is not set. "
              "Please set it to the NebulonDB installation directory.")
        sys.exit(1)
    return NDBConfig()

# ==========================================================
#        Initialize Logger And Config
# ==========================================================

cfg = _load_config()
NEBULONDB_PID_FILE = cfg.NEBULONDB_PID_FILE
log_dir = cfg.NEBULONDB_LOG_PATH

logger_manager = NebulonDBLogger()
logger_manager.configure_file_logging(log_dir=str(log_dir))
logger = logger_manager.get_logger()

# ==========================================================
#        Initialize Colorama
# ==========================================================

init(autoreset=True)

# ==========================================================
#       Helper Functions
# ==========================================================

def _is_server_running(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0


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
#         Setup NebulonDB Paths
# ==========================================================

def _setup_nebulondb_paths(cfg: NDBConfig):
    neb_home = cfg.NEBULONDB_HOME.resolve()
    if not neb_home.is_dir():
        raise EnvironmentError("NEBULONDB_HOME environment variable is not set or invalid")
    if str(neb_home) not in sys.path:
        sys.path.append(str(neb_home))


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


# ==========================================================
#         Start Server Command
# ==========================================================

def start_server(cfg: NDBConfig, foreground: bool = False):
    accounthub_corpus_path = cfg.NEBULONDB_ACCOUNTHUB_CORPUS_PATH
    default_corpus_path = cfg.NEBULONDB_DEFAULT_CORPUS_PATH

    clear_pycache()

    if not (accounthub_corpus_path.exists() and default_corpus_path.exists()):
        logger.info("Please create user credentials first using:")
        logger.info("nebulondb --create-user")
        return

    # Handle stale PID file
    if NEBULONDB_PID_FILE.exists():
        try:
            pid = int(NEBULONDB_PID_FILE.read_text().strip())
            if _is_pid_alive(pid):
                logger.info("Server is already running (PID %s).", pid)
                return
            else:
                logger.warning("Found stale PID file (PID %s). Removing it.", pid)
                NEBULONDB_PID_FILE.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError):
            NEBULONDB_PID_FILE.unlink(missing_ok=True)

    if _is_server_running(cfg.HOST, cfg.PORT):
        logger.info("Server is already listening on %s:%s.", cfg.HOST, cfg.PORT)
        return

    initializer = NebulonInitializer()
    initializer.initialize()

    # --- Resolve gunicorn log file targets (default to NDB log dir) ---
    log_path = cfg.NEBULONDB_LOG_PATH
    if cfg.ACCESS_LOGFILE and cfg.ACCESS_LOGFILE != "-":
        access_logfile = str(cfg.ACCESS_LOGFILE)
    else:
        access_logfile = str(log_path / "access" / "access.log")
    if cfg.ERROR_LOGFILE and cfg.ERROR_LOGFILE != "-":
        error_logfile = str(cfg.ERROR_LOGFILE)
    else:
        error_logfile = str(log_path / "error" / "error.log")

    module_path = "ndb_host.main"
    cmd = [
        sys.executable, "-m", "gunicorn",
        f"{module_path}:app",
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"{cfg.HOST}:{cfg.PORT}",
        "--workers", str(cfg.WORKERS),
        "--access-logfile", access_logfile,
        "--error-logfile", error_logfile,
    ]
    if cfg.TIMEOUT:          cmd += ["--timeout", str(cfg.TIMEOUT)]
    if cfg.KEEP_ALIVE:       cmd += ["--keep-alive", str(cfg.KEEP_ALIVE)]
    if cfg.GRACEFUL_TIMEOUT: cmd += ["--graceful-timeout", str(cfg.GRACEFUL_TIMEOUT)]
    if cfg.LOG_LEVEL:        cmd += ["--log-level", str(cfg.LOG_LEVEL)]

    print(f"Starting {cfg.APP_NAME} on {cfg.HOST}:{cfg.PORT} with {cfg.WORKERS} workers...")
    print(Fore.CYAN + Style.BRIGHT + pyfiglet.figlet_format(cfg.APP_NAME.upper(), font="smslant"))

    if foreground:
        # ---- Foreground mode: output to terminal, block until stopped ----
        stdout = None
        stderr = None
    else:
        # ---- Background mode: always persist gunicorn logs to files ----
        Path(access_logfile).parent.mkdir(parents=True, exist_ok=True)
        Path(error_logfile).parent.mkdir(parents=True, exist_ok=True)
        stdout = open(access_logfile, "a")
        stderr = open(error_logfile, "a")

    kwargs = {
        "stdout": stdout,
        "stderr": stderr,
        "stdin": subprocess.DEVNULL,
        "cwd": cfg.NEBULONDB_HOME,
    }
    if platform.system() != "Windows":
        kwargs["start_new_session"] = True
    if platform.system() == "Windows" and not foreground:
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW

    process = subprocess.Popen(cmd, **kwargs)
    NEBULONDB_PID_FILE.write_text(str(process.pid))
    print(f"Server started with PID {process.pid}.")
    print(f"Web console available at: http://{cfg.HOST}:{cfg.PORT}/api/NebulonDB/dashboard/")
    logger.info("Server started with PID %s.", process.pid)

    if foreground:
        print("Running in foreground. Press Ctrl+C to stop.")
        try:
            process.wait()
            print("Server exited normally.")
        except KeyboardInterrupt:
            print("\nShutting down server...")
            _kill_process(process.pid)
            process.wait()
        finally:
            if NEBULONDB_PID_FILE.exists():
                NEBULONDB_PID_FILE.unlink(missing_ok=True)
    # else: background mode – we return immediately


# ==========================================================
#         Stop Server Command
# ==========================================================

def stop_server(cfg: NDBConfig, force: bool = False):
    if not NEBULONDB_PID_FILE.exists():
        logger.info("PID file not found – server is not managed by this script.")
        if not _is_server_running(cfg.HOST, cfg.PORT):
            return
        if force:
            pid = _find_pid_on_port(cfg.PORT)
            if pid is None:
                logger.error(
                    "Could not find a process listening on %s:%s. Stop it manually.",
                    cfg.HOST, cfg.PORT
                )
                return
            logger.info("Force stopping manually-started server (PID %s)...", pid)
            _kill_process_tree(pid)
            time.sleep(1)
            if not _is_server_running(cfg.HOST, cfg.PORT):
                logger.info("Server stopped successfully.")
            else:
                logger.error("Failed to force stop server. Port %s:%s is still in use.", cfg.HOST, cfg.PORT)
            return
        logger.warning(
            "Server appears to be running on %s:%s but was started manually. "
            "Use 'nebulondb stop --force' to stop it.",
            cfg.HOST, cfg.PORT
        )
        return

    try:
        pid = int(NEBULONDB_PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        logger.error("PID file is corrupt. Removing it.")
        NEBULONDB_PID_FILE.unlink(missing_ok=True)
        return

    if not _is_pid_alive(pid):
        logger.info("Process with PID %s is not running. Cleaning up PID file.", pid)
        NEBULONDB_PID_FILE.unlink(missing_ok=True)
        return

    logger.info("Stopping %s (PID %s)...", cfg.APP_NAME, pid)
    if force:
        _kill_process_tree(pid)
    else:
        _kill_process(pid)
    time.sleep(1)
    if not _is_pid_alive(pid):
        logger.info("Server stopped successfully.")
        NEBULONDB_PID_FILE.unlink(missing_ok=True)
    else:
        logger.error("Failed to stop server. PID %s is still alive.", pid)


# ==========================================================
#         Restart Server Command
# ==========================================================

def restart_server(cfg: NDBConfig, foreground: bool = False, force: bool = False):
    if _is_server_running(cfg.HOST, cfg.PORT) or NEBULONDB_PID_FILE.exists():
        stop_server(cfg, force=force)
        time.sleep(2)
    start_server(cfg, foreground)


# ==========================================================
#         Create User Command
# ==========================================================

def create_user(cfg: NDBConfig):
    accounthub_corpus_path = cfg.NEBULONDB_ACCOUNTHUB_CORPUS_PATH
    default_corpus_path = cfg.NEBULONDB_DEFAULT_CORPUS_PATH

    if accounthub_corpus_path.exists() and default_corpus_path.exists():
        logger.info("Please start the server and create the user through it")
        return

    username = input("Enter username: ").strip()
    password = getpass("Enter password: ").strip()
    confirm = getpass("Confirm password: ").strip()

    if password != confirm:
        logger.info("Passwords do not match. Try again.")
        return

    user_role = input("Enter role (super_user/admin_user/user) [default=user]: ").strip() or "user"
    initializer = NebulonInitializer()
    initializer.bootstrap(username=username, password=password, user_role=user_role)


# ==========================================================
#         Main Entry Point
# ==========================================================

def main():
    if len(sys.argv) < 2:
        logger.info("Usage: nebulondb {start|stop|restart|--create-user} [--foreground|-f] [--force|-F]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command in ("--help", "-h", "help"):
        print("Usage: nebulondb {start|stop|restart|--create-user} [--foreground|-f] [--force|-F]")
        sys.exit(0)

    _setup_nebulondb_paths(cfg)

    # Check for foreground flag
    foreground = "--foreground" in sys.argv or "-f" in sys.argv
    # Check for force flag
    force = "--force" in sys.argv or "-F" in sys.argv

    if command == "start":
        start_server(cfg, foreground=foreground)
    elif command == "stop":
        stop_server(cfg, force=force)
    elif command == "restart":
        restart_server(cfg, foreground=foreground, force=force)
    elif command == "--create-user":
        create_user(cfg)
    else:
        logger.error("Invalid command. Usage: nebulondb {start|stop|restart|--create-user} [--foreground|-f] [--force|-F]")
        sys.exit(1)


if __name__ == "__main__":
    main()