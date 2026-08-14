# ndb_host/tui/server_ops.py
import sys
import time

import pyfiglet
import platform
import subprocess

from pathlib import Path

from getpass import getpass
from colorama import init

from ndb_host.utils.constants import NDBMeta
from ndb_host.db.ndb_settings import NDBConfig

from .context import logger, NEBULONDB_PID_FILE, tui_mode
from ndb_host.utils.logger import Fore, Style
from .bootstrap import NebulonInitializer
from .processes import (
    _is_server_running,
    _is_pid_alive,
    _kill_process_tree,
    _kill_process,
    _find_pid_on_port,
    clear_pycache,
)


# ==========================================================
#        Initialize Colorama
# ==========================================================

init(autoreset=True)

# ==========================================================
#         Start Server Command
# ==========================================================

def start_server(cfg: NDBConfig, foreground: bool = False) -> bool:
    accounthub_corpus_path = cfg.NEBULONDB_ACCOUNTHUB_CORPUS_PATH
    default_corpus_path = cfg.NEBULONDB_DEFAULT_CORPUS_PATH

    if not (accounthub_corpus_path.exists() and default_corpus_path.exists()):
        logger.info("Please create user credentials first using:")
        logger.info("nebulondb --create-user")
        return False

    # Handle stale PID file
    if NEBULONDB_PID_FILE.exists():
        try:
            pid = int(NEBULONDB_PID_FILE.read_text().strip())
            if _is_pid_alive(pid):
                logger.info("Server is already running (PID %s).", pid)
                return False
            else:
                logger.warning("Found stale PID file (PID %s). Removing it.", pid)
                NEBULONDB_PID_FILE.unlink(missing_ok=True)
        except (ValueError, FileNotFoundError):
            NEBULONDB_PID_FILE.unlink(missing_ok=True)

    if _is_server_running(cfg.HOST, cfg.PORT):
        logger.info("Server is already listening on %s:%s.", cfg.HOST, cfg.PORT)
        return False

    logger.info("We are Working on Starting the NebulonDB Server...")
    if cfg.NEBULONDB_CLEAR_CACHE:
        logger.info("Clearing Python bytecode and cache files...")
        clear_pycache()
        logger.info("Cleared Python bytecode and cache files...")
    logger.info("Models will warm up in the background inside each worker after start.")

    # --- Resolve gunicorn log file targets (default to NDB log dir) ---
    log_path = cfg.NEBULONDB_LOG_PATH
    log_file = log_path / time.strftime(NDBMeta.Logging.LOG_FILE)
    if cfg.ACCESS_LOGFILE and cfg.ACCESS_LOGFILE != "-":
        access_logfile = str(cfg.ACCESS_LOGFILE)
    else:
        access_logfile = str(log_file)
    if cfg.ERROR_LOGFILE and cfg.ERROR_LOGFILE != "-":
        error_logfile = str(cfg.ERROR_LOGFILE)
    else:
        error_logfile = str(log_file)

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

    logger.info("Starting %s on %s:%s with %s workers...",
                cfg.APP_NAME, cfg.HOST, cfg.PORT, cfg.WORKERS)
    if not tui_mode():
        print(Fore.CYAN + Style.BRIGHT + pyfiglet.figlet_format(
            cfg.APP_NAME.upper(), font="smslant"))

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
    logger.info("Server started with PID %s.", process.pid)
    logger.info("Web console available at: http://%s:%s/api/NebulonDB/dashboard/",
                cfg.HOST, cfg.PORT)

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

    return True


# ==========================================================
#         Stop Server Command
# ==========================================================

def stop_server(cfg: NDBConfig, force: bool = False) -> bool:
    if not NEBULONDB_PID_FILE.exists():
        logger.info("PID file not found – server is not managed by this script.")
        if not _is_server_running(cfg.HOST, cfg.PORT):
            return False
        if force:
            pid = _find_pid_on_port(cfg.PORT)
            if pid is None:
                logger.error(
                    "Could not find a process listening on %s:%s. Stop it manually.",
                    cfg.HOST, cfg.PORT
                )
                return False
            logger.info("Force stopping manually-started server (PID %s)...", pid)
            _kill_process_tree(pid)
            time.sleep(1)
            if not _is_server_running(cfg.HOST, cfg.PORT):
                logger.info("Server stopped successfully.")
                return True
            logger.error("Failed to force stop server. Port %s:%s is still in use.", cfg.HOST, cfg.PORT)
            return False
        logger.warning(
            "Server appears to be running on %s:%s but was started manually. "
            "Use 'nebulondb stop --force' to stop it.",
            cfg.HOST, cfg.PORT
        )
        return False

    try:
        pid = int(NEBULONDB_PID_FILE.read_text().strip())
    except (ValueError, FileNotFoundError):
        logger.error("PID file is corrupt. Removing it.")
        NEBULONDB_PID_FILE.unlink(missing_ok=True)
        return False

    if not _is_pid_alive(pid):
        logger.info("Process with PID %s is not running. Cleaning up PID file.", pid)
        NEBULONDB_PID_FILE.unlink(missing_ok=True)
        return True

    logger.info("Stopping %s (PID %s)...", cfg.APP_NAME, pid)
    if force:
        _kill_process_tree(pid)
    else:
        _kill_process(pid)
    time.sleep(1)
    if not _is_pid_alive(pid):
        logger.info("Server stopped successfully.")
        NEBULONDB_PID_FILE.unlink(missing_ok=True)
        return True
    logger.error("Failed to stop server. PID %s is still alive.", pid)
    return False


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

def is_initialized(cfg: NDBConfig) -> bool:
    """Return True when the account hub corpus and secrets already exist,
    i.e. the system has been set up (a user has been created)."""
    return (
        cfg.NEBULONDB_ACCOUNTHUB_CORPUS_PATH.exists()
        and cfg.NEBULONDB_DEFAULT_CORPUS_PATH.exists()
    )


def create_user(
    cfg: NDBConfig,
    username: str = None,
    password: str = None,
    user_role: str = None,
) -> bool:
    """Create a user. Prompts interactively when arguments are omitted."""
    if is_initialized(cfg):
        logger.info("System already initialized. Users exist.")
        return False

    if username is None:
        username = input("Enter username: ").strip()

    if password is None:
        password = getpass("Enter password: ").strip()
        confirm = getpass("Confirm password: ").strip()

        if password != confirm:
            logger.info("Passwords do not match. Try again.")
            return False

    if user_role is None:
        user_role = input(
            "Enter role (super_user/admin_user/user) [default=user]: "
        ).strip() or "user"

    if not password or len(password) < 8:
        logger.info("Password must be at least 8 characters long. Try again.")
        return False

    if not username:
        logger.info("Username cannot be empty. Try again.")
        return False

    initializer = NebulonInitializer()
    initializer.bootstrap(username=username, password=password, user_role=user_role)
    return True