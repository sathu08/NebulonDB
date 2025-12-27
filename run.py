import sys
import os
import subprocess
import socket

import platform
import time
import pyfiglet

from getpass import getpass
from colorama import init

from pathlib import Path
from colorama import Fore, Style

from ndb_host.db.ndb_settings import NDBConfig
from ndb_host.utils.bootstrap import NebulonInitializer
from ndb_host.utils.logger import NebulonDBLogger


# ==========================================================
#         NebulonDB Runner
# ==========================================================

def _load_config() -> NDBConfig:
    """Load the NebulonDB configuration file."""
    
    ndb_home = os.environ.get('NEBULONDB_HOME')
    if not ndb_home:
        print(f"NEBULONDB_HOME environment variable is not set. Please set it to the NebulonDB installation directory.")
        sys.exit(1)
    return NDBConfig()

# ==========================================================
#        Initialize Logger
# ==========================================================

cfg = _load_config()
log_dir = Path(cfg.NEBULONDB_LOG)
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
    """Check if the server is running on the specified host and port."""
    
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(1)
        return sock.connect_ex((host, port)) == 0

# ==========================================================
#         Setup NebulonDB Paths
# ==========================================================

def _setup_nebulondb_paths(cfg: NDBConfig):
    """
    Initialize NebulonDB paths and update sys.path.
    Ensures NEBULONDB_HOME is valid and adds it to sys.path.
    """

    neb_home = Path(cfg.NEBULONDB_HOME).resolve()
    if not neb_home.is_dir():
        raise EnvironmentError("NEBULONDB_HOME environment variable is not set or invalid")

    # Add to sys.path
    if str(neb_home) not in sys.path:
        sys.path.append(str(neb_home))  

# ==========================================================
#         Start Server Command
# ==========================================================

def start_server(cfg: NDBConfig):
    """Start the Gunicorn/Uvicorn server."""

    secrets_path = Path(cfg.NEBULONDB_SECRETS)
    
    # === Ensure user credentials exist ===
    if not secrets_path.exists():
        logger.info("Please create user credentials first using:")
        logger.info("python run.py --create-user")
        return
    
    # === Check if server is already running ===
    if _is_server_running(cfg.HOST, cfg.PORT):
        logger.info("Server is already running.")
        return
    
    # === Initialize log directories ===
    initializer = NebulonInitializer()
    initializer.initialize()
    
    # === Start server ===
    module_path = "ndb_host.main"
    cmd = [
        sys.executable, "-m", "gunicorn",
        f"{module_path}:app",
        "-k", "uvicorn.workers.UvicornWorker",
        "--bind", f"{cfg.HOST}:{cfg.PORT}",
        "--workers", str(cfg.WORKERS),
    ]

    if cfg.TIMEOUT: cmd += ["--timeout", str(cfg.TIMEOUT)]
    if cfg.KEEP_ALIVE: cmd += ["--keep-alive", str(cfg.KEEP_ALIVE)]
    if cfg.GRACEFUL_TIMEOUT: cmd += ["--graceful-timeout", str(cfg.GRACEFUL_TIMEOUT)]
    if cfg.ACCESS_LOGFILE: cmd += ["--access-logfile", str(cfg.ACCESS_LOGFILE)]
    if cfg.ERROR_LOGFILE: cmd += ["--error-logfile", str(cfg.ERROR_LOGFILE)]
    if cfg.LOG_LEVEL: cmd += ["--log-level", str(cfg.LOG_LEVEL)]

    print(f"Starting {cfg.APP_NAME} on {cfg.HOST}:{cfg.PORT} with {cfg.WORKERS} workers...")
    
    print("\n")
    print(Fore.CYAN + Style.BRIGHT + pyfiglet.figlet_format((cfg.APP_NAME).upper(), font="smslant"))

    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

# ==========================================================
#         Stop Server Command
# ==========================================================

def stop_server(cfg: NDBConfig):
    """Stop the running server."""

    # === Check if server is running ===
    if not _is_server_running(cfg.HOST, cfg.PORT):
        logger.info("Server is not running.")
        return

    logger.info(f"Stopping {cfg.APP_NAME}...")
    system = platform.system()

    try:
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], stdout=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "gunicorn.*main:app"], stdout=subprocess.DEVNULL)
        logger.info("Server stopped.")
    except Exception as e:
        logger.exception(f"Failed to stop server: {e}")

# ==========================================================
#         Restart Server Command
# ==========================================================

def restart_server(cfg: NDBConfig):
    """Restart the server."""

    # === Check if server is running ===
    if not _is_server_running(cfg.HOST, cfg.PORT):
        logger.info("Server is not running.")
        return
    
    stop_server(cfg)
    time.sleep(2)
    start_server(cfg)

# ==========================================================
#         Create User Command
# ==========================================================

def create_user(cfg: NDBConfig):
    """Create an initial user credential file in NEBULONDB_SECRETS."""

    secrets_path = Path(cfg.NEBULONDB_SECRETS)
    
    if secrets_path.exists():
        logger.info("Please start the server and create the user through it")
        return
    
    secrets_dir = Path(cfg.VECTOR_STORAGE) / secrets_path.stem
    secrets_dir.mkdir(parents=True, exist_ok=True)
    creds_path = secrets_dir / "users.json"

    if creds_path.exists():
        choice = input("User credentials already exist. Overwrite? (y/n): ").strip().lower()
        if choice != "y":
            logger.info("User creation cancelled.")
            return

    username = input("Enter username: ").strip()
    password = getpass("Enter password: ").strip()
    confirm = getpass("Confirm password: ").strip()
    user_role = input("Enter role (super_user/admin_user/user) [default=user]: ").strip() or "user"

    if password != confirm:
        logger.info("Passwords do not match. Try again.")
        return

    initializer = NebulonInitializer()
    initializer.bootstrap(
        username=username, password=password, creds_path=creds_path, 
        secrets_dir=secrets_dir, user_role=user_role)

# ==========================================================
#         Main Entry Point
# ==========================================================

def main():
    if len(sys.argv) < 2:
        logger.info("Usage: python run.py {start|stop|restart|--create-user}")
        sys.exit(1)

    command = sys.argv[1].lower()
    _setup_nebulondb_paths(cfg)

    if command == "start":
        start_server(cfg)
    elif command == "stop":
        stop_server(cfg)
    elif command == "restart":
        restart_server(cfg)
    elif command == "--create-user":
        create_user(cfg)
    else:
        logger.error("Invalid command. Usage: python run.py {start|stop|restart|--create-user}")
        sys.exit(1)


if __name__ == "__main__":
    main()
