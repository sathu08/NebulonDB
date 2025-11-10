import sys
import os
import subprocess
import platform
import time
from getpass import getpass
from pathlib import Path

from ndb_host.db.ndb_settings import NDBConfig, NDBSafeLocker
from ndb_host.utils.models import save_data
from ndb_host.utils.bootstrap import NebulonInitializer


# ==========================================================
# NebulonDB Runner
# ==========================================================

def _load_config() -> NDBConfig:
    """Load the NebulonDB configuration file."""
    
    ndb_home = os.environ.get('NEBULONDB_HOME')
    if not ndb_home:
        print(f"NEBULONDB_HOME environment variable is not set. Please set it to the NebulonDB installation directory.")
        sys.exit(1)
    return NDBConfig()

# ==========================================================
#  Setup NebulonDB Paths
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
#  Start Server Command
# ==========================================================

def start_server(cfg: NDBConfig):
    """Start the Gunicorn/Uvicorn server."""

    secrets_path = Path(cfg.NEBULONDB_SECRETS)
    
    # === Ensure user credentials exist ===
    if not secrets_path.exists():
        print("Please create user credentials first using:")
        print("python run.py --create-user")
        return
    
    # === Ensure default corpus is present ===
    NebulonInitializer().ensure_default_corpus()

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
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

# ==========================================================
#  Stop Server Command
# ==========================================================

def stop_server(cfg: NDBConfig):
    """Stop the running server."""

    print(f"Stopping {cfg.APP_NAME}...")
    system = platform.system()

    try:
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], stdout=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "gunicorn.*main:app"], stdout=subprocess.DEVNULL)
        print("Server stopped.")
    except Exception as e:
        print(f"Failed to stop server: {e}")

# ==========================================================
#  Restart Server Command
# ==========================================================

def restart_server(cfg: NDBConfig):
    """Restart the server."""

    stop_server(cfg)
    time.sleep(2)
    start_server(cfg)


# ==========================================================
#  Create User Command
# ==========================================================

def create_user(cfg: NDBConfig):
    """Create an initial user credential file in NEBULONDB_SECRETS."""

    secrets_path = Path(cfg.NEBULONDB_SECRETS)
    
    if secrets_path.exists():
        print("Please start the server and create the user through it")
        return
    
    secrets_dir = Path(cfg.VECTOR_STORAGE) / secrets_path.stem
    secrets_dir.mkdir(parents=True, exist_ok=True)
    creds_path = secrets_dir / "users.json"

    if creds_path.exists():
        choice = input("User credentials already exist. Overwrite? (y/n): ").strip().lower()
        if choice != "y":
            print("User creation cancelled.")
            return

    username = input("Enter username: ").strip()
    password = getpass("Enter password: ").strip()
    confirm = getpass("Confirm password: ").strip()
    user_role = input("Enter role (super_user/admin_user/user) [default=user]: ").strip() or "user"

    if password != confirm:
        print("Passwords do not match. Try again.")
        return

    try:
        from ndb_host.services.user_service import create_user as service_create_user

        # === Create internal system user first ===
        system_user_data = service_create_user("nebulon-supernova", password, "system", new_creation=True)
        
        # === Create actual user ===
        normal_user_data = service_create_user(username, password, user_role, new_creation=True)

        # === Merge both user records into one dictionary ===
        combined_users = {**system_user_data, **normal_user_data}

        # === Save user database ===
        save_data(data=combined_users, path_loc=str(creds_path))
        print(f"User created successfully and saved at: {creds_path}")

        # === Encrypt credentials with NDBSafeLocker ===
        NDBSafeLocker(str(secrets_dir))
        print("Credentials secured in NDB format.")

    except Exception as e:
        print(f"Failed to create user: {e}")


# ==========================================================
# Main Entry Point
# ==========================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py {start|stop|restart|--create-user}")
        sys.exit(1)

    command = sys.argv[1].lower()
    cfg = _load_config()
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
        print("Invalid command. Usage: python run.py {start|stop|restart|--create-user}")
        sys.exit(1)


if __name__ == "__main__":
    main()
