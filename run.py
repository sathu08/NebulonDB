import os
import sys
import subprocess
import platform
import time
from ndb_host.db.NebulonDBConfig import NebulonDBConfig


# ==========================================================
# NebulonDB Runner
# ==========================================================

def load_config() -> NebulonDBConfig:
    """Load the NebulonDB configuration file."""
    config_path = os.path.join(os.getcwd(), "nebulondb.cfg")
    if not os.path.exists(config_path):
        print(f"X Configuration file not found at {config_path}")
        sys.exit(1)
    return NebulonDBConfig(config_path)

def setup_nebulondb_paths(cfg: NebulonDBConfig):
    """
    Initialize NebulonDB paths and update sys.path.
    Ensures NEBULONDB_HOME is valid and adds it to sys.path.
    Raises EnvironmentError if NEBULONDB_HOME is not set or invalid.
    """
    neb_home = cfg.NEBULONDB_HOME
    if not neb_home or not os.path.isdir(neb_home):
        raise EnvironmentError("NEBULONDB_HOME environment variable is not set or invalid")

    # Add to sys.path
    if neb_home not in sys.path:
        sys.path.append(neb_home)

def start_server(cfg: NebulonDBConfig):
    """Start the Gunicorn/Uvicorn server."""
    module_path = "ndb_host.main"
    main_py = os.path.join(cfg.NEBULONDB_HOME, "ndb_host", "main.py")

    if not os.path.exists(main_py):
        print(f"X Cannot find {main_py}")
        sys.exit(1)

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

    print(f"** Starting {cfg.APP_NAME} on {cfg.HOST}:{cfg.PORT} with {cfg.WORKERS} workers...")
    print("$$ Command:", " ".join(cmd))
    subprocess.run(cmd)


def stop_server(cfg: NebulonDBConfig):
    """Stop the running server."""
    print(f"O Stopping {cfg.APP_NAME}...")
    system = platform.system()

    try:
        if system == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "python.exe"], stdout=subprocess.DEVNULL)
        else:
            subprocess.run(["pkill", "-f", "gunicorn.*main:app"], stdout=subprocess.DEVNULL)
        print("✅ Server stopped.")
    except Exception as e:
        print(f"! Failed to stop server: {e}")


def restart_server(cfg: NebulonDBConfig):
    """Restart the server."""
    stop_server(cfg)
    time.sleep(2)
    start_server(cfg)


def main():
    if len(sys.argv) < 2:
        print("Usage: python run.py {start|stop|restart}")
        sys.exit(1)

    command = sys.argv[1].lower()
    cfg = load_config()
    setup_nebulondb_paths(cfg)
    if command == "start":
        start_server(cfg)
    elif command == "stop":
        stop_server(cfg)
    elif command == "restart":
        restart_server(cfg)
    else:
        print("Invalid command. Usage: python run.py {start|stop|restart}")
        sys.exit(1)


if __name__ == "__main__":
    main()
