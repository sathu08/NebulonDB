# ndb_host/tui/context.py
import logging
import os
import sys

from ndb_host.db.ndb_settings import NDBConfig
from ndb_host.utils.logger import NebulonDBLogger


def _load_config() -> NDBConfig:
    """Load the NebulonDB configuration file."""
    ndb_home = os.environ.get('NEBULONDB_HOME')
    if not ndb_home:
        print("NEBULONDB_HOME environment variable is not set. "
              "Please set it to the NebulonDB installation directory.")
        sys.exit(1)
    return NDBConfig()


# ==========================================================
#        Shared Runtime Context (cfg / logger / pid file)
# ==========================================================

cfg = _load_config()
NEBULONDB_PID_FILE = cfg.NEBULONDB_PID_FILE
log_dir = cfg.NEBULONDB_LOG_PATH

logger_manager = NebulonDBLogger()
logger_manager.configure_file_logging(
    log_dir=str(log_dir),
    retention_days=cfg.LOG_RETENTION_DAYS,
    auto_delete=cfg.LOG_AUTO_DELETE,
)
logger = logger_manager.get_logger()

is_tui = False

def enable_tui_mode():
    """Disable console logging so nothing corrupts the Textual screen."""
    global is_tui
    is_tui = True
    for log in (logger, logging.getLogger("nebulondb.access"),
                logging.getLogger("nebulondb.audit")):
        for handler in list(log.handlers):
            if isinstance(handler, logging.StreamHandler):
                log.removeHandler(handler)

def tui_mode() -> bool:
    """Return whether the TUI is currently active."""
    return is_tui

def setup_nebulondb_paths():
    """Ensure the NebulonDB home directory is importable."""
    neb_home = cfg.NEBULONDB_HOME.resolve()
    if not neb_home.is_dir():
        raise EnvironmentError("NEBULONDB_HOME environment variable is not set or invalid")
    if str(neb_home) not in sys.path:
        sys.path.append(str(neb_home))
