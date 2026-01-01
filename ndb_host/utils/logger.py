import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler

from colorama import Fore, Style, init as colorama_init

# Initialize colorama for cross-platform color support
colorama_init(autoreset=False)

from utils.constants import NDBCorpusMeta


# ==========================================================
#              Single Formatter (TZ + Color)
# ==========================================================
class TZColoredFormatter(logging.Formatter):
    """Formatter with timezone support and colored log levels."""

    COLORS = {
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Style.BRIGHT,
    }

    def formatTime(self, record, datefmt: Optional[str] = None) -> str:
        local_time = datetime.fromtimestamp(record.created).astimezone()
        tz_offset = local_time.strftime("%z")
        if datefmt:
            return local_time.strftime(f"{datefmt} {tz_offset}")
        return local_time.strftime(f"%Y-%m-%d %H:%M:%S {tz_offset}")

    def format(self, record):
        original_levelname = record.levelname

        # Apply color for console
        if original_levelname in self.COLORS:
            record.levelname = (
                f"{self.COLORS[original_levelname]}"
                f"{original_levelname}"
                f"{Style.RESET_ALL}"
            )

        formatted = super().format(record)

        # Restore to avoid side effects
        record.levelname = original_levelname
        return formatted


# ==========================================================
#                 Logger Manager (Singleton)
# ==========================================================
class NebulonDBLogger:
    _instance = None
    _logger = None
    app_name = NDBCorpusMeta.APP_NAME

    def __new__(cls, level=logging.INFO):
        if cls._instance is None:
            if not cls.app_name:
                raise RuntimeError("APP_NAME not configured")
            cls._instance = super().__new__(cls)
            cls._logger = cls._create_logger(cls.app_name, level)
        return cls._instance

    @staticmethod
    def _create_logger(app_name: str, level: int) -> logging.Logger:
        logger = logging.getLogger(app_name)
        logger.setLevel(level)
        logger.propagate = False

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = TZColoredFormatter(
                "[%(asctime)s] [%(process)d] [%(levelname)s] "
                f"{app_name}: %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    # ======================================================
    #              File Logging Configuration
    # ======================================================
    def configure_file_logging(
        self,
        log_dir: str,
        log_structure: list = NDBCorpusMeta.LOG_STRUCTURE
    ):
        app_name = self.app_name
        log_path = Path(log_dir)

        formatter = TZColoredFormatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] "
            f"{app_name}: %(message)s"
        )

        # === App Log ===
        if "app" in log_structure:
            self._add_file_handler(
                self._logger,
                log_path / "app" / "app.log",
                logging.INFO,
                formatter
            )

        # === Error Log ===
        if "error" in log_structure:
            self._add_file_handler(
                self._logger,
                log_path / "error" / "error.log",
                logging.ERROR,
                formatter
            )

        # === Access Log ===
        if "access" in log_structure:
            access_logger = logging.getLogger(f"{app_name}.access")
            access_logger.setLevel(logging.INFO)
            self._add_file_handler(
                access_logger,
                log_path / "access" / "access.log",
                logging.INFO,
                formatter
            )

        # === Audit Log ===
        if "audit" in log_structure:
            audit_logger = logging.getLogger(f"{app_name}.audit")
            audit_logger.setLevel(logging.INFO)
            self._add_file_handler(
                audit_logger,
                log_path / "audit" / "audit.log",
                logging.INFO,
                formatter
            )

    # ======================================================
    #              File Handler Helper
    # ======================================================
    def _add_file_handler(self, logger, file_path, level, formatter):
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Avoid duplicate handlers
        for h in logger.handlers:
            if isinstance(h, RotatingFileHandler) and \
               Path(h.baseFilename).resolve() == file_path.resolve():
                return

        handler = RotatingFileHandler(
            file_path,
            maxBytes=10 * 1024 * 1024,
            backupCount=5
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # ======================================================
    #              Public Logger Access
    # ======================================================
    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        if cls._instance is None:
            raise RuntimeError("NebulonDBLogger not initialized")

        if name == "access":
            return logging.getLogger(f"{cls.app_name}.access")
        if name == "audit":
            return logging.getLogger(f"{cls.app_name}.audit")

        return cls._logger

    # ======================================================
    #        Configure Gunicorn/Uvicorn Colored Logs
    # ======================================================
    @classmethod
    def configure_server_logging(cls):
        """Apply colored formatter to gunicorn and uvicorn loggers."""
        
        formatter = TZColoredFormatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] %(message)s"
        )
        
        # Configure these server-related loggers
        logger_names = [
            "gunicorn.error",
            "gunicorn.access", 
            "uvicorn",
            "uvicorn.error",
            "uvicorn.access"
        ]
        
        for logger_name in logger_names:
            logger = logging.getLogger(logger_name)
            
            # Replace handlers with colored ones
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.StreamHandler):
                    handler.setFormatter(formatter)
