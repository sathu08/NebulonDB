"""
NDB Logger Utility
==========================================================

This module handles logging for the NDB API.

"""

import logging
import time
import contextlib

from datetime import datetime
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

from colorama import Fore, Style, init as colorama_init
from utils.constants import NDBMeta

# Initialize colorama for cross-platform color support
colorama_init(autoreset=False)


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

    def formatTime(self, record, datefmt: str | None = None) -> str:
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
#        Date-Stamped Daily File Handler
# ==========================================================
class DatedTimedRotatingFileHandler(TimedRotatingFileHandler):
    """
    Rotating file handler that names the current file with the date
    (e.g. nebulondb_2026-08-10.log) and rolls over to a new
    date-stamped file at midnight each day.
    """

    def __init__(
        self,
        log_dir,
        filename_template,
        when: str = "midnight",
        backup_count: int = 7,
        encoding: str = "utf-8",
    ):
        self.log_dir = Path(log_dir)
        self.filename_template = filename_template
        self.log_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            str(self._path_for(datetime.now())),
            when=when,
            backupCount=backup_count,
            encoding=encoding,
        )
        self.rolloverAt = self.computeRollover(int(time.time()))

    def _path_for(self, dt: datetime) -> Path:
        return self.log_dir / dt.strftime(self.filename_template)

    def doRollover(self):
        if self.stream:
            self.stream.close()
            self.stream = None

        if self.backupCount > 0:
            self._cleanup_old()

        # Open a new file named after today's date
        self.baseFilename = str(self._path_for(datetime.now()))
        self.mode = "a"
        self.stream = self._open()
        self.rolloverAt = self.computeRollover(int(time.time()))

    def _cleanup_old(self):
        """Delete date-stamped files older than backup_count."""
        template = self.filename_template
        glob_pattern = (
            template.replace("%Y", "????")
            .replace("%m", "??")
            .replace("%d", "??")
        )
        files = sorted(self.log_dir.glob(glob_pattern), key=lambda p: p.name)
        if len(files) <= self.backupCount:
            return
        for old in files[:-self.backupCount]:
            with contextlib.suppress(OSError):
                old.unlink()


# ==========================================================
#                 Logger Manager (Singleton)
# ==========================================================
class NebulonDBLogger:
    _instance = None
    _logger = None
    app_name = NDBMeta.APP_NAME

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
        retention_days: int = NDBMeta.Logging.DEFAULT_RETENTION_DAYS,
        auto_delete: bool = NDBMeta.Logging.DEFAULT_AUTO_DELETE,
    ):
        """
        Configure a single daily-rotating log file shared by the
        main (app/error), access, and audit loggers.

        Args:
            log_dir: Directory where the log file lives.
            retention_days: Number of daily rotated files to keep
                (only used when auto_delete is enabled).
            auto_delete: If True, delete rotated files older than
                retention_days. If False, keep every day-wise file.
        """
        app_name = self.app_name
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        formatter = TZColoredFormatter(
            "[%(asctime)s] [%(process)d] [%(levelname)s] "
            f"{app_name}: %(message)s"
        )

        # Single shared handler for every log type
        handler = self._get_daily_file_handler(
            log_path,
            NDBMeta.Logging.LOG_FILE,
            logging.INFO,
            formatter,
            retention_days=retention_days,
            auto_delete=auto_delete,
        )

        if handler not in self._logger.handlers:
            self._logger.addHandler(handler)

        access_logger = logging.getLogger(f"{app_name}.access")
        access_logger.setLevel(logging.INFO)
        access_logger.propagate = False
        if handler not in access_logger.handlers:
            access_logger.addHandler(handler)

        audit_logger = logging.getLogger(f"{app_name}.audit")
        audit_logger.setLevel(logging.INFO)
        audit_logger.propagate = False
        if handler not in audit_logger.handlers:
            audit_logger.addHandler(handler)

    # ======================================================
    #              Daily File Handler Helper
    # ======================================================
    def _get_daily_file_handler(
        self,
        log_dir,
        filename_template: str = NDBMeta.Logging.LOG_FILE,
        level=logging.INFO,
        formatter=None,
        retention_days: int = NDBMeta.Logging.DEFAULT_RETENTION_DAYS,
        auto_delete: bool = NDBMeta.Logging.DEFAULT_AUTO_DELETE,
    ):
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Reuse existing handler to avoid duplicates
        for h in self._logger.handlers:
            if isinstance(h, DatedTimedRotatingFileHandler):
                return h

        handler = DatedTimedRotatingFileHandler(
            log_path,
            filename_template,
            when="midnight",
            backup_count=retention_days if auto_delete else 0,
            encoding="utf-8",
        )
        handler.setLevel(level)
        handler.setFormatter(formatter)
        return handler

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
