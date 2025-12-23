import logging
from datetime import datetime
from typing import Optional
from pathlib import Path
from logging.handlers import RotatingFileHandler

from utils.constants import NDBCorpusMeta


class TZFormatter(logging.Formatter):
    """Custom formatter that adds the local timezone offset automatically."""

    def formatTime(self, record, datefmt: Optional[str] = None) -> str:
        local_time = datetime.fromtimestamp(record.created).astimezone()
        tz_offset = local_time.strftime("%z")  # e.g., +0530
        if datefmt:
            return local_time.strftime(f"{datefmt} {tz_offset}")
        else:
            return local_time.strftime(f"%Y-%m-%d %H:%M:%S {tz_offset}")


class NebulonDBLogger:
    _instance = None
    _logger = None
    app_name = NDBCorpusMeta.APP_NAME

    def __new__(cls, level=logging.INFO):
        if cls._instance is None:
            if not cls.app_name:
                raise RuntimeError("Logger not initialized with app_name")
            cls._instance = super().__new__(cls)
            cls._logger = cls._create_logger(cls.app_name, level)
        return cls._instance

    @staticmethod
    def _create_logger(app_name: str, level: int) -> logging.Logger:
        logger = logging.getLogger(app_name)
        logger.setLevel(level)

        # Avoid adding duplicate handlers if logger already exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = TZFormatter("[%(asctime)s] [%(process)d] [%(levelname)s] NebulonDB: %(message)s")
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def configure_file_logging(self, log_dir: str, log_structure: list = NDBCorpusMeta.LOG_STRUCTURE):
        """Configure file handlers for the specified log structure."""
        
        app_name = NebulonDBLogger.app_name
        log_path = Path(log_dir)
        formatter = TZFormatter("[%(asctime)s] [%(process)d] [%(levelname)s] NebulonDB: %(message)s")

        # === App Logger (app.log) ===
        if "app" in log_structure:
            file_path = log_path / "app" / "app.log"
            self._add_file_handler(self._logger, file_path, logging.INFO, formatter)

        # === Error Logger (error.log) - attach to main logger but specifically for errors ===
        if "error" in log_structure:
            file_path = log_path / "error" / "error.log"
            self._add_file_handler(self._logger, file_path, logging.ERROR, formatter)

        # === Access Logger ===
        if "access" in log_structure:
            access_logger = logging.getLogger(f"{app_name}.access")
            access_logger.setLevel(logging.INFO)
            file_path = log_path / "access" / "access.log"
            self._add_file_handler(access_logger, file_path, logging.INFO, formatter)

        # === Audit Logger ===
        if "audit" in log_structure:
            audit_logger = logging.getLogger(f"{app_name}.audit")
            audit_logger.setLevel(logging.INFO)
            file_path = log_path / "audit" / "audit.log"
            self._add_file_handler(audit_logger, file_path, logging.INFO, formatter)

    def _add_file_handler(self, logger, file_path, level, formatter):
        
        # === Ensure directory exists ===
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # === Avoid duplicate handlers ===
        for h in logger.handlers:
            if isinstance(h, RotatingFileHandler) and str(h.baseFilename) == str(file_path.resolve()):
                return

        handler = RotatingFileHandler(file_path, maxBytes=10*1024*1024, backupCount=5)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        if cls._instance is None:
             raise RuntimeError("NebulonDBLogger not initialized. Call NebulonDBLogger(app_name) first.")
        
        if name == "access":
            return logging.getLogger(f"{cls.app_name}.access")
        elif name == "audit":
            return logging.getLogger(f"{cls.app_name}.audit")

        if cls._logger is None:
            raise RuntimeError("NebulonDBLogger not initialized correctly.")
            
        return cls._logger