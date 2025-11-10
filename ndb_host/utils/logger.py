import logging
from datetime import datetime, timezone

class TZFormatter(logging.Formatter):
    """Custom formatter that adds the local timezone offset automatically."""

    def formatTime(self, record, datefmt=None):
        local_time = datetime.fromtimestamp(record.created).astimezone()
        tz_offset = local_time.strftime("%z")  # e.g. +0530
        if datefmt:
            return local_time.strftime(f"{datefmt} {tz_offset}")
        else:
            return local_time.strftime(f"%Y-%m-%d %H:%M:%S {tz_offset}")

handler = logging.StreamHandler()
handler.setFormatter(
    TZFormatter("[%(asctime)s] [%(process)d] [%(levelname)s] NebulonDB: %(message)s")
)
logger = logging.getLogger("nebulondb")
logger.setLevel(logging.INFO)
logger.addHandler(handler)