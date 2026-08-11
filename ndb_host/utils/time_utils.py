from datetime import datetime, timezone


def utc_now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string."""
    return utc_now().isoformat()


def utc_timestamp() -> int:
    """Return the current UTC Unix timestamp."""
    return int(utc_now().timestamp())
