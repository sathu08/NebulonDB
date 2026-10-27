"""
NDB Core Permissions
==========================================================

This module handles security for the NDB API.
It provides endpoints for user registration and authentication.

"""

import hashlib
import os
import threading
import time

from passlib.context import CryptContext

from utils.models import AuthenticationConfig
from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Security Context
# ==========================================================

password_context = CryptContext(
    schemes=AuthenticationConfig.PASSWORD_HASH_SCHEMES,
    deprecated=AuthenticationConfig.PASSWORD_HASH_DEPRECATED
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Hashed password from database

    Returns:
        bool: True if password matches, False otherwise
    """
    try:
        return password_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        return False

def hash_password(password: str) -> str:
    """
    Hash a plain text password using bcrypt.

    Args:
        password: Plain text password to hash

    Returns:
        str: Hashed password

    Raises:
        ValueError: If password hashing fails
    """
    try:
        return password_context.hash(password)
    except Exception as e:
        logger.error(f"Password hashing error: {e}")
        raise ValueError("Password hashing failed") from e


# ==========================================================
#        Verified-credential cache
# ==========================================================
#
# Every authenticated API call re-runs bcrypt (~250 ms at the default
# cost). A client such as NebulonMind makes 3-4 backend calls per user
# action, so a single chat save used to spend ~1 s purely re-hashing the
# same credentials. Cache *successful* verifications only, keyed by a
# SHA-256 of the supplied password plus the stored hash, so:
#   * failed attempts are never cached (no brute-force shortcut),
#   * a password change yields a new key automatically (stale entries
#     can never authenticate the old password),
#   * plaintext passwords are never retained in memory.
# Bounded + TTL so memory cannot grow without limit.

_AUTH_CACHE_TTL = float(os.environ.get("NDB_AUTH_CACHE_TTL", "60"))
_AUTH_CACHE_MAX = int(os.environ.get("NDB_AUTH_CACHE_MAX", "1024"))
_auth_cache: dict = {}
_auth_cache_lock = threading.Lock()


def _cache_key(username: str, password: str, hashed_password: str) -> tuple:
    digest = hashlib.sha256(password.encode("utf-8", "ignore")).hexdigest()
    return (username, digest, hashed_password)


def verify_password_cached(username: str, password: str, hashed_password: str) -> bool:
    """``verify_password`` with a bounded TTL cache for successful checks."""
    if _AUTH_CACHE_TTL <= 0:
        return verify_password(password, hashed_password)

    now = time.monotonic()
    key = _cache_key(username, password, hashed_password)
    with _auth_cache_lock:
        expiry = _auth_cache.get(key)
        if expiry is not None and expiry > now:
            return True

    ok = verify_password(password, hashed_password)
    if not ok:
        return False

    with _auth_cache_lock:
        if len(_auth_cache) >= _AUTH_CACHE_MAX:
            for stale in [k for k, v in _auth_cache.items() if v <= now]:
                del _auth_cache[stale]
            while len(_auth_cache) >= _AUTH_CACHE_MAX:
                _auth_cache.pop(next(iter(_auth_cache)))
        _auth_cache[key] = now + _AUTH_CACHE_TTL
    return True


def invalidate_auth_cache(username: str | None = None) -> None:
    """Drop cached verifications (all, or just one user's)."""
    with _auth_cache_lock:
        if username is None:
            _auth_cache.clear()
        else:
            for key in [k for k in _auth_cache if k[0] == username]:
                del _auth_cache[key]

