"""
NDB User Service
==========================================================

This module handles user authentication and authorization for the NDB API.

"""

from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import threading

from typing import Any

from utils.constants import NDBMeta
from db.ndb_settings import NDBConfig
from utils.logger import NebulonDBLogger

from db.index_manager import ComosDBManager
from utils.models import AuthenticationResult, UserRole, StandardErrorResponse

from utils.time_utils import utc_now_iso
from core.security import verify_password, hash_password


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Security and Config Initialization
# ==========================================================

http_basic_security = HTTPBasic()
config_settings = NDBConfig()

# ==========================================================
#        User Manager (In-Memory Singleton)
# ==========================================================

class UserManager:
    """
    Singleton manager for user data, backed by NebulonCosmos.
    In‑memory cache for fast reads; thread‑safe writes and reads.
    """

    _instance = None
    _lock = threading.RLock()
    _users_cache: dict[str, dict[str, Any]]
    _cache_loaded: bool
    _db_manager: ComosDBManager | None
    _segment_name = NDBMeta.Corpus.DEFAULT_SEGMENT_NAME

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._users_cache = {}
                    instance._cache_loaded = False
                    instance._db_manager = None
                    cls._instance = instance
        return cls._instance

    def _get_db_manager(self) -> ComosDBManager:
        """Lazy initialize the Cosmos DB manager."""
        if self._db_manager is None:
            with self._lock:
                if self._db_manager is None:
                    self._db_manager = ComosDBManager(
                        config_settings.NEBULONDB_ACCOUNTHUB_CORPUS_PATH
                    )
                    logger.info("ComosDBManager initialized.")
        return self._db_manager

    def _read_table(self, segment: str) -> list[dict[str, Any]]:
        """Read all records from a segment."""
        return self._get_db_manager().read_data(segment)

    def _ensure_cache_loaded(self):
        """Load all users from the database into the in‑memory cache (under lock)."""
        if not self._cache_loaded:
            with self._lock:
                if not self._cache_loaded:
                    try:
                        users = self._read_table(self._segment_name)
                        cache = {}
                        for doc in users:
                            username = doc.get("username")
                            if username:
                                cache[username] = {
                                    "id": doc.get("_id"),
                                    "password": doc.get("password"),
                                    "role": doc.get("role", "user"),
                                    "created_at": doc.get("created_at")
                                }
                        self._users_cache = cache
                        self._cache_loaded = True
                        logger.info(f"User cache loaded: {len(cache)} users.")
                    except Exception:
                        logger.exception("Failed to load user cache")
                        self._users_cache.clear()
                        self._cache_loaded = False

    # ==========================================================
    #  Public API – thread‑safe, cache‑first
    # ==========================================================

    def get_user(self, username: str) -> dict[str, Any] | None:
        """Retrieve a user by username from cache (loading cache if needed)."""
        self._ensure_cache_loaded()
        with self._lock:
            return self._users_cache.get(username)

    def get_all_users(self) -> list[dict[str, Any]]:
        """Return a list of all users (public fields only, no passwords)."""
        self._ensure_cache_loaded()
        with self._lock:
            return [
                {
                    "id": v["id"],
                    "username": k,
                    "role": v["role"],
                    "created_at": v["created_at"]
                }
                for k, v in self._users_cache.items()
            ]

    def create_user(self, username: str, user_data: dict[str, Any]) -> bool:
        """
        Insert a new user into the database and update the cache.
        user_data must contain 'password', 'role', 'created_at'.
        Returns True on success, False if user already exists.

        Args:
            auto_flush: If True (default), flush immediately after insert.
                        Set to False when batching multiple inserts, then
                        call flush_db() once after all inserts are done.
        """
        with self._lock:
            self._ensure_cache_loaded()  # ensure cache is fresh
            if username in self._users_cache:
                return False

            db = self._get_db_manager()
            # Build the document to store
            doc = {
                "_segment": "users",
                "username": username,
                "password": user_data["password"],
                "role": user_data["role"],
                "created_at": user_data["created_at"]
            }
            try:
                # Insert into the 'users' segment
                record_id = db.insert_data(self._segment_name, doc)
            except Exception:
                logger.exception(f"Failed to insert user '{username}'")
                return False

            # Update the cache
            self._users_cache[username] = {
                "id": record_id,
                "password": user_data["password"],
                "role": user_data["role"],
                "created_at": user_data["created_at"]
            }
            return True

    def delete_user(self, username: str) -> bool:
        """Delete a user from the database and remove from cache."""
        with self._lock:
            self._ensure_cache_loaded()
            if username not in self._users_cache:
                return False

            user_record = self._users_cache[username]
            record_id = user_record["id"]
            db = self._get_db_manager()
            try:
                db.delete_data(self._segment_name, record_id)
            except Exception:
                logger.exception(f"Failed to delete user '{username}'")
                return False

            # Remove from cache
            del self._users_cache[username]
            return True

    def update_password(self, username: str, new_hashed_password: str) -> bool:
        """Update a user's password hash in the database and refresh the cache."""
        with self._lock:
            self._ensure_cache_loaded()
            if username not in self._users_cache:
                return False

            user_record = self._users_cache[username]
            record_id = user_record["id"]
            doc = {
                "_id": record_id,
                "_segment": "users",
                "username": username,
                "password": new_hashed_password,
                "role": user_record["role"],
                "created_at": user_record["created_at"],
                "password_changed_at": utc_now_iso()
            }
            db = self._get_db_manager()
            try:
                db.update_data(self._segment_name, doc)
            except Exception:
                logger.exception(f"Failed to update password for user '{username}'")
                return False

            user_record["password"] = new_hashed_password
            user_record["password_changed_at"] = doc["password_changed_at"]
            return True

# Global Instance
user_manager = UserManager()

def _validate_user_role(user_role: str) -> UserRole:
    try:
        return UserRole(user_role)
    except ValueError:
        return None


# ==========================================================
#        Authentication Functions
# ==========================================================
def get_current_user(credentials: HTTPBasicCredentials = Depends(http_basic_security)) -> AuthenticationResult:
    try:
        logger.debug(f"Attempting authentication for user: {credentials.username}")

        user_record = user_manager.get_user(credentials.username)

        hashed_password = user_record.get("password") if user_record else None
        if not hashed_password or not verify_password(credentials.password, hashed_password):
            logger.warning(f"Authentication failed for user: {credentials.username}")
            return AuthenticationResult(
                username=credentials.username,
                is_authenticated=False,
                message="Invalid credentials"
            )

        user_role = UserRole(user_record.get("role", UserRole.USER.value))

        logger.info(f"Authentication successful for user: {credentials.username}")
        return AuthenticationResult(username=credentials.username, role=user_role)

    except Exception as e:
        logger.error(f"Unexpected authentication error: {e}")
        return AuthenticationResult(
            username=credentials.username if credentials else None,
            is_authenticated=False,
            message="Authentication service error"
        )


# ==========================================================
#        User Management
# ==========================================================

def create_user(username: str, password: str, user_role: str = UserRole.USER.value) -> dict[str, str]:
    try:
        logger.info(f"Attempting to create user: {username} with role: {user_role}")

        if not username or not username.strip():
            return StandardErrorResponse(success=False, message="Username cannot be empty").model_dump()

        if not password or len(password) < 8:
            return StandardErrorResponse(
                success=False, message="Password must be at least 8 characters long"
            ).model_dump()

        validated_role = _validate_user_role(user_role)
        hashed_password = hash_password(password)

        user_data = {
            "password": hashed_password,
            "role": validated_role.value,
            "created_at": utc_now_iso()
        }

        if user_manager.get_user(username):
             logger.warning(f"User creation failed - user already exists: {username}")
             return StandardErrorResponse(success=False, message="User already exists").model_dump()

        success = user_manager.create_user(username, user_data)

        if success:
            logger.info(f"User created successfully: {username} with role: {validated_role.value}")
            return {
                "success":True,
                "message": f"User '{username}' registered successfully with role '{validated_role.value}'",
                "username": username,
                "role": validated_role.value
            }
        else:
             return StandardErrorResponse(success=False, message="User already exists (race condition)").model_dump()

    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return StandardErrorResponse(success=False, message="Error creating user").model_dump()


# ==========================================================
#        User Deletion
# ==========================================================

def delete_user(username: str) -> dict[str, str]:
    try:
        logger.info(f"Attempting to delete user: {username}")

        if not user_manager.get_user(username):
            logger.warning(f"User deletion failed - user not found: {username}")
            return {"success": False, "message": "User not found"}

        user_manager.delete_user(username)

        logger.info(f"User deleted successfully: {username}")
        return {"success": True, "message": f"User '{username}' deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return StandardErrorResponse(success=False, message="Error deleting user").model_dump()


# ==========================================================
#        Password Change
# ==========================================================

def change_password(username: str, current_password: str, new_password: str) -> dict[str, str]:
    try:
        logger.info(f"Attempting to change password for user: {username}")

        user_record = user_manager.get_user(username)
        if not user_record:
            logger.warning(f"Password change failed - user not found: {username}")
            return {"success": False, "message": "User not found"}

        hashed_password = user_record.get("password")
        if not hashed_password or not verify_password(current_password, hashed_password):
            logger.warning(f"Password change failed - current password incorrect: {username}")
            return {"success": False, "message": "Current password is incorrect"}

        if not new_password or len(new_password) < 8:
            return {"success": False, "message": "Password must be at least 8 characters long"}

        new_hashed_password = hash_password(new_password)

        success = user_manager.update_password(username, new_hashed_password)
        if not success:
            return {"success": False, "message": "Failed to update password"}

        logger.info(f"Password changed successfully for user: {username}")
        return {"success": True, "message": f"Password for user '{username}' changed successfully"}

    except Exception as e:
        logger.error(f"Error changing password: {e}")
        return StandardErrorResponse(success=False, message="Error changing password").model_dump()


# ==========================================================
#        User Retrieval
# ==========================================================

def get_all_users() -> dict[str, Any]:
    try:
        logger.info("Retrieving all users")

        users = user_manager.get_all_users()

        safe_users = {
            username: {
                "role": user_data.get("role", UserRole.USER.value),
                "created_at": user_data.get("created_at", "Unknown")
            }
            for username, user_data in users.items()
        }

        logger.info(f"Retrieved {len(safe_users)} users")

        return {
            "success":True,
            "users": safe_users,
            "total_count": len(safe_users)
        }

    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        return StandardErrorResponse(success=False, message="Error retrieving user list").model_dump()

