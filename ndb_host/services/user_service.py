from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

import threading
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime, timezone

from utils.models import load_data, save_data
from core.security import verify_password, hash_password

from utils.models import AuthenticationResult, UserRole, StandardErrorResponse
from db.ndb_settings import NDBConfig, NDBSafeLocker
from utils.logger import NebulonDBLogger


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
    Singleton class to manage users in memory for high performance.
    Reads from disk once (lazy load), writes to disk on change.
    Thread-safe for writes.
    """
    _instance = None
    _lock = threading.RLock() 
    _users_cache: Optional[Dict[str, Any]] = None

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(UserManager, cls).__new__(cls)
        return cls._instance

    def _get_locker(self) -> NDBSafeLocker:
        """Create a fresh locker instance. DO NOT CACHE THIS."""
        secrets_path = Path(config_settings.NEBULONDB_SECRETS)
        return NDBSafeLocker(secrets_path)

    def _ensure_cache_loaded(self):
        """Load users from disk if not already in memory."""
        if self._users_cache is None:
            with self._lock:
                if self._users_cache is None:
                    try:
                        locker = self._get_locker()
                        self._users_cache = load_data(
                            path_loc=locker.read_file(file_path="users.json", as_text=False),
                            is_bytes_input=True
                        )
                        logger.info("User cache loaded from disk.")
                    except Exception as e:
                        logger.error(f"Failed to load user cache: {e}")
                        self._users_cache = {} # Fallback to empty to prevent crash loops

    def get_user(self, username: str) -> Optional[Dict[str, Any]]:
        """Get user from memory cache (Fast)."""
        self._ensure_cache_loaded()
        return self._users_cache.get(username)

    def get_all_users(self) -> Dict[str, Any]:
        """Get all users from memory cache."""
        self._ensure_cache_loaded()
        return self._users_cache.copy()

    def create_user(self, username: str, data: Dict[str, Any]) -> bool:
        """Update memory and write to disk (Thread-safe)."""
        self._ensure_cache_loaded()
        
        with self._lock:
            if username in self._users_cache:
                return False 
            
            # === 1. Update Memory ===
            self._users_cache[username] = data
            
            # 2. Persist to Disk ===
            try:
                locker = self._get_locker()
                locker.write_file("users.json", save_data(self._users_cache, return_bytes=True))
                locker.save() # This closes the zip, which is fine as we discard 'locker'
                return True
            except Exception as e:
                logger.error(f"Failed to persist user creation: {e}")
                del self._users_cache[username]
                raise e

    def delete_user(self, username: str) -> bool:
        """Delete from memory and write to disk (Thread-safe)."""
        self._ensure_cache_loaded()
        
        with self._lock:
            if username not in self._users_cache:
                return False
            
            # === 1. Update Memory ===
            del self._users_cache[username]
            
            # === 2. Persist to Disk ===
            try:
                locker = self._get_locker()
                locker.write_file("users.json", save_data(self._users_cache, return_bytes=True))
                locker.save()
                return True
            except Exception as e:
                logger.error(f"Failed to persist user deletion: {e}")
                # Ideally reload cache from disk to restore state
                self._users_cache = None 
                raise e

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
        
        if not user_record:
            logger.warning("Invalid credentials")
            return AuthenticationResult(username=credentials.username, is_authenticated=False, message="Invalid username")

        hashed_password = user_record.get("password")
        if not hashed_password:
            logger.error(f"User record missing password hash: {credentials.username}")
            return AuthenticationResult(username=credentials.username, is_authenticated=False, message="Invalid password")

        if not verify_password(credentials.password, hashed_password):
            logger.warning(f"Authentication failed - invalid password: {credentials.username}")
            return AuthenticationResult(username=credentials.username, is_authenticated=False, message="Invalid password")

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

def create_user(username: str, password: str, user_role: str = UserRole.USER.value, new_creation: bool = False) -> Dict[str, str]:
    try:
        logger.info(f"Attempting to create user: {username} with role: {user_role}")

        if not username or not username.strip():
            return StandardErrorResponse(success=False, message="Username cannot be empty").model_dump()

        if not password or len(password) < 6:
            return StandardErrorResponse(success=False, message="Password must be at least 6 characters long").model_dump()

        validated_role = _validate_user_role(user_role)
        hashed_password = hash_password(password)

        if not hashed_password:
            return StandardErrorResponse(success=False, message="Password hashing failed").model_dump()

        user_data = {
            "password": hashed_password,
            "role": validated_role.value,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        # === For first-time creation (script usage, bypass manager logic potentially?) ===
        if new_creation:
            # If new_creation is True, it implies we just want the dict back to initialize the DB file manually
            # This is likely used by run.py --create-user before the DB even exists.
            users = {}
            users[username] = user_data
            return users

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

def delete_user(username: str) -> Dict[str, str]:
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
#        User Retrieval
# ==========================================================

def get_all_users() -> Dict[str, Any]:
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

