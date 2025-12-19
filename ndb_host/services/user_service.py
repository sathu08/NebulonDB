from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from typing import Dict, Any
from datetime import datetime, timezone
from pathlib import Path
from functools import lru_cache

from utils.models import AuthenticationResult, UserRole, StandardErrorResponse, load_data, save_data
from utils.logger import logger
from core.security import verify_password, hash_password
from db.ndb_settings import NDBConfig, NDBSafeLocker

# ------------------------------
# Security and Config Initialization 
# ------------------------------
http_basic_security = HTTPBasic()
config_settings = NDBConfig()

# ------------------------------
# Encrypted User Database 
# ------------------------------
@lru_cache(maxsize=1)
def get_user_db_locker() -> NDBSafeLocker:
    """
    Safely initialize and cache the NDBSafeLocker instance.
    """
    secrets_path = Path(config_settings.NEBULONDB_SECRETS)
    if not secrets_path.exists():
        raise FileNotFoundError(f"Secrets directory not found: {secrets_path}")
    return NDBSafeLocker(secrets_path)


def _validate_user_role(user_role: str) -> UserRole:
    try:
        return UserRole(user_role)
    except ValueError:
        return None


# === Authentication Functions ===
def get_current_user(credentials: HTTPBasicCredentials = Depends(http_basic_security)) -> AuthenticationResult:
    try:
        logger.debug(f"Attempting authentication for user: {credentials.username}")

        locker = get_user_db_locker()
        users = load_data(path_loc=locker.read_file(file_path="users.json", as_text=False),is_bytes_input=True)
        user_record = users.get(credentials.username)
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


# === User Management ===
def create_user(username: str, password: str, user_role: str = UserRole.USER.value, new_creation: bool = False) -> Dict[str, str]:
    try:
        logger.info(f"Attempting to create user: {username} with role: {user_role}")

        if not username or not username.strip():
            return StandardErrorResponse(success=False, message="Username cannot be empty").model_dump()

        if not password or len(password) < 6:
            return StandardErrorResponse(success=False, message="Password must be at least 6 characters long").model_dump()

        validated_role = _validate_user_role(user_role)
        hashed_password = hash_password(password)

        # For first-time creation (no existing DB)
        if new_creation:
            users = {}
            users[username] = {
                "password": hashed_password,
                "role": validated_role.value,
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            return users

        locker = get_user_db_locker()

        # Load existing users
        users = load_data(path_loc=locker.read_file(file_path="users.json", as_text=False),is_bytes_input=True)

        if username in users:
            logger.warning(f"User creation failed - user already exists: {username}")
            return StandardErrorResponse(success=False, message="User already exists").model_dump()

        users[username] = {
            "password": hashed_password,
            "role": validated_role.value,
            "created_at": datetime.now(timezone.utc).isoformat()
        }

        locker.write_file("users.json", save_data(users, return_bytes=True))
        locker.save()

        logger.info(f"User created successfully: {username} with role: {validated_role.value}")
        return {
            "success":True,
            "message": f"User '{username}' registered successfully with role '{validated_role.value}'",
            "username": username,
            "role": validated_role.value
        }

    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return StandardErrorResponse(success=False, message="Error creating user").model_dump()


def delete_user(username: str) -> Dict[str, str]:
    try:
        logger.info(f"Attempting to delete user: {username}")

        locker = get_user_db_locker()
        users = load_data(path_loc=locker.read_file(file_path="users.json", as_text=False), is_bytes_input=True)

        if username not in users:
            logger.warning(f"User deletion failed - user not found: {username}")
            return {"success": False, "message": "User not found"}

        del users[username]
        locker.write_file("users.json", save_data(users, return_bytes=True))
        locker.save()

        logger.info(f"User deleted successfully: {username}")
        return {"success": True, "message": f"User '{username}' deleted successfully"}

    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return StandardErrorResponse(success=False, message="Error deleting user").model_dump()


def get_all_users() -> Dict[str, Any]:
    try:
        logger.info("Retrieving all users")

        locker = get_user_db_locker()
        users = load_data(path_loc=locker.read_file(file_path="users.json", as_text=False), is_bytes_input=True)

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
