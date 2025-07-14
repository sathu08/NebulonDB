from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

from typing import Dict, Any
from datetime import datetime, timezone
from pathlib import Path

from utils.models import AuthenticationResult, UserRole, load_data, save_data
from utils.logger import logger
from core.security import verify_password, hash_password
from db.index_manager import NebulonDBConfig

http_basic_security = HTTPBasic()

# === Database Path Configuration ===
USER_DATABASE_PATH = Path(NebulonDBConfig.USER_CREDENTIALS)

def _validate_user_role(user_role: str) -> UserRole:
    try:
        return UserRole(user_role)
    except ValueError:
        return None

# === Authentication Functions ===
def get_current_user(credentials: HTTPBasicCredentials = Depends(http_basic_security)) -> AuthenticationResult:
    try:
        logger.debug(f"Attempting authentication for user: {credentials.username}")
        
        users = load_data(path_loc=USER_DATABASE_PATH)
        user_record = users.get(credentials.username)
        
        if not user_record:
            logger.warning("Invalid credentials")
            return {"success": False, "message": "Invalid username"}
        
        hashed_password = user_record.get("password")
        if not hashed_password:
            logger.error(f"User record missing password hash: {credentials.username}")
            return {"success": False, "message": "Invalid passwords"}
        
        if not verify_password(credentials.password, hashed_password):
            logger.warning(f"Authentication failed - invalid password: {credentials.username}")
            return {"success": False, "message": "Invalid passwords"}
        
        user_role = UserRole(user_record.get("role", UserRole.USER.value))
        
        logger.info(f"Authentication successful for user: {credentials.username}")
        
        return AuthenticationResult(
            username=credentials.username,
            role=user_role
        )
        
    except Exception as e:
        logger.error(f"Unexpected authentication error: {e}")
        return {"success": False, "message": "Authentication service error"}

def create_user(username: str, password: str, user_role: str = UserRole.USER.value) -> Dict[str, str]:
    try:
        logger.info(f"Attempting to create user: {username} with role: {user_role}")
        
        # Validate input parameters
        if not username or not username.strip():
            return {"success": False, "message": "Username cannot be empty"}
        
        if not password or len(password) < 6:
            return{"success": False, "message":"Password must be at least 6 characters long"}
        
        validated_role = _validate_user_role(user_role)
        
        # Load existing users
        users = load_data(path_loc=USER_DATABASE_PATH)
        
        # Check if user already exists
        if username in users:
            logger.warning(f"User creation failed - user already exists: {username}")
            return {"success": False, "message": "User already exists"}
        
        # Hash password and create user record
        hashed_password = hash_password(password)
        
        users[username] = {
            "password": hashed_password,
            "role": validated_role.value,
            "created_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Save updated user database
        save_data(save_data=users, path_loc=USER_DATABASE_PATH)
        
        logger.info(f"User created successfully: {username} with role: {validated_role.value}")
        
        return {
            "message": f"User '{username}' registered successfully with role '{validated_role.value}'",
            "username": username,
            "role": validated_role.value
        }
        
    except Exception as e:
        logger.error(f"Error creating user: {e}")
        return {"success": False, "message": "Error creating user"}

def delete_user(username: str) -> Dict[str, str]:
    try:
        logger.info(f"Attempting to delete user: {username}")
        
        users = load_data(path_loc=USER_DATABASE_PATH)
        
        if username not in users:
            logger.warning(f"User deletion failed - user not found: {username}")
            return {"success": False, "message": "User not found"}
        
        del users[username]
        save_data(save_data=users, path_loc=USER_DATABASE_PATH)
        
        logger.info(f"User deleted successfully: {username}")
        
        return {"success": True, "message": f"User '{username}' deleted successfully"}

        
    except Exception as e:
        logger.error(f"Error deleting user: {e}")
        return {"success": False, "message": "Error deleting user"}
    
def get_all_users() -> Dict[str, Any]:
    try:
        logger.info("Retrieving all users")
        
        users = load_data(path_loc=USER_DATABASE_PATH)
        
        # Remove password information for security
        safe_users = {
            username: {
                "role": user_data.get("role", UserRole.USER.value),
                "created_at": user_data.get("created_at", "Unknown")
            }
            for username, user_data in users.items()
        }
        
        logger.info(f"Retrieved {len(safe_users)} users")
        
        return {
            "users": safe_users,
            "total_count": len(safe_users)
        }
        
    except Exception as e:
        logger.error(f"Error retrieving users: {e}")
        return {"success": False, "message": "Error retrieving user list"}