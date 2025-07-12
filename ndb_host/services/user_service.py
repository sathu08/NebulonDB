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
    """
    Validate and convert user role string to UserRole enum.
    
    Args:
        user_role: Role string to validate
        
    Returns:
        UserRole: Validated user role enum
        
    Raises:
        ValueError: If role is invalid
    """
    try:
        return UserRole(user_role)
    except ValueError:
        valid_roles = [role.value for role in UserRole]
        raise ValueError(f"Invalid role: {user_role}. Must be one of {valid_roles}")

# === Authentication Functions ===
def get_current_user(credentials: HTTPBasicCredentials = Depends(http_basic_security)) -> AuthenticationResult:
    """
    Authenticate user using HTTP Basic Authentication.
    
    Args:
        credentials: HTTP Basic Authentication credentials
        
    Returns:
        AuthenticationResult: Authenticated user information
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        logger.debug(f"Attempting authentication for user: {credentials.username}")
        
        users = load_data(path_loc=USER_DATABASE_PATH)
        user_record = users.get(credentials.username)
        
        if not user_record:
            logger.warning(f"Authentication failed - user not found: {credentials.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        hashed_password = user_record.get("password")
        if not hashed_password:
            logger.error(f"User record missing password hash: {credentials.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        if not verify_password(credentials.password, hashed_password):
            logger.warning(f"Authentication failed - invalid password: {credentials.username}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Basic"},
            )
        
        user_role = UserRole(user_record.get("role", UserRole.USER.value))
        
        logger.info(f"Authentication successful for user: {credentials.username}")
        
        return AuthenticationResult(
            username=credentials.username,
            role=user_role
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service unavailable"
        )

def create_user(username: str, password: str, user_role: str = UserRole.USER.value) -> Dict[str, str]:
    """
    Create a new user in the system.
    
    Args:
        username: Username for the new user
        password: Plain text password for the new user
        user_role: Role to assign to the new user
        
    Returns:
        Dict[str, str]: Success message with user details
        
    Raises:
        ValueError: If input validation fails
        HTTPException: If user creation fails
    """
    try:
        logger.info(f"Attempting to create user: {username} with role: {user_role}")
        
        # Validate input parameters
        if not username or not username.strip():
            raise ValueError("Username cannot be empty")
        
        if not password or len(password) < 6:
            raise ValueError("Password must be at least 6 characters long")
        
        validated_role = _validate_user_role(user_role)
        
        # Load existing users
        users = load_data(path_loc=USER_DATABASE_PATH)
        
        # Check if user already exists
        if username in users:
            logger.warning(f"User creation failed - user already exists: {username}")
            raise ValueError("User already exists")
        
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
        
    except ValueError as e:
        logger.error(f"User creation validation error: {e}")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User already exists"
        )
    except Exception as e:
        logger.error(f"Unexpected error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation service unavailable"
        )

def delete_user(username: str) -> Dict[str, str]:
    """
    Delete a user from the system.
    
    Args:
        username: Username of the user to delete
        
    Returns:
        Dict[str, str]: Success message
        
    Raises:
        ValueError: If user doesn't exist
        HTTPException: If deletion fails
    """
    try:
        logger.info(f"Attempting to delete user: {username}")
        
        users = load_data(path_loc=USER_DATABASE_PATH)
        
        if username not in users:
            logger.warning(f"User deletion failed - user not found: {username}")
            raise ValueError("User not found")
        
        del users[username]
        save_data(save_data=users, path_loc=USER_DATABASE_PATH)
        
        logger.info(f"User deleted successfully: {username}")
        
        return {
            "message": f"User '{username}' deleted successfully"
        }
        
    except ValueError as e:
        logger.error(f"User deletion error: {e}")
        raise ValueError(str(e))
    except Exception as e:
        logger.error(f"Unexpected error deleting user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User deletion service unavailable"
        )

def get_all_users() -> Dict[str, Any]:
    """
    Retrieve all users from the system (without passwords).
    
    Returns:
        Dict[str, Any]: Dictionary containing all users and their roles
        
    Raises:
        HTTPException: If retrieval fails
    """
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User retrieval service unavailable"
        )
