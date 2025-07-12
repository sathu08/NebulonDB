from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from pathlib import Path
import json

from fastapi import HTTPException, status

from utils.logger import logger

# === Constants and Configuration ===
class UserRole(str, Enum):
    """Enumeration of valid user roles in the system."""
    SUPER_USER = "super_user"
    ADMIN_USER = "admin_user"
    USER = "user"

class AuthenticationConfig:
    """Configuration constants for authentication module."""
    PASSWORD_HASH_SCHEMES = ["bcrypt"]
    PASSWORD_HASH_DEPRECATED = "auto"
    ENCODING = "utf-8"
    JSON_INDENT = 4

# === Pydantic Models ===
class UserProfile(BaseModel):
    """User profile model for internal use."""
    username: str
    role: UserRole
    
class UserRecord(BaseModel):
    """User record model for database storage."""
    password: str
    role: UserRole
    created_at: Optional[str] = None
    last_login: Optional[str] = None

class AuthenticationResult(BaseModel):
    """Authentication result model."""
    username: str
    role: UserRole
    is_authenticated: bool = True

class UserRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50, description="Username for the new user")
    password: str = Field(..., min_length=6, description="Password for the new user")
    user_role: str = Field(..., description="Role assigned to the user")

class CorpusQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1, description="Name of the corpus to query")

class UserAuthenticationResponse(BaseModel):
    message: str
    user: Dict[str, Any]

class CorpusExistenceResponse(BaseModel):
    exists: bool
    corpus_name: str
    message: str

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any] = None

# === Helper Functions ===
def load_data(path_loc: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load user database from JSON file.
    
    Returns:
        Dict[str, Dict[str, Any]]: Dictionary containing user data
        
    Raises:
        OSError: If file cannot be read
        json.JSONDecodeError: If JSON is malformed (handled gracefully)
    """
    try:        
        content = path_loc.read_text(
            encoding=AuthenticationConfig.ENCODING
        )
        
        if not content.strip():
            logger.warning("User database file is empty, returning empty dictionary")
            return {}
            
        load_data = json.loads(content)
        return load_data
        
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in user database: {e}")
        logger.warning("Returning empty user database due to corruption")
        return {}
    except OSError as e:
        logger.error(f"Failed to read user database: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to access user database"
        )

def save_data(save_data: Dict[str, Dict[str, Any]], path_loc:Path) -> None:
    """
    Save data to JSON file.
    
    Args:
        save_data: Dictionary containing data to save
        
    Raises:
        OSError: If file cannot be written
        PermissionError: If insufficient permissions
    """
    try:        
        json_content = json.dumps(
            save_data,
            indent=AuthenticationConfig.JSON_INDENT,
            ensure_ascii=False
        )
        
        path_loc.write_text(
            json_content,
            encoding=AuthenticationConfig.ENCODING
        )
                
    except (OSError, PermissionError) as e:
        logger.error(f"Failed to save data : {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unable to save data"
        )
