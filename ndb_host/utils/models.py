from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any
from pathlib import Path
import json

from utils.logger import logger

# === Constants and Configuration ===
class UserRole(str, Enum):
    SUPER_USER = "super_user"
    ADMIN_USER = "admin_user"
    USER = "user"

class AuthenticationConfig:
    PASSWORD_HASH_SCHEMES = ["bcrypt"]
    PASSWORD_HASH_DEPRECATED = "auto"
    ENCODING = "utf-8"
    JSON_INDENT = 4

# === Pydantic Models ===
class UserProfile(BaseModel):
    username: str
    role: UserRole

class UserRecord(BaseModel):
    password: str
    role: UserRole
    created_at: Optional[str] = None
    last_login: Optional[str] = None

class AuthenticationResult(BaseModel):
    username: str
    role: UserRole
    is_authenticated: bool = True

class UserRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    user_role: str

class CorpusQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)

class SegmentQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)
    segment_name: str = Field(..., min_length=1)
    
    @field_validator("segment_name")
    def segment_name_must_be_lowercase(cls, v: str) -> str:
        if not v.islower():
            raise ValueError("segment_name must be lowercase")
        return v

    category: Optional[str] = None

class UserAuthenticationResponse(BaseModel):
    message: str
    user: Dict[str, Any]

class CorpusExistenceResponse(BaseModel):
    exists: bool
    corpus_name: str
    message: str

class SegmentExistenceResponse(BaseModel):
    exists: bool
    corpus_name: str
    segment_name: str
    message: str

class SegmentExistenceResponse(BaseModel):
    exists: bool
    corpus_name: str
    segment_name: str
    message: str

class StandardResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    corpus_name: Optional[str] = None
    segment_name: Optional[str] = None

# === Helper Functions ===
def load_data(path_loc: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load JSON data from file, returning an empty dict if empty or invalid.
    """
    try:
        content = path_loc.read_text(encoding=AuthenticationConfig.ENCODING)

        if not content.strip():
            logger.warning("File is empty. Returning empty dictionary.")
            return {}

        return json.loads(content)

    except json.JSONDecodeError as e:
        logger.error(f"Malformed JSON in file {path_loc}: {e}")
        return {}
    except OSError as e:
        logger.error(f"Error reading file {path_loc}: {e}")
        return {}

def save_data(save_data: Dict[str, Dict[str, Any]], path_loc: Path) -> Dict[str, Any]:
    """
    Save JSON data to a file with safe logging.
    """
    try:
        json_content = json.dumps(
            save_data,
            indent=AuthenticationConfig.JSON_INDENT,
            ensure_ascii=False
        )
        path_loc.write_text(json_content, encoding=AuthenticationConfig.ENCODING)
        logger.info(f"Data successfully saved to {path_loc}")
        return {"success": True, "message": f"Data saved to {path_loc}"}

    except (OSError, PermissionError) as e:
        logger.error(f"Failed to write to {path_loc}: {e}")
        return {"success": False, "message": f"Failed to save data to {path_loc}", "error": str(e)}
