from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import json

from utils.logger import logger

class SemanticEmbeddingModel:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            raise RuntimeError(f"Model loading failed: {str(e)}")

    def encode(self, texts, **kwargs):
        """Encode text(s) into embeddings."""
        return self.model.encode(texts, **kwargs)
        
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

class ColumnPick:
    FIRST_COLUMN = "First Column"
    ALL = "All"
    
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
    role: Optional[UserRole] = None
    is_authenticated: bool = True
    message: Optional[str] = None

class StandardErrorResponse(BaseModel):
    success: bool
    message: str
    role: Optional[str] = None
    
class UserRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)
    user_role: str

class CorpusQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)

class SegmentQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)
    segment_name: str = Field(..., min_length=1)
    segment_dataset: Optional[Dict[str, List[Any]]] = None
    set_columns: Optional[Union[str, List[str]]] = None
    search_item: Optional[str] = None

    @field_validator("segment_dataset", mode="before")
    def ensure_dict(cls, v):
        # If None or already dict, keep as-is
        if v is None or isinstance(v, dict):
            return v
        # If it's a DataFrame or list, try converting to dict
        try:
            return dict(v)
        except Exception:
            return None   # fallback, let your route handle it

    @field_validator("segment_name", mode="before")
    def ensure_lowercase(cls, v: str) -> str:
        if isinstance(v, str):
            return v.lower()
        return v

class UserAuthenticationResponse(BaseModel):
    message: str
    user: Dict[str, Any]

class StandardResponse(BaseModel):
    success: bool
    message: str
    exists: bool = False
    data: Optional[Dict[str, Any]] = None
    corpus_name: Optional[str] = None
    segment_name: Optional[str] = None
    errors: Optional[List[str]] = None

# === Helper Functions ===
def load_data(path_loc: Path, default:Dict = {}) -> Dict[str, Dict[str, Any]]:
    """
    Load JSON data from file, returning an empty dict if empty or invalid.
    """
    try:
        if path_loc.exists():
            content = path_loc.read_text(encoding=AuthenticationConfig.ENCODING)
            return json.loads(content)
        return default
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
