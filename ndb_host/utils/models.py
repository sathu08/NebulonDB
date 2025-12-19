from enum import Enum
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, Any, Union, List
from pathlib import Path
import json
import base64
import threading

from utils.logger import logger

class SemanticEmbeddingModel:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2"):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_model(model_name)
            return cls._instance

    def _init_model(self, model_name: str):
        logger.info(f"Loading embedding model: {model_name} (this happens only once)")
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts, **kwargs):
        return self.model.encode(texts, **kwargs)
        
# === Constants and Configuration ===
class UserRole(str, Enum):
    SYSTEM = "system"
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
    segment_dataset: Optional[Union[Dict[str, List[Any]], List[Dict[str, Any]]]] = None
    set_columns: Optional[Union[str, List[str]]] = None
    search_item: Optional[str] = None
    top_matches: Optional[str] = None
    is_precomputed: Optional[bool] = False

    @field_validator("segment_dataset", mode="before")
    def ensure_dict_or_list(cls, v):
        # Case 1: None → keep None
        if v is None:
            return None

        # Case 2: Already a dict → keep as-is
        if isinstance(v, dict):
            return v

        # Case 3: Already a list of dicts → keep as-is
        if isinstance(v, list) and all(isinstance(i, dict) for i in v):
            return v

        # Case 4: Anything else → reject (return None, let route handle)
        return None

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
def load_data(path_loc: Path, default:Dict = None, is_bytes_input: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load JSON data from file, returning an empty dict if empty or invalid.
    """
    if default is None:
        default = {}
    try:
        if is_bytes_input and isinstance(path_loc, (bytes, bytearray)):
            if not path_loc:  # Handle empty bytes
                logger.warning("Empty bytes received, returning default")
                return default
            
            content = path_loc.decode(AuthenticationConfig.ENCODING, errors="replace")
            content = content.strip()
            
            if not content:  # Handle whitespace-only content
                logger.warning("Whitespace-only content, returning default")
                return default
            
            return json.loads(content)
        
        path_obj = Path(path_loc)
        if not path_obj.exists():
            return default
        
        if path_obj.stat().st_size == 0:
            return default
        
        content = path_obj.read_text(encoding=AuthenticationConfig.ENCODING)
        return json.loads(content)
    
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")

    except (UnicodeDecodeError, OSError, PermissionError, Exception) as e:
        logger.error(f"Error loading data: {e}")

def save_data(data: Dict[str, Any], path_loc: Union[Path, str, None] = None, return_bytes: bool = False) -> Union[Dict[str, Any], bytes]:
    """Save JSON data to file OR return as bytes for NDB."""
    try:
        json_content = json.dumps(
            data,
            indent=AuthenticationConfig.JSON_INDENT,
            ensure_ascii=False
        )

        if return_bytes:
            return json_content.encode(encoding=AuthenticationConfig.ENCODING)
        
        if path_loc is None:
            raise ValueError("path_loc required when return_bytes=False")
        
        path_obj = Path(path_loc)
        path_obj.write_text(json_content, encoding=AuthenticationConfig.ENCODING)
        logger.info(f"Data successfully saved")

        return {"success": True, "message": f"Data saved"}
    
    except (OSError, PermissionError, TypeError) as e:
        logger.error(f"Failed to save data: {e}")
        if return_bytes:
            raise
        return {"success": False, "message": "Failed to save data", "error": str(e)}
