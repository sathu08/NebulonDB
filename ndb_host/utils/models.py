import json
import string
import secrets

from pathlib import Path
from typing import Optional, Dict, Any, Union, List, Tuple

from pydantic import BaseModel, Field, field_validator

from db.ndb_settings import NDBConfig
from utils.logger import NebulonDBLogger
from utils.constants import AuthenticationConfig, NDBMeta, UserRole


# ==========================================================
#        Load Configuration
# ==========================================================

cfg = NDBConfig()

# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Pydantic Models
# ==========================================================

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

class ChangePasswordRequest(BaseModel):
    current_password: str = Field(..., min_length=6)
    new_password: str = Field(..., min_length=6)

class DeleteUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)

class CorpusQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)
    ndb_type: str = NDBMeta.Type.COSMOS

class SegmentQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)
    segment_name: str = Field(..., min_length=1)
    ndb_type: str = NDBMeta.Type.ORBIT
    limit: Optional[int] = None
    segment_dataset: Optional[Union[Dict[str, List[Any]], List[Dict[str, Any]]]] = None
    set_columns: Optional[Union[str, List[str]]] = None
    search_item: Optional[str] = None
    doc_type: Optional[str] = None
    lang_type: Optional[str] = None

    # Nova and Mesh 
    top_matches: Optional[int] = None
    min_score: Optional[float] = None
    is_precomputed: Optional[bool] = False
    rank: Optional[bool] = False
    mode: Optional[str] = None
    graph_start_node: Optional[int] = None
    expand_depth: Optional[int] = None
    graph_boost: Optional[float] = None
    relations: Optional[List[Tuple[int, int, str]]] = None
    source_column: Optional[str] = None
    target_column: Optional[str] = None
    relation_column: Optional[str] = None
    record_id: Optional[int] = None
    node_id: Optional[int] = None
    direction: Optional[str] = "both"
    start_node: Optional[int] = None
    max_depth: Optional[int] = 3
    source: Optional[int] = None
    target: Optional[int] = None
    relation: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    # Bulk graph load (Option A)
    nodes: Optional[List[Dict[str, Any]]] = None
    edges: Optional[List[Dict[str, Any]]] = None

    @field_validator("segment_dataset", mode="before")
    def ensure_dict_or_list(cls, v):
        # === Case 1: None → keep None ===
        if v is None:
            return None

        # === Case 2: Already a dict → keep as-is ===
        if isinstance(v, dict):
            return v

        # === Case 3: Already a list of dicts → keep as-is ===
        if isinstance(v, list) and all(isinstance(i, dict) for i in v):
            return v

        # === Case 4: Anything else → reject (return None, let route handle) ===
        return None


class UserAuthenticationResponse(BaseModel):
    message: str
    user: Dict[str, Any]

class StandardResponse(BaseModel):
    success: bool
    message: str
    exists: bool = False
    data: Optional[Union[Dict[str, Any], List[Any], Any]] = None
    corpus_name: Optional[str] = None
    segment_name: Optional[str] = None
    errors: Optional[List[str]] = None


# ==========================================================
#        Helper Functions
# ==========================================================

def load_data(path_loc: Path, default:Dict = None, is_bytes_input: bool = False) -> Dict[str, Dict[str, Any]]:
    """
    Load JSON data from file, returning an empty dict if empty or invalid.
    Args:
        path_loc (Path): Path to the JSON file.
        default (Dict, optional): Default value to return if file is empty or invalid.
        is_bytes_input (bool, optional): Whether the input is bytes instead of a file path.
    Returns:
        Dict[str, Dict[str, Any]]: Loaded JSON data or default.
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
    """
    Save JSON data to file OR return as bytes for NDB.
    Args:
        data (Dict[str, Any]): Data to save.
        path_loc (Union[Path, str, None], optional): Path to save the JSON file. Required if return_bytes is False.
        return_bytes (bool, optional): Whether to return the JSON data as bytes instead of saving to file.
    Returns:
        Union[Dict[str, Any], bytes]: Result dict if saved to file, or bytes if return_bytes is True.
    """

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


def generate_password(length: int = 16) -> str:
    """Generate a cryptographically secure random password."""
    alphabet = string.ascii_letters + string.digits + string.punctuation
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        # Ensure at least one of each required character class
        if (any(c.islower() for c in password)
                and any(c.isupper() for c in password)
                and any(c.isdigit() for c in password)
                and any(c in string.punctuation for c in password)):
            return password