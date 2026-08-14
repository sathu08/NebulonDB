import json
import string
import secrets

from pathlib import Path
from typing import Any

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
    created_at: str | None = None
    last_login: str | None = None

class AuthenticationResult(BaseModel):
    username: str
    role: UserRole | None = None
    is_authenticated: bool = True
    message: str | None = None

class StandardErrorResponse(BaseModel):
    success: bool
    message: str
    role: str | None = None

class UserRegistrationRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)
    user_role: str

class ChangePasswordRequest(BaseModel):
    current_password: str | None = Field(default=None, max_length=256)
    new_password: str = Field(..., min_length=8)

class DeleteUserRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)

class CorpusQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)
    ndb_type: str = NDBMeta.Type.COSMOS

class SegmentQueryRequest(BaseModel):
    corpus_name: str = Field(..., min_length=1)
    segment_name: str = Field(..., min_length=1)
    ndb_type: str = NDBMeta.Type.ORBIT
    limit: int | None = None
    segment_dataset: dict[str, list[Any]] | list[dict[str, Any]] | None = None
    set_columns: str | list[str] | None = None
    search_item: str | None = None
    query_vector: list[float] | None = None
    doc_type: str | None = None
    lang_type: str | None = None

    # Nova and Mesh
    top_matches: int | None = None
    min_score: float | None = None
    is_precomputed: bool | None = False
    rank: bool | None = False
    mode: str | None = None
    graph_start_node: int | None = None
    expand_depth: int | None = None
    graph_boost: float | None = None
    relations: list[tuple[int, int, str]] | None = None
    source_column: str | None = None
    target_column: str | None = None
    relation_column: str | None = None
    record_id: int | None = None
    node_id: int | None = None
    direction: str | None = "both"
    start_node: int | None = None
    max_depth: int | None = 3
    source: int | None = None
    target: int | None = None
    relation: str | None = None
    metadata: dict[str, Any] | None = None

    # Bulk graph load (Option A)
    nodes: list[dict[str, Any]] | None = None
    edges: list[dict[str, Any]] | None = None

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
    user: dict[str, Any]

class StandardResponse(BaseModel):
    success: bool
    message: str
    exists: bool = False
    data: dict[str, Any] | list[Any] | Any | None = None
    corpus_name: str | None = None
    segment_name: str | None = None
    errors: list[str] | None = None


# ==========================================================
#        Helper Functions
# ==========================================================

def load_data(path_loc: Path, default:dict = None, is_bytes_input: bool = False) -> dict[str, dict[str, Any]]:
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

def save_data(
    data: dict[str, Any],
    path_loc: Path | str | None = None,
    return_bytes: bool = False,
) -> dict[str, Any] | bytes:
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
        logger.info("Data successfully saved")

        return {"success": True, "message": "Data saved"}

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
