import json
import threading

from pathlib import Path
from functools import lru_cache

from typing import Optional, Dict, Any, Union, List, Tuple
from pydantic import BaseModel, Field, field_validator

from db.ndb_settings import NDBConfig
from utils.logger import NebulonDBLogger
from utils.constants import AuthenticationConfig, UserRole, ColumnPick, NDBCorpusMeta


# ==========================================================
#        Load Configuration
# ==========================================================

cfg = NDBConfig()

# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Thread Lock (per worker)
# ==========================================================

_model_lock = threading.Lock()

# ==========================================================
#        Embedding Model Loader with Caching
# ==========================================================

def get_auto_batch_size() -> Tuple[int, str]:
    """Decide batch size automatically based on system/device."""

    import torch
    import psutil
    
    device = "cuda" if torch.cuda.is_available() and not cfg.NEBULONDB_DEFAULT_MODE else "cpu"
    if device == "cuda":
        # GPU memory based logic
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9  # in GB
        if total_mem > 16:
            return 128, device
        elif total_mem > 8:
            return 64, device
        else:
            return 32, device
    else:
        # CPU memory based logic
        ram_gb = psutil.virtual_memory().total / 1e9
        if ram_gb > 16:
            return 32, device
        elif ram_gb > 8:
            return 16, device
        else:
            return 8, device

@lru_cache(maxsize=1)
def get_embedding_model(model_repo_id: str):
    """ Load SentenceTransformer model (only once per worker).Thread-safe + disk-cached."""

    from sentence_transformers import SentenceTransformer

    cache_folder = Path(cfg.NEBULONDB_MODEL_CACHE_DIR)

    # === Update LLM config ===
    
    with _model_lock:
        logger.info(
            f"Loading embedding model: {model_repo_id} "
            f"(cache_dir={cache_folder}, once per worker)"
        )

        return SentenceTransformer(
            model_repo_id,
            cache_folder=str(cache_folder),
            device=cfg.NEBULONDB_MODEL_DEVICE            
        )

# ==========================================================
#        Ensure Model Exists + Load
# ==========================================================

def ensure_embedding_model(model_name: str, prefix: str = "sentence-transformers"):
    """Ensure embedding model exists and load it."""

    repo_id = f"{prefix}/{model_name}"
    return get_embedding_model(repo_id)
    

# ==========================================================
#        Semantic Embedding Model Wrapper
# ==========================================================

class SemanticEmbeddingModel:
    """Wrapper for the embedding model."""

    def __init__(self):
        self.model_name = cfg.NEBULONDB_EMBEDDING_MODEL

    def encode(self, texts, **kwargs):
        """
        Encode texts using the embedding model.
        Args:
            texts (List[str]): List of texts to encode.
        Returns:
            List[List[float]]: List of embeddings.
        """
        
        model = get_embedding_model(self.model_name)
        return model.encode(texts, **kwargs,batch_size=16)

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
