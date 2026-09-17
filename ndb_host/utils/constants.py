"""
NDB Constants and Configurations
==========================================================

This module handles constants and configuration for the NDB API.

"""

from enum import Enum
from pydantic import BaseModel

from typing import Any, Literal

from dataclasses import dataclass


def normalize_doc_type(doc_type: object) -> str:
    """Normalize a free-form doc_type to a safe non-empty string.

    The value is passed through as-is (lowercased/stripped) so whatever
    the caller sends — ``chat_history``, ``chat_mind``, ``pdf``, ... —
    is exactly what lands in record metadata. Empty/missing values fall
    back to ``"other"``.
    """
    text = str(doc_type or "other").strip().lower()
    return text or "other"


def normalize_lang(lang: object) -> str:
    """Normalize a language tag; empty/missing values fall back to ``"en"``."""
    text = str(lang or "en").strip().lower()
    return text or "en"


class NDBMeta:
    APP_NAME = "NebulonDB"

    class User(str, Enum):
        NEBULONDB_USER = "nebulon-supernova"

    class Corpus:
        DEFAULT_CORPUS_NAME = "nebulon_origin"
        DEFAULT_SEGMENT_NAME = "nebulon_userinfo"
        METADATA_SEGMENT_NAME = "nebulon_metadata"

    class Paths:
        STORAGE_DIR = "Storage"
        SECRETS_DIR = "Secrets"
        LOG_DIR = "logs"
        PID_FILE = "nebulon.pid"
        WEB_DIR = "ndb_host/web_dir"

    class Logging:
        LOG_FILE = "nebulondb_%Y-%m-%d.log"
        DEFAULT_RETENTION_DAYS = 7
        DEFAULT_AUTO_DELETE = True

    class Type(str, Enum):
        COSMOS = "cosmos"
        ORBIT = "orbit"

class AuthenticationConfig:
    PASSWORD_HASH_SCHEMES = ["bcrypt"]
    PASSWORD_HASH_DEPRECATED = "auto"
    ENCODING = "utf-8"
    JSON_INDENT = 4

class UserRole(str, Enum):
    SYSTEM = "system"
    SUPER_USER = "super_user"
    ADMIN_USER = "admin_user"
    USER = "user"

class ColumnPick:
    FIRST_COLUMN = "First Column"
    ALL = "All"

class ModelType:
    EMBEDDING: str = "embedding"
    CROSS_ENCODER: str = "cross_encoder"

@dataclass
class BatchConfig:
    batch_size: int
    device: Literal["cpu", "cuda"]
    dtype: Literal["fp32", "fp16", "bf16"]
    use_fp16: bool

class ConfigUpdate(BaseModel):
    config: dict[str, dict[str, Any]]
