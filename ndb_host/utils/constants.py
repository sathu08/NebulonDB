"""
NDB Constants and Configurations
==========================================================

This module handles constants and configuration for the NDB API.

"""

from enum import Enum
from pydantic import BaseModel
from datetime import timedelta

from typing import Dict, Any, Literal

from dataclasses import dataclass

from .time_utils import utc_now

class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    MARKDOWN = "markdown"
    CHAT = "chat"
    CHAT_SUMMARY = "chat_summary"
    IMPORTANT_CHAT = "important_chat"
    WEB_CACHE = "web_cache"
    SESSION = "session"
    OTHER = "other"

class RetentionPolicy(str, Enum):
    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    SESSION = "session"
    CUSTOM = "custom"

class MetadataRetention:
    """Handles automatic expiration of records."""

    DEFAULT_POLICY = {
        DocumentType.PDF: RetentionPolicy.PERMANENT,
        DocumentType.DOCX: RetentionPolicy.PERMANENT,
        DocumentType.TXT: RetentionPolicy.PERMANENT,
        DocumentType.MARKDOWN: RetentionPolicy.PERMANENT,

        DocumentType.CHAT: RetentionPolicy.TEMPORARY,
        DocumentType.CHAT_SUMMARY: RetentionPolicy.TEMPORARY,

        DocumentType.IMPORTANT_CHAT: RetentionPolicy.PERMANENT,

        DocumentType.WEB_CACHE: RetentionPolicy.TEMPORARY,
        DocumentType.SESSION: RetentionPolicy.SESSION,

        DocumentType.OTHER: RetentionPolicy.PERMANENT,
    }

    TTL = {
        RetentionPolicy.TEMPORARY: timedelta(days=10),
        RetentionPolicy.SESSION: timedelta(hours=1),
    }

    @staticmethod
    def _utc_now():
        return utc_now()

    @classmethod
    def apply(cls, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Automatically applies retention policy.

        Rules:
        - PDF -> Never expires
        - Chat -> 10 days
        - Session -> 1 hour
        - Custom -> Uses supplied expires_at
        """

        metadata = metadata.copy()

        doc_type = DocumentType(
            metadata.get("type", DocumentType.OTHER)
        )

        policy = RetentionPolicy(
            metadata.get(
                "retention",
                cls.DEFAULT_POLICY[doc_type]
            )
        )

        metadata["retention"] = policy.value

        # User already supplied expires_at
        if "expires_at" in metadata:
            return metadata

        if policy == RetentionPolicy.PERMANENT:
            metadata["expires_at"] = None

        elif policy == RetentionPolicy.TEMPORARY:
            metadata["expires_at"] = (
                cls._utc_now() +
                cls.TTL[RetentionPolicy.TEMPORARY]
            ).timestamp()

        elif policy == RetentionPolicy.SESSION:
            metadata["expires_at"] = (
                cls._utc_now() +
                cls.TTL[RetentionPolicy.SESSION]
            ).timestamp()

        elif policy == RetentionPolicy.CUSTOM:
            raise ValueError(
                "expires_at must be supplied when using retention='custom'"
            )

        return metadata

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
    config: Dict[str, Dict[str, Any]]