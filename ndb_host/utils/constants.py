from enum import Enum


#=========================================================
#        Constants and Configuration
# ==========================================================

class NDBCorpusMeta:
    APP_NAME = "NebulonDB"
    SEGMENTS_NAME = "segments"
    DEFAULT_CORPUS_NAME = "nebulon_origin"
    LOG_STRUCTURE = ["app", "error", "access", "audit"]

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
