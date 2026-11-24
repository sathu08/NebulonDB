"""
NDB API System Management
==========================================================

This module handles system-level operations for the NDB API.
It provides endpoints for viewing and updating NebulonDB
configuration.
"""

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from core.permissions import check_user_permission
from services.user_service import get_current_user
from utils.logger import NebulonDBLogger
from utils.models import AuthenticationResult, UserRole
from utils.constants import ConfigUpdate
from ndb_host.db.ndb_settings import NDBConfig

# F-01: only these keys may be changed at runtime. Everything else
# ([environment] master key, [paths] home, [powerloss], [server] host/port,
# [llm] cache dir) requires an operator editing the file + restart.
ALLOWED_CONFIG_WRITES: dict[str, set[str]] = {
    "segments": {
        "FLUSH_RECORD_THRESHOLD",
        "FLUSH_INTERVAL",
        "COMPACTION_INTERVAL",
        "MAX_SEGMENTS_BEFORE_COMPACT",
    },
    "vector": {"TOP_MATCHES", "MIN_SCORE"},
    "rank": {"RANK_TOPK"},
}

# ==========================================================
#        Initialize Logger And Config
# ==========================================================

logger = NebulonDBLogger().get_logger("access")

# ==========================================================
#        API Router
# ==========================================================

router = APIRouter()

# ==========================================================
#        Load Configuration
# ==========================================================

config_settings = NDBConfig()
WEB_DIR = config_settings.NEBULONDB_WEB_DIR

# ==========================================================
#        Serve Web Console (index.html)
# ==========================================================

@router.get("/", response_class=HTMLResponse, include_in_schema=False)
@router.get("/index", response_class=HTMLResponse, include_in_schema=False)

async def index_page():
    """Serve the NebulonDB console login page."""

    INDEX_HTML = WEB_DIR / "index.html"

    if not INDEX_HTML.exists():
        raise HTTPException(status_code=404, detail="index.html not found")

    return FileResponse(INDEX_HTML)


# ==========================================================
#        Serve Web Console (dashboard.html)
# ==========================================================

@router.get("/dashboard", response_class=HTMLResponse, include_in_schema=False)
@router.get("/dashboard.html", response_class=HTMLResponse, include_in_schema=False)

async def dashboard_page():
    """Serve the NebulonDB console dashboard page."""

    DASH_HTML = WEB_DIR / "dashboard.html"

    if not DASH_HTML.exists():
        raise HTTPException(status_code=404, detail="dashboard.html not found")

    return FileResponse(DASH_HTML)


# ==========================================================
#        Get Configuration
# ==========================================================

@router.get("/config")
async def get_config(current_user: AuthenticationResult = Depends(get_current_user)):
    if not current_user.is_authenticated:
        raise HTTPException(status_code=401, detail=current_user.message or "Not authenticated")
    return {
        "segments": {
            "wal_auto_flush": config_settings.WAL_AUTO_FLUSH,
            "wal_fsync_interval": config_settings.WAL_FSYNC_INTERVAL,
            "durability_mode": config_settings.DURABILITY_MODE,
            "compress_segments": config_settings.COMPRESS_SEGMENTS,
            "bloom_filter_enabled": config_settings.BLOOM_FILTER_ENABLED,
            "max_open_segments": config_settings.MAX_OPEN_SEGMENTS,
            "compaction_interval": config_settings.COMPACTION_INTERVAL,
            "max_segments_before_compact": config_settings.MAX_SEGMENTS_BEFORE_COMPACT,
            "flush_interval": config_settings.FLUSH_INTERVAL,
            "flush_record_threshold": config_settings.FLUSH_RECORD_THRESHOLD,
        },
        "bloom": {
            "enabled": config_settings.DEFAULT_CORPUS_CONFIG_DATA["bloom_enabled"],
            "bits_per_key": config_settings.DEFAULT_CORPUS_CONFIG_DATA["bloom_bits_per_key"],
            "hash_count": config_settings.DEFAULT_CORPUS_CONFIG_DATA["bloom_hash_count"],
        },
        "vector": {
            "dimension": config_settings.DEFAULT_CORPUS_CONFIG_DATA["dimension"],
            "space": config_settings.DEFAULT_CORPUS_CONFIG_DATA["space"],
            "top_matches": config_settings.DEFAULT_CORPUS_CONFIG_DATA["top_matches"],
            "min_score": config_settings.DEFAULT_CORPUS_CONFIG_DATA["min_score"],
            "save_every_n": config_settings.VECTOR_SAVE_EVERY_N,
            "compaction_threshold": config_settings.VECTOR_COMPACTION_THRESHOLD,
        },
        "hnsw": {
            "m": config_settings.DEFAULT_CORPUS_CONFIG_DATA["m"],
            "ef_construction": config_settings.DEFAULT_CORPUS_CONFIG_DATA["ef_construction"],
            "ef_search": config_settings.DEFAULT_CORPUS_CONFIG_DATA["ef_search"],
        },
        "rank": {
            "rank_topk": config_settings.RANK_TOPK,
            "weight_vector": config_settings.RANK_WEIGHTS["vector"],
            "weight_bm25": config_settings.RANK_WEIGHTS["bm25"],
            "weight_metadata": config_settings.RANK_WEIGHTS["metadata"],
            "weight_importance": config_settings.RANK_WEIGHTS["importance"],
            "weight_freshness": config_settings.RANK_WEIGHTS["freshness"],
        },
        "server": {
            "app_name": config_settings.APP_NAME,
            "host": config_settings.HOST,
            "port": config_settings.PORT,
            "workers": config_settings.WORKERS,
            "single_worker_only": True,
            "engine_state": "in-memory (not shared across processes)",
            "timeout": config_settings.TIMEOUT,
            "keep_alive": config_settings.KEEP_ALIVE,
            "graceful_timeout": config_settings.GRACEFUL_TIMEOUT,
            "access_logfile": config_settings.ACCESS_LOGFILE,
            "error_logfile": config_settings.ERROR_LOGFILE,
            "log_level": config_settings.LOG_LEVEL,
            "url": f"http://{config_settings.HOST}:{config_settings.PORT}",
        },
    }


# ==========================================================
#        Update Configuration
# ==========================================================

@router.put("/config")
async def update_config(
    payload: ConfigUpdate,
    current_user: AuthenticationResult = Depends(get_current_user),
):
    global config_settings
    if not current_user.is_authenticated:
        raise HTTPException(status_code=401, detail=current_user.message or "Not authenticated")
    if not check_user_permission(current_user, UserRole.SYSTEM):
        raise HTTPException(status_code=403, detail="SYSTEM role required to update configuration")
    try:
        updated = []

        for section, values in payload.config.items():
            if section not in ALLOWED_CONFIG_WRITES:
                raise HTTPException(
                    status_code=403,
                    detail=f"Section '{section}' is not writable at runtime."
                )

            if not config_settings._config.has_section(section):
                raise HTTPException(
                    status_code=400,
                    detail=f"Section '{section}' not found."
                )

            allowed_keys = ALLOWED_CONFIG_WRITES[section]
            for key, value in values.items():
                if key not in allowed_keys:
                    raise HTTPException(
                        status_code=403,
                        detail=f"Key '{section}.{key}' is not writable at runtime."
                    )
                text = str(value)
                if "\n" in text or "\r" in text:
                    raise HTTPException(status_code=400, detail="Invalid value: newlines not allowed")
                config_settings._config.set(section, key, text)
                updated.append(f"{section}.{key}")

        # Save to nebulondb.cfg (atomic, F-13) then reload the shared
        # instance in place so other modules see the update (F-15).
        config_settings._write()

        # Reload configuration
        config_settings.reload()

        logger.info("Configuration updated: %s", ", ".join(updated))

        return {
            "success": True,
            "message": "Configuration updated successfully.",
            "updated": updated
        }

    except HTTPException:
        raise

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e
