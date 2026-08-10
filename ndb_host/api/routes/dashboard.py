"""
NDB API System Management
==========================================================

This module handles system-level operations for the NDB API.
It provides endpoints for viewing and updating NebulonDB
configuration.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from utils.logger import NebulonDBLogger
from utils.constants import ConfigUpdate
from ndb_host.db.ndb_settings import NDBConfig

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
async def get_config():
    return {
        "segments": {
            "wal_auto_flush": config_settings.WAL_AUTO_FLUSH,
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
async def update_config(payload: ConfigUpdate):
    global config_settings
    try:
        updated = []

        for section, values in payload.config.items():

            if not config_settings._config.has_section(section):
                raise HTTPException(
                    status_code=400,
                    detail=f"Section '{section}' not found."
                )

            for key, value in values.items():
                config_settings._config.set(section, key, str(value))
                updated.append(f"{section}.{key}")

        # Save to nebulondb.cfg
        config_settings._write()

        # Reload configuration
        config_settings = NDBConfig(config_settings.config_path)

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
        raise HTTPException(status_code=500, detail=str(e))