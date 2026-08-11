"""
NDB Main Application
==========================================================

This module handles the initialization of the NebulonDB API.
It provides endpoints for user registration and authentication.

"""

# === Import routers from API layer ===
from api import create_app

from fastapi.staticfiles import StaticFiles

from api.routes.auth import router as auth_router
from api.routes.corpus import router as corpus_router
from api.routes.segment import router as segment_router
from api.routes.dashboard import router as dashboard_router, WEB_DIR

from utils.logger import NebulonDBLogger

# Configure colored logging for gunicorn/uvicorn
NebulonDBLogger.configure_server_logging()

app = create_app()

app.mount(
    "/assets",
    StaticFiles(directory=WEB_DIR / "assets"),
    name="assets",
)

# === Include route modules ===

app.include_router(auth_router, prefix="/api/NebulonDB/auth", tags=["Authentication"])

app.include_router(corpus_router, prefix="/api/NebulonDB/corpus", tags=["Corpus"])

app.include_router(segment_router, prefix="/api/NebulonDB/segment", tags=["Segment"])

app.include_router(dashboard_router, prefix="/api/NebulonDB/dashboard", tags=["dashboard"])
