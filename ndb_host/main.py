"""
NDB Main Application
==========================================================

This module handles the initialization of the NebulonDB API.
It provides endpoints for user registration and authentication.

"""

# === Import routers from API layer ===
import threading
from contextlib import asynccontextmanager

from api import create_app

from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from api.routes.auth import router as auth_router
from api.routes.corpus import router as corpus_router
from api.routes.segment import router as segment_router
from api.routes.dashboard import router as dashboard_router, WEB_DIR

from utils.bootstrap import _warmup_models
from utils.logger import NebulonDBLogger

# Configure colored logging for gunicorn/uvicorn
NebulonDBLogger.configure_server_logging()

logger = NebulonDBLogger().get_logger()

app = create_app()

@asynccontextmanager
async def _lifespan(app):
    threading.Thread(target=_warmup_models, daemon=True).start()
    yield

app.router.lifespan_context = _lifespan

class NoCacheAssetsMiddleware(BaseHTTPMiddleware):
    """Prevent browsers from serving stale HTML/JS/CSS for the web console."""

    _NO_CACHE_PATHS = ("/assets", "/dashboard", "/dashboard.html", "/index", "/")

    async def dispatch(self, request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/assets") or path.startswith("/api/NebulonDB/dashboard") or path in self._NO_CACHE_PATHS:
            response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        return response


app.add_middleware(NoCacheAssetsMiddleware)

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
