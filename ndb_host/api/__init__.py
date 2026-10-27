"""
NDB API
==========================================================

This module handles the FastAPI application setup and configuration.

"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

import os
import re
import sys


def _configured_worker_count() -> int | None:
    """Best-effort detection of a multi-worker server configuration."""
    raw = os.environ.get("WEB_CONCURRENCY", "").strip()
    if raw.isdigit():
        return int(raw)

    for args in (os.environ.get("GUNICORN_CMD_ARGS", ""), " ".join(sys.argv)):
        match = re.search(r"(?:--workers[=\s]+|-w\s+)(\d+)", args)
        if match:
            return int(match.group(1))
    return None


def _enforce_single_worker() -> None:
    workers = _configured_worker_count()
    if workers is not None and workers > 1:
        raise RuntimeError(
            "NebulonDB requires a single worker: Nova/Mesh state is held in "
            "process memory and is not shared across gunicorn workers. "
            f"Detected {workers} workers."
        )


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        FastAPI: Configured FastAPI instance
    """
    _enforce_single_worker()
    app = FastAPI(
        title="NebulonDB Nova & Mesh API",
        description="Secure API to manage user access and vector database corpus",
        version="0.1.0",
        openapi_tags=[
            {
                "name": "Authentication",
                "description": "User registration and authentication endpoints"
            },
            {
                "name": "Corpus Management",
                "description": "Manage vector corpora (create, list, search)"
            }
        ]
    )

    # === Middleware Setup ===
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app
