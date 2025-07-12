import uvicorn
import logging
from typing import Optional
from utils.logger import logger


def start_app(reload: bool = False):
    """
    Start the FastAPI application using Uvicorn.
    
    Args:
        reload (bool): Enable auto-reload (useful for development)
    """
    try:
        logger.info("Starting NebulonDB Vector API server...")
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=6969,
            reload=reload,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    # Set reload=True during development, False in production
    start_app(reload=False)