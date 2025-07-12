from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI instance
    """
    app = FastAPI(
        title="NebulonDB Vector API",
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