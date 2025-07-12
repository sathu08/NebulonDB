# Import routers from API layer
from api.routes.auth import router as auth_router
from api import create_app
from api.routes.corpus import router as corpus_router

app = create_app()
# Include route modules

app.include_router(auth_router, prefix="/api/NebulonDB/auth", tags=["Authentication"])

app.include_router(corpus_router, prefix="/api/NebulonDB/corpus", tags=["Corpus"])

