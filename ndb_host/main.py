# === Import routers from API layer ===
from pathlib import Path
from api import create_app

from api.routes.auth import router as auth_router
from api.routes.corpus import router as corpus_router
from api.routes.segment import router as segment_router

from db.ndb_settings import NDBConfig
from utils.logger import NebulonDBLogger

# Initialize Logging for the Server Process                                                                 
log_dir = Path(NDBConfig().NEBULONDB_LOG)
NebulonDBLogger().configure_file_logging(log_dir=str(log_dir))

app = create_app()

# === Include route modules ===

app.include_router(auth_router, prefix="/api/NebulonDB/auth", tags=["Authentication"])

app.include_router(corpus_router, prefix="/api/NebulonDB/corpus", tags=["Corpus"])

app.include_router(segment_router, prefix="/api/NebulonDB/segment", tags=["Segment"])
