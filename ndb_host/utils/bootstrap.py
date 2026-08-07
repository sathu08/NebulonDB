"""
NDB Bootstrap
==========================================================

This module handles the initialization of the NebulonDB API.
It provides endpoints for corpus creation, listing, and deletion.

"""

import sys
import shutil

from pathlib import Path

from time import perf_counter

from core.model_hub import get_auto_batch_size
from utils.models import generate_password

from utils.models import UserRole
from utils.constants import NDBMeta
from db.ndb_settings import NDBConfig

from utils.logger import NebulonDBLogger
from core.model_hub import ModelType, NebulonModelHub


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        NebulonInitializer
# ==========================================================

class NebulonInitializer:
    """
    Handles setup and verification of the default NebulonDB corpus.
    """
    def __init__(self):
        """Initialize the NebulonInitializer with configuration settings."""

        self.config = NDBConfig()

    def bootstrap(self, **kwargs):
        """Bootstrap the NebulonInitializer."""
        
        self.bootstrap_users(**kwargs)
        self.bootstrap_default_corpus()
        self.bootstrap_log_dir()

    def initialize(self):
        """Initialize the NebulonInitializer."""
        
        self.initialize_model()

    def initialize_model(self):
        """Set up logging configuration."""

        logger.info("=" * 60)
        logger.info("Initializing NebulonDB AI Model Hub...")
        logger.info("=" * 60)
        
        total_start = perf_counter()

        try:

            # =====================================================
            # EMBEDDING MODEL CONFIG
            # =====================================================
            embed_cfg = get_auto_batch_size(ModelType.EMBEDDING)

            logger.info(
                f"[Embedding] device={embed_cfg.device} "
                f"batch={embed_cfg.batch_size}"
            )
            self.config.update_model_config(
                device=embed_cfg.device,
                batch_size=embed_cfg.batch_size,
                model_type=ModelType.EMBEDDING
            )
            embed_prefix, embed_model_name = (
                self.config.NEBULONDB_EMBEDDING_MODEL.split("/", 1)
            )

            embed_start = perf_counter()
            self.embedding_model = NebulonModelHub().load_model(
                model_repo_id=embed_model_name,
                prefix=embed_prefix,
            )
            logger.info(
                f"[Embedding] loaded in "
                f"{perf_counter() - embed_start:.2f}s"
            )
            logger.info("[Embedding] warmup complete")

            # =====================================================
            # CROSS ENCODER MODEL CONFIG
            # =====================================================

            cross_encoder_cfg = get_auto_batch_size(ModelType.CROSS_ENCODER)

            logger.info(
                f"[Cross Encoder] device={cross_encoder_cfg.device} "
                f"batch={cross_encoder_cfg.batch_size}"
            )
            self.config.update_model_config(
                device=cross_encoder_cfg.device,
                batch_size=cross_encoder_cfg.batch_size,
                model_type=ModelType.CROSS_ENCODER
            )
            cross_encoder_prefix, cross_encoder_model_name = (
                self.config.NEBULONDB_CROSS_ENCODER_MODEL.split("/", 1)
            )

            cross_encoder_start = perf_counter()
            self.cross_encoder_model = NebulonModelHub().load_model(
                model_repo_id=cross_encoder_model_name,
                prefix=cross_encoder_prefix,
                model_type=ModelType.CROSS_ENCODER,
                is_cache_dir=True,
            )
            logger.info(
                f"[Cross Encoder] loaded in "
                f"{perf_counter() - cross_encoder_start:.2f}s"
            )
            logger.info("[Cross Encoder] warmup complete")

            # =====================================================
            # FINAL STATS
            # =====================================================

            logger.info("-" * 60)
            logger.info(
                f"All models initialized successfully "
                f"in {perf_counter() - total_start:.2f}s"
            )
            logger.info("-" * 60)
        except Exception as e:

            logger.exception(
                f"Model initialization failed: {e}"
            ) 

    
    def bootstrap_default_corpus(self) -> None:
        """
        Ensure the default corpus exists, creating it if necessary.

        Returns:            
            None
        """
        
        from db.index_manager import CorpusManager

        try:
            manager = CorpusManager()
            
            corpus_path = self.config.NEBULONDB_DEFAULT_CORPUS_PATH
            if corpus_path.exists():
                return
            
            logger.info(f"Creating default corpus '{NDBMeta.Corpus.DEFAULT_CORPUS_NAME}'...")
            manager.create_corpus(
                corpus_name=NDBMeta.Corpus.DEFAULT_CORPUS_NAME, 
                username=NDBMeta.User.NEBULONDB_USER, 
                status=UserRole.SYSTEM)
            logger.info(f"Corpus '{NDBMeta.Corpus.DEFAULT_CORPUS_NAME}' created successfully.")

        except Exception as e:
            logger.exception(f"Failed to create default corpus '{NDBMeta.Corpus.DEFAULT_CORPUS_NAME}': {e}")
            shutil.rmtree(corpus_path, ignore_errors=True)
            sys.exit(1)

    def bootstrap_log_dir(self):
        """Ensure log directory structure exists."""
        
        try:
            log_dir = NDBConfig().NEBULONDB_LOG_PATH
            if not log_dir.exists():
                for log_type in NDBMeta.Logging.STRUCTURE:
                    (log_dir / log_type).mkdir(parents=True, exist_ok=True)

            logger.debug("Log directory structure verified and file logging configured.")

        except Exception as e:
            logger.exception(f"Failed to create log directory: {e}")
            sys.exit(1)

    def bootstrap_users(
        self,
        username: str,
        password: str,
        secrets_dir: Path = NDBConfig().NEBULONDB_ACCOUNTHUB_CORPUS_PATH,
        user_role: str = UserRole.USER.value,
    ):
        """
        Bootstrap default users if necessary.
        
        Args:
            username (str): Username of the user to create.
            password (str): Password of the user to create.
            creds_path (Path): Path to the credentials file.
            secrets_dir (Path): Path to the secrets directory.
            user_role (str): Role of the user to create.

        """

        from ndb_host.services.user_service import create_user as service_create_user
        
        try:
            system_password = generate_password()
            service_create_user(username=NDBMeta.User.NEBULONDB_USER, password=system_password, user_role=UserRole.SYSTEM)
            service_create_user(username=username, password=password, user_role=user_role)
            print(f"Default users created successfully. System user password: {system_password} Please store this password securely.")

        except Exception as e:
            logger.exception(f"Failed to create user: {e}")
            shutil.rmtree(secrets_dir, ignore_errors=True)
            sys.exit(1)