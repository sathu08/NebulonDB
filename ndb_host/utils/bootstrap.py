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
from core.model_hub import ModelType


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        Model Warmup
# ==========================================================

def _warmup_models(cfg: NDBConfig = None) -> None:
    """Load the embedding model and encode a tiny dummy text so torch/CUDA
    context is initialized before the first real request.

    Skips when ``nebulondb_warm_models = false`` in the ``[llm]`` section
    of the config file. The cross-encoder stays lazy (loaded on first rerank).
    """

    cfg = cfg or NDBConfig()
    if not getattr(cfg, "NEBULONDB_WARM_MODELS", True):
        logger.info("[ModelHub] Warmup disabled (nebulondb_warm_models=False)")
        return

    try:
        from core.model_hub import SemanticEmbeddingModel

        start = perf_counter()
        logger.info("[ModelHub] Background warmup started...")
        SemanticEmbeddingModel().encode(["nebulon warmup"], normalize_embeddings=True)
        logger.info(
            f"[ModelHub] Embedding warmup complete in {perf_counter() - start:.2f}s"
        )
    except Exception:
        logger.exception(
            "[ModelHub] Background warmup failed; the model will load on first use"
        )

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
            _warmup_models(self.config)

            # =====================================================
            # CROSS ENCODER (deferred - loaded lazily on first rerank)
            # =====================================================

            logger.info("[Cross Encoder] deferred: loaded lazily on first rerank")

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
        Ensure that the default corpus exists.

        If the default corpus does not exist, this method creates the corpus
        and loads a sample segment dataset into it.

        Returns:
            None
        """

        import polars as pl

        from db.index_manager import CorpusManager, SegmentManager

        try:
            corpus_name = NDBMeta.Corpus.DEFAULT_CORPUS_NAME
            segment_name = "ndb_sample_data"
            username = NDBMeta.User.NEBULONDB_USER
            ndb_type = NDBMeta.Type.COSMOS
            status = UserRole.SYSTEM.value
            corpus_path = self.config.NEBULONDB_DEFAULT_CORPUS_PATH

            # Default corpus already exists.
            if corpus_path.exists():
                logger.info(f"Default {ndb_type} corpus '{corpus_name}' already exists.")
                return

            logger.info( f"Creating default {ndb_type} corpus '{corpus_name}'...")

            corpus_manager = CorpusManager()

            corpus_manager.create_corpus(
                corpus_name=corpus_name,
                username=username,
                ndb_type=ndb_type,
                status=status,
            )

            logger.info(f"Default {ndb_type} corpus '{corpus_name}' created successfully.")

            segment_manager = SegmentManager(
                corpus_name=corpus_name,
                segment_name=segment_name,
                ndb_type=ndb_type,
            )

            logger.info(f"Loading sample segment '{segment_name}' into {ndb_type} corpus '{corpus_name}'...")

            # Sample NebulonDB dataset.
            segment_dataset = pl.DataFrame(
                {
                    "segment_text": [
                        "NebulonDB stores data using immutable segments",
                        "NebulonDB supports vector and metadata search",
                        "Segment compaction improves storage efficiency",
                    ],
                    "segment_metadata": [
                        "database=nebulondb, type=segment",
                        "database=nebulondb, type=vector",
                        "database=nebulondb, type=metadata",
                    ],
                }
            )

            lang_type = "en"
            doc_type = "other"

            segment_manager.load_segment(
                segment_dataset=segment_dataset,
                columns=[
                    "segment_text",
                    "segment_metadata",
                ],
                doc_type=doc_type,
                lang_type=lang_type,
            )

            logger.info(f"Sample segment '{segment_name}' loaded successfully into {ndb_type} corpus '{corpus_name}'.")

        except Exception as e:
            logger.exception(f"Failed to create default corpus '{corpus_name}': {e}")
            shutil.rmtree(corpus_path, ignore_errors=True)
            sys.exit(1)

    def bootstrap_log_dir(self):
        """Ensure the log directory exists."""

        try:
            log_dir = NDBConfig().NEBULONDB_LOG_PATH
            log_dir.mkdir(parents=True, exist_ok=True)

            logger.debug("Log directory verified and file logging configured.")

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
            service_create_user(
                username=NDBMeta.User.NEBULONDB_USER,
                password=system_password,
                user_role=UserRole.SYSTEM,
            )
            service_create_user(username=username, password=password, user_role=user_role)
            print(
                f"Default users created successfully. System user password: {system_password} "
                "Please store this password securely."
            )

        except Exception as e:
            logger.exception(f"Failed to create user: {e}")
            shutil.rmtree(secrets_dir, ignore_errors=True)
            sys.exit(1)
