import sys
import shutil
from pathlib import Path

from utils.models import load_data, save_data, ensure_embedding_model, get_auto_batch_size

from db.ndb_settings import NDBConfig, NDBSafeLocker
from utils.models import UserRole, NDBCorpusMeta
from utils.logger import NebulonDBLogger


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

        logger.info("Initializing embedding model...")
        
        logger.info("Trying to auto detect device and batch size...")

        batch_size, device = get_auto_batch_size()
        self.config.update_llm_config(device, batch_size)

        logger.info("Device: {}".format(device))
        logger.info("Batch size: {}".format(batch_size))
        
        ensure_embedding_model(self.config.NEBULONDB_EMBEDDING_MODEL)
        logger.info("Embedding model ready.")
    
    def bootstrap_default_corpus(self) -> None:
        """
        Ensure the default corpus exists, creating it if necessary.

        Returns:            
            None
        """
        
        from db.index_manager import CorpusManager
        
        # === Create corpus ===
        try:
            # === Initialize corpus manager ===
            manager = CorpusManager()
            
            # === Initialize corpus path ===
            corpus_path = Path(self.config.VECTOR_STORAGE) / NDBCorpusMeta.DEFAULT_CORPUS_NAME
            
            # === Initialize metadata ===
            metadata = load_data(self.config.VECTOR_METADATA)

            # === Check if corpus exists ===
            if corpus_path.exists():
                return

            # === Check if corpus exists in metadata ===
            if metadata.get(NDBCorpusMeta.DEFAULT_CORPUS_NAME):
                return

            logger.info(f"Creating default corpus '{NDBCorpusMeta.DEFAULT_CORPUS_NAME}'...")
            manager.create_corpus(NDBCorpusMeta.DEFAULT_CORPUS_NAME, self.config.NEBULONDB_USER, status=UserRole.SUPER_USER)
            logger.info(f"Corpus '{NDBCorpusMeta.DEFAULT_CORPUS_NAME}' created successfully.")

        except Exception as e:
            logger.exception(f"Failed to create default corpus '{NDBCorpusMeta.DEFAULT_CORPUS_NAME}': {e}")
            shutil.rmtree(corpus_path, ignore_errors=True)
            sys.exit(1)

    def bootstrap_log_dir(self):
        """Ensure log directory structure exists."""
        
        try:
            log_dir = Path(NDBConfig().NEBULONDB_LOG)
            if not log_dir.exists():
                for log_type in NDBCorpusMeta.LOG_STRUCTURE:
                    (log_dir / log_type).mkdir(parents=True, exist_ok=True)

            logger.debug("Log directory structure verified and file logging configured.")

        except Exception as e:
            logger.exception(f"Failed to create log directory: {e}")
            sys.exit(1)

    def bootstrap_users(
        self,
        username: str,
        password: str,
        creds_path: Path,
        secrets_dir: Path,
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

            # === Create internal system user first ===
            system_user_data = service_create_user(username=self.config.NEBULONDB_USER, password=password, user_role=UserRole.SUPER_USER, new_creation=True)
            
            # === Create actual user ===
            normal_user_data = service_create_user(username=username, password=password, user_role=user_role, new_creation=True)

            # === Merge both user records into one dictionary ===
            combined_users = {**system_user_data, **normal_user_data}

            # === Save user database ===
            save_data(data=combined_users, path_loc=str(creds_path))
            logger.info(f"User created successfully and saved")

            # === Encrypt credentials with NDBSafeLocker ===
            NDBSafeLocker(str(secrets_dir))
            logger.info("Credentials secured in NDB format.")

            # === Initialize metadata file ===
            meta_data = load_data(Path(self.config.VECTOR_METADATA))
            save_data(data=meta_data, path_loc=str(self.config.VECTOR_METADATA))

        except Exception as e:
            logger.exception(f"Failed to create user: {e}")
            shutil.rmtree(secrets_dir, ignore_errors=True)
            sys.exit(1)