import sys
from pathlib import Path
import shutil

from db.ndb_settings import NDBConfig
from ndb_host.utils.models import load_data, save_data
from db.index_manager import CorpusManager
from utils.logger import logger

class NebulonInitializer:
    """
    Handles setup and verification of the default NebulonDB corpus.
    """
    def __init__(self):
        """
        Initialize the NebulonInitializer with configuration settings.
        """
        self.config = NDBConfig()
        self.manager = CorpusManager()

    def ensure_default_corpus(self, corpus_name: str = "nebulon_origin") -> None:
        """
        Ensure the default corpus exists, creating it if necessary.

        Args:
            corpus_name (str): Name of the default corpus to verify or create.
        """
        corpus_path = Path(self.config.VECTOR_STORAGE) / corpus_name
        metadata = load_data(self.config.VECTOR_METADATA)

        if corpus_path.exists():
            return

        if metadata.get(corpus_name):
            return

        try:
            logger.info(f"Creating default corpus '{corpus_name}'...")
            self.manager.create_corpus(corpus_name, self.config.NEBULONDB_USER, status="system")
            logger.info(f"Corpus '{corpus_name}' created successfully.")
        except Exception as e:
            logger.error(f"Failed to create corpus '{corpus_name}': {e}")
            shutil.rmtree(corpus_path, ignore_errors=True)
            sys.exit(1)
 