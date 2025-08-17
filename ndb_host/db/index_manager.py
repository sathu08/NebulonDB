from datetime import datetime, timezone
from configobj import ConfigObj
import os
from typing import List, Dict, Union
from string import Template
from pathlib import Path
import shutil

from utils.models import load_data, save_data

class NebulonDBConfig:
    """
    NebulonDB Configuration Loader

    Loads configuration from `nebulondb.cfg`, supports environment overrides
    and safely resolves variable placeholders using string.Template and os.path.expandvars.
    """

    @staticmethod
    def _resolve_path(path_vars: dict, value: str) -> str:
        """Resolve variables using provided path_vars and environment."""
        combined = dict(path_vars)
        return os.path.expandvars(Template(value).safe_substitute(combined))

    # Load config with comments preserved
    try:
        _config = ConfigObj('nebulondb.cfg', encoding='utf-8', list_values=False)
    except Exception as e:
        raise RuntimeError(f"Failed to load config file: {e}")

    # Validate required sections
    required_sections = ['paths', 'corpus', 'vector_index', 'params']
    for section in required_sections:
        if section not in _config:
            raise KeyError(f"Missing required section: '{section}' in config file.")

    if 'NEBULONDB_HOME' not in _config['paths']:
        raise KeyError("Missing 'NEBULONDB_HOME' in [paths] section.")

    # Apply environment override
    if 'NEBULONDB_HOME' in os.environ:
        env_home = os.environ['NEBULONDB_HOME']
        if _config['paths']['NEBULONDB_HOME'] != env_home:
            _config['paths']['NEBULONDB_HOME'] = env_home
            _config.write()

    # === Paths ===
    NEBULONDB_HOME = _resolve_path(_config['paths'], _config['paths']['NEBULONDB_HOME'])
    VECTOR_STORAGE = _resolve_path(_config['paths'], _config["paths"]["VECTOR_STORAGE"])
    USER_CREDENTIALS = _resolve_path(_config['paths'], _config["paths"]["USER_CREDENTIALS"])
    VECTOR_METADATA = _resolve_path(_config['paths'], _config["paths"]["VECTOR_METADATA"])

    # === Corpus ===
    DEFAULT_CORPUS_CONFIG_STRUCTURES = _resolve_path(_config['paths'], _config["corpus"]["DEFAULT_CORPUS_CONFIG_STRUCTURES"])
    DEFAULT_CORPUS_STRUCTURES = [
        item.strip() for item in _config['corpus']['DEFAULT_CORPUS_STRUCTURES'].split(',')
    ]
    SEGMENT_DIR_NAME = _config["corpus"]["CORPUS_SEGMENT"]

    # === Vector Index Config ===
    DEFAULT_CORPUS_CONFIG_DATA = {
        "dimension": int(_config['vector_index']['dimension']),
        "index_type": _config['vector_index']['index_type'],
        "metric": _config['vector_index']['metric'],
        "params": {
            "nlist": int(_config['params']['nlist']),
            "nprobe": int(_config['params']['nprobe']),
            "m": int(_config['params']['m']),
            "nbits": int(_config['params']['nbits']),
            "hnsw_m": int(_config['params']['hnsw_m']),
            "ef_construction": int(_config['params']['ef_construction']),
            "ef_search": int(_config['params']['ef_search']),
        }
    }

    # === Segments ===
    SEGMENTS_METADATA = _config["segments"]["METADATA_SEGMENTS"]

    # === Segments ===
    SEGMENTS_METADATA = _config["segments"]["METADATA_SEGMENTS"]

class CorpusManager:
    """
    CorpusManager handles validation and retrieval of corpus data and metadata.
    """
    def __init__(self):
        self.vector_storage_path = Path(NebulonDBConfig.VECTOR_STORAGE)
        self.metadata_path = Path(NebulonDBConfig.VECTOR_METADATA)
        self.user_credential_path = Path(NebulonDBConfig.USER_CREDENTIALS)
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Check that essential paths exist."""
        errors = []
        if not self.vector_storage_path.exists() or not self.vector_storage_path.is_dir():
            errors.append(f"Vector storage path missing: {self.vector_storage_path}")
        if not self.metadata_path.exists():
            errors.append(f"Metadata file not found: {self.metadata_path}")
        if not self.user_credential_path.exists():
            errors.append(f"User credentials file not found: {self.user_credential_path}")

        if errors:
            raise FileNotFoundError(" | ".join(errors))

    @staticmethod
    def generate_corpus_metadata(corpus_name: str, created_by: str, status:str="active") -> Dict[str, str]:
        """
        Generate metadata dictionary for a new corpus.

        Args:
            corpus_name (str): Name of the corpus.
            created_by (str): User who created the corpus.

        Returns:
            Dict[str, str]: Metadata entry.
        """
        return {
            "corpus_name": corpus_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "created_by": created_by,
            "status": status,
            "segments": []
        }

    def get_available_corpus_list(self) -> List[str]:
        """
        Get list of corpus directories that have matching metadata.

        Returns:
            List[str]: Matching corpus names.

        """
        try:
            vector_dirs = [d.name for d in self.vector_storage_path.iterdir() if d.is_dir()]
            if not vector_dirs:
                return []

            metadata_names = [meta.get('corpus_name') for meta in load_data(self.metadata_path).values()]
            if not metadata_names:
                return []
            matched_corpora = sorted(set(vector_dirs) & set(metadata_names))
            return matched_corpora
        except Exception as _:
            return []


    def get_corpus_status(self, corpus_name: str) -> str:
        """
        Retrieve the status of a specified corpus.

        Args:
            corpus_name (str): Name of the corpus.

        Returns:
            str: Status of the specified corpus (e.g., 'active', 'deactivate', 'system').

        """
        try:
            metadata = load_data(self.metadata_path)
            return metadata.get(corpus_name, {}).get("status")
        except Exception as _:
            return None


    def set_corpus_status(self, corpus_name: str, status: str) -> None:
        """
        Update the status of a specified corpus.

        Args:
            corpus_name (str): Name of the corpus to update.
            status (str): New status value (e.g., 'active', 'deactivate', 'system').

        """
        try:
            metadata = load_data(self.metadata_path)
            metadata[corpus_name]["status"] = status
            save_data(metadata, self.metadata_path)
            return True
        except Exception as _:
            return False

class  SegmentManager:
    """
    SegmentManager handles validation and retrieval of segment data and metadata.
    """
    def __init__(self, corpus_name:str):
        self.corpus_path = Path(NebulonDBConfig.VECTOR_STORAGE) / corpus_name
        self.vector_storage_path = Path(NebulonDBConfig.VECTOR_STORAGE) / corpus_name
        self.segment_metadata_path = self.vector_storage_path / NebulonDBConfig.SEGMENTS_METADATA
        self._validate_paths()
        

    def _validate_paths(self) -> None:
        """Check that essential paths exist."""
        errors = []
        if not self.vector_storage_path.exists() or not self.vector_storage_path.is_dir():
            errors.append(f"Vector storage path missing: {self.vector_storage_path}")
        if not self.segment_metadata_path.exists():
            errors.append(f"Metadata file not found: {self.segment_metadata_path}")

        if errors:
            raise FileNotFoundError(" | ".join(errors))


    def get_available_segment_list(self) -> List[str]:
        """
        Get list of segment directories that have matching metadata.

        Returns:
            List[str]: Matching segment names.

        """
        try:
            metadata = load_data(self.segment_metadata_path)

            if not metadata:
                return {}
            product_names = list(metadata.keys())
            metadata_names = [meta.get('segment') for meta in metadata.values() if isinstance(meta, dict) and 'segment' in meta]

            return {
                "product_names": product_names,
                "metadata_names": metadata_names
            } if product_names and metadata_names else {}
            
        except Exception as _:
            return {}
    

    
