"""
NDB Configuration
==========================================================

This module handles configuration settings for the NDB API.

"""

import os
import shutil
import base64

import json
import zipfile
import tempfile

from io import BytesIO
from pathlib import Path
from string import Template

from configparser import ConfigParser
from cryptography.fernet import Fernet

from platformdirs import user_cache_dir

from utils.constants import AuthenticationConfig, NDBMeta
from utils.logger import NebulonDBLogger


# ==========================================================
#        Initialize Logger
# ==========================================================

logger = NebulonDBLogger().get_logger()

# ==========================================================
#        NDBConfig
# ==========================================================

class NDBConfig:
    """
    NebulonDB Configuration Loader

    Loads configuration from a specified config file (default: `nebulondb.cfg`),
    supports environment overrides, safely resolves variables using
    string.Template and os.path.expandvars, and enforces secure permissions
    on sensitive directories.
    """

    def __init__(self, config_path: str | Path = None):
        """
        Initialize the NebulonDB configuration loader.

        Args:
            config_path (str): Path to the configuration file.
        """

        if config_path is None:
            neb_home = os.environ.get('NEBULONDB_HOME', os.getcwd())
            config_path = Path(neb_home) / "nebulondb.cfg"
        else:
            config_path = Path(config_path)

        self.config_path = config_path.resolve()
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            self._config = ConfigParser()
            self._config.read(self.config_path, encoding=AuthenticationConfig.ENCODING)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{self.config_path}': {e}")

        self._validate_sections()
        self._apply_env_override()
        self._load_environment()
        self._load_paths()
        self._load_search_config()
        self._load_rank_config()
        self._load_segments()
        self._load_server()
        self._load_llm()

    # ------------------------------
    #  Private Utility Methods 
    # ------------------------------

    @staticmethod
    def _resolve_path(path_vars: dict, value: str) -> Path:
        """Resolve variables using provided path_vars and environment, return a Path."""
        resolved = os.path.expandvars(Template(value).safe_substitute(path_vars))
        return Path(resolved).resolve()
    
    def _write(self):
        """Persist current config state back to disk."""
        with self.config_path.open('w', encoding=AuthenticationConfig.ENCODING) as f:
            self._config.write(f)

    def _validate_sections(self):
            required_sections = ['paths','vector', 'hnsw', 'server', 'environment']
            for section in required_sections:
                if section not in self._config:
                    raise KeyError(f"Missing required section: '{section}' in config file.")
    
            if 'NEBULONDB_HOME' not in self._config['paths']:
                raise KeyError("Missing 'NEBULONDB_HOME' in [paths] section.")
    
            if 'NEBULONDB_MASTER_KEY' not in self._config['environment']:
                raise KeyError("Missing 'NEBULONDB_MASTER_KEY' in [environment] section.")

    def _apply_env_override(self):
        updated = False

        # Override NEBULONDB_HOME
        env_home = os.environ.get('NEBULONDB_HOME')
        if env_home and self._config.get('paths', 'NEBULONDB_HOME') != env_home:
            self._config.set('paths', 'NEBULONDB_HOME', env_home)
            updated = True

        # Ensure NEBULONDB_MASTER_KEY exists
        env_master_key = os.environ.get('NEBULONDB_MASTER_KEY')
        cfg_master_key = self._config.get('environment', 'NEBULONDB_MASTER_KEY', fallback='')

        if not env_master_key and not cfg_master_key:
            generated = Fernet.generate_key().decode()
            logger.warning("NEBULONDB_MASTER_KEY not found; generated new key.")
            self._config.set('environment', 'NEBULONDB_MASTER_KEY', generated)
            updated = True

        if updated:
            self._write()

    # ------------------------------
    # Load Config Sections
    # ------------------------------

    def _load_paths(self):
        paths = dict(self._config['paths'])
        self.NEBULONDB_HOME = Path(self._resolve_path(paths, self._config.get('paths', 'NEBULONDB_HOME')))

        self.NEBULONDB_STORAGE_PATH = self.NEBULONDB_HOME / NDBMeta.Paths.STORAGE_DIR

        self.NEBULONDB_ACCOUNTHUB_CORPUS_PATH = self.NEBULONDB_STORAGE_PATH / NDBMeta.Paths.SECRETS_DIR
        self.NEBULONDB_DEFAULT_CORPUS_PATH = self.NEBULONDB_STORAGE_PATH / NDBMeta.Corpus.DEFAULT_CORPUS_NAME

        self.NEBULONDB_LOG_PATH = self.NEBULONDB_HOME / NDBMeta.Paths.LOG_DIR
        self.NEBULONDB_PID_FILE = self.NEBULONDB_HOME / NDBMeta.Paths.PID_FILE
        self.NEBULONDB_WEB_DIR =  self.NEBULONDB_HOME / NDBMeta.Paths.WEB_DIR

    def _load_environment(self):
        self.ENVIRONMENT_MASTER_KEY = self._config.get('environment', 'NEBULONDB_MASTER_KEY')
        self.KEYRING_ENABLED = self._config.getboolean('environment', 'NEBULONDB_KEYRING_ENABLED', fallback=False)
        self.KEYRING_SERVICE = self._config.get('environment', 'NEBULONDB_KEYRING_SERVICE', fallback='')

    def _load_llm(self):
        self.NEBULONDB_DEFAULT_MODE = self._config.getboolean('llm', 'NEBULONDB_DEFAULT_MODE', fallback=False)

        cache_dir = Path(self._config.get('llm', 'NEBULONDB_MODEL_CACHE_DIR', fallback=''))
        if not cache_dir or not cache_dir.exists():
            cache_dir = str(Path(user_cache_dir(self.APP_NAME)))
            self._config.set('llm', 'NEBULONDB_MODEL_CACHE_DIR', cache_dir)
            self._write()

        self.NEBULONDB_MODEL_CACHE_DIR = cache_dir
        self.NEBULONDB_EMBEDDING_MODEL = self._config.get('llm', 'NEBULONDB_EMBEDDING_MODEL')
        self.NEBULONDB_CROSS_ENCODER_MODEL = self._config.get('llm', 'NEBULONDB_CROSS_ENCODER_MODEL')
        self.NEBULONDB_EMBEDDING_BATCH_SIZE = self._config.getint('llm', 'NEBULONDB_EMBEDDING_BATCH_SIZE', fallback=0)
        self.NEBULONDB_CROSS_ENCODER_BATCH_SIZE = self._config.getint('llm', 'NEBULONDB_CROSS_ENCODER_BATCH_SIZE', fallback=0)
        self.NEBULONDB_EMBEDDING_MODEL_DEVICE = self._config.get('llm', 'NEBULONDB_EMBEDDING_MODEL_DEVICE', fallback='')
        self.NEBULONDB_CROSS_ENCODER_MODEL_DEVICE = self._config.get('llm', 'NEBULONDB_CROSS_ENCODER_MODEL_DEVICE', fallback='')

    def _load_search_config(self):
        self.DEFAULT_CORPUS_CONFIG_DATA = {
            "dimension": self._config.getint('vector', 'DIMENSION'),
            "space": self._config.get('vector', 'SPACE'),
            "top_matches": self._config.getint('vector', 'TOP_MATCHES'),
            "min_score": self._config.getfloat('vector', 'MIN_SCORE'),
            "m": self._config.getint('hnsw', 'M'),
            "ef_construction": self._config.getint('hnsw', 'EF_CONSTRUCTION'),
            "ef_search": self._config.getint('hnsw', 'EF_SEARCH'),
            "bloom_enabled": self._config.get('bloom', 'ENABLED', fallback='false'),
            "bloom_bits_per_key": self._config.getint('bloom', 'BITS_PER_KEY', fallback=10),
            "bloom_hash_count": self._config.getint('bloom', 'HASH_COUNT', fallback=7),
        }
        self.VECTOR_COMPACTION_THRESHOLD = self._config.getfloat("vector", "COMPACTION_THRESHOLD", fallback=0.4)
        self.VECTOR_SAVE_EVERY_N = self._config.getint("vector", "SAVE_EVERY_N", fallback=100)

    def _load_rank_config(self):
        self.RANK_TOPK = self._config.getint('rank', 'RANK_TOPK', fallback=20)
        self.RANK_WEIGHTS = {
            "vector": self._config.getfloat('rank', 'WEIGHT_VECTOR', fallback=0.55),
            "bm25": self._config.getfloat('rank', 'WEIGHT_BM25', fallback=0.20),
            "metadata": self._config.getfloat('rank', 'WEIGHT_METADATA', fallback=0.10),
            "importance": self._config.getfloat('rank', 'WEIGHT_IMPORTANCE', fallback=0.10),
            "freshness": self._config.getfloat('rank', 'WEIGHT_FRESHNESS', fallback=0.05),
        }

    def _load_segments(self):
        self.FLUSH_RECORD_THRESHOLD = self._config.getint('segments', 'FLUSH_RECORD_THRESHOLD')
        self.WAL_AUTO_FLUSH = self._config.getboolean('segments', 'WAL_AUTO_FLUSH', fallback=True)
        self.COMPRESS_SEGMENTS = self._config.getboolean('segments', 'COMPRESS_SEGMENTS', fallback=True)
        self.BLOOM_FILTER_ENABLED = self._config.getboolean('segments', 'BLOOM_FILTER_ENABLED', fallback=True)
        self.MAX_OPEN_SEGMENTS = self._config.getint('segments', 'MAX_OPEN_SEGMENTS', fallback=50)
        self.COMPACTION_INTERVAL = self._config.getfloat('segments', 'COMPACTION_INTERVAL', fallback=60.0)
        self.MAX_SEGMENTS_BEFORE_COMPACT = self._config.getint('segments', 'MAX_SEGMENTS_BEFORE_COMPACT', fallback=10)
        self.FLUSH_INTERVAL = self._config.getfloat('segments', 'FLUSH_INTERVAL', fallback=5.0)

    def _load_server(self):
        self.APP_NAME = self._config.get('server', 'APP_NAME')
        self.HOST = self._config.get('server', 'HOST')
        self.PORT = self._config.getint('server', 'PORT')
        self.WORKERS = self._config.getint('server', 'WORKERS')
        self.TIMEOUT = self._config.getint('server', 'TIMEOUT', fallback=30)
        self.KEEP_ALIVE = self._config.getint('server', 'KEEP_ALIVE', fallback=5)
        self.GRACEFUL_TIMEOUT = self._config.getint('server', 'GRACEFUL_TIMEOUT', fallback=30)
        self.ACCESS_LOGFILE = self._config.get('server', 'ACCESS_LOGFILE', fallback='')
        self.ERROR_LOGFILE = self._config.get('server', 'ERROR_LOGFILE', fallback='')
        self.LOG_LEVEL = self._config.get('server', 'LOG_LEVEL', fallback='info')

    def update_model_config(self, device: str, batch_size: int, model_type: str):
        section = 'llm'
        prefix = f'NEBULONDB_{model_type.upper()}'
        updated = False

        device_key = f'{prefix}_MODEL_DEVICE'
        if self._config.get(section, device_key) != device:
            self._config.set(section, device_key, device)
            updated = True

        batch_key = f'{prefix}_BATCH_SIZE'
        if self._config.getint(section, batch_key) != batch_size:
            self._config.set(section, batch_key, str(batch_size))
            updated = True

        if updated:
            self._write()

# ==========================================================
#        NDBCryptoManager
# ==========================================================

class NDBCryptoManager:
    """
        Handles encryption, decryption, and key management for NDB files.
        - Uses Fernet symmetric encryption for strong confidentiality.
        - Protects per-file NDB key with a persistent master key.
    """

    def __init__(self):
        self.config = NDBConfig()

    # ------------------------------
    # Get Master Key
    # ------------------------------
    def get_master_key(self):
        """
            Retrieve or create a persistent master key.
            Order:
            1. Environment variable
            2. Config file
            3. System keyring (if enabled)
            4. Fallback: generate temporary key
        """

        env_key = os.environ.get("NEBULONDB_MASTER_KEY")
        if env_key:
            return env_key.encode()
    
        config_key = getattr(self.config, "ENVIRONMENT_MASTER_KEY", None)
        if config_key:
            return config_key.encode()
        
        # === As a last resort, generate a temporary in-memory key ===
        logger.warning("[Warning] No valid key found — generating temporary master key.")

    # ------------------------------
    # Encrypt data
    # ------------------------------
    def encrypt_data(self, data: bytes) -> dict[str, str]:
        """
            Encrypt raw data bytes using a generated NDB key and master key.
            Returns a JSON-safe dictionary containing Base64 strings.
        """
        ndb_key = Fernet.generate_key()
        fernet_ndb = Fernet(ndb_key)
        encrypted_data = fernet_ndb.encrypt(data)

        master_key = self.get_master_key()
        fernet_master = Fernet(master_key)
        encrypted_ndb_key = fernet_master.encrypt(ndb_key)

        return {
            "ndb_key": base64.b64encode(encrypted_ndb_key).decode(AuthenticationConfig.ENCODING),
            "ndb_data": base64.b64encode(encrypted_data).decode(AuthenticationConfig.ENCODING)
        }

    # ------------------------------
    # Decrypt data
    # ------------------------------
    def decrypt_data(self, encrypted_content: dict):
        """Decrypt encrypted NDB content and return the original bytes."""
        master_key = self.get_master_key()
        fernet_master = Fernet(master_key)
        ndb_key = fernet_master.decrypt(base64.b64decode(encrypted_content["ndb_key"].encode(AuthenticationConfig.ENCODING)))
        fernet_ndb = Fernet(ndb_key)
        return fernet_ndb.decrypt(base64.b64decode(encrypted_content["ndb_data"].encode(AuthenticationConfig.ENCODING)))


# ==========================================================
#        NDBSafeLocker
# ==========================================================

class NDBSafeLocker:
    """
        Securely manages encrypted .ndb containers (zip + AES encryption).
        Provides methods to:
        - Encrypt folders into NDB
        - List / read / write / delete files
        - Extract all files
        - Save changes securely
    """

    def __init__(self, path, force=False, delete_source=True):
        self.config_settings = NDBConfig()
        self.crypto_manager = NDBCryptoManager()
        self._ndb_path = None
        self._zip_bytes_io = None

        if os.path.isdir(path):
            ndb_file = path.rstrip('/\\') + ".ndb"
            self._ndb_path = ndb_file
            self._encrypt_folder_to_ndb(path, ndb_file, force=force, delete_source=delete_source)
        elif os.path.isfile(path):
            self._ndb_path = path
        else:
            raise ValueError(f"{path} is not a valid folder or file.")

        self._load_ndb(self._ndb_path)

    # ------------------------------
    # Encrypt Folder → .ndb file
    # ------------------------------
    def _encrypt_folder_to_ndb(self, src_folder, output_file, force=False, delete_source=True):
        """Encrypt a folder and save it as an .ndb file."""

        if os.path.exists(output_file) and not force:
            return

        with tempfile.NamedTemporaryFile(delete=False) as tmp_zip:
            tmp_zip_path = tmp_zip.name

        try:
            with zipfile.ZipFile(tmp_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(src_folder):
                    for f in files:
                        abs_path = os.path.join(root, f)
                        rel_path = os.path.relpath(abs_path, src_folder)
                        zipf.write(abs_path, rel_path)

            with open(tmp_zip_path, 'rb') as f:
                zip_bytes = f.read()

            # Use helper to encrypt
            ndb_content = self.crypto_manager.encrypt_data(zip_bytes)

            with open(output_file, 'w', encoding=AuthenticationConfig.ENCODING) as f:
                json.dump(ndb_content, f, ensure_ascii=False)

            if delete_source:
                shutil.rmtree(src_folder)

        finally:
            if os.path.exists(tmp_zip_path):
                os.remove(tmp_zip_path)

    # ------------------------------
    # Load NDB into memory
    # ------------------------------
    def _load_ndb(self, ndb_file):
        """Load the encrypted NDB file into memory."""

        with open(ndb_file, 'r', encoding=AuthenticationConfig.ENCODING) as f:
            content = json.load(f)

        zip_bytes = self.crypto_manager.decrypt_data(content)
        self._zip_bytes_io = BytesIO(zip_bytes)
        self._zipfile = zipfile.ZipFile(self._zip_bytes_io, 'r') 

    # ------------------------------
    # File operations
    # ------------------------------
    def list_files(self):
        """List all files stored in the encrypted NDB."""

        return self._zipfile.namelist()

    def read_file(self, file_path, as_text=True):
        """Read a file from the encrypted NDB."""

        if file_path not in self._zipfile.namelist():
            raise FileNotFoundError(f"{file_path} not found in NDB.")
        data = self._zipfile.read(file_path)
        return data.decode(AuthenticationConfig.ENCODING) if as_text else data

    def write_file(self, file_path, data: bytes):
        """Write or replace a file inside the NDB."""

        temp_io = BytesIO()
        with zipfile.ZipFile(temp_io, 'w') as zipf:
            for f in self._zipfile.namelist():
                if f != file_path:
                    zipf.writestr(f, self._zipfile.read(f))
            zipf.writestr(file_path, data)
        self._zip_bytes_io = temp_io
        self._zipfile = zipfile.ZipFile(self._zip_bytes_io, 'r')

    def delete_file(self, file_path):
        """Remove a file from the NDB."""

        if file_path not in self._zipfile.namelist():
            raise FileNotFoundError(f"{file_path} not found in NDB.")
        temp_io = BytesIO()
        with zipfile.ZipFile(temp_io, 'w') as zipf:
            for f in self._zipfile.namelist():
                if f != file_path:
                    zipf.writestr(f, self._zipfile.read(f))
        self._zip_bytes_io = temp_io
        self._zipfile = zipfile.ZipFile(self._zip_bytes_io, 'r')

    # ------------------------------
    # Utility Additions
    # ------------------------------
    def extract_all(self, output_dir):
        """Extract all files from NDB to the specified output directory."""

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self._zipfile.extractall(output_dir)
        logger.info(f"Extracted all files to '{output_dir}'")

    def print_summary(self):
        """logger summary of files and total compressed size."""

        total_size = sum(zinfo.file_size for zinfo in self._zipfile.infolist())
        logger.info(f"NDB Summary:")
        logger.info(f" - Total Files: {len(self._zipfile.namelist())}")
        logger.info(f" - Total Size: {total_size / 1024:.2f} KB")
        logger.info(f" - Path: {self._ndb_path}")

    # ------------------------------
    # Save changes back
    # ------------------------------
    def save(self):
        """Save all changes to the encrypted NDB file."""

        if not self._ndb_path:
            raise ValueError("No NDB path specified to save.")

        self._zipfile.close()
        self._zip_bytes_io.seek(0)  # pyright: ignore[reportOptionalMemberAccess]
        zip_bytes = self._zip_bytes_io.read()  # pyright: ignore[reportOptionalMemberAccess]

        ndb_content = self.crypto_manager.encrypt_data(zip_bytes)

        with open(self._ndb_path, 'w', encoding=AuthenticationConfig.ENCODING) as f:
            json.dump(ndb_content, f, ensure_ascii=False)



    # ------------------------------
    # Context manager
    # ------------------------------
    def close(self):
        """Close the internal zipfile safely."""

        if hasattr(self, '_zipfile'):
            self._zipfile.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
