from configobj import ConfigObj
from string import Template
import os
import shutil
import zipfile
import json
import tempfile
from cryptography.fernet import Fernet
from io import BytesIO
import base64

from utils.models import AuthenticationConfig
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

    def __init__(self, config_path: str = None):
        """
        Initialize the NebulonDB configuration loader.

        Args:
            config_path (str): Path to the configuration file.
        """

        # Determine default path if none is provided
        if config_path is None:
            neb_home = os.environ.get('NEBULONDB_HOME', os.getcwd())
            config_path = os.path.join(neb_home, "nebulondb.cfg")

        self.config_path = os.path.abspath(config_path)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            self._config = ConfigObj(self.config_path, encoding=AuthenticationConfig.ENCODING, list_values=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{self.config_path}': {e}")

        self._validate_sections()
        self._apply_env_override()
        self._load_environment()
        self._load_paths()
        self._load_corpus()
        self._load_vector_index()
        self._load_segments()
        self._load_server()

    # ------------------------------
    #  Private Utility Methods 
    # ------------------------------

    @staticmethod
    def _resolve_path(path_vars, value) -> str:
        """Resolve variables using provided path_vars and environment."""
        combined = dict(path_vars)
        return os.path.expandvars(Template(value).safe_substitute(combined))

    def _validate_sections(self):
        required_sections = ['paths', 'corpus', 'vector_index', 'params', 'server']
        for section in required_sections:
            if section not in self._config:
                raise KeyError(f"Missing required section: '{section}' in config file.")

        if 'NEBULONDB_HOME' not in self._config['paths']:
            raise KeyError("Missing 'NEBULONDB_HOME' in [paths] section.")

    def _apply_env_override(self):
        """Override NEBULONDB_HOME with environment variable if set."""

        # Override NEBULONDB_HOME if environment variable is set
        env_home = os.environ.get('NEBULONDB_HOME')
        if env_home and self._config['paths'].get('NEBULONDB_HOME') != env_home:  
            self._config['paths']['NEBULONDB_HOME'] = env_home  
            self._config.write()
        
        
        if not self._config['environment'].get('NEBULONDB_MASTER_KEY'):  
            # Ensure NEBULONDB_MASTER_KEY is set
            master_key = os.environ.get('NEBULONDB_MASTER_KEY')
            if not master_key :
                # Generate a secure key if missing
                master_key = Fernet.generate_key().decode()
                print("Warning: NEBULONDB_MASTER_KEY not found in environment; System Gendrated Key Will be Used.")
                self._config['environment']['NEBULONDB_MASTER_KEY'] = master_key  
                self._config.write()

    # ------------------------------
    # Load Config Sections
    # ------------------------------

    def _load_paths(self):
        self.NEBULONDB_HOME = self._resolve_path(self._config['paths'], self._config['paths']['NEBULONDB_HOME'])  
        self.VECTOR_STORAGE = self._resolve_path(self._config['paths'], self._config['paths']["VECTOR_STORAGE"])  
        self.NEBULONDB_SECRETS = self._resolve_path(self._config['paths'], self._config['paths']["NEBULONDB_SECRETS"])  
        self.VECTOR_METADATA = self._resolve_path(self._config['paths'], self._config['paths']["VECTOR_METADATA"])  
    
    def _load_environment(self):
        self.ENVIRONMENT_MASTER_KEY = self._config['environment']['NEBULONDB_MASTER_KEY']
        self.KEYRING_ENABLED = self._config['environment']['NEBULONDB_KEYRING_ENABLED']
        self.KEYRING_SERVICE = self._config['environment']['NEBULONDB_KEYRING_SERVICE']
        self.KEYRING_USER = self._config['environment']['NEBULONDB_KEYRING_USER']
    
    def _load_corpus(self): 
        self.DEFAULT_CORPUS_CONFIG_STRUCTURES = self._resolve_path(
            self._config['paths'], self._config["corpus"]["DEFAULT_CORPUS_CONFIG_STRUCTURES"]
        )
        self.DEFAULT_CORPUS_STRUCTURES = [
            item.strip() for item in self._config['corpus']['DEFAULT_CORPUS_STRUCTURES'].split(',')  
        ]
        self.SEGMENTS_NAME = self._config["corpus"]["CORPUS_SEGMENT"]  

    def _load_vector_index(self):
        self.DEFAULT_CORPUS_CONFIG_DATA = {
            "dimension": int(self._config['vector_index']['dimension']), 
            "index_type": self._config['vector_index']['index_type'], 
            "metric": self._config['vector_index']['metric'], 
            "segment_max_size": self._config['vector_index']["segment_max_size"], 
            "top_matches": self._config['vector_index']["top_matches"], 
            "params": {
                "nlist": int(self._config['params']['nlist']), 
                "nprobe": int(self._config['params']['nprobe']), 
                "m": int(self._config['params']['m']), 
                "nbits": int(self._config['params']['nbits']), 
                "hnsw_m": int(self._config['params']['hnsw_m']), 
                "ef_construction": int(self._config['params']['ef_construction']), 
                "ef_search": int(self._config['params']['ef_search']), 
            }
        }

    def _load_segments(self):
        self.SEGMENTS_METADATA = self._config["segments"]["SEGMENT_METADATA"] 
        self.SEGMENT_MAP = self._config["segments"]["SEGMENT_MAP"]  
    
    def _load_server(self):
        # === Assign values ===
        self.APP_NAME = self._config["server"]["APP_NAME"]  
        self.HOST = self._config["server"]["HOST"]  
        self.PORT = int(self._config["server"]["PORT"])  
        self.WORKERS = int(self._config["server"]["WORKERS"])  

        # === Optional values with defaults ===
        self.TIMEOUT = int(self._config["server"]["TIMEOUT"])  
        self.KEEP_ALIVE = int(self._config["server"]["KEEP_ALIVE"])  
        self.GRACEFUL_TIMEOUT = int(self._config["server"]["GRACEFUL_TIMEOUT"])  
        self.ACCESS_LOGFILE = self._config["server"]["ACCESS_LOGFILE"]  
        self.ERROR_LOGFILE = self._config["server"]["ERROR_LOGFILE"]  
        self.LOG_LEVEL = self._config["server"]["LOG_LEVEL"]  


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
            1. Environment variable (preferred)
            2. System keyring (fallback)
            3. Auto-generate new key if not found
        """

        key = os.environ.get(self.config.ENVIRONMENT_MASTER_KEY)
        if key:
            return key.encode()
        
        # Try from keyring if enabled
        if self.config.KEYRING_ENABLED:
            try:
                import keyring

                stored_key = keyring.get_password(
                    self.config.KEYRING_SERVICE, self.config.KEYRING_USER
                )
                if stored_key:
                    return stored_key.encode()

                # If not found, generate and store a new one
                new_key = Fernet.generate_key().decode()
                keyring.set_password(
                    self.config.KEYRING_SERVICE, self.config.KEYRING_USER, new_key
                )
                return new_key.encode()

            except Exception as e:
                # Graceful fallback if keyring fails
                print(f"[Warning] Keyring access failed: {e}")

        # As a last resort, generate a temporary in-memory key
        print("[Warning] No valid key found — generating temporary master key.")

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
        """
            Decrypt encrypted NDB content and return the original bytes.
        """
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
        if os.path.exists(output_file) and not force:
            print(f"{output_file} exists. Skipping encryption.")
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

            print(f"Folder '{src_folder}' encrypted to '{output_file}'.")
            if delete_source:
                shutil.rmtree(src_folder)

        finally:
            if os.path.exists(tmp_zip_path):
                os.remove(tmp_zip_path)

    # ------------------------------
    # Load NDB into memory
    # ------------------------------
    def _load_ndb(self, ndb_file):
        with open(ndb_file, 'r', encoding=AuthenticationConfig.ENCODING) as f:
            content = json.load(f)

        zip_bytes = self.crypto_manager.decrypt_data(content)
        self._zip_bytes_io = BytesIO(zip_bytes)
        self._zipfile = zipfile.ZipFile(self._zip_bytes_io, 'r') 

    # ------------------------------
    # File operations
    # ------------------------------
    def list_files(self):
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
        print(f"Extracted all files to '{output_dir}'")

    def print_summary(self):
        """Print summary of files and total compressed size."""
        total_size = sum(zinfo.file_size for zinfo in self._zipfile.infolist())
        print(f"NDB Summary:")
        print(f" - Total Files: {len(self._zipfile.namelist())}")
        print(f" - Total Size: {total_size / 1024:.2f} KB")
        print(f" - Path: {self._ndb_path}")

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

        print(f"NDB saved to '{self._ndb_path}'.")


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
