from configobj import ConfigObj
from string import Template
import os
import stat


class NebulonDBConfig:
    """
    NebulonDB Configuration Loader

    Loads configuration from a specified config file (default: `nebulondb.cfg`),
    supports environment overrides, safely resolves variables using
    string.Template and os.path.expandvars, and enforces secure permissions
    on sensitive directories.
    """

    def __init__(self, config_path: str = "nebulondb.cfg"):
        """
        Initialize the NebulonDB configuration loader.

        Args:
            config_path (str): Path to the configuration file.
        """
        self.config_path = os.path.abspath(config_path)
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        try:
            self._config = ConfigObj(self.config_path, encoding='utf-8', list_values=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load config file '{self.config_path}': {e}")

        self._validate_sections()
        self._apply_env_override()
        self._load_paths()
        self._load_corpus()
        self._load_vector_index()
        self._load_segments()
        self._load_server()
        self._secure_directories(
            self.NEBULONDB_HOME,
            self.VECTOR_STORAGE,
            self.VECTOR_METADATA
        )

    # ------------------- Private Utility Methods -------------------

    @staticmethod
    def _resolve_path(path_vars: dict, value: str) -> str:
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
        if 'NEBULONDB_HOME' in os.environ:
            env_home = os.environ['NEBULONDB_HOME']
            if self._config['paths']['NEBULONDB_HOME'] != env_home:
                self._config['paths']['NEBULONDB_HOME'] = env_home
                self._config.write()

    # ------------------- Load Config Sections -------------------

    def _load_paths(self):
        self.NEBULONDB_HOME = self._resolve_path(self._config['paths'], self._config['paths']['NEBULONDB_HOME'])
        self.VECTOR_STORAGE = self._resolve_path(self._config['paths'], self._config['paths']["VECTOR_STORAGE"])
        self.USER_CREDENTIALS = self._resolve_path(self._config['paths'], self._config['paths']["USER_CREDENTIALS"])
        self.VECTOR_METADATA = self._resolve_path(self._config['paths'], self._config['paths']["VECTOR_METADATA"])

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

    # ------------------- Security & Logging -------------------

    @staticmethod
    def _secure_directories(*dirs):
        """
        Secure sensitive directories (700 permissions).
        """
        for d in dirs:
            if os.path.exists(d):
                os.chmod(d, stat.S_IRWXU)
                print(f"🔒 Secured directory: {d}")
            else:
                os.makedirs(d, mode=0o700, exist_ok=True)
                print(f"📁 Created and secured directory: {d}")
