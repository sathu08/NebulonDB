from configobj import ConfigObj
from string import Template
import os

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
    SEGMENTS_NAME = _config["corpus"]["CORPUS_SEGMENT"]

    # === Vector Index Config ===
    DEFAULT_CORPUS_CONFIG_DATA = {
        "dimension": int(_config['vector_index']['dimension']),
        "index_type": _config['vector_index']['index_type'],
        "metric": _config['vector_index']['metric'],
        "segment_max_size":_config['vector_index']["segment_max_size"],
        "top_matches":_config['vector_index']["top_matches"],
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
    SEGMENTS_METADATA = _config["segments"]["SEGMENT_METADATA"]
    SEGMENT_MAP = _config["segments"]["SEGMENT_MAP"]
