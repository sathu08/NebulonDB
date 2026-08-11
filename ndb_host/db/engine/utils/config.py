"""
DatabaseConfig – manages filesystem paths, schema structures, and tuning
parameters for both the core key‑value engine and the vector index store.
"""

from pathlib import Path
from typing import Optional, Union

from dataclasses import dataclass, field

from db.ndb_settings import NDBConfig
from .constants import (
    HEADER_FORMAT, RECORD_HEADER_FORMAT, ENTRY_FORMAT, MAGIC, VERSION,
    HEADER_SIZE, RECORD_HEADER_SIZE, ENTRY_SIZE,
)

config_settings = NDBConfig()

@dataclass
class DatabaseConfig:
    """
    Central configuration object. All paths are computed from `db_dir`.
    Boolean flag `is_vector` adds additional vector‑related paths.
    """
    db_dir: Union[str, Path]
    is_vector: bool = False
    is_graph:  bool = False

    # ---------- Core Database Constants ----------
    MAGIC: bytes = MAGIC
    VERSION: int = VERSION
    BLOOM_FILTER_BITS_PER_KEY: int = 10
    BLOOM_FILTER_HASH_COUNT: int = 4
    HEADER_FORMAT: str = HEADER_FORMAT
    RECORD_HEADER_FORMAT: str = RECORD_HEADER_FORMAT
    ENTRY_FORMAT: str = ENTRY_FORMAT
    HEADER_SIZE: int = HEADER_SIZE
    RECORD_HEADER_SIZE: int = RECORD_HEADER_SIZE
    ENTRY_SIZE: int = ENTRY_SIZE
    FLUSH_SIZE_THRESHOLD: int = 16 * 1024 * 1024

    # ---------- Settings from NDBConfig ----------
    WAL_AUTO_FLUSH: bool = config_settings.WAL_AUTO_FLUSH
    WAL_FSYNC_INTERVAL: int = config_settings.WAL_FSYNC_INTERVAL
    FLUSH_RECORD_THRESHOLD: int = config_settings.FLUSH_RECORD_THRESHOLD
    COMPRESS_SEGMENTS: bool = config_settings.COMPRESS_SEGMENTS
    BLOOM_FILTER_ENABLED: bool = config_settings.BLOOM_FILTER_ENABLED
    MAX_OPEN_SEGMENTS: int = config_settings.MAX_OPEN_SEGMENTS
    COMPACTION_INTERVAL: int = config_settings.COMPACTION_INTERVAL
    MAX_SEGMENTS_BEFORE_COMPACT: int = config_settings.MAX_SEGMENTS_BEFORE_COMPACT
    FLUSH_INTERVAL: float = config_settings.FLUSH_INTERVAL

    # ---------- Vector & Graph Settings ----------
    NOVA_SEGMENT_NAME: str = "nebulon_nova"
    DOCUMENTS_SEGMENT_NAME: str = "nebulon_documents"
    MESH_SEGMENT_NAME: str = "nebulon_mesh"
    MESH_NODE_SEGMENT_NAME: str = "nebulon_mesh_nodes"
    MESH_EDGE_SEGMENT_NAME: str = "nebulon_mesh_edges"
    VECTOR_DIM: int = config_settings.DEFAULT_CORPUS_CONFIG_DATA["dimension"]
    VECTOR_SPACE: str = config_settings.DEFAULT_CORPUS_CONFIG_DATA["space"]
    VECTOR_M: int = config_settings.DEFAULT_CORPUS_CONFIG_DATA["m"]
    VECTOR_EF_CONSTRUCTION: int = config_settings.DEFAULT_CORPUS_CONFIG_DATA["ef_construction"]
    VECTOR_EF_SEARCH: int = config_settings.DEFAULT_CORPUS_CONFIG_DATA["ef_search"]
    TOP_MATCHES: int = config_settings.DEFAULT_CORPUS_CONFIG_DATA["top_matches"]
    COMPACTION_DELETED_RATIO: float = 0.4
    # ---------- Ranking Settings ----------
    WEIGHT: dict = field(default_factory=lambda: dict(config_settings.RANK_WEIGHTS))
    RANK_TOPK: int = config_settings.RANK_TOPK
    # ---------- Computed Paths ----------
    DB_DIR: Path = field(init=False)
    NEBULON_COSMOS_DIR: Path = field(init=False)
    SEG_DIR: Path = field(init=False)
    WAL_FILE: Path = field(init=False)
    INDEX_FILE: Path = field(init=False)
    META_FILE: Path = field(init=False)
    MANIFEST_FILE: Path = field(init=False)

    NEBULON_NOVA_DIR: Optional[Path] = field(init=False, default=None)
    NOVA_CONFIG_JSON: Optional[Path] = field(init=False, default=None)
    NOVA_MANIFEST_FILE_JSON: Optional[Path] = field(init=False, default=None)
    NOVA_WAL: Optional[Path] = field(init=False, default=None)

    def __post_init__(self) -> None:
        self.DB_DIR = Path(self.db_dir)
        self.NEBULON_COSMOS_DIR = self.DB_DIR / "NebulonCosmos"
        self.SEG_DIR = self.NEBULON_COSMOS_DIR / "segments"
        self.WAL_FILE = self.NEBULON_COSMOS_DIR / "wal.log"
        self.INDEX_FILE = self.NEBULON_COSMOS_DIR / "index.idx"
        self.META_FILE = self.NEBULON_COSMOS_DIR / "meta.bin"
        self.MANIFEST_FILE = self.NEBULON_COSMOS_DIR / "manifest.bin"

        self.NEBULON_ORBIT_DIR = self.DB_DIR / "NebulonOrbit"

        if self.is_vector:
            self.NEBULON_NOVA_DIR = self.NEBULON_ORBIT_DIR / "NebulonNova"
            self.NOVA_CONFIG_JSON = self.NEBULON_NOVA_DIR / "config.json"
            self.NOVA_MANIFEST_FILE_JSON = self.NEBULON_NOVA_DIR / "manifest.json"
            self.NOVA_WAL = self.NEBULON_NOVA_DIR / "nova.wal"

        if self.is_graph:
            self.NEBULON_MESH_DIR = self.NEBULON_ORBIT_DIR / "NebulonMesh"
            self.MESH_GRAPH_VIZ_HTML = self.NEBULON_MESH_DIR / "mesh_graph_visualization.html"