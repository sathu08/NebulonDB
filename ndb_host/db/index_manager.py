"""
NDB Index Manager
==========================================================

This module handles index management for the NDB API.
It provides endpoints for index creation, listing, and deletion.

"""

import shutil
import numpy as np
import polars as pl

from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional

from db.ndb_settings import NDBConfig
from core.model_hub import SemanticEmbeddingModel
from utils.constants import ColumnPick, MetadataRetention, NDBMeta

from db.engine import NebulonCosmos, NebulonOrbit, RankConfig

from utils.time_utils import utc_now_iso

# ==========================================================
#        Load Configuration
# ==========================================================

config_settings = NDBConfig()

class ComosDBManager:
    _instances: Dict[str, "ComosDBManager"] = {}

    def __new__(cls, db_path: Path, reset: bool = False):
        key = str(Path(db_path).resolve())
        if reset:
            cls._instances.pop(key, None)
        if key not in cls._instances:
            cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(self, db_path: Path, reset: bool = False):
        _ = str(Path(db_path).resolve())
        if getattr(self, "_initialized", False) and not reset:
            return
        self._reset_db = reset
        self._db = NebulonCosmos(
            db_dir=db_path,
            reset=self._reset_db
        )
        self._initialized = True

    def read_data(self, segment: str, include_internal:bool = False,
                  limit: Optional[int] = None) -> List[Dict[str, Any]]:
        records = self._db.read_all(segment=segment, include_internal=include_internal)
        if limit is not None and limit >= 0:
            return records[:limit]
        return records

    def insert_data(self, segment:str, document: Dict[str, Any]) -> int:
        return self._db.insert(segment, document)

    def delete_data(self, segment:str, record_id: Any) -> int:
        return self._db.delete(segment, record_id)

    def update_data(self, segment:str, document: Dict[str, Any]):
        self._db.update(segment, document)

    def flush(self):
        self._db.flush()
    
    def close(self):
        self._db.close()

class OrbitDBManager:
    def __init__(self, db_path: Path, segment_name: str = "default", reset: bool = False,
                rank_config: Optional[RankConfig] = None): 
        self._db = NebulonOrbit(
            db_dir=db_path,
            segment_name=segment_name,
            reset=reset,
            rank_config=rank_config,
        )

    def initialize_or_flush(self):
        self._db.flush()

    def insert_vec(self, vector: Optional[np.ndarray] = None, text: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[str]]:
        metadata = dict(metadata or {})
        if text is not None and "text" not in metadata:
            metadata["text"] = text
        record_id, err = self._db.insert(vector=vector, metadata=metadata)
        return record_id, err

    def search_vec(self, vector: np.ndarray, filter: Dict, top_k: int, mode="auto",
                   query: Optional[str] = None, rank: bool = False,
                   graph_start_node: Optional[int] = None,
                   expand_depth: int = 1, graph_boost: float = 0.1) -> List[Dict]:
        vector = vector.tolist() if isinstance(vector, np.ndarray) else vector
        mode = mode or "auto"
        expand_depth = expand_depth or 1
        graph_boost = graph_boost if graph_boost is not None else 0.1
        results = self._db.search(
            vector=vector,
            top_k=top_k,
            mode=mode,
            query=query,
            rank=rank,
            graph_start_node=graph_start_node,
            expand_depth=expand_depth,
            graph_boost=graph_boost,
        )

        if filter:
            filtered = []
            for r in results:
                meta = r.get("metadata", {}) or {}
                if all(meta.get(k) == v for k, v in filter.items()):
                    filtered.append(r)
            results = filtered

        return results

    def close(self):
        self._db.close()

    def add_relation(self, source, target, relation: str, weight: Optional[float] = None) -> None:
        """Add a directed relationship between two nodes in the graph."""
        self._db.add_relation(source, target, relation, weight=weight)

    def load_graph(self, nodes: Optional[List[Dict]] = None,
                   edges: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """Bulk graph load (nodes + edges)."""
        return self._db.load_graph(nodes=nodes, edges=edges)

    def remove_relation(self, source: int, target: int, relation: Optional[str] = None) -> None:
        """Remove a relationship between two nodes in the graph."""
        self._db.remove_relation(source, target, relation)

    def get_visualization_html(self) ->  Tuple[Optional[str], Optional[Path]]:
        """Return an HTML string for visualizing the graph."""
        return self._db.get_visualization_html()

    def rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Re-rank existing candidate results using the Orbit ranking engine."""
        return self._db.rerank(
            query=query,
            candidates=candidates,
            top_k=top_k,
        )

    # ------------------------------------------------------------------ #
    # Record / vector inspection                                         #
    # ------------------------------------------------------------------ #
    def count(self) -> int:
        """Number of vectors currently stored."""
        return self._db.count()

    def exists(self, record_id: int) -> bool:
        """Return True if a record with this ID exists."""
        return self._db.exists(record_id)

    def get_record(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Return the full record (id, vector, metadata) or None."""
        return self._db.get(record_id)

    def get_metadata(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Return only the metadata of a record, or None."""
        return self._db.get_metadata(record_id)

    def get_vector(self, record_id: int) -> Optional[List[float]]:
        """Return only the vector of a record, or None."""
        return self._db.get_vector(record_id)

    def update_vec(self, record_id: int, vector: Optional[np.ndarray] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> Tuple[int, Optional[str]]:
        """Update an existing record's vector and/or metadata."""
        if vector is not None and isinstance(vector, np.ndarray):
            vector = vector.tolist()
        return self._db.update(record_id, vector=vector, metadata=metadata)

    def delete_record(self, record_id: int) -> bool:
        """Delete a record. Returns True if it existed and was removed."""
        existed = self._db.exists(record_id)
        if existed:
            self._db.delete(record_id)
        return existed

    def list_ids(self) -> List[int]:
        """Return all record IDs currently stored."""
        return self._db.list_ids()

    def get_all_records(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Return all stored vector records."""
        records = self._db.get_all()
        if limit is not None and limit >= 0:
            return records[:limit]
        return records

    def stats(self) -> Dict[str, Any]:
        """Return a summary snapshot of vector + graph state."""
        return self._db.stats()

    # ------------------------------------------------------------------ #
    # Graph node / edge inspection                                       #
    # ------------------------------------------------------------------ #
    def add_node(self, node_id: int, label: Optional[str] = None) -> None:
        """Create a graph node explicitly (no vector required)."""
        self._db.add_node(node_id, label)

    def remove_node(self, node_id: int) -> None:
        """Remove a graph node and all edges connected to it."""
        self._db.remove_node(node_id)

    def get_node(self, node_id: int) -> Optional[Dict[str, Any]]:
        """Return the graph node's metadata, or None if it does not exist."""
        return self._db.get_node(node_id)

    def has_node(self, node_id: int) -> bool:
        """Return True if the node exists in the graph."""
        return self._db.has_node(node_id)

    def count_nodes(self) -> int:
        """Number of nodes currently in the graph."""
        return self._db.count_nodes()

    def count_edges(self) -> int:
        """Number of directed edges currently in the graph."""
        return self._db.count_edges()

    def has_edges(self) -> bool:
        """Return True if the graph contains at least one edge."""
        return self._db.has_edges()

    def get_edges(self) -> List[Tuple[int, int, str]]:
        """Return all edges as (source, target, relation) tuples."""
        return self._db.get_edges()

    def get_all_nodes(self) -> List[Dict[str, Any]]:
        """Return every graph node as {"id": ..., "metadata": ...}."""
        return self._db.get_all_nodes()

    def edges_by_relation(self, relation: str) -> List[Tuple[int, int, str]]:
        """Return all edges that carry the given relation label."""
        return self._db.edges_by_relation(relation)

    def get_neighbors(self, node_id: int, direction: str = "both") -> List[Tuple[int, str]]:
        """Return neighbours as (neighbor_id, relation) tuples."""
        return self._db.get_neighbors(node_id, direction)

    def get_in_neighbors(self, node_id: int) -> List[Tuple[int, str]]:
        """Neighbours pointing at node_id (incoming edges)."""
        return self._db.get_in_neighbors(node_id)

    def get_out_neighbors(self, node_id: int) -> List[Tuple[int, str]]:
        """Neighbours node_id points at (outgoing edges)."""
        return self._db.get_out_neighbors(node_id)

    def bfs(self, start: int, max_depth: int = 3) -> List[int]:
        """Breadth-first traversal from a start node."""
        return self._db.bfs(start, max_depth)

    def dfs(self, start: int, max_depth: int = 3) -> List[int]:
        """Depth-first traversal from a start node."""
        return self._db.dfs(start, max_depth)

    def shortest_path(self, source: int, target: int) -> Optional[List[int]]:
        """Shortest unweighted path between two nodes, or None."""
        return self._db.shortest_path(source, target)

    def connected_components(self) -> List[Any]:
        """List of connected components (sets of node IDs)."""
        return self._db.connected_components()

# ==========================================================
#        CorpusManager
# ==========================================================

class CorpusManager:
    """CorpusManager handles validation and retrieval of corpus data and metadata."""

    _shared_metadata_db = None

    def __init__(self):
        """Initialize CorpusManager."""
        self.storage_path = config_settings.NEBULONDB_STORAGE_PATH
        self.corpus_metadata_path = config_settings.NEBULONDB_ACCOUNTHUB_CORPUS_PATH
        self.metadata_segment = NDBMeta.Corpus.METADATA_SEGMENT_NAME
        self._validate_paths()

    @property
    def metadata_db(self):
        if CorpusManager._shared_metadata_db is None:
            CorpusManager._shared_metadata_db = ComosDBManager(db_path=self.corpus_metadata_path)
        return CorpusManager._shared_metadata_db
    
    def _validate_paths(self) -> None:
        """Check that essential paths exist."""
        errors = []

        if not self.storage_path.exists() or not self.storage_path.is_dir():
            errors.append(f"Vector storage path missing: {self.storage_path}")
            
        if errors:
            raise FileNotFoundError(" | ".join(errors))
    
    @staticmethod
    def generate_corpus_metadata(corpus_name: str, created_by: str, status:str, ndb_type:NDBMeta.Type.COSMOS) -> Dict[str, str]:
        """
        Generate metadata dictionary for a new corpus.

        Args:
            corpus_name (str): Name of the corpus.
            created_by (str): User who created the corpus.
            status (str): Status of the corpus (e.g., 'active', 'deactivate', 'system').
        Returns:
            Dict[str, str]: Metadata entry.
        """
        return {
            "corpus_name": corpus_name,
            "created_at": utc_now_iso(),
            "created_by": created_by,
            "ndb_type":ndb_type,
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
            vector_dirs = [d.name for d in self.storage_path.iterdir() if d.is_dir()]
            if not vector_dirs:
                return []
            metadata_names = {record["corpus_name"] for record in self.metadata_db.read_data(segment=self.metadata_segment) if record.get("corpus_name")}
            if not metadata_names:
                return []
            return sorted(set(vector_dirs) & set(metadata_names))
        except Exception:
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
            for record in self.metadata_db.read_data(segment=self.metadata_segment):
                if record.get("corpus_name") == corpus_name:
                    return record.get("status")
            return None
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
            for record in self.metadata_db.read_data(segment=self.metadata_segment, include_internal=True,):
                if record.get("corpus_name") == corpus_name:
                    record["status"] = status
                    self.metadata_db.update_data(segment=self.metadata_segment, document=record)
        except Exception as _:
            return False
        
    def create_corpus(self, corpus_name: str, username:str, ndb_type: NDBMeta.Type=NDBMeta.Type.COSMOS, status:str="active") -> None:
        """
        Create a new corpus.

        Args:
            corpus_name (str): Name of the corpus to create.
            username (str): Name of the user creating the corpus.
            ndb_type (NDBMeta.Type): Type of NDB backend to use (default: COSMOS).
            status (str): Status of the corpus (default: 'active').
        """

        corpus_path = self.storage_path / corpus_name

        if ndb_type == NDBMeta.Type.ORBIT:
            # Corpus creation only registers the corpus + creates the directory.
            # The ORBIT storage tree is built lazily on the first segment load,
            # so no default (=spurious) segment directory is created here.
            corpus_path.mkdir(parents=True, exist_ok=True)
        else:
            ComosDBManager(corpus_path)

        metadata = self.generate_corpus_metadata(corpus_name=corpus_name, created_by=username, status=status, ndb_type=ndb_type)
        self.metadata_db.insert_data(segment=self.metadata_segment, document=metadata)

    def delete_corpus(self, corpus_name: str):
        """
        Delete an existing corpus.

        Args:
            corpus_name (str): Name of the corpus to Delete
        """
        corpus_path = self.storage_path / corpus_name
        if corpus_path.exists():
            shutil.rmtree(corpus_path)

        for record in self.metadata_db.read_data(segment=self.metadata_segment, include_internal=True):
            if record.get("corpus_name") == corpus_name:
                self.metadata_db.delete_data(segment=self.metadata_segment, record_id=record["_id"])

# ==========================================================
#        SegmentManager
# ==========================================================

class SegmentManager:
    """
    SegmentManager handles dynamic creation/loading of segments,
    along with vectors, payloads, and ID mapping."""
    
    def __init__(self, corpus_name: str, segment_name: str, ndb_type: NDBMeta.Type = NDBMeta.Type.ORBIT):
        """
        Initialize SegmentManager for a specific corpus.

        Args:
            corpus_name (str): Name of the corpus to manage.
            segment_name (str): Name of the segment to corpus.
        """
        self.corpus_name = corpus_name
        self.segment_name = segment_name
        self.corpus_metadata_path = config_settings.NEBULONDB_ACCOUNTHUB_CORPUS_PATH
        self.corpus_path = config_settings.NEBULONDB_STORAGE_PATH / self.corpus_name
        self.metadata_db = CorpusManager().metadata_db
        self.metadata_segment = CorpusManager().metadata_segment
        self._validate_checks()
        self.ndb_type = ndb_type
        if ndb_type == NDBMeta.Type.ORBIT:
            self.db_manager = OrbitDBManager(self.corpus_path, segment_name=self.segment_name)
        else:
            self.db_manager = ComosDBManager(self.corpus_path)
        self.embedding_model = SemanticEmbeddingModel()
        self._validate_paths()

    RELATION_SOURCE_COLS = ("source", "source_id", "src", "from_id", "from")
    RELATION_TARGET_COLS = ("target", "target_id", "dst", "to_id", "to")
    RELATION_LABEL_COLS = ("relation", "rel", "edge_type", "relationship", "label")

    def _validate_checks(self) -> None:
        """Perform validation checks on corpus metadata."""
        if not self._corpus_exists():
            raise FileNotFoundError(f"Corpus '{self.corpus_name}' not found in metadata.")

    def _corpus_exists(self) -> bool:
        """Return True if this corpus has a metadata record."""
        try:
            for record in self.metadata_db.read_data(segment=self.metadata_segment, include_internal=True):
                if record.get("corpus_name") == self.corpus_name:
                    return True
        except Exception as _:
            return False
        return False

    def _validate_paths(self) -> None:
        """Check that essential paths exist."""
        errors = []
        if not self.corpus_path.exists() or not self.corpus_path.is_dir():
            errors.append(f"Vector storage path missing: {self.corpus_path}")
        if errors:
            raise FileNotFoundError(" | ".join(errors))
        
    def _load_relations(
            self,
            segment_dataset: pl.DataFrame,
            relations: Optional[List[Tuple[int, int, str]]] = None,
            source_column: Optional[str] = None,
            target_column: Optional[str] = None,
            relation_column: Optional[str] = None,
        ) -> List[str]:
            """
            Upload graph relations from explicit tuples and/or DataFrame columns.

            Returns:
                Tuple[int, List[str]]: (number of relations added, error messages).
            """
            errors: List[str] = []
            total_added = 0

            # 1) Explicit (source, target, relation) tuples
            if relations:
                for item in relations:
                    try:
                        source, target, relation = item[0], item[1], item[2]
                    except (IndexError, TypeError):
                        errors.append(f"Invalid relation tuple: {item}")
                        continue
                    try:
                        self.db_manager.add_relation(int(source), int(target), str(relation))
                        total_added += 1
                    except Exception as e:
                        errors.append(f"Relation {source}->{target}: {e}")

            # 2) Relations from DataFrame columns (auto-detect when not specified)
            src_col = source_column or next((c for c in self.RELATION_SOURCE_COLS if c in segment_dataset.columns), None)
            tgt_col = target_column or next((c for c in self.RELATION_TARGET_COLS if c in segment_dataset.columns), None)
            rel_col = relation_column or next((c for c in self.RELATION_LABEL_COLS if c in segment_dataset.columns), None)

            if src_col and tgt_col:
                if src_col not in segment_dataset.columns or tgt_col not in segment_dataset.columns:
                    errors.append(f"Relation columns not found in dataset: '{src_col}', '{tgt_col}'")
                else:
                    default_label = "related"
                    for idx in range(segment_dataset.height):
                        try:
                            source = segment_dataset[src_col][idx]
                            target = segment_dataset[tgt_col][idx]
                            if source is None or target is None:
                                continue
                            label = segment_dataset[rel_col][idx] if rel_col and rel_col in segment_dataset.columns else default_label
                            weight = None
                            if "weight" in segment_dataset.columns:
                                weight = segment_dataset["weight"][idx]
                            self.db_manager.add_relation(source, target, str(label or default_label), weight=weight)
                            total_added += 1
                        except Exception as e:
                            errors.append(f"Relation row {idx}: {e}")

            return total_added, errors


    def get_segment_list(self) -> List[str]:
        """
        Get the list of segments registered for this corpus.

        Returns:
            List[str]: Segment names recorded in the corpus metadata.
        """
        try:
            for record in self.metadata_db.read_data(segment=self.metadata_segment, include_internal=True):
                if record.get("corpus_name") == self.corpus_name:
                    segments = record.get("segments") or []
                    return [s.get("name") for s in segments if isinstance(s, dict) and s.get("name")]
        except Exception as _:
            return []
        return []

    def get_segment_metadata(self) -> List[Dict[str, Any]]:
        """
        Get full segment metadata (name, inserted count, created_at) for this corpus.

        Returns:
            List[Dict[str, Any]]: Segment entries from the corpus metadata.
        """
        try:
            for record in self.metadata_db.read_data(segment=self.metadata_segment, include_internal=True):
                if record.get("corpus_name") == self.corpus_name:
                    return [s for s in (record.get("segments") or []) if isinstance(s, dict)]
        except Exception as _:
            return []
        return []
       
    @staticmethod
    def _determine_column_mode(set_columns) -> tuple[str, list]:
        """
        Determine column selection mode and normalize set_columns.

        Args:
            set_columns (str | list | None): User's column selection

        Returns:
            (mode, columns): A tuple where mode is one of ColumnPick.FIRST_COLUMN, 
                            ColumnPick.ALL, "LIST", or None; and columns is a list of column names (if applicable).
        """
        mode = None
        
        if isinstance(set_columns, str):
            val = set_columns.strip().lower()
            if val in ("first column", "first"):
                mode = ColumnPick.FIRST_COLUMN
            elif val in ("all", "all columns"):
                mode = ColumnPick.ALL
            else:
                mode = "LIST"
            return mode, [set_columns]

        elif isinstance(set_columns, list):
            if len(set_columns) == 1 and str(set_columns[0]).strip().lower() in ("first column", "first", "all"):
                val = str(set_columns[0]).strip().lower()
                if val in ("first column", "first"):
                    mode = ColumnPick.FIRST_COLUMN
                elif val in ("all", "all columns"):
                    mode = ColumnPick.ALL
            else:
                mode = "LIST"
            return mode, set_columns

        return mode, []

    def register_segment(self, corpus_name: str, segment_name: str, inserted: int = 0, created_at: Optional[str] = None) -> bool:
            """
            Record a segment in the corpus metadata.
    
            Adds a segment entry (only once per name) to the corpus record's
            ``segments`` list and persists it.
    
            Args:
                corpus_name: Name of the corpus.
                segment_name: Name of the segment to register.
                inserted: Number of records inserted during the segment load.
                created_at: ISO timestamp for the load (defaults to now).
    
            Returns:
                bool: True if the corpus record was found and updated.
            """
            if inserted <= 0:
                return False
            created_at = created_at
            try:
                for record in self.metadata_db.read_data(segment=self.metadata_segment, include_internal=True):
                    if record.get("corpus_name") == corpus_name:
                        segments = record.setdefault("segments", [])
                        names = {s.get("name") for s in segments if isinstance(s, dict)}
                        if segment_name not in names:
                            segments.append({
                                "name": segment_name,
                                "inserted": inserted,
                                "created_at": created_at,
                            })
                        else:
                            for s in segments:
                                if isinstance(s, dict) and s.get("name") == segment_name:
                                    s["inserted"] = s.get("inserted", 0) + inserted
                                    break
                        record["segments"] = segments
                        self.metadata_db.update_data(segment=self.metadata_segment, document=record)
                        return True
            except Exception as _:
                return False
            return False
      
    @staticmethod   
    def determine_columns_to_process(segment_dataset: pl.DataFrame, set_columns) -> dict:
        """
        Determine which columns to process based on set_column_vector parameter.
        
        Args:
            segment_dataset: Polars DataFrame
            set_column_vector: Column selection criteria
            
        Returns:
            dict: {
                "success": bool,
                "message": list of column names (if success=True),
                "message": str (if success=False)
            }
        """
        
        if isinstance(segment_dataset, dict):
            segment_dataset = pl.DataFrame(segment_dataset)
        elif not isinstance(segment_dataset, pl.DataFrame):
            return {"success": False, "message": "Input 'segment_dataset' must be convertible to a DataFrame"}
        
        # Check if dataset is empty
        if segment_dataset.height == 0:
            return {"success": False, "message": "Dataset is empty","columns": []}
        
        mode, set_columns = SegmentManager._determine_column_mode(set_columns)
        
        if mode == ColumnPick.FIRST_COLUMN:
            if not segment_dataset.columns:
                return {"success": False, "message": "Dataset has no columns", "columns": []}
            columns_to_process = [segment_dataset.columns[0]]
        elif mode == ColumnPick.ALL:
            columns_to_process = [col for col in segment_dataset.columns if segment_dataset[col].dtype == pl.Utf8]
            if not columns_to_process:
                return {"success": False, "message": "No text columns found in dataset","columns": []}
        elif mode == "LIST":
            missing_cols = [col for col in set_columns if col not in segment_dataset.columns]
            if missing_cols:
                return {"success": False, "message": f"Columns not found in dataset: {missing_cols}","columns": []}
            columns_to_process = [col for col in set_columns if col in segment_dataset.columns]
        else:
            return {"success": False, "message": "Invalid column selection mode","columns": []}
        
        if not columns_to_process:
            return {"success": False, "message": "No valid columns found to process","columns": []}
        
        return {"success": True, "message":"Selected Succfully", "columns": columns_to_process}
    
    def get_data(
        self,
        limit: Optional[int] = None,
        include_internal: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve stored records from a backend with an optional row limit.

        Args:
            limit (int, optional): Maximum number of records to return.
            include_internal (bool): Include Cosmos internal fields.

        Returns:
            List[Dict[str, Any]]: Retrieved records (empty if backend unavailable).
        """
        # ndb_type = (ndb_type or "orbit").lower()
        if self.ndb_type == NDBMeta.Type.COSMOS:
            if not isinstance(self.db_manager, ComosDBManager):
                return []
            return self.db_manager.read_data(
                self.segment_name,
                include_internal=include_internal,
                limit=limit,
            )
        if not isinstance(self.db_manager, OrbitDBManager):
            return []
        return self.db_manager.get_all_records(limit=limit)

    def load_segment(
        self, 
        segment_dataset: pl.DataFrame, 
        columns: list[str], 
        is_precomputed: bool = False, 
        lang_type: Optional[str] = None, 
        doc_type: Optional[str] = None,
        relations: Optional[List[Tuple[int, int, str]]] = None,
        source_column: Optional[str] = None,
        target_column: Optional[str] = None,
        relation_column: Optional[str] = None,
        lang: Optional[str] = None,
    ) -> dict:
        """
        Load vectors from one or more columns into OrbitDB, and optionally
        upload graph relations.

        Args:
            segment_dataset: Polars DataFrame.
            columns: Columns to process (text / embedding columns).
            is_precomputed: True if columns already contain embeddings.
            lang_type: Optional language tag stored in metadata.
            doc_type: Optional document type stored in metadata.
            relations: Optional explicit list of (source_id, target_id, relation)
                       tuples to add to the graph.
            lang: Backward-compatible alias for ``lang_type``.
            source_column: Optional column name containing relation source IDs.
                           Auto-detected from common names when omitted.
            target_column: Optional column name containing relation target IDs.
                           Auto-detected from common names when omitted.
            relation_column: Optional column name containing the relation label.
                             Defaults to "related" when omitted.

        Returns:
            dict containing success status and statistics.
        """

        total_inserted = 0
        total_skipped = 0
        total_relations = 0
        errors = []
        created_at = utc_now_iso()

        if lang is not None and lang_type is None:
            lang_type = lang

        is_orbit = self.ndb_type == NDBMeta.Type.ORBIT

        for col in columns:

            if col not in segment_dataset.columns:
                errors.append(f"Column '{col}' not found in dataset")
                continue

            try:
                if not is_orbit:
                    # COSMOS: store documents directly, no vectorisation.
                    texts = segment_dataset[col].fill_null("").to_list()
                    for idx, text in enumerate(texts):
                        if not text.strip():
                            total_skipped += 1
                            continue
                        document = {
                            "text": text,
                            "lang": lang_type,
                            "type": doc_type or "other",
                            "created_at": created_at,
                        }
                        document = MetadataRetention.apply(document)
                        try:
                            self.db_manager.insert_data(
                                segment=self.segment_name,
                                document=document,
                            )
                            total_inserted += 1
                        except Exception as e:
                            errors.append(f"Cosmos Row {idx} in {col}: {e}")
                    continue

                if is_precomputed:
                    vectors_list = segment_dataset[col].to_list()

                    if not vectors_list:
                        total_skipped += 1
                        continue

                    embeddings = np.asarray(vectors_list, dtype=np.float32)
                    texts = [""] * len(embeddings)

                else:
                    texts = segment_dataset[col].fill_null("").to_list()

                    if not any(text.strip() for text in texts):
                        errors.append(f"Column '{col}' has no valid text")
                        total_skipped += 1
                        continue

                    embeddings = self.embedding_model.encode(
                        texts,
                        convert_to_numpy=True,
                        normalize_embeddings=True,
                    ).astype(np.float32)

                for idx, (vec, text) in enumerate(zip(embeddings, texts)):
                    if not text.strip() and is_precomputed:
                        text = ""

                    metadata = {
                        "lang": lang_type,
                        "type": doc_type or "other",
                        "created_at": created_at,
                    }

                    name = None
                    if "name" in segment_dataset.columns:
                        raw_name = segment_dataset["name"][idx]
                        if raw_name is not None:
                            name = str(raw_name)
                    if name:
                        metadata["label"] = name

                    metadata = MetadataRetention.apply(metadata)

                    _, err = self.db_manager.insert_vec(
                        vector=vec.tolist(),
                        text=text,
                        metadata=metadata,
                    )
                    if err:
                        errors.append(f"Row {idx} in {col}: {err}")
                    else:
                        total_inserted += 1

            except Exception as e:
                errors.append(f"{col}: {str(e)}")

        # ---- Graph relation upload (ORBIT only) ----
        if is_orbit:
            relations_added, relation_errors = self._load_relations(
                segment_dataset=segment_dataset,
                relations=relations,
                source_column=source_column,
                target_column=target_column,
                relation_column=relation_column,
            )
            total_relations = relations_added
            errors.extend(relation_errors)

        if is_orbit:
            self.db_manager.initialize_or_flush()

        if total_inserted > 0:
            self.register_segment(
                corpus_name=self.corpus_name,
                segment_name=self.segment_name,
                inserted=total_inserted,
                created_at=created_at,
            )

        return {
            "success": len(errors) == 0,
            "inserted": total_inserted,
            "skipped": total_skipped,
            "relations_added": total_relations,
            "errors": errors,
        }

    def load_graph(
        self,
        nodes: Optional[List[Dict]] = None,
        edges: Optional[List[Dict]] = None,
    ) -> Dict[str, Any]:
        """Bulk graph load into an ORBIT corpus (auto-weight, label resolution)."""
        if self.ndb_type != NDBMeta.Type.ORBIT:
            return {"success": False, "message": "load_graph requires an ORBIT corpus"}
        added = self.db_manager.load_graph(nodes=nodes, edges=edges)
        self.db_manager.initialize_or_flush()
        return {"success": True, **added}

    def get_graph(self) -> Dict[str, Any]:
        """Return the current node + edge snapshot for an ORBIT corpus."""
        if self.ndb_type != NDBMeta.Type.ORBIT:
            return {"nodes": [], "edges": []}
        return {
            "nodes": self.db_manager.get_all_nodes(),
            "edges": self.db_manager.get_edges(),
        }

    def search_vector(
        self, 
        search_item:str,
        top_k: Optional[int] = None,
        set_columns: Optional[List[str]] = None,
        min_score: Optional[float] = None,
        lang_type: Optional[str] = None,
        doc_type: Optional[str] = None,
        mode: Optional[str] = None,
        rank: Optional[bool] = False,
        graph_start_node: Optional[int] = None,
        expand_depth: Optional[int] = None,
        graph_boost: Optional[float] = None,
        lang: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for nearest neighbors of a vector across all segments in a namespace.

        Args:
            search_item (str): The text query.
            top_k (int): Number of top results to return.
            set_columns (list): Restrict returned metadata to these columns.
            min_score (float): Minimum normalised score threshold in [0.0, 1.0].
                               Scores are min-max normalised across the result set
                               so 0.5 means "top half by relative relevance".
            lang_type (str): Optional language filter applied at retrieval time.
            doc_type (str): Optional document-type filter applied at retrieval time.
            mode (str): Search mode ('auto', 'nova', 'hybrid', 'mesh'). Default 'auto'.
            rank (bool): When True, apply multi-signal ranking (vector + BM25 +
                         metadata + importance + freshness) instead of raw
                         retrieval order.
            graph_start_node (int, optional): Seed node for graph traversal.
                         Required for 'mesh' mode; expands neighbours around the
                         seed in 'hybrid' mode.
            expand_depth (int): Max BFS depth for graph expansion. Default 1.
            graph_boost (float): Score assigned to nodes discovered via graph
                         expansion in 'hybrid' mode. Default 0.1.

        Ranking behaviour (RRF fusion, re-ranking, weights, half-life) is
        controlled via the RankConfig passed to OrbitDBManager at construction.

        Returns:
            List[Dict]: Search results with id, normalised score, text, and metadata.
        """
        
        column_mode, set_columns = self._determine_column_mode(set_columns)

        if lang is not None and lang_type is None:
            lang_type = lang

        filter_ = {}
        if lang_type:
            filter_["lang"] = lang_type
        if doc_type:
            filter_["type"] = doc_type
        

        query_vec = self.embedding_model.encode(
                    search_item,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                ).astype("float32")
        
        if hasattr(query_vec, "flatten"):
            query_vec = query_vec.flatten()

        # Perform search using Orbit
        results = self.db_manager.search_vec(
            vector=query_vec.tolist(),
            filter=filter_,
            top_k=top_k or 10,
            mode=mode,
            query=search_item,
            rank=rank,
            graph_start_node=graph_start_node,
            expand_depth=expand_depth,
            graph_boost=graph_boost,
        )
        if results:
            raw_scores = [r.get("score", 0.0) for r in results]
            min_raw = min(raw_scores)
            max_raw = max(raw_scores)
            score_range = max_raw - min_raw

            for r in results:
                if score_range > 0:
                    r["score"] = float(round(float(r.get("score", 0.0) - min_raw) / float(score_range), 6))
                else:
                    # All scores identical → every result is equally relevant
                    r["score"] = 1.0

        # Apply min_score filter against the normalised [0, 1] scores
        if min_score is not None:
            results = [r for r in results if r.get("score", 0.0) >= min_score]

        # Limit metadata to set_columns if requested
        if set_columns and column_mode == "LIST":
            for r in results:
                meta = r.get("metadata", {})
                r["metadata"] = {k: meta.get(k) for k in set_columns if k in meta}

        return results
