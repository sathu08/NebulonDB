from datetime import datetime, timezone
from configobj import ConfigObj
import os
from typing import List, Dict, Tuple, Any,Optional
from string import Template
from pathlib import Path
import shutil
import faiss, os
import numpy as np
import polars as pl 
from sentence_transformers import SentenceTransformer

from utils.models import load_data, save_data
from utils.models import ColumnPick 

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
        except Exception as _:
            return False
        
    def create_corpus(self, corpus_name: str, username:str):
        """
        Create a new corpus.

        Args:
            corpus_name (str): Name of the corpus to create.
            username (str): Name of the user creating the corpus.
        """
        corpus_path = self.vector_storage_path / corpus_name
        os.makedirs(corpus_path, exist_ok=True)
        for corpus_subdir in NebulonDBConfig.DEFAULT_CORPUS_STRUCTURES:
            (corpus_path / corpus_subdir).mkdir(parents=True, exist_ok=True)
                
        corpus_config_path = corpus_path / Path(NebulonDBConfig.DEFAULT_CORPUS_CONFIG_STRUCTURES)
        config_data = NebulonDBConfig.DEFAULT_CORPUS_CONFIG_DATA
        save_data(save_data=config_data, path_loc=corpus_path / corpus_config_path)

        # Store the corpus details
        created_corpus = load_data(path_loc=self.metadata_path)
        created_corpus[corpus_name] = self.generate_corpus_metadata(
            corpus_name=corpus_name,
            created_by=username
        )
        save_data(save_data=created_corpus, path_loc=self.metadata_path)

    def delete_corpus(self, corpus_name: str,):
        """
        Delete an existing corpus.

        Args:
            corpus_name (str): Name of the corpus to Delete
        """
        corpus_path = self.vector_storage_path / corpus_name
        shutil.rmtree(corpus_path)

        corpus_info = load_data(path_loc=self.metadata_path)
        del corpus_info[corpus_name]
        save_data(path_loc=self.metadata_path, save_data=corpus_info)
      
class SegmentManager:
    """
    SegmentManager handles dynamic creation/loading of FAISS segments,
    along with vectors, payloads, and ID mapping.
    """
    model = SentenceTransformer("all-MiniLM-L6-v2")
    def __init__(self, corpus_name:str, model=model):
        """
        Initialize SegmentManager for a specific corpus.

        Args:
            corpus_name (str): Name of the corpus to manage.
            model (_type_, optional): Model to use for embeddings. Defaults to model.
        """
        self.model = model
        self.corpus_path = Path(NebulonDBConfig.VECTOR_STORAGE) / corpus_name
        self.segment_path = self.corpus_path / NebulonDBConfig.SEGMENTS_NAME
        self.segment_metadata_path = self.corpus_path / NebulonDBConfig.SEGMENTS_METADATA
        self.segment_map_path = self.corpus_path / NebulonDBConfig.SEGMENT_MAP
        self.corpus_config = self.corpus_path / NebulonDBConfig.DEFAULT_CORPUS_CONFIG_STRUCTURES
        self.config = self._load_config()
        self._validate_paths()
        
    def _validate_paths(self) -> None:
        """Check that essential paths exist."""
        errors = []
        if not self.corpus_path.exists() or not self.corpus_path.is_dir():
            errors.append(f"Vector storage path missing: {self.corpus_path}")
        if not self.segment_metadata_path.exists():
            errors.append(f"Metadata file not found: {self.segment_metadata_path}")
        try:
            _ = self.model
        except Exception as e:
            errors.append(f"Model loading failed: {str(e)}")
        if errors:
            raise FileNotFoundError(" | ".join(errors))

    def _load_config(self) -> Optional[Dict]:
        """Load corpus configuration."""
        try:
            return load_data(self.corpus_config)
        except Exception as _:
            return None
    
    def _get_next_segment_id(self) -> str:
        """Get the next available segment name."""
        metadata = load_data(self.segment_metadata_path)
        if not metadata:
            return "segment_0"
        try:
            last_segment = max([int(v["segment"].split("_")[1]) for v in metadata.values()])
            return f"segment_{last_segment + 1}"
        except Exception:
            return "segment_0"
    
    def _get_latest_segment_id(self) -> Optional[Tuple[Path, str]]:
        """Return the latest existing segment if available."""
        segments = sorted(self.segment_path.glob("segment_*"))
        if not segments:
            return None
        seg_path = segments[-1]
        return seg_path, seg_path.name
    
    def _load_index(self, seg_path: Path) -> faiss.Index:
        """
        Load or create FAISS index for a segment.
        
        Args:
            seg_path (Path): Path to segment directory
            
        Returns:
            faiss.Index: FAISS index object
        """
        index_path = seg_path / "index.faiss"
        
        if index_path.exists():
            return faiss.read_index(str(index_path))
        
        dim = self.config["dimension"]
        if self.config["index_type"] == "hnsw":
            index = faiss.IndexHNSWFlat(dim, self.config["params"]["hnsw_m"])
            index.hnsw.efConstruction = self.config["params"]["ef_construction"]
            index.hnsw.efSearch = self.config["params"]["ef_search"]
        else:
            index = faiss.IndexFlatL2(dim)
        return index
    
    def _ensure_namespace(self, segment_name: Optional[str] = None) -> Tuple[Path, str]:
        """
        Ensure a segment namespace exists, create if needed.
        
        Args:
            segment_name (Optional[str]): Name of the segment
            
        Returns:
            Tuple[Path, str]: Path to segment and segment ID
        """
        latest = self._get_latest_segment_id()
        segment_map = load_data(self.segment_map_path)

        if segment_name and segment_name in segment_map and latest:
            seg_path, seg_name = latest
            index = self._load_index(seg_path)
            max_size = self.config["segment_max_size"]
            if index.ntotal < max_size:
                return seg_path, seg_name

        segment_id = self._get_next_segment_id()
        ns_path = self.segment_path / segment_id
        ns_path.mkdir(parents=True, exist_ok=True)
        return ns_path, segment_id
    
    def _check_duplicate_entry(self, key: str, vector: np.ndarray, payload: Dict[str, Any]) -> bool: 
        """
        Check if an entry with the same key already exists and if the vector/payload are identical.
        
        Args:
            key (str): Unique external ID
            vector (np.ndarray): Vector embedding
            payload (Dict): Metadata payload
        
        Returns:
            bool: True if duplicate exists, False otherwise
        
        """
        id_map = load_data(self.segment_metadata_path)
        
        if key not in id_map:
            return False
        
        # Key exists, check if vector and payload are identical
        existing_entry = id_map[key]
        segment_id = existing_entry["segment_id"]
        vector_id = existing_entry["vector_id"]
        
        # Load existing vector
        seg_path = self.segment_path / segment_id
        vectors_file = seg_path / "vectors.npy"
        if vectors_file.exists():
            existing_vectors = np.load(vectors_file)
            existing_vector = existing_vectors[vector_id]
            
            # Load existing payload
            payloads_file = seg_path / "payloads.json"
            existing_payloads = load_data(payloads_file)
            existing_payload = existing_payloads.get(str(vector_id), {})
            
            # Compare vector (with tolerance for floating point)
            vector_same = np.allclose(vector.astype('float32'), existing_vector, rtol=1e-5)
            payload_same = payload == existing_payload
            
            if vector_same and payload_same:
                return True  # Exact duplicate
            else:
                return False
        
        return False
    
    def get_next_vector_id(self, column_name: str) -> str:
        """
        Get next available ID for a column.

        Args:
            column_name (str): Name of the column

        Returns:
            str: Next available ID
        """
        id_map = load_data(self.segment_metadata_path)
        existing_ids = [k for k in id_map.keys() if k.startswith(f"{column_name}_")]

        if not existing_ids:
            return f"{column_name}_0"

        max_id = max([int(k.split("_")[-1]) for k in existing_ids])
        return f"{column_name}_{max_id + 1}"
        
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
        mode = None
        
        if isinstance(set_columns, str):
            val = set_columns.strip().lower()
            if val in ("first column", "first"):
                mode = ColumnPick.FIRST_COLUMN
            elif val == "all":
                mode = ColumnPick.ALL
        elif isinstance(set_columns, list):
            if len(set_columns) == 1 and str(set_columns[0]).strip().lower() in (
                "first column", "first", "all"
            ):
                val = str(set_columns[0]).strip().lower()
                if val in ("first column", "first"):
                    mode = ColumnPick.FIRST_COLUMN
                elif val == "all":
                    mode = ColumnPick.ALL
            else:
                mode = "LIST"
        
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
    
    def load_segment(self, segment_name:str, key: str, vector: np.ndarray, payload: Dict[str, Any]) -> None:
        """
        Insert a vector with payload into a segment.
        Args:
            segment_name (str): Segment Name (e.g., table name or product name)
            key (str): Unique external ID (e.g., column name).
            vector (np.ndarray): Vector embedding (1D float32 array).
            payload (Dict): Metadata payload.
        
        Return:
            dict: {
                "success": bool,
                "message": Status /Error (if success=True),
                "message": str (if success=False)
            }
        """
        try:
            if not isinstance(vector, np.ndarray):
                vector = np.array(vector, dtype="float32") 
            
            # Check for duplicates
            if self._check_duplicate_entry(key, vector, payload):
                return {"success": False, "message": f"Duplicate entry skipped for key '{key}'"}

            seg_path, segment_id = self._ensure_namespace(segment_name=segment_name)
            index = self._load_index(seg_path)
            
            # Add vector
            vec = vector.reshape(1, -1).astype("float32")
            index.add(vec)
            faiss.write_index(index, str(seg_path / "index.faiss"))
            
            # Save vectors.npy (append)
            vectors_file = seg_path / "vectors.npy"
            if vectors_file.exists():
                vectors = np.load(vectors_file)
                vectors = np.vstack([vectors, vec])
            else:
                vectors = vec
            np.save(vectors_file, vectors)
            
            # Update payloads.json
            payloads_file = seg_path / "payloads.json"
            payloads = load_data(payloads_file) 
            vector_id = index.ntotal - 1
            payloads[str(vector_id)] = payload
            save_data(payloads, payloads_file)

            # Update id_map.json
            id_map = load_data(self.segment_metadata_path)
            id_map[key] = {"segment_id": segment_id, "vector_id": vector_id}
            save_data(id_map, self.segment_metadata_path)
            
            # Update segment_map.json
            segment_map = load_data(self.segment_map_path)
            if segment_name not in segment_map:
                # Always start with a list of segment IDs
                segment_map[segment_name] = {"segment_ids": [segment_id]}
            else:
                # Append only if not already present
                if segment_id not in segment_map[segment_name]["segment_ids"]:
                    segment_map[segment_name]["segment_ids"].append(segment_id)
            save_data(segment_map, self.segment_map_path)
            return {"success": True, "message": f"Inserted vector for key '{key}'"}
        except Exception as e:
            return {"success": False, "message": f"Failed to insert vector: {str(e)}"}
    
    # def insert_segment(self, segment_dataset: Dict[str, List[Any]], segment_name: str, set_columns) -> Dict[str, Any]:
    #     """
    #         Load multiple vectors from a dataset into segments.
            
    #         Args:
    #             segment_dataset (Dict[str, List[Any]]): Dataset containing text data
    #             segment_name (str): Name of the segment to load data into
    #             set_columns : Columns to process
                
    #         Returns:
    #             Dict[str, Any]: Result dictionary with statistics and status
    #     """
    #     if isinstance(segment_dataset, dict):
    #         try:
    #             segment_dataset = pl.DataFrame(segment_dataset)
    #         except Exception as e:
    #             return {"success": False, "message": f"Failed to convert dataset to DataFrame: {str(e)}"}
        
    #     if not isinstance(segment_dataset, pl.DataFrame) or segment_dataset.height == 0:
    #         return {"success": False, "message": "Invalid or empty dataset"}
        
    #     # Process each column
    #     total_inserted = 0
    #     total_skipped = 0
    #     errors = []
    #     columns = self.determine_columns_to_process(segment_dataset=segment_dataset, set_columns=set_columns)
    #     if not columns["success"]:
    #         return columns["message"]
    #     for col in columns.get("columns",""):
    #         if col not in segment_dataset.columns:
    #             errors.append(f"Column '{col}' not found in dataset")
    #             continue
            
    #         texts = segment_dataset[col].fill_null("").to_list()
    #         if not any(texts):
    #             errors.append(f"Column '{col}' not found in dataset")
    #             continue
    #         try:
    #             embeddings = self.model.encode(texts).tolist()

    #             # Insert row-by-row into SegmentManager
    #             for idx, (txt, vec) in enumerate(zip(texts, embeddings)):
    #                 if not txt.strip():  # Skip empty texts
    #                     continue
                    
    #                 key = self._get_next_vector_id(column_name=col)
    #                 vector = vec
    #                 payload = {col: txt, "row_index": idx}

    #                 result = self._load_segment(
    #                     segment_name=segment_name,
    #                     key=key,  
    #                     vector=vector,
    #                     payload=payload
    #                 )
                    
    #                 if result["success"]:
    #                     total_inserted += 1
    #                 else:
    #                     total_skipped += 1
    #                     if "Duplicate entry" not in result["message"]:
    #                         errors.append(f"Failed to insert {key}: {result['message']}")
                            
    #         except Exception as e:
    #             errors.append(f"Failed to process column '{col}': {str(e)}")
    #     return {
    #         "success": True,
    #         "total_inserted": total_inserted,
    #         "total_skipped": total_skipped,
    #         "errors": errors,
    #         "message": f"Processed {total_inserted} vectors, skipped {total_skipped}"
    #     }
                            
    # def search_vector(self, vector: np.ndarray, top_k: int = 3) -> List[Dict[str, Any]]:
    #     """
    #     Search for nearest neighbors of a vector within the latest segment.

    #     Args:
    #         vector (np.ndarray): Query embedding (1D float32 array).
    #         top_k (int): Number of nearest results.

    #     Returns:
    #         List[Dict]: Search results with id, distance, and payload.
    #     """
    #     if not isinstance(vector, np.ndarray):
    #         vector = np.array(vector, dtype="float32")

    #     # Get last (latest) segment path
    #     seg_path, segment_id = self._ensure_namespace()
    #     index = self._load_index(seg_path)

    #     if index.ntotal == 0:
    #         return []

    #     # Reshape query vector
    #     vec = np.array([vector], dtype="float32")

    #     # Perform FAISS search
    #     distances, indices = index.search(vec, top_k)

    #     # Load mappings
    #     payloads_file = seg_path / "payloads.json"
    #     payloads = load_data(payloads_file)

    #     id_map = load_data(self.segment_metadata_path)

    #     results = []
    #     for dist, idx in zip(distances[0], indices[0]):
    #         if idx == -1:  # FAISS may return -1 if fewer results
    #             continue

    #         # Find external key for this vector_id
    #         external_id = None
    #         for k, v in id_map.items():
    #             if v["segment"] == segment_id and v["vector_id"] == idx:
    #                 external_id = k
    #                 break

    #         payload = payloads.get(str(idx), {})

    #         results.append({
    #             "id": external_id,
    #             "distance": float(dist),
    #             "payload": payload
    #         })

    #     return results

        