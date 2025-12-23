import os
import json
import faiss
import shutil

import numpy as np
import polars as pl 
from pathlib import Path

from datetime import datetime, timezone
from typing import List, Dict, Tuple, Any,Optional

from utils.models import load_data, save_data
from utils.models import ColumnPick , AuthenticationConfig
from utils.constants import NDBCorpusMeta
from db.ndb_settings import NDBConfig, NDBCryptoManager


config_settings = NDBConfig()
crypto_manager = NDBCryptoManager()

# ==========================================================
#        CorpusManager
# ==========================================================

class CorpusManager:
    """CorpusManager handles validation and retrieval of corpus data and metadata."""
    def __init__(self):
        """Initialize CorpusManager."""
        self.vector_storage_path:Path = Path(config_settings.VECTOR_STORAGE)
        self.metadata_path:Path = Path(config_settings.VECTOR_METADATA)
        
        self._validate_paths()

    def _validate_paths(self) -> None:
        """Check that essential paths exist."""
        errors = []

        if not self.vector_storage_path.exists() or not self.vector_storage_path.is_dir():
            errors.append(f"Vector storage path missing: {self.vector_storage_path}")

        if not self.metadata_path.exists():
            errors.append(f"Metadata file not found: {self.metadata_path}")
            
        if errors:
            raise FileNotFoundError(" | ".join(errors))
    
    @staticmethod
    def generate_corpus_metadata(corpus_name: str, created_by: str, status:str) -> Dict[str, str]:
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
        
    def create_corpus(self, corpus_name: str, username:str, status:str="active") -> None:
        """
        Create a new corpus.

        Args:
            corpus_name (str): Name of the corpus to create.
            username (str): Name of the user creating the corpus.
        """
        corpus_path = self.vector_storage_path / corpus_name
        os.makedirs(corpus_path, exist_ok=True)
        for corpus_subdir in config_settings.DEFAULT_CORPUS_STRUCTURES:
            (corpus_path / corpus_subdir).mkdir(parents=True, exist_ok=True)
                
        corpus_config_path = corpus_path / Path(config_settings.DEFAULT_CORPUS_CONFIG_STRUCTURES)
        config_data = config_settings.DEFAULT_CORPUS_CONFIG_DATA
        save_data(data=config_data, path_loc=corpus_path / corpus_config_path)

        # === Store the corpus details ===
        created_corpus = load_data(path_loc=self.metadata_path)
        created_corpus[corpus_name] = self.generate_corpus_metadata(
            corpus_name=corpus_name,
            created_by=username,
            status=status
        )
        save_data(data=created_corpus, path_loc=self.metadata_path)

        # === Create an segment metadata file ===
        segment_metadata_path = corpus_path / Path(config_settings.SEGMENTS_METADATA)
        segment_metadata = load_data(path_loc=segment_metadata_path)
        save_data(data=segment_metadata, path_loc=segment_metadata_path)

        segment_map_path = corpus_path / Path(config_settings.SEGMENT_MAP)
        segment_map = load_data(path_loc=segment_map_path)
        save_data(data=segment_map, path_loc=segment_map_path)

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
        save_data(path_loc=self.metadata_path, data=corpus_info)

# ==========================================================
#        SegmentManager
# ==========================================================

class SegmentManager:
    """
    SegmentManager handles dynamic creation/loading of FAISS segments,
    along with vectors, payloads, and ID mapping."""
    
    def __init__(self, corpus_name:str):
        """
        Initialize SegmentManager for a specific corpus.

        Args:
            corpus_name (str): Name of the corpus to manage.
        """
        self.corpus_name: str = corpus_name
        self.metadata_path:Path = Path(config_settings.VECTOR_METADATA)
        self.corpus_path: Path = Path(config_settings.VECTOR_STORAGE) / self.corpus_name
        self.segment_path: Path = self.corpus_path / NDBCorpusMeta.SEGMENTS_NAME
        self.segment_metadata_path: Path = self.corpus_path / config_settings.SEGMENTS_METADATA
        self.segment_map_path: Path = self.corpus_path / config_settings.SEGMENT_MAP
        self.corpus_config: Path = self.corpus_path / config_settings.DEFAULT_CORPUS_CONFIG_STRUCTURES
 
        self.config = self._load_config()
        self._validate_checks()
        self._validate_paths()

    def _validate_checks(self) -> None:
        """Check that essential paths exist."""
        errors = []
        if not self.corpus_name in self.get_segment_list():
            errors.append(f"Corpus '{self.corpus_name}' not found in metadata.")
        if errors:
            return errors

    def _validate_paths(self) -> None:
        """Check that essential paths exist."""
        errors = []
        if not self.corpus_path.exists() or not self.corpus_path.is_dir():
            errors.append(f"Vector storage path missing: {self.corpus_path}")
        if not self.segment_metadata_path.exists():
            errors.append(f"Metadata file not found: {self.segment_metadata_path}")
        if not self.segment_map_path.exists():
            errors.append(f"Metadata file not found: {self.segment_map_path}")
        if not self.corpus_config.exists():
            errors.append(f"Metadata file not found: {self.corpus_config}")
        if errors:
            raise FileNotFoundError(" | ".join(errors))

    def _load_config(self) -> Optional[Dict]:
        """Load corpus configuration."""
        try:
            return load_data(self.corpus_config)
        except Exception as _:
            return None
    
    def get_segment_id_list(self,segment_name:str)-> List[str]:
        """
        Get a list of segments for a given segment name.

        Args:
            segment_name (str): Name of the segment

        Returns:
            Optional[List]: List of segment IDs or None if not found
        """
        segment_map = load_data(self.segment_map_path)
        return segment_map.get(segment_name, {}).get("segment_ids", [])
    
    def get_segment_list(self)-> List[str]:
        """
        Get a list of segments for a given segment name.

        Args:
            segment_name (str): Name of the segment

        Returns:
            Optional[List]: List of segment IDs or None if not found
        """
        segment_map = load_data(self.metadata_path)
        return segment_map.get(self.corpus_name, {}).get("segments", [])
    
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
        existing_segments = self.get_segment_id_list(segment_name)

        if segment_name and existing_segments and latest:
            seg_path, seg_name = latest
            index = self._load_index(seg_path)
            max_size = self.config["segment_max_size"]
            
            if int(index.ntotal) < int(max_size):
                return seg_path, seg_name

        metadata = load_data(self.metadata_path)
        if segment_name not in metadata[self.corpus_name]["segments"]:
            metadata[self.corpus_name]["segments"].append(segment_name)
        save_data(metadata, self.metadata_path)
        
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
        seg_path = self.segment_path / Path(segment_id)
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
            elif val == "all":
                mode = ColumnPick.ALL
            else:
                mode = "LIST"
                return mode, [set_columns]

        elif isinstance(set_columns, list):
            if len(set_columns) == 1 and str(set_columns[0]).strip().lower() in ("first column", "first", "all"):
                val = str(set_columns[0]).strip().lower()
                if val in ("first column", "first"):
                    mode = ColumnPick.FIRST_COLUMN
                elif val == "all":
                    mode = ColumnPick.ALL
            else:
                mode = "LIST"
                return mode, set_columns

        return mode, []

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

    def get_next_vector_ids(self, column_name: str, count: int) -> List[str]:
        """
        Get next batch of available IDs for a column.
        """
        id_map = load_data(self.segment_metadata_path)
        existing_ids = [k for k in id_map.keys() if k.startswith(f"{column_name}_")]

        if not existing_ids:
            start_id = 0
        else:
            start_id = max([int(k.split("_")[-1]) for k in existing_ids]) + 1
            
        return [f"{column_name}_{start_id + i}" for i in range(count)]
        
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
            
            # # Add vector
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
            if payloads and "ndb_data" in payloads and "ndb_key" in payloads:
                decrypted_bytes = crypto_manager.decrypt_data(payloads)
                decrypted_str = decrypted_bytes.decode(AuthenticationConfig.ENCODING)
                payloads = json.loads(decrypted_str)
            vector_id = int(index.ntotal - 1)
            payloads[str(vector_id)] = payload
            payload_bytes = save_data(payloads, return_bytes=True)
            encrypted_payloads = crypto_manager.encrypt_data(payload_bytes)
            save_data(encrypted_payloads, payloads_file)

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

    def load_segment_batch(self, segment_name:str, keys: List[str], vectors: np.ndarray, payloads: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Batch insert vectors with payloads into a segment.
        Args:
            segment_name (str): Segment Name
            keys (List[str]): List of Unique external IDs
            vectors (np.ndarray): Matrix of Vector embeddings (N, D)
            payloads (List[Dict]): List of Metadata payloads
        
        Return:
            dict: Success/Failure stats
        """
        try:
            if not isinstance(vectors, np.ndarray):
                vectors = np.array(vectors, dtype="float32")
                
            num_vectors = len(keys)
            if vectors.shape[0] != num_vectors or len(payloads) != num_vectors:
                 return {"success": False, "message": "Input lists/arrays must have same length"}

            seg_path, segment_id = self._ensure_namespace(segment_name=segment_name)
            index = self._load_index(seg_path)
            
            # --- Load existing metadata for deduplication ---
            id_map = load_data(self.segment_metadata_path)
            
            # Filter out duplicates
            new_indices = []
            skipped_keys = []
            
            for i, key in enumerate(keys):
                if key in id_map:
                    # For speed, in batch mode we just skip existing keys without checking content equality
                    # (To allow content check, we'd need to load vectors which is slow)
                    skipped_keys.append(key)
                else:
                    new_indices.append(i)
            
            if not new_indices:
                 return {"success": True, "message": "All items were duplicates", "inserted": 0, "skipped": len(keys)}
                 
            # Filter inputs
            keys_to_add = [keys[i] for i in new_indices]
            vectors_to_add = vectors[new_indices]
            payloads_to_add = [payloads[i] for i in new_indices]
            
            # --- FAISS Add ---
            if vectors_to_add.ndim == 1:
                vectors_to_add = vectors_to_add.reshape(1, -1)
                
            index.add(vectors_to_add)
            faiss.write_index(index, str(seg_path / "index.faiss"))
            
            # --- Save Vectors ---
            vectors_file = seg_path / "vectors.npy"
            if vectors_file.exists():
                existing = np.load(vectors_file)
                updated_vectors = np.vstack([existing, vectors_to_add])
            else:
                updated_vectors = vectors_to_add
            np.save(vectors_file, updated_vectors)
            
            # --- Bulk Load & Update Payloads ---
            payloads_file = seg_path / "payloads.json"
            existing_payloads = load_data(payloads_file)
            if existing_payloads and "ndb_data" in existing_payloads and "ndb_key" in existing_payloads:
                decrypted_bytes = crypto_manager.decrypt_data(existing_payloads)
                decrypted_str = decrypted_bytes.decode(AuthenticationConfig.ENCODING)
                existing_payloads = json.loads(decrypted_str)
                
            start_vector_id = int(index.ntotal - len(keys_to_add))
            
            for i, (key, payload) in enumerate(zip(keys_to_add, payloads_to_add)):
                vector_id = start_vector_id + i
                existing_payloads[str(vector_id)] = payload
                
                # Update ID Map
                id_map[key] = {"segment_id": segment_id, "vector_id": vector_id}
                
            # Save Payloads (Encrypted)
            payload_bytes = save_data(existing_payloads, return_bytes=True)
            encrypted_payloads = crypto_manager.encrypt_data(payload_bytes)
            save_data(encrypted_payloads, payloads_file)
            
            # Save ID Map
            save_data(id_map, self.segment_metadata_path)
            
            # Update segment map
            segment_map = load_data(self.segment_map_path)
            if segment_name not in segment_map:
                segment_map[segment_name] = {"segment_ids": [segment_id]}
            elif segment_id not in segment_map[segment_name]["segment_ids"]:
                 segment_map[segment_name]["segment_ids"].append(segment_id)
            save_data(segment_map, self.segment_map_path)
            
            return {
                "success": True, 
                "message": f"Batch inserted {len(keys_to_add)} items", 
                "inserted": len(keys_to_add),
                "skipped": len(skipped_keys)
            }
            
        except Exception as e:
            return {"success": False, "message": f"Batch load failed: {str(e)}"}

    def search_vector(self, segment_name:str, query_vec:np.ndarray, top_k: Optional[int] = None, set_columns: Optional[List[str]] = None) ->Dict[str, Any]:
        """
        Search for nearest neighbors of a vector across all segments in a namespace.

        Args:
            segment_name (str): Name of the segment group/namespace
            query_vec (np.ndarray): Query vector (1D float32 array)
            top_k (int): Number of top results to return.

        Returns:
            List[Dict]: Search results with id, distance, and payload.
        """
        
        mode, set_columns = SegmentManager._determine_column_mode(set_columns)
                
        existing_segments = self.get_segment_id_list(segment_name)
        results = {}
        search_id = 0
        
        # Ensure query_vec is 2D for FAISS
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)
            
        # Load metadata mapping once
        id_map = load_data(self.segment_metadata_path)

        # Build quick lookup for external_id
        id_lookup = {(v["segment_id"], v["vector_id"]): k for k, v in id_map.items()}
        
        num_matches = top_k if top_k is not None else self.config.get("top_matches", 3)
        
        for segment_id in existing_segments:
            segment_id_path = self.segment_path / segment_id
            index = self._load_index(segment_id_path)
            
            if index.ntotal == 0:
                continue  # skip empty index
            
            # FAISS search
            distances, indices = index.search(query_vec, int(num_matches))

            # Load payloads (text/data attached to vectors)
            payloads_file = segment_id_path / "payloads.json"
            payloads = load_data(payloads_file)
            decrypted_bytes = crypto_manager.decrypt_data(payloads)
            decrypted_str = decrypted_bytes.decode("utf-8")
            payloads = json.loads(decrypted_str)
            # Collect results
            for dist, idx in zip(distances[0], indices[0]):
                if idx == -1:
                    continue

                # Match external_id from metadata
                external_id = id_lookup.get((segment_id, idx))

                # Get payload (the actual sentence/data stored)
                payload = payloads.get(str(idx), {})
                
                row_index = payload.get("row_index")

                if row_index is not None and set_columns:
                    matching_entries = [entry for entry in payloads.values() if entry.get("row_index") == row_index]
                    if matching_entries:
                        # Merge all entries with the same row_index
                        merged_payload = {}
                        for entry in matching_entries:
                            merged_payload.update(entry)

                        if mode == "LIST":
                            payload = {col: merged_payload.get(col) for col in set_columns if col in merged_payload}
                        elif mode == ColumnPick.FIRST_COLUMN:
                            if merged_payload:
                                first_col = next(iter(merged_payload))
                                payload = {first_col: merged_payload[first_col]}  
                               
                results[str(search_id)] = {
                    "segment_id": segment_id,
                    "external_id": external_id,
                    "distance": float(dist),
                    "payload": payload
                }
                search_id += 1
                
        return results
