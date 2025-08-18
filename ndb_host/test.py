"""
import faiss, os, yaml, json
import numpy as np
from pathlib import Path

BASE_DIR = Path("NebulonDB")

def load_config():
    with open("config.yaml") as f:
        return yaml.safe_load(f)

def ensure_namespace(namespace: str):
    ns_path = BASE_DIR / namespace / "segments" / "segment_0"
    ns_path.mkdir(parents=True, exist_ok=True)
    return ns_path

def load_or_create_index(ns_path, dim, config):
    index_path = ns_path / "index.faiss"
    if index_path.exists():
        return faiss.read_index(str(index_path))
    # Create HNSW
    index = faiss.IndexHNSWFlat(dim, config["params"]["hnsw_m"])
    index.hnsw.efConstruction = config["params"]["ef_construction"]
    index.hnsw.efSearch = config["params"]["ef_search"]
    return index

def insert_vector(namespace, vector, payload):
    config = load_config()
    ns_path = ensure_namespace(namespace)
    index = load_or_create_index(ns_path, config["dimension"], config)
    vec = np.array([vector], dtype='float32')
    index.add(vec)
    faiss.write_index(index, str(ns_path / "index.faiss"))
    with open(ns_path / "payloads.json", "a") as f:
        f.write(json.dumps(payload) + "\n")
    return {"message": "Inserted successfully."}

def search_vector(namespace, vector, top_k):
    config = load_config()
    ns_path = ensure_namespace(namespace)
    index = load_or_create_index(ns_path, config["dimension"], config)
    vec = np.array([vector], dtype='float32')
    distances, indices = index.search(vec, top_k)
    return {"indices": indices.tolist(), "distances": distances.tolist()}

def list_namespaces():
    return [d.name for d in BASE_DIR.iterdir() if d.is_dir()]
"""
