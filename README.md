# NebulonDB

**NebulonDB** is a high-performance, lightweight Vector Database designed for speed and simplicity. It combines the power of **FAISS** for vector indexing with a JSON-based metadata system, accessible via a robust **FastAPI** interface.

## 🚀 Features

*   **High-Performance Ingestion**: Optimized **Batch Insertion** engine allows ingesting thousands of vectors in milliseconds (~0.06s for 1k vectors).
*   **Hybrid Storage**:
    *   **Vectors**: Managed by FAISS (HNSW/Flat) and NumPy (`.npy`) for extreme speed.
    *   **Metadata**: Stored in JSON (`payloads.json`), supporting flexible schemas.
*   **Security**: Built-in Role-Based Access Control (RBAC) with `BCrypt` encryption and `NDBSafeLocker` for credential management.
*   **REST API**: Full-featured API for managing Corpora, Segments, and Users.
*   **Raw Vector Support**: Bypass internal embedding generation to bring your own vectors (OpenAI, Cohere, etc.) or for pure DB benchmarking.

---

## 🛠️ Installation

1.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd NebulonDB
    ```

2.  **Install Dependencies**
    *Note: We use `bcrypt==4.0.1` to ensure compatibility with `passlib`.*
    ```bash
    pip install "bcrypt==4.0.1" -r requirements.txt
    ```

---

## ⚡ Quick Start

### 1. Create an Admin User
Before starting the server, you must create an admin user.
```bash
python run.py --create-user
# Follow the prompts to set username (e.g., 'admin') and password.
# Select 'admin_user' or 'super_user' as the role.
```

### 2. Start the Server
```bash
python run.py start
```
The server will start on `localhost:8000` (default).

### 3. Usage Example (Python)

Here is how to upload data using the API. You can choose between **Text Mode** (server generates embeddings) or **Raw Mode** (send your own vectors).

```python
import requests
import numpy as np

BASE_URL = "http://localhost:8000/api/NebulonDB"
AUTH = ("admin", "password") # Use credentials created in Step 1

# 1. Create a Corpus (Collection)
requests.post(f"{BASE_URL}/corpus/create_corpus", json={"corpus_name": "my_corpus"}, auth=AUTH)

# 2. Upload Data (Batch)
# OPTION A: Text Mode (Server uses AI model to generate vectors)
payload = {
    "corpus_name": "my_corpus",
    "segment_name": "segment_1",
    "segment_dataset": {
        "description": ["This is doc 1", "This is doc 2"]
    },
    "set_columns": "description"
}
requests.post(f"{BASE_URL}/segment/load_segment", json=payload, auth=AUTH)

# OPTION B: Raw Vector Mode (Faster / Bring Your Own Vectors)
vectors = np.random.rand(10, 384).tolist() # 10 vectors of dim 384
payload_raw = {
    "corpus_name": "my_corpus",
    "segment_name": "segment_raw",
    "segment_dataset": {
        "vector_col": vectors
    },
    "set_columns": "vector_col",
    "is_precomputed": True  # <--- Critical Flag
}
requests.post(f"{BASE_URL}/segment/load_segment", json=payload_raw, auth=AUTH)
```

---

## 📂 Project Structure

*   `run.py`: Main entry point (Start/Stop/Create User).
*   `ndb_host/`: Core application source code.
    *   `api/`: FastAPI routes (`auth`, `corpus`, `segment`).
    *   `db/`: Database Engine.
        *   `index_manager.py`: **Core Logic**. Handles FAISS indexing and Batch Insertion.
    *   `utils/`: Helper models and configuration.
*   `VectorDatabase/`: Data directory where vectors and metadata are stored (Created at runtime).

---

## 📊 Benchmarking

We provide two scripts to verify performance:

1.  **`benchmark.py`** (Internal): Tests the core `SegmentManager` class directly, bypassing the API.
    *   Run: `python benchmark.py`
    *   Expect: ~0.06s ingestion for 1000 vectors.

2.  **`benchmark_api.py`** (End-to-End): Tests the full HTTP API flow including Authentication.
    *   Run: `python benchmark_api.py` (Requires server running)
    *   Mode: Uses `is_precomputed=True` to isolate DB performance from AI model latency.
    *   Expect: ~0.1s - 0.2s total latency for 1000 vectors.
