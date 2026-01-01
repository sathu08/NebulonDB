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

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Virtual environment (recommended)

### Step-by-Step Installation

1.  **Install Python 3.9+**
    
    **Ubuntu/Debian:**
    ```bash
    sudo apt update
    sudo apt install python3.10 python3.10-venv python3-pip
    ```
    
    **macOS (using Homebrew):**
    ```bash
    brew install python@3.10
    ```
    
    **Windows:**
    Download and install from [python.org](https://www.python.org/downloads/)

2.  **Clone the Repository**
    ```bash
    git clone <your-repo-url>
    cd NebulonDB
    ```

3.  **Create and Activate Virtual Environment**
    
    **Linux/macOS:**
    ```bash
    python3 -m venv env
    source env/bin/activate
    ```
    
    **Windows:**
    ```bash
    python -m venv env
    env\Scripts\activate
    ```

4.  **Upgrade pip and Clear Cache**
    ```bash
    # Upgrade pip to latest version
    pip install --upgrade pip
    
    # Clear pip cache (optional, helps with installation issues)
    pip cache purge
    ```

5.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

6.  **Set Environment Variable**
    
    **Linux/macOS (add to ~/.bashrc or ~/.zshrc):**
    ```bash
    export NEBULONDB_HOME=/path/to/NebulonDB
    source ~/.bashrc  # or source ~/.zshrc
    ```
    
    **Windows (Command Prompt):**
    ```cmd
    set NEBULONDB_HOME=C:\path\to\NebulonDB
    ```
    
    **Windows (PowerShell):**
    ```powershell
    $env:NEBULONDB_HOME="C:\path\to\NebulonDB"
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
The server will start on `http://localhost:6969` (default).

### 3. Usage Example (Python)

Here is how to upload data using the API. You can choose between **Text Mode** (server generates embeddings) or **Raw Mode** (send your own vectors).

```python
import requests
import numpy as np

BASE_URL = "http://localhost:6969/api/NebulonDB"
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

## ⚡ Performance Comparison

NebulonDB has been benchmarked against industry-standard vector databases:

### NebulonDB vs FAISS vs ChromaDB

| Feature | NebulonDB | FAISS | ChromaDB |
|---------|-----------|-------|----------|
| **Batch Insertion** | ~0.06s (1k vectors) | ~0.05s | ~0.15s |
| **REST API** | ✅ Built-in | ❌ No | ✅ Built-in |
| **Authentication** | ✅ RBAC | ❌ No | ✅ Basic |
| **Metadata Storage** | ✅ JSON | ❌ No | ✅ SQLite |
| **Ease of Setup** | ✅ Single command | ⚠️ Manual | ⚠️ Docker |
| **Vector Search** | ✅ FAISS-powered | ✅ Native | ✅ HNSW |

### Key Advantages

✅ **All-in-One Solution**: Unlike FAISS (index-only) or ChromaDB (requires Docker), NebulonDB provides a complete, production-ready vector database with authentication, REST API, and metadata management out of the box.

✅ **Lightweight**: No Docker required, minimal dependencies, runs on any machine with Python 3.9+.

✅ **Performance**: Leverages FAISS for vector operations while maintaining competitive performance with additional features like RBAC and flexible metadata schemas.

✅ **Developer-Friendly**: Simple installation, clear API, and comprehensive documentation make integration effortless.

