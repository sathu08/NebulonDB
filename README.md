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

### 3. Authentication APIs

#### 3.1.1 Register User

Create a new user with a specific role.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/auth/register" \
  -u sathya:admin@123 \
  -H "Content-Type: application/json" \
  -d '{
    "username": "ndbadmin1",
    "password": "ndbadmin",
    "user_role": "super_user"
  }'
```

#### Example Roles

* `super_user`
* `admin`
* `user`

---

#### 3.1.2 Verify User Login

Verify login credentials for a registered user.

```bash
curl -X GET "http://localhost:6969/api/NebulonDB/auth/verify" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json"
```

---

#### 3.2 Corpus Management APIs

#### 3.2.1 Create Corpus

Create a new corpus.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/create_corpus" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample"
  }'
```

---

#### 3.2.2 Deactivate Corpus

Deactivate an existing corpus.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/deactivate_corpus" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample"
  }'
```

---

#### 3.2.3 Activate Corpus

Activate a previously deactivated corpus.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/activate_corpus" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample"
  }'
```

---

#### 3.2.4 Delete Corpus

Delete an existing corpus permanently.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/delete_corpus" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample"
  }'
```

---

#### 3.2.5 List All Corpus

Get all available corpus.

```bash
curl -X GET "http://localhost:6969/api/NebulonDB/corpus/list_corpus" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json"
```

---

#### 3.3 Segment Management APIs

#### 3.3.1 Load Segment

Load a new segment into a corpus.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/load_segment" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "nebulon_origin",
    "segment_name": "sample",
    "segment_dataset": {
      "text_col1": [
        "Hello world",
        "AI is amazing",
        "Polars is fast"
      ],
      "text_col2": [
        "Test sentence",
        "Another text",
        "Segment_dataset science"
      ],
      "numeric_col": [1, 2, 3]
    },
    "set_columns": [
      "text_col1",
      "text_col2"
    ]
  }'
```

#### Notes

* `segment_dataset` contains the actual records
* `set_columns` defines which columns are used for vector/text search

---

#### 3.3.2 Search Segment

Search for similar content inside a segment.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/search_segment" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "nebulon_origin",
    "segment_name": "sample",
    "search_item": "Hello World",
    "top_matches": "3",
    "min_score": 0.6
  }'
```

#### Parameters

* `search_item` → text to search
* `top_matches` → number of top results
* `min_score` → minimum similarity score threshold

---

#### 3.3.3 List Segments

List all segments available inside a corpus.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/list_segments" \
  -u ndbadmin1:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "nebulon_origin"
  }'
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

