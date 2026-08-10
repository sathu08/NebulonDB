# NebulonDB

**NebulonDB** is a high-performance, lightweight **Nova + Mesh** database built
on a custom storage engine. It combines a durable **LSM-tree** key/value store
(**NebulonCosmos**) with two search engines under **NebulonOrbit** — **Nova**
(vector similarity) and **Mesh** (graph traversal) — exposed through a secure
**FastAPI** REST API.

> 🔗 **Architecture** — See [Architecture & Engine Overview](ndb_host/db/engine/OVERVIEW.md)
> for detailed diagrams and explanations of the `NebulonCosmos` and
> `NebulonOrbit` engines (memtable/WAL/segments/compaction, HNSW generations,
> Mesh persistence, ranking pipeline).

---

## 🧭 Nova & Mesh — the two search engines

NebulonOrbit is built around two complementary subsystems, defined here and
referred to throughout this document:

* **Nova** — the **vector** subsystem. An HNSW approximate-nearest-neighbor
  (ANN) index (`nova_engine.py`) built on `hnswlib`, with generational saves,
  SHA-256 checksums and manifest-based fallback recovery. Vectors are persisted
  as dedicated records in the `nebulon_nova` Cosmos segment (`nova_store.py`);
  their text/metadata live separately in `nebulon_documents`
  (`document_store.py`). A shared `id` links a document to its vector.
  **Nova answers "what is most similar to this?"**

* **Mesh** — the **graph** subsystem. An in-memory graph of nodes and directed,
  weighted, labeled edges (`mesh_engine.py` + `mesh_store.py`), persisted as
  normalized per-row records in the `nebulon_mesh_nodes` and
  `nebulon_mesh_edges` Cosmos segments, and rendered as an interactive
  Cytoscape.js HTML visualization (`mesh_viz.py`). Edge `weight` is
  auto-computed as the cosine similarity of the two endpoint vectors (or
  `1.0` when either endpoint has no vector), or taken verbatim when supplied.
  **Mesh answers "what is connected to this?"**

Together they power the **hybrid** search mode: Nova retrieves the closest
vector hits, then Mesh expands around them (and an optional seed node) through
graph neighbors.

> Every code module, class and file layout behind Nova and Mesh is documented
> in the [Architecture & Engine Overview](ndb_host/db/engine/OVERVIEW.md).

---

## 🚀 Features

### Engine Layer (`db/engine`)
* **NebulonCosmos** — LSM-tree inspired storage engine:
  * In-memory **memtable** + CRC-checked **Write-Ahead Log (WAL)** for crash safety
  * Immutable, optionally **zlib-compressed** segment files with **Bloom filters**
  * Flat binary index (`record_id → segment, offset, version`) with mmap reads
  * **Background compaction** daemon that merges segments, drops tombstones, and
    **rebuilds the in-memory index from the surviving segments** so reads never hit
    removed files
  * **Globally-unique record IDs** across all tables (single `meta` counter) so
    tables like `nebulon_userinfo` / `nebulon_metadata` can never collide; compaction
    keys merges by `(table, id)` so records are never dropped across tables
  * Atomic metadata / manifest persistence (`meta.bin`, `manifest.bin`)
  * Multi-table ("segment") namespacing and versioned records
* **NebulonOrbit** — unified **Nova** (vector) + **Mesh** (graph) search engine built on Cosmos:
  * **Nova** — **HNSW** ANN index (`hnswlib`) with generational saves, SHA-256
    checksums and manifest-based fallback recovery
  * **Mesh** — graph engine with BFS/DFS, shortest path, connected components
  * **Hybrid search** — Nova vector hits + Mesh neighbor expansion
  * **Ranking pipeline** — BM25, Reciprocal Rank Fusion (RRF), multi-signal
    rank engine (Nova vector + BM25 + metadata + importance + freshness), and an
    optional lazy-loaded cross-encoder re-ranker
  * **LSN-tagged WAL** replay for transaction-safe crash recovery

### API Layer (`ndb_host/api/routes`)
* **Authentication & RBAC** — HTTP Basic auth, bcrypt password hashing, and a
  4-level role hierarchy (`system` → `super_user` → `admin_user` → `user`).
  Users are persisted in NebulonCosmos with an in-memory cache. Supports user
  management: register, change password, and delete users (self-delete blocked).
* **Corpus Management** — create / list / delete / activate / deactivate corpora,
  each backed by either a `cosmos` (key/value) or `orbit` (**Nova** + **Mesh**)
  engine. Corpus metadata is tracked in NebulonCosmos.
* **Segment Ingestion** — load tabular datasets (dict / list / Polars
  `DataFrame`) into a corpus segment:
  * Column selection modes: **First Column**, **All text columns**, or an
    **explicit column list**
  * Automatic text → embedding generation (Sentence-Transformers) or
    **precomputed vectors** (`is_precomputed`)
  * Optional **Mesh relation loading** from explicit tuples **or** auto-detected
    `source`/`target`/`relation` DataFrame columns
  * Per-record metadata (`lang`, `doc_type`, `created_at`) with automatic
    **retention policies** (permanent / temporary TTL / session TTL)
* **Nova & Hybrid Search** — semantic text search with:
  * Search modes: `auto`, `nova` (Nova vector similarity), `mesh` (Mesh graph
    BFS), `hybrid` (Nova + Mesh expansion)
  * `top_matches`, `min_score` (threshold on **min-max normalized** scores),
    `lang_type`/`doc_type` metadata filters, `rank` (multi-signal ranking)
  * Mesh expansion controls: `graph_start_node`, `expand_depth`, `graph_boost`
* **Record & Mesh Inspection** — `segment_stats`, `get_record`, `get_neighbors`
  (in/out/both), `bfs`, `shortest_path`
* **Mesh Mutation** (admin-only) — `add_node`, `add_relation`, `remove_relation`,
  `delete_record`

### Infrastructure
* **Model Hub** (`core/model_hub.py`) — thread-safe singleton model cache with
  automatic batch sizing based on available VRAM/RAM and support for embedding
  + cross-encoder models.
* **Polars**-based ingestion and validation; NumPy float32 vectors throughout.

---

## 🔗 Architecture

The engine is split into two packages under `ndb_host/db/engine/`:

* **`cosmos/`** — the durable LSM storage engine (memtable, WAL, index,
  segment reader/writer, compactor, metadata).
* **`orbit/`** — the **Nova + Mesh** search engine (Nova HNSW vector engine,
  Mesh graph engine, ranking, visualization).

Full architecture diagrams, component tables, on-disk layout, and write/read
path explanations are in the dedicated overview document:

➡️ **[Architecture & Engine Overview](ndb_host/db/engine/OVERVIEW.md)**

```
ndb_host/
├── api/                    # FastAPI application + route modules
│   └── routes/
│       ├── auth.py         # register, verify, change_password, delete_user
│       ├── corpus.py       # create/list/delete/activate/deactivate corpus
│       └── segment.py      # load/list/search segment + Mesh endpoints
├── core/
│   ├── security.py         # bcrypt password hashing (passlib)
│   ├── permissions.py      # RBAC role hierarchy checks
│   └── model_hub.py        # embedding / cross-encoder model loading
├── db/
│   ├── engine/             # NebulonCosmos (LSM) + NebulonOrbit (Nova + Mesh)
│   │   ├── cosmos/         # storage engine submodules
│   │   ├── orbit/          # search engine submodules
│   │   ├── utils/          # config, serializer, bloom, models, constants
│   │   └── OVERVIEW.md     # architecture & engine documentation
│   ├── index_manager.py    # CorpusManager / SegmentManager orchestration
│   └── ndb_settings.py     # NDBConfig (paths, model, tuning settings)
├── services/
│   └── user_service.py     # user manager + authentication (HTTP Basic)
└── utils/
    ├── models.py           # Pydantic request/response models
    ├── constants.py        # roles, doc types, retention policies, column picks
    ├── bootstrap.py        # NebulonInitializer
    └── logger.py           # NebulonDBLogger
```

---

## 🛠️ Installation

### Prerequisites
- **Git** — required to clone the repository
- An internet connection (the first run downloads Python 3.10 and dependencies)

### One-Click Installer (recommended)

Fetch and run the installer directly from GitHub — no manual clone needed.
NebulonDB ships with `install.sh` (Linux/macOS) and `install.bat` (Windows).
The installer handles everything end to end: cloning the repository, installing
[`uv`](https://docs.astral.sh/uv/), provisioning Python 3.10, syncing all
dependencies into a virtual environment, verifying the `nebulondb` CLI, and
persisting `NEBULONDB_HOME`.

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/sathu08/NebulonDB/dev/install.sh | bash
```

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/sathu08/NebulonDB/dev/install.bat -OutFile "$env:TEMP\install.bat"; & "$env:TEMP\install.bat"
```

Alternatively, run the installers from a local clone:

```bash
git clone https://github.com/sathu08/NebulonDB.git
cd NebulonDB
bash install.sh          # Linux/macOS
install.bat              # Windows (Command Prompt)
```

What the installer does:

1. Clones the `dev` branch into `~/CodeBase/NebulonDB` (Linux/macOS) or
   `%USERPROFILE%\CodeBase\NebulonDB` (Windows).
2. Installs `uv` if it is not already available.
3. Installs **Python 3.10** via `uv` (only if not found).
4. Runs `uv sync --python 3.10` to create `.venv` and install dependencies.
5. Verifies the `nebulondb` CLI and runs `nebulondb --help`.
6. Persists `NEBULONDB_HOME` — via `~/.bashrc` on Linux/macOS, or via `setx`
   (user-level environment variable) on Windows.

> Re-running the installer updates an existing clone by fetching the latest
> `dev` branch with `git pull --ff-only`. Dependencies are kept in sync with
> `pyproject.toml`.

### Manual Installation (uv)

Prefer to do it by hand?

1.  **Install Git and [uv](https://docs.astral.sh/uv/).**

2.  **Clone the Repository**
    ```bash
    git clone https://github.com/sathu08/NebulonDB.git
    cd NebulonDB
    ```

3.  **Create the Virtual Environment and Install Dependencies**
    ```bash
    uv sync --python 3.10
    ```

4.  **Set the Environment Variable**

    **Linux/macOS (add to ~/.bashrc or ~/.zshrc):**
    ```bash
    export NEBULONDB_HOME=/path/to/NebulonDB
    source ~/.bashrc  # or source ~/.zshrc
    ```

    **Windows (PowerShell):**
    ```powershell
    $env:NEBULONDB_HOME="C:\path\to\NebulonDB"
    ```

5.  **Activate the Virtual Environment** (each new shell)
    ```bash
    source .venv/bin/activate    # Linux/macOS
    .venv\Scripts\activate       # Windows (Command Prompt)
    ```

---

## ⚡ Quick Start

### 1. Create an Admin User
Before starting the server, you must create an admin user.
```bash
nebulondb --create-user
# Follow the prompts to set username (e.g., 'admin') and password.
# Select 'admin_user' or 'super_user' as the role.
```

### 2. Start the Server
```bash
nebulondb start
```
The server will start on `http://localhost:6969` (default). Interactive API docs
are available at `http://localhost:6969/docs`.

### 3. Authentication & RBAC
All endpoints use **HTTP Basic Authentication**. The user store is persisted in
NebulonCosmos (`nebulon_userinfo` segment) and cached in memory.

| Role | Level | Capabilities |
|------|-------|--------------|
| `user` | 1 | Authenticate, verify, list corpora |
| `admin_user` | 2 | Everything above + create/deactivate/activate/delete corpus, load segments, mutate Mesh |
| `super_user` | 3 | All `admin_user` capabilities |
| `system` | 4 | Highest level; protects system corpora from deletion/deactivation |

---

## 🔐 API Reference

### 1. Authentication (`/api/NebulonDB/auth`)

#### 1.1 `POST /auth/register` — Register a new user
Create a new user with a specific role. The caller must be an authenticated user.
Validates username (≥3 chars) and password (≥6 chars), hashes the password with
bcrypt, and returns a structured response.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/auth/register" \
  -u admin:admin@123 \
  -H "Content-Type: application/json" \
  -d '{
    "username": "ndbadmin",
    "password": "ndbadmin",
    "user_role": "super_user"
  }'
```

#### 1.2 `GET /auth/verify` — Verify user authentication
Returns the current user's authentication status and role details.

```bash
curl -X GET "http://localhost:6969/api/NebulonDB/auth/verify" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json"
```

#### 1.3 `POST /auth/change_password` — Change the current user's password
Changes the password of the currently authenticated user. The caller must supply
their current password (verified against the stored bcrypt hash) and a new
password (≥6 chars). Returns an error if the current password is incorrect.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/auth/change_password" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "current_password": "ndbadmin",
    "new_password": "ndbadmin123"
  }'
```

#### 1.4 `POST /auth/delete_user` — Delete a user account
Permanently deletes a user account (from `nebulon_userinfo` and the in-memory
cache). A user **cannot delete their own account**; use a different
administrator account to delete another user.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/auth/delete_user" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{ "username": "olduser" }'
```

---

### 2. Corpus Management (`/api/NebulonDB/corpus`)

Corpora are top-level database namespaces. Each corpus is backed by either a
`cosmos` engine (durable key/value) or an `orbit` engine (**Nova** + **Mesh**).
Corpus metadata (name, creator, status, segments) is stored in NebulonCosmos
(`nebulon_metadata` segment).

#### 2.1 `POST /corpus/create_corpus` — Create corpus
Creates a new corpus (default type `cosmos`; pass `"ndb_type": "orbit"` to
enable **Nova** (vector) + **Mesh** (graph) features). Requires `admin_user` or
higher. Fails if the corpus already exists.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/create_corpus" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "ndb_type": "orbit"
  }'
```

#### 2.2 `GET /corpus/list_corpus` — List all corpora
Returns every available corpus (union of storage directories and metadata
records) with a total count.

```bash
curl -X GET "http://localhost:6969/api/NebulonDB/corpus/list_corpus" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json"
```

#### 2.3 `POST /corpus/deactivate_corpus` — Deactivate corpus
Marks a corpus as `deactivate` so it is excluded from active use. **System
corpora cannot be deactivated.** Requires `admin_user` or higher.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/deactivate_corpus" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{ "corpus_name": "sample" }'
```

#### 2.4 `POST /corpus/activate_corpus` — Activate corpus
Re-activates a previously deactivated corpus. **System corpora cannot be
activated (they are already active).** Requires `admin_user` or higher.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/activate_corpus" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{ "corpus_name": "sample" }'
```

#### 2.5 `POST /corpus/delete_corpus` — Delete corpus
Permanently removes the corpus directory and its metadata record. **System
corpora cannot be deleted, and a corpus must be deactivated before deletion.**
Requires `admin_user` or higher.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/corpus/delete_corpus" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{ "corpus_name": "sample" }'
```

---

### 3. Segment Management (`/api/NebulonDB/segment`)

Segments are the data collections inside a corpus. The backend is chosen by the
`ndb_type` field passed to each request:
* `orbit` — `load_segment` embeds text into **Nova** vectors (and optionally
  loads **Mesh** relations); `search_segment` performs semantic search.
* `cosmos` — `load_segment` stores documents directly (no embedding) and
  `get_data` returns them as-is.

#### 3.1 `POST /segment/load_segment` — Load a segment
Ingests a tabular dataset into a segment. Supports **dict**, **list of dicts**,
or a Polars `DataFrame` as `segment_dataset`. Behavior depends on `ndb_type`:
* `orbit` — every selected text column is embedded (or used directly when
  `is_precomputed: true`) and stored as a **Nova** vector record with metadata;
  Mesh relations may be loaded too.
* `cosmos` — each non-empty row is stored **directly** as a document (no text
  embedding, no Mesh relations).
Returns `inserted` / `skipped` counts and per-row errors, and registers the
segment in the corpus metadata.

**Request fields**

| Field | Type | Description |
|-------|------|-------------|
| `corpus_name` | str | Target corpus (must be an **orbit** corpus) |
| `segment_name` | str | Name of the segment to create/fill |
| `ndb_type` | str | Backend to use — `orbit` (vector) or `cosmos` (direct document store) |
| `segment_dataset` | dict/list | Column data or rows; converted to a Polars DataFrame |
| `set_columns` | str/list | `"First Column"`, `"All"`, or an explicit column list to embed |
| `is_precomputed` | bool | When true, columns already contain embeddings (no encoding) |
| `doc_type` | str | Document type tag stored in metadata (e.g. `txt`, `markdown`) |
| `lang_type` | str | Language tag stored in metadata (e.g. `en`) |
| `relations` | list | Explicit `[source, target, relation]` tuples to add as **Mesh** graph edges |
| `relations` | list | Explicit `[source, target, relation]` tuples to add as **Mesh** graph edges |
| `source_column` | str | Column name holding relation source IDs (auto-detected if omitted) |
| `target_column` | str | Column name holding relation target IDs (auto-detected if omitted) |
| `relation_column` | str | Column name holding the relation label (defaults to `"related"`) |

> **Node labels** — a `name` column (or a `label` key inside `metadata`) is
> stored as the **Mesh node label** for each record. When `source`/`target`
> columns hold node **labels** (strings) instead of numeric IDs, they are
> resolved to node ids automatically. Edge `weight` is auto-computed from the
> two endpoint vectors unless an explicit `weight` column is present.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/load_segment" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "ndb_type": "orbit",
    "segment_dataset": {
      "text_col1": ["Hello world", "AI is amazing", "Polars is fast"],
      "text_col2": ["Test sentence", "Another text", "Segment_dataset science"],
      "numeric_col": [1, 2, 3]
    },
    "set_columns": ["text_col1", "text_col2"]
  }'
```

**Mesh relation loading** happens in two ways:

* **Explicit tuples** — `"relations": [[1, 2, "links"], [2, 3, "links"]]`
* **DataFrame columns** — when `source`/`target`/`relation` columns exist, they
  are auto-detected. Common aliases: source (`source`, `source_id`, `src`,
  `from_id`, `from`), target (`target`, `target_id`, `dst`, `to_id`, `to`),
  relation (`relation`, `rel`, `edge_type`, `relationship`, `label`). You can
  override with `source_column` / `target_column` / `relation_column`.

**Precomputed vectors** — when `is_precomputed: true`, the selected column holds
embedding arrays directly:

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/load_segment" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "doc_type": "txt",
    "lang_type": "en",
    "segment_dataset": {
      "embeddings": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
    },
    "set_columns": ["embeddings"],
    "is_precomputed": true,
    "relations": [[1, 2, "related"]]
  }'
```

#### 3.2 `GET /segment/list_segment` — List segments
Lists every segment registered in a corpus with its `inserted` record count and
`created_at` timestamp.

```bash
curl -X GET "http://localhost:6969/api/NebulonDB/segment/list_segment?corpus_name=sample" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json"
```

#### 3.3 `POST /segment/get_data` — Get stored segment data
Returns the raw stored records for a segment with an optional `limit`. Works for
both backend types:
* `orbit` → returns the stored document (`text`, `metadata`, `created_at`) plus
  the vector, per shared record `id`.
* `cosmos` → returns the stored documents.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/get_data" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "ndb_type": "cosmos",
    "limit": 10
  }'
```

#### 3.4 `POST /segment/search_segment` — Search a segment
Semantic search over a segment's **Nova** vectors with optional **Mesh**
expansion and ranking.

**Request fields**

| Field | Type | Description |
|-------|------|-------------|
| `search_item` | str | Raw text query (required) |
| `top_matches` | int | Number of results to return (default 10) |
| `min_score` | float | Minimum score threshold; scores are **min-max normalized** to [0,1] |
| `set_columns` | str/list | When a list, restricts returned `metadata` to those keys |
| `lang_type` | str | Metadata filter on `lang` |
| `doc_type` | str | Metadata filter on `type` |
| `mode` | str | `auto` (default), `nova`, `mesh`, `hybrid` |
| `rank` | bool | Apply multi-signal ranking (Nova + BM25 + metadata + importance + freshness) |
| `graph_start_node` | int | Seed node for **Mesh** traversal (`mesh`) or extra expansion seed (`hybrid`) |
| `expand_depth` | int | Max **Mesh** BFS depth for expansion (default 1) |
| `graph_boost` | float | Score assigned to **Mesh**-discovered nodes in `hybrid` mode (default 0.1) |

**Search modes**

| Mode | Behavior |
|------|----------|
| `auto` | Picks `hybrid` when the **Mesh** graph has edges, otherwise `nova` |
| `nova` | Pure **Nova** HNSW vector similarity |
| `mesh` | Pure **Mesh** BFS from `graph_start_node`, scored by `1/(1+depth)` |
| `hybrid` | **Nova** vector search + **Mesh** BFS expansion from the top hits (and `graph_start_node`) |

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/search_segment" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "search_item": "Hello World",
    "top_matches": 3,
    "min_score": 0.5,
    "mode": "hybrid",
    "graph_start_node": 1,
    "expand_depth": 2,
    "graph_boost": 0.1,
    "rank": true
  }'
```

> Responses strip raw vectors from the payload (clients receive
> `id`, `score`, `metadata`). Use `get_record` to fetch the full **Nova** vector.

#### 3.5 `POST /segment/segment_stats` — Segment statistics
Returns **Nova** + **Mesh** statistics for a segment: `vector_count`, `dimension`,
`space`, `node_count`, `edge_count`, `deleted_ratio`, `lsn`.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/segment_stats" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{ "corpus_name": "sample", "segment_name": "seg1" }'
```

#### 3.6 `POST /segment/get_record` — Get a record
Returns the full record (`_id`, **Nova** vector, `metadata`) for a `record_id`.
`exists: false` when the record is not found.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/get_record" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{ "corpus_name": "sample", "segment_name": "seg1", "record_id": 1 }'
```

#### 3.7 `POST /segment/get_neighbors` — Mesh neighbors
Returns the neighbors of a **Mesh** node with a direction filter (`in`, `out`,
`both`). Each entry is `{node_id, relation}`.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/get_neighbors" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "node_id": 1,
    "direction": "both"
  }'
```

#### 3.8 `POST /segment/bfs` — Breadth-first traversal
Returns every node reachable from `start_node` within `max_depth`.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/bfs" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "start_node": 1,
    "max_depth": 3
  }'
```

#### 3.9 `POST /segment/shortest_path` — Shortest path
Finds the shortest unweighted path between `source` and `target`; returns the
node list and `path_length`. `data.path` is `null` when the nodes are
unreachable.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/shortest_path" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "source": 1,
    "target": 4
  }'
```

#### 3.10 `POST /segment/add_node` — Add Mesh node (admin)
Creates a **Mesh** node explicitly without requiring a **Nova** vector record.
An optional `label` may be supplied inside `metadata` (becomes the node's
display label). Requires `admin_user` or higher.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/add_node" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "node_id": 999,
    "metadata": {"label": "virtual", "kind": "virtual"}
  }'
```

#### 3.11 `POST /segment/add_relation` — Add Mesh relation (admin)
Adds a directed **Mesh** edge `source → target` with a relation label. Endpoint
nodes are auto-created if they do not exist; `source`/`target` can be numeric
ids **or** node labels (resolved automatically). When `weight` is omitted it is
auto-computed as the cosine similarity of the two endpoint vectors (or `1.0`
when either endpoint has no vector). Requires `admin_user` or higher.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/add_relation" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "source": 1,
    "target": 2,
    "relation": "friend"
  }'
```

#### 3.12 `POST /segment/remove_relation` — Remove Mesh relation (admin)
Removes a directed **Mesh** edge. When `relation` is omitted, all edges between
the `source` and `target` are removed. Requires `admin_user` or higher.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/remove_relation" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "source": 1,
    "target": 2,
    "relation": "friend"
  }'
```

#### 3.13 `POST /segment/delete_record` — Delete record (admin)
Deletes a record by `record_id`. Works for both backend types:
* `orbit` → deletes the **Nova** vector record and its **Mesh** node/edges, then flushes.
* `cosmos` → deletes the stored document via the Cosmos engine.
Requires `admin_user` or higher. Pass `ndb_type` to select the backend.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/delete_record" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "ndb_type": "orbit",
    "record_id": 1
  }'
```

#### 3.14 `POST /segment/mesh_load_graph` — Bulk Mesh graph load (admin)
Adds **Mesh** nodes and/or edges in a single request (**Option A** bulk graph
loading). Requires `admin_user` or higher and an **ORBIT** corpus.

* `nodes` — optional list of `{id?, label?, metadata?}`. `id` may be omitted to
  auto-create (and resolve) a node by `label`.
* `edges` — optional list of `{from, to, relation?, weight?}`. `from`/`to` are
  endpoint **refs** — either a numeric node `id` or a node **label** (both are
  resolved, auto-creating missing label-nodes). `relation` defaults to the node
  label or `"related"`. When `weight` is omitted it is auto-computed as the
  cosine similarity of the two endpoint vectors (`1.0` when no vector).

Returns `nodes_added` / `edges_added`.

**Request fields**

| Field | Type | Description |
|-------|------|-------------|
| `corpus_name` | str | Target **orbit** corpus |
| `segment_name` | str | Segment whose Mesh graph receives the load |
| `ndb_type` | str | Must be `orbit` |
| `nodes` | list | `{id?, label?, metadata?}` node descriptors |
| `edges` | list | `{from, to, relation?, weight?}` edge descriptors |

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/mesh_load_graph" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{
    "corpus_name": "sample",
    "segment_name": "seg1",
    "ndb_type": "orbit",
    "nodes": [
      {"label": "alpha"},
      {"label": "beta"}
    ],
    "edges": [
      {"from": "alpha", "to": "beta", "relation": "links"},
      {"from": 1, "to": 2, "relation": "links"}
    ]
  }'
```

#### 3.15 `POST /segment/mesh_visualization` — Mesh HTML visualization
Returns the path to the interactive Cytoscape.js HTML visualization for a
segment's **Mesh** graph.

```bash
curl -X POST "http://localhost:6969/api/NebulonDB/segment/mesh_visualization" \
  -u ndbadmin:ndbadmin \
  -H "Content-Type: application/json" \
  -d '{ "corpus_name": "sample", "segment_name": "seg1", "ndb_type": "orbit" }'
```

---

## 🗂️ Storage Layout

```
NEBULONDB_HOME/
└── Storage/
    └── <corpus_name>/           # one directory per corpus
        ├── NebulonCosmos/       # LSM engine: segments/, wal.log, index.idx, meta.bin, manifest.bin
        └── NebulonOrbit/        # Orbit engine: NebulonNova/ (HNSW generations), nova.wal, NebulonMesh/
            ├── NebulonNova/
            │   └── segment_<name>/   # one HNSW directory per segment (no spurious "default" dir)
            └── NebulonMesh/     # Mesh graph visualization HTML (mesh_graph_visualization.html)
```

Corpus metadata (including registered segments) lives in the account hub's
`nebulon_metadata` segment, and user accounts live in `nebulon_userinfo`.

Within each **ORBIT** corpus, the Cosmos engine stores four normalized tables
(keyed by a shared record `id`):

| Table | Fields | Contents |
|-------|--------|----------|
| `nebulon_documents` | `id`, `text`, `metadata` (incl. `label`), `created_at` | Document text + metadata |
| `nebulon_nova` | `id`, `vector`, `created_at` | Embedding vectors only |
| `nebulon_mesh_nodes` | `id`, `label`, `created_at` | Mesh graph nodes |
| `nebulon_mesh_edges` | `edge_id`, `from_id`, `to_id`, `relation`, `weight`, `created_at` | Mesh graph edges |

---

## 🔍 Ranking & Retrieval Details

* **Score normalization** — raw similarity scores are min-max normalized across
  the result set, so `min_score` is a relative threshold (e.g. `0.5` means
  "top half by relevance").
* **Retention policies** — metadata is auto-tagged on ingestion: permanent
  document types (pdf, docx, txt, markdown, important_chat, other) never expire;
  `chat`/`chat_summary`/`web_cache` expire after 10 days; `session` records
  expire after 1 hour; custom retention requires an explicit `expires_at`.
* **Ranking signals** — when `rank: true`, candidates are scored by a weighted
  fusion of **Nova** vector similarity, BM25 text match, metadata rules,
  importance, and freshness (exponential half-life decay), optionally fused
  with Reciprocal Rank Fusion and a cross-encoder re-ranker.

---

## 📝 Changelog — recent changes

### Normalized 4-table storage + bulk graph load (ORBIT)
* **Split ORBIT storage into four normalized Cosmos tables**, shared by a common
  record `id`:
  * `document_store.py` (**new**) → `nebulon_documents` `{id, text, metadata, created_at}`
  * `nova_store.py` → `nebulon_nova` `{id, vector, created_at}` (dropped embedded
    text/metadata)
  * `mesh_store.py` → per-row `nebulon_mesh_nodes` `{id, label, created_at}` and
    `nebulon_mesh_edges` `{edge_id, from_id, to_id, relation, weight, created_at}`
    (replacing the old single-master-document Mesh persistence)
* **Auto edge weight** — when `weight` is omitted, an edge's weight is computed
  as the cosine similarity of its two endpoint vectors (`1.0` when either
  endpoint has no vector).
* **Name / label node resolution** — `source`/`target` edge endpoints and node
  refs can be numeric `id`s **or** string labels; unknown labels auto-create a
  node. A `name`/`label` column becomes each record's node label.
* **New `POST /segment/mesh_load_graph`** — bulk node + edge load (Option A);
  `mesh_viz.py` label fallback reads `label`/`name`/`Node-{id}`.
* **No spurious `segment_default`** — `CorpusManager.create_corpus` no longer
  pre-initializes an ORBIT manager, so a `default` Nova segment directory is no
  longer created at corpus creation; segments are built lazily on first load.
  (Also pinned `mesh_store.load` to read the normalized `id` key.

### Cosmos in the ingestion / retrieval flow
* **`db/index_manager.py` — `SegmentManager.load_segment` now branches on `ndb_type`:**
  * `orbit` → original vector path (text → embeddings → `insert_vec`, Mesh
    relation loading, `initialize_or_flush`) — unchanged.
  * `cosmos` → **direct document insert** (no embedding): each non-empty text
    row is stored via `insert_data(segment, document)` with `text` / `lang` /
    `type` / `created_at` metadata, then the backend is left to persist via the
    engine's threshold / background flush (no forced `flush()`).
* **New `SegmentManager.get_data(limit, include_internal)`** — reads stored
  records from the corpus's own backend:
  * `orbit` → `OrbitDBManager.get_all_records(limit=...)`
  * `cosmos` → `ComosDBManager.read_data(segment, limit=...)`
  An empty list is returned when the requested backend does not match the
  corpus's type.
* **`ComosDBManager.read_data`** and **`OrbitDBManager.get_all_records`** accept
  an optional `limit` argument.
* **`ComosDBManager.delete_data`** now returns the engine's delete result so
  callers can detect a missing record.
* **Removed forced Cosmos `flush()`** in `load_segment` — Cosmos now persists
  only through its WAL / threshold / background flush.

### Flush cleanup
* **`services/user_service.py`** — removed the three root-level `db.flush()`
  calls (create / update / delete user). User documents now persist via
  Cosmos's threshold / background flush. A fresh `UserManager` still reads
  created users back, confirming durability.

### API layer (`api/routes/segment.py`, `utils/models.py`)
* **`SegmentQueryRequest`** gains `ndb_type` (default `orbit`) and `limit`.
* **`load_segment`** passes `ndb_type` through to `SegmentManager`, so a cosmos
  corpus loads documents directly instead of trying to embed.
* **New `POST /segment/get_data`** — retrieves stored records with an optional
  `limit`; returns `{records, total_count, limit}`.
* **`_build_orbit`** honors `ndb_type` so it resolves the correct backend
  manager (Orbit or Cosmos).
* **`delete_record`** now supports **Cosmos** segments (`delete_data`) in
  addition to Orbit (`delete_record` + flush).

### Search serialization fix
* **`db/index_manager.py`** — normalized `score` values are converted to plain
  Python `float` (previously `numpy.float32`), fixing
  `PydanticSerializationError` when `/search_segment` returned JSON.

### Tests
* **`tests/test_cosmos_dual_load.py`** — validates COSMOS direct load +
  `get_data` with `limit`, and ORBIT vector load + `get_data`.
* **`tests/test_user_noflush.py`** — validates user CRUD durability without
  explicit `flush()`.
* **`tests/test_full.py`** — end-to-end API walkthrough (register → change
  password → verify → create ORBIT/COSMOS corpora → load segments → get_data →
  delete record (Cosmos + Orbit) → list/search segments). Run:
  ```bash
  cd tests/ndb_host
  PYTHONPATH=/home/sathyaprakash/CodeBase/tests:/home/sathyaprakash/CodeBase/tests/ndb_host \
      python3 tests/test_full.py
  ```
