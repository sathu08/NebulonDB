# NebulonDB Configuration Reference

`nebulondb.cfg` lives at the NDB home root (the directory `NEBULONDB_HOME` points to).
It uses the standard INI format; section and key names are case-insensitive. Every
option below is read at startup by `NDBConfig` (`ndb_host/db/ndb_settings.py`).

> Missing optional keys fall back to the documented defaults. Two values are
> **required**: `nebulondb_home` (`[paths]`) and `nebulondb_master_key`
> (`[environment]` — auto-generated with a warning if absent).

---

## [paths]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nebulondb_home` | str | — (required) | NDB installation root. All storage/log/web paths resolve from here. Overridable via the `NEBULONDB_HOME` environment variable. |

## [environment]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nebulondb_master_key` | str | — (required) | Fernet master key used by `NDBCryptoManager` to protect on-disk secrets. If missing, a new one is generated and written back with a warning. |
| `nebulondb_keyring_enabled` | bool | `false` | Store the master key in the OS keyring instead of the config file. |
| `nebulondb_keyring_service` | str | `nebulondb-auth-service` | Keyring service name used when keyring is enabled. |

## [llm]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `nebulondb_embedding_model` | str | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace repo id of the **Nova** embedding model. First use downloads it into the model cache dir. |
| `nebulondb_cross_encoder_model` | str | `cross-encoder/ms-marco-MiniLM-L-6-v2` | **Mesh/rerank** cross-encoder used by hybrid ranking. Lazy-loaded — only loaded on the first rerank. |
| `nebulondb_model_cache_dir` | str | platform user cache dir | Directory for HuggingFace model snapshots. Auto-created and written back if missing. |
| `nebulondb_embedding_batch_size` | int | `16` | Embedding encode batch size. Auto-tuned at startup by `get_auto_batch_size`. |
| `nebulondb_cross_encoder_batch_size` | int | `8` | Cross-encoder predict batch size. |
| `nebulondb_embedding_model_device` | str | `cpu` | Device for the embedding model (`cpu` / `cuda`). |
| `nebulondb_cross_encoder_model_device` | str | `cpu` | Device for the cross-encoder (`cpu` / `cuda`). |
| `nebulondb_warm_models` | bool | `true` | Warm the embedding model in the background (one dummy encode) in every worker after server start, so the first real search/insert is fast. Set `false` to skip warmup. |
| `nebulondb_default_mode` | bool | `false` | Force CPU mode and disable CUDA detection at startup (safe fallback on GPU-less machines). |

## [segments]
Cosmos storage-engine (memtable / WAL / immutable segment) tuning.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `wal_auto_flush` | bool | `true` | `fsync` the WAL on every write vs. batching fsyncs. |
| `wal_fsync_interval` | int | `65536` | Bytes written before a forced WAL fsync when auto-flush is off. |
| `compress_segments` | bool | `true` | zlib-compress immutable segment files. |
| `bloom_filter_enabled` | bool | `true` | Attach Bloom filters to segments for fast negative lookups. |
| `max_open_segments` | int | `50` | LRU cap on concurrently open segment files. |
| `compaction_interval` | float | `60` | Seconds between automatic compaction checks. |
| `max_segments_before_compact` | int | `10` | Compaction triggers once segment count exceeds this. |
| `flush_interval` | float | `5` | Seconds between automatic memtable flushes. |
| `flush_record_threshold` | int | `10000` | Record count that forces a memtable flush. |

## [bloom]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | `true` | Enable Bloom filters for newly created corpora. |
| `bits_per_key` | int | `10` | Bloom filter bits allocated per key. |
| `hash_count` | int | `4` | Number of hash functions per Bloom filter. |

## [vector]
Nova (HNSW) index defaults applied to newly created corpora.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `dimension` | int | `384` | Embedding dimension. Must match the embedding model output. |
| `space` | str | `cosine` | Distance metric (`cosine`, `l2`, `ip`). |
| `top_matches` | int | `3` | Default top-k results per search. |
| `min_score` | float | `0.50` | Minimum similarity score for results to be returned. |
| `save_every_n` | int | `100` | Save the index state every N mutations. |
| `compaction_threshold` | float | `0.4` | Deleted-ratio above which the HNSW index is compacted. |

## [hnsw]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `m` | int | `32` | Maximum connections per layer in the HNSW graph. |
| `ef_construction` | int | `200` | Build-time search width (higher = better recall, slower insert). |
| `ef_search` | int | `64` | Query-time search width (higher = better recall, slower query). |

## [rank]
Hybrid ranking weights (Nova similarity + BM25 + metadata + importance + freshness).
Weights should sum to 1.0.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `rank_topk` | int | `20` | Number of candidates reranked per query. |
| `weight_vector` | float | `0.55` | Nova vector similarity weight. |
| `weight_bm25` | float | `0.20` | BM25 keyword match weight. |
| `weight_metadata` | float | `0.10` | Metadata match weight. |
| `weight_importance` | float | `0.10` | Record importance weight. |
| `weight_freshness` | float | `0.05` | Record freshness (age) weight. |

## [logging]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `date` | int | `7` | Log file retention in days. |
| `auto_delete` | bool | `true` | Auto-delete rotated log files older than `date` days. |

## [server]

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `app_name` | str | `NebulonDB` | Display name used by the CLI banner. |
| `host` | str | `0.0.0.0` | Bind address. |
| `port` | int | `6969` | Bind port. |
| `workers` | int | `1` | Gunicorn worker processes. Each worker loads its own model instances. |
| `timeout` | int | `120` | Gunicorn worker timeout (seconds). |
| `keep_alive` | int | `5` | HTTP keep-alive seconds. |
| `graceful_timeout` | int | `10` | Gunicorn graceful shutdown timeout. |
| `access_logfile` | str | `-` (stdout) | Access log target (file path). |
| `error_logfile` | str | `-` (stdout) | Error log target (file path). |
| `log_level` | str | `info` | Gunicorn log level (`debug`/`info`/`warning`/`error`). |
| `nebulondb_clear_cache` | bool | `true` | Purge `__pycache__`/`.pyc` bytecode before starting the server. Set `false` to start faster (stale caches are normally harmless — Python recompiles on `.py` changes automatically). |

---

## Environment overrides

| Variable | Effect |
|----------|--------|
| `NEBULONDB_HOME` | Overrides `nebulondb_home` at load time (written back only if different). |
| `NEBULONDB_MASTER_KEY` | Overrides the master key; otherwise the configured (or generated) one is used. |

## Default `nebulondb.cfg`

> For the newest values, always check the file shipped in the repo root. The
> complete default file is maintained as `nebulondb.cfg` at the repository root;
> run `nebulondb --create-user`/`nebulondb start` after edits.