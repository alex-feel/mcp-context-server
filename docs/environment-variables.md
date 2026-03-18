# Environment Variables Reference

## Introduction

All configuration is via environment variables, set in a `.env` file, directly in the shell, or in your MCP client configuration (`env` block in `.mcp.json`). The canonical source of truth for all variables is [`app/settings.py`](../app/settings.py).

MCP clients such as Claude Code support environment variable expansion using `${VAR}` or `${VAR:-default}` syntax. For details, see: <https://docs.claude.com/en/docs/claude-code/mcp#environment-variable-expansion-in-mcp-json>

## Core Settings

| Variable            | Type    | Default                     | Description                                                                          |
|---------------------|---------|-----------------------------|--------------------------------------------------------------------------------------|
| `LOG_LEVEL`         | string  | `ERROR`                     | Application log level. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`      |
| `STORAGE_BACKEND`   | string  | `sqlite`                    | Database backend. Options: `sqlite`, `postgresql`                                    |
| `DB_PATH`           | path    | `~/.mcp/context_storage.db` | SQLite database file location. Only used when `STORAGE_BACKEND=sqlite`               |
| `MAX_IMAGE_SIZE_MB` | integer | `10`                        | Maximum size per image attachment in megabytes                                       |
| `MAX_TOTAL_SIZE_MB` | integer | `100`                       | Maximum total request size in megabytes                                              |
| `DISABLED_TOOLS`    | string  | _(empty)_                   | Comma-separated list of MCP tools to disable (e.g., `delete_context,update_context`) |

## Transport Settings

| Variable                 | Type    | Default   | Constraints | Description                                                                                                         |
|--------------------------|---------|-----------|-------------|---------------------------------------------------------------------------------------------------------------------|
| `MCP_TRANSPORT`          | string  | `stdio`   |             | Transport mode. Options: `stdio`, `http`, `streamable-http`, `sse`                                                  |
| `FASTMCP_HOST`           | string  | `0.0.0.0` |             | HTTP bind address. Use `0.0.0.0` for Docker                                                                         |
| `FASTMCP_PORT`           | integer | `8000`    | 1-65535     | HTTP port number                                                                                                    |
| `FASTMCP_STATELESS_HTTP` | boolean | `true`    |             | Enable stateless HTTP mode for horizontal scaling. Set to `false` only if you need server-side MCP session tracking |

## Authentication Settings

| Variable             | Type   | Default      | Description                                                                          |
|----------------------|--------|--------------|--------------------------------------------------------------------------------------|
| `MCP_AUTH_PROVIDER`  | string | `none`       | Authentication provider. Options: `none` (no auth), `simple_token` (bearer token)    |
| `MCP_AUTH_TOKEN`     | secret | _(none)_     | Bearer token for HTTP authentication. Required when `MCP_AUTH_PROVIDER=simple_token` |
| `MCP_AUTH_CLIENT_ID` | string | `mcp-client` | Client ID assigned to authenticated requests. Used with `simple_token` provider      |

For detailed authentication setup, see the [Authentication Guide](authentication.md).

## Server Instructions

| Variable                  | Type   | Default              | Description                                                                                                                                                     |
|---------------------------|--------|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `MCP_SERVER_INSTRUCTIONS` | string | _(built-in default)_ | Custom server instructions text sent to MCP clients during initialization. Overrides the built-in default. Set to empty string to disable instructions entirely |

## Ollama Settings (Shared)

| Variable      | Type   | Default                  | Description                                                                             |
|---------------|--------|--------------------------|-----------------------------------------------------------------------------------------|
| `OLLAMA_HOST` | string | `http://localhost:11434` | Ollama server URL. Shared by all features using Ollama (embeddings, summary generation) |

## Embedding Settings

### General

| Variable                       | Type    | Default                | Constraints        | Description                                                                                                                                                          |
|--------------------------------|---------|------------------------|--------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_EMBEDDING_GENERATION`  | boolean | `true`                 |                    | Enable embedding generation for stored context entries. If `true` and dependencies are not met, server will NOT start. Set to `false` to disable embeddings entirely |
| `EMBEDDING_PROVIDER`           | string  | `ollama`               |                    | Embedding provider. Options: `ollama`, `openai`, `azure`, `huggingface`, `voyage`                                                                                    |
| `EMBEDDING_MODEL`              | string  | `qwen3-embedding:0.6b` |                    | Embedding model name (provider-specific)                                                                                                                             |
| `EMBEDDING_DIM`                | integer | `1024`                 | 1-4096, must be >0 | Embedding vector dimensions. Changing after initial setup requires database migration                                                                                |
| `EMBEDDING_TIMEOUT_S`          | float   | `240.0`                | >0, <=300          | Timeout in seconds for embedding generation API calls                                                                                                                |
| `EMBEDDING_RETRY_MAX_ATTEMPTS` | integer | `5`                    | 1-10               | Maximum number of retry attempts for embedding generation                                                                                                            |
| `EMBEDDING_RETRY_BASE_DELAY_S` | float   | `1.0`                  | >0, <=30           | Base delay in seconds between retry attempts (with exponential backoff)                                                                                              |
| `EMBEDDING_MAX_CONCURRENT`     | integer | `3`                    | 1-20               | Maximum concurrent embedding generation operations. Limits parallel provider requests to prevent overload                                                            |

For detailed embedding setup and provider selection, see the [Semantic Search Guide](semantic-search.md).

### Ollama-Specific Embedding Settings

| Variable                    | Type    | Default | Constraints | Description                                                                                                                                                                   |
|-----------------------------|---------|---------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `EMBEDDING_OLLAMA_NUM_CTX`  | integer | `4096`  | 512-2097152 | Ollama embedding context length in tokens. Must match or exceed model capabilities                                                                                            |
| `EMBEDDING_OLLAMA_TRUNCATE` | boolean | `false` |             | Control text truncation when exceeding embedding context length. `false`: returns error on exceeded context. `true`: silently truncates input (may degrade embedding quality) |

### OpenAI-Specific Embedding Settings

| Variable              | Type   | Default  | Description                                |
|-----------------------|--------|----------|--------------------------------------------|
| `OPENAI_API_KEY`      | secret | _(none)_ | OpenAI API key                             |
| `OPENAI_API_BASE`     | string | _(none)_ | Custom base URL for OpenAI-compatible APIs |
| `OPENAI_ORGANIZATION` | string | _(none)_ | OpenAI organization ID                     |

### Azure OpenAI-Specific Embedding Settings

| Variable                                 | Type   | Default      | Description                            |
|------------------------------------------|--------|--------------|----------------------------------------|
| `AZURE_OPENAI_API_KEY`                   | secret | _(none)_     | Azure OpenAI API key                   |
| `AZURE_OPENAI_ENDPOINT`                  | string | _(none)_     | Azure OpenAI endpoint URL              |
| `AZURE_OPENAI_API_VERSION`               | string | `2024-02-01` | Azure OpenAI API version               |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME` | string | _(none)_     | Azure OpenAI embedding deployment name |

### HuggingFace-Specific Embedding Settings

| Variable                   | Type   | Default  | Description               |
|----------------------------|--------|----------|---------------------------|
| `HUGGINGFACEHUB_API_TOKEN` | secret | _(none)_ | HuggingFace Hub API token |

### Voyage AI-Specific Embedding Settings

| Variable            | Type    | Default  | Constraints | Description                                                                                                     |
|---------------------|---------|----------|-------------|-----------------------------------------------------------------------------------------------------------------|
| `VOYAGE_API_KEY`    | secret  | _(none)_ |             | Voyage AI API key                                                                                               |
| `VOYAGE_TRUNCATION` | boolean | `false`  |             | Control text truncation when exceeding context length. `false`: returns error. `true`: silently truncates input |
| `VOYAGE_BATCH_SIZE` | integer | `7`      | 1-128       | Number of texts per API call                                                                                    |

## Summary Generation Settings

| Variable                     | Type    | Default              | Constraints | Description                                                                                                                                                       |
|------------------------------|---------|----------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_SUMMARY_GENERATION`  | boolean | `true`               |             | Enable summary generation for stored context entries. If `true` and dependencies are not met, server will NOT start. Set to `false` to disable summaries entirely |
| `SUMMARY_PROVIDER`           | string  | `ollama`             |             | Summary provider. Options: `ollama`, `openai`, `anthropic`                                                                                                        |
| `SUMMARY_MODEL`              | string  | `qwen3:0.6b`         |             | Summary generation model name. Alternatives: `qwen3:1.7b`, `qwen3:4b`, `qwen3:8b`                                                                                 |
| `SUMMARY_MAX_TOKENS`         | integer | `2000`               | 50-5000     | Maximum output tokens for summary generation. Acts as a safety ceiling passed to the LLM API                                                                      |
| `SUMMARY_TIMEOUT_S`          | float   | `240.0`              | >0, <=300   | Timeout in seconds for summary generation API calls                                                                                                               |
| `SUMMARY_RETRY_MAX_ATTEMPTS` | integer | `5`                  | 1-10        | Maximum number of retry attempts for summary generation                                                                                                           |
| `SUMMARY_RETRY_BASE_DELAY_S` | float   | `1.0`                | >0, <=30    | Base delay in seconds between retry attempts (with exponential backoff)                                                                                           |
| `SUMMARY_MAX_CONCURRENT`     | integer | `3`                  | 1-20        | Maximum concurrent summary generation operations                                                                                                                  |
| `SUMMARY_PROMPT`             | string  | _(built-in default)_ |             | Custom summarization system prompt. Overrides the built-in optimized default. For Qwen3 models, include `/no_think` prefix to disable reasoning mode              |
| `SUMMARY_MIN_CONTENT_LENGTH` | integer | `500`                | 0-10000     | Minimum text content length (characters) to trigger summary generation. Content shorter than this is not summarized. Set to `0` to always generate summaries      |
| `SUMMARY_OLLAMA_NUM_CTX`     | integer | `32768`              | 512-2097152 | Ollama summary context length in tokens. Must match or exceed model capabilities                                                                                  |
| `SUMMARY_OLLAMA_TRUNCATE`    | boolean | `false`              |             | Control text truncation when exceeding summary context length. `false`: returns error. `true`: silently truncates input (summary from incomplete text)            |

For detailed summary setup and provider selection, see the [Summary Generation Guide](summary-generation.md).

## Semantic Search Settings

| Variable                 | Type    | Default | Description                                                                           |
|--------------------------|---------|---------|---------------------------------------------------------------------------------------|
| `ENABLE_SEMANTIC_SEARCH` | boolean | `false` | Enable semantic search tool registration. Requires embedding provider to be available |

For detailed semantic search setup, see the [Semantic Search Guide](semantic-search.md).

## Full-Text Search Settings

| Variable       | Type    | Default   | Description                                                                                                                                                                                                                                                                                                                                                                                             |
|----------------|---------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_FTS`   | boolean | `false`   | Enable full-text search functionality                                                                                                                                                                                                                                                                                                                                                                   |
| `FTS_LANGUAGE` | string  | `english` | Language for FTS stemming. PostgreSQL supports 29 languages. Valid options: `simple`, `arabic`, `armenian`, `basque`, `catalan`, `danish`, `dutch`, `english`, `finnish`, `french`, `german`, `greek`, `hindi`, `hungarian`, `indonesian`, `irish`, `italian`, `lithuanian`, `nepali`, `norwegian`, `portuguese`, `romanian`, `russian`, `serbian`, `spanish`, `swedish`, `tamil`, `turkish`, `yiddish` |

For detailed full-text search setup, see the [Full-Text Search Guide](full-text-search.md).

## FTS Passage Extraction Settings

| Variable                 | Type    | Default | Constraints | Description                                                                  |
|--------------------------|---------|---------|-------------|------------------------------------------------------------------------------|
| `FTS_RERANK_WINDOW_SIZE` | integer | `750`   | 100-2000    | Characters of context around each FTS match for reranking passage extraction |
| `FTS_RERANK_GAP_MERGE`   | integer | `100`   | 0-500       | Merge FTS match regions within this character distance                       |

## Hybrid Search Settings

| Variable                  | Type    | Default | Constraints | Description                                                                    |
|---------------------------|---------|---------|-------------|--------------------------------------------------------------------------------|
| `ENABLE_HYBRID_SEARCH`    | boolean | `false` |             | Enable hybrid search combining FTS and semantic search with RRF fusion         |
| `HYBRID_RRF_K`            | integer | `60`    | 1-1000      | RRF smoothing constant. Higher values give more uniform treatment across ranks |
| `HYBRID_RRF_OVERFETCH`    | integer | `2`     | 1-10        | Multiplier for over-fetching results before RRF fusion                         |
| `HYBRID_FTS_OR_THRESHOLD` | integer | `4`     | 2-20        | Minimum number of significant query terms to switch FTS from AND to OR logic   |

For detailed hybrid search setup, see the [Hybrid Search Guide](hybrid-search.md).

## Search Settings

| Variable                   | Type    | Default     | Constraints | Description                                                                    |
|----------------------------|---------|-------------|-------------|--------------------------------------------------------------------------------|
| `SEARCH_DEFAULT_SORT_BY`   | string  | `relevance` |             | Default sort order for search results. Currently only `relevance` is supported |
| `SEARCH_TRUNCATION_LENGTH` | integer | `300`       | 50-1000     | Maximum character length for truncated `text_content` in search results        |

## Chunking Settings

| Variable                | Type    | Default | Constraints                 | Description                                                       |
|-------------------------|---------|---------|-----------------------------|-------------------------------------------------------------------|
| `ENABLE_CHUNKING`       | boolean | `true`  |                             | Enable text chunking for embedding generation                     |
| `CHUNK_SIZE`            | integer | `1500`  | 100-10000                   | Target chunk size in characters                                   |
| `CHUNK_OVERLAP`         | integer | `150`   | 0-500, must be < CHUNK_SIZE | Overlap between chunks in characters                              |
| `CHUNK_AGGREGATION`     | string  | `max`   |                             | Chunk score aggregation method. Currently only `max` is supported |
| `CHUNK_DEDUP_OVERFETCH` | integer | `5`     | 1-20                        | Multiplier for fetching extra chunks before deduplication         |

## Reranking Settings

| Variable                     | Type    | Default                   | Constraints | Description                                                                                                                  |
|------------------------------|---------|---------------------------|-------------|------------------------------------------------------------------------------------------------------------------------------|
| `ENABLE_RERANKING`           | boolean | `true`                    |             | Enable cross-encoder reranking of search results                                                                             |
| `RERANKING_PROVIDER`         | string  | `flashrank`               |             | Reranking provider                                                                                                           |
| `RERANKING_MODEL`            | string  | `ms-marco-MiniLM-L-12-v2` |             | Reranking model name (~34MB)                                                                                                 |
| `RERANKING_MAX_LENGTH`       | integer | `512`                     | 128-2048    | Maximum input length for reranking in tokens                                                                                 |
| `RERANKING_OVERFETCH`        | integer | `4`                       | 1-20        | Multiplier for over-fetching results before reranking                                                                        |
| `RERANKING_CACHE_DIR`        | string  | _(system cache)_          |             | Directory for caching reranking models                                                                                       |
| `RERANKING_CHARS_PER_TOKEN`  | float   | `4.0`                     | 2.0-8.0     | Estimated characters per token for passage size validation. Default `4.0` for English. Use `3.0`-`3.5` for multilingual/code |
| `RERANKING_INTRA_OP_THREADS` | integer | `0`                       | >=0         | ONNX Runtime intra-operation parallelism threads. `0` = auto-detect. In containers, set to match CPU quota                   |
| `RERANKING_CPU_MEM_ARENA`    | boolean | `false`                   |             | Enable ONNX Runtime CPU memory arena. `false`: reduces RAM usage. `true`: slightly faster inference, higher memory           |
| `RERANKING_BATCH_SIZE`       | integer | `32`                      | >0          | Maximum passages per ONNX Runtime inference batch                                                                            |

## LangSmith Tracing Settings

| Variable             | Type    | Default                           | Description                                                  |
|----------------------|---------|-----------------------------------|--------------------------------------------------------------|
| `LANGSMITH_TRACING`  | boolean | `false`                           | Enable LangSmith tracing for cost tracking and observability |
| `LANGSMITH_API_KEY`  | secret  | _(none)_                          | LangSmith API key for tracing                                |
| `LANGSMITH_ENDPOINT` | string  | `https://api.smith.langchain.com` | LangSmith API endpoint                                       |
| `LANGSMITH_PROJECT`  | string  | `mcp-context-server`              | LangSmith project name for grouping traces                   |

## Metadata Indexing Settings

| Variable                   | Type   | Default                                                                                | Description                                                                                                                                                                                                                                               |
|----------------------------|--------|----------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `METADATA_INDEXED_FIELDS`  | string | `status,agent_name,task_name,project,report_type,references:object,technologies:array` | Comma-separated list of metadata fields to index with optional type hints (`field:type` format). Supported types: `string` (default), `integer`, `boolean`, `float`, `array`, `object`. Array/object types use PostgreSQL GIN indexes (skipped in SQLite) |
| `METADATA_INDEX_SYNC_MODE` | string | `additive`                                                                             | How to handle index mismatches at startup. Options: `strict` (fail if mismatch), `auto` (sync: add missing, drop extra), `warn` (log warnings), `additive` (add missing, never drop)                                                                      |

For detailed metadata usage, see the [Metadata Guide](metadata-addition-updating-and-filtering.md).

## SQLite Connection Pool Settings

These settings apply only when `STORAGE_BACKEND=sqlite`.

| Variable                       | Type    | Default | Description                                                        |
|--------------------------------|---------|---------|--------------------------------------------------------------------|
| `POOL_MAX_READERS`             | integer | `8`     | Maximum number of reader connections in the SQLite connection pool |
| `POOL_MAX_WRITERS`             | integer | `1`     | Maximum number of writer connections in the SQLite connection pool |
| `POOL_CONNECTION_TIMEOUT_S`    | float   | `10.0`  | Connection acquisition timeout in seconds                          |
| `POOL_IDLE_TIMEOUT_S`          | float   | `300.0` | Idle connection timeout in seconds                                 |
| `POOL_HEALTH_CHECK_INTERVAL_S` | float   | `30.0`  | Interval between connection health checks in seconds               |

## SQLite Retry Settings

These settings apply only when `STORAGE_BACKEND=sqlite`.

| Variable               | Type    | Default | Description                                                    |
|------------------------|---------|---------|----------------------------------------------------------------|
| `RETRY_MAX_RETRIES`    | integer | `5`     | Maximum number of retry attempts for transient database errors |
| `RETRY_BASE_DELAY_S`   | float   | `0.5`   | Base delay in seconds between retries                          |
| `RETRY_MAX_DELAY_S`    | float   | `10.0`  | Maximum delay in seconds between retries                       |
| `RETRY_JITTER`         | boolean | `true`  | Add random jitter to retry delays to prevent thundering herd   |
| `RETRY_BACKOFF_FACTOR` | float   | `2.0`   | Exponential backoff multiplier for retry delays                |

## SQLite PRAGMA Settings

These settings apply only when `STORAGE_BACKEND=sqlite`.

| Variable                    | Type    | Default     | Description                                                                                   |
|-----------------------------|---------|-------------|-----------------------------------------------------------------------------------------------|
| `SQLITE_FOREIGN_KEYS`       | boolean | `true`      | Enable foreign key enforcement                                                                |
| `SQLITE_JOURNAL_MODE`       | string  | `WAL`       | SQLite journal mode. `WAL` recommended for concurrent reads                                   |
| `SQLITE_SYNCHRONOUS`        | string  | `NORMAL`    | SQLite synchronous mode. `NORMAL` balances safety and performance                             |
| `SQLITE_TEMP_STORE`         | string  | `MEMORY`    | Where to store temporary tables. `MEMORY` for better performance                              |
| `SQLITE_MMAP_SIZE`          | integer | `268435456` | Memory-mapped I/O size in bytes. Default: 256MB                                               |
| `SQLITE_CACHE_SIZE`         | integer | `-64000`    | SQLite page cache size. Negative value = kilobytes. Default: -64000 (64MB)                    |
| `SQLITE_PAGE_SIZE`          | integer | `4096`      | SQLite page size in bytes                                                                     |
| `SQLITE_WAL_AUTOCHECKPOINT` | integer | `1000`      | Number of WAL frames before auto-checkpoint                                                   |
| `SQLITE_BUSY_TIMEOUT_MS`    | integer | _(derived)_ | SQLite busy timeout in milliseconds. Default: derived from `POOL_CONNECTION_TIMEOUT_S * 1000` |
| `SQLITE_WAL_CHECKPOINT`     | string  | `PASSIVE`   | WAL checkpoint mode                                                                           |

## SQLite Circuit Breaker Settings

These settings apply only when `STORAGE_BACKEND=sqlite`.

| Variable                              | Type    | Default | Description                                                  |
|---------------------------------------|---------|---------|--------------------------------------------------------------|
| `CIRCUIT_BREAKER_FAILURE_THRESHOLD`   | integer | `10`    | Number of consecutive failures before circuit opens          |
| `CIRCUIT_BREAKER_RECOVERY_TIMEOUT_S`  | float   | `30.0`  | Seconds to wait before attempting recovery from open circuit |
| `CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS` | integer | `5`     | Maximum test calls allowed in half-open state                |

## SQLite Operation Timeout Settings

These settings apply only when `STORAGE_BACKEND=sqlite`.

| Variable                  | Type  | Default | Description                                          |
|---------------------------|-------|---------|------------------------------------------------------|
| `SHUTDOWN_TIMEOUT_S`      | float | `10.0`  | Graceful shutdown timeout in seconds                 |
| `SHUTDOWN_TIMEOUT_TEST_S` | float | `5.0`   | Shutdown timeout for test environments in seconds    |
| `QUEUE_TIMEOUT_S`         | float | `1.0`   | Write queue timeout in seconds                       |
| `QUEUE_TIMEOUT_TEST_S`    | float | `0.1`   | Write queue timeout for test environments in seconds |

## PostgreSQL Connection Settings

These settings apply only when `STORAGE_BACKEND=postgresql`.

| Variable                       | Type    | Default       | Description                                                                                                      |
|--------------------------------|---------|---------------|------------------------------------------------------------------------------------------------------------------|
| `POSTGRESQL_CONNECTION_STRING` | secret  | _(none)_      | Full PostgreSQL connection string. When provided, overrides individual host/port/user/password/database settings |
| `POSTGRESQL_HOST`              | string  | `localhost`   | PostgreSQL server host                                                                                           |
| `POSTGRESQL_PORT`              | integer | `5432`        | PostgreSQL server port                                                                                           |
| `POSTGRESQL_USER`              | string  | `postgres`    | PostgreSQL username                                                                                              |
| `POSTGRESQL_PASSWORD`          | secret  | `postgres`    | PostgreSQL password                                                                                              |
| `POSTGRESQL_DATABASE`          | string  | `mcp_context` | PostgreSQL database name                                                                                         |
| `POSTGRESQL_SSL_MODE`          | string  | `prefer`      | PostgreSQL SSL mode. Options: `disable`, `allow`, `prefer`, `require`, `verify-ca`, `verify-full`                |
| `POSTGRESQL_SCHEMA`            | string  | `public`      | PostgreSQL schema name for table and index operations                                                            |

For detailed PostgreSQL setup and Supabase integration, see the [Database Backends Guide](database-backends.md).

## PostgreSQL Connection Pool Settings

These settings apply only when `STORAGE_BACKEND=postgresql`.

| Variable                         | Type    | Default | Constraints | Description                                                                                            |
|----------------------------------|---------|---------|-------------|--------------------------------------------------------------------------------------------------------|
| `POSTGRESQL_POOL_MIN`            | integer | `2`     |             | Minimum connections in the asyncpg connection pool                                                     |
| `POSTGRESQL_POOL_MAX`            | integer | `20`    |             | Maximum connections in the asyncpg connection pool                                                     |
| `POSTGRESQL_POOL_TIMEOUT_S`      | float   | `120.0` |             | Connection acquisition timeout in seconds                                                              |
| `POSTGRESQL_COMMAND_TIMEOUT_S`   | float   | `60.0`  |             | Default command timeout in seconds                                                                     |
| `POSTGRESQL_MIGRATION_TIMEOUT_S` | float   | `300.0` | >0, <=3600  | Timeout in seconds for migration DDL operations (CREATE INDEX, ALTER TABLE). Default: 300s (5 minutes) |

## PostgreSQL Connection Pool Hardening

These settings apply only when `STORAGE_BACKEND=postgresql`.

| Variable                             | Type    | Default | Constraints | Description                                                    |
|--------------------------------------|---------|---------|-------------|----------------------------------------------------------------|
| `POSTGRESQL_MAX_INACTIVE_LIFETIME_S` | float   | `300.0` | >=0         | Close idle connections after this many seconds. `0` to disable |
| `POSTGRESQL_MAX_QUERIES`             | integer | `10000` | >=0         | Recycle connections after this many queries. `0` to disable    |

## PostgreSQL TCP Keepalive Settings

These settings apply only when `STORAGE_BACKEND=postgresql`.

| Variable                               | Type    | Default | Constraints | Description                                                                                |
|----------------------------------------|---------|---------|-------------|--------------------------------------------------------------------------------------------|
| `POSTGRESQL_TCP_KEEPALIVES_IDLE_S`     | integer | `15`    | >=0         | Seconds of idle time before sending first TCP keepalive probe. `0` to disable              |
| `POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S` | integer | `5`     | >=0         | Seconds between subsequent TCP keepalive probes. `0` to disable                            |
| `POSTGRESQL_TCP_KEEPALIVES_COUNT`      | integer | `3`     | >=0         | Number of failed TCP keepalive probes before connection is considered dead. `0` to disable |

## PostgreSQL Prepared Statement Cache Settings

These settings apply only when `STORAGE_BACKEND=postgresql`.

| Variable                                     | Type    | Default | Constraints | Description                                                                                                                      |
|----------------------------------------------|---------|---------|-------------|----------------------------------------------------------------------------------------------------------------------------------|
| `POSTGRESQL_STATEMENT_CACHE_SIZE`            | integer | `100`   | 0-10000     | asyncpg prepared statement cache size. Set to `0` when using external connection poolers (PgBouncer transaction mode, Pgpool-II) |
| `POSTGRESQL_MAX_CACHED_STATEMENT_LIFETIME_S` | integer | `300`   | 0-86400     | Maximum lifetime of cached prepared statements in seconds. No effect when `POSTGRESQL_STATEMENT_CACHE_SIZE=0`                    |
| `POSTGRESQL_MAX_CACHEABLE_STATEMENT_SIZE`    | integer | `15360` | 0-1048576   | Maximum size of statement to cache in bytes (default: 15KB). No effect when `POSTGRESQL_STATEMENT_CACHE_SIZE=0`                  |
