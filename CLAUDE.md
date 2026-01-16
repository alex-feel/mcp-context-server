# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Build and run
uv sync                                    # Install dependencies
uv run mcp-context-server                  # Start server (aliases: mcp-context, python -m app.server)
uvx mcp-context-server                     # Run from PyPI

# Testing
uv run pytest                              # Run all tests
uv run pytest tests/test_server.py -v      # Run specific test file
uv run pytest tests/test_server.py::TestStoreContext::test_store_text_context -v  # Single test
uv run pytest --cov=app --cov-report=html  # Run with coverage
uv run pytest -m "not integration"         # Skip slow tests for quick feedback

# Code quality
uv run pre-commit run --all-files          # Lint + type check (Ruff, mypy, pyright)
uv run ruff check --fix .                  # Ruff linter with autofix
```

Note: Integration tests use SQLite-only temporary databases. PostgreSQL is production-only.

## High-Level Architecture

### MCP Protocol Integration

[Model Context Protocol](https://modelcontextprotocol.io) (MCP) server with JSON-RPC 2.0, automatic tool discovery, Pydantic validation, multi-transport (stdio/HTTP/streamable-http/SSE), and tool annotations (readOnlyHint, destructiveHint, idempotentHint). Compatible with Claude Desktop, Claude Code, LangGraph, and any MCP client.

### MCP Server Architecture

FastMCP 2.0-based server providing persistent context storage for LLM agents:

1. **FastMCP Server Layer** (`app/server.py`, `app/tools/`, `app/startup/`):
   - Entry point with FastMCP instance, lifespan management, and main() function (~550 lines)
   - Tool implementations in `app/tools/` package organized by domain:
     - `context.py`: store_context, get_context_by_ids, update_context, delete_context
     - `search.py`: search_context, semantic_search_context, fts_search_context, hybrid_search_context
     - `discovery.py`: list_threads, get_statistics
     - `batch.py`: store_context_batch, update_context_batch, delete_context_batch
   - Dynamic tool registration via `register_tool()` from `app/tools/__init__.py`
   - Supports multiple transports: stdio (default), HTTP, streamable-http, SSE
   - Provides `/health` endpoint for container orchestration (HTTP transport only)
   - Global state and initialization in `app/startup/` package
   - Database initialization via `init_database()` from `app.startup`
   - Repository access via `ensure_repositories()` from `app.startup`

2. **Authentication Layer** (`app/auth/`):
   - **SimpleTokenVerifier** (`simple_token.py`): Bearer token auth for HTTP transport with constant-time comparison
   - Configured via `FASTMCP_SERVER_AUTH` and `MCP_AUTH_TOKEN`; settings in `AuthSettings`

3. **Storage Backend Layer** (`app/backends/`):
   - **StorageBackend Protocol** (`base.py`): Database-agnostic interface (7 methods)
   - **SQLiteBackend**: Connection pooling, write queue, circuit breaker
   - **PostgreSQLBackend**: Async via asyncpg, connection pooling, MVCC, JSONB
   - **Backend Factory** (`factory.py`): Creates backend based on `STORAGE_BACKEND` env var

4. **Repository Pattern** (`app/repositories/`):
   - **RepositoryContainer**: DI container for all repositories
   - **ContextRepository**: CRUD, search, deduplication, metadata filtering
   - **TagRepository**: Tag normalization, many-to-many relationships
   - **ImageRepository**: Multimodal image attachments
   - **StatisticsRepository**: Thread statistics and DB metrics
   - **EmbeddingRepository**: Vector embeddings for semantic search
   - **FtsRepository**: Full-text search (FTS5/tsvector)
   - All repositories use `StorageBackend` protocol - database-agnostic

5. **Data Models** (`app/models.py`):
   - Pydantic V2 with `StrEnum` for Python 3.12+
   - Main models: `ContextEntry`, `ImageAttachment`, `StoreContextRequest`
   - Base64 image encoding with configurable size limits

6. **Embeddings Layer** (`app/embeddings/`):
   - **EmbeddingProvider Protocol** (`base.py`): `initialize()`, `shutdown()`, `embed_query()`, `embed_documents()`, `is_available()`, `get_dimension()`, `provider_name`
   - **EmbeddingFactory** (`factory.py`): Dynamic import via `PROVIDER_MODULES`/`PROVIDER_CLASSES` dicts
   - **Providers** (`providers/`): LangChain-based (Ollama, OpenAI, Azure, HuggingFace, Voyage)
   - **Retry** (`retry.py`): `with_retry_and_timeout()` using tenacity, handles transient errors across all providers
   - **Context Limits** (`context_limits.py`): `EmbeddingModelSpec` dataclass, `EMBEDDING_MODEL_SPECS` registry

7. **Reranking Layer** (`app/reranking/`):
   - **RerankingProvider Protocol** (`base.py`): `initialize()`, `shutdown()`, `rerank()`, `is_available()`, `provider_name`, `model_name`
   - **RerankingFactory** (`factory.py`): Same dynamic import pattern as embeddings
   - **Providers** (`providers/`): FlashRank (default, 34MB model)

8. **Services Layer** (`app/services/`):
   - **ChunkingService** (`chunking_service.py`): `TextChunk` dataclass, `split_text()` with boundary tracking, LangChain's `RecursiveCharacterTextSplitter`
   - **Passage Extraction** (`passage_extraction_service.py`): `extract_rerank_passage()` for FTS reranking, `HighlightRegion` dataclass, boundary expansion/merging

9. **Metadata Filtering** (`app/metadata_types.py` & `app/query_builder.py`):
   - **MetadataFilter**: 16 operators (eq, ne, gt, lt, contains, etc.)
   - **QueryBuilder**: Backend-aware SQL with nested JSON paths (e.g., "user.preferences.theme")
   - Handles SQLite (`json_extract`) vs PostgreSQL (`->>`/`->`) operators

10. **Fusion Algorithms** (`app/fusion.py`):
    - `reciprocal_rank_fusion()`: RRF algorithm combining FTS + semantic results
    - `count_unique_results()`: Overlap statistics for hybrid search

11. **Database Layer** (`app/schemas/`):
   - 3 tables: `context_entries`, `tags`, `image_attachments`
   - SQLite: JSON, BLOB, WAL mode | PostgreSQL: JSONB, BYTEA, MVCC
   - Thread-scoped isolation, strategic indexing, cascade deletes

### Modular Package Structure

The server codebase is organized into focused packages:

```
app/
├── server.py              # Entry point, lifespan, FastMCP (~550 lines)
├── settings.py            # ALL env vars via get_settings() - centralized configuration
├── types.py               # 40+ TypedDicts for API responses (ScoresDict, ContextEntryDict, etc.)
├── models.py              # Pydantic models (ContextEntry, ImageAttachment, StoreContextRequest)
├── fusion.py              # RRF fusion algorithm for hybrid search
├── tools/                 # MCP tool implementations
│   ├── __init__.py       # Tool registration, TOOL_ANNOTATIONS
│   ├── context.py        # CRUD: store_context, get_context_by_ids, update_context, delete_context
│   ├── search.py         # search_context, semantic_search_context, fts_search_context, hybrid_search_context
│   ├── discovery.py      # list_threads, get_statistics
│   └── batch.py          # store_context_batch, update_context_batch, delete_context_batch
├── startup/               # Server initialization
│   ├── __init__.py       # init_database(), ensure_repositories(), global state
│   └── validation.py     # Parameter validation utilities
├── backends/              # Storage backends (Protocol pattern)
│   ├── base.py           # StorageBackend Protocol (@runtime_checkable)
│   ├── factory.py        # create_backend() with dynamic import
│   ├── sqlite_backend.py # Connection pooling, write queue, circuit breaker
│   └── postgresql_backend.py  # asyncpg, MVCC, JSONB
├── repositories/          # Data access layer (Repository pattern)
│   ├── base.py           # BaseRepository with _placeholder(), _json_extract() helpers
│   ├── __init__.py       # RepositoryContainer (DI)
│   ├── context_repository.py   # CRUD, search, deduplication
│   ├── embedding_repository.py # Vector storage/search
│   ├── fts_repository.py       # Full-text search (FTS5/tsvector)
│   ├── tag_repository.py       # Many-to-many tags
│   ├── image_repository.py     # Binary attachments
│   └── statistics_repository.py
├── embeddings/            # Embedding providers (Protocol pattern)
│   ├── base.py           # EmbeddingProvider Protocol
│   ├── factory.py        # create_embedding_provider() with PROVIDER_MODULES
│   ├── retry.py          # with_retry_and_timeout() using tenacity
│   ├── context_limits.py # EmbeddingModelSpec, EMBEDDING_MODEL_SPECS registry
│   └── providers/        # langchain_ollama.py, langchain_openai.py, etc.
├── reranking/             # Cross-encoder reranking (Protocol pattern)
│   ├── base.py           # RerankingProvider Protocol
│   ├── factory.py        # create_reranking_provider()
│   └── providers/        # flashrank.py
├── services/              # Domain services
│   ├── chunking_service.py       # ChunkingService, TextChunk dataclass
│   └── passage_extraction_service.py  # extract_rerank_passage(), HighlightRegion
├── migrations/            # Auto-applied idempotent migrations
│   ├── semantic.py, fts.py, chunking.py, metadata.py
│   └── dependencies.py, utils.py
├── auth/                  # Authentication
│   └── simple_token.py   # SimpleTokenVerifier for HTTP transport
├── metadata_types.py      # MetadataFilter (16 operators)
├── query_builder.py       # Backend-aware SQL generation
└── schemas/               # SQL schema files (sqlite_schema.sql, postgresql_schema.sql)
```

**Import Patterns:**
- Tools import from `app.startup` for global state and validation
- Tools import from `app.migrations` for migration status
- `app.server` imports from `app.tools` for registration
- Backward compatibility re-exports maintained in `app.server`

### Thread-Based Context Management

Agents share context via `thread_id`. Entries tagged with `source`: 'user' or 'agent'. Filter by thread, source, tags, content type, or metadata (16 operators). Flat structure (no hierarchy).

**Example**: Thread "analyze-q4-sales" - User posts task, Agent 1 fetches data, Agent 2 generates charts, Agent 3 identifies products. All share the same thread_id.

### Database Schema

Tables: `context_entries` (main, with thread_id/source indexes, JSON metadata), `tags` (many-to-many, lowercase), `image_attachments` (binary, cascade delete).

**Performance**: WAL mode, 256MB mmap, compound index (thread_id, source). Indexed metadata: `status`, `agent_name`, `task_name`, `project`, `report_type`. Array/object fields use PostgreSQL GIN (not indexed in SQLite).

### Testing Strategy

**Philosophy**: Tests use SQLite-only temp databases (no PostgreSQL required). Production supports both backends. Always add real server integration tests in `tests/test_real_server.py` for new tools.

**Key Fixtures** (`conftest.py`): `test_db` (direct SQLite), `mock_server_dependencies` (mocked settings), `initialized_server` (full integration), `async_db_initialized` (async backend), `async_db_with_embeddings` (semantic search).

**Skip Markers**: `@requires_ollama`, `@requires_sqlite_vec`, `@requires_numpy`, `@requires_semantic_search`

`prevent_default_db_pollution` (autouse) prevents accidental production DB access.

### Key Implementation Details

1. **Python 3.12+ Type Hints**: `str | None` syntax, `StrEnum`, TypedDicts in `app/types.py`. **NEVER** use `from __future__ import annotations` in server.py (breaks FastMCP).

2. **FastMCP Tool Signatures**: `Literal["user", "agent"]` for source, `Annotated[type, Field(...)]` for docs, `ctx: Context | None = None` as last param (hidden from clients). Returns must be serializable dicts/lists.

3. **Async Operations**: SQLite ops are sync callables wrapped via `execute_write`/`execute_read`. PostgreSQL ops are native async. Repositories detect backend type automatically.

4. **Design Patterns**:
   - **Protocol** (`@runtime_checkable`): `StorageBackend` (backends), `EmbeddingProvider` (embeddings), `RerankingProvider` (reranking)
   - **Repository**: All SQL in `app/repositories/`, `BaseRepository` provides `_placeholder()`, `_placeholders()`, `_json_extract()` helpers
   - **Factory**: `create_backend()`, `create_embedding_provider()`, `create_reranking_provider()` - dynamic imports via `PROVIDER_MODULES` dicts
   - **DI**: `RepositoryContainer` injects all repositories
   - **DTO**: TypedDicts in `app/types.py` (40+ types: `ContextEntryDict`, `ScoresDict`, `HybridSearchResultDict`, etc.)
   - **Dataclass**: `TextChunk` (chunking), `HighlightRegion` (FTS passage), `EmbeddingModelSpec` (model limits)

5. **Error Handling**: Pydantic validation, DB constraints (CHECK clauses), size limits (10MB/image, 100MB total), transaction rollback on failures.

### Semantic Search Implementation

Optional vector similarity search via `semantic_search_context`. SQLite uses `sqlite-vec` (BLOB, `vec_distance_l2()`), PostgreSQL uses `pgvector` (native vector, `<->` L2 distance).

**Extending**: New providers in `app/embeddings/providers/`. Add to `PROVIDER_MODULES`/`PROVIDER_CLASSES` in factory.py. Implement `EmbeddingProvider` protocol.

### Full-Text Search (FTS) Implementation

Optional linguistic search via `fts_search_context`. SQLite: FTS5 with BM25, Porter stemmer. PostgreSQL: tsvector/tsquery with ts_rank, 29 languages.

**Modes**: `match` (default, stemming), `prefix` (autocomplete), `phrase` (exact order), `boolean` (AND/OR/NOT).

**Search tools comparison**: `search_context` (exact keyword), `semantic_search_context` (vector similarity), `fts_search_context` (linguistic + snippets), `hybrid_search_context` (RRF fusion of FTS + semantic).

**Response**: `results` (array), `count` (int), `stats` (only when `explain_query=True`).

### Migration System

Auto-applied idempotent migrations in `app/migrations/`: semantic search, FTS, chunking (1:N embeddings). Changing `FTS_LANGUAGE` requires FTS table rebuild.

### Hybrid Search Implementation

`hybrid_search_context` combines FTS + semantic via RRF: `score(d) = Σ(1 / (k + rank_i(d)))`. Parallel execution, graceful degradation. Requires `ENABLE_HYBRID_SEARCH=true` + at least one of `ENABLE_FTS`/`ENABLE_SEMANTIC_SEARCH`. `HYBRID_RRF_K` controls fusion smoothing.

## Package and Release

uv + Hatchling. Entry points: `mcp-context-server`, `mcp-context`. Python 3.12+. Optional: `uv sync --extra embeddings-ollama` (or `-openai`, `-azure`, `-huggingface`, `-voyage`).

## Release Process

[Release Please](https://github.com/googleapis/release-please) for automated releases via [Conventional Commits](https://www.conventionalcommits.org/). On `release:published`: PyPI package, MCP Registry (`server.json`), GHCR Docker image (amd64/arm64).

## MCP Registry and server.json Maintenance

`server.json` enables MCP client discovery ([spec](https://raw.githubusercontent.com/modelcontextprotocol/registry/refs/heads/main/docs/reference/server-json/generic-server-json.md)). All env vars must match `app/settings.py` Fields with `alias` parameter. Release Please auto-updates version.

## Environment Variables

Configuration via `.env` file or environment. Full list in `app/settings.py`.

**Core**: `STORAGE_BACKEND` (sqlite*/postgresql), `LOG_LEVEL` (ERROR*), `DB_PATH` (~/.mcp/context_storage.db*), `MAX_IMAGE_SIZE_MB` (10*), `MAX_TOTAL_SIZE_MB` (100*), `DISABLED_TOOLS`

**Transport**: `MCP_TRANSPORT` (stdio*/http/streamable-http/sse), `FASTMCP_HOST` (0.0.0.0*), `FASTMCP_PORT` (8000*)

**Auth**: `FASTMCP_SERVER_AUTH` (verifier class), `MCP_AUTH_TOKEN`, `MCP_AUTH_CLIENT_ID` (mcp-client*)

**FTS**: `ENABLE_FTS` (false*), `FTS_LANGUAGE` (english*; PostgreSQL: 29 languages, SQLite: Porter/unicode61), `FTS_RERANK_WINDOW_SIZE` (750*), `FTS_RERANK_GAP_MERGE` (100*)

**Embedding Generation**: `ENABLE_EMBEDDING_GENERATION` (true*). **BREAKING v1.0.0**: When true and dependencies unavailable, server won't start. Set false to disable.

**Semantic Search**: `ENABLE_SEMANTIC_SEARCH` (false*), `EMBEDDING_PROVIDER` (ollama*/openai/azure/huggingface/voyage), `EMBEDDING_MODEL` (qwen3-embedding:0.6b*), `EMBEDDING_DIM` (1024*), `EMBEDDING_TIMEOUT_S` (30*), `EMBEDDING_RETRY_MAX_ATTEMPTS` (3*), `EMBEDDING_RETRY_BASE_DELAY_S` (1.0*)

**Provider-Specific**:
- Ollama: `OLLAMA_HOST` (localhost:11434*), `OLLAMA_TRUNCATE` (false*), `OLLAMA_NUM_CTX` (4096*)
- OpenAI: `OPENAI_API_KEY`
- Azure: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`
- HuggingFace: `HUGGINGFACEHUB_API_TOKEN`
- Voyage: `VOYAGE_API_KEY`, `VOYAGE_TRUNCATION` (false*)

**Hybrid**: `ENABLE_HYBRID_SEARCH` (false*), `HYBRID_RRF_K` (60*), `HYBRID_RRF_OVERFETCH` (2*)

**Chunking**: `ENABLE_CHUNKING` (true*), `CHUNK_SIZE` (1000*), `CHUNK_OVERLAP` (100*), `CHUNK_AGGREGATION` (max*), `CHUNK_DEDUP_OVERFETCH` (5*). Chunk-aware reranking uses chunk boundaries for cross-encoder scoring.

**Reranking**: `ENABLE_RERANKING` (true*), `RERANKING_PROVIDER` (flashrank*), `RERANKING_MODEL` (ms-marco-MiniLM-L-12-v2*), `RERANKING_MAX_LENGTH` (512*), `RERANKING_OVERFETCH` (4*), `RERANKING_CACHE_DIR`, `RERANKING_CHARS_PER_TOKEN` (4.0*; 3.0-3.5 for code)

**Search**: `SEARCH_DEFAULT_SORT_BY` (relevance*)

**Metadata Indexing**: `METADATA_INDEXED_FIELDS` (field:type format; default: status,agent_name,task_name,project,report_type,references:object,technologies:array), `METADATA_INDEX_SYNC_MODE` (additive*/strict/auto/warn)

**PostgreSQL** (when STORAGE_BACKEND=postgresql): `POSTGRESQL_CONNECTION_STRING` (overrides individual settings), `_HOST` (localhost*), `_PORT` (5432*), `_USER` (postgres*), `_PASSWORD`, `_DATABASE` (mcp_context*), `_POOL_MIN` (2*), `_POOL_MAX` (20*), `_POOL_TIMEOUT_S` (10*), `_COMMAND_TIMEOUT_S` (60*), `_SSL_MODE` (prefer*), `_SCHEMA` (public*; required for Supabase)

*\* = default value*. Additional tuning: connection pool, retry, circuit breaker settings in `app/settings.py`.


## Storage Backend Configuration

### SQLite (Default)
Zero-config local storage with connection pooling, write queue, circuit breaker. Single-user deployments.

### PostgreSQL
```bash
docker run --name pgvector18 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mcp_context -p 5432:5432 -d pgvector/pgvector:pg18-trixie
export STORAGE_BACKEND=postgresql
uv run mcp-context-server  # Auto-initializes schema, enables pgvector
```
MVCC (10x+ throughput), asyncpg pooling, JSONB/GIN indexes, pgvector. Multi-user/high-traffic.

### Supabase
`STORAGE_BACKEND=postgresql` + `POSTGRESQL_CONNECTION_STRING`. Session Pooler for IPv4. "getaddrinfo failed" = switch from Direct to Session Pooler. Enable pgvector via Dashboard → Extensions.

### Metadata Field Indexing by Backend

SQLite: B-tree via `json_extract` for scalar fields only. PostgreSQL: B-tree for scalars, GIN for arrays/objects.

| Field | SQLite | PostgreSQL |
|-------|--------|------------|
| `status`, `agent_name`, `task_name`, `project`, `report_type` | B-tree | B-tree |
| `references` (object), `technologies` (array) | **NOT INDEXED** | GIN |

**Note**: Array/object queries require full table scan in SQLite. Use PostgreSQL for high-performance containment queries: `WHERE metadata @> '{"technologies": ["python"]}'`

## Docker Deployment

Multi-stage Dockerfile (uv, non-root UID 10001, `/health` endpoint). Configs in `deploy/docker/`: SQLite, PostgreSQL, Supabase. Ollama sidecar in `deploy/docker/ollama/`.

## Kubernetes Deployment

Helm chart in `deploy/helm/mcp-context-server/`. Profiles: `values-sqlite.yaml`, `values-postgresql.yaml`. Optional Ollama sidecar, ingress with TLS.

## Windows Development Notes

Use `Path` objects (not string concat). Env vars: `set VAR=value &&` (cmd) or `$env:VAR="value";` (PowerShell). DB path: `%USERPROFILE%\.mcp\context_storage.db`. Avoid Unix commands in code. Docker Desktop for PostgreSQL (port 5432, check firewall).

## Debugging and Troubleshooting

```bash
set LOG_LEVEL=DEBUG && uv run mcp-context-server  # Debug logs (Windows)
uv run python -c "from app.startup import init_database; import asyncio; asyncio.run(init_database())"  # Test DB
```

**Common Issues**: Import errors → `uv sync`. Type errors → `uv run mypy app`. Semantic search unavailable → `ENABLE_SEMANTIC_SEARCH=true` + `uv sync --extra embeddings-ollama`. FTS unavailable → `ENABLE_FTS=true`.

## Code Quality Standards

Ruff (127 chars, single quotes), mypy/pyright strict for `app/`. **Never** `from __future__ import annotations` in server.py.

## Critical Implementation Warnings

### Environment Variables - Centralized Configuration

**Never use `os.environ`/`os.getenv()` directly** - always `get_settings()` from `app/settings.py`.

Settings classes: AppSettings, StorageSettings, TransportSettings, AuthSettings, EmbeddingSettings, LangSmithSettings, ChunkingSettings, RerankingSettings, FtsPassageSettings. Use `Field(alias='ENV_VAR_NAME')`. Update `server.json` for new vars.

```python
# WRONG: os.getenv('DB_PATH')
# CORRECT: get_settings().storage.db_path
```

### Settings Class Architecture

**AppSettings must NEVER contain settings fields directly** - it only composes nested settings classes.

When adding new settings:
1. Add to an **existing** settings class if it EXACTLY matches the domain/purpose
2. Create a **new** settings class if no existing class is appropriate - even for a single setting

```python
# WRONG: Adding directly to AppSettings
class AppSettings(CommonSettings):
    my_new_setting: str = Field(...)  # Never do this!

# CORRECT: Create dedicated settings class
class MyFeatureSettings(CommonSettings):
    enabled: bool = Field(default=False, alias='ENABLE_MY_FEATURE')

class AppSettings(CommonSettings):
    my_feature: MyFeatureSettings = Field(default_factory=MyFeatureSettings)
```

Existing settings classes by domain:
- `LoggingSettings`: Log level configuration
- `ToolManagementSettings`: Tool availability (disabled tools)
- `TransportSettings`: HTTP transport (host, port)
- `AuthSettings`: Authentication (tokens, client IDs)
- `StorageSettings`: Database backend configuration (includes metadata indexing settings)
- `EmbeddingSettings`: Embedding provider configuration
- `SemanticSearchSettings`: Semantic search toggle
- `FtsSettings`: Full-text search configuration
- `HybridSearchSettings`: Hybrid search and RRF parameters
- `SearchSettings`: General search behavior
- `ChunkingSettings`: Text chunking parameters
- `RerankingSettings`: Cross-encoder reranking
- `FtsPassageSettings`: FTS passage extraction
- `LangSmithSettings`: LangSmith tracing

### FastMCP-Specific Requirements

1. **Never** `from __future__ import annotations` in server.py (breaks FastMCP)
2. `ctx: Context | None = None` as last param (hidden from clients)
3. Returns: serializable dicts/lists (TypedDicts from `app/types.py`)
4. Params: `Annotated[type, Field(description='...')]`
5. Register via `register_tool()` in lifespan(), not `@mcp.tool()`

### Adding New MCP Tools

```python
# app/tools/<domain>.py
async def my_tool(
    param: Annotated[str, Field(description='...')],
    ctx: Context | None = None,
) -> MyToolResponse:
    repos = await ensure_repositories()
    return {'success': True}
```

**Steps**: 1) Add to `app/tools/<domain>.py` 2) Add to `TOOL_ANNOTATIONS` in `app/tools/__init__.py` 3) Export from `__init__.py` 4) Register in `app/server.py` lifespan() 5) Add TypedDict to `app/types.py` 6) Add tests + real server tests in `test_real_server.py` 7) Update `server.json` if new env vars

**Annotation categories**: READ_ONLY (readOnlyHint=True), ADDITIVE (destructiveHint=False), UPDATE (destructiveHint=True, idempotentHint=False), DELETE (destructiveHint=True, idempotentHint=True)

### Adding New Providers (Embeddings/Reranking)

Both use identical patterns:
1. Create provider class in `app/<layer>/providers/` implementing the Protocol
2. Add to `PROVIDER_MODULES` and `PROVIDER_CLASSES` dicts in factory.py
3. Add install instructions to `PROVIDER_INSTALL_INSTRUCTIONS`
4. Add optional dependency group in `pyproject.toml`

```python
# Example: app/embeddings/providers/langchain_new.py
class NewEmbeddingProvider:
    async def initialize(self) -> None: ...
    async def shutdown(self) -> None: ...
    async def embed_query(self, text: str) -> list[float]: ...
    async def embed_documents(self, texts: list[str]) -> list[list[float]]: ...
    async def is_available(self) -> bool: ...
    def get_dimension(self) -> int: ...
    @property
    def provider_name(self) -> str: return 'new'
```

### Repository Pattern Implementation

All SQL in repositories (never server.py). Use `ensure_repositories()` from `app.startup`. Repositories detect backend type via `self.backend.backend_type`. SQLite: sync functions with `conn`. PostgreSQL: async functions with `conn`. Helper methods: `_placeholder()`, `_json_extract()`.

### Update Context Tool

Partial updates (only provided fields). Immutable: `id`, `thread_id`, `source`, `created_at`. Auto-managed: `content_type`, `updated_at`. Tags/images: replacement (not merge). Transaction-wrapped.

### Batch Operations

`store_context_batch`, `update_context_batch`, `delete_context_batch` (up to 100 entries). `atomic=true` (default): all-or-nothing. `atomic=false`: independent processing with per-entry results.

### Database Best Practices

Repository pattern, automatic connection pooling, parameterized queries, retry with exponential backoff. Monitor via `backend.get_metrics()`.

### Testing Conventions

Unit: `mock_server_dependencies`. Integration: `initialized_server` (SQLite temp DB). Always add `test_real_server.py` tests. SQLite-only test suite (PostgreSQL is production-only).
