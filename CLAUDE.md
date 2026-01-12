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

This server implements the [Model Context Protocol](https://modelcontextprotocol.io) (MCP), enabling:

- **JSON-RPC 2.0 Protocol**: Standardized communication for reliable tool invocation
- **Automatic Tool Discovery**: MCP clients auto-detect available tools and their schemas
- **Strong Typing**: Pydantic models ensure data integrity across client-server boundary
- **Universal Compatibility**: Works with Claude Desktop, Claude Code, LangGraph, and any MCP-compliant client
- **Multi-Transport Support**: Stdio (default), HTTP, streamable-http, or SSE transport modes
- **Tool Annotations**: MCP protocol hints (readOnlyHint, destructiveHint, idempotentHint) for client behavior optimization

### MCP Server Architecture

This is a FastMCP 2.0-based Model Context Protocol server that provides persistent context storage for LLM agents. The architecture consists of:

1. **FastMCP Server Layer** (`app/server.py`, `app/tools/`, `app/startup/`):
   - Entry point with FastMCP instance, lifespan management, and main() function (~415 lines)
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
   - **SimpleTokenVerifier** (`simple_token.py`): Bearer token authentication for HTTP transport
   - Constant-time token comparison to prevent timing attacks
   - Configured via `FASTMCP_SERVER_AUTH` and `MCP_AUTH_TOKEN` environment variables
   - **AuthSettings**: Centralized auth configuration via `app/settings.py`

3. **Storage Backend Layer** (`app/backends/`):
   - **StorageBackend Protocol** (`base.py`): Defines database-agnostic interface with 7 required methods
   - **SQLiteBackend** (`sqlite_backend.py`): Production-grade SQLite implementation with connection pooling, write queue, circuit breaker
   - **PostgreSQLBackend** (`postgresql_backend.py`): Async PostgreSQL implementation using asyncpg with connection pooling, MVCC, JSONB support
   - **Backend Factory** (`factory.py`): Creates appropriate backend based on `STORAGE_BACKEND` environment variable
   - Runtime backend selection enables support for multiple databases (SQLite, PostgreSQL)
   - All backends implement the same protocol for seamless switching

4. **Repository Pattern** (`app/repositories/`):
   - **RepositoryContainer**: Dependency injection container managing all repository instances
   - **ContextRepository**: Manages context entries (CRUD operations, search, deduplication, metadata filtering, updates)
   - **TagRepository**: Handles tag normalization and many-to-many relationships
   - **ImageRepository**: Manages multimodal image attachments
   - **StatisticsRepository**: Provides thread statistics and database metrics
   - **EmbeddingRepository**: Manages vector embeddings for semantic search
   - **FtsRepository**: Handles full-text search operations (FTS5 for SQLite, tsvector for PostgreSQL)
   - Each repository uses `StorageBackend` protocol for database operations
   - Repositories are database-agnostic - work with any backend implementation

5. **Data Models** (`app/models.py`):
   - Pydantic V2 models with `StrEnum` for Python 3.12+ compatibility
   - Strict validation for multimodal content (text + images)
   - Base64 image encoding/decoding with configurable size limits
   - `ContextEntry`, `ImageAttachment`, `StoreContextRequest` as main models

6. **Embeddings Layer** (`app/embeddings/`):
   - **EmbeddingProvider Protocol** (`base.py`): Defines provider-agnostic interface with `embed_query()` and `embed_documents()` methods
   - **EmbeddingFactory** (`factory.py`): Creates appropriate provider based on `EMBEDDING_PROVIDER` environment variable
   - **Providers** (`providers/`): LangChain-based implementations for Ollama, OpenAI, Azure OpenAI, HuggingFace, Voyage AI
   - All providers implement the same protocol for seamless switching between embedding services

7. **Metadata Filtering** (`app/metadata_types.py` & `app/query_builder.py`):
   - **MetadataFilter**: Advanced filter specification with 16 operators (eq, ne, gt, lt, contains, etc.)
   - **QueryBuilder**: Backend-aware SQL generation with proper parameter binding and type casting
   - Supports nested JSON path queries (e.g., "user.preferences.theme")
   - Case-sensitive/insensitive string operations
   - Safe SQL generation with injection prevention
   - Handles SQLite (`json_extract`) vs PostgreSQL (`->>`/`->`) JSON operators

8. **Database Layer** (`app/schemas/`):
   - **SQLite Schema** (`sqlite_schema.sql`): 3 tables with JSON support, BLOB storage
   - **PostgreSQL Schema** (`postgresql_schema.sql`): 3 tables with JSONB support, BYTEA storage
   - Thread-scoped context isolation with strategic indexing
   - Three tables: `context_entries`, `tags`, `image_attachments`
   - Normalized tags table for efficient querying (many-to-many)
   - Binary image storage (BLOB for SQLite, BYTEA for PostgreSQL) with ON DELETE CASCADE
   - WAL mode (SQLite) / MVCC (PostgreSQL) for concurrent access
   - All SQL operations encapsulated in repository classes

### Modular Package Structure

The server codebase is organized into focused packages:

```
app/
├── server.py              # Entry point, lifespan, FastMCP (~415 lines)
├── tools/                 # MCP tool implementations
│   ├── __init__.py       # Tool registration, TOOL_ANNOTATIONS
│   ├── context.py        # Core CRUD operations
│   ├── search.py         # All search operations
│   ├── discovery.py      # Thread listing, statistics
│   └── batch.py          # Bulk operations
├── startup/               # Server initialization
│   ├── __init__.py       # Database init, global state
│   └── validation.py     # Parameter validation utilities
├── migrations/            # Database migrations
│   ├── __init__.py       # Migration exports
│   ├── dependencies.py   # Dependency checking
│   ├── semantic.py       # Semantic search migrations
│   ├── fts.py            # Full-text search migrations
│   ├── metadata.py       # Metadata index management
│   └── utils.py          # Shared utilities
├── backends/              # Storage backends (unchanged)
├── repositories/          # Data access layer (unchanged)
├── embeddings/            # Embedding providers (unchanged)
└── ...
```

**Import Patterns:**
- Tools import from `app.startup` for global state and validation
- Tools import from `app.migrations` for migration status
- `app.server` imports from `app.tools` for registration
- Backward compatibility re-exports maintained in `app.server`

### Thread-Based Context Management

The core concept is thread-based context scoping:
- All agents working on the same task share a `thread_id`
- Context entries are tagged with `source`: 'user' or 'agent'
- Agents can filter context by thread, source, tags, content type, or metadata
- No hierarchical threads - flat structure for simplicity
- Metadata filtering supports 16 operators for complex queries

**Example Multi-Agent Workflow**:
```
Thread: "analyze-q4-sales"
├── User Context: "Analyze our Q4 sales data"
├── Agent 1 Context: "Fetched sales data from database"
├── Agent 2 Context: "Generated charts showing 15% growth"
└── Agent 3 Context: "Identified top performing products"
```
All agents share thread_id="analyze-q4-sales" and can retrieve each other's context.

### Database Schema

Three main tables with strategic indexing:
- `context_entries`: Main storage with thread_id and source indexes, JSON metadata field, updated_at timestamp
- `tags`: Normalized many-to-many relationship, lowercase storage
- `image_attachments`: Binary BLOB storage with foreign key cascade

Performance optimizations:
- WAL mode for better concurrency
- Memory-mapped I/O (256MB)
- Compound index on (thread_id, source) for common queries
- Indexed metadata fields for optimal filtering: `status`, `agent_name`, `task_name`, `project`, `report_type`
- Array/object fields (`technologies`, `references`) use PostgreSQL GIN index (not indexed in SQLite)

### Testing Strategy

The codebase uses a comprehensive multi-layered testing approach:

#### Important Testing Philosophy:
- **Tests use SQLite-only**: All integration tests use temporary SQLite databases for speed and simplicity
- **Production supports both backends**: While tests use SQLite, production code fully supports both SQLite and PostgreSQL
- **No PostgreSQL required for testing**: Developers can run the full test suite without installing PostgreSQL
- **Backend-agnostic implementation**: Repository code works identically with both backends
- **Real server tests required**: When writing unit tests, always add corresponding real server integration tests in `tests/test_real_server.py` to verify tool behavior via the actual MCP protocol

#### Key Test Fixtures (`conftest.py`):

| Fixture | Use Case |
|---------|----------|
| `test_db` | Direct SQLite operations without server layer |
| `mock_server_dependencies` | Server tool testing with mocked settings |
| `initialized_server` | Full integration tests with real database |
| `async_db_initialized` | Async storage backend tests |
| `async_db_with_embeddings` | Semantic search tests |

**Skip Markers**: `@requires_ollama`, `@requires_sqlite_vec`, `@requires_numpy`, `@requires_semantic_search`

All fixtures create SQLite temporary databases. `prevent_default_db_pollution` (autouse) prevents accidental production DB access.

### Key Implementation Details

1. **Python 3.12+ Type Hints**: Uses modern union syntax (`str | None`) instead of `Optional`
   - Do NOT use `from __future__ import annotations` in server.py (breaks FastMCP)
   - Use `StrEnum` instead of `str, Enum` pattern
   - Custom TypedDicts in `app/types.py` for consistent response shapes

2. **FastMCP Tool Signatures**: Tools use specific parameter types:
   - `Literal["user", "agent"]` for source parameter
   - `Annotated[type, Field(...)]` for parameter documentation
   - Returns must be serializable dicts/lists
   - `ctx: Context | None = None` parameter for FastMCP context (hidden from MCP clients)
   - Metadata filters use list of `MetadataFilter` objects for complex queries

3. **Async Context Management**: Server uses async context managers for lifecycle:
   - `@asynccontextmanager` for server startup/shutdown
   - SQLite operations use `loop.run_in_executor()` for async wrappers
   - PostgreSQL operations are natively async via asyncpg

4. **Sync vs Async Operations**:
   - **SQLiteBackend**: Repository operations are **sync callables** wrapped in async via `execute_write`/`execute_read`
     - Example: `def insert_context(conn, text, thread_id): ...` (sync function)
     - Backend wraps it: `await backend.execute_write(insert_context, 'Hello', 'thread-123')`
   - **PostgreSQLBackend**: Repository operations are **async callables** executed directly
     - Example: `async def insert_context(conn, text, thread_id): ...` (async function)
     - Backend calls it: `await backend.execute_write(insert_context, 'Hello', 'thread-123')`
   - Repositories detect backend type and choose appropriate implementation
   - This abstraction allows repositories to work with both backends seamlessly

5. **Design Patterns**:
   - **Protocol Pattern**: `StorageBackend` protocol defines database-agnostic interface
   - **Repository Pattern**: All SQL operations isolated in repository classes (`app/repositories/`)
   - **Dependency Injection**: `RepositoryContainer` provides repositories to server layer
   - **Factory Pattern**: `create_backend()` for backend selection, `get_settings()` for configuration
   - **DTO Pattern**: TypedDicts for data transfer between layers
   - **Context Manager Pattern**: Connection handling via `get_connection()` protocol method

6. **Error Handling**: Comprehensive error handling with specific exceptions:
   - Input validation via Pydantic (strict type checking, field validators)
   - Database constraints via CHECK clauses (source, content_type enums)
   - Size limits enforced before storage (10MB per image, 100MB total)
   - Graceful error responses with detailed messages for debugging
   - Transaction rollback on failures to maintain data integrity

### Semantic Search Implementation

The `semantic_search_context` tool is an optional feature that enables vector similarity search:

**Architecture**:
- **Embedding Providers** (`app/embeddings/`): Multi-provider architecture supporting Ollama, OpenAI, Azure, HuggingFace, Voyage
- **Embedding Repository** (`app/repositories/embedding_repository.py`): Manages vector storage and search
- **Backend-Specific Implementation**:
  - **SQLite**: Uses `sqlite-vec` extension with BLOB storage and `vec_distance_l2()` function
  - **PostgreSQL**: Uses `pgvector` extension with native `vector` type and `<->` operator for L2 distance

**When to Work With Semantic Search**:
- Adding new embedding providers: Create new provider in `app/embeddings/providers/`
- Changing vector dimensions: Update `EMBEDDING_DIM` environment variable
- Performance tuning: Adjust provider-specific settings in `app/settings.py`
- Backend-specific optimizations: Edit `EmbeddingRepository` methods

### Full-Text Search (FTS) Implementation

The `fts_search_context` tool is an optional feature that enables linguistic search with stemming, ranking, and boolean queries:

**Architecture**:
- **FTS Repository** (`app/repositories/fts_repository.py`): Handles search operations and query building
- **Backend-Specific Implementation**:
  - **SQLite**: Uses FTS5 with BM25 ranking. Porter stemmer (English) or unicode61 tokenizer (multilingual, no stemming).
  - **PostgreSQL**: Uses tsvector/tsquery with ts_rank ranking. Supports 29 languages with full stemming.

**Search Modes**:
- `match`: Standard word matching with stemming (default)
- `prefix`: Prefix matching for autocomplete-style search
- `phrase`: Exact phrase matching preserving word order
- `boolean`: Boolean operators (AND, OR, NOT) for complex queries

**Key Differences from Other Search Tools**:
- `search_context`: Exact keyword filtering on text content
- `semantic_search_context`: Vector similarity search for meaning-based retrieval
- `fts_search_context`: Linguistic search with stemming, ranking, and highlighted snippets
- `hybrid_search_context`: Combines FTS and semantic search with RRF fusion for best of both

### Search API Response Structure

All search tools return: `results` (array), `count` (int), `stats` (only when `explain_query=True`).

### Migration System

All migrations in `app/migrations/` are applied automatically on startup and are idempotent:
- **Semantic Search**: `add_semantic_search.sql` (SQLite), `add_semantic_search_postgresql.sql` (PostgreSQL)
- **FTS**: `add_fts.sql` (SQLite), `add_fts_postgresql.sql` (PostgreSQL)

Note: Changing `FTS_LANGUAGE` requires FTS table rebuild.

### Hybrid Search Implementation

The `hybrid_search_context` tool combines FTS and semantic search with Reciprocal Rank Fusion (RRF):

**Architecture**:
- Executes both FTS and semantic search in parallel
- Fuses results using RRF algorithm: `score(d) = Σ(1 / (k + rank_i(d)))`
- Documents appearing in both result sets score higher
- Graceful degradation: works with just FTS or just semantic search if one is unavailable

**Configuration**:
- Requires `ENABLE_HYBRID_SEARCH=true`
- Also requires at least one of `ENABLE_FTS=true` or `ENABLE_SEMANTIC_SEARCH=true`
- `HYBRID_RRF_K` controls fusion smoothing (higher = more weight to lower-ranked documents)

## Package and Release

- **Package Manager**: uv with Hatchling build backend
- **Entry Points**: `mcp-context-server`, `mcp-context` (see `pyproject.toml`)
- **Python**: 3.12+ required
- **Optional**: `uv sync --extra embeddings-ollama` for semantic search (or `embeddings-openai`, `embeddings-azure`, `embeddings-huggingface`, `embeddings-voyage`)

## Release Process

The project uses [Release Please](https://github.com/googleapis/release-please) for automated releases:
- Conventional commits are automatically parsed for CHANGELOG generation
- Version bumping is automated based on commit types
- To trigger a release, merge commits following [Conventional Commits](https://www.conventionalcommits.org/)

### Publish Workflow

On `release:published` event, three jobs run in parallel:
- **PyPI**: `mcp-context-server` package
- **MCP Registry**: Server metadata via `server.json`
- **GHCR**: Docker image `ghcr.io/alex-feel/mcp-context-server` (amd64/arm64)

## MCP Registry and server.json Maintenance

The `server.json` file enables MCP client discovery. Must comply with [MCP Registry specification](https://raw.githubusercontent.com/modelcontextprotocol/registry/refs/heads/main/docs/reference/server-json/generic-server-json.md).

### Synchronization with app/settings.py

All environment variables in `server.json` must match Field definitions in `app/settings.py`:

1. Add Field to appropriate class with `alias` parameter
2. Add corresponding entry to `server.json` environmentVariables array

```python
# settings.py
log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(default='ERROR', alias='LOG_LEVEL')
```

```json
// server.json
{"name": "LOG_LEVEL", "description": "Log level", "format": "string", "isRequired": false, "isSecret": false}
```

Release Please automatically updates `server.json` version during releases.

## Environment Variables

Configuration via `.env` file or environment:

**Core Settings:**
- `STORAGE_BACKEND`: Backend type - `sqlite` (default) or `postgresql`
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: ERROR)
- `DB_PATH`: Custom database location for SQLite (default: ~/.mcp/context_storage.db)
- `MAX_IMAGE_SIZE_MB`: Individual image size limit (default: 10)
- `MAX_TOTAL_SIZE_MB`: Total request size limit (default: 100)
- `DISABLED_TOOLS`: Comma-separated list of tools to disable (e.g., `delete_context,update_context`)

**Transport Settings** (for Docker/remote deployments):
- `MCP_TRANSPORT`: Transport mode - `stdio` (default), `http`, `streamable-http`, or `sse`
- `FASTMCP_HOST`: HTTP bind address (default: 0.0.0.0 for Docker)
- `FASTMCP_PORT`: HTTP port number (default: 8000)

**Authentication Settings** (for HTTP transport):
- `FASTMCP_SERVER_AUTH`: Auth verifier class (e.g., `app.auth.simple_token.SimpleTokenVerifier`)
- `MCP_AUTH_TOKEN`: Bearer token for HTTP authentication (required when using SimpleTokenVerifier)
- `MCP_AUTH_CLIENT_ID`: Client ID to assign to authenticated requests (default: mcp-client)

**Full-Text Search Settings:**
- `ENABLE_FTS`: Enable full-text search functionality (default: false)
- `FTS_LANGUAGE`: Language for stemming and text search (default: english). PostgreSQL supports 29 languages with full stemming. SQLite uses Porter stemmer (English) or unicode61 tokenizer (no stemming).

**Embedding Generation Settings:**
- `ENABLE_EMBEDDING_GENERATION`: Enable embedding generation for stored context entries (default: true).
  **BREAKING CHANGE in v1.0.0:** When true (default) and embedding dependencies are not met, server will NOT start.
  Set to false to explicitly disable embedding generation.
  - `true` (default): Embeddings generated on store/update. Server fails if dependencies unavailable.
  - `false`: No embeddings generated. Server starts without embedding dependencies.
  - Note: `ENABLE_SEMANTIC_SEARCH=true` requires embeddings. If `ENABLE_EMBEDDING_GENERATION=false`,
    the `semantic_search_context` tool will NOT be registered.

**Semantic Search Settings:**
- `ENABLE_SEMANTIC_SEARCH`: Enable semantic_search_context tool registration (default: false).
  Requires `ENABLE_EMBEDDING_GENERATION=true` or defaults. Controls tool availability only;
  embedding generation is controlled by `ENABLE_EMBEDDING_GENERATION`.
- `EMBEDDING_PROVIDER`: Embedding provider - `ollama` (default), `openai`, `azure`, `huggingface`, `voyage`
- `EMBEDDING_MODEL`: Embedding model name (default: qwen3-embedding:0.6b for Ollama)
- `EMBEDDING_DIM`: Embedding vector dimensions (default: 1024)
- `EMBEDDING_TIMEOUT_S`: Timeout in seconds for embedding generation API calls (default: 30.0)
- `EMBEDDING_RETRY_MAX_ATTEMPTS`: Maximum number of retry attempts for embedding generation (default: 3)
- `EMBEDDING_RETRY_BASE_DELAY_S`: Base delay in seconds between retry attempts with exponential backoff (default: 1.0)

**Provider-Specific Settings:**
- `OLLAMA_HOST`: Ollama API host URL (default: http://localhost:11434)
- `OPENAI_API_KEY`: OpenAI API key (required for openai provider)
- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME`: Azure OpenAI settings
- `HUGGINGFACEHUB_API_TOKEN`: HuggingFace API token (required for huggingface provider)
- `VOYAGE_API_KEY`: Voyage AI API key (required for voyage provider)

**Hybrid Search Settings:**
- `ENABLE_HYBRID_SEARCH`: Enable hybrid search combining FTS and semantic with RRF fusion (default: false)
- `HYBRID_RRF_K`: RRF smoothing constant for result fusion (default: 60)

**Metadata Indexing Settings:**
- `METADATA_INDEXED_FIELDS`: Comma-separated list of metadata fields to index with optional type hints.
  Format: `field:type,field2:type2` where type is `string` (default), `integer`, `boolean`, `float`, `array`, or `object`.
  Default: `status,agent_name,task_name,project,report_type,references:object,technologies:array`
- `METADATA_INDEX_SYNC_MODE`: How to handle index mismatches on startup. Options:
  - `strict`: Fail startup if indexes don't match configuration
  - `auto`: Automatically add missing and drop extra indexes
  - `warn`: Log warnings about mismatches but continue
  - `additive` (default): Only add missing indexes, never drop existing ones

**PostgreSQL Settings** (only when STORAGE_BACKEND=postgresql):
- `POSTGRESQL_CONNECTION_STRING`: Full PostgreSQL connection string (if provided, overrides individual host/port/user/password/database settings)
- `POSTGRESQL_HOST`: PostgreSQL server host (default: localhost)
- `POSTGRESQL_PORT`: PostgreSQL server port (default: 5432)
- `POSTGRESQL_USER`: PostgreSQL username (default: postgres)
- `POSTGRESQL_PASSWORD`: PostgreSQL password (default: postgres)
- `POSTGRESQL_DATABASE`: PostgreSQL database name (default: mcp_context)
- `POSTGRESQL_POOL_MIN`: Minimum pool size (default: 2)
- `POSTGRESQL_POOL_MAX`: Maximum pool size (default: 20)
- `POSTGRESQL_POOL_TIMEOUT_S`: Pool connection timeout (default: 10.0)
- `POSTGRESQL_COMMAND_TIMEOUT_S`: Command execution timeout (default: 60.0)
- `POSTGRESQL_SSL_MODE`: SSL mode - disable, allow, prefer, require, verify-ca, verify-full (default: prefer)
- `POSTGRESQL_SCHEMA`: PostgreSQL schema name for table and index operations (default: public). Required for Supabase or multi-schema environments where `current_schema()` may not return 'public'.

Additional tuning parameters (see `app/settings.py` for full list):
- Database connection pool settings
- Retry behavior configuration
- Circuit breaker thresholds


## Storage Backend Configuration

The server uses a protocol-based architecture that supports multiple database backends. The active backend is selected via the `STORAGE_BACKEND` environment variable.

### Supported Backends

#### SQLite (Default)

**Features:**
- Zero-configuration local storage
- Production-grade connection pooling
- Write queue for serialized operations
- Circuit breaker for fault tolerance
- Suitable for single-user deployments

#### PostgreSQL

**Quick Start with Docker (Recommended):**

PostgreSQL setup is incredibly simple using Docker - just 2 commands:

```bash
# 1. Pull and run PostgreSQL with pgvector (all-in-one)
docker run --name pgvector18 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mcp_context \
  -p 5432:5432 \
  -d pgvector/pgvector:pg18-trixie

# 2. Configure environment (minimal setup - just 2 variables)
export STORAGE_BACKEND=postgresql
export ENABLE_SEMANTIC_SEARCH=true  # Optional: only if you need semantic search

# 3. Run server (schema auto-initializes, pgvector extension auto-enables)
uv run mcp-context-server
```

**That's it!** No manual database setup, no extension installation - everything works automatically.

**Features:**
- Concurrent write support via MVCC (10x+ throughput vs SQLite)
- Production-grade connection pooling with asyncpg
- JSONB indexing for fast metadata queries
- Native pgvector support for semantic search
- Explicit transaction management
- Circuit breaker and retry logic
- Suitable for multi-user and high-traffic deployments

**PostgreSQL Initialization:**
The PostgreSQL backend automatically handles initialization on first run:
1. Connects to database using provided credentials
2. Enables pgvector extension (if semantic search is enabled)
3. Creates schema (tables, indexes, constraints)
4. Applies semantic search migration (if enabled)

All initialization is idempotent - safe to run multiple times.

#### Supabase

Supabase is fully compatible via PostgreSQL connection - see README.md for detailed setup instructions.

**Quick Reference:**
- Use `STORAGE_BACKEND=postgresql` with `POSTGRESQL_CONNECTION_STRING`
- Two connection methods: Direct (IPv6 required) or Session Pooler (IPv4 compatible)
- Enable pgvector via Dashboard → Extensions for semantic search
- "getaddrinfo failed" error = switch from Direct to Session Pooler

### Metadata Field Indexing by Backend

The server supports configurable metadata field indexing via `METADATA_INDEXED_FIELDS` environment variable.
Different backends have different capabilities for indexing JSON/JSONB fields:

| Field | Type | SQLite | PostgreSQL | Notes |
|-------|------|--------|------------|-------|
| `status` | string | B-tree (json_extract) | B-tree (->>) | String scalar |
| `agent_name` | string | B-tree (json_extract) | B-tree (->>) | String scalar |
| `task_name` | string | B-tree (json_extract) | B-tree (->>) | String scalar |
| `project` | string | B-tree (json_extract) | B-tree (->>) | String scalar |
| `report_type` | string | B-tree (json_extract) | B-tree (->>) | String scalar |
| `references` | object | **NOT INDEXED** | GIN (containment) | Nested object |
| `technologies` | array | **NOT INDEXED** | GIN (containment) | Array field |

**Important:** Array and nested object fields (`technologies`, `references`) cannot be efficiently indexed in SQLite.
Queries filtering on these fields will require full table scans in SQLite.
For high-performance array/object queries, use PostgreSQL which supports GIN indexes for JSONB containment operations.

**PostgreSQL GIN query examples:**
```sql
-- Find entries with specific technology
SELECT * FROM context_entries WHERE metadata @> '{"technologies": ["python"]}';

-- Find entries referencing specific context ID
SELECT * FROM context_entries WHERE metadata @> '{"references": {"context_ids": [2322]}}';
```

## Docker Deployment

The project supports containerized deployment with HTTP transport for remote access:

- **Multi-stage Dockerfile** with uv, non-root user (UID 10001), health check endpoint
- **Three Docker Compose configurations** in `deploy/docker/`: SQLite, PostgreSQL, External PostgreSQL (Supabase)
- **Ollama sidecar** with automatic embedding model download (`deploy/docker/ollama/`)
- **Health check endpoint** at `/health` for container orchestration

## Kubernetes Deployment

The project includes Helm chart support for Kubernetes deployments:

- **Helm chart** in `deploy/helm/mcp-context-server/` with configurable values
- **Pre-configured profiles**: `values-sqlite.yaml` and `values-postgresql.yaml`
- **Optional Ollama sidecar** for semantic search
- **Ingress support** with TLS configuration

## Windows Development Notes

**Platform-Specific Considerations:**
- **Path Handling**: Use `Path` objects from `pathlib` instead of string concatenation
  - Good: `Path(directory) / 'file.db'`
  - Bad: `directory + '/file.db'`
- **Environment Variables**: Set using `set VAR=value` in cmd.exe or `$env:VAR="value"` in PowerShell
  - Example: `set LOG_LEVEL=DEBUG && uv run mcp-context-server`
- **Database Path**: Default location is `%USERPROFILE%\.mcp\context_storage.db`
- **Line Endings**: Git should handle CRLF ↔ LF automatically via `.gitattributes`
- **Shell Commands**: Avoid Unix-specific commands (grep, cat, tail) in code - use Python equivalents

**Docker on Windows:**
- Use Docker Desktop for Windows to run PostgreSQL container
- Expose port 5432 for local development
- Windows firewall may need to allow Docker connections

## Debugging and Troubleshooting

```bash
# View server logs with debug level
set LOG_LEVEL=DEBUG && uv run mcp-context-server  # Windows cmd
$env:LOG_LEVEL="DEBUG"; uv run mcp-context-server # Windows PowerShell

# Test database connectivity
uv run python -c "from app.server import init_database; init_database()"

# Check database metrics
uv run python -c "from app.server import init_database, _backend; init_database(); print(_backend.get_metrics())"
```

### Common Issues

1. **Module Import Errors**: Run `uv sync` to ensure dependencies are installed
2. **Database Lock Errors**: WAL mode should prevent these, but check for stale processes
3. **Type Checking Errors**: Use `uv run mypy app` to identify type issues
4. **MCP Connection Issues**: Verify server is running and check `.mcp.json` config
5. **Windows Path Issues**: Use `Path` objects or raw strings (r"path\to\file") in Python code
6. **Semantic Search Not Available**: Ensure `ENABLE_SEMANTIC_SEARCH=true` and install an embedding provider: `uv sync --extra embeddings-ollama` (see docs/semantic-search.md for all providers)
7. **Full-Text Search Not Available**: Ensure `ENABLE_FTS=true`. The feature works without additional dependencies. Check logs for migration status on startup.

## Code Quality Standards

- **Ruff**: 127 char lines, Python 3.12+, single quotes, extensive rule sets (see `pyproject.toml`)
- **mypy**: Strict mode for `app/`, relaxed for `tests/`
- **pyright**: Standard mode globally, strict for `app/`
- **Important**: Never use `from __future__ import annotations` in server.py (breaks FastMCP)

## Critical Implementation Warnings

### Environment Variables - Centralized Configuration

**All environment variables MUST be read exclusively from `app/settings.py`**:

1. **Never use `os.environ` or `os.getenv()` directly** in any module except `app/settings.py`
2. **Always use `get_settings()`** to access configuration values throughout the codebase
3. **Add new environment variables** to the appropriate settings class in `app/settings.py`, or create a new class if none fits:
   - `AppSettings`: General application settings (log level, feature flags)
   - `StorageSettings`: Database and backend configuration
   - `TransportSettings`: HTTP transport settings (host, port, transport mode)
   - `AuthSettings`: Authentication settings (tokens, client IDs)
   - `EmbeddingSettings`: Embedding provider configuration (provider selection, API keys, model settings)
   - `LangSmithSettings`: LangSmith tracing configuration (optional observability)
4. **Use `Field(alias='ENV_VAR_NAME')`** to map settings attributes to environment variable names
5. **Update `server.json`** when adding new environment variables (for MCP registry)

```python
# WRONG - never do this
import os
db_path = os.getenv('DB_PATH', 'default.db')

# CORRECT - always use centralized settings
from app.settings import get_settings
settings = get_settings()
db_path = settings.storage.db_path
```

### FastMCP-Specific Requirements

1. **Never add `from __future__ import annotations`** to server.py - it breaks FastMCP's runtime type introspection
2. **Tool signatures must include `ctx: Context | None = None`** as the last parameter (hidden from MCP clients)
3. **Return types must be serializable dicts/lists** - use TypedDicts from `app/types.py`
4. **Use `Annotated` and `Field`** from `typing` and `pydantic` for parameter documentation
5. **Tools are registered dynamically** via `register_tool()` from `app.tools` in `lifespan()`, not with `@mcp.tool()` decorator

### Adding New MCP Tools

Tools are organized in the `app/tools/` package by domain. Follow this pattern:

**Step 1: Define the tool function in the appropriate tools module:**
```python
# In app/tools/<domain>.py (e.g., app/tools/context.py)
async def my_new_tool(
    required_param: Annotated[str, Field(description='Description for MCP clients')],
    optional_param: Annotated[int | None, Field(description='Optional parameter')] = None,
    ctx: Context | None = None,  # ALWAYS last, hidden from MCP clients
) -> MyToolResponse:  # Use TypedDict from app/types.py
    """Tool docstring shown to MCP clients."""
    repos = await ensure_repositories()
    # Use repository methods for database operations
    result = await repos.context.some_method(required_param)
    return {'success': True, 'data': result}
```

**Step 2: Add tool annotations in `app/tools/__init__.py`:**
```python
TOOL_ANNOTATIONS = {
    # ... existing tools ...
    'my_new_tool': {
        'title': 'My New Tool',
        'readOnlyHint': True,  # or False for write operations
    },
}
```

**Step 3: Export from `app/tools/__init__.py`:**
```python
from app.tools.<domain> import my_new_tool

__all__ = [
    # ... existing exports ...
    'my_new_tool',
]
```

**Step 4: Register in `app/server.py` lifespan():**
```python
from app.tools import my_new_tool, register_tool

# In lifespan():
register_tool(mcp, my_new_tool)
```

**Tool Annotation Categories** (defined in `app/tools/__init__.py`):
- `READ_ONLY_ANNOTATIONS`: Search/read tools (readOnlyHint=True)
- `ADDITIVE_ANNOTATIONS`: Store tools (destructiveHint=False)
- `UPDATE_ANNOTATIONS`: Update tools (destructiveHint=True, idempotentHint=False)
- `DELETE_ANNOTATIONS`: Delete tools (destructiveHint=True, idempotentHint=True)

**Checklist for new tools:**
1. Add TypedDict response type to `app/types.py`
2. Add repository methods if database access needed
3. Use `Literal["user", "agent"]` for source parameters
4. Add tool to appropriate domain file in `app/tools/`
5. Add annotations to `TOOL_ANNOTATIONS` in `app/tools/__init__.py`
6. Export from `app/tools/__init__.py`
7. Register in `app/server.py` lifespan() using `register_tool()`
8. Add tests in `tests/test_server.py` or dedicated test file
9. Add real server integration tests in `tests/test_real_server.py`
10. Update server.json if tool adds new environment variables

### Repository Pattern Implementation

1. **All database operations go through repositories** - server.py should never contain SQL
2. **Use `ensure_repositories()` from `app.startup`** - ensures proper initialization
3. **Repository methods return domain objects** - repositories handle all SQL and data mapping
4. **Each repository focuses on a single concern** - context, tags, images, or statistics
5. **Repository methods handle async/sync conversion** - repositories wrap sync DB calls with async
6. **Repositories support multiple backends** - all repositories detect backend type and generate appropriate SQL
7. **SQL dialect handled via helper methods** - `_placeholder()`, `_placeholders()`, `_json_extract()` abstract differences
8. **Writing new repository methods**:
   - SQLite: Write sync functions that accept `conn` parameter
   - PostgreSQL: Write async functions that accept `conn` parameter and use `await` for queries
   - Use `self.backend.backend_type` to conditionally branch if needed

### Update Context Tool Implementation

The `update_context` tool has specific behavior:
1. **Selective Updates**: Only provided fields are updated (partial updates supported)
2. **Immutable Fields**: `id`, `thread_id`, `source`, `created_at` cannot be modified
3. **Auto-managed Fields**: `content_type` recalculates based on images, `updated_at` auto-updates
4. **Full Replacement**: Tags and images use replacement semantics (not merge)
5. **Transaction Safety**: All updates wrapped in transactions for consistency

### Batch Operations

Three batch tools enable efficient bulk processing:
1. **`store_context_batch`**: Store up to 100 entries in one call
2. **`update_context_batch`**: Update up to 100 entries with `metadata_patch` support
3. **`delete_context_batch`**: Delete by IDs, thread IDs, source, or age

**Atomic Mode** (`atomic=true`, default):
- All operations succeed or all fail (transaction rollback on error)
- Returns partial results on failure

**Non-Atomic Mode** (`atomic=false`):
- Processes each entry independently
- Returns individual success/failure for each entry

### Database Best Practices

1. **Use repository pattern for all database operations** - never write SQL in server.py
2. **Repository methods handle connection management** - repositories use `StorageBackend` protocol internally
3. **Connection pooling is automatic** - managed by backend implementation (e.g., `SQLiteBackend`)
4. **Parameterized queries are enforced** - all repositories use parameterized SQL
5. **Handle transient failures** - Backend implementations include retry logic with exponential backoff
6. **Monitor connection health** - check `backend.get_metrics()` for diagnostics

### Testing Conventions

1. **Mock database for unit tests** - use `mock_server_dependencies` fixture
2. **Real database for integration tests** - use `initialized_server` fixture (SQLite temporary database)
3. **Test Windows compatibility** - the project runs on Windows, avoid Unix-specific commands
4. **Use temporary paths** from pytest's `tmp_path` fixture for test isolation
5. **Test update_context thoroughly** - ensure partial updates, field validation, and transaction safety
6. **SQLite-only test suite** - All tests use SQLite temporary databases; PostgreSQL backend is production-only
