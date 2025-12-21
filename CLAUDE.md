# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Essential commands
uv sync                                    # Install dependencies
uv run pytest                              # Run all tests
uv run pytest tests/test_server.py -v      # Run specific test file
uv run pytest tests/test_server.py::TestStoreContext::test_store_text_context -v  # Single test
uv run pre-commit run --all-files          # Lint + type check
uv run mcp-context-server                  # Start server
```

## Common Development Commands

### Building and Running

```bash
# Install dependencies and package
uv sync

# Run the MCP server (multiple options)
uv run mcp-context-server      # Full name entry point
uv run mcp-context             # Short alias
uv run python -m app.server    # As Python module

# Run from anywhere with uvx
uvx mcp-context-server           # From PyPI (published version)
uvx --from . mcp-context-server  # From local directory

# Test server starts correctly
uv run python -m app.server
```

### Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_server.py -v

# Run a single test
uv run pytest tests/test_server.py::TestStoreContext::test_store_text_context -v

# Run with coverage
uv run pytest --cov=app --cov-report=html

# Run integration tests only
uv run pytest -m integration

# Skip slow/integration tests for quick feedback
uv run pytest -m "not integration"

# Run real server integration test
uv run python run_integration_test.py

# Note: Integration tests use SQLite-only temporary databases for speed
# PostgreSQL backend is fully supported in production, but tests don't require PostgreSQL setup
```

### Code Quality and Linting

```bash
# Run all pre-commit hooks (Ruff, mypy, pyright)
uv run pre-commit run --all-files

# Run Ruff linter with autofix
uv run ruff check --fix .

# Run type checking separately
uv run mypy app
uv run pyright app
```

## High-Level Architecture

### MCP Protocol Integration

This server implements the [Model Context Protocol](https://modelcontextprotocol.io) (MCP), enabling:

- **JSON-RPC 2.0 Protocol**: Standardized communication for reliable tool invocation
- **Automatic Tool Discovery**: MCP clients auto-detect available tools and their schemas
- **Strong Typing**: Pydantic models ensure data integrity across client-server boundary
- **Universal Compatibility**: Works with Claude Desktop, LangGraph, and any MCP-compliant client
- **Stdio Transport**: Communication via standard input/output for simple integration

### MCP Server Architecture

This is a FastMCP 2.0-based Model Context Protocol server that provides persistent context storage for LLM agents. The architecture consists of:

1. **FastMCP Server Layer** (`app/server.py`):
   - Exposes 13 MCP tools via JSON-RPC protocol: `store_context`, `search_context`, `get_context_by_ids`, `delete_context`, `update_context`, `list_threads`, `get_statistics`, `semantic_search_context`, `fts_search_context`, `hybrid_search_context`, `store_context_batch`, `update_context_batch`, `delete_context_batch`
   - Handles stdio transport for Claude Desktop/LangGraph integration
   - Manages async request processing with proper lifecycle management
   - Uses `RepositoryContainer` for all database operations (no direct SQL)
   - Database initialization in `init_database()`, repository management via `_ensure_repositories()`

2. **Storage Backend Layer** (`app/backends/`):
   - **StorageBackend Protocol** (`base.py`): Defines database-agnostic interface with 7 required methods
   - **SQLiteBackend** (`sqlite_backend.py`): Production-grade SQLite implementation with connection pooling, write queue, circuit breaker
   - **PostgreSQLBackend** (`postgresql_backend.py`): Async PostgreSQL implementation using asyncpg with connection pooling, MVCC, JSONB support
   - **Backend Factory** (`factory.py`): Creates appropriate backend based on `STORAGE_BACKEND` environment variable
   - Runtime backend selection enables support for multiple databases (SQLite, PostgreSQL)
   - All backends implement the same protocol for seamless switching

3. **Repository Pattern** (`app/repositories/`):
   - **RepositoryContainer**: Dependency injection container managing all repository instances
   - **ContextRepository**: Manages context entries (CRUD operations, search, deduplication, metadata filtering, updates)
   - **TagRepository**: Handles tag normalization and many-to-many relationships
   - **ImageRepository**: Manages multimodal image attachments
   - **StatisticsRepository**: Provides thread statistics and database metrics
   - **EmbeddingRepository**: Manages vector embeddings for semantic search
   - **FtsRepository**: Handles full-text search operations (FTS5 for SQLite, tsvector for PostgreSQL)
   - Each repository uses `StorageBackend` protocol for database operations
   - Repositories are database-agnostic - work with any backend implementation

4. **Data Models** (`app/models.py`):
   - Pydantic V2 models with `StrEnum` for Python 3.12+ compatibility
   - Strict validation for multimodal content (text + images)
   - Base64 image encoding/decoding with configurable size limits
   - `ContextEntry`, `ImageAttachment`, `StoreContextRequest` as main models

5. **Services Layer** (`app/services/`):
   - **EmbeddingService** (`embedding_service.py`): Generates embeddings via Ollama API for semantic search
   - Handles model communication, vector dimension validation, and graceful degradation

6. **Metadata Filtering** (`app/metadata_types.py` & `app/query_builder.py`):
   - **MetadataFilter**: Advanced filter specification with 15 operators (eq, ne, gt, lt, contains, etc.)
   - **QueryBuilder**: Backend-aware SQL generation with proper parameter binding and type casting
   - Supports nested JSON path queries (e.g., "user.preferences.theme")
   - Case-sensitive/insensitive string operations
   - Safe SQL generation with injection prevention
   - Handles SQLite (`json_extract`) vs PostgreSQL (`->>`/`->`) JSON operators

7. **Database Layer** (`app/schemas/`):
   - **SQLite Schema** (`sqlite_schema.sql`): 3 tables with JSON support, BLOB storage
   - **PostgreSQL Schema** (`postgresql_schema.sql`): 3 tables with JSONB support, BYTEA storage
   - Thread-scoped context isolation with strategic indexing
   - Three tables: `context_entries`, `tags`, `image_attachments`
   - Normalized tags table for efficient querying (many-to-many)
   - Binary image storage (BLOB for SQLite, BYTEA for PostgreSQL) with ON DELETE CASCADE
   - WAL mode (SQLite) / MVCC (PostgreSQL) for concurrent access
   - All SQL operations encapsulated in repository classes

### Thread-Based Context Management

The core concept is thread-based context scoping:
- All agents working on the same task share a `thread_id`
- Context entries are tagged with `source`: 'user' or 'agent'
- Agents can filter context by thread, source, tags, content type, or metadata
- No hierarchical threads - flat structure for simplicity
- Metadata filtering supports 15 operators for complex queries

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
- Indexed metadata fields for optimal filtering: `status`, `priority`, `agent_name`, `task_name`, `completed`

### Testing Strategy

The codebase uses a comprehensive multi-layered testing approach:

#### Important Testing Philosophy:
- **Tests use SQLite-only**: All integration tests use temporary SQLite databases for speed and simplicity
- **Production supports both backends**: While tests use SQLite, production code fully supports both SQLite and PostgreSQL
- **No PostgreSQL required for testing**: Developers can run the full test suite without installing PostgreSQL
- **Backend-agnostic implementation**: Repository code works identically with both backends
- **Real server tests required**: When writing unit tests, always add corresponding real server integration tests in `tests/test_real_server.py` to verify tool behavior via the actual MCP protocol

#### Test Database Protection:
- **Automatic Isolation**: All tests use `prevent_default_db_pollution` fixture (session-scoped, autouse)
  - Prevents accidental use of `~/.mcp/context_storage.db` during testing
  - Sets `MCP_TEST_MODE=1` and temporary `DB_PATH` for all tests
  - Raises `RuntimeError` if default database is accessed
- **Fixture Selection**:
  - Use `mock_server_dependencies` + `initialized_server` for integration tests with real database
  - Use `test_db` for direct SQLite operations without server layer
  - Use `async_db_initialized` for async database operations
  - All fixtures create SQLite temporary databases (no PostgreSQL fixtures in test suite)

#### Test Fixtures (`conftest.py`):
- **`test_settings`**: Creates test-specific AppSettings with temp database
- **`temp_db_path`**: Provides temporary database file path
- **`test_db`**: SQLite connection with schema initialization (for direct DB tests)
- **`initialized_server`**: Full server initialization with database (for integration tests)
- **`async_db_initialized`**: Async storage backend with proper lifecycle management
- **`async_db_with_embeddings`**: Async backend with semantic search migration applied (for embedding tests)
- **`mock_context`**: Mock FastMCP Context for unit tests
- **`sample_image_data`**: Base64 encoded test PNG image
- **`multiple_context_entries`**: Pre-populated database entries for testing
- **`mock_server_dependencies`**: Patches server settings for isolated testing
- **`embedding_dim`**: Dynamic embedding dimension from settings (for semantic search tests)

**Conditional Skip Markers** (for optional dependencies):
- `@requires_ollama`: Skip if ollama package not installed
- `@requires_sqlite_vec`: Skip if sqlite-vec package not installed
- `@requires_numpy`: Skip if numpy package not installed
- `@requires_semantic_search`: Skip if any semantic search dependency missing

**Fixture Selection Guide**:
- Direct SQLite testing → use `test_db`
- Server tool testing (mocked) → use `mock_server_dependencies`
- Full integration testing → use `initialized_server` or `async_db_initialized`
- Semantic search testing → use `async_db_with_embeddings` with `@requires_semantic_search`
- All fixtures use SQLite temporary databases for consistency and speed

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
- **Embedding Service** (`app/services/embedding_service.py`): Generates embeddings via Ollama API
- **Embedding Repository** (`app/repositories/embedding_repository.py`): Manages vector storage and search
- **Backend-Specific Implementation**:
  - **SQLite**: Uses `sqlite-vec` extension with BLOB storage and `vec_distance_l2()` function
  - **PostgreSQL**: Uses `pgvector` extension with native `vector` type and `<->` operator for L2 distance

**When to Work With Semantic Search**:
- Adding new embedding models: Modify `app/services/embedding_service.py`
- Changing vector dimensions: Update migration files in `app/migrations/`
- Performance tuning: Adjust `EMBEDDING_DIM` and indexing strategies
- Backend-specific optimizations: Edit `EmbeddingRepository` methods

**Migration System**:
- **Location**: `app/migrations/` directory
- **Backend-Specific Migrations**:
  - `add_semantic_search.sql` - SQLite migration (adds embeddings table with vec0 virtual table)
  - `add_semantic_search_postgresql.sql` - PostgreSQL migration (enables pgvector, adds vector column)
- **Application**: Migrations are applied automatically on server startup if semantic search is enabled
- **Idempotency**: All migrations use `IF NOT EXISTS` or `CREATE EXTENSION IF NOT EXISTS` for safe re-runs
- **Manual Migration** (if needed): Execute migration SQL directly against database
- **Note**: All migrations are version-controlled and tracked in `app/migrations/`. The server automatically detects and applies pending migrations on startup.

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

**Migration System**:
- **Location**: `app/migrations/` directory
- **Backend-Specific Migrations**:
  - `add_fts.sql` - SQLite migration (creates FTS5 virtual table with triggers)
  - `add_fts_postgresql.sql` - PostgreSQL migration (adds tsvector column with GIN index)
- **Language Change**: Changing `FTS_LANGUAGE` requires migration (FTS table rebuild)

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

## Package Configuration

The project uses `uv` as the package manager with `tool.uv.package = true` in pyproject.toml. Key configuration details:

- **Build Backend**: Hatchling for Python package building
- **Entry Points**: Defined in `[project.scripts]` - both `mcp-context-server` and `mcp-context` aliases
- **Dependencies**: Minimal core dependencies (fastmcp, pydantic, python-dotenv, asyncpg)
- **Optional Dependencies**: `semantic-search` group for Ollama, numpy, sqlite-vec, pgvector
- **Python Version**: Requires Python 3.12+ for modern type hints and StrEnum

## Release Process

The project uses [Release Please](https://github.com/googleapis/release-please) for automated releases:
- Conventional commits are automatically parsed for CHANGELOG generation
- Version bumping is automated based on commit types
- PyPI publishing is handled by GitHub Actions
- To trigger a release, merge commits following [Conventional Commits](https://www.conventionalcommits.org/)

## MCP Registry and server.json Maintenance

The `server.json` file enables MCP client discovery and configuration of this server. It must comply with the official [MCP Registry specification](https://raw.githubusercontent.com/modelcontextprotocol/registry/refs/heads/main/docs/reference/server-json/generic-server-json.md).

### Official Specification

- **Schema URL**: `https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/server.schema.json`
- **Specification**: [MCP Registry Generic Server JSON Reference](https://raw.githubusercontent.com/modelcontextprotocol/registry/refs/heads/main/docs/reference/server-json/generic-server-json.md)
- **Purpose**: Provides standardized metadata for MCP client auto-configuration

### Valid Environment Variable Fields

Environment variables in `server.json` support ONLY these fields:

**Required Fields:**
- `name` - Variable identifier (e.g., `LOG_LEVEL`)
- `description` - Description of the input
- `format` - Specifies the input format
- `isRequired` - Boolean requirement flag

**Optional Fields:**
- `isSecret` - Sensitivity indicator (default: false)
- `default` - Default value
- `choices` - Array of allowed values

### Synchronization with app/settings.py

All environment variables in `server.json` must match Field definitions in `app/settings.py`:

**When adding a new environment variable:**
1. Add Field to appropriate class in `app/settings.py` with `alias` parameter
2. Add corresponding entry to `server.json` environmentVariables array
3. Verify synchronization using validation commands below

**Example Field in settings.py:**
```python
log_level: Literal['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'] = Field(
    default='ERROR',
    alias='LOG_LEVEL',
)
```

**Corresponding server.json entry:**
```json
{
    "description": "Log level",
    "isRequired": false,
    "format": "string",
    "isSecret": false,
    "name": "LOG_LEVEL"
}
```

### Validation Commands

**Validate JSON syntax:**
```bash
python -m json.tool server.json > /dev/null && echo "JSON is valid" || echo "JSON is invalid"
```

**Count environment variables:**
```bash
# Should match number of Field aliases in app/settings.py
python -c "import json; print(len(json.load(open('server.json'))['packages'][0]['environmentVariables']))"
```

**List all environment variable names:**
```bash
python -c "import json; print('\n'.join([e['name'] for e in json.load(open('server.json'))['packages'][0]['environmentVariables']]))"
```

### Version Synchronization

The version in `server.json` must match `pyproject.toml`:
- Top-level `version` field
- Package-level `version` field (inside packages array)
- Both must be identical semantic versions (e.g., `0.5.0`)

**Automated version updates:**
- Release Please automatically updates `server.json` version during releases

## Environment Variables

Configuration via `.env` file or environment:

**Core Settings:**
- `STORAGE_BACKEND`: Backend type - `sqlite` (default) or `postgresql`
- `LOG_LEVEL`: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: INFO)
- `DB_PATH`: Custom database location for SQLite (default: ~/.mcp/context_storage.db)
- `MAX_IMAGE_SIZE_MB`: Individual image size limit (default: 10)
- `MAX_TOTAL_SIZE_MB`: Total request size limit (default: 100)

**Full-Text Search Settings:**
- `ENABLE_FTS`: Enable full-text search functionality (default: false)
- `FTS_LANGUAGE`: Language for stemming and text search (default: english). PostgreSQL supports 29 languages with full stemming. SQLite uses Porter stemmer (English) or unicode61 tokenizer (no stemming).

**Semantic Search Settings:**
- `ENABLE_SEMANTIC_SEARCH`: Enable semantic search functionality (default: false)
- `OLLAMA_HOST`: Ollama API host URL (default: http://localhost:11434)
- `EMBEDDING_MODEL`: Embedding model name (default: embeddinggemma:latest)
- `EMBEDDING_DIM`: Embedding vector dimensions (default: 768)

**Hybrid Search Settings:**
- `ENABLE_HYBRID_SEARCH`: Enable hybrid search combining FTS and semantic with RRF fusion (default: false)
- `HYBRID_RRF_K`: RRF smoothing constant for result fusion (default: 60)

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

**Configuration (via environment variables):**
```bash
STORAGE_BACKEND=postgresql
POSTGRESQL_HOST=localhost         # Default
POSTGRESQL_PORT=5432              # Default
POSTGRESQL_USER=postgres          # Default
POSTGRESQL_PASSWORD=postgres      # Change in production!
POSTGRESQL_DATABASE=mcp_context   # Default

# Optional: Connection pool settings
POSTGRESQL_POOL_MIN=2
POSTGRESQL_POOL_MAX=20
POSTGRESQL_POOL_TIMEOUT_S=10.0
POSTGRESQL_COMMAND_TIMEOUT_S=60.0

# Optional: SSL/TLS settings
POSTGRESQL_SSL_MODE=prefer  # Options: disable, allow, prefer, require, verify-ca, verify-full
```

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

### StorageBackend Protocol

All backends implement the `StorageBackend` protocol with these required methods:
- `initialize()`: Set up connection pools and resources
- `shutdown()`: Clean up connections and release resources
- `get_connection(readonly: bool)`: Get a database connection (context manager)
- `execute_write(operation, *args, **kwargs)`: Execute write operation with retry logic
- `execute_read(operation, *args, **kwargs)`: Execute read operation
- `get_metrics()`: Return backend health metrics
- `backend_type` property: Return backend identifier ('sqlite', 'postgresql')

### Backend Benefits

1. **Database-Agnostic Repositories**: Repository code works with any backend without modification
2. **Runtime Selection**: Choose backend via environment variable - no code changes needed
3. **Easy Testing**: Switch to in-memory database for tests via backend factory
4. **Future-Proof**: Add new backends by implementing the protocol
5. **Type-Safe**: Protocol ensures all backends provide required functionality

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

### Common Debug Commands

```bash
# Check if server starts correctly
uv run python -m app.server

# Test database connectivity
uv run python -c "from app.server import init_database; init_database()"

# Check database location and size (Windows)
dir %USERPROFILE%\.mcp\context_storage.db

# View server logs with debug level (Windows - cmd.exe)
set LOG_LEVEL=DEBUG && uv run mcp-context-server

# View server logs with debug level (Windows - PowerShell)
$env:LOG_LEVEL="DEBUG"; uv run mcp-context-server

# Test a specific tool function
uv run pytest tests/test_server.py::TestStoreContext -v

# Check database metrics via Python
uv run python -c "from app.server import init_database, _backend; init_database(); print(_backend.get_metrics())"
```

### Common Issues and Solutions

1. **Module Import Errors**: Run `uv sync` to ensure dependencies are installed
2. **Database Lock Errors**: WAL mode should prevent these, but check for stale processes
3. **Type Checking Errors**: Use `uv run mypy app` to identify type issues
4. **MCP Connection Issues**: Verify server is running and check `.mcp.json` config
5. **Windows Path Issues**: Use `Path` objects or raw strings (r"path\to\file") in Python code
6. **Semantic Search Not Available**: Ensure `ENABLE_SEMANTIC_SEARCH=true` and install optional dependencies with `uv sync --extra semantic-search`
7. **Full-Text Search Not Available**: Ensure `ENABLE_FTS=true`. The feature works without additional dependencies. Check logs for migration status on startup.

## Code Quality Standards

### Ruff Configuration

The project enforces strict code quality with extensive Ruff rules:
- **Line Length**: 127 characters maximum
- **Target Python**: 3.12+
- **Quote Style**: Single quotes for code, double for docstrings
- **Imports**: Forced single-line for clarity
- **Enabled Rule Sets**: FAST (FastAPI), ANN (annotations), ASYNC, B (bugbear), PT (pytest), UP (pyupgrade), FURB (refurb), and many more

### Type Checking

Both mypy and pyright are configured:
- **mypy**: Strict mode for `app/`, relaxed for `tests/`
- **pyright**: Standard mode globally, strict for `app/`
- **Important**: Never use `from __future__ import annotations` in server.py (breaks FastMCP)

## Critical Implementation Warnings

### FastMCP-Specific Requirements

1. **Never add `from __future__ import annotations`** to server.py - it breaks FastMCP's runtime type introspection
2. **Tool signatures must include `ctx: Context | None = None`** as the last parameter (hidden from MCP clients)
3. **Return types must be serializable dicts/lists** - use TypedDicts from `app/types.py`
4. **Tool decorators require specific imports**: Use `Annotated` and `Field` from `typing` and `pydantic`

### Adding New MCP Tools

When adding a new tool to the server, follow this pattern:

```python
# In app/server.py
@mcp.tool()
async def my_new_tool(
    required_param: Annotated[str, Field(description='Description for MCP clients')],
    optional_param: Annotated[int | None, Field(description='Optional parameter')] = None,
    ctx: Context | None = None,  # ALWAYS last, hidden from MCP clients
) -> MyToolResponse:  # Use TypedDict from app/types.py
    """Tool docstring shown to MCP clients."""
    repos = _ensure_repositories()
    # Use repository methods for database operations
    result = await repos.context.some_method(required_param)
    return {'success': True, 'data': result}
```

**Checklist for new tools:**
1. Add TypedDict response type to `app/types.py`
2. Add repository methods if database access needed
3. Use `Literal["user", "agent"]` for source parameters
4. Add tests in `tests/test_server.py` or dedicated test file
5. Add real server integration tests in `tests/test_real_server.py`
6. Update server.json if tool adds new environment variables

### Repository Pattern Implementation

1. **All database operations go through repositories** - server.py should never contain SQL
2. **Use `_ensure_repositories()` to get repository container** - ensures proper initialization
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
