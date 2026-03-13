# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# Build and run
uv sync --all-extras --all-groups          # Install ALL dependencies (dev + all optional)
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
   - Entry point with FastMCP instance, lifespan management, and main() function (~680 lines)
   - Tool implementations in `app/tools/` organized by domain: `context.py` (CRUD), `search.py` (4 search tools), `discovery.py` (list_threads, get_statistics), `batch.py` (batch CRUD), `descriptions.py` (backend-specific dynamic tool descriptions)
   - Dynamic tool registration via `register_tool()` from `app/tools/__init__.py`
   - Supports multiple transports: stdio (default), HTTP, streamable-http, SSE
   - Provides `/health` endpoint for container orchestration (HTTP transport only)
   - Global state and initialization in `app/startup/` package: `init_database()`, `ensure_repositories()`, `set_summary_provider()`/`get_summary_provider()`
   - **Temporary Patches** (`app/patches/`): Monkey-patches for upstream MCP SDK bugs applied at startup. See "Known Upstream Bugs" section.

2. **Authentication Layer** (`app/auth/simple_token.py`): Bearer token auth for HTTP transport with constant-time comparison. Configured via `MCP_AUTH_PROVIDER` and `MCP_AUTH_TOKEN`.

3. **Storage Backend Layer** (`app/backends/`):
   - **StorageBackend Protocol** (`base.py`): Database-agnostic interface (8 methods including `begin_transaction()`)
   - **TransactionContext Protocol** (`base.py`): Provides `connection` and `backend_type` for multi-operation atomic transactions
   - **SQLiteBackend**: Connection pooling, write queue, circuit breaker
   - **PostgreSQLBackend**: Async via asyncpg, connection pooling, MVCC, JSONB, Pgpool-II detection, classified error handling
   - **Backend Factory** (`factory.py`): Creates backend based on `STORAGE_BACKEND` env var

4. **Repository Pattern** (`app/repositories/`):
   - **RepositoryContainer** (`__init__.py`): DI container for all repositories
   - Repositories: Context (CRUD, search, deduplication), Tag (normalization, many-to-many), Image (binary attachments), Statistics, Embedding (vector storage/search), Fts (FTS5/tsvector)
   - All repositories use `StorageBackend` protocol — database-agnostic
   - `BaseRepository` provides `_placeholder()`, `_placeholders()`, `_json_extract()` helpers

5. **Data Models** (`app/models.py`): Pydantic V2 with `StrEnum` for Python 3.12+. Main models: `ContextEntry`, `ImageAttachment`, `StoreContextRequest`. Base64 image encoding with configurable size limits.

6. **Embeddings Layer** (`app/embeddings/`): `EmbeddingProvider` Protocol, factory with dynamic imports (`PROVIDER_MODULES`/`PROVIDER_CLASSES` dicts). Providers: Ollama, OpenAI, Azure, HuggingFace, Voyage. Retry via tenacity (`retry.py`). Context limits registry (`context_limits.py`). LangSmith tracing integration (`tracing.py`).

7. **Reranking Layer** (`app/reranking/`): `RerankingProvider` Protocol, same factory pattern. Provider: FlashRank (default, 34MB model, ONNX inference offloaded to thread pool).

8. **Summary Generation Layer** (`app/summary/`): `SummaryProvider` Protocol, same factory pattern. Providers: Ollama, OpenAI, Anthropic. Retry via tenacity. Default model: `qwen3:1.7b`. Prompt: `DEFAULT_SUMMARY_PROMPT` in `instructions.py`, configurable via `SUMMARY_PROMPT` env var.

9. **Services Layer** (`app/services/`): `ChunkingService` (`TextChunk` dataclass, `split_text()`, LangChain's `RecursiveCharacterTextSplitter`). `PassageExtractionService` (`extract_rerank_passage()`, `HighlightRegion` dataclass).

10. **Metadata Filtering** (`app/metadata_types.py` & `app/query_builder.py`): `MetadataFilter` with 16 operators. `QueryBuilder`: backend-aware SQL with nested JSON paths. Handles SQLite (`json_extract`) vs PostgreSQL (`->>`/`->`) operators.

11. **Other modules**: `app/fusion.py` (RRF algorithm), `app/errors.py` (ConfigurationError exit 78, DependencyError exit 69), `app/instructions.py` (server instructions), `app/types.py` (40+ TypedDicts for API responses), `app/logger_config.py` (logging configuration), `app/schemas/` (SQL schema files).

### Thread-Based Context Management

Agents share context via `thread_id`. Entries tagged with `source`: 'user' or 'agent'. Filter by thread, source, tags, content type, or metadata (16 operators). Flat structure (no hierarchy).

### Database Schema

Tables: `context_entries` (main, with thread_id/source indexes, JSON metadata, summary column), `tags` (many-to-many, lowercase), `image_attachments` (binary, cascade delete).

**Performance**: WAL mode, 256MB mmap, compound index (thread_id, source). Indexed metadata: `status`, `agent_name`, `task_name`, `project`, `report_type`. Array/object fields use PostgreSQL GIN (not indexed in SQLite).

### Search Tools

`search_context` (exact keyword), `semantic_search_context` (vector similarity), `fts_search_context` (linguistic + snippets), `hybrid_search_context` (RRF fusion of FTS + semantic).

**FTS modes**: `match` (default, stemming), `prefix` (autocomplete), `phrase` (exact order), `boolean` (AND/OR/NOT). SQLite: FTS5 with BM25, Porter stemmer. PostgreSQL: tsvector/tsquery with ts_rank, 29 languages.

**Hybrid search**: RRF formula `score(d) = Σ(1 / (k + rank_i(d)))`. Parallel execution, graceful degradation. Adaptive FTS mode: queries with `HYBRID_FTS_OR_THRESHOLD` (default 4) or more significant terms switch from AND to OR for improved recall.

**Response**: `results` (array), `count` (int), `stats` (only when `explain_query=True`).

### Migration System

Auto-applied idempotent migrations in `app/migrations/`: semantic search, FTS, chunking (1:N embeddings), metadata indexing, summary column. PostgreSQL migrations use `POSTGRESQL_MIGRATION_TIMEOUT_S` (300s default) for DDL operations. Changing `FTS_LANGUAGE` requires FTS table rebuild.

### Testing Strategy

**Philosophy**: Tests use SQLite-only temp databases (no PostgreSQL required). Production supports both backends. Always add real server integration tests in `tests/test_real_server.py` for new tools.

**Key Fixtures** (`conftest.py`): `test_db` (direct SQLite), `mock_server_dependencies` (mocked settings), `initialized_server` (full integration), `async_db_initialized` (async backend), `async_db_with_embeddings` (semantic search).

**Skip Markers**: `@requires_ollama`, `@requires_sqlite_vec`, `@requires_numpy`, `@requires_semantic_search`

`prevent_default_db_pollution` (autouse) prevents accidental production DB access.

### Key Implementation Details

1. **Python 3.12+ Type Hints**: `str | None` syntax, `StrEnum`, TypedDicts in `app/types.py`. **NEVER** use `from __future__ import annotations` in server.py (breaks FastMCP).

2. **FastMCP Tool Signatures**: `Literal["user", "agent"]` for source, `Annotated[type, Field(...)]` for docs, `ctx: Context | None = None` as last param (hidden from clients). Returns must be serializable dicts/lists. Register via `register_tool()` in lifespan(), not `@mcp.tool()`.

3. **Async Operations**: SQLite ops are sync callables wrapped via `execute_write`/`execute_read`. PostgreSQL ops are native async. Repositories detect backend type automatically via `self.backend.backend_type`.

4. **Design Patterns**:
   - **Protocol** (`@runtime_checkable`): `StorageBackend`, `TransactionContext`, `EmbeddingProvider`, `SummaryProvider`, `RerankingProvider`
   - **Repository**: All SQL in `app/repositories/`, never in server.py or tools
   - **Factory**: `create_backend()`, `create_embedding_provider()`, `create_summary_provider()`, `create_reranking_provider()` — dynamic imports via `PROVIDER_MODULES` dicts
   - **DI**: `RepositoryContainer` injects all repositories

5. **Error Classification** (`app/errors.py`): `ConfigurationError` (exit 78, supervisor never retries), `DependencyError` (exit 69, may retry with backoff). `classify_provider_error()` classifies embedding/summary provider failures. BSD sysexits.h exit codes for Docker/Kubernetes restart policies.

6. **Server Instructions**: Optional `instructions` field in MCP `InitializeResult`. Configured via `MCP_SERVER_INSTRUCTIONS` env var (overrides `DEFAULT_INSTRUCTIONS` from `app/instructions.py`). Empty string disables.

## Package and Release

uv + Hatchling. Entry points: `mcp-context-server`, `mcp-context`. Python 3.12+. Optional extras: `embeddings-ollama`, `embeddings-openai`, `embeddings-azure`, `embeddings-huggingface`, `embeddings-voyage`, `summary-ollama`, `summary-openai`, `summary-anthropic`, `reranking`, `langsmith`.

[Release Please](https://github.com/googleapis/release-please) for automated releases via [Conventional Commits](https://www.conventionalcommits.org/). On `release:published`: PyPI package, MCP Registry (`server.json`), GHCR Docker image (amd64/arm64).

## CI and Docker Lock File Discipline

The `uv.lock` file is a UNIVERSAL resolution containing ALL dependencies across ALL optional groups and extras. At install time, `uv sync` with selective flags installs only the relevant subset.

**Three defense layers**: Pre-commit `uv-lock` hook (local), `uv lock --check` (CI early step), `uv sync --locked` (CI install).

**Every CI workflow** (`test.yml`, `lint.yml`) MUST run `uv lock --check` then `uv sync --locked --all-extras --all-groups`:

```yaml
# CORRECT: CI workflow pattern
- run: uv lock --check
- run: uv sync --locked --all-extras --all-groups
```

```yaml
# WRONG: Explicit listing misses extras when new ones are added
- run: uv sync --locked --dev --extra embeddings-ollama --extra reranking
```

**Exception**: `publish.yml` and `release-please.yml` run `uv lock` (without `--check`) because Release Please bumps the version, requiring lock file regeneration.

**Docker** MUST use `--locked --no-dev --extra <variant>` for SELECTIVE installation:

```dockerfile
uv sync --locked --no-install-project --extra ${EMBEDDING_EXTRA} --extra ${SUMMARY_EXTRA} --extra reranking --no-dev
uv sync --locked --extra ${EMBEDDING_EXTRA} --extra ${SUMMARY_EXTRA} --extra reranking --no-dev
```

Build args: `EMBEDDING_EXTRA` (default: `embeddings-ollama`), `SUMMARY_EXTRA` (default: `summary-ollama`). Docker intentionally does NOT use `--all-extras`.

## MCP Registry and server.json Maintenance

`server.json` enables MCP client discovery. Every `Field(alias=...)` in `app/settings.py` MUST have a corresponding entry in `server.json` `environmentVariables`. This invariant is enforced by `test_server_json_environment_variables_match_settings`. Release Please auto-updates version.

## Environment Variables

Configuration via `.env` file or environment. **Canonical source**: `app/settings.py` — all env vars with defaults, descriptions, and validation.

**Core**: `STORAGE_BACKEND` (sqlite*/postgresql), `LOG_LEVEL` (ERROR*), `DB_PATH`, `MAX_IMAGE_SIZE_MB` (10*), `MAX_TOTAL_SIZE_MB` (100*), `DISABLED_TOOLS`

**Transport**: `MCP_TRANSPORT` (stdio*/http/streamable-http/sse), `FASTMCP_HOST` (0.0.0.0*), `FASTMCP_PORT` (8000*), `FASTMCP_STATELESS_HTTP` (true*)

**FastMCP Logging** (deployment-only, NOT in `app/settings.py`): `FASTMCP_ENABLE_RICH_LOGGING` (true*; set `false` in Docker/cloud). Read directly by FastMCP at import time — see [FASTMCP_* Governance](#fastmcp_-env-var-governance).

**Auth**: `MCP_AUTH_PROVIDER` (none*/simple_token), `MCP_AUTH_TOKEN`, `MCP_AUTH_CLIENT_ID` (mcp-client*)

**Feature Toggles**: `ENABLE_EMBEDDING_GENERATION` (true*), `ENABLE_SEMANTIC_SEARCH` (false*), `ENABLE_FTS` (false*), `ENABLE_HYBRID_SEARCH` (false*), `ENABLE_CHUNKING` (true*), `ENABLE_RERANKING` (true*), `ENABLE_SUMMARY_GENERATION` (true*)

**Embedding**: `EMBEDDING_PROVIDER` (ollama*/openai/azure/huggingface/voyage), `EMBEDDING_MODEL` (qwen3-embedding:0.6b*), `EMBEDDING_DIM` (1024*), `EMBEDDING_TIMEOUT_S` (30*), `EMBEDDING_MAX_CONCURRENT` (3*)

**Summary**: `SUMMARY_PROVIDER` (ollama*/openai/anthropic), `SUMMARY_MODEL` (qwen3:1.7b*), `SUMMARY_MAX_TOKENS` (2000*), `SUMMARY_MIN_CONTENT_LENGTH` (300*; text shorter than this skips summary; 0 = always generate), `SUMMARY_PROMPT`

**Provider-specific, PostgreSQL, reranking, chunking, hybrid, FTS, search, and metadata indexing vars**: See `app/settings.py` for complete list with defaults and descriptions.

*\* = default value*

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
MVCC, asyncpg pooling, JSONB/GIN indexes, pgvector. Pgpool-II auto-detected (disables prepared statements). Classified error handling: ConfigurationError (exit 78) vs DependencyError (exit 69).

### Supabase
`STORAGE_BACKEND=postgresql` + `POSTGRESQL_CONNECTION_STRING`. Session Pooler for IPv4. "getaddrinfo failed" = switch from Direct to Session Pooler.

### Metadata Field Indexing by Backend

SQLite: B-tree via `json_extract` for scalar fields only. PostgreSQL: B-tree for scalars, GIN for arrays/objects. Array/object queries require full table scan in SQLite.

## Docker Deployment

Multi-stage Dockerfile (uv, non-root UID 10001, `/health` endpoint). Configs in `deploy/docker/`: SQLite, PostgreSQL, Supabase. Ollama sidecar in `deploy/docker/ollama/`. Both embedding and summary models are auto-pulled on first startup.

### Docker-Compose Environment Variable Policy

**CRITICAL:** Compose files MUST contain ONLY variables REQUIRED for the deployment to function. All other settings use defaults from `app/settings.py`. This prevents drift between code defaults and hardcoded compose values.

**Required**: Transport (`MCP_TRANSPORT`, `FASTMCP_HOST`, `FASTMCP_PORT`), `FASTMCP_ENABLE_RICH_LOGGING=false`, `LOG_LEVEL=INFO`, storage backend selection, `OLLAMA_HOST` (sidecar), feature toggles, non-default providers/dimensions, `RERANKING_CACHE_DIR`. Model names on Ollama sidecar (`EMBEDDING_MODEL`, `SUMMARY_MODEL`) are required for auto-pull.

**Do NOT add**: Tuning parameters with sensible defaults, feature-specific settings with correct defaults.

## Kubernetes Deployment

Helm chart in `deploy/helm/mcp-context-server/`. Profiles: `values-sqlite.yaml`, `values-postgresql.yaml`. Optional Ollama sidecar, ingress with TLS.

## Windows Development Notes

Use `Path` objects (not string concat). Env vars: `set VAR=value &&` (cmd) or `$env:VAR="value";` (PowerShell). DB path: `%USERPROFILE%\.mcp\context_storage.db`. Docker Desktop for PostgreSQL.

## Debugging and Troubleshooting

```bash
set LOG_LEVEL=DEBUG && uv run mcp-context-server  # Debug logs (Windows)
uv run python -c "from app.startup import init_database; import asyncio; asyncio.run(init_database())"  # Test DB
```

**Common Issues**: Import errors → `uv sync`. Type errors → `uv run mypy app`. Semantic search unavailable → `ENABLE_SEMANTIC_SEARCH=true` + `uv sync --extra embeddings-ollama`. FTS unavailable → `ENABLE_FTS=true`. Summary generation unavailable → `ENABLE_SUMMARY_GENERATION=true` + `uv sync --extra summary-ollama` + `ollama pull qwen3:1.7b`.

## Code Quality Standards

Ruff (127 chars, single quotes), mypy/pyright strict for `app/`. **Never** `from __future__ import annotations` in server.py.

## Critical Implementation Warnings

### Environment Variables — Centralized Configuration

**Never use `os.environ`/`os.getenv()` directly** — always `get_settings()` from `app/settings.py`.

```python
# WRONG: os.getenv('DB_PATH')
# CORRECT: get_settings().storage.db_path
```

Use `Field(alias='ENV_VAR_NAME')`. Update `server.json` for new env vars.

### Settings Class Architecture

**AppSettings must NEVER contain settings fields directly** — it only composes nested settings classes.

When adding new settings:
1. Add to an **existing** settings class if it EXACTLY matches the domain/purpose
2. Create a **new** settings class if no existing class is appropriate — even for a single setting

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

Existing settings classes: `LoggingSettings`, `ToolManagementSettings`, `TransportSettings`, `AuthSettings`, `InstructionsSettings`, `StorageSettings` (extends `BaseSettings`), `EmbeddingSettings`, `SummarySettings`, `SemanticSearchSettings`, `FtsSettings`, `HybridSearchSettings`, `SearchSettings`, `ChunkingSettings`, `RerankingSettings`, `FtsPassageSettings`, `LangSmithSettings`.

### FASTMCP_* Env Var Governance

A `FASTMCP_*` env var belongs in `app/settings.py` (and `server.json`) ONLY when the project can take **programmatic action** with its value: passed to `mcp.run()`, used in application logic, or sets a different default. Env vars consumed by FastMCP at import time with no `mcp.run()` parameter should NOT be in `settings.py`.

| Env Var                       | In settings.py | Reason                                                        |
|-------------------------------|----------------|---------------------------------------------------------------|
| `FASTMCP_HOST`                | YES            | Passed to `mcp.run(host=...)`                                 |
| `FASTMCP_PORT`                | YES            | Passed to `mcp.run(port=...)`                                 |
| `FASTMCP_STATELESS_HTTP`      | YES            | Passed to `mcp.run(stateless_http=...)`                       |
| `FASTMCP_TRANSPORT`           | NO             | Project uses MCP_TRANSPORT + explicit transport= arg          |
| `FASTMCP_ENABLE_RICH_LOGGING` | NO             | Consumed at FastMCP import time; no `mcp.run()` parameter     |

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

**Steps**: 1) Add to `app/tools/<domain>.py` 2) Add to `TOOL_ANNOTATIONS` in `app/tools/__init__.py` 3) Export from `__init__.py` 4) Register in `app/server.py` lifespan() 5) Add TypedDict to `app/types.py` 6) Add tests + real server tests in `test_real_server.py` 7) Update `server.json` if new env vars 8) For backend-specific descriptions, add generator to `app/tools/descriptions.py`

**Annotation categories**: READ_ONLY (readOnlyHint=True), ADDITIVE (destructiveHint=False), UPDATE (destructiveHint=True, idempotentHint=False), DELETE (destructiveHint=True, idempotentHint=True)

### Adding New Providers (Embeddings/Reranking/Summary)

All three layers use identical patterns:
1. Create provider class in `app/<layer>/providers/` implementing the Protocol
2. Add to `PROVIDER_MODULES` and `PROVIDER_CLASSES` dicts in factory.py
3. Add install instructions to `PROVIDER_INSTALL_INSTRUCTIONS`
4. Add optional dependency group in `pyproject.toml`

### Generation-First Transactional Integrity

**CRITICAL**: When `ENABLE_EMBEDDING_GENERATION=true` or `ENABLE_SUMMARY_GENERATION=true` and generation fails, NO data is saved — transaction rolls back completely. Embeddings and summaries are generated OUTSIDE the transaction via `asyncio.gather()`, then all DB ops (context + tags + images + embeddings + summary) in a single atomic `begin_transaction()`. All repository write methods accept optional `txn: TransactionContext` parameter. `_generate_embeddings_with_timeout` is the single source of truth for the timeout/semaphore pattern used by single-entry operations; batch operations have their own generation loop.

### Deduplication Behavior (store_context)

When `store_context` detects a duplicate (same `thread_id + source + text_content` as the latest entry):

- **Metadata**: Updated via `COALESCE(new, existing)`. `None` preserves existing; explicit value replaces.
- **Tags/Images**: REPLACED (not accumulated) when provided. Preserved when `None`.
- **content_type/updated_at**: Auto-updated.
- **Embeddings**: Storage skipped if already exist; generated if missing.
- **Summary**: Regeneration skipped if already exists. `COALESCE(NULL, existing_summary)` preserves pre-existing summary when generation is skipped.
- **Pre-check optimization**: Read-only check before embedding/summary generation to skip LLM API calls for duplicates.

The `store_context_batch` tool uses the same dedup logic (calls `store_with_deduplication` per entry).

### Update Context and Batch Operations

**Update**: Partial updates (only provided fields). Immutable: `id`, `thread_id`, `source`, `created_at`. Auto-managed: `content_type`, `updated_at`. Tags/images: replacement (not merge). Transaction-wrapped.

**Batch**: `store_context_batch`, `update_context_batch`, `delete_context_batch` (up to 100 entries). `atomic=true` (default): all-or-nothing. `atomic=false`: independent processing with per-entry results.

### Known Upstream Bugs and Temporary Patches

**MCP SDK Session Crash on Client Disconnect** (`app/patches/session_crash.py`):

Temporary monkey-patch for MCP Python SDK where `BaseSession._send_response()` and `send_notification()` don't handle `ClosedResourceError`/`BrokenResourceError` on client disconnect during long-running tool execution. Applied in `app/server.py` lifespan (step 0).

- **Upstream tracking**: [MCP SDK #2064](https://github.com/modelcontextprotocol/python-sdk/issues/2064), PRs [#2072](https://github.com/modelcontextprotocol/python-sdk/pull/2072), [#2184](https://github.com/modelcontextprotocol/python-sdk/pull/2184)
- **Removal**: When upstream MCP SDK fixes this, update `mcp` dependency, delete `app/patches/`, remove patch import/call from `app/server.py`, delete `tests/test_session_crash_patch.py`, remove `test_session_crash_patch_applied` from `tests/test_real_server.py`, remove this section.
