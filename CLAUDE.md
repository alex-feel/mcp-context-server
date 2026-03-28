# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# IMPORTANT: ALL development tasks (coding, testing, type-checking, pre-commit)
# REQUIRE the full sync command below. Bare `uv sync` is insufficient and will
# cause type-checker errors due to missing optional dependencies.

# Build and run
uv sync --all-extras --all-groups          # Install ALL dependencies (REQUIRED for development)
uv run mcp-context-server                  # Start server (aliases: mcp-context, python -m app.server)
uvx mcp-context-server                     # Run from PyPI

# Testing
uv run pytest                              # Run all tests
uv run pytest tests/server/test_server.py -v      # Run specific test file
uv run pytest tests/server/test_server.py::TestStoreContext::test_store_text_context -v  # Single test
uv run pytest --cov=app --cov-report=html  # Run with coverage
uv run pytest -m "not integration"         # Skip slow tests for quick feedback

# Code quality
uv run pre-commit run --all-files          # Lint + type check (Ruff, mypy, pyright)
uv run ruff check --fix .                  # Ruff linter with autofix
```

Note: Integration test infrastructure currently exists only for SQLite. PostgreSQL integration tests are planned for the future.

## High-Level Architecture

### MCP Protocol Integration

[Model Context Protocol](https://modelcontextprotocol.io) (MCP) server with JSON-RPC 2.0, automatic tool discovery, Pydantic validation, multi-transport (default: stdio, HTTP, streamable-http, SSE), and tool annotations (readOnlyHint, destructiveHint, idempotentHint). Compatible with Claude Desktop, Claude Code, LangGraph, and any MCP client.

### MCP Server Architecture

FastMCP 3.1.x-based server providing persistent context storage for LLM agents:

1. **FastMCP Server Layer** (`app/server.py`, `app/tools/`, `app/startup/`):
   - Entry point with FastMCP instance, lifespan management, and main() function
   - Tool implementations in `app/tools/` organized by domain: `context.py` (CRUD), `search.py` (4 search tools), `discovery.py` (list_threads, get_statistics), `batch.py` (batch CRUD), `descriptions.py` (backend-specific dynamic tool descriptions), `_shared.py` (internal shared infrastructure: per-entry processing, image validation, generation with timeout, transaction execution, response message builders -- consumed by `context.py` and `batch.py`, not re-exported via `__init__.py`)
   - Dynamic tool registration via `register_tool()` from `app/tools/__init__.py`
   - Provides `/health` endpoint for container orchestration (HTTP transport only)
   - Global state and initialization in `app/startup/` package: `init_database()`, `ensure_repositories()`, `set_summary_provider()`/`get_summary_provider()`
   - **Temporary Patches** (`app/patches/`): Monkey-patches for upstream MCP SDK bugs applied at startup.
   - **Middleware** (`app/middleware/`): Schema-aware `JsonStringDeserializerMiddleware` for MCP client compatibility (FastMCP 3.1.x Middleware API). Registered via `mcp.add_middleware()` in lifespan() after all tool registrations.
   - Both are documented in "Known Upstream Bugs and Temporary Patches" with upstream tracking and removal instructions.

2. **Authentication Layer** (`app/auth/simple_token.py`): Bearer token auth for HTTP transport with constant-time comparison. Configured via `MCP_AUTH_PROVIDER` and `MCP_AUTH_TOKEN`.

3. **Storage Backend Layer** (`app/backends/`):
   - **StorageBackend Protocol** (`base.py`): Database-agnostic interface (8 methods including `begin_transaction()`)
   - **TransactionContext Protocol** (`base.py`): Provides `connection` and `backend_type` for multi-operation atomic transactions
   - **SQLiteBackend**: Zero-config, connection pooling, write queue, circuit breaker, single-user
   - **PostgreSQLBackend**: Async via asyncpg, connection pooling, MVCC, JSONB/GIN indexes, pgvector, Pgpool-II auto-detection (disables prepared statements)
   - **Backend Factory** (`factory.py`): Creates backend based on `STORAGE_BACKEND` env var

4. **Repository Pattern** (`app/repositories/`):
   - **RepositoryContainer** (`__init__.py`): DI container for all repositories
   - Repositories: Context (CRUD, search, deduplication), Tag (normalization, many-to-many), Image (binary attachments), Statistics, Embedding (vector storage/search), Fts (FTS5/tsvector)
   - All repositories use `StorageBackend` protocol — database-agnostic
   - `BaseRepository` provides `_placeholder()`, `_placeholders()`, `_json_extract()` helpers

5. **Data Models** (`app/models.py`): Pydantic V2 with `StrEnum` for Python 3.12+. Main models: `ContextEntry`, `ImageAttachment`, `StoreContextRequest`. Base64 image encoding with configurable size limits.

6. **Provider Layers** — All use `<Name>Provider` Protocol + factory with dynamic imports (`PROVIDER_MODULES`/`PROVIDER_CLASSES` dicts):
   - **Embeddings** (`app/embeddings/`): Ollama, OpenAI, Azure, HuggingFace, Voyage. Retry via tenacity (`retry.py`). Context limits (`context_limits.py`). LangSmith tracing (`tracing.py`).
   - **Reranking** (`app/reranking/`): FlashRank (default, 34MB model, ONNX inference offloaded to thread pool).
   - **Summary** (`app/summary/`): Ollama, OpenAI, Anthropic. Retry via tenacity. Context limits (`context_limits.py`). Default model: `qwen3:0.6b`. Prompt: `DEFAULT_SUMMARY_PROMPT` in `instructions.py`, configurable via `SUMMARY_PROMPT` env var.

7. **Services Layer** (`app/services/`): `ChunkingService` (`TextChunk` dataclass, `split_text()`, LangChain's `RecursiveCharacterTextSplitter`). `PassageExtractionService` (`extract_rerank_passage()`, `HighlightRegion` dataclass).

8. **Metadata Filtering** (`app/metadata_types.py` & `app/query_builder.py`): `MetadataFilter` with 16 operators. `QueryBuilder`: backend-aware SQL with nested JSON paths. Handles SQLite (`json_extract`) vs PostgreSQL (`->>`/`->`) operators.

9. **Other modules**: `app/fusion.py` (RRF algorithm), `app/errors.py` (error classification + exception formatting, see Key Implementation Details #5), `app/instructions.py` (server instructions), `app/types.py` (40+ TypedDicts for API responses), `app/logger_config.py` (logging configuration), `app/schemas/` (SQL schema files).

### Thread-Based Context Management

Agents share context via `thread_id`. Entries tagged with `source`: 'user' or 'agent'. Filter by thread, source, tags, content type, or metadata (16 operators). Flat structure (no hierarchy).

### Database Schema

Tables: `context_entries` (main, with thread_id/source indexes, JSON metadata, summary column), `tags` (many-to-many, lowercase), `image_attachments` (binary, cascade delete).

**Performance**: WAL mode, 256MB mmap, compound index (thread_id, source). Indexed metadata: `status`, `agent_name`, `task_name`, `project`, `report_type`. See "Metadata Field Indexing by Backend" for per-backend details.

### Search Tools

`search_context` (exact keyword), `semantic_search_context` (vector similarity), `fts_search_context` (linguistic + snippets), `hybrid_search_context` (RRF fusion of FTS + semantic).

**FTS modes**: `match` (default, stemming), `prefix` (autocomplete), `phrase` (exact order), `boolean` (AND/OR/NOT). SQLite: FTS5 with BM25, Porter stemmer. PostgreSQL: tsvector/tsquery with ts_rank, 29 languages.

**Hybrid search**: RRF formula `score(d) = Σ(1 / (k + rank_i(d)))`. Parallel execution, graceful degradation. Adaptive FTS mode: queries with `HYBRID_FTS_OR_THRESHOLD` (default 4) or more significant terms switch from AND to OR for improved recall.

**Response**: `results` (array), `count` (int), `stats` (only when `explain_query=True`).

### Migration System

Auto-applied idempotent migrations in `app/migrations/`: semantic search, FTS, chunking (1:N embeddings), metadata indexing, summary column. PostgreSQL migrations use `POSTGRESQL_MIGRATION_TIMEOUT_S` (300s default) for DDL operations. Changing `FTS_LANGUAGE` requires FTS table rebuild.

### Testing Strategy

**Philosophy**: Tests use SQLite-only temp databases (no PostgreSQL required). Production supports both backends. Always add real server integration tests in `tests/integration/sqlite/test_real_server.py` for new tools.

**Key Files**: `conftest.py` (fixtures, markers), `helpers.py` (shared test utilities -- uses `get_settings()` for configuration), `run_server.py` (subprocess server wrapper for integration tests).

**Key Fixtures** (`conftest.py`): `test_db` (direct SQLite), `mock_server_dependencies` (mocked settings), `initialized_server` (full integration), `async_db_initialized` (async backend), `async_db_with_embeddings` (semantic search).

**Skip Markers**: `@requires_ollama`, `@requires_sqlite_vec`, `@requires_numpy`, `@requires_semantic_search`

`prevent_default_db_pollution` (autouse) prevents accidental production DB access.

### Test Directory Structure

Tests mirror `app/` structure: `tests/<name>/` → `app/<name>/` (package) or `app/<name>.py` (module).

**Non-trivial mappings**:
- `tests/core/` → `app/*.py` small utility root modules (models, errors, fusion, instructions, etc.)
- `tests/server/` → `app/server.py` (dedicated directory; large/complex root modules get their own)
- `tests/settings/` → `app/settings.py` (same reason)
- `tests/integration/sqlite/` → real running server integration tests (no app mirror)

**Shared infrastructure** stays at `tests/` root: `conftest.py`, `helpers.py`, `run_server.py`, `__init__.py`.

**Placement rule**: Follow the PRIMARY source code module under test. Use import analysis as the arbiter when ambiguous.

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

6. **Server Instructions**: Optional `instructions` field in MCP `InitializeResult`. Configured via `MCP_SERVER_INSTRUCTIONS` env var (overrides `DEFAULT_INSTRUCTIONS` from `app/instructions.py`). Empty string disables. Includes `## Skill Integration` section directing agents to discover and apply context-server-related Skills.

## Package and Release

uv + Hatchling. Entry points: `mcp-context-server`, `mcp-context`. Python 3.12+. Optional extras: `embeddings-ollama`, `embeddings-openai`, `embeddings-azure`, `embeddings-huggingface`, `embeddings-voyage`, `summary-ollama`, `summary-openai`, `summary-anthropic`, `reranking`, `langsmith`.

[Release Please](https://github.com/googleapis/release-please) for automated releases via [Conventional Commits](https://www.conventionalcommits.org/). On `release:published`: PyPI package, MCP Registry (`server.json`), GHCR Docker images (amd64/arm64): default Ollama variant and `ollama-openai` variant.

### SECURITY.md Maintenance

When a commit triggers a major version bump (Conventional Commit `!` suffix or `BREAKING CHANGE:` in body/footer), update the Supported Versions table in `SECURITY.md` in the same PR: add the new major version as supported, mark the previous major version as unsupported.

## Documentation Maintenance

When changing core functionality, update the corresponding doc before committing:

- **`README.md`** -- Update when adding, changing, or removing user-facing features. The README is the first thing users read; its Key Features list must reflect current capabilities.
- Update any other related documents in `docs/`.

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

When adding or modifying environment variables in `app/settings.py`, update **both** `server.json` **and** `docs/environment-variables.md`.

## Environment Variables

Configuration via `.env` file or environment. **Canonical source**: `app/settings.py` — all env vars with defaults, descriptions, and validation.

**Core**: `STORAGE_BACKEND` (sqlite*/postgresql), `LOG_LEVEL` (ERROR*), `DB_PATH`, `MAX_IMAGE_SIZE_MB` (10*), `MAX_TOTAL_SIZE_MB` (100*), `DISABLED_TOOLS`

**Transport**: `MCP_TRANSPORT` (stdio*/http/streamable-http/sse), `FASTMCP_HOST` (0.0.0.0*), `FASTMCP_PORT` (8000*), `FASTMCP_STATELESS_HTTP` (true*)

**FastMCP Logging** (deployment-only, NOT in `app/settings.py`): `FASTMCP_ENABLE_RICH_LOGGING` (true*; set `false` in Docker/cloud). Read directly by FastMCP at import time — see [FASTMCP_* Governance](#fastmcp_-env-var-governance).

**Auth**: `MCP_AUTH_PROVIDER` (none*/simple_token), `MCP_AUTH_TOKEN`, `MCP_AUTH_CLIENT_ID` (mcp-client*)

**Feature Toggles**: `ENABLE_EMBEDDING_GENERATION` (true*), `ENABLE_SEMANTIC_SEARCH` (false*), `ENABLE_FTS` (false*), `ENABLE_HYBRID_SEARCH` (false*), `ENABLE_CHUNKING` (true*), `ENABLE_RERANKING` (true*), `ENABLE_SUMMARY_GENERATION` (true*)

**Embedding**: `EMBEDDING_PROVIDER` (ollama*/openai/azure/huggingface/voyage), `EMBEDDING_MODEL` (qwen3-embedding:0.6b*), `EMBEDDING_DIM` (1024*), `EMBEDDING_TIMEOUT_S` (240*), `EMBEDDING_MAX_CONCURRENT` (3*)

**Summary**: `SUMMARY_PROVIDER` (ollama*/openai/anthropic), `SUMMARY_MODEL` (qwen3:0.6b*), `SUMMARY_MAX_TOKENS` (2000*), `SUMMARY_MIN_CONTENT_LENGTH` (500*; text shorter than this skips summary (truncated preview is sufficient); 0 = always generate), `SUMMARY_PROMPT`

**Provider-specific, PostgreSQL, reranking, chunking, hybrid, FTS, search, and metadata indexing vars**: See `app/settings.py` for complete list with defaults and descriptions.

*\* = default value*

## Storage Backend Configuration

### SQLite (Default)
Zero-config local storage. See SQLiteBackend in Architecture for features.

### PostgreSQL
```bash
docker run --name pgvector18 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mcp_context -p 5432:5432 -d pgvector/pgvector:pg18-trixie
export STORAGE_BACKEND=postgresql
uv run mcp-context-server  # Auto-initializes schema, enables pgvector
```

### Supabase
`STORAGE_BACKEND=postgresql` + `POSTGRESQL_CONNECTION_STRING`. Session Pooler for IPv4. "getaddrinfo failed" = switch from Direct to Session Pooler.

### Metadata Field Indexing by Backend

SQLite: B-tree via `json_extract` for scalar fields only. PostgreSQL: B-tree for scalars, GIN for arrays/objects. Array/object queries require full table scan in SQLite.

## Docker Deployment

Multi-stage Dockerfile (uv, non-root UID 10001, `/health` endpoint). Configs in `deploy/docker/`: SQLite, PostgreSQL, Supabase. Ollama sidecar in `deploy/docker/ollama/`. Both embedding and summary models are auto-pulled on first startup.

### Docker Compose File Naming Convention

**Naming formula:** `docker-compose.{storage}.{providers}[.local].yml`

| Segment             | Values                                        | Description                                                                                                                                                               |
|---------------------|-----------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `{storage}`         | `sqlite`, `postgresql`, `postgresql-external` | Database backend                                                                                                                                                          |
| `{providers}`       | `ollama`, `openai`, `ollama-openai`           | Embedding + summary provider combination. Single name when both use same provider; hyphenated `<embedding>-<summary>` when they differ.                                   |
| `.local` (optional) | Present or absent                             | Present ONLY for provider combinations that have a published GHCR image, indicating the file builds that same configuration locally instead of pulling from the registry. |

**Five image source categories:**

| Category                         | Files                        | `image:`                                                    | `pull_policy` | `build:` block                                  |
|----------------------------------|------------------------------|-------------------------------------------------------------|---------------|-------------------------------------------------|
| GHCR pull                        | `*.ollama.yml`               | `ghcr.io/alex-feel/mcp-context-server:latest`               | `always`      | None                                            |
| GHCR pull (variant)              | `*.ollama-openai.yml`        | `ghcr.io/alex-feel/mcp-context-server:latest-ollama-openai` | `always`      | None                                            |
| Local build (GHCR equivalent)    | `*.ollama.local.yml`         | `mcp-context-server`                                        | `build`       | Yes (no custom args, uses Dockerfile defaults)  |
| Local build (variant equivalent) | `*.ollama-openai.local.yml`  | `mcp-context-server`                                        | `build`       | Yes (with SUMMARY_EXTRA=summary-openai)         |
| Local build (provider-specific)  | `*.openai.yml`               | `mcp-context-server`                                        | `build`       | Yes (with EMBEDDING_EXTRA / SUMMARY_EXTRA args) |

**Extensibility rule:** When adding new provider combinations -- if a GHCR image is published for the combination, create both `*.{providers}.yml` (GHCR pull) and `*.{providers}.local.yml` (local build). If no GHCR image exists, create only `*.{providers}.yml` (local build, no `.local` variant needed).

### Docker-Compose Environment Variable Policy

**CRITICAL:** Compose files MUST contain ONLY variables REQUIRED for the deployment to function. All other settings use defaults from `app/settings.py`. This prevents drift between code defaults and hardcoded compose values.

**Configurable variables** use `${VAR:-default}` interpolation (default MUST match previously hardcoded value): `LOG_LEVEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_PROVIDER`, `SUMMARY_MODEL`, `SUMMARY_PROVIDER`. PostgreSQL variants add: `POSTGRESQL_USER`, `POSTGRESQL_PASSWORD`, `POSTGRESQL_DATABASE`. Ollama sidecar model names use the same interpolation to stay in sync.

**Required (hardcoded, NOT configurable)**: Transport (`MCP_TRANSPORT`, `FASTMCP_HOST`, `FASTMCP_PORT`), `FASTMCP_ENABLE_RICH_LOGGING=false`, storage backend selection, `OLLAMA_HOST` (sidecar bind + client URL), `OLLAMA_KEEP_ALIVE=-1`, feature toggles (`ENABLE_*`), container paths (`DB_PATH`, `RERANKING_CACHE_DIR`), Docker networking (`POSTGRESQL_HOST`, `POSTGRESQL_PORT`).

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

**Common Issues**: Import errors → `uv sync`. Type errors → `uv run mypy app`. Semantic search unavailable → `ENABLE_SEMANTIC_SEARCH=true` + `uv sync --extra embeddings-ollama`. FTS unavailable → `ENABLE_FTS=true`. Summary generation unavailable → `ENABLE_SUMMARY_GENERATION=true` + `uv sync --extra summary-ollama` + `ollama pull qwen3:0.6b`.

## Code Quality Standards

Ruff (127 chars, single quotes), mypy/pyright strict for `app/`.

## GitHub Actions Security Policy

All GitHub Actions workflows MUST follow these rules:

1. **Never use mutable branch references** (`@main`, `@master`, `@develop`, `@release/vN`) for third-party actions. Mutable refs can be compromised (e.g., March 2026 `aquasecurity/trivy-action` incident — exfiltrated CI secrets).

2. **Version tag pinning is the project standard.** Pin third-party actions to immutable version tags (e.g., `@v5`, `@v1.13.0`). SHA pinning not required — version tags are immutable enough and Dependabot-compatible.

3. **Verify action runtimes before updating.** Before bumping an action version, check its runtime to avoid deprecated Node.js versions:
   ```bash
   # Check action runtime (note: some actions use action.yaml instead of action.yml)
   gh api repos/OWNER/REPO/contents/action.yml --jq '.content' | base64 -d | grep -E 'using:.*node'
   ```

## Critical Implementation Warnings

### Environment Variables — Centralized Configuration

**Never use `os.environ`/`os.getenv()` directly** — always `get_settings()` from `app/settings.py`.

```python
# WRONG: os.getenv('DB_PATH')
# CORRECT: get_settings().storage.db_path
```

Use `Field(alias='ENV_VAR_NAME')`.

### Settings Singleton Caching (`@lru_cache`)

`get_settings()` is `@lru_cache`-decorated — a process-lifetime singleton. Once called, env var changes are ignored.

**In tests**, use `get_settings.cache_clear()` to invalidate the cache when environment variables change between operations:

```python
from app.settings import get_settings
monkeypatch.setenv('SOME_SETTING', 'new-value')
get_settings.cache_clear()  # Next call creates fresh AppSettings
```

For test modules that modify settings across multiple tests, use an autouse fixture (established pattern from `tests/reranking/conftest.py`):

```python
@pytest.fixture(autouse=True)
def clear_settings_cache() -> None:
    get_settings.cache_clear()
```

**Anti-pattern: Premature `get_settings()` in subprocess scripts.** `tests/run_server.py` configures environment via `os.environ` (intentional — it's an env configurator, not a settings consumer). Call `get_settings.cache_clear()` after all `os.environ` modifications before launching the server, or utility functions like `is_ollama_model_available()` will cache stale defaults.

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

Existing settings classes: `LoggingSettings`, `ToolManagementSettings`, `TransportSettings`, `AuthSettings`, `InstructionsSettings`, `OllamaSettings`, `StorageSettings` (extends `BaseSettings`), `EmbeddingSettings`, `SummarySettings`, `SemanticSearchSettings`, `FtsSettings`, `HybridSearchSettings`, `SearchSettings`, `ChunkingSettings`, `RerankingSettings`, `FtsPassageSettings`, `LangSmithSettings`.

### FASTMCP_* Env Var Governance

`FASTMCP_*` env vars belong in `app/settings.py` (and `server.json`) ONLY when the project takes **programmatic action** with them (passed to `mcp.run()`, used in logic). Import-time-only vars should NOT be in `settings.py`.

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

**Steps**: 1) Add to `app/tools/<domain>.py` 2) Add to `TOOL_ANNOTATIONS` in `app/tools/__init__.py` 3) Export from `__init__.py` 4) Register in `app/server.py` lifespan() 5) Add TypedDict to `app/types.py` 6) Add tests + real server tests in `tests/integration/sqlite/test_real_server.py` 7) Update `server.json` if new env vars 8) For backend-specific descriptions, add generator to `app/tools/descriptions.py` 9) For store/update operations, use shared functions from `app/tools/_shared.py` (image validation, generation with timeout, transaction execution, response builders) to maintain behavioral parity with existing tools

**Annotation categories**: READ_ONLY (readOnlyHint=True), ADDITIVE (destructiveHint=False), UPDATE (destructiveHint=True, idempotentHint=False), DELETE (destructiveHint=True, idempotentHint=True)

### Adding New Providers (Embeddings/Reranking/Summary)

All three layers use identical patterns:
1. Create provider class in `app/<layer>/providers/` implementing the Protocol
2. Add to `PROVIDER_MODULES` and `PROVIDER_CLASSES` dicts in factory.py
3. Add install instructions to `PROVIDER_INSTALL_INSTRUCTIONS`
4. Add optional dependency group in `pyproject.toml`

### Generation-First Transactional Integrity

**CRITICAL**: When generation is enabled and fails, NO data is saved — transaction rolls back. Flow: generate embeddings/summaries OUTSIDE transaction via `asyncio.gather(*tasks, return_exceptions=True)`, then all DB ops in a single atomic `begin_transaction()`. All repository write methods accept optional `txn: TransactionContext`. `generate_embeddings_with_timeout` and `generate_summary_with_timeout` in `app/tools/_shared.py` are the single sources of truth for timeout/semaphore, used by `store_context`, `update_context`, `store_context_batch`, `update_context_batch`. Each `gather` result is independently inspected — failed generation raises (or collected in non-atomic batch mode) without cancelling the other task.

**NEVER propose "graceful skip" of generation when generation is enabled.** If `ENABLE_EMBEDDING_GENERATION=true` and embedding generation fails, or `ENABLE_SUMMARY_GENERATION=true` and summary generation fails, the entry MUST NOT be saved. There is no "store without embeddings" or "store without summary" fallback when generation is enabled. This is non-negotiable mandatory behavior. The only way to skip generation is to explicitly disable it via `ENABLE_EMBEDDING_GENERATION=false` or `ENABLE_SUMMARY_GENERATION=false`. Only the user can make this decision.

### Deduplication Behavior (store_context)

Deduplication serves as **retry protection**: when an MCP client retransmits the same message (network glitch, timeout retry), the server updates the existing entry instead of creating duplicates. This is distinct from a **new conversational turn**, where a user intentionally sends identical text after an agent has responded -- in that case, a new entry must be created to preserve chronological ordering.

**Interleaving check**: Before deduplicating, the server verifies that no opposite-source entries (agent entries for user source, user entries for agent source) were created after the candidate duplicate. If such entries exist, the message represents a new conversational turn and is inserted as a new entry. This uses `id > candidate_id` comparison (immune to clock skew) with the existing `idx_thread_source` index.

When deduplication proceeds (same `thread_id + source + text_content` as the latest entry, with no interleaving):

- **Metadata**: Updated via `COALESCE(new, existing)`. `None` preserves existing; explicit value replaces.
- **Tags/Images**: REPLACED (not accumulated) when provided. Preserved when `None`.
- **content_type/updated_at**: Auto-updated.
- **Embeddings**: Storage skipped if already exist; generated if missing.
- **Summary**: Regeneration skipped if already exists. `COALESCE(NULL, existing_summary)` preserves pre-existing summary when generation is skipped.
- **Pre-check optimization**: Read-only check before embedding/summary generation to skip LLM API calls for duplicates. Applied in both `store_context` and `store_context_batch`.

The `store_context_batch` tool uses the same dedup logic (calls `store_with_deduplication` per entry) with the same pre-check optimization and interleaving check.

### Update Context and Batch Operations

**Update**: Partial updates (only provided fields). Immutable: `id`, `thread_id`, `source`, `created_at`. Auto-managed: `content_type`, `updated_at`. Tags/images: replacement (not merge). Transaction-wrapped.

**Batch**: `store_context_batch`, `update_context_batch`, `delete_context_batch` (up to 100 entries). `atomic=true` (default): all-or-nothing. `atomic=false`: independent processing with per-entry results. Batch and non-batch tools share per-entry processing logic (image validation, transaction execution, response message building) via `app/tools/_shared.py` to guarantee behavioral parity.

### Known Upstream Bugs and Temporary Patches

**MCP SDK Session Crash on Client Disconnect** (`app/patches/session_crash.py`):

Monkey-patch for `BaseSession._send_response()`/`send_notification()` not handling `ClosedResourceError`/`BrokenResourceError` on client disconnect. Applied in `app/server.py` lifespan (step 0).

- **Upstream tracking**: [MCP SDK #2064](https://github.com/modelcontextprotocol/python-sdk/issues/2064), PRs [#2072](https://github.com/modelcontextprotocol/python-sdk/pull/2072), [#2184](https://github.com/modelcontextprotocol/python-sdk/pull/2184)
- **Removal**: When upstream MCP SDK fixes this, update `mcp` dependency, delete `app/patches/`, remove patch import/call from `app/server.py`, delete `tests/patches/test_session_crash_patch.py`, remove `test_session_crash_patch_applied` from `tests/integration/sqlite/test_real_server.py`, remove this section.

**Client JSON String Serialization** (`app/middleware/json_string_deserializer.py`):

Schema-aware FastMCP middleware fixing MCP clients (including Claude Code) intermittently serializing list/dict parameters as JSON strings. Uses `Middleware` base class with `on_call_tool` override. `build_schema_map()` inspects each tool's JSON Schema at startup for `array`/`object` parameters (including `Optional` via `anyOf`/`$ref`). Only those parameters are deserialization candidates — strings are never touched. Handles double-encoding. Registered in `app/server.py` lifespan (step 22) via `mcp.add_middleware()`, after all `register_tool()` calls.

- **Upstream tracking**: [Claude Code #22394](https://github.com/anthropics/claude-code/issues/22394) (closed NOT_PLANNED), [Claude Code #26094](https://github.com/anthropics/claude-code/issues/26094), [FastMCP #932](https://github.com/jlowin/fastmcp/issues/932), Claude Code #5504, #4192, #3084
- **Removal**: When upstream clients fix their serialization, delete `app/middleware/`, remove middleware import and registration block (step 22) from `app/server.py` lifespan(), delete `tests/middleware/test_middleware_json_deserializer.py`, remove middleware integration tests from `tests/integration/sqlite/test_real_server.py`, remove this section.
