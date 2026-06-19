# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Reference

```bash
# IMPORTANT: dev tasks need the full sync below; bare `uv sync` lacks optional
# extras and causes type-checker errors.

# Build and run
uv sync --all-extras --all-groups   # install ALL deps (REQUIRED for dev)
uv run mcp-context-server           # start server (aliases: mcp-context, python -m app.server)
uvx mcp-context-server              # run from PyPI

# Testing
uv run pytest                       # all tests
uv run pytest tests/server/test_server.py -v   # one file
uv run pytest tests/server/test_server.py::TestStoreContext::test_store_text_context -v   # one test
uv run pytest --cov=app --cov-report=html   # coverage
uv run pytest -m "not integration"  # skip slow tests

# Code quality
uv run pre-commit run --all-files   # lint + type check (Ruff, mypy, pyright)
uv run ruff check --fix .           # Ruff autofix
```

Note: Real-server integration tests run against BOTH backends from one shared, backend-parametrized harness (`tests/integration/_harness.py`, `MCPServerIntegrationTest`): SQLite always, and PostgreSQL via a docker-compose pgvector container (the `pg_test_url` fixture / `@requires_docker_postgres` marker; skipped automatically when Docker is unavailable).

## High-Level Architecture

### MCP Protocol Integration

[Model Context Protocol](https://modelcontextprotocol.io) (MCP) server with JSON-RPC 2.0, automatic tool discovery, Pydantic validation, multi-transport (default stdio; also HTTP, streamable-http, SSE), and tool annotations (readOnlyHint, destructiveHint, idempotentHint). Compatible with Claude Desktop, Claude Code, LangGraph, and any MCP client.

### MCP Server Architecture

FastMCP 3.1.x-based server providing persistent context storage for LLM agents:

1. **FastMCP Server Layer** (`app/server.py`, `app/tools/`, `app/startup/`):
   - Entry point: FastMCP instance, lifespan, `main()`, `/health` endpoint (HTTP transport only); tool registration via `register_tool()` from `app/tools/__init__.py`.
   - Tools in `app/tools/` by domain: `context.py` (CRUD), `search.py` (4 search tools), `discovery.py` (list_threads, get_statistics), `batch.py` (batch CRUD), `descriptions.py` (backend-specific dynamic descriptions), `_shared.py` (internal infra: per-entry processing, image validation, generation-with-timeout, transaction execution, response builders — used by `context.py`/`batch.py`, not re-exported via `__init__.py`).
   - Global state/init in `app/startup/`: `init_database()`, `ensure_repositories()`, `set_summary_provider()`/`get_summary_provider()`.
   - **Temporary Patches** (`app/patches/`): startup monkey-patches for upstream MCP SDK bugs. **Middleware** (`app/middleware/`): schema-aware `JsonStringDeserializerMiddleware` (FastMCP 3.1.x Middleware API), registered via `mcp.add_middleware()` in lifespan() after all tool registrations. (Both detailed under "Known Upstream Bugs and Temporary Patches".)

2. **Authentication Layer** (`app/auth/simple_token.py`): Bearer token auth for HTTP transport, constant-time comparison. Via `MCP_AUTH_PROVIDER` + `MCP_AUTH_TOKEN`.

3. **Storage Backend Layer** (`app/backends/`):
   - **StorageBackend / TransactionContext Protocols** (`base.py`): database-agnostic interface (8 methods incl. `begin_transaction()`); `TransactionContext` exposes `connection`/`backend_type` for atomic multi-op transactions.
   - **SQLiteBackend**: zero-config, connection pooling, write queue, circuit breaker, single-user.
   - **PostgreSQLBackend**: async via asyncpg, connection pooling, MVCC, JSONB/GIN indexes, pgvector, Pgpool-II auto-detection (disables prepared statements).
   - **Backend Factory** (`factory.py`): creates backend from `STORAGE_BACKEND`.

4. **Repository Pattern** (`app/repositories/`): **RepositoryContainer** (`__init__.py`) = DI container. Repositories: Context (CRUD, search, deduplication), Tag (normalization, many-to-many), Image (binary attachments), Statistics, Embedding (vector storage/search), Fts (FTS5/tsvector). All use the `StorageBackend` protocol (database-agnostic); `BaseRepository` provides `_placeholder()`, `_placeholders()`, `_json_extract()`.

5. **Data Models** (`app/models.py`): Pydantic V2 with `StrEnum` for Python 3.12+. Main models: `ContextEntry`, `ImageAttachment`, `StoreContextRequest`. Base64 image encoding with configurable size limits.

6. **Provider Layers** — all use `<Name>Provider` Protocol + factory with dynamic imports (`PROVIDER_MODULES`/`PROVIDER_CLASSES` dicts):
   - **Embeddings** (`app/embeddings/`): Ollama, OpenAI, Azure, HuggingFace, Voyage. tenacity retry (`retry.py`), context limits (`context_limits.py`), LangSmith tracing (`tracing.py`).
   - **Reranking** (`app/reranking/`): FlashRank (default, 34MB model, ONNX inference offloaded to thread pool).
   - **Summary** (`app/summary/`): Ollama, OpenAI, Anthropic. tenacity retry, context limits (`context_limits.py`). Default model `qwen3:0.6b`; prompt `DEFAULT_SUMMARY_PROMPT` in `instructions.py` (via `SUMMARY_PROMPT`).
   - **Compression** (`app/compression/`): TurboQuant (default ON in v3.0.0; `ENABLE_EMBEDDING_COMPRESSION=false` to opt out). Wire types in `app/compression/providers/turboquant/_types.py` (discriminated `MSEPayload | IPPayload` union; dispatcher `payload_from_bytes(blob)`). Cached `get_cached_compression_provider()` (`app/compression/factory.py`); startup validator + provenance helpers (`app/startup/compression_validator.py`, `app/compression/provenance.py` — see "Compression Seed-Locked Invariant"). CPU-bound concurrency via `_compression_semaphore` in `app/tools/_shared.py` (held INSIDE `_encode_one`, separate from the I/O-bound embedding/summary semaphores). Read via `EmbeddingRepository.search_compressed()` (from `search()`); write via the chunked flow into `vec_context_embeddings_compressed`. Internals: `docs/embedding-compression.md`.

7. **Services Layer** (`app/services/`): `ChunkingService` (`TextChunk` dataclass, `split_text()`, LangChain `RecursiveCharacterTextSplitter`); `PassageExtractionService` (`extract_rerank_passage()`, `HighlightRegion` dataclass).

8. **Metadata Filtering** (`app/metadata_types.py` & `app/query_builder.py`): `MetadataFilter` (16 operators); `QueryBuilder` builds backend-aware SQL with nested JSON paths — SQLite `json_extract` vs PostgreSQL `->>`/`->`.

9. **Other modules**: `app/fusion.py` (RRF), `app/errors.py` (error classification + exception formatting, see #5), `app/ids.py` (UUIDv7 generation + ID normalization, see #7), `app/cli/migrate.py` (migration CLI, see "Package and Release"), `app/instructions.py` (server instructions), `app/types.py` (40+ API-response TypedDicts), `app/logger_config.py` (logging), `app/schemas/` (SQL schema files).

### Thread-Based Context Management

Agents share context via `thread_id`. Entries tagged `source`: 'user' or 'agent'. Filter by thread, source, tags, content type, or metadata (16 operators). Flat structure (no hierarchy).

### Database Schema

Tables: `context_entries` (main; thread_id/source indexes, JSON metadata, summary column), `tags` (many-to-many, lowercase), `image_attachments` (binary, cascade delete).

**Public primary key (`context_entries.id`)**: 32-character lowercase hex UUIDv7. SQLite: `TEXT NOT NULL UNIQUE`; PostgreSQL: native `UUID NOT NULL PRIMARY KEY`. All foreign keys (`tags.context_entry_id`, `image_attachments.context_entry_id`, `embedding_metadata.context_id`, `embedding_chunks.context_id`, `vec_context_embeddings.context_id`) reference `context_entries(id)` — TEXT on SQLite, UUID on PostgreSQL.

**SQLite `rowid_int` surrogate**: on SQLite, `context_entries` carries an extra `rowid_int INTEGER PRIMARY KEY AUTOINCREMENT` — a stable INTEGER rowid backing the FTS5 external-content table, immune to VACUUM renumbering, never exposed at the MCP boundary. PostgreSQL needs no equivalent (its native `UUID` PRIMARY KEY is layout-independent).

**Embedding chunk and vec0 INTEGER bridge (preserved)**: chunking maintains a 1:N context→embedding mapping via `embedding_chunks` with INTEGER rowid bridges to vec0. `embedding_chunks.id`, `embedding_chunks.vec_rowid`, `vec_context_embeddings.id` stay INTEGER/BIGSERIAL; only the outer `context_id` FK is TEXT/UUID.

**Performance**: WAL mode, 256MB mmap, compound index (thread_id, source). Indexed metadata: `status`, `agent_name`, `task_name`, `project`, `report_type`. See "Metadata Field Indexing by Backend" for per-backend details.

**Optional compressed embedding storage**: with `ENABLE_EMBEDDING_COMPRESSION=true`, the fp32 `vec_context_embeddings` table is REPLACED by `vec_context_embeddings_compressed` (BLOB on SQLite, BYTEA on PostgreSQL) plus a singleton `compression_metadata(id INTEGER PRIMARY KEY CHECK (id = 1), provider, bits, variant, seed, dim, created_at)` provenance row. The `CHECK (id = 1)` singleton blocks accidental duplicate rows (the rotation seed is load-bearing). On PostgreSQL, `idx_vec_context_embeddings_hnsw` is dropped during `--compress` and recreated during `--decompress`. Compressed payloads have no fixed width (depends on `dim`/`bits`/`variant`); wire format (magic prefix, variant code, per-variant headers incl. the IP variant's explicit 1-byte `mse_bits` the decoder reads directly) is detailed in `docs/embedding-compression.md`.

### Search Tools

`search_context` (exact keyword), `semantic_search_context` (vector similarity), `fts_search_context` (linguistic + snippets), `hybrid_search_context` (RRF fusion of FTS + semantic).

**FTS modes**: `match` (default, stemming), `prefix` (autocomplete), `phrase` (exact order), `boolean` (AND/OR/NOT). SQLite: FTS5 with BM25, Porter stemmer. PostgreSQL: tsvector/tsquery with ts_rank, 29 languages.

**Hybrid search**: RRF `score(d) = Σ(1 / (k + rank_i(d)))`. Parallel execution, graceful degradation. Adaptive FTS: queries with ≥ `HYBRID_FTS_OR_THRESHOLD` (default 4) significant terms switch AND→OR for better recall.

**Response**: `results` (array), `count` (int), `stats` (only when `explain_query=True`).

### Migration System

Auto-applied idempotent migrations in `app/migrations/`: semantic search, FTS, chunking (1:N embeddings), metadata indexing, summary column. PostgreSQL DDL uses `POSTGRESQL_MIGRATION_TIMEOUT_S` (300s). Changing `FTS_LANGUAGE` requires an FTS table rebuild.

**Embedding storage is provisioned from `ENABLE_EMBEDDING_GENERATION`, decoupled from the search-tool toggles.** The vec tables and chunk columns (`app/migrations/semantic.py`, `app/migrations/chunking.py`) and the sqlite-vec / pgvector setup (`app/backends/sqlite_backend.py` `_load_sqlite_vec_extension`, `app/backends/postgresql_backend.py` pgvector pre-create) gate on `settings.embedding.generation_enabled` (default true), NOT on `settings.semantic_search.enabled`. So embedding storage exists whenever embeddings are generated, regardless of whether the `semantic_search_context` tool is exposed — turning semantic search on later never requires re-embedding, and the fp32-latent edge (generation on, semantic tool off, compression off) is closed. The FTS migration (`app/migrations/fts.py`) still gates on `settings.fts.enabled`. The migrations' `force` param (CLI bypass) is unchanged.

### Testing Strategy

**Philosophy**: Unit and repository tests use SQLite temp databases; real-server integration tests run the SAME assertions against BOTH backends through the shared parametrized harness `tests/integration/_harness.py` (`MCPServerIntegrationTest(backend='sqlite'|'postgresql')`) — driven by `tests/integration/sqlite/test_real_server.py` (SQLite) and `tests/integration/postgresql/test_real_server.py` (PostgreSQL on docker-compose pgvector, isolated DB). When adding a new tool or a backend-portable behavior, add the assertion as a harness method so it runs on both backends; reserve dedicated per-backend tests for backend-specific paths (e.g. HTTP transport/auth and `/health` under `sqlite/`, PostgreSQL metadata-index creation under `postgresql/`).

**Key Files**: `conftest.py` (fixtures, markers), `helpers.py` (shared utilities; uses `get_settings()`), `run_server.py` (subprocess server wrapper for integration tests).

**Key Fixtures** (`conftest.py`): `test_db` (direct SQLite), `mock_server_dependencies` (mocked settings), `initialized_server` (full integration), `async_db_initialized` (async backend), `async_db_with_embeddings` (semantic search).

**Skip Markers**: `@requires_ollama`, `@requires_sqlite_vec`, `@requires_numpy`, `@requires_semantic_search`, `@requires_docker_postgres` (PostgreSQL integration tests; skipped when Docker is unavailable). `prevent_default_db_pollution` (autouse) prevents accidental production DB access.

### Test Directory Structure

Tests mirror `app/`: `tests/<name>/` → `app/<name>/` (package) or `app/<name>.py` (module).

**Non-trivial mappings**:
- `tests/core/` → `app/*.py` small utility root modules (models, errors, fusion, instructions, etc.)
- `tests/server/` → `app/server.py` (dedicated directory; large/complex root modules get their own)
- `tests/settings/` → `app/settings.py` (same reason)
- `tests/integration/_harness.py` → shared backend-parametrized real-server harness (`MCPServerIntegrationTest`); imported by the per-backend entry points and intentionally NOT named `test_*` so pytest does not collect it directly
- `tests/integration/sqlite/` and `tests/integration/postgresql/` → per-backend real-server entry points and backend-specific tests (no app mirror); PostgreSQL gated on `@requires_docker_postgres`

**Shared infrastructure** stays at `tests/` root: `conftest.py`, `helpers.py`, `run_server.py`, `__init__.py`.

**Placement rule**: follow the PRIMARY source module under test; use import analysis as arbiter when ambiguous.

### Key Implementation Details

1. **Python 3.12+ Type Hints**: `str | None` syntax, `StrEnum`, TypedDicts in `app/types.py`. **NEVER** use `from __future__ import annotations` in server.py (breaks FastMCP).

2. **FastMCP Tool Signatures**: `Literal["user", "agent"]` for source, `Annotated[type, Field(...)]` for docs, `ctx: Context | None = None` last (hidden from clients). Returns must be serializable dicts/lists. Register via `register_tool()` in lifespan(), not `@mcp.tool()`.

3. **Async Operations**: SQLite ops are sync callables wrapped via `execute_write`/`execute_read`; PostgreSQL ops are native async. Repositories detect the backend via `self.backend.backend_type`.

4. **Design Patterns**:
   - **Protocol** (`@runtime_checkable`): `StorageBackend`, `TransactionContext`, `EmbeddingProvider`, `SummaryProvider`, `RerankingProvider`.
   - **Repository**: all SQL in `app/repositories/`, never in server.py or tools.
   - **Factory**: `create_backend()`, `create_embedding_provider()`, `create_summary_provider()`, `create_reranking_provider()` — dynamic imports via `PROVIDER_MODULES` dicts.
   - **DI**: `RepositoryContainer` injects all repositories.

5. **Error Classification** (`app/errors.py`): `ConfigurationError` (exit 78, supervisor never retries), `DependencyError` (exit 69, may retry with backoff). `classify_provider_error()` classifies embedding/summary provider failures. BSD sysexits.h codes for Docker/K8s restart policies.

6. **Server Instructions**: optional `instructions` field in MCP `InitializeResult`. Via `MCP_SERVER_INSTRUCTIONS` (overrides `DEFAULT_INSTRUCTIONS` from `app/instructions.py`); empty string disables. Includes a `## Skill Integration` section pointing agents to context-server Skills.

7. **UUIDv7 ID Generation** (`app/ids.py`): single source of truth for context-entry IDs. Public: `generate_id()`, `generate_id_with_timestamp(created_at)`, `normalize_id(value)`, `is_id_prefix(value)`, `resolve_prefix(prefix, repo)`.

   - **`uuid_utils.uuid7` parameter contract**: `timestamp` is UNIX **SECONDS** (integer), `nanos` the nanosecond fraction within that second. Migration CLI uses `seconds = int(created_at.timestamp())`, `nanos = created_at.microsecond * 1_000`. Passing milliseconds in `timestamp` shifts the embedded timestamp ~1000x into the future ([uuid-utils#73](https://github.com/aminalaee/uuid-utils/issues/73)).
   - **Lex-string ordering**: UUIDv7 hex sorts chronologically at MILLISECOND granularity (first 48 bits = `unix_ts_ms`; sub-ms order is the random tail), so the `id > ?` interleaving check stays monotonic for timestamps >= 1 ms apart.
   - **Lowercase invariant**: `normalize_id` always emits lowercase hex. Under SQLite TEXT BINARY collation uppercase `A-F` sorts before lowercase `a-f`, so mixed-case IDs would corrupt `id > ?` ordering. Tools accept either case; storage is canonical lowercase.
   - **Public display format**: 32-char hyphen-free lowercase hex (e.g. `0190abcdef1234567890abcdef123456`); tools also accept the 36-char hyphenated form via `normalize_id`. Prefix lookup requires 8 to 31 hex characters.
   - **asyncpg `pgproto.UUID` quirk**: asyncpg returns UUID columns as its own `asyncpg.pgproto.pgproto.UUID`, NOT `uuid.UUID`. Detect with `isinstance(x, uuid.UUID)` (asyncpg subclasses it) — MUST NOT use `type(x) is uuid.UUID` (fails at runtime).
   - **Future stdlib `uuid.uuid7()` (Python 3.14)**: parameterless, so unusable by the migration CLI (which needs deterministic generation from a specific `created_at`); `uuid_utils` stays the canonical migration-CLI generator.

## Package and Release

uv + Hatchling. Entry points: `mcp-context-server`, `mcp-context` (→ `app.server:main`), `mcp-context-server-migrate` (migration CLI, → `app.cli.migrate:main`). Python 3.12+. Optional extras: `embeddings-ollama`, `embeddings-openai`, `embeddings-azure`, `embeddings-huggingface`, `embeddings-voyage`, `summary-ollama`, `summary-openai`, `summary-anthropic`, `reranking`, `langsmith`.

The `mcp-context-server-migrate` console script migrates integer-keyed context databases to the UUIDv7 schema. Run manually on a backup of the source DB; accepts `--source-url`/`--target-url`/`--dry-run`/`--report PATH`; supports SQLite→SQLite, PostgreSQL→PostgreSQL, and cross-backend. The PostgreSQL target **schema is auto-initialized** when absent (parity with the SQLite path, via `initialize_target_postgresql`): the CLI builds the fp32 layout (`init_database` + `apply_semantic_search_migration(force=True, embedding_dim=<source dim>)` + jsonb/search_path + `apply_chunking_migration(force=True)`) and NEVER runs the compression migration, so compression stays a separate `--compress` step; you still create the empty PG database yourself. All PG connections honor `POSTGRESQL_SCHEMA` (search_path) and `POSTGRESQL_STATEMENT_CACHE_SIZE` (set 0 for transaction-mode poolers / Supabase 6543) via the shared `build_asyncpg_connect_kwargs()` in `app/backends/postgresql_backend.py`; SSL rides on the DSN. Cross-backend migration copies tags + image attachments (only vector embeddings are dropped) and rebuilds the SQLite FTS index on PostgreSQL→SQLite. Full guide: [`docs/migration-v2-to-v3.md`](docs/migration-v2-to-v3.md).

[Release Please](https://github.com/googleapis/release-please) for automated releases via [Conventional Commits](https://www.conventionalcommits.org/). On `release:published`: PyPI package, MCP Registry (`server.json`), GHCR Docker images (amd64/arm64): default Ollama variant and `ollama-openai` variant.

### SECURITY.md Maintenance

When a commit triggers a major version bump (Conventional Commit `!` suffix or `BREAKING CHANGE:` in body/footer), update the Supported Versions table in `SECURITY.md` in the same PR: add the new major version as supported, mark the previous major version as unsupported.

## Documentation Maintenance

When changing core functionality, update the corresponding doc before committing:

- **`README.md`** — when adding/changing/removing user-facing features; its Key Features list (the first thing users read) must reflect current capabilities.
- Any other related documents in `docs/`.

## CI and Docker Lock File Discipline

`uv.lock` is a UNIVERSAL resolution of ALL dependencies across ALL optional groups/extras; `uv sync` with selective flags installs only the relevant subset.

**Three defense layers**: (1) pre-commit `uv-lock` hook (local); (2) `uv lock --check` (CI early step); (3) `uv sync --locked --all-extras --all-groups` (CI install). Every CI workflow (`test.yml`, `lint.yml`) MUST run both CI steps:

```yaml
# CORRECT
- run: uv lock --check
- run: uv sync --locked --all-extras --all-groups

# WRONG: Explicit listing misses extras when new ones are added
- run: uv sync --locked --dev --extra embeddings-ollama --extra reranking
```

**Exception**: `publish.yml` and `release-please.yml` run `uv lock` (without `--check`) — Release Please bumps the version, requiring lock regeneration.

**Docker** MUST use `--locked --no-dev --extra <variant>` for SELECTIVE installation:

```dockerfile
uv sync --locked --no-install-project --extra ${EMBEDDING_EXTRA} --extra ${SUMMARY_EXTRA} --extra reranking --no-dev
uv sync --locked --extra ${EMBEDDING_EXTRA} --extra ${SUMMARY_EXTRA} --extra reranking --no-dev
```

Build args: `EMBEDDING_EXTRA` (default `embeddings-ollama`), `SUMMARY_EXTRA` (default `summary-ollama`). Docker intentionally does NOT use `--all-extras`.

## MCP Registry and server.json Maintenance

`server.json` enables MCP client discovery. Every `Field(alias=...)` in `app/settings.py` MUST have a matching entry in `server.json` `environmentVariables` — enforced by `test_server_json_environment_variables_match_settings`. Release Please auto-updates version.

When adding or modifying environment variables in `app/settings.py`, update **both** `server.json` **and** `docs/environment-variables.md`.

## Environment Variables

Configuration via `.env` file or environment. **Canonical source**: `app/settings.py` — all env vars with defaults, descriptions, and validation. Full reference: `docs/environment-variables.md`.

**Core**: `STORAGE_BACKEND` (sqlite*/postgresql), `LOG_LEVEL` (ERROR*), `DB_PATH`, `MAX_IMAGE_SIZE_MB` (10*), `MAX_TOTAL_SIZE_MB` (100*), `DISABLED_TOOLS`

**Transport**: `MCP_TRANSPORT` (stdio*/http/streamable-http/sse), `FASTMCP_HOST` (0.0.0.0*), `FASTMCP_PORT` (8000*), `FASTMCP_STATELESS_HTTP` (true*)

**Auth**: `MCP_AUTH_PROVIDER` (none*/simple_token), `MCP_AUTH_TOKEN`, `MCP_AUTH_CLIENT_ID` (mcp-client*)

**FastMCP Logging** (NOT in `app/settings.py`): `FASTMCP_ENABLE_RICH_LOGGING` (true*; set `false` in Docker/cloud) — see [FASTMCP_* Governance](#fastmcp_-env-var-governance).

**Feature Toggles**: `ENABLE_EMBEDDING_GENERATION` (true*), `ENABLE_SEMANTIC_SEARCH` (auto*), `ENABLE_FTS` (auto*), `ENABLE_HYBRID_SEARCH` (auto*), `ENABLE_CHUNKING` (true*), `ENABLE_RERANKING` (true*), `ENABLE_SUMMARY_GENERATION` (true*). The three search toggles are tri-state `Literal['auto','true','false']` (base class `FeatureToggleSettings`; derived read-only `.enabled` property = `mode != 'false'`, so `auto` and `true` are both enabled): `auto` (default) registers the tool when prerequisites are present and skips quietly otherwise, `true` forces it on (warns if unavailable), `false` forces it off. Boolean spellings (true/false/1/0/yes/no/on/off) are coerced via `_normalize_feature_toggle()`. Default `auto` means search works out of the box: semantic registers when an embedding provider is available, FTS always (no extra deps), hybrid when at least one of FTS/semantic is available.

**Embedding**: `EMBEDDING_PROVIDER` (ollama*/openai/azure/huggingface/voyage), `EMBEDDING_MODEL` (qwen3-embedding:0.6b*), `EMBEDDING_DIM` (1024*), `EMBEDDING_TIMEOUT_S` (240*), `EMBEDDING_MAX_CONCURRENT` (3*)

**Summary**: `SUMMARY_PROVIDER` (ollama*/openai/anthropic), `SUMMARY_MODEL` (qwen3:0.6b*), `SUMMARY_MAX_TOKENS` (2000*), `SUMMARY_MIN_CONTENT_LENGTH` (500*; shorter text skips summary; 0 = always), `SUMMARY_PROMPT`

**Retrieval**: `GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY` (false*; tri-state on `get_context_by_ids`: false omits the `summary` key (`get('summary')`→`None`); true returns a stored non-empty summary verbatim, else `''` (NULL/empty DB summary). `text_content` is always full/untruncated.)

**Compression** (default ON in v3.0.0): `ENABLE_EMBEDDING_COMPRESSION` (true*; false to opt out, keep fp32), `COMPRESSION_PROVIDER` (turboquant*), `COMPRESSION_BITS` (4*; range 2-4), `COMPRESSION_VARIANT` (ip*/mse), `COMPRESSION_SEED` (0*; load-bearing, immutable after first compressed row), `COMPRESSION_MAX_CONCURRENT` (min(cpu_count, 4)*; range 1-32). Full reference: `docs/embedding-compression.md`.

**Other vars** (provider-specific, PostgreSQL, reranking, chunking, hybrid, FTS, search, metadata indexing): see `app/settings.py` for the complete list with defaults and descriptions.

*\* = default value*

## Storage Backend Configuration

### SQLite (Default)
Zero-config local storage; see SQLiteBackend (Architecture) for features.

### PostgreSQL
```bash
docker run --name pgvector18 -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mcp_context -p 5432:5432 -d pgvector/pgvector:pg18-trixie
export STORAGE_BACKEND=postgresql
uv run mcp-context-server  # Auto-initializes schema, enables pgvector
```

**Optional: PostgreSQL 18+ `DEFAULT uuidv7()`.** On PG18+, you MAY set `id UUID PRIMARY KEY DEFAULT uuidv7()` (`ALTER TABLE context_entries ALTER COLUMN id SET DEFAULT uuidv7();`) for server-side generation. Pure operator-side optimization, NOT required: the app generates UUIDs Python-side via `app/ids.py` regardless of column default (the migration CLI anchors to each row's `created_at`). PG17 and earlier lack `uuidv7()`.

### Supabase
`STORAGE_BACKEND=postgresql` + `POSTGRESQL_CONNECTION_STRING`. Session Pooler for IPv4. "getaddrinfo failed" = switch from Direct to Session Pooler.

### Metadata Field Indexing by Backend

SQLite: B-tree via `json_extract` for scalars only (array/object queries need a full table scan). PostgreSQL: B-tree for scalars, GIN for arrays/objects.

## Docker Deployment

Multi-stage Dockerfile (uv, non-root UID 10001, `/health` endpoint). Configs in `deploy/docker/`: SQLite, PostgreSQL, Supabase. Ollama sidecar uses the stock `ollama/ollama:latest` image (`:rocm` for AMD GPUs) with an inline `entrypoint:` block per compose file pulling embedding+summary models on first startup.

### Docker Compose File Naming Convention

**Naming formula:** `docker-compose.{storage}.{providers}[.local].yml`

- `{storage}`: `sqlite`, `postgresql`, or `postgresql-external`.
- `{providers}`: embedding+summary combo — single name when both share a provider (`ollama`, `openai`), hyphenated `<embedding>-<summary>` when they differ (`ollama-openai`).
- `.local` (optional): present ONLY for combinations with a published GHCR image, marking the file that builds that config locally instead of pulling from the registry.

**Image sources**: `*.ollama.yml`/`*.ollama-openai.yml` pull the prebuilt GHCR image (`ghcr.io/alex-feel/mcp-context-server:latest` / `:latest-ollama-openai`, `pull_policy: always`); `*.local.yml` and provider-specific `*.openai.yml` build locally (`image: mcp-context-server`, `pull_policy: build`, with `EMBEDDING_EXTRA`/`SUMMARY_EXTRA` build args, e.g. `SUMMARY_EXTRA=summary-openai`).

**Extensibility rule:** if a GHCR image is published for a combo, create both `*.{providers}.yml` (pull) and `*.{providers}.local.yml` (local build); otherwise create only `*.{providers}.yml` (local build, no `.local` variant).

### Docker-Compose Environment Variable Policy

**CRITICAL:** Compose files MUST contain ONLY variables REQUIRED for the deployment to function; all else uses `app/settings.py` defaults. This prevents drift between code defaults and hardcoded compose values.

**Configurable** via `${VAR:-default}` interpolation (default MUST match the previously hardcoded value): `LOG_LEVEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_PROVIDER`, `SUMMARY_MODEL`, `SUMMARY_PROVIDER`; PostgreSQL variants add `POSTGRESQL_USER`, `POSTGRESQL_PASSWORD`, `POSTGRESQL_DATABASE`. Ollama sidecar model names use the same interpolation.

**Required (hardcoded, NOT configurable)**: transport (`MCP_TRANSPORT`, `FASTMCP_HOST`, `FASTMCP_PORT`), `FASTMCP_ENABLE_RICH_LOGGING=false`, storage backend selection, `OLLAMA_HOST` (sidecar bind + client URL), `OLLAMA_KEEP_ALIVE=-1`, feature toggles (`ENABLE_*`), container paths (`DB_PATH`, `RERANKING_CACHE_DIR`), Docker networking (`POSTGRESQL_HOST`, `POSTGRESQL_PORT`).

**Do NOT add**: tuning parameters or feature-specific settings that have correct defaults.

## Kubernetes Deployment

Helm chart in `deploy/helm/mcp-context-server/`. Profiles: `values-sqlite.yaml`, `values-postgresql.yaml`. Optional Ollama sidecar, ingress with TLS.

## Windows Development Notes

Use `Path` objects (not string concat). Env vars: `set VAR=value &&` (cmd) or `$env:VAR="value";` (PowerShell). DB path: `%USERPROFILE%\.mcp\context_storage.db`. Docker Desktop for PostgreSQL.

## Debugging and Troubleshooting

```bash
set LOG_LEVEL=DEBUG && uv run mcp-context-server  # Debug logs (Windows)
uv run python -c "from app.startup import init_database; import asyncio; asyncio.run(init_database())"  # Test DB
```

**Common Issues**: Import errors → `uv sync`. Type errors → `uv run mypy app`. Semantic search unavailable → the tool auto-registers once an embedding provider is present, so `uv sync --extra embeddings-ollama` (provider deps); only set `ENABLE_SEMANTIC_SEARCH=false` if it was force-disabled. FTS unavailable → FTS is `auto` by default (no extra deps); check `ENABLE_FTS` is not set to `false`. Summary generation unavailable → `ENABLE_SUMMARY_GENERATION=true` + `uv sync --extra summary-ollama` + `ollama pull qwen3:0.6b`.

## Code Quality Standards

Ruff (127 chars, single quotes), mypy/pyright strict for `app/`.

## Documentation Style

Markdown files under `docs/` and the top-level `README.md` / `CLAUDE.md` use **one paragraph per physical line**. Do NOT hard-wrap prose at 70 / 80 / 100 columns. Tables, bullet/numbered list items, code blocks, headings, and YAML/JSON blocks retain their natural line breaks.

Rationale: one paragraph per line keeps `git diff` minimal (a wording change touches one line) and `grep -n` pointing at the paragraph start, instead of forcing reflow of neighboring lines.

Existing violations are normalized incrementally: when you touch a paragraph for a content reason, normalize it to one line in the same edit; do not perform standalone reflow sweeps.

## GitHub Actions Security Policy

All GitHub Actions workflows MUST follow these rules:

1. **Never use mutable branch references** (`@main`, `@master`, `@develop`, `@release/vN`) for third-party actions. Mutable refs can be compromised (e.g., March 2026 `aquasecurity/trivy-action` incident — exfiltrated CI secrets).

2. **Version tag pinning is the project standard.** Pin third-party actions to immutable tags (e.g., `@v5`, `@v1.13.0`). SHA pinning not required — tags are immutable enough and Dependabot-compatible.

3. **Verify action runtimes before updating.** Before bumping an action version, check its runtime to avoid deprecated Node.js versions:
   ```bash
   # Check action runtime (note: some actions use action.yaml instead of action.yml)
   gh api repos/OWNER/REPO/contents/action.yml --jq '.content' | base64 -d | grep -E 'using:.*node'
   ```

## Critical Implementation Warnings

### Environment Variables — Centralized Configuration

**Never use `os.environ`/`os.getenv()` directly** — always `get_settings()` from `app/settings.py`. Use `Field(alias='ENV_VAR_NAME')`.

```python
# WRONG: os.getenv('DB_PATH')
# CORRECT: get_settings().storage.db_path
```

### Context Identifier Normalization

`app/ids.py.normalize_id` is the canonical boundary normalizer for context-entry IDs. Any code accepting a `context_id` from outside the storage layer (tool parameters, CLI flags, JSON payloads) MUST route it through `normalize_id` to fold case, strip whitespace, and validate format. The lowercase-hex invariant is load-bearing for SQLite TEXT BINARY ordering — see Key Implementation Details #7.

### Settings Singleton Caching (`@lru_cache`)

`get_settings()` is `@lru_cache`-decorated — a process-lifetime singleton; once called, env var changes are ignored.

**In tests**, call `get_settings.cache_clear()` after changing env vars (e.g. after `monkeypatch.setenv(...)`) so the next `get_settings()` rebuilds `AppSettings`. For modules that change settings across tests, use an autouse fixture calling `get_settings.cache_clear()` (pattern: `tests/reranking/conftest.py`).

**Anti-pattern: Premature `get_settings()` in subprocess scripts.** `tests/run_server.py` configures environment via `os.environ` (intentional: an env configurator, not a consumer). Call `get_settings.cache_clear()` after all `os.environ` changes before launching, or utilities like `is_ollama_model_available()` cache stale defaults.

### Per-test environment overrides for module-level `settings` bindings

Tool modules cache `settings = get_settings()` at import time (e.g., `app/tools/context.py:60`). To flip an env-driven setting per test: `monkeypatch.setenv(...)`, then `get_settings.cache_clear()`, then `monkeypatch.setattr(<tool_module>, 'settings', get_settings())`. The final `setattr` refreshes the module-level binding to the new singleton; without it the module keeps the stale reference. (Refactoring 15+ modules to call `get_settings()` per function would be YAGNI.)

### Settings Class Architecture

**AppSettings must NEVER contain settings fields directly** — it only composes nested settings classes. When adding settings: (1) add to an **existing** class if it EXACTLY matches the domain; (2) else create a **new** class — even for a single setting.

```python
# WRONG: my_new_setting: str = Field(...) directly on AppSettings
# CORRECT: dedicated class, composed in via default_factory
class MyFeatureSettings(CommonSettings):
    enabled: bool = Field(default=False, alias='ENABLE_MY_FEATURE')

class AppSettings(CommonSettings):
    my_feature: MyFeatureSettings = Field(default_factory=MyFeatureSettings)
```

Existing settings classes are enumerated in `app/settings.py` (one per domain, e.g. `EmbeddingSettings`, `SummarySettings`, `RerankingSettings`, `FtsSettings`, `HybridSearchSettings`, `ChunkingSettings`, `RetrievalSettings`, `LangSmithSettings`); `StorageSettings` extends `BaseSettings`, the rest extend `CommonSettings`.

### FASTMCP_* Env Var Governance

`FASTMCP_*` env vars belong in `app/settings.py` (and `server.json`) ONLY when the project takes **programmatic action** with them (passed to `mcp.run()`, used in logic). Import-time-only vars should NOT be in `settings.py`.

- **In `settings.py`** (passed to `mcp.run()`): `FASTMCP_HOST` (`host=`), `FASTMCP_PORT` (`port=`), `FASTMCP_STATELESS_HTTP` (`stateless_http=`).
- **NOT in `settings.py`**: `FASTMCP_TRANSPORT` (project uses `MCP_TRANSPORT` + explicit `transport=` arg), `FASTMCP_ENABLE_RICH_LOGGING` (consumed at FastMCP import time; no `mcp.run()` parameter).

### Adding New MCP Tools

New tools live in `app/tools/<domain>.py` as `async` functions: `Annotated[..., Field(...)]` params, `ctx: Context | None = None` last; call `repos = await ensure_repositories()` and return a serializable dict (`{'success': True, 'context_id': ...}`).

**Steps**: 1) `app/tools/<domain>.py`; 2) add to `TOOL_ANNOTATIONS` in `app/tools/__init__.py`; 3) export from `__init__.py`; 4) register in `app/server.py` lifespan(); 5) add a TypedDict to `app/types.py`; 6) add tests + a real-server harness method in `tests/integration/_harness.py` (runs on BOTH backends via the SQLite and PostgreSQL entry points; reserve dedicated per-backend files only for backend-specific paths); 7) update `server.json` if new env vars; 8) backend-specific descriptions → add a generator to `app/tools/descriptions.py`; 9) store/update ops → reuse `app/tools/_shared.py` shared functions for parity.

**Annotation categories**: READ_ONLY (readOnlyHint=True), ADDITIVE (destructiveHint=False), UPDATE (destructiveHint=True, idempotentHint=False), DELETE (destructiveHint=True, idempotentHint=True)

### Adding New Providers (Embeddings/Reranking/Summary)

All three layers share a pattern: 1) create a provider class in `app/<layer>/providers/` implementing the Protocol; 2) add to `PROVIDER_MODULES`/`PROVIDER_CLASSES` in `factory.py`; 3) add install instructions to `PROVIDER_INSTALL_INSTRUCTIONS`; 4) add an optional dependency group in `pyproject.toml`.

### Generation-First Transactional Integrity

**CRITICAL**: when generation is enabled and fails, NO data is saved — the transaction rolls back. Flow: generate embeddings/summaries OUTSIDE the transaction via `asyncio.gather(*tasks, return_exceptions=True)`, then all DB ops in one atomic `begin_transaction()` (all repo writes accept optional `txn: TransactionContext`). `generate_embeddings_with_timeout`/`generate_summary_with_timeout` in `app/tools/_shared.py` are the single sources of truth for timeout/semaphore, used by `store_context`, `update_context`, `store_context_batch`, `update_context_batch`. Each `gather` result is inspected independently; failed generation raises (or is collected in non-atomic batch mode) without cancelling the other.

**NEVER propose "graceful skip" of generation when generation is enabled.** If `ENABLE_EMBEDDING_GENERATION=true` and embeddings fail, or `ENABLE_SUMMARY_GENERATION=true` and summary fails, the entry MUST NOT be saved — no "store without embeddings"/"store without summary" fallback exists. This is non-negotiable. Skipping requires the user to explicitly set `ENABLE_EMBEDDING_GENERATION=false` or `ENABLE_SUMMARY_GENERATION=false`; only the user can make this decision.

### Deduplication Behavior (store_context)

Deduplication is **retry protection**: when an MCP client retransmits the same message (network glitch, timeout retry), the server updates the existing entry instead of duplicating. A **new conversational turn** (a user deliberately re-sending identical text after an agent responded) instead inserts a new entry, preserving chronological order.

**Interleaving check**: before deduplicating, the server checks for opposite-source entries (agent for user source, user for agent source) after the candidate; if any exist it's a new turn. Uses `id > candidate_id` (immune to clock skew) with the existing `idx_thread_source` index.

When deduplication proceeds (same `thread_id + source + text_content` as the latest entry, no interleaving): **Metadata** via `COALESCE(new, existing)` (`None` keeps existing, explicit value replaces); **Tags/Images** REPLACED (not accumulated) when provided, preserved when `None`; **content_type/updated_at** auto-updated; **Embeddings** skipped if present, else generated; **Summary** regeneration skipped if present (`COALESCE(NULL, existing_summary)` preserves it); **Pre-check** read-only check before generation skips LLM calls for duplicates (in both `store_context` and `store_context_batch`).

`store_context_batch` applies the same dedup logic (`store_with_deduplication` per entry), pre-check, and interleaving check.

### Update Context and Batch Operations

**Update**: Partial updates (only provided fields). Immutable: `id`, `thread_id`, `source`, `created_at`. Auto-managed: `content_type`, `updated_at`. Tags/images: replacement (not merge). Transaction-wrapped.

**Batch**: `store_context_batch`, `update_context_batch`, `delete_context_batch` (up to 100 entries). `atomic=true` (default): all-or-nothing. `atomic=false`: independent processing with per-entry results. Batch and non-batch tools share per-entry logic (image validation, transaction execution, response building) via `app/tools/_shared.py` for parity.

### Compression Seed-Locked Invariant

`COMPRESSION_SEED` defaults to `0` in v3.0.0. On first start the validator (`app/startup/compression_validator.py`) writes a singleton `compression_metadata` row from the runtime config, then treats the DB as source of truth. Later starts where runtime `CompressionSettings` disagrees with the persisted `(provider, bits, variant, seed, dim)` row raise `ConfigurationError` (exit 78); the supervisor will NOT auto-restart, surfacing the misconfiguration loudly instead of silently corrupting search results.

In multi-pod Kubernetes deployments all pods MUST inherit the SAME `COMPRESSION_SEED` (ConfigMap-bound, so every pod resolves the identical value). Changing the seed AFTER any compressed data is stored corrupts every decode/search; the only recovery is restoring from backup. The Helm chart ships an active compression block (`enabled: true`, `seed: 0`) in `values-sqlite.yaml`/`values-postgresql.yaml` under `deploy/helm/mcp-context-server/` (PG profile documents the multi-pod ConfigMap discipline inline); `compression.enabled: false` opts out. See `docs/embedding-compression.md`.

After validation, the lifespan logs the active compression config (singleton `compression_metadata` row + `max_concurrent`); when disabled it reports `Embedding compression disabled (ENABLE_EMBEDDING_COMPRESSION=false)`. `get_statistics` exposes the same values under a `compression` sub-block, plus `embeddings_size_mb` and boolean `embeddings_size_estimated` after total `database_size_mb` — gated on embedding generation OR compression (NOT `ENABLE_SEMANTIC_SEARCH`), `0.0` on failure. Both size keys are per-backend, NOT byte-comparable (formulas in `docs/embedding-compression.md#observability`; on SQLite `embeddings_size_estimated=true` marks a deterministic fp32 estimate). PostgreSQL numeric aggregates are coerced to floats via `_to_float` in `app/repositories/statistics_repository.py` (asyncpg maps `AVG()` to `Decimal`, failing the MCP float output schema).

### Known Upstream Bugs and Temporary Patches

**MCP SDK Session Crash on Client Disconnect** (`app/patches/session_crash.py`):

Monkey-patch for `BaseSession._send_response()`/`send_notification()` not handling `ClosedResourceError`/`BrokenResourceError` on client disconnect. Applied in `app/server.py` lifespan (step 0).

- **Upstream tracking**: [MCP SDK #2064](https://github.com/modelcontextprotocol/python-sdk/issues/2064), PRs [#2072](https://github.com/modelcontextprotocol/python-sdk/pull/2072), [#2184](https://github.com/modelcontextprotocol/python-sdk/pull/2184)
- **Removal** (when upstream MCP SDK fixes this): bump `mcp`, delete `app/patches/` + its patch import/call in `app/server.py`, delete `tests/patches/test_session_crash_patch.py`, remove `test_session_crash_patch_applied` from `tests/integration/_harness.py`, and remove this section.

**Client JSON String Serialization** (`app/middleware/json_string_deserializer.py`):

Schema-aware FastMCP middleware fixing MCP clients (including Claude Code) intermittently serializing list/dict parameters as JSON strings. `Middleware` base class with `on_call_tool` override; `build_schema_map()` inspects each tool's JSON Schema at startup for `array`/`object` params (incl. `Optional` via `anyOf`/`$ref`) — only those are deserialization candidates, strings are never touched. Handles double-encoding. Registered in `app/server.py` lifespan (step 25) via `mcp.add_middleware()`, after all `register_tool()` calls.

- **Upstream tracking**: [Claude Code #22394](https://github.com/anthropics/claude-code/issues/22394) (closed NOT_PLANNED), [Claude Code #26094](https://github.com/anthropics/claude-code/issues/26094), [FastMCP #932](https://github.com/jlowin/fastmcp/issues/932), Claude Code #5504, #4192, #3084
- **Removal** (when upstream clients fix serialization): delete `app/middleware/` + its import/registration block (step 25) in `app/server.py` lifespan(), delete `tests/middleware/test_middleware_json_deserializer.py`, remove middleware integration tests from `tests/integration/_harness.py`, and remove this section.
