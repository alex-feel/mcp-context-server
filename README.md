# MCP Context Server

[![PyPI](https://img.shields.io/pypi/v/mcp-context-server.svg)](https://pypi.org/project/mcp-context-server/) [![MCP Registry](https://img.shields.io/badge/MCP_Registry-listed-blue?logo=anthropic)](https://github.com/modelcontextprotocol/registry) [![GitHub License](https://img.shields.io/github/license/alex-feel/mcp-context-server)](https://github.com/alex-feel/mcp-context-server/blob/main/LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/alex-feel/mcp-context-server)

A high-performance Model Context Protocol (MCP) server providing persistent multimodal context storage for LLM agents. Built with FastMCP, this server enables seamless context sharing across multiple agents working on the same task through thread-based scoping.


## Key Features

- **Multimodal Context Storage**: Store and retrieve both text and images
- **Thread-Based Scoping**: Agents working on the same task share context through thread IDs
- **Flexible Metadata Filtering**: Store custom structured data with any JSON-serializable fields and filter using 16 powerful operators
- **Date Range Filtering**: Filter context entries by creation timestamp using ISO 8601 format
- **Tag-Based Organization**: Efficient context retrieval with normalized, indexed tags
- **Full-Text Search**: Optional linguistic search with stemming, ranking, and boolean queries (FTS5/tsvector)
- **Semantic Search**: Optional vector similarity search for meaning-based retrieval
- **Hybrid Search**: Optional combined FTS + semantic search using Reciprocal Rank Fusion (RRF)
- **Multiple Database Backends**: Choose between SQLite (default, zero-config) or PostgreSQL (high-concurrency, production-grade)
- **High Performance**: WAL mode (SQLite) / MVCC (PostgreSQL), strategic indexing, and async operations
- **MCP Standard Compliance**: Works with Claude Code, LangGraph, and any MCP-compatible client
- **Production Ready**: Comprehensive test coverage, type safety, and robust error handling

## Prerequisites

- `uv` package manager ([install instructions](https://docs.astral.sh/uv/getting-started/installation/))
- An MCP-compatible client (Claude Code, LangGraph, or any MCP client)

## Adding the Server to Claude Code

There are two ways to add the MCP Context Server to Claude Code:

### Method 1: Using CLI Command

```bash
# From PyPI (recommended) - includes reranking enabled by default
claude mcp add context-server -- uvx --python 3.12 --with mcp-context-server[reranking] mcp-context-server

# Or from GitHub (latest development version)
claude mcp add context-server -- uvx --python 3.12 --from git+https://github.com/alex-feel/mcp-context-server --with mcp-context-server[reranking] mcp-context-server

# Or with semantic search using Ollama (for setup instructions, see docs/semantic-search.md)
claude mcp add context-server -- uvx --python 3.12 --with "mcp-context-server[embeddings-ollama,reranking]" mcp-context-server

# Or from GitHub (latest development version) with semantic search
claude mcp add context-server -- uvx --python 3.12 --from git+https://github.com/alex-feel/mcp-context-server --with "mcp-context-server[embeddings-ollama,reranking]" mcp-context-server

# Available embedding providers: embeddings-ollama (default), embeddings-openai, embeddings-azure, embeddings-huggingface, embeddings-voyage
# Note: The `--extra reranking` is necessary to enable reranking.
```

For more details, see: https://docs.claude.com/en/docs/claude-code/mcp#option-1%3A-add-a-local-stdio-server

### Method 2: Direct File Configuration

Add the following to your `.mcp.json` file in your project directory:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[reranking]", "mcp-context-server"],
      "env": {}
    }
  }
}
```

**Note:** The `--extra reranking` is necessary to enable reranking.

For the latest development version from GitHub, use:
```json
"args": ["--python", "3.12", "--from", "git+https://github.com/alex-feel/mcp-context-server", "--with", "mcp-context-server[reranking]", "mcp-context-server"]
```

For configuration file locations and details, see: https://docs.claude.com/en/docs/claude-code/settings#settings-files

### Verifying Installation

```bash
# Start Claude Code
claude

# Check MCP tools are available
/mcp
```

## Environment Configuration

### Environment Variables

You can configure the server using environment variables in your MCP configuration. The server supports environment variable expansion using `${VAR}` or `${VAR:-default}` syntax.

Example configuration with environment variables:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "--with", "mcp-context-server[reranking]", "mcp-context-server"],
      "env": {
        "LOG_LEVEL": "${LOG_LEVEL:-INFO}",
        "DB_PATH": "${DB_PATH:-~/.mcp/context_storage.db}",
        "MAX_IMAGE_SIZE_MB": "${MAX_IMAGE_SIZE_MB:-10}",
        "MAX_TOTAL_SIZE_MB": "${MAX_TOTAL_SIZE_MB:-100}"
      }
    }
  }
}
```

For more details on environment variable expansion, see: https://docs.claude.com/en/docs/claude-code/mcp#environment-variable-expansion-in-mcp-json

### Supported Environment Variables

**Core Settings:**
- **STORAGE_BACKEND**: Database backend - `sqlite` (default) or `postgresql`
- **LOG_LEVEL**: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - defaults to ERROR
- **DB_PATH**: Database file location (SQLite only) - defaults to ~/.mcp/context_storage.db
- **MAX_IMAGE_SIZE_MB**: Maximum size per image in MB - defaults to 10
- **MAX_TOTAL_SIZE_MB**: Maximum total request size in MB - defaults to 100

**Full-Text Search Settings:**
- **ENABLE_FTS**: Enable full-text search functionality (true/false) - defaults to false
- **FTS_LANGUAGE**: Language for stemming and text search - defaults to `english`. PostgreSQL supports 29 languages with full stemming. SQLite uses `english` for Porter stemmer or any other value for unicode61 tokenizer (no stemming).

**Hybrid Search Settings:**
- **ENABLE_HYBRID_SEARCH**: Enable hybrid search combining FTS and semantic search with RRF fusion (true/false) - defaults to false
- **HYBRID_RRF_K**: RRF smoothing constant (1-1000) - defaults to 60. Higher values give more uniform treatment across ranks.

**Chunking Settings** (for improved semantic search on long documents):
- **ENABLE_CHUNKING**: Enable text chunking for embeddings (true/false) - defaults to true
- **CHUNK_SIZE**: Target chunk size in characters - defaults to 1000
- **CHUNK_OVERLAP**: Overlap between chunks in characters - defaults to 100
- **CHUNK_AGGREGATION**: Chunk score aggregation: max (only 'max' supported in current version)

**Reranking Settings** (for improved search precision):
- **ENABLE_RERANKING**: Enable cross-encoder reranking (true/false) - defaults to true
- **RERANKING_PROVIDER**: Reranking provider - defaults to flashrank
- **RERANKING_MODEL**: Reranking model name - defaults to ms-marco-MiniLM-L-12-v2 (~34MB)
- **RERANKING_OVERFETCH**: Multiplier for over-fetching before reranking - defaults to 4

**Semantic Search Settings:**
- **ENABLE_SEMANTIC_SEARCH**: Enable semantic search functionality (true/false) - defaults to false
- **EMBEDDING_PROVIDER**: Embedding provider - `ollama` (default), `openai`, `azure`, `huggingface`, or `voyage`
- **EMBEDDING_MODEL**: Embedding model name - defaults to `qwen3-embedding:0.6b` (provider-specific)
- **EMBEDDING_DIM**: Embedding vector dimensions - defaults to 1024. **Note**: Changing this after initial setup requires database migration (see [Semantic Search Guide](docs/semantic-search.md#changing-embedding-dimensions))

**Provider-Specific Settings** (see [Semantic Search Guide](docs/semantic-search.md) for complete details):
- **OLLAMA_HOST**: Ollama API URL (default: http://localhost:11434)
- **OPENAI_API_KEY**: OpenAI API key (for `openai` provider)
- **AZURE_OPENAI_API_KEY**, **AZURE_OPENAI_ENDPOINT**, **AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME**: Azure OpenAI settings
- **HUGGINGFACEHUB_API_TOKEN**: HuggingFace Hub token (for `huggingface` provider)
- **VOYAGE_API_KEY**: Voyage AI API key (for `voyage` provider)

**LangSmith Tracing** (optional observability - requires `langsmith` extra: `uv sync --extra langsmith`):
- **LANGSMITH_TRACING**: Enable LangSmith tracing (true/false) - defaults to false
- **LANGSMITH_API_KEY**: LangSmith API key
- **LANGSMITH_PROJECT**: Project name for grouping traces - defaults to `mcp-context-server`

**Metadata Indexing Settings:**
- **METADATA_INDEXED_FIELDS**: Comma-separated list of metadata fields to index with optional type hints - defaults to `status,agent_name,task_name,project,report_type,references:object,technologies:array`. Type hints: `string` (default), `integer`, `boolean`, `float`, `array`, `object`. Array/object types use PostgreSQL GIN indexes and are skipped in SQLite.
- **METADATA_INDEX_SYNC_MODE**: How to handle index mismatches at startup - defaults to `additive`. Options: `strict` (fail if mismatch), `auto` (sync - add missing, drop extra), `warn` (log warnings), `additive` (add missing, never drop)

**PostgreSQL Settings** (only when STORAGE_BACKEND=postgresql):
- **POSTGRESQL_HOST**: PostgreSQL server host - defaults to localhost
- **POSTGRESQL_PORT**: PostgreSQL server port - defaults to 5432
- **POSTGRESQL_USER**: PostgreSQL username - defaults to postgres
- **POSTGRESQL_PASSWORD**: PostgreSQL password - defaults to postgres
- **POSTGRESQL_DATABASE**: PostgreSQL database name - defaults to mcp_context

### Advanced Configuration

Additional environment variables are available for advanced server tuning, including:
- Connection pool configuration
- Retry behavior settings
- SQLite performance optimization
- Circuit breaker thresholds
- Operation timeouts

For a complete list of all configuration options, see [app/settings.py](app/settings.py).

## Semantic Search

For detailed instructions on enabling optional semantic search with multiple embedding providers (Ollama, OpenAI, Azure, HuggingFace, Voyage), see the [Semantic Search Guide](docs/semantic-search.md).

## Full-Text Search

For full-text search with linguistic processing, stemming, ranking, and boolean queries, see the [Full-Text Search Guide](docs/full-text-search.md).

## Hybrid Search

For combined FTS + semantic search using Reciprocal Rank Fusion (RRF), see the [Hybrid Search Guide](docs/hybrid-search.md).

## Metadata Filtering

For comprehensive metadata filtering including 16 operators, nested JSON paths, and performance optimization, see the [Metadata Guide](docs/metadata-addition-updating-and-filtering.md).

## Database Backends

The server supports multiple database backends, selectable via the `STORAGE_BACKEND` environment variable. SQLite (default) provides zero-configuration local storage perfect for single-user deployments. PostgreSQL offers high-performance capabilities with 10x+ write throughput for multi-user and high-traffic deployments.

For detailed configuration instructions including PostgreSQL setup with Docker, Supabase integration, connection methods, and troubleshooting, see the [Database Backends Guide](docs/database-backends.md).

## API Reference

The MCP Context Server exposes 13 MCP tools for context management:

**Core Operations:** `store_context`, `search_context`, `get_context_by_ids`, `delete_context`, `update_context`, `list_threads`, `get_statistics`

**Search Tools:** `semantic_search_context`, `fts_search_context`, `hybrid_search_context`

**Batch Operations:** `store_context_batch`, `update_context_batch`, `delete_context_batch`

For complete tool documentation including parameters, return values, filtering options, and examples, see the [API Reference](docs/api-reference.md).

## Docker Deployment

For production deployments with HTTP transport and container orchestration, Docker Compose configurations are available for SQLite, PostgreSQL, and external PostgreSQL (Supabase). See the [Docker Deployment Guide](docs/deployment/docker.md) for setup instructions and client connection details.

## Kubernetes Deployment

For Kubernetes deployments, a Helm chart is provided with configurable values for different environments. See the [Helm Deployment Guide](docs/deployment/helm.md) for installation instructions, or the [Kubernetes Deployment Guide](docs/deployment/kubernetes.md) for general Kubernetes concepts.

## Authentication

For HTTP transport deployments requiring authentication, see the [Authentication Guide](docs/authentication.md) for bearer token, Google OAuth, and Azure AD configuration options.

<!-- mcp-name: io.github.alex-feel/mcp-context-server -->
