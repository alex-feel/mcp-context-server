# MCP Context Server

[![PyPI](https://img.shields.io/pypi/v/mcp-context-server.svg)](https://pypi.org/project/mcp-context-server/) [![MCP Registry](https://img.shields.io/badge/MCP_Registry-listed-blue?logo=anthropic)](https://github.com/modelcontextprotocol/registry) [![GitHub License](https://img.shields.io/github/license/alex-feel/mcp-context-server)](https://github.com/alex-feel/mcp-context-server/blob/main/LICENSE) [![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/alex-feel/mcp-context-server)

A high-performance Model Context Protocol (MCP) server providing persistent multimodal context storage for LLM agents. Built with FastMCP, this server enables seamless context sharing across multiple agents working on the same task through thread-based scoping.


## Key Features

- **Multimodal Context Storage**: Store and retrieve both text and images
- **Thread-Based Scoping**: Agents working on the same task share context through thread IDs
- **Flexible Metadata Filtering**: Store custom structured data with any JSON-serializable fields and filter using 15 powerful operators
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
# From PyPI (recommended)
claude mcp add context-server -- uvx --python 3.12 mcp-context-server

# Or from GitHub (latest development version)
claude mcp add context-server -- uvx --python 3.12 --from git+https://github.com/alex-feel/mcp-context-server mcp-context-server

# Or with semantic search (for setup instructions, see the docs/semantic-search.md)
claude mcp add context-server -- uvx --python 3.12 --with mcp-context-server[semantic-search] mcp-context-server

# Or from GitHub (latest development version) with semantic search (for setup instructions, see docs/semantic-search.md)
claude mcp add context-server -- uvx --python 3.12 --from git+https://github.com/alex-feel/mcp-context-server --with mcp-context-server[semantic-search] mcp-context-server
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
      "args": ["--python", "3.12", "mcp-context-server"],
      "env": {}
    }
  }
}
```

For the latest development version from GitHub, use:
```json
"args": ["--python", "3.12", "--from", "git+https://github.com/alex-feel/mcp-context-server", "mcp-context-server"]
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
      "args": ["--python", "3.12", "mcp-context-server"],
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

**Semantic Search Settings:**
- **ENABLE_SEMANTIC_SEARCH**: Enable semantic search functionality (true/false) - defaults to false
- **OLLAMA_HOST**: Ollama API host URL for embedding generation - defaults to http://localhost:11434
- **EMBEDDING_MODEL**: Embedding model name for semantic search - defaults to embeddinggemma:latest
- **EMBEDDING_DIM**: Embedding vector dimensions - defaults to 768. **Note**: Changing this after initial setup requires database migration (see [Semantic Search Guide](docs/semantic-search.md#changing-embedding-dimensions))

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

### Semantic Search

For detailed instructions on enabling optional semantic search with Ollama and EmbeddingGemma, see the [Semantic Search Guide](docs/semantic-search.md).

### Docker Deployment

For production deployments with HTTP transport and container orchestration, Docker Compose configurations are available for SQLite, PostgreSQL, and external PostgreSQL (Supabase). See the [Docker Deployment Guide](docs/docker-deployment.md) for setup instructions and client connection details.

## Database Backends

The server supports multiple database backends, selectable via the `STORAGE_BACKEND` environment variable.

### SQLite (Default)

Zero-configuration local storage, perfect for single-user deployments.

**Features:**
- No installation required - works out of the box
- Production-grade connection pooling and write queue
- WAL mode for better concurrency
- Suitable for single-user and moderate workloads

**Configuration:** No configuration needed - just start the server!

### PostgreSQL

High-performance backend for multi-user and high-traffic deployments.

**Features:**
- 10x+ write throughput vs SQLite via MVCC
- Native concurrent write support
- JSONB indexing for fast metadata queries
- Production-grade connection pooling with asyncpg
- pgvector extension for semantic search

**Quick Start with Docker:**

Running PostgreSQL with pgvector is incredibly simple - just 2 commands:

```bash
# 1. Pull and run PostgreSQL with pgvector (all-in-one)
docker run --name pgvector18 \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=mcp_context \
  -p 5432:5432 \
  -d pgvector/pgvector:pg18-trixie

# 2. Configure the server (minimal setup - just 2 variables)
export STORAGE_BACKEND=postgresql
export ENABLE_SEMANTIC_SEARCH=true  # Optional: only if you need semantic search
```

**That's it!** The server will automatically:
- Connect to PostgreSQL on startup
- Initialize the schema (creates tables and indexes)
- Enable pgvector extension (comes pre-installed in the Docker image)
- Apply semantic search migration if enabled

**Configuration in .mcp.json:**

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "mcp-context-server"],
      "env": {
        "STORAGE_BACKEND": "postgresql",
        "POSTGRESQL_HOST": "localhost",
        "POSTGRESQL_USER": "postgres",
        "POSTGRESQL_PASSWORD": "postgres",
        "POSTGRESQL_DATABASE": "mcp_context",
        "ENABLE_SEMANTIC_SEARCH": "true"
      }
    }
  }
}
```

**Note:** PostgreSQL settings are only needed when using PostgreSQL. The server uses SQLite by default if `STORAGE_BACKEND` is not set.

### Using with Supabase

Supabase is fully compatible with the PostgreSQL backend using direct database connection. No special configuration needed - Supabase IS PostgreSQL.

**Connection Methods:**

Supabase offers TWO connection methods. Choose based on your network capabilities:

1. **Direct Connection** (IPv6 required, lowest latency)
2. **Session Pooler** (IPv4 compatible, universal)

#### Connection Method 1: Direct Connection (Recommended)

Best for: VMs, servers, and local development with IPv6 support

**Requirements:**
- IPv6 connectivity (or paid dedicated IPv4 add-on)
- Port 5432 accessible
- Lowest latency (~15-20ms)

**Quick Setup:**

1. **Get your database connection details:**

   Navigate to your Supabase Dashboard:
   - Go to **Database → Settings** (left sidebar → Database → Settings)
   - Find the **"Connect to your project"** section
   - Select **"Connection String"** tab, then **"Direct connection"** method
   - You'll see: `postgresql://postgres:[YOUR_PASSWORD]@db.[PROJECT_REF].supabase.co:5432/postgres`

2. **Get or reset your database password:**

   **IMPORTANT:** For security reasons, your database password is **never displayed** in the Supabase Dashboard.

   You must use one of these approaches:
   - **Use the password you set** when creating your Supabase project, OR
   - **Click "Reset database password"** (below the connection string) to generate a new password

   **Note:** Replace `[YOUR_PASSWORD]` in the connection string with your actual database password (NOT API keys - those are for REST/GraphQL APIs).

3. **Configure the connection:**

   Add to your `.mcp.json` with your actual password:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_CONNECTION_STRING": "postgresql://postgres:your-actual-password@db.[PROJECT_REF].supabase.co:5432/postgres"
         }
       }
     }
   }
   ```

   Or using individual environment variables:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_HOST": "db.[PROJECT_REF].supabase.co",
           "POSTGRESQL_PORT": "5432",
           "POSTGRESQL_USER": "postgres",
           "POSTGRESQL_PASSWORD": "your-actual-password",
           "POSTGRESQL_DATABASE": "postgres",
           "ENABLE_SEMANTIC_SEARCH": "true"
         }
       }
     }
   }
   ```

   **Replace** `[PROJECT_REF]` with your actual project reference ID and `your-actual-password` with your database password.

#### Connection Method 2: Session Pooler (IPv4 Compatible)

Best for: Systems without IPv6 support (Windows, corporate networks, restricted environments)

**Requirements:**
- IPv4 connectivity (works universally)
- Port 5432 accessible
- Slightly higher latency (~20-30ms, +5-10ms vs Direct)

**Important Differences from Direct Connection:**
- **Different hostname**: Uses `*.pooler.supabase.com` (NOT `db.*.supabase.co`)
- **Different username format**: `postgres.[PROJECT-REF]` (includes project reference)
- **Same port**: 5432 (NOT 6543 - that's Transaction Pooler for serverless only)
- **IPv4 compatible**: Works on all systems without IPv6 configuration

**Quick Setup:**

1. **Get your Session Pooler connection string:**

   Navigate to your Supabase Dashboard:
   - Go to **Database → Settings** (left sidebar → Database → Settings)
   - Find the **"Connect to your project"** section
   - Select **"Connection String"** tab, then **"Session pooler"** method (NOT "Transaction pooler")
   - You'll see: `postgresql://postgres.[PROJECT-REF]:[YOUR_PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres`

   **Example:**
   ```
   postgresql://postgres.abcdefghijklmno:your-password@aws-0-us-east-1.pooler.supabase.com:5432/postgres
   ```

2. **Get or reset your database password** (same as Direct Connection - see above)

3. **Configure the connection:**

   Add to your `.mcp.json`:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_CONNECTION_STRING": "postgresql://postgres.[PROJECT-REF]:your-actual-password@aws-0-[REGION].pooler.supabase.com:5432/postgres"
         }
       }
     }
   }
   ```

   Or using individual environment variables:

   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_HOST": "aws-0-[REGION].pooler.supabase.com",
           "POSTGRESQL_PORT": "5432",
           "POSTGRESQL_USER": "postgres.[PROJECT-REF]",
           "POSTGRESQL_PASSWORD": "your-actual-password",
           "POSTGRESQL_DATABASE": "postgres",
           "ENABLE_SEMANTIC_SEARCH": "true"
         }
       }
     }
   }
   ```

   **Replace** `[PROJECT-REF]` with your actual project reference, `[REGION]` with your project region (e.g., `us-east-1`), and `your-actual-password` with your database password.

#### Which Connection Method Should I Use?

| Consideration | Direct Connection | Session Pooler |
|--------------|-------------------|----------------|
| **IPv6 Required** | Yes (or paid IPv4 add-on) | No - IPv4 compatible |
| **Latency** | Lowest (~15-20ms) | +5-10ms overhead |
| **Windows Compatibility** | May require IPv6 config | Works universally |
| **Corporate Networks** | May be blocked | Usually works |
| **Configuration** | Simpler (standard PostgreSQL) | Requires correct hostname |
| **Best For** | VMs, servers with IPv6 | Windows, restricted networks |

**Recommendation:**
- **Try Direct Connection first** - it's simpler and faster
- **Switch to Session Pooler if you get "getaddrinfo failed" errors** (indicates IPv6 connectivity issues)

#### Troubleshooting

**"getaddrinfo failed" Error:**

If you see this error with Direct Connection:
```
Error: getaddrinfo ENOTFOUND db.[PROJECT-REF].supabase.co
```

**Solution:** Your system doesn't support IPv6 or it's disabled. Use Session Pooler instead (Method 2 above).

**Why?** Direct Connection (`db.*.supabase.co`) uses IPv6 by default. Session Pooler (`*.pooler.supabase.com`) provides IPv4 compatibility through Supavisor proxy.

**Enabling Semantic Search (pgvector extension):**

If you want to use semantic search with Supabase, you must enable the pgvector extension:

1. **Via Supabase Dashboard** (easiest method):
   - Navigate to **Database → Extensions** (left sidebar)
   - Search for "vector" in the extensions list
   - Find "vector" extension (version 0.8.0+)
   - Click the **toggle switch** to enable it (turns green when enabled)

2. **Via SQL Editor** (alternative method):
   - Navigate to **SQL Editor** in Supabase Dashboard
   - Run: `CREATE EXTENSION IF NOT EXISTS vector;`
   - Verify: `SELECT * FROM pg_extension WHERE extname = 'vector';`

3. **Configure environment**:
   ```json
   {
     "mcpServers": {
       "context-server": {
         "type": "stdio",
         "command": "uvx",
         "args": ["--python", "3.12", "mcp-context-server"],
         "env": {
           "STORAGE_BACKEND": "postgresql",
           "POSTGRESQL_CONNECTION_STRING": "postgresql://postgres:your-actual-password@db.[PROJECT_REF].supabase.co:5432/postgres",
           "ENABLE_SEMANTIC_SEARCH": "true"
         }
       }
     }
   }
   ```

**Note:** The pgvector extension is available on all Supabase projects but must be manually enabled. The server will automatically create the necessary vector columns and indexes on first run.

**Why Direct Connection?**

- **Recommended by Supabase** for backend services and server-side applications
- **Full PostgreSQL capabilities**: pgvector (available, must be enabled), JSONB, transactions, all extensions
- **Better performance**: Lower latency than REST API, native connection pooling
- **Production-ready**: MVCC for concurrent writes, connection pooling with asyncpg
- **Zero special code**: Uses standard PostgreSQL backend - no Supabase-specific implementation needed

**Important Notes:**

- ✅ **Use database password** from Settings → Database section
- ❌ **NOT API keys** - API keys (including legacy service_role key) are for REST/GraphQL APIs, not direct database connection
- ✅ **Use port 5432** for direct connection (recommended for backend services)
- ✅ **pgvector extension** is available on all Supabase projects - enable it via Dashboard → Extensions for semantic search
- ✅ **All PostgreSQL features** work identically - JSONB indexing, metadata filtering, transactions

**Security Best Practices:**

- Store database password in environment variables, not in code
- Use Supabase's connection string format for simplicity
- Enable SSL/TLS by default (handled automatically by Supabase connection)
- Consider using read-only credentials if your use case only needs read access

## API Reference

### Tools

#### store_context

Store a context entry with optional images and flexible metadata.

**Parameters:**
- `thread_id` (str, required): Unique identifier for the conversation/task thread
- `source` (str, required): Either 'user' or 'agent'
- `text` (str, required): Text content to store
- `images` (list, optional): Base64 encoded images with mime_type
- `metadata` (dict, optional): Additional structured data - completely flexible JSON object for your use case
- `tags` (list, optional): Tags for organization (automatically normalized)

**Metadata Flexibility:**
The metadata field accepts any JSON-serializable structure, making the server adaptable to various use cases:
- **Task Management**: Store `status`, `priority`, `assignee`, `due_date`, `completed`
- **Agent Coordination**: Track `agent_name`, `task_name`, `execution_time`, `resource_usage`
- **Knowledge Base**: Include `category`, `relevance_score`, `source_url`, `author`
- **Debugging Context**: Save `error_type`, `stack_trace`, `environment`, `version`
- **Analytics**: Record `user_id`, `session_id`, `event_type`, `timestamp`

**Performance Note:** The following metadata fields are indexed for faster filtering:
- `status`: State information (e.g., 'pending', 'active', 'completed')
- `priority`: Numeric value for range queries
- `agent_name`: Specific agent identifier
- `task_name`: Task title for string searches
- `completed`: Boolean flag for completion state

**Returns:** Dictionary with success status and context_id

#### search_context

Search context entries with powerful filtering including metadata queries and date ranges.

**Parameters:**
- `thread_id` (str, optional): Filter by thread
- `source` (str, optional): Filter by source ('user' or 'agent')
- `tags` (list, optional): Filter by tags (OR logic)
- `content_type` (str, optional): Filter by type ('text' or 'multimodal')
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `limit` (int, optional): Maximum results to return (1-100, default: 30)
- `offset` (int, optional): Pagination offset (default: 0)
- `include_images` (bool, optional): Include image data in response
- `explain_query` (bool, optional): Include query execution statistics

**Metadata Filtering:** Supports simple key=value equality and advanced filtering with 15 operators. See [Metadata Guide](docs/metadata-addition-updating-and-filtering.md).

**Date Filtering:** Supports ISO 8601 date filtering. See sections at the end of this document.

**Returns:** List of matching context entries with optional query statistics

#### get_context_by_ids

Fetch specific context entries by their IDs.

**Parameters:**
- `context_ids` (list, required): List of context entry IDs
- `include_images` (bool, optional): Include image data (default: True)

**Returns:** List of context entries with full content

#### delete_context

Delete context entries by IDs or thread.

**Parameters:**
- `context_ids` (list, optional): Specific IDs to delete
- `thread_id` (str, optional): Delete all entries in a thread

**Returns:** Dictionary with deletion count

#### list_threads

List all active threads with statistics.

**Returns:** Dictionary containing:
- List of threads with entry counts
- Source type distribution
- Multimodal content counts
- Timestamp ranges

#### get_statistics

Get database statistics and usage metrics.

**Returns:** Dictionary with:
- Total entries count
- Breakdown by source and content type
- Total images count
- Unique tags count
- Database size in MB

#### update_context

Update specific fields of an existing context entry.

**Parameters:**
- `context_id` (int, required): ID of the context entry to update
- `text` (str, optional): New text content
- `metadata` (dict, optional): New metadata (full replacement)
- `metadata_patch` (dict, optional): Partial metadata update using RFC 7396 JSON Merge Patch
- `tags` (list, optional): New tags (full replacement)
- `images` (list, optional): New images (full replacement)

**Metadata Update Options:**

Use `metadata` for full replacement or `metadata_patch` for partial updates. These parameters are mutually exclusive.

RFC 7396 JSON Merge Patch semantics (`metadata_patch`):
- New keys are ADDED to existing metadata
- Existing keys are REPLACED with new values
- Null values DELETE keys

```python
# Update single field while preserving others
update_context(context_id=123, metadata_patch={"status": "completed"})

# Add new field and delete another
update_context(context_id=123, metadata_patch={"reviewer": "alice", "draft": None})
```

**Limitations (RFC 7396):** Null values cannot be stored (null means delete key - use full replacement if needed), arrays are replaced entirely (not merged). See [Metadata Guide](docs/metadata-addition-updating-and-filtering.md#partial-metadata-updates-metadata_patch) for details.

**Field Update Rules:**
- **Updatable fields**: text_content, metadata, tags, images
- **Immutable fields**: id, thread_id, source, created_at (preserved for data integrity)
- **Auto-managed fields**: content_type (recalculated based on image presence), updated_at (set to current timestamp)

**Update Behavior:**
- Only provided fields are updated (selective updates)
- Tags and images use full replacement semantics for consistency
- Content type automatically switches between 'text' and 'multimodal' based on image presence
- At least one updatable field must be provided

**Returns:** Dictionary with:
- Success status
- Context ID
- List of updated fields
- Success/error message

#### semantic_search_context

Perform semantic similarity search using vector embeddings.

Note: This tool is only available when semantic search is enabled via `ENABLE_SEMANTIC_SEARCH=true` and all dependencies are installed. The implementation varies by backend:
- **SQLite**: Uses sqlite-vec extension with embedding model via Ollama
- **PostgreSQL**: Uses pgvector extension (pre-installed in pgvector Docker image) with embedding model via Ollama

**Parameters:**
- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](docs/metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string
- List of semantically similar context entries with similarity scores
- Result count
- Model name used for embeddings
- Query execution statistics (only when `explain_query=True`)

**Use Cases:**
- Find related work across different threads based on semantic similarity
- Discover contexts with similar meaning but different wording
- Concept-based retrieval without exact keyword matching
- Find similar content within a specific time period using date filters

**Date Filtering Example:**
```python
# Find similar content from the past week
semantic_search_context(
    query="authentication implementation",
    start_date="2025-11-22",
    end_date="2025-11-29"
)
```

For setup instructions, see the [Semantic Search Guide](docs/semantic-search.md).

#### fts_search_context

Perform full-text search with linguistic processing, relevance ranking, and highlighted snippets.

Note: This tool is only available when FTS is enabled via `ENABLE_FTS=true`. The implementation varies by backend:
- **SQLite**: Uses FTS5 with BM25 ranking. Porter stemmer (English) or unicode61 tokenizer (multilingual).
- **PostgreSQL**: Uses tsvector/tsquery with ts_rank ranking. Supports 29 languages with full stemming.

**Parameters:**
- `query` (str, required): Search query
- `mode` (str, optional): Search mode - `match` (default), `prefix`, `phrase`, or `boolean`
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `highlight` (bool, optional): Include highlighted snippets in results (default: false)
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Search Modes:**
- `match`: Standard word matching with stemming (default)
- `prefix`: Prefix matching for autocomplete-style search
- `phrase`: Exact phrase matching preserving word order
- `boolean`: Boolean operators (AND, OR, NOT) for complex queries

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](docs/metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string and search mode
- List of matching entries with relevance scores and highlighted snippets
- Result count
- FTS availability status

**Example:**
```python
# Search with prefix matching
fts_search_context(
    query="auth",
    mode="prefix",
    thread_id="project-123"
)

# Boolean search with metadata filter
fts_search_context(
    query="authentication AND security",
    mode="boolean",
    metadata_filters=[{"key": "status", "operator": "eq", "value": "active"}]
)
```

#### hybrid_search_context

Perform hybrid search combining FTS and semantic search with Reciprocal Rank Fusion (RRF).

Note: This tool is only available when hybrid search is enabled via `ENABLE_HYBRID_SEARCH=true` and at least one of FTS (`ENABLE_FTS=true`) or semantic search (`ENABLE_SEMANTIC_SEARCH=true`) is enabled. The RRF algorithm combines results from available search methods, boosting documents that appear in both.

**Parameters:**
- `query` (str, required): Natural language search query
- `limit` (int, optional): Maximum results to return (1-100, default: 5)
- `offset` (int, optional): Pagination offset (default: 0)
- `search_modes` (list, optional): Search modes to use - `['fts', 'semantic']` (default: both)
- `fusion_method` (str, optional): Fusion algorithm - `'rrf'` (default)
- `rrf_k` (int, optional): RRF smoothing constant (1-1000, default from HYBRID_RRF_K env var)
- `thread_id` (str, optional): Optional filter by thread
- `source` (str, optional): Filter by source type ('user' or 'agent')
- `tags` (list, optional): Filter by any of these tags (OR logic)
- `content_type` (str, optional): Filter by content type ('text' or 'multimodal')
- `start_date` (str, optional): Filter entries created on or after this date (ISO 8601 format)
- `end_date` (str, optional): Filter entries created on or before this date (ISO 8601 format)
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `include_images` (bool, optional): Include image data in results (default: false)
- `explain_query` (bool, optional): Include query execution statistics (default: false)

**Metadata Filtering:** Supports same filtering syntax as search_context. See [Metadata Guide](docs/metadata-addition-updating-and-filtering.md).

**Returns:** Dictionary with:
- Query string and fusion method
- List of matching entries with combined RRF scores and individual search rankings
- Result count and counts from each search method
- List of search modes actually used
- Query execution statistics (only when `explain_query=True`)

**Scores Breakdown:**
Each result includes a `scores` object with:
- `rrf`: Combined RRF score (higher = better)
- `fts_rank`: Position in FTS results (1-based), null if not in FTS results
- `semantic_rank`: Position in semantic results (1-based), null if not in semantic results
- `fts_score`: Original FTS relevance score (BM25/ts_rank)
- `semantic_distance`: Original semantic distance (L2, lower = more similar)

**Graceful Degradation:**
- If only FTS is available, returns FTS results only
- If only semantic search is available, returns semantic results only
- If neither is available, raises an error

**Example:**
```python
# Full hybrid search
hybrid_search_context(
    query="authentication implementation",
    thread_id="project-123"
)

# Hybrid with metadata filtering
hybrid_search_context(
    query="performance optimization",
    metadata={"status": "completed"},
    metadata_filters=[{"key": "priority", "operator": "gte", "value": 7}]
)

# Single mode through hybrid API (for consistent interface)
hybrid_search_context(
    query="exact phrase",
    search_modes=["fts"]
)
```

For detailed configuration and troubleshooting, see the [Hybrid Search Guide](docs/hybrid-search.md).

### Search Tools Response Structure

All search tools return consistent response structures with common fields and tool-specific additions:

| Field | search_context | semantic_search_context | fts_search_context | hybrid_search_context |
|-------|----------------|------------------------|-------------------|----------------------|
| `results` | List of entries | List of entries | List of entries | List of entries |
| `count` | Yes | Yes | Yes | Yes |
| `query` | No | Yes | Yes | Yes |
| `stats` | explain_query=True | explain_query=True | explain_query=True | explain_query=True |
| `model` | No | Yes (embedding model) | No | No |
| `mode` | No | No | Yes (search mode) | No |
| `language` | No | No | Yes (FTS language) | No |
| `fusion_method` | No | No | No | Yes |
| `search_modes_used` | No | No | No | Yes |
| `fts_count` | No | No | No | Yes |
| `semantic_count` | No | No | No | Yes |

**Entry Fields by Tool:**

| Entry Field | search_context | semantic_search_context | fts_search_context | hybrid_search_context |
|-------------|----------------|------------------------|-------------------|----------------------|
| `id`, `thread_id`, `source`, `content_type` | Yes | Yes | Yes | Yes |
| `text_content` | Truncated (150 chars) | Full | Full | Full |
| `is_truncated` | Yes | No | No | No |
| `metadata`, `tags`, `created_at`, `updated_at` | Yes | Yes | Yes | Yes |
| `images` | include_images=True | include_images=True | include_images=True | include_images=True |
| `distance` | No | Yes (L2 distance) | No | No |
| `score` | No | No | Yes (relevance) | No |
| `highlighted` | No | No | highlight=True | No |
| `scores` (rrf, fts_rank, semantic_rank, etc.) | No | No | No | Yes |

**Notes:**
- `stats` is only included when `explain_query=True` for all search tools
- `search_context` returns truncated text for browsing; use `get_context_by_ids` for full content
- Lower `distance` values indicate higher semantic similarity
- Higher `score` values indicate better FTS relevance

### Batch Operations

The following tools enable efficient batch processing of context entries.

#### store_context_batch

Store multiple context entries in a single batch operation.

**Parameters:**
- `entries` (list, required): List of context entries (max 100). Each entry has:
  - `thread_id` (str, required), `source` (str, required), `text` (str, required)
  - `metadata` (dict, optional), `tags` (list, optional), `images` (list, optional)
- `atomic` (bool, optional): If true, all succeed or all fail (default: true)

**Returns:** Dictionary with success, total, succeeded, failed, results array, message

#### update_context_batch

Update multiple context entries in a single batch operation.

**Parameters:**
- `updates` (list, required): List of update operations (max 100). Each update has:
  - `context_id` (int, required)
  - `text` (str, optional), `metadata` (dict, optional), `metadata_patch` (dict, optional)
  - `tags` (list, optional), `images` (list, optional)
- `atomic` (bool, optional): If true, all succeed or all fail (default: true)

**Note:** `metadata_patch` uses RFC 7396 JSON Merge Patch semantics. See [Metadata Guide](docs/metadata-addition-updating-and-filtering.md#partial-metadata-updates-metadata_patch) for details.

**Returns:** Dictionary with success, total, succeeded, failed, results array, message

#### delete_context_batch

Delete multiple context entries by various criteria. **IRREVERSIBLE.**

**Parameters:**
- `context_ids` (list, optional): Specific context IDs to delete
- `thread_ids` (list, optional): Delete all entries in these threads
- `source` (str, optional): Filter by source ('user' or 'agent') - must combine with another criterion
- `older_than_days` (int, optional): Delete entries older than N days

At least one criterion must be provided. Cascading delete removes associated tags, images, and embeddings.

**Returns:** Dictionary with success, deleted_count, criteria_used, message

### Filtering Reference

The following filtering options apply to `search_context`, `semantic_search_context`, `fts_search_context`, and `hybrid_search_context` tools.

**Metadata Filtering:**

*Simple filtering* (exact match):
```python
metadata={'status': 'active', 'priority': 5}
```

*Advanced filtering* with operators:
```python
metadata_filters=[
    {'key': 'priority', 'operator': 'gt', 'value': 3},
    {'key': 'status', 'operator': 'in', 'value': ['active', 'pending']},
    {'key': 'agent_name', 'operator': 'starts_with', 'value': 'gpt'},
    {'key': 'completed', 'operator': 'eq', 'value': False}
]
```

**Supported Operators:**
- `eq`: Equals (case-insensitive for strings by default)
- `ne`: Not equals
- `gt`, `gte`, `lt`, `lte`: Numeric comparisons
- `in`, `not_in`: List membership
- `exists`, `not_exists`: Field presence
- `contains`, `starts_with`, `ends_with`: String operations
- `is_null`, `is_not_null`: Null checks

All string operators support `case_sensitive: true/false` option.

For comprehensive documentation on metadata filtering including real-world use cases, operator examples, nested JSON paths, and performance optimization, see the [Metadata Guide](docs/metadata-addition-updating-and-filtering.md).

**Date Filtering:**

Filter entries by creation timestamp using ISO 8601 format:
```python
# Find entries from a specific day
search_context(thread_id="project-123", start_date="2025-11-29", end_date="2025-11-29")

# Find entries from a date range
search_context(thread_id="project-123", start_date="2025-11-01", end_date="2025-11-30")

# Find entries with precise timestamp
search_context(thread_id="project-123", start_date="2025-11-29T10:00:00")
```

Supported ISO 8601 formats:
- Date-only: `2025-11-29`
- DateTime: `2025-11-29T10:00:00`
- UTC (Z suffix): `2025-11-29T10:00:00Z`
- Timezone offset: `2025-11-29T10:00:00+02:00`

**Note:** Date-only `end_date` values automatically expand to end-of-day (`T23:59:59.999999`) for intuitive "entire day" behavior. Naive datetime (without timezone) is interpreted as UTC.

<!-- mcp-name: io.github.alex-feel/mcp-context-server -->
