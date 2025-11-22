# MCP Context Server

[![PyPI](https://img.shields.io/pypi/v/mcp-context-server.svg)](https://pypi.org/project/mcp-context-server/) [![MCP Registry](https://img.shields.io/badge/MCP_Registry-listed-blue?logo=anthropic)](https://github.com/modelcontextprotocol/registry) [![GitHub License](https://img.shields.io/github/license/alex-feel/mcp-context-server)](https://github.com/alex-feel/mcp-context-server/blob/main/LICENSE)

A high-performance Model Context Protocol (MCP) server providing persistent multimodal context storage for LLM agents. Built with FastMCP, this server enables seamless context sharing across multiple agents working on the same task through thread-based scoping.


## Key Features

- **Multimodal Context Storage**: Store and retrieve both text and images
- **Thread-Based Scoping**: Agents working on the same task share context through thread IDs
- **Flexible Metadata Filtering**: Store custom structured data with any JSON-serializable fields and filter using 15 powerful operators
- **Tag-Based Organization**: Efficient context retrieval with normalized, indexed tags
- **Semantic Search**: Optional vector similarity search using EmbeddingGemma for meaning-based retrieval
- **High Performance**: SQLite with WAL mode, strategic indexing, and async operations
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
claude mcp add context-server -- uvx mcp-context-server

# Or from GitHub (latest development version)
claude mcp add context-server -- uvx --from git+https://github.com/alex-feel/mcp-context-server mcp-context-server

# Or with semantic search (for setup instructions, see the docs/semantic-search.md)
claude mcp add context-server -- uvx --with mcp-context-server[semantic-search] mcp-context-server

# Or from GitHub (latest development version) with semantic search (for setup instructions, see docs/semantic-search.md)
claude mcp add context-server -- uvx --from git+https://github.com/alex-feel/mcp-context-server --with mcp-context-server[semantic-search] mcp-context-server
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
      "args": ["mcp-context-server"],
      "env": {}
    }
  }
}
```

For the latest development version from GitHub, use:
```json
"args": ["--from", "git+https://github.com/alex-feel/mcp-context-server", "mcp-context-server"]
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
      "args": ["mcp-context-server"],
      "env": {
        "LOG_LEVEL": "${LOG_LEVEL:-INFO}",
        "MCP_CONTEXT_DB": "${MCP_CONTEXT_DB:-~/.mcp/context_storage.db}",
        "MAX_IMAGE_SIZE_MB": "${MAX_IMAGE_SIZE_MB:-10}",
        "MAX_TOTAL_SIZE_MB": "${MAX_TOTAL_SIZE_MB:-100}"
      }
    }
  }
}
```

For more details on environment variable expansion, see: https://docs.claude.com/en/docs/claude-code/mcp#environment-variable-expansion-in-mcp-json

### Supported Environment Variables

- **LOG_LEVEL**: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) - defaults to INFO
- **MCP_CONTEXT_DB**: Database file location - defaults to ~/.mcp/context_storage.db
- **MAX_IMAGE_SIZE_MB**: Maximum size per image in MB - defaults to 10
- **MAX_TOTAL_SIZE_MB**: Maximum total request size in MB - defaults to 100
- **ENABLE_SEMANTIC_SEARCH**: Enable semantic search functionality (true/false) - defaults to false
- **OLLAMA_HOST**: Ollama API host URL for embedding generation - defaults to http://localhost:11434
- **EMBEDDING_MODEL**: Embedding model name for semantic search - defaults to embeddinggemma:latest
- **EMBEDDING_DIM**: Embedding vector dimensions - defaults to 768. **Note**: Changing this after initial setup requires database migration (see [Semantic Search Guide](docs/semantic-search.md#changing-embedding-dimensions))

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

Search context entries with powerful filtering including metadata queries.

**Parameters:**
- `thread_id` (str, optional): Filter by thread
- `source` (str, optional): Filter by source ('user' or 'agent')
- `tags` (list, optional): Filter by tags (OR logic)
- `content_type` (str, optional): Filter by type ('text' or 'multimodal')
- `metadata` (dict, optional): Simple metadata filters (key=value equality)
- `metadata_filters` (list, optional): Advanced metadata filters with operators
- `limit` (int, optional): Maximum results (default: 50, max: 500)
- `offset` (int, optional): Pagination offset
- `include_images` (bool, optional): Include image data in response
- `explain_query` (bool, optional): Include query execution statistics

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

For comprehensive documentation on metadata filtering including real-world use cases, operator examples, nested JSON paths, and performance optimization, see the [Metadata Filtering Guide](docs/metadata-filtering.md).

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
- `tags` (list, optional): New tags (full replacement)
- `images` (list, optional): New images (full replacement)

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

Note: This tool is only available when semantic search is enabled via `ENABLE_SEMANTIC_SEARCH=true` and all dependencies are installed (ollama, numpy, sqlite-vec packages, and EmbeddingGemma model).

**Parameters:**
- `query` (str, required): Natural language search query
- `top_k` (int, optional): Number of results to return (1-100) - defaults to 20
- `thread_id` (str, optional): Filter results to specific thread
- `source` (str, optional): Filter by source type ('user' or 'agent')

**Returns:** Dictionary with:
- Query string
- List of semantically similar context entries with similarity scores
- Result count
- Model name used for embeddings

**Use Cases:**
- Find related work across different threads based on semantic similarity
- Discover contexts with similar meaning but different wording
- Concept-based retrieval without exact keyword matching

For setup instructions, see the [Semantic Search Guide](docs/semantic-search.md).

<!-- mcp-name: io.github.alex-feel/mcp-context-server -->
