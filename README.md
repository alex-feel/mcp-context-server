# MCP Context Server

A high-performance Model Context Protocol (MCP) server providing persistent multimodal context storage for LLM agents. Built with FastMCP, this server enables seamless context sharing across multiple agents working on the same task through thread-based scoping.


## Key Features

- **Multimodal Context Storage**: Store and retrieve both text and images (up to 10MB per image, 100MB total)
- **Thread-Based Scoping**: Agents working on the same task share context through thread IDs
- **Tag-Based Organization**: Efficient context retrieval with normalized, indexed tags
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
claude mcp add context-server -- uvx --from git+https://github.com/alex-feel/mcp-context-server mcp-context-server
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
      "args": [
        "--from",
        "git+https://github.com/alex-feel/mcp-context-server",
        "mcp-context-server"
      ],
      "env": {}
    }
  }
}
```

For configuration file locations and details, see: https://docs.claude.com/en/docs/claude-code/settings#settings-files

### Verifying Installation

```bash
# Start Claude Code
claude

# Check MCP tools are available
/mcp
```

## Troubleshooting

### Common Issues

#### Server Not Starting
- **Verify `uv` installation**: Ensure `uv` is installed and in PATH
- **Network connectivity**: Check access to GitHub for package installation
- **Check configuration**: Verify the server configuration is correct

#### Server Not Listed
- Check MCP tools with `/mcp` command in Claude Code
- Verify the configuration was saved correctly
- Restart Claude Code after configuration changes

### Common Errors

- **"Server not responding"**: Check if `uvx` can access GitHub
- **"Command not found"**: Ensure `uv` is installed and in PATH
- **Configuration issues**: Check your `.mcp.json` file or re-run the CLI command

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
      "args": [
        "--from",
        "git+https://github.com/alex-feel/mcp-context-server",
        "mcp-context-server"
      ],
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

### Advanced Configuration

Additional environment variables are available for advanced server tuning, including:
- Connection pool configuration
- Retry behavior settings
- SQLite performance optimization
- Circuit breaker thresholds
- Operation timeouts

For a complete list of all configuration options, see `app/settings.py`.

## API Reference

### Tools

#### store_context

Store a context entry with optional images.

**Parameters:**
- `thread_id` (str, required): Unique identifier for the conversation/task thread
- `source` (str, required): Either 'user' or 'agent'
- `text` (str, required): Text content to store
- `images` (list, optional): Base64 encoded images with mime_type
- `metadata` (dict, optional): Additional structured data
- `tags` (list, optional): Tags for organization (automatically normalized)

**Returns:** Dictionary with success status and context_id

#### search_context

Search context entries with efficient filtering.

**Parameters:**
- `thread_id` (str, optional): Filter by thread
- `source` (str, optional): Filter by source ('user' or 'agent')
- `tags` (list, optional): Filter by tags (OR logic)
- `content_type` (str, optional): Filter by type ('text' or 'multimodal')
- `limit` (int, optional): Maximum results (default: 50, max: 500)
- `offset` (int, optional): Pagination offset
- `include_images` (bool, optional): Include image data in response

**Returns:** List of matching context entries

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

<!-- mcp-name: io.github.alex-feel/mcp-context-server -->
