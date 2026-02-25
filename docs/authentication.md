# Authentication Guide

## Introduction

This guide covers authentication options for the MCP Context Server when using HTTP transports. Authentication is handled via a configurable provider, with bearer token authentication available for HTTP deployments.

**Key Concepts:**
- Authentication is **only relevant for HTTP transports** (http, sse, streamable-http)
- STDIO transport (default) provides process-level security without authentication
- Two authentication modes available: no auth (STDIO) and bearer token (HTTP)
- Configuration is entirely via environment variables

## Authentication Methods Overview

| Method            | Transport | Use Case                                     |
|-------------------|-----------|----------------------------------------------|
| No Authentication | STDIO     | Local development, Claude Desktop, CLI tools |
| Bearer Token      | HTTP      | Simple API access, CI/CD, internal services  |

## No Authentication (STDIO)

### When to Use

- Claude Desktop and Claude Code CLI (default configuration)
- Local development and testing
- Single-user deployments
- Trusted network environments

### How It Works

When using STDIO transport (`MCP_TRANSPORT=stdio`, which is the default), the MCP server runs as a subprocess spawned by the client. Security is provided at the process level:

1. Client spawns server as a child process
2. Communication occurs via stdin/stdout
3. No network exposure
4. OS-level process isolation

### Configuration

No authentication configuration needed. This is the default behavior:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-context-server"]
    }
  }
}
```

### Security Considerations

- Server only accessible to the parent process
- No network ports exposed
- File system permissions determine database access
- Suitable for personal/development use

## Bearer Token Authentication

### When to Use

- HTTP transport deployments requiring simple authentication
- CI/CD pipelines and automation
- Internal microservices communication
- Docker deployments with controlled access

### How It Works

The `SimpleTokenVerifier` class validates bearer tokens against a static token configured via environment variables. Key security features:

- **SecretStr handling**: Token never exposed in logs or error messages
- **Constant-time comparison**: Prevents timing attacks via `hmac.compare_digest()`
- **Centralized configuration**: Uses `AuthSettings` for consistent settings management

### Configuration

**Required Environment Variables:**

| Variable             | Required | Description                                                  |
|----------------------|----------|--------------------------------------------------------------|
| `MCP_AUTH_PROVIDER`  | Yes      | Set to `simple_token`                                        |
| `MCP_AUTH_TOKEN`     | Yes      | The bearer token for authentication                          |
| `MCP_AUTH_CLIENT_ID` | No       | Client ID for authenticated requests (default: `mcp-client`) |

**Example Configuration:**

```bash
# .env file or environment variables
MCP_TRANSPORT=http
FASTMCP_HOST=0.0.0.0
FASTMCP_PORT=8000
MCP_AUTH_PROVIDER=simple_token
MCP_AUTH_TOKEN=your-secret-token-here
MCP_AUTH_CLIENT_ID=my-service
```

### Client Configuration

Clients must include the bearer token in the `Authorization` header:

```
Authorization: Bearer your-secret-token-here
```

**Claude Code CLI:**

```bash
# Add HTTP server with Bearer token authentication
claude mcp add --transport http context-server http://localhost:8000/mcp --header "Authorization: Bearer your-secret-token-here"
```

**HTTP Client Example (curl):**

```bash
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-secret-token-here" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

**Python Client Example:**

```python
import httpx

headers = {
    "Authorization": "Bearer your-secret-token-here",
    "Content-Type": "application/json"
}

response = httpx.post(
    "http://localhost:8000/mcp",
    headers=headers,
    json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
)
```

### Security Best Practices

1. **Use strong tokens**: Generate cryptographically secure tokens (32+ characters)
   ```bash
   # Generate secure token
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Never commit tokens**: Use environment variables or secrets management

3. **Use HTTPS in production**: Token transmitted in header requires TLS

4. **Rotate tokens regularly**: Change tokens periodically for long-running deployments

## MCP Client Configuration

### Claude Desktop

Claude Desktop configuration varies by authentication method:

**STDIO (No Auth):**

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["mcp-context-server"]
    }
  }
}
```

**HTTP (No Auth):**

```json
{
  "mcpServers": {
    "context-server": {
      "type": "http",
      "url": "http://localhost:8000/mcp"
    }
  }
}
```

**HTTP with Bearer Token:**

Check Claude Desktop documentation for the latest authentication header support. As of this writing, Claude Desktop's HTTP transport may have limited support for custom authentication headers.

### Claude Code CLI

```bash
# Add STDIO server (no auth)
claude mcp add context-server -- uvx mcp-context-server

# Add HTTP server (no auth)
claude mcp add --transport http context-server http://localhost:8000/mcp

# Add HTTP server with Bearer token authentication
claude mcp add --transport http context-server http://localhost:8000/mcp --header "Authorization: Bearer your-secret-token-here"
```

### Custom MCP Clients

For custom clients implementing MCP protocol:

**Bearer Token:**
```python
# Include in all requests
headers = {"Authorization": f"Bearer {token}"}
```

## Environment Variables Reference

### Bearer Token Authentication

| Variable             | Required | Default      | Description                                  |
|----------------------|----------|--------------|----------------------------------------------|
| `MCP_AUTH_PROVIDER`  | Yes      | -            | `simple_token`                               |
| `MCP_AUTH_TOKEN`     | Yes      | -            | Bearer token for validation                  |
| `MCP_AUTH_CLIENT_ID` | No       | `mcp-client` | Client ID assigned to authenticated requests |

## Troubleshooting

### Issue 1: "MCP_AUTH_TOKEN is required" Error

**Symptom:** Server fails to start with token error

**Cause:** `MCP_AUTH_PROVIDER` is set to `simple_token` but `MCP_AUTH_TOKEN` is not set

**Solution:**
```bash
# Set the token
export MCP_AUTH_TOKEN=your-secret-token

# Or disable auth
export MCP_AUTH_PROVIDER=none
```

### Issue 2: Bearer Token Rejected

**Symptom:** HTTP 401 Unauthorized despite correct token

**Causes:**
- Token mismatch (check for trailing whitespace/newlines)
- Missing "Bearer " prefix in header
- Token not properly URL-encoded if special characters

**Solutions:**
```bash
# Verify exact token match
echo -n "$MCP_AUTH_TOKEN" | xxd

# Test with curl
curl -v -H "Authorization: Bearer $MCP_AUTH_TOKEN" http://localhost:8000/mcp
```

### Common Error Messages

| Error                            | Cause                     | Solution                           |
|----------------------------------|---------------------------|------------------------------------|
| `MCP_AUTH_TOKEN cannot be empty` | Token set to empty string | Provide valid token or remove auth |
| `Token validation failed`        | Token mismatch            | Verify token matches exactly       |

## Security Recommendations

### For Bearer Token

1. **Generate strong tokens**: Use `secrets.token_urlsafe(32)` minimum
2. **Use HTTPS**: Required for production to protect token in transit
3. **Rotate periodically**: Change tokens on regular schedule
4. **Limit scope**: Use separate tokens for different services
5. **Monitor usage**: Log authentication events for audit

### General Best Practices

1. **HTTPS everywhere**: Use TLS for all HTTP transport deployments
2. **Principle of least privilege**: Grant minimum necessary access
3. **Audit logging**: Enable logging for authentication events
4. **Regular rotation**: Rotate secrets and tokens periodically
5. **Secure storage**: Use secrets managers for credentials

## Additional Resources

### Related Documentation

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation
- **Database Backends**: [Database Backends Guide](database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata filtering with operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - HTTP transport configuration
- **Main Documentation**: [README.md](../README.md) - overview and quick start
- **FastMCP Authentication**: [FastMCP Auth](https://gofastmcp.com/servers/auth) - FastMCP auth documentation
