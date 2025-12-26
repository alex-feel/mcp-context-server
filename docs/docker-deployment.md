# Docker Deployment Guide

## Introduction

This guide covers deploying the MCP Context Server using Docker and Docker Compose. Docker deployment enables HTTP transport mode for remote client connections, making the server accessible to MCP clients across networks.

**Key Features:**
- HTTP transport mode for remote access (vs. stdio for local)
- Three pre-configured deployment options (SQLite, PostgreSQL, External PostgreSQL)
- Automatic embedding model download on first start
- Health checks and container orchestration support
- Shared Ollama model volume across configurations

## Prerequisites

- **Docker Engine**: 20.10+ or Docker Desktop
- **Docker Compose**: V2 (included with Docker Desktop)
- **Storage**: ~2GB for images and models (600MB for embeddinggemma)
- **Network**: Port 8000 available for HTTP transport

## Deployment Options

Three Docker Compose configurations are provided, each as a standalone file:

| Configuration | File | Database | Use Case |
|--------------|------|----------|----------|
| SQLite | `docker-compose.sqlite.yml` | Local SQLite | Single-user, testing, development |
| PostgreSQL | `docker-compose.postgresql.yml` | Internal PostgreSQL | Multi-user, production |
| External PostgreSQL | `docker-compose.postgresql-external.yml` | Supabase, corporate DB | Existing database infrastructure |

## Quick Start

### SQLite Deployment (Simplest)

```bash
# Build and start
docker compose -f docker-compose.sqlite.yml up -d

# Wait for embedding model download (first run only, ~2-3 minutes)
docker compose -f docker-compose.sqlite.yml logs -f ollama

# Verify server is ready
curl http://localhost:8000/health
```

### PostgreSQL Deployment (Recommended for Production)

```bash
# Build and start (includes PostgreSQL with pgvector)
docker compose -f docker-compose.postgresql.yml up -d

# Wait for all services to be healthy
docker compose -f docker-compose.postgresql.yml ps

# Verify server is ready
curl http://localhost:8000/health
```

### External PostgreSQL Deployment (Supabase, Corporate)

```bash
# 1. Copy and configure environment file
cp .env.postgresql-external.example .env

# 2. Edit .env with your PostgreSQL connection details
# See "External PostgreSQL Configuration" section below

# 3. Build and start
docker compose -f docker-compose.postgresql-external.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

## Client Connection

Once deployed, connect MCP clients via HTTP transport:

### Claude Desktop Configuration

Add to `claude_desktop_config.json`:

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

### Claude Code CLI

```bash
claude mcp add --transport http context-server http://localhost:8000/mcp
```

### Remote Access

Replace `localhost` with your server's IP or hostname for remote connections:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "http",
      "url": "http://your-server.example.com:8000/mcp"
    }
  }
}
```

## Architecture

### Services Overview

Each Docker Compose file defines two or three services:

**mcp-context-server:**
- Production MCP server image
- HTTP transport on port 8000
- Health check endpoint at `/health`
- Non-root user (UID 10001) for security

**ollama:**
- Custom Ollama image with automatic model pulling
- Downloads `embeddinggemma:latest` on first start
- Health check waits for model availability

**postgres (PostgreSQL configurations only):**
- pgvector/pgvector:pg18-trixie image
- Pre-installed pgvector extension for semantic search
- Persistent volume for data

### Volume Management

| Volume | Purpose | Shared Across |
|--------|---------|---------------|
| `mcp-context-sqlite-data` | SQLite database | SQLite config only |
| `mcp-context-postgres-data` | PostgreSQL data | PostgreSQL config only |
| `ollama-models` | Embedding models (~600MB) | All configurations |

The `ollama-models` volume is shared across all configurations, so switching between SQLite and PostgreSQL does not re-download the embedding model.

### Automatic Model Download

The custom Ollama image (`docker/Dockerfile.ollama`) includes an entrypoint script that:

1. Starts Ollama server on a temporary internal port
2. Checks if the configured embedding model exists
3. Pulls the model if not present
4. Restarts Ollama on the production port

This eliminates manual `ollama pull` steps after deployment.

## Configuration

### Environment Variables

All Docker Compose files use environment variables for configuration. Key settings:

**Transport Settings:**

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_TRANSPORT` | `http` | Transport mode (set to `http` for Docker) |
| `FASTMCP_HOST` | `0.0.0.0` | HTTP bind address |
| `FASTMCP_PORT` | `8000` | HTTP port |

**Search Features:**

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_SEMANTIC_SEARCH` | `true` | Enable vector similarity search |
| `ENABLE_FTS` | `true` | Enable full-text search |
| `ENABLE_HYBRID_SEARCH` | `true` | Enable combined FTS + semantic search |
| `EMBEDDING_MODEL` | `embeddinggemma:latest` | Ollama embedding model |
| `EMBEDDING_DIM` | `768` | Embedding vector dimensions |

**Storage Settings (SQLite):**

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_BACKEND` | `sqlite` | Database backend |
| `DB_PATH` | `/data/context_storage.db` | Database path inside container |

**Storage Settings (PostgreSQL):**

| Variable | Default | Description |
|----------|---------|-------------|
| `STORAGE_BACKEND` | `postgresql` | Database backend |
| `POSTGRESQL_HOST` | `postgres` | PostgreSQL hostname |
| `POSTGRESQL_PORT` | `5432` | PostgreSQL port |
| `POSTGRESQL_USER` | `postgres` | PostgreSQL username |
| `POSTGRESQL_PASSWORD` | `postgres` | PostgreSQL password |
| `POSTGRESQL_DATABASE` | `mcp_context` | PostgreSQL database name |

### External PostgreSQL Configuration

For external PostgreSQL (Supabase, corporate databases), copy `.env.postgresql-external.example` to `.env` and configure:

```bash
# Option A: Individual variables (recommended)
POSTGRESQL_HOST=your-db-host.com
POSTGRESQL_PORT=5432
POSTGRESQL_USER=postgres
POSTGRESQL_PASSWORD='your-secure-password'
POSTGRESQL_DATABASE=mcp_context
POSTGRESQL_SSL_MODE=require

# Option B: Connection string
POSTGRESQL_CONNECTION_STRING=postgresql://user:password@host:5432/database?sslmode=require
```

**Important: Special Characters in Passwords**

If your password contains special characters (`$`, `#`, `&`, `*`, etc.), wrap it in **single quotes** (not double quotes):

```bash
# WRONG - $ will be interpreted as a variable
POSTGRESQL_PASSWORD="pass$word"

# CORRECT - single quotes prevent variable interpolation
POSTGRESQL_PASSWORD='pass$word'
```

### Supabase Configuration

For Supabase, use the Session Pooler connection (supports IPv4):

```bash
POSTGRESQL_HOST=aws-0-us-east-1.pooler.supabase.com
POSTGRESQL_PORT=5432
POSTGRESQL_USER=postgres.your-project-ref
POSTGRESQL_PASSWORD='your-database-password'
POSTGRESQL_DATABASE=postgres
POSTGRESQL_SSL_MODE=require
```

See the [Supabase section in README.md](../README.md#using-with-supabase) for detailed connection setup.

## Verification

### Health Check

```bash
# Check server health
curl http://localhost:8000/health

# Expected response
{"status": "ok"}
```

### Container Status

```bash
# SQLite deployment
docker compose -f docker-compose.sqlite.yml ps

# PostgreSQL deployment
docker compose -f docker-compose.postgresql.yml ps

# Expected output: all services "healthy"
NAME                  STATUS
mcp-context-server    Up (healthy)
ollama                Up (healthy)
postgres              Up (healthy)   # PostgreSQL only
```

### Model Availability

```bash
# Check if embedding model is loaded
docker compose -f docker-compose.sqlite.yml exec ollama ollama list

# Expected output includes:
# embeddinggemma:latest    622 MB
```

### Test MCP Connection

```bash
# List available tools via MCP
curl -X POST http://localhost:8000/mcp \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}'
```

## Troubleshooting

### Issue 1: Ollama Health Check Failing

**Symptom:** `ollama` container stays in "starting" state

**Cause:** Embedding model download takes longer than expected (slow network)

**Solution:**
```bash
# Check download progress
docker compose -f docker-compose.sqlite.yml logs -f ollama

# The entrypoint shows: "Pulling model: embeddinggemma:latest..."
# Wait for: "Model pulled successfully!"
```

### Issue 2: PostgreSQL Connection Refused

**Symptom:** Server fails to connect to PostgreSQL

**Causes:**
- PostgreSQL container not yet ready
- Incorrect credentials in `.env`

**Solutions:**
```bash
# Check PostgreSQL logs
docker compose -f docker-compose.postgresql.yml logs postgres

# Verify PostgreSQL is accepting connections
docker compose -f docker-compose.postgresql.yml exec postgres pg_isready

# Test credentials manually
docker compose -f docker-compose.postgresql.yml exec postgres \
  psql -U postgres -d mcp_context -c "SELECT 1"
```

### Issue 3: External PostgreSQL "getaddrinfo failed"

**Symptom:** Cannot connect to Supabase Direct Connection

**Cause:** IPv6 not available on your system

**Solution:** Use Session Pooler connection instead (see Supabase Configuration above)

### Issue 4: Port 8000 Already in Use

**Symptom:** Container fails to start, port binding error

**Solution:** Modify the port mapping in docker-compose file:
```yaml
ports:
  - "8001:8000"  # Use port 8001 on host
```

Then connect clients to `http://localhost:8001/mcp`

### Issue 5: Semantic Search Not Available

**Symptom:** `semantic_search_context` tool not listed

**Causes:**
- Ollama not healthy yet
- Embedding model not downloaded

**Solutions:**
```bash
# Verify Ollama is healthy
docker compose -f docker-compose.sqlite.yml ps ollama

# Check if model exists
docker compose -f docker-compose.sqlite.yml exec ollama ollama list

# If model missing, trigger download
docker compose -f docker-compose.sqlite.yml exec ollama ollama pull embeddinggemma:latest
```

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `connection refused` | Service not running | Check container status with `docker compose ps` |
| `model not found` | Embedding model not pulled | Wait for automatic download or pull manually |
| `permission denied` | Volume permission issue | Check volume ownership matches UID 10001 |
| `database is locked` | SQLite concurrent access | Expected for SQLite; use PostgreSQL for concurrency |

## Advanced Configuration

### Custom Embedding Model

To use a different embedding model, update both services:

```yaml
# In docker-compose file
services:
  mcp-context-server:
    environment:
      - EMBEDDING_MODEL=nomic-embed-text
      - EMBEDDING_DIM=768

  ollama:
    environment:
      - MODEL=nomic-embed-text
```

### GPU Support (Linux)

For GPU-accelerated embedding generation on Linux:

```yaml
services:
  ollama:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### Building Images

Both the MCP server and Ollama images are built locally:

```bash
# Build MCP server image
docker build -t mcp-context-server .

# Build custom Ollama image
docker build -f docker/Dockerfile.ollama -t mcp-ollama .
```

### Production Considerations

1. **Change default PostgreSQL password** in production deployments
2. **Use named volumes** for data persistence (already configured)
3. **Configure resource limits** for container memory/CPU
4. **Set up monitoring** using the `/health` endpoint
5. **Use reverse proxy** (nginx, traefik) for TLS termination

## Files Reference

| File | Description |
|------|-------------|
| `Dockerfile` | Multi-stage MCP server image |
| `docker-compose.sqlite.yml` | SQLite deployment |
| `docker-compose.postgresql.yml` | PostgreSQL with pgvector deployment |
| `docker-compose.postgresql-external.yml` | External PostgreSQL deployment |
| `.env.postgresql-external.example` | Environment template for external DB |
| `docker/Dockerfile.ollama` | Custom Ollama image |
| `docker/ollama-entrypoint.sh` | Auto model pull entrypoint |
| `.dockerignore` | Build context optimization |

## Related Documentation

- [Semantic Search Guide](semantic-search.md) - Vector similarity search configuration
- [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- [Hybrid Search Guide](hybrid-search.md) - Combined search with RRF fusion
- [README.md](../README.md) - Main documentation and API reference
