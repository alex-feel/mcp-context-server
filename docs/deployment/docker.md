# Docker Deployment Guide

## Introduction

This guide covers deploying the MCP Context Server using Docker and Docker Compose. Docker deployment enables HTTP transport mode for remote client connections, making the server accessible to MCP clients across networks.

**Key Features:**
- HTTP transport mode for remote access (vs. stdio for local)
- Six pre-configured deployment options (3 database backends x 2 embedding providers)
- Support for both Ollama (local) and OpenAI (cloud) embeddings
- Health checks and container orchestration support
- Shared Ollama model volume across Ollama configurations

## Prerequisites

- **Docker Engine**: 20.10+ or Docker Desktop
- **Docker Compose**: V2 (included with Docker Desktop)
- **Storage**: ~2GB for images and models (~600MB for qwen3-embedding:0.6b with Ollama)
- **Network**: Port 8000 available for HTTP transport
- **OpenAI API Key**: Required for OpenAI embedding configurations

## Deployment Options

Six Docker Compose configurations are provided, organized by embedding provider:

### Ollama Embeddings (Local, Self-Hosted)

| Configuration                | File                                            | Database               | Use Case                          |
|------------------------------|-------------------------------------------------|------------------------|-----------------------------------|
| SQLite + Ollama              | `docker-compose.sqlite.ollama.yml`              | Local SQLite           | Single-user, testing, development |
| PostgreSQL + Ollama          | `docker-compose.postgresql.ollama.yml`          | Internal PostgreSQL    | Multi-user, production            |
| External PostgreSQL + Ollama | `docker-compose.postgresql-external.ollama.yml` | Supabase, corporate DB | Existing database infrastructure  |

### OpenAI Embeddings (Cloud API)

| Configuration                | File                                            | Database               | Use Case                         |
|------------------------------|-------------------------------------------------|------------------------|----------------------------------|
| SQLite + OpenAI              | `docker-compose.sqlite.openai.yml`              | Local SQLite           | Single-user, cloud embeddings    |
| PostgreSQL + OpenAI          | `docker-compose.postgresql.openai.yml`          | Internal PostgreSQL    | Multi-user, cloud embeddings     |
| External PostgreSQL + OpenAI | `docker-compose.postgresql-external.openai.yml` | Supabase, corporate DB | Enterprise with cloud embeddings |

**Choosing Between Ollama and OpenAI:**

| Factor      | Ollama                          | OpenAI                            |
|-------------|---------------------------------|-----------------------------------|
| Cost        | Free (self-hosted)              | Pay-per-use API                   |
| Privacy     | Data stays local                | Data sent to OpenAI               |
| Setup       | Automatic model download        | Requires API key                  |
| Performance | Depends on hardware             | Consistent cloud performance      |
| Model       | qwen3-embedding:0.6b (1024 dim) | text-embedding-3-small (1536 dim) |

## Quick Start

### Ollama Deployments

#### SQLite + Ollama (Simplest)

```bash
# Build and start
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml up -d

# Wait for embedding model download (first run only, ~2-3 minutes)
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml logs -f ollama

# Verify server is ready
curl http://localhost:8000/health
```

#### PostgreSQL + Ollama (Recommended for Production)

```bash
# Build and start (includes PostgreSQL with pgvector)
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml up -d

# Wait for all services to be healthy
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml ps

# Verify server is ready
curl http://localhost:8000/health
```

#### External PostgreSQL + Ollama (Supabase, Corporate)

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql-external.ollama.example deploy/docker/.env

# 2. Edit .env with your PostgreSQL connection details
# See "External PostgreSQL Configuration" section below

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql-external.ollama.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

### OpenAI Deployments

All OpenAI configurations **require** an `.env` file with your OpenAI API key.

#### SQLite + OpenAI

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-sqlite.openai.example deploy/docker/.env

# 2. Edit .env and add your OPENAI_API_KEY
# OPENAI_API_KEY=sk-your-openai-api-key-here

# 3. Build and start
docker compose -f deploy/docker/docker-compose.sqlite.openai.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

#### PostgreSQL + OpenAI

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql.openai.example deploy/docker/.env

# 2. Edit .env and add your OPENAI_API_KEY
# OPENAI_API_KEY=sk-your-openai-api-key-here

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql.openai.yml up -d

# Verify server is ready
curl http://localhost:8000/health
```

#### External PostgreSQL + OpenAI (Supabase, Corporate)

```bash
# 1. Copy and configure environment file
cp deploy/docker/.env-postgresql-external.openai.example deploy/docker/.env

# 2. Edit .env with your OpenAI API key AND PostgreSQL connection details
# See "External PostgreSQL Configuration" section below

# 3. Build and start
docker compose -f deploy/docker/docker-compose.postgresql-external.openai.yml up -d

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

**Ollama configurations** define two or three services:

**mcp-context-server:**
- Production MCP server image
- HTTP transport on port 8000
- Health check endpoint at `/health`
- Non-root user (UID 10001) for security

**ollama:**
- Custom Ollama image with automatic model pulling
- Downloads `qwen3-embedding:0.6b` on first start
- Health check waits for model availability

**postgres (PostgreSQL configurations only):**
- pgvector/pgvector:pg18-trixie image
- Pre-installed pgvector extension for semantic search
- Persistent volume for data

**OpenAI configurations** are simpler - no Ollama service required:

**mcp-context-server:**
- Same as above, but uses OpenAI API for embeddings
- Requires `OPENAI_API_KEY` in `.env` file

**postgres (PostgreSQL configurations only):**
- Same as Ollama configurations

### Volume Management

| Volume                                 | Purpose                   | Used By                   |
|----------------------------------------|---------------------------|---------------------------|
| `mcp-context-sqlite-ollama-data`       | SQLite database           | SQLite + Ollama           |
| `mcp-context-sqlite-openai-data`       | SQLite database           | SQLite + OpenAI           |
| `mcp-context-postgresql-ollama-data`   | PostgreSQL data           | PostgreSQL + Ollama       |
| `mcp-context-postgresql-openai-data`   | PostgreSQL data           | PostgreSQL + OpenAI       |
| `ollama-models`                        | Embedding models (~600MB) | All Ollama configurations |

The `ollama-models` volume is shared across all Ollama configurations, so switching between SQLite and PostgreSQL does not re-download the embedding model.

### Automatic Model Download (Ollama Only)

The custom Ollama image (`deploy/docker/ollama/Dockerfile`) includes an entrypoint script that:

1. Starts Ollama server on a temporary internal port
2. Checks if the configured embedding model exists
3. Pulls the model if not present
4. Restarts Ollama on the production port

This eliminates manual `ollama pull` steps after deployment.

## Configuration

### Environment Variables

All Docker Compose files use environment variables for configuration. Key settings:

**Transport Settings:**

| Variable        | Default   | Description                               |
|-----------------|-----------|-------------------------------------------|
| `MCP_TRANSPORT` | `http`    | Transport mode (set to `http` for Docker) |
| `FASTMCP_HOST`  | `0.0.0.0` | HTTP bind address                         |
| `FASTMCP_PORT`  | `8000`    | HTTP port                                 |

**Search Features:**

| Variable                 | Default | Description                           |
|--------------------------|---------|---------------------------------------|
| `ENABLE_SEMANTIC_SEARCH` | `true`  | Enable vector similarity search       |
| `ENABLE_FTS`             | `true`  | Enable full-text search               |
| `ENABLE_HYBRID_SEARCH`   | `true`  | Enable combined FTS + semantic search |
| `ENABLE_CHUNKING`        | `true`  | Enable text chunking for embeddings   |
| `ENABLE_RERANKING`       | `true`  | Enable cross-encoder reranking        |

**Embedding Settings (Ollama):**

| Variable             | Default                 | Description                 |
|----------------------|-------------------------|-----------------------------|
| `EMBEDDING_PROVIDER` | `ollama`                | Embedding provider          |
| `EMBEDDING_MODEL`    | `qwen3-embedding:0.6b`  | Ollama embedding model      |
| `EMBEDDING_DIM`      | `1024`                  | Embedding vector dimensions |
| `OLLAMA_HOST`        | `http://ollama:11434`   | Ollama API endpoint         |

**Embedding Settings (OpenAI):**

| Variable             | Default                  | Description                     |
|----------------------|--------------------------|---------------------------------|
| `EMBEDDING_PROVIDER` | `openai`                 | Embedding provider              |
| `EMBEDDING_MODEL`    | `text-embedding-3-small` | OpenAI embedding model          |
| `EMBEDDING_DIM`      | `1536`                   | Embedding vector dimensions     |
| `OPENAI_API_KEY`     | (required)               | OpenAI API key (from .env file) |

**Storage Settings (SQLite):**

| Variable          | Default                    | Description                    |
|-------------------|----------------------------|--------------------------------|
| `STORAGE_BACKEND` | `sqlite`                   | Database backend               |
| `DB_PATH`         | `/data/context_storage.db` | Database path inside container |

**Storage Settings (PostgreSQL):**

| Variable              | Default       | Description              |
|-----------------------|---------------|--------------------------|
| `STORAGE_BACKEND`     | `postgresql`  | Database backend         |
| `POSTGRESQL_HOST`     | `postgres`    | PostgreSQL hostname      |
| `POSTGRESQL_PORT`     | `5432`        | PostgreSQL port          |
| `POSTGRESQL_USER`     | `postgres`    | PostgreSQL username      |
| `POSTGRESQL_PASSWORD` | `postgres`    | PostgreSQL password      |
| `POSTGRESQL_DATABASE` | `mcp_context` | PostgreSQL database name |

**Metadata Indexing Settings:**

| Variable                   | Default                 | Description                                              |
|----------------------------|-------------------------|----------------------------------------------------------|
| `METADATA_INDEXED_FIELDS`  | `status,agent_name,...` | Comma-separated fields to index with optional type hints |
| `METADATA_INDEX_SYNC_MODE` | `additive`              | Index sync mode: `strict`, `auto`, `warn`, `additive`    |

See the [Metadata Guide](../metadata-addition-updating-and-filtering.md#environment-variables) for full details on configurable metadata indexing.

**Chunking and Reranking:**

Text chunking and cross-encoder reranking are enabled by default to improve search quality. These features work automatically without configuration.

| Variable           | Default                   | Description                                     |
|--------------------|---------------------------|-------------------------------------------------|
| `ENABLE_CHUNKING`  | `true`                    | Text chunking for long document embedding       |
| `CHUNK_SIZE`       | `1000`                    | Target chunk size in characters                 |
| `ENABLE_RERANKING` | `true`                    | Cross-encoder result reranking                  |
| `RERANKING_MODEL`  | `ms-marco-MiniLM-L-12-v2` | FlashRank model (~34MB, downloads on first use) |

To disable these features, set the environment variables to `false` in your Docker Compose file:

```yaml
environment:
  - ENABLE_CHUNKING=false
  - ENABLE_RERANKING=false
```

For more configuration options, see [Text Chunking](../semantic-search.md#text-chunking) and [Cross-Encoder Reranking](../semantic-search.md#cross-encoder-reranking).

### External PostgreSQL Configuration

For external PostgreSQL (Supabase, corporate databases), copy the appropriate `.env.example` file to `.env` and configure:

**For Ollama:**
```bash
cp deploy/docker/.env-postgresql-external.ollama.example deploy/docker/.env
```

**For OpenAI:**
```bash
cp deploy/docker/.env-postgresql-external.openai.example deploy/docker/.env
```

Then configure PostgreSQL connection:

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

See the [Supabase section in README.md](../../README.md#using-with-supabase) for detailed connection setup.

### Optional .env for Ollama Configurations

Ollama configurations work without any `.env` file. However, you can optionally create one for additional settings like LangSmith tracing:

```bash
# Copy the optional template
cp deploy/docker/.env-sqlite.ollama.example deploy/docker/.env
# or
cp deploy/docker/.env-postgresql.ollama.example deploy/docker/.env

# Edit .env to enable optional features
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_your-langsmith-api-key
```

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
# Ollama deployment
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml ps

# OpenAI deployment
docker compose -f deploy/docker/docker-compose.sqlite.openai.yml ps

# Expected output: all services "healthy"
NAME                                    STATUS
context-server-mcp-context-server-1     Up (healthy)
context-server-ollama-1                 Up (healthy)   # Ollama only
context-server-postgres-1               Up (healthy)   # PostgreSQL only
```

### Model Availability (Ollama Only)

```bash
# Check if embedding model is loaded
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama list

# Expected output includes:
# qwen3-embedding:0.6b    ~600 MB
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
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml logs -f ollama

# The entrypoint shows: "Pulling model: qwen3-embedding:0.6b..."
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
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml logs postgres

# Verify PostgreSQL is accepting connections
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml exec postgres pg_isready

# Test credentials manually
docker compose -f deploy/docker/docker-compose.postgresql.ollama.yml exec postgres \
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

### Issue 5: Semantic Search Not Available (Ollama)

**Symptom:** `semantic_search_context` tool not listed

**Causes:**
- Ollama not healthy yet
- Embedding model not downloaded

**Solutions:**
```bash
# Verify Ollama is healthy
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml ps ollama

# Check if model exists
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama list

# If model missing, trigger download
docker compose -f deploy/docker/docker-compose.sqlite.ollama.yml exec ollama ollama pull qwen3-embedding:0.6b
```

### Issue 6: OpenAI API Key Missing

**Symptom:** Server fails to start with OpenAI configuration

**Cause:** `.env` file missing or `OPENAI_API_KEY` not set

**Solution:**
```bash
# Ensure .env exists
cp deploy/docker/.env-sqlite.openai.example deploy/docker/.env

# Verify OPENAI_API_KEY is set
grep OPENAI_API_KEY deploy/docker/.env
```

### Common Error Messages

| Error                                  | Cause                          | Solution                                                    |
|----------------------------------------|--------------------------------|-------------------------------------------------------------|
| `connection refused`                   | Service not running            | Check container status with `docker compose ps`             |
| `model not found`                      | Embedding model not pulled     | Wait for automatic download or pull manually                |
| `permission denied`                    | Volume permission issue        | Check volume ownership matches UID 10001                    |
| `database is locked`                   | SQLite concurrent access       | Expected for SQLite; use PostgreSQL for concurrency         |
| `Invalid API Key`                      | OpenAI key incorrect           | Verify OPENAI_API_KEY in .env file                          |
| `password authentication failed`       | Wrong PostgreSQL password      | Fix POSTGRESQL_PASSWORD and restart (exit 78)               |
| `database "..." does not exist`        | PostgreSQL database not found  | Create database or fix POSTGRESQL_DATABASE (exit 78)        |
| `pgvector extension is not installed`  | pgvector not enabled           | Enable via dashboard or CREATE EXTENSION (exit 78)          |
| `insufficient privileges`              | Cannot create pgvector         | Grant permissions via dashboard (exit 78)                   |
| `[Errno 111] Connection refused`       | PostgreSQL not running         | Wait for PostgreSQL to start (exit 69, will retry)          |

## Advanced Configuration

### Custom Embedding Model (Ollama)

To use a different Ollama embedding model, update both services:

```yaml
# In docker-compose file
services:
  mcp-context-server:
    environment:
      - EMBEDDING_MODEL=nomic-embed-text
      - EMBEDDING_DIM=1024

  ollama:
    environment:
      - MODEL=nomic-embed-text
```

### Custom Embedding Model (OpenAI)

To use a different OpenAI embedding model:

```yaml
# In docker-compose file
services:
  mcp-context-server:
    environment:
      - EMBEDDING_MODEL=text-embedding-3-large
      - EMBEDDING_DIM=3072  # Adjust for model
```

### GPU Support (Linux, Ollama Only)

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
docker build -f deploy/docker/ollama/Dockerfile -t mcp-ollama .
```

### Building with Different Embedding Providers

The Dockerfile supports building with different embedding providers via build arguments:

```bash
# Build with Ollama embeddings (default)
docker build -t mcp-context-server .

# Build with OpenAI embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-openai -t mcp-context-server-openai .

# Build with Azure OpenAI embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-azure -t mcp-context-server-azure .

# Build with HuggingFace embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-huggingface -t mcp-context-server-huggingface .

# Build with Voyage embeddings
docker build --build-arg EMBEDDING_EXTRA=embeddings-voyage -t mcp-context-server-voyage .

# Build with all embedding providers
docker build --build-arg EMBEDDING_EXTRA=embeddings-all -t mcp-context-server-all .
```

**Available Embedding Extras:**

| Extra                    | Provider         | Package               |
|--------------------------|------------------|-----------------------|
| `embeddings-ollama`      | Ollama (default) | langchain-ollama      |
| `embeddings-openai`      | OpenAI           | langchain-openai      |
| `embeddings-azure`       | Azure OpenAI     | langchain-openai      |
| `embeddings-huggingface` | HuggingFace      | langchain-huggingface |
| `embeddings-voyage`      | Voyage AI        | langchain-voyageai    |
| `embeddings-all`         | All providers    | All packages          |

**Note:** The OpenAI Docker Compose configurations automatically pass the correct build argument. If using Ollama configurations, no build argument is needed (default is `embeddings-ollama`).

### Restart Policies and Exit Codes

The MCP Context Server uses BSD sysexits.h exit codes to signal different failure types to container supervisors:

| Exit Code | Meaning                  | Docker Restart Behavior                   |
|-----------|--------------------------|-------------------------------------------|
| 0         | Normal shutdown          | No restart                                |
| 1         | General error            | Restart if policy allows                  |
| 69        | Dependency unavailable   | Restart with backoff (may recover)        |
| 78        | Configuration error      | Container halted (prevents restart loops) |

**Exit Code 69 (EX_UNAVAILABLE)** indicates external dependencies are temporarily unavailable:
- Ollama service not running (may start later)
- Database temporarily unreachable (PostgreSQL connection refused, network timeout)
- Network connectivity issues

**Exit Code 78 (EX_CONFIG)** indicates configuration problems requiring human intervention:
- Missing API keys (OPENAI_API_KEY, VOYAGE_API_KEY, etc.)
- Missing required packages
- Invalid configuration values
- Embedding model not found (requires `ollama pull` or fix EMBEDDING_MODEL)
- PostgreSQL authentication errors (wrong password, database doesn't exist, missing pgvector extension)

**For detailed PostgreSQL error scenarios,** see [PostgreSQL Backend Error Scenarios](#postgresql-backend-error-scenarios) below.

### Entrypoint Wrapper Script

The Docker image includes an entrypoint wrapper script (`docker-entrypoint.sh`) that intercepts exit codes to prevent infinite restart loops:

**How it works:**

```
Docker starts container
    |
    v
docker-entrypoint.sh runs Python server
    |
    v
Server exits with code (0, 69, 78, or other)
    |
    v
Entrypoint interprets exit code:
    - Code 0: Normal exit
    - Code 78: Configuration error -> exec sleep infinity (container stays running but idle)
    - Code 69: Dependency error -> exit 69 (Docker may restart)
    - Other: Pass through exit code
```

**Configuration errors (exit code 78) - Container halted:**

When the server encounters a configuration error (e.g., embedding model not found), the entrypoint:

1. Prints a clear error message with troubleshooting steps
2. Replaces itself with `sleep infinity`
3. Container remains "Running" (but idle) instead of entering a restart loop

This allows you to:
- Inspect logs with `docker logs <container>`
- Fix the configuration
- Stop and restart manually: `docker stop <container> && docker compose up -d`

**Dependency errors (exit code 69) - May retry:**

When the server encounters a dependency error (e.g., Ollama not running yet), the entrypoint:

1. Prints an informational message
2. Exits with code 69
3. Docker's `on-failure:5` policy may restart the container (up to 5 times)

**Default restart policy:**

All Docker Compose configurations use `restart: "on-failure:5"` for the MCP Context Server:

```yaml
services:
  mcp-context-server:
    restart: "on-failure:5"  # Restart up to 5 times on failure
```

This provides defense in depth:
- Entrypoint handles configuration errors (code 78) by halting
- Restart policy limits dependency error retries (code 69) to 5 attempts

For production, the built-in health check is already configured:

```yaml
services:
  mcp-context-server:
    restart: "on-failure:5"
    healthcheck:
      test: ["CMD", "python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"]
      interval: 30s
      timeout: 5s
      start_period: 10s
      retries: 3
```

### PostgreSQL Backend Error Scenarios

When using the PostgreSQL backend, the server classifies initialization errors to control container restart behavior. This prevents infinite restart loops on configuration errors while allowing retry for transient connectivity issues.

**Error Classification:**

| Error Scenario            | Exception Type                   | Exit Code               | Container Behavior    | Fix Required                                 |
|---------------------------|----------------------------------|-------------------------|-----------------------|----------------------------------------------|
| PostgreSQL not running    | Connection refused (Errno 111)   | 69 (DependencyError)    | Retry with backoff    | Wait for PostgreSQL to start                 |
| Network timeout           | `TimeoutError`, `InterfaceError` | 69 (DependencyError)    | Retry with backoff    | Check network connectivity                   |
| Too many connections      | `TooManyConnectionsError`        | 69 (DependencyError)    | Retry with backoff    | Increase `max_connections` or wait           |
| Wrong password            | `InvalidPasswordError`           | 78 (ConfigurationError) | Halt (sleep infinity) | Fix `POSTGRESQL_PASSWORD`                    |
| Database does not exist   | `InvalidCatalogNameError`        | 78 (ConfigurationError) | Halt (sleep infinity) | Create database or fix `POSTGRESQL_DATABASE` |
| pgvector not installed    | Extension check failure          | 78 (ConfigurationError) | Halt (sleep infinity) | Enable pgvector extension                    |
| Insufficient privileges   | `InsufficientPrivilegeError`     | 78 (ConfigurationError) | Halt (sleep infinity) | Grant permissions via dashboard              |
| Codec registration failed | pgvector codec error             | 78 (ConfigurationError) | Halt (sleep infinity) | Check pgvector installation                  |

**Examples:**

**Scenario 1: PostgreSQL Not Running (Exit 69)**
```
[ERROR] Failed to connect to PostgreSQL: [Errno 111] Connection refused
[docker-entrypoint] Server exited with code 69
```
- **Container behavior:** Restarts up to 5 times (restart policy)
- **Why retry makes sense:** PostgreSQL service may start soon
- **Fix:** Ensure PostgreSQL container is running and healthy

**Scenario 2: Wrong Database Password (Exit 78)**
```
[ERROR] PostgreSQL authentication failed: password authentication failed
[docker-entrypoint] CONFIGURATION ERROR - CONTAINER HALTED
[docker-entrypoint] Server exited with code 78
```
- **Container behavior:** Halts (remains running but idle with `sleep infinity`)
- **Why halt:** Password won't fix itself - requires human intervention
- **Fix:** Update `POSTGRESQL_PASSWORD` in `.env` file and restart container

**Scenario 3: pgvector Extension Not Installed (Exit 78)**
```
[ERROR] pgvector extension is not installed
[docker-entrypoint] CONFIGURATION ERROR - CONTAINER HALTED
[docker-entrypoint] Server exited with code 78
```
- **Container behavior:** Halts (remains running but idle)
- **Why halt:** Extension must be manually enabled
- **Fix:** Enable pgvector via Supabase Dashboard → Extensions or `CREATE EXTENSION vector;`

**Scenario 4: Database Does Not Exist (Exit 78)**
```
[ERROR] PostgreSQL database does not exist: database "mcp_context" does not exist
[docker-entrypoint] CONFIGURATION ERROR - CONTAINER HALTED
[docker-entrypoint] Server exited with code 78
```
- **Container behavior:** Halts (remains running but idle)
- **Why halt:** Database must be created manually
- **Fix:** Create database or fix `POSTGRESQL_DATABASE` environment variable

**Recovery Steps for Configuration Errors (Exit 78):**

1. **Check container logs** to identify specific error:
   ```bash
   docker logs <container_name>
   ```

2. **Fix the configuration** based on error message:
   - Wrong password: Update `.env` file with correct `POSTGRESQL_PASSWORD`
   - Missing database: Create database or fix `POSTGRESQL_DATABASE`
   - Missing extension: Enable pgvector via dashboard or SQL
   - Permission denied: Grant required permissions (Supabase: use dashboard)

3. **Stop and restart** the container:
   ```bash
   docker stop <container_name>
   docker compose -f <your-compose-file>.yml up -d
   ```

**Understanding Exit Code 69 vs 78:**

- **Exit 69 (Dependency Error):** External dependency temporarily unavailable
  - Examples: PostgreSQL starting up, network hiccup, connection pool exhausted
  - Container restarts automatically (may succeed when dependency becomes available)
  - No human intervention needed

- **Exit 78 (Configuration Error):** Configuration problem requiring human fix
  - Examples: Wrong password, missing database, missing extension
  - Container halts to prevent infinite restart loop
  - Requires manual fix and restart

### Troubleshooting Container Halted State

If your container shows "Running" but the server is not responding:

1. **Check container logs:**
   ```bash
   docker logs <container_name>
   ```

2. **Look for "CONFIGURATION ERROR - CONTAINER HALTED" message:**
   ```
   ==============================================
   [FATAL] CONFIGURATION ERROR - CONTAINER HALTED
   ==============================================
   ```

3. **Fix the configuration** based on the error message (e.g., pull missing model, set API key)

4. **Restart the container:**
   ```bash
   docker stop <container_name>
   docker compose -f <your-compose-file>.yml up -d
   ```

### Production Considerations

1. **Change default PostgreSQL password** in production deployments
2. **Use named volumes** for data persistence (already configured)
3. **Configure resource limits** for container memory/CPU
4. **Set up monitoring** using the `/health` endpoint
5. **Use reverse proxy** (nginx, traefik) for TLS termination
6. **Secure OpenAI API key** - never commit `.env` files to version control
7. **Set restart policies** with limited retry count to prevent infinite loops

## Files Reference

### Docker Compose Files

| File                                            | Description                             |
|-------------------------------------------------|-----------------------------------------|
| `docker-compose.sqlite.ollama.yml`              | SQLite + Ollama embeddings              |
| `docker-compose.postgresql.ollama.yml`          | PostgreSQL + Ollama embeddings          |
| `docker-compose.postgresql-external.ollama.yml` | External PostgreSQL + Ollama embeddings |
| `docker-compose.sqlite.openai.yml`              | SQLite + OpenAI embeddings              |
| `docker-compose.postgresql.openai.yml`          | PostgreSQL + OpenAI embeddings          |
| `docker-compose.postgresql-external.openai.yml` | External PostgreSQL + OpenAI embeddings |

### Environment Templates

| File                                      | Description                                    | Required |
|-------------------------------------------|------------------------------------------------|----------|
| `.env-sqlite.ollama.example`              | Optional config for SQLite + Ollama            | No       |
| `.env-postgresql.ollama.example`          | Optional config for PostgreSQL + Ollama        | No       |
| `.env-postgresql-external.ollama.example` | PostgreSQL connection for external DB + Ollama | Yes      |
| `.env-sqlite.openai.example`              | OpenAI API key for SQLite + OpenAI             | Yes      |
| `.env-postgresql.openai.example`          | OpenAI API key for PostgreSQL + OpenAI         | Yes      |
| `.env-postgresql-external.openai.example` | OpenAI + PostgreSQL for external DB            | Yes      |

### Other Files

| File                                 | Description                                       |
|--------------------------------------|---------------------------------------------------|
| `Dockerfile`                         | Multi-stage MCP server image (repository root)    |
| `deploy/docker/docker-entrypoint.sh` | Exit code handler to prevent infinite restart     |
| `deploy/docker/ollama/Dockerfile`    | Custom Ollama image                               |
| `deploy/docker/ollama/entrypoint.sh` | Auto model pull entrypoint                        |
| `.dockerignore`                      | Build context optimization (repository root)      |

## Additional Resources

### Related Documentation

- **API Reference**: [API Reference](../api-reference.md) - complete tool documentation
- **Database Backends**: [Database Backends Guide](../database-backends.md) - database configuration
- **Semantic Search**: [Semantic Search Guide](../semantic-search.md) - vector similarity search configuration
- **Full-Text Search**: [Full-Text Search Guide](../full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](../hybrid-search.md) - combined search with RRF fusion
- **Metadata Filtering**: [Metadata Guide](../metadata-addition-updating-and-filtering.md) - metadata filtering with operators
- **Authentication**: [Authentication Guide](../authentication.md) - bearer token authentication
- **Main Documentation**: [README.md](../../README.md) - overview and quick start

### Kubernetes Deployment

For Kubernetes deployments, see:
- **Helm Chart**: [Helm Deployment Guide](helm.md)
- **Raw Manifests**: [Kubernetes Guide](kubernetes.md)
