# Database Backends Guide

## Introduction

The MCP Context Server supports multiple database backends, allowing you to choose the right storage solution for your deployment needs. The backend is selected via the `STORAGE_BACKEND` environment variable.

**Supported Backends:**
- **SQLite** (default): Zero-configuration local storage, perfect for single-user deployments
- **PostgreSQL**: High-performance backend for multi-user and high-traffic deployments
- **Supabase**: Fully compatible PostgreSQL cloud database with managed infrastructure

## SQLite (Default)

Zero-configuration local storage, perfect for single-user deployments.

**Features:**
- No installation required - works out of the box
- Production-grade connection pooling and write queue
- WAL mode for better concurrency
- Suitable for single-user and moderate workloads

**Configuration:** No configuration needed - just start the server!

## PostgreSQL

High-performance backend for multi-user and high-traffic deployments.

**Features:**
- 10x+ write throughput vs SQLite via MVCC
- Native concurrent write support
- JSONB indexing for fast metadata queries
- Production-grade connection pooling with asyncpg
- pgvector extension for semantic search

### Quick Start with Docker

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

### Configuration in .mcp.json

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

## External Connection Pooler Compatibility

When using external PostgreSQL connection poolers (PgBouncer in transaction mode, Pgpool-II, AWS RDS Proxy, etc.), you may encounter connection errors caused by asyncpg's prepared statement caching.

### The Problem

asyncpg (the PostgreSQL driver used by this server) caches prepared statements on the backend connection. When an external pooler reassigns backend connections between requests:

1. The client sends a query referencing a cached prepared statement
2. The backend connection has been reassigned - the statement does not exist on this connection
3. Result: `"connection was closed in the middle of operation"` or similar errors

### The Solution

Disable prepared statement caching by setting:

```bash
POSTGRESQL_STATEMENT_CACHE_SIZE=0
```

### Configuration Reference

| Environment Variable                         | Default | Description                                                                                     |
|----------------------------------------------|---------|-------------------------------------------------------------------------------------------------|
| `POSTGRESQL_STATEMENT_CACHE_SIZE`            | 100     | Prepared statement cache size. Set to `0` to disable caching for external pooler compatibility. |
| `POSTGRESQL_MAX_CACHED_STATEMENT_LIFETIME_S` | 300     | Maximum lifetime of cached statements in seconds. Has no effect when cache size is 0.           |
| `POSTGRESQL_MAX_CACHEABLE_STATEMENT_SIZE`    | 15360   | Maximum statement size to cache in bytes (15KB). Has no effect when cache size is 0.            |

### Pooler-Specific Notes

**Works without modification (cache enabled):**
- Direct PostgreSQL connection (no pooler)
- PgBouncer in **session mode** (1:1 client-to-backend mapping)
- Supabase **Session Pooler** (maintains session state)

**Requires `POSTGRESQL_STATEMENT_CACHE_SIZE=0`:**
- PgBouncer in **transaction mode** (reassigns connections between transactions)
- Pgpool-II (all modes - reassigns connections across clients)
- AWS RDS Proxy (transaction-level pooling)
- Supabase **Transaction Pooler** (serverless mode)
- Any pooler that reassigns backend connections

### Example Configuration

For environments using external connection poolers:

```json
{
  "mcpServers": {
    "context-server": {
      "type": "stdio",
      "command": "uvx",
      "args": ["--python", "3.12", "mcp-context-server"],
      "env": {
        "STORAGE_BACKEND": "postgresql",
        "POSTGRESQL_HOST": "your-pooler-host",
        "POSTGRESQL_USER": "postgres",
        "POSTGRESQL_PASSWORD": "your-password",
        "POSTGRESQL_DATABASE": "mcp_context",
        "POSTGRESQL_STATEMENT_CACHE_SIZE": "0"
      }
    }
  }
}
```

### Diagnostic Steps

If you encounter connection errors with PostgreSQL:

1. **Check the error message**: Look for `"connection was closed in the middle of operation"`, `"prepared statement does not exist"`, or similar
2. **Identify your pooler type**: Determine whether you're using an external pooler and its mode
3. **Apply the fix**: Set `POSTGRESQL_STATEMENT_CACHE_SIZE=0` in your environment
4. **Verify**: Restart the server and confirm operations complete successfully

### Additional Considerations

**Pgpool-II users**: Some Pgpool-II versions prior to 4.2.3 have known issues with asyncpg's extended query protocol. If disabling statement caching does not resolve issues, consider upgrading Pgpool-II.

**Performance impact**: Disabling statement caching has minimal performance impact for this server. Embedding generation (which takes 800ms+) dominates latency; the overhead of re-preparing statements is negligible.

### External References

- [asyncpg FAQ: Using asyncpg with external poolers](https://magicstack.github.io/asyncpg/current/faq.html)
- [asyncpg Issue #309: Connection closed during operation](https://github.com/MagicStack/asyncpg/issues/309)
- [asyncpg Issue #573: Pgpool-II compatibility](https://github.com/MagicStack/asyncpg/issues/573)

## Using with Supabase

Supabase is fully compatible with the PostgreSQL backend using direct database connection. No special configuration needed - Supabase IS PostgreSQL.

### Connection Methods

Supabase offers TWO connection methods. Choose based on your network capabilities:

1. **Direct Connection** (IPv6 required, lowest latency)
2. **Session Pooler** (IPv4 compatible, universal)

### Connection Method 1: Direct Connection (Recommended)

Best for: VMs, servers, and local development with IPv6 support

**Requirements:**
- IPv6 connectivity (or paid dedicated IPv4 add-on)
- Port 5432 accessible
- Lowest latency (~15-20ms)

**Quick Setup:**

1. **Get your database connection details:**

   Navigate to your Supabase Dashboard:
   - Go to **Database -> Settings** (left sidebar -> Database -> Settings)
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

### Connection Method 2: Session Pooler (IPv4 Compatible)

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
   - Go to **Database -> Settings** (left sidebar -> Database -> Settings)
   - Find the **"Connect to your project"** section
   - Select **"Connection String"** tab, then **"Session pooler"** method (NOT "Transaction pooler")
   - You'll see: `postgresql://postgres.[PROJECT-REF]:[YOUR_PASSWORD]@aws-0-[REGION].pooler.supabase.com:5432/postgres`

   **Example:**
   ```text
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

### Which Connection Method Should I Use?

| Consideration             | Direct Connection             | Session Pooler               |
|---------------------------|-------------------------------|------------------------------|
| **IPv6 Required**         | Yes (or paid IPv4 add-on)     | No - IPv4 compatible         |
| **Latency**               | Lowest (~15-20ms)             | +5-10ms overhead             |
| **Windows Compatibility** | May require IPv6 config       | Works universally            |
| **Corporate Networks**    | May be blocked                | Usually works                |
| **Configuration**         | Simpler (standard PostgreSQL) | Requires correct hostname    |
| **Best For**              | VMs, servers with IPv6        | Windows, restricted networks |

**Recommendation:**
- **Try Direct Connection first** - it's simpler and faster
- **Switch to Session Pooler if you get "getaddrinfo failed" errors** (indicates IPv6 connectivity issues)

### Session Pooler Connection Limits

**IMPORTANT:** Supabase Session Pooler enforces a `pool_size` limit on total concurrent connections. The default application pool settings (`POSTGRESQL_POOL_MAX=20`) may exceed this limit, causing connection errors.

#### Supabase Session Pooler Limits by Tier

| Plan                     | pool_size | Notes                 |
|--------------------------|-----------|-----------------------|
| Free (Nano)              | 15        | Shared infrastructure |
| Pro (Micro/Small/Medium) | 15        | Dedicated compute     |
| Pro (Large+)             | 30+       | Scales with compute   |

**Note:** These are Session Pooler limits. Direct Connection has higher limits (60-500+ depending on tier). Check your Supabase Dashboard -> Settings -> Database for exact values.

#### Pool Sizing Configuration

Configure the application connection pool to fit within Supabase limits:

**Single replica deployment:**

```json
{
  "env": {
    "POSTGRESQL_POOL_MIN": "1",
    "POSTGRESQL_POOL_MAX": "10"
  }
}
```

**Multi-replica deployment (Kubernetes):**

Use this formula to calculate safe pool settings:

```text
POSTGRESQL_POOL_MAX = floor(pool_size / replica_count) - 1
```

| Replicas | Supabase pool_size | Recommended POSTGRESQL_POOL_MAX |
|----------|--------------------|---------------------------------|
| 1        | 15                 | 10                              |
| 2        | 15                 | 6                               |
| 3        | 15                 | 4                               |
| 4        | 15                 | 3                               |

**Example for 3-replica Kubernetes deployment with Supabase Free tier:**

```json
{
  "env": {
    "POSTGRESQL_POOL_MIN": "1",
    "POSTGRESQL_POOL_MAX": "4"
  }
}
```

**Why leave headroom?** Reserve connections for:
- Administrative queries (Supabase Dashboard)
- Connection spikes during pod restarts
- Other applications sharing the pooler

### "MaxClientsInSessionMode" Error

**Symptom:**
```text
MaxClientsInSessionMode: max clients reached - in Session mode max clients are limited to pool_size
```

**Cause:** Application pool exceeds Supabase Session Pooler's `pool_size` limit.

**Solution:**

1. **Reduce pool settings:**
   ```bash
   export POSTGRESQL_POOL_MIN=1
   export POSTGRESQL_POOL_MAX=5
   ```

2. **Check replica count:** If running multiple replicas, divide pool size across all replicas using the formula above.

3. **Consider upgrading:** Higher Supabase tiers have larger pool_size limits.

4. **Use Direct Connection:** If your network supports IPv6, Direct Connection has much higher connection limits (60-500+).

## Troubleshooting

### "getaddrinfo failed" Error

If you see this error with Direct Connection:
```text
Error: getaddrinfo ENOTFOUND db.[PROJECT-REF].supabase.co
```

**Solution:** Your system doesn't support IPv6 or it's disabled. Use Session Pooler instead (Method 2 above).

**Why?** Direct Connection (`db.*.supabase.co`) uses IPv6 by default. Session Pooler (`*.pooler.supabase.com`) provides IPv4 compatibility through Supavisor proxy.

### Enabling Semantic Search (pgvector extension)

If you want to use semantic search with Supabase, you must enable the pgvector extension:

1. **Via Supabase Dashboard** (easiest method):
   - Navigate to **Database -> Extensions** (left sidebar)
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

## Connection Resilience

The PostgreSQL backend includes multi-layered protection against connection drops caused by network intermediaries (NAT, firewalls, proxies) that may close idle connections.

### TCP Keepalive (Automatic)

TCP keepalive is enabled by default on all PostgreSQL connections:

| Setting        | Env Variable                           | Default | Description                             |
|----------------|----------------------------------------|---------|-----------------------------------------|
| Idle timeout   | `POSTGRESQL_TCP_KEEPALIVES_IDLE_S`     | 15      | Seconds before first keepalive probe    |
| Probe interval | `POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S` | 5       | Seconds between keepalive probes        |
| Probe count    | `POSTGRESQL_TCP_KEEPALIVES_COUNT`      | 3       | Failed probes before connection is dead |

**How it works:**
- **Client-side** (PRIMARY): Configured via `setsockopt()` on each connection's socket. Works through Supavisor, PgBouncer, and all connection proxies.
- **Server-side** (SECONDARY): Configured via PostgreSQL GUC parameters. Effective for direct connections; silently ignored by Supavisor/PgBouncer.

**Cross-platform:** Uses `hasattr()` checks for socket constants. Works on Linux, macOS, and Windows 10 v1703+.

To disable TCP keepalive, set all three values to 0:
```bash
POSTGRESQL_TCP_KEEPALIVES_IDLE_S=0
POSTGRESQL_TCP_KEEPALIVES_INTERVAL_S=0
POSTGRESQL_TCP_KEEPALIVES_COUNT=0
```

### Transaction Heartbeat

Long-running transactions (e.g., storing context with embeddings) include periodic `SELECT 1` heartbeat queries between sequential database operations. This prevents intermediaries from classifying the connection as idle during inter-operation gaps.

### Transaction Retry

If a transaction fails due to a connection error, it is automatically retried up to 2 times with exponential backoff (0.5s, 1.0s). Only connection-level errors trigger retry; logical errors are raised immediately. Operations using deduplication are safe to retry.

## Why Direct Connection?

- **Recommended by Supabase** for backend services and server-side applications
- **Full PostgreSQL capabilities**: pgvector (available, must be enabled), JSONB, transactions, all extensions
- **Better performance**: Lower latency than REST API, native connection pooling
- **Production-ready**: MVCC for concurrent writes, connection pooling with asyncpg
- **Zero special code**: Uses standard PostgreSQL backend - no Supabase-specific implementation needed

## Important Notes

- Use **database password** from Settings -> Database section
- **NOT API keys** - API keys (including legacy service_role key) are for REST/GraphQL APIs, not direct database connection
- Use **port 5432** for direct connection (recommended for backend services)
- **pgvector extension** is available on all Supabase projects - enable it via Dashboard -> Extensions for semantic search
- **All PostgreSQL features** work identically - JSONB indexing, metadata filtering, transactions

## Security Best Practices

- Store database password in environment variables, not in code
- Use Supabase's connection string format for simplicity
- Enable SSL/TLS by default (handled automatically by Supabase connection)
- Consider using read-only credentials if your use case only needs read access

## Additional Resources

### Related Documentation

- **API Reference**: [API Reference](api-reference.md) - complete tool documentation
- **Semantic Search**: [Semantic Search Guide](semantic-search.md) - vector similarity search setup
- **Full-Text Search**: [Full-Text Search Guide](full-text-search.md) - FTS configuration and usage
- **Hybrid Search**: [Hybrid Search Guide](hybrid-search.md) - combined FTS + semantic search
- **Metadata Filtering**: [Metadata Guide](metadata-addition-updating-and-filtering.md) - metadata operators
- **Docker Deployment**: [Docker Deployment Guide](deployment/docker.md) - containerized deployment
- **Authentication**: [Authentication Guide](authentication.md) - HTTP transport authentication
- **Main Documentation**: [README.md](../README.md) - overview and quick start
