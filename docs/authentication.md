# Authentication Guide

## Introduction

This guide covers authentication options for the MCP Context Server when using HTTP transports. Authentication is handled by FastMCP's built-in authentication framework, providing options ranging from simple bearer tokens to enterprise OAuth integrations.

**Key Concepts:**
- Authentication is **only relevant for HTTP transports** (http, sse, streamable-http)
- STDIO transport (default) provides process-level security without authentication
- Three authentication tiers available: no auth, bearer token, and OAuth
- Configuration is entirely via environment variables

## Authentication Tiers Overview

| Tier | Method | Transport | Use Case |
|------|--------|-----------|----------|
| **Tier 1** | No Authentication | STDIO | Local development, Claude Desktop, CLI tools |
| **Tier 2** | Bearer Token | HTTP | Simple API access, CI/CD, internal services |
| **Tier 3** | Google OAuth | HTTP | Google Workspace organizations |
| **Tier 4** | Azure AD OAuth | HTTP | Microsoft 365 / Entra ID organizations |

## Tier 1: No Authentication (STDIO)

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

## Tier 2: Bearer Token Authentication

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

| Variable | Required | Description |
|----------|----------|-------------|
| `FASTMCP_SERVER_AUTH` | Yes | Set to `app.auth.simple_token.SimpleTokenVerifier` |
| `MCP_AUTH_TOKEN` | Yes | The bearer token for authentication |
| `MCP_AUTH_CLIENT_ID` | No | Client ID for authenticated requests (default: `mcp-client`) |

**Example Configuration:**

```bash
# .env file or environment variables
MCP_TRANSPORT=http
FASTMCP_HOST=0.0.0.0
FASTMCP_PORT=8000
FASTMCP_SERVER_AUTH=app.auth.simple_token.SimpleTokenVerifier
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

## Tier 3: Google Workspace OAuth

### When to Use

- Organizations using Google Workspace
- Need to restrict access to corporate accounts only
- Require user identity in audit logs
- Integration with Google Cloud services

### How It Works

FastMCP's built-in `GoogleProvider` handles the complete OAuth 2.1 flow:

1. User redirected to Google's consent screen
2. Google authenticates and authorizes
3. Server receives OAuth tokens
4. Subsequent requests validated via Google's token APIs

### Google Cloud Console Setup

**Step 1: Create/Select Project**

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project or select existing
3. Ensure project is in your Google Workspace organization

**Step 2: Configure OAuth Consent Screen**

1. Navigate to: APIs & Services > OAuth consent screen
2. **User type: Internal** (restricts to organization members only)
3. Fill required fields:
   - App name: "MCP Context Server"
   - User support email: admin@yourcompany.com
   - Developer contact: dev@yourcompany.com
4. Add scopes:
   - `openid` (required)
   - `email` (recommended)
5. Save

**Step 3: Create OAuth Client ID**

1. Navigate to: APIs & Services > Credentials
2. Click "Create Credentials" > "OAuth client ID"
3. Application type: **Web application**
4. Name: "MCP Context Server OAuth"
5. Authorized redirect URIs:
   - Production: `https://your-server.com/auth/callback`
   - Development: `http://localhost:8000/auth/callback`
6. Click Create
7. **Save Client ID and Secret securely**

### Configuration

**Environment Variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `FASTMCP_SERVER_AUTH` | Yes | `fastmcp.server.auth.providers.google.GoogleProvider` |
| `FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_ID` | Yes | OAuth Client ID from Google Console |
| `FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_SECRET` | Yes | OAuth Client Secret |
| `FASTMCP_SERVER_AUTH_GOOGLE_BASE_URL` | Yes | Server's public URL (e.g., `https://your-server.com`) |
| `FASTMCP_SERVER_AUTH_GOOGLE_REQUIRED_SCOPES` | No | Comma-separated scopes (default: `openid`) |

**Example Configuration:**

```bash
# .env file
MCP_TRANSPORT=http
FASTMCP_HOST=0.0.0.0
FASTMCP_PORT=8000

# Google OAuth
FASTMCP_SERVER_AUTH=fastmcp.server.auth.providers.google.GoogleProvider
FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_ID=123456789.apps.googleusercontent.com
FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_SECRET=GOCSPX-your-secret-here
FASTMCP_SERVER_AUTH_GOOGLE_BASE_URL=https://your-server.com
FASTMCP_SERVER_AUTH_GOOGLE_REQUIRED_SCOPES=openid,email
```

### Organization Restriction

Setting the OAuth consent screen to **Internal** automatically restricts access:

- Only users from your Google Workspace organization can authenticate
- Personal Gmail accounts (@gmail.com) cannot authenticate
- Accounts from other organizations cannot authenticate
- No additional configuration needed

### Authentication Flow

1. User accesses MCP endpoint without valid session
2. Redirected to Google sign-in page
3. User authenticates with Google Workspace account
4. Google redirects back to `/auth/callback`
5. Server validates tokens and creates session
6. User can now make authenticated MCP requests

## Tier 4: Azure AD / Microsoft Entra ID OAuth

### When to Use

- Organizations using Microsoft 365 / Azure AD
- Need to restrict access to corporate accounts only
- Require integration with Azure services
- Azure Government cloud deployments

### How It Works

FastMCP's built-in `AzureProvider` handles OAuth 2.1 with Azure AD:

1. User redirected to Microsoft login
2. Azure AD authenticates and validates tenant membership
3. Server receives OAuth tokens
4. Subsequent requests validated via Azure JWKS

### Azure Portal Setup

**Step 1: Create App Registration**

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to: Azure Active Directory > App registrations
3. Click "New registration"
4. Configure:
   - Name: "MCP Context Server"
   - Supported account types: **Accounts in this organizational directory only (Single tenant)**
   - Redirect URI: Web, `https://your-server.com/auth/callback`
5. Click Register
6. **Save the Application (client) ID** and **Directory (tenant) ID**

**Step 2: Create Client Secret**

1. In your App Registration, go to: Certificates & secrets
2. Click "New client secret"
3. Add description and expiration
4. **Save the secret value immediately** (only shown once)

**Step 3: Expose an API (Required)**

1. Go to: Expose an API
2. Set Application ID URI (accept default `api://{client-id}` or customize)
3. Add scope(s):
   - Click "Add a scope"
   - Scope name: `read`
   - Admin consent display name: "Read access"
   - Admin consent description: "Allows read access to MCP tools"
   - State: Enabled
   - Repeat for additional scopes (e.g., `write`)

**Step 4: Set Access Token Version (Critical)**

1. Go to: Manifest
2. Find `"requestedAccessTokenVersion"` (usually `null`)
3. Change to: `"requestedAccessTokenVersion": 2`
4. **Save the manifest**

This is required for FastMCP's AzureProvider to function correctly.

### Configuration

**Environment Variables:**

| Variable | Required | Description |
|----------|----------|-------------|
| `FASTMCP_SERVER_AUTH` | Yes | `fastmcp.server.auth.providers.azure.AzureProvider` |
| `FASTMCP_SERVER_AUTH_AZURE_CLIENT_ID` | Yes | Application (client) ID |
| `FASTMCP_SERVER_AUTH_AZURE_CLIENT_SECRET` | Yes | Client secret value |
| `FASTMCP_SERVER_AUTH_AZURE_TENANT_ID` | Yes | Directory (tenant) ID |
| `FASTMCP_SERVER_AUTH_AZURE_BASE_URL` | Yes | Server's public URL |
| `FASTMCP_SERVER_AUTH_AZURE_REQUIRED_SCOPES` | Yes | Comma-separated scope names (e.g., `read,write`) |

**Example Configuration:**

```bash
# .env file
MCP_TRANSPORT=http
FASTMCP_HOST=0.0.0.0
FASTMCP_PORT=8000

# Azure AD OAuth
FASTMCP_SERVER_AUTH=fastmcp.server.auth.providers.azure.AzureProvider
FASTMCP_SERVER_AUTH_AZURE_CLIENT_ID=835f09b6-0f0f-40cc-85cb-f32c5829a149
FASTMCP_SERVER_AUTH_AZURE_CLIENT_SECRET=your-client-secret-here
FASTMCP_SERVER_AUTH_AZURE_TENANT_ID=08541b6e-646d-43de-a0eb-834e6713d6d5
FASTMCP_SERVER_AUTH_AZURE_BASE_URL=https://your-server.com
FASTMCP_SERVER_AUTH_AZURE_REQUIRED_SCOPES=read,write
```

### Organization Restriction

**Single Tenant Configuration:**

Setting "Supported account types" to "Accounts in this organizational directory only" ensures:

- Only users from your Azure AD tenant can authenticate
- Personal Microsoft accounts cannot authenticate
- Accounts from other organizations cannot authenticate

**Multi-Tenant (Not Recommended for Org-Only Access):**

| tenant_id Value | Access Level |
|-----------------|--------------|
| `<specific-guid>` | Your organization only (recommended) |
| `organizations` | Any Azure AD tenant (multi-tenant) |
| `consumers` | Personal Microsoft accounts only |

### Key Differences from Google OAuth

| Aspect | Google | Azure |
|--------|--------|-------|
| Tenant ID | Implicit (via Internal app) | Explicit (environment variable) |
| Scopes | Optional (defaults to `openid`) | Required (at least one) |
| Scope format | Full URIs | Short names (auto-prefixed) |
| Org restriction | "Internal" app type | "Single tenant" + tenant ID |
| Token format | Opaque (verified via API) | JWT (verified via JWKS) |

### Azure Government Cloud

For Azure Government deployments, add:

```bash
FASTMCP_SERVER_AUTH_AZURE_BASE_AUTHORITY=login.microsoftonline.us
```

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

**HTTP with OAuth:**

OAuth flows require browser-based authentication. The MCP client must support the OAuth redirect flow. Check Claude Desktop documentation for OAuth support status.

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

**OAuth:**
1. Check server's OAuth metadata endpoint (if exposed)
2. Implement OAuth 2.1 authorization code flow with PKCE
3. Use received tokens in subsequent requests

## Environment Variables Reference

### Bearer Token Authentication

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FASTMCP_SERVER_AUTH` | Yes | - | `app.auth.simple_token.SimpleTokenVerifier` |
| `MCP_AUTH_TOKEN` | Yes | - | Bearer token for validation |
| `MCP_AUTH_CLIENT_ID` | No | `mcp-client` | Client ID assigned to authenticated requests |

### Google OAuth

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FASTMCP_SERVER_AUTH` | Yes | - | `fastmcp.server.auth.providers.google.GoogleProvider` |
| `FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_ID` | Yes | - | OAuth Client ID |
| `FASTMCP_SERVER_AUTH_GOOGLE_CLIENT_SECRET` | Yes | - | OAuth Client Secret |
| `FASTMCP_SERVER_AUTH_GOOGLE_BASE_URL` | Yes | - | Server's public URL |
| `FASTMCP_SERVER_AUTH_GOOGLE_REQUIRED_SCOPES` | No | `openid` | Comma-separated scopes |

### Azure AD OAuth

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `FASTMCP_SERVER_AUTH` | Yes | - | `fastmcp.server.auth.providers.azure.AzureProvider` |
| `FASTMCP_SERVER_AUTH_AZURE_CLIENT_ID` | Yes | - | Application (client) ID |
| `FASTMCP_SERVER_AUTH_AZURE_CLIENT_SECRET` | Yes | - | Client secret |
| `FASTMCP_SERVER_AUTH_AZURE_TENANT_ID` | Yes | - | Directory (tenant) ID |
| `FASTMCP_SERVER_AUTH_AZURE_BASE_URL` | Yes | - | Server's public URL |
| `FASTMCP_SERVER_AUTH_AZURE_REQUIRED_SCOPES` | Yes | - | Comma-separated scope names |

## Troubleshooting

### Issue 1: "MCP_AUTH_TOKEN is required" Error

**Symptom:** Server fails to start with token error

**Cause:** `FASTMCP_SERVER_AUTH` is set to `SimpleTokenVerifier` but `MCP_AUTH_TOKEN` is not set

**Solution:**
```bash
# Set the token
export MCP_AUTH_TOKEN=your-secret-token

# Or remove auth requirement
unset FASTMCP_SERVER_AUTH
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

### Issue 3: Google OAuth "Access Denied"

**Symptom:** Personal Gmail or external accounts rejected

**Cause:** OAuth consent screen set to "Internal" (expected behavior)

**Solution:** This is correct - only organization members should access. For external access, reconfigure consent screen (not recommended for security).

### Issue 4: Azure "Invalid Token Version"

**Symptom:** Azure authentication fails with token errors

**Cause:** `requestedAccessTokenVersion` not set to `2`

**Solution:**
1. Go to App Registration > Manifest
2. Set `"requestedAccessTokenVersion": 2`
3. Save and retry

### Issue 5: OAuth Redirect URI Mismatch

**Symptom:** OAuth flow fails with redirect error

**Cause:** `BASE_URL` doesn't match registered redirect URI

**Solution:** Ensure these match exactly:
- `FASTMCP_SERVER_AUTH_*_BASE_URL` (environment)
- Registered redirect URI in Google Console / Azure Portal
- Include protocol (http vs https)

### Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `MCP_AUTH_TOKEN cannot be empty` | Token set to empty string | Provide valid token or remove auth |
| `Token validation failed` | Token mismatch | Verify token matches exactly |
| `Invalid client_id` | OAuth misconfiguration | Check client ID in console |
| `Access denied` | User not in organization | Use organization account |
| `Invalid redirect_uri` | URL mismatch | Match BASE_URL to registered URI |

## Security Recommendations

### For Bearer Token (Tier 2)

1. **Generate strong tokens**: Use `secrets.token_urlsafe(32)` minimum
2. **Use HTTPS**: Required for production to protect token in transit
3. **Rotate periodically**: Change tokens on regular schedule
4. **Limit scope**: Use separate tokens for different services
5. **Monitor usage**: Log authentication events for audit

### For OAuth (Tiers 3-4)

1. **Use Internal/Single-tenant**: Restrict to organization only
2. **Minimum scopes**: Only request necessary OAuth scopes
3. **Short token lifetime**: Configure appropriate expiration
4. **Secure secrets**: Never commit client secrets to source control
5. **Enable MFA**: Require multi-factor for organization accounts

### General Best Practices

1. **HTTPS everywhere**: Use TLS for all HTTP transport deployments
2. **Principle of least privilege**: Grant minimum necessary access
3. **Audit logging**: Enable logging for authentication events
4. **Regular rotation**: Rotate secrets and tokens periodically
5. **Secure storage**: Use secrets managers for credentials

## Related Documentation

- [Docker Deployment Guide](docker-deployment.md) - HTTP transport configuration
- [Semantic Search Guide](semantic-search.md) - Feature configuration
- [README.md](../README.md) - Main documentation and API reference
- [FastMCP Authentication](https://gofastmcp.com/servers/auth) - FastMCP auth documentation
