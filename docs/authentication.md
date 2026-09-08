# Authentication Guide

## Introduction

This guide covers authentication options for the MCP Context Server when using HTTP transports. Authentication is handled via a configurable provider, with bearer token authentication available for HTTP deployments.

**Key Concepts:**
- Authentication is **only relevant for HTTP transports** (http, sse, streamable-http)
- STDIO transport (default) provides process-level security without authentication
- Three authentication modes available: no auth (STDIO), bearer token (HTTP), and JWT verification (HTTP)
- Configuration is entirely via environment variables

## Authentication Methods Overview

| Method            | Transport | Use Case                                             |
|-------------------|-----------|------------------------------------------------------|
| No Authentication | STDIO     | Local development, Claude Desktop, CLI tools         |
| Bearer Token      | HTTP      | Simple API access, CI/CD, internal services          |
| JWT Verification  | HTTP      | Tokens issued by an identity provider (OIDC/OAuth2)  |

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

```text
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

## JWT Authentication

> **Experimental and non-isolating.** The `jwt` provider verifies IdP-issued tokens and rejects unauthenticated requests, but the server does not yet isolate stored data between principals: any successfully authenticated caller can read and write ALL stored context. Use it today only where every token holder is trusted with the full data set; per-principal access control is under active development.

### When to Use

- Tokens are issued by an identity provider (Keycloak, Microsoft Entra ID, Auth0, or any OIDC/OAuth2-compliant issuer)
- Multiple clients or users need individually issued, expiring credentials
- Key rotation should happen at the IdP without server reconfiguration (JWKS mode)

### How It Works

With `MCP_AUTH_PROVIDER=jwt`, the server validates each bearer token as a JWT: signature (static key or JWKS lookup), expiration, and, when configured, issuer and audience. Verified claims from the token (subject, groups, roles) are available to the server for identity resolution.

Two key modes are supported, and exactly one must be configured:

- **JWKS mode** (`MCP_AUTH_JWT_JWKS_URI`): the server fetches signing keys from the IdP's JWKS endpoint and caches them, so IdP-side key rotation needs no server change. This is the recommended mode for real IdPs.
- **Static key mode** (`MCP_AUTH_JWT_PUBLIC_KEY`): a fixed PEM-encoded public key (asymmetric algorithms) or shared secret (HS* algorithms). Suited to testing and closed environments without a reachable JWKS endpoint.

### Configuration

| Variable                  | Required             | Description                                                                          |
|---------------------------|----------------------|--------------------------------------------------------------------------------------|
| `MCP_AUTH_PROVIDER`       | Yes                  | Set to `jwt`                                                                         |
| `MCP_AUTH_JWT_JWKS_URI`   | One of the key pair  | JWKS endpoint URI (mutually exclusive with the public key)                           |
| `MCP_AUTH_JWT_PUBLIC_KEY` | One of the key pair  | PEM public key or HS* shared secret (mutually exclusive with the JWKS URI)           |
| `MCP_AUTH_JWT_ISSUER`     | Recommended          | Expected `iss` claim; unset skips issuer validation                                  |
| `MCP_AUTH_JWT_AUDIENCE`   | Recommended          | Expected `aud` claim; unset skips audience validation                                |
| `MCP_AUTH_JWT_ALGORITHM`  | No (default `RS256`) | Accepted signing algorithm                                                           |
| `MCP_AUTH_GROUPS_CLAIM`   | No (default `groups`)| Claim carrying group memberships; dotted paths and full-URL keys supported           |
| `MCP_AUTH_ROLES_CLAIM`    | No (default `roles`) | Claim carrying roles; dotted paths and full-URL keys supported                       |

**Example Configuration (JWKS mode):**

```bash
MCP_TRANSPORT=http
MCP_AUTH_PROVIDER=jwt
MCP_AUTH_JWT_JWKS_URI=https://idp.example.com/realms/main/protocol/openid-connect/certs
MCP_AUTH_JWT_ISSUER=https://idp.example.com/realms/main
MCP_AUTH_JWT_AUDIENCE=mcp-context-server
```

Clients send the IdP-issued token exactly like a bearer token: `Authorization: Bearer <jwt>`.

### Claim Mapping

Group and role claims differ across identity providers, so the claim keys are configurable:

- A plain key (`groups`) reads the top-level claim. RFC 9068 names `groups` as the standard claim, which is why it is the default.
- A dotted key (`realm_access.roles`) first tries the literal flat key, then traverses nested claim objects.
- A full-URL key (`https://example.com/groups`) is looked up verbatim; the dots inside the URL are never treated as a path.
- A scalar string claim value is treated as a single-element list.

### Per-IdP Setup

#### Keycloak

Keycloak realm roles arrive nested under `realm_access.roles` automatically; set `MCP_AUTH_ROLES_CLAIM=realm_access.roles` to read them. Group membership requires a mapper: in the client's dedicated scope, add a **Group Membership** mapper with token claim name `groups` (disable **Full group path** unless you want `/parent/child` names). The defaults then work unchanged.

```bash
MCP_AUTH_JWT_JWKS_URI=https://keycloak.example.com/realms/<realm>/protocol/openid-connect/certs
MCP_AUTH_JWT_ISSUER=https://keycloak.example.com/realms/<realm>
MCP_AUTH_JWT_AUDIENCE=<client-id>
MCP_AUTH_ROLES_CLAIM=realm_access.roles
```

#### Microsoft Entra ID

Enable group claims in the app registration (**Token configuration** > **Add groups claim**, or `groupMembershipClaims` in the manifest). Entra emits group **object IDs (GUIDs)**, not display names -- grants and policies must use those GUIDs. When a user belongs to more than the token limit (about 200 groups for JWTs), Entra omits the `groups` claim and emits `_claim_names`/`_claim_sources` overage markers instead; this server treats that case as an EMPTY group list (fail-closed) and logs a warning -- it never calls Microsoft Graph to resolve the membership. Avoid overage by filtering to groups assigned to the application in the app registration.

```bash
MCP_AUTH_JWT_JWKS_URI=https://login.microsoftonline.com/<tenant-id>/discovery/v2.0/keys
MCP_AUTH_JWT_ISSUER=https://login.microsoftonline.com/<tenant-id>/v2.0
MCP_AUTH_JWT_AUDIENCE=<application-client-id>
```

#### Auth0

Auth0 strips non-namespaced custom claims from tokens, so groups and roles must be added via a post-login **Action** under full-URL claim keys:

```javascript
exports.onExecutePostLogin = async (event, api) => {
  const namespace = 'https://example.com';
  api.accessToken.setCustomClaim(`${namespace}/groups`, event.user.groups || []);
  api.accessToken.setCustomClaim(`${namespace}/roles`, event.authorization?.roles || []);
};
```

```bash
MCP_AUTH_JWT_JWKS_URI=https://<tenant>.auth0.com/.well-known/jwks.json
MCP_AUTH_JWT_ISSUER=https://<tenant>.auth0.com/
MCP_AUTH_JWT_AUDIENCE=<api-identifier>
MCP_AUTH_GROUPS_CLAIM=https://example.com/groups
MCP_AUTH_ROLES_CLAIM=https://example.com/roles
```

### Security Best Practices

1. **Pin issuer and audience**: leaving `MCP_AUTH_JWT_ISSUER`/`MCP_AUTH_JWT_AUDIENCE` unset skips those checks, so any token signed by the configured key is accepted
2. **Prefer JWKS mode** for real IdPs: key rotation at the IdP is picked up automatically
3. **Use HTTPS in production**: the JWT travels in the Authorization header and requires TLS
4. **Keep token lifetimes short**: expiration is validated on every request

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

### JWT Authentication

| Variable                  | Required            | Default  | Description                                              |
|---------------------------|---------------------|----------|----------------------------------------------------------|
| `MCP_AUTH_PROVIDER`       | Yes                 | -        | `jwt`                                                    |
| `MCP_AUTH_JWT_JWKS_URI`   | One of the key pair | -        | JWKS endpoint URI                                        |
| `MCP_AUTH_JWT_PUBLIC_KEY` | One of the key pair | -        | PEM public key or HS* shared secret                      |
| `MCP_AUTH_JWT_ISSUER`     | No                  | -        | Expected `iss` claim value                               |
| `MCP_AUTH_JWT_AUDIENCE`   | No                  | -        | Expected `aud` claim value                               |
| `MCP_AUTH_JWT_ALGORITHM`  | No                  | `RS256`  | Accepted signing algorithm                               |
| `MCP_AUTH_GROUPS_CLAIM`   | No                  | `groups` | Claim key for group memberships                          |
| `MCP_AUTH_ROLES_CLAIM`    | No                  | `roles`  | Claim key for roles                                      |

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

### Issue 3: JWT Rejected Despite Valid IdP Login

**Symptom:** HTTP 401 Unauthorized with a freshly issued IdP token

**Causes:**
- Issuer mismatch: the token `iss` differs from `MCP_AUTH_JWT_ISSUER` (trailing slash matters)
- Audience mismatch: the token `aud` does not contain `MCP_AUTH_JWT_AUDIENCE`
- Algorithm mismatch: the IdP signs with a different algorithm than `MCP_AUTH_JWT_ALGORITHM`
- Expired token: the `exp` claim is in the past

**Solutions:** decode the token payload (for example at jwt.io or with `python -c "import base64,json,sys; p=sys.argv[1].split('.')[1]; print(json.dumps(json.loads(base64.urlsafe_b64decode(p+'='*(-len(p)%4))), indent=2))" "$TOKEN"`) and compare `iss`, `aud`, `exp`, and the header `alg` against the configured values. Server-side rejection reasons are logged at WARNING level.

### Issue 4: Empty Groups With Entra ID

**Symptom:** Authentication succeeds but the caller has no group memberships, and the log warns about group overage

**Cause:** the user belongs to more groups than Entra ID embeds in a token, so the `groups` claim was replaced by overage markers; the server fails closed to an empty group list and never calls Microsoft Graph

**Solution:** in the app registration, emit only groups assigned to the application (or otherwise filter the group claims) so the token stays under the limit

### Common Error Messages

| Error                            | Cause                             | Solution                                                 |
|----------------------------------|-----------------------------------|----------------------------------------------------------|
| `MCP_AUTH_TOKEN cannot be empty` | Token set to empty string         | Provide valid token or remove auth                       |
| `Token validation failed`        | Token mismatch                    | Verify token matches exactly                             |
| `requires exactly one of ...`    | Neither JWT key source configured | Set `MCP_AUTH_JWT_PUBLIC_KEY` or `MCP_AUTH_JWT_JWKS_URI` |
| `... are mutually exclusive`     | Both JWT key sources configured   | Unset one of the two key sources                         |
| `Unsupported algorithm`          | Invalid `MCP_AUTH_JWT_ALGORITHM`  | Use a supported HS*/RS*/ES*/PS* value                    |

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
