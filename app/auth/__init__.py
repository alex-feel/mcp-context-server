"""Authentication providers for mcp-context-server.

This package provides authentication mechanisms for securing HTTP transports.
The SimpleTokenVerifier is designed to work with FastMCP's FASTMCP_SERVER_AUTH
environment variable mechanism.

Usage:
    ```bash
    # Enable simple bearer token authentication
    export FASTMCP_SERVER_AUTH=app.auth.simple_token.SimpleTokenVerifier
    export MCP_AUTH_TOKEN=your-secret-token

    # Run the server with HTTP transport
    export MCP_TRANSPORT=http
    uv run mcp-context-server
    ```

See also:
    - app.auth.simple_token: SimpleTokenVerifier implementation
    - app.settings.AuthSettings: Authentication configuration
    - FastMCP authentication docs: https://gofastmcp.com/servers/auth
"""

from app.auth.simple_token import SimpleTokenVerifier
from app.settings import AuthSettings

__all__ = [
    'AuthSettings',
    'SimpleTokenVerifier',
]
