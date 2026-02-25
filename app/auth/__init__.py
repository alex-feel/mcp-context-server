"""Authentication providers for mcp-context-server.

This package provides authentication mechanisms for securing HTTP transports.
Configure via MCP_AUTH_PROVIDER environment variable:
- none: No authentication (default)
- simple_token: Bearer token authentication (requires MCP_AUTH_TOKEN)

Usage:
    ```bash
    # Enable simple bearer token authentication
    export MCP_AUTH_PROVIDER=simple_token
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

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from app.auth.simple_token import SimpleTokenVerifier
from app.settings import AuthSettings
from app.settings import get_settings

if TYPE_CHECKING:
    from fastmcp.server.auth import TokenVerifier as AuthProvider

logger = logging.getLogger(__name__)

__all__ = [
    'AuthSettings',
    'SimpleTokenVerifier',
    'create_auth_provider',
]


def create_auth_provider() -> AuthProvider | None:
    """Create an authentication provider based on MCP_AUTH_PROVIDER setting.

    Returns:
        An auth provider instance for FastMCP's auth= parameter, or None for no auth.
    """
    settings = get_settings()
    provider = settings.auth.provider

    if provider == 'none':
        logger.info('Authentication: disabled (MCP_AUTH_PROVIDER=none)')
        return None

    # provider == 'simple_token'
    verifier = SimpleTokenVerifier()
    logger.info('Authentication: SimpleTokenVerifier (bearer token)')
    return verifier
