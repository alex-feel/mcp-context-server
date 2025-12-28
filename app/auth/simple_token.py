"""Simple token verifier for bearer token authentication.

This module provides a simple token verifier that validates tokens against
a single static token configured via environment variables. Designed to work
with FastMCP's `FASTMCP_SERVER_AUTH` auto-configuration mechanism.

Usage:
    Set the following environment variables:
    - FASTMCP_SERVER_AUTH=app.auth.simple_token.SimpleTokenVerifier
    - MCP_AUTH_TOKEN=your-secret-token

Example:
    ```bash
    # Enable simple token authentication
    export FASTMCP_SERVER_AUTH=app.auth.simple_token.SimpleTokenVerifier
    export MCP_AUTH_TOKEN=my-secret-token-123

    # Run the server
    uv run mcp-context-server
    ```

    Clients must include the token in requests:
    ```
    Authorization: Bearer my-secret-token-123
    ```
"""

from __future__ import annotations

import hmac
import logging
from typing import override

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth import TokenVerifier
from pydantic import SecretStr

from app.settings import get_settings

logger = logging.getLogger(__name__)


class SimpleTokenVerifier(TokenVerifier):
    """Token verifier using a single static token from environment.

    This verifier is designed to work with FastMCP's `FASTMCP_SERVER_AUTH`
    environment variable mechanism. When configured, FastMCP will automatically
    import and instantiate this class.

    The verifier reads the expected token from the `MCP_AUTH_TOKEN` environment
    variable (via centralized AuthSettings). Tokens are compared using constant-time
    comparison to prevent timing attacks.

    Attributes:
        _token: The validated token for comparison.
        _client_id: The client ID to assign to authenticated requests.

    Raises:
        ValueError: If MCP_AUTH_TOKEN is not set or is empty.
    """

    def __init__(self) -> None:
        """Initialize the simple token verifier.

        Reads configuration from centralized settings via get_settings().auth.
        Raises an error if MCP_AUTH_TOKEN is not configured.

        Raises:
            ValueError: If MCP_AUTH_TOKEN is not set in the environment.
        """
        # Initialize parent with no required scopes
        super().__init__(required_scopes=None)

        settings = get_settings()
        auth_settings = settings.auth

        # Validate token is set and not empty
        if auth_settings.auth_token is None:
            error_msg = (
                'MCP_AUTH_TOKEN is required for SimpleTokenVerifier. '
                'Set via environment variable or .env file.'
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        token_value = auth_settings.auth_token.get_secret_value()
        if not token_value.strip():
            error_msg = 'MCP_AUTH_TOKEN cannot be empty.'
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Store the validated token for type narrowing
        self._token: SecretStr = auth_settings.auth_token
        self._client_id: str = auth_settings.auth_client_id

        logger.debug('SimpleTokenVerifier initialized successfully')

    @override
    async def verify_token(self, token: str) -> AccessToken | None:
        """Verify a bearer token against the configured token.

        Args:
            token: The bearer token from the Authorization header.

        Returns:
            AccessToken if the token matches, None otherwise.
        """
        # Reject empty tokens
        if not token or not token.strip():
            logger.debug('Rejecting empty token')
            return None

        # Get the expected token value
        expected_token = self._token.get_secret_value()

        # Use constant-time comparison to prevent timing attacks
        if not hmac.compare_digest(token, expected_token):
            logger.debug('Token validation failed: token mismatch')
            return None

        # Return valid AccessToken
        logger.debug('Token validation successful')
        return AccessToken(
            token=token,
            client_id=self._client_id,
            scopes=['tools:read', 'tools:write'],
            expires_at=None,
            claims={},
        )
