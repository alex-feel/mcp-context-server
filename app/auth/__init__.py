"""Authentication providers for mcp-context-server.

This package provides authentication mechanisms for securing HTTP transports.
Configure via MCP_AUTH_PROVIDER environment variable:
- none: No authentication (default)
- simple_token: Bearer token authentication (requires MCP_AUTH_TOKEN)
- jwt: IdP-issued JWT verification (requires exactly one of
  MCP_AUTH_JWT_PUBLIC_KEY or MCP_AUTH_JWT_JWKS_URI)

Usage:
    ```bash
    # Enable simple bearer token authentication
    export MCP_AUTH_PROVIDER=simple_token
    export MCP_AUTH_TOKEN=your-secret-token

    # Or verify JWTs issued by an identity provider
    export MCP_AUTH_PROVIDER=jwt
    export MCP_AUTH_JWT_JWKS_URI=https://idp.example.com/realms/main/protocol/openid-connect/certs
    export MCP_AUTH_JWT_ISSUER=https://idp.example.com/realms/main
    export MCP_AUTH_JWT_AUDIENCE=mcp-context-server

    # Run the server with HTTP transport
    export MCP_TRANSPORT=http
    uv run mcp-context-server
    ```

See also:
    - app.auth.simple_token: SimpleTokenVerifier implementation
    - app.auth.claims: IdP-agnostic claim extraction helpers
    - app.auth.principal: per-request RequestPrincipal resolution
    - app.auth.access: effective-principal resolution and the publish gate
    - app.settings.AuthSettings: Authentication configuration
    - FastMCP authentication docs: https://gofastmcp.com/servers/auth
"""


import logging
from typing import TYPE_CHECKING

from app.auth.access import resolve_effective_principal
from app.auth.access import visibility_denied_reason
from app.auth.principal import RequestPrincipal
from app.auth.principal import resolve_request_principal
from app.auth.simple_token import SimpleTokenVerifier
from app.errors import ConfigurationError
from app.settings import AuthSettings
from app.settings import get_settings

if TYPE_CHECKING:
    from fastmcp.server.auth import TokenVerifier as AuthProvider
    from fastmcp.server.auth.providers.jwt import JWTVerifier

logger = logging.getLogger(__name__)

__all__ = [
    'AuthSettings',
    'RequestPrincipal',
    'SimpleTokenVerifier',
    'create_auth_provider',
    'resolve_effective_principal',
    'resolve_request_principal',
    'visibility_denied_reason',
]


def _create_jwt_verifier(auth_settings: AuthSettings) -> 'JWTVerifier':
    """Construct a JWTVerifier from AuthSettings, validating the key configuration.

    Args:
        auth_settings: The resolved authentication settings.

    Returns:
        A configured fastmcp JWTVerifier.

    Raises:
        ConfigurationError: When the key configuration does not name exactly
            one of MCP_AUTH_JWT_PUBLIC_KEY / MCP_AUTH_JWT_JWKS_URI, or when
            JWTVerifier rejects the configuration (unsupported algorithm,
            symmetric algorithm with a JWKS URI or a PEM public key).
    """
    # Deferred import: the JWT provider pulls in authlib/cryptography, which
    # non-jwt startups (including every stdio startup) should not pay for.
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    # Treat empty/whitespace values as unset, mirroring the MCP_AUTH_TOKEN
    # empty-string handling in SimpleTokenVerifier.
    public_key = (auth_settings.jwt_public_key.get_secret_value() if auth_settings.jwt_public_key else '').strip()
    jwks_uri = (auth_settings.jwt_jwks_uri or '').strip()

    if public_key and jwks_uri:
        raise ConfigurationError(
            'MCP_AUTH_JWT_PUBLIC_KEY and MCP_AUTH_JWT_JWKS_URI are mutually exclusive. '
            'Set exactly one for MCP_AUTH_PROVIDER=jwt.',
        )
    if not public_key and not jwks_uri:
        raise ConfigurationError(
            'MCP_AUTH_PROVIDER=jwt requires exactly one of MCP_AUTH_JWT_PUBLIC_KEY '
            '(static key or shared secret) or MCP_AUTH_JWT_JWKS_URI (JWKS endpoint).',
        )

    try:
        return JWTVerifier(
            public_key=public_key or None,
            jwks_uri=jwks_uri or None,
            issuer=auth_settings.jwt_issuer,
            audience=auth_settings.jwt_audience,
            algorithm=auth_settings.jwt_algorithm,
        )
    except ValueError as e:
        raise ConfigurationError(str(e)) from e


def create_auth_provider() -> 'AuthProvider | None':
    """Create an authentication provider based on MCP_AUTH_PROVIDER setting.

    Returns:
        An auth provider instance for FastMCP's auth= parameter, or None for no auth.

    Raises:
        ConfigurationError: When MCP_AUTH_PROVIDER=simple_token but MCP_AUTH_TOKEN
            is missing or empty, or when MCP_AUTH_PROVIDER=jwt with an invalid key
            configuration (neither or both of MCP_AUTH_JWT_PUBLIC_KEY /
            MCP_AUTH_JWT_JWKS_URI, an unsupported algorithm, or a symmetric
            algorithm combined with a JWKS URI or PEM public key). These are
            startup misconfigurations (missing or contradictory environment
            variables), so they are classified as ConfigurationError (exit 78)
            rather than letting the underlying ValueError reach the generic
            handler and exit 1 -- exit 78 tells a supervisor NOT to crash-loop
            on a restart that cannot fix the config.
    """
    settings = get_settings()
    provider = settings.auth.provider

    if provider == 'none':
        logger.info('Authentication: disabled (MCP_AUTH_PROVIDER=none)')
        return None

    if provider == 'jwt':
        verifier = _create_jwt_verifier(settings.auth)
        logger.info(
            'Authentication: JWTVerifier (%s, algorithm %s)',
            'JWKS endpoint' if verifier.jwks_uri else 'static key',
            verifier.algorithm,
        )
        return verifier

    # provider == 'simple_token'. SimpleTokenVerifier raises ValueError when the
    # token is missing/empty; translate that into a ConfigurationError at this
    # configuration boundary so the server exits 78 (EX_CONFIG) per its documented
    # exit-code contract instead of the generic exit 1.
    try:
        token_verifier = SimpleTokenVerifier()
    except ValueError as e:
        raise ConfigurationError(str(e)) from e
    logger.info('Authentication: SimpleTokenVerifier (bearer token)')
    return token_verifier
