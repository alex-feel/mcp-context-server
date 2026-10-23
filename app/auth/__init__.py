"""Authentication providers for mcp-context-server.

This package provides authentication mechanisms for securing HTTP transports.
Configure via MCP_AUTH_PROVIDER environment variable:
- none: No authentication (default)
- simple_token: Bearer token authentication (requires MCP_AUTH_TOKEN)
- jwt: IdP-issued JWT verification (requires exactly one of
  MCP_AUTH_JWT_PUBLIC_KEY or MCP_AUTH_JWT_JWKS_URI); on a JWKS deployment,
  MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY adds an operator-held static-key verifier
  that accepts only ACCESS_CONTROL_ANONYMOUS_PRINCIPAL, composed with the JWKS
  verifier through fastmcp's MultiAuth

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
    - app.auth.anonymous: AnonymousSubjectVerifier, the anonymous principal's
      static-key verifier on a JWKS deployment
    - app.auth.claims: IdP-agnostic claim extraction helpers
    - app.auth.principal: per-request RequestPrincipal resolution
    - app.auth.access: effective-principal, access-scope and write-scope
      resolution and the publish gate (the scope types live in app.access_scope)
    - app.settings.auth.AuthSettings: Authentication configuration
    - FastMCP authentication docs: https://gofastmcp.com/servers/auth
"""


import logging
from typing import TYPE_CHECKING

from app.auth.access import resolve_access_scope
from app.auth.access import resolve_effective_principal
from app.auth.access import resolve_write_scope
from app.auth.access import visibility_denied_reason
from app.auth.principal import RequestPrincipal
from app.auth.principal import resolve_request_principal
from app.auth.simple_token import SimpleTokenVerifier
from app.errors import ConfigurationError
from app.settings import get_settings
from app.settings.auth import AccessControlSettings
from app.settings.auth import AuthSettings

if TYPE_CHECKING:
    from fastmcp.server.auth import AuthProvider
    from fastmcp.server.auth.providers.jwt import JWTVerifier

    from app.auth.anonymous import AnonymousSubjectVerifier

logger = logging.getLogger(__name__)

__all__ = [
    'AuthSettings',
    'RequestPrincipal',
    'SimpleTokenVerifier',
    'create_auth_provider',
    'resolve_access_scope',
    'resolve_effective_principal',
    'resolve_request_principal',
    'resolve_write_scope',
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
    # Deferred import: the JWT provider pulls in joserfc/cryptography, which
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


def _create_anonymous_verifier(
    auth_settings: AuthSettings,
    access: AccessControlSettings,
) -> 'AnonymousSubjectVerifier | None':
    """Construct the anonymous principal's static-key verifier when its key is configured.

    The verifier pins the primary verifier's issuer and audience and accepts
    only ACCESS_CONTROL_ANONYMOUS_PRINCIPAL. It exists only beside a JWKS
    verifier: a static-key deployment signs the anonymous token with the
    counterpart of its own MCP_AUTH_JWT_PUBLIC_KEY.

    Args:
        auth_settings: The resolved authentication settings.
        access: The resolved access-control settings, which name the anonymous principal.

    Returns:
        The configured verifier, or None when MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY
        is unset, empty or whitespace-only.

    Raises:
        ConfigurationError: When the key is set without MCP_AUTH_JWT_JWKS_URI
            or without ACCESS_CONTROL_ANONYMOUS_PRINCIPAL, or when the JWT
            verifier rejects the key and algorithm (an unsupported algorithm,
            or a PEM key with an HS* algorithm).
    """
    secret = auth_settings.anonymous_jwt_public_key
    public_key = (secret.get_secret_value() if secret else '').strip()
    if not public_key:
        return None
    if not (auth_settings.jwt_jwks_uri or '').strip():
        raise ConfigurationError(
            'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY applies to MCP_AUTH_JWT_JWKS_URI deployments; '
            'with MCP_AUTH_JWT_PUBLIC_KEY, sign the anonymous token with that key',
        )
    if access.anonymous_principal is None:
        raise ConfigurationError(
            'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY requires ACCESS_CONTROL_ANONYMOUS_PRINCIPAL, '
            'the only subject the anonymous verifier accepts',
        )

    # Deferred import, like _create_jwt_verifier's: the verifier pulls in joserfc/cryptography.
    from app.auth.anonymous import AnonymousSubjectVerifier

    try:
        return AnonymousSubjectVerifier(
            subject=access.anonymous_principal,
            public_key=public_key,
            algorithm=auth_settings.anonymous_jwt_algorithm,
            issuer=auth_settings.jwt_issuer,
            audience=auth_settings.jwt_audience,
        )
    except ValueError as e:
        raise ConfigurationError(str(e)) from e


def create_auth_provider() -> 'AuthProvider | None':
    """Create an authentication provider based on MCP_AUTH_PROVIDER setting.

    With MCP_AUTH_PROVIDER=jwt and MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY set, the
    provider is a fastmcp MultiAuth that tries the anonymous static-key
    verifier first and the JWKS verifier second: the local signature check
    fails fast on an identity-provider token, while JWKS-first would send
    every anonymous request through a JWKS key lookup for a key id the
    endpoint never publishes.

    Returns:
        An auth provider instance for FastMCP's auth= parameter, or None for no auth.

    Raises:
        ConfigurationError: When MCP_AUTH_PROVIDER=simple_token but MCP_AUTH_TOKEN
            is missing or empty, or when MCP_AUTH_PROVIDER=jwt with an invalid key
            configuration (neither or both of MCP_AUTH_JWT_PUBLIC_KEY /
            MCP_AUTH_JWT_JWKS_URI, an unsupported algorithm, a symmetric
            algorithm combined with a JWKS URI or PEM public key, or an
            anonymous verifier key without a JWKS URI or without the anonymous
            principal). These are startup misconfigurations (missing or
            contradictory environment variables), so they are classified as
            ConfigurationError (exit 78) rather than letting the underlying
            ValueError reach the generic handler and exit 1 -- exit 78 tells a
            supervisor NOT to crash-loop on a restart that cannot fix the config.
    """
    settings = get_settings()
    provider = settings.auth.provider

    if provider == 'none':
        logger.info('Authentication: disabled (MCP_AUTH_PROVIDER=none)')
        return None

    if provider == 'jwt':
        verifier = _create_jwt_verifier(settings.auth)
        anonymous = _create_anonymous_verifier(settings.auth, settings.access_control)
        if anonymous is not None:
            from fastmcp.server.auth import MultiAuth

            logger.info(
                'Authentication: JWTVerifier (JWKS endpoint, algorithm %s) composed with the anonymous '
                'static-key verifier (algorithm %s, subject %s)',
                verifier.algorithm,
                anonymous.algorithm,
                anonymous.subject,
            )
            return MultiAuth(verifiers={'anonymous': anonymous, 'jwt': verifier})
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
