"""Authentication and access-control settings, and the principal-id grammar the access-control migration shares."""

import re
from typing import Literal

from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator

from app.settings.base import CommonSettings

# A principal id is safe to interpolate into a quoted SQL literal only when it
# cannot break out of the quotes: the access-control migration embeds the
# configured default principal as the ``ADD COLUMN owner_id ... DEFAULT '<value>'``
# backfill literal on both backends. Public because the migration re-checks the
# value defensively right before interpolation, against this same definition.
SAFE_PRINCIPAL_ID_PATTERN = re.compile(r'[A-Za-z0-9._@:+-]{1,128}')


class AuthSettings(CommonSettings):
    """Authentication settings for HTTP transport.

    Configures the authentication provider passed to FastMCP's auth= parameter.
    Supported providers: 'none' (default), 'simple_token', 'jwt'.

    JWT provider configuration is intentionally NOT cross-validated here:
    auth is architecturally inert on stdio transport (create_auth_provider()
    is only called for HTTP transports), so requirements like "exactly one of
    public key / JWKS URI" are enforced at provider construction in app.auth,
    mirroring the simple_token precedent (MCP_AUTH_TOKEN presence is checked
    in SimpleTokenVerifier, not here). A pydantic validator would crash stdio
    startups over configuration that is never used.
    """

    provider: Literal['none', 'simple_token', 'jwt'] = Field(
        default='none',
        alias='MCP_AUTH_PROVIDER',
        description=(
            'Authentication provider: '
            'none (no auth, default), '
            'simple_token (bearer token), '
            'jwt (IdP-issued JWT verification)'
        ),
    )
    auth_token: SecretStr | None = Field(
        default=None,
        alias='MCP_AUTH_TOKEN',
        description='Bearer token for HTTP authentication (required when MCP_AUTH_PROVIDER=simple_token)',
    )
    auth_client_id: str = Field(
        default='mcp-client',
        alias='MCP_AUTH_CLIENT_ID',
        description='Client ID to assign to authenticated requests (used with simple_token)',
    )
    jwt_public_key: SecretStr | None = Field(
        default=None,
        alias='MCP_AUTH_JWT_PUBLIC_KEY',
        description='PEM-encoded public key (asymmetric algorithms) or shared secret (HS* algorithms) '
                    'for JWT verification. Mutually exclusive with MCP_AUTH_JWT_JWKS_URI; '
                    'exactly one is required when MCP_AUTH_PROVIDER=jwt',
    )
    jwt_jwks_uri: str | None = Field(
        default=None,
        alias='MCP_AUTH_JWT_JWKS_URI',
        description='JWKS endpoint URI for JWT verification with automatic key rotation. '
                    'Mutually exclusive with MCP_AUTH_JWT_PUBLIC_KEY; '
                    'exactly one is required when MCP_AUTH_PROVIDER=jwt',
    )
    jwt_issuer: str | None = Field(
        default=None,
        alias='MCP_AUTH_JWT_ISSUER',
        description='Expected issuer (iss) claim value for JWT verification. '
                    'Unset skips issuer validation',
    )
    jwt_audience: str | None = Field(
        default=None,
        alias='MCP_AUTH_JWT_AUDIENCE',
        description='Expected audience (aud) claim value for JWT verification. '
                    'Unset skips audience validation',
    )
    jwt_algorithm: str = Field(
        default='RS256',
        alias='MCP_AUTH_JWT_ALGORITHM',
        description='JWT signing algorithm to accept. '
                    'Supported: HS256/384/512, RS256/384/512, ES256/384/512, PS256/384/512, EdDSA, Ed25519, Ed448',
    )
    groups_claim: str = Field(
        default='groups',
        alias='MCP_AUTH_GROUPS_CLAIM',
        description='Claim carrying the caller group memberships (used with jwt). '
                    'Supports dotted paths (realm_access.roles) and full-URL claim keys '
                    '(https://example.com/groups)',
    )
    roles_claim: str = Field(
        default='roles',
        alias='MCP_AUTH_ROLES_CLAIM',
        description='Claim carrying the caller roles (used with jwt). '
                    'Supports dotted paths and full-URL claim keys',
    )


class AccessControlSettings(CommonSettings):
    """Access-control policy for server-stamped entry ownership and visibility.

    Every stored entry carries a server-stamped ``owner_id`` (the verified
    request principal, or the configured default principal when the request
    carries no verified token) and a ``visibility`` ('private', 'shared', or
    'public'). These settings define the default principal, the default
    visibility for writes that do not specify one, whether the author's group
    memberships become read grants automatically, and which role (if any) is
    required to publish entries as 'public'.
    """

    default_principal: str = Field(
        default='local',
        alias='ACCESS_CONTROL_DEFAULT_PRINCIPAL',
        description='Principal id stamped as owner_id when a request carries no verified '
                    'access token (stdio transport, or MCP_AUTH_PROVIDER none/simple_token '
                    'without a sub claim source). Also the owner backfilled onto rows that '
                    'predate the access-control columns',
    )
    default_visibility: Literal['private', 'shared', 'public'] = Field(
        default='private',
        alias='ACCESS_CONTROL_DEFAULT_VISIBILITY',
        description='Visibility stamped on stored entries when the caller does not '
                    'specify one: private (owner only), shared (owner + explicit grants), '
                    'public (any principal)',
    )
    default_group_grants: Literal['none', 'author_groups'] = Field(
        default='none',
        alias='ACCESS_CONTROL_DEFAULT_GROUP_GRANTS',
        description='Automatic group read grants on newly inserted entries: none (explicit '
                    'shares only, default) or author_groups (every group of the writing '
                    'principal receives a read grant)',
    )
    publish_role: str | None = Field(
        default=None,
        alias='ACCESS_CONTROL_PUBLISH_ROLE',
        description="Role required to set visibility 'public'. Unset (default) lets any "
                    'owner publish; set to a role name to restrict publishing to callers '
                    'whose verified roles claim carries that role',
    )

    @field_validator('default_principal')
    @classmethod
    def _validate_default_principal(cls, value: str) -> str:
        """Restrict the default principal to a DDL-safe identifier.

        The value is stamped into rows AND interpolated as a literal DEFAULT into
        the access-control migration's ``ALTER TABLE ... ADD COLUMN`` backfill on
        both backends, so it must not be able to break out of a quoted SQL
        literal. A conservative character set (no quotes, whitespace, or
        backslashes) makes the interpolation safe by construction.

        Returns:
            The validated principal id.

        Raises:
            ValueError: If the value falls outside the safe character set or
                length bound.
        """
        if not SAFE_PRINCIPAL_ID_PATTERN.fullmatch(value):
            raise ValueError(
                'ACCESS_CONTROL_DEFAULT_PRINCIPAL must be 1-128 characters from '
                'A-Z a-z 0-9 . _ @ : + - (no quotes, spaces, or backslashes)',
            )
        return value
