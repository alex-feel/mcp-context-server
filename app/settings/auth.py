"""Authentication and access-control settings, with the principal-id and PostgreSQL role-name grammars they enforce."""

import re
from typing import Literal
from typing import Self

from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator
from pydantic import model_validator

from app.settings.base import CommonSettings

# A principal id is safe to interpolate into a quoted SQL literal only when it
# cannot break out of the quotes: the access-control migration embeds the
# configured default principal as the ``ADD COLUMN owner_id ... DEFAULT '<value>'``
# backfill literal on both backends. Public because the migration re-checks the
# value defensively right before interpolation, against this same definition.
SAFE_PRINCIPAL_ID_PATTERN = re.compile(r'[A-Za-z0-9._@:+-]{1,128}')

# A PostgreSQL role name that needs no case folding and fits the 63-byte
# identifier limit: lowercase ASCII letters, digits and underscores, not starting
# with a digit. The configured policy role is named in generated policy DDL, in
# privilege checks and in the remediation statements startup prints, so it must
# mean the same role everywhere it appears.
SAFE_PG_ROLE_PATTERN = r'^[a-z_][a-z0-9_]{0,62}$'


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
        description='Client ID that identifies simple_token requests to FastMCP. Not an owner '
                    'identity: simple_token requests own entries as ACCESS_CONTROL_DEFAULT_PRINCIPAL',
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
    """Access-control policy for server-stamped entry ownership and visibility, and its database enforcement.

    Every stored entry carries a server-stamped ``owner_id`` (the verified jwt
    principal, or the configured default principal for stdio, ``none`` and
    ``simple_token`` requests) and a ``visibility``: 'private' (readable by the
    owner and every grantee) or 'public' (readable by everyone). These settings
    define the default principal, the default visibility for writes that do not
    specify one, whether the author's group memberships become read grants
    automatically, which role (if any) is required to publish entries as
    'public', and which jwt subject (if any) is the read-only principal that
    every write tool refuses.

    The application enforces the model on both backends. On PostgreSQL,
    ``ENABLE_POSTGRESQL_RLS`` additionally enforces it in the database through
    row-level security policies that name ``POSTGRESQL_RLS_POLICY_ROLE``, the
    NOLOGIN role the request login role is a member of.
    """

    default_principal: str = Field(
        default='local',
        alias='ACCESS_CONTROL_DEFAULT_PRINCIPAL',
        description='Principal id stamped as owner_id for every request without a jwt '
                    'identity: stdio transport and MCP_AUTH_PROVIDER none or simple_token. '
                    'Also the owner backfilled onto rows that predate the access-control columns',
    )
    default_visibility: Literal['private', 'public'] = Field(
        default='private',
        alias='ACCESS_CONTROL_DEFAULT_VISIBILITY',
        description='Visibility stamped on stored entries when the caller does not '
                    'specify one: private (owner plus every grantee), public (everyone)',
    )
    default_group_grants: Literal['none', 'author_groups'] = Field(
        default='none',
        alias='ACCESS_CONTROL_DEFAULT_GROUP_GRANTS',
        description='Automatic group read grants on newly inserted entries: none (no '
                    'automatic grants, default) or author_groups (every group of the writing '
                    'principal receives a read grant, effective under the default private '
                    'visibility)',
    )
    publish_role: str | None = Field(
        default=None,
        alias='ACCESS_CONTROL_PUBLISH_ROLE',
        description="Role required to set visibility 'public'. Unset (default) lets any "
                    'owner publish; set to a role name to restrict publishing to callers '
                    'whose verified roles claim carries that role',
    )
    anonymous_principal: str | None = Field(
        default=None,
        alias='ACCESS_CONTROL_ANONYMOUS_PRINCIPAL',
        description='Principal id of the read-only jwt subject, matched verbatim against the token sub claim: '
                    'every write tool refuses it, its groups and roles are discarded, and its token must carry '
                    'exp. Unset (default) makes no principal read-only, and a value that differs from the minted '
                    'sub leaves that token an ordinary writable principal, so the startup log names the value. '
                    'Must differ from ACCESS_CONTROL_DEFAULT_PRINCIPAL',
    )
    postgresql_rls_enabled: bool = Field(
        default=False,
        alias='ENABLE_POSTGRESQL_RLS',
        description='Also enforce the access model in PostgreSQL through row-level security. '
                    'Requires STORAGE_BACKEND=postgresql and POSTGRESQL_MIGRATION_USER naming the '
                    'role that owns the tables; requests must connect as a different role that is '
                    'a member of POSTGRESQL_RLS_POLICY_ROLE and neither superuser nor BYPASSRLS',
    )
    postgresql_rls_policy_role: str = Field(
        default='mcp_context_request',
        alias='POSTGRESQL_RLS_POLICY_ROLE',
        pattern=SAFE_PG_ROLE_PATTERN,
        description='NOLOGIN PostgreSQL role the row-level security policies apply to. The operator '
                    'creates it and grants it to the request login role; the server grants it the '
                    'table privileges. 1-63 characters from a-z 0-9 _, not starting with a digit',
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

    @field_validator('anonymous_principal')
    @classmethod
    def _validate_anonymous_principal(cls, value: str | None) -> str | None:
        """Refuse a read-only principal that no jwt subject could ever equal.

        The value is compared verbatim with the token's ``sub`` claim, so any
        charset and length is valid, but an empty, whitespace-only or
        whitespace-padded value would silently bind nobody.

        Returns:
            The validated principal id, or None when unset.

        Raises:
            ValueError: If the value is empty, whitespace-only, or has leading or
                trailing whitespace.
        """
        if value is not None and (not value.strip() or value != value.strip()):
            raise ValueError(
                'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL must be a principal id without leading or trailing '
                'whitespace, matching the read-only token sub claim verbatim; unset it to make no '
                'principal read-only',
            )
        return value

    @model_validator(mode='after')
    def _refuse_read_only_default_principal(self) -> Self:
        """Refuse a read-only principal equal to the default principal.

        The default principal owns every stdio, ``none`` and ``simple_token``
        request and every backfilled row, so making it read-only would refuse
        those writes and hand those rows to the holders of the read-only token.

        Returns:
            The validated settings.

        Raises:
            ValueError: If the read-only principal equals the default principal.
        """
        if self.anonymous_principal is not None and self.anonymous_principal == self.default_principal:
            raise ValueError(
                'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL must differ from ACCESS_CONTROL_DEFAULT_PRINCIPAL, '
                'which owns every request without a jwt identity',
            )
        return self
