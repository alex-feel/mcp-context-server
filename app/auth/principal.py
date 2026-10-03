"""Per-request principal resolution from verified access-token claims.

This module is the single place that turns FastMCP's per-request access token
into an application-level identity. Tool code never inspects raw JWT claims:
it asks for a :class:`RequestPrincipal` and receives the caller's stable
principal id plus normalized group and role sets, extracted via the
IdP-agnostic helpers in :mod:`app.auth.claims`. The principal's id and groups
form the :class:`app.access_scope.AccessScope` its reads and writes run under;
its roles only gate publishing.

Only the jwt provider carries a caller identity. Resolution returns None for
every other request: stdio transport, HTTP with MCP_AUTH_PROVIDER=none, and
HTTP with MCP_AUTH_PROVIDER=simple_token, whose bearer token proves access to
the server but names no principal (its client id identifies the client to
FastMCP, not an owner). Mapping those requests to the configured default
principal is access-control policy, not token plumbing, and lives in
:func:`app.auth.access.resolve_effective_principal`.
"""

from dataclasses import dataclass

from fastmcp.server.dependencies import get_access_token

from app.access_scope import AccessScope
from app.auth.claims import extract_groups
from app.auth.claims import extract_roles
from app.settings import get_settings


@dataclass(frozen=True, slots=True)
class RequestPrincipal:
    """Verified caller identity for one request.

    Attributes:
        principal_id: Stable caller identifier -- the JWT ``sub`` claim when
            present, else the access token's client id.
        groups: Normalized group memberships from the configured groups claim.
        roles: Normalized roles from the configured roles claim.
    """

    principal_id: str
    groups: frozenset[str]
    roles: frozenset[str]

    def access_scope(self) -> AccessScope:
        """Return the scope this principal reads and writes context entries as.

        Roles are not part of the scope: they only gate publishing.

        Returns:
            The principal id and groups as an :class:`AccessScope`.
        """
        return AccessScope(principal_id=self.principal_id, groups=self.groups)


def resolve_request_principal() -> RequestPrincipal | None:
    """Resolve the verified principal for the current request.

    Returns:
        The caller's principal when the jwt provider verified the request's
        access token, or None otherwise (stdio transport, auth disabled, or the
        simple_token provider, whose tokens carry no identity).
    """
    auth_settings = get_settings().auth
    if auth_settings.provider != 'jwt':
        return None

    token = get_access_token()
    if token is None:
        return None

    claims = token.claims
    subject = claims.get('sub')
    principal_id = subject if isinstance(subject, str) and subject else token.client_id

    return RequestPrincipal(
        principal_id=principal_id,
        groups=frozenset(extract_groups(claims, auth_settings.groups_claim)),
        roles=frozenset(extract_roles(claims, auth_settings.roles_claim)),
    )
