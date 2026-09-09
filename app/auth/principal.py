"""Per-request principal resolution from verified access-token claims.

This module is the single place that turns FastMCP's per-request access token
into an application-level identity. Tool code never inspects raw JWT claims:
it asks for a :class:`RequestPrincipal` and receives the caller's stable
principal id plus normalized group and role sets, extracted via the
IdP-agnostic helpers in :mod:`app.auth.claims`.

Resolution returns None when the request carries no verified token (stdio
transport, or HTTP with MCP_AUTH_PROVIDER=none). Mapping that no-token case
to the configured default principal is access-control policy, not token
plumbing, and lives in :func:`app.auth.access.resolve_effective_principal`.
"""

from dataclasses import dataclass

from fastmcp.server.dependencies import get_access_token

from app.auth.claims import extract_groups
from app.auth.claims import extract_roles
from app.settings import get_settings


@dataclass(frozen=True, slots=True)
class RequestPrincipal:
    """Verified caller identity for one request.

    Attributes:
        principal_id: Stable caller identifier -- the JWT ``sub`` claim when
            present, else the access token's client id (the simple_token
            provider issues tokens with empty claims and a configured
            client id).
        groups: Normalized group memberships from the configured groups claim.
        roles: Normalized roles from the configured roles claim.
    """

    principal_id: str
    groups: frozenset[str]
    roles: frozenset[str]


def resolve_request_principal() -> RequestPrincipal | None:
    """Resolve the verified principal for the current request.

    Returns:
        The caller's principal when the request carries a verified access
        token, or None when no token is present (stdio transport, or auth
        disabled).
    """
    token = get_access_token()
    if token is None:
        return None

    claims = token.claims
    subject = claims.get('sub')
    principal_id = subject if isinstance(subject, str) and subject else token.client_id

    auth_settings = get_settings().auth
    return RequestPrincipal(
        principal_id=principal_id,
        groups=frozenset(extract_groups(claims, auth_settings.groups_claim)),
        roles=frozenset(extract_roles(claims, auth_settings.roles_claim)),
    )
