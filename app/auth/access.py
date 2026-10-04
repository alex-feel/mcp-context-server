"""Access-control policy helpers for the tool layer.

This module turns the transport-level principal (:mod:`app.auth.principal`)
into the identity the storage layer stamps on new rows and the scope that
limits which rows a request reaches, and holds the publish-gate rule. It is the
single place that applies ``AccessControlSettings`` policy:

- :func:`resolve_effective_principal` maps every request without a jwt
  identity (stdio transport, and HTTP with ``MCP_AUTH_PROVIDER`` ``none`` or
  ``simple_token``) to the configured default principal with empty groups and
  roles, so every request -- authenticated or not -- resolves to exactly one
  owner identity.
- :func:`resolve_access_scope` returns that same identity as the
  :class:`~app.access_scope.AccessScope` that limits which context entries the
  request may read and write.
- :func:`visibility_denied_reason` enforces ``ACCESS_CONTROL_PUBLISH_ROLE``:
  when the setting names a role, only callers whose verified roles claim
  carries it may set visibility 'public'.
"""

from app.access_scope import AccessScope
from app.auth.principal import RequestPrincipal
from app.auth.principal import resolve_request_principal
from app.settings import get_settings


def resolve_effective_principal() -> RequestPrincipal:
    """Resolve the identity to stamp as ``owner_id`` for the current request.

    Returns:
        The verified request principal when the jwt provider verified the
        request's access token, else the configured
        ``ACCESS_CONTROL_DEFAULT_PRINCIPAL`` with empty groups and roles.
    """
    principal = resolve_request_principal()
    if principal is not None:
        return principal
    access = get_settings().access_control
    return RequestPrincipal(
        principal_id=access.default_principal,
        groups=frozenset(),
        roles=frozenset(),
    )


def resolve_access_scope() -> AccessScope:
    """Resolve the scope the current request reads and writes context entries as.

    Returns:
        The effective principal's id and groups: the verified jwt principal's,
        or ``ACCESS_CONTROL_DEFAULT_PRINCIPAL`` with no groups for every other
        request.
    """
    return resolve_effective_principal().access_scope()


def visibility_denied_reason(visibility: str, principal: RequestPrincipal) -> str | None:
    """Check the publish gate for a requested visibility value.

    Only 'public' is gated: 'private' is always available to the writer (for a
    store) or the owner (for an update; ownership is checked by the caller, not
    here). When ``ACCESS_CONTROL_PUBLISH_ROLE`` is unset, any owner may
    publish.

    Args:
        visibility: The requested visibility value.
        principal: The effective principal making the request.

    Returns:
        A human-readable denial reason when the principal may not set the
        requested visibility, or None when it is allowed.
    """
    if visibility != 'public':
        return None
    publish_role = get_settings().access_control.publish_role
    if publish_role is None or publish_role in principal.roles:
        return None
    return (
        f"Setting visibility 'public' requires the '{publish_role}' role "
        '(ACCESS_CONTROL_PUBLISH_ROLE), which the caller does not have'
    )
