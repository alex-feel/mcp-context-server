"""Access-control policy helpers for the tool layer.

This module turns the transport-level principal (:mod:`app.auth.principal`)
into the identity the storage layer stamps on new rows and the scope that
limits which rows a request reaches, and holds the publish-gate and read-only
rules. It is the single place that applies ``AccessControlSettings`` policy:

- :func:`resolve_effective_principal` maps every request without a jwt
  identity (stdio transport, and HTTP with ``MCP_AUTH_PROVIDER`` ``none`` or
  ``simple_token``) to the configured default principal with empty groups and
  roles, so every request -- authenticated or not -- resolves to exactly one
  owner identity. The read-only principal named by
  ``ACCESS_CONTROL_ANONYMOUS_PRINCIPAL`` resolves with its groups and roles
  discarded, and only on a token that carries an expiry.
- :func:`resolve_access_scope` returns that same identity as the
  :class:`~app.access_scope.AccessScope` that limits which context entries the
  request may read and write. It asserts neither the publish permission nor the
  write permission, so it serves every tool that stores nothing.
- :func:`may_publish` holds the ``ACCESS_CONTROL_PUBLISH_ROLE`` rule: when the
  setting names a role, only callers whose verified roles claim carries it may
  set visibility 'public'; the read-only principal never may.
  :func:`visibility_denied_reason` enforces it in the write tools, and
  :func:`resolve_write_scope` carries its answer in the scope the write tools
  run under, together with the write permission every principal but the
  read-only one holds, which the database checks under row-level security.
- :func:`reject_read_only_principal` is the gate every write tool calls first,
  before any argument check, repository access or generation, refusing the
  read-only principal; :func:`log_read_only_principal` names that principal at
  startup, because an unset or mistyped setting leaves its token writable.
"""

import logging
from dataclasses import replace

from fastmcp.exceptions import ToolError
from fastmcp.server.dependencies import get_access_token

from app.access_scope import AccessScope
from app.auth.principal import RequestPrincipal
from app.auth.principal import resolve_request_principal
from app.settings import get_settings
from app.settings.auth import AccessControlSettings

logger = logging.getLogger(__name__)


def is_read_only(principal: RequestPrincipal) -> bool:
    """Decide whether a principal is the configured read-only principal.

    Args:
        principal: The principal making the request.

    Returns:
        True when ``ACCESS_CONTROL_ANONYMOUS_PRINCIPAL`` is set and equals the
        principal's id verbatim, False otherwise (always False when unset).
    """
    anonymous_principal = get_settings().access_control.anonymous_principal
    return anonymous_principal is not None and anonymous_principal == principal.principal_id


def resolve_effective_principal() -> RequestPrincipal:
    """Resolve the identity to stamp as ``owner_id`` for the current request.

    Returns:
        The verified request principal when the jwt provider verified the
        request's access token, else the configured
        ``ACCESS_CONTROL_DEFAULT_PRINCIPAL`` with empty groups and roles. The
        read-only principal is returned with empty groups and roles, so a token
        minted with either carries neither group access nor the publish role.

    Raises:
        ToolError: If the request is the read-only principal's and its token
            carries no expiry, which would keep a shared credential valid forever.
    """
    principal = resolve_request_principal()
    if principal is None:
        access = get_settings().access_control
        return RequestPrincipal(
            principal_id=access.default_principal,
            groups=frozenset(),
            roles=frozenset(),
        )
    if not is_read_only(principal):
        return principal
    token = get_access_token()
    if token is None or token.expires_at is None:
        raise ToolError("The anonymous principal's token carries no expiry (exp); mint it with a bounded exp")
    return RequestPrincipal(principal_id=principal.principal_id, groups=frozenset(), roles=frozenset())


def resolve_access_scope() -> AccessScope:
    """Resolve the scope the current request reads and writes context entries as.

    The scope asserts no publish permission; the write tools, which may set
    visibility, build theirs with :func:`resolve_write_scope` instead.

    Returns:
        The effective principal's id and groups: the verified jwt principal's,
        or ``ACCESS_CONTROL_DEFAULT_PRINCIPAL`` with no groups for every other
        request.
    """
    return resolve_effective_principal().access_scope()


def resolve_write_scope(principal: RequestPrincipal) -> AccessScope:
    """Return the scope a write tool reads and writes context entries as.

    Args:
        principal: The effective principal making the request.

    Returns:
        The principal's id and groups, with ``may_publish`` set to whether the
        principal may set visibility 'public' and ``may_write`` to whether it may
        store entries at all, which only the read-only principal may not.
    """
    return replace(
        principal.access_scope(), may_publish=may_publish(principal), may_write=not is_read_only(principal),
    )


def may_publish(principal: RequestPrincipal) -> bool:
    """Decide whether a principal may set visibility 'public'.

    Args:
        principal: The effective principal making the request.

    Returns:
        False for the read-only principal. Otherwise True when
        ``ACCESS_CONTROL_PUBLISH_ROLE`` is unset or names a role the principal's
        verified roles carry, False otherwise.
    """
    if is_read_only(principal):
        return False
    publish_role = get_settings().access_control.publish_role
    return publish_role is None or publish_role in principal.roles


def reject_read_only_principal(principal: RequestPrincipal, *, tool: str) -> None:
    """Refuse the read-only principal a write tool.

    Every write tool calls this right after resolving its principal, before any
    argument check, repository access, id resolution or generation, so the
    refusal depends on no row and no argument.

    Args:
        principal: The effective principal making the request.
        tool: The name of the tool being called.

    Raises:
        ToolError: If the principal is the read-only principal.
    """
    if is_read_only(principal):
        raise ToolError(f'The anonymous principal is read-only: {tool} is not permitted')


def log_read_only_principal(access: AccessControlSettings) -> None:
    """Log at startup which principal, if any, is read-only.

    The read-only setting fails open: unset, or set to a value that differs from
    the minted token's subject, the token is an ordinary writable principal.
    Naming the configured value at every start keeps that visible.

    Args:
        access: The access-control settings the server runs with.
    """
    configured = access.anonymous_principal
    logger.info('Read-only principal: %s', 'none' if configured is None else configured)


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
    if visibility != 'public' or may_publish(principal):
        return None
    publish_role = get_settings().access_control.publish_role
    return (
        f"Setting visibility 'public' requires the '{publish_role}' role "
        '(ACCESS_CONTROL_PUBLISH_ROLE), which the caller does not have'
    )
