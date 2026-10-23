"""The sharing tools: share_context, unshare_context and list_context_grants.

Only an entry's owner grants another principal access to it, removes such a grant,
or lists its grants. An entry the caller may not read is reported exactly like an
absent one, and an entry the caller may read but does not own -- through a read or
write grant, or because it is public -- is refused explicitly. A write grant lets its
holder edit an entry, never extend access to it.

Each permission is its own grant row: a share stores at most one row and reports
whether it did, and an unshare removes exactly the row it names and reports the
permissions the same principal still holds on the entry. Grants change neither the
entry's version nor its update time, so they never conflict with a concurrent update.
"""

import asyncio
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Annotated
from typing import Literal

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.access_scope import AccessScope
from app.auth import resolve_access_scope
from app.auth import resolve_effective_principal
from app.auth import resolve_write_scope
from app.auth.access import reject_read_only_principal
from app.errors import ControlFlowError
from app.errors import format_exception_message
from app.ids import resolve_or_normalize_id
from app.models import MAX_GRANT_PRINCIPAL_ID_LENGTH
from app.repositories.grant_repository import GrantPermission
from app.settings import get_settings
from app.startup import ensure_repositories
from app.tools._transactions import EntryNotAuthorizedError
from app.tools._transactions import EntryNotFoundError
from app.tools._transactions import is_connection_error
from app.tools._validation import validate_grant_principal_id
from app.types import ContextGrantDict
from app.types import ListContextGrantsResultDict
from app.types import ShareContextResultDict
from app.types import UnshareContextResultDict

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer

logger = logging.getLogger(__name__)

_CONTEXT_ID_DESCRIPTION = 'Context entry id (32/36-char UUID or 8-31 char hex prefix) of an entry you own'
_PRINCIPAL_TYPE_DESCRIPTION = "'user' for one principal (a token subject) or 'group' for every member of a group"
_PRINCIPAL_ID_DESCRIPTION = (
    f'The user or group id, matched verbatim: case-sensitive, 1 to {MAX_GRANT_PRINCIPAL_ID_LENGTH} '
    'characters, no leading or trailing whitespace'
)

type GrantChangeAction = Literal['share', 'unshare']


class OwnerShareError(ControlFlowError):
    """Internal control-flow signal: the owner named itself as a user grantee.

    The owner already holds every permission on its entry, so such a grant would
    change nothing. The refusal comes after the ownership check, so a caller that
    does not own the entry gets the not-authorized refusal instead, and before the
    insert, so nothing is written. Its message is the exact text ``share_context``
    reports.
    """

    def __init__(self, context_id: str) -> None:
        super().__init__(f'Cannot share context entry with ID {context_id} with its owner')
        self.context_id = context_id


async def _run_owned_grant_change[T](
    repos: 'RepositoryContainer',
    context_id: str,
    *,
    scope: AccessScope,
    action: GrantChangeAction,
    change: Callable[['TransactionContext'], Awaitable[T]],
    max_retries: int = 2,
) -> T:
    """Run one grant change on an entry the caller owns, in one transaction.

    The entry is probed under the caller's scope inside the transaction: an entry
    the caller may not read is not found, and one it may read but does not own is
    refused, before ``change`` runs. Every grant statement also carries the owner
    condition itself, so the change stays owner-only on its own. A transient fault
    (a dropped connection, a SQLite lock held by another process) rolls the whole
    transaction back and is retried with exponential backoff; both changes are
    idempotent, so a retry after an unknown outcome reports the state it finds.
    Refusals are control flow: never retried and never charged to the circuit
    breaker.

    Args:
        repos: Repository container.
        context_id: The resolved id of the entry.
        scope: The caller's scope.
        action: The change, which names the refusal a non-owner gets.
        change: The grant statements, run on the open transaction.
        max_retries: Maximum transient-fault retries.

    Returns:
        What ``change`` returned.

    Raises:
        EntryNotFoundError: If the caller may not read the entry or it does not exist.
        EntryNotAuthorizedError: If the caller may read the entry but does not own it.
    """
    attempt = 0
    while True:
        try:
            async with repos.context.backend.begin_transaction(access_scope=scope) as txn:
                access = await repos.context.probe_ids([context_id], scope=scope, txn=txn)
                if context_id not in access:
                    raise EntryNotFoundError(context_id)
                if not access[context_id].is_owner:
                    raise EntryNotAuthorizedError([context_id], action=action)
                return await change(txn)
        except Exception as exc:
            if is_connection_error(exc) and attempt < max_retries:
                delay = 0.5 * (2 ** attempt)
                attempt += 1
                logger.warning(
                    'Grant change failed with a transient error, retrying in %.1fs (attempt %d/%d): %s',
                    delay, attempt, max_retries, exc,
                )
                await asyncio.sleep(delay)
                continue
            raise


async def share_context(
    context_id: Annotated[str, Field(min_length=8, description=_CONTEXT_ID_DESCRIPTION)],
    principal_type: Annotated[Literal['user', 'group'], Field(description=_PRINCIPAL_TYPE_DESCRIPTION)],
    principal_id: Annotated[str, Field(description=_PRINCIPAL_ID_DESCRIPTION)],
    permission: Annotated[
        Literal['read', 'write'],
        Field(description="'read' to read the entry; 'write' also to edit its text, metadata, tags and images"),
    ],
) -> ShareContextResultDict:
    """Grant a user or a group read or write access to one entry you own.

    Only the entry's owner may share it; a write grant lets its holder edit the entry
    but never share it. A group grant reaches every member of the group. Sharing is
    idempotent: repeating a grant the entry already holds changes nothing and reports
    created=false. Read and write are separate grants, and a write grant also allows
    reading. Sharing leaves the entry's updated_at unchanged and never conflicts with
    a concurrent update of the entry.

    Errors: an entry you may not read is reported as "Context entry with ID {id} not
    found"; one you may read but do not own as "Not authorized to share context entry
    with ID {id}". You cannot share an entry with yourself as a user, and a user grant
    to the anonymous principal is refused: set the entry's visibility to 'public' to
    publish it instead.

    Returns:
        Dict with success, context_id, principal_type, principal_id, permission,
        created (false when the entry already held the grant) and message.

    Raises:
        ToolError: If the principal id is invalid, the entry is not found, the caller
            does not own it, the grant names the owner or the anonymous principal,
            or the share fails.
    """
    try:
        # The caller's identity comes first: the read-only principal is refused before
        # any argument check, repository access or id resolution.
        principal = resolve_effective_principal()
        reject_read_only_principal(principal, tool='share_context')
        scope = resolve_write_scope(principal)

        validate_grant_principal_id(principal_id)
        # No entry may ever be shared with the anonymous principal, so the refusal needs
        # no row: it comes before the id is resolved and names the id as passed.
        if principal_type == 'user' and principal_id == get_settings().access_control.anonymous_principal:
            raise ToolError(
                f'Cannot share context entry with ID {context_id} with the anonymous principal; '
                "set its visibility to 'public' to publish it",
            )

        repos = await ensure_repositories()
        try:
            resolved_id = await resolve_or_normalize_id(context_id, repos.context, scope=scope)
        except ValueError as exc:
            raise ToolError(f'Invalid context ID: {exc}') from exc

        async def _store(txn: 'TransactionContext') -> bool:
            if principal_type == 'user' and principal_id == scope.principal_id:
                raise OwnerShareError(resolved_id)
            return await repos.grants.store_grant(
                resolved_id, principal_type, principal_id, permission, scope=scope, txn=txn,
            )

        created = await _run_owned_grant_change(repos, resolved_id, scope=scope, action='share', change=_store)
        if created:
            message = f'Granted {permission} on context entry {resolved_id} to {principal_type} {principal_id}'
        else:
            message = f'{principal_type} {principal_id} already holds {permission} on context entry {resolved_id}'
        logger.info(message)
        return ShareContextResultDict(
            success=True,
            context_id=resolved_id,
            principal_type=principal_type,
            principal_id=principal_id,
            permission=permission,
            created=created,
            message=message,
        )
    except ToolError:
        raise
    except (EntryNotFoundError, EntryNotAuthorizedError, OwnerShareError) as exc:
        raise ToolError(format_exception_message(exc)) from exc
    except Exception as exc:
        logger.error(f'Error sharing context: {exc}')
        raise ToolError(f'Failed to share context: {format_exception_message(exc)}') from exc


async def unshare_context(
    context_id: Annotated[str, Field(min_length=8, description=_CONTEXT_ID_DESCRIPTION)],
    principal_type: Annotated[Literal['user', 'group'], Field(description=_PRINCIPAL_TYPE_DESCRIPTION)],
    principal_id: Annotated[str, Field(description=_PRINCIPAL_ID_DESCRIPTION)],
    permission: Annotated[Literal['read', 'write'], Field(description='The permission of the grant to remove')],
) -> UnshareContextResultDict:
    """Remove one grant from an entry you own.

    Only the entry's owner may unshare it. Exactly the named grant is removed: removing
    read leaves a write grant of the same principal in place, and a write grant also
    allows reading. remaining_permissions lists what the same principal type and id
    still hold on the entry; membership in a group that holds a grant can still give
    access. Removing a grant the entry does not hold changes nothing and reports
    removed=false. Unsharing leaves the entry's updated_at unchanged and never
    conflicts with a concurrent update of the entry.

    Errors: an entry you may not read is reported as "Context entry with ID {id} not
    found"; one you may read but do not own as "Not authorized to unshare context entry
    with ID {id}".

    Returns:
        Dict with success, context_id, principal_type, principal_id, permission,
        removed (false when the entry held no such grant), remaining_permissions and
        message.

    Raises:
        ToolError: If the principal id is invalid, the entry is not found, the caller
            does not own it, or the unshare fails.
    """
    try:
        # The caller's identity comes first: the read-only principal is refused before
        # any argument check, repository access or id resolution.
        principal = resolve_effective_principal()
        reject_read_only_principal(principal, tool='unshare_context')
        scope = resolve_write_scope(principal)

        validate_grant_principal_id(principal_id)

        repos = await ensure_repositories()
        try:
            resolved_id = await resolve_or_normalize_id(context_id, repos.context, scope=scope)
        except ValueError as exc:
            raise ToolError(f'Invalid context ID: {exc}') from exc

        async def _remove(txn: 'TransactionContext') -> tuple[bool, list[GrantPermission]]:
            removed = await repos.grants.delete_grant(
                resolved_id, principal_type, principal_id, permission, scope=scope, txn=txn,
            )
            grants = await repos.grants.list_grants(resolved_id, scope=scope, txn=txn)
            remaining: list[GrantPermission] = [
                grant.permission
                for grant in grants
                if grant.principal_type == principal_type and grant.principal_id == principal_id
            ]
            return removed, remaining

        removed, remaining = await _run_owned_grant_change(
            repos, resolved_id, scope=scope, action='unshare', change=_remove,
        )
        if removed:
            message = f'Removed {permission} on context entry {resolved_id} from {principal_type} {principal_id}'
        else:
            message = f'No {permission} grant on context entry {resolved_id} for {principal_type} {principal_id}'
        logger.info(message)
        return UnshareContextResultDict(
            success=True,
            context_id=resolved_id,
            principal_type=principal_type,
            principal_id=principal_id,
            permission=permission,
            removed=removed,
            remaining_permissions=remaining,
            message=message,
        )
    except ToolError:
        raise
    except (EntryNotFoundError, EntryNotAuthorizedError) as exc:
        raise ToolError(format_exception_message(exc)) from exc
    except Exception as exc:
        logger.error(f'Error unsharing context: {exc}')
        raise ToolError(f'Failed to unshare context: {format_exception_message(exc)}') from exc


async def list_context_grants(
    context_id: Annotated[str, Field(min_length=8, description=_CONTEXT_ID_DESCRIPTION)],
) -> ListContextGrantsResultDict:
    """List every grant of an entry you own.

    Only the entry's owner may list its grants. Grants come ordered by principal type,
    principal id and permission, each with the principal that granted it and when.

    Errors: an entry you may not read is reported as "Context entry with ID {id} not
    found"; one you may read but do not own as "Not authorized to list the grants of
    context entry with ID {id}".

    Returns:
        Dict with context_id, grants (each with principal_type, principal_id,
        permission, granted_by and created_at) and count.

    Raises:
        ToolError: If the entry is not found, the caller does not own it, or the
            listing fails.
    """
    try:
        scope = resolve_access_scope()
        repos = await ensure_repositories()
        try:
            resolved_id = await resolve_or_normalize_id(context_id, repos.context, scope=scope)
        except ValueError as exc:
            raise ToolError(f'Invalid context ID: {exc}') from exc

        # Both reads run outside a transaction: ownership never changes through the
        # tools, so the probe and the listing cannot disagree about it.
        access = await repos.context.probe_ids([resolved_id], scope=scope)
        if resolved_id not in access:
            raise ToolError(str(EntryNotFoundError(resolved_id)))
        if not access[resolved_id].is_owner:
            raise ToolError(str(EntryNotAuthorizedError([resolved_id], action='list_grants')))
        records = await repos.grants.list_grants(resolved_id, scope=scope)
        grants = [
            ContextGrantDict(
                principal_type=record.principal_type,
                principal_id=record.principal_id,
                permission=record.permission,
                granted_by=record.granted_by,
                created_at=record.created_at,
            )
            for record in records
        ]
        return ListContextGrantsResultDict(context_id=resolved_id, grants=grants, count=len(grants))
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error listing context grants: {exc}')
        raise ToolError(f'Failed to list context grants: {format_exception_message(exc)}') from exc
