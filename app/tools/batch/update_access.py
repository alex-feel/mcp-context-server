"""Write-access checks of the update_context_batch tool.

Two probes decide which updates the caller may apply: the pre-generation check
that every update targets an entry the caller may modify (and that only the
owner changes an entry's visibility), and the in-transaction re-probe that
disambiguates a version compare-and-set that matched zero rows.
"""

from typing import TYPE_CHECKING
from typing import Any
from typing import NoReturn

from fastmcp.exceptions import ToolError

from app.access_scope import AccessScope
from app.auth import RequestPrincipal
from app.auth import visibility_denied_reason
from app.repositories.context_repository.records import VersionConflictError
from app.tools._transactions import EntryNotFoundError

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer


async def authorize_updates(
    repos: 'RepositoryContainer',
    validated_updates: list[dict[str, Any]],
    principal: RequestPrincipal,
    *,
    scope: AccessScope,
    atomic: bool,
) -> tuple[dict[str, str], dict[str, int], list[tuple[int, str, str]]]:
    """Check that the caller may apply every validated update.

    An entry the caller may not read is not found, and one it may read but not
    modify is refused, before any generation is spent on either. A visibility
    change is owner-only, and publishing as 'public' may additionally require
    the configured publish role.

    Args:
        repos: Repository container.
        validated_updates: The updates that passed validation, each carrying its
            original ``index`` and resolved ``context_id``.
        principal: The calling principal, for the ownership and publish-role checks.
        scope: The caller's scope.
        atomic: Whether the batch is all-or-nothing.

    Returns:
        The source of every entry the caller may modify, keyed by context_id; the
        optimistic-concurrency version of each such entry, captured before
        generation; and one ``(index, context_id, error)`` tuple per refused update.

    Raises:
        ToolError: In atomic mode, on the first refused update.
    """
    existence_errors: list[tuple[int, str, str]] = []  # (index, context_id, error)
    entry_sources: dict[str, str] = {}  # context_id -> source
    # context_id -> optimistic-concurrency version captured BEFORE generation;
    # passed to execute_update_in_transaction as the compare-and-set guard so a
    # concurrent writer that commits during generation is detected.
    entry_versions: dict[str, int] = {}

    for update in validated_updates:
        original_idx = update['index']
        context_id = update['context_id']

        # An entry the caller may not read is not found; one it may read but not
        # modify is refused before any generation is spent on it.
        probe = await repos.context.check_entry_exists(context_id, scope=scope)
        if not probe.exists or not probe.can_write:
            denial = (
                f'Context entry {context_id} not found' if not probe.exists
                else f'Not authorized to modify context entry {context_id}'
            )
            if atomic:
                raise ToolError(f'{denial} at index {original_idx}')
            existence_errors.append((original_idx, context_id, denial))
            continue
        assert probe.source is not None
        assert probe.version is not None
        entry_sources[context_id] = probe.source
        entry_versions[context_id] = probe.version

        # Visibility changes are owner-only, and publishing as 'public' may
        # additionally require the configured publish role. owner_id is
        # immutable, so this pre-generation read cannot go stale.
        visibility_change = update.get('visibility')
        if visibility_change is None:
            continue
        auth_error: str | None = None
        if probe.owner_id != principal.principal_id:
            auth_error = f'Only the owner may change the visibility of context {context_id}'
        else:
            auth_error = visibility_denied_reason(visibility_change, principal)
        if auth_error is not None:
            if atomic:
                raise ToolError(f'{auth_error} (index {original_idx})')
            existence_errors.append((original_idx, context_id, auth_error))

    return entry_sources, entry_versions, existence_errors


async def reraise_disambiguated_cas_conflict(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    context_id: str,
    *,
    scope: AccessScope,
) -> NoReturn:
    """Disambiguate a version compare-and-set that matched zero rows.

    Zero matched rows is ambiguous: a concurrent writer bumped the row's
    version (retryable), or the row was deleted or its write access withdrawn
    after its version was captured (permanent). The atomic update batch calls
    this on the OPEN transaction connection to re-probe the row under the
    caller's write predicate, so such a row aborts with the standard not-found
    error every other update path emits instead of concurrent-modification
    retry advice no retry can satisfy.

    Args:
        repos: Repository container.
        txn: The open transaction the compare-and-set ran on.
        context_id: ID of the entry whose compare-and-set matched zero rows.
        scope: The caller's scope.

    Raises:
        EntryNotFoundError: The caller may no longer modify the row, or it is gone.
        VersionConflictError: The row still exists for the caller with a changed
            version; the conflict propagates to the caller's
            concurrent-modification handling.
    """
    if not await repos.context.entry_exists(context_id, scope=scope, txn=txn):
        raise EntryNotFoundError(context_id) from None
    raise VersionConflictError(context_id) from None
