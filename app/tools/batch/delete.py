"""The delete_context_batch tool: delete context entries by IDs, threads, source, or age."""

import logging
from typing import Annotated
from typing import Literal
from typing import cast

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.access_scope import AccessMode
from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.ids import resolve_or_normalize_ids
from app.repositories.context_repository.helpers import describe_batch_delete_criteria
from app.startup import ensure_repositories
from app.tools._delete_cleanup import delete_entries_with_cleanup
from app.tools._transactions import EntryNotAuthorizedError
from app.tools._validation import reject_unstorable_input
from app.types import BulkDeleteResponseDict

logger = logging.getLogger(__name__)


async def delete_context_batch(
    context_ids: Annotated[
        list[str] | None,
        Field(max_length=100, description='Specific context IDs to delete (max 100 per call)'),
    ] = None,
    thread_ids: Annotated[
        list[str] | None,
        Field(max_length=100, description="Delete the caller's own entries in these threads (max 100 per call)"),
    ] = None,
    source: Annotated[
        Literal['user', 'agent'] | None,
        Field(description='Delete only entries from this source (combine with other criteria)'),
    ] = None,
    older_than_days: Annotated[
        int | None,
        Field(description='Delete entries older than N days (combine with other criteria)', gt=0),
    ] = None,
) -> BulkDeleteResponseDict:
    """Delete multiple context entries by various criteria. IRREVERSIBLE.

    Criteria can be combined for targeted deletion:
    - context_ids: Delete specific entries by ID
    - thread_ids: Delete entries in the specified threads
    - source: Filter by source ('user' or 'agent')
    - older_than_days: Delete entries created more than N days ago

    At least one criterion must be provided.
    Note: source alone and older_than_days alone are each insufficient; either must be
    combined with another criterion, because on its own each one matches essentially the
    whole database.
    All associated data (tags, images) is also removed.

    Only the caller's own entries are deleted, and an entry the caller may not read
    is never matched or counted:
    - With context_ids: when any named entry that matches the other criteria is one
      the caller may read but does not own, the call is refused with
      "Not authorized to delete context entries: {ids}" and nothing is deleted.
    - Without context_ids: only the caller's own matching entries are deleted; every
      other matching entry is left in place without an error.

    The context_ids and thread_ids lists each accept at most 100 items per call
    (the same cap as the other batch tools); oversized lists are rejected at the
    tool boundary as validation errors before any database work.

    WARNING: This operation cannot be undone. Verify criteria before deletion.

    Returns:
        BulkDeleteResponseDict with success (bool), deleted_count (int),
        criteria_used (list of str), message (str).

    Raises:
        ToolError: If no criteria provided, a named entry the caller may read is not
            its own, or deletion fails.
    """
    try:
        # Validate at least one criterion is provided
        if not any([context_ids, thread_ids, source, older_than_days]):
            raise ToolError(
                'At least one deletion criterion must be provided: '
                'context_ids, thread_ids, source, or older_than_days',
            )

        # Validate source if provided alone
        if source and not any([context_ids, thread_ids, older_than_days]):
            raise ToolError(
                'source filter must be combined with another criterion '
                '(context_ids, thread_ids, or older_than_days)',
            )

        # older_than_days alone carries the same blast radius as source alone: on any
        # database older than the window, one scalar matches essentially every row, so
        # a single call reaches the whole table irreversibly. Require it to be combined
        # for the same reason source already is; a retention purge stays expressible as
        # older_than_days plus source or thread_ids.
        if older_than_days is not None and not any([context_ids, thread_ids, source]):
            raise ToolError(
                'older_than_days must be combined with another criterion '
                '(context_ids, thread_ids, or source)',
            )

        # Reject an embedded NUL or unpaired UTF-16 surrogate in any thread_id before it
        # reaches the criteria delete's bind, where asyncpg would raise a non-ControlFlowError
        # that charges the circuit breaker (SQLite binds it silently -- a divergence).
        reject_unstorable_input(thread_ids=cast('object', thread_ids))

        repos = await ensure_repositories()
        # A prefix resolves over the entries the caller may read, and only the
        # entries the caller owns are deleted.
        scope = resolve_access_scope()

        # Resolve incoming IDs at the boundary: accept full 32/36-char IDs or
        # 8-31 char hex prefixes (uniform with delete_context).
        if context_ids:
            try:
                context_ids = await resolve_or_normalize_ids(context_ids, repos.context, scope=scope)
            except ValueError as e:
                raise ToolError(f'Invalid context ID: {e}') from e

        # A call that names ids snapshots every matching entry the caller may read,
        # so a named entry the caller sees but does not own reaches the delete and
        # refuses the whole call; without named ids the call is a thread or criteria
        # delete, which snapshots only the caller's own entries and skips the rest.
        # A named id the caller may not read is absent from either snapshot.
        names_ids = bool(context_ids)
        mode: Literal[AccessMode.READ, AccessMode.OWNER] = AccessMode.READ if names_ids else AccessMode.OWNER

        # The snapshot holds the exact ids the COMBINED criteria match (combining
        # context_ids with source/older_than_days never cleans an excluded,
        # surviving entry), and the delete removes EXACTLY those ids, on both
        # backends. On SQLite the fp32 vec0 virtual embedding table lacks FK CASCADE,
        # so its vectors are cleaned explicitly for the snapshot ids; re-running the
        # criteria in the destructive statement would re-evaluate the predicate in a
        # second transaction, so a store committing a matching entry between the
        # cleanup snapshot and the delete would be swept while its embeddings
        # escaped the snapshot and orphaned permanently. Deleting the snapshot keeps
        # the cleaned set and the deleted set identical: an entry inserted after
        # the snapshot is neither cleaned nor deleted (it simply survives the
        # operation). The single predicate evaluation also makes the
        # older_than_days age boundary race-free without a shared absolute cutoff.
        # delete_by_ids chunks the id list, so a criteria match of any size stays
        # under the per-statement bound-parameter limit.
        #
        # Cleanup and delete share ONE transaction with the access probe
        # (delete_entries_with_cleanup, the same helper delete_context uses, which
        # also retries a transient lock collision) so a failure between them can
        # never leave entries stripped of their vectors while their rows survive.
        # The cleanup costs nothing at all on PostgreSQL and whenever CASCADE
        # already covers the embedding rows, and in the fp32 layout that still needs
        # it, issues bounded multi-row statements instead of one write round trip
        # per matched entry.
        affected_ids = await repos.context.get_ids_matching_batch_criteria(
            context_ids=context_ids,
            thread_ids=thread_ids,
            source=source,
            older_than_days=older_than_days,
            scope=scope,
            mode=mode,
        )
        if names_ids:
            # Every snapshot id is a named id (the criteria are AND-combined); keep
            # the caller's order so a refusal lists the ids as the caller named them.
            named_order = {context_id: position for position, context_id in enumerate(context_ids or [])}
            affected_ids.sort(key=named_order.__getitem__)
        deleted_count = 0
        if affected_ids:
            deleted_count = await delete_entries_with_cleanup(
                repos, affected_ids, scope=scope, refuse_unauthorized=names_ids,
            )
        criteria_used = describe_batch_delete_criteria(
            context_ids=context_ids,
            thread_ids=thread_ids,
            source=source,
            older_than_days=older_than_days,
        )

        logger.info(f'Batch delete completed: {deleted_count} entries removed')

        return BulkDeleteResponseDict(
            success=True,
            deleted_count=deleted_count,
            criteria_used=criteria_used,
            message=f'Successfully deleted {deleted_count} context entries',
        )

    except ToolError:
        raise
    except EntryNotAuthorizedError as e:
        raise ToolError(format_exception_message(e)) from e
    except Exception as e:
        logger.error(f'Error in batch delete: {e}')
        raise ToolError(f'Batch delete failed: {format_exception_message(e)}') from e
