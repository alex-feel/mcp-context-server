"""The delete_context tool: delete context entries by ID or by thread."""

import logging
from typing import Annotated

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.access_scope import AccessMode
from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.ids import resolve_or_normalize_ids
from app.startup import ensure_repositories
from app.tools._delete_cleanup import delete_entries_with_cleanup
from app.tools._transactions import EntryNotAuthorizedError
from app.tools._validation import reject_unstorable_input

logger = logging.getLogger(__name__)


async def delete_context(
    context_ids: Annotated[
        list[str] | None,
        Field(
            min_length=1,
            max_length=100,
            description='Specific context entry IDs to delete (max 100 per call; mutually exclusive with thread_id)',
        ),
    ] = None,
    thread_id: Annotated[
        str | None,
        Field(min_length=1, description='Delete ALL entries in thread (mutually exclusive with context_ids)'),
    ] = None,
) -> dict[str, bool | int | str]:
    """Delete context entries by specific IDs or by entire thread. IRREVERSIBLE.

    Provide EITHER context_ids OR thread_id (not both). All associated data
    (tags, images) is also removed.

    context_ids accepts at most 100 IDs per call (the same cap as
    get_context_by_ids and the batch tools); an oversized list is rejected at the
    tool boundary as a validation error before any database work. Delete larger
    sets in successive calls, or delete a whole thread via thread_id.

    WARNING: This operation cannot be undone. Verify IDs/thread before deletion.

    Returns:
        Dict with success (bool), deleted_count (int), and message (str) fields.

    Raises:
        ToolError: If neither context_ids nor thread_id is provided, if BOTH are
            provided, or if deletion fails.
    """
    try:
        # Ensure at least one parameter is provided (business logic validation)
        if not context_ids and not thread_id:
            raise ToolError('Must provide either context_ids or thread_id')

        # Both provided is REFUSED, not silently resolved. The two parameters are
        # documented as mutually exclusive and the dispatch below is if/elif, so
        # accepting the combination would delete only the listed ids while the
        # response reported success for a request that also named a whole thread --
        # a partially executed irreversible delete the caller cannot detect. The
        # sibling delete_context_batch takes the same two argument names and
        # deliberately AND-combines them as criteria, which makes silent precedence
        # here doubly misleading.
        if context_ids and thread_id:
            raise ToolError(
                'context_ids and thread_id are mutually exclusive: provide exactly one. '
                'To delete specific entries within a thread, use delete_context_batch, '
                'which combines its criteria.',
            )

        # Reject an embedded NUL or unpaired UTF-16 surrogate in thread_id before it
        # reaches the thread snapshot's bind: on PostgreSQL asyncpg would raise a
        # non-ControlFlowError that charges the circuit breaker, while SQLite would
        # bind it silently -- a cross-backend divergence on a client-controlled value.
        reject_unstorable_input(thread_id=thread_id)

        # Get repositories first; prefix resolution below needs the context repo.
        repos = await ensure_repositories()
        # A prefix resolves over the entries the caller may read, and only the
        # entries the caller owns are deleted.
        scope = resolve_access_scope()

        # Resolve incoming IDs at the boundary: accept full 32/36-char IDs or
        # 8-31 char hex prefixes (uniform with get_context_by_ids/update_context).
        if context_ids:
            try:
                context_ids = await resolve_or_normalize_ids(context_ids, repos.context, scope=scope)
            except ValueError as e:
                raise ToolError(f'Invalid context ID: {e}') from e

        deleted = 0

        if context_ids:
            # The ids are named, so the caller already knows the entries it can read:
            # one it may read but does not own refuses the whole call before anything
            # is deleted, while an id it may not read counts as absent. The probe,
            # the embedding cleanup and the row delete run in ONE transaction so they
            # commit or roll back together. Deleting the embeddings first closes
            # the orphaned-vector window (a SQLite vec0 row outliving its context
            # row); the transaction closes the complementary one -- a failure or
            # cancellation between the two would otherwise leave entries stripped
            # of their vectors while their rows survive, silently missing from
            # semantic and hybrid search with no path back short of a text edit.
            # The cleanup is a no-op on PostgreSQL, where ON DELETE CASCADE removes
            # the embedding rows in the same statement. A transient lock collision
            # rolls the whole thing back and is retried inside the helper, so a
            # self-clearing SQLITE_BUSY neither fails the delete nor commits it with
            # the vectors left behind.
            deleted = await delete_entries_with_cleanup(repos, context_ids, scope=scope, refuse_unauthorized=True)
            logger.info(f'Deleted {deleted} context entries by IDs')

        elif thread_id:
            # Delete the caller's own entries of a thread, silently leaving every other
            # entry in it. On both backends the delete covers exactly a snapshot of the
            # thread's ids: on SQLite the fp32 vec0 virtual embedding table has no FK
            # CASCADE and is reached only through the embedding_chunks bridge, so its
            # vectors orphan permanently unless cleaned first, and a WHERE thread_id = ?
            # delete would re-evaluate the predicate independently of the cleanup
            # snapshot, sweeping a store committed into the thread in between while its
            # embeddings escaped the cleanup. Constrained to the snapshot ids, the cleaned
            # set and the deleted set are identical: an entry inserted after the snapshot
            # is neither cleaned nor deleted (it simply survives the operation).
            # delete_by_ids chunks the id list, so a very large thread is safe.
            thread_ids_to_delete = await repos.context.get_ids_matching_batch_criteria(
                thread_ids=[thread_id], scope=scope, mode=AccessMode.OWNER,
            )
            if thread_ids_to_delete:
                deleted = await delete_entries_with_cleanup(
                    repos, thread_ids_to_delete, scope=scope, refuse_unauthorized=False,
                )
            logger.info(f'Deleted {deleted} entries from thread {thread_id}')

        return {
            'success': True,
            'deleted_count': deleted,
            'message': f'Successfully deleted {deleted} context entries',
        }
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except EntryNotAuthorizedError as e:
        raise ToolError(format_exception_message(e)) from e
    except Exception as e:
        logger.error(f'Error deleting context: {e}')
        raise ToolError(f'Failed to delete context: {format_exception_message(e)}') from e
