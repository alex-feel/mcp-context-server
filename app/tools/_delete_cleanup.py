"""Entry deletion with owner authorization and the explicit embedding cleanup the storage layout needs.

Used by ``delete_context`` and ``delete_context_batch``: deletes the entries the
caller owns and, on SQLite layouts whose embedding rows do not cascade with the
entry, removes their chunk and vector rows explicitly.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from app.access_scope import AccessScope
from app.backends.sqlite_backend.contention import is_sqlite_locked_error
from app.settings import get_settings
from app.tools._transactions import EntryNotAuthorizedError
from app.tools._transactions import is_connection_error

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer


logger = logging.getLogger(__name__)
settings = get_settings()


def sqlite_embedding_cleanup_required() -> bool:
    """Whether a SQLite delete still needs the explicit per-entry embedding cleanup.

    The cleanup exists for ONE reason: the fp32 ``vec_context_embeddings`` vec0
    VIRTUAL table carries no foreign key and is reachable only through the
    ``embedding_chunks`` bridge, so once that bridge cascades away with the
    context row its vectors orphan permanently.

    With embedding compression enabled that table does not exist at all -- the
    compression migration drops it and the payload table
    ``vec_context_embeddings_compressed`` is an ordinary table with
    ``ON DELETE CASCADE`` on ``context_id``, exactly like ``embedding_metadata``.
    A single ``DELETE FROM context_entries`` then removes every embedding row
    atomically, and the per-entry loop degenerates into one redundant write round
    trip per deleted entry -- which on a thread-wide or criteria-wide delete is
    unbounded and holds the single SQLite writer while every other client stalls.

    Cascade only fires while ``PRAGMA foreign_keys`` is ON, so an operator who
    set ``SQLITE_FOREIGN_KEYS=false`` keeps the explicit cleanup.

    Returns:
        True when the explicit per-entry cleanup is still required on SQLite.
    """
    return not (settings.compression.enabled and settings.storage.sqlite_foreign_keys)


async def cleanup_embeddings_for_delete(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    context_ids: list[str],
) -> None:
    """Delete FK-less SQLite embedding rows for entries about to be removed.

    Runs on the caller's transaction connection so the cleanup and the row
    delete commit or roll back together: without that, a failure between them
    (a client disconnect cancelling the request, a lock-wait timeout, a dropped
    connection) leaves entries stripped of their vectors while their rows
    survive, silently absent from semantic and hybrid search with nothing to
    regenerate them.

    A no-op on PostgreSQL, where ``ON DELETE CASCADE`` removes the embedding
    rows inside the same statement, and a no-op on SQLite whenever cascade
    already covers them (see :func:`sqlite_embedding_cleanup_required`).

    The ids are cleaned in bounded MULTI-ROW statements
    (``delete_all_chunks_bulk``), not one write per entry: the thread-wide and
    criteria-wide delete paths take an id list no client parameter caps, and a
    per-entry loop would hold the single SQLite writer for one round trip per
    matched row while every other client's writes stall behind it.

    A cleanup failure is logged and skipped rather than aborting: deleting an
    entry must stay possible even when its embedding rows are unreadable (a
    missing vec0 module, a corrupted row). On SQLite an error inside a statement
    does not poison the open transaction, so the row delete proceeds normally.
    Because the statements are batched, such a failure leaves the batch's
    remaining vectors orphaned instead of only the offending entry's -- the same
    fail-open direction (never a blocked delete, never lost user data), traded
    for a write path that no longer scales with the number of matched rows.

    Fail-open covers UNREADABLE embedding rows only. Write CONTENTION is the
    opposite case and is re-raised: ``begin_transaction`` opens DEFERRED, so this
    cleanup is the FIRST write of the transaction and therefore exactly where an
    external lock holder surfaces SQLITE_BUSY once ``busy_timeout`` expires.
    Swallowing it would let the row delete commit against a lock that is about to
    clear, permanently orphaning the FK-less vec0 rows this function exists to
    remove -- while reporting success. Re-raising that family propagates out of the
    caller's transaction, rolling the whole delete back so the caller's bounded
    retry (:func:`delete_entries_with_cleanup`) re-runs it atomically once the lock
    clears.

    Args:
        repos: Repository container.
        txn: The open transaction that will also issue the row delete.
        context_ids: The exact ids the delete will remove.
    """
    if not context_ids or txn.backend_type != 'sqlite':
        return
    if not sqlite_embedding_cleanup_required():
        return
    # Gate on whether the embedding tables were ever PROVISIONED, NOT on the
    # runtime ENABLE_EMBEDDING_GENERATION toggle: a prior session may have
    # written embeddings a now-disabled toggle would skip cleaning.
    if not await repos.embeddings.embedding_tables_exist(txn=txn):
        return
    try:
        await repos.embeddings.delete_all_chunks_bulk(context_ids, txn=txn)
    except Exception as exc:
        if is_sqlite_locked_error(exc):
            raise
        logger.warning('Failed to delete embeddings for %d contexts: %s', len(context_ids), exc)


async def delete_entries_with_cleanup(
    repos: 'RepositoryContainer',
    context_ids: list[str],
    *,
    scope: AccessScope,
    refuse_unauthorized: bool,
    max_retries: int = 2,
) -> int:
    """Delete the given entries the caller owns, with their FK-less embedding rows, in one transaction.

    The single chokepoint of every delete: ``delete_context`` by ids and by
    thread, and ``delete_context_batch`` with and without ``context_ids``, so all
    of them get identical authorization, cleanup-then-delete ordering, atomicity
    and transient-fault recovery.

    Deleting is owner-only. Inside the transaction the ids are probed under the
    caller's scope: an id the caller may not read is skipped exactly like an
    absent one, and an entry the caller may read but does not own is either
    skipped (a thread or criteria delete) or, when ``refuse_unauthorized`` is set
    for a delete that names ids, refuses the whole request before anything is
    written. The cleanup and the row delete then cover exactly the owned ids.

    The probe, the cleanup and the row delete share ONE transaction, so a grant
    or visibility change after the probe cannot strip vectors from an entry the
    delete then leaves in place, and a failure between the cleanup and the
    delete can never leave entries stripped of their vectors while their rows
    survive. The bounded retry mirrors the store and update write paths: a
    SQLITE_BUSY from a cross-process lock collision, or a dropped PostgreSQL
    connection, is self-clearing contention rather than a client error, and the
    probe and ``delete_by_ids`` are idempotent, so re-running the rolled-back
    transaction is safe.

    Args:
        repos: Repository container.
        context_ids: The ids to remove: the ids the caller named, or the snapshot
            a thread or criteria delete took.
        scope: The caller's scope.
        refuse_unauthorized: Refuse the request when a readable id is not the
            caller's, instead of skipping it.
        max_retries: Maximum transient-fault retries (exponential backoff).

    Returns:
        The number of context rows deleted.

    Raises:
        EntryNotAuthorizedError: When ``refuse_unauthorized`` is set and the
            caller may read but does not own one of the entries; nothing is
            deleted.
    """
    backend = repos.context.backend
    attempt = 0
    while True:
        try:
            async with backend.begin_transaction() as txn:
                access = await repos.context.probe_ids(context_ids, scope=scope, txn=txn)
                deletable = [i for i in context_ids if i in access and access[i].is_owner]
                if refuse_unauthorized:
                    denied = [i for i in context_ids if i in access and not access[i].is_owner]
                    if denied:
                        raise EntryNotAuthorizedError(denied, action='delete')
                if not deletable:
                    return 0
                await cleanup_embeddings_for_delete(repos, txn, deletable)
                return await repos.context.delete_by_ids(deletable, scope=scope, txn=txn)
        except Exception as exc:
            if is_connection_error(exc) and attempt < max_retries:
                delay = 0.5 * (2 ** attempt)
                attempt += 1
                logger.warning(
                    'Delete transaction failed with a transient error, retrying in %.1fs '
                    '(attempt %d/%d): %s',
                    delay, attempt, max_retries, exc,
                )
                await asyncio.sleep(delay)
                continue
            raise
