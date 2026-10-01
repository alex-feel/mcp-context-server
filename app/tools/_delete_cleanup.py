"""Entry deletion with the explicit embedding cleanup the active storage layout needs.

Used by ``delete_context`` and ``delete_context_batch``: deletes the entries and,
on SQLite layouts whose embedding rows do not cascade with the entry, removes
their chunk and vector rows explicitly.
"""

import asyncio
import logging
from typing import TYPE_CHECKING

from app.backends.sqlite_backend import is_sqlite_locked_error
from app.settings import get_settings
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
    max_retries: int = 2,
) -> int:
    """Delete context entries and their FK-less embedding rows in one transaction.

    The single chokepoint shared by ``delete_context`` (both the by-ids and the
    SQLite by-thread branch) and ``delete_context_batch``'s SQLite criteria
    branch, so all three get identical cleanup-then-delete ordering, identical
    atomicity, and identical transient-fault recovery.

    Cleanup and row delete share ONE transaction so a failure between them can
    never leave entries stripped of their vectors while their rows survive. The
    bounded retry mirrors the store and update write paths: a SQLITE_BUSY from a
    cross-process lock collision, or a dropped PostgreSQL connection, is
    self-clearing contention rather than a client error, and ``delete_by_ids`` is
    idempotent, so re-running the rolled-back transaction is safe.

    Args:
        repos: Repository container.
        context_ids: The exact ids to remove.
        max_retries: Maximum transient-fault retries (exponential backoff).

    Returns:
        The number of context rows deleted.
    """
    backend = repos.context.backend
    attempt = 0
    while True:
        try:
            async with backend.begin_transaction() as txn:
                await cleanup_embeddings_for_delete(repos, txn, context_ids)
                return await repos.context.delete_by_ids(context_ids, txn=txn)
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
