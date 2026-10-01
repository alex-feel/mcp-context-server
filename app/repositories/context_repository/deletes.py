"""Deletion of context entries by id, by thread, and by batch-delete criteria."""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import chunk_ids
from app.repositories.context_repository.helpers import describe_batch_delete_criteria

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


logger = logging.getLogger(__name__)


def _criteria_chunk_pairs(
    context_ids: list[str] | None,
    thread_ids: list[str] | None,
) -> list[tuple[list[str] | None, list[str] | None]]:
    """Pair up bounded chunks of the two client-length-controlled criteria lists.

    The batch-delete criteria combine with AND, and a row has exactly one ``id``
    and one ``thread_id``, so a row matches the full criteria iff it matches
    exactly ONE (id-chunk, thread-chunk) pair -- executing one statement per pair
    and unioning the results is equivalent to the single unchunked statement
    (no duplicates possible) while keeping every statement under the
    per-statement bound-parameter limit. A ``None`` element means that criterion
    is absent from the statement, so when neither list is provided the single
    ``(None, None)`` pair reproduces the unfiltered statement.

    Args:
        context_ids: Specific context entry ids targeted by the criteria, or None.
        thread_ids: Threads whose entries are targeted by the criteria, or None.

    Returns:
        The cross product of the bounded chunks of both lists, with ``None``
        standing in for an absent criterion.
    """
    id_chunks: list[list[str] | None] = [None]
    if context_ids:
        id_chunks = list(chunk_ids(context_ids))
    thread_chunks: list[list[str] | None] = [None]
    if thread_ids:
        thread_chunks = list(chunk_ids(thread_ids))
    return [(id_chunk, thread_chunk) for id_chunk in id_chunks for thread_chunk in thread_chunks]


class ContextDeleteMixin(BaseRepository):
    """Deletion over ``context_entries``.

    Deletes entries by id or by thread, and selects or deletes the entries matching
    batch-delete criteria, binding every client-supplied id and thread list in
    bounded chunks so no statement exceeds a backend's bound-parameter limit.
    """

    async def delete_by_ids(
        self,
        context_ids: list[str],
        txn: 'TransactionContext | None' = None,
    ) -> int:
        """Delete context entries by their IDs.

        Args:
            context_ids: List of context entry IDs to delete
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Number of deleted entries
        """
        # Defensive check: return 0 if no IDs provided
        # Prevents SQL syntax errors when constructing IN clauses
        if not context_ids:
            return 0

        backend_type = txn.backend_type if txn else self.backend.backend_type

        # Issue the delete in bounded chunks so an arbitrarily long id list (e.g. every
        # entry in a large thread) never exceeds a backend's per-statement bound-parameter
        # limit. Each chunk restarts its placeholders at 1 and the per-chunk rowcounts sum.
        chunks = chunk_ids(context_ids)

        if backend_type == 'sqlite':

            def _delete_by_ids_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                deleted = 0
                for chunk in chunks:
                    placeholders = ','.join([self._placeholder(i + 1) for i in range(len(chunk))])
                    cursor.execute(
                        f'DELETE FROM context_entries WHERE id IN ({placeholders})',
                        tuple(chunk),
                    )
                    deleted += cursor.rowcount
                return deleted

            if txn:
                return await self._run_sqlite_txn(_delete_by_ids_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_delete_by_ids_sqlite)

        # PostgreSQL
        async def _delete_by_ids_postgresql(conn: 'asyncpg.Connection') -> int:
            deleted = 0
            for chunk in chunks:
                placeholders = ','.join([self._placeholder(i + 1) for i in range(len(chunk))])
                result = await conn.execute(
                    f'DELETE FROM context_entries WHERE id IN ({placeholders})',
                    *chunk,
                )
                # asyncpg returns "DELETE N" where N is the count
                deleted += int(result.split()[-1]) if result else 0
            return deleted

        if txn:
            return await _delete_by_ids_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_delete_by_ids_postgresql)

    async def delete_by_thread(self, thread_id: str) -> int:
        """Delete all context entries in a thread.

        Args:
            thread_id: Thread ID to delete entries from

        Returns:
            Number of deleted entries
        """
        if self.backend.backend_type == 'sqlite':

            def _delete_by_thread_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                cursor.execute(
                    f'DELETE FROM context_entries WHERE thread_id = {self._placeholder(1)}',
                    (thread_id,),
                )
                return cursor.rowcount

            return await self.backend.execute_write(_delete_by_thread_sqlite)

        # PostgreSQL
        async def _delete_by_thread_postgresql(conn: 'asyncpg.Connection') -> int:
            result = await conn.execute(
                f'DELETE FROM context_entries WHERE thread_id = {self._placeholder(1)}',
                thread_id,
            )
            # asyncpg returns "DELETE N" where N is the count
            return int(result.split()[-1]) if result else 0

        return await self.backend.execute_write(_delete_by_thread_postgresql)

    async def get_ids_matching_batch_criteria(
        self,
        context_ids: list[str] | None = None,
        thread_ids: list[str] | None = None,
        source: str | None = None,
        older_than_days: int | None = None,
    ) -> list[str]:
        """Return context entry IDs matching batch deletion criteria.

        Builds the same AND-combined WHERE clause as delete_contexts_batch but
        executes a SELECT instead of DELETE. On the SQLite delete tool paths the
        returned snapshot is the AUTHORITATIVE delete set: embedding cleanup
        (vec0 virtual tables lack CASCADE) targets exactly these ids, and the
        destructive step then deletes exactly these ids via ``delete_by_ids``
        instead of re-running the criteria, so an entry committed after the
        snapshot survives rather than being deleted without its embedding
        cleanup. The client-length-controlled ``context_ids``/``thread_ids``
        lists are bound in bounded chunks (one statement per
        ``_criteria_chunk_pairs`` pair, all within this single read) so an
        arbitrarily long list never exceeds the per-statement bound-parameter
        limit; every row is still evaluated against the criteria exactly once
        (its id and thread_id select exactly one chunk pair), so the
        ``older_than_days`` age boundary still needs no caller-resolved
        absolute cutoff.

        Args:
            context_ids: Filter by these context IDs (intersected with the others)
            thread_ids: Filter by these thread IDs
            source: Filter by source ('user' or 'agent')
            older_than_days: Filter entries older than N days

        Returns:
            List of matching context entry IDs.
        """
        if self.backend.backend_type == 'sqlite':

            def _select_ids_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                matched: list[str] = []

                for id_chunk, thread_chunk in _criteria_chunk_pairs(context_ids, thread_ids):
                    conditions: list[str] = []
                    params: list[Any] = []

                    if id_chunk:
                        placeholders = ','.join([
                            self._placeholder(len(params) + i + 1) for i in range(len(id_chunk))
                        ])
                        conditions.append(f'id IN ({placeholders})')
                        params.extend(id_chunk)

                    if thread_chunk:
                        placeholders = ','.join([
                            self._placeholder(len(params) + i + 1) for i in range(len(thread_chunk))
                        ])
                        conditions.append(f'thread_id IN ({placeholders})')
                        params.extend(thread_chunk)

                    if source:
                        conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                        params.append(source)

                    if older_than_days is not None:
                        conditions.append(
                            f"created_at < datetime('now', {self._placeholder(len(params) + 1)})",
                        )
                        params.append(f'-{older_than_days} days')

                    if not conditions:
                        return []

                    where_clause = ' AND '.join(conditions)
                    query = f'SELECT id FROM context_entries WHERE {where_clause}'
                    cursor.execute(query, tuple(params))
                    matched.extend(row[0] for row in cursor.fetchall())

                return matched

            return await self.backend.execute_read(_select_ids_sqlite)

        # PostgreSQL: CASCADE handles embedding cleanup, so this method
        # returns an empty list (caller should not need it).
        return []

    async def delete_contexts_batch(
        self,
        context_ids: list[str] | None = None,
        thread_ids: list[str] | None = None,
        source: str | None = None,
        older_than_days: int | None = None,
    ) -> tuple[int, list[str]]:
        """Delete multiple context entries by various criteria.

        At least one criterion must be provided. Criteria can be combined
        for more targeted deletion. Cascading delete removes associated
        tags and images. On PostgreSQL, embedding rows are removed via
        ON DELETE CASCADE on the surviving embedding table for the active
        compression mode (fp32 ``vec_context_embeddings`` when compression
        is disabled; compressed ``vec_context_embeddings_compressed`` when
        enabled). On SQLite the vec0 virtual table is NOT covered by
        CASCADE and requires explicit cleanup via the embedding repository:
        a caller needing that cleanup must pre-query the snapshot via
        ``get_ids_matching_batch_criteria``, clean those ids, and delete
        exactly that snapshot with ``delete_by_ids`` (as the
        ``delete_context_batch`` tool does) instead of calling this method,
        because this criteria-based DELETE re-evaluates the predicate and
        would sweep rows committed after the cleanup snapshot. The
        compressed ``vec_context_embeddings_compressed`` table IS covered
        by CASCADE on SQLite (it is a standard table, not a virtual one).

        The client-length-controlled ``context_ids``/``thread_ids`` lists are
        bound in bounded chunks: one DELETE per ``_criteria_chunk_pairs`` pair,
        all inside the closure's single write (one write-queue transaction on
        SQLite; ``execute_write`` wraps the closure in one transaction on
        PostgreSQL, where ``NOW()`` is also transaction-stable), so the
        AND-combined criteria semantics and atomicity are preserved -- each row
        matches exactly one pair -- while no statement exceeds the
        per-statement bound-parameter limit.

        Args:
            context_ids: Specific context entry IDs to delete
            thread_ids: Delete all entries in these threads
            source: Filter by source ('user' or 'agent') - combine with other criteria
            older_than_days: Delete entries older than N days

        Returns:
            Tuple of (deleted_count, list_of_criteria_used)
        """
        if self.backend.backend_type == 'sqlite':

            def _delete_batch_sqlite(conn: sqlite3.Connection) -> tuple[int, list[str]]:
                cursor = conn.cursor()
                # Built fresh per closure invocation (via the shared helper) so
                # transparent write-retries (e.g. on a transient "database is
                # locked" error) do not accumulate duplicate criteria strings
                # across attempts.
                criteria_used = describe_batch_delete_criteria(
                    context_ids=context_ids,
                    thread_ids=thread_ids,
                    source=source,
                    older_than_days=older_than_days,
                )

                deleted_count = 0
                for id_chunk, thread_chunk in _criteria_chunk_pairs(context_ids, thread_ids):
                    conditions: list[str] = []
                    params: list[Any] = []

                    if id_chunk:
                        placeholders = ','.join([
                            self._placeholder(len(params) + i + 1) for i in range(len(id_chunk))
                        ])
                        conditions.append(f'id IN ({placeholders})')
                        params.extend(id_chunk)

                    if thread_chunk:
                        placeholders = ','.join([
                            self._placeholder(len(params) + i + 1) for i in range(len(thread_chunk))
                        ])
                        conditions.append(f'thread_id IN ({placeholders})')
                        params.extend(thread_chunk)

                    if source:
                        conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                        params.append(source)

                    if older_than_days is not None:
                        conditions.append(
                            f"created_at < datetime('now', {self._placeholder(len(params) + 1)})",
                        )
                        params.append(f'-{older_than_days} days')

                    if not conditions:
                        return 0, criteria_used

                    where_clause = ' AND '.join(conditions)
                    query = f'DELETE FROM context_entries WHERE {where_clause}'
                    cursor.execute(query, tuple(params))
                    deleted_count += cursor.rowcount

                logger.info(f'Batch delete: removed {deleted_count} entries using criteria: {criteria_used}')
                return deleted_count, criteria_used

            return await self.backend.execute_write(_delete_batch_sqlite)

        # PostgreSQL
        async def _delete_batch_postgresql(conn: 'asyncpg.Connection') -> tuple[int, list[str]]:
            # Built fresh per closure invocation (see the SQLite closure) so
            # retried writes do not accumulate duplicate criteria strings across
            # attempts.
            criteria_used = describe_batch_delete_criteria(
                context_ids=context_ids,
                thread_ids=thread_ids,
                source=source,
                older_than_days=older_than_days,
            )

            deleted_count = 0
            for id_chunk, thread_chunk in _criteria_chunk_pairs(context_ids, thread_ids):
                conditions: list[str] = []
                params: list[Any] = []

                if id_chunk:
                    placeholders = ','.join([
                        self._placeholder(len(params) + i + 1) for i in range(len(id_chunk))
                    ])
                    conditions.append(f'id IN ({placeholders})')
                    params.extend(id_chunk)

                if thread_chunk:
                    placeholders = ','.join([
                        self._placeholder(len(params) + i + 1) for i in range(len(thread_chunk))
                    ])
                    conditions.append(f'thread_id IN ({placeholders})')
                    params.extend(thread_chunk)

                if source:
                    conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                    params.append(source)

                if older_than_days is not None:
                    conditions.append(
                        f"created_at < (NOW() - INTERVAL '{older_than_days} days')",
                    )

                if not conditions:
                    return 0, criteria_used

                where_clause = ' AND '.join(conditions)
                query = f'DELETE FROM context_entries WHERE {where_clause}'
                result = await conn.execute(query, *params)

                # asyncpg returns "DELETE N" where N is the count
                deleted_count += int(result.split()[-1]) if result else 0

            logger.info(f'Batch delete: removed {deleted_count} entries using criteria: {criteria_used}')
            return deleted_count, criteria_used

        return await self.backend.execute_write(_delete_batch_postgresql, validate_connection=True)
