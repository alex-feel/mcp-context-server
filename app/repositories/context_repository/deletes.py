"""Deletion of context entries by id, and the id snapshot of batch-delete criteria."""

import sqlite3
from typing import TYPE_CHECKING
from typing import Literal
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import chunk_ids

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


def _criteria_chunk_pairs(
    context_ids: list[str] | None,
    thread_ids: list[str] | None,
) -> list[tuple[list[str] | None, list[str] | None]]:
    """Pair up bounded chunks of the two client-length-controlled criteria lists.

    The batch-delete criteria combine with AND, and a row has exactly one ``id``
    and one ``thread_id``, so a row matches the full criteria iff it matches
    exactly ONE (id-chunk, thread-chunk) pair -- executing one statement per pair
    and unioning the results is equivalent to the single unchunked statement
    (no duplicates possible) while keeping every statement's bind count bounded:
    at most one chunk of each list, one bind each for ``source`` and
    ``older_than_days``, and the access predicate's binds (three in READ mode,
    one in OWNER mode). A ``None`` element means that criterion is absent from
    the statement, so when neither list is provided the single ``(None, None)``
    pair reproduces the unfiltered statement.

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

    Deletes the entries a scope owns by id, and snapshots the ids matching
    batch-delete criteria among the entries a scope may read or owns, binding
    every client-supplied id and thread list in bounded chunks so no statement
    exceeds a backend's bound-parameter limit. A thread or criteria delete
    snapshots its ids first and then deletes exactly that snapshot by id.
    """

    async def delete_by_ids(
        self,
        context_ids: list[str],
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> int:
        """Delete the entries among the given IDs that the scope owns.

        Deleting is owner-only: an entry the scope may read, edit through a write
        grant, or not see at all is left in place and not counted, exactly like
        an ID no entry carries.

        Args:
            context_ids: List of context entry IDs to delete.
            scope: The caller's scope.
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Number of deleted entries.
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

        def _chunk_statement(chunk: list[str]) -> tuple[str, list[object]]:
            placeholders = ','.join([self._placeholder(i + 1) for i in range(len(chunk))])
            owner = build_access_predicate(
                scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='context_entries',
                start=len(chunk) + 1,
            )
            query = f'DELETE FROM context_entries WHERE id IN ({placeholders}){owner.and_clause()}'
            return query, [*chunk, *owner.params]

        if backend_type == 'sqlite':

            def _delete_by_ids_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                deleted = 0
                for chunk in chunks:
                    query, params = _chunk_statement(chunk)
                    cursor.execute(query, tuple(params))
                    deleted += cursor.rowcount
                return deleted

            if txn:
                return await self._run_sqlite_txn(_delete_by_ids_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_delete_by_ids_sqlite)

        # PostgreSQL
        async def _delete_by_ids_postgresql(conn: 'asyncpg.Connection') -> int:
            deleted = 0
            for chunk in chunks:
                query, params = _chunk_statement(chunk)
                result = await conn.execute(query, *params)
                # asyncpg returns "DELETE N" where N is the count
                deleted += int(result.split()[-1]) if result else 0
            return deleted

        if txn:
            return await _delete_by_ids_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_delete_by_ids_postgresql)

    async def get_ids_matching_batch_criteria(
        self,
        context_ids: list[str] | None = None,
        thread_ids: list[str] | None = None,
        source: str | None = None,
        older_than_days: int | None = None,
        *,
        scope: Scope,
        mode: Literal[AccessMode.READ, AccessMode.OWNER],
    ) -> list[str]:
        """Return the IDs of the entries matching batch-delete criteria that the scope may access in ``mode``.

        The criteria are AND-combined, so with ``context_ids`` every returned ID
        is one the caller named. ``mode`` selects which entries the snapshot
        covers: ``READ`` returns every matching entry the scope may read, which a
        delete that names ids needs to refuse an entry the caller sees but does
        not own; ``OWNER`` returns only the scope's own matching entries, which a
        thread or criteria delete removes while skipping every other entry. A call
        without any criterion matches nothing.

        The returned snapshot is the AUTHORITATIVE delete set: the delete removes
        exactly these ids via ``delete_by_ids`` instead of re-running the criteria,
        so an entry committed after the snapshot survives rather than being
        deleted without its embedding cleanup, and the ``older_than_days`` age
        boundary is evaluated once, here. The client-length-controlled
        ``context_ids``/``thread_ids`` lists are bound in bounded chunks (one
        statement per ``_criteria_chunk_pairs`` pair, all within this single read)
        so an arbitrarily long list never exceeds the per-statement
        bound-parameter limit; every row is still evaluated against the criteria
        exactly once (its id and thread_id select exactly one chunk pair).

        Args:
            context_ids: Filter by these context IDs (intersected with the others).
            thread_ids: Filter by these thread IDs.
            source: Filter by source ('user' or 'agent').
            older_than_days: Filter entries created more than N days ago.
            scope: The caller's scope.
            mode: ``READ`` for the entries the scope may read, ``OWNER`` for the
                entries it owns.

        Returns:
            List of matching context entry IDs.
        """
        backend_type = self.backend.backend_type

        def _chunk_statement(
            id_chunk: list[str] | None, thread_chunk: list[str] | None,
        ) -> tuple[str, list[object]] | None:
            conditions: list[str] = []
            params: list[object] = []

            if id_chunk:
                placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(id_chunk))])
                conditions.append(f'id IN ({placeholders})')
                params.extend(id_chunk)

            if thread_chunk:
                placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(thread_chunk))])
                conditions.append(f'thread_id IN ({placeholders})')
                params.extend(thread_chunk)

            if source:
                conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                params.append(source)

            if older_than_days is not None:
                placeholder = self._placeholder(len(params) + 1)
                if backend_type == 'sqlite':
                    conditions.append(f"created_at < datetime('now', {placeholder})")
                    params.append(f'-{older_than_days} days')
                else:
                    conditions.append(f"created_at < NOW() - ({placeholder}::integer * INTERVAL '1 day')")
                    params.append(older_than_days)

            # Without a criterion the statement would select every row the scope
            # reaches, so the guard runs before the access predicate is added.
            if not conditions:
                return None

            access = build_access_predicate(
                scope, mode=mode, backend_type=backend_type, outer='context_entries', start=len(params) + 1,
            )
            query = f'SELECT id FROM context_entries WHERE {" AND ".join(conditions)}{access.and_clause()}'
            return query, [*params, *access.params]

        statements = [
            statement
            for id_chunk, thread_chunk in _criteria_chunk_pairs(context_ids, thread_ids)
            if (statement := _chunk_statement(id_chunk, thread_chunk)) is not None
        ]
        if not statements:
            return []

        if backend_type == 'sqlite':

            def _select_ids_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                matched: list[str] = []
                for query, params in statements:
                    cursor.execute(query, tuple(params))
                    matched.extend(str(row[0]) for row in cursor.fetchall())
                return matched

            return await self.backend.execute_read(_select_ids_sqlite)

        # PostgreSQL
        async def _select_ids_postgresql(conn: 'asyncpg.Connection') -> list[str]:
            matched: list[str] = []
            for query, params in statements:
                # normalize_id(str(...)) returns the canonical hex id the delete and
                # the probe key on, also on a connection without the uuid codec.
                matched.extend(normalize_id(str(row['id'])) for row in await conn.fetch(query, *params))
            return matched

        return await self.backend.execute_read(_select_ids_postgresql)
