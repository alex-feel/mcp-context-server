"""By-id reads, existence probes and id-prefix resolution for context entries."""

import operator
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import chunk_ids
from app.repositories.context_repository.records import CONTEXT_ENTRY_COLUMNS
from app.repositories.context_repository.records import EntryProbe

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


class ContextReadMixin(BaseRepository):
    """By-id reads and existence probes over ``context_entries``.

    Fetches entries by id in bounded chunks, probes an entry's existence, source,
    version and owner for the update paths, reads its content type, and resolves
    an id prefix to the matching full ids.
    """

    async def get_by_ids(self, context_ids: list[str]) -> list[Any]:
        """Get context entries by their IDs.

        Args:
            context_ids: List of context entry IDs

        Returns:
            List of context entry rows (sqlite3.Row or asyncpg.Record depending on
            backend), ordered by ``created_at DESC, id DESC`` across the whole result.
        """
        # Defensive check: return empty list if no IDs provided
        # Prevents SQL syntax errors when constructing IN clauses
        if not context_ids:
            return []

        # Fetch in bounded chunks (mirroring delete_by_ids) so an arbitrarily long id
        # list never exceeds a backend's per-statement bound-parameter limit. Each
        # chunk restarts its placeholders at 1; when more than one statement ran, the
        # accumulated rows are re-sorted in Python so the single-statement
        # ORDER BY created_at DESC, id DESC contract holds across chunk boundaries
        # (the Python tuple sort compares the same TEXT/TIMESTAMPTZ created_at and
        # unique lowercase-hex id values the SQL ORDER BY compares).
        chunks = chunk_ids(context_ids)

        if self.backend.backend_type == 'sqlite':

            def _fetch_sqlite(conn: sqlite3.Connection) -> list[Any]:
                cursor = conn.cursor()
                rows: list[Any] = []
                for chunk in chunks:
                    placeholders = ','.join([self._placeholder(i + 1) for i in range(len(chunk))])
                    # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
                    query = f'''
                        SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries
                        WHERE id IN ({placeholders})
                        ORDER BY created_at DESC, id DESC
                    '''
                    cursor.execute(query, tuple(chunk))
                    rows.extend(cursor.fetchall())
                if len(chunks) > 1:
                    rows.sort(key=operator.itemgetter('created_at', 'id'), reverse=True)
                return rows

            return await self.backend.execute_read(_fetch_sqlite)

        # PostgreSQL
        async def _fetch_postgresql(conn: 'asyncpg.Connection') -> list[Any]:
            rows: list[Any] = []
            for chunk in chunks:
                placeholders = ','.join([self._placeholder(i + 1) for i in range(len(chunk))])
                # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
                query = f'''
                    SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries
                    WHERE id IN ({placeholders})
                    ORDER BY created_at DESC, id DESC
                '''
                rows.extend(await conn.fetch(query, *chunk))
            if len(chunks) > 1:
                rows.sort(key=operator.itemgetter('created_at', 'id'), reverse=True)
            return rows

        return await self.backend.execute_read(_fetch_postgresql)

    async def check_entry_exists(self, context_id: str) -> EntryProbe:
        """Check if a context entry exists and return its source, version, and owner.

        Args:
            context_id: ID of the context entry

        Returns:
            An :class:`EntryProbe`. When ``exists`` is True, ``source`` is
            'user' or 'agent', ``version`` is the current optimistic-concurrency
            token -- update_context captures it BEFORE generation and passes it
            to update_context_entry as the compare-and-set guard, so a concurrent
            writer that commits during generation is detected -- and ``owner_id``
            is the stamped owner backing the owner-only visibility-change check.
        """
        if self.backend.backend_type == 'sqlite':

            def _check_exists_sqlite(conn: sqlite3.Connection) -> EntryProbe:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT source, version, owner_id FROM context_entries '
                    f'WHERE id = {self._placeholder(1)} LIMIT 1',
                    (context_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return EntryProbe(False, None, None, None)
                return EntryProbe(
                    True,
                    cast(str, row['source']),
                    cast(int, row['version']),
                    cast(str, row['owner_id']),
                )

            return await self.backend.execute_read(_check_exists_sqlite)

        # PostgreSQL
        async def _check_exists_postgresql(conn: 'asyncpg.Connection') -> EntryProbe:
            row = await conn.fetchrow(
                f'SELECT source, version, owner_id FROM context_entries '
                f'WHERE id = {self._placeholder(1)} LIMIT 1',
                context_id,
            )
            if row is None:
                return EntryProbe(False, None, None, None)
            return EntryProbe(
                True,
                cast(str, row['source']),
                cast(int, row['version']),
                cast(str, row['owner_id']),
            )

        return await self.backend.execute_read(_check_exists_postgresql)

    async def entry_exists(self, context_id: str, txn: 'TransactionContext | None' = None) -> bool:
        """Return whether a context entry exists, optionally on a transaction connection.

        A lightweight companion to check_entry_exists for callers that need only
        presence (not source/version) and must run inside an open transaction.
        execute_update_in_transaction uses it to confirm the parent row before a
        tags-only or images-only update, whose child writes would otherwise
        violate the foreign key against a missing parent (charging the circuit
        breaker) or orphan the rows.

        When invoked inside a transaction on PostgreSQL, the presence check locks
        the parent row with FOR KEY SHARE so a concurrent DELETE blocks until this
        update transaction commits: without the lock the row can be deleted in the
        window between this check and the subsequent child tag/image writes, whose
        foreign key then fails -- a non-ControlFlowError that charges the circuit
        breaker. FOR KEY SHARE permits concurrent non-key updates while blocking
        row deletion, which is exactly the parent-presence guarantee the child
        writes need. Outside a transaction the lock would release at statement end
        and serve no purpose, so it is applied only on the transaction path.

        Args:
            context_id: ID of the context entry.
            txn: Optional transaction context. When provided the read runs on the
                transaction's own connection instead of acquiring a second pooled
                connection, avoiding a nested pool acquire while a transaction
                connection is already held (PostgreSQL pool-starvation hazard).

        Returns:
            True if the entry exists, False otherwise.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        if backend_type == 'sqlite':

            def _entry_exists_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT 1 FROM context_entries WHERE id = {self._placeholder(1)} LIMIT 1',
                    (context_id,),
                )
                return cursor.fetchone() is not None

            if txn is not None:
                return await self._run_sqlite_txn(_entry_exists_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_entry_exists_sqlite)

        # PostgreSQL
        lock_clause = ' FOR KEY SHARE' if txn is not None else ''

        async def _entry_exists_postgresql(conn: 'asyncpg.Connection') -> bool:
            row = await conn.fetchrow(
                f'SELECT 1 FROM context_entries WHERE id = {self._placeholder(1)} LIMIT 1{lock_clause}',
                context_id,
            )
            return row is not None

        if txn is not None:
            return await _entry_exists_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_entry_exists_postgresql)

    async def get_content_type(self, context_id: str, txn: 'TransactionContext | None' = None) -> str | None:
        """Get the content type of a context entry.

        Args:
            context_id: ID of the context entry
            txn: Optional transaction context. When provided the read runs on the
                transaction's own connection instead of acquiring a second pooled
                connection, avoiding a nested pool acquire while a transaction
                connection is already held (PostgreSQL pool-starvation hazard).

        Returns:
            Content type ('text' or 'multimodal') or None if entry doesn't exist
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        if backend_type == 'sqlite':

            def _get_content_type_sqlite(conn: sqlite3.Connection) -> str | None:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT content_type FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                row = cursor.fetchone()
                return row['content_type'] if row else None

            if txn is not None:
                return await self._run_sqlite_txn(_get_content_type_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_get_content_type_sqlite)

        # PostgreSQL
        async def _get_content_type_postgresql(conn: 'asyncpg.Connection') -> str | None:
            row = await conn.fetchrow(
                f'SELECT content_type FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            return row['content_type'] if row else None

        if txn is not None:
            return await _get_content_type_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_get_content_type_postgresql)

    async def find_ids_by_prefix(self, prefix: str, limit: int = 2) -> list[str]:
        """Find context entry IDs that begin with the given prefix.

        Backs the ID-prefix resolution helper :func:`app.ids.resolve_prefix` (reached via
        :func:`app.ids.resolve_or_normalize_id`), which expands a short user-supplied
        prefix into a full id when ambiguity is unlikely. The caller decides what to do
        when ``limit`` rows are returned; this method's contract is "return up to N
        matches".

        Args:
            prefix: Lowercase hex prefix. ``resolve_prefix`` only calls this for prefixes
                of 8-31 hex characters (the range :func:`app.ids.is_id_prefix` accepts);
                the caller is responsible for normalization and that length check.
            limit: Maximum number of IDs to return. Defaults to 2 so callers can
                detect ambiguity by checking ``len(result) > 1``.

        Returns:
            List of matching context_id strings (UUIDv7 hex, 32 chars), up to ``limit``.
        """
        if self.backend.backend_type == 'sqlite':

            def _find_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT id FROM context_entries
                    WHERE id LIKE {self._placeholder(1)}
                    ORDER BY id
                    LIMIT {self._placeholder(2)}
                    ''',
                    (prefix + '%', limit),
                )
                return [row['id'] for row in cursor.fetchall()]

            return await self.backend.execute_read(_find_sqlite)

        # PostgreSQL: id is a uuid column; canonical text form contains hyphens
        # which are absent from the user-supplied hex prefix. REPLACE() removes
        # hyphens so LIKE matches against the 32-char hex representation.
        async def _find_postgresql(conn: 'asyncpg.Connection') -> list[str]:
            rows = await conn.fetch(
                f'''
                SELECT id FROM context_entries
                WHERE REPLACE(CAST(id AS TEXT), '-', '') LIKE {self._placeholder(1)}
                ORDER BY id
                LIMIT {self._placeholder(2)}
                ''',
                prefix + '%',
                limit,
            )
            # The pool's uuid->str codec (registered in _init_connection) already
            # decodes id to the 32-char hex the SQLite path returns.
            # normalize_id(str(...)) is idempotent defense-in-depth for
            # codec-less connections, so prefix resolution echoes a canonical
            # context_id on both backends (mirrors grep_scan_text_contents).
            return [normalize_id(str(row['id'])) for row in rows]

        return await self.backend.execute_read(_find_postgresql)
