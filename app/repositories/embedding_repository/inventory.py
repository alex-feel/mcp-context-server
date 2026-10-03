"""Existence probes, statistics and storage size of stored embeddings."""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


logger = logging.getLogger(__name__)


class EmbeddingInventoryMixin(BaseRepository):
    """Read-only inventory of stored embeddings.

    Probes whether an entry has an embedding and whether the embedding tables
    exist at all, reports embedding and chunk counts over the entries a scope may
    read, and measures the storage size of the vector payload table the active
    compression mode uses, which covers every entry.
    """

    async def exists(self, context_id: str, txn: 'TransactionContext | None' = None) -> bool:
        """Check if embedding exists for context entry.

        Args:
            context_id: ID of the context entry
            txn: Optional transaction context. When provided the check runs on the
                transaction's own connection instead of acquiring a second pooled
                connection, avoiding a nested pool acquire while a transaction
                connection is already held (PostgreSQL pool-starvation hazard).

        Returns:
            True if embedding exists, False otherwise
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        if backend_type == 'sqlite':

            def _exists_sqlite(conn: sqlite3.Connection) -> bool:
                query = f'SELECT 1 FROM embedding_metadata WHERE context_id = {self._placeholder(1)} LIMIT 1'
                cursor = conn.execute(query, (context_id,))
                return cursor.fetchone() is not None

            if txn is not None:
                return await self._run_sqlite_txn(_exists_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_exists_sqlite)

        # postgresql
        async def _exists_postgresql(conn: 'asyncpg.Connection') -> bool:
            query = f'SELECT 1 FROM embedding_metadata WHERE context_id = {self._placeholder(1)} LIMIT 1'
            row = await conn.fetchrow(query, context_id)
            return row is not None

        if txn is not None:
            return await _exists_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(_exists_postgresql)

    async def embedding_tables_exist(self, txn: 'TransactionContext | None' = None) -> bool:
        """Check whether the embedding_metadata table exists (table-safe).

        Unlike :meth:`exists`, this never raises when embedding storage was never
        provisioned (ENABLE_EMBEDDING_GENERATION has always been false, so the
        semantic/chunking migrations never created the embedding tables). Used to guard
        stale-embedding cleanup on a text update when embedding generation is disabled at
        update time (the entry may still carry chunks from when it WAS enabled).

        Args:
            txn: Optional transaction context. When provided the check runs on the
                transaction's own connection.

        Returns:
            True if the embedding_metadata table is present, False otherwise.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        if backend_type == 'sqlite':

            def _tables_exist_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='embedding_metadata' LIMIT 1",
                )
                return cursor.fetchone() is not None

            if txn is not None:
                return await self._run_sqlite_txn(_tables_exist_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_tables_exist_sqlite)

        # postgresql: to_regclass resolves via the connection's search_path and returns
        # NULL for a missing relation (no UndefinedTableError).
        async def _tables_exist_postgresql(conn: 'asyncpg.Connection') -> bool:
            return await conn.fetchval("SELECT to_regclass('embedding_metadata')") is not None

        if txn is not None:
            return await _tables_exist_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(cast(Any, _tables_exist_postgresql))

    async def get_statistics(self, *, scope: Scope) -> dict[str, Any]:
        """Get embedding statistics, chunk information included, over the entries the scope may read.

        Every count applies the READ predicate: the entry total to ``context_entries``
        and the embedding and chunk counts to the entry each embedding belongs to.

        The chunk-count source is ``embedding_metadata.chunk_count`` (summed)
        on BOTH backends, regardless of compression mode. This is the single
        source-of-truth populated by every write path (fp32 SQLite, fp32
        PostgreSQL, compressed SQLite, compressed PostgreSQL), so the
        reported chunk total stays correct whether compressed payloads or
        fp32 vectors are stored on disk. On PostgreSQL the compression
        migration drops ``vec_context_embeddings``, so reading the count from
        ``embedding_metadata`` is the only query that succeeds in both modes.

        Args:
            scope: The caller's scope.

        Returns:
            Dictionary with statistics (count, coverage, chunk info, etc.)
        """
        backend_type = self.backend.backend_type
        entries = build_access_predicate(scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries')
        parents = build_access_predicate(scope, mode=AccessMode.READ, backend_type=backend_type, outer='ce')
        readable_embeddings = f'embedding_metadata em JOIN context_entries ce ON em.context_id = ce.id{parents.where_clause()}'
        total_sql = f'SELECT COUNT(*) FROM context_entries{entries.where_clause()}'
        embeddings_sql = f'SELECT COUNT(*) FROM {readable_embeddings}'
        chunks_sql = f'SELECT COALESCE(SUM(em.chunk_count), 0) FROM {readable_embeddings}'

        def _figures(total_entries: int, embedding_count: int, total_chunks: int) -> dict[str, Any]:
            coverage_percentage = (embedding_count / total_entries * 100) if total_entries > 0 else 0.0
            average_chunks = round(total_chunks / embedding_count, 2) if embedding_count > 0 else 0.0
            return {
                'total_embeddings': embedding_count,
                'total_entries': total_entries,
                'total_chunks': total_chunks,
                'average_chunks_per_entry': average_chunks,
                'coverage_percentage': round(coverage_percentage, 2),
                'backend': backend_type,
            }

        if backend_type == 'sqlite':

            def _get_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                return _figures(
                    conn.execute(total_sql, entries.params).fetchone()[0],
                    conn.execute(embeddings_sql, parents.params).fetchone()[0],
                    conn.execute(chunks_sql, parents.params).fetchone()[0],
                )

            return await self.backend.execute_read(_get_stats_sqlite)

        # postgresql
        async def _get_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            return _figures(
                await conn.fetchval(total_sql, *entries.params),
                await conn.fetchval(embeddings_sql, *parents.params),
                await conn.fetchval(chunks_sql, *parents.params),
            )

        return await self.backend.execute_read(_get_stats_postgresql)

    async def get_embeddings_size(self) -> tuple[float, bool]:
        """Get the storage size of embedding vector payloads in megabytes.

        The returned size covers only the vector payload table that is active
        for the current compression mode: ``vec_context_embeddings_compressed``
        when compression is enabled, otherwise ``vec_context_embeddings``.

        The number is NOT byte-comparable across backends. On PostgreSQL it is
        the on-disk relation size including indexes (``pg_total_relation_size``).
        On SQLite it is the exact compressed payload bytes (``SUM(LENGTH(payload))``)
        when compression is enabled, or a deterministic fp32 estimate
        (``SUM(chunk_count * dimensions * 4)``) when it is not.

        Any failure (for example a missing table) is logged and reported as a
        zero size so that statistics never fail because of this sub-block.

        Returns:
            Tuple of (size in megabytes rounded to two places, estimated flag).
            The estimated flag is ``True`` only for the SQLite fp32 estimate.
        """
        try:
            backend_type = self.backend.backend_type
            if backend_type == 'sqlite':
                return await self._get_embeddings_size_sqlite()
            if backend_type == 'postgresql':
                return await self._get_embeddings_size_postgresql()
            logger.warning('Unknown backend type %r; embeddings_size_mb reported as 0.0', backend_type)
        except Exception as e:
            logger.warning('Failed to compute embeddings size: %s', e)
        return 0.0, False

    async def _get_embeddings_size_sqlite(self) -> tuple[float, bool]:
        """Get embedding payload size for SQLite (dbstat-free)."""
        from app.settings import get_settings

        if get_settings().compression.enabled:
            # Exact compressed payload bytes.
            def _read_compressed_size(conn: sqlite3.Connection) -> int:
                cursor = conn.execute(
                    'SELECT COALESCE(SUM(LENGTH(payload)), 0) AS size_bytes FROM vec_context_embeddings_compressed',
                )
                return int(cursor.fetchone()['size_bytes'])

            size_bytes = await self.backend.execute_read(_read_compressed_size)
            return round(float(size_bytes) / (1024 * 1024), 2), False

        # Deterministic fp32 estimate: chunks * dimensions * 4 bytes per float.
        def _read_estimate_size(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                'SELECT COALESCE(SUM(chunk_count * dimensions * 4), 0) AS size_bytes FROM embedding_metadata',
            )
            return int(cursor.fetchone()['size_bytes'])

        size_bytes = await self.backend.execute_read(_read_estimate_size)
        return round(float(size_bytes) / (1024 * 1024), 2), True

    async def _get_embeddings_size_postgresql(self) -> tuple[float, bool]:
        """Get embedding payload size for PostgreSQL (on-disk relation size)."""
        from app.settings import get_settings

        active_table = (
            'vec_context_embeddings_compressed' if get_settings().compression.enabled else 'vec_context_embeddings'
        )

        # to_regclass returns NULL for a missing table, avoiding UndefinedTableError
        # (the compression migration drops vec_context_embeddings on PostgreSQL).
        async def _read_relation_size(conn: 'asyncpg.Connection') -> int:
            row = await conn.fetchrow(
                'SELECT COALESCE(pg_total_relation_size(to_regclass($1)), 0) AS size_bytes',
                active_table,
            )
            return int(row['size_bytes']) if row else 0

        size_bytes = await self.backend.execute_read(_read_relation_size)
        return round(float(size_bytes) / (1024 * 1024), 2), False
