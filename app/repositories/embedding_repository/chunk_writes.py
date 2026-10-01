"""Chunk embedding storage and deletion for the fp32 and compressed tables."""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.repositories.base import BaseRepository
from app.repositories.embedding_repository.records import SQLITE_IN_CLAUSE_BATCH
from app.repositories.embedding_repository.records import ChunkEmbedding

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


logger = logging.getLogger(__name__)


class ChunkWriteMixin(BaseRepository):
    """Chunk embedding storage and deletion on both backends and both storage modes.

    ``store_chunked`` writes every chunk of an entry into the fp32 or the compressed
    table, whichever the compression toggle selects, optionally replacing the
    existing chunks first; ``delete_all_chunks`` and ``delete_all_chunks_bulk``
    remove the chunk, vector and ``embedding_metadata`` rows of one or many
    entries from the same table.
    """

    async def store_chunked(
        self,
        context_id: str,
        chunk_embeddings: list[ChunkEmbedding],
        model: str,
        txn: 'TransactionContext | None' = None,
        *,
        upsert: bool = False,
    ) -> None:
        """Store multiple chunk embeddings with boundaries for a context entry atomically.

        All embeddings are stored in a single transaction - either all succeed
        or all fail. Chunk boundaries are stored for chunk-aware reranking.

        When ``settings.compression.enabled`` is true the call is routed to
        the compressed write path which persists provider-encoded payload
        bytes into ``vec_context_embeddings_compressed`` instead of the
        fp32 ``vec_context_embeddings`` table.

        Args:
            context_id: ID of the context entry
            chunk_embeddings: List of ChunkEmbedding objects (embedding + boundaries)
            model: Model identifier (from settings.embedding.model)
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.
            upsert: If True, delete existing embeddings before storing new ones.
                Use this for defense-in-depth when deduplication is possible.
                Default is False for backward compatibility.

        Raises:
            ValueError: If chunk_embeddings list is empty
        """
        if not chunk_embeddings:
            raise ValueError('chunk_embeddings list cannot be empty')

        # Branch on compression toggle. The compressed path expects the
        # caller (via generate_compression_with_timeout in app.tools._generation)
        # to have populated ChunkEmbedding.payload with provider-encoded
        # bytes.
        from app.settings import get_settings
        if get_settings().compression.enabled:
            await self._store_chunked_compressed(
                context_id, chunk_embeddings, model, txn=txn, upsert=upsert,
            )
            return

        # Defense-in-depth: if upsert enabled, delete existing embeddings first
        # This ensures idempotency - calling multiple times produces same result
        if upsert:
            deleted_count = await self.delete_all_chunks(context_id, txn=txn)
            if deleted_count > 0:
                logger.debug(
                    f'UPSERT mode: deleted {deleted_count} existing chunk embeddings '
                    f'for context {context_id} before storing new ones',
                )

        chunk_count = len(chunk_embeddings)
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _store_chunked_sqlite(conn: sqlite3.Connection) -> None:
                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. '
                        'Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)',
                    ) from e

                # Step 1: Get next available rowid for vec0 virtual table
                cursor = conn.execute('SELECT COALESCE(MAX(rowid), 0) + 1 FROM vec_context_embeddings')
                next_rowid = cursor.fetchone()[0]

                vec_rowids: list[int] = []
                for i, chunk_emb in enumerate(chunk_embeddings):
                    vec_rowid = next_rowid + i
                    embedding_blob: bytes = cast(Any, sqlite_vec).serialize_float32(chunk_emb.embedding)
                    conn.execute(
                        'INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (?, ?)',
                        (vec_rowid, embedding_blob),
                    )
                    vec_rowids.append(vec_rowid)

                # Step 2: Insert mapping records into embedding_chunks WITH BOUNDARIES
                for i, vec_rowid in enumerate(vec_rowids):
                    chunk_emb = chunk_embeddings[i]
                    conn.execute(
                        'INSERT INTO embedding_chunks(context_id, vec_rowid, start_index, end_index) VALUES (?, ?, ?, ?)',
                        (context_id, vec_rowid, chunk_emb.start_index, chunk_emb.end_index),
                    )

                # Step 3: Insert embedding_metadata with chunk_count
                conn.execute(
                    '''INSERT INTO embedding_metadata (context_id, model_name, dimensions, chunk_count, created_at, updated_at)
                       VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
                    (context_id, model, len(chunk_embeddings[0].embedding), chunk_count),
                )

            if txn:
                await self._run_sqlite_txn(_store_chunked_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_store_chunked_sqlite)
            logger.debug(f'Stored {chunk_count} chunk embeddings for context {context_id} (SQLite)')

        else:  # postgresql

            async def _store_chunked_postgresql(conn: 'asyncpg.Connection') -> None:
                # Step 1: Insert all embeddings into vec_context_embeddings WITH BOUNDARIES
                # PostgreSQL uses id BIGSERIAL, context_id can repeat (1:N)
                for chunk_emb in chunk_embeddings:
                    await conn.execute(
                        '''INSERT INTO vec_context_embeddings(context_id, embedding, start_index, end_index)
                           VALUES ($1, $2, $3, $4)''',
                        context_id, chunk_emb.embedding, chunk_emb.start_index, chunk_emb.end_index,
                    )

                # Step 2: Insert embedding_metadata with chunk_count
                await conn.execute(
                    '''INSERT INTO embedding_metadata (context_id, model_name, dimensions, chunk_count, created_at, updated_at)
                       VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)''',
                    context_id, model, len(chunk_embeddings[0].embedding), chunk_count,
                )

            if txn:
                await _store_chunked_postgresql(cast('asyncpg.Connection', txn.connection))
            else:
                await self.backend.execute_write(cast(Any, _store_chunked_postgresql))
            logger.debug(f'Stored {chunk_count} chunk embeddings for context {context_id} (PostgreSQL)')

    async def _store_chunked_compressed(
        self,
        context_id: str,
        chunk_embeddings: list[ChunkEmbedding],
        model: str,
        txn: 'TransactionContext | None' = None,
        *,
        upsert: bool = False,
    ) -> None:
        """Persist compressed chunk payloads to vec_context_embeddings_compressed.

        Requires every ``ChunkEmbedding`` to carry a non-None ``payload``;
        the caller (``generate_compression_with_timeout`` in
        ``app.tools._generation``) populates it before invoking the transaction.

        Args:
            context_id: ID of the context entry.
            chunk_embeddings: Compressed-payload chunks (payload is
                non-None for every element).
            model: Embedding model name (recorded in embedding_metadata).
            txn: Optional transaction context for atomic multi-repository
                operations.
            upsert: If True, delete existing compressed rows for the context
                before writing the new ones (defense-in-depth idempotency).

        Raises:
            ValueError: If any chunk lacks the required ``payload`` bytes
                or if the chunk list is empty.
        """
        if not chunk_embeddings:
            raise ValueError('chunk_embeddings list cannot be empty')

        missing_payload = [
            i for i, c in enumerate(chunk_embeddings) if c.payload is None
        ]
        if missing_payload:
            raise ValueError(
                'compressed store path requires payload bytes on every '
                f'ChunkEmbedding; missing at indices: {missing_payload}',
            )

        if upsert:
            deleted = await self._delete_all_chunks_compressed(context_id, txn=txn)
            if deleted > 0:
                logger.debug(
                    'UPSERT mode: deleted %d compressed chunks for context %s',
                    deleted, context_id,
                )

        chunk_count = len(chunk_embeddings)
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _store_compressed_sqlite(conn: sqlite3.Connection) -> None:
                for i, chunk in enumerate(chunk_embeddings):
                    conn.execute(
                        'INSERT INTO vec_context_embeddings_compressed '
                        '(context_id, chunk_index, start_index, end_index, payload) '
                        'VALUES (?, ?, ?, ?, ?)',
                        (
                            context_id,
                            i,
                            chunk.start_index,
                            chunk.end_index,
                            chunk.payload,
                        ),
                    )
                # embedding_metadata stays the source-of-truth for chunk_count
                # and model so existing dedup/exists logic keeps working
                # unchanged.
                conn.execute(
                    'INSERT INTO embedding_metadata '
                    '(context_id, model_name, dimensions, chunk_count, '
                    'created_at, updated_at) '
                    'VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)',
                    (
                        context_id,
                        model,
                        len(chunk_embeddings[0].embedding),
                        chunk_count,
                    ),
                )

            if txn:
                await self._run_sqlite_txn(_store_compressed_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_store_compressed_sqlite)
            logger.debug(
                f'Stored {chunk_count} compressed chunks for context '
                f'{context_id} (SQLite)',
            )
            return

        # postgresql
        async def _store_compressed_pg(conn: 'asyncpg.Connection') -> None:
            for i, chunk in enumerate(chunk_embeddings):
                await conn.execute(
                    'INSERT INTO vec_context_embeddings_compressed '
                    '(context_id, chunk_index, start_index, end_index, payload) '
                    'VALUES ($1, $2, $3, $4, $5)',
                    context_id,
                    i,
                    chunk.start_index,
                    chunk.end_index,
                    chunk.payload,
                )
            await conn.execute(
                'INSERT INTO embedding_metadata '
                '(context_id, model_name, dimensions, chunk_count, '
                'created_at, updated_at) '
                'VALUES ($1, $2, $3, $4, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)',
                context_id,
                model,
                len(chunk_embeddings[0].embedding),
                chunk_count,
            )

        if txn:
            await _store_compressed_pg(cast('asyncpg.Connection', txn.connection))
        else:
            await self.backend.execute_write(cast(Any, _store_compressed_pg))
        logger.debug(
            f'Stored {chunk_count} compressed chunks for context '
            f'{context_id} (PostgreSQL)',
        )

    async def _delete_all_chunks_compressed(
        self,
        context_id: str,
        txn: 'TransactionContext | None' = None,
    ) -> int:
        """Delete compressed chunk rows + embedding_metadata for a context.

        Args:
            context_id: ID of the context entry.
            txn: Optional transaction context for atomic multi-repository
                operations.

        Returns:
            Number of compressed chunk rows deleted.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _delete_compressed_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM vec_context_embeddings_compressed '
                    'WHERE context_id = ?',
                    (context_id,),
                )
                n = int(cursor.fetchone()[0])
                if n == 0:
                    return 0
                conn.execute(
                    'DELETE FROM vec_context_embeddings_compressed '
                    'WHERE context_id = ?',
                    (context_id,),
                )
                conn.execute(
                    'DELETE FROM embedding_metadata WHERE context_id = ?',
                    (context_id,),
                )
                return n

            if txn:
                return await self._run_sqlite_txn(_delete_compressed_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_delete_compressed_sqlite)

        # postgresql
        async def _delete_compressed_pg(conn: 'asyncpg.Connection') -> int:
            count = await conn.fetchval(
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed '
                'WHERE context_id = $1',
                context_id,
            )
            if count == 0:
                return 0
            await conn.execute(
                'DELETE FROM vec_context_embeddings_compressed '
                'WHERE context_id = $1',
                context_id,
            )
            await conn.execute(
                'DELETE FROM embedding_metadata WHERE context_id = $1',
                context_id,
            )
            return int(count)

        if txn:
            return await _delete_compressed_pg(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(cast(Any, _delete_compressed_pg))

    async def _delete_all_chunks_compressed_bulk(
        self,
        context_ids: list[str],
        txn: 'TransactionContext | None' = None,
    ) -> int:
        """Delete compressed chunk rows + embedding_metadata for many contexts on SQLite.

        The bulk counterpart to :meth:`_delete_all_chunks_compressed`, issuing
        one bounded ``IN (...)`` statement per slice instead of one write round
        trip per id. Both deletes cover every id in the slice, matching the fp32
        bulk branch: the callers are delete paths, so the entries (and therefore
        their ``embedding_metadata`` rows) are going away regardless of whether
        any compressed payload row survived for them.

        Args:
            context_ids: The ids whose compressed embedding rows must be removed.
            txn: Optional transaction context so the cleanup commits with the row
                delete. When None, each slice runs as its own write.

        Returns:
            Number of compressed chunk rows deleted.
        """
        deleted_total = 0
        for start in range(0, len(context_ids), SQLITE_IN_CLAUSE_BATCH):
            chunk = context_ids[start : start + SQLITE_IN_CLAUSE_BATCH]

            def _delete_compressed_chunk_sqlite(conn: sqlite3.Connection, ids: list[str] = chunk) -> int:
                placeholders = ', '.join('?' * len(ids))
                cursor = conn.execute(
                    'SELECT COUNT(*) FROM vec_context_embeddings_compressed '
                    f'WHERE context_id IN ({placeholders})',
                    ids,
                )
                deleted = int(cursor.fetchone()[0])
                conn.execute(
                    f'DELETE FROM vec_context_embeddings_compressed WHERE context_id IN ({placeholders})',
                    ids,
                )
                conn.execute(
                    f'DELETE FROM embedding_metadata WHERE context_id IN ({placeholders})',
                    ids,
                )
                return deleted

            if txn:
                deleted_total += await self._run_sqlite_txn(
                    _delete_compressed_chunk_sqlite, cast(sqlite3.Connection, txn.connection),
                )
            else:
                deleted_total += await self.backend.execute_write(_delete_compressed_chunk_sqlite)

        logger.debug(
            f'Deleted {deleted_total} compressed chunk embeddings for {len(context_ids)} contexts (SQLite bulk)',
        )
        return deleted_total

    async def delete_all_chunks(
        self,
        context_id: str,
        txn: 'TransactionContext | None' = None,
    ) -> int:
        """Delete all chunk embeddings for a context entry.

        Used before re-embedding when content is updated.
        For SQLite, also cleans up embedding_chunks mapping table.

        When ``settings.compression.enabled`` is true the call is routed to
        the compressed cleanup path which targets
        ``vec_context_embeddings_compressed`` instead of the fp32 tables.

        Args:
            context_id: ID of the context entry
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Number of chunk embeddings deleted
        """
        # Branch on compression toggle (mirrors the store_chunked branch so
        # cleanup, upsert, and full delete paths all reach the right table).
        from app.settings import get_settings
        if get_settings().compression.enabled:
            return await self._delete_all_chunks_compressed(context_id, txn=txn)

        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _delete_all_chunks_sqlite(conn: sqlite3.Connection) -> int:
                # Step 1: Get vec_rowids from embedding_chunks
                cursor = conn.execute(
                    'SELECT vec_rowid FROM embedding_chunks WHERE context_id = ?',
                    (context_id,),
                )
                vec_rowids = [row[0] for row in cursor.fetchall()]

                if not vec_rowids:
                    return 0

                # Step 2: Delete from vec_context_embeddings (virtual table)
                for vec_rowid in vec_rowids:
                    conn.execute(
                        'DELETE FROM vec_context_embeddings WHERE rowid = ?',
                        (vec_rowid,),
                    )

                # Step 3: Delete from embedding_chunks
                conn.execute(
                    'DELETE FROM embedding_chunks WHERE context_id = ?',
                    (context_id,),
                )

                # Step 4: Delete from embedding_metadata
                conn.execute(
                    'DELETE FROM embedding_metadata WHERE context_id = ?',
                    (context_id,),
                )

                return len(vec_rowids)

            if txn:
                deleted_count = await self._run_sqlite_txn(
                    _delete_all_chunks_sqlite, cast(sqlite3.Connection, txn.connection),
                )
            else:
                deleted_count = await self.backend.execute_write(_delete_all_chunks_sqlite)
            logger.debug(f'Deleted {deleted_count} chunk embeddings for context {context_id} (SQLite)')
            return deleted_count

        # postgresql

        async def _delete_all_chunks_postgresql(conn: 'asyncpg.Connection') -> int:
            # Step 1: Count chunks before delete
            count: int = await conn.fetchval(
                'SELECT COUNT(*) FROM vec_context_embeddings WHERE context_id = $1',
                context_id,
            )

            if count == 0:
                return 0

            # Step 2: Delete from vec_context_embeddings
            await conn.execute(
                'DELETE FROM vec_context_embeddings WHERE context_id = $1',
                context_id,
            )

            # Step 3: Delete from embedding_metadata
            await conn.execute(
                'DELETE FROM embedding_metadata WHERE context_id = $1',
                context_id,
            )

            return count

        if txn:
            deleted_count = await _delete_all_chunks_postgresql(cast('asyncpg.Connection', txn.connection))
        else:
            deleted_count = await self.backend.execute_write(cast(Any, _delete_all_chunks_postgresql))
        logger.debug(f'Deleted {deleted_count} chunk embeddings for context {context_id} (PostgreSQL)')
        return deleted_count

    async def delete_all_chunks_bulk(
        self,
        context_ids: list[str],
        txn: 'TransactionContext | None' = None,
    ) -> int:
        """Delete chunk embeddings for many context entries in bounded multi-row statements.

        The bulk counterpart to :meth:`delete_all_chunks`, for the delete paths whose
        id list is not client-capped (a thread-wide delete, a criteria-wide batch
        delete). Calling the per-id method in a loop costs one write round trip per
        entry -- on SQLite each one hops onto the single writer, so a large thread
        occupies the exclusive writer continuously while every other client stalls.
        Here each bounded chunk of ids costs ONE executor hop issuing multi-row
        statements, turning O(entries) round trips into O(chunks).

        Both SQLite layouts get a bulk implementation, because both are reachable
        with an uncapped list. The fp32 layout needs one because its vectors live
        in the FK-less ``vec_context_embeddings`` vec0 virtual table, reachable
        only through the ``embedding_chunks`` bridge. The compressed layout needs
        one because its ``ON DELETE CASCADE`` only fires while
        ``PRAGMA foreign_keys`` is ON: with ``SQLITE_FOREIGN_KEYS=false`` the
        delete path's own gate still routes every id here, so a per-id fallback
        would reinstate exactly the writer monopolization this method removes.

        PostgreSQL delegates to the per-id method: cascade always removes the
        embedding rows inside the row-delete statement there, so the delete path
        never calls this at all and a second bulk implementation would be
        unexercised.

        Args:
            context_ids: The ids whose embedding rows must be removed.
            txn: Optional transaction context so the cleanup commits with the row
                delete. When None, each chunk runs as its own write.

        Returns:
            Number of chunk embedding rows deleted.
        """
        if not context_ids:
            return 0

        from app.settings import get_settings
        backend_type = txn.backend_type if txn else self.backend.backend_type
        if backend_type != 'sqlite':
            total = 0
            for context_id in context_ids:
                total += await self.delete_all_chunks(context_id, txn=txn)
            return total

        if get_settings().compression.enabled:
            return await self._delete_all_chunks_compressed_bulk(context_ids, txn=txn)

        deleted_total = 0
        for start in range(0, len(context_ids), SQLITE_IN_CLAUSE_BATCH):
            chunk = context_ids[start : start + SQLITE_IN_CLAUSE_BATCH]

            def _delete_chunk_sqlite(conn: sqlite3.Connection, ids: list[str] = chunk) -> int:
                placeholders = ', '.join('?' * len(ids))
                cursor = conn.execute(
                    f'SELECT vec_rowid FROM embedding_chunks WHERE context_id IN ({placeholders})',
                    ids,
                )
                vec_rowids = [row[0] for row in cursor.fetchall()]
                if vec_rowids:
                    # vec0 accepts an ordinary IN (...) predicate on rowid; batch it in
                    # the same bounded slices so one oversized thread cannot exceed
                    # SQLITE_MAX_VARIABLE_NUMBER.
                    for rowid_start in range(0, len(vec_rowids), SQLITE_IN_CLAUSE_BATCH):
                        rowid_chunk = vec_rowids[rowid_start : rowid_start + SQLITE_IN_CLAUSE_BATCH]
                        rowid_placeholders = ', '.join('?' * len(rowid_chunk))
                        conn.execute(
                            f'DELETE FROM vec_context_embeddings WHERE rowid IN ({rowid_placeholders})',
                            rowid_chunk,
                        )
                conn.execute(
                    f'DELETE FROM embedding_chunks WHERE context_id IN ({placeholders})',
                    ids,
                )
                conn.execute(
                    f'DELETE FROM embedding_metadata WHERE context_id IN ({placeholders})',
                    ids,
                )
                return len(vec_rowids)

            if txn:
                deleted_total += await self._run_sqlite_txn(
                    _delete_chunk_sqlite, cast(sqlite3.Connection, txn.connection),
                )
            else:
                deleted_total += await self.backend.execute_write(_delete_chunk_sqlite)

        logger.debug(
            f'Deleted {deleted_total} chunk embeddings for {len(context_ids)} contexts (SQLite bulk)',
        )
        return deleted_total
