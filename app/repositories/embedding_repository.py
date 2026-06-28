"""
Repository for vector embeddings supporting both sqlite-vec and pgvector.

This module provides data access for semantic search embeddings,
handling storage, retrieval, and search operations on vector embeddings
across both SQLite (sqlite-vec) and PostgreSQL (pgvector) backends.
"""


import asyncio
import logging
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext
    from app.compression.types import CompressionMetadata

logger = logging.getLogger(__name__)


# Module-level cache for the compression provenance metadata only.
# Compression provenance is immutable post-bootstrap (validator-enforced
# invariant), so caching it once per process is safe. The metadata is
# bound to a specific StorageBackend, so its cache must live next to
# the repository that consumes it. The cached compression provider
# itself lives in :mod:`app.compression.factory` so the encode and
# search paths share one singleton. A non-reentrant asyncio.Lock,
# constructed at module import time, serializes first-time concurrent
# callers so they observe the same instance.
_compression_metadata: 'CompressionMetadata | None' = None
_compression_metadata_init_lock: asyncio.Lock = asyncio.Lock()


# Decoding the stored TurboQuant payloads (payload_from_bytes per chunk row, each
# doing struct.unpack + np.frombuffer().copy()) and concatenating them is O(N-chunks)
# pure-Python CPU on the compressed read path (search_compressed). For a large
# candidate set this would pin the asyncio event loop and starve other concurrent
# MCP requests, so it is offloaded to a worker thread above this row count -- the
# same discipline the read-path (navigation/grep) and write-leg (chunking/index_tree)
# offloads apply, here measured in chunk rows rather than characters. Small searches
# stay inline to avoid a per-call thread hop. The vectorized provider GEMM
# (estimate_inner_product / decode) is already offloaded inside the provider.
_COMPRESSED_OFFLOAD_MIN_ROWS = 2048

# SQLite caps host parameters per statement at SQLITE_MAX_VARIABLE_NUMBER (as low
# as 999 on older builds), so an unbounded candidate id list cannot be bound as a
# single IN (...) clause without risking 'too many SQL variables'. The compressed
# read batches the candidate ids in chunks below the most conservative limit;
# PostgreSQL is unaffected (it binds the whole set as one = ANY($1::uuid[]) array).
_SQLITE_IN_CLAUSE_BATCH = 900


async def _get_cached_compression_metadata(
    backend: StorageBackend,
) -> 'CompressionMetadata':
    """Return the singleton provenance row, caching it on first call.

    Args:
        backend: Storage backend used to read the row on cache miss.

    Returns:
        The cached :class:`CompressionMetadata` row.

    Raises:
        RuntimeError: If the provenance row is missing. The startup
            validator would normally insert it; absence here indicates the
            validator was bypassed or the database is corrupted.
    """
    global _compression_metadata
    if _compression_metadata is not None:
        return _compression_metadata
    async with _compression_metadata_init_lock:
        if _compression_metadata is None:
            from app.compression.provenance import read_compression_metadata
            meta = await read_compression_metadata(backend)
            if meta is None:
                raise RuntimeError(
                    'compression_metadata row is missing; ensure the server '
                    'started with ENABLE_EMBEDDING_COMPRESSION=true so the '
                    'startup validator bootstrapped the provenance row.',
                )
            _compression_metadata = meta
    return _compression_metadata


def _reset_compression_cache() -> None:
    """Clear the module-level metadata cache and inner LRU factories.

    Tests that switch compression configuration between cases call this
    to avoid leaking metadata state across the process. In addition to
    clearing the local metadata singleton, this function:

    1. Delegates the cached-provider reset to
       :func:`app.compression.factory.reset_cached_compression_provider`
       so both the encode (write) path in :mod:`app.tools._shared` and
       the search (read) path in this module observe a fresh provider
       on the next call.
    2. Invalidates every ``@lru_cache``-decorated factory inside the
       TurboQuant subpackage so a follow-up call with different
       ``(dim, bits, seed, variant)`` constructs fresh rotations,
       codebooks, and quantizers.

    The inner caches are imported lazily inside the function body to keep
    numpy out of the import graph for installations that skipped the
    compression extra.
    """
    global _compression_metadata
    _compression_metadata = None

    # Delegate provider-cache reset to the factory module so all
    # consumers share one truth source.
    from app.compression import reset_cached_compression_provider
    reset_cached_compression_provider()

    # Lazy module imports: the compression extra is optional; this
    # function MUST remain importable even when numpy is absent. Module
    # imports (not symbol imports) avoid the type-checker's private-usage
    # warning for the inner ``_get_cached_*`` factories while still
    # giving us access to their ``.cache_clear()`` method via getattr,
    # which is opaque to private-name analysis.
    try:
        from app.compression.providers.turboquant import _codebook as _codebook_mod
        from app.compression.providers.turboquant import _qjl as _qjl_mod
        from app.compression.providers.turboquant import _rotation as _rotation_mod
        from app.compression.providers.turboquant import encoder as _encoder_mod
    except ImportError:
        # Compression extra not installed; nothing inner-cached to clear.
        return

    # Each inner factory exposes the standard functools.lru_cache.cache_clear
    # callable; reflective access avoids type-checker noise around the
    # leading-underscore naming of the encoder-internal factories.
    for module, attr_name in (
        (_rotation_mod, '_get_cached_rotation'),
        (_qjl_mod, '_get_cached_qjl_impl'),
        (_encoder_mod, '_get_mse_quantizer'),
        (_encoder_mod, '_get_ip_quantizer'),
        (_codebook_mod, 'get_codebook'),
    ):
        factory = getattr(module, attr_name, None)
        if factory is not None and hasattr(factory, 'cache_clear'):
            factory.cache_clear()


__all__ = [
    'ChunkEmbedding',
    'EmbeddingRepository',
    'MetadataFilterValidationError',
    '_reset_compression_cache',
]


@dataclass
class ChunkEmbedding:
    """Embedding data for a single chunk with boundary information.

    This dataclass bundles the embedding vector with its character boundaries
    in the original document, enabling chunk-aware reranking. When embedding
    compression is enabled the optional ``payload`` field carries the
    provider-encoded bytes that the compressed write path persists to the
    ``vec_context_embeddings_compressed`` table; the fp32 write path ignores
    it.

    Attributes:
        embedding: The embedding vector for this chunk.
        start_index: Character offset where chunk starts in original document.
        end_index: Character offset where chunk ends in original document.
        payload: Optional compressed payload bytes produced by the active
            compression provider's ``encode_sync`` method. ``None`` when
            compression is disabled.

    Example:
        >>> chunk_emb = ChunkEmbedding(
        ...     embedding=[0.1, 0.2, 0.3],
        ...     start_index=0,
        ...     end_index=100
        ... )
    """

    embedding: list[float]
    start_index: int
    end_index: int
    payload: bytes | None = None


class MetadataFilterValidationError(Exception):
    """Exception raised when metadata filters fail validation.

    This exception enables unified error handling between search_context
    and semantic_search_context tools.
    """

    def __init__(self, message: str, validation_errors: list[str]) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            validation_errors: List of validation error messages
        """
        super().__init__(message)
        self.message = message
        self.validation_errors = validation_errors


class EmbeddingRepository(BaseRepository):
    """Repository for vector embeddings supporting both sqlite-vec and pgvector.

    This repository handles all database operations for semantic search embeddings,
    using either sqlite-vec extension (SQLite) or pgvector extension (PostgreSQL)
    depending on the configured storage backend.

    Supported backends:
    - SQLite: Uses sqlite-vec with BLOB storage and vec_distance_l2()
    - PostgreSQL: Uses pgvector with native vector type and <-> operator
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the embedding repository.

        Args:
            backend: Storage backend for all database operations
        """
        super().__init__(backend)

    async def store(
        self,
        context_id: str,
        embedding: list[float],
        model: str,
        *,
        start_index: int = 0,
        end_index: int = 0,
    ) -> None:
        """Store embedding for a context entry.

        This is a convenience method for storing a single embedding. It uses the chunked
        storage architecture internally, creating a single-chunk entry for compatibility
        with the 1:N embedding schema.

        Args:
            context_id: ID of the context entry
            embedding: Embedding vector (dimension depends on provider/model configuration)
            model: Model identifier (from settings.embedding.model)
            start_index: Character offset where text starts (default: 0 for full document)
            end_index: Character offset where text ends (default: 0 for legacy/unknown)
        """
        # Delegate to store_chunked with single embedding for unified storage logic
        chunk_emb = ChunkEmbedding(embedding=embedding, start_index=start_index, end_index=end_index)
        await self.store_chunked(context_id, [chunk_emb], model)
        logger.debug(f'Stored embedding for context {context_id}')

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

        This method replaces store() for chunked content. All embeddings are
        stored in a single transaction - either all succeed or all fail.
        Chunk boundaries are stored for chunk-aware reranking.

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
        # caller (via generate_compression_with_timeout in app.tools._shared)
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
                _store_chunked_sqlite(cast(sqlite3.Connection, txn.connection))
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
        ``app.tools._shared``) populates it before invoking the transaction.

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
                _store_compressed_sqlite(cast(sqlite3.Connection, txn.connection))
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
                return _delete_compressed_sqlite(cast(sqlite3.Connection, txn.connection))
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
                deleted_count = _delete_all_chunks_sqlite(cast(sqlite3.Connection, txn.connection))
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

    async def search(
        self,
        query_embedding: list[float],
        limit: int = 20,
        offset: int = 0,
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
        content_type: Literal['text', 'multimodal'] | None = None,
        tags: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        explain_query: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """KNN search with optional filters including date range and metadata.

        SQLite: Uses CTE-based pre-filtering with vec_distance_l2() function
        PostgreSQL: Uses direct JOIN with <-> operator for L2 distance

        Args:
            query_embedding: Query vector for similarity search
            limit: Maximum number of results to return
            offset: Number of results to skip (pagination)
            thread_id: Optional filter by thread
            source: Optional filter by source type
            content_type: Filter by content type (text or multimodal)
            tags: Filter by any of these tags (OR logic)
            start_date: Filter by created_at >= date (ISO 8601 format)
            end_date: Filter by created_at <= date (ISO 8601 format)
            metadata: Simple metadata filters (key=value equality)
            metadata_filters: Advanced metadata filters with operators
            explain_query: If True, include query execution plan in stats

        Returns:
            Tuple of (search results list, statistics dictionary)
        """
        # Dispatch to the compressed read path when the bootstrap-only
        # toggle is on. Settings are read once per process (CLAUDE.md
        # "Settings Singleton Caching") so this branch is stable for the
        # lifetime of the running server.
        from app.settings import get_settings
        if get_settings().compression.enabled:
            return await self.search_compressed(
                query_embedding=query_embedding,
                limit=limit,
                offset=offset,
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata=metadata,
                metadata_filters=metadata_filters,
                explain_query=explain_query,
            )

        if self.backend.backend_type == 'sqlite':

            def _search_sqlite(
                conn: sqlite3.Connection,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                import time as time_module

                start_time = time_module.time()

                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. '
                        'Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)',
                    ) from e

                query_blob: bytes = cast(Any, sqlite_vec).serialize_float32(query_embedding)

                filter_conditions: list[str] = []
                filter_params: list[Any] = []

                # Count filters applied
                filter_count = 0

                if thread_id:
                    filter_conditions.append('thread_id = ?')
                    filter_params.append(thread_id)
                    filter_count += 1

                if source:
                    filter_conditions.append('source = ?')
                    filter_params.append(source)
                    filter_count += 1

                if content_type:
                    filter_conditions.append('content_type = ?')
                    filter_params.append(content_type)
                    filter_count += 1

                # Tag filter (uses subquery with indexed tag table)
                if tags:
                    normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
                    if normalized_tags:
                        tag_placeholders = ','.join(['?' for _ in normalized_tags])
                        filter_conditions.append(f'''
                            id IN (
                                SELECT DISTINCT context_entry_id
                                FROM tags
                                WHERE tag IN ({tag_placeholders})
                            )
                        ''')
                        filter_params.extend(normalized_tags)
                        filter_count += 1

                # Date range filtering - Use datetime() to normalize ISO 8601 input
                # datetime() converts all ISO 8601 formats (T separator, Z suffix, timezone offsets)
                # to SQLite's space-separated format 'YYYY-MM-DD HH:MM:SS' for proper comparison.
                # Without datetime(), TEXT comparison fails because 'T' > ' ' in ASCII ordering.
                if start_date:
                    filter_conditions.append('created_at >= datetime(?)')
                    filter_params.append(start_date)
                    filter_count += 1

                if end_date:
                    filter_conditions.append('created_at <= datetime(?)')
                    filter_params.append(end_date)
                    filter_count += 1

                # Metadata filtering using MetadataQueryBuilder
                metadata_filter_count = 0
                if metadata or metadata_filters:
                    from pydantic import ValidationError

                    from app.metadata_types import MetadataFilter
                    from app.query_builder import MetadataQueryBuilder

                    metadata_builder = MetadataQueryBuilder(backend_type='sqlite')
                    validation_errors: list[str] = []

                    # Simple metadata filters (key=value equality). An invalid KEY is
                    # reported as a structured validation error (NOT silently dropped, which
                    # would widen the result set), consistent with search_context and the
                    # advanced filters below.
                    if metadata:
                        for key, value in metadata.items():
                            try:
                                metadata_builder.add_simple_filter(key, value)
                                metadata_filter_count += 1
                            except ValueError as e:
                                validation_errors.append(f'Invalid metadata key {key!r}: {e}')

                    # Advanced metadata filters with operators
                    if metadata_filters:
                        for filter_dict in metadata_filters:
                            try:
                                filter_spec = MetadataFilter(**filter_dict)
                                metadata_builder.add_advanced_filter(filter_spec)
                                metadata_filter_count += 1
                            except ValidationError as e:
                                error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                                validation_errors.append(error_msg)
                            except ValueError as e:
                                error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                                validation_errors.append(error_msg)
                            except Exception as e:
                                error_msg = f'Unexpected error in metadata filter {filter_dict}: {e}'
                                validation_errors.append(error_msg)
                                logger.error(f'Unexpected error processing metadata filter: {e}')

                    # Raise if ANY (simple key or advanced filter) validation failed,
                    # unified with search_context's structured short-circuit behavior.
                    if validation_errors:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed',
                            validation_errors,
                        )

                    # Add metadata conditions to filter
                    metadata_clause, metadata_params = metadata_builder.build_where_clause()
                    if metadata_clause:
                        filter_conditions.append(metadata_clause)
                        filter_params.extend(metadata_params)
                        filter_count += metadata_filter_count

                where_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ''

                # Use CTE with deduplication by context_id - preserves best chunk boundaries
                # Uses subquery JOIN to identify which chunk had MIN(distance)
                query = f'''
                    WITH filtered_contexts AS (
                        SELECT id
                        FROM context_entries
                        {where_clause}
                    ),
                    chunk_distances AS (
                        SELECT
                            ec.context_id,
                            ec.start_index,
                            ec.end_index,
                            vec_distance_l2(?, ve.embedding) as distance
                        FROM filtered_contexts fc
                        JOIN embedding_chunks ec ON ec.context_id = fc.id
                        JOIN vec_context_embeddings ve ON ve.rowid = ec.vec_rowid
                    ),
                    best_chunks AS (
                        -- One row per context (the nearest chunk). ROW_NUMBER picks a
                        -- single chunk even when two chunks tie at the minimum
                        -- distance, matching the PostgreSQL DISTINCT ON path; the old
                        -- MIN-equality join emitted duplicate rows for one context on
                        -- a tie.
                        SELECT context_id, start_index, end_index, best_distance
                        FROM (
                            SELECT
                                cd.context_id,
                                cd.start_index,
                                cd.end_index,
                                cd.distance as best_distance,
                                ROW_NUMBER() OVER (
                                    PARTITION BY cd.context_id ORDER BY cd.distance
                                ) as rn
                            FROM chunk_distances cd
                        )
                        WHERE rn = 1
                    )
                    SELECT
                        ce.id,
                        ce.thread_id,
                        ce.source,
                        ce.content_type,
                        ce.text_content,
                        ce.metadata,
                        ce.summary,
                        ce.created_at,
                        ce.updated_at,
                        bc.best_distance as distance,
                        bc.start_index as matched_chunk_start,
                        bc.end_index as matched_chunk_end
                    FROM best_chunks bc
                    JOIN context_entries ce ON ce.id = bc.context_id
                    ORDER BY bc.best_distance
                    LIMIT ? OFFSET ?
                '''

                params = filter_params + [query_blob, limit, offset]

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]

                # Calculate execution time and build stats
                execution_time_ms = (time_module.time() - start_time) * 1000
                stats: dict[str, Any] = {
                    'execution_time_ms': round(execution_time_ms, 2),
                    'filters_applied': filter_count,
                    'rows_returned': len(results),
                    'backend': 'sqlite',
                }

                # Get query plan if requested
                if explain_query:
                    cursor = conn.execute(f'EXPLAIN QUERY PLAN {query}', params)
                    plan_rows = cursor.fetchall()
                    plan_data: list[str] = []
                    for row in plan_rows:
                        row_dict = dict(row)
                        id_val = row_dict.get('id', '?')
                        parent_val = row_dict.get('parent', '?')
                        notused_val = row_dict.get('notused', '?')
                        detail_val = row_dict.get('detail', '?')
                        formatted = f'id:{id_val} parent:{parent_val} notused:{notused_val} detail:{detail_val}'
                        plan_data.append(formatted)
                    stats['query_plan'] = '\n'.join(plan_data)

                return results, stats

            return await self.backend.execute_read(_search_sqlite)

        # postgresql
        async def _search_postgresql(
            conn: 'asyncpg.Connection',
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            import time as time_module

            start_time = time_module.time()

            filter_conditions = ['1=1']  # Always true, makes building easier
            filter_params: list[Any] = [query_embedding]
            param_position = 2  # Start at 2 because $1 is embedding

            # Count filters applied
            filter_count = 0

            if thread_id:
                filter_conditions.append(f'ce.thread_id = {self._placeholder(param_position)}')
                filter_params.append(thread_id)
                param_position += 1
                filter_count += 1

            if source:
                filter_conditions.append(f'ce.source = {self._placeholder(param_position)}')
                filter_params.append(source)
                param_position += 1
                filter_count += 1

            if content_type:
                filter_conditions.append(f'ce.content_type = {self._placeholder(param_position)}')
                filter_params.append(content_type)
                param_position += 1
                filter_count += 1

            # Tag filter (uses subquery with indexed tag table)
            if tags:
                normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
                if normalized_tags:
                    tag_placeholders = ','.join([
                        self._placeholder(param_position + i) for i in range(len(normalized_tags))
                    ])
                    filter_conditions.append(f'''
                        ce.id IN (
                            SELECT DISTINCT context_entry_id
                            FROM tags
                            WHERE tag IN ({tag_placeholders})
                        )
                    ''')
                    filter_params.extend(normalized_tags)
                    param_position += len(normalized_tags)
                    filter_count += 1

            # Date range filtering - PostgreSQL uses TIMESTAMPTZ comparison
            # asyncpg requires Python datetime objects, not strings, for TIMESTAMPTZ parameters
            if start_date:
                filter_conditions.append(f'ce.created_at >= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(start_date))
                param_position += 1
                filter_count += 1

            if end_date:
                filter_conditions.append(f'ce.created_at <= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(end_date))
                param_position += 1
                filter_count += 1

            # Metadata filtering using MetadataQueryBuilder
            metadata_filter_count = 0
            if metadata or metadata_filters:
                from pydantic import ValidationError

                from app.metadata_types import MetadataFilter
                from app.query_builder import MetadataQueryBuilder

                # param_offset is the current number of params minus 1 because MetadataQueryBuilder
                # uses 1-based indexing and we need to continue from the current position
                metadata_builder = MetadataQueryBuilder(
                    backend_type='postgresql',
                    param_offset=len(filter_params),
                    table_alias='ce',
                )

                validation_errors: list[str] = []

                # Simple metadata filters (key=value equality). An invalid KEY is reported
                # as a structured validation error (NOT silently dropped, which would widen
                # the result set), consistent with search_context and the advanced filters.
                if metadata:
                    for key, value in metadata.items():
                        try:
                            metadata_builder.add_simple_filter(key, value)
                            metadata_filter_count += 1
                        except ValueError as e:
                            validation_errors.append(f'Invalid metadata key {key!r}: {e}')

                # Advanced metadata filters with operators
                if metadata_filters:
                    for filter_dict in metadata_filters:
                        try:
                            filter_spec = MetadataFilter(**filter_dict)
                            metadata_builder.add_advanced_filter(filter_spec)
                            metadata_filter_count += 1
                        except ValidationError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except ValueError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except Exception as e:
                            error_msg = f'Unexpected error in metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                            logger.error(f'Unexpected error processing metadata filter: {e}')

                # Raise if ANY (simple key or advanced filter) validation failed, unified
                # with search_context's structured short-circuit behavior.
                if validation_errors:
                    raise MetadataFilterValidationError(
                        'Metadata filter validation failed',
                        validation_errors,
                    )

                # The builder emits the metadata conditions already qualified with the
                # 'ce.' table alias (table_alias='ce'), matching the context_entries
                # JOIN target without rewriting the built SQL (a global str.replace
                # would corrupt JSON keys that contain the substring 'metadata').
                metadata_clause, metadata_params = metadata_builder.build_where_clause()
                if metadata_clause:
                    filter_conditions.append(metadata_clause)
                    filter_params.extend(metadata_params)
                    param_position += len(metadata_params)
                    filter_count += metadata_filter_count

            where_clause = ' AND '.join(filter_conditions)

            # Use CTE with DISTINCT ON to preserve best chunk boundaries
            # DISTINCT ON selects first row per context_id when ordered by distance
            query = f'''
                    WITH chunk_distances AS (
                        SELECT
                            ve.context_id,
                            ve.start_index,
                            ve.end_index,
                            ve.embedding <-> {self._placeholder(1)} as distance
                        FROM vec_context_embeddings ve
                        JOIN context_entries ce ON ce.id = ve.context_id
                        WHERE {where_clause}
                    ),
                    best_chunks AS (
                        SELECT DISTINCT ON (context_id)
                            context_id,
                            start_index,
                            end_index,
                            distance as best_distance
                        FROM chunk_distances
                        ORDER BY context_id, distance
                    )
                    SELECT
                        ce.id,
                        ce.thread_id,
                        ce.source,
                        ce.content_type,
                        ce.text_content,
                        ce.metadata,
                        ce.summary,
                        ce.created_at,
                        ce.updated_at,
                        bc.best_distance as distance,
                        bc.start_index as matched_chunk_start,
                        bc.end_index as matched_chunk_end
                    FROM best_chunks bc
                    JOIN context_entries ce ON ce.id = bc.context_id
                    ORDER BY bc.best_distance
                    LIMIT {self._placeholder(param_position)} OFFSET {self._placeholder(param_position + 1)}
                '''

            filter_params.extend([limit, offset])

            rows = await conn.fetch(query, *filter_params)
            results = [dict(row) for row in rows]

            # Calculate execution time and build stats
            execution_time_ms = (time_module.time() - start_time) * 1000
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': filter_count,
                'rows_returned': len(results),
                'backend': 'postgresql',
            }

            # Get query plan if requested
            if explain_query:
                plan_result = await conn.fetch(f'EXPLAIN {query}', *filter_params)
                plan_data = [str(row[0]) for row in plan_result]
                stats['query_plan'] = '\n'.join(plan_data)

            return results, stats

        return await self.backend.execute_read(_search_postgresql)

    async def search_compressed(
        self,
        query_embedding: list[float],
        limit: int = 20,
        offset: int = 0,
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
        content_type: Literal['text', 'multimodal'] | None = None,
        tags: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        explain_query: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """KNN search over compressed embeddings (TurboQuant payloads).

        Algorithm:
            1. Resolve the singleton provenance row + cached provider.
            2. Filter ``context_entries`` exactly as :meth:`search` does to
               narrow the candidate set.
            3. Read the matching rows from
               ``vec_context_embeddings_compressed``.
            4. For ``variant='ip'`` use the provider's unbiased
               inner-product estimator; convert to a distance via
               ``distance = -ip`` so smaller-is-better matches the existing
               result ordering.
            5. For ``variant='mse'`` decode each payload and compute the L2
               distance to the query.
            6. Aggregate per ``context_id`` (MIN distance per context,
               matches the best-chunk-per-context semantics of the fp32
               read path).
            7. Sort ASC by distance, slice ``[offset : offset + limit]``,
               hydrate from ``context_entries``.

        Return shape is IDENTICAL to :meth:`search` so the calling tool
        layer needs no compression-specific branching.

        Args:
            query_embedding: Query vector for similarity search.
            limit: Maximum number of results to return.
            offset: Number of results to skip (pagination).
            thread_id: Optional filter by thread.
            source: Optional filter by source type.
            content_type: Filter by content type (text or multimodal).
            tags: Filter by any of these tags (OR logic).
            start_date: Filter by created_at >= date (ISO 8601 format).
            end_date: Filter by created_at <= date (ISO 8601 format).
            metadata: Simple metadata filters (key=value equality).
            metadata_filters: Advanced metadata filters with operators.
            explain_query: If True, include query execution plan in stats.

        Returns:
            Tuple of (search results list, statistics dictionary). Each
            result carries ``distance``, ``matched_chunk_start``,
            ``matched_chunk_end`` plus the standard ``context_entries``
            columns.

        Raises:
            RuntimeError: If the singleton compression provenance row is
                missing or the query embedding length does not match the
                stored compression dimension.
        """
        import time as time_module

        import numpy as np

        from app.compression import get_cached_compression_provider

        provider = await get_cached_compression_provider()
        comp_meta = await _get_cached_compression_metadata(self.backend)
        variant = comp_meta.variant

        if len(query_embedding) != comp_meta.dim:
            raise RuntimeError(
                f'query_embedding length {len(query_embedding)} does not '
                f'match compression metadata dim {comp_meta.dim}',
            )

        start_time = time_module.time()
        query_matrix = np.asarray([query_embedding], dtype=np.float32)

        # Compute filter count for the stats dict in the same way the fp32
        # search does so external observers see a consistent shape.
        filter_count = 0
        if thread_id:
            filter_count += 1
        if source:
            filter_count += 1
        if content_type:
            filter_count += 1
        normalized_tags: list[str] = []
        if tags:
            normalized_tags = [t.strip().lower() for t in tags if t.strip()]
            if normalized_tags:
                filter_count += 1
        if start_date:
            filter_count += 1
        if end_date:
            filter_count += 1

        # Build the candidate-id query against context_entries reusing the
        # same building blocks as :meth:`search` (tags, dates, metadata).
        if self.backend.backend_type == 'sqlite':

            def _candidates_sqlite(
                conn: sqlite3.Connection,
            ) -> tuple[list[str], int, str | None]:
                conditions: list[str] = []
                params: list[Any] = []
                metadata_filter_count = 0

                if thread_id:
                    conditions.append('thread_id = ?')
                    params.append(thread_id)
                if source:
                    conditions.append('source = ?')
                    params.append(source)
                if content_type:
                    conditions.append('content_type = ?')
                    params.append(content_type)
                if normalized_tags:
                    placeholders = ','.join(['?' for _ in normalized_tags])
                    conditions.append(
                        'id IN (SELECT DISTINCT context_entry_id '
                        f'FROM tags WHERE tag IN ({placeholders}))',
                    )
                    params.extend(normalized_tags)
                if start_date:
                    conditions.append('created_at >= datetime(?)')
                    params.append(start_date)
                if end_date:
                    conditions.append('created_at <= datetime(?)')
                    params.append(end_date)

                if metadata or metadata_filters:
                    from pydantic import ValidationError

                    from app.metadata_types import MetadataFilter
                    from app.query_builder import MetadataQueryBuilder

                    builder = MetadataQueryBuilder(backend_type='sqlite')
                    errs: list[str] = []
                    # An invalid simple-metadata KEY is reported as a structured validation
                    # error (NOT silently dropped, which would widen the result set),
                    # consistent with search_context and the advanced filters below.
                    if metadata:
                        for key, value in metadata.items():
                            try:
                                builder.add_simple_filter(key, value)
                                metadata_filter_count += 1
                            except ValueError as e:
                                errs.append(f'Invalid metadata key {key!r}: {e}')
                    if metadata_filters:
                        for filter_dict in metadata_filters:
                            try:
                                spec = MetadataFilter(**filter_dict)
                                builder.add_advanced_filter(spec)
                                metadata_filter_count += 1
                            except (ValidationError, ValueError) as e:
                                errs.append(
                                    f'Invalid metadata filter {filter_dict}: {e}',
                                )
                            except Exception as e:
                                errs.append(
                                    f'Unexpected error in metadata filter '
                                    f'{filter_dict}: {e}',
                                )
                                logger.error(
                                    'Unexpected error processing metadata filter: %s',
                                    e,
                                )
                    if errs:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed', errs,
                        )

                    clause, mparams = builder.build_where_clause()
                    if clause:
                        conditions.append(clause)
                        params.extend(mparams)

                where_clause = (
                    f'WHERE {" AND ".join(conditions)}' if conditions else ''
                )
                cand_sql = f'SELECT id FROM context_entries {where_clause}'
                cursor = conn.execute(cand_sql, params)
                cand_ids = [str(row[0]) for row in cursor.fetchall()]

                plan: str | None = None
                if explain_query:
                    cursor = conn.execute(
                        f'EXPLAIN QUERY PLAN {cand_sql}', params,
                    )
                    plan_rows = cursor.fetchall()
                    formatted: list[str] = []
                    for row in plan_rows:
                        row_dict = dict(row)
                        formatted.append(
                            f"id:{row_dict.get('id', '?')} "
                            f"parent:{row_dict.get('parent', '?')} "
                            f"notused:{row_dict.get('notused', '?')} "
                            f"detail:{row_dict.get('detail', '?')}",
                        )
                    plan = '\n'.join(formatted)
                return cand_ids, metadata_filter_count, plan

            candidate_ids, meta_count, query_plan = await self.backend.execute_read(
                _candidates_sqlite,
            )
        else:
            async def _candidates_pg(
                conn: 'asyncpg.Connection',
            ) -> tuple[list[str], int, str | None]:
                conditions: list[str] = ['1=1']
                params: list[Any] = []
                position = 1
                metadata_filter_count = 0

                if thread_id:
                    conditions.append(f'ce.thread_id = ${position}')
                    params.append(thread_id)
                    position += 1
                if source:
                    conditions.append(f'ce.source = ${position}')
                    params.append(source)
                    position += 1
                if content_type:
                    conditions.append(f'ce.content_type = ${position}')
                    params.append(content_type)
                    position += 1
                if normalized_tags:
                    placeholders = ','.join(
                        f'${position + i}' for i in range(len(normalized_tags))
                    )
                    conditions.append(
                        'ce.id IN (SELECT DISTINCT context_entry_id '
                        f'FROM tags WHERE tag IN ({placeholders}))',
                    )
                    params.extend(normalized_tags)
                    position += len(normalized_tags)
                if start_date:
                    conditions.append(f'ce.created_at >= ${position}')
                    params.append(self._parse_date_for_postgresql(start_date))
                    position += 1
                if end_date:
                    conditions.append(f'ce.created_at <= ${position}')
                    params.append(self._parse_date_for_postgresql(end_date))
                    position += 1

                if metadata or metadata_filters:
                    from pydantic import ValidationError

                    from app.metadata_types import MetadataFilter
                    from app.query_builder import MetadataQueryBuilder

                    builder = MetadataQueryBuilder(
                        backend_type='postgresql',
                        param_offset=len(params),
                        table_alias='ce',
                    )
                    errs: list[str] = []
                    # An invalid simple-metadata KEY is reported as a structured validation
                    # error (NOT silently dropped, which would widen the result set),
                    # consistent with search_context and the advanced filters below.
                    if metadata:
                        for key, value in metadata.items():
                            try:
                                builder.add_simple_filter(key, value)
                                metadata_filter_count += 1
                            except ValueError as e:
                                errs.append(f'Invalid metadata key {key!r}: {e}')
                    if metadata_filters:
                        for filter_dict in metadata_filters:
                            try:
                                spec = MetadataFilter(**filter_dict)
                                builder.add_advanced_filter(spec)
                                metadata_filter_count += 1
                            except (ValidationError, ValueError) as e:
                                errs.append(
                                    f'Invalid metadata filter {filter_dict}: {e}',
                                )
                            except Exception as e:
                                errs.append(
                                    f'Unexpected error in metadata filter '
                                    f'{filter_dict}: {e}',
                                )
                                logger.error(
                                    'Unexpected error processing metadata filter: %s',
                                    e,
                                )
                    if errs:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed', errs,
                        )
                    clause, mparams = builder.build_where_clause()
                    if clause:
                        # The builder emits the clause already qualified with the 'ce.'
                        # alias (table_alias='ce'), matching the JOIN target -- no
                        # str.replace that would corrupt 'metadata'-containing keys.
                        conditions.append(clause)
                        params.extend(mparams)
                        position += len(mparams)

                where_clause = ' AND '.join(conditions)
                cand_sql = (
                    f'SELECT ce.id FROM context_entries ce WHERE {where_clause}'
                )
                rows = await conn.fetch(cand_sql, *params)

                def _build_ids() -> list[str]:
                    return [str(row['id']) for row in rows]

                # O(N-contexts) id construction on the event loop for PostgreSQL;
                # offload a large candidate set, mirroring _read_compressed_pg.
                cand_ids = (
                    await asyncio.to_thread(_build_ids)
                    if len(rows) > _COMPRESSED_OFFLOAD_MIN_ROWS
                    else _build_ids()
                )

                plan: str | None = None
                if explain_query:
                    plan_rows = await conn.fetch(
                        f'EXPLAIN {cand_sql}', *params,
                    )
                    plan = '\n'.join(str(row[0]) for row in plan_rows)
                return cand_ids, metadata_filter_count, plan

            candidate_ids, meta_count, query_plan = await self.backend.execute_read(
                cast(Any, _candidates_pg),
            )

        filter_count += meta_count

        if not candidate_ids:
            stats: dict[str, Any] = {
                'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
                'filters_applied': filter_count,
                'rows_returned': 0,
                'backend': self.backend.backend_type,
            }
            if explain_query and query_plan is not None:
                stats['query_plan'] = query_plan
            return [], stats

        # Read compressed rows for the candidate context_ids.
        if self.backend.backend_type == 'sqlite':

            def _read_compressed_sqlite(
                conn: sqlite3.Connection,
            ) -> list[tuple[str, int, int, int, bytes]]:
                rows: list[tuple[str, int, int, int, bytes]] = []
                for start in range(0, len(candidate_ids), _SQLITE_IN_CLAUSE_BATCH):
                    batch = candidate_ids[start:start + _SQLITE_IN_CLAUSE_BATCH]
                    placeholders = ','.join('?' for _ in batch)
                    cursor = conn.execute(
                        'SELECT context_id, chunk_index, start_index, end_index, '
                        f'payload FROM vec_context_embeddings_compressed '
                        f'WHERE context_id IN ({placeholders})',
                        batch,
                    )
                    rows.extend(
                        (str(r[0]), int(r[1]), int(r[2]), int(r[3]), bytes(r[4]))
                        for r in cursor.fetchall()
                    )
                return rows

            payload_rows = await self.backend.execute_read(_read_compressed_sqlite)
        else:
            async def _read_compressed_pg(
                conn: 'asyncpg.Connection',
            ) -> list[tuple[str, int, int, int, bytes]]:
                rows = await conn.fetch(
                    'SELECT context_id, chunk_index, start_index, end_index, '
                    'payload FROM vec_context_embeddings_compressed '
                    'WHERE context_id = ANY($1::uuid[])',
                    candidate_ids,
                )

                def _build_rows() -> list[tuple[str, int, int, int, bytes]]:
                    return [
                        (
                            str(r['context_id']),
                            int(r['chunk_index']),
                            int(r['start_index']),
                            int(r['end_index']),
                            bytes(r['payload']),
                        )
                        for r in rows
                    ]

                # Building payload_rows copies each BYTEA payload (bytes(...)) and is
                # O(N-chunks) pure-Python over the unbounded candidate set; on
                # PostgreSQL this async read callable runs on the event loop (the
                # SQLite callable already runs inside execute_read's worker thread), so
                # a large candidate set is offloaded so it cannot pin the loop (see
                # _COMPRESSED_OFFLOAD_MIN_ROWS). The fetched asyncpg Records are fully
                # materialized and detached, so accessing them off-loop is safe.
                if len(rows) > _COMPRESSED_OFFLOAD_MIN_ROWS:
                    return await asyncio.to_thread(_build_rows)
                return _build_rows()

            payload_rows = await self.backend.execute_read(cast(Any, _read_compressed_pg))

        if not payload_rows:
            stats = {
                'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
                'filters_applied': filter_count,
                'rows_returned': 0,
                'backend': self.backend.backend_type,
            }
            if explain_query and query_plan is not None:
                stats['query_plan'] = query_plan
            return [], stats

        from app.compression.providers.turboquant._types import IPPayload
        from app.compression.providers.turboquant._types import MSEPayload
        from app.compression.providers.turboquant._types import payload_from_bytes

        def _decode_and_concat() -> bytes:
            # Decode every stored payload via the wire-format dispatcher and
            # concatenate same-variant payloads into ONE synthetic payload so the
            # provider's scoring call runs exactly ONCE per query (O(1) GEMM)
            # instead of once per candidate row. This decode+concat is O(N-chunks)
            # pure-Python CPU (payload_from_bytes does struct.unpack + frombuffer
            # copies per row); the caller offloads it for a large candidate set
            # (see _COMPRESSED_OFFLOAD_MIN_ROWS) so it cannot pin the event loop.
            decoded_payloads: list[MSEPayload | IPPayload] = [
                payload_from_bytes(payload_bytes)
                for _, _, _, _, payload_bytes in payload_rows
            ]
            if variant == 'ip':
                ip_subtypes: list[IPPayload] = []
                for p in decoded_payloads:
                    if not isinstance(p, IPPayload):
                        raise RuntimeError(
                            f'Compression metadata variant=ip but stored payload is '
                            f'{type(p).__name__}; storage corruption suspected',
                        )
                    ip_subtypes.append(p)
                return IPPayload.concat(ip_subtypes).to_bytes()
            mse_subtypes: list[MSEPayload] = []
            for p in decoded_payloads:
                if not isinstance(p, MSEPayload):
                    raise RuntimeError(
                        f'Compression metadata variant=mse but stored payload is '
                        f'{type(p).__name__}; storage corruption suspected',
                    )
                mse_subtypes.append(p)
            return MSEPayload.concat(mse_subtypes).to_bytes()

        if len(payload_rows) > _COMPRESSED_OFFLOAD_MIN_ROWS:
            combined_bytes = await asyncio.to_thread(_decode_and_concat)
        else:
            combined_bytes = _decode_and_concat()

        # Distance polarity convention (matches the fp32 read path):
        #   smaller distance = closer / more similar. The provider's async GEMM
        # wrapper offloads the matmul via asyncio.to_thread; the per-chunk distances
        # + per-context MIN aggregation + sort below are themselves O(N-chunks)
        # pure-Python/numpy work over the same unbounded candidate set, so they are
        # offloaded too (see _COMPRESSED_OFFLOAD_MIN_ROWS). Together with the decode
        # above, this leaves NO per-chunk work on the event loop for a large search;
        # only the bounded offset/limit page slice + hydration run inline.
        if variant == 'ip':
            # estimate_inner_product returns shape (nq, n_total); nq=1 slices to a
            # 1-D array of length n_total in payload_rows order (concat preserves it).
            gemm_output = await provider.estimate_inner_product(combined_bytes, query_matrix)
        else:  # variant == 'mse'
            # decode returns (n_total, d) for an L2 distance to the single query vector.
            gemm_output = await provider.decode(combined_bytes)

        def _rank_from_gemm() -> list[tuple[str, tuple[float, int, int]]]:
            if variant == 'ip':
                # IP negated so a larger IP (more similar) maps to a smaller distance.
                distances = [-float(x) for x in gemm_output[0].tolist()]
            else:
                diff = gemm_output - query_matrix[0]
                distances = [float(x) for x in np.linalg.norm(diff, axis=1).tolist()]
            best_by_context: dict[str, tuple[float, int, int]] = {}
            for (context_id, _chunk_index, start_index, end_index, _payload), dist in zip(
                payload_rows, distances, strict=True,
            ):
                current = best_by_context.get(context_id)
                if current is None or dist < current[0]:
                    best_by_context[context_id] = (dist, start_index, end_index)
            return sorted(best_by_context.items(), key=lambda kv: kv[1][0])

        if len(payload_rows) > _COMPRESSED_OFFLOAD_MIN_ROWS:
            ranked = await asyncio.to_thread(_rank_from_gemm)
        else:
            ranked = _rank_from_gemm()

        page = ranked[offset : offset + limit]
        if not page:
            stats = {
                'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
                'filters_applied': filter_count,
                'rows_returned': 0,
                'backend': self.backend.backend_type,
            }
            if explain_query and query_plan is not None:
                stats['query_plan'] = query_plan
            return [], stats

        page_ids = [cid for cid, _ in page]

        # Hydrate the result rows from context_entries.
        if self.backend.backend_type == 'sqlite':

            def _hydrate_sqlite(
                conn: sqlite3.Connection,
            ) -> dict[str, dict[str, Any]]:
                # page_ids is bounded by the overfetch limit, not by the final
                # page size, so it can exceed SQLITE_MAX_VARIABLE_NUMBER; bind it
                # in bounded batches like the compressed candidate read.
                out: dict[str, dict[str, Any]] = {}
                for start in range(0, len(page_ids), _SQLITE_IN_CLAUSE_BATCH):
                    batch = page_ids[start:start + _SQLITE_IN_CLAUSE_BATCH]
                    placeholders = ','.join('?' for _ in batch)
                    cursor = conn.execute(
                        'SELECT id, thread_id, source, content_type, text_content, '
                        'metadata, summary, created_at, updated_at FROM context_entries '
                        f'WHERE id IN ({placeholders})',
                        batch,
                    )
                    out.update({str(dict(r)['id']): dict(r) for r in cursor.fetchall()})
                return out

            hydrated = await self.backend.execute_read(_hydrate_sqlite)
        else:
            async def _hydrate_pg(
                conn: 'asyncpg.Connection',
            ) -> dict[str, dict[str, Any]]:
                rows = await conn.fetch(
                    'SELECT id, thread_id, source, content_type, text_content, '
                    'metadata, summary, created_at, updated_at FROM context_entries '
                    'WHERE id = ANY($1::uuid[])',
                    page_ids,
                )
                return {str(dict(r)['id']): dict(r) for r in rows}

            hydrated = await self.backend.execute_read(cast(Any, _hydrate_pg))

        results: list[dict[str, Any]] = []
        for context_id, (dist, start_index, end_index) in page:
            row = hydrated.get(context_id)
            if row is None:
                # Compressed row points at a context_id that vanished
                # between the candidate scan and the hydration query
                # (concurrent deletion). Skip rather than fabricate a row.
                continue
            row['distance'] = dist
            row['matched_chunk_start'] = start_index
            row['matched_chunk_end'] = end_index
            results.append(row)

        stats = {
            'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
            'filters_applied': filter_count,
            'rows_returned': len(results),
            'backend': self.backend.backend_type,
        }
        if explain_query and query_plan is not None:
            stats['query_plan'] = query_plan
        return results, stats

    async def update(
        self,
        context_id: str,
        chunk_embeddings: list[ChunkEmbedding],
        model: str,
    ) -> None:
        """Update embeddings for a context entry (delete old, store new).

        This method replaces all existing chunk embeddings with new ones atomically.
        Uses delete-all + store-chunked pattern for consistency.

        Args:
            context_id: ID of the context entry
            chunk_embeddings: New ChunkEmbedding objects (embedding + boundaries)
            model: Model identifier
        """
        # Delete existing chunks
        await self.delete_all_chunks(context_id)

        # Store new chunks
        await self.store_chunked(context_id, chunk_embeddings, model)

        logger.debug(f'Updated {len(chunk_embeddings)} embeddings for context {context_id}')

    async def delete(self, context_id: str) -> None:
        """Delete all embeddings for a context entry.

        Delegates to delete_all_chunks() for proper cleanup of chunked embeddings.

        Args:
            context_id: ID of the context entry
        """
        await self.delete_all_chunks(context_id)

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
                return _exists_sqlite(cast(sqlite3.Connection, txn.connection))
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
                return _tables_exist_sqlite(cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_read(_tables_exist_sqlite)

        # postgresql: to_regclass resolves via the connection's search_path and returns
        # NULL for a missing relation (no UndefinedTableError).
        async def _tables_exist_postgresql(conn: 'asyncpg.Connection') -> bool:
            return await conn.fetchval("SELECT to_regclass('embedding_metadata')") is not None

        if txn is not None:
            return await _tables_exist_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_read(cast(Any, _tables_exist_postgresql))

    async def get_statistics(self, thread_id: str | None = None) -> dict[str, Any]:
        """Get embedding statistics including chunk information.

        The chunk-count source is ``embedding_metadata.chunk_count`` (summed)
        on BOTH backends, regardless of compression mode. This is the single
        source-of-truth populated by every write path (fp32 SQLite, fp32
        PostgreSQL, compressed SQLite, compressed PostgreSQL), so the
        reported chunk total stays correct whether compressed payloads or
        fp32 vectors are stored on disk.

        Args:
            thread_id: Optional filter by thread

        Returns:
            Dictionary with statistics (count, coverage, chunk info, etc.)
        """
        if self.backend.backend_type == 'sqlite':

            def _get_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                if thread_id:
                    query1 = f'SELECT COUNT(*) FROM context_entries WHERE thread_id = {self._placeholder(1)}'
                    cursor = conn.execute(query1, (thread_id,))
                else:
                    cursor = conn.execute('SELECT COUNT(*) FROM context_entries')

                total_entries = cursor.fetchone()[0]

                if thread_id:
                    query2 = f'''
                        SELECT COUNT(*)
                        FROM embedding_metadata em
                        JOIN context_entries ce ON em.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                    cursor = conn.execute(query2, (thread_id,))
                else:
                    cursor = conn.execute('SELECT COUNT(*) FROM embedding_metadata')

                embedding_count = cursor.fetchone()[0]

                # Chunk total is the sum of embedding_metadata.chunk_count,
                # the single source-of-truth populated by every write path
                # (fp32 + compressed) so the value stays correct in every
                # backend/compression-mode combination.
                if thread_id:
                    query3 = f'''
                        SELECT COALESCE(SUM(em.chunk_count), 0)
                        FROM embedding_metadata em
                        JOIN context_entries ce ON em.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                    cursor = conn.execute(query3, (thread_id,))
                else:
                    cursor = conn.execute('SELECT COALESCE(SUM(chunk_count), 0) FROM embedding_metadata')

                total_chunks = cursor.fetchone()[0]

                coverage_percentage = (embedding_count / total_entries * 100) if total_entries > 0 else 0.0
                average_chunks = round(total_chunks / embedding_count, 2) if embedding_count > 0 else 0.0

                return {
                    'total_embeddings': embedding_count,
                    'total_entries': total_entries,
                    'total_chunks': total_chunks,
                    'average_chunks_per_entry': average_chunks,
                    'coverage_percentage': round(coverage_percentage, 2),
                    'backend': 'sqlite',
                }

            return await self.backend.execute_read(_get_stats_sqlite)

        # postgresql
        async def _get_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            if thread_id:
                query1 = f'SELECT COUNT(*) FROM context_entries WHERE thread_id = {self._placeholder(1)}'
                total_entries = await conn.fetchval(query1, thread_id)
            else:
                total_entries = await conn.fetchval('SELECT COUNT(*) FROM context_entries')

            if thread_id:
                query2 = f'''
                        SELECT COUNT(*)
                        FROM embedding_metadata em
                        JOIN context_entries ce ON em.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                embedding_count = await conn.fetchval(query2, thread_id)
            else:
                embedding_count = await conn.fetchval('SELECT COUNT(*) FROM embedding_metadata')

            # Chunk total is the sum of embedding_metadata.chunk_count,
            # the single source-of-truth populated by every write path
            # (fp32 + compressed). On PostgreSQL the compression migration
            # drops vec_context_embeddings, so reading the count from
            # embedding_metadata is the only query that succeeds in both
            # compression-enabled and compression-disabled deployments.
            if thread_id:
                query3 = f'''
                        SELECT COALESCE(SUM(em.chunk_count), 0)
                        FROM embedding_metadata em
                        JOIN context_entries ce ON em.context_id = ce.id
                        WHERE ce.thread_id = {self._placeholder(1)}
                    '''
                total_chunks = await conn.fetchval(query3, thread_id)
            else:
                total_chunks = await conn.fetchval(
                    'SELECT COALESCE(SUM(chunk_count), 0) FROM embedding_metadata',
                )

            coverage_percentage = (embedding_count / total_entries * 100) if total_entries > 0 else 0.0
            average_chunks = round(total_chunks / embedding_count, 2) if embedding_count > 0 else 0.0

            return {
                'total_embeddings': embedding_count,
                'total_entries': total_entries,
                'total_chunks': total_chunks,
                'average_chunks_per_entry': average_chunks,
                'coverage_percentage': round(coverage_percentage, 2),
                'backend': 'postgresql',
            }

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

    async def get_table_dimension(self) -> int | None:
        """Get the dimension of the existing vector table.

        This is useful for diagnostics and validation to check if the configured
        EMBEDDING_DIM matches the actual table dimension.

        Returns:
            Dimension of existing embeddings, or None if no embeddings exist
        """
        if self.backend.backend_type == 'sqlite':

            def _get_dimension_sqlite(conn: sqlite3.Connection) -> int | None:
                cursor = conn.execute('SELECT dimensions FROM embedding_metadata LIMIT 1')
                row = cursor.fetchone()
                return row[0] if row else None

            return await self.backend.execute_read(_get_dimension_sqlite)

        # postgresql
        async def _get_dimension_postgresql(conn: 'asyncpg.Connection') -> int | None:
            row = await conn.fetchrow('SELECT dimensions FROM embedding_metadata LIMIT 1')
            return row['dimensions'] if row else None

        return await self.backend.execute_read(_get_dimension_postgresql)
