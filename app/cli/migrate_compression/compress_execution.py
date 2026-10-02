"""Transactional fp32-to-compressed data movement for ``--compress``, one branch per backend."""

import logging
import sqlite3
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.cli.migrate_compression import storage
from app.cli.migrate_compression.storage import fp32_blob_to_list_sqlite
from app.cli.migrate_compression.storage import make_2d_array
from app.cli.migrate_compression.storage import raise_pg_migration_budget
from app.compression.base import CompressionProvider
from app.compression.types import CompressionMetadata
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations._pg_ddl import fetch_migration
from app.migrations._pg_ddl import fetchval_migration
from app.migrations.compression import ensure_sqlite_codebook_fingerprint_column
from app.settings import get_settings

logger = logging.getLogger(__name__)


async def execute_compress(
    *,
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
) -> None:
    """Encode every fp32 row and write the compressed payload table.

    The DATA MOVEMENT runs inside a single
    :meth:`StorageBackend.begin_transaction` transaction on both backends,
    committing on success and rolling back on exception. The SCHEMA step does not
    share that guarantee on SQLite, where the ``IF NOT EXISTS`` table DDL autocommits
    through ``executescript()`` before ``BEGIN IMMEDIATE`` opens (see the branch
    docstrings); on PostgreSQL the DDL is transactional and rolls back with the rest. Each branch locks
    the source table (``BEGIN IMMEDIATE`` / ``LOCK TABLE ... ACCESS
    EXCLUSIVE``) BEFORE streaming and recounts it INSIDE the transaction, so
    no planning-time row count is consumed here.
    """
    if backend.backend_type == 'sqlite':
        await _execute_compress_sqlite(
            backend, provider, provenance,
        )
        return
    await _execute_compress_postgresql(
        backend, provider, provenance,
    )


async def _execute_compress_sqlite(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
) -> None:
    """SQLite branch of :func:`execute_compress`.

    Streams fp32 rows in batches of :data:`storage.MIGRATION_BATCH_SIZE` from
    ``vec_context_embeddings`` (joined to ``embedding_chunks`` for the
    chunk metadata), encodes each batch in-process, and INSERTs each
    batch into ``vec_context_embeddings_compressed``. The whole streamed
    migration -- every batch INSERT plus the provenance row and the
    source table DROP -- runs inside ONE
    :meth:`StorageBackend.begin_transaction` so any failure rolls back
    all work and the source table remains intact.

    Bounded peak memory: ``O(storage.MIGRATION_BATCH_SIZE * dim * 4)`` bytes
    (~40 MB at ``dim == 1024``).

    Raises:
        ValueError: If the streamed read did not cover every fp32 row in
            ``vec_context_embeddings`` (an orphan lacking an embedding_chunks
            bridge), so the transaction aborts rather than dropping unread data.
    """
    settings = get_settings()
    dim = settings.embedding.dim
    batch_size = storage.MIGRATION_BATCH_SIZE

    async with backend.begin_transaction() as txn:
        conn = cast(sqlite3.Connection, txn.connection)

        # Create the compressed tables inline (we cannot use the
        # migration loader because it drops vec_context_embeddings up
        # front, which would invalidate the source rows being streamed
        # in this transaction).
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                payload BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_vec_compressed_context
                ON vec_context_embeddings_compressed(context_id);
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
                variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
                seed INTEGER NOT NULL CHECK (seed >= 0),
                dim INTEGER NOT NULL CHECK (dim > 0),
                codebook_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )

        # This CLI creates the table itself -- deliberately, because the migration
        # loader's leading DROP would remove the fp32 source table mid-stream -- so
        # it shares the loader's idempotent fingerprint-column add rather than
        # repeating it. Without it, a table created by a build predating the column
        # would reach the seven-column provenance INSERT below and fail with 'no
        # such column' AFTER the entire encode pass had already run.
        ensure_sqlite_codebook_fingerprint_column(conn)

        # Take the write lock BEFORE the first streamed read: sqlite3's
        # legacy transaction control would otherwise open the implicit
        # transaction only at the first INSERT, so a concurrent-process
        # writer could commit between the planning count / early batches and
        # that first INSERT (an equal-count replacement would slip past the
        # count guard below and the DROP would destroy never-encoded rows).
        # BEGIN IMMEDIATE freezes the database for the whole stream + swap;
        # executescript above ran first because it COMMITS any open
        # transaction (its IF NOT EXISTS DDL is idempotent autocommit).
        conn.execute('BEGIN IMMEDIATE')

        # Idempotency check: a populated provenance singleton means the
        # migration has already been applied. Re-running is a no-op.
        existing_provenance = conn.execute(
            'SELECT COUNT(*) FROM compression_metadata WHERE id = 1',
        ).fetchone()
        if existing_provenance and int(existing_provenance[0]) == 1:
            logger.info(
                'Compression migration already applied (provenance row '
                'present); --compress is a no-op.',
            )
            return

        # Stream fp32 rows in batches; encode + INSERT each batch.
        # ORDER BY (context_id, id) is stable so the batch composition is
        # deterministic across runs and per-batch memory is bounded.
        rows_processed = 0
        offset = 0
        # chunk_index is the per-context sequential position (0, 1, 2, ...),
        # matching the live compressed write path so the on-disk contract is
        # identical regardless of how a compressed row was produced. The
        # streamed read is ordered by (context_id, id), so a context's chunks
        # arrive contiguously and in order across batch boundaries.
        current_ctx: str | None = None
        chunk_seq = 0
        while True:
            cur = conn.execute(
                'SELECT ec.context_id, ec.id, ec.start_index, ec.end_index, '
                'v.embedding FROM embedding_chunks ec '
                'JOIN vec_context_embeddings v ON v.rowid = ec.vec_rowid '
                'ORDER BY ec.context_id, ec.id '
                'LIMIT ? OFFSET ?',
                (batch_size, offset),
            )
            batch = cur.fetchall()
            if not batch:
                break
            for ctx_id, _chunk_id, start_index, end_index, blob in batch:
                ctx_str = str(ctx_id)
                if ctx_str != current_ctx:
                    current_ctx = ctx_str
                    chunk_seq = 0
                else:
                    chunk_seq += 1
                vec_list = fp32_blob_to_list_sqlite(bytes(blob), dim)
                payload = provider.encode_sync(make_2d_array(vec_list))
                conn.execute(
                    'INSERT INTO vec_context_embeddings_compressed '
                    '(context_id, chunk_index, start_index, end_index, payload) '
                    'VALUES (?, ?, ?, ?, ?)',
                    (
                        ctx_str, chunk_seq, int(start_index),
                        int(end_index), payload,
                    ),
                )
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard: the streamed read joins embedding_chunks to
        # vec_context_embeddings, so a vec row with no embedding_chunks bridge
        # (an orphan from a corrupted source) would be silently skipped and then
        # destroyed by the DROP below. Recount INSIDE the write transaction
        # (the planning-time count ran outside it and may predate a
        # concurrent commit that BEGIN IMMEDIATE has since locked out) and
        # abort rather than drop unread fp32 data, mirroring the decompress
        # rebuild's mismatch guard.
        live_count = int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings',
            ).fetchone()[0],
        )
        if rows_processed != live_count:
            raise ValueError(
                f'compress read {rows_processed} fp32 row(s) but '
                f'vec_context_embeddings holds {live_count} '
                f'(mismatch of {abs(live_count - rows_processed)} row(s); fewer reads '
                'mean orphaned vec rows lack an embedding_chunks bridge, more mean '
                'duplicate bridges). Aborting so no fp32 vector is dropped.',
            )

        # Provenance INSERT + source DROP last inside the same
        # transaction. The CHECK (id = 1) constraint guarantees this
        # INSERT is the only one ever applied.
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            'VALUES (1, ?, ?, ?, ?, ?, ?)',
            (
                provenance.provider,
                provenance.bits,
                provenance.variant,
                provenance.seed,
                provenance.dim,
                provenance.codebook_fingerprint,
            ),
        )
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')

    logger.info(
        'Compressed %d fp32 row(s) into vec_context_embeddings_compressed; '
        'dropped legacy table.',
        rows_processed,
    )


async def _execute_compress_postgresql(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
) -> None:
    """PostgreSQL branch of :func:`execute_compress`.

    Streams fp32 rows in batches of :data:`storage.MIGRATION_BATCH_SIZE` from
    ``vec_context_embeddings`` using LIMIT/OFFSET pagination ordered by
    ``(context_id, id)`` and encodes each batch in-process. Server-side
    cursors would keep a portal open against the source table that
    PostgreSQL would reject when the trailing ``DROP TABLE`` runs in the
    same transaction. The source table is locked ``ACCESS EXCLUSIVE``
    BEFORE the first batch: the stream runs under READ COMMITTED, so a
    concurrent writer replacing rows mid-stream with an equal-cardinality
    set would otherwise slip past a count-only guard and the trailing DROP
    would destroy never-encoded rows. Every batch INSERT plus the
    provenance row, the HNSW index DROP, and the source table DROP run
    inside ONE :meth:`StorageBackend.begin_transaction` so any failure
    rolls back all work and the source table remains intact.

    Bounded peak memory: ``O(storage.MIGRATION_BATCH_SIZE * dim * 4)`` bytes
    (~40 MB at ``dim == 1024``).

    Raises:
        ValueError: If the streamed read did not cover every fp32 row the
            in-transaction recount sees under the lock, so the transaction
            aborts rather than dropping unread data.
    """
    batch_size = storage.MIGRATION_BATCH_SIZE
    migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s

    async with backend.begin_transaction() as txn:
        conn = cast('asyncpg.Connection', txn.connection)
        # Raise this transaction's statement budget so the heavy vec-table DDL below
        # (DROP TABLE / DROP INDEX) and the streamed batch reads run under the migration
        # budget, not the pool's ~60s command_timeout. See storage.raise_pg_migration_budget.
        await raise_pg_migration_budget(conn, migration_timeout_s)

        # Bare table names rely on PostgreSQL's search_path resolution;
        # this matches the project-wide pattern in
        # ``app/repositories/embedding_repository/`` and ``postgresql_schema.sql``.
        # Operators using a non-default schema configure ``search_path``
        # accordingly. Create the compressed tables inline (the
        # migration loader would drop the source vec table at this
        # point, invalidating the streamed read).
        await conn.execute(
            'CREATE TABLE IF NOT EXISTS '
            'vec_context_embeddings_compressed ('
            '  id BIGSERIAL PRIMARY KEY,'
            '  context_id UUID NOT NULL,'
            '  chunk_index INTEGER NOT NULL,'
            '  start_index INTEGER NOT NULL DEFAULT 0,'
            '  end_index INTEGER NOT NULL DEFAULT 0,'
            '  payload BYTEA NOT NULL,'
            '  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,'
            '  FOREIGN KEY (context_id) REFERENCES context_entries(id) '
            '  ON DELETE CASCADE'
            ')',
        )
        await conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_vec_compressed_context '
            'ON vec_context_embeddings_compressed(context_id)',
        )
        await conn.execute(
            'CREATE TABLE IF NOT EXISTS compression_metadata ('
            '  id INTEGER PRIMARY KEY CHECK (id = 1),'
            '  provider TEXT NOT NULL,'
            '  bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),'
            "  variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),"
            '  seed BIGINT NOT NULL CHECK (seed >= 0),'
            '  dim INTEGER NOT NULL CHECK (dim > 0),'
            '  codebook_fingerprint TEXT,'
            '  created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP'
            ')',
        )
        # CREATE TABLE IF NOT EXISTS never ADDS a column to an existing table, so a
        # compression_metadata table created by a build predating
        # codebook_fingerprint (a documented in-place-upgrade shape the provenance
        # reader tolerates) survives the statement above unchanged and the
        # six-parameter provenance INSERT below would fail with UndefinedColumnError
        # AFTER the entire encode pass had already run. This CLI creates the table
        # itself -- deliberately, because the migration loader would drop the fp32
        # source table mid-stream -- so it must also mirror that loader's idempotent
        # add. The ALTER is a no-op on a current-shape table.
        await conn.execute(
            'ALTER TABLE compression_metadata ADD COLUMN IF NOT EXISTS codebook_fingerprint TEXT',
        )

        # Idempotency check: skip work if provenance already populated.
        existing_count = await conn.fetchval(
            'SELECT COUNT(*) FROM compression_metadata WHERE id = 1',
        )
        if int(existing_count or 0) == 1:
            logger.info(
                'Compression migration already applied (provenance row '
                'present); --compress is a no-op.',
            )
            return

        # Freeze the source table BEFORE streaming: the batches read under
        # READ COMMITTED, so a concurrent writer replacing rows mid-stream
        # with an equal-cardinality set would defeat a count-only guard --
        # the stream would have encoded the OLD rows and the DROP below
        # would destroy the never-encoded replacements. ACCESS EXCLUSIVE
        # (waiting out in-flight writers, bounded by the migration budget)
        # makes every batch, the recount, and the DROP see one frozen state.
        await execute_migration_ddl(
            conn,
            'LOCK TABLE vec_context_embeddings IN ACCESS EXCLUSIVE MODE',
            migration_timeout_s,
        )
        # Route the recount through the migration budget (not a bare fetchval,
        # which inherits the pool's ~60s command_timeout): this COUNT(*) is the
        # first full-table scan under the lock, and on a large corpus it must
        # use POSTGRESQL_MIGRATION_TIMEOUT_S like the streamed reads that follow.
        live_count = int(
            await fetchval_migration(
                conn,
                'SELECT COUNT(*) FROM vec_context_embeddings',
                migration_timeout_s,
            )
            or 0,
        )

        # Stream fp32 rows in batches via LIMIT/OFFSET pagination.
        # Server-side cursors would keep a portal open against the
        # source table that PostgreSQL would reject when the trailing
        # DROP TABLE runs in the same transaction. The
        # ``ORDER BY (context_id, id)`` clause is backed by the
        # composite (context_id, id) ordering on the source so the
        # per-batch read cost stays bounded.
        rows_processed = 0
        offset = 0
        # chunk_index is the per-context sequential position (0, 1, 2, ...),
        # matching the live compressed write path. The streamed read is ordered
        # by (context_id, id), so a context's chunks arrive contiguously and in
        # order across batch boundaries.
        current_ctx: str | None = None
        chunk_seq = 0
        while True:
            batch = await fetch_migration(
                conn,
                'SELECT context_id, id, start_index, end_index, embedding '
                'FROM vec_context_embeddings '
                'ORDER BY context_id, id '
                'LIMIT $1 OFFSET $2',
                migration_timeout_s,
                batch_size, offset,
            )
            if not batch:
                break
            for r in batch:
                ctx_str = str(r['context_id'])
                if ctx_str != current_ctx:
                    current_ctx = ctx_str
                    chunk_seq = 0
                else:
                    chunk_seq += 1
                vec_list = [float(x) for x in r['embedding']]
                # Validate the stored fp32 dimension against the configured compression
                # dim BEFORE encoding. SQLite fails safe (storage.fp32_blob_to_list_sqlite raises
                # on a size mismatch); PostgreSQL would otherwise feed a wrong-width vector
                # to the encoder, silently corrupting the payload and then DROPping the
                # fp32 source in this same transaction. Abort instead so EMBEDDING_DIM can
                # be set to the stored dimension and the run retried -- no fp32 vector is
                # corrupted or dropped.
                if len(vec_list) != provenance.dim:
                    raise ValueError(
                        f'fp32 vector for context {ctx_str} has dimension {len(vec_list)} but '
                        f'compression is configured for dim {provenance.dim} (EMBEDDING_DIM); '
                        f'aborting so no fp32 vector is corrupted or dropped.',
                    )
                payload = provider.encode_sync(make_2d_array(vec_list))
                await conn.execute(
                    'INSERT INTO vec_context_embeddings_compressed '
                    '(context_id, chunk_index, start_index, end_index, payload) '
                    'VALUES ($1, $2, $3, $4, $5)',
                    ctx_str, chunk_seq,
                    int(r['start_index']), int(r['end_index']), payload,
                )
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard: abort rather than drop vec_context_embeddings if
        # the streamed read did not cover every fp32 row the in-transaction
        # recount saw under the ACCESS EXCLUSIVE lock. With the table frozen
        # since before the first batch, a mismatch signals OFFSET-pagination
        # drift or an anomaly; abort so no fp32 vector is lost.
        if rows_processed != live_count:
            raise ValueError(
                f'compress read {rows_processed} fp32 row(s) but '
                f'vec_context_embeddings holds {live_count}; aborting so no '
                'fp32 vector is dropped.',
            )

        # Provenance INSERT + HNSW index DROP + source DROP last inside
        # the same transaction.
        await conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            'VALUES (1, $1, $2, $3, $4, $5, $6)',
            provenance.provider,
            provenance.bits,
            provenance.variant,
            provenance.seed,
            provenance.dim,
            provenance.codebook_fingerprint,
        )
        await execute_migration_ddl(
            conn,
            'DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw',
            migration_timeout_s,
        )
        await execute_migration_ddl(
            conn,
            'DROP TABLE IF EXISTS vec_context_embeddings',
            migration_timeout_s,
        )

    logger.info(
        'Compressed %d fp32 row(s) into vec_context_embeddings_compressed; '
        'dropped legacy table and HNSW index.',
        rows_processed,
    )
