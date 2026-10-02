"""Transactional compressed-to-fp32 data movement for ``--decompress``, one branch per backend."""

import logging
import sqlite3
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.cli.migrate_compression import storage
from app.cli.migrate_compression.decompress_empty import execute_decompress_empty
from app.cli.migrate_compression.storage import list_to_fp32_blob_sqlite
from app.cli.migrate_compression.storage import raise_pg_migration_budget
from app.compression.base import CompressionProvider
from app.compression.types import CompressionMetadata
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations._pg_ddl import fetch_migration
from app.migrations._pg_ddl import fetchval_migration
from app.settings import get_settings

logger = logging.getLogger(__name__)


async def execute_decompress(
    *,
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """Decode every compressed row and write the fp32 vec table.

    The DATA MOVEMENT runs inside a single
    :meth:`StorageBackend.begin_transaction` transaction on both backends,
    committing on success and rolling back on exception. The SCHEMA step does not
    share that guarantee on SQLite, where the ``IF NOT EXISTS`` table DDL autocommits
    through ``executescript()`` before ``BEGIN IMMEDIATE`` opens (see the branch
    docstrings); on PostgreSQL the DDL is transactional and rolls back with the rest. With ZERO compressed rows the
    zero-data reverse path runs instead: it drops the empty compressed table
    and clears the provenance row WITHOUT provisioning any fp32
    infrastructure, so the disable direction also works on deployments whose
    embedding storage (embedding_chunks, sqlite-vec, the pgvector extension)
    was never provisioned.
    """
    if row_count == 0:
        await execute_decompress_empty(backend)
        return
    if backend.backend_type == 'sqlite':
        await _execute_decompress_sqlite(
            backend, provider, provenance, row_count,
        )
        return
    await _execute_decompress_postgresql(
        backend, provider, provenance, row_count,
    )


async def _execute_decompress_sqlite(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """SQLite branch of :func:`execute_decompress`.

    Symmetric streamed reverse migration. Reads compressed rows in
    batches of :data:`storage.MIGRATION_BATCH_SIZE` from
    ``vec_context_embeddings_compressed`` (ordered by
    ``(context_id, chunk_index)``), decodes each batch in-process, and
    rebuilds BOTH the recreated ``vec_context_embeddings`` fp32 table and
    the ``embedding_chunks`` bridge directly from each compressed row's own
    ``context_id``/``start_index``/``end_index``. The rebuild does NOT depend
    on any pre-existing ``embedding_chunks`` row: a server-compressed-from-
    start database never wrote ``embedding_chunks`` (the live compressed
    write path stores only ``vec_context_embeddings_compressed`` with a
    per-context sequential ``chunk_index``), so on a default deployment the
    compressed rows themselves are the only record of each vector's context
    and span. ``BEGIN IMMEDIATE`` takes the write lock BEFORE the first
    streamed read, so a concurrent-process writer cannot commit (or replace
    rows with an equal-cardinality set) mid-stream. Provenance DELETE +
    compressed source DROP run last inside the same transaction.

    Raises:
        ValueError: When the recount inside the transaction disagrees with
            the number of rows streamed -- with the write lock held since
            before the first batch this signals pagination drift or an
            anomaly, and the drop below could destroy payloads that were
            never decoded. Mirrors the compress path's integrity guard; the
            transaction rolls back.
    """
    batch_size = storage.MIGRATION_BATCH_SIZE

    async with backend.begin_transaction() as txn:
        conn = cast(sqlite3.Connection, txn.connection)

        # Recreate the fp32 virtual table (idempotent).
        conn.executescript(
            f'CREATE VIRTUAL TABLE IF NOT EXISTS vec_context_embeddings '
            f'USING vec0(embedding float[{provenance.dim}])',
        )

        # Take the write lock BEFORE the first streamed read (see
        # compress_execution._execute_compress_sqlite): without BEGIN
        # IMMEDIATE the implicit transaction opens only at the first DML,
        # leaving early batch reads exposed to a
        # concurrent-process writer whose equal-count replacement would slip
        # past the recount guard below. executescript above ran first because
        # it COMMITS any open transaction (its IF NOT EXISTS DDL is
        # idempotent autocommit).
        conn.execute('BEGIN IMMEDIATE')

        # Idempotency check: if the compressed source table no longer
        # exists, the reverse migration has already run.
        check = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='vec_context_embeddings_compressed'",
        ).fetchone()
        if not check:
            logger.info(
                'Reverse migration already applied (compressed source '
                'table absent); --decompress is a no-op.',
            )
            return

        # Clear the (possibly stale or empty) chunk->vec bridge and rebuild
        # it alongside vec_context_embeddings from the compressed rows
        # themselves, mirroring the context_id-based PostgreSQL reverse path.
        # A fresh running rowid links each decoded fp32 vector to its new
        # embedding_chunks row so the SQLite fp32 search join
        # (embedding_chunks.vec_rowid = vec_context_embeddings.rowid) resolves
        # after decompress regardless of how the compressed rows were produced.
        conn.execute('DELETE FROM embedding_chunks')
        next_rowid = int(
            conn.execute(
                'SELECT COALESCE(MAX(rowid), 0) + 1 FROM vec_context_embeddings',
            ).fetchone()[0],
        )

        # Stream compressed rows; decode and INSERT each batch.
        rows_processed = 0
        offset = 0
        while True:
            cur = conn.execute(
                'SELECT context_id, chunk_index, start_index, end_index, '
                'payload FROM vec_context_embeddings_compressed '
                'ORDER BY context_id, chunk_index '
                'LIMIT ? OFFSET ?',
                (batch_size, offset),
            )
            batch = cur.fetchall()
            if not batch:
                break
            for ctx_id, _chunk_idx, start_index, end_index, payload in batch:
                vec = provider.decode_sync(bytes(payload))[0]
                blob = list_to_fp32_blob_sqlite(
                    [float(x) for x in vec.tolist()],
                )
                conn.execute(
                    'INSERT INTO vec_context_embeddings (rowid, embedding) '
                    'VALUES (?, ?)',
                    (next_rowid, blob),
                )
                conn.execute(
                    'INSERT INTO embedding_chunks '
                    '(context_id, vec_rowid, start_index, end_index) '
                    'VALUES (?, ?, ?, ?)',
                    (str(ctx_id), next_rowid, int(start_index), int(end_index)),
                )
                next_rowid += 1
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard mirroring the compress path: with BEGIN IMMEDIATE
        # holding the write lock since before the first batch, a mismatch
        # against the in-transaction recount signals pagination drift or an
        # anomaly; abort rather than drop undecoded payloads.
        live_count = int(
            conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
            ).fetchone()[0],
        )
        if live_count != rows_processed:
            raise ValueError(
                f'decompress streamed {rows_processed} compressed row(s) but '
                f'vec_context_embeddings_compressed holds {live_count} at drop '
                'time -- a concurrent writer modified the table while '
                '--decompress was running. Aborting so no undecoded payload is '
                'dropped; stop the server writing to this database and re-run '
                '--decompress.',
            )

        # Provenance DELETE + compressed source DROP last inside the
        # same transaction.
        conn.execute('DROP TABLE IF EXISTS vec_context_embeddings_compressed')
        conn.execute('DELETE FROM compression_metadata WHERE id = 1')

    logger.info(
        'Decompressed %d compressed row(s) into vec_context_embeddings; '
        'dropped compressed table and provenance row.',
        rows_processed or row_count,
    )


async def _execute_decompress_postgresql(
    backend: StorageBackend,
    provider: CompressionProvider,
    provenance: CompressionMetadata,
    row_count: int,
) -> None:
    """PostgreSQL branch of :func:`execute_decompress`.

    Symmetric streamed reverse migration using LIMIT/OFFSET pagination
    ordered by ``(context_id, chunk_index)``. Server-side cursors would
    keep a portal open against the source table that PostgreSQL would
    reject when the trailing ``DROP TABLE`` runs in the same
    transaction. The source table is locked ``ACCESS EXCLUSIVE`` BEFORE
    the first batch: the stream runs under READ COMMITTED, so a concurrent
    same-chunk-count update committing mid-stream would otherwise replace
    rows the stream had already decoded without changing the cardinality a
    trailing recount could check. Recreates ``vec_context_embeddings`` +
    HNSW index inside the same transaction that streams decompressed rows
    in batches.

    Raises:
        ValueError: When the number of rows streamed disagrees with the
            recount taken under the lock before streaming -- with the table
            frozen this signals OFFSET-pagination drift or an anomaly, and
            the drop below could destroy payloads that were never decoded.
            Mirrors the compress path's integrity guard; the transaction
            rolls back.
    """
    batch_size = storage.MIGRATION_BATCH_SIZE
    migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s

    async with backend.begin_transaction() as txn:
        conn = cast('asyncpg.Connection', txn.connection)
        # Raise this transaction's statement budget so the heavy vec-table DDL below
        # (the HNSW CREATE INDEX / DROP TABLE) and the streamed batch reads run under the
        # migration budget, not the pool's ~60s command_timeout. See
        # storage.raise_pg_migration_budget.
        await raise_pg_migration_budget(conn, migration_timeout_s)

        # Bare table names rely on PostgreSQL's search_path resolution;
        # this matches the project-wide pattern in
        # ``app/repositories/embedding_repository/`` and ``postgresql_schema.sql``.
        # Operators using a non-default schema configure ``search_path``
        # accordingly. The existence probe below uses ``to_regclass`` so
        # it resolves through the SAME search_path as the bare-name DML
        # and the trailing DROP TABLE.
        await conn.execute(
            f'CREATE TABLE IF NOT EXISTS vec_context_embeddings ('
            f'  id BIGSERIAL PRIMARY KEY,'
            f'  context_id UUID NOT NULL,'
            f'  embedding vector({provenance.dim}),'
            f'  start_index INTEGER NOT NULL DEFAULT 0,'
            f'  end_index INTEGER NOT NULL DEFAULT 0,'
            f'  FOREIGN KEY (context_id) REFERENCES context_entries(id) '
            f'  ON DELETE CASCADE'
            f')',
        )
        await conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_vec_embeddings_context_id '
            'ON vec_context_embeddings(context_id)',
        )

        # Idempotency check: the compressed source table must exist for
        # reverse migration to have work to do.
        source_reachable = await conn.fetchval(
            "SELECT to_regclass('vec_context_embeddings_compressed') IS NOT NULL",
        )
        if not source_reachable:
            logger.info(
                'Reverse migration already applied (compressed source '
                'table absent); --decompress is a no-op.',
            )
            return

        # Freeze the source table BEFORE streaming: the batches read under
        # READ COMMITTED, so a concurrent writer replacing an entry's N
        # compressed rows with N new ones mid-stream (a same-chunk-count
        # update, the most common write shape) would defeat a count-only
        # recount taken after the fact -- the stream would have decoded the
        # OLD rows, the cardinality would still match, and the DROP below
        # would destroy the never-decoded new payloads while fp32 kept the
        # stale ones. ACCESS EXCLUSIVE (waiting out in-flight writers,
        # bounded by the migration budget) makes every batch, the recount,
        # and the DROP see one frozen state.
        await execute_migration_ddl(
            conn,
            'LOCK TABLE vec_context_embeddings_compressed '
            'IN ACCESS EXCLUSIVE MODE',
            migration_timeout_s,
        )
        # Route the recount through the migration budget (see
        # compress_execution._execute_compress_postgresql): a bare fetchval
        # would cap this first under-lock full-table scan at the pool
        # command_timeout instead of the migration budget.
        live_count = int(
            await fetchval_migration(
                conn,
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
                migration_timeout_s,
            )
            or 0,
        )

        # Stream compressed rows in batches via LIMIT/OFFSET pagination.
        # Server-side cursors would keep a portal open against the
        # source table that PostgreSQL would reject when the trailing
        # DROP TABLE runs in the same transaction.
        rows_processed = 0
        offset = 0
        while True:
            batch = await fetch_migration(
                conn,
                'SELECT context_id, chunk_index, start_index, end_index, '
                'payload FROM vec_context_embeddings_compressed '
                'ORDER BY context_id, chunk_index '
                'LIMIT $1 OFFSET $2',
                migration_timeout_s,
                batch_size, offset,
            )
            if not batch:
                break
            for r in batch:
                vec = provider.decode_sync(bytes(r['payload']))[0]
                await conn.execute(
                    'INSERT INTO vec_context_embeddings '
                    '(context_id, embedding, start_index, end_index) '
                    'VALUES ($1, $2, $3, $4)',
                    str(r['context_id']),
                    [float(x) for x in vec.tolist()],
                    int(r['start_index']),
                    int(r['end_index']),
                )
            rows_processed += len(batch)
            offset += len(batch)

        # Integrity guard mirroring the compress path: with the ACCESS
        # EXCLUSIVE lock held since before the first batch, the table cannot
        # have changed mid-stream; a mismatch against the under-lock recount
        # signals OFFSET-pagination drift or an anomaly, and aborting is
        # still safer than dropping an undecoded payload.
        if live_count != rows_processed:
            raise ValueError(
                f'decompress streamed {rows_processed} compressed row(s) but '
                f'vec_context_embeddings_compressed holds {live_count} at drop '
                'time -- a concurrent writer modified the table while '
                '--decompress was running. Aborting so no undecoded payload is '
                'dropped; stop the server writing to this database and re-run '
                '--decompress.',
            )

        # HNSW index CREATE + compressed source DROP + provenance DELETE
        # last inside the same transaction.
        await execute_migration_ddl(
            conn,
            'CREATE INDEX IF NOT EXISTS idx_vec_context_embeddings_hnsw '
            'ON vec_context_embeddings '
            'USING hnsw (embedding vector_l2_ops) '
            'WITH (m = 16, ef_construction = 64)',
            migration_timeout_s,
        )
        await execute_migration_ddl(
            conn,
            'DROP TABLE IF EXISTS vec_context_embeddings_compressed',
            migration_timeout_s,
        )
        await conn.execute(
            'DELETE FROM compression_metadata WHERE id = 1',
        )

    logger.info(
        'Decompressed %d compressed row(s) into vec_context_embeddings; '
        'dropped compressed table and recreated HNSW index.',
        rows_processed or row_count,
    )
