"""Zero-data ``--decompress`` path: drop the empty compressed table without any fp32 infrastructure."""

import logging
import sqlite3
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.cli.migrate_compression.storage import raise_pg_migration_budget
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations._pg_ddl import fetchval_migration
from app.settings import get_settings

logger = logging.getLogger(__name__)


async def execute_decompress_empty(backend: StorageBackend) -> None:
    """Zero-data reverse migration: drop the empty compressed table, clear the row.

    With no compressed rows there is nothing to decode, so the fp32
    infrastructure the full reverse path provisions (the vec0 virtual table /
    the pgvector-typed table plus the embedding_chunks rebuild) is not needed
    -- and may be genuinely absent: a deployment running with
    ``ENABLE_EMBEDDING_GENERATION=false`` never provisioned embedding_chunks,
    sqlite-vec, or the pgvector extension, yet could still carry the
    compression schema and a provenance row; a reverse path that required that
    infrastructure would leave such a deployment no way to turn compression off.
    Dropping the empty table and deleting the provenance row is the complete
    reverse migration for that state; the next startup provisions fp32
    storage per ``ENABLE_EMBEDDING_GENERATION`` as usual. The caller has
    already verified both the provenance row and the compressed table exist.

    The planning-time row count ran OUTSIDE this transaction, so emptiness is
    re-checked INSIDE it before the drop: a compression-on server running
    concurrently could commit compressed rows in the gap, and dropping them
    here would destroy embeddings that were never decoded. On PostgreSQL the
    table is locked ACCESS EXCLUSIVE first (waiting out in-flight writers) so
    the recount is authoritative; on SQLite an explicit ``BEGIN IMMEDIATE``
    takes the write lock up front -- sqlite3's legacy transaction control
    opens the implicit transaction only before DML, so a bare SELECT pins no
    snapshot and a DROP with no transaction open would run in AUTOCOMMIT,
    committing past the guard -- making the recount authoritative and the
    drop + provenance delete one atomic commit; a concurrent write-lock
    holder surfaces as SQLITE_BUSY, bounded by the busy timeout.

    Raises:
        ValueError: When the recount inside the transaction finds compressed
            rows -- a concurrent writer stored embeddings after the zero-data
            path was planned. The transaction rolls back and nothing is
            dropped.
    """
    if backend.backend_type == 'sqlite':
        async with backend.begin_transaction() as txn:
            conn = cast(sqlite3.Connection, txn.connection)
            conn.execute('BEGIN IMMEDIATE')
            live_row = conn.execute(
                'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
            ).fetchone()
            live_count = int(live_row[0])
            if live_count != 0:
                raise ValueError(
                    f'vec_context_embeddings_compressed holds {live_count} row(s) at '
                    'drop time but was empty when the zero-data reverse path was '
                    'planned -- a concurrent writer stored compressed embeddings '
                    'while --decompress was running. Aborting so no compressed '
                    'payload is dropped; stop the server writing to this database '
                    'and re-run --decompress.',
                )
            conn.execute('DROP TABLE IF EXISTS vec_context_embeddings_compressed')
            conn.execute('DELETE FROM compression_metadata WHERE id = 1')
    else:
        migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s
        async with backend.begin_transaction() as txn:
            pg_conn = cast('asyncpg.Connection', txn.connection)
            await raise_pg_migration_budget(pg_conn, migration_timeout_s)
            await execute_migration_ddl(
                pg_conn,
                'LOCK TABLE vec_context_embeddings_compressed '
                'IN ACCESS EXCLUSIVE MODE',
                migration_timeout_s,
            )
            # Route the under-lock recount through the migration budget (not a bare
            # fetchval, which inherits the pool's ~60s command_timeout), matching the
            # compress/decompress recounts: this COUNT(*) runs under the lock and must
            # use POSTGRESQL_MIGRATION_TIMEOUT_S on a large table.
            live_count = int(
                await fetchval_migration(
                    pg_conn,
                    'SELECT COUNT(*) FROM vec_context_embeddings_compressed',
                    migration_timeout_s,
                )
                or 0,
            )
            if live_count != 0:
                raise ValueError(
                    f'vec_context_embeddings_compressed holds {live_count} row(s) at '
                    'drop time but was empty when the zero-data reverse path was '
                    'planned -- a concurrent writer stored compressed embeddings '
                    'while --decompress was running. Aborting so no compressed '
                    'payload is dropped; stop the server writing to this database '
                    'and re-run --decompress.',
                )
            await execute_migration_ddl(
                pg_conn,
                'DROP TABLE IF EXISTS vec_context_embeddings_compressed',
                migration_timeout_s,
            )
            await pg_conn.execute('DELETE FROM compression_metadata WHERE id = 1')
    logger.info(
        'No compressed rows to decode; dropped the empty compressed table '
        'and cleared the provenance row.',
    )
