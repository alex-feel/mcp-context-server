"""Compression migration loader for mcp-context-server.

Applies the schema that replaces the fp32 vector tables with compressed-payload
tables when ``ENABLE_EMBEDDING_COMPRESSION=true``. Follows the same shape as
``app.migrations.chunking`` for consistency:

- Selects the SQLite or PostgreSQL SQL file based on ``backend.backend_type``.
- For PostgreSQL, acquires an advisory lock, sets the migration statement
  timeout, and parses statements respecting ``$$`` dollar-quoted blocks.
- Early-returns when ``settings.compression.enabled`` is false.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.compression.provenance import read_compression_metadata
from app.errors import ConfigurationError
from app.errors import format_exception_message
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


async def _fp32_table_has_rows(backend: StorageBackend) -> bool:
    """Return True when the fp32 ``vec_context_embeddings`` table exists and holds rows.

    Used by the pre-drop guard in :func:`apply_compression_migration`: when no
    compression provenance row exists, a populated fp32 table is the
    AUTHORITATIVE embedding store, and the migration's ``DROP TABLE IF EXISTS
    vec_context_embeddings`` would destroy it irrecoverably.

    Args:
        backend: Storage backend instance.

    Returns:
        True when the table exists and holds at least one row (or, on SQLite,
        when the vec0 module cannot read it -- refuse rather than risk dropping
        shadow-table data that becomes readable once the extension loads);
        False when the table is absent or empty (the drop is harmless).
    """
    if backend.backend_type == 'sqlite':

        def _probe_sqlite(conn: sqlite3.Connection) -> bool:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_context_embeddings'",
            )
            if cursor.fetchone() is None:
                return False
            try:
                row = conn.execute(
                    'SELECT EXISTS (SELECT 1 FROM vec_context_embeddings)',
                ).fetchone()
            except sqlite3.OperationalError:
                return True
            return bool(row and row[0])

        return await backend.execute_read(_probe_sqlite)

    async def _probe_pg(conn: asyncpg.Connection) -> bool:
        schema = settings.storage.postgresql_schema
        exists = await conn.fetchval(
            '''
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = $1
                  AND table_name = 'vec_context_embeddings'
            )
            ''',
            schema,
        )
        if not exists:
            return False
        populated = await conn.fetchval(
            'SELECT EXISTS (SELECT 1 FROM vec_context_embeddings)',
        )
        return bool(populated)

    return await backend.execute_read(cast(Any, _probe_pg))


def _check_compression_migration_applied_sqlite(conn: sqlite3.Connection) -> bool:
    """Check whether the compression schema is already present on SQLite.

    Args:
        conn: SQLite connection.

    Returns:
        True when ``compression_metadata`` exists (migration already applied).
    """
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='compression_metadata'",
    )
    return cursor.fetchone() is not None


async def _check_compression_migration_applied_postgresql(conn: asyncpg.Connection) -> bool:
    """Check whether the compression schema is already present on PostgreSQL.

    Both tables are required: ``--decompress`` drops
    ``vec_context_embeddings_compressed`` while leaving the
    ``compression_metadata`` table behind (it only deletes the provenance
    row), so a marker-table-only probe would treat a later re-enable as
    already applied and skip re-creating the payload table -- wedging every
    embedding store and search on a table that does not exist. SQLite has no
    equivalent gap because its branch re-runs the idempotent script every
    start.

    Args:
        conn: PostgreSQL connection.

    Returns:
        True when ``compression_metadata`` AND
        ``vec_context_embeddings_compressed`` both exist in the configured
        schema.
    """
    schema = settings.storage.postgresql_schema
    result = await conn.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1
              AND table_name = 'compression_metadata'
        ) AND EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1
              AND table_name = 'vec_context_embeddings_compressed'
        )
        ''',
        schema,
    )
    return bool(result)


async def apply_compression_migration(backend: StorageBackend) -> None:
    """Apply the compression schema migration.

    Args:
        backend: Storage backend instance.

    Raises:
        ConfigurationError: When this would be the FIRST application of the
            compression schema (no provenance row) on a database whose fp32
            ``vec_context_embeddings`` table holds rows. The migration's
            ``DROP TABLE IF EXISTS vec_context_embeddings`` would destroy
            those embeddings irrecoverably, so a bare
            ``ENABLE_EMBEDDING_COMPRESSION=true`` flip (or removing a
            ``false`` override, since the default is true) on a populated
            fp32 deployment refuses loudly (exit 78) and directs the
            operator to ``mcp-context-server-migrate --compress``, which
            encodes the embeddings before swapping the tables. Mirrors the
            disable-direction guard in the compression validator.
        RuntimeError: If migration execution fails or the SQL file is missing.

    Notes:
        - Only applies when ``ENABLE_EMBEDDING_COMPRESSION=true``.
        - Idempotent: uses ``DROP IF EXISTS`` + ``CREATE IF NOT EXISTS``.
        - Must run AFTER the semantic-search and chunking migrations.
    """
    if not settings.compression.enabled:
        return

    # Compression is a storage format FOR embeddings, and embedding storage is
    # provisioned from ENABLE_EMBEDDING_GENERATION (the semantic/chunking
    # migrations gate on it). With generation off nothing can ever write a
    # compressed payload, so a database WITHOUT the compression schema gets
    # none -- otherwise the validator would seed a provenance row for a schema
    # that can never hold data, and a later ENABLE_EMBEDDING_COMPRESSION=false
    # flip would wedge behind the disable-direction guard's --decompress
    # instruction on a deployment whose embedding_chunks / vector
    # infrastructure was never provisioned. A database that ALREADY carries
    # the schema (data compressed while generation was on) falls through so
    # the already-applied branch keeps the fingerprint column ensured for the
    # decode path.
    if not settings.embedding.generation_enabled:
        if backend.backend_type == 'postgresql':
            applied = await backend.execute_read(
                cast(Any, _check_compression_migration_applied_postgresql),
            )
        else:
            applied = await backend.execute_read(
                _check_compression_migration_applied_sqlite,
            )
        if not applied:
            return

    # Enable-direction data guard: on FIRST-TIME application (no provenance
    # row yet -- read_compression_metadata returns None when the table is
    # absent or empty), a populated fp32 table is the authoritative embedding
    # store and must not be dropped by a bare env flip. The --compress CLI is
    # the sanctioned path: it encodes every fp32 row into compressed payloads
    # and writes the provenance row, after which this startup migration only
    # finalizes the swap. A database that already carries a provenance row
    # proceeds normally (the fp32 table there is a stray artifact, not data).
    if await read_compression_metadata(backend) is None and await _fp32_table_has_rows(backend):
        raise ConfigurationError(
            'ENABLE_EMBEDDING_COMPRESSION is true but this database still holds '
            'uncompressed fp32 embeddings in vec_context_embeddings and has never '
            'been compressed (no compression_metadata provenance row). Applying '
            'the compression schema now would DROP that table and permanently '
            'destroy every stored embedding. Run "mcp-context-server-migrate '
            '--compress" to encode the existing embeddings into compressed '
            'storage first, or set ENABLE_EMBEDDING_COMPRESSION=false to keep '
            'serving the fp32 data.',
        )

    backend_type = backend.backend_type

    migration_filename = (
        'add_compression_postgresql.sql'
        if backend_type == 'postgresql'
        else 'add_compression_sqlite.sql'
    )

    migration_path = Path(__file__).parent / migration_filename

    if not migration_path.exists():
        error_msg = f'Compression migration file not found: {migration_path}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        migration_sql_template = migration_path.read_text(encoding='utf-8')
        await _apply_compression_migration_with_backend(
            backend, migration_sql_template,
        )
    except Exception as e:
        logger.error(f'Failed to apply compression migration: {e}')
        raise RuntimeError(
            f'Compression migration failed: {format_exception_message(e)}',
        ) from e


async def _apply_compression_migration_with_backend(
    manager: StorageBackend,
    migration_sql_template: str,
) -> None:
    """Apply the compression migration with a concrete backend.

    Args:
        manager: The storage backend to use.
        migration_sql_template: Raw SQL template. The compression
            migration uses BARE table names; the loader does not
            substitute {SCHEMA} for this migration. Operators with a
            non-default POSTGRESQL_SCHEMA configure search_path on
            their connection pool.
    """
    if manager.backend_type == 'postgresql':
        already_applied = await manager.execute_read(
            cast(Any, _check_compression_migration_applied_postgresql),
        )

        async def _ensure_fingerprint_column(conn: asyncpg.Connection) -> None:
            # A compression_metadata table may predate the codebook_fingerprint
            # column (CREATE TABLE IF NOT EXISTS never adds a column to an
            # existing table). Add it idempotently so the read path -- which
            # SELECTs the column -- works after an in-place upgrade.
            await conn.execute(
                'ALTER TABLE compression_metadata '
                'ADD COLUMN IF NOT EXISTS codebook_fingerprint TEXT',
            )

        if already_applied:
            # Both tables exist; only the fingerprint column may be missing.
            # Skip the destructive table swap.
            await manager.execute_write(cast(Any, _ensure_fingerprint_column))
            logger.info(
                'Compression migration: already applied for PostgreSQL, '
                'codebook_fingerprint column ensured',
            )
            return

        migration_sql = migration_sql_template

        migration_timeout_s = settings.storage.postgresql_migration_timeout_s

        async def _apply_postgresql(conn: asyncpg.Connection) -> None:
            # Raise the transaction-scoped statement_timeout to the migration budget and take
            # the shared advisory lock under it. SET LOCAL auto-reverts on COMMIT/ROLLBACK, so
            # there is no finally-restore SET -- which, in an aborted transaction, would raise
            # 25P02 and mask the real DDL error. The lock acquire and the DDL run under the
            # migration timeout so the pool's shorter command_timeout cannot cancel them.
            await begin_migration(conn, migration_timeout_s)

            # Parse SQL statements respecting dollar-quoted DO blocks.
            statements: list[str] = []
            current_stmt: list[str] = []
            in_dollar_quote = False

            for line in migration_sql.split('\n'):
                stripped = line.strip()
                if stripped.startswith('--') and not in_dollar_quote:
                    continue
                if '$$' in stripped:
                    in_dollar_quote = not in_dollar_quote
                if stripped:
                    current_stmt.append(line)
                if stripped.endswith(';') and not in_dollar_quote:
                    statements.append('\n'.join(current_stmt))
                    current_stmt = []

            if current_stmt:
                statements.append('\n'.join(current_stmt))

            for stmt in statements:
                stmt_clean = stmt.strip()
                if stmt_clean and not stmt_clean.startswith('--'):
                    await execute_migration_ddl(conn, stmt_clean, migration_timeout_s)

            # The script's CREATE TABLE IF NOT EXISTS keeps a surviving
            # compression_metadata table as-is (the payload table can be
            # re-created around it, e.g. after --decompress dropped only the
            # payload table), so a table predating the codebook_fingerprint
            # column still needs the idempotent ALTER on this path too.
            await execute_migration_ddl(
                conn,
                'ALTER TABLE compression_metadata '
                'ADD COLUMN IF NOT EXISTS codebook_fingerprint TEXT',
                migration_timeout_s,
            )

        await manager.execute_write(cast(Any, _apply_postgresql))
        logger.info(
            'Applied compression migration for PostgreSQL: '
            'vec_context_embeddings_compressed + compression_metadata created',
        )
        return

    # SQLite branch
    already_applied = await manager.execute_read(_check_compression_migration_applied_sqlite)

    def _apply_sqlite(conn: sqlite3.Connection) -> None:
        conn.executescript(migration_sql_template)
        # Add codebook_fingerprint to a pre-existing compression_metadata table that
        # predates the column (CREATE TABLE IF NOT EXISTS never adds a column to an
        # existing table). SQLite has no ADD COLUMN IF NOT EXISTS, so probe PRAGMA
        # first; the ALTER is then idempotent across restarts and in-place upgrades.
        existing_cols = [
            row[1] for row in conn.execute('PRAGMA table_info(compression_metadata)').fetchall()
        ]
        if 'codebook_fingerprint' not in existing_cols:
            conn.execute(
                'ALTER TABLE compression_metadata ADD COLUMN codebook_fingerprint TEXT',
            )

    await manager.execute_write(_apply_sqlite)

    if not already_applied:
        logger.info(
            'Applied compression migration for SQLite: '
            'vec_context_embeddings_compressed + compression_metadata created',
        )
    else:
        logger.info('Compression migration: verified for SQLite')
