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
from app.errors import format_exception_message
from app.migrations._pg_ddl import execute_migration_ddl
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


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

    Args:
        conn: PostgreSQL connection.

    Returns:
        True when ``compression_metadata`` exists in the configured schema.
    """
    schema = settings.storage.postgresql_schema
    result = await conn.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_schema = $1
              AND table_name = 'compression_metadata'
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
        RuntimeError: If migration execution fails or the SQL file is missing.

    Notes:
        - Only applies when ``ENABLE_EMBEDDING_COMPRESSION=true``.
        - Idempotent: uses ``DROP IF EXISTS`` + ``CREATE IF NOT EXISTS``.
        - Must run AFTER the semantic-search and chunking migrations.
    """
    if not settings.compression.enabled:
        return

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

        if already_applied:
            # The table exists, but it may predate the codebook_fingerprint column
            # (CREATE TABLE IF NOT EXISTS never adds a column to an existing table).
            # Add it idempotently so the read path -- which SELECTs the column -- works
            # after an in-place upgrade, without re-running the destructive table swap.
            async def _ensure_fingerprint_column(conn: asyncpg.Connection) -> None:
                await conn.execute(
                    'ALTER TABLE compression_metadata '
                    'ADD COLUMN IF NOT EXISTS codebook_fingerprint TEXT',
                )

            await manager.execute_write(cast(Any, _ensure_fingerprint_column))
            logger.info(
                'Compression migration: already applied for PostgreSQL, '
                'codebook_fingerprint column ensured',
            )
            return

        migration_sql = migration_sql_template

        migration_timeout_s = settings.storage.postgresql_migration_timeout_s

        async def _apply_postgresql(conn: asyncpg.Connection) -> None:
            # The advisory-lock acquire and the DDL below run under the migration
            # timeout so the pool's shorter command_timeout cannot cancel them before
            # the server-side statement_timeout applies.
            await execute_migration_ddl(
                conn,
                "SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))",
                migration_timeout_s,
            )

            migration_timeout_ms = int(migration_timeout_s * 1000)
            await conn.execute(f'SET statement_timeout = {migration_timeout_ms}')

            try:
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
            finally:
                default_timeout_ms = int(
                    settings.storage.postgresql_command_timeout_s * 1000 * 0.9,
                )
                await conn.execute(f'SET statement_timeout = {default_timeout_ms}')

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
