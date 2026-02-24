"""
Semantic search migration functions for mcp-context-server.

This module handles:
- Semantic search table creation (vector storage)
- jsonb_merge_patch function migration for PostgreSQL
- Function search_path security fix migration
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends import create_backend
from app.migrations.utils import format_exception_message
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Database path for backward compatibility mode
DB_PATH = settings.storage.db_path


async def apply_semantic_search_migration(backend: StorageBackend | None = None) -> None:
    """Apply semantic search migration if enabled.

    Args:
        backend: Optional backend to use. If None, creates temporary backend for backward compatibility.

    This function can work in two modes:
    1. With backend parameter (normal server startup): Uses provided backend, no temp backend created
    2. Without backend parameter (tests/direct calls): Creates temporary backend for isolation

    This function:
    1. Checks if vector table already exists with embeddings
    2. Validates dimension compatibility (existing vs configured)
    3. Templates the migration SQL with configured embedding dimension
    4. Applies the migration if safe to proceed

    Raises:
        RuntimeError: If migration fails or dimension mismatch detected
    """
    if not settings.semantic_search.enabled:
        return

    # Determine backend type to select correct migration file
    if backend is not None:
        backend_type = backend.backend_type
    else:
        # Create temporary backend to determine type
        temp_backend = create_backend(backend_type=None, db_path=DB_PATH)
        backend_type = temp_backend.backend_type

    # Select migration file based on backend type
    migration_filename = ('add_semantic_search_postgresql.sql' if backend_type == 'postgresql'
                          else 'add_semantic_search_sqlite.sql')

    migration_path = Path(__file__).parent / migration_filename

    if not migration_path.exists():
        error_msg = f'Semantic search migration file not found: {migration_path}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        # Read migration SQL template
        migration_sql_template = migration_path.read_text(encoding='utf-8')

        # Apply migration - use provided backend or create temporary one
        if backend is not None:
            # Use provided backend (normal server startup)
            await _apply_migration_with_backend(backend, migration_sql_template)
        else:
            # Backward compatibility: create temporary backend for tests
            temp_manager = create_backend(backend_type=None, db_path=DB_PATH)
            await temp_manager.initialize()
            try:
                await _apply_migration_with_backend(temp_manager, migration_sql_template)
            finally:
                await temp_manager.shutdown()
    except Exception as e:
        logger.error(f'Failed to apply semantic search migration: {e}')
        raise RuntimeError(f'Semantic search migration failed: {e}') from e


async def _apply_migration_with_backend(manager: StorageBackend, migration_sql_template: str) -> None:
    """Helper function to apply migration with a given backend.

    Args:
        manager: The backend to use for migration
        migration_sql_template: The migration SQL template with {EMBEDDING_DIM} placeholder

    Raises:
        RuntimeError: If migration fails or dimension mismatch detected
    """
    # Check for existing table and dimension compatibility - backend-specific
    if manager.backend_type == 'sqlite':

        def _check_existing_dimension_sqlite(conn: sqlite3.Connection) -> tuple[bool, int | None]:
            # Check if vector table exists
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_context_embeddings'",
            )
            table_exists = cursor.fetchone() is not None

            if not table_exists:
                return False, None

            # Get existing dimension from any embedding metadata
            cursor = conn.execute('SELECT dimensions FROM embedding_metadata LIMIT 1')
            row = cursor.fetchone()
            existing_dim = row[0] if row else None

            return True, existing_dim

        table_exists, existing_dim = await manager.execute_read(_check_existing_dimension_sqlite)
    else:  # postgresql

        async def _check_existing_dimension_postgresql(conn: asyncpg.Connection) -> tuple[bool, int | None]:
            # Use configured schema (default: 'public') instead of hardcoded value
            # which may not match actual schema in Supabase environments
            schema = settings.storage.postgresql_schema
            # Check if vector table exists
            row = await conn.fetchrow(
                'SELECT tablename FROM pg_tables WHERE schemaname = $1 AND tablename = $2',
                schema,
                'vec_context_embeddings',
            )
            table_exists = row is not None

            if not table_exists:
                return False, None

            # Get existing dimension from any embedding metadata
            row = await conn.fetchrow('SELECT dimensions FROM embedding_metadata LIMIT 1')
            existing_dim = row['dimensions'] if row else None

            return True, existing_dim

        table_exists, existing_dim = await manager.execute_read(cast(Any, _check_existing_dimension_postgresql))

    # Validate dimension compatibility
    if table_exists and existing_dim is not None and existing_dim != settings.embedding.dim:
        db_path = str(DB_PATH).replace('\\', '/')
        raise RuntimeError(
            f'Embedding dimension mismatch detected!\n'
            f'  Existing database dimension: {existing_dim}\n'
            f'  Configured EMBEDDING_DIM: {settings.embedding.dim}\n\n'
            f'To change embedding dimensions, you must:\n'
            f'  1. Back up your database: {db_path}\n'
            f'  2. Delete or rename the database file\n'
            f'  3. Restart the server to create new tables with dimension {settings.embedding.dim}\n'
            f'  4. Re-import your context data (embeddings will be regenerated)\n\n'
            f'Note: Changing dimensions will lose all existing embeddings.',
        )

    # Template the migration SQL with configured dimension and schema
    migration_sql = migration_sql_template.replace(
        '{EMBEDDING_DIM}',
        str(settings.embedding.dim),
    ).replace(
        '{SCHEMA}',
        settings.storage.postgresql_schema,
    )

    # Apply migration - backend-specific
    if manager.backend_type == 'sqlite':

        def _apply_migration_sqlite(conn: sqlite3.Connection) -> None:
            # Load sqlite-vec extension before executing migration
            try:
                import sqlite_vec

                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.enable_load_extension(False)
                logger.debug('sqlite-vec extension loaded for migration')
            except ImportError:
                raise RuntimeError(
                    'sqlite-vec package required for semantic search migration. '
                    'Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)',
                ) from None
            except AttributeError:
                raise RuntimeError(
                    'SQLite does not support extension loading. Semantic search requires SQLite with extension support.',
                ) from None
            except Exception as e:
                raise RuntimeError(f'Failed to load sqlite-vec extension: {e}') from e

            # Now safe to execute migration with vec0 module
            conn.executescript(migration_sql)

        await manager.execute_write(_apply_migration_sqlite)
    else:  # postgresql

        async def _apply_migration_postgresql(conn: asyncpg.Connection) -> None:
            # PostgreSQL: pgvector extension registration happens in backend initialization
            # Acquire transaction-scoped advisory lock for multi-pod DDL safety.
            # pg_advisory_xact_lock releases automatically on COMMIT or ROLLBACK,
            # aligning with execute_write()'s conn.transaction() wrapper.
            await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")

            # Parse and execute migration SQL statements
            statements: list[str] = []
            current_stmt: list[str] = []
            in_function = False

            for line in migration_sql.split('\n'):
                stripped = line.strip()
                # Skip comment-only lines
                if stripped.startswith('--'):
                    continue
                # Track dollar-quoted strings (function bodies)
                if '$$' in stripped:
                    in_function = not in_function
                if stripped:
                    current_stmt.append(line)
                # End of statement: semicolon when not in dollar quotes
                if stripped.endswith(';') and not in_function:
                    statements.append('\n'.join(current_stmt))
                    current_stmt = []

            # Add any remaining statement
            if current_stmt:
                statements.append('\n'.join(current_stmt))

            # Execute each statement
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and not stmt.startswith('--'):
                    await conn.execute(stmt)

        await manager.execute_write(cast(Any, _apply_migration_postgresql))

    # Check table existence (not row existence) to determine if migration was applied
    if not table_exists:
        logger.info(
            f'Semantic search migration applied successfully with dimension: {settings.embedding.dim}',
        )
    else:
        logger.info('Semantic search migration: tables already exist, skipping')


async def _check_jsonb_merge_patch_exists(conn: asyncpg.Connection) -> bool:
    """Check if jsonb_merge_patch function already exists in PostgreSQL.

    Args:
        conn: PostgreSQL connection

    Returns:
        True if the function exists, False otherwise
    """
    # Use configured schema (default: 'public') instead of hardcoded value
    # which may not match actual schema in Supabase environments
    schema = settings.storage.postgresql_schema
    result = await conn.fetchval('''
        SELECT EXISTS (
            SELECT 1 FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = $1
              AND p.proname = 'jsonb_merge_patch'
        )
    ''', schema)
    return bool(result)


async def apply_jsonb_merge_patch_migration(backend: StorageBackend | None = None) -> None:
    """Apply jsonb_merge_patch function migration for PostgreSQL.

    This migration creates the jsonb_merge_patch() PL/pgSQL function that implements
    TRUE RFC 7396 recursive deep merge semantics. The function is required by the
    context_repository.patch_metadata() method for PostgreSQL backends.

    Args:
        backend: Optional backend to use. If None, creates temporary backend.

    Raises:
        RuntimeError: If migration execution fails.

    Note:
        - Only applies to PostgreSQL backends (SQLite uses native json_patch)
        - Idempotent: Uses CREATE OR REPLACE FUNCTION
        - Must be called after init_database() to ensure tables exist
    """
    # Determine backend type
    if backend is not None:
        backend_type = backend.backend_type
    else:
        temp_backend = create_backend(backend_type=None, db_path=DB_PATH)
        backend_type = temp_backend.backend_type

    # Only apply to PostgreSQL backends
    if backend_type != 'postgresql':
        return

    migration_path = Path(__file__).parent / 'add_jsonb_merge_patch_postgresql.sql'

    if not migration_path.exists():
        error_msg = f'jsonb_merge_patch migration file not found: {migration_path}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        migration_sql_template = migration_path.read_text(encoding='utf-8')

        # Template the migration SQL with configured schema
        schema = settings.storage.postgresql_schema
        migration_sql = migration_sql_template.replace('{SCHEMA}', schema)

        if backend is not None:
            # Check if function already exists before applying
            function_exists = await backend.execute_read(cast(Any, _check_jsonb_merge_patch_exists))

            async def _apply_jsonb_merge_patch(conn: asyncpg.Connection) -> None:
                # Acquire transaction-scoped advisory lock for multi-pod DDL safety.
                # pg_advisory_xact_lock releases automatically on COMMIT or ROLLBACK,
                # aligning with execute_write()'s conn.transaction() wrapper.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")

                # Parse SQL statements, handling dollar-quoted function bodies
                statements: list[str] = []
                current_stmt: list[str] = []
                in_function = False

                for line in migration_sql.split('\n'):
                    stripped = line.strip()
                    # Skip comment-only lines (but preserve function comments)
                    if stripped.startswith('--') and not in_function:
                        continue
                    # Track dollar-quoted strings (function bodies)
                    if '$$' in stripped:
                        in_function = not in_function
                    if stripped:
                        current_stmt.append(line)
                    # End of statement: semicolon when not in dollar quotes
                    if stripped.endswith(';') and not in_function:
                        statements.append('\n'.join(current_stmt))
                        current_stmt = []

                # Add any remaining statement
                if current_stmt:
                    statements.append('\n'.join(current_stmt))

                # Execute each statement
                for stmt in statements:
                    stmt = stmt.strip()
                    if stmt and not stmt.startswith('--'):
                        await conn.execute(stmt)

            await backend.execute_write(cast(Any, _apply_jsonb_merge_patch))

            # Verify function was actually created after migration
            verification_result = await backend.execute_read(cast(Any, _check_jsonb_merge_patch_exists))
            if not verification_result:
                raise RuntimeError(
                    'jsonb_merge_patch migration applied but function verification failed. '
                    'Check PostgreSQL permissions and error logs.',
                )

            if function_exists:
                logger.debug('jsonb_merge_patch function already exists, verified')
            else:
                logger.info('Applied jsonb_merge_patch migration for PostgreSQL')
        else:
            # Backward compatibility: create temporary backend
            temp_manager = create_backend(backend_type=None, db_path=DB_PATH)
            await temp_manager.initialize()
            try:
                # Check if function already exists before applying
                function_exists = await temp_manager.execute_read(cast(Any, _check_jsonb_merge_patch_exists))

                async def _apply_jsonb_merge_patch_temp(conn: asyncpg.Connection) -> None:
                    # Acquire transaction-scoped advisory lock for multi-pod DDL safety.
                    # pg_advisory_xact_lock releases automatically on COMMIT or ROLLBACK,
                    # aligning with execute_write()'s conn.transaction() wrapper.
                    await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")

                    statements: list[str] = []
                    current_stmt: list[str] = []
                    in_function = False

                    for line in migration_sql.split('\n'):
                        stripped = line.strip()
                        if stripped.startswith('--') and not in_function:
                            continue
                        if '$$' in stripped:
                            in_function = not in_function
                        if stripped:
                            current_stmt.append(line)
                        if stripped.endswith(';') and not in_function:
                            statements.append('\n'.join(current_stmt))
                            current_stmt = []

                    if current_stmt:
                        statements.append('\n'.join(current_stmt))

                    for stmt in statements:
                        stmt = stmt.strip()
                        if stmt and not stmt.startswith('--'):
                            await conn.execute(stmt)

                await temp_manager.execute_write(cast(Any, _apply_jsonb_merge_patch_temp))

                # Verify function was actually created after migration
                verification_result = await temp_manager.execute_read(cast(Any, _check_jsonb_merge_patch_exists))
                if not verification_result:
                    raise RuntimeError(
                        'jsonb_merge_patch migration applied but function verification failed. '
                        'Check PostgreSQL permissions and error logs.',
                    )

                if function_exists:
                    logger.debug('jsonb_merge_patch function already exists, verified')
                else:
                    logger.info('Applied jsonb_merge_patch migration for PostgreSQL')
            finally:
                await temp_manager.shutdown()
    except Exception as e:
        logger.error(f'Failed to apply jsonb_merge_patch migration: {e}')
        raise RuntimeError(f'jsonb_merge_patch migration failed: {format_exception_message(e)}') from e


async def apply_function_search_path_migration(backend: StorageBackend | None = None) -> None:
    """Apply search_path fix for PostgreSQL functions (CVE-2018-1058 mitigation).

    This migration sets search_path for all PostgreSQL functions to prevent
    potential search_path hijacking attacks. The migration is idempotent
    and can be safely run multiple times.

    Args:
        backend: Optional backend to use. If None, creates temporary backend.

    Raises:
        RuntimeError: If migration execution fails.

    Note:
        - Only applies to PostgreSQL backends
        - Idempotent: ALTER FUNCTION SET is safe to run repeatedly
        - Must be called after all function-creating migrations
    """
    # Determine backend type
    if backend is not None:
        backend_type = backend.backend_type
    else:
        temp_backend = create_backend(backend_type=None, db_path=DB_PATH)
        backend_type = temp_backend.backend_type

    # Only apply to PostgreSQL backends
    if backend_type != 'postgresql':
        return

    migration_path = Path(__file__).parent / 'fix_function_search_path_postgresql.sql'

    if not migration_path.exists():
        error_msg = f'Function search_path migration file not found: {migration_path}'
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    try:
        migration_sql_template = migration_path.read_text(encoding='utf-8')

        # Template the migration SQL with configured schema
        schema = settings.storage.postgresql_schema
        migration_sql = migration_sql_template.replace('{SCHEMA}', schema)

        if backend is not None:

            async def _apply_search_path_fix(conn: asyncpg.Connection) -> None:
                # Acquire transaction-scoped advisory lock for multi-pod DDL safety.
                # pg_advisory_xact_lock releases automatically on COMMIT or ROLLBACK,
                # aligning with execute_write()'s conn.transaction() wrapper.
                await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")

                # Parse SQL statements, handling dollar-quoted DO blocks
                statements: list[str] = []
                current_stmt: list[str] = []
                in_dollar_quote = False

                for line in migration_sql.split('\n'):
                    stripped = line.strip()
                    # Skip comment-only lines outside dollar quotes
                    if stripped.startswith('--') and not in_dollar_quote:
                        continue
                    # Track dollar-quoted strings (DO blocks and function bodies)
                    if '$$' in stripped:
                        in_dollar_quote = not in_dollar_quote
                    if stripped:
                        current_stmt.append(line)
                    # End of statement: semicolon when not in dollar quotes
                    if stripped.endswith(';') and not in_dollar_quote:
                        statements.append('\n'.join(current_stmt))
                        current_stmt = []

                # Add any remaining statement
                if current_stmt:
                    statements.append('\n'.join(current_stmt))

                for stmt in statements:
                    stmt = stmt.strip()
                    if stmt and not stmt.startswith('--'):
                        await conn.execute(stmt)

            await backend.execute_write(cast(Any, _apply_search_path_fix))
            logger.info('Applied function search_path security fix for PostgreSQL')
        else:
            # Backward compatibility: create temporary backend
            temp_manager = create_backend(backend_type=None, db_path=DB_PATH)
            await temp_manager.initialize()
            try:

                async def _apply_search_path_fix_temp(conn: asyncpg.Connection) -> None:
                    # Acquire transaction-scoped advisory lock for multi-pod DDL safety.
                    # pg_advisory_xact_lock releases automatically on COMMIT or ROLLBACK,
                    # aligning with execute_write()'s conn.transaction() wrapper.
                    await conn.execute("SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))")

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
                        stmt = stmt.strip()
                        if stmt and not stmt.startswith('--'):
                            await conn.execute(stmt)

                await temp_manager.execute_write(cast(Any, _apply_search_path_fix_temp))
                logger.info('Applied function search_path security fix for PostgreSQL')
            finally:
                await temp_manager.shutdown()
    except Exception as e:
        logger.error(f'Failed to apply function search_path migration: {e}')
        raise RuntimeError(f'Function search_path migration failed: {e}') from e
