"""
MCP Context Server implementation using FastMCP.

This server provides persistent multimodal context storage capabilities for LLM agents,
enabling shared memory across different conversation threads with support for text and images.
"""

import asyncio
import base64
import contextlib
import json
import logging
import operator
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

import asyncpg
from fastmcp import Context
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from app.backends import StorageBackend
from app.backends import create_backend
from app.logger_config import config_logger
from app.repositories import RepositoryContainer
from app.repositories.fts_repository import FtsRepository
from app.settings import get_settings
from app.types import BulkDeleteResponseDict
from app.types import BulkStoreResponseDict
from app.types import BulkStoreResultItemDict
from app.types import BulkUpdateResponseDict
from app.types import BulkUpdateResultItemDict
from app.types import ContextEntryDict
from app.types import JsonValue
from app.types import MetadataDict
from app.types import StoreContextSuccessDict
from app.types import ThreadListDict
from app.types import UpdateContextSuccessDict

# Get setting
settings = get_settings()
# Configure logging
config_logger(settings.log_level)
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = settings.storage.db_path
MAX_IMAGE_SIZE_MB = settings.storage.max_image_size_mb
MAX_TOTAL_SIZE_MB = settings.storage.max_total_size_mb

# Global connection manager and repositories
_backend: StorageBackend | None = None
_repositories: RepositoryContainer | None = None

# Global embedding service (only if semantic search enabled and available)
_embedding_service: Any | None = None


# Tool Annotations for MCP protocol hints
# Read-only tools: do not modify environment
READ_ONLY_ANNOTATIONS = ToolAnnotations(readOnlyHint=True)

# Additive tools: create new entries (not destructive)
ADDITIVE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=False,
)

# Delete tools: destructive but idempotent (deleting twice is no-op)
DELETE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=True,
)

# Update tools: destructive and NOT idempotent (same update may have different effects)
UPDATE_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    destructiveHint=True,
    idempotentHint=False,
)


@dataclass
class FtsMigrationStatus:
    """Track FTS migration state for graceful degradation.

    When FTS language/tokenizer settings change, migration must occur.
    During migration, the fts_search_context tool remains available but
    returns informative status instead of search results.
    """

    in_progress: bool = False
    started_at: datetime | None = None
    estimated_seconds: int | None = None
    backend: str | None = None
    old_language: str | None = None
    new_language: str | None = None
    records_count: int | None = None


# Global FTS migration status (module-level for graceful degradation)
_fts_migration_status: FtsMigrationStatus = FtsMigrationStatus()


def _reset_fts_migration_status() -> None:
    """Reset FTS migration status to default (not in progress)."""
    global _fts_migration_status
    _fts_migration_status = FtsMigrationStatus()


def estimate_migration_time(records_count: int) -> int:
    """Estimate FTS migration time based on record count.

    Based on empirical testing:
    - 1,000 records: ~1-2 seconds
    - 10,000 records: ~5-15 seconds
    - 100,000 records: ~30-120 seconds
    - 1,000,000+ records: ~2-10 minutes

    Args:
        records_count: Number of records to migrate

    Returns:
        Estimated migration time in seconds (conservative estimate)
    """
    if records_count <= 1_000:
        return 2
    if records_count <= 10_000:
        return 15
    if records_count <= 100_000:
        return 120
    if records_count <= 1_000_000:
        return 600  # 10 minutes
    return 1200  # 20 minutes for very large datasets


# Dependency check functions for semantic search
async def check_semantic_search_dependencies(backend_type: str = 'sqlite') -> bool:
    """Check all semantic search dependencies.

    Performs comprehensive checks for:
    - Python packages: ollama, numpy, sqlite_vec (SQLite) or pgvector (PostgreSQL)
    - Ollama service availability
    - EmbeddingGemma model availability
    - sqlite-vec extension loading (SQLite only)

    Args:
        backend_type: Either 'sqlite' or 'postgresql'

    Returns:
        True if all dependencies are available, False otherwise
    """
    logger.info('Checking semantic search dependencies...')

    # Check ollama package
    try:
        import ollama

        logger.debug('[OK] ollama package available')
    except ImportError as e:
        logger.warning(f'[X] ollama package not available: {e}')
        return False

    # Check numpy package
    try:
        import importlib.util

        if importlib.util.find_spec('numpy') is None:
            logger.warning('[X] numpy package not available')
            return False
        logger.debug('[OK] numpy package available')
    except ImportError as e:
        logger.warning(f'[X] numpy package not available: {e}')
        return False

    # Check sqlite_vec package (SQLite only)
    if backend_type == 'sqlite':
        try:
            import importlib.util as sqlite_vec_util

            if sqlite_vec_util.find_spec('sqlite_vec') is None:
                logger.warning('[X] sqlite_vec package not available')
                return False
            logger.debug('[OK] sqlite_vec package available')
        except ImportError as e:
            logger.warning(f'[X] sqlite_vec package not available: {e}')
            return False

    # Check pgvector package (PostgreSQL only)
    if backend_type == 'postgresql':
        try:
            import importlib.util as pgvector_util

            if pgvector_util.find_spec('pgvector') is None:
                logger.warning('[X] pgvector package not available')
                logger.warning('  Install with: uv sync --extra semantic-search')
                return False
            logger.debug('[OK] pgvector package available')
        except ImportError as e:
            logger.warning(f'[X] pgvector package not available: {e}')
            logger.warning('  Install with: uv sync --extra semantic-search')
            return False

    # Check Ollama service
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(settings.ollama_host, timeout=1.0)
            if response.status_code == 200:
                logger.debug('[OK] Ollama service running')
            else:
                logger.warning(f'[X] Ollama service returned status {response.status_code}')
                return False
    except Exception as e:
        logger.warning(f'[X] Ollama service not accessible: {e}')
        return False

    # Check EmbeddingGemma model
    try:
        ollama_client = ollama.Client(host=settings.ollama_host)
        ollama_client.show(settings.embedding_model)
        logger.debug(f'[OK] EmbeddingGemma model "{settings.embedding_model}" available')
    except Exception as e:
        logger.warning(f'[X] EmbeddingGemma model not available: {e}')
        logger.warning(f'  Run: ollama pull {settings.embedding_model}')
        return False

    # Check sqlite-vec extension loading (SQLite only)
    if backend_type == 'sqlite':
        try:
            import sqlite3

            import sqlite_vec as sqlite_vec_ext

            test_conn = sqlite3.connect(':memory:')
            test_conn.enable_load_extension(True)
            sqlite_vec_ext.load(test_conn)
            test_conn.enable_load_extension(False)
            test_conn.close()
            logger.debug('[OK] sqlite-vec extension loads successfully')
        except Exception as e:
            logger.warning(f'[X] sqlite-vec extension failed to load: {e}')
            return False

    logger.info('[OK] All semantic search dependencies available')
    return True


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
    if not settings.enable_semantic_search:
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

    migration_path = Path(__file__).parent / 'migrations' / migration_filename

    if not migration_path.exists():
        logger.warning(f'Semantic search migration file not found: {migration_path}')
        return

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
            # Check if vector table exists
            row = await conn.fetchrow(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public' AND tablename = 'vec_context_embeddings'",
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
    if table_exists and existing_dim is not None and existing_dim != settings.embedding_dim:
        db_path = str(DB_PATH).replace('\\', '/')
        raise RuntimeError(
            f'Embedding dimension mismatch detected!\n'
            f'  Existing database dimension: {existing_dim}\n'
            f'  Configured EMBEDDING_DIM: {settings.embedding_dim}\n\n'
            f'To change embedding dimensions, you must:\n'
            f'  1. Back up your database: {db_path}\n'
            f'  2. Delete or rename the database file\n'
            f'  3. Restart the server to create new tables with dimension {settings.embedding_dim}\n'
            f'  4. Re-import your context data (embeddings will be regenerated)\n\n'
            f'Note: Changing dimensions will lose all existing embeddings.',
        )

    # Template the migration SQL with configured dimension
    migration_sql = migration_sql_template.replace(
        '{EMBEDDING_DIM}',
        str(settings.embedding_dim),
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
                    'sqlite-vec package required for semantic search migration. Install with: uv sync --extra semantic-search',
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
            # Just execute the migration SQL statements
            statements = []
            current_stmt = []
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
            f'Semantic search migration applied successfully with dimension: {settings.embedding_dim}',
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
    result = await conn.fetchval('''
        SELECT EXISTS (
            SELECT 1 FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = 'public'
              AND p.proname = 'jsonb_merge_patch'
        )
    ''')
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

    migration_path = Path(__file__).parent / 'migrations' / 'add_jsonb_merge_patch_postgresql.sql'

    if not migration_path.exists():
        logger.warning(f'jsonb_merge_patch migration file not found: {migration_path}')
        return

    try:
        migration_sql = migration_path.read_text(encoding='utf-8')

        if backend is not None:
            # Check if function already exists before applying
            function_exists = await backend.execute_read(cast(Any, _check_jsonb_merge_patch_exists))

            async def _apply_jsonb_merge_patch(conn: asyncpg.Connection) -> None:
                # Parse SQL statements, handling dollar-quoted function bodies
                statements = []
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
                    statements = []
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

                if function_exists:
                    logger.debug('jsonb_merge_patch function already exists, verified')
                else:
                    logger.info('Applied jsonb_merge_patch migration for PostgreSQL')
            finally:
                await temp_manager.shutdown()
    except Exception as e:
        logger.error(f'Failed to apply jsonb_merge_patch migration: {e}')
        raise RuntimeError(f'jsonb_merge_patch migration failed: {e}') from e


async def apply_fts_migration(backend: StorageBackend | None = None, repos: RepositoryContainer | None = None) -> None:
    """Apply full-text search migration if enabled, with language-aware tokenizer selection.

    Args:
        backend: Optional backend to use. If None, creates temporary backend.
        repos: Optional repository container. If None, creates temporary one.

    This function applies the FTS migration (FTS5 for SQLite, tsvector for PostgreSQL)
    when ENABLE_FTS=true. For SQLite, it selects the appropriate tokenizer based on
    FTS_LANGUAGE setting:
    - english (or not set) -> 'porter unicode61' (English stemming)
    - other languages -> 'unicode61' (multilingual, no stemming)

    If the language/tokenizer setting changes, migration will be triggered automatically.
    """
    # Skip if FTS is not enabled
    if not settings.enable_fts:
        logger.debug('FTS disabled (ENABLE_FTS=false), skipping migration')
        return

    # Determine backend type and get manager
    own_backend = False
    if backend is not None:
        manager = backend
        backend_type = backend.backend_type
    else:
        manager = create_backend(backend_type=None, db_path=DB_PATH)
        await manager.initialize()
        backend_type = manager.backend_type
        own_backend = True

    # Create repository if not provided
    fts_repo = repos.fts if repos else None
    if fts_repo is None:
        from app.repositories.fts_repository import FtsRepository

        fts_repo = FtsRepository(manager)

    try:
        # Check if FTS is already initialized
        fts_exists = await fts_repo.is_available()

        if fts_exists:
            # FTS exists - check if tokenizer/language matches current settings
            await _check_and_migrate_fts_if_needed(fts_repo, backend_type)
            if own_backend:
                await manager.shutdown()
            return

        # FTS doesn't exist - apply initial migration
        await _apply_initial_fts_migration(manager, backend_type)

    except Exception as e:
        # FTS migration failure should be logged but not fatal
        logger.warning(f'FTS migration may have already been applied or failed: {e}')
    finally:
        if own_backend:
            await manager.shutdown()


async def _check_and_migrate_fts_if_needed(fts_repo: FtsRepository, backend_type: str) -> None:
    """Check if FTS tokenizer/language matches settings and migrate if needed.

    Args:
        fts_repo: FTS repository instance
        backend_type: 'sqlite' or 'postgresql'
    """
    global _fts_migration_status

    if backend_type == 'sqlite':
        # Check current tokenizer
        current_tokenizer = await fts_repo.get_current_tokenizer()
        desired_tokenizer = await fts_repo.get_desired_tokenizer(settings.fts_language)

        if current_tokenizer != desired_tokenizer:
            logger.info(
                f'FTS tokenizer mismatch: current="{current_tokenizer}", desired="{desired_tokenizer}". '
                'Starting migration...',
            )

            # Get entry count for estimation
            stats = await fts_repo.get_statistics()
            records_count = stats.get('total_entries', 0)
            estimated_time = estimate_migration_time(records_count)

            # Set migration status for graceful degradation
            _fts_migration_status = FtsMigrationStatus(
                in_progress=True,
                started_at=datetime.now(tz=UTC),
                estimated_seconds=estimated_time,
                backend='sqlite',
                old_language=current_tokenizer,
                new_language=desired_tokenizer,
                records_count=records_count,
            )

            try:
                # Perform migration
                result = await fts_repo.migrate_tokenizer(desired_tokenizer)
                logger.info(
                    f'FTS tokenizer migration completed: {result["entries_migrated"]} entries '
                    f'migrated from "{result["old_tokenizer"]}" to "{result["new_tokenizer"]}"',
                )
            finally:
                # Reset migration status
                _reset_fts_migration_status()
        else:
            logger.debug(f'FTS tokenizer matches settings: "{current_tokenizer}"')

    else:  # postgresql
        # Check current language
        current_language = await fts_repo.get_current_language()
        desired_language = settings.fts_language

        if current_language and current_language != desired_language:
            logger.info(
                f'FTS language mismatch: current="{current_language}", desired="{desired_language}". '
                'Starting migration...',
            )

            # Get entry count for estimation
            stats = await fts_repo.get_statistics()
            records_count = stats.get('total_entries', 0)
            estimated_time = estimate_migration_time(records_count)

            # Set migration status for graceful degradation
            _fts_migration_status = FtsMigrationStatus(
                in_progress=True,
                started_at=datetime.now(tz=UTC),
                estimated_seconds=estimated_time,
                backend='postgresql',
                old_language=current_language,
                new_language=desired_language,
                records_count=records_count,
            )

            try:
                # Perform migration
                result = await fts_repo.migrate_language(desired_language)
                logger.info(
                    f'FTS language migration completed: {result["entries_migrated"]} entries '
                    f'migrated from "{result["old_language"]}" to "{result["new_language"]}"',
                )
            finally:
                # Reset migration status
                _reset_fts_migration_status()
        else:
            logger.debug(f'FTS language matches settings: "{current_language}"')


async def _apply_initial_fts_migration(manager: StorageBackend, backend_type: str) -> None:
    """Apply initial FTS migration for a fresh database.

    Args:
        manager: Storage backend
        backend_type: 'sqlite' or 'postgresql'
    """
    if backend_type == 'sqlite':
        # Read SQLite migration template (consistent with PostgreSQL approach)
        migration_path = Path(__file__).parent / 'migrations' / 'add_fts_sqlite.sql'
        if not migration_path.exists():
            logger.warning(f'FTS migration file not found: {migration_path}')
            return

        migration_sql = migration_path.read_text(encoding='utf-8')

        # Determine tokenizer based on language setting
        # - 'porter unicode61' for English (enables stemming: "running" matches "run")
        # - 'unicode61' for other languages (multilingual support, no stemming)
        tokenizer = 'porter unicode61' if settings.fts_language.lower() == 'english' else 'unicode61'
        migration_sql = migration_sql.replace('{TOKENIZER}', tokenizer)

        def _apply_fts_sqlite(conn: sqlite3.Connection) -> None:
            conn.executescript(migration_sql)

        await manager.execute_write(_apply_fts_sqlite)
        logger.info(f'Applied FTS migration (SQLite FTS5) with tokenizer: {tokenizer}')

    else:  # postgresql
        # Read PostgreSQL migration template
        migration_path = Path(__file__).parent / 'migrations' / 'add_fts_postgresql.sql'
        if not migration_path.exists():
            logger.warning(f'FTS migration file not found: {migration_path}')
            return

        migration_sql = migration_path.read_text(encoding='utf-8')
        migration_sql = migration_sql.replace('{FTS_LANGUAGE}', settings.fts_language)

        async def _apply_fts_pg(conn: asyncpg.Connection) -> None:
            statements: list[str] = []
            current_stmt: list[str] = []
            in_function = False

            for line in migration_sql.split('\n'):
                stripped = line.strip()
                if stripped.startswith('--'):
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

        await manager.execute_write(cast(Any, _apply_fts_pg))
        logger.info(f'Applied FTS migration (PostgreSQL tsvector) with language: {settings.fts_language}')


def _is_tool_disabled(tool_name: str) -> bool:
    """Check if a tool is in the disabled list.

    Args:
        tool_name: The name of the tool to check

    Returns:
        True if tool is disabled, False otherwise
    """
    return tool_name.lower() in settings.disabled_tools


def _register_tool(
    func: Callable[..., Any],
    name: str | None = None,
    annotations: ToolAnnotations | None = None,
) -> bool:
    """Register a tool only if it's not in the disabled list.

    Args:
        func: The tool function to register
        name: Optional explicit tool name (defaults to function name)
        annotations: Optional ToolAnnotations for hints

    Returns:
        True if tool was registered, False if disabled
    """
    tool_name = name or func.__name__

    if _is_tool_disabled(tool_name):
        logger.info(f'[!] {tool_name} not registered (in DISABLED_TOOLS)')
        return False

    # Register with annotations if provided
    if annotations is not None:
        mcp.tool(annotations=annotations)(func)
    else:
        mcp.tool()(func)

    logger.info(f'[OK] {tool_name} registered')
    return True


# Lifespan context manager for FastMCP
@asynccontextmanager
async def lifespan(_: FastMCP[None]) -> AsyncGenerator[None, None]:
    """Manage server lifecycle - initialize on startup, cleanup on shutdown.

    This ensures that the database manager's background tasks run in the
    same event loop as FastMCP, preventing the hanging issue.

    Args:
        _: The FastMCP server instance (unused but required by signature)

    Yields:
        None: Control is yielded back to FastMCP during server operation
    """
    global _backend, _repositories, _embedding_service

    # Startup
    try:
        # Create backend ONCE at the start - used throughout initialization and runtime
        _backend = create_backend(backend_type=None, db_path=DB_PATH)
        await _backend.initialize()
        # 1) Ensure schema exists using the shared backend
        await init_database(backend=_backend)
        # 2) Apply semantic search migration if enabled using the shared backend
        await apply_semantic_search_migration(backend=_backend)
        # 3) Apply jsonb_merge_patch migration for PostgreSQL (required for metadata_patch)
        await apply_jsonb_merge_patch_migration(backend=_backend)
        # 4) Apply FTS migration if enabled
        await apply_fts_migration(backend=_backend)
        # 5) Initialize repositories with the backend
        _repositories = RepositoryContainer(_backend)

        # 6) Register core tools with appropriate annotations
        # Additive tools (create new entries)
        _register_tool(store_context, annotations=ADDITIVE_ANNOTATIONS)
        _register_tool(store_context_batch, annotations=ADDITIVE_ANNOTATIONS)

        # Read-only tools (no modifications)
        _register_tool(search_context, annotations=READ_ONLY_ANNOTATIONS)
        _register_tool(get_context_by_ids, annotations=READ_ONLY_ANNOTATIONS)
        _register_tool(list_threads, annotations=READ_ONLY_ANNOTATIONS)
        _register_tool(get_statistics, annotations=READ_ONLY_ANNOTATIONS)

        # Update tools (destructive, not idempotent)
        _register_tool(update_context, annotations=UPDATE_ANNOTATIONS)
        _register_tool(update_context_batch, annotations=UPDATE_ANNOTATIONS)

        # Delete tools (destructive, idempotent)
        _register_tool(delete_context, annotations=DELETE_ANNOTATIONS)
        _register_tool(delete_context_batch, annotations=DELETE_ANNOTATIONS)

        # 7) Initialize semantic search if enabled
        if settings.enable_semantic_search:
            semantic_available = await check_semantic_search_dependencies(_backend.backend_type)

            if semantic_available:
                try:
                    from app.services.embedding_service import EmbeddingService

                    _embedding_service = EmbeddingService()
                    # Register the semantic search tool dynamically (DISABLED_TOOLS takes priority)
                    _register_tool(semantic_search_context, annotations=READ_ONLY_ANNOTATIONS)
                    logger.info('[OK] Semantic search enabled and available')
                except Exception as e:
                    logger.error(f'Failed to initialize embedding service: {e}')
                    _embedding_service = None
                    logger.warning('[!] Semantic search enabled but initialization failed - feature disabled')
                    logger.info('[!] semantic_search_context not registered (initialization failed)')
            else:
                _embedding_service = None
                logger.warning('[!] Semantic search enabled but dependencies not met - feature disabled')
                logger.warning('  Install dependencies: uv sync --extra semantic-search')
                logger.warning(f'  Download model: ollama pull {settings.embedding_model}')
                logger.info('[!] semantic_search_context not registered (dependencies not met)')
        else:
            _embedding_service = None
            logger.info('Semantic search disabled (ENABLE_SEMANTIC_SEARCH=false)')
            logger.info('[!] semantic_search_context not registered (feature disabled)')

        # 8) Register FTS tool if enabled - ALWAYS register when ENABLE_FTS=true
        # The tool handles graceful degradation during migration
        if settings.enable_fts:
            # Always register the FTS tool when enabled (DISABLED_TOOLS takes priority)
            # The tool itself checks migration status and returns informative response
            _register_tool(fts_search_context, annotations=READ_ONLY_ANNOTATIONS)

            # Check if FTS is available and log status
            fts_available = await _repositories.fts.is_available()
            if fts_available:
                logger.info('[OK] Full-text search enabled and available')
            else:
                logger.warning('[!] FTS enabled but index may need initialization or migration')
        else:
            logger.info('Full-text search disabled (ENABLE_FTS=false)')
            logger.info('[!] fts_search_context not registered (feature disabled)')

        # 9) Register Hybrid Search tool if enabled AND at least one search mode is available
        if settings.enable_hybrid_search:
            semantic_available_for_hybrid = settings.enable_semantic_search and _embedding_service is not None
            fts_available_for_hybrid = settings.enable_fts

            if semantic_available_for_hybrid or fts_available_for_hybrid:
                # DISABLED_TOOLS takes priority over ENABLE_HYBRID_SEARCH
                _register_tool(hybrid_search_context, annotations=READ_ONLY_ANNOTATIONS)
                modes_available = []
                if fts_available_for_hybrid:
                    modes_available.append('fts')
                if semantic_available_for_hybrid:
                    modes_available.append('semantic')
                logger.info(f'[OK] hybrid_search_context modes available: {modes_available}')
            else:
                logger.warning(
                    '[!] Hybrid search enabled but no search modes available - feature disabled. '
                    'Enable ENABLE_FTS=true and/or ENABLE_SEMANTIC_SEARCH=true.',
                )
                logger.info('[!] hybrid_search_context not registered (no search modes available)')
        else:
            logger.info('Hybrid search disabled (ENABLE_HYBRID_SEARCH=false)')
            logger.info('[!] hybrid_search_context not registered (feature disabled)')

        logger.info(f'MCP Context Server initialized (backend: {_backend.backend_type})')
    except Exception as e:
        logger.error(f'Failed to initialize server: {e}')
        if _backend:
            await _backend.shutdown()
        raise

    # Yield control to FastMCP
    yield

    # Shutdown
    logger.info('Shutting down MCP Context Server')
    # At this point, startup succeeded and _backend must be set
    assert _backend is not None
    try:
        await _backend.shutdown()
    except Exception as e:
        logger.error(f'Error during shutdown: {e}')
    finally:
        _backend = None
        _repositories = None
        _embedding_service = None
    logger.info('MCP Context Server shutdown complete')


# Initialize FastMCP server with lifespan management
# mask_error_details=False exposes validation errors for LLM autocorrection
mcp = FastMCP(name='mcp-context-server', lifespan=lifespan, mask_error_details=False)


@mcp.custom_route('/health', methods=['GET'])
async def health(_: Request) -> JSONResponse:
    """Health check endpoint for container orchestration.

    Returns simple status for Docker/Kubernetes liveness probes.
    This endpoint is only available when running in HTTP transport mode.
    """
    return JSONResponse({'status': 'ok'})


async def init_database(backend: StorageBackend | None = None) -> None:
    """Initialize database schema.

    Args:
        backend: Optional backend to use. If None, creates temporary backend for backward compatibility.

    This function can work in two modes:
    1. With backend parameter (normal server startup): Uses provided backend, no temp backend created
    2. Without backend parameter (tests/direct calls): Creates temporary backend for isolation

    Raises:
        RuntimeError: If no schema file found or backend initialization fails.
    """
    try:
        # Ensure database path exists (only for file-based backends)
        if DB_PATH:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not DB_PATH.exists():
                DB_PATH.touch()

        # Determine backend type to select correct schema file
        if backend is not None:
            backend_type = backend.backend_type
        else:
            # Create temporary backend to determine type
            temp_backend = create_backend(backend_type=None, db_path=DB_PATH)
            backend_type = temp_backend.backend_type

        # Select schema file based on backend type
        schema_filename = 'postgresql_schema.sql' if backend_type == 'postgresql' else 'sqlite_schema.sql'

        schema_path = Path(__file__).parent / 'schemas' / schema_filename

        # Read schema from file
        if schema_path.exists():
            schema_sql = schema_path.read_text(encoding='utf-8')
        else:
            raise RuntimeError(f'Schema file not found: {schema_path}')

        # Apply schema - backend-specific approach
        if backend is not None:
            # Use provided backend (normal server startup)
            if backend.backend_type == 'sqlite':

                def _init_schema_sqlite(conn: sqlite3.Connection) -> None:
                    # Single executescript to create all objects atomically
                    conn.executescript(schema_sql)

                await backend.execute_write(_init_schema_sqlite)
            else:  # postgresql

                async def _init_schema_postgresql(conn: asyncpg.Connection) -> None:
                    # PostgreSQL: parse and execute statements individually
                    statements = []
                    current_stmt = []
                    in_function = False

                    for line in schema_sql.split('\n'):
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

                await backend.execute_write(cast(Any, _init_schema_postgresql))
            logger.info(f'Database schema initialized successfully ({backend.backend_type})')
        else:
            # Backward compatibility: create temporary backend for tests
            temp_manager = create_backend(backend_type=None, db_path=DB_PATH)
            await temp_manager.initialize()
            try:
                if temp_manager.backend_type == 'sqlite':

                    def _init_schema_sqlite(conn: sqlite3.Connection) -> None:
                        conn.executescript(schema_sql)

                    await temp_manager.execute_write(_init_schema_sqlite)
                else:  # postgresql

                    async def _init_schema_postgresql(conn: asyncpg.Connection) -> None:
                        # PostgreSQL: parse and execute statements individually
                        statements = []
                        current_stmt = []
                        in_function = False

                        for line in schema_sql.split('\n'):
                            stripped = line.strip()
                            if stripped.startswith('--'):
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

                    await temp_manager.execute_write(cast(Any, _init_schema_postgresql))
                logger.info(f'Database schema initialized successfully ({temp_manager.backend_type})')
            finally:
                # Always shutdown to stop background tasks and close connections
                await temp_manager.shutdown()
    except Exception as e:
        logger.error(f'Failed to initialize database: {e}')
        raise


# Utility functions


async def _ensure_backend() -> StorageBackend:
    """Ensure a connection manager exists and is initialized.

    In tests, FastMCP lifespan isn't running, so tools need a lazy
    initializer to operate directly.

    Returns:
        Initialized `StorageBackend` singleton to use for DB ops.
    """
    global _backend
    if _backend is None:
        manager = create_backend(backend_type=None, db_path=DB_PATH)
        await manager.initialize()
        _backend = manager
    return _backend


async def _ensure_repositories() -> RepositoryContainer:
    """Ensure repositories are initialized.

    Returns:
        Initialized repository container.
    """
    global _repositories
    if _repositories is None:
        manager = await _ensure_backend()
        _repositories = RepositoryContainer(manager)
    return _repositories


def deserialize_json_param(
    value: JsonValue | None,
) -> JsonValue | None:
    """Deserialize JSON string parameters if needed with enhanced safety checks.

    COMPATIBILITY NOTE: This function works around a known issue where some MCP clients
    (including Claude Code) send complex parameters as JSON strings instead of native
    Python objects. This is documented in multiple GitHub issues:
    - FastMCP #932: JSON Arguments Encapsulated as String Cause Validation Failure
    - Claude Code #5504: JSON objects converted to quoted strings
    - Claude Code #4192: Consecutive parameter calls fail
    - Claude Code #3084: Pydantic model parameters cause validation errors

    Enhanced to handle:
    - Double-encoding issues (JSON within JSON)
    - Single string values that should be treated as tags
    - Edge cases with special characters like forward slashes

    This function can be removed when the upstream issues are resolved.

    Args:
        value: The parameter value which might be a JSON string

    Returns:
        The deserialized value if it was a JSON string, or the original value
    """
    if isinstance(value, str):
        try:
            result = json.loads(value)
            # Check for double-encoding (JSON string within JSON)
            if isinstance(result, str):
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    # Try to decode again in case of double-encoding
                    result = json.loads(result)
            return cast(JsonValue | None, result)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON - check if it's meant to be a single tag
            if value.strip():
                # For tags parameter, a single string should become a list
                # This helps handle edge cases where a single tag is passed as string
                # The caller will need to handle this appropriately
                pass
            return value
    return value


def truncate_text(text: str | None, max_length: int = 150) -> tuple[str | None, bool]:
    """
    Truncate text at word boundary when possible.

    Args:
        text: The text to truncate
        max_length: Maximum character length (default: 150)

    Returns:
        tuple: (truncated_text, is_truncated)
    """
    if not text or len(text) <= max_length:
        return text, False

    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > max_length * 0.7:  # Only use word boundary if it's not too short
        truncated = truncated[:last_space]

    return truncated + '...', True


def validate_date_param(date_str: str | None, param_name: str) -> str | None:
    """Validate and normalize date parameter for database filtering.

    Accepts ISO 8601 format dates and returns the validated string for database use.
    Both date-only (YYYY-MM-DD) and datetime (YYYY-MM-DDTHH:MM:SS) formats are supported.
    Timezone suffixes (Z or +HH:MM) are also accepted.

    For end_date with date-only format: automatically expands to end-of-day (T23:59:59)
    to match user expectations. This follows Elasticsearch's precedent where missing time
    components are replaced with max values for 'lte' (less-than-or-equal) operations.
    See: https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-range-query

    Args:
        date_str: ISO 8601 date string or None
        param_name: Parameter name for error messages (e.g., 'start_date', 'end_date')

    Returns:
        Validated date string (possibly expanded for end_date) or None if input was None

    Raises:
        ToolError: If date format is invalid
    """
    from datetime import date
    from datetime import datetime as dt

    if date_str is None:
        return None

    # Detect date-only format by checking for absence of time separators
    # Date-only: '2025-11-29' (no 'T' or space separator)
    # Datetime: '2025-11-29T10:00:00' or '2025-11-29 10:00:00'
    is_date_only = 'T' not in date_str and ' ' not in date_str

    # Validate the date string
    try:
        if is_date_only:
            # Parse as date-only format (YYYY-MM-DD)
            date.fromisoformat(date_str)
        else:
            # Parse as datetime format (with optional timezone)
            # Python 3.11+ handles 'Z' natively
            dt.fromisoformat(date_str)
    except ValueError:
        raise ToolError(
            f'Invalid {param_name} format: "{date_str}". '
            f'Use ISO 8601 format (e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ) from None

    # For end_date with date-only format: expand to end-of-day with microsecond precision
    # This follows Elasticsearch precedent where missing time components are replaced
    # with max values for 'lte' operations, matching user expectations that
    # end_date='2025-11-29' should include ALL entries on November 29th.
    #
    # Uses T23:59:59.999999 (microsecond precision) for PostgreSQL compatibility:
    # PostgreSQL's CURRENT_TIMESTAMP stores microseconds (e.g., 23:59:59.500000),
    # so T23:59:59 (microsecond=0) would exclude entries at 23:59:59.xxx.
    # SQLite is unaffected as CURRENT_TIMESTAMP stores second precision only.
    if param_name == 'end_date' and is_date_only:
        date_str = f'{date_str}T23:59:59.999999'

    return date_str


def validate_date_range(start_date: str | None, end_date: str | None) -> None:
    """Validate that start_date is not after end_date.

    Args:
        start_date: Validated start date string
        end_date: Validated end date string

    Raises:
        ToolError: If start_date is after end_date
    """
    from datetime import date
    from datetime import datetime as dt

    def _parse_and_normalize(date_str: str) -> dt:
        """Parse date string and normalize to naive datetime for comparison.

        Handles all ISO 8601 formats: date-only, datetime, datetime+tz, datetime+Z.
        Strips timezone info to allow comparison between mixed formats.

        Returns:
            Naive datetime object for comparison purposes.
        """
        # Handle Z suffix - replace with +00:00 for fromisoformat
        normalized = date_str.replace('Z', '+00:00') if date_str.endswith('Z') else date_str

        try:
            parsed = dt.fromisoformat(normalized)
            # Strip timezone info for comparison (we just need relative ordering)
            return parsed.replace(tzinfo=None)
        except ValueError:
            # Date-only format - convert to datetime for comparison
            return dt.combine(date.fromisoformat(date_str), dt.min.time())

    if start_date and end_date:
        start_dt = _parse_and_normalize(start_date)
        end_dt = _parse_and_normalize(end_date)

        if start_dt > end_dt:
            raise ToolError(
                f'Invalid date range: start_date ({start_date}) is after end_date ({end_date})',
            )


# MCP Tools


async def store_context(
    thread_id: Annotated[str, Field(min_length=1, description='Unique identifier for the conversation/task thread')],
    source: Annotated[Literal['user', 'agent'], Field(description="Either 'user' or 'agent'")],
    text: Annotated[str, Field(min_length=1, description='Text content to store')],
    images: Annotated[
        list[dict[str, str]] | None,
        Field(description='List of base64 encoded images with mime_type. Each image max 10MB, total max 100MB'),
    ] = None,
    metadata: Annotated[
        MetadataDict | None,
        Field(
            description='Additional structured data. For optimal performance, consider using indexed field names: '
            'status (state information), priority (numeric value for range queries), '
            'agent_name (specific agent identifier), task_name (task title for string searches), '
            'completed (boolean flag for completion state). '
            'These fields are indexed for faster filtering but not required.',
        ),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='List of tags (normalized to lowercase)')] = None,
    ctx: Context | None = None,
) -> StoreContextSuccessDict:
    """Store a context entry with optional images and metadata.

    All agents working on the same task should use the same thread_id to share context.
    If an entry with identical thread_id, source, and text already exists, it will be
    updated instead of creating a duplicate (deduplication).

    Notes:
    - Tags are normalized to lowercase and stored separately for efficient filtering
    - If semantic search is enabled, an embedding is automatically generated
    - Use indexed metadata fields (status, priority, agent_name, task_name, completed)
      for faster filtering in search_context

    Returns: {
        success: bool,
        context_id: int,
        thread_id: str,
        message: str
    }
    """
    try:
        # Clean input strings - defensive try/except handles edge cases where Pydantic validation bypassed
        try:
            thread_id = thread_id.strip()
        except AttributeError:
            raise ToolError('thread_id is required') from None
        try:
            text = text.strip()
        except AttributeError:
            raise ToolError('text is required') from None

        # Business logic: empty strings after stripping are not allowed
        if not thread_id:
            raise ToolError('thread_id cannot be empty or whitespace')
        if not text:
            raise ToolError('text cannot be empty or whitespace')

        # Validate images if provided
        if images:
            for idx, img in enumerate(images):
                if 'data' in img:
                    try:
                        base64.b64decode(img['data'])
                    except Exception:
                        raise ToolError(f'Invalid base64 encoded data in image {idx}') from None

        # Log info if context is available
        if ctx:
            await ctx.info(f'Storing context for thread: {thread_id}')

        # Deserialize JSON parameters if needed
        images_raw = deserialize_json_param(cast(JsonValue | None, images))
        images = cast(list[dict[str, str]] | None, images_raw)
        tags_raw = deserialize_json_param(cast(JsonValue | None, tags))
        tags = cast(list[str] | None, tags_raw)
        metadata_raw = deserialize_json_param(cast(JsonValue | None, metadata))
        metadata = cast(MetadataDict | None, metadata_raw)

        # Get repositories
        repos = await _ensure_repositories()

        # Determine content type
        content_type = 'multimodal' if images else 'text'

        # Store context entry with deduplication
        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            text_content=text,
            metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None,
        )

        # Ensure we got a valid ID (not None or 0)
        if not context_id:
            raise ToolError('Failed to store context')

        # Store normalized tags
        if tags:
            await repos.tags.store_tags(context_id, tags)

        # Store images if provided
        total_size: float = 0.0
        valid_image_count = 0
        if images:
            # Pre-validate ALL images before storing any
            for idx, img in enumerate(images):
                # Validate required data field
                if 'data' not in img:
                    raise ToolError(f'Image {idx} is missing required "data" field')

                img_data_str = img.get('data', '')
                if not img_data_str or not img_data_str.strip():
                    raise ToolError(f'Image {idx} has empty "data" field')

                # mime_type is optional - defaults to 'image/png' if not provided
                if 'mime_type' not in img:
                    img['mime_type'] = 'image/png'

                # Validate base64 encoding
                try:
                    image_binary = base64.b64decode(img_data_str)
                except Exception as e:
                    raise ToolError(f'Image {idx} has invalid base64 encoding: {str(e)}') from None

                # Validate image size
                image_size_mb = len(image_binary) / (1024 * 1024)

                if image_size_mb > MAX_IMAGE_SIZE_MB:
                    raise ToolError(f'Image {idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit')

                total_size += image_size_mb
                if total_size > MAX_TOTAL_SIZE_MB:
                    raise ToolError(f'Total size exceeds {MAX_TOTAL_SIZE_MB}MB limit')

                valid_image_count += 1

            # All validations passed, store the images
            logger.debug(f'Pre-validation passed for {valid_image_count} images, total size: {total_size:.2f}MB')
            try:
                await repos.images.store_images(context_id, images)
            except Exception as e:
                raise ToolError(f'Failed to store images: {str(e)}') from e

        # Generate embedding if semantic search is available (non-blocking)
        embedding_generated = False
        if _embedding_service is not None:
            try:
                embedding = await _embedding_service.generate_embedding(text)
                await repos.embeddings.store(
                    context_id=context_id,
                    embedding=embedding,
                    model=settings.embedding_model,
                )
                embedding_generated = True
                logger.debug(f'Generated embedding for context {context_id}')
            except Exception as e:
                logger.warning(f'Failed to generate/store embedding for context {context_id}: {e}')
                # Non-blocking: continue even if embedding fails

        action = 'updated' if was_updated else 'stored'
        logger.info(f'{action.capitalize()} context {context_id} in thread {thread_id}')

        return StoreContextSuccessDict(
            success=True,
            context_id=context_id,
            thread_id=thread_id,
            message=f'Context {action} with {len(images) if images else 0} images'
            + (' (embedding generated)' if embedding_generated else ''),
        )
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error storing context: {e}')
        raise ToolError(f'Failed to store context: {str(e)}') from e


async def search_context(
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 30)')] = 30,
    thread_id: Annotated[str | None, Field(min_length=1, description='Filter by thread (indexed)')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Filter by source type (indexed)')] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    content_type: Annotated[Literal['text', 'multimodal'] | None, Field(description='Filter by content type')] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null',
        ),
    ] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Search context entries with filtering. Returns TRUNCATED text_content (150 chars max).

    Use get_context_by_ids to retrieve full content for specific entries of interest.

    Filtering options:
    - thread_id, source: Indexed for fast filtering (always prefer specifying thread_id)
    - tags: OR logic (matches ANY of provided tags)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)
    - start_date/end_date: Filter by creation timestamp (ISO 8601)

    Performance tips:
    - Always specify thread_id to reduce search space
    - Use indexed metadata fields: status, priority, agent_name, task_name, completed

    Returns: {
        results: [{id, thread_id, source, text_content, metadata, tags, ...}],
        count: int,
        stats: {...}  # Only when explain_query=True
    }
    """
    try:
        # Validate date parameters
        start_date = validate_date_param(start_date, 'start_date')
        end_date = validate_date_param(end_date, 'end_date')
        validate_date_range(start_date, end_date)

        if ctx:
            await ctx.info(f'Searching context with filters: thread_id={thread_id}, source={source}')

        # Get repositories
        repos = await _ensure_repositories()

        # Use the improved search_contexts method that now supports metadata and date filtering
        result = await repos.context.search_contexts(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            metadata=metadata,
            metadata_filters=metadata_filters,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
            offset=offset,
            explain_query=explain_query,
        )

        # Always expect tuple from repository
        rows, stats = result

        # Check for validation errors in stats
        if 'error' in stats:
            # Return the error response with validation details
            error_response: dict[str, Any] = {
                'results': [],
                'count': 0,
                'error': stats.get('error', 'Unknown error'),
            }
            if 'validation_errors' in stats:
                error_response['validation_errors'] = stats['validation_errors']
            return error_response

        entries: list[ContextEntryDict] = []

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))

            # Parse JSON metadata - database stores as JSON string
            metadata_raw = entry.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    entry['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    entry['metadata'] = None

            # Get normalized tags
            entry_id_raw = entry.get('id')
            if entry_id_raw is not None:
                entry_id = int(entry_id_raw)
                tags_result = await repos.tags.get_tags_for_context(entry_id)
                entry['tags'] = tags_result
            else:
                entry['tags'] = []

            # Apply text truncation for search_context
            text_content = entry.get('text_content', '')
            truncated_text, is_truncated = truncate_text(text_content)
            entry['text_content'] = truncated_text
            entry['is_truncated'] = is_truncated

            # Fetch images if requested and applicable
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id = int(entry.get('id', 0))
                images_result = await repos.images.get_images_for_context(entry_id, include_data=True)
                entry['images'] = cast(list[dict[str, str]], images_result)

            entries.append(entry)

        # Return dict with results, count, and optional stats
        response: dict[str, Any] = {'results': entries, 'count': len(entries)}
        if explain_query:
            response['stats'] = stats
        return response
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error searching context: {e}')
        raise ToolError(f'Failed to search context: {str(e)}') from e


async def get_context_by_ids(
    context_ids: Annotated[list[int], Field(min_length=1, description='List of context entry IDs to retrieve')],
    include_images: Annotated[bool, Field(description='Whether to include image data')] = True,
    ctx: Context | None = None,
) -> list[ContextEntryDict]:
    """Fetch specific context entries by their IDs with FULL (non-truncated) text content.

    Use this after search_context to retrieve complete content for entries of interest,
    or when you have specific context IDs from previous operations.

    Workflow: search_context (browse, truncated) -> get_context_by_ids (retrieve full content)

    Returns: [
        {id, thread_id, source, text_content, metadata, tags, images, created_at, updated_at}
    ]
    """
    try:
        if ctx:
            await ctx.info(f'Fetching context entries: {context_ids}')

        # Get repositories
        repos = await _ensure_repositories()

        # Fetch context entries using repository
        rows = await repos.context.get_by_ids(context_ids)
        entries: list[ContextEntryDict] = []

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))

            # Parse JSON metadata - database stores as JSON string
            metadata_raw = entry.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    entry['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    entry['metadata'] = None

            # Get normalized tags
            entry_id_raw = entry.get('id')
            if entry_id_raw is not None:
                entry_id = int(entry_id_raw)
                tags_result = await repos.tags.get_tags_for_context(entry_id)
                entry['tags'] = tags_result
            else:
                entry['tags'] = []

            # Fetch images
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id_img = entry.get('id')
                if entry_id_img is not None:
                    images_result = await repos.images.get_images_for_context(int(entry_id_img), include_data=True)
                    entry['images'] = cast(list[dict[str, str]], images_result)
                else:
                    entry['images'] = []

            entries.append(entry)

        return entries
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error fetching context by IDs: {e}')
        raise ToolError(f'Failed to fetch context entries: {str(e)}') from e


async def delete_context(
    context_ids: Annotated[
        list[int] | None,
        Field(min_length=1, description='Specific context entry IDs to delete (mutually exclusive with thread_id)'),
    ] = None,
    thread_id: Annotated[
        str | None,
        Field(min_length=1, description='Delete ALL entries in thread (mutually exclusive with context_ids)'),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, bool | int | str]:
    """Delete context entries by specific IDs or by entire thread. IRREVERSIBLE.

    Provide EITHER context_ids OR thread_id (not both). Cascading delete removes
    associated tags, images, and embeddings.

    WARNING: This operation cannot be undone. Verify IDs/thread before deletion.

    Returns: {
        success: bool,
        deleted_count: int,
        message: str
    }
    """
    try:
        # Ensure at least one parameter is provided (business logic validation)
        if not context_ids and not thread_id:
            raise ToolError('Must provide either context_ids or thread_id')

        if ctx:
            await ctx.info(f'Deleting context: ids={context_ids}, thread={thread_id}')

        # Get repositories
        repos = await _ensure_repositories()

        deleted = 0

        if context_ids:
            # Delete embeddings first (explicit cleanup)
            if settings.enable_semantic_search:
                for context_id in context_ids:
                    try:
                        await repos.embeddings.delete(context_id)
                    except Exception as e:
                        logger.warning(f'Failed to delete embedding for context {context_id}: {e}')
                        # Non-blocking: continue even if embedding deletion fails

            deleted = await repos.context.delete_by_ids(context_ids)
            logger.info(f'Deleted {deleted} context entries by IDs')

        elif thread_id:
            # Get all context IDs in thread for embedding cleanup
            if settings.enable_semantic_search:
                try:
                    # Get all context IDs in this thread
                    results = await repos.context.search_contexts(
                        thread_id=thread_id,
                        limit=10000,  # Large limit to get all
                        offset=0,
                        explain_query=False,
                    )
                    rows, _ = results

                    # Delete embeddings for all contexts in thread
                    for row in rows:
                        context_id = row['id']  # sqlite3.Row supports __getitem__
                        if context_id:
                            try:
                                await repos.embeddings.delete(int(context_id))
                            except Exception as e:
                                logger.warning(f'Failed to delete embedding for context {context_id}: {e}')
                except Exception as e:
                    logger.warning(f'Failed to cleanup embeddings for thread {thread_id}: {e}')
                    # Non-blocking: continue with context deletion

            deleted = await repos.context.delete_by_thread(thread_id)
            logger.info(f'Deleted {deleted} entries from thread {thread_id}')

        return {
            'success': True,
            'deleted_count': deleted,
            'message': f'Successfully deleted {deleted} context entries',
        }
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error deleting context: {e}')
        raise ToolError(f'Failed to delete context: {str(e)}') from e


async def update_context(
    context_id: Annotated[int, Field(gt=0, description='ID of the context entry to update')],
    text: Annotated[str | None, Field(min_length=1, description='New text content (replaces existing)')] = None,
    metadata: Annotated[MetadataDict | None, Field(description='New metadata (FULL REPLACEMENT)')] = None,
    metadata_patch: Annotated[
        MetadataDict | None,
        Field(
            description='Partial metadata update (RFC 7396 JSON Merge Patch): new keys added, '
            'existing updated, null values DELETE keys. MUTUALLY EXCLUSIVE with metadata.',
        ),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='New tags list (REPLACES all existing)')] = None,
    images: Annotated[
        list[dict[str, str]] | None,
        Field(description='New images with base64 data and mime_type (REPLACES all existing)'),
    ] = None,
    ctx: Context | None = None,
) -> UpdateContextSuccessDict:
    """Update an existing context entry. Only provided fields are modified.

    Immutable fields: id, thread_id, source, created_at (cannot be changed)
    Auto-managed: content_type (recalculated based on images), updated_at

    Metadata options (MUTUALLY EXCLUSIVE):
    - metadata: FULL REPLACEMENT of entire metadata object
    - metadata_patch: RFC 7396 JSON Merge Patch - merge with existing
      - New keys added, existing keys updated, null values DELETE keys
      - Limitation: Cannot store null values (use full replacement instead)
      - Limitation: Arrays replaced entirely (no element-wise merge)

    Tags and images use REPLACEMENT semantics (not merge).

    Returns: {
        success: bool,
        context_id: int,
        updated_fields: [str, ...],
        message: str
    }
    """
    try:
        # Clean text input if provided
        if text is not None:
            text = text.strip()
            # Business logic: if text provided, it cannot be empty after stripping
            if not text:
                raise ToolError('text cannot be empty or contain only whitespace')

        # Validate mutual exclusivity: metadata and metadata_patch cannot be used together
        # RFC 7396 Note: metadata_patch is for partial updates (merge), metadata is for full replacement
        if metadata is not None and metadata_patch is not None:
            raise ToolError(
                'Cannot use both metadata and metadata_patch parameters together. '
                'Use metadata for full replacement or metadata_patch for partial updates.',
            )

        # Validate that at least one field is provided for update
        # Note: metadata_patch is also a valid update field
        if text is None and metadata is None and metadata_patch is None and tags is None and images is None:
            raise ToolError('At least one field must be provided for update')

        if ctx:
            await ctx.info(f'Updating context entry {context_id}')

        # Get repositories
        repos = await _ensure_repositories()

        # Check if entry exists
        exists = await repos.context.check_entry_exists(context_id)
        if not exists:
            raise ToolError(f'Context entry with ID {context_id} not found')

        updated_fields: list[str] = []

        # Start transaction-like operations
        try:
            # Update text content and/or metadata (full replacement) if provided
            if text is not None or metadata is not None:
                # Prepare metadata JSON string if provided
                metadata_str: str | None = None
                if metadata is not None:
                    metadata_str = json.dumps(metadata, ensure_ascii=False)

                # Update context entry
                success, fields = await repos.context.update_context_entry(
                    context_id=context_id,
                    text_content=text,
                    metadata=metadata_str,
                )

                if not success:
                    raise ToolError('Failed to update context entry')

                updated_fields.extend(fields)

            # Apply metadata patch (partial update) if provided
            # RFC 7396 JSON Merge Patch: merges with existing metadata
            # - New keys are added
            # - Existing keys are replaced with new values
            # - null values DELETE keys (cannot store null values with patch)
            if metadata_patch is not None:
                success, fields = await repos.context.patch_metadata(
                    context_id=context_id,
                    patch=metadata_patch,
                )

                if not success:
                    raise ToolError('Failed to patch metadata')

                updated_fields.extend(fields)
                logger.debug(f'Applied metadata patch to context {context_id}')

            # Replace tags if provided
            if tags is not None:
                await repos.tags.replace_tags_for_context(context_id, tags)
                updated_fields.append('tags')
                logger.debug(f'Replaced tags for context {context_id}')

            # Replace images if provided
            if images is not None:
                # If images list is empty (removing all images), update content_type to text
                if len(images) == 0:
                    await repos.images.replace_images_for_context(context_id, [])
                    await repos.context.update_content_type(context_id, 'text')
                    updated_fields.extend(['images', 'content_type'])
                    logger.debug(f'Removed all images from context {context_id}')
                else:
                    # Validate image data first
                    total_size = 0.0
                    for img in images:
                        if 'data' not in img or 'mime_type' not in img:
                            raise ToolError('Each image must have "data" and "mime_type" fields')

                        # Check individual image size
                        try:
                            img_data = base64.b64decode(img['data'])
                        except Exception:
                            raise ToolError('Invalid base64 image data') from None

                        img_size_mb = len(img_data) / (1024 * 1024)
                        total_size += img_size_mb

                        if img_size_mb > MAX_IMAGE_SIZE_MB:
                            raise ToolError(f'Image exceeds size limit of {MAX_IMAGE_SIZE_MB}MB')

                    # Check total size
                    if total_size > MAX_TOTAL_SIZE_MB:
                        raise ToolError(f'Total image size {total_size:.2f}MB exceeds limit of {MAX_TOTAL_SIZE_MB}MB')

                    # Replace images
                    await repos.images.replace_images_for_context(context_id, images)
                    updated_fields.append('images')

                    # Update content_type to multimodal if images were added
                    await repos.context.update_content_type(context_id, 'multimodal')
                    updated_fields.append('content_type')
                    logger.debug(f'Replaced images for context {context_id}')

            # Check if we need to update content_type based on current state
            if images is None and (text is not None or metadata is not None):
                # Check if there are existing images to determine content_type
                image_count = await repos.images.count_images_for_context(context_id)
                current_content_type = 'multimodal' if image_count > 0 else 'text'

                # Get the stored content type
                stored_content_type = await repos.context.get_content_type(context_id)

                # Update if different
                if stored_content_type != current_content_type:
                    await repos.context.update_content_type(context_id, current_content_type)
                    updated_fields.append('content_type')

            # Regenerate embedding if text was changed and semantic search is available (non-blocking)
            if text is not None and _embedding_service is not None:
                try:
                    new_embedding = await _embedding_service.generate_embedding(text)

                    # Check if embedding exists
                    embedding_exists = await repos.embeddings.exists(context_id)

                    if embedding_exists:
                        await repos.embeddings.update(
                            context_id=context_id,
                            embedding=new_embedding,
                        )
                        logger.debug(f'Updated embedding for context {context_id}')
                    else:
                        await repos.embeddings.store(
                            context_id=context_id,
                            embedding=new_embedding,
                            model=settings.embedding_model,
                        )
                        logger.debug(f'Created embedding for context {context_id}')

                    updated_fields.append('embedding')
                except Exception as e:
                    logger.warning(f'Failed to update embedding for context {context_id}: {e}')
                    # Non-blocking: continue even if embedding update fails

            logger.info(f'Successfully updated context {context_id}, fields: {updated_fields}')

            return UpdateContextSuccessDict(
                success=True,
                context_id=context_id,
                updated_fields=updated_fields,
                message=f'Successfully updated {len(updated_fields)} field(s)',
            )

        except ToolError:
            raise  # Re-raise ToolError as-is
        except Exception as update_error:
            logger.error(f'Error during context update: {update_error}')
            raise ToolError(f'Update operation failed: {str(update_error)}') from update_error

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error updating context: {e}')
        raise ToolError(f'Unexpected error: {str(e)}') from e


# MCP Resources for read-only access


async def list_threads(ctx: Context | None = None) -> ThreadListDict:
    """List all threads with entry statistics. Use for thread discovery and overview.

    Returns: {
      threads: [{thread_id, entry_count, source_types, multimodal_count, first_entry, last_entry, last_id}],
      total_threads: int
    }

    Fields explained:
    - entry_count: Total context entries in thread
    - source_types: Number of distinct sources (1=user only or agent only, 2=both)
    - multimodal_count: Entries containing images
    - first_entry/last_entry: ISO timestamps of earliest/latest entries
    - last_id: ID of most recent entry (useful for pagination)
    """
    try:
        if ctx:
            await ctx.info('Listing all threads')

        # Get repositories
        repos = await _ensure_repositories()

        # Use statistics repository to get thread list
        threads = await repos.statistics.get_thread_list()

        return {
            'threads': threads,
            'total_threads': len(threads),
        }
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error listing threads: {e}')
        raise ToolError(f'Failed to list threads: {str(e)}') from e


async def get_statistics(ctx: Context | None = None) -> dict[str, Any]:
    """Get database statistics for monitoring and debugging.

    Returns: {
        total_contexts: int,
        total_threads: int,
        total_images: int,
        total_tags: int,
        database_size_mb: float,
        connection_metrics: {...},
        semantic_search: {enabled, available, model, dimensions, embedding_count, coverage_percentage},
        fts: {enabled, available, language, backend, engine, indexed_entries, coverage_percentage}
    }

    Use for: capacity planning, debugging performance issues, verifying search status.
    """
    try:
        if ctx:
            await ctx.info('Getting database statistics')

        # Get repositories
        repos = await _ensure_repositories()

        # Use statistics repository to get database stats
        stats = await repos.statistics.get_database_statistics(DB_PATH)

        # Ensure backend for metrics
        manager = await _ensure_backend()

        # Add connection manager metrics for monitoring
        stats['connection_metrics'] = manager.get_metrics()

        # Add semantic search metrics if available
        if settings.enable_semantic_search:
            if _embedding_service is not None:
                embedding_stats = await repos.embeddings.get_statistics()
                stats['semantic_search'] = {
                    'enabled': True,
                    'available': True,
                    'backend': embedding_stats['backend'],
                    'model': settings.embedding_model,
                    'dimensions': settings.embedding_dim,
                    'embedding_count': embedding_stats['total_embeddings'],
                    'coverage_percentage': embedding_stats['coverage_percentage'],
                }
            else:
                stats['semantic_search'] = {
                    'enabled': True,
                    'available': False,
                    'message': 'Dependencies not met or initialization failed',
                }
        else:
            stats['semantic_search'] = {
                'enabled': False,
                'available': False,
            }

        # Add FTS metrics if available
        if settings.enable_fts:
            fts_available = await repos.fts.is_available()
            if fts_available:
                fts_stats = await repos.fts.get_statistics()
                stats['fts'] = {
                    'enabled': True,
                    'available': True,
                    'language': settings.fts_language,
                    'backend': fts_stats['backend'],
                    'engine': fts_stats['engine'],
                    'indexed_entries': fts_stats['indexed_entries'],
                    'coverage_percentage': fts_stats['coverage_percentage'],
                }
            else:
                stats['fts'] = {
                    'enabled': True,
                    'available': False,
                    'message': 'FTS migration not applied',
                }
        else:
            stats['fts'] = {
                'enabled': False,
                'available': False,
            }

        return stats
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error getting statistics: {e}')
        raise ToolError(f'Failed to get statistics: {str(e)}') from e


async def semantic_search_context(
    query: Annotated[str, Field(min_length=1, description='Natural language search query')],
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 5)')] = 5,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    thread_id: Annotated[str | None, Field(min_length=1, description='Optional filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Optional filter by source type')] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None, Field(description='Filter by content type (text or multimodal)'),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null',
        ),
    ] = None,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Find semantically similar context using vector embeddings with optional metadata filtering.

    Unlike keyword search (search_context), this finds entries with similar MEANING
    even without matching keywords. Use for: finding related concepts, similar discussions,
    thematic grouping.

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)

    Returns: {
      query: str,
      results: [{id, thread_id, source, text_content, metadata, distance, tags, ...}],
      count: int,
      model: str,
      stats: {...}  # Only when explain_query=True
    }

    The `distance` field is L2 Euclidean distance - LOWER values mean HIGHER similarity.
    Typical interpretation: <0.5 very similar, 0.5-1.0 related, >1.0 less related.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Check if semantic search is available
    if _embedding_service is None:
        raise ToolError(
            'Semantic search is not available. '
            'Ensure ENABLE_SEMANTIC_SEARCH=true and all dependencies are installed. '
            f'Run: uv sync --extra semantic-search && ollama pull {settings.embedding_model}',
        )

    try:
        if ctx:
            await ctx.info(f'Performing semantic search: "{query[:50]}..."')

        # Get repositories
        repos = await _ensure_repositories()

        # Generate embedding for query
        try:
            query_embedding = await _embedding_service.generate_embedding(query)
        except Exception as e:
            logger.error(f'Failed to generate query embedding: {e}')
            raise ToolError(f'Failed to generate embedding for query: {str(e)}') from e

        # Perform similarity search with optional filtering (date and metadata)
        # Import exception here to avoid circular imports at module level
        from app.repositories.embedding_repository import MetadataFilterValidationError

        try:
            # Unpack tuple; stats used when explain_query=True
            search_results, search_stats = await repos.embeddings.search(
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
        except MetadataFilterValidationError as e:
            # Return error response (unified with search_context behavior)
            return {
                'query': query,
                'results': [],
                'count': 0,
                'model': settings.embedding_model,
                'error': e.message,
                'validation_errors': e.validation_errors,
            }
        except Exception as e:
            logger.error(f'Semantic search failed: {e}')
            raise ToolError(f'Semantic search failed: {str(e)}') from e

        # Enrich results with tags and optionally images
        for result in search_results:
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(int(context_id))
                result['tags'] = tags_result
                # Fetch images if requested and applicable
                if include_images and result.get('content_type') == 'multimodal':
                    images_result = await repos.images.get_images_for_context(int(context_id), include_data=True)
                    result['images'] = images_result
            else:
                result['tags'] = []

        logger.info(f'Semantic search found {len(search_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'results': search_results,
            'count': len(search_results),
            'model': settings.embedding_model,
        }
        if explain_query:
            response['stats'] = search_stats
        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in semantic search: {e}')
        raise ToolError(f'Semantic search failed: {str(e)}') from e


async def fts_search_context(
    query: Annotated[str, Field(min_length=1, description='Full-text search query')],
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 5)')] = 5,
    mode: Annotated[
        Literal['match', 'prefix', 'phrase', 'boolean'],
        Field(
            description="Search mode: 'match' (default, natural language), "
            "'prefix' (wildcard with *), 'phrase' (exact phrase), "
            "'boolean' (AND/OR/NOT operators)",
        ),
    ] = 'match',
    thread_id: Annotated[str | None, Field(min_length=1, description='Optional filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Optional filter by source type')] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None, Field(description='Filter by content type (text or multimodal)'),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null',
        ),
    ] = None,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    highlight: Annotated[bool, Field(description='Include highlighted snippets in results')] = False,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Full-text search with linguistic analysis (stemming, ranking, boolean queries).

    Unlike keyword filtering (search_context) or semantic similarity (semantic_search_context),
    FTS provides:
    - Stemming: "running" matches "run", "runs", "runner"
    - Stop word handling: common words like "the", "is" are ignored
    - Boolean operators: AND, OR, NOT for precise queries
    - BM25/ts_rank relevance scoring

    Search modes:
    - match: Natural language query (default) - words are stemmed and matched
    - prefix: Wildcard search - "search*" matches "searching", "searched"
    - phrase: Exact phrase matching - "exact phrase" must appear as-is
    - boolean: Boolean operators - "python AND (async OR await) NOT blocking"

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)

    Returns: {
        query: str,
        mode: str,
        results: [{id, thread_id, source, text_content, metadata, score, highlighted, tags, ...}],
        count: int,
        language: str
    }

    The `score` field is relevance score - HIGHER values mean BETTER match.
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Check if FTS is enabled
    if not settings.enable_fts:
        raise ToolError(
            'Full-text search is not available. '
            'Set ENABLE_FTS=true to enable this feature.',
        )

    # Check if migration is in progress - return informative response for graceful degradation
    if _fts_migration_status.in_progress:
        if _fts_migration_status.started_at is not None and _fts_migration_status.estimated_seconds is not None:
            elapsed = (datetime.now(tz=UTC) - _fts_migration_status.started_at).total_seconds()
            remaining = max(0, _fts_migration_status.estimated_seconds - int(elapsed))
        else:
            remaining = 60  # Default estimate if no timing info available

        old_lang = _fts_migration_status.old_language or 'unknown'
        new_lang = _fts_migration_status.new_language or settings.fts_language

        return {
            'migration_in_progress': True,
            'message': f'FTS index is being rebuilt with language/tokenizer "{new_lang}". '
            'Search functionality will be available shortly.',
            'started_at': _fts_migration_status.started_at.isoformat() if _fts_migration_status.started_at else '',
            'estimated_remaining_seconds': remaining,
            'old_language': old_lang,
            'new_language': new_lang,
            'suggestion': f'Please retry in {remaining + 5} seconds.',
        }

    try:
        if ctx:
            await ctx.info(f'Performing FTS search: "{query[:50]}..." (mode={mode})')

        # Get repositories
        repos = await _ensure_repositories()

        # Check if FTS is properly initialized
        if not await repos.fts.is_available():
            raise ToolError(
                'FTS index not found. The database may need migration. '
                'Restart the server with ENABLE_FTS=true to apply migrations.',
            )

        # Import exception here to avoid circular imports
        from app.repositories.fts_repository import FtsValidationError

        try:
            search_results, stats = await repos.fts.search(
                query=query,
                mode=mode,
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
                highlight=highlight,
                language=settings.fts_language,
                explain_query=explain_query,
            )
        except FtsValidationError as e:
            # Return error response (unified with search_context behavior)
            error_response: dict[str, Any] = {
                'query': query,
                'mode': mode,
                'results': [],
                'count': 0,
                'language': settings.fts_language,
                'error': e.message,
                'validation_errors': e.validation_errors,
            }
            if explain_query:
                error_response['stats'] = {
                    'execution_time_ms': 0.0,
                    'filters_applied': 0,
                    'rows_returned': 0,
                }
            return error_response
        except Exception as e:
            logger.error(f'FTS search failed: {e}')
            raise ToolError(f'FTS search failed: {str(e)}') from e

        # Process results: parse metadata and enrich with tags
        for result in search_results:
            # Parse JSON metadata - database stores as JSON string
            metadata_raw = result.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    result['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    result['metadata'] = None

            # Get normalized tags
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(int(context_id))
                result['tags'] = tags_result
                # Fetch images if requested and applicable
                if include_images and result.get('content_type') == 'multimodal':
                    images_result = await repos.images.get_images_for_context(int(context_id), include_data=True)
                    result['images'] = images_result
            else:
                result['tags'] = []

        logger.info(f'FTS search found {len(search_results)} results for query: "{query[:50]}..."')

        response: dict[str, Any] = {
            'query': query,
            'mode': mode,
            'results': search_results,
            'count': len(search_results),
            'language': settings.fts_language,
        }
        if explain_query:
            response['stats'] = stats
        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in FTS search: {e}')
        raise ToolError(f'FTS search failed: {str(e)}') from e


async def hybrid_search_context(
    query: Annotated[str, Field(min_length=1, description='Natural language search query')],
    limit: Annotated[int, Field(ge=1, le=100, description='Maximum results to return (1-100, default: 5)')] = 5,
    offset: Annotated[int, Field(ge=0, description='Pagination offset (default: 0)')] = 0,
    search_modes: Annotated[
        list[Literal['fts', 'semantic']] | None,
        Field(
            description="Search modes to use: 'fts' (full-text), 'semantic' (vector similarity), "
            "or both ['fts', 'semantic'] (default). Modes are executed in parallel.",
        ),
    ] = None,
    fusion_method: Annotated[
        Literal['rrf'],
        Field(description="Fusion algorithm: 'rrf' (Reciprocal Rank Fusion, default)"),
    ] = 'rrf',
    rrf_k: Annotated[
        int | None,
        Field(
            ge=1,
            le=1000,
            description='RRF smoothing constant (default from HYBRID_RRF_K env var, typically 60). '
            'Higher values give more weight to lower-ranked documents.',
        ),
    ] = None,
    thread_id: Annotated[str | None, Field(min_length=1, description='Optional filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Optional filter by source type')] = None,
    content_type: Annotated[
        Literal['text', 'multimodal'] | None, Field(description='Filter by content type (text or multimodal)'),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags (OR logic)')] = None,
    start_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at >= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T10:00:00")',
        ),
    ] = None,
    end_date: Annotated[
        str | None,
        Field(
            description='Filter by created_at <= date (ISO 8601 format, e.g., "2025-11-29" or "2025-11-29T23:59:59")',
        ),
    ] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description='Advanced metadata filters: [{"key": "priority", "operator": "gt", "value": 5}]. '
            'Operators: eq, ne, gt, gte, lt, lte, in, not_in, exists, not_exists, contains, '
            'starts_with, ends_with, is_null, is_not_null',
        ),
    ] = None,
    include_images: Annotated[bool, Field(description='Include image data (only for multimodal entries)')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """Hybrid search combining FTS and semantic search with Reciprocal Rank Fusion (RRF).

    Executes both full-text search and semantic search in parallel, then fuses results
    using RRF algorithm. Documents appearing in both result sets score higher.

    RRF Formula: score(d) = sum(1 / (k + rank_i(d))) for each search method i

    Graceful degradation:
    - If only FTS is available, returns FTS results only
    - If only semantic search is available, returns semantic results only
    - If neither is available, raises ToolError

    Filtering options (all combinable):
    - thread_id/source: Basic entry filtering
    - content_type: Filter by text or multimodal entries
    - tags: OR logic (matches ANY of provided tags)
    - start_date/end_date: Date range filtering (ISO 8601)
    - metadata: Simple key=value equality matching
    - metadata_filters: Advanced operators (gt, lt, contains, exists, etc.)

    Returns: {
        query: str,
        results: [{id, thread_id, source, text_content, metadata, scores, tags, ...}],
        count: int,
        fusion_method: 'rrf',
        search_modes_used: ['fts', 'semantic'],  # Actual modes executed
        fts_count: int,
        semantic_count: int,
        stats: {...}  # Only when explain_query=True
    }

    The `scores` field contains: rrf (combined), fts_rank, semantic_rank,
    fts_score, semantic_distance.

    When explain_query=True, the `stats` field contains:
    - execution_time_ms: Total hybrid search time
    - fts_stats: {execution_time_ms, filters_applied, rows_returned} or None
    - semantic_stats: {execution_time_ms, embedding_generation_ms, filters_applied, rows_returned} or None
    - fusion_stats: {rrf_k, total_unique_documents, documents_in_both, documents_fts_only, documents_semantic_only}
    """
    # Validate date parameters
    start_date = validate_date_param(start_date, 'start_date')
    end_date = validate_date_param(end_date, 'end_date')
    validate_date_range(start_date, end_date)

    # Check if hybrid search is enabled
    if not settings.enable_hybrid_search:
        raise ToolError(
            'Hybrid search is not available. '
            'Set ENABLE_HYBRID_SEARCH=true to enable this feature. '
            'Also ensure ENABLE_FTS=true and/or ENABLE_SEMANTIC_SEARCH=true.',
        )

    # Use default search modes if not specified
    if search_modes is None:
        search_modes = ['fts', 'semantic']

    # Use settings default if rrf_k not specified
    effective_rrf_k = rrf_k if rrf_k is not None else settings.hybrid_rrf_k

    # Determine available search modes
    fts_available = settings.enable_fts
    semantic_available = settings.enable_semantic_search and _embedding_service is not None

    # Filter requested modes to available ones
    available_modes: list[str] = []
    if 'fts' in search_modes and fts_available:
        available_modes.append('fts')
    if 'semantic' in search_modes and semantic_available:
        available_modes.append('semantic')

    if not available_modes:
        unavailable_reasons = []
        if 'fts' in search_modes and not fts_available:
            unavailable_reasons.append('FTS requires ENABLE_FTS=true')
        if 'semantic' in search_modes and not semantic_available:
            unavailable_reasons.append('Semantic search requires ENABLE_SEMANTIC_SEARCH=true and Ollama')
        raise ToolError(
            f'No search modes available. Requested: {search_modes}. '
            f'Issues: {"; ".join(unavailable_reasons)}',
        )

    try:
        import time as time_module

        total_start_time = time_module.time()

        if ctx:
            await ctx.info(f'Performing hybrid search: "{query[:50]}..." (modes={available_modes})')

        # Import fusion module
        from app.fusion import count_unique_results
        from app.fusion import reciprocal_rank_fusion

        # Get repositories
        repos = await _ensure_repositories()

        # Over-fetch for better fusion quality
        # Must account for offset to ensure all entries are fetched for proper pagination
        over_fetch_limit = (limit + offset) * 2

        # Execute searches in parallel
        fts_results: list[dict[str, Any]] = []
        semantic_results: list[dict[str, Any]] = []
        fts_error: str | None = None
        semantic_error: str | None = None

        # Stats collection for explain_query
        fts_stats: dict[str, Any] | None = None
        semantic_stats: dict[str, Any] | None = None

        async def run_fts_search() -> None:
            nonlocal fts_results, fts_error, fts_stats
            try:
                # Check if FTS migration is in progress
                if _fts_migration_status.in_progress:
                    fts_error = 'FTS migration in progress'
                    return

                # Check if FTS is properly initialized
                if not await repos.fts.is_available():
                    fts_error = 'FTS index not available'
                    return

                from app.repositories.fts_repository import FtsValidationError

                try:
                    results, stats = await repos.fts.search(
                        query=query,
                        mode='match',
                        limit=over_fetch_limit,
                        offset=0,
                        thread_id=thread_id,
                        source=source,
                        content_type=content_type,
                        tags=tags,
                        start_date=start_date,
                        end_date=end_date,
                        metadata=metadata,
                        metadata_filters=metadata_filters,
                        highlight=False,
                        language=settings.fts_language,
                        explain_query=explain_query,
                    )
                    fts_results = results
                    if explain_query:
                        fts_stats = stats
                except FtsValidationError as e:
                    fts_error = str(e.message)
                except Exception as e:
                    fts_error = str(e)
            except Exception as e:
                fts_error = str(e)

        async def run_semantic_search() -> None:
            nonlocal semantic_results, semantic_error, semantic_stats
            try:
                if _embedding_service is None:
                    semantic_error = 'Embedding service not available'
                    return

                # Track embedding generation time for explain_query
                embedding_start_time = time_module.time() if explain_query else 0.0

                # Generate embedding for query
                try:
                    query_embedding = await _embedding_service.generate_embedding(query)
                except Exception as e:
                    semantic_error = f'Failed to generate embedding: {e}'
                    return

                embedding_generation_ms = (
                    (time_module.time() - embedding_start_time) * 1000 if explain_query else 0.0
                )

                from app.repositories.embedding_repository import MetadataFilterValidationError

                try:
                    # Unpack tuple; stats will be captured for explain_query
                    results, search_stats = await repos.embeddings.search(
                        query_embedding=query_embedding,
                        limit=over_fetch_limit,
                        offset=0,
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
                    semantic_results = results

                    # Build semantic stats with embedding generation time
                    if explain_query:
                        semantic_stats = {
                            'execution_time_ms': round(search_stats.get('execution_time_ms', 0.0), 2),
                            'embedding_generation_ms': round(embedding_generation_ms, 2),
                            'filters_applied': search_stats.get('filters_applied', 0),
                            'rows_returned': search_stats.get('rows_returned', 0),
                            'backend': search_stats.get('backend', 'unknown'),
                            'query_plan': search_stats.get('query_plan'),
                        }
                except MetadataFilterValidationError as e:
                    semantic_error = str(e.message)
                except Exception as e:
                    semantic_error = str(e)
            except Exception as e:
                semantic_error = str(e)

        # Run searches in parallel
        tasks = []
        if 'fts' in available_modes:
            tasks.append(run_fts_search())
        if 'semantic' in available_modes:
            tasks.append(run_semantic_search())

        await asyncio.gather(*tasks)

        # Check if both searches failed
        if fts_error and semantic_error:
            raise ToolError(
                f'All search modes failed. FTS: {fts_error}. Semantic: {semantic_error}',
            )

        # Determine which modes actually returned results
        modes_used: list[str] = []
        if fts_results:
            modes_used.append('fts')
        if semantic_results:
            modes_used.append('semantic')

        # Parse FTS metadata (returned as JSON strings from DB)
        for result in fts_results:
            metadata_raw = result.get('metadata')
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):
                try:
                    result['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    result['metadata'] = None

        # Fuse results using RRF
        # Over-fetch to handle offset, then apply offset after fusion
        fused_results = reciprocal_rank_fusion(
            fts_results=fts_results,
            semantic_results=semantic_results,
            k=effective_rrf_k,
            limit=limit + offset,  # Over-fetch to handle offset
        )

        # Apply offset after fusion
        fused_results = fused_results[offset:]

        # Enrich results with tags and optionally images (cast to Any for mutation compatibility)
        fused_results_any: list[dict[str, Any]] = cast(list[dict[str, Any]], fused_results)
        for result in fused_results_any:
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(int(context_id))
                result['tags'] = tags_result
                # Fetch images if requested and applicable
                if include_images and result.get('content_type') == 'multimodal':
                    images_result = await repos.images.get_images_for_context(int(context_id), include_data=True)
                    result['images'] = images_result
            else:
                result['tags'] = []

        logger.info(
            f'Hybrid search found {len(fused_results_any)} results for query: "{query[:50]}..." '
            f'(fts={len(fts_results)}, semantic={len(semantic_results)}, modes={modes_used})',
        )

        # Build response
        response: dict[str, Any] = {
            'query': query,
            'results': fused_results_any,
            'count': len(fused_results_any),
            'fusion_method': fusion_method,
            'search_modes_used': modes_used,
            'fts_count': len(fts_results),
            'semantic_count': len(semantic_results),
        }

        # Add stats if explain_query is enabled
        if explain_query:
            # Calculate fusion stats
            fts_only, semantic_only, overlap = count_unique_results(fts_results, semantic_results)
            fusion_stats: dict[str, Any] = {
                'rrf_k': effective_rrf_k,
                'total_unique_documents': fts_only + semantic_only + overlap,
                'documents_in_both': overlap,
                'documents_fts_only': fts_only,
                'documents_semantic_only': semantic_only,
            }

            # Calculate total execution time
            total_execution_time_ms = (time_module.time() - total_start_time) * 1000

            response['stats'] = {
                'execution_time_ms': round(total_execution_time_ms, 2),
                'fts_stats': fts_stats,
                'semantic_stats': semantic_stats,
                'fusion_stats': fusion_stats,
            }

        return response

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in hybrid search: {e}')
        raise ToolError(f'Hybrid search failed: {str(e)}') from e


# Bulk Operation MCP Tools


async def store_context_batch(
    entries: Annotated[
        list[dict[str, Any]],
        Field(
            description='List of context entries to store. Each entry must have: '
            'thread_id (str), source ("user" or "agent"), text (str). '
            'Optional: metadata (dict), tags (list[str]), images (list[dict]).',
            min_length=1,
            max_length=100,
        ),
    ],
    atomic: Annotated[
        bool,
        Field(
            description='If true, ALL entries must succeed or NONE are stored (transaction rollback). '
            'If false, partial success is allowed with per-item error reporting.',
        ),
    ] = True,
    ctx: Context | None = None,
) -> BulkStoreResponseDict:
    """Store multiple context entries in a single batch operation.

    Batch processing is significantly faster than individual store_context calls
    when storing many entries. Use for migrations, imports, or bulk operations.

    Atomicity modes:
    - atomic=True (default): All-or-nothing. If ANY entry fails, ALL are rolled back.
    - atomic=False: Best-effort. Each entry processed independently; partial success possible.

    Returns: {
        success: bool (true if ALL succeeded),
        total: int,
        succeeded: int,
        failed: int,
        results: [{index, success, context_id, error}, ...],
        message: str
    }

    Size limits:
    - Maximum 100 entries per batch
    - Image limits per entry: 10MB each, 100MB total
    - Standard tag normalization (lowercase)
    """
    try:
        if ctx:
            await ctx.info(f'Batch storing {len(entries)} context entries (atomic={atomic})')

        repos = await _ensure_repositories()

        # Phase 1: Validate all entries before processing
        validated_entries: list[dict[str, Any]] = []
        validation_errors: list[tuple[int, str]] = []

        for idx, entry in enumerate(entries):
            # Validate required fields
            if 'thread_id' not in entry or not entry.get('thread_id'):
                validation_errors.append((idx, 'Missing required field: thread_id'))
                continue
            if 'source' not in entry or entry.get('source') not in ('user', 'agent'):
                validation_errors.append((idx, 'Missing or invalid source (must be "user" or "agent")'))
                continue
            if 'text' not in entry or not entry.get('text'):
                validation_errors.append((idx, 'Missing required field: text'))
                continue

            # Clean input strings
            thread_id = str(entry['thread_id']).strip()
            text = str(entry['text']).strip()

            if not thread_id:
                validation_errors.append((idx, 'thread_id cannot be empty or whitespace'))
                continue
            if not text:
                validation_errors.append((idx, 'text cannot be empty or whitespace'))
                continue

            # Validate images if present
            images = entry.get('images', [])
            content_type = 'multimodal' if images else 'text'

            if images:
                total_size = 0.0
                for img_idx, img in enumerate(images):
                    if 'data' not in img:
                        validation_errors.append((idx, f'Image {img_idx} is missing required "data" field'))
                        break
                    try:
                        img_data = base64.b64decode(img['data'])
                        img_size_mb = len(img_data) / (1024 * 1024)
                        if img_size_mb > MAX_IMAGE_SIZE_MB:
                            validation_errors.append((idx, f'Image {img_idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit'))
                            break
                        total_size += img_size_mb
                        if total_size > MAX_TOTAL_SIZE_MB:
                            validation_errors.append((idx, f'Total image size exceeds {MAX_TOTAL_SIZE_MB}MB limit'))
                            break
                    except Exception:
                        validation_errors.append((idx, f'Image {img_idx} has invalid base64 encoding'))
                        break
                else:
                    # All images valid for this entry
                    pass

            # Check if entry already had validation errors from images
            if any(idx == err[0] for err in validation_errors):
                continue

            # Prepare validated entry
            metadata = entry.get('metadata')
            validated_entries.append({
                'index': idx,
                'thread_id': thread_id,
                'source': entry['source'],
                'text_content': text,
                'metadata': json.dumps(metadata, ensure_ascii=False) if metadata else None,
                'content_type': content_type,
                'tags': entry.get('tags', []),
                'images': images,
            })

        # Phase 2: In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at index {first_error[0]}: {first_error[1]}',
            )

        # Phase 3: Process validated entries through repository
        # Build results list including validation errors
        results: list[BulkStoreResultItemDict] = []

        # Add validation errors to results
        for idx, error in validation_errors:
            results.append(BulkStoreResultItemDict(
                index=idx,
                success=False,
                context_id=None,
                error=error,
            ))

        if validated_entries:
            # Prepare entries for repository batch operation
            repo_entries = [
                {
                    'thread_id': e['thread_id'],
                    'source': e['source'],
                    'text_content': e['text_content'],
                    'metadata': e['metadata'],
                    'content_type': e['content_type'],
                }
                for e in validated_entries
            ]

            # Execute batch store
            batch_results = await repos.context.store_contexts_batch(repo_entries)

            # Process repository results and store tags/images
            for repo_idx, ctx_id, repo_error in batch_results:
                original_entry = validated_entries[repo_idx]
                original_idx = original_entry['index']

                if ctx_id is not None and repo_error is None:
                    # Store tags if provided
                    if original_entry.get('tags'):
                        await repos.tags.store_tags(ctx_id, original_entry['tags'])

                    # Store images if provided
                    if original_entry.get('images'):
                        await repos.images.store_images(ctx_id, original_entry['images'])

                    # Generate embedding if semantic search enabled (non-blocking)
                    if _embedding_service is not None:
                        try:
                            embedding = await _embedding_service.generate_embedding(original_entry['text_content'])
                            await repos.embeddings.store(
                                context_id=ctx_id,
                                embedding=embedding,
                                model=settings.embedding_model,
                            )
                        except Exception as emb_err:
                            logger.warning(f'Failed to generate embedding for context {ctx_id}: {emb_err}')

                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=True,
                        context_id=ctx_id,
                        error=None,
                    ))
                else:
                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=False,
                        context_id=None,
                        error=repo_error or 'Unknown error',
                    ))

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(entries) - succeeded

        logger.info(f'Batch store completed: {succeeded}/{len(entries)} succeeded')

        return BulkStoreResponseDict(
            success=failed == 0,
            total=len(entries),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=f'Stored {succeeded}/{len(entries)} entries successfully',
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch store: {e}')
        raise ToolError(f'Batch store failed: {str(e)}') from e


async def update_context_batch(
    updates: Annotated[
        list[dict[str, Any]],
        Field(
            description='List of update operations. Each must have context_id (int). '
            'Optional: text (str), metadata (dict - full replace), '
            'metadata_patch (dict - RFC 7396 merge), tags (list[str]), images (list[dict]).',
            min_length=1,
            max_length=100,
        ),
    ],
    atomic: Annotated[
        bool,
        Field(
            description='If true, ALL updates succeed or NONE are applied. '
            'If false, partial success allowed.',
        ),
    ] = True,
    ctx: Context | None = None,
) -> BulkUpdateResponseDict:
    """Update multiple context entries in a single batch operation.

    Similar semantics to update_context but for multiple entries:
    - Each update is identified by context_id
    - Only provided fields are modified
    - metadata vs metadata_patch are mutually exclusive per entry
    - Tags and images use replacement semantics

    Atomicity modes:
    - atomic=True (default): All-or-nothing transaction
    - atomic=False: Best-effort with per-item error reporting

    Returns: {
        success: bool (true if ALL succeeded),
        total: int,
        succeeded: int,
        failed: int,
        results: [{index, context_id, success, updated_fields, error}, ...],
        message: str
    }
    """
    try:
        if ctx:
            await ctx.info(f'Batch updating {len(updates)} context entries (atomic={atomic})')

        repos = await _ensure_repositories()

        # Phase 1: Validate all updates before processing
        validated_updates: list[dict[str, Any]] = []
        validation_errors: list[tuple[int, int, str]] = []  # (index, context_id, error)

        for idx, update in enumerate(updates):
            # Validate required context_id
            if 'context_id' not in update:
                validation_errors.append((idx, 0, 'Missing required field: context_id'))
                continue

            context_id = update['context_id']
            if not isinstance(context_id, int) or context_id <= 0:
                validation_errors.append((idx, 0, 'context_id must be a positive integer'))
                continue

            # Validate mutual exclusivity of metadata and metadata_patch
            if update.get('metadata') is not None and update.get('metadata_patch') is not None:
                validation_errors.append((
                    idx,
                    context_id,
                    'Cannot use both metadata and metadata_patch. Use one or the other.',
                ))
                continue

            # Validate text if provided
            text = update.get('text')
            if text is not None:
                text = str(text).strip()
                if not text:
                    validation_errors.append((idx, context_id, 'text cannot be empty or whitespace'))
                    continue

            # Check that at least one field is provided for update
            has_update = any(
                update.get(field) is not None
                for field in ['text', 'metadata', 'metadata_patch', 'tags', 'images']
            )
            if not has_update:
                validation_errors.append((idx, context_id, 'At least one field must be provided for update'))
                continue

            # Validate images if provided
            images = update.get('images')
            if images is not None and len(images) > 0:
                total_size = 0.0
                for img_idx, img in enumerate(images):
                    if 'data' not in img:
                        validation_errors.append((idx, context_id, f'Image {img_idx} missing "data" field'))
                        break
                    try:
                        img_data = base64.b64decode(img['data'])
                        img_size_mb = len(img_data) / (1024 * 1024)
                        if img_size_mb > MAX_IMAGE_SIZE_MB:
                            validation_errors.append((
                                idx,
                                context_id,
                                f'Image {img_idx} exceeds {MAX_IMAGE_SIZE_MB}MB',
                            ))
                            break
                        total_size += img_size_mb
                        if total_size > MAX_TOTAL_SIZE_MB:
                            validation_errors.append((
                                idx,
                                context_id,
                                f'Total size exceeds {MAX_TOTAL_SIZE_MB}MB',
                            ))
                            break
                    except Exception:
                        validation_errors.append((idx, context_id, f'Image {img_idx} has invalid base64'))
                        break

            # Check if entry already had validation errors from images
            if any(idx == err[0] for err in validation_errors):
                continue

            # Prepare validated update
            validated_updates.append({
                'index': idx,
                'context_id': context_id,
                'text': text,
                'metadata': update.get('metadata'),
                'metadata_patch': update.get('metadata_patch'),
                'tags': update.get('tags'),
                'images': images,
            })

        # Phase 2: In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at context_id {first_error[1]}: {first_error[2]}',
            )

        # Phase 3: Process validated updates
        results: list[BulkUpdateResultItemDict] = []

        # Add validation errors to results
        for idx, context_id, error in validation_errors:
            results.append(BulkUpdateResultItemDict(
                index=idx,
                context_id=context_id,
                success=False,
                updated_fields=None,
                error=error,
            ))

        # Process each validated update
        for update in validated_updates:
            original_idx = update['index']
            context_id = update['context_id']
            updated_fields: list[str] = []

            try:
                # Check if entry exists
                exists = await repos.context.check_entry_exists(context_id)
                if not exists:
                    results.append(BulkUpdateResultItemDict(
                        index=original_idx,
                        context_id=context_id,
                        success=False,
                        updated_fields=None,
                        error=f'Context entry {context_id} not found',
                    ))
                    continue

                # Update text and/or metadata (full replacement)
                if update.get('text') is not None or update.get('metadata') is not None:
                    metadata_str = None
                    if update.get('metadata') is not None:
                        metadata_str = json.dumps(update['metadata'], ensure_ascii=False)

                    success, fields = await repos.context.update_context_entry(
                        context_id=context_id,
                        text_content=update.get('text'),
                        metadata=metadata_str,
                    )
                    if success:
                        updated_fields.extend(fields)

                # Apply metadata patch if provided
                if update.get('metadata_patch') is not None:
                    success, fields = await repos.context.patch_metadata(
                        context_id=context_id,
                        patch=update['metadata_patch'],
                    )
                    if success:
                        updated_fields.extend(fields)

                # Replace tags if provided
                if update.get('tags') is not None:
                    await repos.tags.replace_tags_for_context(context_id, update['tags'])
                    updated_fields.append('tags')

                # Replace images if provided
                if update.get('images') is not None:
                    images = update['images']
                    if len(images) == 0:
                        await repos.images.replace_images_for_context(context_id, [])
                        await repos.context.update_content_type(context_id, 'text')
                        updated_fields.extend(['images', 'content_type'])
                    else:
                        await repos.images.replace_images_for_context(context_id, images)
                        await repos.context.update_content_type(context_id, 'multimodal')
                        updated_fields.extend(['images', 'content_type'])

                # Regenerate embedding if text changed and semantic search available
                if update.get('text') is not None and _embedding_service is not None:
                    try:
                        new_embedding = await _embedding_service.generate_embedding(update['text'])
                        embedding_exists = await repos.embeddings.exists(context_id)
                        if embedding_exists:
                            await repos.embeddings.update(context_id=context_id, embedding=new_embedding)
                        else:
                            await repos.embeddings.store(
                                context_id=context_id,
                                embedding=new_embedding,
                                model=settings.embedding_model,
                            )
                        updated_fields.append('embedding')
                    except Exception as emb_err:
                        logger.warning(f'Failed to update embedding for context {context_id}: {emb_err}')

                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=True,
                    updated_fields=updated_fields,
                    error=None,
                ))

            except Exception as e:
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=str(e),
                ))

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(updates) - succeeded

        logger.info(f'Batch update completed: {succeeded}/{len(updates)} succeeded')

        return BulkUpdateResponseDict(
            success=failed == 0,
            total=len(updates),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=f'Updated {succeeded}/{len(updates)} entries successfully',
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch update: {e}')
        raise ToolError(f'Batch update failed: {str(e)}') from e


async def delete_context_batch(
    context_ids: Annotated[
        list[int] | None,
        Field(description='Specific context IDs to delete'),
    ] = None,
    thread_ids: Annotated[
        list[str] | None,
        Field(description='Delete ALL entries in these threads'),
    ] = None,
    source: Annotated[
        Literal['user', 'agent'] | None,
        Field(description='Delete only entries from this source (combine with other criteria)'),
    ] = None,
    older_than_days: Annotated[
        int | None,
        Field(description='Delete entries older than N days', gt=0),
    ] = None,
    ctx: Context | None = None,
) -> BulkDeleteResponseDict:
    """Delete multiple context entries by various criteria. IRREVERSIBLE.

    Criteria can be combined for targeted deletion:
    - context_ids: Delete specific entries by ID
    - thread_ids: Delete all entries in specified threads
    - source: Filter by source ('user' or 'agent')
    - older_than_days: Delete entries created more than N days ago

    At least one criterion must be provided.
    Cascading delete removes associated tags, images, and embeddings.

    WARNING: This operation cannot be undone. Verify criteria before deletion.

    Returns: {
        success: bool,
        deleted_count: int,
        criteria_used: [str, ...],
        message: str
    }
    """
    try:
        # Validate at least one criterion is provided
        if not any([context_ids, thread_ids, source, older_than_days]):
            raise ToolError(
                'At least one deletion criterion must be provided: '
                'context_ids, thread_ids, source, or older_than_days',
            )

        # Validate source if provided alone
        if source and not any([context_ids, thread_ids, older_than_days]):
            raise ToolError(
                'source filter must be combined with another criterion '
                '(context_ids, thread_ids, or older_than_days)',
            )

        if ctx:
            criteria_summary = []
            if context_ids:
                criteria_summary.append(f'{len(context_ids)} IDs')
            if thread_ids:
                criteria_summary.append(f'{len(thread_ids)} threads')
            if source:
                criteria_summary.append(f'source={source}')
            if older_than_days:
                criteria_summary.append(f'older_than={older_than_days}d')
            await ctx.info(f'Batch delete with criteria: {", ".join(criteria_summary)}')

        repos = await _ensure_repositories()

        # Delete embeddings first if context_ids are specified
        if settings.enable_semantic_search and context_ids:
            for cid in context_ids:
                try:
                    await repos.embeddings.delete(cid)
                except Exception as e:
                    logger.warning(f'Failed to delete embedding for context {cid}: {e}')

        # Execute batch delete through repository
        deleted_count, criteria_used = await repos.context.delete_contexts_batch(
            context_ids=context_ids,
            thread_ids=thread_ids,
            source=source,
            older_than_days=older_than_days,
        )

        logger.info(f'Batch delete completed: {deleted_count} entries removed')

        return BulkDeleteResponseDict(
            success=True,
            deleted_count=deleted_count,
            criteria_used=criteria_used,
            message=f'Successfully deleted {deleted_count} context entries',
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch delete: {e}')
        raise ToolError(f'Batch delete failed: {str(e)}') from e


# Main entry point
def main() -> None:
    """Main entry point for the MCP Context Server.

    Supports both stdio (default) and HTTP transport modes:
    - stdio: Default for local process spawning (uv run mcp-context-server)
    - http: For Docker/remote deployments (set MCP_TRANSPORT=http)

    Initialization and shutdown are handled by the @mcp.startup and @mcp.shutdown decorators.
    """
    try:
        transport = settings.transport.transport

        if transport == 'stdio':
            logger.info('Transport: STDIO')
            mcp.run()
        else:
            host = settings.transport.host
            port = settings.transport.port
            logger.info('Transport: HTTP')
            logger.info(f'Server URL: http://{host}:{port}/mcp')
            mcp.run(
                transport=cast(Literal['stdio', 'http', 'sse', 'streamable-http'], transport),
                host=host,
                port=port,
            )

    except KeyboardInterrupt:
        logger.info('Server shutdown requested')
    except Exception as e:
        logger.error(f'Server error: {e}')
        raise


if __name__ == '__main__':
    main()
