"""
Full-text search migration functions for mcp-context-server.

This module handles:
- FTS table creation (FTS5 for SQLite, tsvector for PostgreSQL)
- Tokenizer/language migration when settings change
- Migration status tracking for graceful degradation
"""

import logging
import sqlite3
from dataclasses import dataclass
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends import create_backend
from app.repositories.fts_repository import FtsRepository
from app.settings import get_settings

if TYPE_CHECKING:
    from app.repositories import RepositoryContainer

logger = logging.getLogger(__name__)
settings = get_settings()

# Database path for backward compatibility mode
DB_PATH = settings.storage.db_path


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


def get_fts_migration_status() -> FtsMigrationStatus:
    """Get the current FTS migration status.

    Returns:
        Current FtsMigrationStatus instance
    """
    return _fts_migration_status


def reset_fts_migration_status() -> None:
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


async def apply_fts_migration(backend: StorageBackend | None = None, repos: 'RepositoryContainer | None' = None) -> None:
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
    # Import here to avoid circular import (repos type hint uses string annotation above)

    # Skip if FTS is not enabled
    if not settings.fts.enabled:
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
        desired_tokenizer = await fts_repo.get_desired_tokenizer(settings.fts.language)

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
                reset_fts_migration_status()
        else:
            logger.debug(f'FTS tokenizer matches settings: "{current_tokenizer}"')

    else:  # postgresql
        # Check current language
        current_language = await fts_repo.get_current_language()
        desired_language = settings.fts.language

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
                reset_fts_migration_status()
        else:
            logger.debug(f'FTS language matches settings: "{current_language}"')


async def _apply_initial_fts_migration(manager: StorageBackend, backend_type: str) -> None:
    """Apply initial FTS migration for a fresh database.

    Args:
        manager: Storage backend
        backend_type: 'sqlite' or 'postgresql'

    Raises:
        RuntimeError: If FTS migration file is not found
    """
    if backend_type == 'sqlite':
        # Read SQLite migration template (consistent with PostgreSQL approach)
        migration_path = Path(__file__).parent / 'add_fts_sqlite.sql'
        if not migration_path.exists():
            error_msg = f'FTS migration file not found: {migration_path}'
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        migration_sql = migration_path.read_text(encoding='utf-8')

        # Determine tokenizer based on language setting
        # - 'porter unicode61' for English (enables stemming: "running" matches "run")
        # - 'unicode61' for other languages (multilingual support, no stemming)
        tokenizer = 'porter unicode61' if settings.fts.language.lower() == 'english' else 'unicode61'
        migration_sql = migration_sql.replace('{TOKENIZER}', tokenizer)

        def _apply_fts_sqlite(conn: sqlite3.Connection) -> None:
            conn.executescript(migration_sql)

        await manager.execute_write(_apply_fts_sqlite)
        logger.info(f'Applied FTS migration (SQLite FTS5) with tokenizer: {tokenizer}')

    else:  # postgresql
        # Read PostgreSQL migration template
        migration_path = Path(__file__).parent / 'add_fts_postgresql.sql'
        if not migration_path.exists():
            error_msg = f'FTS migration file not found: {migration_path}'
            logger.error(error_msg)
            raise RuntimeError(error_msg)

        migration_sql = migration_path.read_text(encoding='utf-8')
        migration_sql = migration_sql.replace('{FTS_LANGUAGE}', settings.fts.language)

        async def _apply_fts_pg(conn: asyncpg.Connection) -> None:
            # Acquire advisory lock for multi-pod DDL safety
            await conn.execute("SELECT pg_advisory_lock(hashtext('mcp_context_schema_init'))")
            try:
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
            finally:
                # Always release lock, even on error
                await conn.execute("SELECT pg_advisory_unlock(hashtext('mcp_context_schema_init'))")

        await manager.execute_write(cast(Any, _apply_fts_pg))
        logger.info(f'Applied FTS migration (PostgreSQL tsvector) with language: {settings.fts.language}')
