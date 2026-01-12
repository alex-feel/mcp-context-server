"""
Server initialization and lifecycle management for mcp-context-server.

This package contains:
- Database initialization (init_database)
- Global state management (_backend, _repositories, _embedding_provider)
- Lazy initialization helpers (_ensure_backend, _ensure_repositories)
- Configuration constants (DB_PATH, MAX_IMAGE_SIZE_MB, MAX_TOTAL_SIZE_MB)

Global State Architecture:
    The _backend, _repositories, and _embedding_provider variables are module-level
    singletons that are initialized once during server lifespan and accessed by
    MCP tool functions. Direct mutation is allowed from server.py's lifespan().

Usage:
    # In server.py lifespan():
    from app.startup import (
        set_backend, set_repositories, set_embedding_provider,
        init_database, DB_PATH,
    )

    # Initialize
    set_backend(create_backend(backend_type=None, db_path=DB_PATH))
    await _backend.initialize()
    await init_database(backend=_backend)
    set_repositories(RepositoryContainer(_backend))

    # In MCP tools (app/tools/*.py):
    from app.startup import ensure_repositories
    repos = await ensure_repositories()
"""

import logging
import sqlite3
from pathlib import Path
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends import create_backend
from app.embeddings import EmbeddingProvider
from app.repositories import RepositoryContainer
from app.settings import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Configuration constants from settings
DB_PATH = settings.storage.db_path
MAX_IMAGE_SIZE_MB = settings.storage.max_image_size_mb
MAX_TOTAL_SIZE_MB = settings.storage.max_total_size_mb

# Global state - initialized during server lifespan
_backend: StorageBackend | None = None
_repositories: RepositoryContainer | None = None
_embedding_provider: EmbeddingProvider | None = None


def set_backend(backend: StorageBackend | None) -> None:
    """Set the global backend instance.

    Called from server.py lifespan() during startup/shutdown.
    """
    global _backend
    _backend = backend


def set_repositories(repos: RepositoryContainer | None) -> None:
    """Set the global repositories instance.

    Called from server.py lifespan() during startup/shutdown.
    """
    global _repositories
    _repositories = repos


def set_embedding_provider(provider: EmbeddingProvider | None) -> None:
    """Set the global embedding provider instance.

    Called from server.py lifespan() during startup/shutdown.
    """
    global _embedding_provider
    _embedding_provider = provider


def get_backend() -> StorageBackend | None:
    """Get the current backend instance (read-only access)."""
    return _backend


def get_repositories() -> RepositoryContainer | None:
    """Get the current repositories instance (read-only access)."""
    return _repositories


def get_embedding_provider() -> EmbeddingProvider | None:
    """Get the current embedding provider instance (read-only access)."""
    return _embedding_provider


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

        schema_path = Path(__file__).parent.parent / 'schemas' / schema_filename

        # Read schema from file
        if schema_path.exists():
            schema_sql_template = schema_path.read_text(encoding='utf-8')
        else:
            raise RuntimeError(f'Schema file not found: {schema_path}')

        # Template the schema SQL with configured schema name (PostgreSQL only)
        if backend_type == 'postgresql':
            schema_sql = schema_sql_template.replace('{SCHEMA}', settings.storage.postgresql_schema)
        else:
            schema_sql = schema_sql_template

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
                    statements: list[str] = []
                    current_stmt: list[str] = []
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
                        statements: list[str] = []
                        current_stmt: list[str] = []
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


async def ensure_backend() -> StorageBackend:
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


async def ensure_repositories() -> RepositoryContainer:
    """Ensure repositories are initialized.

    Returns:
        Initialized repository container.
    """
    global _repositories
    if _repositories is None:
        manager = await ensure_backend()
        _repositories = RepositoryContainer(manager)
    return _repositories


__all__ = [
    # Configuration constants
    'DB_PATH',
    'MAX_IMAGE_SIZE_MB',
    'MAX_TOTAL_SIZE_MB',
    # Global state (direct access - for lifespan read/write via setters)
    '_backend',
    '_repositories',
    '_embedding_provider',
    # Setters (for lifespan initialization)
    'set_backend',
    'set_repositories',
    'set_embedding_provider',
    # Getters (read-only access)
    'get_backend',
    'get_repositories',
    'get_embedding_provider',
    # Initialization functions
    'init_database',
    'ensure_backend',
    'ensure_repositories',
]
