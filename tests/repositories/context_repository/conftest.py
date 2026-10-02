"""Fixtures for the context repository tests.

context_test_db is a SQLite backend with the base schema, and context_repo and repos wrap it in a ContextRepository
and a RepositoryContainer. backend is a separate schema-initialized SQLite backend with a time-bounded shutdown; the
deduplication modules build their own repos on it.
"""

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio

from app.backends import create_backend
from app.backends.base import StorageBackend
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from app.schemas import load_schema


@pytest_asyncio.fixture
async def context_test_db(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a test database for context repository testing."""
    db_path = tmp_path / 'context_test.db'

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    yield backend

    await backend.shutdown()


@pytest_asyncio.fixture
async def context_repo(context_test_db: StorageBackend) -> ContextRepository:
    """Create a context repository for testing."""
    return ContextRepository(context_test_db)


@pytest_asyncio.fixture
async def repos(context_test_db: StorageBackend) -> RepositoryContainer:
    """Create full repository container."""
    return RepositoryContainer(context_test_db)


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a StorageBackend with a test database (SQLite only)."""
    db_path = tmp_path / 'test.db'

    # Initialize database with schema
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn.executescript(schema_sql)
    conn.close()

    # Create backend with db_path
    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    yield backend

    # Proper async cleanup with timeout protection to prevent hangs
    try:
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)
    except TimeoutError:
        import logging
        logging.getLogger(__name__).warning('Backend shutdown timed out after 5 seconds')
    except Exception as e:
        import logging
        logging.getLogger(__name__).error(f'Error during backend shutdown: {e}')
