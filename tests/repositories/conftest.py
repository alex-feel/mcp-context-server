"""Fixtures shared by the repository test modules."""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.repositories import RepositoryContainer
from app.repositories.statistics_repository import StatisticsRepository
from app.schemas import load_schema


@pytest_asyncio.fixture
async def backend_with_repos(temp_db_path: Path) -> 'AsyncGenerator[tuple[StorageBackend, RepositoryContainer], None]':
    """Create backend and repository container for transaction tests."""
    # Initialize database schema first
    import sqlite3 as stdlib_sqlite3

    from app.schemas import load_schema

    conn = stdlib_sqlite3.connect(str(temp_db_path))
    try:
        schema_sql = load_schema('sqlite')
        conn.executescript(schema_sql)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.commit()
    finally:
        conn.close()

    # Create backend and initialize
    backend = create_backend(backend_type='sqlite', db_path=str(temp_db_path))
    await backend.initialize()

    repos = RepositoryContainer(backend)

    yield backend, repos

    await backend.shutdown()


@pytest_asyncio.fixture
async def stats_test_db(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """Create a test database with the statistics repository."""
    db_path = tmp_path / 'stats_test.db'

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    # Initialize schema
    schema_sql = load_schema('sqlite')

    def _init_schema(conn: sqlite3.Connection) -> None:
        conn.executescript(schema_sql)

    await backend.execute_write(_init_schema)

    yield backend

    await backend.shutdown()


@pytest_asyncio.fixture
async def stats_repo(stats_test_db: StorageBackend) -> StatisticsRepository:
    """Create a statistics repository for testing."""
    return StatisticsRepository(stats_test_db)
