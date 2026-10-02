"""Fixtures for the context repository tests.

A SQLite backend with the base schema, a ContextRepository and a RepositoryContainer.
"""

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
