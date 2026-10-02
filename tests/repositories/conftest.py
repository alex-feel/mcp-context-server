"""Fixtures shared by the repository test modules."""

from pathlib import Path
from typing import TYPE_CHECKING

import pytest_asyncio

from app.backends import create_backend
from app.repositories import RepositoryContainer

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from app.backends import StorageBackend


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
