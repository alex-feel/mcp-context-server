"""Shared fixtures for the tool tests."""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest_asyncio

import app.startup
from app.backends import StorageBackend
from app.backends import create_backend
from app.repositories import RepositoryContainer


@pytest_asyncio.fixture
async def nav_backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend wired into app.startup so the tools resolve repositories."""
    from app.schemas import load_schema

    db_path = tmp_path / 'nav.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    repos = RepositoryContainer(backend)
    app.startup.set_backend(backend)
    app.startup.set_repositories(repos)
    try:
        yield backend
    finally:
        await backend.shutdown()
        app.startup.set_backend(None)
        app.startup.set_repositories(None)
