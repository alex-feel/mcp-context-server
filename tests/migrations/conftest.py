"""Fixtures shared by the compression migration test modules.

Provides a ``get_settings`` cache reset and a SQLite backend whose database
carries the standard schema. Both are non-autouse: each compression module
requests the cache reset through its ``pytestmark``.
"""

import asyncio
import contextlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Generator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends import create_backend
from app.settings import get_settings


@pytest.fixture
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset ``get_settings`` cache before and after each test that uses it.

    Env-var monkeypatching for compression toggles would otherwise leak
    into unrelated tests because the settings singleton is process-global.

    Yields:
        Control to the test body; setup and teardown invalidate the cache.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest_asyncio.fixture
async def backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with the standard schema pre-applied."""
    db_path = tmp_path / 'test_compression.db'

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn.executescript(schema_sql)
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()

    yield backend

    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=5.0)
