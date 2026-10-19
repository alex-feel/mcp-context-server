"""Fixtures shared by the migration test modules.

Provides a ``get_settings`` cache reset, a SQLite backend whose database carries the
standard schema, and a SQLite backend whose database predates the access-control
columns and table. All are non-autouse: each compression module requests the cache
reset through its ``pytestmark``, and the access-control and ingest index modules
request the pre-access-control backend by name.
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

# The context_entries columns without owner_id and visibility: the shape of a
# database that predates the access-control columns.
_PRE_MIGRATION_CONTEXT_ENTRIES_DDL = '''
    CREATE TABLE context_entries (
        rowid_int INTEGER PRIMARY KEY AUTOINCREMENT,
        id TEXT NOT NULL UNIQUE,
        thread_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
        content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
        text_content TEXT,
        metadata JSON,
        summary TEXT,
        content_hash TEXT,
        version INTEGER NOT NULL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''


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


@pytest_asyncio.fixture
async def backend_pre_migration(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend whose schema predates the access-control columns and table."""
    db_path = tmp_path / 'test_access_control_pre_migration.db'

    with sqlite3.connect(str(db_path)) as conn:
        conn.execute(_PRE_MIGRATION_CONTEXT_ENTRIES_DDL)
        conn.execute('CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id)')
        conn.execute(
            'CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source)',
        )

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    try:
        yield backend
    finally:
        await backend.shutdown()
