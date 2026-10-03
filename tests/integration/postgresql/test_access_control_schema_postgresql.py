"""PostgreSQL coverage for the ``visibility`` CHECK constraint.

The column accepts exactly 'private' and 'public' on PostgreSQL, both in a fresh
database built from the base schema and in a database whose ``context_entries``
table predates the access-control columns and gains them through
:func:`app.migrations.access_control.apply_access_control_migration`. The SQLite
counterparts live in ``tests/migrations/test_access_control_migration.py``.

Each test runs against an isolated database on the pgvector container of the
``pg_test_url`` fixture (``@requires_docker_postgres``, skipped cleanly without
Docker). The tables live in the default ``public`` schema.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from app.backends import create_backend
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.migrations.access_control import apply_access_control_migration
from app.schemas import load_schema

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

_DB_NAME = 'mcp_access_control_schema_e2e'

_VISIBILITY_CASES = [('private', True), ('public', True), ('shared', False), ('everyone', False)]

# The PostgreSQL context_entries columns MINUS owner_id and visibility: the shape
# of a database created before the access-control columns joined the base schema.
_PRE_MIGRATION_CONTEXT_ENTRIES_DDL = '''
    CREATE TABLE context_entries (
        id UUID NOT NULL PRIMARY KEY,
        thread_id TEXT NOT NULL,
        source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
        content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
        text_content TEXT,
        metadata JSONB,
        summary TEXT,
        content_hash TEXT,
        version BIGINT NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
'''

_INSERT_WITH_VISIBILITY = (
    'INSERT INTO context_entries '
    '(id, thread_id, source, content_type, text_content, owner_id, visibility) '
    "VALUES (gen_random_uuid(), 't1', 'agent', 'text', 'x', 'local', $1)"
)


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database name replaced by ``new_db``."""
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{new_db}', parts.query, parts.fragment))


async def _make_isolated_db(pg_test_url: str) -> str:
    """Create or recreate the isolated database and return its connection URL."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {_DB_NAME}')
        await admin.execute(f'CREATE DATABASE {_DB_NAME}')
    finally:
        await admin.close()
    return _replace_db_name(pg_test_url, _DB_NAME)


async def _drop_isolated_db(pg_test_url: str) -> None:
    """Drop the isolated database, ignoring failures."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {_DB_NAME}')
    finally:
        await admin.close()


async def _assert_visibility_check(pg_url: str, visibility: str, accepted: bool) -> None:
    """Insert one row with ``visibility`` and assert the CHECK outcome."""
    conn = await asyncpg.connect(pg_url)
    try:
        if accepted:
            await conn.execute(_INSERT_WITH_VISIBILITY, visibility)
        else:
            with pytest.raises(asyncpg.exceptions.CheckViolationError):
                await conn.execute(_INSERT_WITH_VISIBILITY, visibility)
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def pg_base_schema_url(pg_test_url: str) -> AsyncIterator[str]:
    """Isolated database built from the PostgreSQL base schema.

    Yields:
        Connection string of the isolated database.
    """
    target_url = await _make_isolated_db(pg_test_url)
    schema_sql = load_schema('postgresql').replace('{SCHEMA}', quote_pg_identifier('public'))
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute(schema_sql)
    finally:
        await setup.close()

    try:
        yield target_url
    finally:
        await _drop_isolated_db(pg_test_url)


@pytest_asyncio.fixture
async def pg_migrated_url(pg_test_url: str) -> AsyncIterator[str]:
    """Isolated database whose pre-access-control table was upgraded by the migration.

    Yields:
        Connection string of the isolated database after the migration ran.
    """
    target_url = await _make_isolated_db(pg_test_url)
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute(_PRE_MIGRATION_CONTEXT_ENTRIES_DDL)
    finally:
        await setup.close()

    backend = create_backend(backend_type='postgresql', connection_string=target_url)
    await backend.initialize()
    try:
        await apply_access_control_migration(backend)
    finally:
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    try:
        yield target_url
    finally:
        await _drop_isolated_db(pg_test_url)


class TestVisibilityCheckPostgreSQL:
    """The visibility column accepts only 'private' and 'public' (PostgreSQL)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('visibility', 'accepted'), _VISIBILITY_CASES)
    async def test_base_schema_visibility_check(
        self, pg_base_schema_url: str, visibility: str, accepted: bool,
    ) -> None:
        """A fresh base-schema database stores 'private' and 'public' and rejects any other value."""
        await _assert_visibility_check(pg_base_schema_url, visibility, accepted)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('visibility', 'accepted'), _VISIBILITY_CASES)
    async def test_migrated_column_visibility_check(
        self, pg_migrated_url: str, visibility: str, accepted: bool,
    ) -> None:
        """The column added by the access-control migration carries the same CHECK."""
        await _assert_visibility_check(pg_migrated_url, visibility, accepted)
