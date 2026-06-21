"""PostgreSQL coverage for the ``version`` column in-place upgrade migration.

The SQLite branch of :func:`app.migrations.version.apply_version_migration` is
covered by ``tests/migrations/test_version_migration.py``. This module closes
the matching gap for the PostgreSQL branch: the
``ALTER TABLE context_entries ADD COLUMN IF NOT EXISTS version BIGINT NOT NULL
DEFAULT 0`` issued under the schema-init advisory lock.

It mirrors the SQLite test's structure -- build a ``context_entries`` table that
OMITS the ``version`` column (the pre-migration shape of a database created
before the column joined the base schema), run the migration, and assert the
column is added as ``BIGINT NOT NULL DEFAULT 0``, backfills existing rows, and
is idempotent -- against a live pgvector container via the ``pg_test_url``
fixture (``@requires_docker_postgres``, skipped cleanly without Docker).

The isolated-database construction (admin asyncpg connection that drops/creates
a per-test database, then a ``PostgreSQLBackend`` built via
:func:`app.backends.create_backend` and ``await backend.initialize()``) follows
the pattern established by the neighboring PostgreSQL integration tests
(``test_cross_backend_tags_images.py``, ``test_migrations_non_default_schema.py``).
The table lives in the default ``public`` schema, so the migration's bare
``ALTER TABLE context_entries`` resolves there via the operator's implicit
``search_path`` without any search_path patching.
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
from app.migrations.version import apply_version_migration

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

_DB_NAME = 'mcp_version_migration_e2e'

# The PostgreSQL context_entries columns MINUS ``version`` -- the pre-migration
# shape of a database created before the column joined the base schema. Mirrors
# app/schemas/postgresql_schema.sql's context_entries columns (id, thread_id,
# source, content_type, text_content, metadata, summary, content_hash,
# created_at, updated_at) with the ``version BIGINT NOT NULL DEFAULT 0`` line
# removed. Indexes and the updated_at trigger are omitted: the migration does
# not require them.
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
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
'''


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database name replaced by ``new_db``."""
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{new_db}', parts.query, parts.fragment))


async def _make_isolated_db(pg_test_url: str) -> str:
    """Create or recreate the isolated PG database; return its connection URL."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {_DB_NAME}')
        await admin.execute(f'CREATE DATABASE {_DB_NAME}')
    finally:
        await admin.close()
    return _replace_db_name(pg_test_url, _DB_NAME)


async def _drop_isolated_db(pg_test_url: str) -> None:
    """Best-effort drop of the isolated database."""
    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {_DB_NAME}')
    finally:
        await admin.close()


async def _version_column_info(pg_url: str) -> asyncpg.Record | None:
    """Return the ``version`` column's information_schema row, or ``None``."""
    conn = await asyncpg.connect(pg_url)
    try:
        return await conn.fetchrow(
            '''
            SELECT data_type, is_nullable, column_default
            FROM information_schema.columns
            WHERE table_schema = 'public'
              AND table_name = 'context_entries'
              AND column_name = 'version'
            ''',
        )
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def pg_pre_migration_url(pg_test_url: str) -> AsyncIterator[str]:
    """Isolated PG database whose context_entries table OMITS ``version``.

    Creates a fresh per-test database, builds the pre-migration
    ``context_entries`` table (no ``version`` column) via an admin asyncpg
    connection, yields the connection URL, and drops the database at teardown.

    Yields:
        Connection string pointing at the isolated database where
        ``context_entries`` exists without the ``version`` column.
    """
    target_url = await _make_isolated_db(pg_test_url)
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute(_PRE_MIGRATION_CONTEXT_ENTRIES_DDL)
    finally:
        await setup.close()

    try:
        yield target_url
    finally:
        await _drop_isolated_db(pg_test_url)


class TestVersionMigrationPostgreSQL:
    """apply_version_migration adds the version column on an in-place upgrade (PostgreSQL)."""

    def test_migration_adds_version_column_with_default_zero(self, pg_pre_migration_url: str) -> None:
        """The PostgreSQL branch adds ``version`` as BIGINT NOT NULL DEFAULT 0,
        backfills a plain INSERT to 0, and is idempotent on re-run."""
        async def _scenario() -> None:
            # Precondition: the pre-migration table has NO version column.
            assert await _version_column_info(pg_pre_migration_url) is None

            backend = create_backend(
                backend_type='postgresql',
                connection_string=pg_pre_migration_url,
            )
            await backend.initialize()
            try:
                await apply_version_migration(backend)
            finally:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(backend.shutdown(), timeout=10.0)

        asyncio.run(_scenario())

        # The column is now present as BIGINT NOT NULL with a DEFAULT of 0.
        info = asyncio.run(_version_column_info(pg_pre_migration_url))
        assert info is not None
        assert info['data_type'] == 'bigint'
        assert info['is_nullable'] == 'NO'
        assert '0' in (info['column_default'] or '')

        # A plain INSERT that omits version reads back the DEFAULT-backfilled 0.
        async def _insert_and_read() -> int | None:
            conn = await asyncpg.connect(pg_pre_migration_url)
            try:
                await conn.execute(
                    "INSERT INTO context_entries (id, thread_id, source, content_type, text_content) "
                    "VALUES (gen_random_uuid(), 't1', 'agent', 'text', 'hello')",
                )
                return await conn.fetchval(
                    "SELECT version FROM context_entries WHERE thread_id = 't1'",
                )
            finally:
                await conn.close()

        assert asyncio.run(_insert_and_read()) == 0

        # Re-applying the migration is idempotent: no error, column unchanged.
        async def _reapply() -> None:
            backend = create_backend(
                backend_type='postgresql',
                connection_string=pg_pre_migration_url,
            )
            await backend.initialize()
            try:
                await apply_version_migration(backend)
            finally:
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(backend.shutdown(), timeout=10.0)

        asyncio.run(_reapply())

        info_after = asyncio.run(_version_column_info(pg_pre_migration_url))
        assert info_after is not None
        assert info_after['data_type'] == 'bigint'
        assert info_after['is_nullable'] == 'NO'
        # The previously-inserted row still reads 0 (value unchanged by re-run).
        assert asyncio.run(_version_column_info(pg_pre_migration_url)) is not None

        async def _read_existing_version() -> int | None:
            conn = await asyncpg.connect(pg_pre_migration_url)
            try:
                return await conn.fetchval(
                    "SELECT version FROM context_entries WHERE thread_id = 't1'",
                )
            finally:
                await conn.close()

        assert asyncio.run(_read_existing_version()) == 0
