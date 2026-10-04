"""PostgreSQL coverage for ``mcp-context-server-migrate --reassign-owner``.

Each test drives the dispatcher ``main()`` against an isolated database on the
pgvector container of the ``pg_test_url`` fixture (``@requires_docker_postgres``,
skipped cleanly without Docker). The database is built from the PostgreSQL base
schema, so the ``update_context_entries_updated_at`` trigger is present, and is
seeded with entries owned by two principals plus grant rows naming them. The
SQLite counterpart lives in ``tests/cli/test_migrate_reassign_owner.py``.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import AsyncIterator
from datetime import UTC
from datetime import datetime
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from app.backends.postgresql_backend.session import quote_pg_identifier
from app.cli.migrate import main as cli_main
from app.schemas import load_schema
from app.settings.auth import SAFE_PRINCIPAL_ID_PATTERN

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

_DB_NAME = 'mcp_migrate_reassign_owner_e2e'

_SEEDED_UPDATED_AT = datetime(2000, 1, 1, tzinfo=UTC)

# entry id -> (owner_id, version)
_SEED: dict[str, tuple[str, int]] = {
    'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa': ('local', 3),
    'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb': ('local', 0),
    'cccccccccccccccccccccccccccccccc': ('bob', 5),
}

# (context_entry_id, principal_type, principal_id, permission, granted_by)
_GRANTS: list[tuple[str, str, str, str, str]] = [
    ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'user', 'bob', 'read', 'local'),
    ('cccccccccccccccccccccccccccccccc', 'user', 'local', 'write', 'bob'),
]


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database name replaced by ``new_db``."""
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{new_db}', parts.query, parts.fragment))


async def _seed(pg_url: str) -> None:
    """Apply the base schema and insert the seeded entries and grants."""
    schema_sql = load_schema('postgresql').replace('{SCHEMA}', quote_pg_identifier('public'))
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(schema_sql)
        for entry_id, (owner_id, version) in _SEED.items():
            await conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, owner_id, version, updated_at) '
                "VALUES ($1, 'thread-a', 'user', 'text', 'doc', $2, $3, $4)",
                entry_id, owner_id, version, _SEEDED_UPDATED_AT,
            )
        await conn.executemany(
            'INSERT INTO context_entry_grants '
            '(context_entry_id, principal_type, principal_id, permission, granted_by) '
            'VALUES ($1, $2, $3, $4, $5)',
            _GRANTS,
        )
    finally:
        await conn.close()


async def _entries(pg_url: str) -> dict[str, tuple[str, int, datetime]]:
    """Return ``id -> (owner_id, version, updated_at)`` for every context entry."""
    conn = await asyncpg.connect(pg_url)
    try:
        rows = await conn.fetch('SELECT id, owner_id, version, updated_at FROM context_entries')
    finally:
        await conn.close()
    return {r['id'].hex: (str(r['owner_id']), int(r['version']), r['updated_at']) for r in rows}


async def _grants(pg_url: str) -> list[tuple[str, str, str, str, str]]:
    """Return every grant row as a sorted list of tuples."""
    conn = await asyncpg.connect(pg_url)
    try:
        rows = await conn.fetch(
            'SELECT context_entry_id, principal_type, principal_id, permission, granted_by '
            'FROM context_entry_grants',
        )
    finally:
        await conn.close()
    return sorted(
        (
            r['context_entry_id'].hex,
            str(r['principal_type']),
            str(r['principal_id']),
            str(r['permission']),
            str(r['granted_by']),
        )
        for r in rows
    )


def _unchanged_entries() -> dict[str, tuple[str, int, datetime]]:
    """Return the seeded entry state as :func:`_entries` reports it."""
    return {entry_id: (owner, version, _SEEDED_UPDATED_AT) for entry_id, (owner, version) in _SEED.items()}


@pytest_asyncio.fixture
async def pg_reassign_url(pg_test_url: str) -> AsyncIterator[str]:
    """Isolated database built from the base schema and seeded with two owners.

    Yields:
        Connection string of the isolated database.
    """
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {_DB_NAME}')
        await admin.execute(f'CREATE DATABASE {_DB_NAME}')
    finally:
        await admin.close()

    target_url = _replace_db_name(pg_test_url, _DB_NAME)
    await _seed(target_url)

    try:
        yield target_url
    finally:
        admin = await asyncpg.connect(pg_test_url)
        try:
            with contextlib.suppress(Exception):
                await admin.execute(f'DROP DATABASE IF EXISTS {_DB_NAME}')
        finally:
            await admin.close()


def test_dry_run_prints_count_and_changes_nothing(
    pg_reassign_url: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """--dry-run reports the matching row count and leaves every row as it was."""
    rc = cli_main(['--source-url', pg_reassign_url, '--reassign-owner', 'local', 'alice', '--dry-run'])

    assert rc == 0
    assert '[DRY-RUN] 2 context entries' in capsys.readouterr().err
    assert asyncio.run(_entries(pg_reassign_url)) == _unchanged_entries()
    assert asyncio.run(_grants(pg_reassign_url)) == sorted(_GRANTS)


def test_reassigns_exactly_the_from_rows(
    pg_reassign_url: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """Only FROM's rows change owner, to a subject outside the default-principal charset.

    Grants and versions stay as they were, and updated_at moves on the
    reassigned rows only.
    """
    subject = 'auth0|abc'
    assert SAFE_PRINCIPAL_ID_PATTERN.fullmatch(subject) is None

    rc = cli_main(['--source-url', pg_reassign_url, '--reassign-owner', 'local', subject])

    assert rc == 0
    assert f"Reassigned 2 context entries from 'local' to '{subject}'" in capsys.readouterr().err

    after = asyncio.run(_entries(pg_reassign_url))
    for entry_id, (owner, version) in _SEED.items():
        new_owner, new_version, new_updated_at = after[entry_id]
        assert new_version == version
        if owner == 'local':
            assert new_owner == subject
            assert new_updated_at > _SEEDED_UPDATED_AT
        else:
            assert new_owner == owner
            assert new_updated_at == _SEEDED_UPDATED_AT
    assert asyncio.run(_grants(pg_reassign_url)) == sorted(_GRANTS)


def test_no_matching_rows_is_a_success_no_op(
    pg_reassign_url: str, capsys: pytest.CaptureFixture[str],
) -> None:
    """A FROM that owns nothing exits 0 with a message and changes nothing."""
    rc = cli_main(['--source-url', pg_reassign_url, '--reassign-owner', 'mcp-client', 'local'])

    assert rc == 0
    assert "No context entries are owned by 'mcp-client'" in capsys.readouterr().err
    assert asyncio.run(_entries(pg_reassign_url)) == _unchanged_entries()
