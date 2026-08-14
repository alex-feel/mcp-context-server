"""Regression tests for the migration CLI's PostgreSQL connection budget.

``POSTGRESQL_CONNECT_TIMEOUT_S`` bounds connection ESTABLISHMENT (TCP connect plus the
PostgreSQL startup handshake). The server pool applies it to every connection it opens,
so a DSN whose handshake needs a longer budget than asyncpg's built-in 60-second default
-- a managed instance behind a session pooler or a VPN -- boots the server fine. Unless
the migration CLI passes the same value, its own connections silently keep the 60-second
default and abort a migration the operator explicitly configured against; the mirror case
is equally wrong, since a deliberately short budget would not fail fast either.
"""

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from app.cli.migrate import MigrationOptions
from app.cli.migrate import _pg_connect_kwargs
from app.cli.migrate import run_migration_mixed_sqlite_to_postgresql
from app.settings import get_settings

_INTEGER_KEYED_SCHEMA_SQL = '''
CREATE TABLE IF NOT EXISTS context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    summary TEXT,
    content_hash TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
'''


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings cache around every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _seed_single_row_source(path: Path) -> None:
    """Create a minimal integer-keyed source database with one row."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_INTEGER_KEYED_SCHEMA_SQL)
        conn.execute(
            'INSERT INTO context_entries '
            '(id, thread_id, source, content_type, text_content, metadata, created_at, updated_at) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (1, 't1', 'user', 'text', 'hello', json.dumps({'task_name': 'audit'}),
             '2025-01-01 12:00:00', '2025-01-01 12:00:00'),
        )
        conn.commit()
    finally:
        conn.close()


class _FakeTargetConn:
    """Minimal async stand-in for the asyncpg target connection."""

    async def execute(self, _query: str, *_args: object) -> str:
        """Accept any statement the copy loops and the session setup run."""
        return 'OK'

    async def close(self) -> None:
        """Accept the close the migration performs in its finally block."""


class TestPgConnectKwargs:
    """The shared CLI connect kwargs carry the configured establishment budget."""

    def test_default_matches_the_configured_default(self) -> None:
        """With no override the CLI applies the settings default explicitly."""
        kwargs = _pg_connect_kwargs()

        assert kwargs['timeout'] == get_settings().storage.postgresql_connect_timeout_s

    @pytest.mark.parametrize('configured', ['110', '3.5'])
    def test_configured_value_is_applied(self, monkeypatch: pytest.MonkeyPatch, configured: str) -> None:
        """A longer or shorter configured budget reaches asyncpg instead of its own default."""
        monkeypatch.setenv('POSTGRESQL_CONNECT_TIMEOUT_S', configured)
        get_settings.cache_clear()

        kwargs = _pg_connect_kwargs()

        assert kwargs['timeout'] == float(configured)
        # The statement-cache parameter shared with the server pool stays; the startup
        # packet stays empty so an external pooler cannot refuse the connection over it.
        assert 'server_settings' not in kwargs
        assert 'statement_cache_size' in kwargs


class TestMigrationConnectionsHonorTheBudget:
    """A real migration run opens its connections with the configured budget."""

    @pytest.mark.asyncio
    async def test_connect_receives_the_configured_timeout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every asyncpg.connect the migration opens carries POSTGRESQL_CONNECT_TIMEOUT_S."""
        monkeypatch.setenv('POSTGRESQL_CONNECT_TIMEOUT_S', '7')
        get_settings.cache_clear()

        source = tmp_path / 'source.db'
        _seed_single_row_source(source)
        recorded: list[dict[str, Any]] = []

        async def _fake_connect(*_args: object, **kwargs: Any) -> _FakeTargetConn:
            recorded.append(kwargs)
            return _FakeTargetConn()

        async def _has_data(*_args: object, **_kwargs: object) -> bool:
            return False

        async def _table_exists(*_args: object, **_kwargs: object) -> bool:
            return True

        async def _ensure_fts(*_args: object, **_kwargs: object) -> None:
            return None

        options = MigrationOptions(
            source_url=f'sqlite:///{source.as_posix()}',
            target_url='postgresql://user:pass@localhost:5432/db',
            dry_run=False,
            report_path=None,
        )
        with (
            mock.patch('asyncpg.connect', _fake_connect),
            mock.patch('app.cli.migrate._target_pg_has_data', _has_data),
            mock.patch('app.cli.migrate._pg_table_exists', _table_exists),
            mock.patch('app.cli.migrate.ensure_target_pg_fts', _ensure_fts),
        ):
            stats = await run_migration_mixed_sqlite_to_postgresql(options)

        assert stats.rows_migrated == 1
        assert recorded
        assert all(kwargs.get('timeout') == 7.0 for kwargs in recorded)
