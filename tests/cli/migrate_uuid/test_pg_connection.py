"""Tests for the PostgreSQL connection helpers: the target emptiness probe and the connection budget.

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
from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from app.cli.migrate_uuid.pg_connection import pg_connect_kwargs
from app.cli.migrate_uuid.records import MigrationOptions
from app.cli.migrate_uuid.sqlite_to_postgresql import run_migration_mixed_sqlite_to_postgresql
from app.settings import get_settings
from tests.cli.migrate_uuid._sources import INTEGER_KEYED_ENTRIES_SCHEMA_SQL


class TestTargetPgHasDataSchemaQuoting:
    """The empty-target COUNT(*) probe quotes POSTGRESQL_SCHEMA via the shared helper.

    A schema name containing a double-quote is a valid quoted PostgreSQL identifier and is
    reachable operator config (POSTGRESQL_SCHEMA has no charset validation). The COUNT(*)
    target-emptiness probe must double the embedded quote exactly as CREATE SCHEMA and the
    search_path builder do -- all three route through quote_pg_identifier -- so the sites
    cannot drift and a pathological schema name does not abort the migration with a raw
    PostgresSyntaxError.
    """

    @pytest.mark.asyncio
    async def test_count_probe_doubles_embedded_quote_in_schema(self) -> None:
        """A schema with an embedded double-quote yields the doubled-quote COUNT(*),
        not the malformed single-quote-wrapped form."""
        from typing import Any
        from typing import cast

        from app.backends.postgresql_backend.session import quote_pg_identifier
        from app.cli.migrate_uuid.pg_connection import target_pg_has_data

        captured: list[str] = []

        class _RecordingConn:
            async def fetchval(self, query: str, *_args: object) -> object:
                captured.append(query)
                if 'information_schema.tables' in query:
                    return True  # the table exists in the probed schema
                return 5  # non-zero row count

        has_data = await target_pg_has_data(cast(Any, _RecordingConn()), schema='weird"schema')

        assert has_data is True
        count_query = next(q for q in captured if 'COUNT(*)' in q)
        assert quote_pg_identifier('weird"schema') == '"weird""schema"'
        assert '"weird""schema".context_entries' in count_query
        # The malformed single-quote-wrapped form must NOT appear.
        assert '"weird"schema".context_entries' not in count_query

    @pytest.mark.asyncio
    async def test_count_probe_with_none_schema_is_unqualified(self) -> None:
        """schema=None keeps the unqualified current_schema() COUNT(*), unchanged."""
        from typing import Any
        from typing import cast

        from app.cli.migrate_uuid.pg_connection import target_pg_has_data

        captured: list[str] = []

        class _RecordingConn:
            async def fetchval(self, query: str, *_args: object) -> object:
                captured.append(query)
                if 'information_schema.tables' in query:
                    return True
                return 0

        has_data = await target_pg_has_data(cast(Any, _RecordingConn()), schema=None)

        assert has_data is False
        count_query = next(q for q in captured if 'COUNT(*)' in q)
        assert count_query == 'SELECT COUNT(*) FROM context_entries'


def _seed_single_row_source(path: Path) -> None:
    """Create a minimal integer-keyed source database with one row."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(INTEGER_KEYED_ENTRIES_SCHEMA_SQL)
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
        kwargs = pg_connect_kwargs()

        assert kwargs['timeout'] == get_settings().storage.postgresql_connect_timeout_s

    @pytest.mark.parametrize('configured', ['110', '3.5'])
    def test_configured_value_is_applied(self, monkeypatch: pytest.MonkeyPatch, configured: str) -> None:
        """A longer or shorter configured budget reaches asyncpg instead of its own default."""
        monkeypatch.setenv('POSTGRESQL_CONNECT_TIMEOUT_S', configured)
        get_settings.cache_clear()

        kwargs = pg_connect_kwargs()

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
            mock.patch('app.cli.migrate_uuid.sqlite_to_postgresql.target_pg_has_data', _has_data),
            mock.patch('app.cli.migrate_uuid.sqlite_to_postgresql.pg_table_exists', _table_exists),
            mock.patch('app.cli.migrate_uuid.sqlite_to_postgresql.ensure_target_pg_fts', _ensure_fts),
        ):
            stats = await run_migration_mixed_sqlite_to_postgresql(options)

        assert stats.rows_migrated == 1
        assert recorded
        assert all(kwargs.get('timeout') == 7.0 for kwargs in recorded)
