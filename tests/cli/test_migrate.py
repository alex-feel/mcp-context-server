"""Tests for the integer-keyed to UUIDv7 migration CLI.

The fixture-defined source schema in this module is the integer-keyed
shape that the CLI accepts as input. Production code under ``app/``
does not include this shape; the fixture is the canonical reference
for the integer-keyed source layout used by the migration tests.
"""

import hashlib
import json
import re
import sqlite3
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pytest

from app.cli.migrate import MigrationOptions
from app.cli.migrate import MigrationStats
from app.cli.migrate import build_id_mapping
from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main
from app.cli.migrate import mask_credentials
from app.cli.migrate import parse_backend_url
from app.cli.migrate import rewrite_metadata_references
from app.cli.migrate import run_migration_sqlite_to_sqlite

HEX_32_RE = re.compile(r'^[0-9a-f]{32}$')


# ---------------------------------------------------------------------------
# Integer-keyed source schema (fixture-local)
# ---------------------------------------------------------------------------

INTEGER_KEYED_SCHEMA_SQL = '''
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

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS image_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
    image_data BLOB NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSON,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);
'''


def _seed_source_db(path: Path, rows: list[dict[str, object]]) -> None:
    """Create an integer-keyed source DB at ``path`` and seed it.

    Each row dict must include ``id``, ``thread_id``, ``source``,
    ``content_type``, ``text_content``, ``metadata`` (or None), and
    ``created_at`` (str or datetime). Optional keys: ``summary``,
    ``content_hash``, ``updated_at``.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(INTEGER_KEYED_SCHEMA_SQL)
        for row in rows:
            metadata_text = row.get('metadata')
            if isinstance(metadata_text, (dict, list)):
                metadata_text = json.dumps(metadata_text)
            created_at = row['created_at']
            if isinstance(created_at, datetime):
                created_at = created_at.isoformat()
            updated_at = row.get('updated_at', created_at)
            if isinstance(updated_at, datetime):
                updated_at = updated_at.isoformat()
            text_content = row.get('text_content')
            content_hash = row.get('content_hash')
            if content_hash is None and isinstance(text_content, str):
                content_hash = hashlib.sha256(text_content.encode('utf-8')).hexdigest()
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, metadata, '
                'summary, content_hash, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    row['id'],
                    row['thread_id'],
                    row['source'],
                    row['content_type'],
                    text_content,
                    metadata_text,
                    row.get('summary'),
                    content_hash,
                    created_at,
                    updated_at,
                ),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def legacy_source_db(tmp_path: Path) -> Path:
    """Create a temporary integer-keyed source DB with a small sample set."""
    path = tmp_path / 'source.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 'thread-alpha',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'Hello world',
            'metadata': {'agent_name': 'orchestrator'},
            'created_at': '2024-06-15 12:00:00',
        },
        {
            'id': 2,
            'thread_id': 'thread-alpha',
            'source': 'agent',
            'content_type': 'text',
            'text_content': 'See Report ID: 8944 and entries 9044, 14226',
            'metadata': {
                'agent_name': 'analyst',
                'references': {'context_ids': [1]},
            },
            'summary': 'A reply that mentions ID 8944 inline.',
            'created_at': '2025-01-01 09:30:00',
        },
        {
            'id': 3,
            'thread_id': 'thread-beta',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'Third entry',
            'metadata': {
                'references': {'context_ids': [1, 2]},
            },
            'created_at': '2026-02-20 17:45:00',
        },
    ]
    _seed_source_db(path, rows)
    return path


@pytest.fixture
def new_target_db_path(tmp_path: Path) -> Path:
    """Return a path to a non-existent target SQLite file."""
    return tmp_path / 'target.db'


def _build_options(source: Path, target: Path, *, dry_run: bool = False,
                   report: Path | None = None) -> MigrationOptions:
    """Build a :class:`MigrationOptions` for direct unit-test invocation."""
    return MigrationOptions(
        source_url=f'sqlite:///{source.as_posix()}',
        target_url=f'sqlite:///{target.as_posix()}',
        dry_run=dry_run,
        report_path=report,
    )


def _decode_unix_ts_ms(uuid_hex: str) -> int:
    """Extract the 48-bit unix_ts_ms field embedded in a UUIDv7 hex value."""
    return int(uuid_hex[:12], 16)


# ---------------------------------------------------------------------------
# UUID mapping behaviour
# ---------------------------------------------------------------------------


class TestUuidMappingDeterminism:
    """Determinism guarantees for the integer-to-UUIDv7 mapping table."""

    def test_mapping_repeatable_at_ms_granularity(self) -> None:
        """Two runs against the same rows produce UUIDs whose unix_ts_ms field matches row-by-row."""
        created_a = datetime(2025, 3, 4, 12, 0, 0, tzinfo=UTC)
        created_b = datetime(2025, 3, 4, 12, 0, 1, tzinfo=UTC)
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, created_a.isoformat()))
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, created_b.isoformat()))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        first = build_id_mapping(rows)
        second = build_id_mapping(rows)
        for row_id in first:
            assert _decode_unix_ts_ms(first[row_id]) == _decode_unix_ts_ms(second[row_id])


# ---------------------------------------------------------------------------
# NULL created_at tolerance and dry-run target isolation
# ---------------------------------------------------------------------------


class TestNullCreatedAtTolerance:
    """A schema-legal NULL created_at must not abort the migration."""

    def test_build_id_mapping_anchors_null_created_at(self) -> None:
        """build_id_mapping anchors a NULL created_at row instead of raising."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, '2025-06-24T12:00:00+00:00'))
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, None))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1, 2}
        assert all(HEX_32_RE.match(value) for value in mapping.values())

    def test_build_id_mapping_anchors_pre_epoch_created_at(self) -> None:
        """A pre-1970 created_at is anchored (no uuid7 OverflowError on negative epoch)."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, '1969-06-15T12:00:00+00:00'))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1}
        assert HEX_32_RE.match(mapping[1])

    def test_build_id_mapping_handles_numeric_epoch_created_at(self) -> None:
        """A numeric Unix-epoch created_at (some non-app sources store it) is coerced via
        epoch+timedelta, not aborted; a negative (pre-1970) epoch is anchored like NULL."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at)')  # typeless: keep the raw int
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, 1700000000))  # 2023 epoch seconds
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, -100))  # pre-1970 epoch -> anchored
        conn.execute('INSERT INTO r VALUES (?, ?)', (3, 1700000000.5))  # float epoch
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1, 2, 3}
        assert all(HEX_32_RE.match(value) for value in mapping.values())

    def test_migration_succeeds_with_null_created_at_row(
        self,
        tmp_path: Path,
        new_target_db_path: Path,
    ) -> None:
        """A source row with NULL created_at migrates without aborting the run."""
        source = tmp_path / 'null_created_at_source.db'
        _seed_source_db(source, [
            {
                'id': 1, 'thread_id': 't', 'source': 'user', 'content_type': 'text',
                'text_content': 'has timestamp', 'metadata': None,
                'created_at': '2025-06-24 12:00:00',
            },
            {
                'id': 2, 'thread_id': 't', 'source': 'agent', 'content_type': 'text',
                'text_content': 'null timestamp', 'metadata': None,
                'created_at': None,
            },
        ])
        stats = run_migration_sqlite_to_sqlite(_build_options(source, new_target_db_path))
        assert not stats.errors
        assert stats.rows_migrated == 2
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            total = conn.execute('SELECT COUNT(*) FROM context_entries').fetchone()[0]
            null_ts = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE created_at IS NULL',
            ).fetchone()[0]
        finally:
            conn.close()
        assert total == 2
        assert null_ts == 1

    def test_build_id_mapping_anchors_malformed_string_created_at(self) -> None:
        """A malformed non-ISO string created_at is anchored (not aborted) for id derivation.

        Mirrors the NULL / pre-epoch / numeric-epoch tolerance: an arbitrary non-app source
        database may store created_at as a non-ISO string, and one such row must not abort the
        whole migration via an uncaught _coerce_datetime ValueError.
        """
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at)')  # typeless: keep the raw text
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, '2024/01/01 12:00:00'))  # non-ISO slashes
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, '15-06-2024 10:00:00'))  # day-first
        conn.execute('INSERT INTO r VALUES (?, ?)', (3, 'not-a-date'))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1, 2, 3}
        assert all(HEX_32_RE.match(value) for value in mapping.values())

    def test_stored_datetime_or_none_tolerates_malformed_and_null(self) -> None:
        """The verbatim-bind helper yields None for NULL/malformed input and a datetime for valid."""
        from app.cli.migrate import _stored_datetime_or_none

        assert _stored_datetime_or_none(None) is None
        assert _stored_datetime_or_none('2024/01/01 12:00:00') is None  # non-ISO -> None, not raise
        assert _stored_datetime_or_none('not-a-date') is None
        valid = _stored_datetime_or_none('2025-06-24T12:00:00+00:00')
        assert valid is not None
        assert valid.year == 2025

    def test_migration_succeeds_with_malformed_string_created_at_row(
        self,
        tmp_path: Path,
        new_target_db_path: Path,
    ) -> None:
        """A source row with a malformed string created_at migrates without aborting the run."""
        source = tmp_path / 'malformed_created_at_source.db'
        _seed_source_db(source, [
            {
                'id': 1, 'thread_id': 't', 'source': 'user', 'content_type': 'text',
                'text_content': 'well-formed timestamp', 'metadata': None,
                'created_at': '2025-06-24 12:00:00',
            },
            {
                'id': 2, 'thread_id': 't', 'source': 'agent', 'content_type': 'text',
                'text_content': 'malformed timestamp', 'metadata': None,
                'created_at': '2024/01/01 12:00:00',  # non-ISO -> id anchored, value preserved
            },
        ])
        stats = run_migration_sqlite_to_sqlite(_build_options(source, new_target_db_path))
        assert not stats.errors
        assert stats.rows_migrated == 2
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            total = conn.execute('SELECT COUNT(*) FROM context_entries').fetchone()[0]
        finally:
            conn.close()
        assert total == 2


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

        from app.backends.postgresql_backend import quote_pg_identifier
        from app.cli.migrate import _target_pg_has_data

        captured: list[str] = []

        class _RecordingConn:
            async def fetchval(self, query: str, *_args: object) -> object:
                captured.append(query)
                if 'information_schema.tables' in query:
                    return True  # the table exists in the probed schema
                return 5  # non-zero row count

        has_data = await _target_pg_has_data(cast(Any, _RecordingConn()), schema='weird"schema')

        assert has_data is True
        count_query = next(q for q in captured if 'COUNT(*)' in q)
        assert quote_pg_identifier('weird"schema') == '"weird""schema"'
        assert '"weird""schema".context_entries' in count_query
        # The malformed single-quote-wrapped form (pre-fix) must NOT appear.
        assert '"weird"schema".context_entries' not in count_query

    @pytest.mark.asyncio
    async def test_count_probe_with_none_schema_is_unqualified(self) -> None:
        """schema=None keeps the unqualified current_schema() COUNT(*), unchanged."""
        from typing import Any
        from typing import cast

        from app.cli.migrate import _target_pg_has_data

        captured: list[str] = []

        class _RecordingConn:
            async def fetchval(self, query: str, *_args: object) -> object:
                captured.append(query)
                if 'information_schema.tables' in query:
                    return True
                return 0

        has_data = await _target_pg_has_data(cast(Any, _RecordingConn()), schema=None)

        assert has_data is False
        count_query = next(q for q in captured if 'COUNT(*)' in q)
        assert count_query == 'SELECT COUNT(*) FROM context_entries'


class TestDryRunNoTargetWrites:
    """A dry run must not create or write the SQLite target on disk."""

    def test_dry_run_does_not_create_target_file(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """--dry-run leaves no file at a non-existent SQLite target path."""
        assert not new_target_db_path.exists()
        stats = run_migration_sqlite_to_sqlite(
            _build_options(legacy_source_db, new_target_db_path, dry_run=True),
        )
        assert not stats.errors
        assert not new_target_db_path.exists()


class TestEnsureTargetPgFts:
    """FTS backstop for a PRE-EXISTING PostgreSQL target.

    ``initialize_target_postgresql`` provisions FTS only when the target had
    no ``context_entries`` table at all; a pre-existing target (bootstrapped
    by a server run with ``ENABLE_FTS=false``, or by any means other than
    this CLI) previously lost the source's full-text search silently -- the
    exact regression class the source-presence gate closed for freshly
    initialized targets. Unlike embeddings (not derivable -> abort), FTS is
    fully derivable from the copied rows, so the backstop provisions it.
    """

    @staticmethod
    def _column_probe_conn(has_fts_column: bool) -> object:
        """Build a fake target connection answering the column probe.

        Args:
            has_fts_column: Whether the probe reports text_search_vector.

        Returns:
            An object exposing the async ``fetchval`` surface the probe uses.
        """

        class _Conn:
            async def fetchval(self, query: str, *args: object) -> bool:
                del query, args
                return has_fts_column

        return _Conn()

    @pytest.mark.asyncio
    async def test_noop_when_source_has_no_fts(self) -> None:
        """No probe, no provisioning, no note when the source lacks FTS."""
        from typing import Any
        from typing import cast

        from app.cli.migrate import ensure_target_pg_fts

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, object()),
            target_schema='public', source_has_fts=False,
            dry_run=False, stats=stats,
        )
        assert stats.warnings == []
        assert stats.errors == []

    @pytest.mark.asyncio
    async def test_noop_when_target_already_has_fts(self) -> None:
        """A target already carrying text_search_vector is left untouched."""
        from typing import Any
        from typing import cast

        from app.cli.migrate import ensure_target_pg_fts

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, self._column_probe_conn(True)),
            target_schema='public', source_has_fts=True,
            dry_run=False, stats=stats,
        )
        assert stats.warnings == []

    @pytest.mark.asyncio
    async def test_dry_run_records_the_provisioning_plan(self) -> None:
        """A dry run records the plan instead of touching the target."""
        from typing import Any
        from typing import cast

        from app.cli.migrate import ensure_target_pg_fts

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, self._column_probe_conn(False)),
            target_schema='public', source_has_fts=True,
            dry_run=True, stats=stats,
        )
        assert any('would be provisioned' in w for w in stats.warnings)
        assert stats.errors == []

    @pytest.mark.asyncio
    async def test_real_run_applies_the_fts_migration_with_force(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A real run applies apply_fts_migration(force=True) against the target."""
        from typing import Any
        from typing import cast
        from unittest.mock import AsyncMock

        import app.backends as backends_module
        import app.migrations.fts as fts_module
        from app.cli.migrate import ensure_target_pg_fts

        fake_backend = AsyncMock()
        monkeypatch.setattr(
            backends_module, 'create_backend', lambda **_kw: fake_backend,
        )
        apply_spy = AsyncMock()
        monkeypatch.setattr(fts_module, 'apply_fts_migration', apply_spy)

        stats = MigrationStats()
        await ensure_target_pg_fts(
            'postgresql://ignored', cast(Any, self._column_probe_conn(False)),
            target_schema='public', source_has_fts=True,
            dry_run=False, stats=stats,
        )

        apply_spy.assert_awaited_once_with(fake_backend, force=True)
        fake_backend.initialize.assert_awaited_once()
        fake_backend.shutdown.assert_awaited_once()
        assert any('provisioned full-text search' in w for w in stats.warnings)


class TestSettingsValidationExitCode:
    """A settings ValidationError exits EX_CONFIG (78) on every CLI path.

    The server's guarded import (app/server.py) classifies an import-time
    settings ValidationError as a permanent misconfiguration and exits 78,
    but mcp-context-server-migrate never imports app.server: its in-place
    flows reach get_settings() through the lazily imported backend modules,
    and previously a value like RETRY_MAX_RETRIES=0 surfaced as a raw
    multi-frame pydantic traceback with the generic exit 1 -- the exact
    failure mode the server guard eliminated, alive on a second first-class
    entry point.
    """

    def test_in_place_dispatch_maps_validation_error_to_exit_78(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """--compress under an invalid env exits 78 with the pydantic detail."""
        from app.errors import ConfigurationError
        from app.settings import get_settings

        db = tmp_path / 'src.db'
        sqlite3.connect(str(db)).close()
        monkeypatch.setenv('RETRY_MAX_RETRIES', '0')
        get_settings.cache_clear()
        try:
            rc = cli_main(['--source-url', f'sqlite:///{db}', '--compress', '--dry-run'])
        finally:
            monkeypatch.delenv('RETRY_MAX_RETRIES', raising=False)
            get_settings.cache_clear()

        assert rc == ConfigurationError.EXIT_CODE
        err = capsys.readouterr().err
        assert 'Configuration invalid' in err
        assert 'RETRY_MAX_RETRIES' in err or 'retry_max_retries' in err


class TestUuidMappingFormat:
    """Format invariants for produced UUIDv7 hex strings."""

    def test_mapping_emits_32char_lowercase_hex(self) -> None:
        """Each value in the mapping matches ``^[0-9a-f]{32}$``."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        for i, year in enumerate((2024, 2025, 2026), start=1):
            conn.execute(
                'INSERT INTO r VALUES (?, ?)',
                (i, datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()),
            )
        rows = list(conn.execute('SELECT * FROM r'))
        mapping = build_id_mapping(rows)
        for value in mapping.values():
            assert HEX_32_RE.match(value), f'{value!r} is not 32-char lowercase hex'


class TestUuidYearAnchor:
    """Decoded year-from-UUIDv7 matches the source created_at year."""

    def test_uuid_year_matches_source_created_at_year(self) -> None:
        """Decoded UUIDv7 timestamp lands in the source's year, NOT year ~50,000."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        years = (2024, 2025, 2026)
        for i, year in enumerate(years, start=1):
            conn.execute(
                'INSERT INTO r VALUES (?, ?)',
                (i, datetime(year, 6, 15, 12, 0, 0, tzinfo=UTC).isoformat()),
            )
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        for row_id, expected_year in zip(sorted(mapping.keys()), years, strict=True):
            unix_ts_ms = _decode_unix_ts_ms(mapping[row_id])
            decoded = datetime.fromtimestamp(unix_ts_ms / 1000.0, tz=UTC)
            assert decoded.year == expected_year, (
                f'row {row_id}: decoded year {decoded.year}, expected {expected_year}'
            )


# ---------------------------------------------------------------------------
# metadata.references rewrite behaviour
# ---------------------------------------------------------------------------


class TestMetadataReferencesRewrite:
    """Rewrite behaviour for integer entries inside references.context_ids."""

    def test_top_level_references_remapped(self) -> None:
        """An integer list at the canonical path is rewritten to UUID strings."""
        stats = MigrationStats()
        mapping = {3: 'aaaa' * 8, 7: 'bbbb' * 8}
        metadata = {'references': {'context_ids': [3, 7]}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), mapping, stats, row_pk=99)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == ['aaaa' * 8, 'bbbb' * 8]
        assert stats.references_rewritten == 2
        assert stats.orphan_references == 0
        assert stats.malformed_references == 0

    def test_nested_references_remapped(self) -> None:
        """References buried inside a list-of-objects are rewritten recursively."""
        stats = MigrationStats()
        mapping = {5: 'cccc' * 8}
        metadata = {
            'history': [
                {'note': 'first'},
                {'references': {'context_ids': [5]}},
            ],
        }
        rewritten = rewrite_metadata_references(json.dumps(metadata), mapping, stats, row_pk=10)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['history'][1]['references']['context_ids'] == ['cccc' * 8]
        assert stats.references_rewritten == 1

    def test_string_references_preserved(self) -> None:
        """String entries inside context_ids are kept unchanged."""
        stats = MigrationStats()
        existing_uuid = 'd' * 32
        metadata = {'references': {'context_ids': [existing_uuid]}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), {}, stats, row_pk=1)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == [existing_uuid]
        assert stats.references_rewritten == 0
        assert stats.orphan_references == 0

    def test_orphan_reference_warning_emitted(self) -> None:
        """An integer with no mapping entry produces a warning and is kept as integer."""
        stats = MigrationStats()
        metadata = {'references': {'context_ids': [9999]}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), {}, stats, row_pk=42)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == [9999]
        assert stats.orphan_references == 1
        assert any('9999' in w for w in stats.warnings)

    def test_malformed_references_recorded_in_errors(self) -> None:
        """A non-list ``context_ids`` value is flagged and preserved unchanged."""
        stats = MigrationStats()
        metadata = {'references': {'context_ids': 'not-a-list'}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), {}, stats, row_pk=7)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == 'not-a-list'
        assert stats.malformed_references == 1
        assert any('not a list' in e for e in stats.errors)


# ---------------------------------------------------------------------------
# Ghost references (free-form text never rewritten)
# ---------------------------------------------------------------------------


class TestGhostReferences:
    """Free-form text columns are never rewritten."""

    def test_text_content_not_rewritten(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """text_content containing integer-style mentions is copied verbatim."""
        options = _build_options(legacy_source_db, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated == 3

        # Confirm via hash equality between source and target text_content.
        src_conn = sqlite3.connect(str(legacy_source_db))
        tgt_conn = sqlite3.connect(str(new_target_db_path))
        try:
            src_conn.row_factory = sqlite3.Row
            tgt_conn.row_factory = sqlite3.Row
            src_rows = list(src_conn.execute(
                'SELECT text_content FROM context_entries ORDER BY created_at ASC, id ASC',
            ))
            tgt_rows = list(tgt_conn.execute(
                'SELECT text_content FROM context_entries ORDER BY created_at ASC',
            ))
            assert len(src_rows) == len(tgt_rows)
            for src_row, tgt_row in zip(src_rows, tgt_rows, strict=True):
                src_hash = hashlib.sha256(
                    (src_row['text_content'] or '').encode('utf-8'),
                ).hexdigest()
                tgt_hash = hashlib.sha256(
                    (tgt_row['text_content'] or '').encode('utf-8'),
                ).hexdigest()
                assert src_hash == tgt_hash
        finally:
            src_conn.close()
            tgt_conn.close()

    def test_summary_not_rewritten(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """summary column is preserved byte-for-byte across migration."""
        options = _build_options(legacy_source_db, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)
        tgt_conn = sqlite3.connect(str(new_target_db_path))
        try:
            tgt_conn.row_factory = sqlite3.Row
            row = tgt_conn.execute(
                "SELECT summary FROM context_entries WHERE summary LIKE '%8944%'",
            ).fetchone()
            assert row is not None
            assert '8944' in row['summary']
        finally:
            tgt_conn.close()


# ---------------------------------------------------------------------------
# Embedding preservation
# ---------------------------------------------------------------------------


def _sqlite_vec_available() -> bool:
    """Return True iff the sqlite-vec extension can be loaded."""
    try:
        import sqlite_vec
    except ImportError:
        return False
    conn = sqlite3.connect(':memory:')
    try:
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        return True
    except (AttributeError, sqlite3.OperationalError, sqlite3.NotSupportedError):
        return False
    finally:
        conn.close()


requires_sqlite_vec = pytest.mark.skipif(
    not _sqlite_vec_available(),
    reason='sqlite-vec extension not loadable on this platform',
)


@pytest.fixture
def source_with_embeddings(tmp_path: Path) -> Path:
    """Create a source DB that includes embedding metadata, chunks, and vec0 rows."""
    if not _sqlite_vec_available():
        pytest.skip('sqlite-vec required for this fixture')
    import sqlite_vec

    path = tmp_path / 'source-with-vec.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 'thread-x',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'embedding row one',
            'metadata': None,
            'created_at': '2025-05-01 09:00:00',
        },
        {
            'id': 2,
            'thread_id': 'thread-x',
            'source': 'agent',
            'content_type': 'text',
            'text_content': 'embedding row two',
            'metadata': None,
            'created_at': '2025-05-01 09:01:00',
        },
    ]
    _seed_source_db(path, rows)

    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        conn.execute('CREATE VIRTUAL TABLE vec_context_embeddings USING vec0(embedding float[4])')
        conn.execute(
            'CREATE TABLE embedding_metadata ('
            'context_id INTEGER NOT NULL PRIMARY KEY, '
            'model_name TEXT NOT NULL, '
            'dimensions INTEGER NOT NULL, '
            'chunk_count INTEGER NOT NULL DEFAULT 1, '
            'created_at TEXT NOT NULL, '
            'updated_at TEXT NOT NULL)',
        )
        conn.execute(
            'CREATE TABLE embedding_chunks ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'context_id INTEGER NOT NULL, '
            'vec_rowid INTEGER NOT NULL, '
            'start_index INTEGER NOT NULL DEFAULT 0, '
            'end_index INTEGER NOT NULL DEFAULT 0, '
            'created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)',
        )
        emb_a = bytes([1, 2, 3, 4] * 4)  # 16 bytes = 4 float32 values
        emb_b = bytes([5, 6, 7, 8] * 4)
        conn.execute('INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (1, ?)', (emb_a,))
        conn.execute('INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (2, ?)', (emb_b,))
        conn.execute(
            'INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)',
            (1, 'model-a', 4, 1, '2025-05-01 09:00:00', '2025-05-01 09:00:00'),
        )
        conn.execute(
            'INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)',
            (2, 'model-a', 4, 1, '2025-05-01 09:01:00', '2025-05-01 09:01:00'),
        )
        conn.execute(
            'INSERT INTO embedding_chunks (id, context_id, vec_rowid, start_index, end_index) '
            'VALUES (?, ?, ?, ?, ?)',
            (1, 1, 1, 0, 17),
        )
        conn.execute(
            'INSERT INTO embedding_chunks (id, context_id, vec_rowid, start_index, end_index) '
            'VALUES (?, ?, ?, ?, ?)',
            (2, 2, 2, 0, 17),
        )
        conn.commit()
    finally:
        conn.close()
    return path


class TestEmbeddingPreservation:
    """Embeddings and chunk internals are preserved verbatim."""

    @requires_sqlite_vec
    def test_embeddings_copied_not_regenerated(
        self,
        source_with_embeddings: Path,
        new_target_db_path: Path,
    ) -> None:
        """vec_context_embeddings blob content matches between source and target."""
        import sqlite_vec

        options = _build_options(source_with_embeddings, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert not stats.errors, f'unexpected errors: {stats.errors}'

        src = sqlite3.connect(str(source_with_embeddings))
        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            for conn in (src, tgt):
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
                conn.row_factory = sqlite3.Row
            src_rows = list(src.execute(
                'SELECT rowid, embedding FROM vec_context_embeddings ORDER BY rowid',
            ))
            tgt_rows = list(tgt.execute(
                'SELECT rowid, embedding FROM vec_context_embeddings ORDER BY rowid',
            ))
            assert len(src_rows) == len(tgt_rows)
            for src_row, tgt_row in zip(src_rows, tgt_rows, strict=True):
                assert src_row['rowid'] == tgt_row['rowid']
                assert bytes(src_row['embedding']) == bytes(tgt_row['embedding'])
        finally:
            src.close()
            tgt.close()

    @requires_sqlite_vec
    def test_embedding_chunks_internals_preserved(
        self,
        source_with_embeddings: Path,
        new_target_db_path: Path,
    ) -> None:
        """``embedding_chunks.id`` and ``vec_rowid`` match between source and target."""
        options = _build_options(source_with_embeddings, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)

        src = sqlite3.connect(str(source_with_embeddings))
        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            src.row_factory = sqlite3.Row
            tgt.row_factory = sqlite3.Row
            src_rows = list(src.execute(
                'SELECT id, vec_rowid FROM embedding_chunks ORDER BY id',
            ))
            tgt_rows = list(tgt.execute(
                'SELECT id, vec_rowid FROM embedding_chunks ORDER BY id',
            ))
            assert len(src_rows) == len(tgt_rows)
            for src_row, tgt_row in zip(src_rows, tgt_rows, strict=True):
                assert src_row['id'] == tgt_row['id']
                assert src_row['vec_rowid'] == tgt_row['vec_rowid']
        finally:
            src.close()
            tgt.close()


@pytest.fixture
def source_pre_boundary_columns(tmp_path: Path) -> Path:
    """Create a source DB whose embedding_chunks predates start_index/end_index (pre-f36266c)."""
    if not _sqlite_vec_available():
        pytest.skip('sqlite-vec required for this fixture')
    import sqlite_vec

    path = tmp_path / 'source-pre-boundary.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 'thread-x',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'legacy embedding row',
            'metadata': None,
            'created_at': '2025-05-01 09:00:00',
        },
    ]
    _seed_source_db(path, rows)

    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    try:
        conn.execute('CREATE VIRTUAL TABLE vec_context_embeddings USING vec0(embedding float[4])')
        conn.execute(
            'CREATE TABLE embedding_metadata ('
            'context_id INTEGER NOT NULL PRIMARY KEY, '
            'model_name TEXT NOT NULL, '
            'dimensions INTEGER NOT NULL, '
            'chunk_count INTEGER NOT NULL DEFAULT 1, '
            'created_at TEXT NOT NULL, '
            'updated_at TEXT NOT NULL)',
        )
        # Pre-f36266c schema: NO start_index / end_index columns.
        conn.execute(
            'CREATE TABLE embedding_chunks ('
            'id INTEGER PRIMARY KEY AUTOINCREMENT, '
            'context_id INTEGER NOT NULL, '
            'vec_rowid INTEGER NOT NULL, '
            'created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)',
        )
        conn.execute('INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (1, ?)', (bytes([1, 2, 3, 4] * 4),))
        conn.execute(
            'INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)',
            (1, 'model-a', 4, 1, '2025-05-01 09:00:00', '2025-05-01 09:00:00'),
        )
        conn.execute(
            'INSERT INTO embedding_chunks (id, context_id, vec_rowid, created_at) VALUES (?, ?, ?, ?)',
            (1, 1, 1, '2025-05-01 09:00:00'),
        )
        conn.commit()
    finally:
        conn.close()
    return path


class TestPreBoundaryColumnSource:
    """A source embedding_chunks table lacking start_index/end_index is tolerated."""

    @requires_sqlite_vec
    def test_migration_succeeds_and_defaults_boundaries_to_zero(
        self,
        source_pre_boundary_columns: Path,
        new_target_db_path: Path,
    ) -> None:
        """The unguarded boundary-column SELECT would raise OperationalError; the guarded
        copier instead migrates the row and defaults start_index/end_index to 0 on a target
        that has the columns.
        """
        options = _build_options(source_pre_boundary_columns, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)

        assert not stats.errors, f'unexpected errors: {stats.errors}'
        assert stats.embedding_chunks_migrated == 1

        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            tgt.row_factory = sqlite3.Row
            row = tgt.execute(
                'SELECT id, vec_rowid, start_index, end_index FROM embedding_chunks',
            ).fetchone()
            assert row['id'] == 1
            assert row['vec_rowid'] == 1
            assert row['start_index'] == 0
            assert row['end_index'] == 0
        finally:
            tgt.close()


# ---------------------------------------------------------------------------
# Dry-run behaviour
# ---------------------------------------------------------------------------


class TestDryRun:
    """`--dry-run` performs all logic without writing to the target."""

    def test_dry_run_no_writes_to_target(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """Target ends up with zero rows after a dry-run."""
        options = _build_options(legacy_source_db, new_target_db_path, dry_run=True)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated > 0
        # File may exist (schema initialized) but context_entries must be empty.
        if new_target_db_path.exists():
            conn = sqlite3.connect(str(new_target_db_path))
            try:
                row = conn.execute('SELECT COUNT(*) AS c FROM context_entries').fetchone()
                assert row is not None
                assert row[0] == 0
            finally:
                conn.close()

    def test_dry_run_returns_zero_exit_code(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """``main`` returns 0 in dry-run mode when no errors occur."""
        exit_code = cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
                '--dry-run',
            ],
        )
        assert exit_code == 0


# ---------------------------------------------------------------------------
# Report output
# ---------------------------------------------------------------------------


class TestReportOutput:
    """JSON report path and stdout summary behaviour."""

    def test_report_written_to_path(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        tmp_path: Path,
    ) -> None:
        """``--report`` writes a JSON file matching MigrationStats.to_dict() shape."""
        report = tmp_path / 'report.json'
        exit_code = cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
                '--report',
                str(report),
            ],
        )
        assert exit_code == 0
        assert report.exists()
        loaded = json.loads(report.read_text(encoding='utf-8'))
        assert 'rows_migrated' in loaded
        assert 'references_rewritten' in loaded
        assert 'warnings' in loaded
        assert 'errors' in loaded

    def test_summary_printed_to_stdout(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without ``--report``, the summary appears on stdout."""
        cli_main(
            [
                '--source-url',
                f'sqlite:///{legacy_source_db.as_posix()}',
                '--target-url',
                f'sqlite:///{new_target_db_path.as_posix()}',
            ],
        )
        captured = capsys.readouterr()
        assert 'Migration summary' in captured.out
        assert 'rows migrated' in captured.out


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


class TestSchemaDetection:
    """Source schema detection guards against unnecessary work."""

    def test_already_migrated_source_warned_and_exit_zero(
        self,
        tmp_path: Path,
    ) -> None:
        """A TEXT-keyed source DB produces a warning and exits 0 without writes."""
        source = tmp_path / 'already.db'
        conn = sqlite3.connect(str(source))
        try:
            conn.execute(
                'CREATE TABLE context_entries ('
                'rowid_int INTEGER PRIMARY KEY AUTOINCREMENT, '
                'id TEXT NOT NULL UNIQUE, '
                'thread_id TEXT NOT NULL, '
                'source TEXT NOT NULL, '
                'content_type TEXT NOT NULL, '
                'text_content TEXT, '
                'metadata JSON, '
                'created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)',
            )
            conn.commit()
        finally:
            conn.close()
        target = tmp_path / 'target.db'
        options = _build_options(source, target)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated == 0
        assert any('nothing to migrate' in w for w in stats.warnings)
        assert not stats.errors

    def test_legacy_source_proceeds_normally(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """An integer-keyed source database is fully migrated."""
        options = _build_options(legacy_source_db, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated == 3
        assert not stats.errors

    def test_target_must_be_empty(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """If the target file contains rows, the CLI refuses with code 1."""
        # Pre-populate the target.
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            conn.executescript(INTEGER_KEYED_SCHEMA_SQL)
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, created_at, updated_at) '
                "VALUES (1, 't', 'user', 'text', 'existing', '2024-01-01', '2024-01-01')",
            )
            conn.commit()
        finally:
            conn.close()
        options = _build_options(legacy_source_db, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.errors
        assert any('already contains' in e for e in stats.errors)
        assert any('Recovery' in e for e in stats.errors)


# ---------------------------------------------------------------------------
# FTS rebuild
# ---------------------------------------------------------------------------


@pytest.fixture
def source_with_fts(tmp_path: Path) -> Path:
    """Create a source DB that includes the FTS5 virtual table."""
    path = tmp_path / 'source-fts.db'
    rows: list[dict[str, object]] = [
        {
            'id': 1,
            'thread_id': 't',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'the quick brown fox jumps over the lazy dog',
            'metadata': None,
            'created_at': '2025-08-10 10:00:00',
        },
        {
            'id': 2,
            'thread_id': 't',
            'source': 'agent',
            'content_type': 'text',
            'text_content': 'sphinx of black quartz judge my vow',
            'metadata': None,
            'created_at': '2025-08-10 10:05:00',
        },
    ]
    _seed_source_db(path, rows)
    conn = sqlite3.connect(str(path))
    try:
        # Build an FTS5 table that mirrors the legacy external-content shape so
        # that detect_optional_tables returns True for context_entries_fts.
        conn.execute(
            "CREATE VIRTUAL TABLE context_entries_fts USING fts5("
            "text_content, content='context_entries', content_rowid='id', "
            "tokenize='porter unicode61')",
        )
        conn.execute("INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')")
        conn.commit()
    finally:
        conn.close()
    return path


class TestFtsRebuild:
    """FTS5 index is rebuilt against the target's rowid_int surrogate."""

    def test_fts_rebuilt_after_migration_sqlite(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
    ) -> None:
        """An FTS5 MATCH against the target returns the migrated rows."""
        options = _build_options(source_with_fts, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.fts_rebuilt
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            conn.row_factory = sqlite3.Row
            rows = list(conn.execute(
                "SELECT rowid FROM context_entries_fts WHERE text_content MATCH 'fox'",
            ))
            assert len(rows) >= 1
        finally:
            conn.close()

    def test_fts_join_uses_rowid_int_surrogate(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
    ) -> None:
        """``context_entries_fts.rowid`` lines up with ``context_entries.rowid_int``."""
        options = _build_options(source_with_fts, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            conn.row_factory = sqlite3.Row
            rows = list(conn.execute(
                'SELECT ce.rowid_int AS rowid_int, ce.id AS public_id, fts.rowid AS fts_rowid '
                'FROM context_entries ce '
                'JOIN context_entries_fts fts ON fts.rowid = ce.rowid_int',
            ))
            assert len(rows) >= 1
            for row in rows:
                assert row['rowid_int'] == row['fts_rowid']
                assert HEX_32_RE.match(row['public_id'])
        finally:
            conn.close()

    @staticmethod
    def _read_target_fts_ddl(target_path: Path) -> str:
        """Return the stored CREATE statement for the target's FTS5 table."""
        conn = sqlite3.connect(str(target_path))
        try:
            row = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type = 'table' AND name = 'context_entries_fts'",
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0]
        return str(row[0])

    def test_fts_tokenizer_defaults_to_porter_for_english(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
    ) -> None:
        """With the default English language, the target FTS5 table uses the Porter stemmer."""
        options = _build_options(source_with_fts, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.fts_rebuilt
        ddl = self._read_target_fts_ddl(new_target_db_path)
        assert "tokenize='porter unicode61'" in ddl

    def test_fts_tokenizer_drops_porter_for_non_english_language(
        self,
        source_with_fts: Path,
        new_target_db_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-English FTS_LANGUAGE migrates to plain unicode61, not the English Porter stemmer."""
        from app.settings import get_settings

        monkeypatch.setenv('FTS_LANGUAGE', 'russian')
        get_settings.cache_clear()
        try:
            options = _build_options(source_with_fts, new_target_db_path)
            stats = run_migration_sqlite_to_sqlite(options)
            assert stats.fts_rebuilt
            ddl = self._read_target_fts_ddl(new_target_db_path)
            assert "tokenize='unicode61'" in ddl
            assert "tokenize='porter unicode61'" not in ddl
        finally:
            # Restore the cached singleton so the override does not leak into later tests.
            get_settings.cache_clear()


# ---------------------------------------------------------------------------
# vec0 preservation
# ---------------------------------------------------------------------------


class TestVec0Preservation:
    """The vec0 rowid identifier is preserved verbatim between source and target."""

    @requires_sqlite_vec
    def test_vec0_rowid_unchanged(
        self,
        source_with_embeddings: Path,
        new_target_db_path: Path,
    ) -> None:
        """Target vec_context_embeddings.rowid matches the source values."""
        import sqlite_vec

        options = _build_options(source_with_embeddings, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)
        src = sqlite3.connect(str(source_with_embeddings))
        tgt = sqlite3.connect(str(new_target_db_path))
        try:
            for conn in (src, tgt):
                conn.enable_load_extension(True)
                sqlite_vec.load(conn)
            src_ids = [row[0] for row in src.execute('SELECT rowid FROM vec_context_embeddings')]
            tgt_ids = [row[0] for row in tgt.execute('SELECT rowid FROM vec_context_embeddings')]
            assert sorted(src_ids) == sorted(tgt_ids)
        finally:
            src.close()
            tgt.close()


# ---------------------------------------------------------------------------
# CLI argument handling
# ---------------------------------------------------------------------------


class TestCliArgs:
    """Argparse plumbing for the migrate CLI."""

    def test_help_runs_and_exits_zero(self, capsys: pytest.CaptureFixture[str]) -> None:
        """``--help`` exits with code 0 and prints usage."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(['--help'])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert 'mcp-context-server-migrate' in captured.out
        assert '--source-url' in captured.out

    def test_missing_source_url_errors(self) -> None:
        """Calling without ``--source-url`` triggers a non-zero exit."""
        with pytest.raises(SystemExit) as exc_info:
            cli_main(['--target-url', 'sqlite:///dummy.db'])
        assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# Second-precision warning
# ---------------------------------------------------------------------------


class TestSecondPrecisionWarning:
    """The CLI logs an info line when source created_at has zero microsecond precision."""

    def test_second_precision_source_logs_info(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """A second-precision source triggers the precision info log."""
        options = _build_options(legacy_source_db, new_target_db_path)
        with caplog.at_level('INFO', logger='app.cli.migrate'):
            run_migration_sqlite_to_sqlite(options)
        assert any('precision' in record.message for record in caplog.records)


# ---------------------------------------------------------------------------
# Auxiliary helpers
# ---------------------------------------------------------------------------


class TestUrlHelpers:
    """URL parsing and credential masking helpers."""

    def test_parse_backend_url_sqlite_form(self) -> None:
        """``sqlite://`` URLs resolve to a SQLite address."""
        kind, addr = parse_backend_url('sqlite:///tmp/file.db')
        assert kind == 'sqlite'
        assert addr.endswith('tmp/file.db')

    def test_parse_backend_url_sqlalchemy_posix_absolute_form(self) -> None:
        """``sqlite:////abs/path`` collapses to a single-slash absolute path.

        The SQLAlchemy absolute form on POSIX uses four slashes; keeping the
        double-slash prefix would later be parsed as a file-URI authority by
        the SQLite backend and rejected.
        """
        kind, addr = parse_backend_url('sqlite:////tmp/file.db')
        assert kind == 'sqlite'
        assert addr == '/tmp/file.db'

    def test_parse_backend_url_windows_drive_scheme_form(self) -> None:
        """``sqlite:///C:/foo`` strips the leading slash before the drive letter."""
        kind, addr = parse_backend_url('sqlite:///C:/data/file.db')
        assert kind == 'sqlite'
        assert addr == 'C:/data/file.db'

    def test_parse_backend_url_postgresql_form(self) -> None:
        """``postgresql://`` URLs resolve to a PostgreSQL address."""
        kind, addr = parse_backend_url('postgresql://u:p@h/db')
        assert kind == 'postgresql'
        assert addr == 'postgresql://u:p@h/db'

    def test_parse_backend_url_windows_backslash_path(self) -> None:
        """A bare Windows absolute path with backslashes is treated as SQLite.

        ``urlparse`` would misread the ``C:`` drive letter as a URL scheme; the
        parser must recognize the drive-letter form and return it verbatim.
        """
        win_path = 'C:\\Users\\me\\AppData\\Local\\Temp\\v2_source.db'
        kind, addr = parse_backend_url(win_path)
        assert kind == 'sqlite'
        assert addr == win_path

    def test_parse_backend_url_windows_forwardslash_path(self) -> None:
        """A bare Windows absolute path with forward slashes is treated as SQLite."""
        kind, addr = parse_backend_url('D:/data/v3_target.db')
        assert kind == 'sqlite'
        assert addr == 'D:/data/v3_target.db'

    def test_parse_backend_url_posix_path(self) -> None:
        """A bare POSIX absolute path is still treated as SQLite (no regression)."""
        kind, addr = parse_backend_url('/home/me/db.sqlite')
        assert kind == 'sqlite'
        assert addr == '/home/me/db.sqlite'

    def test_mask_credentials_redacts_password(self) -> None:
        """The password segment of a PostgreSQL URL is masked."""
        masked = mask_credentials('postgresql://user:secret@host/db')
        assert 'secret' not in masked
        assert 'user' in masked
        assert '***' in masked

    def test_build_parser_has_required_flags(self) -> None:
        """The argparse parser declares all expected options."""
        parser = build_parser()
        actions = {action.dest for action in parser._actions}
        assert {'source_url', 'target_url', 'dry_run', 'report'}.issubset(actions)


@pytest.mark.asyncio
async def test_pg_pg_migration_closes_source_when_target_connect_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed target connect closes the source and propagates the original error.

    Mutation guard for the R2 None-guard fix (run_migration_postgresql): target_conn
    is opened INSIDE the try with a None-guarded finally, so a target connect failure
    does NOT leak the already-open source connection and does NOT raise
    UnboundLocalError / AttributeError from a None.close() in the finally.
    """
    import asyncpg as asyncpg_mod

    from app.cli.migrate import run_migration_postgresql

    closed = {'source': False}

    class _FakeConn:
        async def close(self) -> None:
            closed['source'] = True

    calls = {'n': 0}

    async def _fake_connect(_url: str, **_kwargs: object) -> _FakeConn:
        calls['n'] += 1
        if calls['n'] == 1:
            return _FakeConn()  # source connection succeeds
        raise OSError('target unreachable')  # target connect fails

    monkeypatch.setattr(asyncpg_mod, 'connect', _fake_connect)

    options = MigrationOptions(
        source_url='postgresql://u:p@localhost/src',
        target_url='postgresql://u:p@localhost/tgt',
    )
    with pytest.raises(OSError, match='target unreachable'):
        await run_migration_postgresql(options)
    # The source was closed by the finally; no UnboundLocalError / None.close().
    assert closed['source'] is True
