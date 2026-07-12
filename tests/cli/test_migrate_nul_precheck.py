"""Regression tests for the SQLite->PostgreSQL NUL / unstorable-string pre-check.

A NUL (U+0000) or unpaired UTF-16 surrogate is legal in a SQLite TEXT value but
fatal on PostgreSQL: asyncpg rejects a NUL text bind (``CharacterNotInRepertoireError``,
SQLSTATE 22021) and PostgreSQL's jsonb parser rejects the ``\\u0000`` escape the
store path emits for a metadata NUL (SQLSTATE 22P05). Before the pre-check, such a
value aborted the whole cross-backend migration mid-transaction with only a raw
driver error and no row identification, and ``--dry-run`` could not surface it
first. These tests prove each offending row is now identified and skipped, the run
does not crash, and ``--dry-run`` surfaces the same rows without inserting.

The full-run tests drive the real ``run_migration_mixed_sqlite_to_postgresql``
against a fake asyncpg target connection (the PostgreSQL probe helpers are patched
out), so no live PostgreSQL is required; the per-row copy loops -- the code under
test -- run unchanged.
"""

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate import MigrationOptions
from app.cli.migrate import MigrationStats
from app.cli.migrate import _pg_unstorable_column_reason
from app.cli.migrate import run_migration_mixed_sqlite_to_postgresql
from app.settings import get_settings

# Integer-keyed source schema (the shape the CLI accepts as input). Production
# code under app/ no longer uses this layout; each migration test file defines
# its own bootstrap copy so it stays self-contained.
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


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings cache around every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _seed_source(
    path: Path,
    entries: list[dict[str, object]],
    tags: list[tuple[int, str]] | None = None,
    images: list[tuple[int, str, str | None]] | None = None,
) -> None:
    """Create an integer-keyed source DB at ``path`` and seed it.

    Each entry dict needs ``id``, ``thread_id``, ``source``, ``content_type``,
    ``text_content``, ``metadata`` (a dict serialized to JSON, or None), and
    ``created_at``. Tags are ``(context_entry_id, tag)`` pairs; images are
    ``(context_entry_id, mime_type, image_metadata_json)`` triples.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_INTEGER_KEYED_SCHEMA_SQL)
        for entry in entries:
            metadata = entry.get('metadata')
            if isinstance(metadata, (dict, list)):
                metadata = json.dumps(metadata)
            created_at = entry['created_at']
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, metadata, '
                'summary, content_hash, created_at, updated_at) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    entry['id'],
                    entry['thread_id'],
                    entry['source'],
                    entry['content_type'],
                    entry.get('text_content'),
                    metadata,
                    entry.get('summary'),
                    entry.get('content_hash'),
                    created_at,
                    created_at,
                ),
            )
        for context_entry_id, tag in tags or []:
            conn.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (context_entry_id, tag),
            )
        for context_entry_id, mime_type, image_metadata in images or []:
            conn.execute(
                'INSERT INTO image_attachments '
                '(context_entry_id, image_data, mime_type, image_metadata, position) '
                'VALUES (?, ?, ?, ?, ?)',
                (context_entry_id, b'\x89PNG\r\n', mime_type, image_metadata, 0),
            )
        conn.commit()
    finally:
        conn.close()


class _FakeTargetConn:
    """Minimal async stand-in for the asyncpg target connection.

    Records every ``execute`` call so a test can assert which rows were inserted
    (or, under ``--dry-run``, that none were). The PostgreSQL probe helpers are
    patched out, so this only needs the ``execute`` / ``close`` surface the copy
    loops touch.
    """

    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []
        self.closed = False

    async def execute(self, query: str, *args: object) -> str:
        """Record the SQL and its bound parameters, returning a status string."""
        self.executed.append((query, args))
        return 'OK'

    async def close(self) -> None:
        """Mark the connection closed."""
        self.closed = True

    def inserts(self, table: str) -> list[tuple[object, ...]]:
        """Return the bound-parameter tuples of every ``INSERT INTO <table>`` call."""
        prefix = f'INSERT INTO {table}'
        return [args for query, args in self.executed if query.startswith(prefix)]

    def statements(self) -> list[str]:
        """Return every executed SQL statement in call order."""
        return [query for query, _ in self.executed]


async def _run_with_fake_target(
    source: Path,
    *,
    dry_run: bool,
) -> tuple[MigrationStats, _FakeTargetConn]:
    """Run the SQLite->PostgreSQL migration against a fake target connection.

    Patches ``asyncpg.connect`` and the PostgreSQL probe helpers so the real
    per-row copy loops run without a live PostgreSQL server.

    Returns:
        The populated migration stats and the fake connection that recorded every
        executed statement.
    """
    fake_conn = _FakeTargetConn()

    async def _fake_connect(*_args: object, **_kwargs: object) -> _FakeTargetConn:
        return fake_conn

    async def _has_data(*_args: object, **_kwargs: object) -> bool:
        return False

    async def _table_exists(*_args: object, **_kwargs: object) -> bool:
        # Report the target as already initialized so auto-init is skipped and the
        # FTS backstop (also patched out) is taken instead.
        return True

    async def _ensure_fts(*_args: object, **_kwargs: object) -> None:
        return None

    options = MigrationOptions(
        source_url=f'sqlite:///{source.as_posix()}',
        target_url='postgresql://user:pass@localhost:5432/db',
        dry_run=dry_run,
        report_path=None,
    )
    with (
        mock.patch('asyncpg.connect', _fake_connect),
        mock.patch('app.cli.migrate._target_pg_has_data', _has_data),
        mock.patch('app.cli.migrate._pg_table_exists', _table_exists),
        mock.patch('app.cli.migrate.ensure_target_pg_fts', _ensure_fts),
    ):
        stats = await run_migration_mixed_sqlite_to_postgresql(options)
    return stats, fake_conn


def _mixed_source_with_nul_rows(path: Path) -> None:
    """Seed a source exercising every NUL guard plus the skipped-parent cascade."""
    _seed_source(
        path,
        entries=[
            {
                'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                'text_content': 'clean entry', 'metadata': {'a': 'b'},
                'created_at': '2025-01-01 12:00:00',
            },
            {
                'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                'text_content': 'poisoned\x00text', 'metadata': None,
                'created_at': '2025-01-02 12:00:00',
            },
            {
                'id': 3, 'thread_id': 't3', 'source': 'user', 'content_type': 'text',
                'text_content': 'ok text', 'metadata': {'note': 'has\x00nul'},
                'created_at': '2025-01-03 12:00:00',
            },
        ],
        tags=[
            (1, 'good-tag'),
            (1, 'bad\x00tag'),
            (2, 'child-of-skipped-parent'),
        ],
        images=[
            (1, 'image/png', json.dumps({'caption': 'fine'})),
            (1, 'image/png', json.dumps({'caption': 'has\x00nul'})),
            (2, 'image/png', json.dumps({'caption': 'fine'})),
        ],
    )


class TestUnstorableColumnReason:
    """Unit coverage for the per-column detection helper, including the jsonb escape."""

    def test_raw_nul_and_surrogate_detected_clean_passes(self) -> None:
        """A raw TEXT NUL/surrogate is flagged; clean text and None pass."""
        assert _pg_unstorable_column_reason('a\x00b', is_jsonb=False) is not None
        assert _pg_unstorable_column_reason('\ud800', is_jsonb=False) is not None
        assert _pg_unstorable_column_reason('clean value', is_jsonb=False) is None
        assert _pg_unstorable_column_reason(None, is_jsonb=False) is None

    def test_jsonb_escape_needs_the_decoded_check(self) -> None:
        """A metadata NUL serializes to the \\u0000 escape (no literal byte); only the
        decoded-structure check under is_jsonb=True catches it."""
        escaped = json.dumps({'note': 'has\x00nul'}, ensure_ascii=False)
        assert '\x00' not in escaped  # stored as the six-char escape, not a literal NUL
        # The raw-string check alone misses the escape ...
        assert _pg_unstorable_column_reason(escaped, is_jsonb=False) is None
        # ... but the jsonb path decodes and detects it.
        assert _pg_unstorable_column_reason(escaped, is_jsonb=True) is not None

    def test_jsonb_literal_nul_and_malformed_json(self) -> None:
        """A literal NUL in the serialized jsonb is flagged; unparseable JSON destined for a
        jsonb column is itself unstorable (the ``::jsonb`` cast rejects it mid-transaction),
        while the same unparseable content in a raw TEXT column has no cast and passes."""
        assert _pg_unstorable_column_reason('{"k": "a\x00b"}', is_jsonb=True) is not None
        assert _pg_unstorable_column_reason('{not valid json', is_jsonb=True) is not None
        assert _pg_unstorable_column_reason('{not valid json', is_jsonb=False) is None


class TestSqliteToPostgresqlNulPrecheck:
    """The real cross-backend copy loops skip unstorable rows instead of aborting."""

    @pytest.mark.asyncio
    async def test_nul_rows_skipped_and_run_completes(self, tmp_path: Path) -> None:
        """Every unstorable row is identified and skipped; the clean data still migrates
        and the transaction commits rather than rolling back."""
        source = tmp_path / 'nul_source.db'
        _mixed_source_with_nul_rows(source)

        stats, conn = await _run_with_fake_target(source, dry_run=False)

        # Only the single clean context row is copied.
        assert stats.rows_migrated == 1
        assert len(conn.inserts('context_entries')) == 1
        assert stats.tags_migrated == 1
        assert len(conn.inserts('tags')) == 1
        assert stats.images_migrated == 1
        assert len(conn.inserts('image_attachments')) == 1

        # Each unstorable row is reported with source id and offending column.
        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "column 'text_content'" in errors
        assert 'context_entries row id=3' in errors
        assert "column 'metadata'" in errors
        assert 'tags row context_entry_id=1' in errors
        assert "column 'tag'" in errors
        assert 'image_attachments row context_entry_id=1' in errors
        assert "column 'image_metadata'" in errors

        # Children of the skipped parent (id=2) are skipped, not orphaned onto a
        # never-inserted parent (which would raise an FK violation on PostgreSQL).
        warnings = '\n'.join(stats.warnings)
        assert warnings.count('parent context_entries row was skipped') == 2

        # The run committed; it neither crashed nor rolled back.
        assert 'COMMIT' in conn.statements()
        assert 'ROLLBACK' not in conn.statements()

    @pytest.mark.asyncio
    async def test_dry_run_surfaces_nul_rows_without_inserting(self, tmp_path: Path) -> None:
        """--dry-run reports every unstorable row (restoring the preview guarantee) yet
        writes nothing to the target."""
        source = tmp_path / 'nul_source_dry.db'
        _mixed_source_with_nul_rows(source)

        stats, conn = await _run_with_fake_target(source, dry_run=True)

        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert 'context_entries row id=3' in errors

        # No write of any kind reaches the target under a dry run.
        assert all(not statement.startswith('INSERT') for statement in conn.statements())
        assert 'BEGIN' not in conn.statements()
        assert 'COMMIT' not in conn.statements()
        # The clean row is still counted (matching existing dry-run counting semantics).
        assert stats.rows_migrated == 1

    @pytest.mark.asyncio
    async def test_malformed_metadata_json_skipped_not_bound_to_jsonb(self, tmp_path: Path) -> None:
        """A row whose metadata is unparseable JSON is identified and skipped, never bound into
        the target's ``metadata::jsonb`` cast (which would reject it mid-transaction and abort
        the whole run). The clean row still migrates and the run commits.
        """
        source = tmp_path / 'malformed_metadata_source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': {'a': 'b'},
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'ok text', 'metadata': '{not valid json',
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, conn = await _run_with_fake_target(source, dry_run=False)

        # Only the clean row is copied; the malformed-metadata row is skipped.
        assert stats.rows_migrated == 1
        context_inserts = conn.inserts('context_entries')
        assert len(context_inserts) == 1

        # The malformed row is reported with its source id and the offending column.
        errors = '\n'.join(stats.errors)
        assert 'context_entries row id=2' in errors
        assert "column 'metadata'" in errors

        # The verbatim malformed string never reaches an INSERT bind: had it been bound to the
        # $n::jsonb cast, PostgreSQL would have aborted the transaction mid-run.
        assert all('{not valid json' not in args for args in context_inserts)

        # The run committed rather than rolling back on a mid-transaction jsonb parse error.
        assert 'COMMIT' in conn.statements()
        assert 'ROLLBACK' not in conn.statements()
