"""Regression tests for the SQLite->PostgreSQL btree index-tuple pre-check.

SQLite indexes a value of any size; PostgreSQL refuses an index tuple larger than
BTMaxItemSize (2704 bytes on btree version 4). A legacy corpus predating the
write-path length caps can therefore hold a thread_id, a tag, or an indexed metadata
value the PostgreSQL target cannot index, and binding it aborts the INSERT
mid-transaction, ROLLBACKs the whole run, and reports a raw driver error naming no
source row. The pre-check converts that into a per-row skip-and-warn, so these tests
pin two properties:

* the payload budget is derived from the widest index each value feeds, so a value
  that passes really is indexable (thread_id shares its tuple with the ``source`` and
  ``content_hash`` columns of idx_context_entries_dedup_hash, while a tag and a
  string-typed metadata field are indexed on their own);
* every column the target schema indexes is covered, including the values stored
  under string-typed ``METADATA_INDEXED_FIELDS`` keys, whose expression index
  ``idx_metadata_<field>`` carries the same ceiling.

The full-run tests drive the real ``run_migration_mixed_sqlite_to_postgresql`` against
a fake asyncpg target connection (the PostgreSQL probe helpers are patched out), so no
live PostgreSQL is required; the per-row copy loops -- the code under test -- run
unchanged.
"""

import json
import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest import mock

import pytest

from app.cli.migrate import _PG_BTREE_MAX_ITEM_BYTES
from app.cli.migrate import _PG_MAX_INDEXED_THREAD_ID_BYTES
from app.cli.migrate import _PG_MAX_INDEXED_VALUE_BYTES
from app.cli.migrate import MigrationOptions
from app.cli.migrate import MigrationStats
from app.cli.migrate import _first_pg_unindexable_metadata_field
from app.cli.migrate import _pg_unindexable_column_reason
from app.cli.migrate import run_migration_mixed_sqlite_to_postgresql
from app.settings import get_settings

# Bytes an index tuple spends on things other than the checked payload, used by the
# arithmetic assertions below: the MAXALIGNed IndexTupleData header with a null bitmap
# (16) plus the long varlena header a text datum past 126 bytes carries (4).
_INDEX_TUPLE_HEADER_BYTES = 16 + 4
# The trailing columns of idx_context_entries_dedup_hash(thread_id, source, content_hash):
# 'agent' as a short-header varlena (6) and a 64-character SHA-256 hex string (65).
_DEDUP_TRAILING_COLUMN_BYTES = 6 + 65

# Integer-keyed source schema (the shape the CLI accepts as input). Production code
# under app/ no longer uses this layout; each migration test file defines its own
# bootstrap copy so it stays self-contained.
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
) -> None:
    """Create an integer-keyed source DB at ``path`` and seed it.

    Each entry dict needs ``id``, ``thread_id``, ``source``, ``content_type``,
    ``text_content``, ``metadata`` (a dict serialized to JSON, or None), and
    ``created_at``. Tags are ``(context_entry_id, tag)`` pairs.
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
        conn.commit()
    finally:
        conn.close()


class _FakeTargetConn:
    """Minimal async stand-in for the asyncpg target connection."""

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


async def _run_with_fake_target(
    source: Path,
    *,
    dry_run: bool = False,
) -> tuple[MigrationStats, _FakeTargetConn]:
    """Run the SQLite->PostgreSQL migration against a fake target connection.

    Patches ``asyncpg.connect`` and the PostgreSQL probe helpers so the real per-row
    copy loops run without a live PostgreSQL server.

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


class TestBtreeBudgets:
    """The accepted payload sizes really fit the index tuples they land in."""

    def test_thread_id_budget_fits_the_dedup_index_tuple(self) -> None:
        """A thread_id at the budget still fits idx_context_entries_dedup_hash.

        The base schema declares that index on (thread_id, source, content_hash), so the
        thread_id payload shares its tuple with two more datums. A budget that accounts
        only for the tuple header lets a value pass the guard and then abort the whole
        run with ``index row size ... exceeds btree version 4 maximum 2704``.
        """
        widest_tuple = (
            _INDEX_TUPLE_HEADER_BYTES + _PG_MAX_INDEXED_THREAD_ID_BYTES + _DEDUP_TRAILING_COLUMN_BYTES
        )
        assert widest_tuple <= _PG_BTREE_MAX_ITEM_BYTES

    def test_single_column_budget_fits_its_index_tuple(self) -> None:
        """A tag or string-typed metadata value at the budget fits its own index tuple."""
        assert _INDEX_TUPLE_HEADER_BYTES + _PG_MAX_INDEXED_VALUE_BYTES <= _PG_BTREE_MAX_ITEM_BYTES

    def test_thread_id_budget_is_the_narrower_one(self) -> None:
        """thread_id gets less room than a value indexed on its own."""
        assert _PG_MAX_INDEXED_THREAD_ID_BYTES < _PG_MAX_INDEXED_VALUE_BYTES

    def test_boundary_values_are_accepted_and_rejected(self) -> None:
        """Each budget accepts its exact size and rejects one byte more."""
        for budget in (_PG_MAX_INDEXED_THREAD_ID_BYTES, _PG_MAX_INDEXED_VALUE_BYTES):
            assert _pg_unindexable_column_reason('a' * budget, budget) is None
            reason = _pg_unindexable_column_reason('a' * (budget + 1), budget)
            assert reason is not None
            assert str(budget) in reason

    def test_multibyte_values_are_measured_in_utf8_bytes(self) -> None:
        """A short string of wide code points can still exceed the byte budget."""
        # Each code point encodes to 3 UTF-8 bytes, so half the budget in characters
        # is one and a half budgets in bytes.
        value = '中' * _PG_MAX_INDEXED_THREAD_ID_BYTES
        assert _pg_unindexable_column_reason(value, _PG_MAX_INDEXED_THREAD_ID_BYTES) is not None

    def test_none_and_ordinary_values_pass(self) -> None:
        """Absent and normally sized values are never flagged."""
        assert _pg_unindexable_column_reason(None, _PG_MAX_INDEXED_VALUE_BYTES) is None
        assert _pg_unindexable_column_reason('thread-1', _PG_MAX_INDEXED_VALUE_BYTES) is None


class TestIndexedMetadataFields:
    """Values under string-typed METADATA_INDEXED_FIELDS keys are checked too."""

    def test_oversized_string_under_an_indexed_key_is_flagged(self) -> None:
        """A default indexed field carrying an oversized value names the field."""
        metadata = json.dumps({'task_name': 'a' * 3000})
        found = _first_pg_unindexable_metadata_field(metadata)

        assert found is not None
        column, reason = found
        assert column == 'metadata.task_name'
        assert '3000 UTF-8 bytes' in reason

    def test_oversized_value_under_a_non_indexed_key_passes(self) -> None:
        """Only indexed keys are capped; jsonb itself imposes no such limit."""
        metadata = json.dumps({'notes': 'a' * 5000})
        assert _first_pg_unindexable_metadata_field(metadata) is None

    def test_array_and_object_fields_are_not_expression_indexed(self) -> None:
        """The default ``references``/``technologies`` fields are served by the GIN index.

        They are excluded from expression indexing on both backends, so a large value
        under them is perfectly storable and must not cost the row its migration.
        """
        metadata = json.dumps(
            {
                'references': {'context_ids': [f'{index:032x}' for index in range(300)]},
                'technologies': ['python'] * 2000,
            },
        )
        assert len(metadata) > _PG_MAX_INDEXED_VALUE_BYTES
        assert _first_pg_unindexable_metadata_field(metadata) is None

    def test_container_under_a_string_typed_key_is_measured_as_serialized_text(self) -> None:
        """``metadata->>'<field>'`` yields a container's serialized form, which is indexed."""
        metadata = json.dumps({'project': ['a' * 100] * 40})
        found = _first_pg_unindexable_metadata_field(metadata)

        assert found is not None
        assert found[0] == 'metadata.project'

    def test_typed_field_holding_an_uncastable_value_is_flagged(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value the target's index cast rejects loses its row, not the whole run.

        SQLite indexes ``{"priority": "high"}`` under an integer-typed field without a
        cast and stores it happily; the PostgreSQL expression index evaluates
        ``(metadata->>'priority')::INTEGER`` on every INSERT and refuses it, aborting
        the transaction with a raw driver error that names no source row.

        Args:
            monkeypatch: Used to configure a typed indexed field.
        """
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'priority:integer')
        get_settings.cache_clear()

        found = _first_pg_unindexable_metadata_field(json.dumps({'priority': 'high'}))

        assert found is not None
        column, reason = found
        assert column == 'metadata.priority'
        assert 'not a valid integer' in reason
        assert 'skipped' in reason

    def test_typed_field_holding_a_castable_value_passes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A value the cast accepts migrates untouched.

        Args:
            monkeypatch: Used to configure a typed indexed field.
        """
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'priority:integer')
        get_settings.cache_clear()

        assert _first_pg_unindexable_metadata_field(json.dumps({'priority': 7})) is None
        assert _first_pg_unindexable_metadata_field(json.dumps({'priority': '7'})) is None

    def test_json_null_and_small_values_pass(self) -> None:
        """A JSON null indexes as SQL NULL (excluded by the index predicate)."""
        assert _first_pg_unindexable_metadata_field(json.dumps({'task_name': None})) is None
        assert _first_pg_unindexable_metadata_field(json.dumps({'task_name': 'audit'})) is None
        assert _first_pg_unindexable_metadata_field(json.dumps({'status': 42})) is None

    def test_absent_unparseable_and_non_object_metadata_pass(self) -> None:
        """Shapes the jsonb bind rejects on its own are left to the unstorable check."""
        assert _first_pg_unindexable_metadata_field(None) is None
        assert _first_pg_unindexable_metadata_field('{not json') is None
        assert _first_pg_unindexable_metadata_field('[1, 2, 3]') is None


class TestSqliteToPostgresqlIndexPrecheck:
    """The real cross-backend copy loops skip unindexable rows instead of aborting."""

    @pytest.mark.asyncio
    async def test_oversized_indexed_metadata_row_is_skipped(self, tmp_path: Path) -> None:
        """A row whose indexed metadata value cannot be indexed is skipped, not fatal."""
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': {'task_name': 'audit'},
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'legacy entry', 'metadata': {'task_name': 'a' * 3000},
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
            tags=[(1, 'good-tag'), (2, 'child-of-skipped-parent')],
        )

        stats, fake = await _run_with_fake_target(source)

        assert stats.rows_migrated == 1
        assert len(fake.inserts('context_entries')) == 1
        skip_errors = [error for error in stats.errors if 'id=2' in error]
        assert len(skip_errors) == 1
        assert "'metadata.task_name'" in skip_errors[0]
        # The skipped parent takes its children with it (an FK violation would relocate
        # the very abort this guard prevents).
        assert [args[1] for args in fake.inserts('tags')] == ['good-tag']
        assert stats.tags_migrated == 1

    @pytest.mark.asyncio
    async def test_oversized_thread_id_row_is_skipped(self, tmp_path: Path) -> None:
        """A thread_id between the single-column and compound budgets is skipped.

        Such a value fits an index tuple of its own but not the dedup index tuple it
        actually lands in, so it must be caught by the pre-check rather than by the
        driver mid-transaction.
        """
        oversized = 'a' * (_PG_MAX_INDEXED_THREAD_ID_BYTES + 1)
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': oversized, 'source': 'user', 'content_type': 'text',
                    'text_content': 'legacy entry', 'metadata': None,
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': None,
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, fake = await _run_with_fake_target(source)

        assert stats.rows_migrated == 1
        assert len(fake.inserts('context_entries')) == 1
        assert any("column 'thread_id' skipped" in error for error in stats.errors)

    @pytest.mark.asyncio
    async def test_skipped_row_does_not_count_its_reference_rewrites(self, tmp_path: Path) -> None:
        """Remappings inside a skipped row never reach the target, so they are not counted."""
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'clean entry', 'metadata': None,
                    'created_at': '2025-01-01 12:00:00',
                },
                {
                    'id': 2, 'thread_id': 't2', 'source': 'agent', 'content_type': 'text',
                    'text_content': 'legacy entry',
                    'metadata': {'task_name': 'a' * 3000, 'references': {'context_ids': [1, 1, 1]}},
                    'created_at': '2025-01-02 12:00:00',
                },
            ],
        )

        stats, _ = await _run_with_fake_target(source)

        assert stats.rows_migrated == 1
        assert stats.references_rewritten == 0

    @pytest.mark.asyncio
    async def test_dry_run_surfaces_the_skip_without_inserting(self, tmp_path: Path) -> None:
        """--dry-run reports the same unindexable rows before a real run touches the target."""
        source = tmp_path / 'source.db'
        _seed_source(
            source,
            entries=[
                {
                    'id': 1, 'thread_id': 't1', 'source': 'user', 'content_type': 'text',
                    'text_content': 'legacy entry', 'metadata': {'project': 'a' * 4000},
                    'created_at': '2025-01-01 12:00:00',
                },
            ],
        )

        stats, fake = await _run_with_fake_target(source, dry_run=True)

        assert stats.rows_migrated == 0
        assert fake.inserts('context_entries') == []
        assert any("'metadata.project'" in error for error in stats.errors)
