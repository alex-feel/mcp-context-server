"""Regression tests for the SQLite->SQLite compressed-target backstop.

A v3 server started once against a fresh database path leaves an EMPTY database
that already carries compression provenance (compression is default-on):
``compression_metadata`` with a singleton row, ``vec_context_embeddings_compressed``,
and zero ``context_entries`` rows. Migrating an embedding-carrying source into that
database used to report complete success while silently condemning every migrated
vector: ``initialize_target_sqlite`` re-creates the fp32 ``vec_context_embeddings``
table (``CREATE VIRTUAL TABLE IF NOT EXISTS`` masks the compressed layout), the
follow-up ``--compress`` is a no-op because a provenance row already exists, and the
next server start applies the compression migration whose leading
``DROP TABLE IF EXISTS vec_context_embeddings`` destroys the copied vectors.

The PostgreSQL runner has always refused this shape; these tests pin the symmetric
SQLite refusal, including the ``--dry-run`` preview (which must probe the REAL target
file, not the in-memory dry-run handle).
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from app.cli.migrate import MigrationOptions
from app.cli.migrate import _read_schema_file
from app.cli.migrate import run_migration_sqlite_to_sqlite
from app.cli.migrate import target_sqlite_is_compressed
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
'''

_EMBEDDING_METADATA_DDL = (
    'CREATE TABLE embedding_metadata ('
    'context_id INTEGER NOT NULL PRIMARY KEY, '
    'model_name TEXT NOT NULL, '
    'dimensions INTEGER NOT NULL, '
    'chunk_count INTEGER NOT NULL DEFAULT 1, '
    'created_at TEXT NOT NULL, '
    'updated_at TEXT NOT NULL)'
)

_COMPRESSION_METADATA_DDL = (
    'CREATE TABLE compression_metadata ('
    'id INTEGER PRIMARY KEY CHECK (id = 1), '
    'provider TEXT NOT NULL, '
    'bits INTEGER NOT NULL, '
    'variant TEXT NOT NULL, '
    'seed INTEGER NOT NULL, '
    'dim INTEGER NOT NULL, '
    'codebook_fingerprint TEXT, '
    'created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)'
)

_COMPRESSED_PAYLOAD_DDL = (
    'CREATE TABLE vec_context_embeddings_compressed ('
    'id INTEGER PRIMARY KEY AUTOINCREMENT, '
    'context_id TEXT NOT NULL, '
    'chunk_index INTEGER NOT NULL, '
    'payload BLOB NOT NULL)'
)

_TARGET_CONTEXT_ENTRIES_DDL = (
    'CREATE TABLE context_entries ('
    'id TEXT NOT NULL UNIQUE, '
    'thread_id TEXT NOT NULL, '
    'source TEXT NOT NULL, '
    'content_type TEXT NOT NULL, '
    'text_content TEXT)'
)


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings cache around every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _build_options(source: Path, target: Path, *, dry_run: bool = False) -> MigrationOptions:
    """Build a :class:`MigrationOptions` for direct invocation.

    Returns:
        Options addressing ``source`` and ``target`` as SQLite URLs.
    """
    return MigrationOptions(
        source_url=f'sqlite:///{source.as_posix()}',
        target_url=f'sqlite:///{target.as_posix()}',
        dry_run=dry_run,
        report_path=None,
    )


def _seed_source_with_embeddings(path: Path) -> None:
    """Create an integer-keyed source carrying an ``embedding_metadata`` table.

    The backstop keys on the SOURCE having embeddings, which
    ``detect_optional_tables`` reports from table presence alone, so no vec0
    virtual table (and therefore no sqlite-vec extension) is needed here.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_INTEGER_KEYED_SCHEMA_SQL)
        conn.execute(
            'INSERT INTO context_entries '
            '(id, thread_id, source, content_type, text_content, created_at, updated_at) '
            "VALUES (1, 'thread-c', 'user', 'text', 'entry with an embedding', "
            "'2025-06-01 10:00:00', '2025-06-01 10:00:00')",
        )
        conn.execute(_EMBEDDING_METADATA_DDL)
        conn.execute(
            'INSERT INTO embedding_metadata VALUES (?, ?, ?, ?, ?, ?)',
            (1, 'model-a', 4, 1, '2025-06-01 10:00:00', '2025-06-01 10:00:00'),
        )
        conn.commit()
    finally:
        conn.close()


def _make_compressed_target(path: Path, *, with_provenance_row: bool = True) -> None:
    """Create the shape a v3 server leaves behind after one start against a new path.

    The real packaged base schema is applied so the target is indistinguishable from
    a server-initialized empty database; the compression tables are layered on top,
    exactly as the default-on compression migration would leave them.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(_read_schema_file('sqlite_schema.sql'))
        conn.execute(_COMPRESSION_METADATA_DDL)
        conn.execute(_COMPRESSED_PAYLOAD_DDL)
        if with_provenance_row:
            conn.execute(
                'INSERT INTO compression_metadata '
                '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
                "VALUES (1, 'turboquant', 4, 'ip', 0, 4, NULL)",
            )
        conn.commit()
    finally:
        conn.close()


def _table_names(path: Path) -> set[str]:
    """Return every table name present in the SQLite database at ``path``.

    Returns:
        The set of table names.
    """
    conn = sqlite3.connect(str(path))
    try:
        return {
            str(row[0])
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    finally:
        conn.close()


def _target_row_count(path: Path) -> int:
    """Return the number of ``context_entries`` rows in the database at ``path``.

    Returns:
        The row count.
    """
    conn = sqlite3.connect(str(path))
    try:
        return int(conn.execute('SELECT COUNT(*) FROM context_entries').fetchone()[0])
    finally:
        conn.close()


class TestTargetSqliteIsCompressed:
    """The target probe reads the REAL file and recognizes both compression markers."""

    def test_missing_or_empty_file_is_not_compressed(self, tmp_path: Path) -> None:
        """A nonexistent or zero-byte target file is treated as uncompressed."""
        assert target_sqlite_is_compressed(str(tmp_path / 'absent.db')) is False
        empty = tmp_path / 'empty.db'
        empty.write_bytes(b'')
        assert target_sqlite_is_compressed(str(empty)) is False

    def test_fp32_target_is_not_compressed(self, tmp_path: Path) -> None:
        """A target carrying neither compression marker is uncompressed."""
        path = tmp_path / 'fp32.db'
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(_TARGET_CONTEXT_ENTRIES_DDL)
            conn.commit()
        finally:
            conn.close()
        assert target_sqlite_is_compressed(str(path)) is False

    def test_provenance_row_marks_compressed(self, tmp_path: Path) -> None:
        """A populated compression_metadata row marks the target compressed."""
        path = tmp_path / 'provenance.db'
        _make_compressed_target(path)
        assert target_sqlite_is_compressed(str(path)) is True

    def test_compressed_payload_table_marks_compressed(self, tmp_path: Path) -> None:
        """The compressed payload table alone marks the target compressed."""
        path = tmp_path / 'payload_only.db'
        _make_compressed_target(path, with_provenance_row=False)
        assert target_sqlite_is_compressed(str(path)) is True

    def test_cleared_provenance_table_alone_is_not_compressed(self, tmp_path: Path) -> None:
        """A compression_metadata table with no row and no payload table is uncompressed.

        ``--decompress`` clears the provenance row and drops the payload table, so
        this shape is the documented recovery state and must stay migratable.
        """
        path = tmp_path / 'cleared.db'
        conn = sqlite3.connect(str(path))
        try:
            conn.execute(_TARGET_CONTEXT_ENTRIES_DDL)
            conn.execute(_COMPRESSION_METADATA_DDL)
            conn.commit()
        finally:
            conn.close()
        assert target_sqlite_is_compressed(str(path)) is False


class TestCompressedTargetBackstop:
    """An embedding-carrying source is refused against a compressed SQLite target."""

    def test_real_run_aborts_before_target_initialization(self, tmp_path: Path) -> None:
        """The run records an error and never re-creates the fp32 vec table."""
        source = tmp_path / 'source_emb.db'
        _seed_source_with_embeddings(source)
        target = tmp_path / 'compressed_target.db'
        _make_compressed_target(target)

        stats = run_migration_sqlite_to_sqlite(_build_options(source, target))

        assert stats.rows_migrated == 0
        assert any('configured for compressed embeddings' in e for e in stats.errors)
        assert any('--decompress' in e for e in stats.errors)
        assert any('Aborting to avoid silently dropping embeddings' in e for e in stats.errors)

        # The refusal lands BEFORE initialize_target_sqlite, so the fp32 vec table it
        # would have re-created (masking the compressed layout) is still absent and the
        # target still holds no rows.
        names = _table_names(target)
        assert 'vec_context_embeddings' not in names
        assert 'vec_context_embeddings_compressed' in names
        assert _target_row_count(target) == 0

    def test_dry_run_reports_the_refusal(self, tmp_path: Path) -> None:
        """--dry-run inspects the REAL target file and previews the abort."""
        source = tmp_path / 'source_emb_dry.db'
        _seed_source_with_embeddings(source)
        target = tmp_path / 'compressed_target_dry.db'
        _make_compressed_target(target)

        stats = run_migration_sqlite_to_sqlite(_build_options(source, target, dry_run=True))

        assert not stats.errors
        assert any('configured for compressed embeddings' in w for w in stats.warnings)
        assert any('a real run would abort' in w for w in stats.warnings)

    def test_uncompressed_target_still_migrates(self, tmp_path: Path) -> None:
        """The backstop does not fire for an ordinary fresh target."""
        source = tmp_path / 'source_emb_ok.db'
        _seed_source_with_embeddings(source)
        target = tmp_path / 'fresh_target.db'

        stats = run_migration_sqlite_to_sqlite(_build_options(source, target))

        assert not any('compressed embeddings' in e for e in stats.errors)
        assert stats.rows_migrated == 1

    def test_embeddingless_source_is_not_refused(self, tmp_path: Path) -> None:
        """A source with no embedding tables migrates into a compressed target.

        Nothing can be lost when the source carries no vectors, so the backstop
        must not block this shape.
        """
        source = tmp_path / 'source_plain.db'
        conn = sqlite3.connect(str(source))
        try:
            conn.executescript(_INTEGER_KEYED_SCHEMA_SQL)
            conn.execute(
                'INSERT INTO context_entries '
                '(id, thread_id, source, content_type, text_content, created_at, updated_at) '
                "VALUES (1, 'thread-p', 'user', 'text', 'plain entry', "
                "'2025-06-01 10:00:00', '2025-06-01 10:00:00')",
            )
            conn.commit()
        finally:
            conn.close()
        target = tmp_path / 'compressed_target_plain.db'
        _make_compressed_target(target)

        stats = run_migration_sqlite_to_sqlite(_build_options(source, target))

        assert not any('compressed embeddings' in e for e in stats.errors)
        assert stats.rows_migrated == 1
