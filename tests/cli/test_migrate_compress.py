"""Unit tests for the compression CLI handler.

These tests cover the CLI surface only: argparse flag handling,
``--dry-run`` output shape, idempotency logic, and env-var validation.
End-to-end coverage with real SQLite/PostgreSQL databases lives in
``tests/integration/sqlite/test_migrate_compress_e2e.py`` and
``tests/integration/postgresql/test_migrate_compress_e2e_postgresql.py``.
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main
from app.cli.migrate_compression import WARNING_BORDER
from app.cli.migrate_compression import run_compress
from app.cli.migrate_compression import run_decompress
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset settings + compression caches around every test."""
    get_settings.cache_clear()
    _reset_compression_cache()
    yield
    get_settings.cache_clear()
    _reset_compression_cache()


def _bootstrap_schema(path: Path) -> None:
    """Apply the SQLite schema to ``path``."""
    from app.schemas import load_schema

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(load_schema('sqlite'))
        # The semantic-search migration owns embedding_metadata in
        # production; recreate the table directly so the CLI's structural
        # checks pass without dragging in sqlite-vec.
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                context_id TEXT NOT NULL PRIMARY KEY,
                model_name TEXT NOT NULL,
                dimensions INTEGER NOT NULL,
                chunk_count INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS embedding_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                vec_rowid INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            ''',
        )
    finally:
        conn.close()


def _create_fp32_vec_table(path: Path) -> None:
    """Create a minimal stand-in for the fp32 ``vec_context_embeddings`` table.

    The real table is a sqlite-vec virtual table; the CLI's preflight check
    just verifies the name exists, so a plain table with the same column
    name is sufficient for flag-handling tests that do NOT execute the
    encode pass.
    """
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                embedding BLOB
            );
            ''',
        )
    finally:
        conn.close()


def test_build_parser_accepts_compress_flag() -> None:
    """``--compress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.compress is True
    assert args.decompress is False


def test_build_parser_accepts_decompress_flag() -> None:
    """``--decompress`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--decompress',
    ])
    assert args.decompress is True
    assert args.compress is False


def test_build_parser_rejects_both_flags() -> None:
    """``--compress`` and ``--decompress`` are mutually exclusive."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--decompress',
        ])


def test_build_parser_target_url_optional_when_compress() -> None:
    """``--target-url`` is no longer required when --compress is set."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress',
    ])
    assert args.target_url is None


def test_main_requires_compression_enabled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_compress`` exits 1 when ENABLE_EMBEDDING_COMPRESSION is unset."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--compress'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'ENABLE_EMBEDDING_COMPRESSION=true' in err
    assert 'BACKUP REQUIRED' in err  # warning was printed before validation


def test_main_rejects_decompress_when_compression_enabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_decompress`` exits 1 when ENABLE_EMBEDDING_COMPRESSION is true."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'must be unset' in err


def test_main_compress_aborts_when_fp32_table_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_compress`` exits 1 when the fp32 source table is absent."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)  # schema only; no vec_context_embeddings table

    rc = run_compress(f'sqlite:///{db}', dry_run=True)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'vec_context_embeddings not present' in err


def test_main_compress_dry_run_prints_plan_for_empty_table(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Dry-run prints the plan and exits 0 even when the fp32 table is empty."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    _create_fp32_vec_table(db)

    rc = run_compress(f'sqlite:///{db}', dry_run=True)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'BACKUP REQUIRED' in err
    assert WARNING_BORDER in err
    assert '[DRY-RUN]' in err
    assert 'from_table:    vec_context_embeddings' in err
    assert 'to_table:      vec_context_embeddings_compressed' in err

    # Verify dry-run made no destructive changes.
    conn = sqlite3.connect(str(db))
    try:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
    finally:
        conn.close()
    assert 'vec_context_embeddings' in tables
    # compression_metadata is created by the migration that the dry-run
    # also applies (idempotent CREATE IF NOT EXISTS); a dry-run does not
    # roll the migration back -- the migration is non-destructive.
    # The singleton row, however, is NOT inserted by the dry run.
    cursor = conn = sqlite3.connect(str(db))
    try:
        if 'compression_metadata' in tables:
            count = cursor.execute(
                'SELECT COUNT(*) FROM compression_metadata',
            ).fetchone()[0]
            assert count == 0
    finally:
        cursor.close()


def test_main_compress_idempotent_when_already_compressed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Running ``--compress`` twice no-ops with an informational message."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    # Pre-seed the database with the compressed schema + singleton row so
    # the second --compress call recognizes "already compressed" state.
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                payload BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
                variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
                seed INTEGER NOT NULL CHECK (seed >= 0),
                dim INTEGER NOT NULL CHECK (dim > 0),
                codebook_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            INSERT INTO compression_metadata
            (id, provider, bits, variant, seed, dim)
            VALUES (1, 'turboquant', 4, 'ip', 42, 1024);
            ''',
        )
    finally:
        conn.close()

    rc = run_compress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'already present' in err
    assert 'bits=4 variant=ip dim=1024 seed=42' in err


def test_main_decompress_noop_when_nothing_to_do(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` no-ops when the source is already fp32-only."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    _create_fp32_vec_table(db)

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Nothing to do' in err


def test_main_decompress_errors_when_provenance_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--decompress`` exits 1 when the provenance row is absent but the
    compressed table exists (corrupt state)."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                payload BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL,
                variant TEXT NOT NULL,
                seed INTEGER NOT NULL,
                dim INTEGER NOT NULL,
                codebook_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )
    finally:
        conn.close()

    rc = run_decompress(f'sqlite:///{db}', dry_run=False)

    assert rc == 1
    err = capsys.readouterr().err
    assert 'compression_metadata row missing' in err
