"""Tests for ``--compress`` planning: environment validation, preflight aborts, the dry-run plan and idempotency.

End-to-end coverage with real SQLite and PostgreSQL databases lives in
``tests/integration/sqlite/test_migrate_compress_e2e.py`` and
``tests/integration/postgresql/test_migrate_compress_e2e_postgresql.py``.
"""

import sqlite3
from pathlib import Path

import pytest

from app.cli.migrate import main as cli_main
from app.cli.migrate_compression.compress import run_compress
from app.cli.migrate_compression.console import WARNING_BORDER
from app.pgvector_limits import PGVECTOR_INDEX_DIM_LIMIT
from app.settings import get_settings
from tests.cli.migrate_compression._sqlite_db import bootstrap_schema
from tests.cli.migrate_compression._sqlite_db import count_fp32
from tests.cli.migrate_compression._sqlite_db import create_fp32_vec_table
from tests.cli.migrate_compression._sqlite_db import enable_compression
from tests.cli.migrate_compression._sqlite_db import seed_fp32_database
from tests.cli.migrate_compression._sqlite_db import table_exists


def test_main_requires_compression_enabled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_compress`` exits 1 when ENABLE_EMBEDDING_COMPRESSION is unset."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--compress'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'ENABLE_EMBEDDING_COMPRESSION=true' in err
    assert 'BACKUP REQUIRED' in err  # warning was printed before validation


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
    bootstrap_schema(db)  # schema only; no vec_context_embeddings table

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
    bootstrap_schema(db)
    create_fp32_vec_table(db)

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
    bootstrap_schema(db)

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


def test_compress_invalid_env_surfaces_clean_cli_error(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """run_compress shares the clean env-validation error path with run_decompress."""
    from app.errors import ConfigurationError

    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_DIM', str(PGVECTOR_INDEX_DIM_LIMIT + 1))
    get_settings.cache_clear()

    rc = run_compress('postgresql://u:p@localhost:5432/ctx', dry_run=False)

    assert rc == ConfigurationError.EXIT_CODE
    err = capsys.readouterr().err
    assert 'Configuration invalid' in err
    assert 'pgvector index limit' in err


@pytest.mark.integration
def test_compress_aborts_on_byte_alignment_violation_sqlite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A misaligned EMBEDDING_DIM aborts --compress BEFORE any destructive DROP.

    The per-row encode would succeed and DROP the fp32 table, but every compressed
    search (and server startup) would then fail. The CLI must reject the config up
    front and leave the fp32 source intact.
    """
    db = tmp_path / 'misaligned.db'
    seed_fp32_database(db, monkeypatch, n_docs=3)
    enable_compression(monkeypatch)
    # 1020 * (4 - 1) = 3060, not a multiple of 8 -> compressed read would crash.
    monkeypatch.setenv('EMBEDDING_DIM', '1020')
    get_settings.cache_clear()

    rc = run_compress(f'sqlite:///{db}', dry_run=False)
    assert rc == 1
    # The fp32 source is preserved (never dropped); no compressed table created.
    assert table_exists(db, 'vec_context_embeddings')
    assert count_fp32(db) == 3
    assert not table_exists(db, 'vec_context_embeddings_compressed')
