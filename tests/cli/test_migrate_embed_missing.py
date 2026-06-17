"""Unit tests for the embed-missing CLI handler.

These tests cover the CLI surface only: argparse flag handling,
dispatcher logic, dry-run output shape, and env-var validation.
End-to-end coverage with real SQLite/PostgreSQL databases lives in
``tests/integration/sqlite/test_migrate_embed_missing_e2e.py``.
"""

import sqlite3
from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import pytest

from app.cli.migrate import build_parser
from app.cli.migrate import main as cli_main
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings cache around every test."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _bootstrap_schema(path: Path) -> None:
    """Apply the SQLite schema + the embedding_metadata table to ``path``."""
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
            ''',
        )
    finally:
        conn.close()


def test_build_parser_accepts_embed_missing_flag() -> None:
    """``--embed-missing`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--embed-missing',
    ])
    assert args.embed_missing is True
    assert args.compress is False
    assert args.decompress is False


def test_build_parser_embed_missing_compose_with_compress() -> None:
    """``--compress --embed-missing`` is NOT mutually exclusive."""
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--compress', '--embed-missing',
    ])
    assert args.compress is True
    assert args.embed_missing is True


def test_build_parser_embed_missing_with_decompress_parses() -> None:
    """``--decompress --embed-missing`` is accepted by argparse.

    The flag is OUTSIDE ``mode_group``, so argparse does not reject the
    combination. The dispatcher in ``main()`` honors ``--decompress``
    and skips ``--embed-missing`` per design. Document the parser-level
    acceptance here.
    """
    parser = build_parser()
    args = parser.parse_args([
        '--source-url', 'sqlite:///fake.db',
        '--decompress', '--embed-missing',
    ])
    assert args.decompress is True
    assert args.embed_missing is True


def test_main_requires_generation_enabled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_embed_missing`` exits 1 when ENABLE_EMBEDDING_GENERATION=false."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'ENABLE_EMBEDDING_GENERATION=true' in err
    assert 'EMBEDDING BACKFILL' in err  # warning printed before validation


def test_no_missing_entries_returns_zero(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When the database has no context_entries rows, exits 0 with count=0."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Found 0 entries with missing embeddings' in err


def test_dry_run_emits_count_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` reports the missing count without invoking the provider."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    # Seed 2 context_entries rows without embedding_metadata counterparts.
    conn = sqlite3.connect(str(db))
    try:
        conn.execute(
            "INSERT INTO context_entries "
            '(id, thread_id, source, content_type, text_content) '
            "VALUES ('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'thread-a', 'user', 'text', 'doc A')",
        )
        conn.execute(
            "INSERT INTO context_entries "
            '(id, thread_id, source, content_type, text_content) '
            "VALUES ('bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'thread-b', 'user', 'text', 'doc B')",
        )
        conn.commit()
    finally:
        conn.close()

    rc = cli_main([
        '--source-url', f'sqlite:///{db}',
        '--embed-missing', '--dry-run',
    ])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Found 2 entries with missing embeddings' in err
    assert '[DRY-RUN] Provider not called' in err


def test_embedding_metadata_table_missing_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When ``embedding_metadata`` is absent, the CLI surfaces a clear error."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'

    # Apply ONLY the base schema, NOT the embedding_metadata table.
    from app.schemas import load_schema
    conn = sqlite3.connect(str(db))
    try:
        conn.executescript(load_schema('sqlite'))
    finally:
        conn.close()

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'embedding_metadata table not present' in err


def test_dispatch_compress_then_embed_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--compress --embed-missing`` invokes both, in order."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    call_order: list[str] = []

    def _fake_compress(*_args: object, **_kwargs: object) -> int:
        call_order.append('compress')
        return 0

    def _fake_embed(*_args: object, **_kwargs: object) -> int:
        call_order.append('embed')
        return 0

    with (
        patch('app.cli.migrate_compression.run_compress', side_effect=_fake_compress),
        patch('app.cli.migrate_embeddings.run_embed_missing', side_effect=_fake_embed),
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--embed-missing',
        ])

    assert rc == 0
    assert call_order == ['compress', 'embed']


def test_dispatch_embed_missing_skipped_when_compress_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When --compress returns non-zero, --embed-missing is NOT attempted."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_compression.run_compress', return_value=2) as mock_compress,
        patch('app.cli.migrate_embeddings.run_embed_missing') as mock_embed,
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--compress', '--embed-missing',
        ])

    assert rc == 2
    mock_compress.assert_called_once()
    mock_embed.assert_not_called()


def test_dispatch_embed_missing_standalone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``--embed-missing`` standalone dispatches directly to run_embed_missing."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_compression.run_compress') as mock_compress,
        patch('app.cli.migrate_compression.run_decompress') as mock_decompress,
        patch('app.cli.migrate_embeddings.run_embed_missing', return_value=0) as mock_embed,
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--embed-missing',
        ])

    assert rc == 0
    mock_compress.assert_not_called()
    mock_decompress.assert_not_called()
    mock_embed.assert_called_once()


def _seed_entry_with_embedding(
    path: Path, entry_id: str, text: str, model: str, dim: int,
) -> None:
    """Insert a context entry plus an embedding_metadata marker row."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            'INSERT INTO context_entries '
            '(id, thread_id, source, content_type, text_content) '
            "VALUES (?, 'thread-a', 'user', 'text', ?)",
            (entry_id, text),
        )
        conn.execute(
            'INSERT INTO embedding_metadata '
            '(context_id, model_name, dimensions, chunk_count) VALUES (?, ?, ?, 1)',
            (entry_id, model, dim),
        )
        conn.commit()
    finally:
        conn.close()


def test_embed_missing_refuses_model_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Existing embeddings from a different model block the backfill."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_MODEL', 'new-model')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    # One entry already embedded under a DIFFERENT model.
    _seed_entry_with_embedding(
        db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A', 'old-model', 1024,
    )

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert "'old-model'" in err
    assert '--re-embed' in err  # points to the whole-corpus path


def test_embed_missing_refuses_dim_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Existing embeddings at a different dimension block the backfill."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_MODEL', 'same-model')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    # Existing embedding recorded at dimension 512 (config wants 1024).
    _seed_entry_with_embedding(
        db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A', 'same-model', 512,
    )

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'EMBEDDING_DIM=1024' in err
    assert '[512]' in err


def _seed_compression_metadata(path: Path, dim: int) -> None:
    """Create + populate the singleton compression_metadata row at ``dim``."""
    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
                variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
                seed INTEGER NOT NULL CHECK (seed >= 0),
                dim INTEGER NOT NULL CHECK (dim > 0),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )
        conn.execute(
            'INSERT INTO compression_metadata (id, provider, bits, variant, seed, dim) '
            "VALUES (1, 'turboquant', 4, 'ip', 0, ?)",
            (dim,),
        )
        conn.commit()
    finally:
        conn.close()


def test_embed_missing_refuses_compression_dim_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A seed-locked compression dim different from EMBEDDING_DIM is refused.

    Regression test for the corruption path where the compressed database is
    sealed at one dimension (compression_metadata) but embedding_metadata is
    still empty: the per-row dimension scan finds nothing, so the guard MUST
    consult compression_metadata to block the mismatched backfill.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '0')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('EMBEDDING_DIM', '512')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    # Database sealed at dim=1024 (no embeddings yet); config wants 512.
    _seed_compression_metadata(db, 1024)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--embed-missing'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'EMBEDDING_DIM=512' in err
    assert 'compression_metadata' in err
