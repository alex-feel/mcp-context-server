"""Unit tests for the --re-embed CLI handler.

These tests cover the CLI surface only: argparse flag handling, mutual
exclusivity, dispatcher logic, dry-run output shape, env-var validation,
and the dimension-change refusal. End-to-end coverage with a real SQLite
database lives in ``tests/integration/sqlite/test_migrate_reembed_e2e.py``.
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


def _bootstrap_schema(path: Path, *, with_embedding_metadata: bool = True) -> None:
    """Apply the SQLite base schema and (optionally) embedding_metadata."""
    from app.schemas import load_schema

    conn = sqlite3.connect(str(path))
    try:
        conn.executescript(load_schema('sqlite'))
        if with_embedding_metadata:
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


def _seed_entry(path: Path, entry_id: str, text: str) -> None:
    """Insert one context_entries row with text and no embeddings."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            'INSERT INTO context_entries '
            '(id, thread_id, source, content_type, text_content) '
            "VALUES (?, 'thread-a', 'user', 'text', ?)",
            (entry_id, text),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_embedding_metadata(path: Path, context_id: str, model: str, dim: int) -> None:
    """Insert one embedding_metadata row (existing-embedding marker)."""
    conn = sqlite3.connect(str(path))
    try:
        conn.execute(
            'INSERT INTO embedding_metadata '
            '(context_id, model_name, dimensions, chunk_count) VALUES (?, ?, ?, 1)',
            (context_id, model, dim),
        )
        conn.commit()
    finally:
        conn.close()


def test_build_parser_accepts_re_embed_flag() -> None:
    """``--re-embed`` is parsed as a boolean toggle."""
    parser = build_parser()
    args = parser.parse_args(['--source-url', 'sqlite:///fake.db', '--re-embed'])
    assert args.re_embed is True
    assert args.compress is False
    assert args.decompress is False


def test_re_embed_mutually_exclusive_with_compress() -> None:
    """``--re-embed --compress`` is rejected by argparse (mode_group)."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--source-url', 'sqlite:///fake.db', '--re-embed', '--compress'])


def test_re_embed_mutually_exclusive_with_decompress() -> None:
    """``--re-embed --decompress`` is rejected by argparse (mode_group)."""
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(['--source-url', 'sqlite:///fake.db', '--re-embed', '--decompress'])


def test_main_requires_generation_enabled_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``run_reembed`` exits 1 when ENABLE_EMBEDDING_GENERATION=false."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'ENABLE_EMBEDDING_GENERATION=true' in err
    assert 'RE-EMBED ALL' in err  # warning printed before validation


def test_embedding_metadata_table_missing_returns_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """When ``embedding_metadata`` is absent, the CLI surfaces a clear error."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db, with_embedding_metadata=False)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'embedding_metadata table not present' in err


def test_dry_run_emits_plan_without_provider_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--dry-run`` reports the entry count without invoking the provider."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    _seed_entry(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A')
    _seed_entry(db, 'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', 'doc B')

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed', '--dry-run'])

    assert rc == 0
    err = capsys.readouterr().err
    assert 'Re-embedding 2 entries' in err
    assert '[DRY-RUN] Provider not called' in err


def test_dimension_change_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A configured EMBEDDING_DIM different from the stored dim is refused."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('DB_PATH', str(tmp_path / 'test.db'))
    get_settings.cache_clear()
    db = tmp_path / 'test.db'
    _bootstrap_schema(db)
    _seed_entry(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'doc A')
    # Existing embedding recorded at a DIFFERENT dimension (512 != 1024).
    _seed_embedding_metadata(db, 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', 'old-model', 512)

    rc = cli_main(['--source-url', f'sqlite:///{db}', '--re-embed'])

    assert rc == 1
    err = capsys.readouterr().err
    assert 'EMBEDDING_DIM=1024' in err
    assert '[512]' in err


def test_dispatch_re_embed_standalone(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--re-embed`` standalone dispatches directly to run_reembed."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_compression.run_compress') as mock_compress,
        patch('app.cli.migrate_compression.run_decompress') as mock_decompress,
        patch('app.cli.migrate_reembed.run_reembed', return_value=0) as mock_reembed,
    ):
        rc = cli_main(['--source-url', 'sqlite:///fake.db', '--re-embed'])

    assert rc == 0
    mock_compress.assert_not_called()
    mock_decompress.assert_not_called()
    mock_reembed.assert_called_once()


def test_dispatch_re_embed_supersedes_embed_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """``--re-embed --embed-missing`` runs only re-embed (the superset)."""
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'true')
    get_settings.cache_clear()

    with (
        patch('app.cli.migrate_reembed.run_reembed', return_value=0) as mock_reembed,
        patch('app.cli.migrate_embeddings.run_embed_missing') as mock_embed,
    ):
        rc = cli_main([
            '--source-url', 'sqlite:///fake.db',
            '--re-embed', '--embed-missing',
        ])

    assert rc == 0
    mock_reembed.assert_called_once()
    mock_embed.assert_not_called()
