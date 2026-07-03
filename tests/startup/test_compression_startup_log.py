"""Tests for the inline compression startup announcement.

``lifespan()`` in ``app/server.py`` emits an INFO-level announcement of the
active compression configuration immediately after
``validate_compression_provenance(backend=backend)``. The announcement
mirrors the sibling feature announcements (embedding generation, reranking,
chunking, summary) in style and depth: an operator-facing INFO line that
surfaces the active configuration at a glance when ``LOG_LEVEL=INFO``.

The announcement uses an f-string and lives inline in the lifespan body --
no helper module.

These tests invoke ``lifespan`` directly with a real ``FastMCP`` instance
and assert the expected INFO record appears. Other feature subsystems
(embeddings, summary, FTS) are disabled via env vars so lifespan completes
without external service dependencies.
"""

import logging
import os
import tempfile
from collections.abc import Generator
from pathlib import Path

import pytest

import app.server as server_module
from app.settings import get_settings


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings singleton before and after each test.

    Compression env flips would otherwise leak via the process-global
    ``get_settings()`` lru_cache and pollute unrelated tests.

    Yields:
        Control to the test body.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _baseline_disable_external_services(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Disable every feature that would require an external service.

    The lifespan flow initializes embedding/summary providers and queries
    Ollama; this test only cares about the compression INFO log, so we
    disable those subsystems to keep the lifespan fully local.
    """
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'false')
    monkeypatch.setenv('ENABLE_FTS', 'false')
    monkeypatch.setenv('ENABLE_HYBRID_SEARCH', 'false')
    monkeypatch.setenv('ENABLE_CHUNKING', 'false')
    monkeypatch.setenv('ENABLE_RERANKING', 'false')
    monkeypatch.setenv('ENABLE_SUMMARY_GENERATION', 'false')


async def _run_lifespan_to_yield(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drive ``server.lifespan`` through its startup block.

    Constructs a real ``FastMCP`` instance, enters the context manager,
    and exits cleanly so the shutdown phase runs too. Refreshes module
    bindings so the lifespan sees the env-driven settings flipped in the
    test.
    """
    from fastmcp import FastMCP

    # Refresh the cached settings in server.py so the lifespan sees the
    # env-flipped values.
    get_settings.cache_clear()
    monkeypatch.setattr(server_module, 'settings', get_settings())
    # lifespan() creates the backend from server.py's own module-level DB_PATH
    # (imported `from app.startup import DB_PATH`, bound at import time to the
    # default DB). Repoint it at the per-test isolated DB so the lifespan never
    # touches the developer's default database -- otherwise the enabled test
    # would bootstrap a compression_metadata row into it and the disabled test
    # would then trip the validator's disabled-branch guard against it.
    monkeypatch.setattr(server_module, 'DB_PATH', Path(os.environ['DB_PATH']))
    # Refresh the cached settings in modules touched by lifespan that bind
    # at import time.
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    mcp: FastMCP[None] = FastMCP('test-compression-startup-log')
    cm = server_module.lifespan(mcp)
    async with cm:
        pass  # Lifespan startup ran; immediately yield + shut down.


@pytest.fixture
def isolated_db_path(monkeypatch: pytest.MonkeyPatch) -> Generator[str, None, None]:
    """Provide a clean SQLite DB path scoped to this test only.

    Yields:
        Path string for the per-test SQLite database. Teardown removes the
        temporary directory; if SQLite WAL files keep file handles open the
        directory survives -- pytest's tmp cleanup eventually reclaims it.
    """
    import shutil
    tmpdir = tempfile.mkdtemp(prefix='test_compression_log_')
    db_path = str(Path(tmpdir) / 'test.db')
    monkeypatch.setenv('DB_PATH', db_path)
    monkeypatch.setenv('STORAGE_BACKEND', 'sqlite')
    try:
        yield db_path
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.asyncio
async def test_compression_startup_log_enabled(
    monkeypatch: pytest.MonkeyPatch,
    isolated_db_path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lifespan emits the compression-enabled INFO log inline after the validator.

    Asserts on the substring ``Embedding compression enabled with provider:
    turboquant`` plus the configured bits/variant/dim/seed/max_concurrent
    parameters so the lifespan announcement mirrors the sibling feature
    announcements in style and depth.

    The baseline disables embedding generation (to keep the lifespan free of
    external services), and compression provisioning follows embedding
    storage, so a provenance row is pre-seeded here: the announcement fires
    for a database that already carries compressed provenance, matching a
    real deployment whose data was compressed while generation was on. The
    fingerprint is left NULL (the documented pre-fingerprint state) so the
    validator warns and proceeds instead of deriving a rotation.
    """
    _baseline_disable_external_services(monkeypatch)
    # Enable compression with deterministic values for the assertion.
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_PROVIDER', 'turboquant')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('COMPRESSION_SEED', '7')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')

    import sqlite3

    conn = sqlite3.connect(isolated_db_path)
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
                codebook_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                payload BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 7, 1024, NULL)",
        )
        conn.commit()
    finally:
        conn.close()

    caplog.set_level(logging.INFO, logger='app.server')

    await _run_lifespan_to_yield(monkeypatch)

    log_text = caplog.text
    assert 'Embedding compression enabled with provider: turboquant' in log_text, (
        'Lifespan did not emit the inline INFO line announcing '
        f'compression configuration. caplog text:\n{log_text}'
    )
    # Assert the announcement carries the configured parameters so the
    # operator can verify them at a glance.
    assert 'bits=4' in log_text
    assert 'variant=ip' in log_text
    assert 'dim=1024' in log_text
    assert 'seed=7' in log_text
    assert 'max_concurrent=' in log_text


@pytest.mark.asyncio
async def test_compression_startup_log_idle_when_generation_disabled(
    monkeypatch: pytest.MonkeyPatch,
    isolated_db_path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lifespan announces the idle state on a generation-off fresh database.

    Compression provisioning follows embedding storage, so with generation
    disabled and nothing previously compressed there is no schema and no
    provenance row -- an expected idle state the lifespan must announce as
    such, not as a missing-row inconsistency.
    """
    del isolated_db_path  # Implicit through DB_PATH env var
    _baseline_disable_external_services(monkeypatch)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')

    caplog.set_level(logging.INFO, logger='app.server')

    await _run_lifespan_to_yield(monkeypatch)

    log_text = caplog.text
    assert 'Embedding compression enabled but idle' in log_text, (
        'Lifespan did not emit the idle-state INFO line for a generation-off '
        f'fresh database. caplog text:\n{log_text}'
    )
    assert 'provenance row missing' not in log_text


@pytest.mark.asyncio
async def test_compression_startup_log_disabled(
    monkeypatch: pytest.MonkeyPatch,
    isolated_db_path: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Lifespan emits the compression-disabled INFO log when feature is off."""
    del isolated_db_path
    _baseline_disable_external_services(monkeypatch)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')

    caplog.set_level(logging.INFO, logger='app.server')

    await _run_lifespan_to_yield(monkeypatch)

    log_text = caplog.text
    assert (
        'Embedding compression disabled (ENABLE_EMBEDDING_COMPRESSION=false)'
        in log_text
    ), (
        'Lifespan did not emit the inline INFO line announcing '
        f'compression disabled. caplog text:\n{log_text}'
    )
