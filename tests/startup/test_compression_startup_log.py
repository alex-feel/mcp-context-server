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
    """
    del isolated_db_path  # Implicit through DB_PATH env var
    _baseline_disable_external_services(monkeypatch)
    # Enable compression with deterministic values for the assertion.
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_PROVIDER', 'turboquant')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    monkeypatch.setenv('COMPRESSION_SEED', '7')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')

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
