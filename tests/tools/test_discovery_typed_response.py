"""Tests for the StatisticsResponseDict typed surface of ``get_statistics``.

``app.tools.discovery.get_statistics`` returns a ``StatisticsResponseDict``
declared in ``app/types.py``. The TypedDict gives static type checkers
information about the response shape without changing runtime behavior.

Two assertions in this test:

1. The runtime shape matches the TypedDict: every documented top-level
   key is present at runtime when the corresponding subsystem is enabled.
   This catches the silent-drift case where the TypedDict declaration
   misses a field that the producer actually emits.
2. The producer's API field name ``embedding_count`` is preserved.
   ``embedding_count`` counts the persisted embedding rows on disk. The
   text chunks themselves are NOT persisted (chunking is a transient
   pre-processing step for embedding quality), so the field name reflects
   what is actually stored.
"""

import asyncio
import contextlib
import importlib
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Generator
from pathlib import Path
from typing import Any
from typing import cast

import pytest
import pytest_asyncio

import app.tools.discovery as discovery_module
from app.backends import StorageBackend
from app.backends import create_backend
from app.compression.provenance import insert_compression_metadata
from app.compression.types import CompressionMetadata
from app.migrations.compression import apply_compression_migration
from app.settings import get_settings
from app.startup import set_backend
from app.startup import set_repositories


def _get_statistics_response_dict() -> object:
    """Resolve StatisticsResponseDict at call time.

    Resolving at call time means the test file loads even if the TypedDict
    has not yet been added to ``app.types`` (TDD: this test must fail on
    the missing attribute, not crash at import).

    Returns:
        The StatisticsResponseDict TypedDict class.
    """
    types_module = importlib.import_module('app.types')
    if not hasattr(types_module, 'StatisticsResponseDict'):
        pytest.fail(
            'StatisticsResponseDict is missing from app.types',
        )
    return types_module.StatisticsResponseDict


@pytest.fixture(autouse=True)
def clear_settings_cache() -> Generator[None, None, None]:
    """Reset the settings singleton + startup globals between tests.

    Yields:
        Control to the test body.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    set_backend(None)
    set_repositories(None)


@pytest_asyncio.fixture
async def fresh_backend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend with compression enabled and provenance bootstrapped.

    Yields:
        Initialized backend wired into the startup-layer globals so the
        discovery tool can resolve it via ``ensure_backend()``.
    """
    for var in (
        'ENABLE_EMBEDDING_COMPRESSION', 'COMPRESSION_SEED',
        'COMPRESSION_BITS', 'COMPRESSION_VARIANT', 'COMPRESSION_PROVIDER',
        'EMBEDDING_DIM',
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    get_settings.cache_clear()
    monkeypatch.setattr(discovery_module, 'settings', get_settings())
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    db_path = tmp_path / 'test_typed_response.db'
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    from app.schemas import load_schema
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    await apply_compression_migration(backend=backend)
    await insert_compression_metadata(
        backend,
        CompressionMetadata(
            provider='turboquant', bits=4, variant='ip', seed=42, dim=1024,
        ),
    )

    set_backend(backend)

    try:
        yield backend
    finally:
        set_backend(None)
        set_repositories(None)
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(backend.shutdown(), timeout=5.0)


@pytest.mark.asyncio
async def test_get_statistics_returns_typed_dict_shape(
    fresh_backend: StorageBackend,
) -> None:
    """Runtime shape matches StatisticsResponseDict TypedDict declaration.

    Every required top-level key is present and the sub-block keys for
    semantic_search/fts/chunking/reranking/summary/compression at least
    contain ``enabled`` and ``available`` so the union-of-shape
    assertions on disabled/enabled paths can both pass.
    """
    del fresh_backend  # Implicit via set_backend
    stats = await discovery_module.get_statistics(ctx=None)

    # The function's return TYPE must be StatisticsResponseDict. We can't
    # directly inspect the function's static type at runtime, but we can
    # introspect its annotation.
    import inspect
    sig = inspect.signature(discovery_module.get_statistics)
    expected = _get_statistics_response_dict()
    assert sig.return_annotation is expected, (
        f'get_statistics return annotation is '
        f'{sig.return_annotation}, expected StatisticsResponseDict'
    )

    # Top-level keys produced by StatisticsRepository.get_database_statistics.
    required_keys = {
        'total_entries', 'by_source', 'by_content_type', 'total_images',
        'unique_tags', 'total_threads', 'avg_entries_per_thread',
        'most_active_threads', 'top_tags', 'backend',
    }
    for key in required_keys:
        assert key in stats, f'missing required top-level key: {key}'

    # Sub-blocks added by discovery.py. StatisticsResponseDict has all
    # fields declared total=False; cast the response to a plain dict here
    # so per-key lookups read as runtime structural assertions.
    stats_dict = cast(dict[str, Any], stats)
    for block_name in ('semantic_search', 'fts', 'chunking', 'reranking', 'summary', 'compression'):
        assert block_name in stats_dict, f'missing sub-block: {block_name}'
        assert 'enabled' in stats_dict[block_name], f'{block_name} sub-block missing enabled'
        assert 'available' in stats_dict[block_name], f'{block_name} sub-block missing available'

    # connection_metrics is always present (backend-driven shape)
    assert 'connection_metrics' in stats


@pytest.mark.asyncio
async def test_semantic_search_embedding_count_field_name_preserved(
    fresh_backend: StorageBackend,
) -> None:
    """The API field ``semantic_search.embedding_count`` is preserved.

    The field counts persisted embedding rows on disk. Text chunks are
    NOT persisted (chunking is a transient pre-processing step purely
    for embedding quality), so ``embedding_count`` -- not
    ``total_chunks`` -- is the correct name for what the storage layer
    actually holds. This test guards against accidental drift to a
    misleading ``total_chunks`` rename.
    """
    del fresh_backend  # Implicit via set_backend
    stats = await discovery_module.get_statistics(ctx=None)
    semantic = stats.get('semantic_search', {})
    # When semantic search is disabled (default), the block is just
    # {enabled, available}; we still want to assert that if either appears
    # it never uses 'total_chunks' as a key.
    assert 'total_chunks' not in semantic, (
        'The semantic block must use embedding_count as the field name '
        'for persisted embedding rows. Chunks are transient and not '
        f"persisted. Found 'total_chunks' in semantic block: {semantic}"
    )


@pytest.mark.asyncio
async def test_embeddings_size_reported_when_generation_or_compression_enabled(
    fresh_backend: StorageBackend,
) -> None:
    """Gating regression: embeddings_size_mb is gated on generation OR compression.

    The field is reported whenever ``embedding.generation_enabled`` (default
    True) or ``compression.enabled`` is set, NOT tied to
    ``semantic_search.enabled``. It is a non-negative float carrying a boolean
    estimated flag, reported alongside ``database_size_mb``.
    """
    del fresh_backend  # Implicit via set_backend
    stats = cast(dict[str, Any], await discovery_module.get_statistics(ctx=None))

    assert 'embeddings_size_mb' in stats
    assert isinstance(stats['embeddings_size_mb'], float)
    assert stats['embeddings_size_mb'] >= 0.0
    assert 'embeddings_size_estimated' in stats
    assert isinstance(stats['embeddings_size_estimated'], bool)
    assert 'database_size_mb' in stats
