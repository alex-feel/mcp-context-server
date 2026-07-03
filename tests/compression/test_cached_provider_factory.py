"""Regression tests for the cached compression provider in factory.py.

Verifies that :func:`app.compression.factory.get_cached_compression_provider`
is the single truth source for compression provider construction (both
read and write paths share one cached instance) and that the reset hook
clears the cache cleanly between tests that swap compression
configuration.
"""

import asyncio

import pytest

pytest.importorskip('numpy')


@pytest.mark.asyncio
async def test_cached_helper_returns_same_instance_on_repeat_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two consecutive get_cached_compression_provider() calls return one instance."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('EMBEDDING_DIM', '128')
    from app.settings import get_settings
    get_settings.cache_clear()

    from app.compression import get_cached_compression_provider
    from app.compression import reset_cached_compression_provider

    reset_cached_compression_provider()
    try:
        p1 = await get_cached_compression_provider()
        p2 = await get_cached_compression_provider()
        assert p1 is p2
    finally:
        reset_cached_compression_provider()
        get_settings.cache_clear()


@pytest.mark.asyncio
async def test_reset_clears_cached_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reset_cached_compression_provider() forces a fresh construction."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('EMBEDDING_DIM', '128')
    from app.settings import get_settings
    get_settings.cache_clear()

    from app.compression import get_cached_compression_provider
    from app.compression import reset_cached_compression_provider

    reset_cached_compression_provider()
    try:
        p1 = await get_cached_compression_provider()
        reset_cached_compression_provider()
        p2 = await get_cached_compression_provider()
        assert p1 is not p2
    finally:
        reset_cached_compression_provider()
        get_settings.cache_clear()


def test_reset_compression_cache_clears_factory_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_reset_compression_cache() in embedding_repository delegates to factory."""
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('EMBEDDING_DIM', '128')
    from app.settings import get_settings
    get_settings.cache_clear()

    from app.compression import get_cached_compression_provider
    from app.compression import reset_cached_compression_provider
    from app.repositories.embedding_repository import _reset_compression_cache

    reset_cached_compression_provider()
    try:
        p1 = asyncio.run(get_cached_compression_provider())
        _reset_compression_cache()
        p2 = asyncio.run(get_cached_compression_provider())
        assert p1 is not p2
    finally:
        reset_cached_compression_provider()
        get_settings.cache_clear()
