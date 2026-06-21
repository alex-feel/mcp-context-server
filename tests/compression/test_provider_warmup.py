"""Regression tests for the compression provider warmup (PG store hang fix).

store_context fans per-chunk encodes across worker threads via asyncio.to_thread.
On a cold process that first concurrent batch deadlocked when NumPy's lazy
submodule import (triggered by rotation-matrix generation) raced threadpoolctl's
dl_iterate_phdr (the dynamic-loader lock taken under the Python import lock),
wedging every concurrent compression. get_cached_compression_provider now warms
the provider single-threaded before returning, so the rotation cache and every
lazy import are primed before any concurrency.
"""

import asyncio

import pytest

pytest.importorskip('numpy')

import numpy as np

from app.compression.factory import get_cached_compression_provider
from app.compression.factory import reset_cached_compression_provider
from app.compression.providers.turboquant._rotation import _get_cached_rotation
from app.compression.providers.turboquant.provider import TurboQuantProvider
from app.repositories.embedding_repository import _reset_compression_cache


def test_warmup_primes_rotation_cache_single_threaded() -> None:
    """warmup() populates the (dim, seed) rotation cache so the later threaded
    encodes never generate it concurrently (the lazy-import deadlock locus)."""
    reset_cached_compression_provider()
    _get_cached_rotation.cache_clear()
    provider = TurboQuantProvider()
    assert _get_cached_rotation.cache_info().currsize == 0
    provider.warmup()
    assert _get_cached_rotation.cache_info().currsize >= 1
    # Idempotent: warming an already-warm provider must not raise.
    provider.warmup()


@pytest.mark.asyncio
async def test_cached_provider_is_warm_on_acquire() -> None:
    """get_cached_compression_provider warms before returning, so the rotation
    cache is already populated when the caller starts its concurrent encodes."""
    reset_cached_compression_provider()
    _reset_compression_cache()  # clears ALL turboquant LRU factories (rotation + quantizers)
    assert _get_cached_rotation.cache_info().currsize == 0
    provider = await get_cached_compression_provider()
    assert _get_cached_rotation.cache_info().currsize >= 1
    assert provider is not None
    reset_cached_compression_provider()


@pytest.mark.asyncio
async def test_concurrent_encodes_complete() -> None:
    """The exact fan-out store_context uses (gather of per-chunk to_thread encodes)
    completes; the cold-start deadlock regression would hang past the timeout."""
    reset_cached_compression_provider()
    provider = await get_cached_compression_provider()
    dim = provider.metadata.dim
    vectors = [np.ones((1, dim), dtype=np.float32) for _ in range(4)]
    results = await asyncio.wait_for(
        asyncio.gather(*[asyncio.to_thread(provider.encode_sync, v) for v in vectors]),
        timeout=30,
    )
    assert len(results) == 4
    assert all(isinstance(payload, bytes) for payload in results)
    reset_cached_compression_provider()
