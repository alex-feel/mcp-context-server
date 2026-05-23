"""Regression tests for _reset_compression_cache().

Verifies that calling _reset_compression_cache() invalidates every
@lru_cache-decorated factory inside the TurboQuant subpackage so
configuration swaps between tests do not leak rotation/codebook/
quantizer state.
"""

import pytest

# Skip if compression extra not installed
pytest.importorskip('numpy')

from app.compression.providers.turboquant._codebook import get_codebook
from app.compression.providers.turboquant._qjl import _get_cached_qjl_impl
from app.compression.providers.turboquant._rotation import _get_cached_rotation
from app.compression.providers.turboquant.encoder import _get_ip_quantizer
from app.compression.providers.turboquant.encoder import _get_mse_quantizer
from app.repositories.embedding_repository import _reset_compression_cache


def test_reset_clears_all_five_lru_factories() -> None:
    """Populate every LRU cache then verify _reset_compression_cache empties them."""
    # Populate each cache with one entry
    _get_cached_rotation(dim=128, seed=0)
    _get_cached_qjl_impl(dim=128, m=128, seed=0)
    _get_mse_quantizer(dim=128, bits=4, seed=0)
    _get_ip_quantizer(dim=128, bits=4, seed=0)
    get_codebook(dim=128, bits=4)

    # Sanity: each cache reports a non-zero current size
    assert _get_cached_rotation.cache_info().currsize >= 1
    assert _get_cached_qjl_impl.cache_info().currsize >= 1
    assert _get_mse_quantizer.cache_info().currsize >= 1
    assert _get_ip_quantizer.cache_info().currsize >= 1
    assert get_codebook.cache_info().currsize >= 1

    _reset_compression_cache()

    # All five caches must be empty
    assert _get_cached_rotation.cache_info().currsize == 0
    assert _get_cached_qjl_impl.cache_info().currsize == 0
    assert _get_mse_quantizer.cache_info().currsize == 0
    assert _get_ip_quantizer.cache_info().currsize == 0
    assert get_codebook.cache_info().currsize == 0
