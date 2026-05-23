"""Tests for the top-level encoder/decoder orchestrators.

Exercises :func:`encoder.encode` together with :func:`decoder.decode`
and :func:`decoder.estimate_inner_product`. The cache test verifies
that the rotation cache keys on ``(dim, seed)``.
"""

import numpy as np

from app.compression.providers.turboquant import decoder
from app.compression.providers.turboquant import encoder


class TestFunctionalAPI:
    """Smoke tests for the encoder/decoder orchestrators."""

    def test_mse_round_trip(self) -> None:
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(16, 64), dtype=np.float32)
        payload = encoder.encode(x, bits=3, variant='mse', seed=0)
        x_hat = decoder.decode(payload)
        assert x_hat.shape == x.shape
        cos = np.sum(x * x_hat, axis=-1) / (
            np.linalg.norm(x, axis=-1) * np.linalg.norm(x_hat, axis=-1) + 1e-12
        )
        assert cos.mean() > 0.9

    def test_ip_round_trip(self) -> None:
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(8, 64), dtype=np.float32)
        payload = encoder.encode(x, bits=3, variant='ip', seed=0)
        x_hat = decoder.decode(payload)
        assert x_hat.shape == x.shape

    def test_estimate_ip(self) -> None:
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(8, 64), dtype=np.float32)
        y = rng.standard_normal(size=(8, 64), dtype=np.float32)
        payload = encoder.encode(x, bits=4, variant='ip', seed=0)
        # decoder.estimate_inner_product returns (nq, n) for query rows.
        est_matrix = decoder.estimate_inner_product(payload, y)
        assert est_matrix.shape == (8, 8)
        true_ip = (x * y).sum(axis=-1)
        est_diag = np.diag(est_matrix)
        assert (np.abs(est_diag - true_ip) < np.abs(true_ip) * 2.0 + 5.0).all()

    def test_functional_cache_keyed_by_dim_and_seed(self) -> None:
        """Rotation cache MUST key on ``(dim, seed)``."""
        from app.compression.providers.turboquant._rotation import _get_cached_rotation

        _get_cached_rotation.cache_clear()
        rot_a = _get_cached_rotation(dim=64, seed=0)
        rot_b = _get_cached_rotation(dim=64, seed=0)
        assert rot_a is rot_b, 'Same (dim, seed) must hit the cache'

        rot_c = _get_cached_rotation(dim=128, seed=0)
        assert rot_c is not rot_a, 'Different dim must miss'

        rot_d = _get_cached_rotation(dim=64, seed=1)
        assert rot_d is not rot_a, 'Different seed must miss'
