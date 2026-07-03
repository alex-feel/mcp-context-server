"""Tests for :mod:`app.compression.providers.turboquant._rotation`."""

import numpy as np
import pytest

from app.compression.providers.turboquant._rotation import RandomRotation
from app.compression.providers.turboquant._rotation import random_rotate
from app.compression.providers.turboquant._rotation import random_rotate_inverse


class TestRandomRotation:
    """Tests for :class:`RandomRotation`."""

    @pytest.mark.parametrize('d', [16, 64, 128, 256])
    def test_orthogonality(self, d: int) -> None:
        rot = RandomRotation(dim=d, seed=0)
        eye = np.eye(d, dtype=np.float32)
        product = rot.matrix.T @ rot.matrix
        np.testing.assert_allclose(product, eye, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize('d', [16, 64, 128, 256])
    def test_norm_preservation(self, d: int) -> None:
        rot = RandomRotation(dim=d, seed=42)
        rng = np.random.Generator(np.random.PCG64(99))
        x = rng.standard_normal(size=(32, d), dtype=np.float32)
        y = rot.forward(x)
        norms_x = np.linalg.norm(x, axis=-1)
        norms_y = np.linalg.norm(y, axis=-1)
        np.testing.assert_allclose(norms_x, norms_y, atol=1e-5, rtol=1e-5)

    def test_deterministic_seeding(self) -> None:
        r1 = RandomRotation(dim=64, seed=123)
        r2 = RandomRotation(dim=64, seed=123)
        np.testing.assert_allclose(r1.matrix, r2.matrix)

    def test_different_seeds_differ(self) -> None:
        r1 = RandomRotation(dim=64, seed=0)
        r2 = RandomRotation(dim=64, seed=1)
        assert not np.allclose(r1.matrix, r2.matrix)

    def test_inverse_round_trip(self) -> None:
        rot = RandomRotation(dim=128, seed=7)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(16, 128), dtype=np.float32)
        y = rot.forward(x)
        x_hat = rot.inverse(y)
        np.testing.assert_allclose(x_hat, x, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize('d', [32, 64])
    def test_batch_dimensions(self, d: int) -> None:
        rot = RandomRotation(dim=d, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(4, 8, d), dtype=np.float32)
        y = rot.forward(x)
        assert y.shape == x.shape
        x_hat = rot.inverse(y)
        np.testing.assert_allclose(x_hat, x, atol=1e-5, rtol=1e-5)


class TestFunctionalAPI:
    """Tests for the functional wrappers."""

    def test_round_trip(self) -> None:
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(10, 64), dtype=np.float32)
        y = random_rotate(x, seed=42)
        x_hat = random_rotate_inverse(y, seed=42)
        np.testing.assert_allclose(x_hat, x, atol=1e-5, rtol=1e-5)

    def test_wrong_seed_does_not_recover(self) -> None:
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(10, 64), dtype=np.float32)
        y = random_rotate(x, seed=42)
        x_bad = random_rotate_inverse(y, seed=99)
        assert not np.allclose(x_bad, x, atol=1e-3)
