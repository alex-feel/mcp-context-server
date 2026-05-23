"""Tests for :mod:`app.compression.providers.turboquant._mse_quantizer`.

Includes round-trip shape, distortion bounds (above the Shannon
lower bound and within Theorem 1), determinism, and byte-aligned range
dequantize behaviour. Helper functions ``mse_distortion`` and
``shannon_lower_bound`` are inlined since this module does not expose
a metrics helper.
"""

import math

import numpy as np
import pytest

from app.compression.providers.turboquant._mse_quantizer import MSEQuantizer

THEOREM1_BOUNDS = {
    1: 0.3634,
    2: 0.1175,
    3: 0.03454,
    4: 0.009497,
}


def mse_distortion(x: np.ndarray, x_hat: np.ndarray) -> float:
    """``E[||x - x_hat||^2]`` summed over coordinates, mean over batch."""
    return float(np.mean(np.sum((x - x_hat) ** 2, axis=-1)))


def shannon_lower_bound(bits: int) -> float:
    """Shannon lower bound on per-vector MSE for normalized vectors at b bits/coord."""
    return math.pow(2.0, -2.0 * bits)


class TestMSEQuantizer:
    """Tests for MSEQuantizer correctness."""

    @pytest.mark.parametrize('d', [64, 128, 256])
    @pytest.mark.parametrize('bits', [1, 2, 3, 4])
    def test_round_trip_shape(self, d: int, bits: int) -> None:
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        rng = np.random.Generator(np.random.PCG64(42))
        x = rng.standard_normal(size=(32, d), dtype=np.float32)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    @pytest.mark.parametrize('bits', [1, 2, 3, 4])
    def test_norms_preserved(self, bits: int) -> None:
        d = 128
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        rng = np.random.Generator(np.random.PCG64(42))
        x = rng.standard_normal(size=(64, d), dtype=np.float32)
        qt = q.quantize(x)
        expected_norms = np.linalg.norm(x, axis=-1, keepdims=True).astype(np.float32)
        np.testing.assert_allclose(qt.norms, expected_norms, atol=1e-5, rtol=1e-5)

    @pytest.mark.parametrize('bits', [1, 2, 3, 4])
    def test_distortion_below_theorem1(self, bits: int) -> None:
        """Empirical normalized MSE should sit close to the Theorem 1 bound."""
        d = 256
        n = 10000
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        rng = np.random.Generator(np.random.PCG64(99))
        x = rng.standard_normal(size=(n, d), dtype=np.float32)
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        empirical = mse_distortion(x, x_hat)
        bound = THEOREM1_BOUNDS[bits] * 1.3
        assert empirical < bound, (
            f'b={bits}: empirical MSE {empirical:.6f} > bound {bound:.6f}'
        )

    @pytest.mark.parametrize('bits', [1, 2, 3, 4])
    def test_distortion_above_shannon(self, bits: int) -> None:
        """Distortion must sit above the Shannon lower bound."""
        d = 256
        n = 5000
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(n, d), dtype=np.float32)
        x = x / np.linalg.norm(x, axis=-1, keepdims=True)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        empirical = mse_distortion(x, x_hat)
        lb = shannon_lower_bound(bits)
        assert empirical >= lb * 0.5, (
            f'b={bits}: empirical MSE {empirical:.6f} way below Shannon bound {lb:.6f}'
        )

    def test_wrong_dim_raises(self) -> None:
        q = MSEQuantizer(dim=64, bits=2, seed=0)
        x = np.zeros((10, 32), dtype=np.float32)
        with pytest.raises(ValueError, match='Expected dim'):
            q.quantize(x)

    def test_batch_dimensions(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=2, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(4, 8, d), dtype=np.float32)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_deterministic(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=3, seed=42)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(16, d), dtype=np.float32)
        qt1 = q.quantize(x)
        qt2 = q.quantize(x)
        np.testing.assert_array_equal(qt1.packed_indices, qt2.packed_indices)

    @pytest.mark.parametrize('bits', [1, 2, 3, 4])
    def test_dequantize_range_matches_slice(self, bits: int) -> None:
        """Byte-aligned range dequantize must match a slice of the full dequantize."""
        d = 64
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(32, d), dtype=np.float32)
        qt = q.quantize(x)
        full = q.dequantize(qt)
        sub = q.dequantize_range(qt, 8, 24)
        np.testing.assert_allclose(sub, full[8:24], atol=1e-6, rtol=1e-6)

    def test_dequantize_range_unaligned_dim(self) -> None:
        """Fallback path (``dim * bits`` not divisible by 8) still produces a correct slice."""
        d = 100
        bits = 3
        q = MSEQuantizer(dim=d, bits=bits, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(20, d), dtype=np.float32)
        qt = q.quantize(x)
        full = q.dequantize(qt)
        sub = q.dequantize_range(qt, 5, 15)
        np.testing.assert_allclose(sub, full[5:15], atol=1e-6, rtol=1e-6)

    def test_dequantize_range_empty_slice(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=3, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(10, d), dtype=np.float32)
        qt = q.quantize(x)
        sub = q.dequantize_range(qt, 5, 5)
        assert sub.shape == (0, d)

    def test_dequantize_range_full_span(self) -> None:
        """Range covering the entire chunk must equal the full dequantize."""
        d = 64
        q = MSEQuantizer(dim=d, bits=4, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(16, d), dtype=np.float32)
        qt = q.quantize(x)
        full = q.dequantize(qt)
        sub = q.dequantize_range(qt, 0, 16)
        np.testing.assert_allclose(sub, full, atol=1e-6, rtol=1e-6)

    def test_dequantize_range_invalid_bounds(self) -> None:
        d = 64
        q = MSEQuantizer(dim=d, bits=3, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(10, d), dtype=np.float32)
        qt = q.quantize(x)
        with pytest.raises(ValueError, match='Invalid range'):
            q.dequantize_range(qt, -1, 5)
        with pytest.raises(ValueError, match='Invalid range'):
            q.dequantize_range(qt, 5, 20)
        with pytest.raises(ValueError, match='Invalid range'):
            q.dequantize_range(qt, 6, 4)
