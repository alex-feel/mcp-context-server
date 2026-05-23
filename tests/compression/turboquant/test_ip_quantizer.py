"""Tests for :mod:`app.compression.providers.turboquant._ip_quantizer`.

The :func:`test_bits_too_low_raises` case verifies the library-enforced
minimum ``bits=2`` guard on :class:`InnerProductQuantizer`.
"""

import numpy as np
import pytest

from app.compression.providers.turboquant._ip_quantizer import InnerProductQuantizer


class TestInnerProductQuantizer:
    """Tests for InnerProductQuantizer."""

    @pytest.mark.parametrize('bits', [2, 3, 4])
    def test_round_trip_shape(self, bits: int) -> None:
        d = 64
        q = InnerProductQuantizer(dim=d, bits=bits, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(16, d), dtype=np.float32)
        qt = q.quantize(x)
        x_hat = q.dequantize(qt)
        assert x_hat.shape == x.shape

    def test_bits_too_low_raises(self) -> None:
        """Library guard: InnerProductQuantizer rejects ``bits < 2``."""
        with pytest.raises(ValueError, match='bits >= 2'):
            InnerProductQuantizer(dim=64, bits=1)

    @pytest.mark.slow
    @pytest.mark.parametrize('bits', [2, 3, 4])
    def test_ip_estimation_unbiased(self, bits: int) -> None:
        """Mean estimated IP should be close to the true IP over many seeds."""
        d = 128
        n_seeds = 200
        rng = np.random.Generator(np.random.PCG64(42))
        x = rng.standard_normal(size=d, dtype=np.float32)
        y = rng.standard_normal(size=d, dtype=np.float32)
        true_ip = float((x * y).sum())

        estimates = []
        for seed in range(n_seeds):
            q = InnerProductQuantizer(dim=d, bits=bits, seed=seed)
            qt = q.quantize(x[np.newaxis, :])
            est = float(q.estimate_inner_product(qt, y[np.newaxis, :])[0])
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        tolerance = abs(true_ip) * 0.25 + 1.0
        assert abs(mean_est - true_ip) < tolerance, (
            f'b={bits}: mean IP est {mean_est:.4f} vs true {true_ip:.4f}'
        )

    def test_dequantize_is_mse_only(self) -> None:
        """Dequantize must return the MSE reconstruction (no QJL residual)."""
        d = 64
        from app.compression.providers.turboquant._mse_quantizer import MSEQuantizer

        ip_q = InnerProductQuantizer(dim=d, bits=3, seed=0)
        mse_q = MSEQuantizer(dim=d, bits=2, seed=0)

        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(8, d), dtype=np.float32)

        qt_ip = ip_q.quantize(x)
        x_hat_ip = ip_q.dequantize(qt_ip)

        qt_mse = mse_q.quantize(x)
        x_hat_mse = mse_q.dequantize(qt_mse)

        np.testing.assert_allclose(x_hat_ip, x_hat_mse, atol=1e-5, rtol=1e-5)

    def test_deterministic(self) -> None:
        d = 64
        q = InnerProductQuantizer(dim=d, bits=3, seed=42)
        rng = np.random.Generator(np.random.PCG64(0))
        x = rng.standard_normal(size=(8, d), dtype=np.float32)
        qt1 = q.quantize(x)
        qt2 = q.quantize(x)
        np.testing.assert_array_equal(qt1.qjl_bits, qt2.qjl_bits)
        np.testing.assert_allclose(qt1.residual_norms, qt2.residual_norms)
