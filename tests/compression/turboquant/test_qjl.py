"""Tests for :mod:`app.compression.providers.turboquant._qjl`.

Two Monte Carlo tests (unbiasedness and variance bound) are marked
``@pytest.mark.slow`` because they run many trials.
"""

import numpy as np
import pytest

from app.compression.providers.turboquant._qjl import QJLTransform


class TestQJLTransform:
    """Tests for QJL sketch and inner-product estimation."""

    def test_quantize_shape(self) -> None:
        d, m = 64, 64
        qjl = QJLTransform(dim=d, m=m, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        r = rng.standard_normal(size=(16, d), dtype=np.float32)
        packed = qjl.quantize(r)
        expected_bytes = (m + 7) // 8
        assert packed.shape == (16, expected_bytes)

    @pytest.mark.slow
    def test_unbiasedness(self) -> None:
        """Estimator mean should converge to the true inner product."""
        d = 128
        n_trials = 50000
        rng = np.random.Generator(np.random.PCG64(42))
        r = rng.standard_normal(size=d, dtype=np.float32)
        r = (r / np.linalg.norm(r)).astype(np.float32)
        y = rng.standard_normal(size=d, dtype=np.float32)
        y = (y / np.linalg.norm(y)).astype(np.float32)

        true_ip = float((r * y).sum())

        estimates = []
        for trial in range(n_trials):
            qjl = QJLTransform(dim=d, seed=trial)
            packed = qjl.quantize(r[np.newaxis, :])
            r_norm = np.array([float(np.linalg.norm(r))], dtype=np.float32)
            est = float(
                qjl.estimate_inner_product(packed, y[np.newaxis, :], r_norm)[0],
            )
            estimates.append(est)

        mean_est = sum(estimates) / len(estimates)
        assert abs(mean_est - true_ip) < 0.05, (
            f'Mean estimate {mean_est:.4f} vs true {true_ip:.4f}'
        )

    @pytest.mark.slow
    def test_variance_bound(self) -> None:
        """Variance of the QJL estimator scales roughly as pi/(2m) * ||r||^2 * ||y||^2."""
        d = 64
        m = 64
        n_trials = 10000
        rng = np.random.Generator(np.random.PCG64(0))
        r = rng.standard_normal(size=d, dtype=np.float32)
        y = rng.standard_normal(size=d, dtype=np.float32)

        estimates = []
        for trial in range(n_trials):
            qjl = QJLTransform(dim=d, m=m, seed=trial)
            packed = qjl.quantize(r[np.newaxis, :])
            r_norm = np.array([float(np.linalg.norm(r))], dtype=np.float32)
            est = float(
                qjl.estimate_inner_product(packed, y[np.newaxis, :], r_norm)[0],
            )
            estimates.append(est)

        arr = np.array(estimates, dtype=np.float64)
        empirical_var = float(arr.var())
        expected_order = (
            float(np.linalg.norm(r)) ** 2
            * float(np.linalg.norm(y)) ** 2
        )
        assert empirical_var < expected_order * 5.0, (
            f'Variance {empirical_var:.4f} too large vs {expected_order:.4f}'
        )

    def test_deterministic(self) -> None:
        d = 32
        qjl = QJLTransform(dim=d, seed=42)
        rng = np.random.Generator(np.random.PCG64(0))
        r = rng.standard_normal(size=(4, d), dtype=np.float32)
        p1 = qjl.quantize(r)
        p2 = qjl.quantize(r)
        np.testing.assert_array_equal(p1, p2)

    def test_batch_estimate(self) -> None:
        d = 64
        n = 16
        qjl = QJLTransform(dim=d, seed=0)
        rng = np.random.Generator(np.random.PCG64(0))
        r = rng.standard_normal(size=(n, d), dtype=np.float32)
        y = rng.standard_normal(size=d, dtype=np.float32)
        norms = np.linalg.norm(r, axis=-1).astype(np.float32)
        safe_norms = np.maximum(norms, np.float32(1e-12))
        r_normalized = (r / safe_norms[:, np.newaxis]).astype(np.float32)
        packed = qjl.quantize(r_normalized)
        y_broadcast = np.broadcast_to(y, (n, d)).astype(np.float32, copy=True)
        est_loop = qjl.estimate_inner_product(packed, y_broadcast, norms)
        est_batch = qjl.estimate_inner_product_batch(packed, y, norms)
        np.testing.assert_allclose(est_loop, est_batch, atol=1e-5, rtol=1e-5)
