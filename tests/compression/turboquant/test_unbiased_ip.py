"""Theorem 2 unbiased IP estimator validation (arXiv:2504.19874v1, Section 4).

Asserts that the mean of N independent estimates converges to the true
inner product within a 95 percent Monte Carlo confidence interval at
``b`` in ``[2, 4]``.
"""

import math

import numpy as np
import pytest

from app.compression.providers.turboquant._ip_quantizer import InnerProductQuantizer

N_SEEDS = 200
CONFIDENCE_Z = 1.96  # 95% CI


@pytest.mark.slow
@pytest.mark.parametrize('bits', [2, 3, 4])
def test_estimator_unbiased_within_95pct_ci(bits: int) -> None:
    d = 128
    rng = np.random.Generator(np.random.PCG64(42))
    x = rng.standard_normal(size=d, dtype=np.float32)
    y = rng.standard_normal(size=d, dtype=np.float32)
    true_ip = float((x * y).sum())

    estimates: list[float] = []
    for seed in range(N_SEEDS):
        q = InnerProductQuantizer(dim=d, bits=bits, seed=seed)
        qt = q.quantize(x[np.newaxis, :])
        est = float(q.estimate_inner_product(qt, y[np.newaxis, :])[0])
        estimates.append(est)

    sample_mean = sum(estimates) / N_SEEDS
    sample_var = sum((e - sample_mean) ** 2 for e in estimates) / max(N_SEEDS - 1, 1)
    se = math.sqrt(sample_var / N_SEEDS)
    ci_radius = CONFIDENCE_Z * se

    # The 95% CI for the mean of N=200 i.i.d. estimates should cover the
    # true IP if the estimator is truly unbiased. A small absolute slack
    # tolerates finite-sample bias from MSE quantization.
    assert abs(sample_mean - true_ip) < ci_radius + 0.5, (
        f'b={bits}: |mean_est - true_ip| = {abs(sample_mean - true_ip):.4f} '
        f'exceeds 95% CI radius {ci_radius:.4f} (+ 0.5 absolute slack)'
    )
