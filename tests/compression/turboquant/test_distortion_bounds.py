"""Theorem 1 distortion bounds (arXiv:2504.19874v1, Section 3) for b in [2, 4].

Empirical MSE distortion on unit-normalized vectors must converge to the
paper's published per-coordinate MSE values for the hardcoded ``N(0, 1)``
codebooks at ``b`` in ``{2, 3, 4}``. ``b=1`` is excluded from the
user-facing surface and is not exercised here.

Tolerance is 5% to keep flake probability low at Monte Carlo
``n = 20000`` while remaining strict enough to detect a real correctness
regression.
"""

import numpy as np
import pytest

from app.compression.providers.turboquant._mse_quantizer import MSEQuantizer

PAPER_MSE_VALUES = {2: 0.1175, 3: 0.03454, 4: 0.009497}


@pytest.mark.parametrize('bits', [2, 3, 4])
def test_distortion_within_5pct_of_theorem1(bits: int) -> None:
    d = 256
    n = 20000
    q = MSEQuantizer(dim=d, bits=bits, seed=0)
    rng = np.random.Generator(np.random.PCG64(99))
    x = rng.standard_normal(size=(n, d), dtype=np.float32)
    x = (x / np.linalg.norm(x, axis=-1, keepdims=True)).astype(np.float32)
    qt = q.quantize(x)
    x_hat = q.dequantize(qt)
    empirical = float(np.mean(np.sum((x - x_hat) ** 2, axis=-1)))
    expected = PAPER_MSE_VALUES[bits]
    rel_err = abs(empirical - expected) / expected
    # 5% tolerance accounts for Monte Carlo variance at n=20000.
    assert rel_err < 0.05, (
        f'b={bits}: empirical MSE {empirical:.6f} vs paper {expected:.6f}; '
        f'relative error {rel_err:.4f} exceeds 5%'
    )
