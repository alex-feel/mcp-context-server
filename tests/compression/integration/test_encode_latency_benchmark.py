"""Encode-latency benchmark for the TurboQuant compression provider.

Measures per-call encode latency across a full ``(dim, bits, variant)``
matrix and asserts a hard gate on the ``(d=1024, b=4, variant='ip')``
cell.

Matrix:
    dims: [768, 1024, 1536, 2048, 3072]
    variants: ['ip', 'mse']
    bits: [2, 3, 4]
    iterations: >= 1000 per cell

Output: each measurement prints ``(dim, bits, variant, mean_us, p50_us,
p95_us, p99_us)``. The gate assertion runs ONLY on the
``(d=1024, bits=4, variant='ip')`` cell. Other cells are informational.

Gate thresholds (microseconds per encode at the gate cell):
    * PASS  <=  500 us
    * WARN  <=  700 us  (test fails so the regression is visible)
    * FAIL  >   700 us
"""

import statistics
import time
from typing import Literal

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from app.compression.providers.turboquant.encoder import encode

ITERATIONS = 1000
DIMS = [768, 1024, 1536, 2048, 3072]
VARIANTS: list[Literal['mse', 'ip']] = ['mse', 'ip']
BITS_VALUES = [2, 3, 4]

# Gate thresholds in microseconds (1 us = 1e-6 s)
GATE_PASS_US = 500.0
GATE_WARN_US = 700.0
GATE_CELL: tuple[int, int, Literal['mse', 'ip']] = (1024, 4, 'ip')

# RNG seed for reproducible benchmark inputs
RNG_SEED = 31415


def _measure_encode(
    dim: int, bits: int, variant: Literal['mse', 'ip'],
) -> dict[str, float]:
    """Return per-iteration latency statistics (microseconds).

    Args:
        dim: Vector dimension.
        bits: Bit-width per coordinate.
        variant: Compression variant.

    Returns:
        Mapping with ``mean``, ``median``, ``p95``, ``p99`` keys
        expressed in microseconds.
    """
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    # Pre-generate input vectors so the timing window excludes RNG.
    vectors_pool = rng.standard_normal(size=(ITERATIONS, dim), dtype=np.float32)
    timings: list[float] = []
    # Pin BLAS threads for reproducible per-call latency on hosts with many
    # logical CPUs, where default thread counts cause oversubscription on
    # the n=1 sequential workload measured here.
    with threadpool_limits(limits=2, user_api='blas'):
        # Warm up to populate codebook and rotation caches.
        for _ in range(10):
            encode(vectors_pool[0:1], bits=bits, variant=variant, seed=0)
        for i in range(ITERATIONS):
            v = vectors_pool[i:i + 1]
            start = time.perf_counter()
            encode(v, bits=bits, variant=variant, seed=0)
            elapsed_us = (time.perf_counter() - start) * 1e6
            timings.append(elapsed_us)
    sorted_t = sorted(timings)
    return {
        'mean': statistics.mean(timings),
        'median': statistics.median(timings),
        'p95': sorted_t[int(0.95 * len(sorted_t))],
        'p99': sorted_t[int(0.99 * len(sorted_t))],
    }


@pytest.mark.performance
def test_encode_latency_full_matrix() -> None:
    """Sweep the full matrix and print results; assert only on the gate cell."""
    results: list[tuple[int, int, Literal['mse', 'ip'], dict[str, float]]] = []
    for dim in DIMS:
        for bits in BITS_VALUES:
            for variant in VARIANTS:
                stats = _measure_encode(dim, bits, variant)
                results.append((dim, bits, variant, stats))

    # Human-readable table for developer review.
    print(f'\nEncode latency benchmark (microseconds, ITERATIONS={ITERATIONS}):')
    print(
        f"{'dim':>5} {'bits':>4} {'variant':>7} {'mean':>10} "
        f"{'p50':>10} {'p95':>10} {'p99':>10}",
    )
    for dim, bits, variant, s in results:
        print(
            f'{dim:>5} {bits:>4} {variant:>7} {s["mean"]:>10.2f} '
            f'{s["median"]:>10.2f} {s["p95"]:>10.2f} {s["p99"]:>10.2f}',
        )

    gate = next(
        s for (d, b, v, s) in results if (d, b, v) == GATE_CELL
    )
    gate_mean = gate['mean']

    if gate_mean <= GATE_PASS_US:
        print(
            f'\nGATE PASS: encode latency {gate_mean:.2f} us '
            f'<= {GATE_PASS_US} us at d=1024 b=4 ip',
        )
        return

    if gate_mean <= GATE_WARN_US:
        pytest.fail(
            f'GATE WARN ({GATE_PASS_US:.0f}-{GATE_WARN_US:.0f} us): encode latency '
            f'{gate_mean:.2f} us at d=1024 b=4 ip exceeds the PASS threshold.',
        )

    pytest.fail(
        f'GATE FAIL (>{GATE_WARN_US:.0f} us): encode latency '
        f'{gate_mean:.2f} us at d=1024 b=4 ip. Investigate the NumPy '
        f'hot path for regressions.',
    )
