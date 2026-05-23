"""Encode-latency benchmark for the TurboQuant compression provider.

Measures per-row encode latency across a full ``(dim, bits, variant)``
matrix using batched encodes (n=64 per call) that match the bulk-write
pattern of CLI migrate-compress and chunked-document ingestion. Asserts
a hard gate on the ``(d=1024, b=4, variant='ip')`` cell.

Matrix:
    dims: [768, 1024, 1536, 2048, 3072]
    variants: ['ip', 'mse']
    bits: [2, 3, 4]
    iterations: 100 batches x BATCH_SIZE rows per cell
        (effective per-cell encode count: 6400 rows)

Output: each measurement prints ``(dim, bits, variant, mean_us, p50_us,
p95_us, p99_us)`` PER ROW. The gate assertion runs ONLY on the
``(d=1024, bits=4, variant='ip')`` cell. Other cells are informational.

Gate thresholds (microseconds per row at the gate cell):
    * PASS  <=  500 us
    * WARN  <=  700 us  (test fails so the regression is visible)
    * FAIL  >   700 us

Methodology rationale: n=1 encodes are dominated by BLAS launch overhead
on hosts with non-trivial CPU jitter. n=64 batched encodes match the
production chunked-write pattern (CLI migrate-compress and large-document
ingestion both encode many rows together) and produce per-row
measurements with substantially lower variance.
"""

import statistics
import time
from typing import Literal

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from app.compression.providers.turboquant.encoder import encode

ITERATIONS = 100
BATCH_SIZE = 64
DIMS = [768, 1024, 1536, 2048, 3072]
VARIANTS: list[Literal['mse', 'ip']] = ['mse', 'ip']
BITS_VALUES = [2, 3, 4]

# Gate thresholds in microseconds per row (1 us = 1e-6 s)
GATE_PASS_US = 500.0
GATE_WARN_US = 700.0
GATE_CELL: tuple[int, int, Literal['mse', 'ip']] = (1024, 4, 'ip')

# RNG seed for reproducible benchmark inputs
RNG_SEED = 31415


def _measure_encode(
    dim: int, bits: int, variant: Literal['mse', 'ip'],
) -> dict[str, float]:
    """Return per-ROW latency statistics (microseconds) using n=64 batches.

    Each timing iteration encodes a ``(BATCH_SIZE, dim)`` batch in a
    single ``encode()`` call; the elapsed time is divided by
    ``BATCH_SIZE`` to yield a per-row figure. This matches the
    production chunked-write pattern (CLI migrate-compress and
    large-document ingestion both encode many rows together).

    Args:
        dim: Vector dimension.
        bits: Bit-width per coordinate.
        variant: Compression variant.

    Returns:
        Mapping with ``mean``, ``median``, ``p95``, ``p99`` keys
        expressed in microseconds PER ROW.
    """
    rng = np.random.Generator(np.random.PCG64(RNG_SEED))
    # Pre-generate input batches so the timing window excludes RNG.
    # Total rows pre-allocated = ITERATIONS * BATCH_SIZE.
    vectors_pool = rng.standard_normal(
        size=(ITERATIONS * BATCH_SIZE, dim), dtype=np.float32,
    )
    timings_per_row: list[float] = []
    # Pin BLAS threads for reproducible per-call latency on hosts with many
    # logical CPUs. Mirrors the BLAS thread pin now applied inside
    # TurboQuantProvider.{encode_sync,decode_sync,estimate_inner_product_sync}
    # so the benchmark measures behavior representative of production.
    with threadpool_limits(limits=2, user_api='blas'):
        # Warm up to populate codebook and rotation caches.
        for _ in range(5):
            encode(vectors_pool[0:BATCH_SIZE], bits=bits, variant=variant, seed=0)
        for i in range(ITERATIONS):
            batch = vectors_pool[i * BATCH_SIZE : (i + 1) * BATCH_SIZE]
            start = time.perf_counter()
            encode(batch, bits=bits, variant=variant, seed=0)
            elapsed_us = (time.perf_counter() - start) * 1e6
            timings_per_row.append(elapsed_us / BATCH_SIZE)
    sorted_t = sorted(timings_per_row)
    return {
        'mean': statistics.mean(timings_per_row),
        'median': statistics.median(timings_per_row),
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
    print(
        f'\nEncode latency benchmark (microseconds PER ROW, '
        f'ITERATIONS={ITERATIONS} batches of {BATCH_SIZE} rows = '
        f'{ITERATIONS * BATCH_SIZE} effective encodes per cell):',
    )
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
            f'\nGATE PASS: encode latency {gate_mean:.2f} us/row '
            f'<= {GATE_PASS_US} us at d=1024 b=4 ip (n={BATCH_SIZE} batched)',
        )
        return

    if gate_mean <= GATE_WARN_US:
        pytest.fail(
            f'GATE WARN ({GATE_PASS_US:.0f}-{GATE_WARN_US:.0f} us): encode latency '
            f'{gate_mean:.2f} us/row at d=1024 b=4 ip (n={BATCH_SIZE} batched) '
            f'exceeds the PASS threshold.',
        )

    pytest.fail(
        f'GATE FAIL (>{GATE_WARN_US:.0f} us): encode latency '
        f'{gate_mean:.2f} us/row at d=1024 b=4 ip (n={BATCH_SIZE} batched). '
        f'Investigate the NumPy hot path for regressions.',
    )
