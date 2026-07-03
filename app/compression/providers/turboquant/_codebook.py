"""Lloyd-Max codebooks for TurboQuant scalar quantization.

Ships hardcoded codebooks for ``b`` in ``[1, 4]``. The supported public
range is ``[2, 4]``; ``b=1`` is exposed only because it is consumed
internally by the inner-product quantizer (which calls the MSE
quantizer at ``bits - 1``).

Hardcoded centroid values are the optimal Lloyd-Max centroids for the
standard normal distribution ``N(0, 1)``.
"""

import math
from functools import lru_cache

import numpy as np

from app.compression.providers.turboquant._types import Codebook

# Optimal Lloyd-Max centroids for N(0, 1).
_KNOWN_CENTROIDS: dict[int, list[float]] = {
    1: [-0.7978845608, 0.7978845608],
    2: [-1.5104176088, -0.4527800398, 0.4527800398, 1.5104176088],
    3: [
        -2.1519742685, -1.3439092613, -0.7560052489, -0.2451209529,
        0.2451209529, 0.7560052489, 1.3439092613, 2.1519742685,
    ],
    4: [
        -2.7326368225, -2.0690571770, -1.6180378132, -1.2561836443,
        -0.9423401764, -0.6567588956, -0.3880170670, -0.1284042432,
        0.1284042432, 0.3880170670, 0.6567588956, 0.9423401764,
        1.2561836443, 1.6180378132, 2.0690571770, 2.7326368225,
    ],
}

# Empirical MSE_cost values from arXiv:2504.19874v1 (per-coordinate MSE on N(0,1)).
_KNOWN_MSE_COSTS: dict[int, float] = {
    1: 0.3634,
    2: 0.1175,
    3: 0.03454,
    4: 0.009497,
}

_SUPPORTED_BITS = (1, 2, 3, 4)


def _compute_gaussian_codebook(bits: int) -> Codebook:
    """Return the raw N(0,1) codebook for ``bits`` in ``[1, 4]``.

    Args:
        bits: Bit-width in the supported range.

    Returns:
        Unscaled :class:`Codebook` with N(0,1) centroids.

    Raises:
        ValueError: If ``bits`` is outside the supported range.
    """
    if bits not in _SUPPORTED_BITS:
        raise ValueError(
            f'bits must be in {_SUPPORTED_BITS}; got {bits}.',
        )
    centroids = np.asarray(_KNOWN_CENTROIDS[bits], dtype=np.float32)
    boundaries = (centroids[:-1] + centroids[1:]) / np.float32(2.0)
    return Codebook(
        centroids=centroids,
        boundaries=boundaries,
        bits=bits,
        mse_cost=_KNOWN_MSE_COSTS[bits],
    )


@lru_cache(maxsize=32)
def get_codebook(dim: int, bits: int) -> Codebook:
    """Load (cached) a codebook scaled by ``1/sqrt(dim)`` for the input dim.

    Cache size 32 is bounded by (4 supported bits) * (a handful of
    common dims). Memory per entry is negligible (a few KB).

    Args:
        dim: Vector dimension; must be positive.
        bits: Bit-width in the supported range ``[1, 4]``.

    Returns:
        Scaled :class:`Codebook` whose centroids are pre-divided by
        ``sqrt(dim)``.

    Raises:
        ValueError: If ``bits`` is outside ``[1, 4]`` or ``dim`` is non-positive.
    """
    if bits < 1 or bits > 4:
        raise ValueError(f'bits must be in [1, 4], got {bits}')
    if dim <= 0:
        raise ValueError(f'dim must be positive, got {dim}')

    base = _compute_gaussian_codebook(bits)
    scale = np.float32(1.0 / math.sqrt(dim))
    return Codebook(
        centroids=base.centroids * scale,
        boundaries=base.boundaries * scale,
        bits=bits,
        mse_cost=base.mse_cost / dim,
    )
