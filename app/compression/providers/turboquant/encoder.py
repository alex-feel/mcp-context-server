"""Public encode orchestrator for the TurboQuant provider.

Exposes a single function ``encode`` that wraps either MSEQuantizer or
InnerProductQuantizer based on the requested variant and returns a
serializable discriminated payload (:class:`MSEPayload` or
:class:`IPPayload`).
"""

from functools import lru_cache
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from app.compression.providers.turboquant._ip_quantizer import InnerProductQuantizer
from app.compression.providers.turboquant._mse_quantizer import MSEQuantizer
from app.compression.providers.turboquant._types import CompressedPayload
from app.compression.providers.turboquant._types import IPPayload
from app.compression.providers.turboquant._types import MSEPayload


@lru_cache(maxsize=8)
def _get_mse_quantizer(dim: int, bits: int, seed: int) -> MSEQuantizer:
    """Cached factory keyed by ``(dim, bits, seed)``.

    Reuses the same :class:`MSEQuantizer` across encode/decode calls so
    quantizer construction (rotation + codebook lookup) is amortized.

    Args:
        dim: Vector dimension.
        bits: Bit-width per coordinate.
        seed: Rotation seed.

    Returns:
        Memoized :class:`MSEQuantizer` instance.
    """
    return MSEQuantizer(dim=dim, bits=bits, seed=seed)


@lru_cache(maxsize=8)
def _get_ip_quantizer(dim: int, bits: int, seed: int) -> InnerProductQuantizer:
    """Cached factory keyed by ``(dim, bits, seed)``.

    Reuses the same :class:`InnerProductQuantizer` across encode/decode
    calls so quantizer construction (MSE rotation + QJL projection) is
    amortized.

    Args:
        dim: Vector dimension.
        bits: Total bit budget per coordinate.
        seed: Rotation seed.

    Returns:
        Memoized :class:`InnerProductQuantizer` instance.
    """
    return InnerProductQuantizer(dim=dim, bits=bits, seed=seed)


def get_mse_quantizer(dim: int, bits: int, seed: int) -> MSEQuantizer:
    """Public wrapper around the cached :class:`MSEQuantizer` factory.

    Args:
        dim: Vector dimension.
        bits: Bit-width per coordinate.
        seed: Rotation seed.

    Returns:
        Memoized :class:`MSEQuantizer` instance.
    """
    return _get_mse_quantizer(dim=dim, bits=bits, seed=seed)


def get_ip_quantizer(dim: int, bits: int, seed: int) -> InnerProductQuantizer:
    """Public wrapper around the cached :class:`InnerProductQuantizer` factory.

    Args:
        dim: Vector dimension.
        bits: Total bit budget per coordinate.
        seed: Rotation seed.

    Returns:
        Memoized :class:`InnerProductQuantizer` instance.
    """
    return _get_ip_quantizer(dim=dim, bits=bits, seed=seed)


def encode(
    vectors: NDArray[np.float32],
    *,
    bits: int,
    variant: Literal['mse', 'ip'],
    seed: int,
) -> CompressedPayload:
    """Encode a batch of vectors using the requested variant.

    Args:
        vectors: Float32 array of shape (n, d).
        bits: Bit budget per coordinate. For variant='ip', bits must
            be >= 2 (enforced by InnerProductQuantizer).
        variant: 'mse' (Algorithm 1) or 'ip' (Algorithm 2 with QJL).
        seed: Rotation seed.

    Returns:
        CompressedPayload ready for to_bytes() serialization.

    Raises:
        ValueError: If vectors are not 2-D or variant is unrecognized.
    """
    if vectors.ndim != 2:
        raise ValueError(f'vectors must be 2-D (n, d), got shape {vectors.shape}')
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)
    n, dim = vectors.shape

    if variant == 'mse':
        q_mse = _get_mse_quantizer(dim, bits, seed)
        qt_mse = q_mse.quantize(vectors)
        return MSEPayload(
            bits=bits,
            dim=dim,
            seed=seed,
            n_rows=n,
            norms=qt_mse.norms.reshape(n).astype(np.float32),
            packed_indices=qt_mse.packed_indices,
        )
    if variant == 'ip':
        q_ip = _get_ip_quantizer(dim, bits, seed)
        qt_ip = q_ip.quantize(vectors)
        # Ensure qjl_bits has shape (n, n_bytes) for serialization
        qjl_2d = qt_ip.qjl_bits
        if qjl_2d.ndim == 1:
            qjl_2d = qjl_2d.reshape(n, -1)
        return IPPayload(
            bits=bits,
            mse_bits=bits - 1,
            dim=dim,
            seed=seed,
            n_rows=n,
            norms=qt_ip.mse_data.norms.reshape(n).astype(np.float32),
            packed_indices=qt_ip.mse_data.packed_indices,
            qjl_bits=qjl_2d,
            residual_norms=qt_ip.residual_norms.reshape(n).astype(np.float32),
        )
    raise ValueError(f"variant must be 'mse' or 'ip', got {variant!r}")
