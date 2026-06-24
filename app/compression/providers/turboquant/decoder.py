"""Public decode and IP estimation orchestrator for the TurboQuant provider.

Reconstructs QuantizedMSE/QuantizedIP from a discriminated payload
(:class:`MSEPayload` or :class:`IPPayload`) and exposes the dequantize
and estimate_inner_product operations against externally supplied
queries. Variant dispatch happens via ``match payload:`` so additions
to the sum type surface as exhaustiveness warnings at type-check time.
"""

import numpy as np
from numpy.typing import NDArray

from app.compression.providers.turboquant._qjl import get_cached_qjl
from app.compression.providers.turboquant._types import CompressedPayload
from app.compression.providers.turboquant._types import IPPayload
from app.compression.providers.turboquant._types import MSEPayload
from app.compression.providers.turboquant._types import QuantizedMSE
from app.compression.providers.turboquant.encoder import get_mse_quantizer


def decode(payload: CompressedPayload) -> NDArray[np.float32]:
    """Reconstruct float32 vectors from a compressed payload (MSE component).

    For :class:`IPPayload`, the QJL residual is NOT recovered (lossy).
    Use :func:`estimate_inner_product` for IP-preserving search.

    Args:
        payload: Compressed payload from ``encode``.

    Raises:
        TypeError: If ``payload`` is neither :class:`MSEPayload` nor :class:`IPPayload`.

    Returns:
        Reconstructed array of shape ``(n_rows, dim)``.
    """
    match payload:
        case MSEPayload():
            q_mse = get_mse_quantizer(payload.dim, payload.bits, payload.seed)
            qt = QuantizedMSE(
                packed_indices=payload.packed_indices,
                norms=payload.norms.reshape(payload.n_rows, 1),
                dim=payload.dim,
                bits=payload.bits,
                seed=payload.seed,
            )
            return q_mse.dequantize(qt)
        case IPPayload():
            # decode() returns the MSE reconstruction only (the QJL residual is
            # lossy and not recovered here -- use estimate_inner_product for
            # IP-preserving search). Decode the inner MSE component at the
            # bit-width stored EXPLICITLY in the wire format (mse_bits) rather
            # than deriving the codebook from bits-1, so a future encoder that
            # varies mse_bits independently decodes correctly; this mirrors
            # estimate_inner_product, which builds its synthetic MSE payload from
            # mse_bits. Behavior is unchanged while mse_bits == bits-1 (today).
            q_mse = get_mse_quantizer(payload.dim, payload.mse_bits, payload.seed)
            qt = QuantizedMSE(
                packed_indices=payload.packed_indices,
                norms=payload.norms.reshape(payload.n_rows, 1),
                dim=payload.dim,
                bits=payload.mse_bits,
                seed=payload.seed,
            )
            return q_mse.dequantize(qt)
        case _:
            raise TypeError(
                f'Unsupported payload type: {type(payload).__name__}',
            )


def estimate_inner_product(
    payload: CompressedPayload,
    queries: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Estimate <db, query> for each (db_row, query_row) pair.

    For :class:`MSEPayload`, falls back to ``<decode(payload), queries>``
    since the MSE quantizer offers no unbiased IP estimator.
    :class:`IPPayload` uses the paper's unbiased estimator.

    Args:
        payload: Compressed payload with N database rows.
        queries: Query matrix of shape ``(nq, d)``.

    Raises:
        TypeError: If ``payload`` is neither :class:`MSEPayload` nor :class:`IPPayload`.

    Returns:
        Estimate matrix of shape ``(nq, n_rows)``.
    """
    match payload:
        case MSEPayload():
            db = decode(payload)  # (n, d)
            return (queries.astype(np.float32, copy=False) @ db.T).astype(np.float32)
        case IPPayload():
            qjl = get_cached_qjl(payload.dim, payload.dim, payload.seed + 1)
            # MSE contribution: reuse the decoder against a synthetic
            # MSE-only payload constructed from the inner-MSE bit-width
            # stored explicitly in the wire format.
            mse_only_payload = MSEPayload(
                bits=payload.mse_bits,
                dim=payload.dim,
                seed=payload.seed,
                n_rows=payload.n_rows,
                norms=payload.norms,
                packed_indices=payload.packed_indices,
            )
            x_hat = decode(mse_only_payload)  # (n, d)
            mse_ip = queries.astype(np.float32, copy=False) @ x_hat.T  # (nq, n)
            qjl_ip = qjl.estimate_inner_product_batch_queries(
                payload.qjl_bits, queries, payload.residual_norms,
            )  # (nq, n)
            return (mse_ip + qjl_ip).astype(np.float32)
        case _:
            raise TypeError(
                f'Unsupported payload type: {type(payload).__name__}',
            )
