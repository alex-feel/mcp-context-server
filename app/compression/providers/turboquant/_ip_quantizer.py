"""Inner-product TurboQuant quantizer (Algorithm 2 of arXiv:2504.19874v1).

Two-stage quantization: MSE quantize at ``bits - 1``, then apply 1-bit
QJL to the residual. This enables unbiased inner-product estimation.
"""

from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.compression.providers.turboquant._mse_quantizer import MSEQuantizer
from app.compression.providers.turboquant._qjl import QJLTransform
from app.compression.providers.turboquant._qjl import get_cached_qjl
from app.compression.providers.turboquant._types import QuantizedIP


class InnerProductQuantizer:
    """Two-stage inner-product-preserving quantizer.

    Stage 1: MSE-optimal quantization at ``(bits - 1)`` bits.
    Stage 2: 1-bit QJL transform on the quantization residual.

    The total budget is ``bits`` bits per coordinate: ``(bits - 1)`` for
    MSE plus 1 for QJL.

    Args:
        dim: Vector dimension d.
        bits: Total bit budget per coordinate (must be >= 2).
        seed: Deterministic seed.

    Raises:
        ValueError: If ``bits < 2`` (one bit is reserved for QJL,
            leaving zero for MSE which is not meaningful).
    """

    def __init__(self, dim: int, bits: int = 4, seed: int = 0) -> None:
        if bits < 2:
            raise ValueError(f'InnerProductQuantizer needs bits >= 2, got {bits}')
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self._mse = MSEQuantizer(dim, bits=bits - 1, seed=seed)
        self._qjl = get_cached_qjl(dim, dim, seed + 1)

    @property
    def mse_quantizer(self) -> MSEQuantizer:
        """The inner MSE quantizer (at ``bits - 1`` bits)."""
        return self._mse

    @property
    def qjl_transform(self) -> QJLTransform:
        """The QJL transform applied to MSE residuals."""
        return self._qjl

    def quantize(self, x: NDArray[np.float32]) -> QuantizedIP:
        """Quantize vectors using Algorithm 2 (Inner-Product TurboQuant).

        Args:
            x: Input array of shape ``(*, d)``.

        Returns:
            :class:`QuantizedIP` holding MSE data and QJL sign bits.
        """
        leading = x.shape[:-1]
        flat = x.reshape(-1, self.dim).astype(np.float32, copy=False)

        mse_qt, x_hat = self._mse.quantize_with_reconstruction(flat)

        residual = flat - x_hat
        residual_norms = np.linalg.norm(residual, axis=-1).astype(np.float32)
        safe_norms = np.maximum(residual_norms, np.float32(1e-12))
        residual_normalized = (residual / safe_norms[:, np.newaxis]).astype(np.float32)

        qjl_bits = self._qjl.quantize(residual_normalized)

        return QuantizedIP(
            mse_data=mse_qt,
            qjl_bits=qjl_bits,
            residual_norms=residual_norms.reshape(leading),
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
        )

    def dequantize(self, qt: QuantizedIP) -> NDArray[np.float32]:
        """Reconstruct vectors (MSE component only; QJL is lossy).

        The QJL bits only aid inner-product estimation and cannot
        reconstruct the residual direction, so this returns the MSE
        reconstruction.

        Args:
            qt: Quantized representation from :meth:`quantize`.

        Returns:
            Reconstructed array of shape matching the original input.
        """
        return self._mse.dequantize(qt.mse_data)

    def dequantize_range(
        self, qt: QuantizedIP, start: int, end: int,
    ) -> NDArray[np.float32]:
        """Reconstruct vectors in ``[start, end)`` from a chunk.

        Thin wrapper around :meth:`MSEQuantizer.dequantize_range`: only
        the MSE component is reconstructed.

        Args:
            qt: Quantized representation from :meth:`quantize`.
            start: Inclusive start vector index.
            end: Exclusive end vector index.

        Returns:
            Reconstructed array of shape ``(end - start, dim)``.
        """
        return self._mse.dequantize_range(qt.mse_data, start, end)

    def estimate_inner_product(
        self, qt: QuantizedIP, y: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Estimate ``<x, y>`` from the quantized representation.

        Uses ``<x, y> ~ <x_hat_mse, y> + QJL_estimate(<r, y>)``.

        Args:
            qt: Quantized data for the database vectors.
            y: Query vectors of shape ``(*, d)``.

        Returns:
            Estimated inner products, shape ``qt.residual_norms.shape``.
        """
        x_hat = self._mse.dequantize(qt.mse_data)
        flat_x_hat = x_hat.reshape(-1, self.dim)
        flat_y = y.reshape(-1, self.dim).astype(np.float32, copy=False)

        mse_ip = (flat_x_hat * flat_y).sum(axis=-1).astype(np.float32)

        qjl_ip = self._qjl.estimate_inner_product(
            qt.qjl_bits.reshape(-1, qt.qjl_bits.shape[-1]),
            flat_y,
            qt.residual_norms.reshape(-1),
        )

        combined = (mse_ip + qjl_ip).reshape(qt.residual_norms.shape).astype(np.float32)
        return cast('NDArray[np.float32]', combined)
