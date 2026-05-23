"""MSE-optimal TurboQuant quantizer (Algorithm 1 of arXiv:2504.19874v1).

Performs: normalize -> random rotate -> per-coordinate scalar quantize
-> pack. Dequantize reverses the process.
"""

import numpy as np
from numpy.typing import NDArray

from app.compression.providers.turboquant._codebook import get_codebook
from app.compression.providers.turboquant._packed import pack_indices
from app.compression.providers.turboquant._packed import unpack_indices
from app.compression.providers.turboquant._rotation import get_cached_rotation
from app.compression.providers.turboquant._types import Codebook
from app.compression.providers.turboquant._types import QuantizedMSE


class MSEQuantizer:
    """Data-oblivious MSE-optimal vector quantizer.

    Applies a seeded random rotation then per-coordinate Lloyd-Max
    quantization at the given bit-width. The distortion approaches the
    Shannon lower bound ``1/(d * 4^b)`` for large ``d`` (Theorem 1 of
    the paper).

    Args:
        dim: Vector dimension d.
        bits: Bit-width b (supported range 1-4).
        seed: Deterministic seed for the rotation matrix.
    """

    def __init__(self, dim: int, bits: int, seed: int = 0) -> None:
        self.dim = dim
        self.bits = bits
        self.seed = seed
        self._rotation = get_cached_rotation(dim, seed)
        self._codebook: Codebook = get_codebook(dim, bits)

    @property
    def codebook(self) -> Codebook:
        """The precomputed scaled codebook used for quantization."""
        return self._codebook

    def quantize(self, x: NDArray[np.float32]) -> QuantizedMSE:
        """Quantize vectors using Algorithm 1 (MSE TurboQuant).

        Args:
            x: Input array of shape ``(*, d)``.

        Returns:
            :class:`QuantizedMSE` holding packed indices and norms.

        Raises:
            ValueError: If the trailing dim does not match this quantizer.
        """
        leading = x.shape[:-1]
        d = x.shape[-1]
        if d != self.dim:
            raise ValueError(f'Expected dim={self.dim}, got {d}')

        flat = x.reshape(-1, d).astype(np.float32, copy=False)
        norms = np.linalg.norm(flat, axis=-1, keepdims=True).astype(np.float32)
        safe_norms = np.maximum(norms, np.float32(1e-12))
        normalized = (flat / safe_norms).astype(np.float32)

        rotated = self._rotation.forward(normalized)

        boundaries = self._codebook.boundaries
        indices = np.searchsorted(boundaries, rotated).astype(np.int32)
        indices = np.clip(indices, 0, (1 << self.bits) - 1).astype(np.int32)

        packed = pack_indices(indices.reshape(-1), self.bits)

        return QuantizedMSE(
            packed_indices=packed,
            norms=norms.reshape(*leading, 1),
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
        )

    def quantize_with_reconstruction(
        self, x: NDArray[np.float32],
    ) -> tuple[QuantizedMSE, NDArray[np.float32]]:
        """Quantize and also return the reconstructed vectors.

        Avoids an extra dequantize pass when the caller needs ``x_hat``
        immediately (e.g. inner-product quantization of the residual).

        Args:
            x: Input array of shape ``(*, d)``.

        Returns:
            Tuple ``(qt, x_hat)`` where ``qt`` is the
            :class:`QuantizedMSE` representation and ``x_hat`` is the
            float32 reconstruction with the same shape as ``x``.

        Raises:
            ValueError: If the trailing dim does not match this quantizer.
        """
        leading = x.shape[:-1]
        d = x.shape[-1]
        if d != self.dim:
            raise ValueError(f'Expected dim={self.dim}, got {d}')

        flat = x.reshape(-1, d).astype(np.float32, copy=False)
        norms = np.linalg.norm(flat, axis=-1, keepdims=True).astype(np.float32)
        safe_norms = np.maximum(norms, np.float32(1e-12))
        normalized = (flat / safe_norms).astype(np.float32)

        rotated = self._rotation.forward(normalized)

        boundaries = self._codebook.boundaries
        indices = np.searchsorted(boundaries, rotated).astype(np.int32)
        indices = np.clip(indices, 0, (1 << self.bits) - 1).astype(np.int32)

        centroids = self._codebook.centroids[indices]
        rotated_back = self._rotation.inverse(centroids)
        x_hat = (rotated_back * safe_norms).reshape(*leading, d).astype(np.float32)

        packed = pack_indices(indices.reshape(-1), self.bits)
        qt = QuantizedMSE(
            packed_indices=packed,
            norms=norms.reshape(*leading, 1),
            dim=self.dim,
            bits=self.bits,
            seed=self.seed,
        )
        return qt, x_hat

    def dequantize(self, qt: QuantizedMSE) -> NDArray[np.float32]:
        """Reconstruct vectors from a quantized representation.

        Args:
            qt: Quantized representation from :meth:`quantize`.

        Returns:
            Reconstructed array of shape matching the original input.
        """
        n_vectors = qt.norms.reshape(-1).shape[0]
        count = n_vectors * self.dim

        indices = unpack_indices(qt.packed_indices, qt.bits, count)
        centroids = self._codebook.centroids
        reconstructed = centroids[indices.astype(np.int64)].reshape(n_vectors, self.dim)

        rotated_back = self._rotation.inverse(reconstructed)

        norms_flat = qt.norms.reshape(n_vectors, 1)
        result = (rotated_back * norms_flat).reshape(*qt.norms.shape[:-1], self.dim)
        return result.astype(np.float32)

    def dequantize_range(
        self, qt: QuantizedMSE, start: int, end: int,
    ) -> NDArray[np.float32]:
        """Reconstruct vectors in the index range ``[start, end)``.

        When ``dim * qt.bits`` is a multiple of 8, each vector occupies
        an integer number of bytes in ``qt.packed_indices``, enabling
        cheap byte-aligned slicing so peak fp32 memory during
        reconstruction is bounded by ``(end - start) * dim * 4`` bytes.
        Otherwise this falls back to a full dequantize plus slice.

        Args:
            qt: Quantized representation from :meth:`quantize`.
            start: Inclusive start vector index.
            end: Exclusive end vector index.

        Returns:
            Reconstructed array of shape ``(end - start, dim)``.

        Raises:
            ValueError: If the range is invalid for the chunk size.
        """
        n_total = qt.norms.reshape(-1).shape[0]
        if not (0 <= start <= end <= n_total):
            raise ValueError(
                f'Invalid range [{start}, {end}) for chunk of size {n_total}',
            )
        m = end - start
        if m == 0:
            return np.empty((0, self.dim), dtype=np.float32)

        bits = qt.bits
        total_bits_per_vec = self.dim * bits

        if total_bits_per_vec % 8 != 0:
            full = self.dequantize(qt).reshape(-1, self.dim)
            return full[start:end].astype(np.float32)

        bytes_per_vec = total_bits_per_vec // 8
        packed_sub = qt.packed_indices[
            start * bytes_per_vec: end * bytes_per_vec
        ]
        indices = unpack_indices(packed_sub, bits, m * self.dim)
        centroids = self._codebook.centroids
        reconstructed = centroids[indices.astype(np.int64)].reshape(m, self.dim)

        rotated_back = self._rotation.inverse(reconstructed)
        norms_slice = qt.norms.reshape(-1)[start:end][:, np.newaxis]
        return (rotated_back * norms_slice).astype(np.float32)
