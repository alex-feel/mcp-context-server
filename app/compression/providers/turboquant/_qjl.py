"""Quantized Johnson-Lindenstrauss (QJL) transform (Definition 1 of arXiv:2504.19874v1).

Projects vectors via a random Gaussian matrix and takes the sign,
producing 1-bit sketches that allow unbiased inner-product estimation.
"""

import math
from functools import lru_cache
from typing import cast

import numpy as np
from numpy.typing import NDArray

from app.compression.providers.turboquant._packed import pack_bits_batch
from app.compression.providers.turboquant._packed import unpack_bits_batch


@lru_cache(maxsize=4)
def _get_cached_qjl_impl(dim: int, m: int, seed: int) -> 'QJLTransform':
    """Cached :class:`QJLTransform` factory keyed on ``(dim, m, seed)``.

    Memory math: one projection matrix at ``(m, dim)`` float32 occupies
    ``4 * m * dim`` bytes. At ``m == dim == 3072`` that is approximately
    36 MB per entry; with ``maxsize=4`` the worst-case peak is ~144 MB.
    Production deployments hold a single entry per ``(dim, seed)``.

    Args:
        dim: Input dimension d.
        m: Number of projection rows (sketch dimension).
        seed: Deterministic seed for the projection matrix.

    Returns:
        Memoized :class:`QJLTransform` instance.
    """
    return QJLTransform(dim=dim, m=m, seed=seed)


def get_cached_qjl(dim: int, m: int, seed: int) -> 'QJLTransform':
    """Public wrapper around the cached :class:`QJLTransform` factory.

    Args:
        dim: Input dimension d.
        m: Number of projection rows (sketch dimension).
        seed: Deterministic seed for the projection matrix.

    Returns:
        Memoized :class:`QJLTransform` instance keyed by ``(dim, m, seed)``.
    """
    return _get_cached_qjl_impl(dim=dim, m=m, seed=seed)


class QJLTransform:
    """1-bit Quantized Johnson-Lindenstrauss transform.

    Given a random Gaussian matrix ``S`` of shape ``(m, d)``, the QJL
    sketch of a vector ``r`` is ``sign(S @ r)``. The inner product
    ``<x, y>`` can be estimated unbiasedly from the sketches.

    Args:
        dim: Input dimension d.
        m: Number of projection rows (sketch dimension). Defaults to dim.
        seed: Deterministic seed for the projection matrix.
    """

    def __init__(self, dim: int, m: int | None = None, seed: int = 0) -> None:
        self.dim = dim
        self.m = m if m is not None else dim
        self.seed = seed
        self._S = self._generate()

    def _generate(self) -> NDArray[np.float32]:
        rng = np.random.Generator(np.random.PCG64(self.seed))
        return rng.standard_normal(size=(self.m, self.dim), dtype=np.float32)

    @property
    def projection_matrix(self) -> NDArray[np.float32]:
        """The ``(m, d)`` random Gaussian projection matrix ``S``."""
        return self._S

    def quantize(self, r: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Compute the QJL sketch: packed sign bits of ``S @ r``.

        Args:
            r: Residual vectors of shape ``(*, d)``.

        Returns:
            Packed uint8 array of sign bits, leading shape preserved.
        """
        leading = r.shape[:-1]
        flat = r.reshape(-1, self.dim).astype(np.float32, copy=False)  # (n, d)
        proj = flat @ self._S.T  # (n, m)
        signs = (proj >= 0).astype(np.uint8)
        packed = pack_bits_batch(signs)
        n_bytes = packed.shape[-1]
        return packed.reshape(*leading, n_bytes)

    def estimate_inner_product(
        self,
        packed_z: NDArray[np.uint8],
        y: NDArray[np.float32],
        residual_norms: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Estimate ``<r, y>`` from the QJL sketch.

        Uses the unbiased estimator
        ``<r, y>_est = sqrt(pi/2) / m * ||r|| * sum_j z_j * (S_j . y)``
        where ``z_j`` in ``{-1, +1}`` are the unpacked sign bits.

        Args:
            packed_z: Packed sign bits from :meth:`quantize`, shape
                ``(*, n_bytes)``.
            y: Query vectors of shape ``(*, d)``.
            residual_norms: L2 norms ``||r||``, shape ``(*,)``.

        Returns:
            Estimated inner products, shape ``(*,)``.
        """
        leading = packed_z.shape[:-1]
        flat_z = packed_z.reshape(-1, packed_z.shape[-1])
        flat_y = y.reshape(-1, self.dim).astype(np.float32, copy=False)
        flat_norms = residual_norms.reshape(-1).astype(np.float32, copy=False)

        sy = flat_y @ self._S.T  # (n, m)
        signs = unpack_bits_batch(flat_z, self.m)
        scale = np.float32(math.sqrt(math.pi / 2.0) / self.m)
        ip = (signs * sy).sum(axis=-1)
        result = (scale * flat_norms * ip).reshape(leading).astype(np.float32)
        return cast('NDArray[np.float32]', result)

    def estimate_inner_product_batch(
        self,
        packed_z: NDArray[np.uint8],
        y: NDArray[np.float32],
        residual_norms: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Vectorized batch estimation (one query against n database entries).

        Args:
            packed_z: Packed sign bits, shape ``(n, n_bytes)``.
            y: Single query vector of shape ``(d,)``.
            residual_norms: Norms of shape ``(n,)``.

        Returns:
            Estimated inner products, shape ``(n,)``.
        """
        signs = unpack_bits_batch(packed_z, self.m)
        sy = y.astype(np.float32, copy=False) @ self._S.T  # (m,)
        scale = np.float32(math.sqrt(math.pi / 2.0) / self.m)
        ip = (signs * sy[np.newaxis, :]).sum(axis=-1)
        result = (scale * residual_norms.astype(np.float32, copy=False) * ip).astype(np.float32)
        return cast('NDArray[np.float32]', result)

    def estimate_inner_product_batch_queries(
        self,
        packed_z: NDArray[np.uint8],
        queries: NDArray[np.float32],
        residual_norms: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """All-queries variant: ``nq`` queries against ``n`` database rows.

        Args:
            packed_z: Packed sign bits, shape ``(n, n_bytes)``.
            queries: Query matrix, shape ``(nq, d)``.
            residual_norms: Norms, shape ``(n,)``.

        Returns:
            Array of shape ``(nq, n)`` with ``[j, i]`` the estimate for
            DB row ``i`` vs query ``j``.
        """
        signs = unpack_bits_batch(packed_z, self.m)  # (n, m)
        sy = queries.astype(np.float32, copy=False) @ self._S.T  # (nq, m)
        scale = np.float32(math.sqrt(math.pi / 2.0) / self.m)
        ip = signs @ sy.T  # (n, nq)
        return (scale * residual_norms[np.newaxis, :].astype(np.float32) * ip.T).astype(np.float32)
