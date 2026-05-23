"""Random orthogonal rotations for TurboQuant.

Uses :class:`numpy.random.Generator` with the PCG64 BitGenerator for
deterministic seeded streams, and :func:`numpy.linalg.qr` for the QR
decomposition that produces a uniform Haar measure over the orthogonal
group ``O(d)``.

A bounded :func:`functools.lru_cache` (size 4) memoizes rotation
matrices keyed by ``(dim, seed)``. Memory math:

* One rotation matrix at dim ``d`` is ``d * d`` float32 = ``4 * d**2``
  bytes.
* At the largest practical embedding dim (``d=3072`` for
  ``text-embedding-3-large``) one entry is approximately 36 MB.
* With ``maxsize=4`` the worst-case peak is approximately 144 MB.

Production deployments are expected to hold one entry (one seed,
one dim); the higher cache ceiling accommodates test scenarios that
sweep multiple ``(dim, seed)`` pairs in quick succession.

``dtype`` is locked to ``np.float32``. If float16 rotations are ever
introduced, the cache key MUST be extended to include dtype to prevent
silent reuse of mismatched rotations.
"""

from functools import lru_cache

import numpy as np
from numpy.typing import NDArray


class RandomRotation:
    """Deterministic random orthogonal rotation seeded by an integer.

    After rotation, each coordinate of a unit vector on ``S^{d-1}``
    follows a Beta distribution (paper Lemma 1), enabling per-
    coordinate scalar quantization.

    Args:
        dim: Vector dimension d.
        seed: Deterministic seed (>= 0).

    Raises:
        ValueError: If ``dim`` is non-positive or ``seed`` is negative.
    """

    def __init__(self, dim: int, seed: int = 0) -> None:
        if dim <= 0:
            raise ValueError(f'dim must be positive, got {dim}')
        if seed < 0:
            raise ValueError(f'seed must be non-negative, got {seed}')
        self.dim = dim
        self.seed = seed
        self._matrix: NDArray[np.float32] = self._generate()

    def _generate(self) -> NDArray[np.float32]:
        rng = np.random.Generator(np.random.PCG64(self.seed))
        gauss = rng.standard_normal(size=(self.dim, self.dim), dtype=np.float32)
        q, r = np.linalg.qr(gauss)
        # Fix sign of R diagonal to give uniform Haar measure over O(d)
        diag_sign = np.sign(np.diag(r)).astype(np.float32)
        diag_sign[diag_sign == 0] = np.float32(1.0)
        q = q * diag_sign[np.newaxis, :]
        return np.ascontiguousarray(q, dtype=np.float32)

    @property
    def matrix(self) -> NDArray[np.float32]:
        """The d x d orthogonal matrix Pi."""
        return self._matrix

    def forward(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply rotation: ``y = x @ Pi.T``.

        Args:
            x: Input array; last axis matches ``dim``.

        Returns:
            Rotated array of the same shape as ``x``.
        """
        return np.ascontiguousarray(x @ self._matrix.T, dtype=np.float32)

    def inverse(self, y: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply inverse rotation: ``x = y @ Pi``.

        Args:
            y: Rotated array; last axis matches ``dim``.

        Returns:
            Inverse-rotated array of the same shape as ``y``.
        """
        return np.ascontiguousarray(y @ self._matrix, dtype=np.float32)


@lru_cache(maxsize=4)
def _get_cached_rotation(dim: int, seed: int) -> RandomRotation:
    """Cached factory keyed by ``(dim, seed)``.

    The cache is process-local and non-invalidatable; swapping seeds
    or dims at runtime requires a process restart.

    Args:
        dim: Vector dimension.
        seed: Deterministic seed.

    Returns:
        Memoized :class:`RandomRotation` instance.
    """
    return RandomRotation(dim=dim, seed=seed)


def get_cached_rotation(dim: int, seed: int) -> RandomRotation:
    """Public wrapper around the cached rotation factory.

    Exposes the memoized :class:`RandomRotation` so that quantizer
    constructors share rotation matrices across encode calls instead of
    paying the QR factorization cost on every instance.

    Args:
        dim: Vector dimension.
        seed: Deterministic seed.

    Returns:
        Memoized :class:`RandomRotation` instance keyed by ``(dim, seed)``.
    """
    return _get_cached_rotation(dim=dim, seed=seed)


def random_rotate(
    x: NDArray[np.float32], seed: int = 0,
) -> NDArray[np.float32]:
    """Functional API: apply a seeded random rotation to ``x``.

    Args:
        x: Input array; last axis defines the rotation dimension.
        seed: Deterministic seed.

    Returns:
        Rotated array.
    """
    rot = _get_cached_rotation(dim=x.shape[-1], seed=seed)
    return rot.forward(x)


def random_rotate_inverse(
    y: NDArray[np.float32], seed: int = 0,
) -> NDArray[np.float32]:
    """Functional API: apply the inverse of a seeded random rotation.

    Args:
        y: Rotated array; last axis defines the rotation dimension.
        seed: Deterministic seed used at rotation time.

    Returns:
        Inverse-rotated array.
    """
    rot = _get_cached_rotation(dim=y.shape[-1], seed=seed)
    return rot.inverse(y)
