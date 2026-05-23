"""TurboQuant :class:`CompressionProvider` implementation.

Composes the internal encoder/decoder modules and presents both sync
(hot-path) and async (event-loop-friendly) interfaces.

The async wrappers use bare :func:`asyncio.to_thread`; an outer
semaphore (``COMPRESSION_MAX_CONCURRENT``) may be layered on top by
the caller to bound CPU concurrency.
"""

import asyncio
import logging
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from app.compression.providers.turboquant import decoder
from app.compression.providers.turboquant import encoder
from app.compression.providers.turboquant._types import payload_from_bytes
from app.compression.types import CompressionMetadata
from app.settings import get_settings

logger = logging.getLogger(__name__)


class TurboQuantProvider:
    """TurboQuant compression provider.

    Accepts explicit construction parameters; falls back to the app-wide
    settings singleton when a parameter is left at its default ``None``.
    Mixing per-parameter overrides with settings defaults is supported:
    any parameter passed explicitly takes precedence; remaining
    parameters are read from settings.

    The ``seed`` parameter is load-bearing for codebook geometry and the
    rotation matrix. The settings layer always resolves it to a concrete
    non-negative integer (default 0); production deployments validate
    seed stability at the startup-validator layer via
    ``ConfigurationError`` (exit 78) when the env value diverges from
    the persisted ``compression_metadata`` row.

    Args:
        bits: Bit-width per coordinate; supported range is [2, 4]. When
            ``None``, reads ``settings.compression.bits``.
        variant: Compression variant identifier. When ``None``, reads
            ``settings.compression.variant``.
        seed: Rotation-matrix seed. When ``None``, reads
            ``settings.compression.seed``.
        dim: Vector dimension d. When ``None``, reads
            ``settings.embedding.dim``.

    Raises:
        ValueError: If ``variant`` is not in ``{'mse', 'ip'}``.
    """

    def __init__(
        self,
        *,
        bits: int | None = None,
        variant: Literal['mse', 'ip'] | None = None,
        seed: int | None = None,
        dim: int | None = None,
    ) -> None:
        # Resolve each parameter: explicit kwarg wins; settings fallback
        # is consulted only when the kwarg is None.
        settings = get_settings()
        comp = settings.compression
        emb = settings.embedding

        resolved_bits = comp.bits if bits is None else bits
        resolved_variant_raw: str = comp.variant if variant is None else variant
        resolved_seed = comp.seed if seed is None else seed
        resolved_dim = emb.dim if dim is None else dim

        self._variant: Literal['mse', 'ip']
        if resolved_variant_raw == 'mse':
            self._variant = 'mse'
        elif resolved_variant_raw == 'ip':
            self._variant = 'ip'
        else:
            raise ValueError(
                f"variant must be 'mse' or 'ip', got {resolved_variant_raw!r}",
            )

        self._bits = resolved_bits
        self._seed = resolved_seed
        self._dim = resolved_dim

        logger.debug(
            'TurboQuantProvider initialized: bits=%d variant=%s dim=%d seed=%d',
            self._bits, self._variant, self._dim, self._seed,
        )

    @property
    def provider_name(self) -> str:
        """Provider identifier."""
        return 'turboquant'

    @property
    def metadata(self) -> CompressionMetadata:
        """Provenance metadata describing the current configuration."""
        return CompressionMetadata(
            provider='turboquant',
            bits=self._bits,
            variant=self._variant,
            seed=self._seed,
            dim=self._dim,
        )

    def encode_sync(self, vectors: NDArray[np.float32]) -> bytes:
        """Encode vectors and return a serialized payload.

        Args:
            vectors: Float32 array of shape ``(n, d)``.

        Returns:
            Serialized payload bytes.
        """
        payload = encoder.encode(
            vectors,
            bits=self._bits,
            variant=self._variant,
            seed=self._seed,
        )
        return payload.to_bytes()

    def decode_sync(self, payload: bytes) -> NDArray[np.float32]:
        """Deserialize a payload and reconstruct vectors.

        Args:
            payload: Bytes produced by :meth:`encode_sync`.

        Returns:
            Reconstructed float32 vectors.
        """
        decoded_payload = payload_from_bytes(payload)
        return decoder.decode(decoded_payload)

    def estimate_inner_product_sync(
        self,
        payload: bytes,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Estimate inner products between DB rows and queries.

        Args:
            payload: Bytes produced by :meth:`encode_sync`.
            queries: Query matrix of shape ``(nq, d)``.

        Returns:
            Float32 array of shape ``(nq, n_rows)``.
        """
        decoded_payload = payload_from_bytes(payload)
        return decoder.estimate_inner_product(decoded_payload, queries)

    async def encode(self, vectors: NDArray[np.float32]) -> bytes:
        """Async wrapper around :meth:`encode_sync` via :func:`asyncio.to_thread`.

        Args:
            vectors: Float32 array of shape ``(n, d)``.

        Returns:
            Serialized payload bytes.
        """
        return await asyncio.to_thread(self.encode_sync, vectors)

    async def decode(self, payload: bytes) -> NDArray[np.float32]:
        """Async wrapper around :meth:`decode_sync`.

        Args:
            payload: Bytes produced by :meth:`encode_sync`.

        Returns:
            Reconstructed float32 vectors.
        """
        return await asyncio.to_thread(self.decode_sync, payload)
