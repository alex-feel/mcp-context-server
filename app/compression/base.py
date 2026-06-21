"""Protocol definition for embedding compression providers.

Mirrors the architecture pattern in :mod:`app.embeddings.base` so that
provider implementations remain interchangeable.

:class:`CompressionProvider` is the only Protocol exported. Provider-
internal helper Protocols live alongside their implementations.
"""

from typing import Protocol
from typing import runtime_checkable

import numpy as np
from numpy.typing import NDArray

from app.compression.types import CompressionMetadata


@runtime_checkable
class CompressionProvider(Protocol):
    """Protocol defining the interface for embedding compression providers.

    All compression providers must implement this protocol. The surface
    includes both synchronous (NumPy hot path) and asynchronous (event-
    loop-friendly) operations.

    Compression is CPU-bound (NumPy GEMM), unlike embedding generation
    which is I/O-bound HTTP. Async wrappers offload work to a worker
    thread via :func:`asyncio.to_thread`; a dedicated concurrency
    semaphore (``COMPRESSION_MAX_CONCURRENT``) may be applied at higher
    layers.
    """

    @property
    def provider_name(self) -> str:
        """Provider identifier (e.g. ``'turboquant'``)."""
        ...

    @property
    def metadata(self) -> CompressionMetadata:
        """Provenance metadata describing this provider's configuration.

        Returned metadata MUST be deterministic for a given
        ``(provider, bits, variant, dim, seed)`` tuple so that startup
        validation can verify env-vs-DB consistency.
        """
        ...

    # Synchronous (hot-path) API ----------------------------------------
    def encode_sync(self, vectors: NDArray[np.float32]) -> bytes:
        """Encode a batch of float32 vectors to a single compressed payload.

        Args:
            vectors: Array of shape ``(n, d)`` with dtype float32.

        Returns:
            Opaque bytes payload. Format is provider-specific; only
            :meth:`decode_sync` of the same provider+config can reverse it.
        """
        ...

    def decode_sync(self, payload: bytes) -> NDArray[np.float32]:
        """Reconstruct float32 vectors from a compressed payload.

        Args:
            payload: Compressed bytes payload from :meth:`encode_sync`.

        Returns:
            Float32 array of reconstructed vectors. The reconstruction
            is LOSSY: decoded vectors approximate the original MSE
            component. For variant ``'ip'``, the QJL residual is NOT
            recoverable here; use :meth:`estimate_inner_product_sync`
            for IP-preserving search.
        """
        ...

    def estimate_inner_product_sync(
        self,
        payload: bytes,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Estimate ``<db, query>`` for each ``(db_row, query_row)`` pair.

        Args:
            payload: Compressed payload from :meth:`encode_sync`
                (n database rows).
            queries: Query matrix of shape ``(nq, d)``.

        Returns:
            Estimate matrix of shape ``(nq, n)``.

        Notes:
            Only meaningful when variant is ``'ip'``. For variant ``'mse'``,
            implementations MAY fall back to
            ``<decode_sync(payload), queries>`` but should document this
            clearly.
        """
        ...

    def warmup(self) -> None:
        """Prime lazy imports, caches, and native thread-pool probes single-threaded.

        Implementations that lazily import native libraries (e.g. NumPy submodules)
        or call thread-pool introspection (e.g. ``threadpoolctl``) MUST make a
        single-threaded warmup safe and idempotent. ``get_cached_compression_provider``
        calls this once before any concurrent ``asyncio.to_thread`` encode/decode
        fan-out, so the first concurrent batch hits only warm caches and cannot
        deadlock on the import-lock vs dynamic-loader-lock cycle.
        """
        ...

    # Asynchronous (event-loop-friendly) wrappers -----------------------
    async def encode(self, vectors: NDArray[np.float32]) -> bytes:
        """Async wrapper around :meth:`encode_sync` via :func:`asyncio.to_thread`.

        Args:
            vectors: Array of shape ``(n, d)`` with dtype float32.

        Returns:
            Opaque bytes payload.
        """
        ...

    async def decode(self, payload: bytes) -> NDArray[np.float32]:
        """Async wrapper around :meth:`decode_sync`.

        Args:
            payload: Compressed bytes payload.

        Returns:
            Float32 array of reconstructed vectors.
        """
        ...

    async def estimate_inner_product(
        self,
        payload: bytes,
        queries: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Async wrapper around :meth:`estimate_inner_product_sync`.

        Offloads the GEMM into a worker thread via
        :func:`asyncio.to_thread` so callers inside an active event loop
        (notably :meth:`EmbeddingRepository.search_compressed`) do not
        block other concurrent MCP requests.

        Args:
            payload: Compressed payload from :meth:`encode_sync`
                (n database rows).
            queries: Query matrix of shape ``(nq, d)``.

        Returns:
            Estimate matrix of shape ``(nq, n)``.
        """
        ...
