"""Process-wide cache of the compression provenance row.

The cached row, its initialization lock, the cached getter and the test reset
hook live together in this one module, so every reader, writer and reset of
the cache refers to the same single instance.
"""

import asyncio
from typing import TYPE_CHECKING

from app.backends.base import StorageBackend

if TYPE_CHECKING:
    from app.compression.types import CompressionMetadata


# Module-level cache for the compression provenance metadata only.
# Compression provenance is immutable post-bootstrap (validator-enforced
# invariant), so caching it once per process is safe. The metadata is
# bound to a specific StorageBackend, so its cache must live next to
# the repository that consumes it. The cached compression provider
# itself lives in :mod:`app.compression.factory` so the encode and
# search paths share one singleton. A non-reentrant asyncio.Lock,
# constructed at module import time, serializes first-time concurrent
# callers so they observe the same instance.
_compression_metadata: 'CompressionMetadata | None' = None
_compression_metadata_init_lock: asyncio.Lock = asyncio.Lock()


async def get_cached_compression_metadata(
    backend: StorageBackend,
) -> 'CompressionMetadata':
    """Return the singleton provenance row, caching it on first call.

    Args:
        backend: Storage backend used to read the row on cache miss.

    Returns:
        The cached :class:`CompressionMetadata` row.

    Raises:
        RuntimeError: If the provenance row is missing. The startup
            validator would normally insert it; absence here indicates the
            validator was bypassed or the database is corrupted.
    """
    global _compression_metadata
    if _compression_metadata is not None:
        return _compression_metadata
    async with _compression_metadata_init_lock:
        if _compression_metadata is None:
            from app.compression.provenance import read_compression_metadata
            meta = await read_compression_metadata(backend)
            if meta is None:
                raise RuntimeError(
                    'compression_metadata row is missing; ensure the server '
                    'started with ENABLE_EMBEDDING_COMPRESSION=true so the '
                    'startup validator bootstrapped the provenance row.',
                )
            _compression_metadata = meta
    return _compression_metadata


def _reset_compression_cache() -> None:
    """Clear the module-level metadata cache and inner LRU factories.

    Tests that switch compression configuration between cases call this
    to avoid leaking metadata state across the process. In addition to
    clearing the local metadata singleton, this function:

    1. Delegates the cached-provider reset to
       :func:`app.compression.factory.reset_cached_compression_provider`
       so both the encode (write) path in :mod:`app.tools._generation` and
       the search (read) path in
       :mod:`app.repositories.embedding_repository.compressed_search`
       observe a fresh provider on the next call.
    2. Invalidates every ``@lru_cache``-decorated factory inside the
       TurboQuant subpackage so a follow-up call with different
       ``(dim, bits, seed, variant)`` constructs fresh rotations,
       codebooks, and quantizers.

    The inner caches are imported lazily inside the function body to keep
    numpy out of the import graph for installations that skipped the
    compression extra.
    """
    global _compression_metadata
    _compression_metadata = None

    # Delegate provider-cache reset to the factory module so all
    # consumers share one truth source.
    from app.compression import reset_cached_compression_provider
    reset_cached_compression_provider()

    # Lazy module imports: the compression extra is optional; this
    # function MUST remain importable even when numpy is absent. Module
    # imports (not symbol imports) avoid the type-checker's private-usage
    # warning for the inner ``_get_cached_*`` factories while still
    # giving us access to their ``.cache_clear()`` method via getattr,
    # which is opaque to private-name analysis.
    try:
        from app.compression.providers.turboquant import _codebook as _codebook_mod
        from app.compression.providers.turboquant import _qjl as _qjl_mod
        from app.compression.providers.turboquant import _rotation as _rotation_mod
        from app.compression.providers.turboquant import encoder as _encoder_mod
    except ImportError:
        # Compression extra not installed; nothing inner-cached to clear.
        return

    # Each inner factory exposes the standard functools.lru_cache.cache_clear
    # callable; reflective access avoids type-checker noise around the
    # leading-underscore naming of the encoder-internal factories.
    for module, attr_name in (
        (_rotation_mod, '_get_cached_rotation'),
        (_qjl_mod, '_get_cached_qjl_impl'),
        (_encoder_mod, '_get_mse_quantizer'),
        (_encoder_mod, '_get_ip_quantizer'),
        (_codebook_mod, 'get_codebook'),
    ):
        factory = getattr(module, attr_name, None)
        if factory is not None and hasattr(factory, 'cache_clear'):
            factory.cache_clear()


__all__ = ['_reset_compression_cache', 'get_cached_compression_metadata']
