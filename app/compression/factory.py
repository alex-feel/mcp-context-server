"""Factory for creating compression provider instances.

Follows the pattern from :mod:`app.reranking.factory` for consistency.
Dynamic imports avoid loading numpy when compression is disabled.

This module also owns the process-wide cached provider singleton so
both the encode (write) path in :mod:`app.tools._shared` and the
search (read) path in :mod:`app.repositories.embedding_repository`
share a single rotation matrix and codebook instance per process.
"""

import asyncio
import importlib
import logging
from typing import TYPE_CHECKING

from app.settings import get_settings

if TYPE_CHECKING:
    from app.compression.base import CompressionProvider

logger = logging.getLogger(__name__)

# Module-level cached provider and an asyncio.Lock initialized at module
# import time. asyncio.Lock() has been parameterless since Python 3.10
# (the loop parameter was removed); construction does NOT require a
# running event loop, so it is safe to bind at module scope. The lock
# serializes first-time provider construction across concurrent
# coroutines so the rotation matrix and codebook are built exactly once
# per process.
_cached_provider: 'CompressionProvider | None' = None
_cached_provider_lock: asyncio.Lock = asyncio.Lock()

PROVIDER_MODULES: dict[str, str] = {
    'turboquant': 'app.compression.providers.turboquant',
}

PROVIDER_CLASSES: dict[str, str] = {
    'turboquant': 'TurboQuantProvider',
}

PROVIDER_INSTALL_INSTRUCTIONS: dict[str, str] = {
    'turboquant': 'uv sync',
}


def create_compression_provider(
    provider: str | None = None,
) -> 'CompressionProvider':
    """Create a compression provider based on configuration.

    Args:
        provider: Override provider selection. If None, uses
            ``settings.compression.provider``.

    Returns:
        Initialized :class:`CompressionProvider` instance.

    Raises:
        ValueError: If provider is not supported.
        ImportError: If required dependencies are not installed (e.g.
            numpy missing when the core dependency set is incomplete).
    """
    if provider is None:
        settings = get_settings()
        provider_name: str = settings.compression.provider
    else:
        provider_name = provider

    if provider_name not in PROVIDER_MODULES:
        supported = ', '.join(PROVIDER_MODULES.keys())
        raise ValueError(
            f"Unsupported compression provider: '{provider_name}'. "
            f'Supported providers: {supported}',
        )

    module_path = PROVIDER_MODULES[provider_name]
    class_name = PROVIDER_CLASSES[provider_name]

    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        logger.debug(f'Creating compression provider: {provider_name}')
        provider_instance: CompressionProvider = provider_class()
        return provider_instance
    except ImportError as e:
        install_cmd = PROVIDER_INSTALL_INSTRUCTIONS[provider_name]
        raise ImportError(
            f"Optional dependencies for '{provider_name}' not installed. "
            f'Install via: {install_cmd}',
        ) from e


async def get_cached_compression_provider() -> 'CompressionProvider':
    """Return the process-wide compression provider, constructing once.

    The provider holds the rotation matrix and codebook arrays, which
    are reused for every encode/decode/search call. Caching avoids
    re-running the deterministic provider constructor on every call site
    and ensures the encode (write) path in :mod:`app.tools._shared` and
    the search (read) path in
    :mod:`app.repositories.embedding_repository` share a single
    rotation matrix and codebook per process.

    Returns:
        The cached :class:`CompressionProvider` instance.
    """
    global _cached_provider
    if _cached_provider is not None:
        return _cached_provider
    async with _cached_provider_lock:
        if _cached_provider is None:
            _cached_provider = create_compression_provider()
    return _cached_provider


def reset_cached_compression_provider() -> None:
    """Clear the cached provider so the next
    :func:`get_cached_compression_provider` call constructs a fresh
    instance.

    Test fixtures that swap compression configuration between cases
    call this to ensure the next invocation observes the new settings.
    The module-level lock is NOT replaced; it remains valid across the
    reset and continues to serialize first-time construction in
    subsequent test cases.
    """
    global _cached_provider
    _cached_provider = None
