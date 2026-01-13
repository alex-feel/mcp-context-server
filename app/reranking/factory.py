"""
Factory for creating reranking provider instances.

Follows the pattern from app/embeddings/factory.py for consistency.
"""

from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

from app.settings import get_settings

if TYPE_CHECKING:
    from app.reranking.base import RerankingProvider

logger = logging.getLogger(__name__)

# Provider module mappings
PROVIDER_MODULES: dict[str, str] = {
    'flashrank': 'app.reranking.providers.flashrank',
}

PROVIDER_CLASSES: dict[str, str] = {
    'flashrank': 'FlashRankProvider',
}

PROVIDER_INSTALL_INSTRUCTIONS: dict[str, str] = {
    'flashrank': 'uv sync --extra reranking',
}


def create_reranking_provider() -> RerankingProvider:
    """Create a reranking provider based on settings.

    Reads RERANKING_PROVIDER from settings and dynamically imports
    the appropriate provider class.

    Returns:
        Initialized RerankingProvider instance

    Raises:
        ValueError: If provider is not supported
        ImportError: If provider dependencies are not installed
    """
    settings = get_settings()
    provider_name = settings.reranking.provider.lower()

    if provider_name not in PROVIDER_MODULES:
        supported = ', '.join(PROVIDER_MODULES.keys())
        raise ValueError(
            f"Unsupported reranking provider: '{provider_name}'. "
            f'Supported providers: {supported}',
        )

    module_path = PROVIDER_MODULES[provider_name]
    class_name = PROVIDER_CLASSES[provider_name]
    install_cmd = PROVIDER_INSTALL_INSTRUCTIONS[provider_name]

    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        provider: RerankingProvider = provider_class()

        logger.info(f'Created reranking provider: {provider_name}')
        return provider
    except ImportError as e:
        raise ImportError(
            f"Optional dependencies for '{provider_name}' not installed. "
            f'Install with: {install_cmd}',
        ) from e
