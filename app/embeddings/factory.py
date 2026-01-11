"""
Embedding provider factory with dynamic import.

This module provides the factory function for creating embedding providers
based on the EMBEDDING_PROVIDER environment variable setting.

The factory uses dynamic imports to avoid loading unused dependencies,
allowing users to install only the provider-specific packages they need.
"""

from __future__ import annotations

import importlib
import logging
from typing import cast

from app.embeddings.base import EmbeddingProvider
from app.settings import get_settings

logger = logging.getLogger(__name__)

# Provider module mapping
PROVIDER_MODULES = {
    'ollama': 'app.embeddings.providers.langchain_ollama',
    'openai': 'app.embeddings.providers.langchain_openai',
    'azure': 'app.embeddings.providers.langchain_azure',
    'huggingface': 'app.embeddings.providers.langchain_huggingface',
    'voyage': 'app.embeddings.providers.langchain_voyage',
}

PROVIDER_CLASSES = {
    'ollama': 'OllamaEmbeddingProvider',
    'openai': 'OpenAIEmbeddingProvider',
    'azure': 'AzureEmbeddingProvider',
    'huggingface': 'HuggingFaceEmbeddingProvider',
    'voyage': 'VoyageEmbeddingProvider',
}

# Provider-specific installation instructions
PROVIDER_INSTALL_INSTRUCTIONS = {
    'ollama': 'uv sync --extra embeddings-ollama',
    'openai': 'uv sync --extra embeddings-openai',
    'azure': 'uv sync --extra embeddings-azure',
    'huggingface': 'uv sync --extra embeddings-huggingface',
    'voyage': 'uv sync --extra embeddings-voyage',
}


def create_embedding_provider(
    provider: str | None = None,
) -> EmbeddingProvider:
    """Create embedding provider based on configuration.

    Auto-imports the provider module to avoid loading unused dependencies.
    Users must install appropriate optional dependencies for their chosen provider.

    Args:
        provider: Override provider selection. If None, uses EMBEDDING_PROVIDER setting.

    Returns:
        Initialized embedding provider instance

    Raises:
        ImportError: If required optional dependencies not installed
        ValueError: If provider is not supported

    Example:
        # Using default provider from settings
        provider = create_embedding_provider()

        # Override provider selection
        provider = create_embedding_provider('openai')
    """
    if provider is None:
        settings = get_settings()
        # Get provider from settings.embedding.provider if available (Phase 3+),
        # otherwise default to 'ollama' for backward compatibility
        embedding_settings = getattr(settings, 'embedding', None)
        if embedding_settings is not None:
            provider_name: str = getattr(embedding_settings, 'provider', 'ollama')
        else:
            provider_name = 'ollama'
    else:
        provider_name = provider

    if provider_name not in PROVIDER_MODULES:
        supported = ', '.join(PROVIDER_MODULES.keys())
        raise ValueError(
            f"Unsupported embedding provider: '{provider_name}'. "
            f'Supported providers: {supported}',
        )

    # Dynamic import to avoid loading unused dependencies
    module_path = PROVIDER_MODULES[provider_name]
    class_name = PROVIDER_CLASSES[provider_name]
    install_cmd = PROVIDER_INSTALL_INSTRUCTIONS[provider_name]

    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        logger.debug(f'Creating embedding provider: {provider_name}')
        # Cast required because provider_class is dynamically loaded
        return cast(EmbeddingProvider, provider_class())
    except ImportError as e:
        raise ImportError(
            f"Optional dependencies for '{provider_name}' not installed. "
            f'Install with: {install_cmd}',
        ) from e
