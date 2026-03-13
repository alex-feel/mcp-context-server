"""Summary provider factory with dynamic import.

This module provides the factory function for creating summary providers
based on the SUMMARY_PROVIDER environment variable setting.

The factory uses dynamic imports to avoid loading unused dependencies,
allowing users to install only the provider-specific packages they need.
"""

from __future__ import annotations

import importlib
import logging
from typing import cast

from app.settings import get_settings
from app.summary.base import SummaryProvider

logger = logging.getLogger(__name__)

# Provider module mapping
PROVIDER_MODULES: dict[str, str] = {
    'ollama': 'app.summary.providers.langchain_ollama',
    'openai': 'app.summary.providers.langchain_openai',
    'anthropic': 'app.summary.providers.langchain_anthropic',
}

PROVIDER_CLASSES: dict[str, str] = {
    'ollama': 'OllamaSummaryProvider',
    'openai': 'OpenAISummaryProvider',
    'anthropic': 'AnthropicSummaryProvider',
}

# Provider-specific installation instructions
PROVIDER_INSTALL_INSTRUCTIONS: dict[str, str] = {
    'ollama': 'uv sync --extra summary-ollama',
    'openai': 'uv sync --extra summary-openai',
    'anthropic': 'uv sync --extra summary-anthropic',
}


def create_summary_provider(
    provider: str | None = None,
) -> SummaryProvider:
    """Create summary provider based on configuration.

    Auto-imports the provider module to avoid loading unused dependencies.
    Users must install appropriate optional dependencies for their chosen provider.

    Args:
        provider: Override provider selection. If None, uses SUMMARY_PROVIDER setting.

    Returns:
        Summary provider instance (not yet initialized)

    Raises:
        ImportError: If required optional dependencies not installed
        ValueError: If provider is not supported
    """
    if provider is None:
        settings = get_settings()
        provider_name: str = settings.summary.provider
    else:
        provider_name = provider

    if provider_name not in PROVIDER_MODULES:
        supported = ', '.join(PROVIDER_MODULES.keys())
        raise ValueError(
            f"Unsupported summary provider: '{provider_name}'. "
            f'Supported providers: {supported}',
        )

    # Dynamic import to avoid loading unused dependencies
    module_path = PROVIDER_MODULES[provider_name]
    class_name = PROVIDER_CLASSES[provider_name]

    try:
        module = importlib.import_module(module_path)
        provider_class = getattr(module, class_name)
        logger.debug(f'Creating summary provider: {provider_name}')
        # Cast required because provider_class is dynamically loaded
        return cast(SummaryProvider, provider_class())
    except ImportError as e:
        raise ImportError(
            f"Optional dependencies for '{provider_name}' not installed",
        ) from e
