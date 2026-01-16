"""
LangSmith tracing integration for embedding providers.

Provides conditional tracing decorator that integrates with LangSmith
when LANGSMITH_TRACING=true AND langsmith package is installed.

LangSmith requires the optional 'langsmith' dependency group:
    uv sync --extra langsmith

For uvx (combined with embeddings):
    uvx --python 3.12 --with "mcp-context-server[embeddings-ollama,langsmith]" mcp-context-server
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable
from collections.abc import Callable
from functools import wraps
from typing import Any

from app.settings import get_settings

logger = logging.getLogger(__name__)

# Track if we've logged the missing package warning (log once per process)
_warned_missing_package = False


def traced_embedding[**P, R](func: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    """Decorator to trace embedding operations with LangSmith.

    Only applies tracing when:
    1. LANGSMITH_TRACING=true AND
    2. langsmith package is installed

    Uses run_type='embedding' for proper categorization in LangSmith UI.
    Automatically populates ls_provider and ls_model_name from the provider instance
    to enable model identification in LangSmith UI for cost mapping.

    Graceful degradation:
    - If langsmith not installed: logs warning once, returns original function
    - If tracing disabled: returns original function
    - If tracing enabled and langsmith installed: wraps with @traceable decorator

    Args:
        func: Async function to decorate (must be a method on a provider instance)

    Returns:
        Decorated function with LangSmith tracing (if enabled) or original function

    Example:
        @traced_embedding
        async def embed_query(self, text: str) -> list[float]:
            ...
    """
    global _warned_missing_package

    settings = get_settings()
    if not settings.langsmith.tracing:
        return func

    try:
        from langsmith import traceable
    except ImportError:
        # langsmith not installed - log warning once and return original function
        if not _warned_missing_package:
            logger.warning(
                'LangSmith tracing enabled (LANGSMITH_TRACING=true) but langsmith '
                'package not installed. Tracing disabled. Install with: '
                'uv sync --extra langsmith',
            )
            _warned_missing_package = True
        return func

    @wraps(func)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        # Extract model info from provider instance for LangSmith metadata
        metadata: dict[str, Any] = {}

        # Get the provider instance (first positional arg is 'self')
        if args:
            provider_instance = args[0]

            # Extract model info from provider instance using getattr for type safety
            model_name: str | None = None
            provider_name: str | None = None

            # Get provider_name from property
            provider_name = getattr(provider_instance, 'provider_name', None)

            # Get model_name - Azure uses _deployment, others use _model
            model_name = getattr(provider_instance, '_model', None)
            if model_name is None:
                model_name = getattr(provider_instance, '_deployment', None)

            # Build metadata dict with ls_ prefixed fields for LangSmith model identification
            if provider_name:
                metadata['ls_provider'] = provider_name
            if model_name:
                metadata['ls_model_name'] = model_name
                # Also set invocation_params for fallback compatibility
                metadata['ls_invocation_params'] = {'model': model_name}

        # Apply traceable decorator with dynamic metadata at call time
        # This ensures metadata is passed when the run tree is created
        traced_func: Any = traceable(
            name=func.__name__,
            run_type='embedding',
            metadata=metadata or None,
        )(func)

        result: R = await traced_func(*args, **kwargs)
        return result

    return wrapper
