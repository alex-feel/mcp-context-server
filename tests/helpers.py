"""Shared test helper functions.

Provides utility functions used across multiple test infrastructure files
(conftest.py, run_server.py) to avoid code duplication.
"""

from __future__ import annotations


def is_ollama_model_available(
    model: str | None = None,
    host: str | None = None,
) -> bool:
    """Check if an Ollama model is available for testing.

    Performs two checks:
    1. Ollama service is running at the resolved host
    2. The specified model (or any candidate model) is installed

    Args:
        model: Model name to check. If None, checks candidate models
            in priority order: all-minilm, qwen3-embedding:0.6b.
        host: Ollama host URL. If None, resolves from settings
            (default: http://localhost:11434).

    Returns:
        True if the Ollama service is running and the model is available,
        False otherwise.
    """
    try:
        import httpx
        import ollama
    except ImportError:
        return False

    # Resolve host: explicit parameter > settings > hardcoded default
    if host is None:
        try:
            from app.settings import get_settings

            host = get_settings().ollama.host
        except Exception:
            host = 'http://localhost:11434'

    try:
        # Check 1: Service is running (short timeout)
        with httpx.Client(timeout=2.0) as client:
            response = client.get(host)
            if response.status_code != 200:
                return False

        # Check 2: Model is available
        ollama_client = ollama.Client(host=host, timeout=5.0)

        if model is not None:
            ollama_client.show(model)
            return True

        # No model specified -- check candidate models in priority order
        candidate_models = ['all-minilm', 'qwen3-embedding:0.6b']
        for candidate in candidate_models:
            try:
                ollama_client.show(candidate)
                return True
            except Exception:
                continue
        return False

    except Exception:
        return False
