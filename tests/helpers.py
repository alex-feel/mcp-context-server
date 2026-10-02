"""Shared test helper functions.

Provides utility functions used across test infrastructure files
(conftest.py, run_server.py) and test modules to avoid code duplication.
Application modules are imported inside the helpers, so importing this
module never loads the application or reads settings.
"""

import importlib
import os
import pkgutil
from collections.abc import Generator
from contextlib import AbstractContextManager
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

from pydantic import ValidationError as PydanticValidationError

if TYPE_CHECKING:
    import pytest
    from fastmcp.exceptions import ValidationError as FastMCPValidationError
    from pydantic_core import ErrorDetails

    from app.repositories.embedding_repository import EmbeddingRepository
    from app.settings import AppSettings


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


async def store_single_chunk_embedding(
    repo: 'EmbeddingRepository',
    context_id: str,
    embedding: list[float],
    model: str = 'test-model',
) -> None:
    """Store one embedding for a context entry as a single chunk.

    Seeds the embedding tables through ``EmbeddingRepository.store_chunked``
    with one chunk whose start and end offsets are both zero.

    Args:
        repo: Embedding repository to write through.
        context_id: ID of the context entry the embedding belongs to.
        embedding: Embedding vector.
        model: Model identifier recorded in ``embedding_metadata``.
    """
    from app.repositories.embedding_repository.records import ChunkEmbedding

    chunk = ChunkEmbedding(embedding=embedding, start_index=0, end_index=0)
    await repo.store_chunked(context_id, [chunk], model)


@contextmanager
def env_var(key: str, value: str | None) -> Generator[None, None, None]:
    """Context manager for temporarily setting an environment variable."""
    original = os.environ.get(key)
    try:
        if value is not None:
            os.environ[key] = value
        elif key in os.environ:
            del os.environ[key]
        yield
    finally:
        if original is not None:
            os.environ[key] = original
        elif key in os.environ:
            del os.environ[key]


@contextmanager
def env_vars(**kwargs: str | None) -> Generator[None, None, None]:
    """Context manager for temporarily setting multiple environment variables."""
    originals = {key: os.environ.get(key) for key in kwargs}
    try:
        for key, value in kwargs.items():
            if value is not None:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
        yield
    finally:
        for key, original in originals.items():
            if original is not None:
                os.environ[key] = original
            elif key in os.environ:
                del os.environ[key]


def rebind_package_settings(monkeypatch: 'pytest.MonkeyPatch', package: ModuleType, settings: 'AppSettings') -> None:
    """Rebind the module-level ``settings`` of every submodule of a package that binds one.

    Modules bind ``settings = get_settings()`` at import time, so clearing the
    ``get_settings`` cache leaves those bindings on the old object. In a package
    whose submodules each hold their own binding (the storage backends),
    rebinding only some of them leaves the rest silently on stale values.

    Args:
        monkeypatch: Fixture that restores every binding after the test.
        package: The imported package whose submodules are rebound.
        settings: The settings object installed on every binding.
    """
    for info in pkgutil.iter_modules(package.__path__, f'{package.__name__}.'):
        module = importlib.import_module(info.name)
        if 'settings' in vars(module):
            monkeypatch.setattr(module, 'settings', settings)


def patch_database_setup_steps() -> AbstractContextManager[Any]:
    """Neutralize the schema and migration steps that ``prepare_database`` runs.

    Patches them in ``app.startup.database_setup`` with one ``patch.multiple``, so the
    enclosing ``with (...)`` statement of a lifespan test stays under CPython's static
    nested-block limit: each parenthesized context manager is a nested block. A step
    missing here runs against the test's MagicMock backend and raises ``object MagicMock
    can't be used in 'await'``. The compression migration and the compression provenance
    validator are not patched.

    Returns:
        The ``patch.multiple`` context manager neutralizing those steps.
    """
    return patch.multiple(
        'app.startup.database_setup',
        init_database=AsyncMock(),
        handle_metadata_indexes=AsyncMock(),
        guard_compression_disable_over_populated=AsyncMock(),
        apply_semantic_search_migration=AsyncMock(),
        apply_jsonb_merge_patch_migration=AsyncMock(),
        apply_function_search_path_migration=AsyncMock(),
        apply_fts_migration=AsyncMock(),
        apply_chunking_migration=AsyncMock(),
        apply_index_tree_migration=AsyncMock(),
        apply_summary_migration=AsyncMock(),
        apply_content_hash_migration=AsyncMock(),
        apply_version_migration=AsyncMock(),
        apply_access_control_migration=AsyncMock(),
        apply_tag_uniqueness_migration=AsyncMock(),
    )


def argument_errors(exc_info: 'pytest.ExceptionInfo[FastMCPValidationError]') -> 'list[ErrorDetails]':
    """Return the pydantic error details behind a FastMCP argument-validation failure.

    ``Tool.run`` reports a call whose arguments fail schema validation as
    ``fastmcp.exceptions.ValidationError`` chained from the pydantic error.

    Args:
        exc_info: The captured FastMCP validation error.

    Returns:
        The error details of the pydantic ``ValidationError`` the FastMCP error was raised from.
    """
    cause = exc_info.value.__cause__
    assert isinstance(cause, PydanticValidationError), f'expected a pydantic ValidationError cause, got {cause!r}'
    return cause.errors()
