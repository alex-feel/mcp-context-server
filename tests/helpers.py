"""Shared test helper functions.

Provides utility functions used across test infrastructure files
(conftest.py, run_server.py) and test modules to avoid code duplication.
Application modules are imported inside the helpers, apart from the
standard-library-only ``app.access_scope``, so importing this module never
loads the application or reads settings.
"""

import importlib
import os
import pkgutil
import sqlite3
from collections.abc import Generator
from collections.abc import Iterable
from contextlib import AbstractContextManager
from contextlib import contextmanager
from types import ModuleType
from typing import TYPE_CHECKING
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import patch

from pydantic import ValidationError as PydanticValidationError

from app.access_scope import AccessScope

if TYPE_CHECKING:
    import asyncpg
    import pytest
    from fastmcp.exceptions import ValidationError as FastMCPValidationError
    from pydantic_core import ErrorDetails

    from app.backends.base import StorageBackend
    from app.repositories.embedding_repository import EmbeddingRepository
    from app.settings import AppSettings

# The scope of a request without a jwt identity under the default
# ACCESS_CONTROL_DEFAULT_PRINCIPAL ('local'), which carries no groups.
LOCAL_SCOPE = AccessScope('local', frozenset())


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


@contextmanager
def preserve_summary_state() -> Generator[None, None, None]:
    """Restore the summary and embedding providers and reset the summary-model semaphore around a block.

    Captures both providers from ``app.startup`` and resets the summary-model
    semaphore in ``app.tools._generation`` on entry; on exit it restores both
    providers and resets the semaphore again, so a test that installs a provider
    or holds the semaphore leaves no state behind.

    Yields:
        None.
    """
    import app.startup
    import app.tools._generation as generation_module

    original_summary_provider = app.startup.get_summary_provider()
    original_embedding_provider = app.startup.get_embedding_provider()
    generation_module._reset_summary_model_semaphore()

    try:
        yield
    finally:
        app.startup.set_summary_provider(original_summary_provider)
        app.startup.set_embedding_provider(original_embedding_provider)
        generation_module._reset_summary_model_semaphore()


def enable_compression(monkeypatch: 'pytest.MonkeyPatch') -> None:
    """Flip the compression toggle and refresh module-level settings caches."""
    from app.settings import get_settings

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    # COMPRESSION_SEED is required for runtime but not for the migration loader
    # which only inspects the enabled flag.
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


def disable_compression(monkeypatch: 'pytest.MonkeyPatch') -> None:
    """Reset compression toggle to off."""
    from app.settings import get_settings

    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    monkeypatch.delenv('COMPRESSION_SEED', raising=False)
    get_settings.cache_clear()
    import app.migrations.compression as compression_module
    monkeypatch.setattr(compression_module, 'settings', get_settings())


@contextmanager
def as_principal(
    principal_id: str,
    groups: Iterable[str] = (),
    roles: Iterable[str] = (),
) -> Generator[None, None, None]:
    """Run the block as a verified request principal.

    Patches ``app.auth.access.resolve_request_principal``, the lookup behind both
    ``resolve_effective_principal`` and ``resolve_access_scope``, so both return
    this identity inside the block wherever they are called from.

    Args:
        principal_id: The principal id the request resolves to.
        groups: The principal's group ids.
        roles: The principal's roles.

    Yields:
        None.
    """
    from app.auth.principal import RequestPrincipal

    principal = RequestPrincipal(principal_id=principal_id, groups=frozenset(groups), roles=frozenset(roles))
    with patch('app.auth.access.resolve_request_principal', return_value=principal):
        yield


_INSERT_GRANT_COLUMNS = '(context_entry_id, principal_type, principal_id, permission, granted_by)'
_READ_GRANTS_SQL = (
    'SELECT principal_type, principal_id, permission, granted_by FROM context_entry_grants '
    'WHERE context_entry_id = {placeholder} ORDER BY principal_type, principal_id, permission'
)


async def insert_grant(
    backend: 'StorageBackend',
    context_id: str,
    principal_type: str,
    principal_id: str,
    permission: str,
    granted_by: str,
) -> None:
    """Insert one access grant row through raw SQL on either backend.

    The application writes only the group read grants of
    ``ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups``, so tests seed user
    grants and write grants with this helper.

    Args:
        backend: The backend whose database receives the row.
        context_id: ID of the granted context entry.
        principal_type: ``'user'`` or ``'group'``.
        principal_id: The grantee principal or group id.
        permission: ``'read'`` or ``'write'``.
        granted_by: The principal id recorded as the grantor.
    """
    values = (context_id, principal_type, principal_id, permission, granted_by)

    if backend.backend_type == 'sqlite':

        def _insert_sqlite(conn: sqlite3.Connection) -> None:
            conn.execute(f'INSERT INTO context_entry_grants {_INSERT_GRANT_COLUMNS} VALUES (?, ?, ?, ?, ?)', values)

        await backend.execute_write(_insert_sqlite)
        return

    async def _insert_postgresql(conn: 'asyncpg.Connection') -> None:
        await conn.execute(f'INSERT INTO context_entry_grants {_INSERT_GRANT_COLUMNS} VALUES ($1, $2, $3, $4, $5)', *values)

    await backend.execute_write(_insert_postgresql)


async def read_grants(backend: 'StorageBackend', context_id: str) -> list[tuple[str, str, str, str]]:
    """Read the grant rows of one context entry through raw SQL on either backend.

    Args:
        backend: The backend whose database holds the rows.
        context_id: ID of the context entry.

    Returns:
        ``(principal_type, principal_id, permission, granted_by)`` tuples ordered by
        principal type, principal id and permission.
    """
    if backend.backend_type == 'sqlite':

        def _read_sqlite(conn: sqlite3.Connection) -> list[tuple[str, str, str, str]]:
            rows = conn.execute(_READ_GRANTS_SQL.format(placeholder='?'), (context_id,)).fetchall()
            return [(row[0], row[1], row[2], row[3]) for row in rows]

        return await backend.execute_read(_read_sqlite)

    async def _read_postgresql(conn: 'asyncpg.Connection') -> list[tuple[str, str, str, str]]:
        rows = await conn.fetch(_READ_GRANTS_SQL.format(placeholder='$1'), context_id)
        return [(row[0], row[1], row[2], row[3]) for row in rows]

    return await backend.execute_read(_read_postgresql)
