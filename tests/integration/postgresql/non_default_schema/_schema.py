"""Environment setup and catalog probes for the non-default POSTGRESQL_SCHEMA tests.

``install_search_path_patch`` wraps ``asyncpg.create_pool`` so every backend
connection a test opens resolves bare names through the given schema.
``configure_non_default_env``, ``configure_default_schema_env`` and
``refresh_module_settings`` point the settings singleton and the module-level
``settings`` bindings at the test database and schema.
``table_exists_in_schema``, ``function_exists_in_schema``,
``column_exists_in_schema`` and ``proconfig_for_function`` read the
PostgreSQL catalogs over a direct connection.
"""

from __future__ import annotations

from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast

import asyncpg
import pytest

import app.backends.postgresql_backend as postgresql_backend_module
import app.migrations.chunking as chunking_module
import app.migrations.compression as compression_module
import app.migrations.semantic as semantic_module
import app.startup as startup_module
from app.settings import get_settings
from tests.helpers import rebind_package_settings
from tests.integration.postgresql.conftest import NON_DEFAULT_SCHEMA

_InitCallback = Callable[[asyncpg.Connection], Awaitable[None]]


def install_search_path_patch(
    monkeypatch: pytest.MonkeyPatch, schema: str,
) -> None:
    """Wrap ``asyncpg.create_pool`` so every backend connection sets search_path.

    The wrapper runs ``SET search_path = <schema>, public`` ahead of the
    pool's own ``init=`` and ``setup=`` callbacks, so the statement runs
    when a connection is created and again on every acquire, after the
    pool's ``RESET ALL`` on release. The production ``setup`` callback
    (``setup_pool_connection``) runs after it and applies the configured
    search_path itself; ``test_production_pool_search_path_survives_reset_all``
    covers that production path without this wrapper.

    Args:
        monkeypatch: The pytest monkeypatch fixture.
        schema: The schema name to prepend to ``search_path``. Tests
            running against the default schema pass ``'public'`` to
            ensure the user-implicit ``"$user", public`` default does
            not accidentally route DDL into a same-named schema that
            exists in the test database.
    """
    original_create_pool = asyncpg.create_pool

    async def _set_search_path(conn: asyncpg.Connection) -> None:
        await conn.execute(
            f'SET search_path = {schema}, public',
        )

    def _patched_create_pool(
        *args: object,
        **kwargs: object,
    ) -> object:
        user_init = cast(
            '_InitCallback | None', kwargs.get('init'),
        )
        user_setup = cast(
            '_InitCallback | None', kwargs.get('setup'),
        )

        async def _composed_init(conn: asyncpg.Connection) -> None:
            await _set_search_path(conn)
            if user_init is not None:
                await user_init(conn)

        async def _composed_setup(conn: asyncpg.Connection) -> None:
            # ``setup`` runs after asyncpg's ``reset=`` callback (which
            # issues ``RESET ALL`` and clears any session-scoped
            # ``SET search_path``). Re-apply search_path here so every
            # acquire delivers a connection bound to the test schema.
            await _set_search_path(conn)
            if user_setup is not None:
                await user_setup(conn)

        kwargs['init'] = _composed_init
        kwargs['setup'] = _composed_setup
        # asyncpg.create_pool exposes a long, optional-heavy signature.
        # The wrapper forwards all kwargs through to the original; the
        # cast keeps the return type compatible with the original
        # callable's contract.
        original_fn = cast(
            'Callable[..., object]', original_create_pool,
        )
        return original_fn(*args, **kwargs)

    # PostgreSQLBackend.initialize looks create_pool up on the asyncpg
    # module at call time, so patching the module attribute reaches it.
    monkeypatch.setattr(asyncpg, 'create_pool', _patched_create_pool)


def refresh_module_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refresh module-level ``settings`` bindings against the live env.

    Several modules cache ``settings = get_settings()`` at import time;
    the per-test ``get_settings.cache_clear()`` invalidates the singleton
    but leaves the cached module-level reference in place. Re-binding
    each module's ``settings`` attribute is required for env changes to
    take effect during the test.
    """
    fresh = get_settings()
    monkeypatch.setattr(semantic_module, 'settings', fresh)
    monkeypatch.setattr(chunking_module, 'settings', fresh)
    monkeypatch.setattr(compression_module, 'settings', fresh)
    monkeypatch.setattr(startup_module, 'settings', fresh)
    rebind_package_settings(monkeypatch, postgresql_backend_module, fresh)


def configure_non_default_env(
    monkeypatch: pytest.MonkeyPatch, pg_url: str,
) -> None:
    """Apply the env vars every non-default-schema test needs."""
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_url)
    monkeypatch.setenv('POSTGRESQL_SCHEMA', NON_DEFAULT_SCHEMA)
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    refresh_module_settings(monkeypatch)


def configure_default_schema_env(
    monkeypatch: pytest.MonkeyPatch, pg_url: str,
) -> None:
    """Apply env vars for the default-schema (``public``) baseline test."""
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_url)
    monkeypatch.setenv('POSTGRESQL_SCHEMA', 'public')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    refresh_module_settings(monkeypatch)


async def table_exists_in_schema(
    pg_url: str, schema: str, table: str,
) -> bool:
    """Return True when ``schema.table`` is present."""
    conn = await asyncpg.connect(pg_url)
    try:
        result = await conn.fetchval(
            '''
            SELECT EXISTS (
                SELECT 1 FROM information_schema.tables
                WHERE table_schema = $1 AND table_name = $2
            )
            ''',
            schema, table,
        )
        return bool(result)
    finally:
        await conn.close()


async def function_exists_in_schema(
    pg_url: str, schema: str, function: str,
) -> bool:
    """Return True when ``schema.function`` exists."""
    conn = await asyncpg.connect(pg_url)
    try:
        result = await conn.fetchval(
            '''
            SELECT EXISTS (
                SELECT 1 FROM pg_proc p
                JOIN pg_namespace n ON p.pronamespace = n.oid
                WHERE n.nspname = $1 AND p.proname = $2
            )
            ''',
            schema, function,
        )
        return bool(result)
    finally:
        await conn.close()


async def column_exists_in_schema(
    pg_url: str, schema: str, table: str, column: str,
) -> bool:
    """Return True when ``schema.table.column`` exists."""
    conn = await asyncpg.connect(pg_url)
    try:
        result = await conn.fetchval(
            '''
            SELECT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = $1
                  AND table_name = $2
                  AND column_name = $3
            )
            ''',
            schema, table, column,
        )
        return bool(result)
    finally:
        await conn.close()


async def proconfig_for_function(
    pg_url: str, schema: str, function: str,
) -> list[str] | None:
    """Return ``pg_proc.proconfig`` for the named function in the schema."""
    conn = await asyncpg.connect(pg_url)
    try:
        row = await conn.fetchrow(
            '''
            SELECT proconfig FROM pg_proc p
            JOIN pg_namespace n ON p.pronamespace = n.oid
            WHERE n.nspname = $1 AND p.proname = $2
            ''',
            schema, function,
        )
    finally:
        await conn.close()
    if row is None:
        return None
    cfg = row['proconfig']
    return list(cfg) if cfg is not None else None
