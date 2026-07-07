"""Migration idempotency tests under a non-default POSTGRESQL_SCHEMA.

Default CI runs every PostgreSQL migration against
``POSTGRESQL_SCHEMA=public``. These tests parameterize the same set of
migrations against ``POSTGRESQL_SCHEMA=mcp_test`` so the project-wide
BARE-DDL convention (TABLE/INDEX DDL uses BARE names + operator's
``search_path``; FUNCTION DDL stays schema-qualified for CVE-2018-1058
mitigation) is verified end-to-end under both schemas.

The ``pg_non_default_schema_db`` fixture (see ``conftest.py``)
provisions an isolated database with the ``mcp_test`` schema
pre-created. Each test sets ``POSTGRESQL_SCHEMA=mcp_test``, invalidates
the settings singleton, and enforces ``search_path = mcp_test, public``
on every connection by patching :func:`asyncpg.create_pool` to inject a
session-level ``SET search_path`` ``init=`` callback. This pattern
keeps every backend operation -- including DDL issued by migration
helpers and reads issued by the idempotency-check filters -- aligned
with the operator's intended schema.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Generator
from typing import cast

import asyncpg
import pytest

import app.backends.postgresql_backend as postgresql_backend_module
import app.migrations.chunking as chunking_module
import app.migrations.compression as compression_module
import app.migrations.semantic as semantic_module
import app.startup as startup_module
from app.backends import create_backend
from app.migrations.chunking import apply_chunking_migration
from app.migrations.compression import apply_compression_migration
from app.migrations.semantic import apply_function_search_path_migration
from app.migrations.semantic import apply_jsonb_merge_patch_migration
from app.migrations.semantic import apply_semantic_search_migration
from app.repositories.embedding_repository import _reset_compression_cache
from app.settings import get_settings
from app.startup import init_database
from tests.integration.postgresql.conftest import NON_DEFAULT_SCHEMA

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

_InitCallback = Callable[[asyncpg.Connection], Awaitable[None]]


@pytest.fixture(autouse=True)
def reset_caches() -> Generator[None, None, None]:
    """Invalidate settings + compression caches around every test.

    Yields:
        ``None`` (sentinel); caches are cleared again at teardown.
    """
    get_settings.cache_clear()
    _reset_compression_cache()
    yield None
    get_settings.cache_clear()
    _reset_compression_cache()


def _install_search_path_patch(
    monkeypatch: pytest.MonkeyPatch, schema: str,
) -> None:
    """Wrap ``asyncpg.create_pool`` so every backend connection sets search_path.

    Production ``PostgreSQLBackend`` carries the operator's
    ``search_path`` via the asyncpg ``server_settings`` startup packet
    (see ``build_asyncpg_connect_kwargs``); that startup-packet value
    becomes the connection's reset value, so it SURVIVES the
    ``RESET ALL`` the pool issues on every release (proven by
    ``test_production_pool_search_path_survives_reset_all``). This helper
    instead drives search_path with a runtime ``SET search_path``, which
    ``RESET ALL`` DOES clear, so it re-applies the ``SET`` on every
    acquire (via both an ``init=`` and a ``setup=`` callback). It exists
    so the test's own ad-hoc bookkeeping resolves to the configured
    schema regardless of that mechanism difference, NOT because the
    production server_settings path needs it; production DOES configure
    ``init=``/``setup=`` callbacks, but neither sets search_path.

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

    # Patch both the asyncpg module itself and the dotted reference
    # used inside PostgreSQLBackend.initialize so the backend's call
    # site picks up the wrapper regardless of how it imported asyncpg.
    monkeypatch.setattr(asyncpg, 'create_pool', _patched_create_pool)
    monkeypatch.setattr(
        'app.backends.postgresql_backend.asyncpg.create_pool',
        _patched_create_pool,
    )


def _refresh_module_settings(monkeypatch: pytest.MonkeyPatch) -> None:
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
    # The backend reads POSTGRESQL_SCHEMA (via build_asyncpg_connect_kwargs) off
    # its own module-level ``settings`` binding; without this rebind it would
    # send the import-time default schema in server_settings, so the production
    # pool's search_path test would resolve to ``public`` instead of the
    # configured schema.
    monkeypatch.setattr(postgresql_backend_module, 'settings', fresh)


def _configure_non_default_env(
    monkeypatch: pytest.MonkeyPatch, pg_url: str,
) -> None:
    """Apply the env vars every non-default-schema test needs."""
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_url)
    monkeypatch.setenv('POSTGRESQL_SCHEMA', NON_DEFAULT_SCHEMA)
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    _refresh_module_settings(monkeypatch)


def _configure_default_schema_env(
    monkeypatch: pytest.MonkeyPatch, pg_url: str,
) -> None:
    """Apply env vars for the default-schema (``public``) baseline test."""
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_url)
    monkeypatch.setenv('POSTGRESQL_SCHEMA', 'public')
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    _refresh_module_settings(monkeypatch)


async def _table_exists_in_schema(
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


async def _function_exists_in_schema(
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


async def _column_exists_in_schema(
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


async def _proconfig_for_function(
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


def test_compression_migration_idempotent_under_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply compression migration twice against the default schema.

    Establishes the baseline: under ``POSTGRESQL_SCHEMA=public`` the
    migration MUST be idempotent (second call is a no-op) AND the
    compressed table MUST exist in ``public``. Uses the
    ``pg_non_default_schema_db`` fixture to obtain an isolated
    database, but configures ``POSTGRESQL_SCHEMA=public`` so the
    BARE-DDL contract resolves to the default schema and pre-existing
    tables in ``mcp_test`` are not consulted.
    """
    _install_search_path_patch(monkeypatch, 'public')
    _configure_default_schema_env(monkeypatch, pg_non_default_schema_db)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()
    monkeypatch.setattr(compression_module, 'settings', get_settings())
    monkeypatch.setattr(semantic_module, 'settings', get_settings())
    monkeypatch.setattr(chunking_module, 'settings', get_settings())

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
            await apply_chunking_migration(backend=backend)
            await apply_compression_migration(backend=backend)
            await apply_compression_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, 'public',
        'vec_context_embeddings_compressed',
    ))
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, 'public', 'compression_metadata',
    ))


def test_compression_migration_idempotent_under_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as the default-schema control but under POSTGRESQL_SCHEMA=mcp_test.

    Verifies the BARE-DDL contract: tables MUST be created in
    ``mcp_test`` (not ``public``); second migration call MUST be a
    no-op. This is the direct regression test for the project-wide
    bare-DDL conversion.
    """
    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    monkeypatch.setenv('COMPRESSION_BITS', '4')
    monkeypatch.setenv('COMPRESSION_VARIANT', 'ip')
    get_settings.cache_clear()
    monkeypatch.setattr(compression_module, 'settings', get_settings())
    monkeypatch.setattr(semantic_module, 'settings', get_settings())
    monkeypatch.setattr(chunking_module, 'settings', get_settings())

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
            await apply_chunking_migration(backend=backend)
            await apply_compression_migration(backend=backend)
            await apply_compression_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings_compressed',
    ))
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'compression_metadata',
    ))
    assert not asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, 'public',
        'vec_context_embeddings_compressed',
    ))


def test_compression_guard_sees_legacy_fp32_in_public_under_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A populated legacy fp32 table in ``public`` trips the enable-direction guard.

    The migration's ``DROP TABLE IF EXISTS vec_context_embeddings`` uses a
    BARE name resolved through ``search_path`` (configured schema first,
    then ``public``), so a legacy fp32 deployment living in ``public`` is
    reachable by the drop even after the operator sets a non-default
    ``POSTGRESQL_SCHEMA``. The guard probe must resolve the same way: a
    probe pinned to the configured schema reported the table absent, let
    the guard pass, and the migration destroyed every stored embedding.
    The guard must refuse loudly (exit 78) and leave the table intact.
    """
    from app.errors import ConfigurationError

    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'true')
    monkeypatch.setenv('COMPRESSION_SEED', '42')
    get_settings.cache_clear()
    monkeypatch.setattr(compression_module, 'settings', get_settings())

    async def _seed_legacy_public_fp32() -> None:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            await conn.execute(
                'CREATE TABLE public.vec_context_embeddings '
                '(id BIGSERIAL PRIMARY KEY, context_id UUID)',
            )
            await conn.execute(
                'INSERT INTO public.vec_context_embeddings (context_id) '
                "VALUES ('00000000-0000-0000-0000-000000000001')",
            )
        finally:
            await conn.close()

    async def _scenario() -> None:
        await _seed_legacy_public_fp32()
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            with pytest.raises(ConfigurationError, match='uncompressed fp32'):
                await apply_compression_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    # The legacy table survived, still populated, and no compression schema
    # was provisioned anywhere.
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, 'public', 'vec_context_embeddings',
    ))
    for schema in ('public', NON_DEFAULT_SCHEMA):
        assert not asyncio.run(_table_exists_in_schema(
            pg_non_default_schema_db, schema,
            'vec_context_embeddings_compressed',
        ))


def test_chunking_migration_idempotent_under_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply chunking migration twice under POSTGRESQL_SCHEMA=mcp_test.

    Verifies that the chunking migration (BARE table DDL +
    ``current_schema()`` idempotency filters) creates the expected
    columns in ``mcp_test`` and is a no-op on a second invocation.
    """
    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
            await apply_chunking_migration(backend=backend)
            await apply_chunking_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    # Chunking-added columns must be present in mcp_test.
    assert asyncio.run(_column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'embedding_metadata', 'chunk_count',
    ))
    assert asyncio.run(_column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings', 'id',
    ))
    assert asyncio.run(_column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings', 'start_index',
    ))
    assert asyncio.run(_column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings', 'end_index',
    ))
    # The chunking DDL must NOT have leaked into ``public``.
    assert not asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, 'public', 'vec_context_embeddings',
    ))


def test_semantic_search_migration_function_in_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The semantic-search trigger function lives in ``mcp_test``.

    The FUNCTION DDL stays schema-qualified for CVE-2018-1058
    mitigation; combined with the operator's ``search_path``, the
    function and the trigger that references it must both resolve to
    ``mcp_test`` when ``POSTGRESQL_SCHEMA=mcp_test``.
    """
    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    assert asyncio.run(_function_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'update_embedding_metadata_timestamp',
    ))
    # The function MUST NOT have been created in ``public``.
    assert not asyncio.run(_function_exists_in_schema(
        pg_non_default_schema_db, 'public',
        'update_embedding_metadata_timestamp',
    ))
    # The embedding_metadata table must also exist in mcp_test (the
    # trigger references it).
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'embedding_metadata',
    ))


def test_jsonb_merge_patch_function_in_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``jsonb_merge_patch`` is created in ``mcp_test`` and is callable.

    Verifies the FUNCTION DDL substitution in
    ``add_jsonb_merge_patch_postgresql.sql`` resolves to
    ``mcp_test.jsonb_merge_patch`` and that the function executes RFC
    7396 null-deletion semantics correctly.
    """
    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_jsonb_merge_patch_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    assert asyncio.run(_function_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'jsonb_merge_patch',
    ))
    assert not asyncio.run(_function_exists_in_schema(
        pg_non_default_schema_db, 'public', 'jsonb_merge_patch',
    ))

    # Invoke the function with the schema-qualified name to confirm
    # RFC 7396 semantics work end-to-end under the non-default schema.
    async def _invoke_jsonb_merge_patch() -> str:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            result = await conn.fetchval(
                f'SELECT {NON_DEFAULT_SCHEMA}.jsonb_merge_patch('
                "'{\"a\":\"b\"}'::jsonb, '{\"a\":null}'::jsonb)",
            )
        finally:
            await conn.close()
        return str(result)

    merged = asyncio.run(_invoke_jsonb_merge_patch())
    assert merged == '{}'


def test_fix_function_search_path_under_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ALTER FUNCTION targets resolve to ``mcp_test`` functions.

    After running the semantic-search and jsonb-merge-patch migrations
    followed by ``apply_function_search_path_migration``, every
    schema-qualified function in ``mcp_test`` must carry the hardened
    ``search_path=pg_catalog, pg_temp`` configuration.
    """
    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
            await apply_jsonb_merge_patch_migration(backend=backend)
            await apply_function_search_path_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    expected_setting = 'search_path=pg_catalog, pg_temp'
    for function in (
        'update_updated_at_column',
        'update_embedding_metadata_timestamp',
        'jsonb_merge_patch',
    ):
        cfg = asyncio.run(_proconfig_for_function(
            pg_non_default_schema_db, NON_DEFAULT_SCHEMA, function,
        ))
        assert cfg is not None, (
            f'Function {NON_DEFAULT_SCHEMA}.{function} should exist and '
            f'carry a proconfig entry after fix_function_search_path '
            f'migration ran.'
        )
        assert expected_setting in cfg, (
            f'Function {NON_DEFAULT_SCHEMA}.{function} must have '
            f'{expected_setting!r} in pg_proc.proconfig; observed {cfg!r}.'
        )


def test_function_ddl_quotes_mixed_case_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Schema-qualified function DDL must quote a mixed-case POSTGRESQL_SCHEMA.

    ``CREATE/EXECUTE/ALTER FUNCTION {SCHEMA}.foo()`` built by raw substitution
    of an unquoted schema folds a mixed-case (or reserved-word) name to
    lowercase at parse time, so boot crashes with 'schema "..." does not
    exist' even though the connection search_path (quoted) reaches the schema.
    init_database and the three function migrations MUST succeed against a
    quoted mixed-case schema created via quoted DDL, and the functions MUST
    land in that schema.
    """
    mixed = 'MixedCaseSchema'

    async def _create_mixed_schema() -> None:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{mixed}"')
        finally:
            await conn.close()

    asyncio.run(_create_mixed_schema())

    # Drive the REAL production pool (which quotes search_path via
    # server_settings), NOT the raw-SET test shim, so a mixed-case schema
    # resolves the same way production resolves it.
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_non_default_schema_db)
    monkeypatch.setenv('POSTGRESQL_SCHEMA', mixed)
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    _refresh_module_settings(monkeypatch)

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            # Pre-fix, init_database itself crashes here on the unquoted
            # CREATE FUNCTION "mixedcaseschema".update_updated_at_column().
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
            await apply_jsonb_merge_patch_migration(backend=backend)
            await apply_function_search_path_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    for function in (
        'update_updated_at_column',
        'update_embedding_metadata_timestamp',
        'jsonb_merge_patch',
    ):
        assert asyncio.run(_function_exists_in_schema(
            pg_non_default_schema_db, mixed, function,
        )), f'{mixed}.{function} must exist after the quoted function DDL ran'
        assert not asyncio.run(_function_exists_in_schema(
            pg_non_default_schema_db, mixed.lower(), function,
        )), f'the function must not fold into a lowercased schema {mixed.lower()!r}'


def test_patch_metadata_runtime_quotes_mixed_case_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patch_metadata's RUNTIME jsonb_merge_patch() call must quote a mixed-case schema.

    Quoting only the DDL that CREATEs the function makes a mixed-case schema boot, but the
    runtime caller in ContextRepository.patch_metadata interpolated the schema UNQUOTED, so
    every metadata_patch case-folded to a nonexistent lowercase schema (SQLSTATE 3F000/
    42883) -- a loud boot failure converted into a latent per-request failure. The runtime
    call must resolve to the SAME quoted "MixedCaseSchema".jsonb_merge_patch the DDL created.
    """
    from app.repositories import RepositoryContainer

    mixed = 'MixedCaseSchema'

    async def _create_mixed_schema() -> None:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            await conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{mixed}"')
        finally:
            await conn.close()

    asyncio.run(_create_mixed_schema())

    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_non_default_schema_db)
    monkeypatch.setenv('POSTGRESQL_SCHEMA', mixed)
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    get_settings.cache_clear()
    _refresh_module_settings(monkeypatch)

    result: dict[str, object] = {}

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_jsonb_merge_patch_migration(backend=backend)
            repos = RepositoryContainer(backend)
            context_id, _ = await repos.context.store_with_deduplication(
                thread_id='mixed-patch', source='user', content_type='text',
                text_content='patch target', metadata=None,
            )
            # Pre-fix this raises SQLSTATE 3F000/42883 (mixedcaseschema.jsonb_merge_patch
            # does not exist); post-fix it resolves to the quoted "MixedCaseSchema" object.
            success, fields = await repos.context.patch_metadata(context_id, {'b': 2})
            result['success'] = success
            result['fields'] = fields
            entries = await repos.context.get_by_ids([context_id])
            result['metadata'] = entries[0]['metadata'] if entries else None
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    # success=True + the updated field prove the runtime jsonb_merge_patch() resolved to the
    # quoted "MixedCaseSchema" function instead of raising SQLSTATE 3F000/42883 -- the fix.
    assert result['success'] is True
    assert result['fields'] == ['metadata']
    import json as _json

    merged = result['metadata']
    if isinstance(merged, str):
        merged = _json.loads(merged)
    assert merged == {'b': 2}


def test_generation_off_infra_present_reprovisions_vector_layout(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation off + embedding infra present + pgvector absent still boots.

    The semantic/chunking migrations re-provision the fp32 vector layout on an
    infra-carrying generation-off database (embedding_metadata exists), but the
    pgvector extension pre-create + codec must follow the SAME gate. With the
    extension dropped, the migration's ``CREATE TABLE ... vector(dim)`` crashed
    boot. The backend now resolves the provision-vector decision as
    ``generation OR infra-present`` and re-creates the extension, so the
    migration succeeds.
    """
    # Use the fixture's own schema (which matches the connecting role, so the
    # raw-connection CREATE EXTENSION lands in the same schema the migration's
    # search_path resolves through). Compression OFF so the fp32 vector layout
    # (the vector-type user) is actually provisioned in phase 1.
    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    _refresh_module_settings(monkeypatch)

    async def _vector_type_present() -> bool:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            return bool(await conn.fetchval("SELECT to_regtype('vector') IS NOT NULL"))
        finally:
            await conn.close()

    async def _phase1_generation_on() -> None:
        backend = create_backend(
            backend_type='postgresql', connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_phase1_generation_on())
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'embedding_metadata',
    ))
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'vec_context_embeddings',
    ))

    async def _drop_extension_keep_infra() -> None:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            # Model a schema-only restore that omitted the pgvector extension and
            # its vector table but kept embedding_metadata: drop the fp32 table +
            # HNSW index first, then the extension (so nothing depends on the
            # vector type). This is the C9 precondition -- the vector type is
            # absent while the embedding infrastructure signal survives.
            await conn.execute('DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw')
            await conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')
            await conn.execute('DROP EXTENSION IF EXISTS vector')
        finally:
            await conn.close()

    asyncio.run(_drop_extension_keep_infra())
    # The vector type and table are gone; embedding_metadata (the infra signal)
    # survives, so the infra-present provisioning decision must still fire.
    assert not asyncio.run(_vector_type_present())
    assert not asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'vec_context_embeddings',
    ))
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'embedding_metadata',
    ))

    # Phase 2: generation OFF + compression OFF. A fresh backend must re-create
    # the pgvector extension (infra-present decision) so the semantic migration's
    # CREATE TABLE ... vector(dim) type-checks instead of crashing on
    # 'type "vector" does not exist' (which it does even when the relation
    # already exists, because CREATE TABLE IF NOT EXISTS type-checks first).
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    _refresh_module_settings(monkeypatch)

    async def _phase2_generation_off() -> None:
        backend = create_backend(
            backend_type='postgresql', connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            # Pre-fix, this raised 'type "vector" does not exist' at boot.
            await apply_semantic_search_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_phase2_generation_off())
    # The extension was re-created (infra-present decision), so the vector type
    # is available again and the migration re-provisioned the fp32 table.
    assert asyncio.run(_vector_type_present())
    assert asyncio.run(_table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'vec_context_embeddings',
    ))


def test_no_project_tables_in_public_when_non_default_schema_used(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the full migration sequence under POSTGRESQL_SCHEMA=mcp_test,
    NO project tables MUST appear in ``public``.

    Catches accidental table-leakage regressions in any migration that
    misses the bare-DDL conversion.
    """
    _install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_semantic_search_migration(backend=backend)
            await apply_chunking_migration(backend=backend)
            await apply_jsonb_merge_patch_migration(backend=backend)
            await apply_function_search_path_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    project_tables = (
        'context_entries', 'tags', 'image_attachments',
        'vec_context_embeddings', 'embedding_metadata',
    )
    for table in project_tables:
        assert not asyncio.run(_table_exists_in_schema(
            pg_non_default_schema_db, 'public', table,
        )), (
            f"Project table '{table}' MUST NOT exist in 'public' "
            f"when POSTGRESQL_SCHEMA={NON_DEFAULT_SCHEMA}"
        )
        assert asyncio.run(_table_exists_in_schema(
            pg_non_default_schema_db, NON_DEFAULT_SCHEMA, table,
        )), (
            f"Project table '{table}' MUST exist in "
            f"'{NON_DEFAULT_SCHEMA}' when POSTGRESQL_SCHEMA="
            f"{NON_DEFAULT_SCHEMA}"
        )


def test_production_pool_search_path_survives_reset_all(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production server_settings search_path survives RESET ALL on re-acquire.

    Unlike every other test in this module, this one does NOT install the
    ``SET``-based per-acquire search_path patch: it exercises the REAL
    production pool, whose search_path is carried solely by the asyncpg
    ``server_settings`` startup packet. The pool issues ``RESET ALL`` on
    every connection release; this test acquires, releases, then RE-acquires
    the SAME physical connection (matched by ``pg_backend_pid()``) and
    asserts the configured schema still governs bare-name resolution. That
    proves the startup-packet search_path is the connection's reset value
    (NOT reverted to the role/database default), which is the behavior the
    production code deliberately relies on -- and which the SET-based helper
    above would otherwise mask.
    """
    _configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    async def _scenario() -> tuple[str, str, str, str, bool]:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            # First transaction: capture the physical connection identity and
            # confirm the configured schema governs bare-name resolution. Each
            # begin_transaction acquires a pooled connection and releases it on
            # exit, which runs the pool's RESET ALL.
            async with backend.begin_transaction() as txn:
                conn = cast('asyncpg.Connection', txn.connection)
                pid = await conn.fetchval('SELECT pg_backend_pid()')
                sp1 = await conn.fetchval('SHOW search_path')
                await conn.execute('CREATE TABLE sp_probe_first (id int)')
                landed1 = await conn.fetchval(
                    "SELECT schemaname FROM pg_tables "
                    "WHERE tablename = 'sp_probe_first'",
                )
            # Re-enter transactions until the pool hands back the SAME backend
            # pid (a reused connection past one RESET ALL) and re-check that the
            # configured schema STILL governs -- proving the server_settings
            # search_path is the connection's reset value, not reverted to the
            # role/database default.
            sp2 = ''
            landed2 = ''
            reused = False
            for _ in range(50):
                async with backend.begin_transaction() as txn:
                    conn = cast('asyncpg.Connection', txn.connection)
                    if await conn.fetchval('SELECT pg_backend_pid()') != pid:
                        continue
                    sp2 = await conn.fetchval('SHOW search_path')
                    await conn.execute('CREATE TABLE sp_probe_second (id int)')
                    landed2 = await conn.fetchval(
                        "SELECT schemaname FROM pg_tables "
                        "WHERE tablename = 'sp_probe_second'",
                    )
                    reused = True
                    break
            return sp1, landed1, sp2, landed2, reused
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    sp1, landed1, sp2, landed2, reused = asyncio.run(_scenario())

    assert NON_DEFAULT_SCHEMA in sp1, f'first-acquire search_path was {sp1!r}'
    assert landed1 == NON_DEFAULT_SCHEMA, f'first bare table landed in {landed1!r}'
    assert reused, 'pool never handed back the same physical connection to re-test'
    # Load-bearing assertion: after a RESET ALL on release, the reused
    # connection STILL resolves bare names to the configured schema.
    assert NON_DEFAULT_SCHEMA in sp2, (
        f're-acquire search_path was {sp2!r} -- RESET ALL wiped the '
        f'server_settings search_path (production bug)'
    )
    assert landed2 == NON_DEFAULT_SCHEMA, (
        f'post-RESET-ALL bare table landed in {landed2!r}, not '
        f'{NON_DEFAULT_SCHEMA!r}'
    )
