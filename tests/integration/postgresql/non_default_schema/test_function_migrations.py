"""Function migrations under a non-default POSTGRESQL_SCHEMA.

FUNCTION DDL stays schema-qualified (CVE-2018-1058 mitigation), so the
semantic-search trigger function and ``jsonb_merge_patch`` land in the
configured schema with a pinned ``search_path``, and a mixed-case schema is
quoted both in the DDL and in the runtime ``jsonb_merge_patch()`` call.
"""

from __future__ import annotations

import asyncio
import contextlib

import asyncpg
import pytest

from app.backends import create_backend
from app.migrations.semantic import apply_function_search_path_migration
from app.migrations.semantic import apply_jsonb_merge_patch_migration
from app.migrations.semantic import apply_semantic_search_migration
from app.settings import get_settings
from app.startup import init_database
from tests.integration.postgresql.conftest import NON_DEFAULT_SCHEMA
from tests.integration.postgresql.non_default_schema._schema import configure_non_default_env
from tests.integration.postgresql.non_default_schema._schema import function_exists_in_schema
from tests.integration.postgresql.non_default_schema._schema import install_search_path_patch
from tests.integration.postgresql.non_default_schema._schema import proconfig_for_function
from tests.integration.postgresql.non_default_schema._schema import refresh_module_settings
from tests.integration.postgresql.non_default_schema._schema import table_exists_in_schema

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


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
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)

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

    assert asyncio.run(function_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'update_embedding_metadata_timestamp',
    ))
    # The function MUST NOT have been created in ``public``.
    assert not asyncio.run(function_exists_in_schema(
        pg_non_default_schema_db, 'public',
        'update_embedding_metadata_timestamp',
    ))
    # The embedding_metadata table must also exist in mcp_test (the
    # trigger references it).
    assert asyncio.run(table_exists_in_schema(
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
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)

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

    assert asyncio.run(function_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'jsonb_merge_patch',
    ))
    assert not asyncio.run(function_exists_in_schema(
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
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)

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
        cfg = asyncio.run(proconfig_for_function(
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

    # Drive the REAL production pool (whose setup callback applies the quoted
    # search_path with SET), NOT the raw-SET test patch, so a mixed-case schema
    # resolves the same way production resolves it.
    monkeypatch.setenv('STORAGE_BACKEND', 'postgresql')
    monkeypatch.setenv('POSTGRESQL_CONNECTION_STRING', pg_non_default_schema_db)
    monkeypatch.setenv('POSTGRESQL_SCHEMA', mixed)
    monkeypatch.setenv('EMBEDDING_DIM', '1024')
    monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'true')
    get_settings.cache_clear()
    refresh_module_settings(monkeypatch)

    async def _scenario() -> None:
        backend = create_backend(
            backend_type='postgresql',
            connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            # An unquoted schema would make init_database itself crash here on
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
        assert asyncio.run(function_exists_in_schema(
            pg_non_default_schema_db, mixed, function,
        )), f'{mixed}.{function} must exist after the quoted function DDL ran'
        assert not asyncio.run(function_exists_in_schema(
            pg_non_default_schema_db, mixed.lower(), function,
        )), f'the function must not fold into a lowercased schema {mixed.lower()!r}'


def test_patch_metadata_runtime_quotes_mixed_case_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """patch_metadata's RUNTIME jsonb_merge_patch() call must quote a mixed-case schema.

    Quoting only the DDL that CREATEs the function makes a mixed-case schema boot, but a
    runtime caller in ContextRepository.patch_metadata that interpolates the schema UNQUOTED
    case-folds every metadata_patch to a nonexistent lowercase schema (SQLSTATE 3F000/
    42883) -- a loud boot failure turned into a latent per-request failure. The runtime
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
    refresh_module_settings(monkeypatch)

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
                owner_id='local',
                visibility='private',
                thread_id='mixed-patch', source='user', content_type='text',
                text_content='patch target', metadata=None,
            )
            # An unquoted call would raise SQLSTATE 3F000/42883 (mixedcaseschema.jsonb_merge_patch
            # does not exist); the quoted call resolves to the "MixedCaseSchema" object.
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
    # quoted "MixedCaseSchema" function instead of raising SQLSTATE 3F000/42883.
    assert result['success'] is True
    assert result['fields'] == ['metadata']
    import json as _json

    merged = result['metadata']
    if isinstance(merged, str):
        merged = _json.loads(merged)
    assert merged == {'b': 2}
