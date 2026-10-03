"""Table migrations under a non-default POSTGRESQL_SCHEMA.

Bare-name TABLE/INDEX DDL resolves through ``search_path``, so the compression
and chunking tables land in ``mcp_test``, never in ``public``, and each
migration is idempotent; the default-schema compression test is the
``public`` baseline, and the full migration sequence leaves no project table
in ``public``.
"""

from __future__ import annotations

import asyncio
import contextlib

import asyncpg
import pytest

import app.migrations.chunking as chunking_module
import app.migrations.compression as compression_module
import app.migrations.semantic as semantic_module
from app.backends import create_backend
from app.migrations.chunking import apply_chunking_migration
from app.migrations.compression import apply_compression_migration
from app.migrations.semantic import apply_function_search_path_migration
from app.migrations.semantic import apply_jsonb_merge_patch_migration
from app.migrations.semantic import apply_semantic_search_migration
from app.settings import get_settings
from app.startup import init_database
from tests.integration.postgresql.conftest import NON_DEFAULT_SCHEMA
from tests.integration.postgresql.non_default_schema._schema import column_exists_in_schema
from tests.integration.postgresql.non_default_schema._schema import configure_default_schema_env
from tests.integration.postgresql.non_default_schema._schema import configure_non_default_env
from tests.integration.postgresql.non_default_schema._schema import install_search_path_patch
from tests.integration.postgresql.non_default_schema._schema import table_exists_in_schema

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


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
    install_search_path_patch(monkeypatch, 'public')
    configure_default_schema_env(monkeypatch, pg_non_default_schema_db)
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

    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, 'public',
        'vec_context_embeddings_compressed',
    ))
    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, 'public', 'compression_metadata',
    ))


def test_compression_migration_idempotent_under_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same as the default-schema control but under POSTGRESQL_SCHEMA=mcp_test.

    Verifies the BARE-DDL contract: tables MUST be created in
    ``mcp_test`` (not ``public``); second migration call MUST be a
    no-op. It pins the project-wide bare-DDL convention.
    """
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)
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

    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings_compressed',
    ))
    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'compression_metadata',
    ))
    assert not asyncio.run(table_exists_in_schema(
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
    probe pinned to the configured schema would report the table absent,
    let the guard pass, and let the migration destroy every stored
    embedding. The guard must refuse loudly (exit 78) and leave the table
    intact.
    """
    from app.errors import ConfigurationError

    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)
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
    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, 'public', 'vec_context_embeddings',
    ))
    for schema in ('public', NON_DEFAULT_SCHEMA):
        assert not asyncio.run(table_exists_in_schema(
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
            await apply_chunking_migration(backend=backend)
            await apply_chunking_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())

    # Chunking-added columns must be present in mcp_test.
    assert asyncio.run(column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'embedding_metadata', 'chunk_count',
    ))
    assert asyncio.run(column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings', 'id',
    ))
    assert asyncio.run(column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings', 'start_index',
    ))
    assert asyncio.run(column_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA,
        'vec_context_embeddings', 'end_index',
    ))
    # The chunking DDL must NOT have leaked into ``public``.
    assert not asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, 'public', 'vec_context_embeddings',
    ))


def test_no_project_tables_in_public_when_non_default_schema_used(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After the full migration sequence under POSTGRESQL_SCHEMA=mcp_test,
    NO project tables MUST appear in ``public``.

    Catches table leakage from any migration that does not follow the
    bare-DDL convention.
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
        assert not asyncio.run(table_exists_in_schema(
            pg_non_default_schema_db, 'public', table,
        )), (
            f"Project table '{table}' MUST NOT exist in 'public' "
            f"when POSTGRESQL_SCHEMA={NON_DEFAULT_SCHEMA}"
        )
        assert asyncio.run(table_exists_in_schema(
            pg_non_default_schema_db, NON_DEFAULT_SCHEMA, table,
        )), (
            f"Project table '{table}' MUST exist in "
            f"'{NON_DEFAULT_SCHEMA}' when POSTGRESQL_SCHEMA="
            f"{NON_DEFAULT_SCHEMA}"
        )
