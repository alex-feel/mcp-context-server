"""Table migrations under a non-default POSTGRESQL_SCHEMA.

Bare-name TABLE/INDEX DDL resolves through ``search_path``, so the compression
and chunking tables land in ``mcp_test``, never in ``public``, and each
migration is idempotent; the default-schema compression test is the
``public`` baseline, and the full migration sequence leaves no project table
in ``public``. The ingest index and the ingest statistics land on
``mcp_test.context_entries``, and their probes look for them on that table
alone, so an index or a statistics object of the same name that a deployment
left in ``public`` neither hides them nor is touched.
"""

from __future__ import annotations

import asyncio
import contextlib

import asyncpg
import pytest

import app.migrations.chunking as chunking_module
import app.migrations.compression as compression_module
import app.migrations.ingest as ingest_module
import app.migrations.semantic as semantic_module
from app.backends import create_backend
from app.ingest_sql import INGEST_INDEX_NAME
from app.ingest_sql import ingest_index_ddl
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations.chunking import apply_chunking_migration
from app.migrations.compression import apply_compression_migration
from app.migrations.ingest import INGEST_STATISTICS_NAME
from app.migrations.ingest import apply_ingest_index_migration
from app.migrations.ingest import ingest_statistics_ddl
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


async def _ingest_index_tables(pg_url: str, schema: str) -> list[str]:
    """Return the table of every index named like the ingest index in ``schema``."""
    conn = await asyncpg.connect(pg_url)
    try:
        rows = await conn.fetch(
            'SELECT tablename FROM pg_indexes WHERE schemaname = $1 AND indexname = $2', schema, INGEST_INDEX_NAME,
        )
        return [str(row['tablename']) for row in rows]
    finally:
        await conn.close()


async def _ingest_statistics_objects(pg_url: str) -> list[tuple[str, str, str]]:
    """Return the schema of every statistics object named like the ingest statistics, with its table's schema and name."""
    conn = await asyncpg.connect(pg_url)
    try:
        rows = await conn.fetch(
            'SELECT statistics_schema.nspname AS statistics_schema, table_schema.nspname AS table_schema, '
            'relation.relname AS table_name '
            'FROM pg_statistic_ext AS statistics '
            'JOIN pg_namespace AS statistics_schema ON statistics_schema.oid = statistics.stxnamespace '
            'JOIN pg_class AS relation ON relation.oid = statistics.stxrelid '
            'JOIN pg_namespace AS table_schema ON table_schema.oid = relation.relnamespace '
            'WHERE statistics.stxname = $1 ORDER BY statistics_schema.nspname',
            INGEST_STATISTICS_NAME,
        )
        return [(str(row['statistics_schema']), str(row['table_schema']), str(row['table_name'])) for row in rows]
    finally:
        await conn.close()


def _first_build_steps() -> list[str]:
    """Return the steps of the first application on a database holding neither the ingest index nor its statistics."""
    return ['begin_migration', ingest_index_ddl('postgresql'), ingest_statistics_ddl(), 'ANALYZE context_entries']


def _apply_ingest_index_migration_twice(pg_url: str, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Prepare the base schema, apply the ingest index migration twice and return the lock and DDL steps it ran."""
    steps: list[str] = []

    async def _begin(conn: asyncpg.Connection, timeout_s: float) -> None:
        steps.append('begin_migration')
        await begin_migration(conn, timeout_s)

    async def _ddl(conn: asyncpg.Connection, statement: str, timeout_s: float) -> None:
        steps.append(statement)
        await execute_migration_ddl(conn, statement, timeout_s)

    monkeypatch.setattr(ingest_module, 'settings', get_settings())
    monkeypatch.setattr(ingest_module, 'begin_migration', _begin)
    monkeypatch.setattr(ingest_module, 'execute_migration_ddl', _ddl)

    async def _scenario() -> None:
        backend = create_backend(backend_type='postgresql', connection_string=pg_url)
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_ingest_index_migration(backend)
            await apply_ingest_index_migration(backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())
    return steps


def test_ingest_index_migration_idempotent_under_non_default_schema(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ingest index and statistics land in ``mcp_test`` once; a second application runs no DDL; ``public`` has neither."""
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    steps = _apply_ingest_index_migration_twice(pg_non_default_schema_db, monkeypatch)

    assert steps == _first_build_steps()
    assert asyncio.run(_ingest_index_tables(pg_non_default_schema_db, NON_DEFAULT_SCHEMA)) == ['context_entries']
    assert asyncio.run(_ingest_index_tables(pg_non_default_schema_db, 'public')) == []
    assert asyncio.run(_ingest_statistics_objects(pg_non_default_schema_db)) == [
        (NON_DEFAULT_SCHEMA, NON_DEFAULT_SCHEMA, 'context_entries'),
    ]


def test_ingest_probes_ignore_a_same_named_index_and_statistics_in_public(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ingest index and statistics a deployment left in ``public`` do not stand in for the configured schema's.

    ``search_path`` lists ``public`` after ``mcp_test``, so a bare ``to_regclass`` of the
    index name would find the ``public`` index and skip the build; the probes instead look
    for the index and the statistics on the table the DDL targets, and leave the ``public``
    objects as they are.
    """
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)

    async def _seed_public_index() -> None:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            await conn.execute('SET search_path = public')
            await conn.execute('CREATE TABLE context_entries (id UUID PRIMARY KEY, thread_id TEXT NOT NULL, metadata JSONB)')
            await conn.execute(ingest_index_ddl('postgresql'))
            await conn.execute(ingest_statistics_ddl())
        finally:
            await conn.close()

    asyncio.run(_seed_public_index())

    steps = _apply_ingest_index_migration_twice(pg_non_default_schema_db, monkeypatch)

    assert steps == _first_build_steps()
    assert asyncio.run(_ingest_index_tables(pg_non_default_schema_db, NON_DEFAULT_SCHEMA)) == ['context_entries']
    assert asyncio.run(_ingest_index_tables(pg_non_default_schema_db, 'public')) == ['context_entries']
    assert asyncio.run(_ingest_statistics_objects(pg_non_default_schema_db)) == [
        (NON_DEFAULT_SCHEMA, NON_DEFAULT_SCHEMA, 'context_entries'),
        ('public', 'public', 'context_entries'),
    ]
