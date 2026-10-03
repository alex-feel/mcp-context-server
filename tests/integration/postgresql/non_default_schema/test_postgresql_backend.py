"""PostgreSQL backend pool behavior under a non-default POSTGRESQL_SCHEMA.

Covers the provision-vector decision on a generation-off database that still
carries embedding infrastructure, and the configured ``search_path``
surviving the ``RESET ALL`` the pool issues on every connection release.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import cast

import asyncpg
import pytest

from app.backends import create_backend
from app.migrations.semantic import apply_semantic_search_migration
from app.settings import get_settings
from app.startup import init_database
from tests.integration.postgresql.conftest import NON_DEFAULT_SCHEMA
from tests.integration.postgresql.non_default_schema._schema import configure_non_default_env
from tests.integration.postgresql.non_default_schema._schema import install_search_path_patch
from tests.integration.postgresql.non_default_schema._schema import refresh_module_settings
from tests.integration.postgresql.non_default_schema._schema import table_exists_in_schema

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


def test_generation_off_infra_present_reprovisions_vector_layout(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Generation off + embedding infra present + pgvector absent still boots.

    The semantic/chunking migrations re-provision the fp32 vector layout on an
    infra-carrying generation-off database (embedding_metadata exists), but the
    pgvector extension pre-create + codec must follow the SAME gate. With the
    extension dropped, the migration's ``CREATE TABLE ... vector(dim)`` fails
    at boot unless the backend's provision-vector decision covers this
    infra-present database and re-creates the extension first.
    """
    # Use the fixture's own schema (which matches the connecting role, so the
    # raw-connection CREATE EXTENSION lands in the same schema the migration's
    # search_path resolves through). Compression OFF so the fp32 vector layout
    # (the vector-type user) is actually provisioned in phase 1.
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)
    monkeypatch.setenv('ENABLE_EMBEDDING_COMPRESSION', 'false')
    get_settings.cache_clear()
    refresh_module_settings(monkeypatch)

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
    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'embedding_metadata',
    ))
    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'vec_context_embeddings',
    ))

    async def _drop_extension_keep_infra() -> None:
        conn = await asyncpg.connect(pg_non_default_schema_db)
        try:
            # Model a schema-only restore that omitted the pgvector extension and
            # its vector table but kept embedding_metadata: drop the fp32 table +
            # HNSW index first, then the extension (so nothing depends on the
            # vector type). This is the precondition under test -- the vector type
            # is absent while the embedding infrastructure signal survives.
            await conn.execute('DROP INDEX IF EXISTS idx_vec_context_embeddings_hnsw')
            await conn.execute('DROP TABLE IF EXISTS vec_context_embeddings')
            await conn.execute('DROP EXTENSION IF EXISTS vector')
        finally:
            await conn.close()

    asyncio.run(_drop_extension_keep_infra())
    # The vector type and table are gone; embedding_metadata (the infra signal)
    # survives, so the infra-present provisioning decision must still fire.
    assert not asyncio.run(_vector_type_present())
    assert not asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'vec_context_embeddings',
    ))
    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'embedding_metadata',
    ))

    # Phase 2: generation OFF + compression OFF. A fresh backend must re-create
    # the pgvector extension (infra-present decision) so the semantic migration's
    # CREATE TABLE ... vector(dim) type-checks instead of crashing on
    # 'type "vector" does not exist' (which it does even when the relation
    # already exists, because CREATE TABLE IF NOT EXISTS type-checks first).
    monkeypatch.setenv('ENABLE_EMBEDDING_GENERATION', 'false')
    get_settings.cache_clear()
    refresh_module_settings(monkeypatch)

    async def _phase2_generation_off() -> None:
        backend = create_backend(
            backend_type='postgresql', connection_string=pg_non_default_schema_db,
        )
        await backend.initialize()
        try:
            # Without the re-created extension this raises 'type "vector" does not exist' at boot.
            await apply_semantic_search_migration(backend=backend)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_phase2_generation_off())
    # The extension was re-created (infra-present decision), so the vector type
    # is available again and the migration re-provisioned the fp32 table.
    assert asyncio.run(_vector_type_present())
    assert asyncio.run(table_exists_in_schema(
        pg_non_default_schema_db, NON_DEFAULT_SCHEMA, 'vec_context_embeddings',
    ))


def test_production_pool_search_path_survives_reset_all(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Production search_path survives RESET ALL on re-acquire.

    This test does NOT install the ``SET``-based test patch: it exercises
    the REAL production pool, whose ``setup`` callback
    (``setup_pool_connection``) applies the configured search_path with
    ``SET`` on every acquire. The pool issues ``RESET ALL`` on every
    connection release; this test acquires, releases, then RE-acquires the
    SAME physical connection (matched by ``pg_backend_pid()``) and asserts
    the configured schema still governs bare-name resolution instead of the
    role/database default. Production relies on that behavior, and the
    ``SET``-based test patch would mask it.
    """
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)

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
            # configured schema STILL governs -- proving the setup callback
            # restores the configured search_path after RESET ALL instead of
            # leaving the role/database default.
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
        f'configured search_path (production bug)'
    )
    assert landed2 == NON_DEFAULT_SCHEMA, (
        f'post-RESET-ALL bare table landed in {landed2!r}, not '
        f'{NON_DEFAULT_SCHEMA!r}'
    )
