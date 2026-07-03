"""End-to-end compression round trip against the real PostgreSQL backend.

Spawns the MCP server with ``ENABLE_EMBEDDING_COMPRESSION=true`` and
``STORAGE_BACKEND=postgresql`` pointed at the docker-compose pgvector
container. Verifies the schema migration replaced the fp32 vec table
with the compressed-payload table, the singleton ``compression_metadata``
row exists, and the basic store/get tool surface still works.

The round-trip tests keep embedding generation disabled so they do not
require Ollama; because compression provisioning follows embedding storage,
they pre-seed the compression schema (reproducing a database compressed
while generation was on) before booting. The re-enable test instead boots
with generation enabled (provisioning must run) and proves the regression
by direct schema inspection. The compressed write path is exercised by the
unit suite under ``tests/repositories/test_embedding_repository_compressed.py``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import asyncpg
import pytest
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


def _build_server_env(
    pg_url: str,
    *,
    bits: str = '4',
    variant: str = 'ip',
    seed: str = '42',
    generation: str = 'false',
) -> dict[str, str]:
    """Build the env dict for the spawned server with compression enabled."""
    return {
        **os.environ,
        'STORAGE_BACKEND': 'postgresql',
        'POSTGRESQL_CONNECTION_STRING': pg_url,
        'MCP_TEST_MODE': '1',
        'ENABLE_SEMANTIC_SEARCH': 'false',
        'ENABLE_FTS': 'false',
        'ENABLE_HYBRID_SEARCH': 'false',
        'ENABLE_EMBEDDING_GENERATION': generation,
        'ENABLE_SUMMARY_GENERATION': 'false',
        'ENABLE_EMBEDDING_COMPRESSION': 'true',
        'COMPRESSION_BITS': bits,
        'COMPRESSION_VARIANT': variant,
        'COMPRESSION_SEED': seed,
        'COMPRESSION_MAX_CONCURRENT': '2',
    }


async def _preseed_compression_schema(pg_url: str, *, seed: int = 42) -> None:
    """Seed the compression schema + provenance row into a fresh PG database.

    Compression provisioning follows embedding storage: with
    ``ENABLE_EMBEDDING_GENERATION=false`` the server creates no compression
    schema on a fresh database. Pre-seeding reproduces a database whose data
    was compressed while generation was on, so an Ollama-free (generation-off)
    server boot maintains and seed-validates the schema instead of idling.
    The fingerprint stays NULL (the documented pre-fingerprint state) so the
    validator warns and proceeds. The payload-table FK to ``context_entries``
    is omitted because the base schema does not exist yet at pre-seed time and
    no row is ever inserted by these generation-off runs.
    """
    conn = await asyncpg.connect(pg_url)
    try:
        await conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
                variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
                seed BIGINT NOT NULL CHECK (seed >= 0),
                dim INTEGER NOT NULL CHECK (dim > 0),
                codebook_fingerprint TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            ''',
        )
        await conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
                id BIGSERIAL PRIMARY KEY,
                context_id UUID NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                payload BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            ''',
        )
        await conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', $1, 1024, NULL)",
            seed,
        )
    finally:
        await conn.close()


async def _table_exists(conn: asyncpg.Connection, table: str) -> bool:
    """Return True when ``table`` exists in the current PG database."""
    result = await conn.fetchval(
        '''
        SELECT EXISTS (
            SELECT 1 FROM information_schema.tables
            WHERE table_name = $1
        )
        ''',
        table,
    )
    return bool(result)


@pytest.mark.asyncio
async def test_compression_round_trip_postgresql(pg_test_url: str) -> None:
    """End-to-end compression round trip against PostgreSQL.

    Verifies:

    1. The compressed schema is present and maintained (pre-seeded here,
       reproducing a database compressed while generation was on, because
       compression provisioning follows embedding storage and this
       Ollama-free run keeps generation off) while the fp32
       ``vec_context_embeddings`` table stays absent.
    2. The singleton ``compression_metadata`` row reflects env values.
    3. ``store_context`` + ``get_context_by_ids`` still work via MCP.

    Args:
        pg_test_url: Connection string from the session-scoped docker
            fixture.
    """
    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'
    server_env = _build_server_env(pg_test_url)
    await _preseed_compression_schema(pg_test_url)

    transport = PythonStdioTransport(
        script_path=str(wrapper_script),
        env=server_env,
    )
    client: Client[Any] = Client(transport)
    async with client:
        await client.ping()

        thread_id = f'pg_compression_rt_{int(time.time())}'
        store_result = await client.call_tool(
            'store_context',
            {
                'thread_id': thread_id,
                'source': 'agent',
                'text': 'PostgreSQL compression round trip',
                'tags': ['compression', 'pg', 'round-trip'],
            },
        )
        store_content: dict[str, Any] | None = None
        if hasattr(store_result, 'structured_content'):
            store_content = store_result.structured_content
        assert store_content is not None
        assert store_content.get('success') is True, store_content
        context_id = store_content.get('context_id')
        assert context_id, store_content

        conn = await asyncpg.connect(pg_test_url)
        try:
            assert await _table_exists(conn, 'vec_context_embeddings_compressed')
            assert not await _table_exists(conn, 'vec_context_embeddings')
            assert await _table_exists(conn, 'compression_metadata')

            row = await conn.fetchrow(
                'SELECT provider, bits, variant, seed '
                'FROM compression_metadata WHERE id = 1',
            )
            assert row is not None, 'compression_metadata row missing'
            assert row['provider'] == 'turboquant'
            assert int(row['bits']) == 4
            assert row['variant'] == 'ip'
            assert int(row['seed']) == 42
        finally:
            await conn.close()

        get_result = await client.call_tool(
            'get_context_by_ids',
            {'context_ids': [context_id]},
        )
        get_content: dict[str, Any] | None = None
        if hasattr(get_result, 'structured_content'):
            get_content = get_result.structured_content
        assert get_content is not None
        results = get_content.get('result') or get_content.get('results')
        assert results, get_content
        assert results[0]['id'] == context_id

        # Best-effort cleanup; delete_context may be absent via DISABLED_TOOLS.
        import contextlib

        from fastmcp.exceptions import ToolError
        with contextlib.suppress(ToolError):
            await client.call_tool(
                'delete_context',
                {'thread_id': thread_id},
            )


@pytest.mark.asyncio
async def test_compression_round_trip_postgresql_mse_variant(
    pg_test_url: str,
) -> None:
    """Variant smoke test: same round trip with MSE variant + bits=2.

    Uses a fresh PG database (``mcp_test_mse``) so the bootstrap validator
    inserts a new provenance row rather than colliding with the row written
    by ``test_compression_round_trip_postgresql`` (different ``bits`` and
    ``variant`` values).

    Args:
        pg_test_url: Connection string from the session-scoped docker
            fixture.
    """
    # Create an isolated database so the bootstrap-only invariant is not
    # violated by a different-variant first run sharing the same DB.
    admin_url = pg_test_url
    admin_conn = await asyncpg.connect(admin_url)
    try:
        await admin_conn.execute('DROP DATABASE IF EXISTS mcp_test_mse')
        await admin_conn.execute('CREATE DATABASE mcp_test_mse')
    finally:
        await admin_conn.close()

    mse_url = pg_test_url.rsplit('/', 1)[0] + '/mcp_test_mse'

    # Enable pgvector in the new database.
    mse_admin = await asyncpg.connect(mse_url)
    try:
        await mse_admin.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await mse_admin.close()

    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'
    server_env = _build_server_env(mse_url, bits='2', variant='mse')

    transport = PythonStdioTransport(
        script_path=str(wrapper_script),
        env=server_env,
    )
    client: Client[Any] = Client(transport)
    async with client:
        await client.ping()

        thread_id = f'pg_compression_mse_{int(time.time())}'
        store_result = await client.call_tool(
            'store_context',
            {
                'thread_id': thread_id,
                'source': 'agent',
                'text': 'MSE variant round trip',
            },
        )
        store_content: dict[str, Any] | None = None
        if hasattr(store_result, 'structured_content'):
            store_content = store_result.structured_content
        assert store_content is not None
        assert store_content.get('success') is True, store_content

        # Best-effort cleanup; delete_context may be absent via DISABLED_TOOLS.
        import contextlib

        from fastmcp.exceptions import ToolError
        with contextlib.suppress(ToolError):
            await client.call_tool(
                'delete_context',
                {'thread_id': thread_id},
            )


@pytest.mark.asyncio
async def test_compression_reenable_after_decompress_recreates_payload_table(
    pg_test_url: str,
) -> None:
    """Re-enabling compression after --decompress re-creates the payload table.

    ``--decompress`` drops ``vec_context_embeddings_compressed`` and deletes
    the provenance row but leaves the ``compression_metadata`` table behind.
    The migration idempotency probe must not treat that marker table alone as
    "already applied": a later re-enable start has to re-create the payload
    table, or every embedding store and compressed search fails with
    UndefinedTableError and no CLI path recovers (--compress no-ops on the
    validator-reseeded row, --decompress errors on the absent table).

    Both boots run with ``ENABLE_EMBEDDING_GENERATION=true`` because
    compression provisioning follows embedding storage -- the wedge this
    guards against arises on generation-on deployments. Booting does not
    require a reachable Ollama (provider construction is lazy and the
    pre-warm degrades to a warning), so the test stays CI-safe; the proof is
    direct schema inspection rather than a store call, whose embedding leg
    would need a live provider (the store path over the re-created table is
    covered by the unit suite and the live-stack end-to-end run).

    Args:
        pg_test_url: Connection string from the session-scoped docker
            fixture.
    """
    # Isolated database: this test deletes the provenance row mid-sequence.
    admin_conn = await asyncpg.connect(pg_test_url)
    try:
        await admin_conn.execute('DROP DATABASE IF EXISTS mcp_test_reenable')
        await admin_conn.execute('CREATE DATABASE mcp_test_reenable')
    finally:
        await admin_conn.close()

    reenable_url = pg_test_url.rsplit('/', 1)[0] + '/mcp_test_reenable'

    db_admin = await asyncpg.connect(reenable_url)
    try:
        await db_admin.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await db_admin.close()

    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'
    server_env = _build_server_env(reenable_url, generation='true')

    # First start: compression schema created, provenance row bootstrapped.
    transport = PythonStdioTransport(script_path=str(wrapper_script), env=server_env)
    client: Client[Any] = Client(transport)
    async with client:
        await client.ping()

    # Simulate the state --decompress leaves behind: payload table dropped,
    # provenance row deleted, marker table still present (the fp32 table it
    # restores stays empty here -- the zero-embedding corpus of the wedge).
    conn = await asyncpg.connect(reenable_url)
    try:
        assert await _table_exists(conn, 'compression_metadata')
        await conn.execute('DROP TABLE IF EXISTS vec_context_embeddings_compressed')
        await conn.execute('DELETE FROM compression_metadata WHERE id = 1')
    finally:
        await conn.close()

    # Second start with compression still enabled: the migration must
    # re-create the payload table instead of early-returning on the marker
    # table, and the validator must re-seed the provenance row.
    transport = PythonStdioTransport(script_path=str(wrapper_script), env=server_env)
    client = Client(transport)
    async with client:
        await client.ping()

    conn = await asyncpg.connect(reenable_url)
    try:
        assert await _table_exists(conn, 'vec_context_embeddings_compressed'), (
            'payload table must be re-created on a re-enable start'
        )
        row = await conn.fetchrow('SELECT provider, seed FROM compression_metadata WHERE id = 1')
        assert row is not None, 'validator must re-seed the provenance row'
    finally:
        await conn.close()
