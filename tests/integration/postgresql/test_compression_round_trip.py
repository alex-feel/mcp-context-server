"""End-to-end compression round trip against the real PostgreSQL backend.

Spawns the MCP server with ``ENABLE_EMBEDDING_COMPRESSION=true`` and
``STORAGE_BACKEND=postgresql`` pointed at the docker-compose pgvector
container. Verifies the schema migration replaced the fp32 vec table
with the compressed-payload table, the singleton ``compression_metadata``
row exists, and the basic store/get tool surface still works.

Embedding generation is disabled so the test does not require Ollama;
the compressed write path is exercised by the unit suite under
``tests/repositories/test_embedding_repository_compressed.py``.
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
        'ENABLE_EMBEDDING_GENERATION': 'false',
        'ENABLE_SUMMARY_GENERATION': 'false',
        'ENABLE_EMBEDDING_COMPRESSION': 'true',
        'COMPRESSION_BITS': bits,
        'COMPRESSION_VARIANT': variant,
        'COMPRESSION_SEED': seed,
        'COMPRESSION_MAX_CONCURRENT': '2',
    }


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

    1. The migration replaces ``vec_context_embeddings`` with
       ``vec_context_embeddings_compressed``.
    2. The singleton ``compression_metadata`` row reflects env values.
    3. ``store_context`` + ``get_context_by_ids`` still work via MCP.

    Args:
        pg_test_url: Connection string from the session-scoped docker
            fixture.
    """
    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'
    server_env = _build_server_env(pg_test_url)

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
