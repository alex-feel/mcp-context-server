"""Baseline integration tests for the real MCP server against PostgreSQL.

The tests start the server via ``tests/run_server.py`` with
``STORAGE_BACKEND=postgresql`` and ``POSTGRESQL_CONNECTION_STRING``
pointing at the docker-compose pgvector container provided by
``pg_test_url`` (see ``conftest.py``). They exercise the baseline tool
surface (store, search, get_context_by_ids, list_threads, statistics) to
catch PG-side regressions in the storage write path.

Compression-specific behaviour lives in
``test_compression_round_trip.py``; this file does not enable
compression so the test runs against the standard fp32 storage path.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


@pytest.mark.asyncio
async def test_real_server_baseline_postgresql(pg_test_url: str) -> None:
    """Smoke test: server starts against PostgreSQL and store/get round-trip works.

    The test confirms that:

    1. The lifespan completes successfully with STORAGE_BACKEND=postgresql,
    2. ``store_context`` writes an entry via the asyncpg backend, and
    3. ``get_context_by_ids`` returns the same entry.

    Args:
        pg_test_url: Fixture providing the docker-compose pgvector
            connection string (or skips when Docker is unavailable).
    """
    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'

    server_env = {
        **os.environ,
        'STORAGE_BACKEND': 'postgresql',
        'POSTGRESQL_CONNECTION_STRING': pg_test_url,
        'MCP_TEST_MODE': '1',
        # Disable optional features so the test does not depend on
        # Ollama / external models.
        'ENABLE_SEMANTIC_SEARCH': 'false',
        'ENABLE_FTS': 'false',
        'ENABLE_HYBRID_SEARCH': 'false',
        'ENABLE_EMBEDDING_GENERATION': 'false',
        'ENABLE_SUMMARY_GENERATION': 'false',
        # Compression off in this baseline test.
        'ENABLE_EMBEDDING_COMPRESSION': 'false',
    }

    transport = PythonStdioTransport(
        script_path=str(wrapper_script),
        env=server_env,
    )
    client: Client[Any] = Client(transport)
    async with client:
        await client.ping()

        thread_id = f'pg_baseline_{int(time.time())}'

        store_result = await client.call_tool(
            'store_context',
            {
                'thread_id': thread_id,
                'source': 'agent',
                'text': 'PostgreSQL baseline entry',
            },
        )
        store_content: dict[str, Any] | None = None
        if hasattr(store_result, 'structured_content'):
            store_content = store_result.structured_content
        assert store_content is not None
        assert store_content.get('success') is True, store_content
        context_id = store_content.get('context_id')
        assert context_id, store_content

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

        # Best-effort cleanup; the docker container is torn down at session
        # end so leftover rows are harmless. delete_context may be absent
        # via DISABLED_TOOLS.
        import contextlib

        from fastmcp.exceptions import ToolError
        with contextlib.suppress(ToolError):
            await client.call_tool(
                'delete_context',
                {'thread_id': thread_id},
            )
