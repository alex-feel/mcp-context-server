"""Real-server PostgreSQL metadata-index creation integration test.

Exercises the PostgreSQL-specific branch of ``handle_metadata_indexes``
(``app/migrations/metadata.py``) end-to-end against a REAL MCP server
subprocess. This code path has NO SQLite equivalent and no prior
real-server coverage: on PostgreSQL each configured SCALAR metadata field
gets a typed expression btree index (``idx_metadata_{field}`` over
``((metadata->>'{field}')::CAST)``), while the always-present GIN index
``idx_metadata_gin`` over ``metadata jsonb_path_ops`` (declared in
``app/schemas/postgresql_schema.sql``) serves jsonb-containment queries.

The server applies the metadata-index migration during lifespan startup
(``app/server.py`` -> ``handle_metadata_indexes``) in the default
``additive`` sync mode, which CREATES the scalar indexes named in
``METADATA_INDEXED_FIELDS``. These scalar expression indexes are NOT
declared in ``app/schemas/postgresql_schema.sql``; the sync layer is their
single source of truth, so a database initialized only by the migrate CLI
carries none of them until first server startup provisions them. The GIN
index ``idx_metadata_gin`` is the one metadata index still declared in the
base schema, because it is always required and not sync-managed. The scalar
index identifiers are emitted quoted (``"idx_metadata_{field}"``); for the
lowercase field names used here quoting yields the identical catalog name.
After the server starts, the test connects to the isolated database with
asyncpg and asserts ``pg_indexes`` contains the expected typed expression
indexes and the GIN index, then drives a real ``search_context``
``metadata_filters`` query through the MCP client to confirm the indexed
fields return correct rows end-to-end.

The run uses an ISOLATED PostgreSQL database (created fresh + dropped at
teardown) so it never pollutes the shared ``mcp_test`` database nor the
seed-locked ``compression_metadata`` singleton other PG tests rely on.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

# Scalar fields configured for indexing. 'priority' exercises the typed-cast
# branch (::INTEGER) of _generate_create_index_postgresql; 'status' and
# 'category' exercise the plain (no cast) string branch. Array/object fields
# are deliberately omitted: metadata.py routes those to the existing GIN index
# and creates NO dedicated expression index for them.
INDEXED_FIELDS = 'priority:integer,status,category:string'

# Index name -> expected substring inside pg_indexes.indexdef. metadata.py
# names every scalar expression index idx_metadata_{field}. The integer field
# carries an explicit cast; the string fields carry none. PostgreSQL normalizes
# the indexdef expression, so assert on stable substrings rather than an exact
# match.
EXPECTED_SCALAR_INDEXES: dict[str, str] = {
    'idx_metadata_priority': "(metadata ->> 'priority'::text))::integer",
    'idx_metadata_status': "(metadata ->> 'status'::text)",
    'idx_metadata_category': "(metadata ->> 'category'::text)",
}

GIN_INDEX_NAME = 'idx_metadata_gin'


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database-name path component replaced."""
    parts = urlsplit(pg_url)
    return urlunsplit((
        parts.scheme,
        parts.netloc,
        f'/{new_db}',
        parts.query,
        parts.fragment,
    ))


@pytest_asyncio.fixture
async def isolated_metadata_index_db(pg_test_url: str) -> AsyncIterator[str]:
    """Provision an isolated PostgreSQL database for the metadata-index run.

    Creates a fresh database, installs the pgvector extension inside it, yields
    the DSN, and drops the database (FORCE, terminating any lingering pool
    connections) at teardown. Isolation keeps this test independent of the
    shared ``mcp_test`` database mutated by other PG tests.

    Args:
        pg_test_url: docker-compose pgvector DSN (skips when Docker is absent).

    Yields:
        Connection string pointing at the isolated metadata-index database.
    """
    db_name = 'mcp_metadata_indexing_e2e'
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {db_name} WITH (FORCE)')
        await admin.execute(f'CREATE DATABASE {db_name}')
    finally:
        await admin.close()

    target_url = _replace_db_name(pg_test_url, db_name)
    setup = await asyncpg.connect(target_url)
    try:
        await setup.execute('CREATE EXTENSION IF NOT EXISTS vector')
    finally:
        await setup.close()

    yield target_url

    admin = await asyncpg.connect(pg_test_url)
    try:
        with contextlib.suppress(Exception):
            await admin.execute(f'DROP DATABASE IF EXISTS {db_name} WITH (FORCE)')
    finally:
        await admin.close()


def _extract_content(result: object) -> dict[str, Any]:
    """Extract the structured payload from a FastMCP CallToolResult.

    Mirrors the extraction logic in :mod:`tests.integration._harness`: search
    responses carry ``results`` + ``count`` directly, while ``store_context``
    returns a flat dict with ``success``.

    Args:
        result: CallToolResult object returned by ``client.call_tool``.

    Returns:
        The tool's structured-content dict.
    """
    content = getattr(result, 'structured_content', None)
    if isinstance(content, dict):
        if 'results' in content and 'count' in content:
            return {'success': True, **content} if 'success' not in content else content
        return content
    return {'success': False, 'error': 'Unable to extract content from result'}


async def _query_pg_indexes(pg_url: str) -> dict[str, str]:
    """Return a mapping of index name -> indexdef for context_entries.

    All asyncpg I/O happens inside this single coroutine so the connection
    stays bound to one event loop.

    Args:
        pg_url: PostgreSQL connection string for the isolated database.

    Returns:
        Dict mapping ``indexname`` to its ``indexdef`` for every index on
        ``context_entries`` in the ``public`` schema.
    """
    conn = await asyncpg.connect(pg_url)
    try:
        rows = await conn.fetch(
            "SELECT indexname, indexdef FROM pg_indexes "
            "WHERE tablename = 'context_entries' AND schemaname = 'public'",
        )
    finally:
        await conn.close()
    return {row['indexname']: row['indexdef'] for row in rows}


@pytest.mark.asyncio
async def test_metadata_indexing_real_server_postgresql(
    isolated_metadata_index_db: str,
) -> None:
    """Real server creates typed expression + GIN metadata indexes on PG.

    Launches the real MCP server against an isolated pgvector database with
    ``METADATA_INDEXED_FIELDS`` configured for scalar fields, then asserts the
    PostgreSQL-only metadata indexes the lifespan migration creates, and that a
    ``metadata_filters`` search over the indexed fields returns correct rows.

    Args:
        isolated_metadata_index_db: Isolated pgvector DSN from the fixture.
    """
    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'

    # The MCP SDK env whitelist strips app-specific vars when env=None, so the
    # backend routing, the indexed-field config, and the feature toggles MUST be
    # passed explicitly. ENABLE_EMBEDDING_GENERATION=false keeps run_server.py
    # from probing Ollama; the other toggles keep startup fast and avoid model
    # dependencies. METADATA_INDEX_SYNC_MODE is left at its default 'additive',
    # which creates the missing scalar indexes.
    server_env = {
        **os.environ,
        'STORAGE_BACKEND': 'postgresql',
        'POSTGRESQL_CONNECTION_STRING': isolated_metadata_index_db,
        'MCP_TEST_MODE': '1',
        'DISABLED_TOOLS': '',
        'METADATA_INDEXED_FIELDS': INDEXED_FIELDS,
        'ENABLE_EMBEDDING_GENERATION': 'false',
        'ENABLE_SEMANTIC_SEARCH': 'false',
        'ENABLE_FTS': 'false',
        'ENABLE_HYBRID_SEARCH': 'false',
        'ENABLE_SUMMARY_GENERATION': 'false',
        'ENABLE_EMBEDDING_COMPRESSION': 'false',
    }

    transport = PythonStdioTransport(
        script_path=str(wrapper_script),
        env=server_env,
    )
    client: Client[Any] = Client(transport)
    await client.__aenter__()
    try:
        await client.ping()

        # The server lifespan has now applied handle_metadata_indexes. Verify
        # the PG-specific index set directly against the database.
        indexes = await _query_pg_indexes(isolated_metadata_index_db)

        for index_name, expected_expr in EXPECTED_SCALAR_INDEXES.items():
            assert index_name in indexes, (
                f'Expected scalar expression index {index_name!r} missing. '
                f'Present indexes: {sorted(indexes)}'
            )
            indexdef = indexes[index_name].lower()
            assert expected_expr.lower() in indexdef, (
                f'Index {index_name!r} indexdef does not contain expected '
                f'expression {expected_expr!r}. Actual: {indexes[index_name]}'
            )

        # The integer field's index MUST carry the typed cast; the string
        # fields' indexes MUST NOT (plain text expression branch).
        assert '::integer' in indexes['idx_metadata_priority'].lower()
        assert '::integer' not in indexes['idx_metadata_status'].lower()
        assert '::integer' not in indexes['idx_metadata_category'].lower()

        # The always-present GIN index (schema DDL, jsonb_path_ops) must exist.
        assert GIN_INDEX_NAME in indexes, (
            f'GIN index {GIN_INDEX_NAME!r} missing. Present: {sorted(indexes)}'
        )
        assert 'gin' in indexes[GIN_INDEX_NAME].lower()

        # End-to-end: store entries whose metadata uses the indexed scalar
        # fields, then confirm metadata_filters returns the correct rows.
        thread_id = 'pg-metadata-index-e2e'
        entries = [
            {'status': 'active', 'priority': 9, 'category': 'backend'},
            {'status': 'active', 'priority': 3, 'category': 'frontend'},
            {'status': 'failed', 'priority': 7, 'category': 'backend'},
        ]
        for i, meta in enumerate(entries):
            store_result = await client.call_tool(
                'store_context',
                {
                    'thread_id': thread_id,
                    'source': 'agent',
                    'text': f'metadata index entry {i}',
                    'metadata': meta,
                },
            )
            store_data = _extract_content(store_result)
            assert store_data.get('success'), f'store_context failed: {store_data}'

        # Scalar-equality on an indexed string field (->> branch).
        active_result = await client.call_tool(
            'search_context',
            {
                'thread_id': thread_id,
                'limit': 50,
                'metadata': {'status': 'active'},
            },
        )
        active_data = _extract_content(active_result)
        assert len(active_data.get('results', [])) == 2, (
            f"Expected 2 'active' rows, got {active_data.get('results')}"
        )

        # Typed numeric comparison on the integer-indexed field (::NUMERIC/cast
        # branch in query_builder for the gt operator).
        high_priority_result = await client.call_tool(
            'search_context',
            {
                'thread_id': thread_id,
                'limit': 50,
                'metadata_filters': [
                    {'key': 'priority', 'operator': 'gt', 'value': 5},
                ],
            },
        )
        high_priority_data = _extract_content(high_priority_result)
        assert len(high_priority_data.get('results', [])) == 2, (
            f"Expected 2 rows with priority>5, got {high_priority_data.get('results')}"
        )

        # Combined scalar filter on two indexed fields.
        combined_result = await client.call_tool(
            'search_context',
            {
                'thread_id': thread_id,
                'limit': 50,
                'metadata': {'category': 'backend'},
                'metadata_filters': [
                    {'key': 'priority', 'operator': 'gte', 'value': 8},
                ],
            },
        )
        combined_data = _extract_content(combined_result)
        assert len(combined_data.get('results', [])) == 1, (
            f"Expected 1 backend row with priority>=8, got {combined_data.get('results')}"
        )
    finally:
        with contextlib.suppress(Exception):
            await client.__aexit__(None, None, None)
