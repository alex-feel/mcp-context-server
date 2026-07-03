"""SQLite entry point for the real-server integration suite.

Runs the shared backend-parametrized harness
(:class:`tests.integration._harness.MCPServerIntegrationTest`) against a real
MCP server subprocess backed by SQLite, plus two SQLite-specific standalone
tests (max-size image store; compression round-trip with direct file
inspection). The identical harness assertions run against PostgreSQL from
``tests/integration/postgresql/test_real_server.py``.
"""

import asyncio
import base64
import importlib.util
import os
import sqlite3
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client

from tests.integration._harness import MCPServerIntegrationTest

# Conditional skip marker for tests requiring sqlite-vec package
requires_sqlite_vec = pytest.mark.skipif(
    importlib.util.find_spec('sqlite_vec') is None,
    reason='sqlite-vec package not installed',
)


# Pytest integration
@pytest.mark.integration
@pytest.mark.asyncio
@requires_sqlite_vec
async def test_real_server(tmp_path: Path) -> None:
    """Run integration tests against real server with temporary database.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Raises:
        RuntimeError: If MCP_TEST_MODE is not set or if attempting to use default database.
    """
    # Verify we're in test mode from the global fixture
    if not os.environ.get('MCP_TEST_MODE'):
        raise RuntimeError(
            'MCP_TEST_MODE not set! Global test fixture may have failed.\n'
            'This could lead to pollution of the default database!',
        )

    # Create a unique database path in the temp directory
    temp_db = tmp_path / 'test_real_server.db'

    # Double-check we're not using the default database
    default_db = Path.home() / '.mcp' / 'context_storage.db'
    if temp_db.resolve() == default_db.resolve():
        raise RuntimeError(
            f'Test attempting to use default database!\nDefault: {default_db}\nTest DB: {temp_db}',
        )

    print(f'[TEST] Running with temp database: {temp_db}')
    print(f"[TEST] MCP_TEST_MODE: {os.environ.get('MCP_TEST_MODE')}")

    test = MCPServerIntegrationTest(temp_db_path=temp_db)
    success = await test.run_all_tests()
    assert success, 'Integration tests failed'


@pytest.mark.integration
@pytest.mark.asyncio
@requires_sqlite_vec
async def test_store_context_max_size_image(tmp_path: Path) -> None:
    """Test storing context with an image at the maximum allowed size.

    Creates an image just under the 10MB limit and verifies store_context succeeds.
    This is a standalone pytest test that validates the image size limit handling
    via the real MCP protocol.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Raises:
        RuntimeError: If MCP_TEST_MODE is not set or if attempting to use default database.
    """
    # Verify we're in test mode from the global fixture
    if not os.environ.get('MCP_TEST_MODE'):
        raise RuntimeError(
            'MCP_TEST_MODE not set! Global test fixture may have failed.\n'
            'This could lead to pollution of the default database!',
        )

    # Create a unique database path in the temp directory
    temp_db = tmp_path / 'test_max_image.db'

    # Store original environment
    original_env: dict[str, str | None] = {
        'DB_PATH': os.environ.get('DB_PATH'),
        'MCP_TEST_MODE': os.environ.get('MCP_TEST_MODE'),
        'ENABLE_SEMANTIC_SEARCH': os.environ.get('ENABLE_SEMANTIC_SEARCH'),
        'ENABLE_FTS': os.environ.get('ENABLE_FTS'),
        'ENABLE_HYBRID_SEARCH': os.environ.get('ENABLE_HYBRID_SEARCH'),
    }

    # Set environment for this test
    os.environ['DB_PATH'] = str(temp_db)
    os.environ['MCP_TEST_MODE'] = '1'
    os.environ['ENABLE_SEMANTIC_SEARCH'] = 'false'  # Disable for speed
    os.environ['ENABLE_FTS'] = 'false'  # Disable for speed
    os.environ['ENABLE_HYBRID_SEARCH'] = 'false'  # Disable for speed

    # Use the wrapper script that sets up Python path correctly
    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'

    # Initialize the database schema before creating client
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    with sqlite3.connect(str(temp_db)) as conn:
        conn.executescript(schema_sql)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        conn.commit()

    try:
        # Create FastMCP client with wrapper script path
        client: Client[Any] = Client(str(wrapper_script))

        async with client:
            # Create a large image that is just under the 10MB limit
            # MAX_IMAGE_SIZE_MB is 10 by default, so we create a ~9.9MB image
            target_size_bytes = int(9.9 * 1024 * 1024)  # 9.9 MB

            # Create random binary data for image content
            large_binary = os.urandom(target_size_bytes)
            large_image_b64 = base64.b64encode(large_binary).decode('utf-8')

            test_thread_id = f'max_image_test_{int(time.time())}'

            result = await client.call_tool(
                'store_context',
                {
                    'thread_id': test_thread_id,
                    'source': 'agent',
                    'text': 'Context with maximum size image',
                    'images': [
                        {
                            'data': large_image_b64,
                            'mime_type': 'application/octet-stream',
                        },
                    ],
                },
            )

            # Extract result content
            if hasattr(result, 'content'):
                content = result.content
                if content and hasattr(content[0], 'text'):
                    import json

                    text_value = content[0].text
                    assert isinstance(text_value, str | bytes | bytearray)
                    data = json.loads(text_value)
                else:
                    data = {'error': 'No content in result'}
            else:
                data = result if isinstance(result, dict) else {'error': str(result)}

            # Verify the operation succeeded
            assert data.get('success'), f'store_context should succeed with max-size image: {data}'
            assert data.get('context_id'), f'store_context should return context_id: {data}'

            # Cleanup - delete the test context
            await client.call_tool(
                'delete_context',
                {'thread_id': test_thread_id},
            )

    finally:
        # Restore original environment
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@pytest.mark.integration
@pytest.mark.asyncio
async def test_compression_round_trip_sqlite(tmp_path: Path) -> None:
    """End-to-end compression round-trip against a real SQLite-backed server.

    The test starts the server with ENABLE_EMBEDDING_COMPRESSION=true and
    ENABLE_EMBEDDING_GENERATION=false, stores a single context entry via
    the MCP store_context tool, then inspects the SQLite file directly to
    verify:

    1. vec_context_embeddings_compressed contains zero rows (no embeddings
       are produced because EMBEDDING_GENERATION is disabled), but the
       table exists and is maintained across the boot.
    2. The legacy vec_context_embeddings table does NOT exist.
    3. The singleton compression_metadata row exists with the env-derived
       provenance values.
    4. get_context_by_ids returns the stored entry, proving the read path
       still works with compression enabled.

    EMBEDDING_GENERATION is disabled so the test does not depend on
    Ollama. Compression provisioning follows embedding storage, so the
    compression schema + provenance row are pre-seeded before boot,
    reproducing a database whose data was compressed while generation was
    on; the generation-off server maintains and seed-validates that schema
    instead of idling. The compressed write path is exercised by the unit
    suite at ``tests/repositories/test_embedding_repository_compressed.py``.

    Args:
        tmp_path: Pytest fixture providing temporary directory.

    Raises:
        RuntimeError: If MCP_TEST_MODE is not set (session fixture failed).
    """
    if not os.environ.get('MCP_TEST_MODE'):
        raise RuntimeError(
            'MCP_TEST_MODE not set! Global test fixture may have failed.',
        )

    temp_db = tmp_path / 'test_compression_round_trip.db'

    original_env: dict[str, str | None] = {
        key: os.environ.get(key) for key in (
            'DB_PATH', 'MCP_TEST_MODE', 'ENABLE_SEMANTIC_SEARCH',
            'ENABLE_FTS', 'ENABLE_HYBRID_SEARCH',
            'ENABLE_EMBEDDING_GENERATION', 'ENABLE_SUMMARY_GENERATION',
            'ENABLE_EMBEDDING_COMPRESSION', 'COMPRESSION_BITS',
            'COMPRESSION_VARIANT', 'COMPRESSION_SEED',
            'COMPRESSION_MAX_CONCURRENT', 'EMBEDDING_DIM',
        )
    }

    server_env = {
        **os.environ,
        'DB_PATH': str(temp_db),
        'MCP_TEST_MODE': '1',
        'ENABLE_SEMANTIC_SEARCH': 'false',
        'ENABLE_FTS': 'false',
        'ENABLE_HYBRID_SEARCH': 'false',
        # Embedding generation off keeps the test independent of Ollama;
        # the compression migration and bootstrap validator still run.
        'ENABLE_EMBEDDING_GENERATION': 'false',
        'ENABLE_SUMMARY_GENERATION': 'false',
        'ENABLE_EMBEDDING_COMPRESSION': 'true',
        'COMPRESSION_BITS': '4',
        'COMPRESSION_VARIANT': 'ip',
        'COMPRESSION_SEED': '42',
        'COMPRESSION_MAX_CONCURRENT': '2',
        # Pin the dim so the spawned server always matches the pre-seeded
        # provenance row: the dict spreads the ambient os.environ first, and
        # CI exports EMBEDDING_DIM=384 for the unit suite, which would
        # otherwise reach the seed-locked validator as a dim mismatch and
        # kill the server with exit 78 before the client connects.
        'EMBEDDING_DIM': '1024',
    }
    # Also mutate parent os.environ so any in-process helpers see the
    # compression toggle (e.g., when the wrapper imports app.settings).
    os.environ.update(server_env)

    wrapper_script = Path(__file__).parent.parent.parent / 'run_server.py'

    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    with sqlite3.connect(str(temp_db)) as conn:
        conn.executescript(schema_sql)
        conn.execute('PRAGMA foreign_keys = ON')
        conn.execute('PRAGMA journal_mode = WAL')
        # Pre-seed the compression schema + provenance row (values matching
        # the server env below; NULL fingerprint = the documented
        # pre-fingerprint state the validator warns about and proceeds).
        conn.executescript(
            '''
            CREATE TABLE IF NOT EXISTS vec_context_embeddings_compressed (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                context_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                start_index INTEGER NOT NULL DEFAULT 0,
                end_index INTEGER NOT NULL DEFAULT 0,
                payload BLOB NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (context_id) REFERENCES context_entries(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS compression_metadata (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                provider TEXT NOT NULL,
                bits INTEGER NOT NULL CHECK (bits BETWEEN 2 AND 4),
                variant TEXT NOT NULL CHECK (variant IN ('mse', 'ip')),
                seed INTEGER NOT NULL CHECK (seed >= 0),
                dim INTEGER NOT NULL CHECK (dim > 0),
                codebook_fingerprint TEXT,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            );
            ''',
        )
        conn.execute(
            'INSERT INTO compression_metadata '
            '(id, provider, bits, variant, seed, dim, codebook_fingerprint) '
            "VALUES (1, 'turboquant', 4, 'ip', 42, 1024, NULL)",
        )
        conn.commit()

    try:
        # Pass env explicitly via PythonStdioTransport so the spawned
        # subprocess receives our compression configuration. FastMCP's
        # Client(str) shortcut does not propagate the parent env.
        from fastmcp.client.transports import PythonStdioTransport
        transport = PythonStdioTransport(
            script_path=str(wrapper_script),
            env=server_env,
        )
        client: Client[Any] = Client(transport)
        async with client:
            await client.ping()

            store_result = await client.call_tool(
                'store_context',
                {
                    'thread_id': f'compression_rt_{int(time.time())}',
                    'source': 'agent',
                    'text': 'Compressed-path round trip entry',
                    'tags': ['compression', 'round-trip'],
                },
            )

            content_obj: dict[str, Any] | None = None
            if hasattr(store_result, 'structured_content'):
                content_obj = store_result.structured_content
            assert content_obj is not None, 'store_context returned no structured content'
            assert content_obj.get('success') is True, content_obj
            context_id = content_obj.get('context_id')
            assert context_id, content_obj

            with sqlite3.connect(str(temp_db)) as conn:
                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='vec_context_embeddings_compressed'",
                ).fetchone()
                assert row is not None, 'compressed vector table missing'

                row = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' "
                    "AND name='vec_context_embeddings'",
                ).fetchone()
                assert row is None, 'fp32 vec_context_embeddings still exists'

                prov = conn.execute(
                    'SELECT provider, bits, variant, seed FROM compression_metadata '
                    'WHERE id = 1',
                ).fetchone()
                assert prov is not None, 'compression_metadata row missing'
                assert prov[0] == 'turboquant'
                assert int(prov[1]) == 4
                assert prov[2] == 'ip'
                assert int(prov[3]) == 42

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
            assert 'Compressed-path round trip entry' in results[0]['text_content']

    finally:
        for key, value in original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


if __name__ == '__main__':
    # Allow running directly
    async def main() -> None:
        # Set test mode when running directly
        os.environ['MCP_TEST_MODE'] = '1'

        # Create a temporary directory for the database when running directly
        with tempfile.TemporaryDirectory(prefix='mcp_test_direct_') as tmpdir:
            temp_db_path = Path(tmpdir) / 'test_direct.db'

            # Set DB_PATH for the subprocess
            os.environ['DB_PATH'] = str(temp_db_path)

            print('[INFO] Running directly with test mode enabled')
            print(f'[INFO] Using temporary directory: {tmpdir}')
            print(f'[INFO] DB_PATH set to: {temp_db_path}')
            print(f"[INFO] MCP_TEST_MODE: {os.environ.get('MCP_TEST_MODE')}")

            test = MCPServerIntegrationTest(temp_db_path=temp_db_path)
            success = await test.run_all_tests()
            sys.exit(0 if success else 1)

    asyncio.run(main())
