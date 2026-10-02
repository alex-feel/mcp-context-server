"""Shared plumbing of the real-server integration harness.

``HarnessCore`` owns the per-run state: the server environment for each
backend, the FastMCP client and the second-server helper, image and result
extraction, and the result list every check reports into. ``ClientMode``,
``CLIENT_MODES`` and ``ERA_PROTOCOL_VERSIONS`` name the protocol eras a
harness client negotiates.
"""

import base64
import contextlib
import os
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from typing import Literal

from anyio import Path as AsyncPath
from fastmcp import Client
from fastmcp.client.transports import ClientTransport
from fastmcp.client.transports import PythonStdioTransport
from mcp.client.stdio import get_default_environment
from mcp.types.version import HANDSHAKE_PROTOCOL_VERSIONS
from mcp.types.version import MODERN_PROTOCOL_VERSIONS

# Protocol era a harness client negotiates. 'auto' adopts the sessionless
# 2026-07-28 era (no initialize handshake) over stdio and streamable HTTP;
# 'legacy' forces the initialize handshake that handshake-era clients such as
# Claude Code use. The entry points run the full harness once per mode, and
# ERA_PROTOCOL_VERSIONS lists the protocol versions each mode may negotiate.
type ClientMode = Literal['auto', 'legacy']
CLIENT_MODES: tuple[ClientMode, ...] = ('auto', 'legacy')
ERA_PROTOCOL_VERSIONS: dict[ClientMode, tuple[str, ...]] = {
    'auto': MODERN_PROTOCOL_VERSIONS,
    'legacy': HANDSHAKE_PROTOCOL_VERSIONS,
}


class HarnessCore:
    """Server, client and result plumbing shared by every check mixin."""

    def __init__(
        self,
        temp_db_path: Path | None = None,
        *,
        backend: str = 'sqlite',
        pg_url: str | None = None,
        client_mode: ClientMode = 'auto',
    ) -> None:
        """Initialize the integration test suite.

        Args:
            temp_db_path: Optional path to a temporary SQLite database.
            backend: Storage backend to exercise ('sqlite' or 'postgresql').
            pg_url: PostgreSQL connection string; required when
                ``backend='postgresql'``.
            client_mode: Protocol era every harness client negotiates
                (see :data:`ClientMode`).
        """
        self.client: Client[Any] | None = None
        self.test_results: list[tuple[str, bool, str]] = []
        self.test_thread_id = f'integration_test_{int(time.time())}'
        self.temp_db_path = temp_db_path
        self.backend = backend
        self.pg_url = pg_url
        self.client_mode: ClientMode = client_mode
        self.registered_tools: frozenset[str] = frozenset()

    def _new_client(self, transport: ClientTransport) -> Client[Any]:
        """Build a client that negotiates the harness's protocol era.

        Every harness connection, including the short-lived second servers some
        tests start, is built here, so one run exercises the tool surface in
        exactly the era ``client_mode`` selects.

        Args:
            transport: The transport to connect through.

        Returns:
            An unconnected client.
        """
        return Client(transport, mode=self.client_mode)

    async def start_server(self) -> bool:
        """Start the MCP server via FastMCP Client.

        Returns:
            bool: True (server starts automatically with Client).
        """
        print('[OK] Server will be started by FastMCP Client')
        return True

    async def connect_client(self) -> bool:
        """Connect FastMCP client to server.

        Returns:
            bool: True if client connected successfully.

        Raises:
            RuntimeError: If attempting to use default database in test mode.
        """
        try:
            # Use the wrapper script that sets up Python path correctly
            wrapper_script = Path(__file__).parents[2] / 'run_server.py'
            print(f'[INFO] Connecting to server via wrapper: {wrapper_script}')

            if self.backend == 'postgresql':
                # PostgreSQL path: the server auto-initializes its schema on
                # startup, so there is no SQLite pre-init step. The backend and
                # DSN MUST be passed explicitly via PythonStdioTransport(env=...)
                # because a bare script path applies the MCP SDK env whitelist
                # that strips app-specific vars. Inherit the parent env (carries
                # EMBEDDING_MODEL / EMBEDDING_DIM in CI) and override routing.
                server_env = {
                    **os.environ,
                    'STORAGE_BACKEND': 'postgresql',
                    'POSTGRESQL_CONNECTION_STRING': self.pg_url or '',
                    'MCP_TEST_MODE': '1',
                    # Neutralize any DISABLED_TOOLS inherited from the developer's
                    # shell / .env. The SQLite path starts from the MCP SDK's
                    # default stdio environment, which carries no DISABLED_TOOLS,
                    # so it runs the full tool surface; PG must match for true
                    # parity (an inherited DISABLED_TOOLS would skip whole tools).
                    'DISABLED_TOOLS': '',
                    'ENABLE_SEMANTIC_SEARCH': 'true',
                    'ENABLE_FTS': 'true',
                    'ENABLE_HYBRID_SEARCH': 'true',
                    # Run the parity suite against the fp32 vector layout. This
                    # deliberately exercises the pgvector <-> / HNSW / DISTINCT ON
                    # path (high-value PG divergence not covered by the compressed
                    # tests) and keeps the run independent of the seed-locked
                    # compression_metadata singleton sealed by the dedicated
                    # compression round-trip tests.
                    'ENABLE_EMBEDDING_COMPRESSION': 'false',
                    'SUMMARY_OPENAI_REASONING_EFFORT': 'low',
                    'SUMMARY_ANTHROPIC_EFFORT': 'low',
                }
                transport = PythonStdioTransport(
                    script_path=str(wrapper_script),
                    env=server_env,
                )
                self.client = self._new_client(transport)
                await self.client.__aenter__()
                self.registered_tools = frozenset(tool.name for tool in await self.client.list_tools())
                print(f'[OK] Client connected successfully (postgresql, {self.client_mode} mode)')
                return True

            # SQLite path: the environment is passed explicitly so the server
            # opens this test's own database. It starts from the MCP SDK's
            # default stdio environment, the same clean base a bare script path
            # gets, so nothing else from the developer's shell reaches the server.
            if self.temp_db_path is None:
                raise RuntimeError('The SQLite harness requires temp_db_path')
            default_db = Path.home() / '.mcp' / 'context_storage.db'
            if self.temp_db_path.resolve() == default_db.resolve():
                raise RuntimeError(
                    'CRITICAL: Attempting to use default database in test!\n'
                    f'Default: {default_db}\n'
                    f'Current: {self.temp_db_path}',
                )
            server_env = {
                **get_default_environment(),
                'STORAGE_BACKEND': 'sqlite',
                'DB_PATH': str(self.temp_db_path),
                'MCP_TEST_MODE': '1',
                'ENABLE_SEMANTIC_SEARCH': 'true',
                'ENABLE_FTS': 'true',
                'ENABLE_HYBRID_SEARCH': 'true',
                'SUMMARY_OPENAI_REASONING_EFFORT': 'low',
                'SUMMARY_ANTHROPIC_EFFORT': 'low',
            }
            print(f'[INFO] Using temporary database: {self.temp_db_path}')
            transport = PythonStdioTransport(script_path=str(wrapper_script), env=server_env)
            self.client = self._new_client(transport)

            # Connect to server
            await self.client.__aenter__()

            # Confirm the server answers a request in the negotiated era and record
            # the tools it registered; feature-gated checks follow this list
            self.registered_tools = frozenset(tool.name for tool in await self.client.list_tools())

            print(f'[OK] Client connected successfully ({self.client_mode} mode)')
            return True

        except Exception as e:
            print(f'[ERROR] Failed to connect client: {e}')
            import traceback

            traceback.print_exc()
            return False

    def _create_test_image(self) -> str:
        """Create a small test image as base64.

        Returns:
            str: Base64 encoded test image.
        """
        # Create a simple 1x1 pixel PNG image
        png_header = b'\x89PNG\r\n\x1a\n'
        ihdr = b'\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89'
        idat = b'\x00\x00\x00\x0bIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05W\xbf\xaa\xd4'
        iend = b'\x00\x00\x00\x00IEND\xaeB`\x82'
        png_data = png_header + ihdr + idat + iend
        return base64.b64encode(png_data).decode('utf-8')

    def _extract_content(self, result: object) -> dict[str, Any]:
        """Extract content from FastMCP CallToolResult.

        Args:
            result: CallToolResult object from FastMCP.

        Returns:
            dict: The actual result content.
        """
        # FastMCP CallToolResult has structured_content attribute
        content = getattr(result, 'structured_content', None)
        if content is not None:
            if isinstance(content, dict):
                # Handle wrapped results
                if 'result' in content:
                    if isinstance(content['result'], list):
                        return {'success': True, 'results': content['result']}
                    if isinstance(content['result'], dict):
                        return content['result']
                # Special handling for search responses - return full content as-is
                # (search_context, semantic_search, fts_search, hybrid_search all return 'results' and 'count')
                if 'results' in content and 'count' in content:
                    # Add success=True if not present, preserve all other fields (error, stats, etc.)
                    if 'success' not in content:
                        return {'success': True, **content}
                    return content
                # Special handling for list_threads - it returns threads directly
                if 'threads' in content:
                    return {'success': True, 'threads': content['threads'], 'total_threads': content.get('total_threads', 0)}
                # Special handling for get_statistics - it returns stats directly
                if 'total_entries' in content:
                    return {'success': True, **content}  # Include all statistics fields
                # Direct dict results
                return content
            # List results
            if isinstance(content, list):
                return {'success': True, 'results': content}

        # Should not reach here with current FastMCP, but return error for safety
        return {'success': False, 'error': 'Unable to extract content from result'}

    @contextlib.asynccontextmanager
    async def _second_server(self, extra_env: dict[str, str]) -> AsyncIterator[Client[Any]]:
        """Run a SECOND server subprocess carrying extra environment overrides.

        Some behavior is decided once, at server startup -- a feature toggle, a pool
        timeout, the indexed-metadata configuration -- so it cannot be exercised against
        the long-lived server ``connect_client`` connected to. The overrides are applied
        on top of that same environment, and backend routing matches the harness's own:
        PostgreSQL reuses the shared test database, while SQLite gets an isolated
        temporary file so the second server never contends with the primary client's
        database.

        ``PythonStdioTransport(env=...)`` passes the dict explicitly on both backends,
        because the MCP SDK env whitelist applied to a bare script path strips
        app-specific variables.

        Args:
            extra_env: Environment overrides applied last, so they win over the
                harness defaults.

        Yields:
            A connected client for the second server.
        """
        wrapper_script = Path(__file__).parents[2] / 'run_server.py'
        server_env: dict[str, str] = {
            **os.environ,
            'MCP_TEST_MODE': '1',
            'DISABLED_TOOLS': '',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'ENABLE_FTS': 'true',
            'ENABLE_HYBRID_SEARCH': 'true',
            'ENABLE_EMBEDDING_COMPRESSION': 'false',
            'SUMMARY_OPENAI_REASONING_EFFORT': 'low',
            'SUMMARY_ANTHROPIC_EFFORT': 'low',
        }
        second_db_path: Path | None = None
        if self.backend == 'postgresql':
            server_env['STORAGE_BACKEND'] = 'postgresql'
            server_env['POSTGRESQL_CONNECTION_STRING'] = self.pg_url or ''
        else:
            server_env['STORAGE_BACKEND'] = 'sqlite'
            second_db_dir = tempfile.mkdtemp(prefix='mcp_harness_second_')
            second_db_path = Path(second_db_dir) / 'second_server.db'
            server_env['DB_PATH'] = str(second_db_path)
        server_env.update(extra_env)

        transport = PythonStdioTransport(script_path=str(wrapper_script), env=server_env)
        second_client = self._new_client(transport)
        try:
            await second_client.__aenter__()
            await second_client.list_tools()
            yield second_client
        finally:
            with contextlib.suppress(Exception):
                await second_client.__aexit__(None, None, None)
            if second_db_path is not None:
                for suffix in ('-wal', '-shm', ''):
                    candidate = AsyncPath(str(second_db_path) + suffix)
                    with contextlib.suppress(Exception):
                        if await candidate.exists():
                            await candidate.unlink()
                with contextlib.suppress(Exception):
                    await AsyncPath(second_db_path.parent).rmdir()

    async def cleanup(self) -> None:
        """Clean up server and resources."""
        try:
            # Disconnect client (this also stops the server subprocess)
            if self.client:
                await self.client.__aexit__(None, None, None)
                print('[OK] Client disconnected and server stopped')

            # Clean up temporary database file if it exists
            if self.temp_db_path:
                async_temp_db_path = AsyncPath(self.temp_db_path)
                if await async_temp_db_path.exists():
                    try:
                        # Remove WAL and SHM files if they exist
                        wal_file = AsyncPath(str(self.temp_db_path) + '-wal')
                        shm_file = AsyncPath(str(self.temp_db_path) + '-shm')
                        if await wal_file.exists():
                            await wal_file.unlink()
                        if await shm_file.exists():
                            await shm_file.unlink()
                        # Remove main database file
                        await async_temp_db_path.unlink()
                        print(f'[OK] Temporary database cleaned up: {self.temp_db_path}')
                    except Exception as cleanup_err:
                        print(f'[WARNING] Could not clean up temp database: {cleanup_err}')

        except Exception as e:
            print(f'[WARNING] Cleanup error: {e}')
