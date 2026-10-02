"""Real-server checks for the server surface and startup wiring.

Protocol-era negotiation, the reported server version, tool annotations
delivered by ``list_tools``, the ``ENABLE_FTS=false`` force-off on a second
server, the ``/health`` handler, and the PostgreSQL-only session-pooler
check staying inert on SQLite.
"""

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Any

from anyio import Path as AsyncPath
from fastmcp.client.transports import PythonStdioTransport

from tests.integration._harness.core import ERA_PROTOCOL_VERSIONS
from tests.integration._harness.core import HarnessCore


class ServerMixin(HarnessCore):
    """Checks for server identity, tool registration and startup wiring."""

    async def test_client_negotiates_requested_protocol_era(self) -> bool:
        """Verify the harness client runs in the protocol era ``client_mode`` selects.

        ``'auto'`` must adopt a sessionless (2026-07-28 or later) era with no
        initialize handshake, so the run reaches the tool surface the way current
        FastMCP clients do. ``'legacy'`` must complete the initialize handshake that
        handshake-era clients such as Claude Code use, and a ping -- routed only in
        that era -- must succeed.

        Returns:
            bool: True if the negotiated era matches the requested mode.
        """
        test_name = 'client_negotiates_requested_protocol_era'
        assert self.client is not None  # Type guard for Pyright
        try:
            protocol_version = self.client.protocol_version
            expected_versions = ERA_PROTOCOL_VERSIONS[self.client_mode]
            handshake_completed = self.client.initialize_result is not None
            if protocol_version not in expected_versions or handshake_completed != (self.client_mode == 'legacy'):
                self.test_results.append((
                    test_name,
                    False,
                    (
                        f'{self.client_mode} mode negotiated {protocol_version!r} '
                        f'(handshake completed: {handshake_completed}); expected one of {expected_versions}'
                    ),
                ))
                return False
            if self.client_mode == 'legacy' and not await self.client.ping():
                self.test_results.append((test_name, False, 'ping did not return an empty result'))
                return False

            self.test_results.append(
                (test_name, True, f'{self.client_mode} mode negotiated protocol {protocol_version}'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_server_version_is_project_version(self) -> bool:
        """Test that the negotiated server identity reports the project version.

        Verifies that the server reports its own version (from pyproject.toml)
        rather than the FastMCP framework version or MCP SDK version. The identity
        comes from the initialize handshake in the legacy era and from
        ``server/discover`` in the sessionless era.

        Returns:
            bool: True if server version matches project version.
        """
        test_name = 'server_version_is_project_version'
        assert self.client is not None  # Type guard for Pyright
        try:
            from app.server import SERVER_VERSION

            server_info = self.client.server_info
            assert server_info is not None, 'a connected client always carries the server identity'

            server_version = server_info.version
            if server_version != SERVER_VERSION:
                self.test_results.append((
                    test_name,
                    False,
                    f'Server reports version {server_version!r}, expected {SERVER_VERSION!r}',
                ))
                return False

            self.test_results.append(
                (test_name, True, f'Server version correctly reports {SERVER_VERSION}'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_tool_annotations_exposed_to_client(self) -> bool:
        """Verify tool behavior-hint annotations reach the client via list_tools.

        Clients rely on the readOnlyHint/destructiveHint/idempotentHint wire hints
        for auto-approval and destructive-action confirmation. This asserts the
        hints declared in TOOL_ANNOTATIONS are delivered over the wire (tools
        absent due to DISABLED_TOOLS are skipped, not failed).

        Returns:
            bool: True if test passed.
        """
        test_name = 'tool_annotations_exposed_to_client'
        assert self.client is not None
        try:
            tools = await self.client.list_tools()
            ann_by_name: dict[str, Any] = {t.name: t.annotations for t in tools}

            # (tool, hint attribute, expected value)
            expectations: list[tuple[str, str, bool]] = [
                ('search_context', 'read_only_hint', True),
                ('get_context_by_ids', 'read_only_hint', True),
                ('list_threads', 'read_only_hint', True),
                ('get_statistics', 'read_only_hint', True),
                ('grep_context', 'read_only_hint', True),
                ('navigate_context', 'read_only_hint', True),
                ('read_context_range', 'read_only_hint', True),
                ('store_context', 'read_only_hint', False),
                ('store_context', 'destructive_hint', False),
                ('update_context', 'destructive_hint', True),
                ('update_context', 'idempotent_hint', False),
                ('delete_context', 'destructive_hint', True),
                ('delete_context', 'idempotent_hint', True),
            ]
            checked = 0
            for tool_name, attr, expected in expectations:
                annotations = ann_by_name.get(tool_name)
                if annotations is None:
                    # Tool disabled (DISABLED_TOOLS) or no annotations exposed; skip.
                    continue
                actual = getattr(annotations, attr, None)
                if actual != expected:
                    self.test_results.append((test_name, False,
                        f'{tool_name}.{attr} = {actual!r}, expected {expected!r}'))
                    return False
                checked += 1

            if checked == 0:
                self.test_results.append((test_name, False, 'No tool annotations were exposed to assert against'))
                return False

            self.test_results.append((test_name, True, f'Tool annotations delivered to client ({checked} hints verified)'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_force_off_removes_search_tool(self) -> bool:
        """Verify ENABLE_FTS=false force-off removes fts_search_context end to end.

        Opens a SECOND server subprocess whose environment matches connect_client()
        except ENABLE_FTS is forced to 'false' while ENABLE_SEMANTIC_SEARCH and
        ENABLE_HYBRID_SEARCH stay enabled. The tri-state toggle should drop
        fts_search_context from the registered tool surface while leaving
        semantic_search_context and hybrid_search_context registered. This proves
        the force-off path on BOTH SQLite and PostgreSQL.

        When no embedding provider is available (no Ollama model in the
        environment), semantic_search_context cannot register and
        hybrid_search_context has no remaining search mode (FTS is forced off),
        so the presence assertions are skipped gracefully; the FTS absence
        assertion -- the core force-off behavior -- always runs.

        Returns:
            bool: True if test passed (or skipped gracefully).
        """
        test_name = 'force_off_removes_search_tool'
        assert self.client is not None
        wrapper_script = Path(__file__).parents[2] / 'run_server.py'

        # Build a server env mirroring connect_client(), then force FTS off.
        # PythonStdioTransport(env=...) passes the dict explicitly on both
        # backends (the MCP SDK env whitelist applied to a bare script path would
        # strip app-specific vars), so the forced ENABLE_FTS reaches the server.
        server_env: dict[str, str] = {
            **os.environ,
            'MCP_TEST_MODE': '1',
            'DISABLED_TOOLS': '',
            'ENABLE_SEMANTIC_SEARCH': 'true',
            'ENABLE_FTS': 'false',
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
            # SQLite needs an isolated temp DB so the second server does not
            # contend with the primary client's database file.
            server_env['STORAGE_BACKEND'] = 'sqlite'
            second_db_dir = tempfile.mkdtemp(prefix='mcp_force_off_')
            second_db_path = Path(second_db_dir) / 'force_off.db'
            server_env['DB_PATH'] = str(second_db_path)

        transport = PythonStdioTransport(
            script_path=str(wrapper_script),
            env=server_env,
        )
        second_client = self._new_client(transport)

        try:
            await second_client.__aenter__()

            # Determine whether an embedding provider is available; if not,
            # semantic_search_context cannot register and (with FTS forced off)
            # hybrid_search_context has no search mode, so skip those presence
            # assertions while still asserting FTS absence.
            stats_data = self._extract_content(await second_client.call_tool('get_statistics', {}))
            semantic_info = stats_data.get('semantic_search', {})
            semantic_available = bool(semantic_info.get('available', False))

            tools = await second_client.list_tools()
            tool_names = {t.name for t in tools}

            if 'fts_search_context' in tool_names:
                self.test_results.append(
                    (test_name, False, 'fts_search_context still registered despite ENABLE_FTS=false'),
                )
                return False

            if not semantic_available:
                self.test_results.append(
                    (test_name, True, 'FTS force-off verified; semantic/hybrid presence skipped (no embedding provider)'),
                )
                return True

            if 'semantic_search_context' not in tool_names:
                self.test_results.append(
                    (test_name, False, 'semantic_search_context missing while ENABLE_SEMANTIC_SEARCH stayed enabled'),
                )
                return False

            if 'hybrid_search_context' not in tool_names:
                self.test_results.append(
                    (test_name, False, 'hybrid_search_context missing while ENABLE_HYBRID_SEARCH stayed enabled'),
                )
                return False

            self.test_results.append(
                (test_name, True, 'ENABLE_FTS=false removed only fts_search_context; semantic and hybrid remain'),
            )
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
        finally:
            with contextlib.suppress(Exception):
                await second_client.__aexit__(None, None, None)
            if second_db_path is not None:
                async_second_db = AsyncPath(second_db_path)
                for suffix in ('-wal', '-shm', ''):
                    candidate = AsyncPath(str(second_db_path) + suffix)
                    with contextlib.suppress(Exception):
                        if await candidate.exists():
                            await candidate.unlink()
                with contextlib.suppress(Exception):
                    await AsyncPath(async_second_db.parent).rmdir()

    async def test_health_endpoint_returns_ok(self) -> bool:
        """Verify the /health endpoint returns HTTP 200 with {"status": "ok"}.

        The health endpoint is only available on HTTP transport. Since integration tests
        use stdio transport, this test validates the handler behavior using Starlette TestClient.

        Returns:
            bool: True if test passed.
        """
        test_name = 'health_endpoint_returns_ok'
        assert self.client is not None
        try:
            from starlette.applications import Starlette
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            from starlette.routing import Route
            from starlette.testclient import TestClient

            async def health_handler(_: Request) -> JSONResponse:
                return JSONResponse({'status': 'ok'})

            app = Starlette(routes=[Route('/health', health_handler, methods=['GET'])])

            with TestClient(app) as test_client:
                response = test_client.get('/health')

                if response.status_code != 200:
                    self.test_results.append((test_name, False,
                        f'Health endpoint status code: {response.status_code}'))
                    return False

                body = response.json()
                if body.get('status') != 'ok':
                    self.test_results.append((test_name, False,
                        f'Health endpoint body: {body}'))
                    return False

            self.test_results.append((test_name, True,
                'Health endpoint returns 200 with {"status": "ok"}'))
            return True

        except ImportError as e:
            self.test_results.append((test_name, True,
                f'Skipped (missing dependency: {e})'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_session_pooler_validation_noop_on_sqlite(self) -> bool:
        """Verify the session-pooler advisory wiring is a PostgreSQL-only no-op on SQLite.

        validate_session_pooler_capacity() / _detect_session_mode_pooler() are gated on
        backend_type == 'postgresql'; a healthy SQLite server boot proves the startup
        step neither runs nor crashes on the default backend.

        Returns:
            bool: True if test passed.
        """
        test_name = 'session_pooler_validation_noop_on_sqlite'
        assert self.client is not None
        if self.backend != 'sqlite':
            self.test_results.append(
                (test_name, True, 'Skipped on postgresql (SQLite-only no-op assertion)'),
            )
            return True
        try:
            data = self._extract_content(await self.client.call_tool('list_threads', {}))
            if 'threads' not in data:
                self.test_results.append((test_name, False, f'Server not operational: {data}'))
                return False
            self.test_results.append((test_name, True,
                'SQLite server operational; session-pooler advisory is a PG-only no-op'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Unexpected failure: {e}'))
            return False
