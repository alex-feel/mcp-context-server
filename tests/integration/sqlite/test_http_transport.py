"""End-to-end HTTP transport tests for the real MCP server (SQLite backend).

The rest of the integration suite drives the server over stdio, so two
user-facing HTTP behaviors have no end-to-end coverage:

1. Bearer-token authentication (``MCP_AUTH_PROVIDER=simple_token`` plus
   ``MCP_AUTH_TOKEN``). It is wired only on HTTP transports (``app/server.py``
   ``main()`` calls ``create_auth_provider()`` when ``transport != 'stdio'``)
   and is otherwise only unit-tested against ``SimpleTokenVerifier`` directly.
2. The real ``/health`` route registered via ``mcp.custom_route('/health', ...)``
   for non-stdio transports. The existing harness ``test_health_endpoint_returns_ok``
   builds its own throwaway Starlette app and never hits the live route.

These tests launch the actual server as an HTTP server by spawning
``tests/run_server.py`` via ``subprocess.Popen`` with an explicit environment
(the MCP SDK strips app-specific env vars when ``env=None``, so the full env is
passed as a dict). Embeddings/semantic/FTS/summary/compression are disabled for
fast startup. Each test picks a free ephemeral loopback port and terminates the
subprocess in a ``finally`` block so no orphan server or port binding leaks.
"""

import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

# The HTTP transport mode the project exposes. main() registers /health and
# wires auth for every non-stdio transport; 'http' maps to FastMCP's
# streamable-http MCP endpoint mounted at /mcp.
HTTP_TRANSPORT = 'http'
TEST_TOKEN = 'integration-secret-token-123'

# run_server.py wrapper that configures sys.path and test mode, then calls main().
WRAPPER_SCRIPT = Path(__file__).parent.parent.parent / 'run_server.py'


def _free_port() -> int:
    """Return an unused TCP port on the loopback interface."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def _build_env(*, db_path: Path, port: int, auth: bool) -> dict[str, str]:
    """Build the explicit subprocess environment for a fast-startup HTTP server.

    The MCP SDK whitelists env vars when spawning subprocesses, so all
    app-specific configuration is passed explicitly. Generation features are
    disabled so the server starts without Ollama/LLM dependencies.

    Args:
        db_path: Temporary SQLite database path for this server instance.
        port: Loopback TCP port the HTTP server should bind.
        auth: When True, enable simple_token bearer auth with TEST_TOKEN.

    Returns:
        A complete environment dict for ``subprocess.Popen``.
    """
    env = {
        **os.environ,
        'STORAGE_BACKEND': 'sqlite',
        'DB_PATH': str(db_path),
        'MCP_TEST_MODE': '1',
        'MCP_TRANSPORT': HTTP_TRANSPORT,
        'FASTMCP_HOST': '127.0.0.1',
        'FASTMCP_PORT': str(port),
        # Disable all generation/search subsystems for fast, dependency-free startup.
        'ENABLE_EMBEDDING_GENERATION': 'false',
        'ENABLE_SEMANTIC_SEARCH': 'false',
        'ENABLE_FTS': 'false',
        'ENABLE_HYBRID_SEARCH': 'false',
        'ENABLE_SUMMARY_GENERATION': 'false',
        'ENABLE_EMBEDDING_COMPRESSION': 'false',
        # Avoid noisy rich logging in subprocess output.
        'FASTMCP_ENABLE_RICH_LOGGING': 'false',
    }
    if auth:
        env['MCP_AUTH_PROVIDER'] = 'simple_token'
        env['MCP_AUTH_TOKEN'] = TEST_TOKEN
    else:
        env['MCP_AUTH_PROVIDER'] = 'none'
        env.pop('MCP_AUTH_TOKEN', None)
    return env


def _wait_for_health(base_url: str, proc: 'subprocess.Popen[bytes]', timeout_s: float = 60.0) -> None:
    """Poll ``GET {base_url}/health`` until it returns HTTP 200.

    Args:
        base_url: Server origin, e.g. ``http://127.0.0.1:8123``.
        proc: The running server subprocess (checked for premature exit).
        timeout_s: Maximum time to wait for readiness.

    Raises:
        RuntimeError: If the subprocess exits before becoming ready.
        TimeoutError: If the server does not become healthy in time.
    """
    deadline = time.time() + timeout_s
    last_exc: Exception | None = None
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(
                f'Server subprocess exited prematurely with code {proc.returncode} '
                'before /health became ready',
            )
        try:
            resp = httpx.get(f'{base_url}/health', timeout=2.0)
            if resp.status_code == 200:
                return
        except httpx.HTTPError as exc:
            last_exc = exc
        time.sleep(0.25)
    raise TimeoutError(
        f'Server at {base_url} did not become healthy within {timeout_s}s '
        f'(last error: {last_exc!r})',
    )


def _terminate(proc: 'subprocess.Popen[bytes]') -> None:
    """Terminate the server subprocess, escalating to kill, leaving no orphan.

    Args:
        proc: The server subprocess to stop.
    """
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=10)


@pytest.fixture
def http_auth_server(tmp_path: Path) -> Iterator[str]:
    """Launch a real HTTP server with bearer-token auth; yield its base URL.

    Spawns ``tests/run_server.py`` with an explicit env (auth enabled), waits
    for ``/health`` readiness, yields the loopback origin, and terminates the
    subprocess in teardown so no orphan process or port binding leaks.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    port = _free_port()
    base_url = f'http://127.0.0.1:{port}'
    env = _build_env(db_path=tmp_path / 'http_auth.db', port=port, auth=True)
    proc = subprocess.Popen(
        [sys.executable, str(WRAPPER_SCRIPT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(base_url, proc)
        yield base_url
    finally:
        _terminate(proc)


@pytest.fixture
def http_noauth_server(tmp_path: Path) -> Iterator[str]:
    """Launch a real HTTP server WITHOUT auth; yield its base URL.

    Used by the ``/health`` test: the health endpoint is unauthenticated even
    when auth is on, but a no-auth server keeps that test independent of the
    auth configuration under test.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    port = _free_port()
    base_url = f'http://127.0.0.1:{port}'
    env = _build_env(db_path=tmp_path / 'http_health.db', port=port, auth=False)
    proc = subprocess.Popen(
        [sys.executable, str(WRAPPER_SCRIPT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health(base_url, proc)
        yield base_url
    finally:
        _terminate(proc)


def _initialize_payload() -> dict[str, object]:
    """Return a minimal MCP ``initialize`` JSON-RPC request body."""
    return {
        'jsonrpc': '2.0',
        'id': 1,
        'method': 'initialize',
        'params': {
            'protocolVersion': '2025-06-18',
            'capabilities': {},
            'clientInfo': {'name': 'http-transport-test', 'version': '0.0.0'},
        },
    }


@pytest.mark.integration
def test_http_request_without_auth_header_is_rejected(http_auth_server: str) -> None:
    """An MCP request with NO Authorization header is rejected as unauthorized."""
    resp = httpx.post(
        f'{http_auth_server}/mcp',
        json=_initialize_payload(),
        headers={'Accept': 'application/json, text/event-stream'},
        timeout=10.0,
    )
    assert resp.status_code == 401, (
        f'Expected 401 for missing Authorization header, got {resp.status_code}: '
        f'{resp.text[:300]}'
    )


@pytest.mark.integration
def test_http_request_with_wrong_token_is_rejected(http_auth_server: str) -> None:
    """An MCP request with a WRONG bearer token is rejected as unauthorized."""
    resp = httpx.post(
        f'{http_auth_server}/mcp',
        json=_initialize_payload(),
        headers={
            'Accept': 'application/json, text/event-stream',
            'Authorization': 'Bearer totally-wrong-token',
        },
        timeout=10.0,
    )
    assert resp.status_code == 401, (
        f'Expected 401 for wrong bearer token, got {resp.status_code}: '
        f'{resp.text[:300]}'
    )


@pytest.mark.integration
@pytest.mark.asyncio
async def test_http_request_with_correct_token_is_accepted(http_auth_server: str) -> None:
    """The CORRECT bearer token authenticates and can list and call tools."""
    transport = StreamableHttpTransport(
        url=f'{http_auth_server}/mcp',
        headers={'Authorization': f'Bearer {TEST_TOKEN}'},
    )
    async with Client(transport) as client:
        await client.ping()

        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}
        assert 'store_context' in tool_names, f'store_context missing from tools: {tool_names}'

        thread_id = f'http_auth_test_{int(time.time())}'
        store_result = await client.call_tool(
            'store_context',
            {
                'thread_id': thread_id,
                'source': 'agent',
                'text': 'HTTP bearer-auth end-to-end probe entry.',
            },
        )
        store_data = store_result.structured_content
        assert store_data is not None, 'store_context returned no structured content'
        assert store_data.get('success') is True, f'store_context failed: {store_data}'

        search_result = await client.call_tool('search_context', {'thread_id': thread_id})
        search_data = search_result.structured_content
        assert search_data is not None, 'search_context returned no structured content'
        assert search_data.get('count') == 1, f'Expected 1 stored entry, got {search_data}'


@pytest.mark.integration
def test_real_health_endpoint_returns_ok(http_noauth_server: str) -> None:
    """The live ``/health`` route returns HTTP 200 with body ``{"status": "ok"}``.

    Unlike the harness ``test_health_endpoint_returns_ok`` (which builds its own
    Starlette app), this hits the actual route registered by ``main()`` on the
    running HTTP server.
    """
    resp = httpx.get(f'{http_noauth_server}/health', timeout=10.0)
    assert resp.status_code == 200, f'Expected 200 from /health, got {resp.status_code}'
    assert resp.json() == {'status': 'ok'}, f'Unexpected /health body: {resp.text[:300]}'
