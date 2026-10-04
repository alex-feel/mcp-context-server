"""End-to-end HTTP transport tests for the real MCP server (SQLite backend).

The rest of the integration suite drives the server over stdio, so two
user-facing HTTP behaviors have no end-to-end coverage:

1. Bearer-token authentication (``MCP_AUTH_PROVIDER=simple_token`` plus
   ``MCP_AUTH_TOKEN``) and JWT verification (``MCP_AUTH_PROVIDER=jwt``). Both are
   wired only on HTTP transports (``app/server.py`` ``main()`` calls
   ``create_auth_provider()`` when ``transport != 'stdio'``).
2. The real ``/health`` route registered via ``mcp.custom_route('/health', ...)``
   for non-stdio transports. The existing harness ``test_health_endpoint_returns_ok``
   builds its own throwaway Starlette app and never hits the live route.

These tests launch the actual server as an HTTP server through the helpers of
:mod:`tests.integration._http_jwt`: an explicit environment built from the MCP SDK's
default environment, generation disabled for a fast start, a free ephemeral loopback
port, and termination of the subprocess when the test ends so no orphan server or port
binding leaks.
"""

import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.auth.providers.jwt import RSAKeyPair

from tests.integration._harness.core import CLIENT_MODES
from tests.integration._harness.core import ERA_PROTOCOL_VERSIONS
from tests.integration._harness.core import ClientMode
from tests.integration._http_jwt import JWT_AUDIENCE
from tests.integration._http_jwt import JWT_ISSUER
from tests.integration._http_jwt import TEST_TOKEN
from tests.integration._http_jwt import build_env
from tests.integration._http_jwt import build_jwt_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import running_http_server


@pytest.fixture
def http_auth_server(tmp_path: Path) -> Iterator[str]:
    """Launch a real HTTP server with bearer-token auth; yield its base URL.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    env = build_env(backend='sqlite', database=str(tmp_path / 'http_auth.db'), port=free_port(), auth=True)
    with running_http_server(env) as base_url:
        yield base_url


@pytest.fixture
def http_noauth_server(tmp_path: Path) -> Iterator[str]:
    """Launch a real HTTP server WITHOUT auth; yield its base URL.

    Used by the ``/health`` test: the health endpoint is unauthenticated even
    when auth is on, but a no-auth server keeps that test independent of the
    auth configuration under test.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    env = build_env(backend='sqlite', database=str(tmp_path / 'http_health.db'), port=free_port(), auth=False)
    with running_http_server(env) as base_url:
        yield base_url


@pytest.fixture
def http_jwt_server(tmp_path: Path, jwt_key_pair: RSAKeyPair) -> Iterator[str]:
    """Launch a real HTTP server verifying JWTs against the test key pair.

    The server runs ``MCP_AUTH_PROVIDER=jwt`` with the test key pair's public key,
    issuer and audience pinned.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    env = build_jwt_env(
        backend='sqlite',
        database=str(tmp_path / 'http_jwt.db'),
        port=free_port(),
        public_key=jwt_key_pair.public_key,
    )
    with running_http_server(env) as base_url:
        yield base_url


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
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_http_request_with_correct_token_is_accepted(http_auth_server: str, client_mode: ClientMode) -> None:
    """The CORRECT bearer token authenticates and can list and call tools in either protocol era."""
    transport = StreamableHttpTransport(
        url=f'{http_auth_server}/mcp',
        headers={'Authorization': f'Bearer {TEST_TOKEN}'},
    )
    async with Client(transport, mode=client_mode) as client:
        assert client.protocol_version in ERA_PROTOCOL_VERSIONS[client_mode], (
            f'{client_mode} mode negotiated {client.protocol_version!r}'
        )

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
def test_http_jwt_request_without_token_is_rejected(http_jwt_server: str) -> None:
    """An MCP request with NO Authorization header is rejected by the jwt provider."""
    resp = httpx.post(
        f'{http_jwt_server}/mcp',
        json=_initialize_payload(),
        headers={'Accept': 'application/json, text/event-stream'},
        timeout=10.0,
    )
    assert resp.status_code == 401, (
        f'Expected 401 for missing Authorization header, got {resp.status_code}: '
        f'{resp.text[:300]}'
    )


@pytest.mark.integration
def test_http_jwt_invalid_tokens_are_rejected(http_jwt_server: str, jwt_key_pair: RSAKeyPair) -> None:
    """Wrong-signature, expired, and wrong-audience JWTs are all rejected."""
    other_pair = RSAKeyPair.generate()
    invalid_tokens = {
        'wrong signature': other_pair.create_token(
            subject='mallory',
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
        ),
        'expired': jwt_key_pair.create_token(
            subject='alice',
            issuer=JWT_ISSUER,
            audience=JWT_AUDIENCE,
            expires_in_seconds=-60,
        ),
        'wrong audience': jwt_key_pair.create_token(
            subject='alice',
            issuer=JWT_ISSUER,
            audience='some-other-service',
        ),
    }
    for reason, token in invalid_tokens.items():
        resp = httpx.post(
            f'{http_jwt_server}/mcp',
            json=_initialize_payload(),
            headers={
                'Accept': 'application/json, text/event-stream',
                'Authorization': f'Bearer {token}',
            },
            timeout=10.0,
        )
        assert resp.status_code == 401, (
            f'Expected 401 for {reason} token, got {resp.status_code}: '
            f'{resp.text[:300]}'
        )


@pytest.mark.integration
@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_http_jwt_valid_token_is_accepted(
    http_jwt_server: str,
    jwt_key_pair: RSAKeyPair,
    client_mode: ClientMode,
) -> None:
    """A valid minted JWT authenticates and can list and call tools in either protocol era."""
    token = jwt_key_pair.create_token(
        subject='integration-user',
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        additional_claims={'groups': ['team-a'], 'roles': ['publisher']},
    )
    transport = StreamableHttpTransport(
        url=f'{http_jwt_server}/mcp',
        headers={'Authorization': f'Bearer {token}'},
    )
    async with Client(transport, mode=client_mode) as client:
        assert client.protocol_version in ERA_PROTOCOL_VERSIONS[client_mode], (
            f'{client_mode} mode negotiated {client.protocol_version!r}'
        )

        tools = await client.list_tools()
        tool_names = {tool.name for tool in tools}
        assert 'store_context' in tool_names, f'store_context missing from tools: {tool_names}'

        thread_id = f'http_jwt_test_{int(time.time())}'
        store_result = await client.call_tool(
            'store_context',
            {
                'thread_id': thread_id,
                'source': 'agent',
                'text': 'HTTP JWT-auth end-to-end probe entry.',
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
