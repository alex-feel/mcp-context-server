"""Real HTTP servers for the end-to-end transport and access tests, and the clients of the JWT scenarios.

The server is ``tests/run_server.py`` spawned with ``subprocess.Popen`` as an HTTP
server on a free loopback port. Its environment starts from the MCP SDK's
``get_default_environment()`` plus explicit overrides, so nothing from the developer's
shell (a configured storage backend, credentials, feature toggles) reaches the server;
generation is disabled, so it starts without any model provider.

The JWT scenarios run against these servers: owners, grantees and sharing in
:mod:`tests.integration._http_jwt_access`, the read-only anonymous principal in
:mod:`tests.integration._http_jwt_anonymous`. They share the principals, token minting,
clients and tool-call helpers here; the read helpers check that every entry a read
returns reports its visibility and whether the caller owns it, and never the owner's
id. :func:`serve_jwks` publishes a key pair the way an identity provider's JWKS
endpoint does.
"""

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import AsyncGenerator
from collections.abc import Generator
from collections.abc import Sequence
from contextlib import asynccontextmanager
from contextlib import contextmanager
from functools import partial
from http.server import SimpleHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import override

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from joserfc.jwk import RSAKey
from mcp.client.stdio import get_default_environment
from mcp.types import TextContent

from tests.integration._harness.core import ClientMode

if TYPE_CHECKING:
    import pytest

    from app.backends.base import StorageBackend

# main() registers /health and wires auth for every non-stdio transport; 'http' maps
# to FastMCP's streamable-http MCP endpoint mounted at /mcp.
HTTP_TRANSPORT = 'http'
TEST_TOKEN = 'integration-secret-token-123'
JWT_ISSUER = 'https://issuer.integration.test'
JWT_AUDIENCE = 'mcp-context-server-test'

# run_server.py configures sys.path and test mode, then calls main().
WRAPPER_SCRIPT = Path(__file__).parent.parent / 'run_server.py'

type BackendName = Literal['sqlite', 'postgresql']

ALICE = 'alice'
BOB = 'bob'
CAROL = 'carol'
TEAM = 'team-x'


def free_port() -> int:
    """Return an unused TCP port on the loopback interface.

    Returns:
        The port number.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return int(s.getsockname()[1])


def storage_env(backend: BackendName, database: str) -> dict[str, str]:
    """Route a server to one database.

    Args:
        backend: The storage backend.
        database: The SQLite file path, or the PostgreSQL connection string.

    Returns:
        The routing variables, the PostgreSQL schema pinned to ``public``.
    """
    if backend == 'sqlite':
        return {'STORAGE_BACKEND': 'sqlite', 'DB_PATH': database}
    return {'STORAGE_BACKEND': 'postgresql', 'POSTGRESQL_CONNECTION_STRING': database, 'POSTGRESQL_SCHEMA': 'public'}


def build_env(*, backend: BackendName, database: str, port: int, auth: bool) -> dict[str, str]:
    """Build the explicit environment of a fast-starting HTTP server.

    Args:
        backend: The storage backend.
        database: The SQLite file path, or the PostgreSQL connection string.
        port: Loopback TCP port the HTTP server binds.
        auth: Enable simple_token bearer auth with ``TEST_TOKEN``.

    Returns:
        A complete environment for ``subprocess.Popen``.
    """
    env = {
        **get_default_environment(),
        **storage_env(backend, database),
        'MCP_TEST_MODE': '1',
        'MCP_TRANSPORT': HTTP_TRANSPORT,
        'FASTMCP_HOST': '127.0.0.1',
        'FASTMCP_PORT': str(port),
        # Disable every generation and search subsystem for a fast, dependency-free start.
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
    return env


def build_jwt_env(*, backend: BackendName, database: str, port: int, public_key: str) -> dict[str, str]:
    """Build the environment of a server verifying minted JWTs.

    The jwt provider checks the signature against a static public key and pins the
    issuer and audience, so only tokens minted by the test key pair are accepted.

    Args:
        backend: The storage backend.
        database: The SQLite file path, or the PostgreSQL connection string.
        port: Loopback TCP port the HTTP server binds.
        public_key: PEM-encoded public key of the test RSA key pair.

    Returns:
        A complete environment for ``subprocess.Popen``.
    """
    return {
        **build_env(backend=backend, database=database, port=port, auth=False),
        'MCP_AUTH_PROVIDER': 'jwt',
        'MCP_AUTH_JWT_PUBLIC_KEY': public_key,
        'MCP_AUTH_JWT_ISSUER': JWT_ISSUER,
        'MCP_AUTH_JWT_AUDIENCE': JWT_AUDIENCE,
    }


def build_access_env(*, backend: BackendName, database: str, port: int, public_key: str) -> dict[str, str]:
    """Build the environment of the JWT server the access scenario runs against.

    Full-text search is on, and every store grants each of its author's groups read
    access to the new entry.

    Args:
        backend: The storage backend.
        database: The SQLite file path, or the PostgreSQL connection string.
        port: Loopback TCP port the HTTP server binds.
        public_key: PEM-encoded public key of the test RSA key pair.

    Returns:
        A complete environment for ``subprocess.Popen``.
    """
    return {
        **build_jwt_env(backend=backend, database=database, port=port, public_key=public_key),
        'ENABLE_FTS': 'true',
        'ACCESS_CONTROL_DEFAULT_GROUP_GRANTS': 'author_groups',
    }


def wait_for_health(base_url: str, proc: 'subprocess.Popen[bytes]', timeout_s: float = 60.0) -> None:
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


def terminate(proc: 'subprocess.Popen[bytes]') -> None:
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


@contextmanager
def running_http_server(env: dict[str, str]) -> Generator[str]:
    """Run the server with ``env`` until the block exits.

    Waits for ``/health`` before yielding and terminates the subprocess afterwards,
    so no orphan process or port binding leaks.

    Args:
        env: The complete server environment; ``FASTMCP_PORT`` names the port.

    Yields:
        The server origin URL, e.g. ``http://127.0.0.1:<port>``.
    """
    base_url = f'http://127.0.0.1:{env["FASTMCP_PORT"]}'
    proc = subprocess.Popen(
        [sys.executable, str(WRAPPER_SCRIPT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_health(base_url, proc)
        yield base_url
    finally:
        terminate(proc)


class _QuietFileHandler(SimpleHTTPRequestHandler):
    """Serve the files of one directory, keeping the request log out of the test output."""

    @override
    def log_message(self, format: str, *args: Any) -> None:
        """Log nothing."""


@contextmanager
def serve_jwks(key_pair: RSAKeyPair, kid: str) -> Generator[str]:
    """Publish the public key of ``key_pair`` as an identity provider's JWKS endpoint.

    The endpoint is a loopback HTTP server in a daemon thread, alive until the block
    exits; a server started with ``MCP_AUTH_JWT_JWKS_URI`` set to the yielded URL
    verifies tokens that ``key_pair`` signs with the header ``kid``.

    Args:
        key_pair: The identity provider's key pair.
        kid: The key id the JWKS document gives the key.

    Yields:
        The JWKS URI.
    """
    jwk = {**RSAKey.import_key(key_pair.public_key).as_dict(), 'kid': kid, 'use': 'sig', 'alg': 'RS256'}
    with tempfile.TemporaryDirectory(prefix='jwks_') as directory:
        (Path(directory) / 'jwks.json').write_text(json.dumps({'keys': [jwk]}), encoding='utf-8')
        endpoint = ThreadingHTTPServer(('127.0.0.1', 0), partial(_QuietFileHandler, directory=directory))
        thread = threading.Thread(target=endpoint.serve_forever, daemon=True)
        thread.start()
        try:
            yield f'http://127.0.0.1:{endpoint.server_port}/jwks.json'
        finally:
            endpoint.shutdown()
            endpoint.server_close()
            thread.join(timeout=10)


@asynccontextmanager
async def server_database(
    backend: BackendName, database: str, monkeypatch: 'pytest.MonkeyPatch',
) -> AsyncGenerator['StorageBackend']:
    """Open the database a running server serves, to read rows no tool reports.

    Args:
        backend: The storage backend.
        database: The SQLite file path, or the PostgreSQL connection string.
        monkeypatch: Restores the settings bindings the PostgreSQL backend reads.

    Yields:
        An initialized backend on the server's database.
    """
    from app.backends.factory import create_backend

    if backend == 'sqlite':
        storage = create_backend(backend_type='sqlite', db_path=database)
    else:
        import app.backends.postgresql_backend as postgresql_backend_package
        from app.settings import get_settings
        from tests.helpers import rebind_package_settings

        # The server's tables live in the public schema; pin it for this process too.
        monkeypatch.setenv('POSTGRESQL_SCHEMA', 'public')
        get_settings.cache_clear()
        rebind_package_settings(monkeypatch, postgresql_backend_package, get_settings())
        storage = create_backend(backend_type='postgresql', connection_string=database, provision_vector=False)
    await storage.initialize()
    try:
        yield storage
    finally:
        await storage.shutdown()


def mint_token(key_pair: RSAKeyPair, subject: str, groups: Sequence[str] = ()) -> str:
    """Mint a JWT the access scenario's server accepts.

    Args:
        key_pair: The test key pair the server verifies against.
        subject: The principal id.
        groups: The principal's groups; no groups claim when empty.

    Returns:
        The encoded token.
    """
    return key_pair.create_token(
        subject=subject,
        issuer=JWT_ISSUER,
        audience=JWT_AUDIENCE,
        additional_claims={'groups': list(groups)} if groups else None,
    )


def jwt_client(base_url: str, token: str, client_mode: ClientMode) -> Client[Any]:
    """Build an unconnected client that authenticates with ``token``.

    Args:
        base_url: Server origin.
        token: The bearer token.
        client_mode: Protocol era the client negotiates.

    Returns:
        The client.
    """
    transport = StreamableHttpTransport(url=f'{base_url}/mcp', headers={'Authorization': f'Bearer {token}'})
    return Client(transport, mode=client_mode)


async def tool_content(client: Client[Any], tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool that must succeed and return its structured content.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        The structured content.
    """
    result = await client.call_tool(tool, arguments)
    assert result.structured_content is not None, f'{tool} returned no structured content'
    return result.structured_content


async def tool_error(client: Client[Any], tool: str, arguments: dict[str, Any]) -> str:
    """Call a tool that must refuse the call and return its error text.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        The error text.
    """
    result = await client.call_tool(tool, arguments, raise_on_error=False)
    assert result.is_error, f'{tool}({arguments}) succeeded: {result.structured_content}'
    return ' '.join(block.text for block in result.content if isinstance(block, TextContent))


async def _rows(client: Client[Any], tool: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the entry rows a read, search or grep tool returned.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        The rows.
    """
    content = await tool_content(client, tool, arguments)
    return list(content['result'] if tool == 'get_context_by_ids' else content['results'])


def _row_id(tool: str, row: dict[str, Any]) -> str:
    """Return the entry id of one row a read, search or grep tool returned.

    Args:
        tool: The tool name.
        row: The row.

    Returns:
        The entry id.
    """
    return str(row['context_id' if tool == 'grep_context' else 'id'])


async def tool_ids(client: Client[Any], tool: str, arguments: dict[str, Any]) -> set[str]:
    """Collect the entry ids a read, search or grep tool returned.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        The returned ids.
    """
    return {_row_id(tool, row) for row in await _rows(client, tool, arguments)}


def entry_read_back(entry: dict[str, Any]) -> tuple[str, bool]:
    """Return the access fields one entry-carrying response reports.

    Args:
        entry: A row, or a ``navigate_context`` or ``read_context_range`` result.

    Returns:
        The entry's visibility and whether the caller owns it.
    """
    assert 'owner_id' not in entry, f'the owner id was reported: {entry}'
    assert type(entry['is_owner']) is bool, f'is_owner is not a boolean: {entry}'
    return str(entry['visibility']), entry['is_owner']


async def read_back(client: Client[Any], tool: str, arguments: dict[str, Any]) -> dict[str, tuple[str, bool]]:
    """Collect the access fields a read, search or grep tool reported for each returned entry.

    A grep in ``content`` mode returns one row per match; every row of one entry must
    report the same fields.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        ``(visibility, is_owner)`` keyed by entry id.
    """
    reported: dict[str, tuple[str, bool]] = {}
    for row in await _rows(client, tool, arguments):
        fields = entry_read_back(row)
        assert reported.setdefault(_row_id(tool, row), fields) == fields, f'{tool} rows of one entry disagree: {row}'
    return reported


async def share(client: Client[Any], context_id: str, principal_type: str, principal_id: str, permission: str) -> None:
    """Grant access to an entry through ``share_context``, which must create the grant.

    Args:
        client: The owner's client.
        context_id: The entry id.
        principal_type: ``'user'`` or ``'group'``.
        principal_id: The grantee.
        permission: ``'read'`` or ``'write'``.
    """
    shared = await tool_content(client, 'share_context', {
        'context_id': context_id, 'principal_type': principal_type, 'principal_id': principal_id, 'permission': permission,
    })
    assert shared['created'] is True, shared


async def listed_grants(client: Client[Any], context_id: str) -> list[tuple[str, str, str, str]]:
    """List the grants of an entry through ``list_context_grants``.

    Args:
        client: The owner's client.
        context_id: The entry id.

    Returns:
        ``(principal_type, principal_id, permission, granted_by)`` per grant, in listing order.
    """
    listed = await tool_content(client, 'list_context_grants', {'context_id': context_id})
    assert (listed['context_id'], listed['count']) == (context_id, len(listed['grants'])), listed
    assert all(isinstance(grant['created_at'], str) and grant['created_at'] for grant in listed['grants']), listed
    return [
        (grant['principal_type'], grant['principal_id'], grant['permission'], grant['granted_by'])
        for grant in listed['grants']
    ]
