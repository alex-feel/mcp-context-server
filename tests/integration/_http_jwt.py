"""Real HTTP servers for the end-to-end transport and access tests, and the JWT access scenario.

The server is ``tests/run_server.py`` spawned with ``subprocess.Popen`` as an HTTP
server on a free loopback port. Its environment starts from the MCP SDK's
``get_default_environment()`` plus explicit overrides, so nothing from the developer's
shell (a configured storage backend, credentials, feature toggles) reaches the server;
generation is disabled, so it starts without any model provider.

``verify_jwt_access_scoping`` is the access scenario both backends run
(``tests/integration/sqlite/test_http_jwt_access.py`` and
``tests/integration/postgresql/test_http_jwt_access.py``): three principals with
minted JWTs -- ``alice`` and ``bob`` in group ``team-x``, ``carol`` in none -- against a
server that stamps a group read grant for every author group
(``ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups``). The application writes no user
grant and no write grant, so the scenario inserts carol's two grants straight into the
server's database with ``insert_grant``.
"""

import socket
import subprocess
import sys
import time
from collections.abc import AsyncIterator
from collections.abc import Iterator
from collections.abc import Sequence
from contextlib import asynccontextmanager
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal

import httpx
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from mcp.client.stdio import get_default_environment
from mcp.types import TextContent

from tests.helpers import insert_grant
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
SCENARIO_TAG = 'jwt-access'
SCENARIO_WORD = 'quasarfern'
GROUP_THREAD = 'jwt_access_group'
GRANTED_THREAD = 'jwt_access_granted'


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
def running_http_server(env: dict[str, str]) -> Iterator[str]:
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


@asynccontextmanager
async def server_database(
    backend: BackendName, database: str, monkeypatch: 'pytest.MonkeyPatch',
) -> AsyncIterator['StorageBackend']:
    """Open the database a running server serves, to write what no tool writes.

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


async def _content(client: Client[Any], tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
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


async def _error(client: Client[Any], tool: str, arguments: dict[str, Any]) -> str:
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


async def _ids(client: Client[Any], tool: str, arguments: dict[str, Any]) -> set[str]:
    """Collect the entry ids a read, search or grep tool returned.

    Args:
        client: The calling principal's client.
        tool: The tool name.
        arguments: The tool arguments.

    Returns:
        The returned ids.
    """
    content = await _content(client, tool, arguments)
    rows = content['result'] if tool == 'get_context_by_ids' else content['results']
    key = 'context_id' if tool == 'grep_context' else 'id'
    return {str(row[key]) for row in rows}


async def _store_alice_entries(alice: Client[Any]) -> dict[str, str]:
    """Store alice's four entries under the default ``private`` visibility.

    Args:
        alice: alice's client.

    Returns:
        Entry ids keyed ``group`` (alone in its thread), ``read`` and ``write`` (granted to
        carol below) and ``team`` (readable through the group grant only).
    """
    entries: dict[str, str] = {}
    threads = {'group': GROUP_THREAD, 'read': GRANTED_THREAD, 'write': GRANTED_THREAD, 'team': GRANTED_THREAD}
    for label, thread in threads.items():
        stored = await _content(alice, 'store_context', {
            'thread_id': thread,
            'source': 'agent',
            'text': f'# {label.title()} entry\n\nAlice {label} entry carrying {SCENARIO_WORD}.',
            'tags': [SCENARIO_TAG],
        })
        entries[label] = str(stored['context_id'])
    return entries


async def verify_jwt_access_scoping(
    base_url: str, key_pair: RSAKeyPair, client_mode: ClientMode, database: 'StorageBackend',
) -> None:
    """Prove what each JWT principal reads, modifies and deletes on a running server.

    bob reads all of alice's private entries through the group grant ``author_groups``
    stamped at store time; carol, in no group, reads only the entry granted to her for
    read and the one granted to her for write. carol's write grant authorizes a text
    update but neither a visibility change nor a delete; her read grant authorizes no
    update; and a thread delete by a grantee deletes nothing.

    Args:
        base_url: Origin of the server started with ``build_access_env``.
        key_pair: The key pair the server verifies against.
        client_mode: Protocol era every client negotiates.
        database: A backend on the server's database, for the grants no tool writes.
    """
    async with (
        jwt_client(base_url, mint_token(key_pair, ALICE, [TEAM]), client_mode) as alice,
        jwt_client(base_url, mint_token(key_pair, BOB, [TEAM]), client_mode) as bob,
        jwt_client(base_url, mint_token(key_pair, CAROL), client_mode) as carol,
    ):
        entry = await _store_alice_entries(alice)
        await insert_grant(database, entry['read'], 'user', CAROL, 'read', ALICE)
        await insert_grant(database, entry['write'], 'user', CAROL, 'write', ALICE)
        all_ids = set(entry.values())
        carol_ids = {entry['read'], entry['write']}

        for client, expected, threads, totals in (
            (alice, all_ids, {GROUP_THREAD: 1, GRANTED_THREAD: 3}, (4, 2)),
            (bob, all_ids, {GROUP_THREAD: 1, GRANTED_THREAD: 3}, (4, 2)),
            (carol, carol_ids, {GRANTED_THREAD: 2}, (2, 1)),
        ):
            assert await _ids(client, 'search_context', {'tags': [SCENARIO_TAG]}) == expected
            assert await _ids(client, 'get_context_by_ids', {'context_ids': sorted(all_ids)}) == expected
            assert await _ids(client, 'fts_search_context', {'query': SCENARIO_WORD}) == expected
            assert await _ids(client, 'grep_context', {'pattern': SCENARIO_WORD}) == expected
            listed = (await _content(client, 'list_threads', {}))['threads']
            assert {row['thread_id']: row['entry_count'] for row in listed} == threads
            stats = await _content(client, 'get_statistics', {})
            assert (stats['total_entries'], stats['total_threads']) == totals

        hidden_from_carol = f'Context entry not found: {entry["group"]}'
        assert await _error(carol, 'navigate_context', {'context_id': entry['group']}) == hidden_from_carol
        assert await _error(carol, 'read_context_range', {
            'context_id': entry['group'], 'start_line': 1, 'end_line': 1,
        }) == hidden_from_carol
        for client, context_id in ((bob, entry['group']), (carol, entry['read']), (carol, entry['write'])):
            outline = await _content(client, 'navigate_context', {'context_id': context_id})
            assert outline['context_id'] == context_id
            span = await _content(client, 'read_context_range', {'context_id': context_id, 'start_line': 1, 'end_line': 1})
            assert span['text'].startswith('# ')

        new_text = f'Carol rewrote the write entry, still carrying {SCENARIO_WORD}.'
        updated = await _content(carol, 'update_context', {'context_id': entry['write'], 'text': new_text})
        assert updated['success'] is True
        rows = (await _content(alice, 'get_context_by_ids', {'context_ids': [entry['write']]}))['result']
        assert rows[0]['text_content'] == new_text
        assert await _error(carol, 'update_context', {'context_id': entry['read'], 'text': 'x'}) == (
            f'Not authorized to modify context entry with ID {entry["read"]}'
        )
        assert await _error(carol, 'update_context', {'context_id': entry['group'], 'text': 'x'}) == (
            f'Context entry with ID {entry["group"]} not found'
        )
        assert await _error(carol, 'update_context', {'context_id': entry['write'], 'visibility': 'public'}) == (
            f'Only the owner may change the visibility of context {entry["write"]}'
        )
        assert await _error(bob, 'update_context', {'context_id': entry['group'], 'text': 'x'}) == (
            f'Not authorized to modify context entry with ID {entry["group"]}'
        )

        for client, context_id in ((carol, entry['write']), (bob, entry['group'])):
            refusal = f'Not authorized to delete context entries: {context_id}'
            assert await _error(client, 'delete_context', {'context_ids': [context_id]}) == refusal
            assert await _error(client, 'delete_context_batch', {'context_ids': [context_id]}) == refusal
        for client in (bob, carol):
            assert (await _content(client, 'delete_context', {'thread_id': GRANTED_THREAD}))['deleted_count'] == 0
            assert (await _content(client, 'delete_context_batch', {'thread_ids': [GRANTED_THREAD]}))['deleted_count'] == 0
        assert await _ids(alice, 'get_context_by_ids', {'context_ids': sorted(all_ids)}) == all_ids

        deleted = await _content(alice, 'delete_context_batch', {'thread_ids': [GROUP_THREAD]})
        assert deleted['deleted_count'] == 1
        assert await _ids(bob, 'get_context_by_ids', {'context_ids': [entry['group']]}) == set()
