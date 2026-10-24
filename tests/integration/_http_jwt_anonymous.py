"""The read-only anonymous principal over HTTP: what it reads, what it is refused, and how it is verified.

A server started with ``ACCESS_CONTROL_ANONYMOUS_PRINCIPAL=anonymous`` treats every
token whose ``sub`` is ``anonymous`` as one shared, read-only identity. Both entry points
(``tests/integration/sqlite/test_http_jwt_access.py`` and
``tests/integration/postgresql/test_http_jwt_access.py``, the PostgreSQL one also with
row-level security on) run three scenarios against the access scenario's server
configuration:

- ``verify_anonymous_arm``: two holders of distinct anonymous tokens read exactly
  alice's public entries through every read tool, and every write tool and both
  sharing tools refuse them before touching a row; nobody can share an entry with the
  anonymous principal; an anonymous token without an expiry is refused; ``/health``
  answers without a token while the MCP endpoint demands one.
- ``verify_read_only_setting_fails_open``: a server without the setting accepts the
  same token's store, which is exactly what an operator's check before distributing
  the token detects.
- ``verify_jwks_composition``: on a JWKS deployment an operator-held static key
  verifies the anonymous token beside the identity provider's keys, and verifies no
  token for any other subject.
"""

import sqlite3
import time
import uuid
from typing import TYPE_CHECKING
from typing import Any

import httpx
from fastmcp import Client
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from joserfc import jwt
from joserfc.jwk import RSAKey
from mcp.types import TextContent

from app.backends.base import MAINTENANCE
from tests.helpers import read_grants
from tests.integration._harness.core import ClientMode
from tests.integration._harness.sync import listed_source_ids
from tests.integration._harness.sync import managed_entry
from tests.integration._harness.sync import manifest_item
from tests.integration._harness.sync import meta_digest
from tests.integration._harness.sync import plan_shape
from tests.integration._harness.sync import plan_with
from tests.integration._http_jwt import ALICE
from tests.integration._http_jwt import JWT_AUDIENCE
from tests.integration._http_jwt import JWT_ISSUER
from tests.integration._http_jwt import TEAM
from tests.integration._http_jwt import BackendName
from tests.integration._http_jwt import build_access_env
from tests.integration._http_jwt import build_env
from tests.integration._http_jwt import entry_read_back
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import jwt_client
from tests.integration._http_jwt import mint_token
from tests.integration._http_jwt import read_back
from tests.integration._http_jwt import tool_content
from tests.integration._http_jwt import tool_error

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import StorageBackend

ANONYMOUS = 'anonymous'
READ_ONLY_SETTING = 'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL'
EXPIRY_REFUSAL = "The anonymous principal's token carries no expiry (exp); mint it with a bounded exp"
JWKS_KID = 'integration-identity-provider'
ANONYMOUS_THREAD = 'jwt_anonymous'
ANONYMOUS_CORPUS = 'kb:jwt_anonymous'
ANONYMOUS_TAG = 'jwt-anonymous'
ANONYMOUS_WORD = 'lumenwick'
JWKS_THREAD = 'jwt_jwks'
CORPUS_DIGEST = meta_digest('jwt-anonymous')
CORPUS_TEXTS = {'public.md': 'Alice public managed entry.', 'private.md': 'Alice private managed entry.'}
# The read-only tools of the access scenario's server configuration; the arm calls each.
READ_TOOLS = frozenset({
    'get_context_by_ids',
    'search_context',
    'fts_search_context',
    'grep_context',
    'navigate_context',
    'read_context_range',
    'list_threads',
    'get_statistics',
    'plan_context_sync',
    'list_ingest_state',
    'list_context_grants',
})


def read_only_refusal(tool: str) -> str:
    """Return the text every write tool refuses the anonymous principal with.

    Args:
        tool: The tool name.

    Returns:
        The refusal text.
    """
    return f'The anonymous principal is read-only: {tool} is not permitted'


def build_anonymous_env(*, backend: BackendName, database: str, port: int, public_key: str) -> dict[str, str]:
    """Build the access scenario's server environment with the anonymous principal read-only.

    Args:
        backend: The storage backend.
        database: The SQLite file path, or the PostgreSQL connection string.
        port: Loopback TCP port the HTTP server binds.
        public_key: PEM-encoded public key of the test RSA key pair.

    Returns:
        A complete environment for ``subprocess.Popen``.
    """
    access_env = build_access_env(backend=backend, database=database, port=port, public_key=public_key)
    return {**access_env, READ_ONLY_SETTING: ANONYMOUS}


def without_read_only_principal(env: dict[str, str]) -> dict[str, str]:
    """Return a server environment with the read-only setting unset, on a new port.

    Args:
        env: A server environment that names the read-only principal.

    Returns:
        The same environment without ``ACCESS_CONTROL_ANONYMOUS_PRINCIPAL``.
    """
    return {**{key: value for key, value in env.items() if key != READ_ONLY_SETTING}, 'FASTMCP_PORT': str(free_port())}


def build_jwks_env(
    *, backend: BackendName, database: str, port: int, jwks_uri: str, anonymous_public_key: str,
) -> dict[str, str]:
    """Build the environment of a JWKS deployment that also verifies the anonymous token.

    The JWKS verifier trusts the identity provider's keys; the anonymous verifier trusts
    ``anonymous_public_key`` for the subject ``anonymous`` alone. Both pin the same
    issuer and audience.

    Args:
        backend: The storage backend.
        database: The SQLite file path, or the PostgreSQL connection string.
        port: Loopback TCP port the HTTP server binds.
        jwks_uri: The identity provider's JWKS endpoint.
        anonymous_public_key: PEM-encoded public key of the operator's anonymous key pair.

    Returns:
        A complete environment for ``subprocess.Popen``.
    """
    return {
        **build_env(backend=backend, database=database, port=port, auth=False),
        'MCP_AUTH_PROVIDER': 'jwt',
        'MCP_AUTH_JWT_JWKS_URI': jwks_uri,
        'MCP_AUTH_JWT_ISSUER': JWT_ISSUER,
        'MCP_AUTH_JWT_AUDIENCE': JWT_AUDIENCE,
        'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY': anonymous_public_key,
        READ_ONLY_SETTING: ANONYMOUS,
    }


def mint_anonymous_token(key_pair: RSAKeyPair) -> str:
    """Mint an anonymous token with an expiry and a token id of its own.

    Args:
        key_pair: The key pair the server verifies anonymous tokens against.

    Returns:
        The encoded token.
    """
    return key_pair.create_token(
        subject=ANONYMOUS, issuer=JWT_ISSUER, audience=JWT_AUDIENCE, additional_claims={'jti': uuid.uuid4().hex},
    )


def mint_token_without_expiry(key_pair: RSAKeyPair, subject: str) -> str:
    """Mint a token that carries no ``exp`` claim, which the JWT verifier accepts forever.

    Args:
        key_pair: The key pair the server verifies against.
        subject: The principal id.

    Returns:
        The encoded token.
    """
    claims = {'sub': subject, 'iss': JWT_ISSUER, 'aud': JWT_AUDIENCE, 'iat': int(time.time()), 'jti': uuid.uuid4().hex}
    key = RSAKey.import_key(key_pair.private_key.get_secret_value())
    return jwt.encode({'alg': 'RS256'}, claims, key, algorithms=['RS256'])


async def mcp_status(base_url: str, token: str | None) -> int:
    """Return the HTTP status a raw request to the MCP endpoint gets.

    Args:
        base_url: Server origin.
        token: The bearer token, or None to send none.

    Returns:
        The status code.
    """
    headers = {'Accept': 'application/json, text/event-stream'}
    if token is not None:
        headers['Authorization'] = f'Bearer {token}'
    async with httpx.AsyncClient(timeout=10.0) as http:
        response = await http.post(
            f'{base_url}/mcp', headers=headers, json={'jsonrpc': '2.0', 'id': 1, 'method': 'tools/list'},
        )
    return response.status_code


async def count_owned_entries(database: 'StorageBackend', owner_id: str) -> int:
    """Count the entries one principal owns, through raw SQL as the table owner.

    Args:
        database: A backend on the server's database that no policy filters.
        owner_id: The principal id.

    Returns:
        The number of entries whose ``owner_id`` is ``owner_id``.
    """
    if database.backend_type == 'sqlite':

        def _count_sqlite(conn: sqlite3.Connection) -> int:
            return int(conn.execute('SELECT COUNT(*) FROM context_entries WHERE owner_id = ?', (owner_id,)).fetchone()[0])

        return await database.execute_read(_count_sqlite, access_scope=MAINTENANCE)

    async def _count_postgresql(conn: 'asyncpg.Connection') -> int:
        return int(await conn.fetchval('SELECT COUNT(*) FROM context_entries WHERE owner_id = $1', owner_id))

    return await database.execute_read(_count_postgresql, access_scope=MAINTENANCE)


async def _tool_names(client: Client[Any]) -> tuple[set[str], set[str]]:
    """Split the tools a client is offered into read-only tools and the rest.

    Args:
        client: The client.

    Returns:
        The names of the tools annotated read-only, and the names of every other tool.
    """
    tools = await client.list_tools()
    read_only = {tool.name for tool in tools if tool.annotations is not None and tool.annotations.read_only_hint}
    return read_only, {tool.name for tool in tools} - read_only


async def _store_alice_rows(alice: Client[Any]) -> dict[str, str]:
    """Store alice's public and private entries.

    Args:
        alice: alice's client.

    Returns:
        Entry ids keyed ``note`` and ``guide`` (public), ``private``, and ``public.md`` and
        ``private.md`` (the managed entries of the corpus, public and private).
    """
    ids: dict[str, str] = {}
    for label, visibility in (('note', 'public'), ('guide', 'public'), ('private', 'private')):
        stored = await tool_content(alice, 'store_context', {
            'thread_id': ANONYMOUS_THREAD,
            'source': 'agent',
            'text': f'# Alice {label}\n\n## Body\n\nAlice {visibility} {label} carrying {ANONYMOUS_WORD}.',
            'tags': [ANONYMOUS_TAG],
            'visibility': visibility,
        })
        ids[label] = str(stored['context_id'])
    entries = [
        managed_entry(ANONYMOUS_CORPUS, source, text, CORPUS_DIGEST, visibility=source.removesuffix('.md'))
        for source, text in CORPUS_TEXTS.items()
    ]
    stored_batch = await tool_content(alice, 'store_context_batch', {'entries': entries})
    for item in stored_batch['results']:
        ids[list(CORPUS_TEXTS)[item['index']]] = str(item['context_id'])
    return ids


async def _verify_reads_public_entries_only(anonymous: Client[Any], ids: dict[str, str]) -> None:
    """Prove every read tool shows the anonymous principal exactly alice's public entries.

    Args:
        anonymous: An anonymous token holder's client.
        ids: alice's entry ids, as :func:`_store_alice_rows` returns them.
    """
    public = dict.fromkeys((ids['note'], ids['guide'], ids['public.md']), ('public', False))
    tagged = dict.fromkeys((ids['note'], ids['guide']), ('public', False))
    hidden = ids['private']
    assert await read_back(anonymous, 'get_context_by_ids', {'context_ids': sorted(ids.values())}) == public
    assert await read_back(anonymous, 'search_context', {'tags': [ANONYMOUS_TAG]}) == tagged
    assert await read_back(anonymous, 'fts_search_context', {'query': ANONYMOUS_WORD}) == tagged
    assert await read_back(anonymous, 'grep_context', {'pattern': ANONYMOUS_WORD, 'output_mode': 'content'}) == tagged
    outline = await tool_content(anonymous, 'navigate_context', {'context_id': ids['guide']})
    assert entry_read_back(outline) == ('public', False)
    span = await tool_content(anonymous, 'read_context_range', {'context_id': ids['note'], 'start_line': 1, 'end_line': 1})
    assert span['text'].startswith('# Alice note')
    assert entry_read_back(span) == ('public', False)
    for tool, arguments in (
        ('navigate_context', {'context_id': hidden}),
        ('read_context_range', {'context_id': hidden, 'start_line': 1, 'end_line': 1}),
    ):
        assert await tool_error(anonymous, tool, arguments) == f'Context entry not found: {hidden}'
    threads = (await tool_content(anonymous, 'list_threads', {}))['threads']
    assert {row['thread_id']: row['entry_count'] for row in threads} == {ANONYMOUS_THREAD: 2, ANONYMOUS_CORPUS: 1}
    stats = await tool_content(anonymous, 'get_statistics', {})
    assert (stats['total_entries'], stats['total_threads']) == (3, 2)
    manifest = [manifest_item(source, text, CORPUS_DIGEST) for source, text in CORPUS_TEXTS.items()]
    assert await plan_with(anonymous, ANONYMOUS_CORPUS, manifest) == plan_shape(creates=[0, 1], foreign=1)
    assert await listed_source_ids(anonymous, ANONYMOUS_CORPUS) == []
    assert await tool_error(anonymous, 'list_context_grants', {'context_id': ids['note']}) == (
        f'Not authorized to list the grants of context entry with ID {ids["note"]}'
    )
    assert await tool_error(anonymous, 'list_context_grants', {'context_id': hidden}) == (
        f'Context entry with ID {hidden} not found'
    )


async def _verify_writes_refused(anonymous: Client[Any], ids: dict[str, str], write_tools: set[str]) -> None:
    """Prove every write tool and both sharing tools refuse the anonymous principal.

    Each call names an entry the anonymous principal may read, and the store calls
    name a valid new entry, so every refusal is the read-only gate's and none is a
    missing entry or an invalid argument.

    Args:
        anonymous: An anonymous token holder's client.
        ids: alice's entry ids, as :func:`_store_alice_rows` returns them.
        write_tools: Every tool the server offers that is not annotated read-only.
    """
    note = ids['note']
    new_entry = {'thread_id': ANONYMOUS_THREAD, 'source': 'agent', 'text': 'Written by an anonymous holder.'}
    calls: dict[str, dict[str, Any]] = {
        'store_context': new_entry,
        'store_context_batch': {'entries': [new_entry]},
        'update_context': {'context_id': note, 'text': 'Rewritten by an anonymous holder.'},
        'update_context_batch': {'updates': [{'context_id': note, 'text': 'Rewritten by an anonymous holder.'}]},
        'delete_context': {'context_ids': [note]},
        'delete_context_batch': {'context_ids': [note]},
        'share_context': {'context_id': note, 'principal_type': 'group', 'principal_id': 'everyone', 'permission': 'read'},
        'unshare_context': {'context_id': note, 'principal_type': 'group', 'principal_id': TEAM, 'permission': 'read'},
    }
    assert write_tools == set(calls), write_tools
    for tool, arguments in calls.items():
        refusal = await tool_error(anonymous, tool, arguments)
        assert refusal == read_only_refusal(tool), refusal


async def verify_anonymous_arm(
    base_url: str, key_pair: RSAKeyPair, client_mode: ClientMode, database: 'StorageBackend', token: str,
) -> None:
    """Prove the anonymous principal reads exactly alice's public entries and changes nothing.

    alice stores public and private entries and managed corpus entries, and is refused a
    user grant to the anonymous principal. Two anonymous holders, one with ``token`` and
    one with a token minted beside it, see the same public entries through every read
    tool, each reported as public and not theirs, and are refused every write and sharing
    tool. Afterwards the anonymous principal owns no entry and every grant is the one
    ``author_groups`` stored. An anonymous token without an expiry is refused even on a
    read tool, and only the MCP endpoint demands a token.

    Args:
        base_url: Origin of the server started with :func:`build_anonymous_env`.
        key_pair: The key pair the server verifies against.
        client_mode: Protocol era every client negotiates.
        database: A backend on the server's database that no policy filters.
        token: An anonymous token minted with :func:`mint_anonymous_token`.
    """
    async with jwt_client(base_url, mint_token(key_pair, ALICE, [TEAM]), client_mode) as alice:
        ids = await _store_alice_rows(alice)
        before = {row['id']: row for row in (await tool_content(alice, 'get_context_by_ids', {
            'context_ids': sorted(ids.values()),
        }))['result']}
        assert await tool_error(alice, 'share_context', {
            'context_id': ids['note'], 'principal_type': 'user', 'principal_id': ANONYMOUS, 'permission': 'read',
        }) == (
            f'Cannot share context entry with ID {ids["note"]} with the anonymous principal; '
            "set its visibility to 'public' to publish it"
        )

    second = mint_anonymous_token(key_pair)
    assert second != token
    for holder_token in (token, second):
        async with jwt_client(base_url, holder_token, client_mode) as anonymous:
            read_tools, write_tools = await _tool_names(anonymous)
            assert read_tools == READ_TOOLS, read_tools
            await _verify_reads_public_entries_only(anonymous, ids)
            await _verify_writes_refused(anonymous, ids, write_tools)

    assert await count_owned_entries(database, ANONYMOUS) == 0
    for context_id in ids.values():
        assert await read_grants(database, context_id) == [('group', TEAM, 'read', ALICE)]
    async with jwt_client(base_url, mint_token(key_pair, ALICE, [TEAM]), client_mode) as alice:
        after = (await tool_content(alice, 'get_context_by_ids', {'context_ids': sorted(ids.values())}))['result']
        assert {row['id']: (row['text_content'], row['updated_at']) for row in after} == {
            context_id: (row['text_content'], row['updated_at']) for context_id, row in before.items()
        }

    async with jwt_client(base_url, mint_token_without_expiry(key_pair, ANONYMOUS), client_mode) as unbounded:
        assert await tool_error(unbounded, 'get_statistics', {}) == EXPIRY_REFUSAL
    async with httpx.AsyncClient(timeout=10.0) as http:
        assert (await http.get(f'{base_url}/health')).status_code == 200
    assert await mcp_status(base_url, None) == 401
    assert await mcp_status(base_url, token) != 401


async def verify_read_only_setting_fails_open(base_url: str, token: str, client_mode: ClientMode) -> None:
    """Prove a server without the read-only setting treats an anonymous token as writable.

    This is the outcome the operator's check before distributing the token exists to
    catch: the store succeeds, no read-only refusal appears, and the token's holder owns
    the new entry.

    Args:
        base_url: Origin of a server started without ``ACCESS_CONTROL_ANONYMOUS_PRINCIPAL``.
        token: An anonymous token the server verifies.
        client_mode: Protocol era the client negotiates.
    """
    async with jwt_client(base_url, token, client_mode) as anonymous:
        result = await anonymous.call_tool('store_context', {
            'thread_id': ANONYMOUS_THREAD, 'source': 'agent', 'text': 'Written by the anonymous token.',
        }, raise_on_error=False)
        text = ' '.join(block.text for block in result.content if isinstance(block, TextContent))
        assert not result.is_error, text
        assert read_only_refusal('store_context') not in text
        assert result.structured_content is not None
        context_id = str(result.structured_content['context_id'])
        assert await read_back(anonymous, 'get_context_by_ids', {'context_ids': [context_id]}) == {
            context_id: ('private', True),
        }


async def verify_jwks_composition(
    base_url: str, idp_key_pair: RSAKeyPair, anonymous_key_pair: RSAKeyPair, client_mode: ClientMode,
) -> None:
    """Prove a JWKS deployment verifies the operator-minted anonymous token and no forged one.

    alice's identity-provider token, signed with the key the JWKS endpoint publishes,
    stores a public and a private entry. The anonymous token, signed with the operator's
    static key, reads only the public one and is refused a store, an update and a delete.
    A token the operator's key signs for any other subject is rejected before any tool
    runs.

    Args:
        base_url: Origin of the server started with :func:`build_jwks_env`.
        idp_key_pair: The identity provider's key pair, published under ``JWKS_KID``.
        anonymous_key_pair: The operator's anonymous key pair.
        client_mode: Protocol era every client negotiates.
    """
    idp_token = idp_key_pair.create_token(subject=ALICE, issuer=JWT_ISSUER, audience=JWT_AUDIENCE, kid=JWKS_KID)
    async with jwt_client(base_url, idp_token, client_mode) as alice:
        ids: dict[str, str] = {}
        for visibility in ('public', 'private'):
            stored = await tool_content(alice, 'store_context', {
                'thread_id': JWKS_THREAD, 'source': 'agent', 'text': f'Alice {visibility} entry.', 'visibility': visibility,
            })
            ids[visibility] = str(stored['context_id'])
        assert await read_back(alice, 'get_context_by_ids', {'context_ids': sorted(ids.values())}) == {
            ids['public']: ('public', True), ids['private']: ('private', True),
        }

    async with jwt_client(base_url, mint_anonymous_token(anonymous_key_pair), client_mode) as anonymous:
        public = {ids['public']: ('public', False)}
        assert await read_back(anonymous, 'get_context_by_ids', {'context_ids': sorted(ids.values())}) == public
        assert await read_back(anonymous, 'search_context', {'thread_id': JWKS_THREAD}) == public
        for tool, arguments in (
            ('store_context', {'thread_id': JWKS_THREAD, 'source': 'agent', 'text': 'Written by an anonymous holder.'}),
            ('update_context', {'context_id': ids['public'], 'text': 'Rewritten by an anonymous holder.'}),
            ('delete_context', {'context_ids': [ids['public']]}),
        ):
            assert await tool_error(anonymous, tool, arguments) == read_only_refusal(tool)

    assert await mcp_status(base_url, idp_token) != 401
    for subject in (ALICE, 'mallory'):
        forged = anonymous_key_pair.create_token(subject=subject, issuer=JWT_ISSUER, audience=JWT_AUDIENCE)
        assert await mcp_status(base_url, forged) == 401, subject
