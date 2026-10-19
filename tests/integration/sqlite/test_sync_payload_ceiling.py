"""Corpus sync payloads at the limits of the Streamable HTTP transport (SQLite, over HTTP).

The caps of the sync protocol keep every request and every response under limits
the transport fixes and no deployment configures: the MCP SDK answers a request
body above 4,194,304 bytes with HTTP 413 before parsing it, and a Python SDK client
fails on a Server-Sent Events message above 1,048,576 bytes, the framing of every
response in the legacy protocol era. These tests drive the sync tools of a real HTTP
server with the largest values the caps admit, once per protocol era: a manifest of
the most items, each with the largest source id, is planned; one item more is refused
with the validation error, not a 413; the largest plan response (an update for every
item and the most deletes, each naming the largest source id) and the largest
ingest-state page arrive. A raw request one byte over the body limit is answered 413,
while one at the limit is not, which pins the limit the sync client keeps below.

The server is ``tests/run_server.py`` started by ``running_http_server`` from the
explicit environment ``build_env`` builds, with generation off.
"""

from collections.abc import Iterator
from operator import itemgetter
from typing import Any

import httpx
import pytest
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from mcp.types import TextContent

from app.ingest import MAX_INGEST_STATE_PAGE
from app.ingest import MAX_MANIFEST_ITEMS
from app.ingest import MAX_PLAN_DELETES
from app.ingest import MAX_SOURCE_ID_LENGTH
from tests.integration._harness.core import CLIENT_MODES
from tests.integration._harness.core import ClientMode
from tests.integration._harness.sync import managed_entry
from tests.integration._harness.sync import manifest_item
from tests.integration._harness.sync import maximal_source_id
from tests.integration._harness.sync import meta_digest
from tests.integration._harness.sync import plan_shape
from tests.integration._harness.sync import plan_with
from tests.integration._http_jwt import build_env
from tests.integration._http_jwt import free_port
from tests.integration._http_jwt import running_http_server

pytestmark = pytest.mark.integration

# The request body limit of the MCP SDK's HTTP transports.
REQUEST_BODY_LIMIT = 4_194_304
# The most entries store_context_batch takes in one call.
BATCH_SIZE = 100


@pytest.fixture(scope='module')
def payload_server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """Run one HTTP server without authentication for every test of the module.

    Yields:
        The server origin URL.
    """
    database = tmp_path_factory.mktemp('sync_payload') / 'sync_payload.db'
    env = build_env(backend='sqlite', database=str(database), port=free_port(), auth=False)
    with running_http_server(env) as base_url:
        yield base_url


def _client(base_url: str, client_mode: ClientMode) -> Client[Any]:
    """Build an unconnected client of the server's MCP endpoint.

    Returns:
        The client, negotiating ``client_mode``.
    """
    return Client(StreamableHttpTransport(url=f'{base_url}/mcp'), mode=client_mode)


def _maximal_manifest(count: int, digest: str) -> list[dict[str, str]]:
    """Return ``count`` manifest items, each naming the largest source id.

    Returns:
        The items, their texts distinct.
    """
    return [manifest_item(maximal_source_id(number), f'Document {number}.', digest) for number in range(count)]


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_manifest_at_the_cap_is_planned(payload_server: str, client_mode: ClientMode) -> None:
    """The most items, each with a source id of the most four-byte code points, are planned."""
    manifest = _maximal_manifest(MAX_MANIFEST_ITEMS, meta_digest('cap'))
    assert len(manifest[0]['source_id'].encode('utf-8')) == 4 * MAX_SOURCE_ID_LENGTH

    async with _client(payload_server, client_mode) as client:
        plan = await plan_with(client, f'kb:payload-cap-{client_mode}', manifest)

    assert plan == plan_shape(creates=list(range(MAX_MANIFEST_ITEMS)))


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_manifest_past_the_cap_gets_the_validation_error(payload_server: str, client_mode: ClientMode) -> None:
    """One item past the cap reaches the tool and is refused by validation, not by the transport."""
    manifest = _maximal_manifest(MAX_MANIFEST_ITEMS + 1, meta_digest('past-cap'))

    async with _client(payload_server, client_mode) as client:
        result = await client.call_tool(
            'plan_context_sync',
            {'thread_id': f'kb:payload-past-cap-{client_mode}', 'manifest': manifest},
            raise_on_error=False,
        )

    assert result.is_error
    text = ' '.join(block.text for block in result.content if isinstance(block, TextContent))
    assert f'List should have at most {MAX_MANIFEST_ITEMS} items' in text


@pytest.mark.asyncio
@pytest.mark.parametrize('client_mode', CLIENT_MODES)
async def test_largest_plan_response_and_state_page_arrive(payload_server: str, client_mode: ClientMode) -> None:
    """A plan of the most updates and deletes, and a full state page, all naming the largest source ids, arrive.

    The corpus holds one entry per manifest item and the most deletes beyond them. The
    manifest names every entry but the last ``MAX_PLAN_DELETES`` with another metadata
    digest, so every item is a metadata update and every unnamed entry an absent delete.
    """
    thread = f'kb:payload-response-{client_mode}'
    stored_digest, planned_digest = meta_digest('stored'), meta_digest('planned')
    total = MAX_MANIFEST_ITEMS + MAX_PLAN_DELETES
    entries = [
        managed_entry(thread, maximal_source_id(number), f'Document {number}.', stored_digest) for number in range(total)
    ]

    async with _client(payload_server, client_mode) as client:
        ids: list[str] = []
        for start in range(0, total, BATCH_SIZE):
            batch = entries[start:start + BATCH_SIZE]
            stored = await client.call_tool('store_context_batch', {'entries': batch, 'atomic': True})
            assert stored.structured_content is not None
            assert stored.structured_content['succeeded'] == len(batch)
            results = sorted(stored.structured_content['results'], key=itemgetter('index'))
            ids.extend(str(item['context_id']) for item in results)
        plan = await plan_with(client, thread, _maximal_manifest(MAX_MANIFEST_ITEMS, planned_digest))
        page = await client.call_tool('list_ingest_state', {'thread_id': thread, 'limit': MAX_INGEST_STATE_PAGE})

    assert plan['creates'] == []
    assert plan['updates'] == [
        {'index': index, 'context_id': ids[index], 'change': 'metadata'} for index in range(MAX_MANIFEST_ITEMS)
    ]
    assert plan['deletes_total'] == MAX_PLAN_DELETES
    assert {delete['context_id']: delete for delete in plan['deletes']} == {
        ids[number]: {'context_id': ids[number], 'source_id': maximal_source_id(number), 'reason': 'absent'}
        for number in range(MAX_MANIFEST_ITEMS, total)
    }
    assert page.structured_content is not None
    listed = page.structured_content['entries']
    assert page.structured_content['count'] == len(listed) == MAX_INGEST_STATE_PAGE
    assert page.structured_content['next_cursor'] is not None
    assert all(len(entry['source_id']) == MAX_SOURCE_ID_LENGTH for entry in listed)


def test_request_body_over_the_limit_is_refused_with_413(payload_server: str) -> None:
    """A body of exactly the limit passes the transport, and one byte more is answered 413."""
    message = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    at_limit = message + b' ' * (REQUEST_BODY_LIMIT - len(message))
    headers = {'Content-Type': 'application/json', 'Accept': 'application/json, text/event-stream'}
    url = f'{payload_server}/mcp'

    assert httpx.post(url, content=at_limit, headers=headers, timeout=60.0).status_code != 413
    assert httpx.post(url, content=at_limit + b' ', headers=headers, timeout=60.0).status_code == 413
