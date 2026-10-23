"""The JWT access scenario: what owners and grantees read, modify, share and delete over HTTP.

``verify_jwt_access_scoping`` runs on both backends
(``tests/integration/sqlite/test_http_jwt_access.py`` and
``tests/integration/postgresql/test_http_jwt_access.py``): three principals with
minted JWTs -- ``alice`` and ``bob`` in group ``team-x``, ``carol`` in none -- against a
server that stamps a group read grant for every author group
(``ACCESS_CONTROL_DEFAULT_GROUP_GRANTS=author_groups``). alice shares two of her entries
with carol through ``share_context``, one for read and one for write, lists every grant
of her entries, and later unshares carol's read grant; bob and carol, who may read but do
not own those entries, are refused every sharing tool. Every entry a principal reads
reports its visibility and whether that principal owns it. The scenario ends with a
corpus sync, which manages only the caller's own entries whatever grants reach the
others. ``verify_reassigned_owner_manages_grants`` follows it on the same server: the
operator CLI ``mcp-context-server-migrate --reassign-owner`` moves alice's entries to
``erin``, who then manages the grants of an entry alice shared, while alice, still a
reader through her group, may not list, add or remove them. PostgreSQL runs both
a second time with row-level security on, where the database enforces the same grants
and its grants policies decide every sharing call.
"""

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

from fastmcp import Client
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from mcp.client.stdio import get_default_environment

from tests.integration._harness.core import ClientMode
from tests.integration._harness.sync import ingest_block
from tests.integration._harness.sync import listed_source_ids
from tests.integration._harness.sync import managed_entry
from tests.integration._harness.sync import manifest_item
from tests.integration._harness.sync import meta_digest
from tests.integration._harness.sync import plan_shape
from tests.integration._harness.sync import plan_with
from tests.integration._http_jwt import ALICE
from tests.integration._http_jwt import BOB
from tests.integration._http_jwt import CAROL
from tests.integration._http_jwt import TEAM
from tests.integration._http_jwt import entry_read_back
from tests.integration._http_jwt import jwt_client
from tests.integration._http_jwt import listed_grants
from tests.integration._http_jwt import mint_token
from tests.integration._http_jwt import read_back
from tests.integration._http_jwt import share
from tests.integration._http_jwt import tool_content
from tests.integration._http_jwt import tool_error
from tests.integration._http_jwt import tool_ids

SCENARIO_TAG = 'jwt-access'
SCENARIO_WORD = 'quasarfern'
GROUP_THREAD = 'jwt_access_group'
GRANTED_THREAD = 'jwt_access_granted'
SYNC_THREAD = 'kb:jwt_access'
# alice's managed sources: carol is granted read on the first and write on the second,
# and only the group grant reaches the third.
SYNC_SOURCES = ('granted-read.md', 'granted-write.md', 'team-only.md')
REASSIGNED_THREAD = 'jwt_access_reassigned'
# The principal the operator hands alice's entries to.
ERIN = 'erin'
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CLI_TIMEOUT_S = 120


async def _store_alice_entries(alice: Client[Any]) -> dict[str, str]:
    """Store alice's four entries under the default ``private`` visibility.

    Args:
        alice: alice's client.

    Returns:
        Entry ids keyed ``group`` (alone in its thread), ``read`` and ``write`` (shared with
        carol) and ``team`` (readable through the group grant only).
    """
    entries: dict[str, str] = {}
    threads = {'group': GROUP_THREAD, 'read': GRANTED_THREAD, 'write': GRANTED_THREAD, 'team': GRANTED_THREAD}
    for label, thread in threads.items():
        stored = await tool_content(alice, 'store_context', {
            'thread_id': thread,
            'source': 'agent',
            'text': f'# {label.title()} entry\n\nAlice {label} entry carrying {SCENARIO_WORD}.',
            'tags': [SCENARIO_TAG],
        })
        entries[label] = str(stored['context_id'])
    return entries


async def verify_jwt_access_scoping(base_url: str, key_pair: RSAKeyPair, client_mode: ClientMode) -> None:
    """Prove what each JWT principal reads, modifies, shares and deletes on a running server.

    bob reads all of alice's private entries through the group grant ``author_groups``
    stamped at store time; carol, in no group, reads only the entry alice shares with her
    for read and the one alice shares with her for write. Every entry read reports its
    visibility and whether the reader owns it. Only alice may list, add or remove the
    grants of her entries. carol's write grant authorizes a text update but neither a
    visibility change nor a delete; her read grant authorizes no update; and a thread
    delete by a grantee deletes nothing. Once alice unshares carol's read grant, carol
    cannot reach that entry. A corpus sync by bob or carol counts alice's readable
    managed entries as foreign and never plans them.

    Args:
        base_url: Origin of the server started with ``build_access_env``.
        key_pair: The key pair the server verifies against.
        client_mode: Protocol era every client negotiates.
    """
    async with (
        jwt_client(base_url, mint_token(key_pair, ALICE, [TEAM]), client_mode) as alice,
        jwt_client(base_url, mint_token(key_pair, BOB, [TEAM]), client_mode) as bob,
        jwt_client(base_url, mint_token(key_pair, CAROL), client_mode) as carol,
    ):
        entry = await _store_alice_entries(alice)
        await share(alice, entry['read'], 'user', CAROL, 'read')
        await share(alice, entry['write'], 'user', CAROL, 'write')
        all_ids = set(entry.values())
        carol_ids = {entry['read'], entry['write']}

        for client, expected, owner, threads, totals in (
            (alice, all_ids, True, {GROUP_THREAD: 1, GRANTED_THREAD: 3}, (4, 2)),
            (bob, all_ids, False, {GROUP_THREAD: 1, GRANTED_THREAD: 3}, (4, 2)),
            (carol, carol_ids, False, {GRANTED_THREAD: 2}, (2, 1)),
        ):
            fields = dict.fromkeys(expected, ('private', owner))
            assert await read_back(client, 'search_context', {'tags': [SCENARIO_TAG]}) == fields
            assert await read_back(client, 'get_context_by_ids', {'context_ids': sorted(all_ids)}) == fields
            assert await read_back(client, 'fts_search_context', {'query': SCENARIO_WORD}) == fields
            for output_mode in ('files_with_matches', 'content', 'count'):
                grep = {'pattern': SCENARIO_WORD, 'output_mode': output_mode}
                assert await read_back(client, 'grep_context', grep) == fields
            listed = (await tool_content(client, 'list_threads', {}))['threads']
            assert {row['thread_id']: row['entry_count'] for row in listed} == threads
            stats = await tool_content(client, 'get_statistics', {})
            assert (stats['total_entries'], stats['total_threads']) == totals

        await _verify_sharing_is_owner_only(alice, bob, carol, entry)

        hidden_from_carol = f'Context entry not found: {entry["group"]}'
        assert await tool_error(carol, 'navigate_context', {'context_id': entry['group']}) == hidden_from_carol
        assert await tool_error(carol, 'read_context_range', {
            'context_id': entry['group'], 'start_line': 1, 'end_line': 1,
        }) == hidden_from_carol
        for client, context_id, owner in (
            (alice, entry['team'], True),
            (bob, entry['group'], False),
            (carol, entry['read'], False),
            (carol, entry['write'], False),
        ):
            outline = await tool_content(client, 'navigate_context', {'context_id': context_id})
            assert (outline['context_id'], entry_read_back(outline)) == (context_id, ('private', owner))
            span = await tool_content(client, 'read_context_range', {'context_id': context_id, 'start_line': 1, 'end_line': 1})
            assert span['text'].startswith('# ')
            assert entry_read_back(span) == ('private', owner)

        new_text = f'Carol rewrote the write entry, still carrying {SCENARIO_WORD}.'
        updated = await tool_content(carol, 'update_context', {'context_id': entry['write'], 'text': new_text})
        assert updated['success'] is True
        rows = (await tool_content(alice, 'get_context_by_ids', {'context_ids': [entry['write']]}))['result']
        assert rows[0]['text_content'] == new_text
        assert await tool_error(carol, 'update_context', {'context_id': entry['read'], 'text': 'x'}) == (
            f'Not authorized to modify context entry with ID {entry["read"]}'
        )
        assert await tool_error(carol, 'update_context', {'context_id': entry['group'], 'text': 'x'}) == (
            f'Context entry with ID {entry["group"]} not found'
        )
        assert await tool_error(carol, 'update_context', {'context_id': entry['write'], 'visibility': 'public'}) == (
            f'Only the owner may change the visibility of context {entry["write"]}'
        )
        assert await tool_error(bob, 'update_context', {'context_id': entry['group'], 'text': 'x'}) == (
            f'Not authorized to modify context entry with ID {entry["group"]}'
        )

        for client, context_id in ((carol, entry['write']), (bob, entry['group'])):
            refusal = f'Not authorized to delete context entries: {context_id}'
            assert await tool_error(client, 'delete_context', {'context_ids': [context_id]}) == refusal
            assert await tool_error(client, 'delete_context_batch', {'context_ids': [context_id]}) == refusal
        for client in (bob, carol):
            assert (await tool_content(client, 'delete_context', {'thread_id': GRANTED_THREAD}))['deleted_count'] == 0
            deleted = await tool_content(client, 'delete_context_batch', {'thread_ids': [GRANTED_THREAD]})
            assert deleted['deleted_count'] == 0
        assert await tool_ids(alice, 'get_context_by_ids', {'context_ids': sorted(all_ids)}) == all_ids

        deleted = await tool_content(alice, 'delete_context_batch', {'thread_ids': [GROUP_THREAD]})
        assert deleted['deleted_count'] == 1
        assert await tool_ids(bob, 'get_context_by_ids', {'context_ids': [entry['group']]}) == set()

        await _verify_unshare_revokes_access(alice, bob, carol, entry)
        await _verify_sync_manages_own_entries(alice, bob, carol)


async def _verify_sharing_is_owner_only(
    alice: Client[Any], bob: Client[Any], carol: Client[Any], entry: dict[str, str],
) -> None:
    """Prove only alice lists the grants of her entries, and only she may add or remove one.

    alice sees team-x's ``author_groups`` read grant on every entry and carol's user grant
    on each entry shared with her, all granted by alice. carol (a write grantee) and bob
    (a group reader) may read the entries but are refused every sharing tool on them,
    and an entry carol cannot read is not found to her.

    Args:
        alice: alice's client.
        bob: bob's client.
        carol: carol's client.
        entry: alice's entry ids, as :func:`_store_alice_entries` returns them.
    """
    team = ('group', TEAM, 'read', ALICE)
    for context_id, grants in (
        (entry['group'], [team]),
        (entry['read'], [team, ('user', CAROL, 'read', ALICE)]),
        (entry['write'], [team, ('user', CAROL, 'write', ALICE)]),
        (entry['team'], [team]),
    ):
        assert await listed_grants(alice, context_id) == grants

    for client, context_id, grantee in ((carol, entry['write'], BOB), (bob, entry['group'], CAROL)):
        grant = {'context_id': context_id, 'principal_type': 'user', 'principal_id': grantee, 'permission': 'read'}
        assert await tool_error(client, 'share_context', grant) == (
            f'Not authorized to share context entry with ID {context_id}'
        )
        assert await tool_error(client, 'unshare_context', grant) == (
            f'Not authorized to unshare context entry with ID {context_id}'
        )
        assert await tool_error(client, 'list_context_grants', {'context_id': context_id}) == (
            f'Not authorized to list the grants of context entry with ID {context_id}'
        )
    hidden = entry['group']
    not_found = f'Context entry with ID {hidden} not found'
    assert await tool_error(carol, 'share_context', {
        'context_id': hidden, 'principal_type': 'user', 'principal_id': BOB, 'permission': 'read',
    }) == not_found
    assert await tool_error(carol, 'list_context_grants', {'context_id': hidden}) == not_found
    assert await listed_grants(alice, entry['group']) == [team]


async def _verify_unshare_revokes_access(
    alice: Client[Any], bob: Client[Any], carol: Client[Any], entry: dict[str, str],
) -> None:
    """Prove an unshare removes exactly the named grant and the access only it gave.

    alice removes carol's read grant, then removes it again, which changes nothing.
    carol then cannot read that entry but still reads the one shared with her for write,
    and bob keeps reading it through the group grant.

    Args:
        alice: alice's client.
        bob: bob's client.
        carol: carol's client.
        entry: alice's entry ids, as :func:`_store_alice_entries` returns them.
    """
    grant = {'context_id': entry['read'], 'principal_type': 'user', 'principal_id': CAROL, 'permission': 'read'}
    for removed in (True, False):
        result = await tool_content(alice, 'unshare_context', grant)
        assert (result['removed'], result['remaining_permissions']) == (removed, []), result
    assert await listed_grants(alice, entry['read']) == [('group', TEAM, 'read', ALICE)]
    assert await tool_ids(carol, 'get_context_by_ids', {'context_ids': [entry['read'], entry['write']]}) == {entry['write']}
    assert await tool_error(carol, 'navigate_context', {'context_id': entry['read']}) == (
        f'Context entry not found: {entry["read"]}'
    )
    assert await tool_ids(bob, 'get_context_by_ids', {'context_ids': [entry['read']]}) == {entry['read']}


async def _verify_sync_manages_own_entries(alice: Client[Any], bob: Client[Any], carol: Client[Any]) -> None:
    """Prove a corpus sync plans and lists only the caller's own managed entries.

    alice stores three managed entries of one corpus, each readable to bob through the
    ``author_groups`` grant, and shares the first with carol for read and the second for
    write. Each principal plans the full manifest: alice finds her entries unchanged,
    while bob and carol plan every source as a create and count the managed entries
    they can read as foreign -- a write grant included -- so no plan ever updates or
    deletes alice's entries, not even an empty one. Only alice's listing shows them.
    carol may still rewrite the block of the entry she may write; that changes nothing
    anyone may read or write, and alice's next plan updates the entry's metadata.

    Args:
        alice: alice's client.
        bob: bob's client.
        carol: carol's client.
    """
    digest = meta_digest('jwt-access')
    texts = {source: f'Alice managed entry {source} of the corpus.' for source in SYNC_SOURCES}
    stored = await tool_content(alice, 'store_context_batch', {
        'entries': [managed_entry(SYNC_THREAD, source, text, digest) for source, text in texts.items()],
        'atomic': False,
    })
    assert stored['succeeded'] == len(SYNC_SOURCES), stored
    ids = {SYNC_SOURCES[item['index']]: str(item['context_id']) for item in stored['results']}
    await share(alice, ids['granted-read.md'], 'user', CAROL, 'read')
    await share(alice, ids['granted-write.md'], 'user', CAROL, 'write')

    manifest = [manifest_item(source, text, digest) for source, text in texts.items()]
    creates = list(range(len(SYNC_SOURCES)))
    for client, expected, expected_empty, listed in (
        (alice, plan_shape(unchanged=3), None, sorted(SYNC_SOURCES)),
        (bob, plan_shape(creates=creates, foreign=3), plan_shape(foreign=3), []),
        (carol, plan_shape(creates=creates, foreign=2), plan_shape(foreign=2), []),
    ):
        assert await plan_with(client, SYNC_THREAD, manifest) == expected
        if expected_empty is not None:
            assert await plan_with(client, SYNC_THREAD, [], allow_empty_manifest=True) == expected_empty
        assert await listed_source_ids(client, SYNC_THREAD) == listed

    # carol's write grant lets her rewrite the block of alice's entry. The block decides
    # no access: the rewrite only makes alice's next plan restore her metadata.
    rewritten = 'granted-write.md'
    patched = await tool_content(carol, 'update_context', {
        'context_id': ids[rewritten],
        'metadata_patch': {'ingest': ingest_block(SYNC_THREAD, rewritten, meta_digest('rewritten'))},
    })
    assert patched['success'] is True
    assert await plan_with(alice, SYNC_THREAD, manifest) == plan_shape(
        updates=[{'index': SYNC_SOURCES.index(rewritten), 'context_id': ids[rewritten], 'change': 'metadata'}],
        unchanged=2,
    )
    assert await plan_with(carol, SYNC_THREAD, manifest) == plan_shape(creates=creates, foreign=2)
    assert await tool_ids(carol, 'get_context_by_ids', {'context_ids': sorted(ids.values())}) == {
        ids['granted-read.md'], ids[rewritten],
    }
    assert await tool_error(carol, 'delete_context', {'context_ids': [ids[rewritten]]}) == (
        f'Not authorized to delete context entries: {ids[rewritten]}'
    )
    assert await tool_ids(bob, 'get_context_by_ids', {'context_ids': sorted(ids.values())}) == set(ids.values())


def reassign_owner(source_url: str, from_principal: str, to_principal: str) -> None:
    """Hand every entry of one principal to another with the operator CLI.

    ``mcp-context-server-migrate --reassign-owner`` runs as its own process from the MCP
    SDK's default environment, so nothing from the test process's shell reaches it. The
    server stays up, with no request in flight while the CLI runs.

    Args:
        source_url: The SQLite file path, or a PostgreSQL connection string with the
            credentials of the role that owns the tables.
        from_principal: The current owner.
        to_principal: The new owner.
    """
    command = [sys.executable, '-m', 'app.cli.migrate', '--source-url', source_url]
    completed = subprocess.run(
        [*command, '--reassign-owner', from_principal, to_principal],
        env=get_default_environment(),
        cwd=REPOSITORY_ROOT,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=CLI_TIMEOUT_S,
        check=False,
    )
    stderr = completed.stderr.decode('utf-8', errors='replace')
    assert completed.returncode == 0, stderr
    assert f'from {from_principal!r} to {to_principal!r}' in stderr, stderr


async def verify_reassigned_owner_manages_grants(
    base_url: str, key_pair: RSAKeyPair, client_mode: ClientMode, source_url: str,
) -> None:
    """Prove the owner an operator reassigns an entry to manages its grants, and the previous owner does not.

    alice stores a private entry and shares it with carol for read; ``--reassign-owner``
    then hands alice's entries to erin. erin reads the entry as hers and lists both of
    its grants, each still recorded as granted by alice; she removes carol's, so carol
    cannot reach the entry, and adds one for bob, recorded as granted by her. alice,
    still a reader through team-x's ``author_groups`` grant, reads the entry as not hers
    and is refused every sharing tool on it.

    Args:
        base_url: Origin of the server started with ``build_access_env``.
        key_pair: The key pair the server verifies against.
        client_mode: Protocol era every client negotiates.
        source_url: The server's database, as :func:`reassign_owner` takes it.
    """
    async with jwt_client(base_url, mint_token(key_pair, ALICE, [TEAM]), client_mode) as alice:
        stored = await tool_content(alice, 'store_context', {
            'thread_id': REASSIGNED_THREAD, 'source': 'agent', 'text': 'Alice entry the operator hands to erin.',
        })
        context_id = str(stored['context_id'])
        await share(alice, context_id, 'user', CAROL, 'read')

    await asyncio.to_thread(reassign_owner, source_url, ALICE, ERIN)

    async with (
        jwt_client(base_url, mint_token(key_pair, ERIN), client_mode) as erin,
        jwt_client(base_url, mint_token(key_pair, ALICE, [TEAM]), client_mode) as alice,
        jwt_client(base_url, mint_token(key_pair, CAROL), client_mode) as carol,
    ):
        lookup = {'context_ids': [context_id]}
        assert await read_back(erin, 'get_context_by_ids', lookup) == {context_id: ('private', True)}
        assert await read_back(alice, 'get_context_by_ids', lookup) == {context_id: ('private', False)}
        team = ('group', TEAM, 'read', ALICE)
        assert await listed_grants(erin, context_id) == [team, ('user', CAROL, 'read', ALICE)]

        grant = {'context_id': context_id, 'principal_type': 'user', 'principal_id': CAROL, 'permission': 'read'}
        for tool, arguments, action in (
            ('share_context', grant, 'share'),
            ('unshare_context', grant, 'unshare'),
            ('list_context_grants', {'context_id': context_id}, 'list the grants of'),
        ):
            refusal = await tool_error(alice, tool, arguments)
            assert refusal == f'Not authorized to {action} context entry with ID {context_id}', refusal

        removed = await tool_content(erin, 'unshare_context', grant)
        assert (removed['removed'], removed['remaining_permissions']) == (True, []), removed
        assert await tool_ids(carol, 'get_context_by_ids', lookup) == set()
        await share(erin, context_id, 'user', BOB, 'write')
        assert await listed_grants(erin, context_id) == [team, ('user', BOB, 'write', ERIN)]
