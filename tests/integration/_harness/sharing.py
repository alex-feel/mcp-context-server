"""Real-server checks for the sharing tools and the read-back of visibility and ownership.

The primary client owns every entry it stores. The checks drive the three sharing
tools against such entries: registration (ungated), an idempotent share whose grant
records the primary principal as its grantor, an unshare that removes exactly the
row it names and reports what the grantee keeps, grants that leave the entry's
``version`` and ``updated_at`` untouched, the refusal of invalid arguments, the
``visibility`` and ``is_owner`` every entry-carrying read reports to the owner, and
the grants removed together with their entry. A real grantee reading through its
grants is the second principal of ``access_scoping.py``, which shares the
module-level helpers: the sharing calls, the refusal texts a non-owner gets, the
grant rows and the read-back check.

``version`` and the grant rows are no tool's output, so they are read through the
database itself: the primary server's SQLite file, or the superuser connection of
``pg_admin_url``, who sees every row whatever the database policies admit.
"""

import asyncio
import sqlite3
from collections.abc import Mapping
from typing import Any

from mcp.types import TextContent

from app.settings.auth import AccessControlSettings
from tests.integration._harness.core import HarnessCore

SHARING_TOOLS = frozenset({'share_context', 'unshare_context', 'list_context_grants'})
GRANTEE = 'harness-reader'
GRANTEE_GROUP = 'harness-team'
READ_BACK_TOKEN = 'quillfeather'
# A well-formed id no entry carries.
ABSENT_ID = '0190abcdef1234567890abcdef0eeeee'
# What each sharing tool answers a caller that may read an entry but does not own it.
NOT_OWNER_REFUSALS = {
    'share_context': 'Not authorized to share context entry with ID {id}',
    'unshare_context': 'Not authorized to unshare context entry with ID {id}',
    'list_context_grants': 'Not authorized to list the grants of context entry with ID {id}',
}

# A tool call's structured content, or its error text when the tool refused the call.
type SharingOutcome = tuple[dict[str, Any] | None, str | None]
# A tool name and its arguments.
type SharingCall = tuple[str, dict[str, Any]]
# The rows one read tool returned for one entry.
type ReadBackRows = list[tuple[str, dict[str, Any]]]


def grant_calls(context_id: str, principal_id: str) -> list[SharingCall]:
    """Return a share and an unshare of a user read grant on one entry, and its grant listing.

    Args:
        context_id: The entry's id, or the ``{id}`` placeholder of an argument template.
        principal_id: The user the share and the unshare name.

    Returns:
        One call per sharing tool.
    """
    grant = {'context_id': context_id, 'principal_type': 'user', 'principal_id': principal_id, 'permission': 'read'}
    return [('share_context', grant), ('unshare_context', grant), ('list_context_grants', {'context_id': context_id})]


def primary_principal(server_env: Mapping[str, str] | None) -> str:
    """Return the principal the primary server runs every request as.

    Args:
        server_env: The environment the primary server was started with.

    Returns:
        Its ``ACCESS_CONTROL_DEFAULT_PRINCIPAL``, or the setting's default.
    """
    configured = (server_env or {}).get('ACCESS_CONTROL_DEFAULT_PRINCIPAL')
    return configured or str(AccessControlSettings.model_fields['default_principal'].default)


def grant_rows(grants: list[dict[str, Any]]) -> list[tuple[str, str, str, str]]:
    """Reduce a ``list_context_grants`` result to comparable rows.

    Args:
        grants: The listed grants.

    Returns:
        ``(principal_type, principal_id, permission, granted_by)`` per grant, in order.
    """
    return [(g['principal_type'], g['principal_id'], g['permission'], g['granted_by']) for g in grants]


def read_back_problems(rows: ReadBackRows, visibility: str, *, is_owner: bool) -> list[str]:
    """Check the access fields of entry rows a read tool returned.

    Args:
        rows: ``(tool, row)`` pairs, the row being an entry, a grep row or a response.
        visibility: The visibility the entry was stored with.
        is_owner: Whether the reading principal owns the entry.

    Returns:
        The problems found.
    """
    problems: list[str] = []
    for tool, row in rows:
        reported = row.get('is_owner')
        if row.get('visibility') != visibility or type(reported) is not bool or reported is not is_owner:
            problems.append(
                f'{tool} reported visibility {row.get("visibility")!r} and is_owner {reported!r}, '
                f'expected {visibility!r} and {is_owner}',
            )
        if 'owner_id' in row:
            problems.append(f'{tool} returned owner_id')
    return problems


def outline_access_field_problems(root: dict[str, Any] | None) -> list[str]:
    """Check that no node of a ``navigate_context`` outline carries the access fields.

    Args:
        root: The outline's root node.

    Returns:
        The problems found; an outline without a root or without a heading node is one.
    """
    if root is None or not root.get('children'):
        return [f'navigate_context returned the outline {root!r}, expected a root with heading nodes']
    problems: list[str] = []
    nodes = [root]
    while nodes:
        node = nodes.pop()
        carried = sorted({'visibility', 'is_owner', 'owner_id'} & set(node))
        if carried:
            problems.append(f'navigate_context reported {carried} on the outline node {node.get("node_id")!r}')
        nodes.extend(node.get('children', []))
    return problems


class SharingMixin(HarnessCore):
    """Checks for the sharing tools and the read-back on a real server."""

    async def _sharing_outcome(self, tool: str, arguments: dict[str, Any]) -> SharingOutcome:
        """Call a tool through the primary client without raising on a tool error.

        Args:
            tool: The tool name.
            arguments: The tool arguments.

        Returns:
            The extracted content and None, or None and the error text.
        """
        assert self.client is not None
        result = await self.client.call_tool(tool, arguments, raise_on_error=False)
        if result.is_error:
            return None, ' '.join(block.text for block in result.content if isinstance(block, TextContent))
        return self._extract_content(result), None

    async def _sharing_content(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool through the primary client that must succeed.

        Args:
            tool: The tool name.
            arguments: The tool arguments.

        Returns:
            The extracted content.

        Raises:
            AssertionError: If the tool reported an error.
        """
        data, error = await self._sharing_outcome(tool, arguments)
        if data is None:
            raise AssertionError(f'{tool} failed: {error}')
        return data

    async def _store_owned(self, suffix: str, text: str, *, visibility: str = 'private') -> str:
        """Store an entry as the primary principal in a thread of its own.

        Args:
            suffix: Names the thread.
            text: The entry's text.
            visibility: The entry's visibility.

        Returns:
            The stored entry's id.
        """
        stored = await self._sharing_content('store_context', {
            'thread_id': f'{self.test_thread_id}_sharing_{suffix}', 'source': 'agent', 'text': text,
            'visibility': visibility,
        })
        return str(stored['context_id'])

    async def _stored_state(self, context_id: str) -> tuple[int | None, int]:
        """Read an entry's ``version`` and its number of grant rows through the database.

        Args:
            context_id: The entry's id.

        Returns:
            The version (None when no such entry exists) and the grant row count.
        """
        if self.backend == 'postgresql':
            import asyncpg

            assert self.pg_admin_url is not None  # __init__ requires it on PostgreSQL
            conn = await asyncpg.connect(self.pg_admin_url)
            try:
                version = await conn.fetchval('SELECT version FROM context_entries WHERE id = $1::uuid', context_id)
                grants = await conn.fetchval(
                    'SELECT count(*) FROM context_entry_grants WHERE context_entry_id = $1::uuid', context_id,
                )
            finally:
                await conn.close()
            return (None if version is None else int(version)), int(grants)

        assert self.temp_db_path is not None  # connect_client requires it on SQLite

        def read() -> tuple[int | None, int]:
            with sqlite3.connect(f'file:{self.temp_db_path}?mode=ro', uri=True, timeout=30) as conn:
                row = conn.execute('SELECT version FROM context_entries WHERE id = ?', (context_id,)).fetchone()
                (grants,) = conn.execute(
                    'SELECT count(*) FROM context_entry_grants WHERE context_entry_id = ?', (context_id,),
                ).fetchone()
            return (None if row is None else int(row[0])), int(grants)

        return await asyncio.to_thread(read)

    async def _updated_at(self, context_id: str) -> str | None:
        """Read an entry's ``updated_at`` through ``get_context_by_ids``.

        Args:
            context_id: The entry's id.

        Returns:
            The timestamp, or None when the entry is not returned.
        """
        data = await self._sharing_content('get_context_by_ids', {'context_ids': [context_id]})
        rows = [row for row in data.get('results', []) if row.get('id') == context_id]
        return rows[0].get('updated_at') if rows else None

    def _sharing_record(self, test_name: str, problems: list[str], passed: str) -> bool:
        """Record a check's outcome.

        Args:
            test_name: The check's name.
            problems: The problems found.
            passed: The detail recorded when there are none.

        Returns:
            bool: True when there are no problems.
        """
        self.test_results.append((test_name, not problems, '; '.join(problems) or passed))
        return not problems

    async def test_sharing_tools_registered(self) -> bool:
        """Verify the server registers the three sharing tools.

        Not gated on ``registered_tools``: the sharing tools are core tools with no
        prerequisite, so a server that lacks one has a registration defect.

        Returns:
            bool: True if test passed.
        """
        test_name = 'sharing_tools_registered'
        assert self.client is not None
        try:
            registered = {tool.name for tool in await self.client.list_tools()}
            missing = sorted(SHARING_TOOLS - registered)
            problems = [f'the sharing tools {missing} are not registered'] if missing else []
            return self._sharing_record(test_name, problems, 'share, unshare and list tools registered')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_share_context_idempotent(self) -> bool:
        """Share one grant twice, then a second permission, and list the grants.

        The first share creates the grant and the repeat creates nothing; the listing
        holds that one grant, recorded as granted by the primary principal. A write
        grant to a group is a row of its own, listed in principal type, id and
        permission order.

        Returns:
            bool: True if test passed.
        """
        test_name = 'share_context_idempotent'
        try:
            context_id = await self._store_owned('share', 'Entry the primary principal shares twice.')
            grantor = primary_principal(self.server_env)
            grant = {'context_id': context_id, 'principal_type': 'user', 'principal_id': GRANTEE, 'permission': 'read'}
            problems: list[str] = []
            first = await self._sharing_content('share_context', grant)
            second = await self._sharing_content('share_context', grant)
            for share, created in ((first, True), (second, False)):
                identity = tuple(share.get(key) for key in ('context_id', 'principal_type', 'principal_id', 'permission'))
                if (share.get('success'), share.get('created'), identity) != (True, created, tuple(grant.values())):
                    problems.append(f'a share expected to report created={created} returned {share}')
            listed = await self._sharing_content('list_context_grants', {'context_id': context_id})
            if (listed.get('context_id'), listed.get('count')) != (context_id, 1):
                problems.append(f'list_context_grants after the repeated share returned {listed}')
            if grant_rows(listed.get('grants', [])) != [('user', GRANTEE, 'read', grantor)]:
                problems.append(f'the listed grant is {listed.get("grants")}, expected read for {GRANTEE} by {grantor}')

            await self._sharing_content('share_context', {
                'context_id': context_id, 'principal_type': 'group', 'principal_id': GRANTEE_GROUP, 'permission': 'write',
            })
            listed = await self._sharing_content('list_context_grants', {'context_id': context_id})
            expected = [('group', GRANTEE_GROUP, 'write', grantor), ('user', GRANTEE, 'read', grantor)]
            if grant_rows(listed.get('grants', [])) != expected or listed.get('count') != 2:
                problems.append(f'the grants are {listed.get("grants")}, expected {expected}')
            if (await self._stored_state(context_id))[1] != 2:
                problems.append('the database does not hold exactly the two listed grant rows')
            return self._sharing_record(test_name, problems, 'one row per grant; repeats change nothing')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_unshare_context_exact_row(self) -> bool:
        """Unshare one grant twice beside a second permission of the same grantee.

        The grantee holds read and write. Removing read reports removed and the write
        the grantee keeps; repeating it removes nothing; removing write leaves no
        grant.

        Returns:
            bool: True if test passed.
        """
        test_name = 'unshare_context_exact_row'
        try:
            context_id = await self._store_owned('unshare', 'Entry the primary principal unshares exactly.')
            grant = {'context_id': context_id, 'principal_type': 'user', 'principal_id': GRANTEE}
            for permission in ('read', 'write'):
                await self._sharing_content('share_context', {**grant, 'permission': permission})
            problems: list[str] = []
            for permission, removed, remaining in (('read', True, ['write']), ('read', False, ['write']), ('write', True, [])):
                data = await self._sharing_content('unshare_context', {**grant, 'permission': permission})
                outcome = (data.get('success'), data.get('permission'), data.get('removed'), data.get('remaining_permissions'))
                if outcome != (True, permission, removed, remaining):
                    problems.append(
                        f'unshare of {permission} returned {data}, expected removed={removed} and remaining {remaining}',
                    )
            listed = await self._sharing_content('list_context_grants', {'context_id': context_id})
            if listed.get('grants') != [] or listed.get('count') != 0:
                problems.append(f'grants remain after both unshares: {listed}')
            if (await self._stored_state(context_id))[1] != 0:
                problems.append('grant rows remain in the database after both unshares')
            return self._sharing_record(test_name, problems, 'each unshare removed exactly the named grant')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_grants_leave_version_and_updated_at(self) -> bool:
        """Verify a share and an unshare change neither the entry's version nor its updated_at.

        Grants sit outside the update compare-and-set, so a concurrent update of the
        entry never conflicts with one.

        Returns:
            bool: True if test passed.
        """
        test_name = 'grants_leave_version_and_updated_at'
        try:
            context_id = await self._store_owned('version', 'Entry whose version grants never move.')
            grant = {'context_id': context_id, 'principal_type': 'user', 'principal_id': GRANTEE, 'permission': 'write'}
            states = [(await self._stored_state(context_id))[0], await self._updated_at(context_id)]
            problems: list[str] = []
            if states[0] is None or states[1] is None:
                problems.append(f'the stored entry reads as version {states[0]!r}, updated_at {states[1]!r}')
            for tool in ('share_context', 'unshare_context'):
                # Cross a whole-second boundary so a moved updated_at is observable rather
                # than rounded away by SQLite's second-granularity CURRENT_TIMESTAMP.
                await asyncio.sleep(1.1)
                await self._sharing_content(tool, grant)
                after = [(await self._stored_state(context_id))[0], await self._updated_at(context_id)]
                if after != states:
                    problems.append(f'{tool} moved (version, updated_at) from {states} to {after}')
            return self._sharing_record(test_name, problems, 'version and updated_at unchanged by share and unshare')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_sharing_rejects_invalid_arguments(self) -> bool:
        """Verify the sharing tools refuse invalid principal ids, a self share and an absent entry.

        Returns:
            bool: True if test passed.
        """
        test_name = 'sharing_rejects_invalid_arguments'
        try:
            context_id = await self._store_owned('invalid', 'Entry the invalid sharing arguments name.')
            owner = primary_principal(self.server_env)
            problems: list[str] = []
            refusals: list[tuple[str, str, str, str]] = [
                (tool, principal_type, principal_id, expected)
                for tool in ('share_context', 'unshare_context')
                for principal_type, principal_id, expected in (
                    ('user', '', 'principal_id cannot be empty or whitespace'),
                    ('group', '   ', 'principal_id cannot be empty or whitespace'),
                    ('user', f' {GRANTEE}', 'principal_id cannot have leading or trailing whitespace'),
                    ('group', 'x' * 257, 'principal_id exceeds 256 characters'),
                    ('user', 'nul\x00id', 'principal_id: '),
                )
            ]
            self_share = f'Cannot share context entry with ID {context_id} with its owner'
            refusals.append(('share_context', 'user', owner, self_share))
            for tool, principal_type, principal_id, expected in refusals:
                _, error = await self._sharing_outcome(tool, {
                    'context_id': context_id, 'principal_type': principal_type, 'principal_id': principal_id,
                    'permission': 'read',
                })
                if error is None or not error.startswith(expected):
                    problems.append(f'{tool} for {principal_type} {principal_id[:16]!r} returned {error!r}')

            not_found = f'Context entry with ID {ABSENT_ID} not found'
            for tool, arguments in grant_calls(ABSENT_ID, GRANTEE):
                _, error = await self._sharing_outcome(tool, arguments)
                if error != not_found:
                    problems.append(f'{tool} on an absent entry returned {error!r}, expected {not_found!r}')
            grant = {'principal_type': 'user', 'principal_id': GRANTEE, 'permission': 'read'}
            for arguments in ({**grant, 'context_id': context_id, 'permission': 'admin'}, {**grant, 'context_id': 'abc'}):
                data, error = await self._sharing_outcome('share_context', arguments)
                if error is None:
                    problems.append(f'share_context accepted {arguments}: {data}')

            listed = await self._sharing_content('list_context_grants', {'context_id': context_id})
            if listed.get('count') != 0:
                problems.append(f'a refused call stored a grant: {listed}')
            return self._sharing_record(test_name, problems, 'invalid arguments refused, nothing stored')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_read_back_on_own_entries(self) -> bool:
        """Verify every entry-carrying read reports visibility and is_owner to the owner.

        A private and a public entry of the primary principal are read through
        ``get_context_by_ids``, the four searches, ``grep_context`` in each output mode,
        ``navigate_context`` and ``read_context_range``: each row reports the stored
        visibility and ``is_owner`` true as a boolean, never ``owner_id``, and the
        outline nodes carry neither field.

        Returns:
            bool: True if test passed.
        """
        test_name = 'read_back_on_own_entries'
        try:
            thread = f'{self.test_thread_id}_sharing_readback'
            entries: dict[str, str] = {}
            texts: dict[str, str] = {}
            for visibility in ('private', 'public'):
                text = f'# Read-back {visibility}\n\nA {visibility} entry of the primary principal carrying {READ_BACK_TOKEN}.'
                stored = await self._sharing_content('store_context', {
                    'thread_id': thread, 'source': 'agent', 'text': text, 'visibility': visibility,
                })
                entries[str(stored['context_id'])] = visibility
                texts[str(stored['context_id'])] = text

            reads: list[tuple[str, dict[str, Any], str]] = [
                ('get_context_by_ids', {'context_ids': list(entries)}, 'id'),
                ('search_context', {'thread_id': thread}, 'id'),
            ]
            for tool in ('semantic_search_context', 'fts_search_context', 'hybrid_search_context'):
                if tool in self.registered_tools:
                    query = next(iter(texts.values())) if tool == 'semantic_search_context' else READ_BACK_TOKEN
                    reads.append((tool, {'query': query, 'thread_id': thread}, 'id'))
            reads.extend(
                ('grep_context', {'pattern': READ_BACK_TOKEN, 'thread_id': thread, 'output_mode': mode}, 'context_id')
                for mode in ('files_with_matches', 'content', 'count')
            )

            problems: list[str] = []
            rows_by_entry: dict[str, ReadBackRows] = {context_id: [] for context_id in entries}
            for tool, arguments, key in reads:
                data = await self._sharing_content(tool, arguments)
                label = f'{tool}({arguments.get("output_mode", "")})' if tool == 'grep_context' else tool
                returned = {str(row[key]) for row in data.get('results', [])}
                if returned != set(entries):
                    problems.append(f'{label} returned {sorted(returned)}, expected both entries')
                for row in data.get('results', []):
                    rows_by_entry.setdefault(str(row[key]), []).append((label, row))
            for context_id in entries:
                for tool, arguments in (
                    ('navigate_context', {'context_id': context_id}),
                    ('read_context_range', {'context_id': context_id, 'start_line': 1, 'end_line': 1}),
                ):
                    data = await self._sharing_content(tool, arguments)
                    rows_by_entry[context_id].append((tool, data))
                    if tool == 'navigate_context':
                        problems.extend(outline_access_field_problems(data.get('root')))

            for context_id, visibility in entries.items():
                problems.extend(read_back_problems(rows_by_entry[context_id], visibility, is_owner=True))
            return self._sharing_record(test_name, problems, f'{len(reads) + 2} read tools report the access fields')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_removes_grants(self) -> bool:
        """Verify deleting an entry removes its grants.

        The grant rows are counted through the database before and after the delete,
        and the listing of the deleted entry reports it not found.

        Returns:
            bool: True if test passed.
        """
        test_name = 'delete_removes_grants'
        try:
            context_id = await self._store_owned('delete', 'Entry deleted together with its grants.')
            for principal_type, principal_id in (('user', GRANTEE), ('group', GRANTEE_GROUP)):
                await self._sharing_content('share_context', {
                    'context_id': context_id, 'principal_type': principal_type, 'principal_id': principal_id,
                    'permission': 'read',
                })
            problems: list[str] = []
            before = (await self._stored_state(context_id))[1]
            if before != 2:
                problems.append(f'{before} grant rows before the delete, expected 2')
            deleted = await self._sharing_content('delete_context', {'context_ids': [context_id]})
            if deleted.get('deleted_count') != 1:
                problems.append(f'delete_context returned {deleted}')
            _, error = await self._sharing_outcome('list_context_grants', {'context_id': context_id})
            if error != f'Context entry with ID {context_id} not found':
                problems.append(f'list_context_grants on the deleted entry returned {error!r}')
            after = await self._stored_state(context_id)
            if after != (None, 0):
                problems.append(f'after the delete the database holds version {after[0]} and {after[1]} grant rows')
            return self._sharing_record(test_name, problems, 'the delete removed the entry and its grants')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
