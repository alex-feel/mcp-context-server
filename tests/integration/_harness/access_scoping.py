"""Real-server checks that a second principal reaches only what it may read and modify.

The primary client runs as the default principal ``local``. A second server process
runs as ``bob``: it starts from the primary server's own environment and serves the
same database, so both processes share the rows, the embedding configuration and the
compression layout, and only the principal differs. The primary stores one private
and one public entry; the checks then drive every tool as ``bob``:

- the private entry is unreachable through all twenty-one tools: it is never returned,
  matched, listed or counted, an update and every sharing tool report it not found, a
  delete deletes nothing, and a store of identical text inserts ``bob``'s own entry
  instead of merging into it; the primary reads it as its owner;
- the public entry is readable and reads as public and not owned, while an update
  reports ``Not authorized to modify``, a delete that names it, through either delete
  tool, reports ``Not authorized to delete context entries`` and deletes nothing, and
  each sharing tool refuses it explicitly; a thread delete skips it silently;
- a hidden entry and an absent one produce identical delete counts, not-found
  messages and prefix messages;
- a private entry the primary shares with ``bob`` reads as private and not owned
  while a read grant lasts, refuses his updates and every sharing tool, and accepts
  his text edit but not a visibility change once the grant is a write grant; the
  primary lists both grants as granted by itself, and once it unshares them ``bob``
  reads the entry through the write grant alone and then not at all;
- a corpus sync manages only the caller's own entries: the primary's private managed
  entry is absent from ``bob``'s plan and listing, exactly like a source no entry
  carries, and its public one is counted as foreign, never updated or deleted; a
  ``deduplicate: false`` store by ``bob`` inserts his own copy and leaves the
  primary's entry untouched.
"""

from dataclasses import dataclass
from typing import Any

from fastmcp import Client
from mcp.types import TextContent

from tests.integration._harness.core import HarnessCore
from tests.integration._harness.sharing import NOT_OWNER_REFUSALS
from tests.integration._harness.sharing import grant_calls
from tests.integration._harness.sharing import grant_rows
from tests.integration._harness.sharing import primary_principal
from tests.integration._harness.sharing import read_back_problems
from tests.integration._harness.sync import SYNC_TOOLS
from tests.integration._harness.sync import listed_source_ids
from tests.integration._harness.sync import managed_entry
from tests.integration._harness.sync import manifest_item
from tests.integration._harness.sync import meta_digest
from tests.integration._harness.sync import plan_shape
from tests.integration._harness.sync import plan_with

SECOND_PRINCIPAL = 'bob'
# A grantee no server of the run serves, named by the shares the second principal attempts.
THIRD_PRINCIPAL = 'carol'
PRIVATE_TOKEN = 'zephyrquill'
PUBLIC_TOKEN = 'auroraplume'
SHARED_TOKEN = 'emberlattice'
# A well-formed id no entry carries, and a prefix no UUIDv7 id begins with, so they
# stand for an absent entry next to the hidden one.
ABSENT_ID = '0190abcdef1234567890abcdef0fffff'
ABSENT_PREFIX = 'f' * 24
# Long enough to reach the random bits of a UUIDv7, so the prefix of the hidden
# entry never also names an entry the second principal may read.
PREFIX_LENGTH = 24
RANKED_SEARCH_TOOLS = ('semantic_search_context', 'fts_search_context', 'hybrid_search_context')

# A tool call's structured content, or its error text when the tool refused the call.
type ToolOutcome = tuple[dict[str, Any] | None, str | None]
# A tool name and its arguments.
type ToolCall = tuple[str, dict[str, Any]]
# (total_entries, total_threads) shown to the primary and to the second principal.
type PrincipalTotals = tuple[tuple[int, int], tuple[int, int]]


@dataclass(frozen=True, slots=True)
class ScopingEntries:
    """The two entries the primary principal stores for the checks."""

    private_id: str
    private_thread: str
    private_text: str
    public_id: str
    public_thread: str
    public_text: str


class AccessScopingMixin(HarnessCore):
    """Checks that a second principal sees and changes only what the access model allows."""

    async def test_access_scoping_second_principal(self) -> bool:
        """Drive every tool as a second principal against the primary principal's entries.

        Records one result per aspect: the scoped counts, the private entry, the
        public entry, hidden-equals-absent and the store path.

        Returns:
            bool: True if every aspect passed.
        """
        test_name = 'access_scoping_second_principal'
        assert self.client is not None
        try:
            async with self._second_server(
                {'ACCESS_CONTROL_DEFAULT_PRINCIPAL': SECOND_PRINCIPAL}, share_primary_database=True,
            ) as bob:
                counts_before = (await self._counts(self.client), await self._counts(bob))
                entries = await self._store_scoping_entries()
                counts_after = (await self._counts(self.client), await self._counts(bob))
                originals = await self._entries_by_id(self.client, [entries.private_id, entries.public_id])
                aspects = [
                    ('access_scoping_counts', self._count_problems(counts_before, counts_after)),
                    ('access_scoping_private_entry_hidden', await self._private_entry_problems(bob, entries)),
                    ('access_scoping_public_entry_read_only', await self._public_entry_problems(bob, entries)),
                    ('access_scoping_hidden_matches_absent', await self._hidden_matches_absent_problems(bob, entries)),
                    ('access_scoping_store_never_merges_foreign', await self._store_problems(bob, entries, originals)),
                    ('access_scoping_sync_manages_own_entries', await self._sync_problems(bob)),
                    ('access_scoping_grants_round_trip', await self._grants_round_trip_problems(bob)),
                ]
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

        for aspect, problems in aspects:
            self.test_results.append((aspect, not problems, '; '.join(problems) or f'{SECOND_PRINCIPAL} scoped as expected'))
        return all(not problems for _, problems in aspects)

    async def _outcome(self, client: Client[Any], tool: str, arguments: dict[str, Any]) -> ToolOutcome:
        """Call a tool without raising on a tool error.

        Args:
            client: The client to call through.
            tool: The tool name.
            arguments: The tool arguments.

        Returns:
            The extracted content and None, or None and the error text.
        """
        result = await client.call_tool(tool, arguments, raise_on_error=False)
        if result.is_error:
            return None, ' '.join(block.text for block in result.content if isinstance(block, TextContent))
        return self._extract_content(result), None

    async def _content(self, client: Client[Any], tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool that must succeed.

        Args:
            client: The client to call through.
            tool: The tool name.
            arguments: The tool arguments.

        Returns:
            The extracted content.

        Raises:
            AssertionError: If the tool reported an error.
        """
        data, error = await self._outcome(client, tool, arguments)
        if data is None:
            raise AssertionError(f'{tool} failed: {error}')
        return data

    async def _store_scoping_entries(self) -> ScopingEntries:
        """Store the primary principal's private and public entries.

        Returns:
            The stored entries.
        """
        assert self.client is not None
        private_thread = f'{self.test_thread_id}_scoping_private'
        public_thread = f'{self.test_thread_id}_scoping_public'
        private_text = f'# Scoping private\n\nPrivate entry of the primary principal carrying {PRIVATE_TOKEN}.'
        public_text = f'# Scoping public\n\nPublic entry of the primary principal carrying {PUBLIC_TOKEN}.'
        private = await self._content(self.client, 'store_context', {
            'thread_id': private_thread, 'source': 'agent', 'text': private_text, 'visibility': 'private',
        })
        public = await self._content(self.client, 'store_context', {
            'thread_id': public_thread, 'source': 'agent', 'text': public_text, 'visibility': 'public',
        })
        return ScopingEntries(
            private_id=str(private['context_id']),
            private_thread=private_thread,
            private_text=private_text,
            public_id=str(public['context_id']),
            public_thread=public_thread,
            public_text=public_text,
        )

    async def _counts(self, client: Client[Any]) -> tuple[int, int]:
        """Read the entry and thread totals a principal is shown.

        Args:
            client: The client of the principal.

        Returns:
            ``(total_entries, total_threads)`` from ``get_statistics``.
        """
        stats = await self._content(client, 'get_statistics', {})
        return int(stats['total_entries']), int(stats['total_threads'])

    async def _entries_by_id(self, client: Client[Any], context_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch entries through ``get_context_by_ids``.

        Args:
            client: The client of the reading principal.
            context_ids: The ids to fetch.

        Returns:
            The returned entries keyed by id.
        """
        data = await self._content(client, 'get_context_by_ids', {'context_ids': context_ids})
        return {str(row['id']): row for row in data.get('results', [])}

    async def _thread_ids(self, client: Client[Any]) -> set[str]:
        """List the threads a principal is shown.

        Args:
            client: The client of the principal.

        Returns:
            The listed thread ids.
        """
        data = await self._content(client, 'list_threads', {})
        return {str(thread['thread_id']) for thread in data.get('threads', [])}

    async def _matched_ids(self, client: Client[Any], tool: str, arguments: dict[str, Any]) -> set[str]:
        """Run a search or grep tool and collect the ids it returned.

        Args:
            client: The client of the searching principal.
            tool: A search tool or ``grep_context``.
            arguments: The tool arguments.

        Returns:
            The returned entry ids.
        """
        data = await self._content(client, tool, arguments)
        key = 'context_id' if tool == 'grep_context' else 'id'
        return {str(row[key]) for row in data.get('results', [])}

    @staticmethod
    def _count_problems(before: PrincipalTotals, after: PrincipalTotals) -> list[str]:
        """Compare the totals each principal is shown before and after the primary's two stores.

        The primary owns both new entries in two new threads; the second principal can
        read only the public one, so its totals move by exactly one entry and one thread.

        Args:
            before: The totals of the primary and of the second principal.
            after: The same totals after the stores.

        Returns:
            The problems found.
        """
        problems: list[str] = []
        for who, old, new, expected in (
            ('primary', before[0], after[0], (2, 2)),
            (SECOND_PRINCIPAL, before[1], after[1], (1, 1)),
        ):
            if (new[0] - old[0], new[1] - old[1]) != expected:
                problems.append(f'{who} totals (entries, threads) moved from {old} to {new}, expected a change of {expected}')
        return problems

    async def _private_entry_problems(self, bob: Client[Any], entries: ScopingEntries) -> list[str]:
        """Check that no tool lets the second principal reach the private entry.

        Every read is paired with the primary's own call, so a check never passes
        only because the query matches nothing at all.

        Args:
            bob: The second principal's client.
            entries: The primary's entries.

        Returns:
            The problems found.
        """
        assert self.client is not None
        private_id, thread = entries.private_id, entries.private_thread
        problems: list[str] = []

        for client, expected, who in ((self.client, {private_id}, 'primary'), (bob, set(), SECOND_PRINCIPAL)):
            fetched = set(await self._entries_by_id(client, [private_id]))
            if fetched != expected:
                problems.append(f'get_context_by_ids for {who} returned {sorted(fetched)}')
            browsed = await self._matched_ids(client, 'search_context', {'thread_id': thread})
            if browsed != expected:
                problems.append(f'search_context for {who} returned {sorted(browsed)}')
            grepped = await self._matched_ids(client, 'grep_context', {'pattern': PRIVATE_TOKEN, 'thread_id': thread})
            if grepped != expected:
                problems.append(f'grep_context for {who} returned {sorted(grepped)}')
            for tool in RANKED_SEARCH_TOOLS:
                if tool not in self.registered_tools:
                    continue
                query = entries.private_text if tool == 'semantic_search_context' else PRIVATE_TOKEN
                ranked = await self._matched_ids(client, tool, {'query': query, 'thread_id': thread})
                if ranked != expected:
                    problems.append(f'{tool} for {who} returned {sorted(ranked)}')
            listed = thread in await self._thread_ids(client)
            if listed != bool(expected):
                problems.append(f'list_threads for {who} listed the private thread: {listed}')

        own = (await self._entries_by_id(self.client, [private_id])).get(private_id, {})
        problems.extend(read_back_problems([('get_context_by_ids', own)], 'private', is_owner=True))

        not_found = f'Context entry not found: {private_id}'
        for tool, arguments in (
            ('navigate_context', {'context_id': private_id}),
            ('read_context_range', {'context_id': private_id, 'start_line': 1, 'end_line': 1}),
        ):
            _, error = await self._outcome(bob, tool, arguments)
            if error != not_found:
                problems.append(f'{tool} for {SECOND_PRINCIPAL} returned {error!r}, expected {not_found!r}')

        refused: list[ToolCall] = [
            ('update_context', {'context_id': private_id, 'text': 'Overwritten.'}),
            *grant_calls(private_id, THIRD_PRINCIPAL),
        ]
        for tool, arguments in refused:
            _, error = await self._outcome(bob, tool, arguments)
            if error != f'Context entry with ID {private_id} not found':
                problems.append(f'{tool} returned {error!r}')
        _, error = await self._outcome(bob, 'update_context_batch', {
            'updates': [{'context_id': private_id, 'text': 'Overwritten.'}], 'atomic': True,
        })
        if error is None or f'Context entry {private_id} not found at index 0' not in error:
            problems.append(f'update_context_batch returned {error!r}')

        deletes: tuple[ToolCall, ...] = (
            ('delete_context', {'context_ids': [private_id]}),
            ('delete_context', {'thread_id': thread}),
            ('delete_context_batch', {'context_ids': [private_id]}),
            ('delete_context_batch', {'thread_ids': [thread]}),
        )
        for tool, arguments in deletes:
            data, error = await self._outcome(bob, tool, arguments)
            if data is None or data.get('deleted_count') != 0:
                problems.append(f'{tool}({arguments}) returned {data or error!r}, expected deleted_count 0')

        surviving = await self._entries_by_id(self.client, [private_id])
        if surviving.get(private_id, {}).get('text_content') != entries.private_text:
            problems.append('the private entry did not survive unchanged')
        return problems

    async def _public_entry_problems(self, bob: Client[Any], entries: ScopingEntries) -> list[str]:
        """Check that the second principal reads the public entry but cannot modify or delete it.

        Args:
            bob: The second principal's client.
            entries: The primary's entries.

        Returns:
            The problems found.
        """
        assert self.client is not None
        public_id, thread = entries.public_id, entries.public_thread
        problems: list[str] = []

        fetched = await self._entries_by_id(bob, [public_id])
        if fetched.get(public_id, {}).get('text_content') != entries.public_text:
            problems.append(f'get_context_by_ids returned {sorted(fetched)} for the public entry')
        problems.extend(read_back_problems([('get_context_by_ids', fetched.get(public_id, {}))], 'public', is_owner=False))
        problems.extend(await self._grant_refusal_problems(bob, public_id, THIRD_PRINCIPAL))
        reads: list[ToolCall] = [
            ('search_context', {'thread_id': thread}),
            ('grep_context', {'pattern': PUBLIC_TOKEN, 'thread_id': thread}),
        ]
        reads.extend(
            (tool, {'query': PUBLIC_TOKEN, 'thread_id': thread})
            for tool in ('fts_search_context', 'hybrid_search_context') if tool in self.registered_tools
        )
        for tool, arguments in reads:
            matched = await self._matched_ids(bob, tool, arguments)
            if matched != {public_id}:
                problems.append(f'{tool} returned {sorted(matched)} for the public entry')
        if thread not in await self._thread_ids(bob):
            problems.append('list_threads does not list the public thread')
        for tool, arguments in (
            ('navigate_context', {'context_id': public_id}),
            ('read_context_range', {'context_id': public_id, 'start_line': 1, 'end_line': 1}),
        ):
            data, error = await self._outcome(bob, tool, arguments)
            if data is None or data.get('context_id') != public_id:
                problems.append(f'{tool} on the public entry returned {data or error!r}')

        not_modifiable = f'Not authorized to modify context entry with ID {public_id}'
        for arguments in ({'text': 'Overwritten.'}, {'visibility': 'private'}):
            _, error = await self._outcome(bob, 'update_context', {'context_id': public_id, **arguments})
            if error != not_modifiable:
                problems.append(f'update_context({arguments}) returned {error!r}, expected {not_modifiable!r}')
        _, error = await self._outcome(bob, 'update_context_batch', {
            'updates': [{'context_id': public_id, 'text': 'Overwritten.'}], 'atomic': True,
        })
        if error is None or f'Not authorized to modify context entry {public_id} at index 0' not in error:
            problems.append(f'update_context_batch returned {error!r}')

        not_deletable = f'Not authorized to delete context entries: {public_id}'
        for tool in ('delete_context', 'delete_context_batch'):
            _, error = await self._outcome(bob, tool, {'context_ids': [public_id]})
            if error != not_deletable:
                problems.append(f'{tool} by id returned {error!r}, expected {not_deletable!r}')
        thread_deletes: tuple[ToolCall, ...] = (
            ('delete_context', {'thread_id': thread}),
            ('delete_context_batch', {'thread_ids': [thread]}),
        )
        for tool, arguments in thread_deletes:
            data, error = await self._outcome(bob, tool, arguments)
            if data is None or data.get('deleted_count') != 0:
                problems.append(f'{tool} by thread returned {data or error!r}, expected deleted_count 0')

        surviving = await self._entries_by_id(self.client, [public_id])
        if surviving.get(public_id, {}).get('text_content') != entries.public_text:
            problems.append('the public entry did not survive unchanged')
        return problems

    async def _hidden_matches_absent_problems(self, bob: Client[Any], entries: ScopingEntries) -> list[str]:
        """Check that a hidden entry answers exactly like an absent one.

        The same calls are made with the hidden entry's id (or prefix) and with an
        absent id (or prefix); the outcomes must be identical once the id itself is
        masked out of the messages.

        Args:
            bob: The second principal's client.
            entries: The primary's entries.

        Returns:
            The problems found.
        """
        hidden_prefix = entries.private_id[:PREFIX_LENGTH]
        templates: list[ToolCall] = [
            ('get_context_by_ids', {'context_ids': ['{id}']}),
            ('update_context', {'context_id': '{id}', 'text': 'Overwritten.'}),
            ('navigate_context', {'context_id': '{id}'}),
            ('read_context_range', {'context_id': '{id}', 'start_line': 1, 'end_line': 1}),
            ('delete_context', {'context_ids': ['{id}']}),
            ('delete_context_batch', {'context_ids': ['{id}']}),
            *grant_calls('{id}', THIRD_PRINCIPAL),
        ]
        calls: list[tuple[str, str, str, dict[str, Any]]] = []
        for hidden, absent in ((entries.private_id, ABSENT_ID), (hidden_prefix, ABSENT_PREFIX)):
            calls.extend((hidden, absent, tool, arguments) for tool, arguments in templates)
        problems: list[str] = []
        for hidden, absent, tool, template in calls:
            hidden_outcome = self._masked(await self._outcome(bob, tool, self._filled(template, hidden)), hidden)
            absent_outcome = self._masked(await self._outcome(bob, tool, self._filled(template, absent)), absent)
            if hidden_outcome != absent_outcome:
                problems.append(f'{tool} with {hidden!r} gave {hidden_outcome!r}, with an absent id {absent_outcome!r}')
        return problems

    @staticmethod
    def _filled(template: dict[str, Any], value: str) -> dict[str, Any]:
        """Substitute an id or prefix for the ``{id}`` placeholder of argument templates.

        Args:
            template: Tool arguments whose strings may hold ``{id}``.
            value: The id or prefix.

        Returns:
            The concrete arguments.
        """
        def fill(item: object) -> object:
            if isinstance(item, str):
                return item.replace('{id}', value)
            if isinstance(item, list):
                return [fill(member) for member in item]
            return item

        return {key: fill(item) for key, item in template.items()}

    @staticmethod
    def _masked(outcome: ToolOutcome, value: str) -> tuple[str, str]:
        """Render an outcome with the id or prefix masked, for comparison.

        Args:
            outcome: The tool outcome.
            value: The id or prefix the call named.

        Returns:
            ``('ok', content)`` or ``('error', message)`` with ``value`` masked.
        """
        data, error = outcome
        if error is not None:
            return 'error', error.replace(value, '<id>')
        return 'ok', repr(data).replace(value, '<id>')

    async def _store_problems(
        self, bob: Client[Any], entries: ScopingEntries, originals: dict[str, dict[str, Any]],
    ) -> list[str]:
        """Check that storing the primary's text never merges into the primary's entry.

        The private entry is hidden from the second principal, so its store lands as
        a new entry; the public entry is readable but foreign, so it is never merged
        into either. Neither of the primary's entries changes.

        Args:
            bob: The second principal's client.
            entries: The primary's entries.
            originals: The primary's entries as stored, keyed by id.

        Returns:
            The problems found.
        """
        assert self.client is not None
        problems: list[str] = []
        batch = await self._content(bob, 'store_context_batch', {
            'entries': [{'thread_id': entries.private_thread, 'source': 'agent', 'text': entries.private_text}],
        })
        batch_ids = {str(item.get('context_id')) for item in batch.get('results', []) if item.get('success')}
        if len(batch_ids) != 1 or entries.private_id in batch_ids:
            problems.append(f'store_context_batch of the private text returned {batch}')
        for target_id, thread, text in (
            (entries.private_id, entries.private_thread, entries.private_text),
            (entries.public_id, entries.public_thread, entries.public_text),
        ):
            stored = await self._content(bob, 'store_context', {'thread_id': thread, 'source': 'agent', 'text': text})
            if stored.get('context_id') in {None, target_id}:
                problems.append(f'store_context merged into {target_id}: {stored}')

        current = await self._entries_by_id(self.client, [entries.private_id, entries.public_id])
        for context_id, original in originals.items():
            row = current.get(context_id, {})
            if (row.get('text_content'), row.get('updated_at')) != (original.get('text_content'), original.get('updated_at')):
                problems.append(f'the entry {context_id} changed: {original} became {row}')
        if await self._entries_by_id(bob, [entries.private_id]):
            problems.append(f'the private entry became readable to {SECOND_PRINCIPAL}')
        return problems

    async def _sync_problems(self, bob: Client[Any]) -> list[str]:
        """Check that a corpus sync by the second principal manages only his own entries.

        The primary stores one private and one public managed entry of the same source,
        each alone in its own corpus thread. The second principal's plan of the private
        corpus equals his plan of a corpus no entry exists in: the source is planned as
        a create, and both listings are empty. His plan of the public corpus plans a
        create too and counts the primary's entry as foreign, never as an update or a
        delete, even with an empty manifest. His ``deduplicate: false`` store of the
        source inserts his own copy, which his next plan finds unchanged, while the
        primary's entry stays as it was and the primary, which cannot read the copy,
        still plans only its own entry.

        Args:
            bob: The second principal's client.

        Returns:
            The problems found.
        """
        if not SYNC_TOOLS.issubset(self.registered_tools):
            return []
        assert self.client is not None
        base = f'kb:{self.test_thread_id}_scoping_sync'
        private_thread, public_thread, absent_thread = f'{base}_private', f'{base}_public', f'{base}_absent'
        digest = meta_digest('scoping')
        text = 'Managed entry of the primary principal.'
        manifest = [manifest_item('p.md', text, digest)]
        stored = await self._content(self.client, 'store_context_batch', {'entries': [
            managed_entry(private_thread, 'p.md', text, digest, visibility='private'),
            managed_entry(public_thread, 'p.md', text, digest, visibility='public'),
        ], 'atomic': False})
        public_id = next(str(item['context_id']) for item in stored['results'] if item['index'] == 1)
        problems: list[str] = []

        plans: list[tuple[str, Client[Any], str, list[dict[str, Any]], dict[str, Any], dict[str, Any]]] = [
            ('primary', self.client, private_thread, manifest, {}, plan_shape(unchanged=1)),
            ('primary', self.client, public_thread, manifest, {}, plan_shape(unchanged=1)),
            (SECOND_PRINCIPAL, bob, private_thread, manifest, {}, plan_shape(creates=[0])),
            (SECOND_PRINCIPAL, bob, absent_thread, manifest, {}, plan_shape(creates=[0])),
            (SECOND_PRINCIPAL, bob, public_thread, manifest, {}, plan_shape(creates=[0], foreign=1)),
            (SECOND_PRINCIPAL, bob, public_thread, [], {'allow_empty_manifest': True}, plan_shape(foreign=1)),
        ]
        for who, client, thread, items, arguments, expected in plans:
            plan = await plan_with(client, thread, items, **arguments)
            if plan != expected:
                problems.append(f'{who} planned {thread} with {len(items)} items as {plan}, expected {expected}')
        for who, client, thread, expected_sources in (
            ('primary', self.client, private_thread, ['p.md']),
            (SECOND_PRINCIPAL, bob, private_thread, []),
            (SECOND_PRINCIPAL, bob, absent_thread, []),
            (SECOND_PRINCIPAL, bob, public_thread, []),
        ):
            listed = await listed_source_ids(client, thread)
            if listed != expected_sources:
                problems.append(f'{who} listed {listed} in {thread}, expected {expected_sources}')

        original = (await self._entries_by_id(self.client, [public_id]))[public_id]
        copy = await self._content(bob, 'store_context_batch', {
            'entries': [managed_entry(public_thread, 'p.md', text, digest)], 'atomic': False,
        })
        if copy.get('succeeded') != 1 or copy['results'][0].get('context_id') == public_id:
            problems.append(f'{SECOND_PRINCIPAL} store without deduplication returned {copy}')
        for who, client, expected in (
            (SECOND_PRINCIPAL, bob, plan_shape(unchanged=1, foreign=1)),
            ('primary', self.client, plan_shape(unchanged=1)),
        ):
            plan = await plan_with(client, public_thread, manifest)
            if plan != expected:
                problems.append(f'{who} planned the public corpus as {plan} after the copy, expected {expected}')
        current = (await self._entries_by_id(self.client, [public_id])).get(public_id, {})
        fields = ('text_content', 'metadata', 'updated_at')
        if tuple(current.get(field) for field in fields) != tuple(original.get(field) for field in fields):
            problems.append(f'the public managed entry changed: {original} became {current}')
        return problems

    async def _grant_refusal_problems(self, bob: Client[Any], context_id: str, principal_id: str) -> list[str]:
        """Check that every sharing tool refuses the second principal an entry it reads but does not own.

        Args:
            bob: The second principal's client.
            context_id: The entry's id.
            principal_id: The user the attempted share and unshare name.

        Returns:
            The problems found.
        """
        problems: list[str] = []
        for tool, arguments in grant_calls(context_id, principal_id):
            expected = NOT_OWNER_REFUSALS[tool].format(id=context_id)
            _, error = await self._outcome(bob, tool, arguments)
            if error != expected:
                problems.append(f'{tool} by {SECOND_PRINCIPAL} returned {error!r}, expected {expected!r}')
        return problems

    async def _grants_round_trip_problems(self, bob: Client[Any]) -> list[str]:
        """Share a private entry of the primary with the second principal, upgrade the grant and revoke it.

        While ``bob`` holds read, he reads the entry as private and not owned and finds
        it in a search, while an update and every sharing tool are refused; his own
        grant included, he cannot unshare it. A write grant beside it lets him edit the
        text but not change the visibility. The primary lists both grants as granted by
        itself; removing read leaves him the write grant, which still lets him read,
        and removing write hides the entry from him again.

        Args:
            bob: The second principal's client.

        Returns:
            The problems found.
        """
        assert self.client is not None
        thread = f'{self.test_thread_id}_scoping_shared'
        text = f'# Scoping shared\n\nPrivate entry of the primary principal shared with {SECOND_PRINCIPAL}: {SHARED_TOKEN}.'
        stored = await self._content(self.client, 'store_context', {
            'thread_id': thread, 'source': 'agent', 'text': text, 'visibility': 'private',
        })
        shared_id = str(stored['context_id'])
        grant = {'context_id': shared_id, 'principal_type': 'user', 'principal_id': SECOND_PRINCIPAL}
        problems: list[str] = []

        await self._content(self.client, 'share_context', {**grant, 'permission': 'read'})
        entry = (await self._entries_by_id(bob, [shared_id])).get(shared_id, {})
        if entry.get('text_content') != text:
            problems.append(f'{SECOND_PRINCIPAL} cannot read the entry shared with him: {entry}')
        problems.extend(read_back_problems([('get_context_by_ids', entry)], 'private', is_owner=False))
        if await self._matched_ids(bob, 'search_context', {'thread_id': thread}) != {shared_id}:
            problems.append(f'search_context by {SECOND_PRINCIPAL} does not find the shared entry')
        _, error = await self._outcome(bob, 'update_context', {'context_id': shared_id, 'text': 'Overwritten.'})
        if error != f'Not authorized to modify context entry with ID {shared_id}':
            problems.append(f'update_context under a read grant returned {error!r}')
        problems.extend(await self._grant_refusal_problems(bob, shared_id, SECOND_PRINCIPAL))

        await self._content(self.client, 'share_context', {**grant, 'permission': 'write'})
        edited = f'{text}\n\nEdited by {SECOND_PRINCIPAL} through his write grant.'
        _, error = await self._outcome(bob, 'update_context', {'context_id': shared_id, 'text': edited})
        current = (await self._entries_by_id(self.client, [shared_id])).get(shared_id, {})
        if error is not None or current.get('text_content') != edited:
            problems.append(f'the edit under a write grant returned {error!r} and left {current.get("text_content")!r}')
        _, error = await self._outcome(bob, 'update_context', {'context_id': shared_id, 'visibility': 'public'})
        if error != f'Only the owner may change the visibility of context {shared_id}':
            problems.append(f'a visibility change under a write grant returned {error!r}')

        listed = await self._content(self.client, 'list_context_grants', {'context_id': shared_id})
        grantor = primary_principal(self.server_env)
        expected_grants = [('user', SECOND_PRINCIPAL, permission, grantor) for permission in ('read', 'write')]
        if grant_rows(listed.get('grants', [])) != expected_grants:
            problems.append(f'the primary lists {listed.get("grants")}, expected {expected_grants}')
        for permission, remaining, readable in (('read', ['write'], True), ('write', [], False)):
            removed = await self._content(self.client, 'unshare_context', {**grant, 'permission': permission})
            if (removed.get('removed'), removed.get('remaining_permissions')) != (True, remaining):
                problems.append(f'unshare of {permission} returned {removed}, expected the remaining {remaining}')
            if (shared_id in await self._entries_by_id(bob, [shared_id])) != readable:
                problems.append(f'after the unshare of {permission} {SECOND_PRINCIPAL} reads the entry: {not readable}')
        _, error = await self._outcome(bob, 'navigate_context', {'context_id': shared_id})
        if error != f'Context entry not found: {shared_id}':
            problems.append(f'navigate_context after both unshares returned {error!r}')
        return problems
