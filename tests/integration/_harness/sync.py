"""Real-server checks for corpus sync: ``plan_context_sync`` and ``list_ingest_state``.

A corpus lives in a ``kb:`` thread whose managed entries carry the reserved
``metadata.ingest`` block naming that thread as their scope. The checks store such
entries the way a sync client does, through ``store_context_batch`` with
``deduplicate: false`` per entry, and drive the two read-only tools against them:
registration, a plan that converges once applied, plans confined to their source-id
ranges, the paged listing, the refusal of invalid manifests, a manifest at the item
cap, and the refusal of a malformed block on every write tool. Unmanaged entries
and blocks naming another scope sit in the same threads and are never planned or
listed.

The module-level helpers build blocks, manifest items and store entries for the
other vehicles that exercise the sync tools: the second-principal checks in
``access_scoping.py`` and the JWT scenario in ``tests/integration/_http_jwt.py``.
"""

from operator import itemgetter
from typing import Any

from fastmcp import Client

from app.digests import canonical_json_digest
from app.digests import content_hash
from app.digests import normalize_stored_text
from app.ingest import MAX_MANIFEST_ITEMS
from app.ingest import MAX_SOURCE_ID_LENGTH
from tests.integration._harness.core import HarnessCore

SYNC_TOOLS = frozenset({'plan_context_sync', 'list_ingest_state'})
SYNCED_AT = '2026-10-18T12:00:00Z'
# A four-byte UTF-8 code point: a source id of MAX_SOURCE_ID_LENGTH of them is the
# largest one a manifest item or a stored block admits.
ASTRAL_FILL = '\U0001f600'
INGEST_SHAPE_ERROR = 'an object with exactly the keys meta_digest, scope, source_id, synced_at'
SOURCES = ('a.md', 'b.md', 'c.md', 'd.md')


def meta_digest(label: str) -> str:
    """Return a metadata digest computed the way the sync client computes one.

    Args:
        label: A value that makes the digest differ from the digests of other labels.

    Returns:
        64 lowercase hexadecimal characters.
    """
    return canonical_json_digest({'metadata': {'label': label}, 'tags': [], 'visibility': 'private'})


def ingest_block(scope: str, source_id: str, digest: str) -> dict[str, str]:
    """Return a valid ingest block.

    Args:
        scope: The corpus thread the block names.
        source_id: The source document's id.
        digest: The metadata digest.

    Returns:
        The block.
    """
    return {'scope': scope, 'source_id': source_id, 'meta_digest': digest, 'synced_at': SYNCED_AT}


def manifest_item(source_id: str, text: str, digest: str) -> dict[str, str]:
    """Return the manifest item of a source document whose text is ``text``.

    Args:
        source_id: The source document's id.
        text: The text a sync would store for it.
        digest: Its metadata digest.

    Returns:
        The item, its text digest the server's content hash of the stored text.
    """
    return {'source_id': source_id, 'text_digest': content_hash(normalize_stored_text(text)), 'meta_digest': digest}


def keep_item(source_id: str) -> dict[str, Any]:
    """Return the keep item of a source document the client could not read.

    Args:
        source_id: The source document's id.

    Returns:
        The item.
    """
    return {'source_id': source_id, 'keep': True}


def managed_entry(
    thread_id: str, source_id: str, text: str, digest: str, *, visibility: str | None = None,
) -> dict[str, Any]:
    """Return a ``store_context_batch`` entry that stores a managed entry without deduplication.

    Args:
        thread_id: The corpus thread, which the block names as its scope.
        source_id: The source document's id.
        text: The text.
        digest: The metadata digest.
        visibility: An explicit visibility, or None for the server default.

    Returns:
        The batch entry.
    """
    entry: dict[str, Any] = {
        'thread_id': thread_id,
        'source': 'user',
        'text': text,
        'metadata': {'ingest': ingest_block(thread_id, source_id, digest)},
        'deduplicate': False,
    }
    if visibility is not None:
        entry['visibility'] = visibility
    return entry


def plan_shape(
    *,
    creates: list[int] | None = None,
    updates: list[dict[str, Any]] | None = None,
    deletes: list[dict[str, str]] | None = None,
    unchanged: int = 0,
    kept: int = 0,
    foreign: int = 0,
) -> dict[str, Any]:
    """Return the expected plan of a call whose deletes all fit in one response.

    Args:
        creates: The manifest indices to create.
        updates: The planned updates.
        deletes: The planned deletes.
        unchanged: The expected ``unchanged_count``.
        kept: The expected ``kept_count``.
        foreign: The expected ``foreign_count``.

    Returns:
        The plan without its ``thread_id``.
    """
    return {
        'creates': creates or [],
        'updates': updates or [],
        'deletes': deletes or [],
        'deletes_total': len(deletes or []),
        'unchanged_count': unchanged,
        'kept_count': kept,
        'foreign_count': foreign,
    }


def plan_body(plan: dict[str, Any]) -> dict[str, Any]:
    """Return a plan without its ``thread_id``, for comparison with :func:`plan_shape`.

    Args:
        plan: A ``plan_context_sync`` result.

    Returns:
        The plan's other fields.
    """
    return {key: value for key, value in plan.items() if key != 'thread_id'}


def maximal_source_id(number: int) -> str:
    """Return a distinct source id of the largest size a manifest item admits.

    Args:
        number: Distinguishes the id from the ids of other numbers below 0xEFFFF.

    Returns:
        ``MAX_SOURCE_ID_LENGTH`` four-byte code points.
    """
    return chr(0x10000 + number) + ASTRAL_FILL * (MAX_SOURCE_ID_LENGTH - 1)


class SyncMixin(HarnessCore):
    """Checks for the corpus sync tools on a real server."""

    async def _sync_content(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call a tool through the primary client that must succeed.

        Args:
            tool: The tool name.
            arguments: The tool arguments.

        Returns:
            The extracted content.

        Raises:
            AssertionError: If the tool reported an error.
        """
        data, error = await self._outcome(tool, arguments)
        if data is None:
            raise AssertionError(f'{tool} failed: {error}')
        return data

    async def _store_managed(self, entries: list[dict[str, Any]]) -> list[str]:
        """Store entries in one non-atomic batch and return their ids in order.

        Args:
            entries: The batch entries.

        Returns:
            The stored ids, one per entry.

        Raises:
            AssertionError: If any entry was not stored.
        """
        data = await self._sync_content('store_context_batch', {'entries': entries, 'atomic': False})
        if data.get('succeeded') != len(entries):
            raise AssertionError(f'store_context_batch stored {data.get("succeeded")} of {len(entries)}: {data}')
        return [str(item['context_id']) for item in sorted(data['results'], key=itemgetter('index'))]

    async def _store_decoys(self, thread_id: str, source_id: str, text: str, digest: str) -> None:
        """Store an unmanaged entry and one whose block names another scope in a corpus thread.

        Neither is a member of the thread: the first carries no block, and the second
        carries the same source id and digests as a member but names another thread.

        Args:
            thread_id: The corpus thread.
            source_id: A member's source id the second decoy repeats.
            text: That member's text.
            digest: That member's metadata digest.
        """
        await self._store_managed([
            {'thread_id': thread_id, 'source': 'user', 'text': 'Unmanaged note beside the corpus.', 'deduplicate': False},
            {
                'thread_id': thread_id,
                'source': 'user',
                'text': text,
                'metadata': {'ingest': ingest_block(f'{thread_id}-elsewhere', source_id, digest)},
                'deduplicate': False,
            },
        ])

    async def _plan(self, thread_id: str, manifest: list[dict[str, Any]], **arguments: Any) -> dict[str, Any]:
        """Plan a corpus thread against a manifest.

        Args:
            thread_id: The corpus thread.
            manifest: The manifest items.
            arguments: Further tool arguments such as the range bounds.

        Returns:
            The plan.

        Raises:
            AssertionError: If the plan names another thread.
        """
        plan = await self._sync_content('plan_context_sync', {'thread_id': thread_id, 'manifest': manifest, **arguments})
        if plan.get('thread_id') != thread_id:
            raise AssertionError(f'plan_context_sync answered for {plan.get("thread_id")!r} instead of {thread_id!r}')
        return plan

    def _sync_skipped(self, test_name: str) -> bool:
        """Record a skip when the server did not register the sync tools.

        Args:
            test_name: The check's name.

        Returns:
            bool: True when the check is skipped.
        """
        if SYNC_TOOLS.issubset(self.registered_tools):
            return False
        self.test_results.append((test_name, True, 'Skipped (corpus sync tools not registered)'))
        return True

    async def test_sync_tools_registered(self) -> bool:
        """Verify the server registers every annotated tool, the two sync tools included.

        Not gated on ``registered_tools``: the other sync checks are, and they would
        skip silently if registration regressed. Every tool of ``TOOL_ANNOTATIONS``
        registers by default except ``semantic_search_context``, which needs an
        embedding provider, so the expected set leaves it out exactly when the server
        reports semantic search unavailable.

        Returns:
            bool: True if test passed.
        """
        test_name = 'sync_tools_registered'
        assert self.client is not None
        try:
            from app.tools import TOOL_ANNOTATIONS

            registered = {tool.name for tool in await self.client.list_tools()}
            stats = await self._sync_content('get_statistics', {})
            semantic_available = bool(stats.get('semantic_search', {}).get('available', False))
            expected = set(TOOL_ANNOTATIONS) - (set() if semantic_available else {'semantic_search_context'})
            problems: list[str] = []
            if not SYNC_TOOLS.issubset(registered):
                problems.append(f'the sync tools {sorted(SYNC_TOOLS - registered)} are not registered')
            if registered != expected:
                problems.append(
                    f'registered tools differ from the annotated set: missing {sorted(expected - registered)}, '
                    f'unexpected {sorted(registered - expected)}',
                )
            return self._record(test_name, problems, f'{len(registered)} tools registered, both sync tools included')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_plan_context_sync_converges(self) -> bool:
        """Plan, apply and re-plan a corpus until nothing changes.

        An empty thread plans every item as a create; once stored, every item is
        unchanged. A changed text plans a text update, a changed metadata digest a
        metadata update, a dropped source an absent delete, and a keep item nothing.
        Applying the plan through the batch tools brings the next plan back to
        unchanged. A seeded duplicate is planned as a duplicate delete, and two plans
        over the duplicated state name the same survivor. The thread's unmanaged entry
        and its block naming another scope are never planned.

        Returns:
            bool: True if test passed.
        """
        test_name = 'plan_context_sync_converges'
        if self._sync_skipped(test_name):
            return True
        try:
            thread = f'kb:{self.test_thread_id}_sync_converges'
            texts = {source: f'Corpus document {source} of the converging sync.' for source in SOURCES}
            first, second = meta_digest('first'), meta_digest('second')
            manifest = [manifest_item(source, text, first) for source, text in texts.items()]
            problems: list[str] = []

            plan = await self._plan(thread, manifest)
            if plan_body(plan) != plan_shape(creates=[0, 1, 2, 3]):
                problems.append(f'empty thread planned {plan}')

            ids = dict(zip(texts, await self._store_managed(
                [managed_entry(thread, source, text, first) for source, text in texts.items()],
            ), strict=True))
            await self._store_decoys(thread, 'a.md', texts['a.md'], first)
            plan = await self._plan(thread, manifest)
            if plan_body(plan) != plan_shape(unchanged=4):
                problems.append(f'stored corpus planned {plan}')

            revised = 'Corpus document a.md, revised by the converging sync.'
            changed: list[dict[str, Any]] = [
                manifest_item('a.md', revised, first), manifest_item('b.md', texts['b.md'], second), keep_item('d.md'),
            ]
            plan = await self._plan(thread, changed)
            expected = plan_shape(
                updates=[
                    {'index': 0, 'context_id': ids['a.md'], 'change': 'text'},
                    {'index': 1, 'context_id': ids['b.md'], 'change': 'metadata'},
                ],
                deletes=[{'context_id': ids['c.md'], 'source_id': 'c.md', 'reason': 'absent'}],
                kept=1,
            )
            if plan_body(plan) != expected:
                problems.append(f'changed manifest planned {plan}, expected {expected}')

            await self._sync_content('update_context_batch', {'updates': [
                {'context_id': ids['a.md'], 'text': revised, 'metadata': {'ingest': ingest_block(thread, 'a.md', first)}},
                {'context_id': ids['b.md'], 'metadata': {'ingest': ingest_block(thread, 'b.md', second)}},
            ], 'atomic': True})
            await self._sync_content('delete_context_batch', {'context_ids': [ids['c.md']]})
            converged = [
                manifest_item('a.md', revised, first),
                manifest_item('b.md', texts['b.md'], second),
                manifest_item('d.md', texts['d.md'], first),
            ]
            plan = await self._plan(thread, converged)
            if plan_body(plan) != plan_shape(unchanged=3):
                problems.append(f'applied plan left {plan}')

            duplicate = (await self._store_managed([managed_entry(thread, 'a.md', revised, first)]))[0]
            plans = [await self._plan(thread, converged) for _ in range(2)]
            # Both copies match the item exactly, so the newest id survives.
            expected = plan_shape(
                deletes=[{'context_id': min(ids['a.md'], duplicate), 'source_id': 'a.md', 'reason': 'duplicate'}],
                unchanged=3,
            )
            if plans[0] != plans[1]:
                problems.append(f'two plans over the same duplicates differ: {plans}')
            if plan_body(plans[0]) != expected:
                problems.append(f'duplicate planned {plans[0]}, expected {expected}')
            await self._sync_content('delete_context_batch', {'context_ids': [min(ids['a.md'], duplicate)]})
            plan = await self._plan(thread, converged)
            if plan_body(plan) != plan_shape(unchanged=3):
                problems.append(f'removed duplicate left {plan}')

            return self._record(test_name, problems, 'plan, apply and re-plan converged; one survivor per source')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_plan_context_sync_ranges(self) -> bool:
        """Verify each slice of a manifest plans deletes only inside its own range.

        Four sources are stored; a first slice ``(open, r/b]`` names only ``r/a`` and a
        second slice ``(r/b, open)`` names only ``r/d``. Each plans the delete of the
        one stored source of its own range that it does not name, and nothing outside
        it. Planning only reads, so all four remain.

        Returns:
            bool: True if test passed.
        """
        test_name = 'plan_context_sync_ranges'
        if self._sync_skipped(test_name):
            return True
        try:
            thread = f'kb:{self.test_thread_id}_sync_ranges'
            digest = meta_digest('ranges')
            texts = {source: f'Ranged document {source}.' for source in ('r/a', 'r/b', 'r/c', 'r/d')}
            ids = dict(zip(texts, await self._store_managed(
                [managed_entry(thread, source, text, digest) for source, text in texts.items()],
            ), strict=True))
            problems: list[str] = []
            for manifest_source, bounds, deleted in (
                ('r/a', {'source_id_through': 'r/b'}, 'r/b'),
                ('r/d', {'source_id_after': 'r/b'}, 'r/c'),
            ):
                plan = await self._plan(thread, [manifest_item(manifest_source, texts[manifest_source], digest)], **bounds)
                expected = plan_shape(
                    deletes=[{'context_id': ids[deleted], 'source_id': deleted, 'reason': 'absent'}], unchanged=1,
                )
                if plan_body(plan) != expected:
                    problems.append(f'slice {bounds} planned {plan}, expected {expected}')
            plan = await self._plan(thread, [manifest_item(source, text, digest) for source, text in texts.items()])
            if plan_body(plan) != plan_shape(unchanged=4):
                problems.append(f'the whole corpus planned {plan} after the slices')
            return self._record(test_name, problems, 'each slice planned deletes inside its own range only')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_list_ingest_state_paged(self) -> bool:
        """Verify pages of the ingest state cover the managed entries once, in source-id order.

        Four managed entries are stored out of order beside an unmanaged entry and a
        block naming another scope. Pages of two list the four in ``(source_id,
        context_id)`` order with their digests and sync time; the second page fills
        exactly, so its cursor is null.

        Returns:
            bool: True if test passed.
        """
        test_name = 'list_ingest_state_paged'
        if self._sync_skipped(test_name):
            return True
        try:
            thread = f'kb:{self.test_thread_id}_sync_list'
            digest = meta_digest('listing')
            texts = {source: f'Listed document {source}.' for source in ('l/c', 'l/a', 'l/d', 'l/b')}
            ids = dict(zip(texts, await self._store_managed(
                [managed_entry(thread, source, text, digest) for source, text in texts.items()],
            ), strict=True))
            await self._store_decoys(thread, 'l/a', texts['l/a'], digest)

            problems: list[str] = []
            first = await self._sync_content('list_ingest_state', {'thread_id': thread, 'limit': 2})
            if first.get('next_cursor') is None:
                problems.append(f'the first page has no cursor: {first}')
                pages = [first]
            else:
                second = await self._sync_content(
                    'list_ingest_state', {'thread_id': thread, 'limit': 2, 'cursor': first['next_cursor']},
                )
                if second.get('next_cursor') is not None:
                    problems.append(f'the exactly filled last page has a cursor: {second}')
                pages = [first, second]
            listed = [entry for page in pages for entry in page['entries']]
            expected = [
                {
                    'context_id': ids[source],
                    'source_id': source,
                    'text_digest': content_hash(texts[source]),
                    'meta_digest': digest,
                    'synced_at': SYNCED_AT,
                }
                for source in sorted(texts)
            ]
            if listed != expected:
                problems.append(f'pages listed {listed}, expected {expected}')
            problems.extend(
                f'page header disagrees with its entries: {page}'
                for page in pages
                if page.get('count') != len(page['entries']) or page.get('thread_id') != thread
            )
            return self._record(test_name, problems, 'two pages listed the four managed entries in order')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_plan_context_sync_rejects_invalid_manifest(self) -> bool:
        """Verify an invalid manifest is refused with its reason.

        A repeated source id, a malformed digest, a keep item carrying digests, one
        item past the cap, an item outside the planned range and an empty manifest
        without the opt-in are each refused.

        Returns:
            bool: True if test passed.
        """
        test_name = 'plan_context_sync_rejects_invalid_manifest'
        if self._sync_skipped(test_name):
            return True
        try:
            thread = f'kb:{self.test_thread_id}_sync_invalid'
            digest = meta_digest('invalid')
            item = manifest_item('x.md', 'Invalid manifest document.', digest)
            cases: list[tuple[dict[str, Any], str]] = [
                ({'manifest': [item, item]}, "manifest repeats source_id values (each may appear once): 'x.md'"),
                ({'manifest': [{**item, 'text_digest': 'A' * 64}]}, 'String should match pattern'),
                ({'manifest': [{**keep_item('x.md'), 'meta_digest': digest}]},
                 'a keep item carries no text_digest or meta_digest'),
                ({'manifest': [keep_item(f'{index:04d}') for index in range(MAX_MANIFEST_ITEMS + 1)]},
                 f'List should have at most {MAX_MANIFEST_ITEMS} items'),
                ({'manifest': [item], 'source_id_after': 'x.md'},
                 "manifest item 0 has source_id 'x.md', outside the planned range (source_id_after, source_id_through]"),
                ({'manifest': []},
                 ('manifest is empty: planning an empty manifest deletes every managed entry in the range; '
                  'pass allow_empty_manifest=true to confirm')),
            ]
            problems: list[str] = []
            for arguments, message in cases:
                data, error = await self._outcome('plan_context_sync', {'thread_id': thread, **arguments})
                if error is None or message not in error:
                    problems.append(f'{message!r} not reported: {data or error!r}')
            return self._record(test_name, problems, f'{len(cases)} invalid manifests refused with their reasons')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_plan_context_sync_manifest_at_cap(self) -> bool:
        """Verify a manifest of the most items with the largest source ids is planned.

        Each of the items carries a source id of ``MAX_SOURCE_ID_LENGTH`` four-byte code
        points; the thread is unknown, so every item is planned as a create.

        Returns:
            bool: True if test passed.
        """
        test_name = 'plan_context_sync_manifest_at_cap'
        if self._sync_skipped(test_name):
            return True
        try:
            thread = f'kb:{self.test_thread_id}_sync_cap'
            digest = meta_digest('cap')
            manifest = [
                manifest_item(maximal_source_id(index), f'Document {index}.', digest) for index in range(MAX_MANIFEST_ITEMS)
            ]
            plan = await self._plan(thread, manifest)
            problems: list[str] = []
            if plan_body(plan) != plan_shape(creates=list(range(MAX_MANIFEST_ITEMS))):
                problems.append(f'the manifest at the cap planned {plan_body(plan) | {"creates": len(plan["creates"])}}')
            return self._record(test_name, problems, f'{MAX_MANIFEST_ITEMS} maximal items planned as creates')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_ingest_block_rejected(self) -> bool:
        """Verify every write tool refuses a malformed ingest block with the same reason.

        ``store_context`` and ``update_context`` refuse the call, the batch tools record
        the reason on the entry; nothing is stored and the entry an update names keeps
        its block.

        Returns:
            bool: True if test passed.
        """
        test_name = 'ingest_block_rejected'
        try:
            thread = f'kb:{self.test_thread_id}_sync_rejected'
            digest = meta_digest('rejected')
            text = 'Document whose block the write tools guard.'
            target = (await self._store_managed([managed_entry(thread, 'g.md', text, digest)]))[0]
            malformed = {'ingest': {**ingest_block(thread, 'g.md', digest), 'meta_digest': 'A' * 64}}
            message = 'metadata.ingest.meta_digest must be 64 lowercase hexadecimal characters'
            problems: list[str] = []

            refused_calls: list[tuple[str, dict[str, Any], str]] = [
                ('store_context', {'thread_id': thread, 'source': 'user', 'text': 'Refused.', 'metadata': malformed}, message),
                ('update_context', {'context_id': target, 'metadata': malformed}, message),
                ('update_context', {'context_id': target, 'metadata_patch': {'ingest': {'scope': thread}}},
                 f'metadata_patch.ingest must be null or {INGEST_SHAPE_ERROR}'),
            ]
            for tool, arguments, expected in refused_calls:
                data, error = await self._outcome(tool, arguments)
                if error != expected:
                    problems.append(f'{tool}({arguments}) returned {data or error!r}, expected {expected!r}')
            batch_calls: list[tuple[str, dict[str, Any]]] = [
                ('store_context_batch', {'entries': [
                    {'thread_id': thread, 'source': 'user', 'text': 'Refused.', 'metadata': malformed},
                ], 'atomic': False}),
                ('update_context_batch', {'updates': [{'context_id': target, 'metadata': malformed}], 'atomic': False}),
            ]
            for tool, arguments in batch_calls:
                data = await self._sync_content(tool, arguments)
                errors = [item.get('error') for item in data.get('results', [])]
                if data.get('succeeded') != 0 or errors != [message]:
                    problems.append(f'{tool} returned {data}, expected the entry error {message!r}')

            stored = await self._sync_content('search_context', {'thread_id': thread, 'limit': 10})
            if [row['id'] for row in stored.get('results', [])] != [target]:
                problems.append(f'the thread holds {stored.get("results")}, expected only {target}')
            rows = (await self._sync_content('get_context_by_ids', {'context_ids': [target]})).get('results', [])
            if not rows or rows[0].get('metadata') != {'ingest': ingest_block(thread, 'g.md', digest)}:
                problems.append(f'the guarded entry changed: {rows}')
            return self._record(test_name, problems, 'every write tool refused the malformed block')
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False


async def plan_with(client: Client[Any], thread_id: str, manifest: list[dict[str, Any]], **arguments: Any) -> dict[str, Any]:
    """Plan a corpus thread through any client and return the plan without its thread.

    Args:
        client: The planning principal's client.
        thread_id: The corpus thread.
        manifest: The manifest items.
        arguments: Further tool arguments.

    Returns:
        The plan without its ``thread_id``.

    Raises:
        AssertionError: If the tool refused the call.
    """
    result = await client.call_tool(
        'plan_context_sync', {'thread_id': thread_id, 'manifest': manifest, **arguments}, raise_on_error=False,
    )
    if result.is_error or result.structured_content is None:
        raise AssertionError(f'plan_context_sync failed: {result.content}')
    return plan_body(result.structured_content)


async def listed_source_ids(client: Client[Any], thread_id: str) -> list[str]:
    """Return the source ids of the first ingest-state page a principal is shown.

    Args:
        client: The listing principal's client.
        thread_id: The corpus thread.

    Returns:
        The listed source ids in order.

    Raises:
        AssertionError: If the tool refused the call.
    """
    result = await client.call_tool('list_ingest_state', {'thread_id': thread_id}, raise_on_error=False)
    if result.is_error or result.structured_content is None:
        raise AssertionError(f'list_ingest_state failed: {result.content}')
    return [str(entry['source_id']) for entry in result.structured_content['entries']]
