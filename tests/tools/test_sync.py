"""Tests for the corpus sync tools: ``plan_context_sync`` and ``list_ingest_state``.

Both tools run through an in-process FastMCP server that registers them the way the
server does and installs the JSON string middleware over them, against a SQLite
database that carries the ingest index, as every database the server's startup prepares
does, and whose managed entries are inserted through the repository. Argument errors
are checked at the boundary a client meets, the plan and the listing against the
stored rows, and the middleware's repair of a stringified manifest.
"""

import base64
import json
from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from fastmcp import Client
from fastmcp import FastMCP

import app.tools.sync
from app.access_scope import AccessScope
from app.auth import resolve_access_scope
from app.backends import StorageBackend
from app.digests import content_hash
from app.ids import generate_id
from app.migrations import apply_ingest_index_migration
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from app.startup import ensure_repositories
from app.startup.tool_registration import install_json_string_middleware
from app.tools import list_ingest_state
from app.tools import plan_context_sync
from app.tools import register_tool
from tests.helpers import LOCAL_SCOPE

THREAD = 'kb:handbook'
BOB = AccessScope('bob', frozenset())
META = 'b' * 64
OTHER_META = 'c' * 64
SYNCED_AT = '2026-10-18T12:00:00Z'


def _block(source_id: str, *, meta_digest: str = META, scope: str = THREAD) -> dict[str, str]:
    """Return an ingest block for ``source_id``."""
    return {'scope': scope, 'source_id': source_id, 'meta_digest': meta_digest, 'synced_at': SYNCED_AT}


def _item(source_id: str, text: str, *, meta_digest: str = META) -> dict[str, object]:
    """Return the manifest item of a source whose text is ``text``."""
    return {'source_id': source_id, 'text_digest': content_hash(text), 'meta_digest': meta_digest}


async def _insert(
    repos: RepositoryContainer,
    text: str,
    metadata: dict[str, object] | None,
    *,
    owner: AccessScope = LOCAL_SCOPE,
    visibility: str = 'private',
) -> str:
    """Insert one fresh entry of :data:`THREAD` owned by ``owner`` and return its id."""
    return await repos.context.insert_entry(
        thread_id=THREAD,
        source='user',
        content_type='text',
        text_content=text,
        scope=owner,
        visibility=visibility,
        metadata=json.dumps(metadata) if metadata is not None else None,
    )


async def _member(
    repos: RepositoryContainer,
    source_id: str,
    text: str,
    *,
    meta_digest: str = META,
    owner: AccessScope = LOCAL_SCOPE,
    visibility: str = 'private',
) -> str:
    """Insert a managed entry of :data:`THREAD` for ``source_id`` and return its id."""
    metadata: dict[str, object] = {'ingest': _block(source_id, meta_digest=meta_digest)}
    return await _insert(repos, text, metadata, owner=owner, visibility=visibility)


def _cursor(value: object) -> str:
    """Encode a value the way the listing encodes its cursor."""
    return _raw_cursor(json.dumps(value, separators=(',', ':')).encode('utf-8'))


def _raw_cursor(data: bytes) -> str:
    """Encode bytes as unpadded base64url."""
    return base64.urlsafe_b64encode(data).decode('ascii').rstrip('=')


def _cursor_with_stray_character() -> str:
    """Return a well-formed cursor with a character outside the base64url alphabet inserted.

    The cursor's length is a multiple of four, so a lenient decoder that drops the
    stray character finds a complete encoding and decodes the cursor anyway.

    Returns:
        The cursor with ``.`` inserted after its eighth character.

    Raises:
        AssertionError: Never; one of three consecutive payload lengths is a multiple of three.
    """
    context_id = generate_id()
    for source_id in ('a', 'ab', 'abc'):
        cursor = _cursor([source_id, context_id])
        if len(cursor) % 4 == 0:
            return f'{cursor[:8]}.{cursor[8:]}'
    raise AssertionError('one of three consecutive payload lengths is a multiple of three')


@pytest_asyncio.fixture
async def repos(nav_backend: StorageBackend) -> RepositoryContainer:
    """The repositories the tools resolve, over the fixture's SQLite database once it carries the ingest index."""
    await apply_ingest_index_migration(nav_backend)
    return await ensure_repositories()


@pytest_asyncio.fixture
async def client(repos: RepositoryContainer) -> AsyncGenerator[Client[Any], None]:
    """A client of an in-process server registering both sync tools behind the JSON string middleware."""
    _ = repos
    mcp: FastMCP[None] = FastMCP(name='sync-tools', mask_error_details=False, strict_input_validation=False)
    assert register_tool(mcp, plan_context_sync) is True
    assert register_tool(mcp, list_ingest_state) is True
    await install_json_string_middleware(mcp)
    async with Client(mcp) as connected:
        yield connected


async def _ok(client: Client[Any], name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Call a tool and return its structured result, failing on a tool error."""
    result = await client.call_tool(name, arguments, raise_on_error=False)
    assert result.is_error is False, result.content
    assert result.structured_content is not None
    return result.structured_content


async def _error(client: Client[Any], name: str, arguments: dict[str, Any]) -> str:
    """Call a tool that must fail and return its error text."""
    result = await client.call_tool(name, arguments, raise_on_error=False)
    assert result.is_error is True
    return ' '.join(getattr(block, 'text', '') for block in result.content)


async def _plan(client: Client[Any], manifest: object, **arguments: Any) -> dict[str, Any]:
    """Plan :data:`THREAD` against a manifest."""
    return await _ok(client, 'plan_context_sync', {'thread_id': THREAD, 'manifest': manifest, **arguments})


async def _list(client: Client[Any], **arguments: Any) -> dict[str, Any]:
    """List one page of :data:`THREAD`."""
    return await _ok(client, 'list_ingest_state', {'thread_id': THREAD, **arguments})


_VALID = _item('a', 'alpha')

_PLAN_ARGUMENT_ERRORS: list[tuple[dict[str, Any], str]] = [
    ({'manifest': [{'source_id': f'{index:04d}', 'keep': True} for index in range(1001)]},
     'List should have at most 1000 items'),
    ({'manifest': [{**_VALID, 'text_digest': 'A' * 64}]}, 'String should match pattern'),
    ({'manifest': [{'source_id': 'a', 'keep': True, 'text_digest': 'a' * 64}]},
     'a keep item carries no text_digest or meta_digest'),
    ({'manifest': [{'source_id': 'a', 'text_digest': 'a' * 64}]},
     'an item without keep requires both text_digest and meta_digest'),
    ({'manifest': [{'source_id': 'a', 'kep': True}]}, 'Extra inputs are not permitted'),
    ({'manifest': [{'source_id': 'x' * 257, 'keep': True}]}, 'String should have at most 256 characters'),
    ({'manifest': [_item('b', 'one'), _item('a', 'two'), _item('b', 'three'), _item('a', 'four')]},
     "manifest repeats source_id values (each may appear once): 'b', 'a'"),
    ({'manifest': [_item('m', 'one')], 'source_id_after': 'm'},
     "manifest item 0 has source_id 'm', outside the planned range (source_id_after, source_id_through]"),
    ({'manifest': [_VALID], 'source_id_after': 'b', 'source_id_through': 'b'},
     'source_id_after must sort before source_id_through'),
    ({'manifest': []},
     ('manifest is empty: planning an empty manifest deletes every managed entry in the range; '
      'pass allow_empty_manifest=true to confirm')),
    ({'manifest': [_VALID], 'thread_id': '   '}, 'thread_id cannot be empty or whitespace'),
    ({'manifest': [{'source_id': 'a\x00b', 'keep': True}]}, 'manifest_source_ids: A string value contains'),
    ({'manifest': [_VALID], 'source_id_through': 'z\x00'}, 'source_id_through: A string value contains'),
]


class TestPlanArguments:
    """A malformed request is refused with its reason before the stored rows are read."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(('arguments', 'message'), _PLAN_ARGUMENT_ERRORS)
    async def test_refused_before_any_query(
        self,
        client: Client[Any],
        monkeypatch: pytest.MonkeyPatch,
        arguments: dict[str, Any],
        message: str,
    ) -> None:
        read = AsyncMock(return_value=[])
        monkeypatch.setattr(ContextRepository, 'ingest_rows_in_range', read)

        text = await _error(client, 'plan_context_sync', {'thread_id': THREAD, **arguments})

        assert message in text
        read.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_empty_manifest_with_opt_in_deletes_only_managed_members(
        self, client: Client[Any], repos: RepositoryContainer,
    ) -> None:
        member = await _member(repos, 'a', 'alpha')
        await _insert(repos, 'an unmanaged note', {'topic': 'notes'})
        await _insert(repos, 'a member of another corpus', {'ingest': _block('b', scope='kb:other')})

        plan = await _plan(client, [], allow_empty_manifest=True)

        assert plan['deletes'] == [{'context_id': member, 'source_id': 'a', 'reason': 'absent'}]
        assert plan['deletes_total'] == 1


class TestPlanContextSync:
    """The plan compares the manifest with the caller's own members of the planned range."""

    @pytest.mark.asyncio
    async def test_unknown_thread_plans_every_ordinary_item_as_a_create(self, client: Client[Any]) -> None:
        manifest = [_item('a', 'alpha'), {'source_id': 'b', 'keep': True}, _item('c', 'charlie')]

        plan = await _plan(client, manifest)

        assert plan == {
            'thread_id': THREAD,
            'creates': [0, 2],
            'updates': [],
            'deletes': [],
            'deletes_total': 0,
            'unchanged_count': 0,
            'kept_count': 1,
            'foreign_count': 0,
        }

    @pytest.mark.asyncio
    async def test_compares_text_with_the_stored_content_hash(
        self, client: Client[Any], repos: RepositoryContainer,
    ) -> None:
        await _member(repos, 'a', 'alpha')
        bravo = await _member(repos, 'b', 'bravo')
        charlie = await _member(repos, 'c', 'charlie')
        delta = await _member(repos, 'd', 'delta')
        echo = await _member(repos, 'e', 'echo')
        manifest = [
            _item('a', 'alpha'),
            _item('b', 'bravo, revised'),
            _item('c', 'charlie', meta_digest=OTHER_META),
            _item('d', 'delta, revised', meta_digest=OTHER_META),
            _item('f', 'foxtrot'),
        ]

        plan = await _ok(client, 'plan_context_sync', {'thread_id': f'  {THREAD}  ', 'manifest': manifest})

        assert plan == {
            'thread_id': THREAD,
            'creates': [4],
            'updates': [
                {'index': 1, 'context_id': bravo, 'change': 'text'},
                {'index': 2, 'context_id': charlie, 'change': 'metadata'},
                {'index': 3, 'context_id': delta, 'change': 'both'},
            ],
            'deletes': [{'context_id': echo, 'source_id': 'e', 'reason': 'absent'}],
            'deletes_total': 1,
            'unchanged_count': 1,
            'kept_count': 0,
            'foreign_count': 0,
        }

    @pytest.mark.asyncio
    async def test_keeps_the_newest_exact_duplicate(self, client: Client[Any], repos: RepositoryContainer) -> None:
        older = await _member(repos, 'a', 'alpha')
        newer = await _member(repos, 'a', 'alpha')

        plan = await _plan(client, [_item('a', 'alpha')])

        assert plan['deletes'] == [{'context_id': older, 'source_id': 'a', 'reason': 'duplicate'}]
        assert plan['unchanged_count'] == 1
        assert newer > older

    @pytest.mark.asyncio
    async def test_counts_readable_foreign_members_and_never_plans_them(
        self, client: Client[Any], repos: RepositoryContainer,
    ) -> None:
        await _member(repos, 'a', 'alpha of bob', owner=BOB, visibility='public')
        await _member(repos, 'b', 'bravo of bob', owner=BOB)

        plan = await _plan(client, [_item('a', 'alpha')])

        assert plan['creates'] == [0]
        assert plan['deletes'] == []
        assert plan['foreign_count'] == 1

    @pytest.mark.asyncio
    async def test_lists_the_first_hundred_deletes_and_reports_the_total(
        self, client: Client[Any], repos: RepositoryContainer,
    ) -> None:
        ids = {source_id: await _member(repos, source_id, f'document {source_id}')
               for source_id in (f'{index:03d}' for index in range(101))}

        plan = await _plan(client, [], allow_empty_manifest=True)

        assert plan['deletes_total'] == 101
        assert plan['deletes'] == [
            {'context_id': ids[source_id], 'source_id': source_id, 'reason': 'absent'} for source_id in sorted(ids)[:100]
        ]

    @pytest.mark.asyncio
    async def test_plans_only_inside_the_range(self, client: Client[Any], repos: RepositoryContainer) -> None:
        await _member(repos, 'a', 'alpha')
        bravo = await _member(repos, 'b', 'bravo')
        await _member(repos, 'c', 'charlie')

        plan = await _plan(client, [], source_id_after='a', source_id_through='b', allow_empty_manifest=True)

        assert plan['deletes'] == [{'context_id': bravo, 'source_id': 'b', 'reason': 'absent'}]

    @pytest.mark.asyncio
    async def test_resolves_the_caller_once(self, client: Client[Any], monkeypatch: pytest.MonkeyPatch) -> None:
        resolver = MagicMock(wraps=resolve_access_scope)
        monkeypatch.setattr(app.tools.sync, 'resolve_access_scope', resolver)

        await _plan(client, [_item('a', 'alpha')])
        await _list(client)

        assert resolver.call_count == 2


class TestListIngestState:
    """The listing pages through the caller's own members in ``(source_id, context_id)`` order."""

    @pytest.mark.asyncio
    async def test_pages_cover_every_owned_member_once_in_order(
        self, client: Client[Any], repos: RepositoryContainer,
    ) -> None:
        ids = {source_id: await _member(repos, source_id, f'document {source_id}') for source_id in 'ecadb'}
        await _member(repos, 'f', 'bob', owner=BOB, visibility='public')

        listed: list[dict[str, Any]] = []
        cursor: str | None = None
        pages = 0
        while True:
            page = await _list(client, limit=2, **({'cursor': cursor} if cursor is not None else {}))
            pages += 1
            assert page['thread_id'] == THREAD
            assert page['count'] == len(page['entries'])
            listed.extend(page['entries'])
            cursor = page['next_cursor']
            if cursor is None:
                break

        assert pages == 3
        assert listed == [
            {
                'context_id': ids[source_id],
                'source_id': source_id,
                'text_digest': content_hash(f'document {source_id}'),
                'meta_digest': META,
                'synced_at': SYNCED_AT,
            }
            for source_id in 'abcde'
        ]

    @pytest.mark.asyncio
    async def test_cursor_encodes_the_last_row_of_the_page(
        self, client: Client[Any], repos: RepositoryContainer,
    ) -> None:
        await _member(repos, 'a', 'alpha')
        bravo = await _member(repos, 'b', 'bravo')
        await _member(repos, 'c', 'charlie')

        page = await _list(client, limit=2)

        cursor = page['next_cursor']
        assert isinstance(cursor, str)
        assert '=' not in cursor
        decoded = json.loads(base64.urlsafe_b64decode(cursor + '=' * (-len(cursor) % 4)).decode('utf-8'))
        assert decoded == ['b', bravo]

    @pytest.mark.asyncio
    async def test_exact_fit_returns_no_cursor(self, client: Client[Any], repos: RepositoryContainer) -> None:
        await _member(repos, 'a', 'alpha')
        await _member(repos, 'b', 'bravo')

        page = await _list(client, limit=2)

        assert page['count'] == 2
        assert page['next_cursor'] is None

    @pytest.mark.asyncio
    async def test_a_row_with_an_invalid_block_never_stalls_paging(
        self, client: Client[Any], repos: RepositoryContainer,
    ) -> None:
        await _insert(repos, 'a malformed member', {'ingest': _block('a', meta_digest='not a digest')})
        bravo = await _member(repos, 'b', 'bravo')

        first = await _list(client, limit=1)
        second = await _list(client, limit=1, cursor=first['next_cursor'])

        assert first['entries'] == []
        assert first['count'] == 0
        assert first['next_cursor'] is not None
        assert [entry['context_id'] for entry in second['entries']] == [bravo]
        assert second['next_cursor'] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'cursor',
        [
            'not*base64',
            _cursor_with_stray_character(),
            _raw_cursor(b'\xff\xfe'),
            _raw_cursor(b'not json'),
            _cursor({'source_id': 'a'}),
            _cursor('a'),
            _cursor(['a']),
            _cursor(['a', generate_id(), 'b']),
            _cursor(['a', 5]),
            _cursor(['', generate_id()]),
            _cursor(['x' * 257, generate_id()]),
            _cursor(['a', 'not an id']),
        ],
    )
    async def test_invalid_cursor_is_refused(self, client: Client[Any], cursor: str) -> None:
        text = await _error(client, 'list_ingest_state', {'thread_id': THREAD, 'cursor': cursor})

        assert text == 'invalid cursor'

    @pytest.mark.asyncio
    @pytest.mark.parametrize('limit', [0, 201])
    async def test_limit_outside_the_page_cap_is_refused(self, client: Client[Any], limit: int) -> None:
        text = await _error(client, 'list_ingest_state', {'thread_id': THREAD, 'limit': limit})

        assert 'limit' in text

    @pytest.mark.asyncio
    async def test_whitespace_thread_is_refused(self, client: Client[Any]) -> None:
        text = await _error(client, 'list_ingest_state', {'thread_id': '  '})

        assert text == 'thread_id cannot be empty or whitespace'

    @pytest.mark.asyncio
    async def test_unstorable_thread_is_refused_before_any_query(
        self, client: Client[Any], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        read = AsyncMock(return_value=[])
        monkeypatch.setattr(ContextRepository, 'owned_ingest_page', read)

        text = await _error(client, 'list_ingest_state', {'thread_id': 'kb:\x00'})

        assert text.startswith('thread_id: A string value contains')
        read.assert_not_awaited()


class TestStringifiedManifest:
    """The JSON string middleware repairs a stringified manifest but not stringified items."""

    @pytest.mark.asyncio
    async def test_stringified_manifest_is_repaired(self, client: Client[Any]) -> None:
        plan = await _plan(client, json.dumps([_item('a', 'alpha')]))

        assert plan['creates'] == [0]

    @pytest.mark.asyncio
    async def test_double_encoded_manifest_is_repaired(self, client: Client[Any]) -> None:
        plan = await _plan(client, json.dumps(json.dumps([_item('a', 'alpha')])))

        assert plan['creates'] == [0]

    @pytest.mark.asyncio
    async def test_stringified_items_are_refused(self, client: Client[Any]) -> None:
        text = await _error(
            client, 'plan_context_sync', {'thread_id': THREAD, 'manifest': [json.dumps(_item('a', 'alpha'))]},
        )

        assert 'Input should be a valid dictionary or instance of SyncManifestItem' in text
