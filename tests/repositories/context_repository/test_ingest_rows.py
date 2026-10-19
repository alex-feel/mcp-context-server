"""Tests for the reads of ingest-managed entries: a source-id range and a page of the caller's own entries.

An entry is a member of thread T when its ``metadata.ingest`` block carries a source id
and names T as its scope. ``ingest_rows_in_range`` returns every member the caller may
read inside an exclusive-inclusive source-id range and flags the ones the caller owns;
``owned_ingest_page`` pages through the members the caller owns in ``(source_id, id)``
order. The SQLite tests run both reads on a real database; the PostgreSQL tests pin the
rendered statements and the decoding of their rows through a recording connection.
"""

import json
import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from typing import cast
from unittest.mock import Mock

import pytest

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from app.backends.base import base_relation
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from app.repositories.context_repository.records import IngestRow
from tests.helpers import LOCAL_SCOPE

THREAD = 'kb:handbook'
OTHER_THREAD = 'kb:other'
BOB = AccessScope('bob', frozenset())
DIGEST = 'a' * 64
SYNCED_AT = '2026-10-18T12:00:00Z'


def _block(source_id: str, *, scope: str = THREAD) -> dict[str, str]:
    """Return a valid ingest block for ``source_id`` in ``scope``."""
    return {'scope': scope, 'source_id': source_id, 'meta_digest': DIGEST, 'synced_at': SYNCED_AT}


async def _store(
    repos: RepositoryContainer,
    text: str,
    *,
    metadata: Mapping[str, object] | None,
    scope: AccessScope = LOCAL_SCOPE,
    visibility: str = 'private',
    thread_id: str = THREAD,
) -> str:
    """Store one entry owned by ``scope`` and return its id."""
    context_id, merged = await repos.context.store_with_deduplication(
        thread_id=thread_id,
        source='user',
        content_type='text',
        text_content=text,
        scope=scope,
        visibility=visibility,
        metadata=json.dumps(metadata) if metadata is not None else None,
    )
    assert not merged
    return context_id


async def _member(
    repos: RepositoryContainer,
    source_id: str,
    *,
    scope: AccessScope = LOCAL_SCOPE,
    visibility: str = 'private',
) -> str:
    """Store a member of :data:`THREAD` whose block carries ``source_id``."""
    return await _store(
        repos, f'Document {source_id} of {scope.principal_id}', metadata={'ingest': _block(source_id)},
        scope=scope, visibility=visibility,
    )


async def _stored_hash(backend: StorageBackend, context_id: str) -> str | None:
    """Read the stored content hash of an entry through raw SQL."""

    def _read(conn: sqlite3.Connection) -> str | None:
        row = conn.execute('SELECT content_hash FROM context_entries WHERE id = ?', (context_id,)).fetchone()
        return cast(str | None, row[0])

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


async def _range_ids(
    repos: RepositoryContainer,
    *,
    scope: Scope = LOCAL_SCOPE,
    after: str | None = None,
    through: str | None = None,
    thread_id: str = THREAD,
) -> list[str]:
    """Return the source ids of the members in the range, sorted."""
    rows = await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=thread_id, source_id_after=after, source_id_through=through,
    )
    return sorted(row.source_id for row in rows)


async def _page(
    repos: RepositoryContainer,
    *,
    scope: Scope = LOCAL_SCOPE,
    after: tuple[str, str] | None = None,
    limit: int = 50,
) -> list[tuple[str, str]]:
    """Return ``(source_id, context_id)`` of one owned page, in the order the read returns it."""
    rows = await repos.context.owned_ingest_page(scope=scope, thread_id=THREAD, after=after, limit=limit)
    return [(row.source_id, row.context_id) for row in rows]


class TestIngestRowsInRange:
    """``ingest_rows_in_range`` reads the members the caller may read inside a source-id range."""

    @pytest.mark.asyncio
    async def test_returns_each_member_with_its_block_hash_and_ownership(
        self, repos: RepositoryContainer, context_test_db: StorageBackend,
    ) -> None:
        """A member comes back with its id, source id, parsed block, stored content hash and owner flag."""
        context_id = await _member(repos, 'guides/setup.md')

        rows = await repos.context.ingest_rows_in_range(
            scope=LOCAL_SCOPE, thread_id=THREAD, source_id_after=None, source_id_through=None,
        )

        stored_hash = await _stored_hash(context_test_db, context_id)
        assert stored_hash is not None
        assert rows == [
            IngestRow(
                context_id=context_id,
                source_id='guides/setup.md',
                block=_block('guides/setup.md'),
                content_hash=stored_hash,
                is_owner=True,
            ),
        ]

    @pytest.mark.asyncio
    async def test_membership_needs_a_source_id_and_the_thread_as_scope(self, repos: RepositoryContainer) -> None:
        """Rows without a block, with another scope, in another thread or without a source id are not members."""
        await _member(repos, 'member.md')
        await _store(repos, 'Unmanaged entry', metadata={'title': 'not managed'})
        await _store(repos, 'Other scope', metadata={'ingest': _block('other-scope.md', scope=OTHER_THREAD)})
        await _store(repos, 'Other thread', metadata={'ingest': _block('other-thread.md')}, thread_id=OTHER_THREAD)
        await _store(repos, 'No source id', metadata={'ingest': {'scope': THREAD, 'meta_digest': DIGEST}})
        await _store(repos, 'Scalar block', metadata={'ingest': 'guides/setup.md'})
        await _store(repos, 'No metadata', metadata=None)

        assert await _range_ids(repos) == ['member.md']
        assert await _range_ids(repos, thread_id=OTHER_THREAD) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('after', 'through', 'expected'),
        [
            pytest.param(None, None, ['a.md', 'b.md', 'c.md', 'd.md'], id='unbounded'),
            pytest.param('a.md', None, ['b.md', 'c.md', 'd.md'], id='after-is-exclusive'),
            pytest.param(None, 'c.md', ['a.md', 'b.md', 'c.md'], id='through-is-inclusive'),
            pytest.param('a.md', 'c.md', ['b.md', 'c.md'], id='both-bounds'),
            pytest.param('b.md', 'b.md', [], id='empty-range'),
        ],
    )
    async def test_range_is_exclusive_after_and_inclusive_through(
        self, repos: RepositoryContainer, after: str | None, through: str | None, expected: list[str],
    ) -> None:
        """``source_id_after`` excludes its own value and ``source_id_through`` includes its own."""
        for source_id in ('c.md', 'a.md', 'd.md', 'b.md'):
            await _member(repos, source_id)

        assert await _range_ids(repos, after=after, through=through) == expected

    @pytest.mark.asyncio
    async def test_bounds_compare_by_code_point(self, repos: RepositoryContainer) -> None:
        """Uppercase sorts before lowercase and a non-ASCII id after both: no case folding, no locale."""
        for source_id in ('B.md', 'a.md', 'z.md', 'é.md'):
            await _member(repos, source_id)

        assert await _range_ids(repos, after='B.md', through='z.md') == ['a.md', 'z.md']
        assert await _range_ids(repos, after='z.md') == ['é.md']
        assert await _range_ids(repos, through='B.md') == ['B.md']

    @pytest.mark.asyncio
    async def test_flags_ownership_and_hides_unreadable_members(self, repos: RepositoryContainer) -> None:
        """Another principal's public member reads as not owned; its private member is absent."""
        own = await _member(repos, 'own.md')
        foreign_public = await _member(repos, 'foreign-public.md', scope=BOB, visibility='public')
        await _member(repos, 'foreign-private.md', scope=BOB)

        rows = await repos.context.ingest_rows_in_range(
            scope=LOCAL_SCOPE, thread_id=THREAD, source_id_after=None, source_id_through=None,
        )

        assert sorted((row.context_id, row.is_owner) for row in rows) == sorted([(own, True), (foreign_public, False)])

    @pytest.mark.asyncio
    async def test_system_scope_reads_every_member_as_owned(self, repos: RepositoryContainer) -> None:
        """The system scope's predicate admits every row, so every member reads as owned."""
        await _member(repos, 'own.md')
        await _member(repos, 'foreign-private.md', scope=BOB)

        rows = await repos.context.ingest_rows_in_range(
            scope=SYSTEM_SCOPE, thread_id=THREAD, source_id_after=None, source_id_through=None,
        )

        assert sorted((row.source_id, row.is_owner) for row in rows) == [('foreign-private.md', True), ('own.md', True)]

    @pytest.mark.asyncio
    async def test_a_row_without_a_content_hash_reads_as_none(
        self, repos: RepositoryContainer, context_test_db: StorageBackend,
    ) -> None:
        """A row stored before content hashes existed reads with ``content_hash`` None."""
        context_id = generate_id()

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id) '
                "VALUES (?, ?, 'user', 'text', 'Legacy document', ?, ?)",
                (context_id, THREAD, json.dumps({'ingest': _block('legacy.md')}), LOCAL_SCOPE.principal_id),
            )

        await context_test_db.execute_write(_insert, access_scope=MAINTENANCE)

        rows = await repos.context.ingest_rows_in_range(
            scope=LOCAL_SCOPE, thread_id=THREAD, source_id_after=None, source_id_through=None,
        )

        assert [(row.context_id, row.content_hash) for row in rows] == [(context_id, None)]


class TestOwnedIngestPage:
    """``owned_ingest_page`` pages through the caller's own members in ``(source_id, id)`` order."""

    @pytest.mark.asyncio
    async def test_orders_by_source_id_then_id(self, repos: RepositoryContainer) -> None:
        """Members sort by source id, and two members of one source id by their ids."""
        c_id = await _member(repos, 'c.md')
        a_id = await _member(repos, 'a.md')
        b_first = await _member(repos, 'b.md')
        b_second = await _store(repos, 'Second copy of b.md', metadata={'ingest': _block('b.md')})

        assert await _page(repos) == [
            ('a.md', a_id), ('b.md', min(b_first, b_second)), ('b.md', max(b_first, b_second)), ('c.md', c_id),
        ]

    @pytest.mark.asyncio
    async def test_cursor_resumes_strictly_after_its_source_id_and_id(self, repos: RepositoryContainer) -> None:
        """The keyset continues after the cursor row, also between two members of one source id."""
        await _member(repos, 'a.md')
        b_first = await _member(repos, 'b.md')
        b_second = await _store(repos, 'Second copy of b.md', metadata={'ingest': _block('b.md')})
        c_id = await _member(repos, 'c.md')
        low, high = sorted((b_first, b_second))

        assert await _page(repos, after=('b.md', low)) == [('b.md', high), ('c.md', c_id)]
        assert await _page(repos, after=('b.md', high)) == [('c.md', c_id)]
        assert await _page(repos, after=('c.md', c_id)) == []

    @pytest.mark.asyncio
    async def test_limit_bounds_the_page(self, repos: RepositoryContainer) -> None:
        """At most ``limit`` rows come back, the first ones in key order."""
        for source_id in ('c.md', 'a.md', 'b.md'):
            await _member(repos, source_id)

        assert [source_id for source_id, _ in await _page(repos, limit=2)] == ['a.md', 'b.md']

    @pytest.mark.asyncio
    async def test_lists_only_the_callers_own_members(self, repos: RepositoryContainer) -> None:
        """A readable public member and a hidden member of another principal never take a page slot."""
        await _member(repos, 'a.md', scope=BOB, visibility='public')
        await _member(repos, 'b.md', scope=BOB)
        own = await _member(repos, 'c.md')

        assert await _page(repos, limit=1) == [('c.md', own)]
        assert [source_id for source_id, _ in await _page(repos, scope=BOB)] == ['a.md', 'b.md']

    @pytest.mark.asyncio
    async def test_every_record_reads_as_owned(self, repos: RepositoryContainer) -> None:
        """The page holds only owned rows, so every record carries ``is_owner`` True and its block."""
        context_id = await _member(repos, 'a.md')

        rows = await repos.context.owned_ingest_page(scope=LOCAL_SCOPE, thread_id=THREAD, after=None, limit=10)

        assert [(row.context_id, row.is_owner, row.block) for row in rows] == [(context_id, True, _block('a.md'))]

    @pytest.mark.asyncio
    async def test_membership_needs_a_source_id_and_the_thread_as_scope(self, repos: RepositoryContainer) -> None:
        """Owned rows without a block, with another scope or in another thread are never listed."""
        await _member(repos, 'member.md')
        await _store(repos, 'Unmanaged entry', metadata={'title': 'not managed'})
        await _store(repos, 'Other scope', metadata={'ingest': _block('other-scope.md', scope=OTHER_THREAD)})
        await _store(repos, 'Other thread', metadata={'ingest': _block('other-thread.md')}, thread_id=OTHER_THREAD)

        assert [source_id for source_id, _ in await _page(repos)] == ['member.md']

    @pytest.mark.asyncio
    async def test_system_scope_lists_every_member(self, repos: RepositoryContainer) -> None:
        """The system scope's owner predicate is empty, so it lists the members of every owner."""
        await _member(repos, 'a.md', scope=BOB)
        await _member(repos, 'b.md')

        assert [source_id for source_id, _ in await _page(repos, scope=SYSTEM_SCOPE)] == ['a.md', 'b.md']


# PostgreSQL: the rendered statements and the decoding of their rows.

type Statement = tuple[str, tuple[object, ...]]

_SOURCE_ID = "(ce.metadata #>> '{ingest,source_id}')"
_SCOPE = "(ce.metadata #>> '{ingest,scope}')"
_KEY = "((ce.metadata #>> '{ingest,source_id}') COLLATE \"C\")"
_SELECT = f"SELECT ce.id, {_SOURCE_ID} AS source_id, ce.metadata -> 'ingest' AS block, ce.content_hash"


def _membership(thread: str) -> str:
    """Return the membership conditions with the thread placeholder ``thread``."""
    return f'ce.thread_id = {thread} AND {_SOURCE_ID} IS NOT NULL AND {_SCOPE} = ce.thread_id'


def _recording_postgresql_repo(rows: list[Mapping[str, object]]) -> tuple[ContextRepository, list[Statement]]:
    """Build a PostgreSQL repository whose connection returns ``rows`` and records every statement.

    Args:
        rows: The rows every ``fetch`` returns.

    Returns:
        The repository and the list the recorded ``(statement, arguments)`` pairs are appended to.
    """
    statements: list[Statement] = []

    class _RecordingConnection:
        async def fetch(self, query: str, *args: object) -> list[Mapping[str, object]]:
            statements.append((query, args))
            return rows

    pg_backend = Mock()
    pg_backend.backend_type = 'postgresql'
    pg_backend.readable_relation = base_relation

    async def _execute_read(closure: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        _ = access_scope
        return await closure(_RecordingConnection())

    pg_backend.execute_read = _execute_read
    return ContextRepository(cast(StorageBackend, pg_backend)), statements


class TestPostgresqlStatements:
    """The PostgreSQL reads bound and order the source id under the C collation and end with the access predicate."""

    @pytest.mark.asyncio
    async def test_range_statement_with_both_bounds(self) -> None:
        """The owner flag binds first, then the thread and the bounds, then the read predicate after every condition."""
        repo, statements = _recording_postgresql_repo([])
        read = build_access_predicate(LOCAL_SCOPE, mode=AccessMode.READ, backend_type='postgresql', outer='ce', start=5)

        await repo.ingest_rows_in_range(scope=LOCAL_SCOPE, thread_id=THREAD, source_id_after='a.md', source_id_through='c.md')

        expected = (
            f'{_SELECT}, CASE WHEN ce.owner_id = $1 THEN 1 ELSE 0 END AS is_owner FROM context_entries ce '
            f'WHERE {_membership("$2")} AND {_KEY} > $3 AND {_KEY} <= $4 AND {read.sql}'
        )
        assert statements == [(expected, ('local', THREAD, 'a.md', 'c.md', *read.params))]

    @pytest.mark.asyncio
    async def test_range_statement_without_bounds(self) -> None:
        """Without bounds the read predicate's placeholders follow the thread directly."""
        repo, statements = _recording_postgresql_repo([])
        read = build_access_predicate(LOCAL_SCOPE, mode=AccessMode.READ, backend_type='postgresql', outer='ce', start=3)

        await repo.ingest_rows_in_range(scope=LOCAL_SCOPE, thread_id=THREAD, source_id_after=None, source_id_through=None)

        expected = (
            f'{_SELECT}, CASE WHEN ce.owner_id = $1 THEN 1 ELSE 0 END AS is_owner FROM context_entries ce '
            f'WHERE {_membership("$2")} AND {read.sql}'
        )
        assert statements == [(expected, ('local', THREAD, *read.params))]

    @pytest.mark.asyncio
    async def test_range_statement_for_the_system_scope(self) -> None:
        """The system scope binds no identity: the owner flag is the constant 1 and no predicate follows."""
        repo, statements = _recording_postgresql_repo([])

        await repo.ingest_rows_in_range(scope=SYSTEM_SCOPE, thread_id=THREAD, source_id_after=None, source_id_through='c.md')

        assert statements == [(
            f'{_SELECT}, 1 AS is_owner FROM context_entries ce WHERE {_membership("$1")} AND {_KEY} <= $2',
            (THREAD, 'c.md'),
        )]

    @pytest.mark.asyncio
    async def test_owned_page_statement_with_a_cursor(self) -> None:
        """The keyset bounds the key, then compares ``(key, id)``; the owner predicate precedes ORDER BY and LIMIT."""
        repo, statements = _recording_postgresql_repo([])
        cursor_id = generate_id()

        await repo.owned_ingest_page(scope=LOCAL_SCOPE, thread_id=THREAD, after=('b.md', cursor_id), limit=11)

        expected = (
            f'{_SELECT} FROM context_entries ce WHERE {_membership("$1")} AND {_KEY} >= $2 AND ({_KEY}, ce.id) > ($3, $4) '
            f'AND ce.owner_id = $5 ORDER BY {_KEY}, ce.id LIMIT $6'
        )
        assert statements == [(expected, (THREAD, 'b.md', 'b.md', cursor_id, 'local', 11))]

    @pytest.mark.asyncio
    async def test_owned_page_statement_without_a_cursor(self) -> None:
        """The first page carries the membership conditions and the owner predicate only."""
        repo, statements = _recording_postgresql_repo([])

        await repo.owned_ingest_page(scope=LOCAL_SCOPE, thread_id=THREAD, after=None, limit=3)

        expected = (
            f'{_SELECT} FROM context_entries ce WHERE {_membership("$1")} AND ce.owner_id = $2 '
            f'ORDER BY {_KEY}, ce.id LIMIT $3'
        )
        assert statements == [(expected, (THREAD, 'local', 3))]

    @pytest.mark.asyncio
    async def test_owned_page_statement_for_the_system_scope(self) -> None:
        """The system scope's owner predicate is empty, so the limit follows the membership conditions."""
        repo, statements = _recording_postgresql_repo([])

        await repo.owned_ingest_page(scope=SYSTEM_SCOPE, thread_id=THREAD, after=None, limit=3)

        assert statements == [(
            f'{_SELECT} FROM context_entries ce WHERE {_membership("$1")} ORDER BY {_KEY}, ce.id LIMIT $2',
            (THREAD, 3),
        )]

    @pytest.mark.asyncio
    async def test_rows_decode_the_block_text(self) -> None:
        """The jsonb block arrives as JSON text and is parsed; the owner flag becomes a boolean."""
        context_id = generate_id()
        block = _block('a.md')
        repo, _statements = _recording_postgresql_repo([
            {'id': context_id, 'source_id': 'a.md', 'block': json.dumps(block), 'content_hash': 'f' * 64, 'is_owner': 0},
        ])

        range_rows = await repo.ingest_rows_in_range(
            scope=LOCAL_SCOPE, thread_id=THREAD, source_id_after=None, source_id_through=None,
        )
        page_rows = await repo.owned_ingest_page(scope=LOCAL_SCOPE, thread_id=THREAD, after=None, limit=1)

        assert range_rows == [IngestRow(context_id, 'a.md', block, 'f' * 64, False)]
        assert page_rows == [IngestRow(context_id, 'a.md', block, 'f' * 64, True)]
