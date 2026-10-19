"""Query-plan pins for the SQLite reads of ingest-managed entries.

Both reads select the source id, the block, the content hash and the access columns of
the members of one thread. The ingest index carries every one of them, so SQLite answers
the reads from index pages and never walks a row's text overflow pages to reach
``metadata``: the range read searches the index on the thread and the source-id bounds,
and the owner page reads it in ``(source_id, id)`` order, so neither sorts.

The pins run on a database that the server's startup preparation (``prepare_database``)
builds, so every index a server has competes with the ingest index, and that holds
members of a corpus thread beside rows of other threads. The database carries no
``sqlite_stat1`` table, so the plans are the ones a database gets before any ``ANALYZE``
has run. Each read runs through the repository as a request principal with and without
groups, and its statement is planned with the parameters it bound.
"""

import re
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Awaitable
from collections.abc import Callable
from dataclasses import dataclass
from typing import cast

import pytest
import pytest_asyncio

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.base import BackendAccess
from app.ingest_sql import INGEST_INDEX_NAME
from app.repositories import RepositoryContainer
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_layouts import sqlite_scoped_db

CORPUS = 'kb:plans'
MEMBERS = 200
FILLER_ROWS = 200
# A plan step that reads context_entries under the alias both reads use.
_ENTRY_STEP = re.compile(r'^(?:SCAN|SEARCH) ce\b')
# A search of the ingest index that starts at the corpus thread, never a scan of every managed row.
_INGEST_INDEX_STEP = re.compile(rf'^SEARCH ce USING (?:COVERING )?INDEX {INGEST_INDEX_NAME} \(thread_id=\?')
_SORT_STEP = 'USE TEMP B-TREE FOR ORDER BY'

_SCOPES = [
    pytest.param(AccessScope('alice', frozenset()), id='principal-only'),
    pytest.param(AccessScope('alice', frozenset({'team-x'})), id='principal-with-groups'),
]

# Members and filler rows: alice owns the even members, bob the odd ones; every fifth is public.
_SEED_SQL = (
    'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) '
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id, visibility) '
    "SELECT lower(hex(randomblob(16))), ?, 'agent', 'text', 'Document ' || i, "
    "json_object('ingest', json_object('scope', ?, 'source_id', printf('doc-%04d', i), "
    "'meta_digest', ?, 'synced_at', '2026-10-18T12:00:00Z')), "
    "CASE WHEN i % 2 = 0 THEN 'alice' ELSE 'bob' END, "
    "CASE WHEN i % 5 = 0 THEN 'public' ELSE 'private' END FROM n"
)
_FILLER_SQL = (
    'WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < ?) '
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id, visibility) '
    "SELECT lower(hex(randomblob(16))), 'plan-filler', 'agent', 'text', 'Filler ' || i, "
    "json_object('label', 'filler'), 'alice', 'private' FROM n"
)


@dataclass(frozen=True, slots=True)
class Captured:
    """One statement a repository sent, with the parameters it bound."""

    sql: str
    params: tuple[object, ...]


class _RecordingConnection:
    """A connection that records every statement before running it on the real connection."""

    def __init__(self, conn: sqlite3.Connection, statements: list[Captured]) -> None:
        self._conn = conn
        self._statements = statements

    def execute(self, sql: str, params: tuple[object, ...] = ()) -> sqlite3.Cursor:
        self._statements.append(Captured(sql, tuple(params)))
        return self._conn.execute(sql, params)


class _CapturingBackend:
    """A backend whose reads record the statements the repositories send."""

    def __init__(self, backend: StorageBackend) -> None:
        self.backend = backend
        self.statements: list[Captured] = []

    @property
    def backend_type(self) -> str:
        return self.backend.backend_type

    def readable_relation(self, table: str, *, alias: str) -> str:
        return self.backend.readable_relation(table, alias=alias)

    async def execute_read[T](self, operation: Callable[..., T], *, access_scope: BackendAccess) -> T:
        def _recorded(conn: sqlite3.Connection) -> T:
            return operation(_RecordingConnection(conn, self.statements))

        return await self.backend.execute_read(_recorded, access_scope=access_scope)


@pytest_asyncio.fixture
async def plan_db(async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[ScopedDb]:
    """A database prepared by server startup, holding the members of :data:`CORPUS` and rows of another thread."""
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        await db.execute(_SEED_SQL, (MEMBERS, CORPUS, CORPUS, '0' * 64))
        await db.execute(_FILLER_SQL, (FILLER_ROWS,))
        yield db


type Read = Callable[[RepositoryContainer, AccessScope], Awaitable[object]]


async def _range_unbounded(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after=None, source_id_through=None,
    )


async def _range_bounded(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after='doc-0050', source_id_through='doc-0100',
    )


async def _range_after(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after='doc-0150', source_id_through=None,
    )


async def _range_through(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after=None, source_id_through='doc-0050',
    )


async def _first_page(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.owned_ingest_page(scope=scope, thread_id=CORPUS, after=None, limit=21)


async def _next_page(repos: RepositoryContainer, scope: AccessScope) -> object:
    return await repos.context.owned_ingest_page(scope=scope, thread_id=CORPUS, after=('doc-0100', 'f' * 32), limit=21)


# Each read and the constraints its index search must carry beyond the thread equality.
READS: dict[str, tuple[Read, tuple[str, ...]]] = {
    'range-unbounded': (_range_unbounded, ()),
    'range-bounded': (_range_bounded, ('<expr>>?', '<expr><?')),
    'range-after': (_range_after, ('<expr>>?',)),
    'range-through': (_range_through, ('<expr><?',)),
    'first-page': (_first_page, ()),
    'next-page': (_next_page, ('<expr>>?',)),
}


async def _captured_statement(db: ScopedDb, read: Read, scope: AccessScope) -> Captured:
    """Run ``read`` as ``scope`` and return the one statement it sent."""
    capturing = _CapturingBackend(db.backend)
    rows = await read(RepositoryContainer(cast(StorageBackend, capturing)), scope)
    assert rows, 'the read returned no rows, so its plan would not be the plan of a read that finds members'
    assert len(capturing.statements) == 1, capturing.statements
    return capturing.statements[0]


async def _plan(backend: StorageBackend, statement: Captured) -> list[str]:
    """Return the detail column of every ``EXPLAIN QUERY PLAN`` row of ``statement``."""

    def _explain(conn: sqlite3.Connection) -> list[str]:
        return [str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {statement.sql}', statement.params).fetchall()]

    return await backend.execute_read(_explain, access_scope=MAINTENANCE)


@pytest.mark.parametrize('scope', _SCOPES)
@pytest.mark.parametrize('read', list(READS))
@pytest.mark.asyncio
async def test_ingest_reads_search_the_ingest_index_without_sorting(
    read: str, scope: AccessScope, plan_db: ScopedDb,
) -> None:
    run, constraints = READS[read]
    statement = await _captured_statement(plan_db, run, scope)

    plan = await _plan(plan_db.backend, statement)

    entry_steps = [step for step in plan if _ENTRY_STEP.match(step)]
    assert len(entry_steps) == 1, f'{statement.sql!r} reads context_entries more than once or never: {plan}'
    step = entry_steps[0]
    assert _INGEST_INDEX_STEP.match(step), f'{statement.sql!r} reads context_entries outside the ingest index: {plan}'
    for constraint in constraints:
        assert constraint in step, f'{statement.sql!r} does not search the index on {constraint}: {plan}'
    assert not any(_SORT_STEP in planned for planned in plan), f'{statement.sql!r} sorts: {plan}'
