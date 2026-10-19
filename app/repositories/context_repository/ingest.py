"""Reads of ingest-managed context entries: a source-id range and a page of the caller's own entries.

An entry is a member of thread T when its ``metadata.ingest`` block carries a source id and
names T as its scope; membership is decided here, at read time, and the block itself takes
no part in any access decision. Both reads name the relation through the read seam, apply
the access predicate after every other condition, and compare and order source ids through
the fragments of :mod:`app.ingest_sql`, from which the ingest index is rendered as well.
"""

import json
import sqlite3
from typing import TYPE_CHECKING
from typing import Protocol
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.ids import normalize_id
from app.ingest_sql import block_sql
from app.ingest_sql import scope_sql
from app.ingest_sql import source_id_key_sql
from app.ingest_sql import source_id_sql
from app.repositories.base import BaseRepository
from app.repositories.context_repository.records import IngestRow

if TYPE_CHECKING:
    import asyncpg


def _select_list(backend_type: str) -> str:
    """Return the columns both reads select: the id, the source id, the block and the content hash."""
    source_id = source_id_sql(backend_type, 'ce')
    block = block_sql(backend_type, 'ce')
    return f'ce.id, {source_id} AS source_id, {block} AS block, ce.content_hash'


def _membership_conditions(backend_type: str, thread_placeholder: str) -> list[str]:
    """Return the conditions that make a row a member of the thread bound at ``thread_placeholder``.

    The row sits in the thread, its block carries a source id (which also satisfies the
    ingest index's partial predicate), and the block names the row's own thread as its scope.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        thread_placeholder: The placeholder the thread id binds to.

    Returns:
        The conditions, to be joined with ``AND``.
    """
    source_id = source_id_sql(backend_type, 'ce')
    block_scope = scope_sql(backend_type, 'ce')
    return [f'ce.thread_id = {thread_placeholder}', f'{source_id} IS NOT NULL', f'{block_scope} = ce.thread_id']


class _SelectedRow(Protocol):
    """A selected row as either backend's driver returns it, read by column name."""

    def __getitem__(self, key: str, /) -> object: ...


def _ingest_row(row: _SelectedRow, *, context_id: str, is_owner: bool) -> IngestRow:
    """Build the record of one selected row; the block arrives as JSON text on both backends."""
    return IngestRow(
        context_id=context_id,
        source_id=str(row['source_id']),
        block=json.loads(cast(str, row['block'])),
        content_hash=cast(str | None, row['content_hash']),
        is_owner=is_owner,
    )


class ContextIngestMixin(BaseRepository):
    """Reads of the entries a corpus sync manages.

    Reads the members of a thread the caller may read inside a source-id range, each flagged
    with whether the caller owns it, and pages through the members the caller owns in
    ``(source_id, id)`` order.
    """

    async def ingest_rows_in_range(
        self,
        *,
        scope: Scope,
        thread_id: str,
        source_id_after: str | None,
        source_id_through: str | None,
    ) -> list[IngestRow]:
        """Read the members of a thread the scope may read whose source ids fall in a range.

        The range is ``(source_id_after, source_id_through]`` in code-point order; an
        omitted bound leaves that side open. The READ predicate is the last condition, so a
        member the scope may not read is never returned, and the OWNER predicate in the
        select list flags each returned member. The rows come back in no defined order.

        Args:
            scope: The caller's scope.
            thread_id: The thread whose members are read.
            source_id_after: Exclusive lower bound of the source ids, or None for no lower bound.
            source_id_through: Inclusive upper bound of the source ids, or None for no upper bound.

        Returns:
            One record per readable member in the range.
        """
        backend_type = self.backend.backend_type
        entries = self._readable('context_entries', alias='ce')
        # Placeholder order follows the text: the OWNER binds of the select list, the thread,
        # the bounds, then the READ binds of the last condition.
        owner = build_access_predicate(scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='ce', start=1)
        params: list[object] = [*owner.params, thread_id]
        conditions = _membership_conditions(backend_type, self._placeholder(len(params)))
        key = source_id_key_sql(backend_type, 'ce')
        if source_id_after is not None:
            params.append(source_id_after)
            conditions.append(f'{key} > {self._placeholder(len(params))}')
        if source_id_through is not None:
            params.append(source_id_through)
            conditions.append(f'{key} <= {self._placeholder(len(params))}')
        read = build_access_predicate(
            scope, mode=AccessMode.READ, backend_type=backend_type, outer='ce', start=len(params) + 1,
        )
        params.extend(read.params)
        where = ' AND '.join(conditions)
        query = (
            f'SELECT {_select_list(backend_type)}, {owner.admits_expression()} AS is_owner '
            f'FROM {entries} WHERE {where}{read.and_clause()}'
        )

        if backend_type == 'sqlite':

            def _read_sqlite(conn: sqlite3.Connection) -> list[IngestRow]:
                return [
                    _ingest_row(row, context_id=str(row['id']), is_owner=bool(row['is_owner']))
                    for row in conn.execute(query, tuple(params)).fetchall()
                ]

            return await self.backend.execute_read(_read_sqlite, access_scope=scope)

        # PostgreSQL
        async def _read_postgresql(conn: 'asyncpg.Connection') -> list[IngestRow]:
            return [
                _ingest_row(row, context_id=normalize_id(str(row['id'])), is_owner=bool(row['is_owner']))
                for row in await conn.fetch(query, *params)
            ]

        return await self.backend.execute_read(_read_postgresql, access_scope=scope)

    async def owned_ingest_page(
        self,
        *,
        scope: Scope,
        thread_id: str,
        after: tuple[str, str] | None,
        limit: int,
    ) -> list[IngestRow]:
        """Read one page of the members of a thread the scope owns, in ``(source_id, id)`` order.

        The page continues strictly after the ``(source_id, context_id)`` cursor ``after``;
        the redundant ``source_id >=`` bound gives SQLite a range constraint on the index.
        The OWNER predicate precedes ORDER BY and LIMIT, so a member of another principal,
        readable or hidden, never takes a page slot. Under the system scope the predicate is
        empty and every member is listed. A caller that needs to know whether another page
        follows asks for one row more than it shows.

        Args:
            scope: The caller's scope.
            thread_id: The thread whose members are listed.
            after: The source id and id of the last row of the previous page, or None for
                the first page.
            limit: The most rows to return.

        Returns:
            The page's records, each owned by the caller.
        """
        backend_type = self.backend.backend_type
        entries = self._readable('context_entries', alias='ce')
        params: list[object] = [thread_id]
        conditions = _membership_conditions(backend_type, self._placeholder(1))
        key = source_id_key_sql(backend_type, 'ce')
        if after is not None:
            after_source_id, after_id = after
            first = len(params) + 1
            params.extend([after_source_id, after_source_id, after_id])
            conditions.append(
                f'{key} >= {self._placeholder(first)} AND ({key}, ce.id) > '
                f'({self._placeholder(first + 1)}, {self._placeholder(first + 2)})',
            )
        owner = build_access_predicate(
            scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='ce', start=len(params) + 1,
        )
        params.extend(owner.params)
        params.append(limit)
        where = ' AND '.join(conditions)
        query = (
            f'SELECT {_select_list(backend_type)} FROM {entries} WHERE {where}{owner.and_clause()} '
            f'ORDER BY {key}, ce.id LIMIT {self._placeholder(len(params))}'
        )

        if backend_type == 'sqlite':

            def _page_sqlite(conn: sqlite3.Connection) -> list[IngestRow]:
                return [
                    _ingest_row(row, context_id=str(row['id']), is_owner=True)
                    for row in conn.execute(query, tuple(params)).fetchall()
                ]

            return await self.backend.execute_read(_page_sqlite, access_scope=scope)

        # PostgreSQL
        async def _page_postgresql(conn: 'asyncpg.Connection') -> list[IngestRow]:
            return [
                _ingest_row(row, context_id=normalize_id(str(row['id'])), is_owner=True)
                for row in await conn.fetch(query, *params)
            ]

        return await self.backend.execute_read(_page_postgresql, access_scope=scope)
