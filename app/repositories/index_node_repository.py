"""Index-tree node repository for per-node summaries.

Sole writer of ``context_index_nodes`` -- the table holding optional per-node LLM
summaries for the navigate_context index_tree. Mirrors the delete-then-insert
replacement pattern of TagRepository and is parity-matched across backends
(TEXT FK on SQLite, UUID FK on PostgreSQL). The table exists only when
per-node summaries are enabled; read methods degrade to empty when it is absent.
"""

import contextlib
import sqlite3
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Any
from typing import NamedTuple
from typing import cast

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext
else:
    with contextlib.suppress(ImportError):
        import asyncpg


@dataclass(frozen=True, slots=True)
class IndexNodeRow:
    """One persisted index_tree node (an outline section with an LLM summary)."""

    node_id: str
    level: int
    ordinal: int
    title: str
    node_summary: str
    char_start: int
    char_end: int


class StoredNodeSummaries(NamedTuple):
    """Stored node summaries for one entry, indexed by both attachment keys.

    ``by_node_id`` is the primary index the navigate_context reader attaches
    with; ``by_span`` keys the same summaries by ``(char_start, char_end)`` so
    rows written by an older slug algorithm still re-attach to the sections
    the current parse computes for the unchanged text.
    """

    by_node_id: dict[str, str]
    by_span: dict[tuple[int, int], str]


class IndexNodeRepository(BaseRepository):
    """Repository for context_index_nodes (index_tree per-node summaries)."""

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the index-node repository.

        Args:
            backend: Storage backend for executing database operations.
        """
        super().__init__(backend)

    async def replace_nodes_for_context(
        self,
        context_id: str,
        nodes: list[IndexNodeRow],
        txn: 'TransactionContext | None' = None,
    ) -> None:
        """Replace all index-tree nodes for a context entry (delete then insert).

        Wholesale replacement keeps stored node summaries consistent with the
        current text: an empty ``nodes`` list simply clears them. Runs inside the
        caller's transaction when ``txn`` is provided so node writes commit
        atomically with the entry.

        Args:
            context_id: Canonical id of the owning context entry.
            nodes: Node rows to store (each carries a non-empty summary).
            txn: Optional transaction context for atomic multi-repository writes.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _replace_sqlite(conn: sqlite3.Connection) -> None:
                # Existence pre-check: when per-node summaries were never enabled
                # the table does not exist. Skipping here (rather than letting the
                # DELETE raise) keeps node writes additive -- a missing table
                # never aborts the surrounding store transaction.
                exists = conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='context_index_nodes'",
                ).fetchone()
                if exists is None:
                    return
                cursor = conn.cursor()
                cursor.execute(
                    f'DELETE FROM context_index_nodes WHERE context_id = {self._placeholder(1)}',
                    (context_id,),
                )
                for node in nodes:
                    cursor.execute(
                        'INSERT INTO context_index_nodes '
                        '(context_id, node_id, level, ordinal, title, node_summary, char_start, char_end) '
                        f'VALUES ({self._placeholders(8)})',
                        (
                            context_id, node.node_id, node.level, node.ordinal,
                            node.title, node.node_summary, node.char_start, node.char_end,
                        ),
                    )

            if txn:
                await self._run_sqlite_txn(_replace_sqlite, cast(sqlite3.Connection, txn.connection))
            else:
                await self.backend.execute_write(_replace_sqlite)
            return

        async def _replace_postgresql(conn: 'asyncpg.Connection') -> None:
            # Existence pre-check (see SQLite branch). to_regclass resolves via
            # search_path and returns NULL when the table is absent; checking
            # first avoids a failed statement that would poison the transaction.
            if await conn.fetchval("SELECT to_regclass('context_index_nodes')") is None:
                return
            await conn.execute(
                f'DELETE FROM context_index_nodes WHERE context_id = {self._placeholder(1)}',
                context_id,
            )
            for node in nodes:
                await conn.execute(
                    'INSERT INTO context_index_nodes '
                    '(context_id, node_id, level, ordinal, title, node_summary, char_start, char_end) '
                    f'VALUES ({self._placeholders(8)})',
                    context_id, node.node_id, node.level, node.ordinal,
                    node.title, node.node_summary, node.char_start, node.char_end,
                )

        if txn:
            await _replace_postgresql(cast('asyncpg.Connection', txn.connection))
        else:
            await self.backend.execute_write(cast(Any, _replace_postgresql))

    async def get_nodes_for_context(self, context_id: str) -> StoredNodeSummaries:
        """Return stored node summaries indexed by node id AND by char span.

        The primary attachment key is ``node_id``. The span index exists so a
        row written by an OLDER slug algorithm -- whose node_id the current
        on-demand parse no longer computes for the SAME section -- still
        re-attaches: spans derive from the text rather than the slug, and
        node rows are replaced on every text change, so an exact-span match
        against the same revision is deterministic.

        Returns empty indexes when the table is absent (node summaries
        disabled) or the entry has no stored node summaries.

        Args:
            context_id: Canonical id of the owning context entry.

        Returns:
            :class:`StoredNodeSummaries` with the node-id and span indexes.
        """
        if self.backend.backend_type == 'sqlite':

            def _get_sqlite(conn: sqlite3.Connection) -> StoredNodeSummaries:
                try:
                    cursor = conn.execute(
                        f'SELECT node_id, node_summary, char_start, char_end '
                        f'FROM context_index_nodes '
                        f'WHERE context_id = {self._placeholder(1)}',
                        (context_id,),
                    )
                except sqlite3.OperationalError:
                    return StoredNodeSummaries({}, {})
                rows = [row for row in cursor.fetchall() if row['node_summary']]
                return StoredNodeSummaries(
                    by_node_id={row['node_id']: row['node_summary'] for row in rows},
                    by_span={
                        (int(row['char_start']), int(row['char_end'])): row['node_summary']
                        for row in rows
                    },
                )

            return await self.backend.execute_read(_get_sqlite)

        async def _get_postgresql(conn: 'asyncpg.Connection') -> StoredNodeSummaries:
            try:
                rows = await conn.fetch(
                    f'SELECT node_id, node_summary, char_start, char_end '
                    f'FROM context_index_nodes '
                    f'WHERE context_id = {self._placeholder(1)}',
                    context_id,
                )
            except asyncpg.UndefinedTableError:
                return StoredNodeSummaries({}, {})
            kept = [row for row in rows if row['node_summary']]
            return StoredNodeSummaries(
                by_node_id={row['node_id']: row['node_summary'] for row in kept},
                by_span={
                    (int(row['char_start']), int(row['char_end'])): row['node_summary']
                    for row in kept
                },
            )

        return await self.backend.execute_read(cast(Any, _get_postgresql))

    async def count_all_nodes(self) -> int:
        """Return the total number of stored index_tree nodes (0 if table absent)."""
        if self.backend.backend_type == 'sqlite':

            def _count_sqlite(conn: sqlite3.Connection) -> int:
                try:
                    cursor = conn.execute('SELECT COUNT(*) AS n FROM context_index_nodes')
                except sqlite3.OperationalError:
                    return 0
                row = cursor.fetchone()
                return int(row['n']) if row is not None else 0

            return await self.backend.execute_read(_count_sqlite)

        async def _count_postgresql(conn: 'asyncpg.Connection') -> int:
            try:
                value = await conn.fetchval('SELECT COUNT(*) FROM context_index_nodes')
            except asyncpg.UndefinedTableError:
                return 0
            return int(value or 0)

        return await self.backend.execute_read(cast(Any, _count_postgresql))
