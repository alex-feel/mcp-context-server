"""Query-plan pins for scoped SQLite statements over ``context_entries``.

The READ predicate ORs an owner match, a public match and a correlated EXISTS over
``context_entry_grants``. SQLite cannot answer that OR through separate index lookups,
so a scoped statement without an indexable client filter visits every
``context_entries`` row, and ``owner_id``/``visibility`` sit after ``text_content`` in
the row, so reading them from the table walks each row's text overflow pages. The
SQLite covering access indexes the access-control migration provisions let these
statements read index pages only. Each pin runs ``EXPLAIN QUERY PLAN`` on a database
initialized the way server startup initializes it (base schema, then the migration)
and requires every step that reads ``context_entries`` to use one of those indexes.
The database carries no ``sqlite_stat1`` table, so the plans are the ones a database
gets before any ``ANALYZE`` has run.
"""

import re
import sqlite3
from collections.abc import AsyncGenerator
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import pytest_asyncio

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.backends import StorageBackend
from app.backends import create_backend
from app.migrations.access_control import apply_access_control_migration
from app.repositories.statistics_repository import StatisticsRepository
from app.startup import init_database

# A plan step that reads context_entries, under its table name or the statistics alias.
_ENTRY_STEP = re.compile(r'^(?:SCAN|SEARCH) (?:context_entries|ce)\b')
_COVERING_ACCESS_INDEX = 'USING COVERING INDEX idx_context_access_'

_SCOPES = [
    pytest.param(AccessScope('local', frozenset()), id='principal-only'),
    pytest.param(AccessScope('alice', frozenset({'team-a', 'team-b'})), id='principal-with-groups'),
]


@pytest_asyncio.fixture
async def startup_backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend initialized like server startup: base schema, then the migration."""
    backend = create_backend(backend_type='sqlite', db_path=str(tmp_path / 'test_access_plans.db'))
    await backend.initialize()
    try:
        await init_database(backend=backend)
        await apply_access_control_migration(backend)
        yield backend
    finally:
        await backend.shutdown()


async def _plan(backend: StorageBackend, sql: str, params: list[Any]) -> list[str]:
    """Return the detail column of every ``EXPLAIN QUERY PLAN`` row for one statement."""

    def _explain(conn: sqlite3.Connection) -> list[str]:
        return [str(row[3]) for row in conn.execute(f'EXPLAIN QUERY PLAN {sql}', params).fetchall()]

    return await backend.execute_read(_explain)


def _assert_entries_read_from_covering_index(sql: str, plan: list[str]) -> None:
    """Require at least one context_entries step and a covering access index on every one."""
    entry_steps = [step for step in plan if _ENTRY_STEP.match(step)]
    assert entry_steps, f'no context_entries step in the plan of {sql!r}: {plan}'
    uncovered = [step for step in entry_steps if _COVERING_ACCESS_INDEX not in step]
    assert not uncovered, f'{sql!r} reads context_entries outside a covering access index: {plan}'


async def _statistics_statements(
    backend: StorageBackend, scope: AccessScope, monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Run ``get_database_statistics`` and capture every statement it executed.

    Returns:
        The executed statements in order. The SQLite trace callback reports each one
        with its bound values inlined, so each text is the exact statement the
        repository ran.
    """
    statements: list[str] = []
    execute_read = backend.execute_read

    async def _tracing_execute_read(operation: Callable[..., object], *args: Any, **kwargs: Any) -> object:
        def _traced(conn: sqlite3.Connection, *inner_args: Any, **inner_kwargs: Any) -> object:
            conn.set_trace_callback(statements.append)
            try:
                return operation(conn, *inner_args, **inner_kwargs)
            finally:
                conn.set_trace_callback(None)

        return await execute_read(_traced, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(backend, 'execute_read', _tracing_execute_read)
        await StatisticsRepository(backend).get_database_statistics(scope=scope)
    return statements


class TestScopedStatementPlans:
    """Scoped statements over context_entries read only a covering access index."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize('scope', _SCOPES)
    async def test_statistics_statements_use_covering_access_indexes(
        self, startup_backend: StorageBackend, scope: AccessScope, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every get_statistics statement -- the totals, the source, content-type and
        thread aggregates, and the image and tag figures -- reads context_entries from
        a covering access index."""
        statements = await _statistics_statements(startup_backend, scope, monkeypatch)

        assert any(statement.startswith('SELECT COUNT(*) AS count FROM context_entries') for statement in statements)
        assert any('GROUP BY source' in statement for statement in statements)
        assert any('FROM tags t' in statement for statement in statements)
        assert any('FROM image_attachments i' in statement for statement in statements)
        for statement in statements:
            _assert_entries_read_from_covering_index(statement, await _plan(startup_backend, statement, []))

    @pytest.mark.asyncio
    @pytest.mark.parametrize('scope', _SCOPES)
    @pytest.mark.parametrize(
        ('client_filter', 'client_params'),
        [
            pytest.param('', [], id='unfiltered'),
            pytest.param('source = ? AND ', ['agent'], id='source'),
            pytest.param('thread_id = ? AND ', ['thread-a'], id='thread'),
        ],
    )
    async def test_candidate_statement_uses_covering_access_index(
        self,
        startup_backend: StorageBackend,
        scope: AccessScope,
        client_filter: str,
        client_params: list[Any],
    ) -> None:
        """The semantic-search candidate statement -- the client filters, then the READ
        predicate -- reads context_entries from a covering access index."""
        read = build_access_predicate(scope, mode=AccessMode.READ, backend_type='sqlite', outer='context_entries')
        sql = f'SELECT id FROM context_entries WHERE {client_filter}{read.sql}'

        plan = await _plan(startup_backend, sql, [*client_params, *read.params])

        _assert_entries_read_from_covering_index(sql, plan)
