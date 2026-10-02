"""Tests for the compression CLI storage helpers: the PostgreSQL migration budget and the budgeted row count."""

import asyncio
from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from typing import cast

from tests.cli.migrate_compression._pg_fakes import RecordingConn
from tests.cli.migrate_compression._pg_fakes import budget_set_local
from tests.cli.migrate_compression._pg_fakes import expected_client_timeout


class _ReadBackend:
    """A backend whose execute_read invokes the async operation on a recording conn."""

    backend_type = 'postgresql'

    def __init__(self, conn: RecordingConn) -> None:
        self._conn = conn

    async def execute_read(self, operation: Callable[[RecordingConn], Awaitable[int]]) -> int:
        return await operation(self._conn)


def test_raise_pg_migration_budget_floors_sub_millisecond_budget() -> None:
    """A sub-millisecond migration budget must not become statement_timeout = 0.

    PostgreSQL treats ``SET LOCAL statement_timeout = 0`` as UNLIMITED, so a
    truncating conversion would silently remove the server-side backstop and
    leave only the client-side deadline -- which surfaces a non-retryable
    ``asyncio.TimeoutError`` instead of the retryable ``QueryCanceledError``
    this budget exists to produce.
    """
    from app.cli.migrate_compression.storage import raise_pg_migration_budget

    conn = RecordingConn()
    asyncio.run(raise_pg_migration_budget(cast(Any, conn), 0.0005))
    assert conn.execute_calls == [('SET LOCAL statement_timeout = 1', None)]


def test_count_table_pg_runs_under_migration_budget() -> None:
    """The pre-lock estimate COUNT(*) carries the migration budget, not the pool timeout.

    On a large corpus a bare fetchval on the borrowed pool would be cancelled at the ~60s
    command_timeout before the budgeted transaction begins, aborting the CLI where raising
    POSTGRESQL_MIGRATION_TIMEOUT_S is meant to help. The pre-lock count runs in a short read
    transaction that raises both the server-side (SET LOCAL) and client-side deadlines.
    """
    from app.cli.migrate_compression.storage import count_table

    conn = RecordingConn()
    backend = _ReadBackend(conn)
    count = asyncio.run(count_table(cast(Any, backend), 'vec_context_embeddings'))
    assert count == 0
    # The read transaction raises the server-side budget.
    set_local = [stmt for stmt, _ in conn.execute_calls if stmt.startswith('SET LOCAL statement_timeout')]
    assert set_local == [budget_set_local()]
    # The COUNT(*) carries the explicit client-side deadline (not the bare pool timeout).
    counts = [t for stmt, t in conn.fetchval_calls if 'COUNT(*) FROM vec_context_embeddings' in stmt]
    assert counts
    assert all(t == expected_client_timeout() for t in counts)
