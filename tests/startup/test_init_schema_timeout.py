"""Tests that PostgreSQL schema-init statements run under the migration budget.

``init_database`` builds the base schema on a fresh PostgreSQL database by
splitting the schema file into statements and executing each one inside a single
``execute_write`` transaction. Those statements (large index builds, table
creation) can legitimately run longer than the connection pool's regular
``command_timeout`` (default 60s), which asyncpg applies as the CLIENT-side
per-call deadline for any ``conn.execute`` that carries no explicit ``timeout``.
The migration transaction raises the SERVER-side ``statement_timeout`` to
``POSTGRESQL_MIGRATION_TIMEOUT_S`` (default 300s), but a bare ``conn.execute``
would still be cancelled client-side at 60s -- crash-looping the boot despite the
raised server budget.

These tests drive the real ``_init_schema_postgresql`` closure (via a mock
backend whose ``execute_write`` invokes the closure with a mock asyncpg
connection) and assert that every schema statement is executed with the
migration-budget client-side deadline (``migration_timeout_s +
_CLIENT_TIMEOUT_MARGIN_S``), for BOTH the provided-backend path and the
temporary-backend backward-compatibility path.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.migrations._pg_ddl import _CLIENT_TIMEOUT_MARGIN_S
from app.migrations._pg_ddl import SCHEMA_INIT_ADVISORY_LOCK_SQL
from app.startup import init_database

# A minimal multi-statement PostgreSQL schema: enough to exercise the
# statement-splitting loop with more than one statement so the assertion covers
# every iteration, not just a single call.
_FAKE_SCHEMA_SQL = (
    'CREATE TABLE context_entries (id UUID PRIMARY KEY);\n'
    'CREATE INDEX idx_thread_source ON context_entries (id);\n'
    'CREATE TABLE tags (context_entry_id UUID);\n'
)

# The migration budget the timeout assertions expect. init_database reads
# settings.storage.postgresql_migration_timeout_s; the mock settings below pin it
# to this value so the expected client-side deadline is deterministic.
_MIGRATION_TIMEOUT_S = 300.0


def _make_pg_backend() -> tuple[MagicMock, MagicMock]:
    """Build a mock PostgreSQL backend that runs the init closure for real.

    Returns:
        A tuple ``(backend, conn)`` where ``backend.execute_write`` invokes the
        closure it is handed with ``conn`` (an ``AsyncMock`` asyncpg connection),
        so the closure's real statement loop runs against a capturing mock.
    """
    conn = MagicMock()
    conn.execute = AsyncMock(return_value=None)

    async def _run_write(fn: Callable[[Any], Awaitable[None]]) -> None:
        await fn(conn)

    backend = MagicMock()
    backend.backend_type = 'postgresql'
    backend.execute_write = AsyncMock(side_effect=_run_write)
    return backend, conn


def _schema_statement_calls(conn: MagicMock) -> list[tuple[str, dict[str, Any]]]:
    """Extract the schema-DDL ``conn.execute`` calls, excluding migration setup.

    The ``SET LOCAL statement_timeout`` bootstrap and the advisory-lock
    acquisition are migration-setup calls, not schema statements; filter them out
    so the assertions target the schema DDL the fix protects.

    Returns:
        A list of ``(sql, kwargs)`` for each schema-statement execute call.
    """
    calls: list[tuple[str, dict[str, Any]]] = []
    for call in conn.execute.await_args_list:
        sql = call.args[0]
        if sql.startswith('SET LOCAL statement_timeout'):
            continue
        if sql == SCHEMA_INIT_ADVISORY_LOCK_SQL:
            continue
        calls.append((sql, dict(call.kwargs)))
    return calls


def _mock_settings() -> MagicMock:
    """Mock settings pinning the migration timeout to the expected budget."""
    mock_settings = MagicMock()
    mock_settings.storage.postgresql_migration_timeout_s = _MIGRATION_TIMEOUT_S
    return mock_settings


@pytest.mark.asyncio
async def test_provided_backend_schema_statements_carry_migration_timeout() -> None:
    """Provided-backend path: every schema statement gets the migration budget.

    Drives ``init_database(backend=...)`` with a PostgreSQL mock backend and
    asserts each schema-DDL ``conn.execute`` call carries
    ``timeout == migration_timeout_s + _CLIENT_TIMEOUT_MARGIN_S`` rather than
    inheriting the pool's shorter ``command_timeout``.
    """
    backend, conn = _make_pg_backend()
    expected_timeout = _MIGRATION_TIMEOUT_S + _CLIENT_TIMEOUT_MARGIN_S

    with (
        patch('app.startup.settings', _mock_settings()),
        patch('pathlib.Path.exists', return_value=True),
        patch('pathlib.Path.read_text', return_value=_FAKE_SCHEMA_SQL),
        patch('app.backends.postgresql_backend.quote_pg_identifier', return_value='"public"'),
    ):
        await init_database(backend=backend)

    schema_calls = _schema_statement_calls(conn)
    # The fake schema has three statements; all three must run.
    assert len(schema_calls) == 3
    for sql, kwargs in schema_calls:
        assert kwargs.get('timeout') == expected_timeout, (
            f'schema statement executed without the migration-budget client deadline: {sql!r} '
            f'(kwargs={kwargs})'
        )


@pytest.mark.asyncio
async def test_temp_backend_schema_statements_carry_migration_timeout() -> None:
    """Temporary-backend path: every schema statement gets the migration budget.

    ``init_database()`` with no backend argument creates a temporary backend for
    backward compatibility; that path has its own ``_init_schema_postgresql``
    closure. This asserts the fix is applied there too, so a first-boot schema
    statement is not cancelled client-side at the pool's shorter deadline.
    """
    backend, conn = _make_pg_backend()
    backend.initialize = AsyncMock(return_value=None)
    backend.shutdown = AsyncMock(return_value=None)
    expected_timeout = _MIGRATION_TIMEOUT_S + _CLIENT_TIMEOUT_MARGIN_S

    with (
        patch('app.startup.settings', _mock_settings()),
        patch('app.startup.create_backend', return_value=backend),
        patch('pathlib.Path.exists', return_value=True),
        patch('pathlib.Path.read_text', return_value=_FAKE_SCHEMA_SQL),
        patch('app.backends.postgresql_backend.quote_pg_identifier', return_value='"public"'),
    ):
        await init_database()

    schema_calls = _schema_statement_calls(conn)
    assert len(schema_calls) == 3
    for sql, kwargs in schema_calls:
        assert kwargs.get('timeout') == expected_timeout, (
            f'schema statement executed without the migration-budget client deadline: {sql!r} '
            f'(kwargs={kwargs})'
        )
