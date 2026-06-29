"""Shared helpers for running PostgreSQL migration DDL under the migration timeout.

The connection pool is created with ``command_timeout = POSTGRESQL_COMMAND_TIMEOUT_S``
(default 60s), which asyncpg applies as the DEFAULT CLIENT-SIDE deadline for every
``conn.execute`` that does not pass an explicit ``timeout``. Migration DDL additionally
raises the SERVER-SIDE ``statement_timeout`` to ``POSTGRESQL_MIGRATION_TIMEOUT_S``
(default 300s) so long index builds and table rewrites get a larger budget -- but
asyncpg cancels the call client-side at the 60s default FIRST (raising
``asyncio.TimeoutError``, which is not a retryable ``QueryCanceledError``), so the
larger server-side budget never takes effect and a slow migration is killed at ~60s and
crashes startup.

:func:`execute_migration_ddl` passes the migration timeout as the explicit per-call
``timeout`` so asyncpg's client-side deadline matches the server-side budget, plus a
small margin so an overrun is cancelled SERVER-side first -- a retryable
``QueryCanceledError`` (57014) the write-retry loop recovers from -- rather than
client-side, which surfaces as a non-retryable ``asyncio.TimeoutError``.

:func:`begin_migration` raises the server-side budget for the whole migration
transaction via ``SET LOCAL`` and acquires the shared schema-init advisory lock under
it. ``SET LOCAL`` is transaction-scoped: PostgreSQL auto-reverts it to the session
value on COMMIT or ROLLBACK, so no ``finally`` restore is needed. A plain ``SET`` paired
with a ``finally`` restore must NOT be used: when a DDL statement fails the transaction
enters the aborted state, and the ``finally``'s restore ``SET`` would itself raise
``InFailedSQLTransactionError`` (25P02), MASKING the real DDL error and converting a
retryable ``QueryCanceledError`` into a non-retryable one.
"""

import asyncpg

# The advisory-lock key every schema migration takes so concurrent pods -- and
# concurrent migrations within one pod -- serialize their DDL against one another.
SCHEMA_INIT_ADVISORY_LOCK_SQL = "SELECT pg_advisory_xact_lock(hashtext('mcp_context_schema_init'))"

# Seconds by which asyncpg's client-side per-call deadline exceeds the server-side
# ``statement_timeout`` budget. Keeping the client deadline slightly larger lets the
# SERVER cancel an overrunning statement first (a retryable ``QueryCanceledError``)
# instead of the client (a non-retryable ``asyncio.TimeoutError``). The margin only
# needs to cover the round-trip on which PostgreSQL reports the cancellation.
_CLIENT_TIMEOUT_MARGIN_S = 5.0


async def execute_migration_ddl(
    conn: asyncpg.Connection,
    statement: str,
    timeout_s: float,
) -> None:
    """Execute one migration statement under the migration budget.

    Args:
        conn: PostgreSQL connection.
        statement: A single SQL statement -- migration DDL or the advisory-lock
            acquisition, both of which may legitimately run longer than the regular
            per-query command timeout.
        timeout_s: The server-side ``statement_timeout`` budget in seconds (pass
            ``settings.storage.postgresql_migration_timeout_s``). asyncpg's client-side
            per-call deadline is set to ``timeout_s + _CLIENT_TIMEOUT_MARGIN_S`` so the
            pool's shorter ``command_timeout`` does not pre-empt the configured budget
            and an overrun is cancelled server-side (retryable) before the client gives
            up (non-retryable).
    """
    await conn.execute(statement, timeout=timeout_s + _CLIENT_TIMEOUT_MARGIN_S)


async def begin_migration(
    conn: asyncpg.Connection,
    migration_timeout_s: float,
) -> None:
    """Prepare a migration transaction: raise the statement budget, take the shared lock.

    MUST be called as the first action inside ``execute_write()``'s transaction. It:

    1. Raises the SERVER-SIDE ``statement_timeout`` to the migration budget for THIS
       transaction only, via ``SET LOCAL`` -- which PostgreSQL auto-reverts to the
       session value on COMMIT or ROLLBACK. ``SET LOCAL`` (not a plain ``SET`` + a
       ``finally`` restore) is required: a failed DDL aborts the transaction, and a
       restore ``SET`` in a ``finally`` would then raise ``InFailedSQLTransactionError``
       (25P02), masking the real error and defeating the ``QueryCanceledError`` retry.
    2. Acquires the shared schema-init advisory lock under the migration budget -- both
       the server-side budget from step 1 and asyncpg's client-side per-call deadline
       (via :func:`execute_migration_ddl`) -- so a multi-pod peer's slow first-time
       build can be waited out instead of being cancelled at the pool's shorter
       ``command_timeout``. The lock is transaction-scoped (``pg_advisory_xact_lock``),
       releasing automatically on COMMIT or ROLLBACK.

    Args:
        conn: PostgreSQL connection, already inside ``execute_write()``'s transaction.
        migration_timeout_s: The migration budget in seconds
            (``settings.storage.postgresql_migration_timeout_s``).
    """
    migration_timeout_ms = int(migration_timeout_s * 1000)
    await conn.execute(f'SET LOCAL statement_timeout = {migration_timeout_ms}')
    await execute_migration_ddl(conn, SCHEMA_INIT_ADVISORY_LOCK_SQL, migration_timeout_s)
