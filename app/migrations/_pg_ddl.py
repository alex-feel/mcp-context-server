"""Shared helper for executing PostgreSQL migration DDL under the migration timeout.

The connection pool is created with ``command_timeout = POSTGRESQL_COMMAND_TIMEOUT_S``
(default 60s), which asyncpg applies as the DEFAULT CLIENT-SIDE deadline for every
``conn.execute`` that does not pass an explicit ``timeout``. Migration DDL additionally
raises the SERVER-SIDE ``statement_timeout`` to ``POSTGRESQL_MIGRATION_TIMEOUT_S``
(default 300s) so long index builds and table rewrites get a larger budget -- but
asyncpg cancels the call client-side at the 60s default FIRST (raising
``asyncio.TimeoutError``, which is not a retryable ``QueryCanceledError``), so the
larger server-side budget never takes effect and a slow migration is killed at ~60s and
crashes startup.

Passing the migration timeout as the explicit per-call ``timeout`` raises asyncpg's
client-side deadline to match the server-side budget, so migration DDL -- and the
advisory-lock acquisition that may wait on a peer pod's migration -- get their full
configured time instead of the regular per-query command timeout.
"""

import asyncpg


async def execute_migration_ddl(
    conn: asyncpg.Connection,
    statement: str,
    timeout_s: float,
) -> None:
    """Execute one migration statement with ``timeout_s`` as asyncpg's per-call timeout.

    Args:
        conn: PostgreSQL connection.
        statement: A single SQL statement -- migration DDL or the advisory-lock
            acquisition, both of which may legitimately run longer than the regular
            per-query command timeout.
        timeout_s: Per-call client-side timeout in seconds. Pass
            ``settings.storage.postgresql_migration_timeout_s`` so the pool's shorter
            ``command_timeout`` does not pre-empt the configured migration budget.
    """
    await conn.execute(statement, timeout=timeout_s)
