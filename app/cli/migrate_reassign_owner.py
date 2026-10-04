"""Reassign every context entry owned by one principal to another.

Implements the ``--reassign-owner FROM TO`` flag of
``mcp-context-server-migrate``. It rewrites ``context_entries.owner_id`` from
FROM to TO in one statement, for example to hand the rows written under
``ACCESS_CONTROL_DEFAULT_PRINCIPAL`` to an operator's identity-provider subject
after switching to ``MCP_AUTH_PROVIDER=jwt``.

Both values are bound as statement parameters and carry no character-set
limit, so subjects outside ``SAFE_PRINCIPAL_ID_PATTERN`` (such as the Auth0
forms ``auth0|...`` and ``google-oauth2|...``) are accepted here even though
the environment variable cannot hold them.

The statement sets ``updated_at`` to the current time on both backends and
leaves ``version`` alone: the version token guards content against concurrent
updates, and the server is stopped while the reassignment runs.
``context_entry_grants`` is not touched, because ``granted_by`` records who
created a grant and the grantee rows name principals, not owners.

The CLI is single-backend: the source is the database under operation and
``--target-url`` is ignored (mirrors ``--compress``, ``--decompress``,
``--re-embed`` and ``--embed-missing``).
"""

import asyncio
import logging
import sqlite3
import sys
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.cli._backend import make_backend
from app.cli._backend import shutdown_backend
from app.cli._database_url import mask_credentials
from app.settings import get_settings

logger = logging.getLogger(__name__)


def run_reassign_owner(source_url: str, from_principal: str, to_principal: str, *, dry_run: bool) -> int:
    """Public entry point for the ``--reassign-owner`` flag.

    Args:
        source_url: Database URL passed to ``--source-url``.
        from_principal: Current owner whose rows are reassigned.
        to_principal: New owner of those rows.
        dry_run: When True, report the matching row count only; no writes.

    Returns:
        Process exit code: 0 on success or success-no-op, 1 on invalid
        arguments, 2 on an unrecoverable failure.
    """
    if not from_principal or not to_principal:
        print('[ERROR] --reassign-owner FROM and TO must not be empty.', file=sys.stderr)
        return 1
    if from_principal == to_principal:
        print(
            f'[ERROR] --reassign-owner FROM and TO are the same principal ({from_principal!r}); '
            'nothing to reassign.',
            file=sys.stderr,
        )
        return 1

    # Resolve the settings before the async body so an invalid environment
    # reaches the dispatcher, which exits EX_CONFIG (78), instead of the
    # generic failure path below.
    get_settings()

    lines = [
        f'Reassigning context entry ownership: {from_principal!r} -> {to_principal!r}',
        f'Source: {mask_credentials(source_url)}',
        'Stop the server before running; grants are not changed.',
    ]
    if dry_run:
        lines.append('DRY-RUN: no rows are changed; only the matching row count is reported')
    print('\n'.join(lines), file=sys.stderr)

    try:
        return asyncio.run(_reassign_async(source_url, from_principal, to_principal, dry_run=dry_run))
    except Exception as exc:
        logger.exception('owner reassignment failed: %s', exc)
        return 2


def _entries_phrase(count: int) -> str:
    """Return ``'<count> context entry'`` or ``'<count> context entries'``."""
    return f'{count} context {"entry" if count == 1 else "entries"}'


async def _reassign_async(source_url: str, from_principal: str, to_principal: str, *, dry_run: bool) -> int:
    """Async body for :func:`run_reassign_owner`.

    Args:
        source_url: URL passed to ``--source-url``.
        from_principal: Current owner whose rows are reassigned.
        to_principal: New owner of those rows.
        dry_run: When True, report the matching row count only.

    Returns:
        Process exit code: 0 on success or success-no-op.
    """
    # The statements touch no vector column, so the pgvector extension and its
    # codec are never provisioned for this mode.
    backend = make_backend(source_url, provision_vector=False)
    await backend.initialize()
    try:
        if dry_run:
            count = await _count_owned(backend, from_principal)
            print(
                f'[DRY-RUN] {_entries_phrase(count)} owned by {from_principal!r} '
                f'would be reassigned to {to_principal!r}; rerun without --dry-run to apply.',
                file=sys.stderr,
            )
            return 0

        reassigned = await _reassign(backend, from_principal, to_principal)
        if reassigned == 0:
            print(f'No context entries are owned by {from_principal!r}; nothing reassigned.', file=sys.stderr)
        else:
            print(
                f'Reassigned {_entries_phrase(reassigned)} from {from_principal!r} to {to_principal!r}.',
                file=sys.stderr,
            )
        return 0
    finally:
        await shutdown_backend(backend)


async def _count_owned(backend: StorageBackend, principal_id: str) -> int:
    """Count the context entries owned by ``principal_id``.

    Args:
        backend: Initialized storage backend.
        principal_id: Owner to count.

    Returns:
        Number of matching ``context_entries`` rows.
    """
    if backend.backend_type == 'sqlite':

        def _count_sqlite(conn: sqlite3.Connection) -> int:
            row = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE owner_id = ?', (principal_id,),
            ).fetchone()
            return int(row[0])

        return await backend.execute_read(_count_sqlite)

    async def _count_pg(conn: asyncpg.Connection) -> int:
        value = await conn.fetchval('SELECT COUNT(*) FROM context_entries WHERE owner_id = $1', principal_id)
        return int(value)

    count: int = await backend.execute_read(cast(Any, _count_pg))
    return count


async def _reassign(backend: StorageBackend, from_principal: str, to_principal: str) -> int:
    """Rewrite ``owner_id`` from ``from_principal`` to ``to_principal`` in one statement.

    No SQLite trigger maintains ``updated_at``, so the statement sets it
    itself; on PostgreSQL the ``update_context_entries_updated_at`` trigger
    sets the same value.

    Args:
        backend: Initialized storage backend.
        from_principal: Current owner whose rows are reassigned.
        to_principal: New owner of those rows.

    Returns:
        Number of rows reassigned.
    """
    if backend.backend_type == 'sqlite':

        def _update_sqlite(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                'UPDATE context_entries SET owner_id = ?, updated_at = CURRENT_TIMESTAMP WHERE owner_id = ?',
                (to_principal, from_principal),
            )
            return cursor.rowcount

        return await backend.execute_write(_update_sqlite)

    async def _update_pg(conn: asyncpg.Connection) -> int:
        result = await conn.execute(
            'UPDATE context_entries SET owner_id = $1, updated_at = CURRENT_TIMESTAMP WHERE owner_id = $2',
            to_principal,
            from_principal,
        )
        return int(result.split()[-1]) if result else 0

    reassigned: int = await backend.execute_write(cast(Any, _update_pg))
    return reassigned


__all__ = ['run_reassign_owner']
