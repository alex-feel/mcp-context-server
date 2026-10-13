"""Backend construction, the owner precondition and bounded shutdown shared by the maintenance CLI modes.

Every mode builds its backend with :func:`make_backend`, which runs the maintenance
factory: a PostgreSQL URL is dialed with its own credentials under the maintenance
session profile, which binds no identity and turns ``row_security`` off. Before its first
data statement each PostgreSQL mode calls :func:`assert_maintenance_connection`, which
refuses a role that row-level security filters on any managed table: such a role would
read the rows it may not see as absent and act on that, for example dropping embedding
payloads it never read. ``row_security = off`` is the backstop behind it, turning any
filtered statement that still runs into an error, and :func:`owner_refusal` maps that
error, like the refusal, to the remediation the dispatcher prints.
"""

import asyncio
import contextlib
from typing import Any
from typing import Final
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends import create_maintenance_backend
from app.backends.base import MAINTENANCE
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.cli._database_url import parse_backend_url
from app.errors import format_exception_message
from app.migrations.rls_ddl import MANAGED_TABLES

# $1 the schema-qualified managed tables: the connected role, and each table that
# row-level security filters for it. row_security_active() answers whatever the
# row_security setting says, so the check holds behind a pooler that drops session SETs.
FILTERED_TABLES_SQL: Final = (
    'SELECT current_user::text, '
    'COALESCE(array_agg(c.relname::text ORDER BY c.relname) '
    'FILTER (WHERE pg_catalog.row_security_active(c.oid)), ARRAY[]::text[]) '
    'FROM pg_catalog.pg_class c '
    'WHERE c.oid IN (SELECT pg_catalog.to_regclass(t) FROM pg_catalog.unnest($1::text[]) AS t)'
)
_REMEDIATION: Final = "Rerun with the owner's connection string (docs/migration-v2-to-v3.md)."
_INSUFFICIENT_PRIVILEGE: Final = '42501'


class MaintenanceConnectionError(Exception):
    """The command connected as a role that row-level security filters, so it touches no data."""


def make_backend(
    source_url: str, *, provision_vector: bool | None = None,
) -> StorageBackend:
    """Build the maintenance backend pointed at ``source_url``.

    Args:
        source_url: URL passed to ``--source-url``.
        provision_vector: Explicit pgvector-provisioning decision forwarded to
            the PostgreSQL backend (ignored for SQLite, whose sqlite-vec load is
            unconditional). ``None`` lets the backend resolve it from settings
            and a probe at ``initialize()`` time; ``False`` skips ``CREATE
            EXTENSION vector`` so the zero-data reverse path runs on a
            pgvector-less host; ``True`` provisions the extension for the fp32
            rebuild. See
            :func:`app.cli.migrate_compression.decompress._decompress_needs_vector`.

    Returns:
        Constructed (not yet initialized) :class:`StorageBackend` matching the
        URL scheme. A PostgreSQL URL keeps its own credentials. The URL
        classification is performed by :func:`parse_backend_url` which raises
        ``ValueError`` for unrecognized schemes; the exception propagates to the
        caller.
    """
    backend_kind, address = parse_backend_url(source_url)
    if backend_kind == 'sqlite':
        return create_maintenance_backend(db_path=address)
    return create_maintenance_backend(connection_string=address, provision_vector=provision_vector)


async def assert_maintenance_connection(target: 'asyncpg.Connection[Any] | StorageBackend', schema: str) -> None:
    """Refuse to go on when row-level security filters the connected role on any managed table.

    Each PostgreSQL mode calls it before its first data statement. The table owner passes,
    because row-level security is enabled without FORCE; a SQLite backend has nothing to
    check.

    Args:
        target: A raw connection or a backend of the database under operation.
        schema: The schema of the tables, unquoted (``POSTGRESQL_SCHEMA``).

    Raises:
        MaintenanceConnectionError: If row-level security filters the connected role; the
            message names the role, the tables and the remediation.
    """
    names = [f'{quote_pg_identifier(schema)}.{table}' for table in MANAGED_TABLES]

    async def _filtered(conn: 'asyncpg.Connection[Any]') -> tuple[str, list[str]]:
        row = await conn.fetchrow(FILTERED_TABLES_SQL, names)
        if row is None:
            return '', []
        return str(row[0]), [str(table) for table in row[1]]

    if not isinstance(target, StorageBackend):
        current_user, tables = await _filtered(target)
    elif target.backend_type == 'postgresql':
        current_user, tables = await target.execute_read(cast(Any, _filtered), access_scope=MAINTENANCE)
    else:
        return
    if tables:
        raise MaintenanceConnectionError(
            "This command must connect with the table owner's credentials: row-level security filters "
            f'{current_user} on {", ".join(tables)}. {_REMEDIATION}',
        )


def owner_refusal(exc: BaseException) -> str | None:
    """Return the remediation for a refused or unprivileged maintenance connection, or None.

    The precondition's refusal keeps its own message. A privilege error (SQLSTATE 42501)
    anywhere in the chain of causes and contexts, raised for example by a statement that
    row-level security would filter while ``row_security`` is off, maps to the same
    remediation, quoting the failure.

    Args:
        exc: The exception a mode raised.

    Returns:
        The message the dispatcher prints, or None for any other failure.
    """
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, MaintenanceConnectionError):
            return str(current)
        if getattr(current, 'sqlstate', None) == _INSUFFICIENT_PRIVILEGE:
            failure = format_exception_message(exc) if isinstance(exc, Exception) else str(exc)
            return (
                "This command must connect with the table owner's credentials: the database refused a statement "
                f'({failure}). {_REMEDIATION}'
            )
        pending.extend(link for link in (current.__cause__, current.__context__) if link is not None)
    return None


async def shutdown_backend(backend: StorageBackend) -> None:
    """Suppress timeouts during backend shutdown.

    A long-running CLI invocation may keep PG sessions open until the
    process exits; bound the wait so a stuck connection does not block
    teardown.
    """
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=10.0)
