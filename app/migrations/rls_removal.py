"""PostgreSQL row-level security: taking the installed objects out again, and refusing to start over them switched off.

:func:`remove_rls` drops, as the table owner, under the schema-init lock and in one
transaction, everything :mod:`app.migrations.rls` installs: the read view and its write
rejection, every policy of ours on every managed table, the guard trigger and the
helpers. It disables row-level security only on the tables whose read policy recorded
that it was off before the migration first enabled it, and reports every table it leaves
enabled. The objects are found by their fixed names, so removal does not depend on the
configured policy role.

:func:`guard_rls_disable_over_enabled` refuses a PostgreSQL start with
``ENABLE_POSTGRESQL_RLS`` off while any policy of ours or the read view exists: a request
role that does not own the tables would then read only public rows and fail every write.
"""

import logging
from dataclasses import dataclass
from typing import Any
from typing import Final
from typing import cast

import asyncpg

from app.access_scope import READ_VIEW
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.errors import ConfigurationError
from app.errors import format_exception_message
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import ObjectKey
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import disable_statement
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import policy_names
from app.migrations.rls_ddl import read_view_drop
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_state import RlsState
from app.migrations.rls_state import read_rls_state
from app.settings import AppSettings

logger = logging.getLogger(__name__)

PREENABLED: Final = 'preenabled=true'
NO_READABLE_COMMENT: Final = 'no readable comment'

# $1 schema name, $2 every policy name of ours: the tables carrying one of them.
_OUR_POLICY_TABLES_SQL: Final = (
    'SELECT c.relname, p.polname FROM pg_catalog.pg_policy p '
    'JOIN pg_catalog.pg_class c ON c.oid = p.polrelid JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    'WHERE n.nspname = $1 AND p.polname = ANY($2::text[])'
)
# $1 schema name, $2 the view's name.
_VIEW_EXISTS_SQL: Final = (
    'SELECT EXISTS (SELECT 1 FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    "WHERE n.nspname = $1 AND c.relname = $2 AND c.relkind = 'v')"
)


@dataclass(frozen=True, slots=True)
class RlsRemovalReport:
    """What one run of :func:`remove_rls` did, or would do in a dry run.

    Attributes:
        statements: The statements issued, in order; empty when nothing was installed.
        disabled: The tables whose row-level security was switched off.
        left_enabled: Each managed table left with row-level security on, with the
            reason: ``preenabled=true`` when it was on before the migration enabled it,
            ``no readable comment`` when nothing of ours says it was not.
    """

    statements: list[str]
    disabled: list[str]
    left_enabled: dict[str, str]


def plan_removal(state: RlsState, *, schema: str) -> RlsRemovalReport:
    """Return what removing every object of ours from ``state`` takes, in reverse dependency order.

    Args:
        state: What the catalog holds.
        schema: The schema of the tables, quoted.

    Returns:
        The statements, the tables to disable and the tables left enabled.
    """
    statements: list[str] = []
    if ObjectKey(RlsObjectKind.VIEW, READ_VIEW, f'{schema}.{READ_VIEW}') in state.objects:
        statements.append(read_view_drop(schema))
    statements.extend(_drops(state, (reject_function(schema).key,)))

    disabled: list[str] = []
    left_enabled: dict[str, str] = {}
    for table in (table for table in MANAGED_TABLES if table in state.tables):
        keys = [ObjectKey(RlsObjectKind.POLICY, name, f'{schema}.{table}') for name in policy_names(table)]
        statements.extend(_drops(state, keys))
        if not state.tables[table]:
            continue
        read = state.objects.get(keys[0])
        recorded = None if read is None or read.fingerprint is None else read.fingerprint.preenabled
        if recorded is False:
            statements.append(disable_statement(table, schema=schema))
            disabled.append(table)
        else:
            left_enabled[table] = PREENABLED if recorded else NO_READABLE_COMMENT

    statements.extend(_drops(state, (guard_trigger(schema).key, guard_function(schema).key)))
    statements.extend(_drops(state, tuple(helper.key for helper in helper_functions(schema))))
    return RlsRemovalReport(statements=statements, disabled=disabled, left_enabled=left_enabled)


async def remove_rls(backend: StorageBackend, settings: AppSettings, *, dry_run: bool) -> RlsRemovalReport:
    """Drop every row-level security object of ours, under the lock and in one transaction.

    It must run as the table owner, while no server instance runs. A table keeps
    row-level security when it was on before the migration first enabled it, or when no
    readable record of ours says otherwise; the report names each such table.

    Args:
        backend: A maintenance backend, logged in as the table owner.
        settings: The application settings: the schema and the migration budget.
        dry_run: Report what would be dropped and disabled without changing anything.

    Returns:
        The statements issued, or that a dry run would issue, and the tables disabled and
        left enabled; empty on SQLite, which has no row-level security.

    Raises:
        RuntimeError: If reading the catalog or dropping an object fails.
    """
    if backend.backend_type != 'postgresql':
        return RlsRemovalReport(statements=[], disabled=[], left_enabled={})
    schema_name = settings.storage.postgresql_schema
    schema = quote_pg_identifier(schema_name)
    timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _remove(conn: asyncpg.Connection) -> RlsRemovalReport:
        await begin_migration(conn, timeout_s)
        report = plan_removal(await read_rls_state(conn, schema_name=schema_name, role=None), schema=schema)
        if not dry_run:
            for statement in report.statements:
                await execute_migration_ddl(conn, statement, timeout_s)
        return report

    try:
        report: RlsRemovalReport = await backend.execute_write(cast(Any, _remove), access_scope=MAINTENANCE)
    except Exception as e:
        logger.error(f'Failed to remove row-level security: {e}')
        raise RuntimeError(f'Row-level security removal failed: {format_exception_message(e)}') from e
    if not dry_run and report.statements:
        logger.info(f'Removed row-level security from schema {schema}: disabled on {", ".join(report.disabled) or "no table"}')
    return report


async def guard_rls_disable_over_enabled(backend: StorageBackend, settings: AppSettings) -> None:
    """Refuse a PostgreSQL start with ENABLE_POSTGRESQL_RLS off while the objects it installs exist.

    Runs only on PostgreSQL with the flag off. It reads the catalog alone, so any role can
    run it.

    Args:
        backend: A backend of the database.
        settings: The application settings: the flag and the schema.

    Raises:
        ConfigurationError: If a policy of ours or the read view exists; the message names
            where and how to remove them.
    """
    if backend.backend_type != 'postgresql' or settings.access_control.postgresql_rls_enabled:
        return
    schema_name = settings.storage.postgresql_schema
    ours = {table: set(policy_names(table)) for table in MANAGED_TABLES}

    async def _installed(conn: asyncpg.Connection) -> list[str]:
        names = sorted({name for table_names in ours.values() for name in table_names})
        rows = await conn.fetch(_OUR_POLICY_TABLES_SQL, schema_name, names)
        tables = {str(row[0]) for row in rows if str(row[1]) in ours.get(str(row[0]), set())}
        relations = [table for table in MANAGED_TABLES if table in tables]
        if await conn.fetchval(_VIEW_EXISTS_SQL, schema_name, READ_VIEW):
            relations.append(READ_VIEW)
        return relations

    relations: list[str] = await backend.execute_read(cast(Any, _installed), access_scope=MAINTENANCE)
    if relations:
        raise ConfigurationError(
            'ENABLE_POSTGRESQL_RLS=false, but this database carries the row-level security policies and read view '
            f'this server installed (tables: {", ".join(relations)}). Starting with the flag off would leave a '
            'non-owner request role reading only public rows. Either set ENABLE_POSTGRESQL_RLS=true again, or remove '
            'them first: stop every server instance, run `mcp-context-server-migrate --disable-rls --source-url '
            '<table-owner connection>`, then start with the flag off (procedure per deployment: '
            "docs/database-backends.md, 'Disabling row-level security').",
        )


def _drops(state: RlsState, keys: tuple[ObjectKey, ...] | list[ObjectKey]) -> list[str]:
    """Return the drop of each of ``keys`` that ``state`` holds, in order."""
    return [key.drop_statement for key in keys if key in state.objects]
