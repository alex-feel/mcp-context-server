"""PostgreSQL row-level security: keeping the objects :mod:`app.migrations.rls_ddl` renders installed and current.

With ``ENABLE_POSTGRESQL_RLS`` on a PostgreSQL backend, :func:`apply_rls` runs as the
table owner and brings the database to what the code renders: the helper functions,
created under a search path of pg_catalog alone so the names in their bodies are bound
there, and the policy role's EXECUTE on them, the guard trigger on ``context_entries``,
the policies of every managed table that exists, ``ENABLE ROW LEVEL SECURITY`` on each
such table where it is off, and last the read view of ``context_entries`` (:func:`create_read_view`)
with SELECT for the policy role as the only privilege anyone but its owner holds on it.
Every object carries a fingerprint comment built from the ``sha`` of its rendered
statement and the digest of the server's own deparse (:func:`read_definition`), read back
in the same transaction; each table's read policy also records whether row-level
security was on before the migration first enabled it.

The step uses double-checked locking. It first reads the catalog without any lock
(:func:`~app.migrations.rls_state.read_rls_state`); when nothing differs it returns with
no lock taken and no DDL issued, which is what rolling restarts of several instances
need. Otherwise it takes the schema-init lock, reads again and changes only what still
differs (:func:`~app.migrations.rls_state.plan_rls`), naming in a WARNING every object
that was changed outside the server and every grantee whose privileges on the view it
takes away. Tables created after a boot get their policies at the next one. The FTS
language rebuild, which must drop the view to drop a column, shares
:func:`drop_read_view` and :func:`create_read_view`.
"""

import logging
from typing import Any
from typing import cast

import asyncpg

from app.access_scope import READ_VIEW
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.errors import format_exception_message
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations.rls_ddl import CONTEXT_TABLE
from app.migrations.rls_ddl import PIN_SEARCH_PATH
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import read_view_drop
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_ddl import reject_trigger
from app.migrations.rls_ddl import restore_search_path
from app.migrations.rls_state import COLUMNS_SQL
from app.migrations.rls_state import RlsPlan
from app.migrations.rls_state import plan_rls
from app.migrations.rls_state import policy_definition
from app.migrations.rls_state import read_rls_state
from app.migrations.rls_state import read_view_acl
from app.migrations.rls_state import view_acl_repair
from app.settings import AppSettings

logger = logging.getLogger(__name__)

_SEARCH_PATH_SQL = "SELECT pg_catalog.current_setting('search_path')"
_FUNCTION_DEFINITION_SQL = 'SELECT pg_catalog.pg_get_functiondef(pg_catalog.to_regprocedure($1))'
_VIEW_DEFINITION_SQL = 'SELECT pg_catalog.pg_get_viewdef(pg_catalog.to_regclass($1), true)'
_TRIGGER_DEFINITION_SQL = (
    'SELECT pg_catalog.pg_get_triggerdef(t.oid) FROM pg_catalog.pg_trigger t '
    'WHERE t.tgrelid = pg_catalog.to_regclass($1) AND t.tgname = $2'
)
_POLICY_DEFINITION_SQL = (
    'SELECT p.polcmd::text, p.polpermissive, '
    'ARRAY(SELECT pg_catalog.pg_get_userbyid(r) FROM pg_catalog.unnest(p.polroles) AS r ORDER BY 1), '
    'pg_catalog.pg_get_expr(p.polqual, p.polrelid), pg_catalog.pg_get_expr(p.polwithcheck, p.polrelid) '
    'FROM pg_catalog.pg_policy p WHERE p.polrelid = pg_catalog.to_regclass($1) AND p.polname = $2'
)


async def apply_rls(backend: StorageBackend, settings: AppSettings) -> None:
    """Bring the row-level security helpers, guard trigger, policies and read view to what the code renders.

    Runs only on PostgreSQL with ``ENABLE_POSTGRESQL_RLS``; otherwise it returns without
    touching the database. It must run as the table owner, after every table this boot
    provisions exists: a table created later gets its policies on the next boot, and the
    read view lists the columns ``context_entries`` has when it runs. A database that
    already holds exactly what the code renders is only read: no lock, no DDL.

    Args:
        backend: The maintenance backend, logged in as the table owner.
        settings: The application settings: the schema, the policy role and the migration
            budget.

    Raises:
        RuntimeError: If reading the catalog or changing an object fails.
    """
    if backend.backend_type != 'postgresql' or not settings.access_control.postgresql_rls_enabled:
        return

    schema_name = settings.storage.postgresql_schema
    policy_role = settings.access_control.postgresql_rls_policy_role
    schema = quote_pg_identifier(schema_name)
    role = quote_pg_identifier(policy_role)
    timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _apply(conn: asyncpg.Connection) -> bool:
        state = await read_rls_state(conn, schema_name=schema_name, role=role)
        if plan_rls(state, schema=schema, role_name=policy_role).clean:
            return False
        await begin_migration(conn, timeout_s)
        state = await read_rls_state(conn, schema_name=schema_name, role=role)
        plan = plan_rls(state, schema=schema, role_name=policy_role)
        await _repair(conn, plan, schema_name=schema_name, policy_role=policy_role, timeout_s=timeout_s)
        return not plan.clean

    try:
        changed = await backend.execute_write(cast(Any, _apply), access_scope=MAINTENANCE)
    except Exception as e:
        logger.error(f'Failed to apply the row-level security migration: {e}')
        raise RuntimeError(f'Row-level security migration failed: {format_exception_message(e)}') from e
    if changed:
        logger.info(f'Applied row-level security for the role {role} in schema {schema}')
    else:
        logger.debug(f'Row-level security for the role {role} in schema {schema} is current')


async def create_read_view(conn: asyncpg.Connection, *, schema: str, policy_role: str, timeout_s: float) -> None:
    """Create the read view of ``context_entries`` with SELECT for the policy role alone, read-only and fingerprinted.

    The view lists the columns ``context_entries`` has in this transaction, in attribute
    order. Default privileges may hand others privileges on the new view the moment it is
    created; every privilege anyone but its owner holds is revoked except the policy role's
    SELECT, which is granted when missing, and one WARNING names each revoked grantee and
    privilege. The trigger that refuses writes through the view and the fingerprint
    comments of the view, the function and the trigger follow. It runs inside the caller's
    migration transaction, as the table owner, and the view must not exist yet.

    Args:
        conn: The migration transaction's connection, logged in as the table owner.
        schema: The schema of the tables.
        policy_role: The NOLOGIN role the policies and the view's membership test name.
        timeout_s: The server-side budget of each statement, in seconds.
    """
    quoted_schema = quote_pg_identifier(schema)
    columns = [
        quote_pg_identifier(str(row[0])) for row in await conn.fetch(COLUMNS_SQL, f'{quoted_schema}.{CONTEXT_TABLE}')
    ]
    view = read_view(columns, schema=quoted_schema, role_name=policy_role)
    await execute_migration_ddl(conn, view.statement, timeout_s)
    await _normalize_view_acl(conn, schema=quoted_schema, policy_role=policy_role, timeout_s=timeout_s)
    function = reject_function(quoted_schema)
    trigger = reject_trigger(quoted_schema)
    await execute_migration_ddl(conn, function.statement, timeout_s)
    await execute_migration_ddl(conn, trigger.statement, timeout_s)
    for obj in (function, trigger, view):
        await _fingerprint(conn, obj, timeout_s, preenabled=None)


async def drop_read_view(conn: asyncpg.Connection, *, schema: str, policy_role: str, timeout_s: float) -> None:
    """Drop the read view of ``context_entries`` when it exists, naming every grant the drop takes away.

    The drop takes every privilege on the view along with it. One WARNING names each
    grantee that held more than the policy role's SELECT, which :func:`create_read_view`
    grants again, with what it held. It runs inside the caller's migration transaction, as
    the table owner.

    Args:
        conn: The migration transaction's connection, logged in as the table owner.
        schema: The schema of the tables.
        policy_role: The NOLOGIN role the policies and the view's membership test name.
        timeout_s: The server-side budget of the statement, in seconds.
    """
    quoted_schema = quote_pg_identifier(schema)
    acl = await read_view_acl(conn, f'{quoted_schema}.{READ_VIEW}')
    taken = view_acl_repair(acl, schema=quoted_schema, role_name=policy_role)
    await execute_migration_ddl(conn, read_view_drop(quoted_schema), timeout_s)
    _warn_revoked(schema=quoted_schema, policy_role=policy_role, revoked=taken.revoked)


async def read_definition(conn: asyncpg.Connection, obj: RlsObject) -> str:
    """Read the server's own deparse of ``obj``, the text its ``def`` fingerprint digests.

    A function reads as ``pg_get_functiondef``, the view as its pretty-printed
    ``pg_get_viewdef`` and a trigger as ``pg_get_triggerdef``; a policy reads as its
    command, whether it is permissive, its roles and its USING and WITH CHECK expressions as
    ``pg_get_expr`` renders them, one per line.

    Args:
        conn: A connection to the database holding the object.
        obj: The object.

    Returns:
        The deparsed definition.

    Raises:
        LookupError: If the object does not exist.
    """
    if obj.kind is RlsObjectKind.POLICY:
        row = await conn.fetchrow(_POLICY_DEFINITION_SQL, obj.anchor, obj.name)
        if row is None:
            raise LookupError(f'{obj.target} does not exist')
        return policy_definition(str(row[0]), bool(row[1]), [str(role) for role in row[2]], row[3], row[4])
    if obj.kind is RlsObjectKind.TRIGGER:
        definition = await conn.fetchval(_TRIGGER_DEFINITION_SQL, obj.anchor, obj.name)
    elif obj.kind is RlsObjectKind.VIEW:
        definition = await conn.fetchval(_VIEW_DEFINITION_SQL, obj.anchor)
    else:
        definition = await conn.fetchval(_FUNCTION_DEFINITION_SQL, obj.anchor)
    if definition is None:
        raise LookupError(f'{obj.target} does not exist')
    return str(definition)


async def _repair(
    conn: asyncpg.Connection, plan: RlsPlan, *, schema_name: str, policy_role: str, timeout_s: float,
) -> None:
    """Change what ``plan`` lists, in dependency order, inside the locked migration transaction."""
    schema = quote_pg_identifier(schema_name)
    for target in plan.drifted:
        logger.warning(f'{target} was changed outside the server; re-creating it')
    if plan.helpers:
        await _create_helpers(conn, plan.helpers, timeout_s)
    if plan.grant_helpers:
        await execute_migration_ddl(conn, helper_grant(schema, quote_pg_identifier(policy_role)), timeout_s)
    for helper in plan.helpers:
        await _fingerprint(conn, helper, timeout_s, preenabled=None)
    if plan.guard_function is not None:
        await execute_migration_ddl(conn, plan.guard_function.statement, timeout_s)
        await _fingerprint(conn, plan.guard_function, timeout_s, preenabled=None)
    if plan.guard_trigger is not None:
        await _recreate(conn, plan.guard_trigger, timeout_s, preenabled=None)
    for repair in plan.tables:
        for policy, preenabled in repair.policies:
            await _recreate(conn, policy, timeout_s, preenabled=preenabled)
        if repair.enable:
            await execute_migration_ddl(conn, enable_statement(repair.table, schema=schema), timeout_s)
    if plan.view is not None:
        # CREATE OR REPLACE VIEW cannot drop or reorder columns, so the view is re-created.
        await drop_read_view(conn, schema=schema_name, policy_role=policy_role, timeout_s=timeout_s)
        await create_read_view(conn, schema=schema_name, policy_role=policy_role, timeout_s=timeout_s)
    elif plan.view_acl:
        await _normalize_view_acl(conn, schema=schema, policy_role=policy_role, timeout_s=timeout_s)


async def _create_helpers(conn: asyncpg.Connection, helpers: tuple[RlsObject, ...], timeout_s: float) -> None:
    """Create or replace ``helpers`` with every name in their bodies resolved in pg_catalog, then restore the search path.

    Their SQL-standard bodies are bound when they are created, so the pinned search path
    decides what they call for good. The connection's own search path is set back for the
    rest of the transaction, under which the fingerprints read every deparse.
    """
    search_path = str(await conn.fetchval(_SEARCH_PATH_SQL))
    await execute_migration_ddl(conn, PIN_SEARCH_PATH, timeout_s)
    for helper in helpers:
        await execute_migration_ddl(conn, helper.statement, timeout_s)
    await execute_migration_ddl(conn, restore_search_path(search_path), timeout_s)


async def _normalize_view_acl(conn: asyncpg.Connection, *, schema: str, policy_role: str, timeout_s: float) -> None:
    """Leave the policy role's SELECT as the only privilege anyone but the owner holds on the read view.

    Every other grantee, PUBLIC included, loses everything; the policy role loses every
    privilege but SELECT and gets SELECT when it lacks it. One WARNING names each revoked
    grantee with the privileges it held.
    """
    acl = await read_view_acl(conn, f'{schema}.{READ_VIEW}')
    repair = view_acl_repair(acl, schema=schema, role_name=policy_role)
    for statement in repair.statements:
        await execute_migration_ddl(conn, statement, timeout_s)
    _warn_revoked(schema=schema, policy_role=policy_role, revoked=repair.revoked)


def _warn_revoked(*, schema: str, policy_role: str, revoked: list[str]) -> None:
    """Log one WARNING naming each grantee that lost privileges on the read view, if any did."""
    if revoked:
        logger.warning(
            f'Revoked privileges on {schema}.{READ_VIEW}, where only the policy role {policy_role} may hold SELECT: '
            f'{"; ".join(revoked)}',
        )


async def _fingerprint(conn: asyncpg.Connection, obj: RlsObject, timeout_s: float, *, preenabled: bool | None) -> None:
    """Attach ``obj``'s fingerprint, reading its deparse back first."""
    definition = digest(await read_definition(conn, obj))
    await execute_migration_ddl(conn, comment_statement(obj, definition=definition, preenabled=preenabled), timeout_s)


async def _recreate(conn: asyncpg.Connection, obj: RlsObject, timeout_s: float, *, preenabled: bool | None) -> None:
    """Drop ``obj`` when it exists, create it and attach its fingerprint."""
    await execute_migration_ddl(conn, obj.drop_statement, timeout_s)
    await execute_migration_ddl(conn, obj.statement, timeout_s)
    await _fingerprint(conn, obj, timeout_s, preenabled=preenabled)
