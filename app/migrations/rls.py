"""PostgreSQL row-level security: installing the objects :mod:`app.migrations.rls_ddl` renders.

With ``ENABLE_POSTGRESQL_RLS`` on a PostgreSQL backend, :func:`apply_rls` runs as the table
owner and installs, in one transaction under the schema-init advisory lock: the helper
functions and the policy role's EXECUTE on them, the guard trigger on ``context_entries``,
the policies of every managed table that exists, ``ENABLE ROW LEVEL SECURITY`` on each
such table where it is still off, and last the read view of ``context_entries``
(:func:`create_read_view`). Every object then gets its fingerprint comment, built from the
``sha`` of its rendered statement and the digest of the server's own deparse of the object
(:func:`read_definition`), read back in the same transaction.
"""

import logging
from typing import Any
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.errors import format_exception_message
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations.rls_ddl import CONTEXT_TABLE
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import read_view_drop
from app.migrations.rls_ddl import read_view_grant
from app.migrations.rls_ddl import read_view_revoke
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_ddl import reject_trigger
from app.migrations.rls_ddl import table_policies
from app.settings import AppSettings

logger = logging.getLogger(__name__)

# The managed tables that exist in the schema, with whether row-level security is enabled on each.
_MANAGED_TABLES_SQL = (
    'SELECT c.relname, c.relrowsecurity FROM pg_catalog.pg_class c '
    'JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    "WHERE n.nspname = $1 AND c.relkind IN ('r', 'p') AND c.relname = ANY($2::text[])"
)
# The live columns of a table, in the order SELECT * lists them.
_COLUMNS_SQL = (
    'SELECT a.attname FROM pg_catalog.pg_attribute a '
    'WHERE a.attrelid = pg_catalog.to_regclass($1) AND a.attnum > 0 AND NOT a.attisdropped '
    'ORDER BY a.attnum'
)
# Every privilege a relation's ACL holds for a role other than its owner; PUBLIC reads as NULL.
_ACL_SQL = (
    'SELECT r.rolname, a.privilege_type FROM pg_catalog.pg_class c '
    'CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) AS a '
    'LEFT JOIN pg_catalog.pg_roles r ON r.oid = a.grantee '
    'WHERE c.oid = pg_catalog.to_regclass($1) AND a.grantee <> c.relowner '
    'ORDER BY r.rolname NULLS FIRST, a.privilege_type'
)
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
    """Install the row-level security helpers, guard trigger, policies and read view with their fingerprints.

    Runs only on PostgreSQL with ``ENABLE_POSTGRESQL_RLS``; otherwise it returns without
    touching the database. It must run as the table owner, after every table this boot
    provisions exists: a table created later gets its policies on the next boot, and the
    read view lists the columns ``context_entries`` has when it runs.

    Args:
        backend: The maintenance backend, logged in as the table owner.
        settings: The application settings: the schema, the policy role and the migration
            budget.

    Raises:
        RuntimeError: If installing an object fails.
    """
    if backend.backend_type != 'postgresql' or not settings.access_control.postgresql_rls_enabled:
        return

    schema_name = settings.storage.postgresql_schema
    policy_role = settings.access_control.postgresql_rls_policy_role
    schema = quote_pg_identifier(schema_name)
    role = quote_pg_identifier(policy_role)
    timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _apply(conn: asyncpg.Connection) -> None:
        await begin_migration(conn, timeout_s)
        helpers = helper_functions(schema)
        for helper in helpers:
            await execute_migration_ddl(conn, helper.statement, timeout_s)
        await execute_migration_ddl(conn, helper_grant(schema, role), timeout_s)
        for helper in helpers:
            await _fingerprint(conn, helper, timeout_s)

        function = guard_function(schema)
        await execute_migration_ddl(conn, function.statement, timeout_s)
        await _fingerprint(conn, function, timeout_s)
        await _recreate(conn, guard_trigger(schema), timeout_s)

        rows = await conn.fetch(_MANAGED_TABLES_SQL, schema_name, list(MANAGED_TABLES))
        enabled = {str(row[0]): bool(row[1]) for row in rows}
        for table in (table for table in MANAGED_TABLES if table in enabled):
            for policy in table_policies(table, schema=schema, role=role):
                await _recreate(conn, policy, timeout_s)
            if not enabled[table]:
                await execute_migration_ddl(conn, enable_statement(table, schema=schema), timeout_s)

        # CREATE OR REPLACE VIEW cannot drop or reorder columns, so the view is re-created.
        if CONTEXT_TABLE in enabled:
            await execute_migration_ddl(conn, read_view_drop(schema), timeout_s)
            await create_read_view(conn, schema=schema_name, policy_role=policy_role, timeout_s=timeout_s)

    try:
        await backend.execute_write(cast(Any, _apply), access_scope=MAINTENANCE)
    except Exception as e:
        logger.error(f'Failed to apply the row-level security migration: {e}')
        raise RuntimeError(f'Row-level security migration failed: {format_exception_message(e)}') from e
    logger.info(f'Applied row-level security policies for the role {role} in schema {schema}')


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
        quote_pg_identifier(str(row[0])) for row in await conn.fetch(_COLUMNS_SQL, f'{quoted_schema}.{CONTEXT_TABLE}')
    ]
    view = read_view(columns, schema=quoted_schema, role_name=policy_role)
    await execute_migration_ddl(conn, view.statement, timeout_s)
    await _normalize_view_acl(conn, view, schema=quoted_schema, policy_role=policy_role, timeout_s=timeout_s)
    function = reject_function(quoted_schema)
    trigger = reject_trigger(quoted_schema)
    await execute_migration_ddl(conn, function.statement, timeout_s)
    await execute_migration_ddl(conn, trigger.statement, timeout_s)
    for obj in (function, trigger, view):
        await _fingerprint(conn, obj, timeout_s)


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
        command, permissive, roles, using, check = row[0], row[1], row[2], row[3], row[4]
        return '\n'.join((
            str(command), 'permissive' if permissive else 'restrictive', ','.join(roles), using or '', check or '',
        ))
    if obj.kind is RlsObjectKind.TRIGGER:
        definition = await conn.fetchval(_TRIGGER_DEFINITION_SQL, obj.anchor, obj.name)
    elif obj.kind is RlsObjectKind.VIEW:
        definition = await conn.fetchval(_VIEW_DEFINITION_SQL, obj.anchor)
    else:
        definition = await conn.fetchval(_FUNCTION_DEFINITION_SQL, obj.anchor)
    if definition is None:
        raise LookupError(f'{obj.target} does not exist')
    return str(definition)


async def _normalize_view_acl(
    conn: asyncpg.Connection, view: RlsObject, *, schema: str, policy_role: str, timeout_s: float,
) -> None:
    """Leave the policy role's SELECT as the only privilege anyone but the owner holds on ``view``.

    Every other grantee, PUBLIC included, loses everything; the policy role loses every
    privilege but SELECT and gets SELECT when it lacks it. One WARNING names each revoked
    grantee with the privileges it held.
    """
    held: dict[str | None, list[str]] = {}
    for row in await conn.fetch(_ACL_SQL, view.anchor):
        held.setdefault(None if row[0] is None else str(row[0]), []).append(str(row[1]))

    role = quote_pg_identifier(policy_role)
    statements: list[str] = []
    revoked: list[str] = []
    for grantee, privileges in held.items():
        if grantee == policy_role:
            continue
        statements.append(read_view_revoke(schema, 'PUBLIC' if grantee is None else quote_pg_identifier(grantee), ('ALL',)))
        revoked.append(f'{"PUBLIC" if grantee is None else grantee} ({", ".join(privileges)})')
    policy_role_privileges = held.get(policy_role, [])
    beyond_select = [privilege for privilege in policy_role_privileges if privilege != 'SELECT']
    if beyond_select:
        statements.append(read_view_revoke(schema, role, beyond_select))
        revoked.append(f'{policy_role} ({", ".join(beyond_select)})')
    if 'SELECT' not in policy_role_privileges:
        statements.append(read_view_grant(schema, role))

    for statement in statements:
        await execute_migration_ddl(conn, statement, timeout_s)
    if revoked:
        logger.warning(
            f'Revoked privileges on {view.anchor}, where only the policy role {policy_role} may hold SELECT: '
            f'{"; ".join(revoked)}',
        )


async def _fingerprint(conn: asyncpg.Connection, obj: RlsObject, timeout_s: float) -> None:
    """Attach ``obj``'s fingerprint, reading its deparse back first."""
    definition = digest(await read_definition(conn, obj))
    await execute_migration_ddl(conn, comment_statement(obj, definition=definition), timeout_s)


async def _recreate(conn: asyncpg.Connection, obj: RlsObject, timeout_s: float) -> None:
    """Drop ``obj`` when it exists, create it and attach its fingerprint."""
    await execute_migration_ddl(conn, obj.drop_statement, timeout_s)
    await execute_migration_ddl(conn, obj.statement, timeout_s)
    await _fingerprint(conn, obj, timeout_s)
