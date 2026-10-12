"""The row-level security objects a database holds, and what bringing them to what the code renders takes.

:func:`read_rls_state` reads, in one connection and without any lock or write, every
object :mod:`app.migrations.rls_ddl` renders that the schema holds: each with its parsed
fingerprint comment and the digest of the server's own deparse, plus whether row-level
security is on for each managed table, the live columns of ``context_entries``, the read
view's owner, options and ACL, and whether the policy role may execute the helpers.
:func:`plan_rls` compares that state with what the code renders and returns exactly what
differs. An object differs when it is missing, when the ``sha`` of its comment names
another rendered statement (this code renders it differently now, for example after the
policy role, the schema or the columns of ``context_entries`` changed), or when it was
changed outside the server: its comment is missing or unreadable, its deparse differs
from the digest recorded right after it was created, it is a disabled trigger, or it is
the read view owned by someone else or carrying options. A database whose state yields
an empty plan needs no lock and no DDL.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from enum import auto
from typing import Final

import asyncpg

from app.access_scope import READ_VIEW
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.migrations.rls_ddl import CONTEXT_TABLE
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import Fingerprint
from app.migrations.rls_ddl import ObjectKey
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import parse_fingerprint
from app.migrations.rls_ddl import policy_names
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import read_view_grant
from app.migrations.rls_ddl import read_view_revoke
from app.migrations.rls_ddl import records_preenabled
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_ddl import reject_trigger
from app.migrations.rls_ddl import table_policies

# $1 schema name, $2 candidate tables: each managed table that exists, with whether row-level security is on.
MANAGED_TABLES_SQL: Final = (
    'SELECT c.relname, c.relrowsecurity FROM pg_catalog.pg_class c '
    'JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    "WHERE n.nspname = $1 AND c.relkind IN ('r', 'p') AND c.relname = ANY($2::text[])"
)
# $1 a quoted, schema-qualified table: its live columns, in the order SELECT * lists them.
COLUMNS_SQL: Final = (
    'SELECT a.attname FROM pg_catalog.pg_attribute a '
    'WHERE a.attrelid = pg_catalog.to_regclass($1) AND a.attnum > 0 AND NOT a.attisdropped '
    'ORDER BY a.attnum'
)
# $1 a quoted, schema-qualified relation: every privilege a role other than its owner holds; PUBLIC reads as NULL.
ACL_SQL: Final = (
    'SELECT r.rolname, a.privilege_type FROM pg_catalog.pg_class c '
    'CROSS JOIN LATERAL pg_catalog.aclexplode(c.relacl) AS a '
    'LEFT JOIN pg_catalog.pg_roles r ON r.oid = a.grantee '
    'WHERE c.oid = pg_catalog.to_regclass($1) AND a.grantee <> c.relowner '
    'ORDER BY r.rolname NULLS FIRST, a.privilege_type'
)
# $1 schema name, $2 tables: every policy on them, as read_definition deparses it.
_POLICIES_SQL: Final = (
    "SELECT c.relname, p.polname, pg_catalog.obj_description(p.oid, 'pg_policy'), p.polcmd::text, p.polpermissive, "
    'ARRAY(SELECT pg_catalog.pg_get_userbyid(r) FROM pg_catalog.unnest(p.polroles) AS r ORDER BY 1), '
    'pg_catalog.pg_get_expr(p.polqual, p.polrelid), pg_catalog.pg_get_expr(p.polwithcheck, p.polrelid) '
    'FROM pg_catalog.pg_policy p JOIN pg_catalog.pg_class c ON c.oid = p.polrelid '
    'JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    'WHERE n.nspname = $1 AND c.relname = ANY($2::text[])'
)
# $1 function signatures, $2 the quoted policy role or NULL: each function that exists, with
# whether the role may execute it.
_FUNCTIONS_SQL: Final = (
    "SELECT s.signature, pg_catalog.obj_description(f.oid, 'pg_proc'), pg_catalog.pg_get_functiondef(f.oid), "
    'CASE WHEN $2::text IS NULL OR pg_catalog.to_regrole($2) IS NULL THEN false '
    "ELSE pg_catalog.has_function_privilege(pg_catalog.to_regrole($2)::oid, f.oid, 'EXECUTE') END "
    'FROM pg_catalog.unnest($1::text[]) AS s(signature) '
    'CROSS JOIN LATERAL (SELECT pg_catalog.to_regprocedure(s.signature)::oid AS oid) AS f '
    'WHERE f.oid IS NOT NULL'
)
# $1 the quoted, schema-qualified relation, $2 the trigger name.
_TRIGGER_SQL: Final = (
    "SELECT pg_catalog.obj_description(t.oid, 'pg_trigger'), pg_catalog.pg_get_triggerdef(t.oid), t.tgenabled <> 'D' "
    'FROM pg_catalog.pg_trigger t WHERE t.tgrelid = pg_catalog.to_regclass($1) AND t.tgname = $2'
)
# $1 the quoted, schema-qualified view, $2 the quoted, schema-qualified context table.
_VIEW_SQL: Final = (
    "SELECT pg_catalog.obj_description(v.oid, 'pg_class'), pg_catalog.pg_get_viewdef(v.oid, true), "
    'v.relowner = t.relowner, v.reloptions IS NULL '
    'FROM pg_catalog.pg_class v LEFT JOIN pg_catalog.pg_class t ON t.oid = pg_catalog.to_regclass($2) '
    "WHERE v.oid = pg_catalog.to_regclass($1) AND v.relkind = 'v'"
)


@dataclass(frozen=True, slots=True)
class InstalledObject:
    """One object of ours as the catalog holds it.

    Attributes:
        fingerprint: Its parsed comment; None when it has none or another one.
        definition: The digest of the server's current deparse of it.
        enabled: False for a disabled trigger; True otherwise.
    """

    fingerprint: Fingerprint | None
    definition: str
    enabled: bool


@dataclass(frozen=True, slots=True)
class InstalledView:
    """What the catalog holds about the read view beyond its definition.

    Attributes:
        owned_by_table_owner: Whether the owner of ``context_entries`` owns it.
        plain: Whether it carries no option, such as ``security_barrier``.
        acl: Each role but its owner that holds a privilege on it, None standing for
            PUBLIC, with those privileges.
    """

    owned_by_table_owner: bool
    plain: bool
    acl: Mapping[str | None, tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class RlsState:
    """The row-level security objects of one schema, as one read of the catalog found them.

    Attributes:
        tables: Each managed table that exists, with whether row-level security is on.
        objects: Each object of ours that exists, by key.
        columns: The live columns of ``context_entries``, quoted, in attribute order;
            empty when the table does not exist.
        view: The read view's ownership, options and ACL; None when it does not exist.
        helpers_executable: Whether the policy role may execute all three helpers.
    """

    tables: Mapping[str, bool]
    objects: Mapping[ObjectKey, InstalledObject]
    columns: tuple[str, ...]
    view: InstalledView | None
    helpers_executable: bool


@dataclass(frozen=True, slots=True)
class TableRepair:
    """What one managed table needs.

    Attributes:
        table: The table.
        policies: Each policy to re-create, with the ``preenabled`` value its comment
            records: a bool for the read policy, None for the others.
        enable: Whether row-level security must be enabled on the table.
    """

    table: str
    policies: tuple[tuple[RlsObject, bool | None], ...]
    enable: bool


@dataclass(frozen=True, slots=True)
class RlsPlan:
    """Everything that differs from what the code renders, in the order the migration repairs it.

    Attributes:
        helpers: The helper functions to create or replace.
        grant_helpers: Whether the policy role must be granted EXECUTE on the helpers.
        guard_function: The guard trigger function to create or replace, if it differs.
        guard_trigger: The guard trigger to re-create, if it differs.
        tables: Each managed table that needs a policy re-created or row-level security
            enabled.
        view: The read view to re-create with its write rejection, if any of them differs.
        view_acl: Whether the ACL of the otherwise unchanged view must be repaired.
        drifted: The targets of the objects changed outside the server, which the
            migration names in a WARNING when it re-creates them.
    """

    helpers: tuple[RlsObject, ...]
    grant_helpers: bool
    guard_function: RlsObject | None
    guard_trigger: RlsObject | None
    tables: tuple[TableRepair, ...]
    view: RlsObject | None
    view_acl: bool
    drifted: tuple[str, ...]

    @property
    def clean(self) -> bool:
        """Whether nothing differs, so the migration needs no lock and no DDL."""
        return not (
            self.helpers or self.grant_helpers or self.guard_function or self.guard_trigger
            or self.tables or self.view or self.view_acl
        )


@dataclass(frozen=True, slots=True)
class ViewAclRepair:
    """The statements that leave the policy role's SELECT as the only privilege on the read view.

    Attributes:
        statements: The REVOKE and GRANT statements, in order; empty when the ACL is right.
        revoked: Each grantee that loses something, with what it loses, for the WARNING.
    """

    statements: list[str]
    revoked: list[str]


class _Status(Enum):
    """How an installed object compares with what the code renders."""

    CURRENT = auto()
    MISSING = auto()
    STALE = auto()
    DRIFTED = auto()


def policy_definition(command: str, permissive: bool, roles: list[str], using: str | None, check: str | None) -> str:
    """Return the text a policy's ``def`` fingerprint digests.

    Args:
        command: The policy's command code, as ``pg_policy.polcmd`` reads.
        permissive: Whether the policy is permissive.
        roles: The roles it applies to, by name, in name order.
        using: Its USING expression as ``pg_get_expr`` renders it, if any.
        check: Its WITH CHECK expression as ``pg_get_expr`` renders it, if any.

    Returns:
        The five parts, one per line.
    """
    return '\n'.join((command, 'permissive' if permissive else 'restrictive', ','.join(roles), using or '', check or ''))


def view_acl_repair(acl: Mapping[str | None, tuple[str, ...]], *, schema: str, role_name: str) -> ViewAclRepair:
    """Return what leaves SELECT for the policy role as the only privilege anyone but the owner holds on the view.

    Every other grantee, PUBLIC included, loses everything; the policy role loses every
    privilege but SELECT and gets SELECT when it lacks it.

    Args:
        acl: Each grantee but the owner, None standing for PUBLIC, with its privileges.
        schema: The schema of the tables, quoted.
        role_name: The policy role's own name.

    Returns:
        The statements and the revoked grantees.
    """
    role = quote_pg_identifier(role_name)
    statements: list[str] = []
    revoked: list[str] = []
    for grantee, privileges in acl.items():
        if grantee == role_name:
            continue
        statements.append(read_view_revoke(schema, 'PUBLIC' if grantee is None else quote_pg_identifier(grantee), ('ALL',)))
        revoked.append(f'{"PUBLIC" if grantee is None else grantee} ({", ".join(privileges)})')
    held = acl.get(role_name, ())
    beyond_select = [privilege for privilege in held if privilege != 'SELECT']
    if beyond_select:
        statements.append(read_view_revoke(schema, role, beyond_select))
        revoked.append(f'{role_name} ({", ".join(beyond_select)})')
    if 'SELECT' not in held:
        statements.append(read_view_grant(schema, role))
    return ViewAclRepair(statements=statements, revoked=revoked)


async def read_view_acl(conn: asyncpg.Connection, view: str) -> dict[str | None, tuple[str, ...]]:
    """Return each grantee but the owner of ``view``, None standing for PUBLIC, with the privileges it holds.

    Args:
        conn: A connection to the database.
        view: The view, quoted and schema-qualified.

    Returns:
        The ACL; empty when only the owner holds privileges.
    """
    held: dict[str | None, list[str]] = {}
    for row in await conn.fetch(ACL_SQL, view):
        held.setdefault(None if row[0] is None else str(row[0]), []).append(str(row[1]))
    return {grantee: tuple(privileges) for grantee, privileges in held.items()}


async def read_rls_state(conn: asyncpg.Connection, *, schema_name: str, role: str | None) -> RlsState:
    """Read the row-level security objects of ``schema_name`` and what they depend on.

    Every read is a catalog query; nothing is locked or written.

    Args:
        conn: A connection to the database, normally the table owner's.
        schema_name: The schema of the tables, unquoted.
        role: The policy role, quoted, whose EXECUTE on the helpers is read; None when
            no role matters, which reads it as missing.

    Returns:
        The state.
    """
    schema = quote_pg_identifier(schema_name)
    tables = {str(row[0]): bool(row[1]) for row in await conn.fetch(MANAGED_TABLES_SQL, schema_name, list(MANAGED_TABLES))}
    objects = await _read_policies(conn, schema_name=schema_name, tables=list(tables))
    helpers = helper_functions(schema)
    functions, executable = await _read_functions(
        conn, (*helpers, guard_function(schema), reject_function(schema)), role=role,
    )
    objects.update(functions)
    for trigger in (guard_trigger(schema), reject_trigger(schema)):
        trigger_row = await conn.fetchrow(_TRIGGER_SQL, trigger.anchor, trigger.name)
        if trigger_row is not None:
            objects[trigger.key] = InstalledObject(
                fingerprint=parse_fingerprint(trigger_row[0]),
                definition=digest(str(trigger_row[1])),
                enabled=bool(trigger_row[2]),
            )

    view_anchor = f'{schema}.{READ_VIEW}'
    view: InstalledView | None = None
    view_row = await conn.fetchrow(_VIEW_SQL, view_anchor, f'{schema}.{CONTEXT_TABLE}')
    if view_row is not None:
        objects[ObjectKey(RlsObjectKind.VIEW, READ_VIEW, view_anchor)] = InstalledObject(
            fingerprint=parse_fingerprint(view_row[0]), definition=digest(str(view_row[1])), enabled=True,
        )
        view = InstalledView(
            owned_by_table_owner=bool(view_row[2]), plain=bool(view_row[3]), acl=await read_view_acl(conn, view_anchor),
        )

    columns: tuple[str, ...] = ()
    if CONTEXT_TABLE in tables:
        column_rows = await conn.fetch(COLUMNS_SQL, f'{schema}.{CONTEXT_TABLE}')
        columns = tuple(quote_pg_identifier(str(row[0])) for row in column_rows)
    return RlsState(
        tables=tables,
        objects=objects,
        columns=columns,
        view=view,
        helpers_executable=all(helper.key in executable for helper in helpers),
    )


def plan_rls(state: RlsState, *, schema: str, role_name: str) -> RlsPlan:
    """Return what differs between ``state`` and what the code renders for the policy role.

    Args:
        state: What the catalog holds.
        schema: The schema of the tables, quoted.
        role_name: The policy role's own name.

    Returns:
        The plan; empty when nothing differs.
    """
    role = quote_pg_identifier(role_name)
    drifted: list[str] = []

    def _differs(obj: RlsObject, status: _Status) -> bool:
        if status is _Status.DRIFTED:
            drifted.append(obj.target)
        return status is not _Status.CURRENT

    helpers = tuple(helper for helper in helper_functions(schema) if _differs(helper, _status(state, helper)))
    function = guard_function(schema)
    guard = function if _differs(function, _status(state, function)) else None
    trigger = guard_trigger(schema)
    has_context_table = CONTEXT_TABLE in state.tables
    guard_on_entries = trigger if has_context_table and _differs(trigger, _status(state, trigger)) else None

    repairs: list[TableRepair] = []
    for table in (table for table in MANAGED_TABLES if table in state.tables):
        policies = tuple(
            (policy, _preenabled(state, policy, table) if records_preenabled(policy) else None)
            for policy in table_policies(table, schema=schema, role=role)
            if _differs(policy, _status(state, policy))
        )
        if policies or not state.tables[table]:
            repairs.append(TableRepair(table=table, policies=policies, enable=not state.tables[table]))

    view: RlsObject | None = None
    view_acl = False
    if has_context_table:
        expected = read_view(state.columns, schema=schema, role_name=role_name)
        rejection = (reject_function(schema), reject_trigger(schema))
        # Every member of the group is compared, so each one changed outside the server is named.
        differences = [_differs(expected, _view_status(state, expected))]
        differences.extend(_differs(obj, _status(state, obj)) for obj in rejection)
        if any(differences):
            view = expected
        elif state.view is not None:
            view_acl = bool(view_acl_repair(state.view.acl, schema=schema, role_name=role_name).statements)

    return RlsPlan(
        helpers=helpers,
        grant_helpers=not state.helpers_executable,
        guard_function=guard,
        guard_trigger=guard_on_entries,
        tables=tuple(repairs),
        view=view,
        view_acl=view_acl,
        drifted=tuple(drifted),
    )


async def _read_policies(conn: asyncpg.Connection, *, schema_name: str, tables: list[str]) -> dict[ObjectKey, InstalledObject]:
    """Read every policy of ours on ``tables``, by key."""
    schema = quote_pg_identifier(schema_name)
    objects: dict[ObjectKey, InstalledObject] = {}
    for row in await conn.fetch(_POLICIES_SQL, schema_name, tables):
        table, name = str(row[0]), str(row[1])
        if name not in policy_names(table):
            continue
        definition = policy_definition(str(row[3]), bool(row[4]), [str(role) for role in row[5]], row[6], row[7])
        objects[ObjectKey(RlsObjectKind.POLICY, name, f'{schema}.{table}')] = InstalledObject(
            fingerprint=parse_fingerprint(row[2]), definition=digest(definition), enabled=True,
        )
    return objects


async def _read_functions(
    conn: asyncpg.Connection, functions: tuple[RlsObject, ...], *, role: str | None,
) -> tuple[dict[ObjectKey, InstalledObject], set[ObjectKey]]:
    """Read each of ``functions`` that exists, and which of them the policy role may execute."""
    by_anchor = {obj.anchor: obj for obj in functions}
    objects: dict[ObjectKey, InstalledObject] = {}
    executable: set[ObjectKey] = set()
    for row in await conn.fetch(_FUNCTIONS_SQL, list(by_anchor), role):
        obj = by_anchor[str(row[0])]
        objects[obj.key] = InstalledObject(fingerprint=parse_fingerprint(row[1]), definition=digest(str(row[2])), enabled=True)
        if row[3]:
            executable.add(obj.key)
    return objects, executable


def _status(state: RlsState, obj: RlsObject) -> _Status:
    """Compare the installed ``obj`` with its rendering."""
    installed = state.objects.get(obj.key)
    if installed is None:
        return _Status.MISSING
    fingerprint = installed.fingerprint
    if fingerprint is not None and fingerprint.sha != obj.sha:
        return _Status.STALE
    if (
        fingerprint is None
        or fingerprint.definition != installed.definition
        or not installed.enabled
        or (records_preenabled(obj) and fingerprint.preenabled is None)
    ):
        return _Status.DRIFTED
    return _Status.CURRENT


def _view_status(state: RlsState, view: RlsObject) -> _Status:
    """Compare the read view with its rendering, its ownership and its options included."""
    status = _status(state, view)
    if status is _Status.CURRENT and state.view is not None and not (state.view.owned_by_table_owner and state.view.plain):
        return _Status.DRIFTED
    return status


def _preenabled(state: RlsState, read_policy: RlsObject, table: str) -> bool:
    """Return the ``preenabled`` value a re-created read policy records.

    The value the installed policy's comment recorded is kept. Without one it is whether
    row-level security is on right now, before the migration enables it, so a table found
    enabled with no readable record of ours counts as enabled before the migration.

    Returns:
        The value to record.
    """
    installed = state.objects.get(read_policy.key)
    if installed is not None and installed.fingerprint is not None and installed.fingerprint.preenabled is not None:
        return installed.fingerprint.preenabled
    return state.tables[table]
