"""The PostgreSQL role checks of the server start, around the database preparation.

Requests log in with the request connection settings; migrations log in as the table
owner through ``POSTGRESQL_MIGRATION_USER`` and ``POSTGRESQL_MIGRATION_PASSWORD``. When a
migration user is configured, :func:`discover_request_role` reads on one connection with
the request connection settings which role requests act as, as the server itself reports
it. With ``ENABLE_POSTGRESQL_RLS`` on, :func:`refuse_unsafe_request_role` refuses before
any DDL a request role that the policies could not hold: a superuser, a role with
BYPASSRLS, or one that is not a member of the policy role (or a policy role that does not
exist). :func:`split_request_role` then compares it with the role the maintenance backend
acts as and refuses, with the flag on, request and migration credentials that resolve to
one role, which only the two ``current_user`` values can show: a pooler-forced login, a
Supabase project-ref user name or ``ALTER ROLE ... SET role`` make different configured
names act as one role.

After the preparation, :func:`verify_rls_posture` runs on the request backend whenever
the roles are split: in either flag state the request role must be a member of the policy
role and hold every privilege the start grants it; with the flag off no managed table may
filter it through row-level security enabled outside the server; with the flag on every
managed table must have row-level security enabled, applying to the request role and not
forced, no policy but ours may apply to the request role, every object of ours must be
the one this code renders, and each read view must be owned by the owner of the table it
lists, carry no options, refuse writes and be readable by the policy role alone. Every refusal is a
:class:`~app.errors.ConfigurationError` (exit 78) that names the fix.
:func:`warn_jwt_without_rls` warns when verified JWT identities reach a PostgreSQL
database that enforces access in the application layer only.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import Final
from typing import cast

import asyncpg

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session import apply_session_gucs
from app.backends.postgresql_backend.session import build_asyncpg_connect_kwargs
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.backends.postgresql_backend.session import request_connection_string
from app.errors import ConfigurationError
from app.errors import DependencyError
from app.migrations.request_role_grants import HeldPrivileges
from app.migrations.request_role_grants import plan_grants
from app.migrations.request_role_grants import read_held_privileges
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import disable_statement
from app.migrations.rls_ddl import enable_statement
from app.migrations.rls_ddl import helper_grant
from app.migrations.rls_ddl import policy_names
from app.migrations.rls_state import RlsPlan
from app.migrations.rls_state import RlsState
from app.migrations.rls_state import plan_rls
from app.migrations.rls_state import read_rls_state
from app.migrations.rls_state import view_acl_repair
from app.migrations.rls_view_ddl import READ_VIEW_TABLES
from app.migrations.rls_view_ddl import REJECT_TRIGGER
from app.migrations.rls_view_ddl import reject_trigger
from app.settings import AppSettings

logger = logging.getLogger(__name__)

DOCS: Final = "docs/database-backends.md, 'Row-level security (PostgreSQL)'"
JWT_WITHOUT_RLS_WARNING: Final = (
    'MCP_AUTH_PROVIDER=jwt on PostgreSQL without ENABLE_POSTGRESQL_RLS: '
    'access isolation is enforced in the application layer only.'
)

# $1 the quoted policy role: the role this connection acts as, its attributes, whether the
# policy role exists and whether this role holds its privileges.
REQUEST_ROLE_SQL: Final = (
    'SELECT current_user, r.rolsuper, r.rolbypassrls, pg_catalog.to_regrole($1) IS NOT NULL, '
    'CASE WHEN pg_catalog.to_regrole($1) IS NULL THEN false '
    "ELSE pg_catalog.pg_has_role(current_user, pg_catalog.to_regrole($1), 'USAGE') END "
    'FROM pg_catalog.pg_roles r WHERE r.rolname = current_user'
)
# $1 schema name, $2 candidate tables: each managed table that exists, whether row-level
# security is enabled, forced and active for this role, and whether this role holds the
# privileges of the table's owner.
TABLE_SECURITY_SQL: Final = (
    'SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity, pg_catalog.row_security_active(c.oid), '
    "pg_catalog.pg_has_role(current_user, c.relowner, 'USAGE') "
    'FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    "WHERE n.nspname = $1 AND c.relkind IN ('r', 'p') AND c.relname = ANY($2::text[])"
)
# $1 schema name, $2 tables: every policy on them that applies to this role: one for
# PUBLIC, or for a role whose privileges this role holds, the test PostgreSQL itself
# applies to decide which policies govern a query.
APPLYING_POLICIES_SQL: Final = (
    'SELECT c.relname, p.polname, p.polpermissive, '
    "ARRAY(SELECT CASE WHEN r = 0::pg_catalog.oid THEN 'PUBLIC' ELSE pg_catalog.pg_get_userbyid(r)::text END "
    'FROM pg_catalog.unnest(p.polroles) AS r ORDER BY 1) '
    'FROM pg_catalog.pg_policy p JOIN pg_catalog.pg_class c ON c.oid = p.polrelid '
    'JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    'WHERE n.nspname = $1 AND c.relname = ANY($2::text[]) '
    'AND (0::pg_catalog.oid = ANY(p.polroles) OR EXISTS (SELECT 1 FROM pg_catalog.unnest(p.polroles) AS r '
    "WHERE r <> 0::pg_catalog.oid AND pg_catalog.pg_has_role(current_user, r, 'USAGE'))) "
    'ORDER BY c.relname, p.polname'
)


@dataclass(frozen=True, slots=True)
class RequestRole:
    """The role the request connection settings act as, as the server reports it.

    Attributes:
        name: Its ``current_user``.
        is_superuser: Whether it is a superuser.
        bypass_rls: Whether it has BYPASSRLS.
        policy_role_exists: Whether the policy role exists.
        policy_member: Whether it holds the privileges of the policy role.
    """

    name: str
    is_superuser: bool
    bypass_rls: bool
    policy_role_exists: bool
    policy_member: bool


@dataclass(frozen=True, slots=True)
class TableSecurity:
    """Row-level security on one managed table, as the request role meets it.

    Attributes:
        enabled: Whether row-level security is enabled on the table.
        forced: Whether it is forced, which subjects the owner to the policies too.
        active: Whether it filters the request role's queries.
        owned: Whether the request role holds the privileges of the table's owner.
    """

    enabled: bool
    forced: bool
    active: bool
    owned: bool


@dataclass(frozen=True, slots=True)
class ApplyingPolicy:
    """A policy that applies to the request role.

    Attributes:
        table: The table it is defined on.
        name: Its name.
        permissive: Whether it is permissive; a restrictive one narrows the access model.
        roles: The roles it names, ``PUBLIC`` for everyone.
    """

    table: str
    name: str
    permissive: bool
    roles: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Posture:
    """What the request backend reports about the request role and the objects it meets.

    Attributes:
        policy_member: Whether the request role holds the privileges of the policy role.
        held: The privileges it holds on what the start grants the policy role.
        tables: Each managed table that exists, with its row-level security.
        applying_policies: The policies on the managed tables that apply to it, ours
            included; read only with row-level security on.
        rls: Our row-level security objects; read only with row-level security on.
    """

    policy_member: bool
    held: HeldPrivileges
    tables: Mapping[str, TableSecurity]
    applying_policies: tuple[ApplyingPolicy, ...]
    rls: RlsState | None


async def discover_request_role(settings: AppSettings) -> RequestRole | None:
    """Read the role the request connection settings act as, on one connection of its own.

    Args:
        settings: The application settings.

    Returns:
        The role; None when no migration user is configured, which makes requests and
        migrations log in alike.

    Raises:
        ConfigurationError: If the request connection settings cannot log in for a reason
            only the configuration can fix.
        DependencyError: If the server cannot be reached.
    """
    if not settings.storage.postgresql_migration_user:
        return None
    try:
        conn = await asyncpg.connect(
            request_connection_string(settings),
            timeout=settings.storage.postgresql_connect_timeout_s,
            **build_asyncpg_connect_kwargs(settings, credentials=NO_CREDENTIAL_OVERRIDE),
        )
        try:
            await apply_session_gucs(conn, settings, profile_statements=())
            row = await conn.fetchrow(
                REQUEST_ROLE_SQL, quote_pg_identifier(settings.access_control.postgresql_rls_policy_role),
            )
        finally:
            await conn.close()
    except (
        asyncpg.exceptions.ClientConfigurationError,
        asyncpg.exceptions.InvalidAuthorizationSpecificationError,
        asyncpg.exceptions.InvalidCatalogNameError,
        asyncpg.exceptions.InsufficientPrivilegeError,
        ValueError,
    ) as e:
        raise ConfigurationError(
            f'Cannot read the request role with the request connection settings: {e}. Check '
            'POSTGRESQL_CONNECTION_STRING, or POSTGRESQL_USER, POSTGRESQL_PASSWORD and POSTGRESQL_DATABASE, '
            'and the server pg_hba.conf rules.',
        ) from e
    except Exception as e:
        raise DependencyError(
            f'PostgreSQL connection failed while reading the request role: {e}. '
            'Ensure PostgreSQL is running and accessible.',
        ) from e
    if row is None:
        raise DependencyError('PostgreSQL reported no role for the request connection settings.')
    return RequestRole(
        name=str(row[0]),
        is_superuser=bool(row[1]),
        bypass_rls=bool(row[2]),
        policy_role_exists=bool(row[3]),
        policy_member=bool(row[4]),
    )


def refuse_unsafe_request_role(request_role: RequestRole | None, settings: AppSettings) -> None:
    """With row-level security on, refuse a request role the policies could not hold.

    Args:
        request_role: The discovered request role; None when no migration user is set.
        settings: The application settings.

    Raises:
        ConfigurationError: If the request role is a superuser, has BYPASSRLS or does not
            hold the privileges of the policy role, or the policy role does not exist.
    """
    if request_role is None or not settings.access_control.postgresql_rls_enabled:
        return
    role = quote_pg_identifier(settings.access_control.postgresql_rls_policy_role)
    login = quote_pg_identifier(request_role.name)
    problems: list[str] = []
    if request_role.is_superuser or request_role.bypass_rls:
        problems.append(
            f'row-level security never filters a superuser or a role with BYPASSRLS ({_attributes(request_role)}); '
            'point the request connection settings at a dedicated login role: '
            f"CREATE ROLE <login> LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS; GRANT {role} TO <login>;",
        )
    if not request_role.policy_role_exists:
        problems.append(
            f'the policy role {role} that POSTGRESQL_RLS_POLICY_ROLE names does not exist; as a role '
            f'administrator, create it and make the request role its member: CREATE ROLE {role} NOLOGIN; '
            f'GRANT {role} TO {login};',
        )
    elif not request_role.policy_member:
        problems.append(f'it is not a member of the policy role {role}, which every policy names: GRANT {role} TO {login};')
    if problems:
        raise ConfigurationError(
            f"ENABLE_POSTGRESQL_RLS=true, but the request role '{request_role.name}' cannot be held by row-level "
            f'security: {" Also, ".join(problems)} ({DOCS}).',
        )


async def read_login_role(backend: StorageBackend) -> str:
    """Return the role the maintenance backend acts as.

    Args:
        backend: The initialized maintenance backend.

    Returns:
        Its ``current_user``.
    """

    async def _current_user(conn: asyncpg.Connection) -> str:
        return str(await conn.fetchval('SELECT current_user'))

    role: str = await backend.execute_read(cast(Any, _current_user), access_scope=MAINTENANCE)
    return role


def split_request_role(
    request_role: RequestRole | None, maintenance_role: str, settings: AppSettings,
) -> RequestRole | None:
    """Return the request role when requests and migrations act as two roles, None otherwise.

    Args:
        request_role: The discovered request role; None when no migration user is set.
        maintenance_role: The role the maintenance backend acts as.
        settings: The application settings.

    Returns:
        ``request_role`` when it differs from ``maintenance_role``; None otherwise.

    Raises:
        ConfigurationError: If row-level security is on and the two are one role, which
            owns the tables and therefore bypasses every policy.
    """
    if request_role is not None and request_role.name != maintenance_role:
        return request_role
    if not settings.access_control.postgresql_rls_enabled:
        return None
    role = quote_pg_identifier(settings.access_control.postgresql_rls_policy_role)
    request_name = maintenance_role if request_role is None else request_role.name
    raise ConfigurationError(
        'ENABLE_POSTGRESQL_RLS=true, but the request and migration credentials resolve to the same PostgreSQL role '
        f"'{maintenance_role}' (request current_user '{request_name}', "
        f"POSTGRESQL_MIGRATION_USER='{settings.storage.postgresql_migration_user}'). The table owner bypasses "
        'row-level security, so nothing would be enforced. Create a separate login role for requests '
        f"(CREATE ROLE {role} NOLOGIN; CREATE ROLE <login> LOGIN PASSWORD '...' NOSUPERUSER NOBYPASSRLS; "
        f'GRANT {role} TO <login>;) and point the request connection settings at it ({DOCS}).',
    )


async def read_posture(conn: asyncpg.Connection, settings: AppSettings) -> Posture:
    """Read what the posture checks judge, as the role ``conn`` acts as.

    Args:
        conn: A connection of the request backend.
        settings: The application settings: the schema, the policy role and the flag.

    Returns:
        The posture; the applying policies and our objects are read only with row-level
        security on.
    """
    schema_name = settings.storage.postgresql_schema
    role = quote_pg_identifier(settings.access_control.postgresql_rls_policy_role)
    login = await conn.fetchrow(REQUEST_ROLE_SQL, role)
    login_name = '' if login is None else str(login[0])
    tables = {
        str(row[0]): TableSecurity(enabled=bool(row[1]), forced=bool(row[2]), active=bool(row[3]), owned=bool(row[4]))
        for row in await conn.fetch(TABLE_SECURITY_SQL, schema_name, list(MANAGED_TABLES))
    }
    held = await read_held_privileges(conn, schema_name=schema_name, role_name=login_name)
    applying: tuple[ApplyingPolicy, ...] = ()
    rls: RlsState | None = None
    if settings.access_control.postgresql_rls_enabled:
        applying = tuple(
            ApplyingPolicy(table=str(row[0]), name=str(row[1]), permissive=bool(row[2]), roles=tuple(str(r) for r in row[3]))
            for row in await conn.fetch(APPLYING_POLICIES_SQL, schema_name, list(tables))
        )
        rls = await read_rls_state(conn, schema_name=schema_name, role=role)
    return Posture(
        policy_member=login is not None and bool(login[4]),
        held=held,
        tables=tables,
        applying_policies=applying,
        rls=rls,
    )


def posture_problems(posture: Posture, settings: AppSettings, *, request_role: str) -> list[str]:
    """Return everything about ``posture`` the server must refuse to start with, each with its fix.

    Args:
        posture: What the request backend reported.
        settings: The application settings: the schema, the policy role and the flag.
        request_role: The request role's name.

    Returns:
        One sentence per problem; empty when the posture is right.
    """
    schema = quote_pg_identifier(settings.storage.postgresql_schema)
    policy_role = settings.access_control.postgresql_rls_policy_role
    role = quote_pg_identifier(policy_role)
    problems: list[str] = []
    if not posture.policy_member:
        problems.append(
            f'It is not a member of the policy role {role}, through which it holds every privilege the start grants. '
            f'Fix: GRANT {role} TO {quote_pg_identifier(request_role)};',
        )
    else:
        missing = plan_grants(posture.held, schema=schema, role=role)
        if missing:
            problems.append(
                f'It lacks privileges the start granted the policy role {role}, so they were revoked since. '
                f'Fix, as the table owner: {"; ".join(missing)};',
            )
    if not settings.access_control.postgresql_rls_enabled:
        active = [table for table, security in posture.tables.items() if security.active]
        if active:
            fixes = ' '.join(f'{disable_statement(table, schema=schema)};' for table in active)
            problems.append(
                f'Row-level security filters it on {", ".join(active)} although ENABLE_POSTGRESQL_RLS=false: it was '
                'enabled outside this server (a dashboard, a restore), so its reads return no rows and its writes fail. '
                f'Fix: {fixes} Or start with ENABLE_POSTGRESQL_RLS=true.',
            )
        return problems
    problems.extend(_table_problems(posture.tables, schema=schema))
    problems.extend(
        _foreign_policy_problem(policy, schema=schema)
        for policy in posture.applying_policies
        if policy.name not in policy_names(policy.table)
    )
    if posture.rls is not None:
        problems.extend(_object_problems(posture.rls, schema=schema, policy_role=policy_role))
    return problems


async def verify_rls_posture(
    backend: StorageBackend, settings: AppSettings, *, request_role: RequestRole, maintenance_role: str,
) -> None:
    """Refuse to serve when the request role or the objects it meets are not what the start prepared.

    Runs on the request backend, so every check sees what requests will see, after the
    database preparation and whenever requests and migrations act as two roles.

    Args:
        backend: The initialized request backend.
        settings: The application settings.
        request_role: The discovered request role.
        maintenance_role: The role the maintenance backend acts as, which owns the tables.

    Raises:
        ConfigurationError: Naming every problem and its fix.
    """

    async def _read(conn: asyncpg.Connection) -> Posture:
        return await read_posture(conn, settings)

    posture: Posture = await backend.execute_read(cast(Any, _read), access_scope=MAINTENANCE)
    problems = posture_problems(posture, settings, request_role=request_role.name)
    if not problems:
        return
    owner = any(security.owned for security in posture.tables.values())
    flag = 'true' if settings.access_control.postgresql_rls_enabled else 'false'
    raise ConfigurationError(
        f"The request role '{request_role.name}' ({_attributes(request_role)}, member of the policy role "
        f'{quote_pg_identifier(settings.access_control.postgresql_rls_policy_role)}: {_yes(posture.policy_member)}, '
        f"privileges of the table owner '{maintenance_role}': {_yes(owner)}) fails the checks of a start with "
        f'ENABLE_POSTGRESQL_RLS={flag}:\n- ' + '\n- '.join(problems) + f'\nSee {DOCS}.',
    )


def warn_jwt_without_rls(settings: AppSettings) -> None:
    """Warn when verified JWT identities reach PostgreSQL over HTTP without row-level security.

    Args:
        settings: The application settings.
    """
    if (
        settings.auth.provider == 'jwt'
        and settings.storage.backend_type == 'postgresql'
        and not settings.access_control.postgresql_rls_enabled
        and settings.transport.transport != 'stdio'
    ):
        logger.warning(JWT_WITHOUT_RLS_WARNING)


def _table_problems(tables: Mapping[str, TableSecurity], *, schema: str) -> list[str]:
    """Return the problems of row-level security on the managed tables with the flag on."""
    problems: list[str] = []
    disabled = [table for table, security in tables.items() if not security.enabled]
    if disabled:
        fixes = ' '.join(f'{enable_statement(table, schema=schema)};' for table in disabled)
        problems.append(
            f'Row-level security is not enabled on {", ".join(disabled)}. Fix: {fixes} Or restart, so the migration '
            'enables it.',
        )
    inactive = [table for table, security in tables.items() if security.enabled and not security.active]
    if inactive:
        problems.append(
            f'Row-level security does not apply to it on {", ".join(inactive)}: it is a superuser, has BYPASSRLS or '
            'holds the privileges of the table owner. Fix: point the request connection settings at a login role '
            'created NOSUPERUSER NOBYPASSRLS that is not a member of the table owner.',
        )
    forced = [table for table, security in tables.items() if security.forced]
    if forced:
        fixes = ' '.join(f'ALTER TABLE {schema}.{table} NO FORCE ROW LEVEL SECURITY;' for table in forced)
        problems.append(
            f'Row-level security is forced on {", ".join(forced)}, which subjects the table owner to the policies and '
            f'leaves the read views empty. Fix: {fixes}',
        )
    return problems


def _foreign_policy_problem(policy: ApplyingPolicy, *, schema: str) -> str:
    """Return the problem of one policy other than ours that applies to the request role."""
    kind = 'PERMISSIVE' if policy.permissive else 'RESTRICTIVE'
    return (
        f'The policy {policy.name} on {schema}.{policy.table} ({kind} TO {", ".join(policy.roles)}) is not one of this '
        "server's and applies to it: a permissive policy widens the access model, a restrictive one narrows it. Fix: "
        f'DROP POLICY {quote_pg_identifier(policy.name)} ON {schema}.{policy.table}; or limit it to roles the request '
        'role does not hold.'
    )


def _object_problems(state: RlsState, *, schema: str, policy_role: str) -> list[str]:
    """Return the problems of our row-level security objects and of the read views."""
    role = quote_pg_identifier(policy_role)
    problems: list[str] = []
    changed = _changed_objects(plan_rls(state, schema=schema, role_name=policy_role))
    if changed:
        problems.append(
            f'These row-level security objects are missing, out of date or changed outside the server: '
            f'{"; ".join(changed)}. Fix: restart, so the migration re-creates them.',
        )
    for name, table in READ_VIEW_TABLES.items():
        installed = state.views.get(name)
        if installed is None or table not in state.tables:
            continue
        view = f'{schema}.{name}'
        if not installed.owned_by_table_owner:
            problems.append(
                f'The read view {view} is not owned by the owner of {schema}.{table}. Fix: restart, so the '
                'migration re-creates it.',
            )
        if not installed.plain:
            problems.append(
                f'The read view {view} carries options such as security_barrier or security_invoker, which change how '
                'it reads. Fix: restart, so the migration re-creates it.',
            )
        trigger = state.objects.get(reject_trigger(schema, view=name).key)
        if trigger is None or not trigger.enabled:
            problems.append(
                f'The trigger {REJECT_TRIGGER} that refuses writes through the read view {view} is '
                f'{"missing" if trigger is None else "disabled"}. Fix: restart, so the migration re-creates it.',
            )
        repair = view_acl_repair(installed.acl, schema=schema, view=name, role_name=policy_role)
        if repair.statements:
            granted = f': {"; ".join(repair.revoked)}' if repair.revoked else ''
            problems.append(
                f'The privileges on the read view {view} are not exactly SELECT for {role}{granted}. Fix: '
                f'{"; ".join(repair.statements)};',
            )
    if not state.helpers_executable:
        problems.append(
            f'The policy role {role} may not execute the helper functions the policies call. Fix: '
            f'{helper_grant(schema, role)};',
        )
    return problems


def _changed_objects(plan: RlsPlan) -> list[str]:
    """Return every object ``plan`` re-creates, as COMMENT ON and DROP name it."""
    changed = [helper.target for helper in plan.helpers]
    changed.extend(obj.target for obj in (plan.guard_function, plan.guard_trigger) if obj is not None)
    for repair in plan.tables:
        changed.extend(policy.target for policy, _ in repair.policies)
    changed.extend(view.target for view in plan.views)
    return changed


def _attributes(request_role: RequestRole) -> str:
    """Return the request role's superuser and BYPASSRLS attributes as the catalog names them."""
    return f'rolsuper={str(request_role.is_superuser).lower()}, rolbypassrls={str(request_role.bypass_rls).lower()}'


def _yes(value: bool) -> str:
    """Return ``value`` as yes or no."""
    return 'yes' if value else 'no'
