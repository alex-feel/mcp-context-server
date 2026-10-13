"""The privileges a request login role needs, granted to the policy role it is a member of.

When requests log in as a role other than the table owner, the owner grants the NOLOGIN
policy role (``POSTGRESQL_RLS_POLICY_ROLE``), and through membership the request role,
exactly what the server's request path uses: USAGE on the schema; SELECT, INSERT, UPDATE
and DELETE on every managed table that exists; SELECT on the compression provenance; USAGE
and SELECT on the sequences the managed tables own; and EXECUTE on the merge-patch
function. Nothing is granted on the read view, on any other table or sequence of a schema
that may be shared, or for TRUNCATE.

Grants are computed missing-only from the catalog, one privilege check per privilege, so
a boot that finds everything in place writes no catalog row and takes no lock; when
something is missing, the step takes the schema-init advisory lock, which also serializes
concurrent pods, reads again and issues only what is still missing. It runs whenever the
roles are split, whether or not row-level security is enabled.
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
from app.backends.postgresql_backend.session import quote_pg_identifier
from app.errors import ConfigurationError
from app.errors import format_exception_message
from app.migrations._pg_ddl import begin_migration
from app.migrations._pg_ddl import execute_migration_ddl
from app.migrations.rls_ddl import MANAGED_TABLES
from app.settings import AppSettings

logger = logging.getLogger(__name__)

TABLE_PRIVILEGES: Final = ('SELECT', 'INSERT', 'UPDATE', 'DELETE')
SEQUENCE_PRIVILEGES: Final = ('USAGE', 'SELECT')
PROVENANCE_TABLE: Final = 'compression_metadata'
MERGE_PATCH_SIGNATURE: Final = 'jsonb_merge_patch(jsonb, jsonb)'

# $1 schema name, $2 policy role, $3 candidate tables: the four privileges the role holds on each that exists.
TABLE_PRIVILEGES_SQL: Final = (
    'SELECT c.relname, '
    + ', '.join(f"pg_catalog.has_table_privilege($2, c.oid, '{privilege}')" for privilege in TABLE_PRIVILEGES)
    + ' FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace '
    "WHERE n.nspname = $1 AND c.relkind IN ('r', 'p') AND c.relname = ANY($3::text[])"
)
# $1 schema name, $2 policy role, $3 managed tables: every sequence a column of a managed table
# owns (serial columns depend automatically, identity columns internally, the set
# pg_get_serial_sequence reports), with its schema and the privileges the role holds on it.
OWNED_SEQUENCES_SQL: Final = (
    'SELECT sn.nspname, s.relname, '
    + ', '.join(f"pg_catalog.has_sequence_privilege($2, s.oid, '{privilege}')" for privilege in SEQUENCE_PRIVILEGES)
    + ' FROM pg_catalog.pg_depend d '
    'JOIN pg_catalog.pg_class s ON s.oid = d.objid '
    'JOIN pg_catalog.pg_namespace sn ON sn.oid = s.relnamespace '
    'JOIN pg_catalog.pg_class t ON t.oid = d.refobjid '
    'JOIN pg_catalog.pg_namespace tn ON tn.oid = t.relnamespace '
    "WHERE d.classid = 'pg_catalog.pg_class'::pg_catalog.regclass "
    "AND d.refclassid = 'pg_catalog.pg_class'::pg_catalog.regclass "
    "AND d.deptype IN ('a', 'i') AND s.relkind = 'S' "
    'AND tn.nspname = $1 AND t.relname = ANY($3::text[]) '
    'ORDER BY pg_catalog.array_position($3::text[], t.relname::text), s.relname'
)
_SCHEMA_USAGE_SQL: Final = "SELECT pg_catalog.has_schema_privilege($2, $1, 'USAGE')"
# NULL when the function does not exist.
_MERGE_PATCH_EXECUTE_SQL: Final = (
    'SELECT CASE WHEN pg_catalog.to_regprocedure($1) IS NULL THEN NULL '
    "ELSE pg_catalog.has_function_privilege($2, pg_catalog.to_regprocedure($1)::pg_catalog.oid, 'EXECUTE') END"
)


@dataclass(frozen=True, slots=True)
class GrantReport:
    """What one run of :func:`apply_request_role_grants` did.

    Attributes:
        statements: The GRANT statements issued, in order; empty when nothing was missing.
        locked: Whether the run took the schema-init advisory lock, which it does only to
            grant.
    """

    statements: list[str]
    locked: bool


@dataclass(frozen=True, slots=True)
class HeldPrivileges:
    """The privileges the policy role holds, as the catalog reports them.

    Attributes:
        schema_usage: Whether it holds USAGE on the schema.
        tables: Each managed table and the provenance table that exists, with the table
            privileges the role holds on it.
        sequences: Each sequence a managed table owns, schema-qualified and quoted, with
            the sequence privileges the role holds on it.
        merge_patch: Whether it may execute the merge-patch function; None when the
            function does not exist.
    """

    schema_usage: bool
    tables: Mapping[str, frozenset[str]]
    sequences: Mapping[str, frozenset[str]]
    merge_patch: bool | None


def plan_grants(held: HeldPrivileges, *, schema: str, role: str) -> list[str]:
    """Return the GRANT statements for every privilege the policy role needs and does not hold.

    Args:
        held: What the role holds.
        schema: The schema of the tables, quoted.
        role: The policy role, quoted.

    Returns:
        The statements: the schema, the managed tables in managed order, the provenance
        table, the owned sequences and the merge-patch function.
    """
    statements: list[str] = []
    if not held.schema_usage:
        statements.append(f'GRANT USAGE ON SCHEMA {schema} TO {role}')
    needed: dict[str, tuple[str, ...]] = {**dict.fromkeys(MANAGED_TABLES, TABLE_PRIVILEGES), PROVENANCE_TABLE: ('SELECT',)}
    for table, privileges in needed.items():
        if table not in held.tables:
            continue
        missing = [privilege for privilege in privileges if privilege not in held.tables[table]]
        if missing:
            statements.append(f'GRANT {", ".join(missing)} ON TABLE {schema}.{table} TO {role}')
    for sequence, sequence_held in held.sequences.items():
        missing = [privilege for privilege in SEQUENCE_PRIVILEGES if privilege not in sequence_held]
        if missing:
            statements.append(f'GRANT {", ".join(missing)} ON SEQUENCE {sequence} TO {role}')
    if held.merge_patch is False:
        statements.append(f'GRANT EXECUTE ON FUNCTION {schema}.{MERGE_PATCH_SIGNATURE} TO {role}')
    return statements


async def read_held_privileges(conn: asyncpg.Connection, *, schema_name: str, role_name: str) -> HeldPrivileges:
    """Read which of the needed privileges the policy role holds, one privilege check each.

    A privilege counts as held however the role holds it: granted to it, inherited from a
    role it is a member of, or granted to PUBLIC.

    Args:
        conn: A connection of the table owner.
        schema_name: The schema of the tables, unquoted.
        role_name: The policy role, unquoted.

    Returns:
        The held privileges.
    """
    schema = quote_pg_identifier(schema_name)
    schema_usage = bool(await conn.fetchval(_SCHEMA_USAGE_SQL, schema_name, role_name))
    table_rows = await conn.fetch(TABLE_PRIVILEGES_SQL, schema_name, role_name, [*MANAGED_TABLES, PROVENANCE_TABLE])
    sequence_rows = await conn.fetch(OWNED_SEQUENCES_SQL, schema_name, role_name, list(MANAGED_TABLES))
    merge_patch = await conn.fetchval(_MERGE_PATCH_EXECUTE_SQL, f'{schema}.{MERGE_PATCH_SIGNATURE}', role_name)
    return HeldPrivileges(
        schema_usage=schema_usage,
        tables={
            str(row[0]): frozenset(privilege for privilege, held in zip(TABLE_PRIVILEGES, row[1:], strict=True) if held)
            for row in table_rows
        },
        sequences={
            f'{quote_pg_identifier(str(row[0]))}.{quote_pg_identifier(str(row[1]))}': frozenset(
                privilege for privilege, held in zip(SEQUENCE_PRIVILEGES, row[2:], strict=True) if held
            )
            for row in sequence_rows
        },
        merge_patch=None if merge_patch is None else bool(merge_patch),
    )


async def apply_request_role_grants(
    backend: StorageBackend,
    settings: AppSettings,
    *,
    request_role: str | None,
) -> GrantReport:
    """Grant the policy role the privileges the request role needs, issuing only the missing ones.

    Runs on PostgreSQL when ``request_role`` is given and differs from the role the
    maintenance backend logs in as; otherwise it grants nothing. It works whether or not
    row-level security is enabled.

    Args:
        backend: The maintenance backend, logged in as the table owner.
        settings: The application settings: the schema, the policy role and the migration
            budget.
        request_role: The role requests log in as, as the database reports it; None when
            requests use the maintenance credentials.

    Returns:
        The statements issued and whether the lock was taken.

    Raises:
        ConfigurationError: If the policy role does not exist; the message carries the
            statements that create it and its membership.
        RuntimeError: If reading the catalog or granting fails.
    """
    if request_role is None or backend.backend_type != 'postgresql':
        return GrantReport(statements=[], locked=False)

    schema_name = settings.storage.postgresql_schema
    schema = quote_pg_identifier(schema_name)
    role_name = settings.access_control.postgresql_rls_policy_role
    role = quote_pg_identifier(role_name)
    timeout_s = settings.storage.postgresql_migration_timeout_s

    async def _grant(conn: asyncpg.Connection) -> GrantReport:
        if await conn.fetchval('SELECT current_user') == request_role:
            return GrantReport(statements=[], locked=False)
        if not await conn.fetchval('SELECT pg_catalog.to_regrole($1) IS NOT NULL', role):
            raise ConfigurationError(_missing_policy_role_message(role, quote_pg_identifier(request_role)))
        held = await read_held_privileges(conn, schema_name=schema_name, role_name=role_name)
        if not plan_grants(held, schema=schema, role=role):
            return GrantReport(statements=[], locked=False)
        await begin_migration(conn, timeout_s)
        held = await read_held_privileges(conn, schema_name=schema_name, role_name=role_name)
        statements = plan_grants(held, schema=schema, role=role)
        for statement in statements:
            await execute_migration_ddl(conn, statement, timeout_s)
        return GrantReport(statements=statements, locked=True)

    try:
        report: GrantReport = await backend.execute_write(cast(Any, _grant), access_scope=MAINTENANCE)
    except ConfigurationError:
        raise
    except Exception as e:
        logger.error(f'Failed to grant the request role privileges: {e}')
        raise RuntimeError(f'Request role grants failed: {format_exception_message(e)}') from e
    if report.statements:
        logger.info(f'Granted {role} {len(report.statements)} missing privilege statements for the request role')
    return report


def _missing_policy_role_message(role: str, request_role: str) -> str:
    """Return the refusal for a policy role that does not exist, with the statements that fix it."""
    return (
        f'POSTGRESQL_RLS_POLICY_ROLE names the PostgreSQL role {role}, which does not exist, so the request login '
        f'role {request_role} cannot be granted the privileges it needs. As a role administrator, create the role '
        f'and make the login role its member, then restart: CREATE ROLE {role} NOLOGIN; GRANT {role} TO '
        f"{request_role}; (docs/database-backends.md, 'Row-level security (PostgreSQL)')."
    )
