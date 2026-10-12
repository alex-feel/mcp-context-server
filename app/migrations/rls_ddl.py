"""Rendering of the PostgreSQL row-level security objects: statement text only, no I/O.

Row-level security enforces the access model of :mod:`app.access_scope` in the database
for the NOLOGIN policy role every request login role is a member of. This module renders
the objects that do it:

- three helper functions that hand the transaction's bound identity to the policies, and
  the grant that lets the policy role execute them;
- per-command PERMISSIVE policies on ``context_entries`` (READ for SELECT, the owner
  stamp and the publish gate for INSERT, WRITE for UPDATE, OWNER for DELETE), on
  ``context_entry_grants`` (the caller's own grants; inserts by the entry's owner) and on
  every child table (reads follow the parent's read policy; inserts and deletes need
  WRITE on the parent; no update policy, because no child row is ever updated);
- the guard trigger that keeps ``owner_id`` immutable, visibility changes owner-only and
  publishing gated, which a policy cannot express because WITH CHECK does not see the old
  row.

Every policy expression of ``context_entries`` and of a child's parent comes from
:func:`app.access_scope.build_policy_predicate`, the composer the application predicate
uses, so the two cannot drift apart. Policies apply to the policy role only, never to
PUBLIC, so every other role without the table owner's bypass reads nothing. Row-level
security is enabled without FORCE: the table owner, which runs migrations and the
command-line tools, keeps reading every row.

Every object carries a fingerprint comment: the ``sha`` of its rendered statement, which
changes whenever this module renders something different, and the digest of the server's
own deparse of the object, which changes when the object is altered out of band.

Schema and role names are taken already quoted as SQL identifiers.
"""

import hashlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from app.access_scope import GROUPS_FUNCTION
from app.access_scope import MAY_PUBLISH_FUNCTION
from app.access_scope import PRINCIPAL_FUNCTION
from app.access_scope import AccessMode
from app.access_scope import build_policy_predicate
from app.access_scope import policy_operands

FINGERPRINT_PREFIX: Final = 'mcp-context-server rls v1'

CONTEXT_TABLE: Final = 'context_entries'
GRANTS_TABLE: Final = 'context_entry_grants'
# Every table whose rows belong to one context entry, with the column naming that entry.
CHILD_PARENT_KEYS: Final[Mapping[str, str]] = MappingProxyType({
    'tags': 'context_entry_id',
    'image_attachments': 'context_entry_id',
    'embedding_metadata': 'context_id',
    'vec_context_embeddings': 'context_id',
    'vec_context_embeddings_compressed': 'context_id',
    'context_index_nodes': 'context_id',
})
# The tables row-level security governs; each layout and feature set provisions a subset.
MANAGED_TABLES: Final[tuple[str, ...]] = (CONTEXT_TABLE, GRANTS_TABLE, *CHILD_PARENT_KEYS)

GUARD_FUNCTION: Final = 'mcp_context_guard_access_columns'
GUARD_TRIGGER: Final = 'guard_context_entry_access_columns'

# Alias of the parent entry in the subqueries of the grants and child policies.
_PARENT: Final = 'ce'
_HEX_DIGEST = re.compile(r'[0-9a-f]{64}')

# Each helper reads one transaction-local setting the request profile binds. An unset
# setting reads NULL and a setting cleared by COMMIT or RESET ALL reads '', and NULLIF turns
# both into no principal, no groups and no publish right.
_HELPER_BODIES: Final = (
    (PRINCIPAL_FUNCTION, 'text', "SELECT NULLIF(pg_catalog.current_setting('app.principal_id', true), '')"),
    (
        GROUPS_FUNCTION,
        'text[]',
        "SELECT COALESCE(NULLIF(pg_catalog.current_setting('app.groups', true), '')::text[], '{}'::text[])",
    ),
    (
        MAY_PUBLISH_FUNCTION,
        'boolean',
        "SELECT COALESCE(NULLIF(pg_catalog.current_setting('app.may_publish', true), '') = 'on', false)",
    ),
)


class RlsObjectKind(StrEnum):
    """The kinds of database object row-level security creates, as SQL names them."""

    FUNCTION = 'FUNCTION'
    POLICY = 'POLICY'
    TRIGGER = 'TRIGGER'


@dataclass(frozen=True, slots=True)
class RlsObject:
    """One row-level security object and the statement that creates it.

    Attributes:
        kind: Function, policy or trigger.
        name: The object's own name.
        anchor: For a function, its schema-qualified signature; for a policy or a
            trigger, the schema-qualified table it is defined on.
        statement: The statement that creates the object; functions are created or
            replaced, policies and triggers are created after their drop.
    """

    kind: RlsObjectKind
    name: str
    anchor: str
    statement: str

    @property
    def sha(self) -> str:
        """The digest of :attr:`statement`, which embeds the schema and the policy role."""
        return digest(self.statement)

    @property
    def target(self) -> str:
        """The object as COMMENT ON and DROP name it, such as ``POLICY tags_read ON "public".tags``."""
        if self.kind is RlsObjectKind.FUNCTION:
            return f'{self.kind} {self.anchor}'
        return f'{self.kind} {self.name} ON {self.anchor}'

    @property
    def drop_statement(self) -> str:
        """The statement that drops the object when it exists."""
        kind, _, reference = self.target.partition(' ')
        return f'DROP {kind} IF EXISTS {reference}'


def digest(text: str) -> str:
    """Return the lowercase hex SHA-256 of ``text``.

    Args:
        text: The text to digest.

    Returns:
        64 hex characters.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def fingerprint(*, sha: str, definition: str) -> str:
    """Return the fingerprint comment of an object.

    Args:
        sha: The digest of the object's rendered statement.
        definition: The digest of the server's deparse of the object.

    Returns:
        ``'mcp-context-server rls v1 sha=<sha> def=<definition>'``.
    """
    return f'{FINGERPRINT_PREFIX} sha={sha} def={definition}'


def comment_statement(obj: RlsObject, *, definition: str) -> str:
    """Return the COMMENT that attaches ``obj``'s fingerprint to it.

    Args:
        obj: The object.
        definition: The digest of the server's deparse of the object.

    Returns:
        The COMMENT ON statement.

    Raises:
        ValueError: If ``definition`` is not a lowercase hex SHA-256 digest; it is
            interpolated into a string literal.
    """
    if _HEX_DIGEST.fullmatch(definition) is None:
        raise ValueError(f'{definition!r} is not a SHA-256 hex digest')
    return f"COMMENT ON {obj.target} IS '{fingerprint(sha=obj.sha, definition=definition)}'"


def helper_functions(schema: str) -> tuple[RlsObject, RlsObject, RlsObject]:
    """Return the principal, groups and publish-right helpers the policy operands call.

    They are STABLE and PARALLEL SAFE SQL functions running as the invoker with a pinned
    search_path.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The three helper functions, in that order.
    """
    principal, groups, may_publish = (
        RlsObject(
            kind=RlsObjectKind.FUNCTION,
            name=name,
            anchor=f'{schema}.{name}()',
            statement=(
                f'CREATE OR REPLACE FUNCTION {schema}.{name}() RETURNS {returns} '
                'LANGUAGE sql STABLE PARALLEL SAFE SET search_path = pg_catalog, pg_temp '
                f'AS $$ {body} $$'
            ),
        )
        for name, returns, body in _HELPER_BODIES
    )
    return principal, groups, may_publish


def helper_grant(schema: str, role: str) -> str:
    """Return the grant that lets the policy role execute the helper functions.

    Args:
        schema: The schema of the tables, quoted.
        role: The policy role, quoted.

    Returns:
        The GRANT EXECUTE statement.
    """
    functions = ', '.join(helper.anchor for helper in helper_functions(schema))
    return f'GRANT EXECUTE ON FUNCTION {functions} TO {role}'


def guard_function(schema: str) -> RlsObject:
    """Return the trigger function guarding the access columns of ``context_entries``.

    A transaction that binds no principal passes through: the request role then matches no
    update policy, so its updates never reach the trigger, and a maintenance transaction,
    such as the owner reassignment of the command line, must pass.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The trigger function.
    """
    return RlsObject(
        kind=RlsObjectKind.FUNCTION,
        name=GUARD_FUNCTION,
        anchor=f'{schema}.{GUARD_FUNCTION}()',
        statement=(
            f'CREATE OR REPLACE FUNCTION {schema}.{GUARD_FUNCTION}() RETURNS trigger\n'
            '    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$\n'
            'DECLARE\n'
            f'    principal text := {schema}.{PRINCIPAL_FUNCTION}();\n'
            'BEGIN\n'
            '    IF principal IS NULL THEN\n'
            '        RETURN NEW;\n'
            '    END IF;\n'
            '    IF NEW.owner_id IS DISTINCT FROM OLD.owner_id THEN\n'
            "        RAISE EXCEPTION 'owner_id of context entry % is immutable', OLD.id USING ERRCODE = '42501';\n"
            '    END IF;\n'
            '    IF NEW.visibility IS DISTINCT FROM OLD.visibility THEN\n'
            '        IF OLD.owner_id IS DISTINCT FROM principal THEN\n'
            "            RAISE EXCEPTION 'only the owner may change the visibility of context entry %', OLD.id "
            "USING ERRCODE = '42501';\n"
            '        END IF;\n'
            f"        IF NEW.visibility = 'public' AND NOT {schema}.{MAY_PUBLISH_FUNCTION}() THEN\n"
            "            RAISE EXCEPTION 'publishing context entry % requires the publish role', OLD.id "
            "USING ERRCODE = '42501';\n"
            '        END IF;\n'
            '    END IF;\n'
            '    RETURN NEW;\n'
            'END\n'
            '$$'
        ),
    )


def guard_trigger(schema: str) -> RlsObject:
    """Return the trigger that runs the guard for updates naming ``owner_id`` or ``visibility``.

    Updates that set neither column, such as text and metadata edits, never fire it.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The trigger.
    """
    return RlsObject(
        kind=RlsObjectKind.TRIGGER,
        name=GUARD_TRIGGER,
        anchor=f'{schema}.{CONTEXT_TABLE}',
        statement=(
            f'CREATE TRIGGER {GUARD_TRIGGER} BEFORE UPDATE OF owner_id, visibility ON {schema}.{CONTEXT_TABLE} '
            f'FOR EACH ROW EXECUTE FUNCTION {schema}.{GUARD_FUNCTION}()'
        ),
    )


def policy_names(table: str) -> tuple[str, ...]:
    """Return the names of ``table``'s policies.

    Args:
        table: A managed table.

    Returns:
        The table name joined with each command the table has a policy for.

    Raises:
        ValueError: If ``table`` is not managed.
    """
    if table == CONTEXT_TABLE:
        commands: tuple[str, ...] = ('read', 'insert', 'update', 'delete')
    elif table == GRANTS_TABLE:
        commands = ('read', 'insert')
    elif table in CHILD_PARENT_KEYS:
        commands = ('read', 'insert', 'delete')
    else:
        raise ValueError(f'{table!r} is not a table row-level security manages')
    return tuple(f'{table}_{command}' for command in commands)


def table_policies(table: str, *, schema: str, role: str) -> tuple[RlsObject, ...]:
    """Return ``table``'s policies, in the order of :func:`policy_names`.

    Args:
        table: A managed table; :func:`policy_names` refuses any other.
        schema: The schema of the tables, quoted.
        role: The policy role, quoted.

    Returns:
        The policies.
    """
    names = policy_names(table)
    if table == CONTEXT_TABLE:
        clauses = _context_entries_clauses(schema)
    elif table == GRANTS_TABLE:
        clauses = _grants_clauses(schema)
    else:
        clauses = _child_clauses(table, CHILD_PARENT_KEYS[table], schema)
    return tuple(
        RlsObject(
            kind=RlsObjectKind.POLICY,
            name=name,
            anchor=f'{schema}.{table}',
            statement=f'CREATE POLICY {name} ON {schema}.{table} AS PERMISSIVE FOR {command} TO {role}{clause}',
        )
        for name, (command, clause) in zip(names, clauses, strict=True)
    )


def enable_statement(table: str, *, schema: str) -> str:
    """Return the statement that enables row-level security on ``table``, never forcing it.

    Args:
        table: A managed table.
        schema: The schema of the tables, quoted.

    Returns:
        The ALTER TABLE statement.

    Raises:
        ValueError: If ``table`` is not managed.
    """
    if table not in MANAGED_TABLES:
        raise ValueError(f'{table!r} is not a table row-level security manages')
    return f'ALTER TABLE {schema}.{table} ENABLE ROW LEVEL SECURITY'


def _context_entries_clauses(schema: str) -> tuple[tuple[str, str], ...]:
    """Return the command and the USING and WITH CHECK clauses of each context_entries policy."""
    read = build_policy_predicate(AccessMode.READ, outer=CONTEXT_TABLE, schema=schema)
    write = build_policy_predicate(AccessMode.WRITE, outer=CONTEXT_TABLE, schema=schema)
    owner = build_policy_predicate(AccessMode.OWNER, outer=CONTEXT_TABLE, schema=schema)
    may_publish = policy_operands(schema).may_publish
    return (
        ('SELECT', f' USING ({read})'),
        ('INSERT', f" WITH CHECK ({owner} AND ({CONTEXT_TABLE}.visibility <> 'public' OR {may_publish}))"),
        ('UPDATE', f' USING ({write}) WITH CHECK ({write})'),
        ('DELETE', f' USING ({owner})'),
    )


def _grants_clauses(schema: str) -> tuple[tuple[str, str], ...]:
    """Return the command and clause of each context_entry_grants policy.

    The read policy calls the helpers directly instead of through their scalar subqueries,
    so it holds no subquery at all. PostgreSQL rejects as infinitely recursive a policy
    with a subquery that is reached again while its own table's policies expand, and the
    insert check reaches the read policy again: it reads ``context_entries``, whose read
    policy reads this table. A read policy without a subquery ends the expansion there.

    Returns:
        The SELECT and INSERT commands with their clauses.
    """
    operands = policy_operands(schema)
    principal, groups, _ = helper_functions(schema)
    own_grants = (
        f"(principal_type = 'user' AND principal_id = {principal.anchor}) "
        f"OR (principal_type = 'group' AND principal_id = ANY({groups.anchor}))"
    )
    owner = build_policy_predicate(AccessMode.OWNER, outer=_PARENT, schema=schema)
    granted_entry = (
        f'EXISTS (SELECT 1 FROM {schema}.{CONTEXT_TABLE} {_PARENT} '
        f'WHERE {_PARENT}.id = {GRANTS_TABLE}.context_entry_id AND {owner})'
    )
    return (
        ('SELECT', f' USING ({own_grants})'),
        ('INSERT', f' WITH CHECK (granted_by = {operands.principal} AND {granted_entry})'),
    )


def _child_clauses(table: str, key: str, schema: str) -> tuple[tuple[str, str], ...]:
    """Return the command and clause of each policy of the child ``table``, whose ``key`` names its entry.

    The read policy's subquery reads the parent through the parent's own read policy,
    because policy subqueries run as the invoker. An invisible parent and a missing one
    refuse an insert alike, so the foreign key reveals nothing.

    Returns:
        The SELECT, INSERT and DELETE commands with their clauses.
    """
    parent = f'EXISTS (SELECT 1 FROM {schema}.{CONTEXT_TABLE} {_PARENT} WHERE {_PARENT}.id = {table}.{key}'
    writable = f'{parent} AND {build_policy_predicate(AccessMode.WRITE, outer=_PARENT, schema=schema)})'
    return (
        ('SELECT', f' USING ({parent}))'),
        ('INSERT', f' WITH CHECK ({writable})'),
        ('DELETE', f' USING ({writable})'),
    )
