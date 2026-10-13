"""Rendering of the PostgreSQL row-level security read views: statement text only, no I/O.

A request role under row-level security reads ``context_entries`` and ``tags`` through two
views the table owner owns: ``context_entries_readable``, whose condition is the read
predicate gated on membership of the policy role, and ``tags_readable``, which lists the
tags of the entries that view returns. Neither carries an option: each runs as its owner,
whom the policies do not filter, and the planner pulls both up into the query, so the
query's own conditions keep their indexes, and a read of the tags joins them to the
readable entries instead of running the tags read policy, which checks the entry of every
tag row once more. This module renders the views, the order they are created in, the
statements that leave SELECT for the policy role as the only privilege anyone but the
owner holds on each, and the function and INSTEAD OF triggers that keep them read-only.

Schema and role names are taken already quoted as SQL identifiers; a view is named by
its own unquoted name, :data:`~app.access_scope.READ_VIEW` or
:data:`~app.access_scope.TAGS_READ_VIEW`.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from types import MappingProxyType
from typing import Final

from app.access_scope import READ_VIEW
from app.access_scope import TAGS_READ_VIEW
from app.access_scope import build_gated_read_predicate
from app.migrations.rls_ddl import CHILD_PARENT_KEYS
from app.migrations.rls_ddl import CONTEXT_TABLE
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import string_literal

TAGS_TABLE: Final = 'tags'
# Each read view, in the order they are created, with the table whose rows it lists. A view
# reads through the views before it, so they are dropped in the reverse order.
READ_VIEW_TABLES: Final[Mapping[str, str]] = MappingProxyType({READ_VIEW: CONTEXT_TABLE, TAGS_READ_VIEW: TAGS_TABLE})

REJECT_FUNCTION: Final = 'mcp_context_reject_view_write'
REJECT_TRIGGER: Final = 'reject_writes'

# The aliases of the listed tables in the views' statements.
_ENTRY: Final = 'ce'
_TAG: Final = 't'
# The privileges a view's ACL holds, as GRANT and REVOKE name them, and ALL.
_ACL_PRIVILEGES: Final = frozenset({
    'ALL', 'SELECT', 'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE', 'REFERENCES', 'TRIGGER', 'MAINTAIN',
})


def read_view(columns: Sequence[str], *, schema: str, role_name: str) -> RlsObject:
    """Return the view through which request roles read ``context_entries``.

    The view lists ``columns`` in order and keeps a row when the reading role is a member
    of the policy role and the bound identity may read the row. It carries no option: it
    runs as its owner, the table owner, whom the policies do not filter, and the planner
    pulls it up into the query, so its conditions and the query's own compete by cost
    alone and the query's conditions keep their indexes. The membership test sits inside
    the identity operands and the public arm of the read predicate
    (:func:`app.access_scope.build_gated_read_predicate`), each a scalar subquery the
    planner evaluates once per statement, so it never stands above a join of the view as
    a condition of its own. For a role outside the policy role the identity is NULL, the
    groups are empty and the public arm is false, so such a role that obtains SELECT on
    the view reads nothing, as the policies give it nothing.

    Args:
        columns: Every column of ``context_entries``, quoted, in attribute order.
        schema: The schema of the tables, quoted.
        role_name: The policy role's own name, which the membership test takes as a
            string literal.

    Returns:
        The view.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if not columns:
        raise ValueError('the read view needs at least one column of context_entries')
    select_list = ', '.join(f'{_ENTRY}.{column}' for column in columns)
    gate = f"pg_catalog.pg_has_role(CURRENT_USER, {string_literal(role_name)}, 'USAGE')"
    readable = build_gated_read_predicate(outer=_ENTRY, schema=schema, gate=gate)
    return RlsObject(
        kind=RlsObjectKind.VIEW,
        name=READ_VIEW,
        anchor=f'{schema}.{READ_VIEW}',
        statement=(
            f'CREATE VIEW {schema}.{READ_VIEW} AS SELECT {select_list} '
            f'FROM {schema}.{CONTEXT_TABLE} AS {_ENTRY} WHERE {readable}'
        ),
    )


def tags_read_view(columns: Sequence[str], *, schema: str) -> RlsObject:
    """Return the view through which request roles read ``tags``.

    The view lists ``columns`` of every tag whose entry the read view of
    ``context_entries`` returns, so a reading role sees the tags of the entries it may
    read and a role outside the policy role sees none: the identity and the membership
    test are the read view's. It carries no option, like that view, so it runs as its
    owner and the planner pulls both views up into the query.

    Args:
        columns: Every column of ``tags``, quoted, in attribute order.
        schema: The schema of the tables, quoted.

    Returns:
        The view.

    Raises:
        ValueError: If ``columns`` is empty.
    """
    if not columns:
        raise ValueError('the tags read view needs at least one column of tags')
    select_list = ', '.join(f'{_TAG}.{column}' for column in columns)
    return RlsObject(
        kind=RlsObjectKind.VIEW,
        name=TAGS_READ_VIEW,
        anchor=f'{schema}.{TAGS_READ_VIEW}',
        statement=(
            f'CREATE VIEW {schema}.{TAGS_READ_VIEW} AS SELECT {select_list} FROM {schema}.{TAGS_TABLE} AS {_TAG} '
            f'JOIN {schema}.{READ_VIEW} AS {_ENTRY} ON {_ENTRY}.id = {_TAG}.{CHILD_PARENT_KEYS[TAGS_TABLE]}'
        ),
    )


def read_views(columns: Mapping[str, Sequence[str]], *, schema: str, role_name: str) -> tuple[RlsObject, ...]:
    """Return the read views the existing tables call for, in the order they are created.

    The read view of ``context_entries`` needs that table, and ``tags_readable``, which
    reads through it, needs ``tags`` as well.

    Args:
        columns: The columns of each table of :data:`READ_VIEW_TABLES` that exists,
            quoted, in attribute order.
        schema: The schema of the tables, quoted.
        role_name: The policy role's own name.

    Returns:
        The views; none without ``context_entries``.
    """
    if CONTEXT_TABLE not in columns:
        return ()
    entries = read_view(columns[CONTEXT_TABLE], schema=schema, role_name=role_name)
    if TAGS_TABLE not in columns:
        return (entries,)
    return entries, tags_read_view(columns[TAGS_TABLE], schema=schema)


def read_view_drop(schema: str, *, view: str) -> str:
    """Return the statement that drops ``view``, and its trigger with it, when it exists.

    It names no CASCADE: dropping the read view of ``context_entries`` while
    ``tags_readable`` reads through it fails instead of taking it along.

    Args:
        schema: The schema of the tables, quoted.
        view: The view's own name.

    Returns:
        The DROP VIEW statement.
    """
    return f'DROP VIEW IF EXISTS {schema}.{view}'


def read_view_grant(schema: str, role: str, *, view: str) -> str:
    """Return the grant that lets the policy role read through ``view``, and nothing more.

    Args:
        schema: The schema of the tables, quoted.
        role: The policy role, quoted.
        view: The view's own name.

    Returns:
        The GRANT SELECT statement.
    """
    return f'GRANT SELECT ON {schema}.{view} TO {role}'


def read_view_revoke(schema: str, grantee: str, privileges: Sequence[str], *, view: str) -> str:
    """Return the statement that revokes ``privileges`` on ``view`` from ``grantee``.

    Args:
        schema: The schema of the tables, quoted.
        grantee: The role, quoted, or ``PUBLIC``.
        privileges: The privileges as the view's ACL names them, or ``ALL``.
        view: The view's own name.

    Returns:
        The REVOKE statement.

    Raises:
        ValueError: If ``privileges`` is empty or names anything but a table privilege;
            each is interpolated into the statement.
    """
    if not privileges or any(privilege not in _ACL_PRIVILEGES for privilege in privileges):
        raise ValueError(f'{list(privileges)!r} is not a list of table privileges')
    return f'REVOKE {", ".join(privileges)} ON {schema}.{view} FROM {grantee}'


def reject_function(schema: str) -> RlsObject:
    """Return the trigger function that refuses every write through either read view.

    The read view of ``context_entries`` is automatically updatable, and a write through
    either view would run as its owner, past every write policy, so the function raises
    insufficient privilege naming the view written to instead, should a later grant ever
    give anyone a write privilege on a view.

    Args:
        schema: The schema of the tables, quoted.

    Returns:
        The trigger function.
    """
    return RlsObject(
        kind=RlsObjectKind.FUNCTION,
        name=REJECT_FUNCTION,
        anchor=f'{schema}.{REJECT_FUNCTION}()',
        statement=(
            f'CREATE OR REPLACE FUNCTION {schema}.{REJECT_FUNCTION}() RETURNS trigger\n'
            '    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$\n'
            'BEGIN\n'
            "    RAISE EXCEPTION '% is read-only', TG_TABLE_NAME USING ERRCODE = '42501';\n"
            'END\n'
            '$$'
        ),
    )


def reject_trigger(schema: str, *, view: str) -> RlsObject:
    """Return the trigger that runs :func:`reject_function` instead of every write on ``view``.

    Args:
        schema: The schema of the tables, quoted.
        view: The view's own name.

    Returns:
        The trigger.
    """
    return RlsObject(
        kind=RlsObjectKind.TRIGGER,
        name=REJECT_TRIGGER,
        anchor=f'{schema}.{view}',
        statement=(
            f'CREATE TRIGGER {REJECT_TRIGGER} INSTEAD OF INSERT OR UPDATE OR DELETE ON {schema}.{view} '
            f'FOR EACH ROW EXECUTE FUNCTION {schema}.{REJECT_FUNCTION}()'
        ),
    )


def write_rejection(schema: str, *, view: str) -> tuple[RlsObject, ...]:
    """Return what keeps ``view`` read-only, in the order it is created after the view.

    The function both views share is created with the read view of ``context_entries``,
    which is created first; each view has its own trigger.

    Args:
        schema: The schema of the tables, quoted.
        view: The view's own name.

    Returns:
        The function and the trigger for the read view of ``context_entries``; the
        trigger alone for any later view.
    """
    trigger = reject_trigger(schema, view=view)
    return (reject_function(schema), trigger) if view == READ_VIEW else (trigger,)
