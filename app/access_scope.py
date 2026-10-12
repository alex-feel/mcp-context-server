"""Caller scopes and the SQL access predicate for context-entry statements.

A scope names whose ``context_entries`` rows a statement may touch: an
:class:`AccessScope` built from the request's effective principal, or
:data:`SYSTEM_SCOPE` for schema maintenance that runs outside any request.
:func:`build_access_predicate` turns a scope into the SQL fragment that limits a
statement to the rows the scope may read, modify or owns:

- ``READ``: the caller owns the row, the row is ``public``, or a ``read`` or
  ``write`` grant names the caller or one of its groups.
- ``WRITE``: the caller owns the row, or a ``write`` grant names the caller or
  one of its groups.
- ``OWNER``: the caller owns the row.

:func:`build_readable_parent_predicate` renders the ``READ`` predicate for a child
row instead: the child's key must name an entry the scope may read.

:func:`build_policy_predicate` renders the same three modes for PostgreSQL
row-level security policies and the read view: the identity comes from
:func:`policy_operands`, scalar subqueries around the schema's identity
functions, instead of bound parameters. Both renderers compose their arms
through one function, so the application predicate and the database policies
cannot drift apart.

A grant carries no visibility condition, so it is effective under either
visibility, and a write grant implies read. Comparisons are exact and
case-sensitive, matching how owner ids and grant principals are stored.

The predicate is a standalone fragment that never passes through the metadata
query builder or its filter counters, so it never counts as a client filter and
never draws on the metadata bind budget. Its bind count is constant per mode on
both backends (three for ``READ`` and ``WRITE``, one for ``OWNER``, none for the
system scope) because the group set binds as ONE parameter: a text array on
PostgreSQL, a JSON array expanded by ``json_each`` on SQLite. An empty group set
therefore needs no special case and never produces an empty ``IN ()`` list.

This module imports only the standard library, so the repositories and
:mod:`app.ids` can import it without pulling in settings or the MCP framework.
"""

import json
from dataclasses import KW_ONLY
from dataclasses import dataclass
from enum import StrEnum
from typing import Final
from typing import final


@dataclass(frozen=True, slots=True)
class AccessScope:
    """The identity a request reads and writes context entries as.

    Attributes:
        principal_id: The caller's principal id, matched against ``owner_id``
            and against user grants.
        groups: The caller's group ids, matched against group grants.
        may_publish: Whether the caller may set visibility ``public`` under
            ``ACCESS_CONTROL_PUBLISH_ROLE``. It is part of the caller's identity,
            not a row filter: the access predicates never read it. Only scopes
            built for the write tools assert it.
    """

    principal_id: str
    groups: frozenset[str]
    _: KW_ONLY
    may_publish: bool = False

    def groups_list(self) -> list[str]:
        """Return the groups sorted, so the bound value is deterministic across processes.

        Returns:
            The group ids in ascending order.
        """
        return sorted(self.groups)

    def groups_json(self) -> str:
        """Return the sorted groups as a JSON array, the SQLite group bind.

        Returns:
            The JSON text of :meth:`groups_list`.
        """
        return json.dumps(self.groups_list())


@final
class SystemScope:
    """Unrestricted scope for schema maintenance that runs outside any request.

    It carries no principal and is never built from request data.
    """

    __slots__ = ()


SYSTEM_SCOPE = SystemScope()

type Scope = AccessScope | SystemScope


class AccessMode(StrEnum):
    """Which rows a predicate admits."""

    READ = 'read'
    WRITE = 'write'
    OWNER = 'owner'


@dataclass(frozen=True, slots=True)
class AccessPredicate:
    """A SQL fragment restricting a statement to the rows a scope may access.

    Attributes:
        sql: Empty for the system scope; otherwise one boolean expression with
            no leading ``AND``, parenthesized whenever it has more than one arm.
        params: Values to bind, in placeholder order: text order on SQLite,
            ``$start`` upward on PostgreSQL.
    """

    sql: str
    params: list[str | list[str]]

    @property
    def bind_count(self) -> int:
        """Number of parameters the predicate binds, the same on both backends."""
        return len(self.params)

    def and_clause(self) -> str:
        """Return the predicate as an ``AND`` term to append to an existing ``WHERE``.

        Returns:
            ``' AND <sql>'``, or an empty string for the system scope.
        """
        return f' AND {self.sql}' if self.sql else ''

    def where_clause(self) -> str:
        """Return the predicate as a complete ``WHERE`` clause.

        Returns:
            ``' WHERE <sql>'``, or an empty string for the system scope.
        """
        return f' WHERE {self.sql}' if self.sql else ''


def build_access_predicate(
    scope: Scope,
    *,
    mode: AccessMode,
    backend_type: str,
    outer: str,
    start: int = 1,
) -> AccessPredicate:
    """Build the predicate limiting a statement to the rows ``scope`` may access in ``mode``.

    The principal binds twice for ``READ`` and ``WRITE`` (the owner arm and the
    user-grant arm) on both backends, so callers advance their placeholder
    position by ``bind_count`` with the same arithmetic everywhere.

    Args:
        scope: The caller's scope; the system scope yields an empty predicate.
        mode: ``READ``, ``WRITE`` or ``OWNER``.
        backend_type: ``'sqlite'`` or ``'postgresql'``, the backend's ``backend_type``.
        outer: The table name or alias of ``context_entries`` in the enclosing
            statement; it qualifies ``id``, ``owner_id`` and ``visibility``,
            because the grant subquery's own table also has an ``id`` column.
            It must be a trusted identifier, never request data.
        start: The first PostgreSQL ``$n`` the predicate may use; ignored on SQLite.

    Returns:
        The predicate text and its parameters.
    """
    if isinstance(scope, SystemScope):
        return AccessPredicate(sql='', params=[])

    principal = scope.principal_id
    if backend_type == 'sqlite':
        owner_operand = '?'
        grantee_operand = '?'
        group_membership = 'IN (SELECT value FROM json_each(?))'
        groups_param: str | list[str] = scope.groups_json()
    else:  # postgresql
        owner_operand = f'${start}'
        grantee_operand = f'${start + 1}'
        group_membership = f'= ANY(${start + 2}::text[])'
        groups_param = scope.groups_list()

    sql = _compose(
        mode,
        outer=outer,
        owner_operand=owner_operand,
        grantee_operand=grantee_operand,
        group_membership=group_membership,
        grants_relation='context_entry_grants',
    )
    params: list[str | list[str]] = [principal] if mode is AccessMode.OWNER else [principal, principal, groups_param]
    return AccessPredicate(sql=sql, params=params)


@dataclass(frozen=True, slots=True)
class PolicyOperands:
    """The SQL operands through which PostgreSQL policies read the transaction's bound identity.

    Each is a scalar subquery around one of the schema's identity functions, so
    PostgreSQL evaluates it once per statement.

    Attributes:
        principal: The bound principal; NULL when none is bound.
        groups: The bound groups as a ``text[]``; empty when none are bound. The
            explicit cast makes ``= ANY(<groups>)`` compare with the elements of
            the array: without it, ``= ANY((SELECT ...))`` compares with each row
            of the subquery.
        may_publish: Whether the bound caller may set visibility ``public``.
    """

    principal: str
    groups: str
    may_publish: str


# The schema functions that hand the bound identity to the policies.
PRINCIPAL_FUNCTION: Final = 'mcp_context_principal_id'
GROUPS_FUNCTION: Final = 'mcp_context_groups'
MAY_PUBLISH_FUNCTION: Final = 'mcp_context_may_publish'


def policy_operands(schema: str) -> PolicyOperands:
    """Return the operands of the identity functions in ``schema``.

    Args:
        schema: The schema holding the identity functions, already quoted as a
            SQL identifier.

    Returns:
        The principal, groups and publish-right operands.
    """
    return PolicyOperands(
        principal=f'(SELECT {schema}.{PRINCIPAL_FUNCTION}())',
        groups=f'(SELECT {schema}.{GROUPS_FUNCTION}())::text[]',
        may_publish=f'(SELECT {schema}.{MAY_PUBLISH_FUNCTION}())',
    )


def build_policy_predicate(mode: AccessMode, *, outer: str, schema: str) -> str:
    """Render the PostgreSQL row-level security form of the ``mode`` predicate.

    The arms are those of :func:`build_access_predicate`; only the operands
    differ. The principal and the groups are the :func:`policy_operands` of
    ``schema``, and the grants table is qualified by ``schema``. The text binds
    no parameter, so it serves as a policy expression and as the read view's
    ``WHERE`` conjunct alike.

    Args:
        mode: ``READ``, ``WRITE`` or ``OWNER``.
        outer: The table name or alias of ``context_entries`` the predicate
            qualifies, a trusted identifier: the table name in a policy on
            ``context_entries``, the alias in the read view or in a child
            table's policy subquery.
        schema: The schema holding the tables and the identity functions,
            already quoted as a SQL identifier, so this module needs nothing
            beyond the standard library.

    Returns:
        The predicate text.
    """
    operands = policy_operands(schema)
    return _compose(
        mode,
        outer=outer,
        owner_operand=operands.principal,
        grantee_operand=operands.principal,
        group_membership=f'= ANY({operands.groups})',
        grants_relation=f'{schema}.context_entry_grants',
    )


def _compose(
    mode: AccessMode,
    *,
    outer: str,
    owner_operand: str,
    grantee_operand: str,
    group_membership: str,
    grants_relation: str,
) -> str:
    """Compose the arms of the ``mode`` predicate around the given operands.

    Args:
        mode: ``READ``, ``WRITE`` or ``OWNER``.
        outer: The table name or alias of ``context_entries``; it qualifies
            ``id``, ``owner_id`` and ``visibility``, because the grant subquery's
            own table also has an ``id`` column.
        owner_operand: The value ``owner_id`` must equal.
        grantee_operand: The value a user grant's ``principal_id`` must equal.
        group_membership: The test a group grant's ``principal_id`` must pass,
            written to follow the column, such as ``= ANY(...)``.
        grants_relation: The grants table as the statement names it.

    Returns:
        ``<outer>.owner_id = <owner>`` for ``OWNER``; otherwise the owner arm,
        for ``READ`` the public arm, and the grant arm, parenthesized as one
        disjunction.
    """
    owner_arm = f'{outer}.owner_id = {owner_operand}'
    if mode is AccessMode.OWNER:
        return owner_arm

    if mode is AccessMode.READ:
        public_arm = f" OR {outer}.visibility = 'public'"
        permission = "g.permission IN ('read', 'write')"
    else:  # WRITE
        public_arm = ''
        permission = "g.permission = 'write'"
    grant_arm = (
        f'EXISTS (SELECT 1 FROM {grants_relation} g WHERE g.context_entry_id = {outer}.id AND {permission} '
        f"AND ((g.principal_type = 'user' AND g.principal_id = {grantee_operand}) "
        f"OR (g.principal_type = 'group' AND g.principal_id {group_membership})))"
    )
    return f'({owner_arm}{public_arm} OR {grant_arm})'


def build_readable_parent_predicate(
    scope: Scope,
    *,
    child_key: str,
    parent_key: str,
    backend_type: str,
    start: int = 1,
) -> AccessPredicate:
    """Build the predicate limiting child rows to those whose parent entry ``scope`` may read.

    The predicate tests the child's key for membership in the keys of the readable
    entries: ``<child_key> IN (SELECT ce.<parent_key> FROM context_entries ce WHERE <READ>)``.
    SQLite statements use it in place of a join to each child's parent row. A join
    looks each parent up by ``id`` or ``rowid_int``, and neither lookup holds
    ``owner_id`` or ``visibility``, so every child row reads its parent's table row and
    walks the text overflow pages stored before those columns; the key set instead comes
    from one scan of a covering access index. The parameters are those of the ``READ``
    predicate.

    Args:
        scope: The caller's scope; the system scope yields an empty predicate.
        child_key: The child column holding the parent key, qualified by the child's
            name or alias in the enclosing statement. It must be a trusted identifier,
            never request data.
        parent_key: The ``context_entries`` column ``child_key`` references, a trusted
            identifier: ``id``, or ``rowid_int`` for the SQLite FTS index.
        backend_type: ``'sqlite'`` or ``'postgresql'``, the backend's ``backend_type``.
        start: The first PostgreSQL ``$n`` the predicate may use; ignored on SQLite.

    Returns:
        The predicate text and its parameters.
    """
    read = build_access_predicate(scope, mode=AccessMode.READ, backend_type=backend_type, outer='ce', start=start)
    if not read.sql:
        return read
    return AccessPredicate(
        sql=f'{child_key} IN (SELECT ce.{parent_key} FROM context_entries ce WHERE {read.sql})',
        params=read.params,
    )
