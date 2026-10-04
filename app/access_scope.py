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
from dataclasses import dataclass
from enum import StrEnum
from typing import final


@dataclass(frozen=True, slots=True)
class AccessScope:
    """The identity a request reads and writes context entries as.

    Attributes:
        principal_id: The caller's principal id, matched against ``owner_id``
            and against user grants.
        groups: The caller's group ids, matched against group grants.
    """

    principal_id: str
    groups: frozenset[str]

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
        owner_placeholder = '?'
        grantee_placeholder = '?'
        group_membership = 'g.principal_id IN (SELECT value FROM json_each(?))'
        groups_param: str | list[str] = scope.groups_json()
    else:  # postgresql
        owner_placeholder = f'${start}'
        grantee_placeholder = f'${start + 1}'
        group_membership = f'g.principal_id = ANY(${start + 2}::text[])'
        groups_param = scope.groups_list()

    owner_arm = f'{outer}.owner_id = {owner_placeholder}'
    if mode is AccessMode.OWNER:
        return AccessPredicate(sql=owner_arm, params=[principal])

    if mode is AccessMode.READ:
        public_arm = f" OR {outer}.visibility = 'public'"
        permission = "g.permission IN ('read', 'write')"
    else:  # WRITE
        public_arm = ''
        permission = "g.permission = 'write'"
    grant_arm = (
        f'EXISTS (SELECT 1 FROM context_entry_grants g WHERE g.context_entry_id = {outer}.id AND {permission} '
        f"AND ((g.principal_type = 'user' AND g.principal_id = {grantee_placeholder}) "
        f"OR (g.principal_type = 'group' AND {group_membership})))"
    )
    return AccessPredicate(
        sql=f'({owner_arm}{public_arm} OR {grant_arm})',
        params=[principal, principal, groups_param],
    )


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
