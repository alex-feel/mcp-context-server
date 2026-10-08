"""Session profiles: how a PostgreSQL backend presents the caller's identity to the database.

A profile is injected into the backend at construction. The entry points hand every
transaction's identity to :meth:`SessionProfile.bind` as the first statement inside the
transaction, and a read runs inside such a transaction when the profile wraps reads, so a
profile that enforces row-level security can scope the transaction to the caller and a
profile that does not can leave it untouched. The profile also names the relation each
plain read selects from (:meth:`SessionProfile.readable_relation`) and the session
statements every pooled connection runs on acquire.

Three profiles exist: :class:`UnboundProfile` for a request backend whose database
enforces no row-level security, :class:`RlsRequestProfile` for a request backend whose
database does, and :class:`MaintenanceProfile` for the backend that owns the tables and
runs the migrations, startup checks and command-line tools.
"""

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final
from typing import Protocol
from typing import runtime_checkable

import asyncpg

from app.access_scope import AccessScope
from app.backends.base import BackendAccess
from app.backends.base import base_relation

# Binds the caller for the rest of the open transaction in one round trip. Every value is
# transaction-local, so it vanishes at COMMIT or ROLLBACK and never reaches the next user of
# a pooled connection or of a transaction-mode pooler's server connection. The groups travel
# as a text[] rendered to its exact array literal, which escapes commas, quotes, braces and
# backslashes. jit is read when the later statements of the transaction are planned, and the
# policy-heavy plans gain nothing from compiling. One statement rather than a CTE, because
# SQL does not fix the evaluation order of CTEs.
BIND_IDENTITY_STATEMENT: Final = (
    "SELECT pg_catalog.set_config('app.principal_id', $1, true), "
    "pg_catalog.set_config('app.groups', $2::text[]::text, true), "
    "pg_catalog.set_config('app.may_publish', $3, true), "
    "pg_catalog.set_config('jit', 'off', true)"
)

# The relation a request backend under row-level security reads each mapped table through:
# an owner-owned view gated on the policy role whose condition is the read predicate.
RLS_READ_VIEWS: Final[Mapping[str, str]] = MappingProxyType({'context_entries': 'context_entries_readable'})

_VIEW_NAME = re.compile(r'[a-z_][a-z0-9_]*')


@runtime_checkable
class SessionProfile(Protocol):
    """How a backend's connections and transactions present the caller's identity."""

    @property
    def wraps_reads(self) -> bool:
        """Whether reads run inside a transaction so a bound identity holds for them."""
        ...

    def session_statements(self) -> list[str]:
        """Return the session-level statements every pooled connection runs on acquire.

        Returns:
            The statements, in order; empty when the profile needs none.
        """
        ...

    async def bind(self, conn: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]', access: BackendAccess) -> None:
        """Present ``access`` to the database for the rest of the open transaction.

        Args:
            conn: The connection whose transaction is open.
            access: The identity the transaction runs as.
        """
        ...

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name the relation a plain read selects ``table``'s rows from.

        Args:
            table: A routable table.
            alias: The name the statement qualifies the table's columns with.

        Returns:
            The FROM or JOIN item to place in the statement.

        Raises:
            ValueError: If ``table`` is not routable or ``alias`` is not a lowercase
                identifier.
        """
        ...


class UnboundProfile:
    """The profile of a request backend whose database enforces no row-level security.

    The application predicate in every statement is the whole access control, so the
    profile sets no session state, wraps no read, binds nothing and names every table by
    itself: the backend runs exactly the statements its callers issue.
    """

    @property
    def wraps_reads(self) -> bool:
        """Reads run as issued, outside any transaction the profile would open."""
        return False

    def session_statements(self) -> list[str]:
        """Return no session statements.

        Returns:
            An empty list.
        """
        return []

    async def bind(self, conn: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]', access: BackendAccess) -> None:
        """Bind nothing: the database has no policy that reads an identity.

        Args:
            conn: The connection whose transaction is open.
            access: The identity the transaction runs as.
        """
        _ = conn
        _ = access

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name ``table`` by itself: no access view replaces it.

        Args:
            table: A routable table.
            alias: The name the statement qualifies the table's columns with.

        Returns:
            The base form of the relation (:func:`app.backends.base.base_relation`).
        """
        return base_relation(table, alias=alias)


class RlsRequestProfile:
    """The profile of a request backend whose database enforces row-level security.

    Every read runs in a transaction, and every transaction of a request scope starts with
    :data:`BIND_IDENTITY_STATEMENT`, so the policies and the read views see the caller. A
    scope without a principal (the system scope, maintenance) binds nothing, and the
    policies then match no private row: the backend fails closed. Plain reads of a mapped
    table name its read view; every other table keeps its own name.
    """

    def __init__(self, *, read_views: Mapping[str, str]) -> None:
        """Map each routable table to the view its plain reads select from.

        Args:
            read_views: Routable table name to read view name.

        Raises:
            ValueError: If a table is not routable or a view name is not a lowercase
                identifier.
        """
        for table, view in read_views.items():
            base_relation(table, alias=table)
            if _VIEW_NAME.fullmatch(view) is None:
                raise ValueError(f'{view!r} is not a valid read view name')
        self._read_views = dict(read_views)

    @property
    def wraps_reads(self) -> bool:
        """Reads run inside a transaction, so the bound identity holds for every statement."""
        return True

    def session_statements(self) -> list[str]:
        """Return no session statements: the identity is bound per transaction.

        Returns:
            An empty list.
        """
        return []

    async def bind(self, conn: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]', access: BackendAccess) -> None:
        """Bind a request scope's principal, groups and publish right for the open transaction.

        Args:
            conn: The connection whose transaction is open.
            access: The identity the transaction runs as; only a request scope binds.
        """
        if not isinstance(access, AccessScope):
            return
        await conn.execute(
            BIND_IDENTITY_STATEMENT,
            access.principal_id,
            access.groups_list(),
            'on' if access.may_publish else 'off',
        )

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name a mapped table's read view under the statement's alias and any other table by itself.

        Args:
            table: A routable table.
            alias: The name the statement qualifies the table's columns with.

        Returns:
            ``'<view> AS <alias>'`` for a mapped table, otherwise the base form.
        """
        relation = base_relation(table, alias=alias)
        view = self._read_views.get(table)
        return relation if view is None else f'{view} AS {alias}'


class MaintenanceProfile:
    """The profile of the backend that owns the tables: migrations, startup and the command line.

    It runs as the table owner, which row-level security without FORCE does not filter, so
    it binds nothing and reads the base tables. Every pooled connection turns
    ``row_security`` off, which makes any statement a policy would filter raise instead of
    returning a filtered result.
    """

    @property
    def wraps_reads(self) -> bool:
        """Reads run as issued, outside any transaction the profile would open."""
        return False

    def session_statements(self) -> list[str]:
        """Return the statement that turns row security off for the session.

        Returns:
            ``['SET row_security = off']``.
        """
        return ['SET row_security = off']

    async def bind(self, conn: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]', access: BackendAccess) -> None:
        """Bind nothing: the table owner is not subject to the policies.

        Args:
            conn: The connection whose transaction is open.
            access: The identity the transaction runs as.
        """
        _ = conn
        _ = access

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name ``table`` by itself: the owner reads the base tables.

        Args:
            table: A routable table.
            alias: The name the statement qualifies the table's columns with.

        Returns:
            The base form of the relation (:func:`app.backends.base.base_relation`).
        """
        return base_relation(table, alias=alias)
