"""Session profiles: how a PostgreSQL backend presents the caller's identity to the database.

A profile is injected into the backend at construction. The write entry points hand
every transaction's identity to :meth:`SessionProfile.bind` as the first statement inside
the transaction, so a profile that enforces row-level security can scope the transaction
to the caller and a profile that does not can leave it untouched.
"""

from typing import Protocol
from typing import runtime_checkable

import asyncpg

from app.backends.base import BackendAccess


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


class UnboundProfile:
    """The profile of a request backend whose database enforces no row-level security.

    The application predicate in every statement is the whole access control, so the
    profile sets no session state, wraps no read and binds nothing: the backend runs
    exactly the statements its callers issue.
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
