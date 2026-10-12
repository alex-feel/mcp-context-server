"""Session profiles the row-level security tests run repositories on.

:class:`PinnedIdentityProfile` behaves like the request profile of a row-level security
deployment, except that every transaction binds one fixed identity whatever access the
repository passes. A repository called with ``scope=SYSTEM_SCOPE`` renders no application
predicate, so on a backend with this profile the rows a read returns are the database's
answer alone for the pinned identity: the read view's for a plain read, the policies' for
anything else. :func:`pinned_repositories` opens such a backend as the request login role.
"""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import asyncpg

from app.access_scope import AccessScope
from app.backends.base import BackendAccess
from app.backends.postgresql_backend import PostgreSQLBackend
from app.backends.postgresql_backend.session import NO_CREDENTIAL_OVERRIDE
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.backends.postgresql_backend.session_profiles import RlsRequestProfile
from app.repositories import RepositoryContainer


class PinnedIdentityProfile:
    """The row-level security request profile with one identity bound in every transaction.

    Reads run in a transaction and plain reads of ``context_entries`` name its read view,
    as on a request backend under row-level security; the bind ignores the transaction's
    own access and binds the pinned identity instead, or nothing when none is pinned.
    """

    def __init__(self, scope: AccessScope | None) -> None:
        """Pin ``scope`` as the identity of every transaction.

        Args:
            scope: The identity to bind; None binds nothing, as a transaction without a
                principal does.
        """
        self._scope = scope
        self._request = RlsRequestProfile(read_views=RLS_READ_VIEWS)

    @property
    def wraps_reads(self) -> bool:
        """Reads run inside a transaction, so the pinned identity holds for every statement."""
        return self._request.wraps_reads

    def session_statements(self) -> list[str]:
        """Return the request profile's session statements.

        Returns:
            The statements every pooled connection runs on acquire.
        """
        return self._request.session_statements()

    async def bind(self, conn: 'asyncpg.pool.PoolConnectionProxy[asyncpg.Record]', access: BackendAccess) -> None:
        """Bind the pinned identity for the open transaction, whatever ``access`` is.

        Args:
            conn: The connection whose transaction is open.
            access: The access the repository passed, which the pinned identity replaces.
        """
        _ = access
        if self._scope is not None:
            await self._request.bind(conn, self._scope)

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name the relation a plain read selects ``table`` from, as the request profile does.

        Args:
            table: A routable table.
            alias: The name the statement qualifies the table's columns with.

        Returns:
            The read view under ``alias`` for a mapped table, otherwise the base form.
        """
        return self._request.readable_relation(table, alias=alias)


@asynccontextmanager
async def pinned_repositories(url: str, scope: AccessScope | None) -> AsyncGenerator[RepositoryContainer]:
    """Open repositories over a backend that binds ``scope`` in every transaction.

    Args:
        url: Connection string of the request login role.
        scope: The identity to pin; None binds nothing.

    Yields:
        The repositories; their backend is shut down afterwards.
    """
    backend = PostgreSQLBackend(
        connection_string=url,
        profile=PinnedIdentityProfile(scope),
        credentials=NO_CREDENTIAL_OVERRIDE,
    )
    await backend.initialize()
    try:
        yield RepositoryContainer(backend)
    finally:
        await backend.shutdown()
