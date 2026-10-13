"""Backend stubs that record the SQL the statistics repository sends, for tests that pin its statements.

:class:`RecordingPgBackend` selects the PostgreSQL branch of every statistics method and
answers each statement with an empty result, so a test reads the PostgreSQL SQL without a
server. :class:`RecordingSqliteBackend` wraps a real SQLite backend and records every
statement its cursors execute, so a test reads the SQLite SQL the same data produced.
Statements are recorded with their whitespace collapsed to single spaces.
"""

import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Sequence
from typing import Any
from typing import TypeVar

from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from app.backends.base import base_relation

T = TypeVar('T')


def _collapsed(query: str) -> str:
    """Return ``query`` with every run of whitespace collapsed to one space."""
    return ' '.join(query.split())


class RecordingPgConnection:
    """Async connection stub recording every statement the PostgreSQL paths emit."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    async def fetch(self, query: str, *_args: object) -> list[dict[str, object]]:
        """Record the statement and return no rows.

        Args:
            query: SQL text emitted by the repository.
            _args: Bound parameters (unused).

        Returns:
            An empty row list.
        """
        self.queries.append(_collapsed(query))
        return []

    async def fetchrow(self, query: str, *_args: object) -> dict[str, object] | None:
        """Record the statement and return no row.

        Args:
            query: SQL text emitted by the repository.
            _args: Bound parameters (unused).

        Returns:
            ``None``, which every caller tolerates as an empty result.
        """
        self.queries.append(_collapsed(query))
        return None

    async def fetchval(self, query: str, *_args: object) -> int:
        """Record the statement and return a zero scalar.

        Args:
            query: SQL text emitted by the repository.
            _args: Bound parameters (unused).

        Returns:
            Zero.
        """
        self.queries.append(_collapsed(query))
        return 0


class RecordingPgBackend:
    """Backend stub selecting the PostgreSQL branch and capturing its SQL."""

    backend_type = 'postgresql'

    def __init__(self) -> None:
        self.connection = RecordingPgConnection()

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name every table by its own name, as a backend without access views does."""
        return base_relation(table, alias=alias)

    async def execute_read(self, operation: Callable[[Any], Awaitable[T]], *, access_scope: BackendAccess) -> T:
        """Run the repository's async closure against the recording connection.

        Args:
            operation: The async closure the repository passes to the backend.
            access_scope: The identity the read runs as; the fake binds nothing.

        Returns:
            Whatever the closure returns.
        """
        _ = access_scope
        return await operation(self.connection)


class _RecordingCursor:
    """A SQLite cursor that records each statement before executing it."""

    def __init__(self, cursor: sqlite3.Cursor, queries: list[str]) -> None:
        self._cursor = cursor
        self._queries = queries

    def execute(self, query: str, params: Sequence[str] = ()) -> sqlite3.Cursor:
        """Record the statement and execute it on the real cursor.

        Args:
            query: SQL text emitted by the repository.
            params: Bound parameters.

        Returns:
            The real cursor.
        """
        self._queries.append(_collapsed(query))
        return self._cursor.execute(query, params)

    def __getattr__(self, name: str) -> object:
        return getattr(self._cursor, name)


class _RecordingConnection:
    """A SQLite connection whose cursors record what they execute."""

    def __init__(self, conn: sqlite3.Connection, queries: list[str]) -> None:
        self._conn = conn
        self._queries = queries

    def cursor(self) -> _RecordingCursor:
        """Return a recording cursor over a cursor of the real connection."""
        return _RecordingCursor(self._conn.cursor(), self._queries)

    def __getattr__(self, name: str) -> object:
        return getattr(self._conn, name)


class RecordingSqliteBackend:
    """A real SQLite backend whose reads record the statements the repository's cursors execute."""

    backend_type = 'sqlite'

    def __init__(self, backend: StorageBackend) -> None:
        self._backend = backend
        self.queries: list[str] = []

    def readable_relation(self, table: str, *, alias: str) -> str:
        """Name each table as the wrapped backend does."""
        return self._backend.readable_relation(table, alias=alias)

    async def execute_read(self, operation: Callable[[Any], T], *, access_scope: BackendAccess) -> T:
        """Run the repository's closure on the wrapped backend through a recording connection.

        Args:
            operation: The synchronous closure the repository passes to the backend.
            access_scope: The identity the read runs as.

        Returns:
            Whatever the closure returns.
        """

        def _recorded(conn: sqlite3.Connection) -> T:
            return operation(_RecordingConnection(conn, self.queries))

        return await self._backend.execute_read(_recorded, access_scope=access_scope)
