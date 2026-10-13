"""A stand-in for the raw PostgreSQL connections of the CLI unit tests that answers the owner precondition.

Every raw connection a maintenance mode opens runs
:func:`app.cli._backend.assert_maintenance_connection` before any data statement. The
fake connections of the unit tests inherit :class:`AnswersAsOwner`, so that check sees
the table owner, whom row-level security filters on no table.
"""

from app.cli._backend import FILTERED_TABLES_SQL

OWNER_ANSWER: tuple[str, list[str]] = ('mcp_owner', [])


class AnswersAsOwner:
    """Answers the owner precondition as the table owner; any other ``fetchrow`` is a test error."""

    async def fetchrow(self, query: str, *args: object) -> tuple[str, list[str]]:
        """Answer the precondition: the table owner, filtered on no table.

        Args:
            query: The statement; only the precondition is expected.
            args: Its parameters, the schema-qualified managed tables.

        Returns:
            The connected role and the tables row-level security filters for it.
        """
        assert query == FILTERED_TABLES_SQL, f'unexpected fetchrow: {query!r}'
        assert len(args) == 1
        return OWNER_ANSWER
