"""Two-principal cases of the grant statements (G1, G2).

A grant statement inserts a grant only on an entry the caller owns and records the caller
as the grantor: the owner predicate sits inside the INSERT, so a caller who does not own
the entry inserts nothing, whatever it may read. Each case adds one private entry of
carol's and grants read access on it to carol's own group, a grant every database policy
on the grants table lets carol both insert and read back, then reads the grant rows of
that entry through the seed channel. Each row reports the owner the grant records: its
``entry_owner_id`` on PostgreSQL, and the entry's own owner on SQLite, which keeps no copy.

- G1: ``store_group_read_grants`` on the caller's transaction stores the grant for the
  entry's owner and nothing for anyone else.
- G2: the statement ``_grant_insert_sql`` renders, run on the caller's transaction,
  inserts one row for the entry's owner and none for anyone else.
"""

import sqlite3
from typing import TYPE_CHECKING
from typing import cast

from app.access_scope import AccessScope
from app.access_scope import Scope
from app.backends.base import TransactionContext
from app.repositories.base import BaseRepository
from app.repositories.base import Statement
from tests.repositories._access_scope_cases import TEAM_GROUP
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import per_request_scope

if TYPE_CHECKING:
    import asyncpg

type GrantRow = tuple[str, str, str, str, str]
type InsertObservable = tuple[int, tuple[GrantRow, ...]]

GRANTED_LABEL = 'carol_private_granted'
GRANT_OWNER = 'carol'
# The grant carol's own read grant to her group stores: (type, grantee, permission, grantor, entry owner).
_CAROL_GROUP_GRANT: GrantRow = ('group', TEAM_GROUP, 'read', GRANT_OWNER, GRANT_OWNER)

_GRANT_ROWS_SQL = {
    'sqlite': (
        'SELECT g.principal_type, g.principal_id, g.permission, g.granted_by, ce.owner_id '
        'FROM context_entry_grants g JOIN context_entries ce ON ce.id = g.context_entry_id '
        'WHERE g.context_entry_id = ? ORDER BY g.principal_type, g.principal_id, g.permission'
    ),
    'postgresql': (
        'SELECT g.principal_type, g.principal_id, g.permission, g.granted_by, g.entry_owner_id '
        'FROM context_entry_grants g '
        'WHERE g.context_entry_id = ? ORDER BY g.principal_type, g.principal_id, g.permission'
    ),
}


async def _add_granted_entry(db: ScopedDb) -> None:
    """Add the private entry of carol's the cases grant access to."""
    await db.add_entry(GRANTED_LABEL, owner=GRANT_OWNER, visibility='private')


async def _grant_rows(db: ScopedDb) -> tuple[GrantRow, ...]:
    """Read every grant of the granted entry through the seed channel."""
    rows = await db.fetch_all(_GRANT_ROWS_SQL[db.seed_backend.backend_type], (db.ids[GRANTED_LABEL],))
    return tuple((str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])) for row in rows)


def _request_scope(scope: Scope) -> AccessScope:
    """Return the scope as the caller's own scope, which every grant statement requires."""
    assert isinstance(scope, AccessScope), scope
    return scope


async def _store_group_read_grant(db: ScopedDb, scope: Scope) -> tuple[GrantRow, ...]:
    """Grant carol's group read access on the entry as the scope, then report the entry's grants."""
    caller = _request_scope(scope)
    async with db.backend.begin_transaction(access_scope=caller) as txn:
        await db.repos.grants.store_group_read_grants(db.ids[GRANTED_LABEL], {TEAM_GROUP}, scope=caller, txn=txn)
    return await _grant_rows(db)


async def _inserted_rows(txn: TransactionContext, statement: Statement) -> int:
    """Run one INSERT on the transaction's connection and return how many rows it inserted."""
    sql, params = statement
    if txn.backend_type == 'sqlite':

        def _insert_sqlite(conn: sqlite3.Connection) -> int:
            return conn.execute(sql, params).rowcount

        return await BaseRepository._run_sqlite_txn(_insert_sqlite, cast(sqlite3.Connection, txn.connection))
    status = await cast('asyncpg.Connection', txn.connection).execute(sql, *params)
    return int(status.rsplit(' ', 1)[1])


async def _insert_through_the_builder(db: ScopedDb, scope: Scope) -> InsertObservable:
    """Run the rendered grant INSERT as the scope, then report the rows it inserted and the entry's grants."""
    caller = _request_scope(scope)
    statement = db.repos.grants._grant_insert_sql(
        db.backend.backend_type, db.ids[GRANTED_LABEL], 'group', TEAM_GROUP, 'read', scope=caller,
    )
    async with db.backend.begin_transaction(access_scope=caller) as txn:
        inserted = await _inserted_rows(txn, statement)
    return inserted, await _grant_rows(db)


def _expected_grants(scope_name: str) -> tuple[GrantRow, ...]:
    """The entry's owner stores the grant as its grantor; every other scope stores nothing."""
    return (_CAROL_GROUP_GRANT,) if scope_name == GRANT_OWNER else ()


def _expected_insert(scope_name: str) -> InsertObservable:
    """The entry's owner inserts the one row; every other scope inserts none."""
    grants = _expected_grants(scope_name)
    return len(grants), grants


def _grant_cases(layout: Layout, suffix: str) -> tuple[AccessCase, ...]:
    """The grant cases on one layout."""
    return (
        AccessCase(
            'G1', _store_group_read_grant, per_request_scope(_expected_grants),
            layout=layout, variant=suffix, setup=_add_granted_entry, reports_write_access=True,
        ),
        AccessCase(
            'G2', _insert_through_the_builder, per_request_scope(_expected_insert),
            layout=layout, variant=suffix, setup=_add_granted_entry, reports_write_access=True,
        ),
    )


GRANT_CASES: tuple[AccessCase, ...] = (*_grant_cases('compressed', ''), *_grant_cases('fp32', 'fp32'))
