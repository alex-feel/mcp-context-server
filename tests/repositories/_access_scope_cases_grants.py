"""Two-principal cases of the grant statements (G1 to G5).

A grant statement acts only on an entry the caller owns: the owner predicate sits inside every
INSERT, DELETE and SELECT, so a caller who does not own the entry stores, removes and lists
nothing, whatever it may read or write. An INSERT also records the caller as the grantor.
Every store and removal runs on a transaction of its own, as a tool runs it. The observables
read the grant rows of an entry through the seed channel; each row reports the owner the grant
records: its ``entry_owner_id`` on PostgreSQL, and the entry's own owner on SQLite, which keeps
no copy.

- G1: ``store_group_read_grants`` stores a group's read grant for the entry's owner and
  nothing for anyone else: on a private entry of carol's for her own group, and on alice's
  private entry for a group alice is not a member of (``foreign-group``).
- G2: the statement ``_grant_insert_sql`` renders inserts one row for the entry's owner and
  none for anyone else: carol's grant to her own group, alice's grant to a user
  (``user-grantee``) and alice's grant to a group she is not a member of (``foreign-group``).
- G3: ``store_grant`` stores a user's write grant twice on each of a private entry bob may
  write, a public entry carol may write and bob's own entry: the owner creates it once and
  finds it present the second time; every other caller, a write grantee included, stores
  nothing.
- G4: ``delete_grant`` removes each of bob's read and write grants, carol's write grant and a
  grant of carol's on bob's entry twice: the owner removes it once and finds it gone the
  second time; every other caller, the grantee itself included, removes nothing.
- G5: ``list_grants`` lists every grant of each seed entry, outside and inside a transaction:
  the owner sees each grant with its grantor and a canonical creation time, in principal
  type, principal id and permission order; every other caller sees none.

Under row-level security a grant to a user or to a group its grantor is not a member of is a
row only the owner arm of the grants policies admits, and the removals and listings of G4 and
G5 rest on that arm too; carol's grant to her own group is a row the grantee arm admits as well.
"""

import re
import sqlite3
from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Sequence
from typing import TYPE_CHECKING
from typing import cast

from app.access_scope import AccessScope
from app.access_scope import Scope
from app.backends.base import TransactionContext
from app.repositories.base import BaseRepository
from app.repositories.base import Statement
from app.repositories.grant_repository import GrantRecord
from tests.helpers import insert_grant
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import SEED_ROWS
from tests.repositories._access_scope_cases import TEAM_GROUP
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Grant
from tests.repositories._access_scope_cases import Invoker
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import Setup
from tests.repositories._access_scope_cases import per_request_scope

if TYPE_CHECKING:
    import asyncpg

type GrantRow = tuple[str, str, str, str, str]
type InsertObservable = tuple[int, tuple[GrantRow, ...]]
type ChangeObservable = tuple[str, bool, bool, tuple[GrantRow, ...]]
type ListedGrant = tuple[str, str, str, str, bool]
type ListObservable = tuple[str, tuple[ListedGrant, ...], tuple[ListedGrant, ...]]

GRANTED_LABEL = 'carol_private_granted'
GRANT_OWNER = 'carol'
# A private entry of alice's that carries no grant.
UNGRANTED_LABEL = 'alice_private'

_GROUP_READ = Grant('group', TEAM_GROUP, 'read')
_GROUP_WRITE = Grant('group', TEAM_GROUP, 'write')
_USER_READ = Grant('user', 'dave', 'read')
# The grant G3 stores.
_STORED_GRANT = Grant('user', 'dave', 'write')
# The grant the G4 and G5 cases add to bob's private entry, so that bob owns a grant too.
_BOB_GRANT = Grant('user', 'carol', 'read')

_OWNER = {**{row.label: row.owner for row in SEED_ROWS}, GRANTED_LABEL: GRANT_OWNER}
_SEEDED = {row.label: row.grants for row in SEED_ROWS}
_WITH_BOB_GRANT = {**_SEEDED, 'bob_private': (*_SEEDED['bob_private'], _BOB_GRANT)}

# G3 stores on a private entry bob may write, a public entry carol may write and bob's own entry.
_STORE_TARGETS = ('alice_private_bob_write', 'alice_public_carol_write', 'bob_private')
# G4 removes the one grant of each of these entries: bob's read and write grants, carol's write
# grant on a public entry and carol's read grant on bob's entry.
_DELETE_TARGETS = ('alice_private_bob_read', 'alice_private_bob_write', 'alice_public_carol_write', 'bob_private')

# The canonical UTC form every timestamp a tool returns takes.
_CANONICAL_TIMESTAMP = re.compile(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z')

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


def _stored(label: str, grants: Iterable[Grant]) -> tuple[GrantRow, ...]:
    """Return the rows ``grants`` store on the entry, each granted by the entry's owner, in listing order."""
    owner = _OWNER[label]
    return tuple(sorted((grant.principal_type, grant.principal_id, grant.permission, owner, owner) for grant in grants))


async def _add_granted_entry(db: ScopedDb) -> None:
    """Add the private entry of carol's the own-group cases grant access to."""
    await db.add_entry(GRANTED_LABEL, owner=GRANT_OWNER, visibility='private')


async def _grant_on_bob_private(db: ScopedDb) -> None:
    """Grant carol read access on bob's private entry, recorded as granted by bob."""
    await insert_grant(
        db.seed_backend, db.ids['bob_private'],
        _BOB_GRANT.principal_type, _BOB_GRANT.principal_id, _BOB_GRANT.permission, 'bob',
    )


async def _grant_rows(db: ScopedDb, label: str) -> tuple[GrantRow, ...]:
    """Read every grant of the entry through the seed channel."""
    rows = await db.fetch_all(_GRANT_ROWS_SQL[db.seed_backend.backend_type], (db.ids[label],))
    return tuple((str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])) for row in rows)


def _request_scope(scope: Scope) -> AccessScope:
    """Return the scope as the caller's own scope, which every grant statement requires."""
    assert isinstance(scope, AccessScope), scope
    return scope


def _group_read_grant_on(label: str) -> Invoker:
    """Return an invoker that grants carol's group read access on the entry, then reports its grants."""

    async def _store(db: ScopedDb, scope: Scope) -> tuple[GrantRow, ...]:
        caller = _request_scope(scope)
        async with db.backend.begin_transaction(access_scope=caller) as txn:
            await db.repos.grants.store_group_read_grants(db.ids[label], {TEAM_GROUP}, scope=caller, txn=txn)
        return await _grant_rows(db, label)

    return _store


async def _inserted_rows(txn: TransactionContext, statement: Statement) -> int:
    """Run one INSERT on the transaction's connection and return how many rows it inserted."""
    sql, params = statement
    if txn.backend_type == 'sqlite':

        def _insert_sqlite(conn: sqlite3.Connection) -> int:
            return conn.execute(sql, params).rowcount

        return await BaseRepository._run_sqlite_txn(_insert_sqlite, cast(sqlite3.Connection, txn.connection))
    status = await cast('asyncpg.Connection', txn.connection).execute(sql, *params)
    return int(status.rsplit(' ', 1)[1])


def _insert_through_the_builder(label: str, grant: Grant) -> Invoker:
    """Return an invoker that runs the rendered grant INSERT, then reports the rows it inserted and the entry's grants."""

    async def _insert(db: ScopedDb, scope: Scope) -> InsertObservable:
        caller = _request_scope(scope)
        statement = db.repos.grants._grant_insert_sql(
            db.backend.backend_type, db.ids[label], grant.principal_type, grant.principal_id, grant.permission,
            scope=caller,
        )
        async with db.backend.begin_transaction(access_scope=caller) as txn:
            inserted = await _inserted_rows(txn, statement)
        return inserted, await _grant_rows(db, label)

    return _insert


def _expected_grants(label: str, grant: Grant) -> Callable[[str], tuple[GrantRow, ...]]:
    """The entry's owner stores the grant as its grantor; every other scope stores nothing."""
    return lambda scope_name: _stored(label, (grant,)) if scope_name == _OWNER[label] else ()


def _expected_insert(label: str, grant: Grant) -> Callable[[str], InsertObservable]:
    """The entry's owner inserts the one row; every other scope inserts none."""

    def _expected(scope_name: str) -> InsertObservable:
        grants = _expected_grants(label, grant)(scope_name)
        return len(grants), grants

    return _expected


async def _store_twice(db: ScopedDb, scope: Scope) -> tuple[ChangeObservable, ...]:
    """Store the G3 grant twice on each target as the scope, then report what each call created and the grants."""
    caller = _request_scope(scope)
    observed: list[ChangeObservable] = []
    for label in _STORE_TARGETS:
        created: list[bool] = []
        for _attempt in range(2):
            async with db.backend.begin_transaction(access_scope=caller) as txn:
                created.append(await db.repos.grants.store_grant(
                    db.ids[label], _STORED_GRANT.principal_type, _STORED_GRANT.principal_id, _STORED_GRANT.permission,
                    scope=caller, txn=txn,
                ))
        observed.append((label, created[0], created[1], await _grant_rows(db, label)))
    return tuple(observed)


def _expected_store(scope_name: str) -> tuple[ChangeObservable, ...]:
    """The owner creates the grant once beside the seeded grants; every other scope leaves the seeded grants alone."""
    return tuple(
        (label, True, False, _stored(label, (*_SEEDED[label], _STORED_GRANT)))
        if scope_name == _OWNER[label]
        else (label, False, False, _stored(label, _SEEDED[label]))
        for label in _STORE_TARGETS
    )


async def _delete_twice(db: ScopedDb, scope: Scope) -> tuple[ChangeObservable, ...]:
    """Remove the one grant of each G4 target twice as the scope, then report what each call removed and the grants."""
    caller = _request_scope(scope)
    observed: list[ChangeObservable] = []
    for label in _DELETE_TARGETS:
        (grant,) = _WITH_BOB_GRANT[label]
        removed: list[bool] = []
        for _attempt in range(2):
            async with db.backend.begin_transaction(access_scope=caller) as txn:
                removed.append(await db.repos.grants.delete_grant(
                    db.ids[label], grant.principal_type, grant.principal_id, grant.permission, scope=caller, txn=txn,
                ))
        observed.append((label, removed[0], removed[1], await _grant_rows(db, label)))
    return tuple(observed)


def _expected_delete(scope_name: str) -> tuple[ChangeObservable, ...]:
    """The owner removes the grant once and leaves none; every other scope removes nothing and the grant survives."""
    return tuple(
        (label, True, False, ())
        if scope_name == _OWNER[label]
        else (label, False, False, _stored(label, _WITH_BOB_GRANT[label]))
        for label in _DELETE_TARGETS
    )


def _listed(records: Sequence[GrantRecord]) -> tuple[ListedGrant, ...]:
    """Return each record's key and grantor, and whether its creation time is canonical."""
    return tuple(
        (
            record.principal_type, record.principal_id, record.permission, record.granted_by,
            _CANONICAL_TIMESTAMP.fullmatch(record.created_at) is not None,
        )
        for record in records
    )


async def _list_outside_and_inside(db: ScopedDb, scope: Scope) -> tuple[ListObservable, ...]:
    """List the grants of every seed entry as the scope, on a connection of its own and on a transaction."""
    caller = _request_scope(scope)
    observed: list[ListObservable] = []
    for label in SEED_LABELS:
        outside = await db.repos.grants.list_grants(db.ids[label], scope=caller)
        async with db.backend.begin_transaction(access_scope=caller) as txn:
            inside = await db.repos.grants.list_grants(db.ids[label], scope=caller, txn=txn)
        observed.append((label, _listed(outside), _listed(inside)))
    return tuple(observed)


def _expected_listing(scope_name: str) -> tuple[ListObservable, ...]:
    """The owner lists every grant of its entries, granted by itself; every other scope lists none."""

    def _owned(label: str) -> tuple[ListedGrant, ...]:
        if scope_name != _OWNER[label]:
            return ()
        return tuple(
            (principal_type, principal_id, permission, granted_by, True)
            for principal_type, principal_id, permission, granted_by, _owner in _stored(label, _WITH_BOB_GRANT[label])
        )

    return tuple((label, _owned(label), _owned(label)) for label in SEED_LABELS)


def _grant_cases(layout: Layout, suffix: str) -> tuple[AccessCase, ...]:
    """The grant cases on one layout."""

    def _variant(name: str) -> str:
        return '-'.join(part for part in (name, suffix) if part)

    def _case(
        case_id: str, invoker: Invoker, build: Callable[[str], object], *, variant: str = '', setup: Setup | None = None,
    ) -> AccessCase:
        return AccessCase(
            case_id, invoker, per_request_scope(build),
            layout=layout, variant=_variant(variant), setup=setup, reports_write_access=True,
        )

    return (
        _case(
            'G1', _group_read_grant_on(GRANTED_LABEL), _expected_grants(GRANTED_LABEL, _GROUP_READ),
            setup=_add_granted_entry,
        ),
        _case(
            'G1', _group_read_grant_on(UNGRANTED_LABEL), _expected_grants(UNGRANTED_LABEL, _GROUP_READ),
            variant='foreign-group',
        ),
        _case(
            'G2', _insert_through_the_builder(GRANTED_LABEL, _GROUP_READ), _expected_insert(GRANTED_LABEL, _GROUP_READ),
            setup=_add_granted_entry,
        ),
        _case(
            'G2', _insert_through_the_builder(UNGRANTED_LABEL, _USER_READ), _expected_insert(UNGRANTED_LABEL, _USER_READ),
            variant='user-grantee',
        ),
        _case(
            'G2', _insert_through_the_builder(UNGRANTED_LABEL, _GROUP_WRITE),
            _expected_insert(UNGRANTED_LABEL, _GROUP_WRITE), variant='foreign-group',
        ),
        _case('G3', _store_twice, _expected_store),
        _case('G4', _delete_twice, _expected_delete, setup=_grant_on_bob_private),
        _case('G5', _list_outside_and_inside, _expected_listing, setup=_grant_on_bob_private),
    )


GRANT_CASES: tuple[AccessCase, ...] = (*_grant_cases('compressed', ''), *_grant_cases('fp32', 'fp32'))
