"""Tests for the access scope types and the access predicate builder.

Covers app.access_scope: the exact predicate text per mode, backend and outer
qualifier; the parameter shape every predicate guarantees (a constant bind
count, one group bind whatever the group count, contiguous PostgreSQL
placeholders from ``start``); the empty SystemScope predicate; the
readable-parent predicate that tests a child row's parent key for membership in
the readable entries; the scope value objects; which rows each predicate admits
in a real SQLite database seeded with owners, visibilities and grants; and the
module's standard-library-only imports.
"""

import ast
import dataclasses
import itertools
import json
import re
import sqlite3
import sys
from pathlib import Path

import pytest

import app.access_scope
from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessMode
from app.access_scope import AccessPredicate
from app.access_scope import AccessScope
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.access_scope import build_readable_parent_predicate
from app.backends import StorageBackend
from app.ids import generate_id
from tests.helpers import insert_grant
from tests.helpers import read_grants

BOB_TEAMS = AccessScope('bob', frozenset({'team-y', 'team-x'}))

# ============================================================================
# Exact predicate text
# ============================================================================

_EXACT_CASES = [
    pytest.param(
        AccessMode.READ, 'sqlite', 'context_entries', 1,
        "(context_entries.owner_id = ? OR context_entries.visibility = 'public' OR EXISTS (SELECT 1 FROM "
        "context_entry_grants g WHERE g.context_entry_id = context_entries.id AND g.permission IN ('read', 'write') "
        "AND ((g.principal_type = 'user' AND g.principal_id = ?) OR (g.principal_type = 'group' AND g.principal_id "
        'IN (SELECT value FROM json_each(?))))))',
        ['bob', 'bob', '["team-x", "team-y"]'],
        id='read-sqlite-context_entries',
    ),
    pytest.param(
        AccessMode.READ, 'sqlite', 'ce', 5,
        "(ce.owner_id = ? OR ce.visibility = 'public' OR EXISTS (SELECT 1 FROM context_entry_grants g WHERE "
        "g.context_entry_id = ce.id AND g.permission IN ('read', 'write') AND ((g.principal_type = 'user' AND "
        "g.principal_id = ?) OR (g.principal_type = 'group' AND g.principal_id IN (SELECT value FROM json_each(?))))))",
        ['bob', 'bob', '["team-x", "team-y"]'],
        id='read-sqlite-ce',
    ),
    pytest.param(
        AccessMode.READ, 'postgresql', 'context_entries', 1,
        "(context_entries.owner_id = $1 OR context_entries.visibility = 'public' OR EXISTS (SELECT 1 FROM "
        "context_entry_grants g WHERE g.context_entry_id = context_entries.id AND g.permission IN ('read', 'write') "
        "AND ((g.principal_type = 'user' AND g.principal_id = $2) OR (g.principal_type = 'group' AND g.principal_id "
        '= ANY($3::text[])))))',
        ['bob', 'bob', ['team-x', 'team-y']],
        id='read-postgresql-context_entries',
    ),
    pytest.param(
        AccessMode.READ, 'postgresql', 'ce', 5,
        "(ce.owner_id = $5 OR ce.visibility = 'public' OR EXISTS (SELECT 1 FROM context_entry_grants g WHERE "
        "g.context_entry_id = ce.id AND g.permission IN ('read', 'write') AND ((g.principal_type = 'user' AND "
        "g.principal_id = $6) OR (g.principal_type = 'group' AND g.principal_id = ANY($7::text[])))))",
        ['bob', 'bob', ['team-x', 'team-y']],
        id='read-postgresql-ce',
    ),
    pytest.param(
        AccessMode.WRITE, 'sqlite', 'context_entries', 1,
        "(context_entries.owner_id = ? OR EXISTS (SELECT 1 FROM context_entry_grants g WHERE g.context_entry_id = "
        "context_entries.id AND g.permission = 'write' AND ((g.principal_type = 'user' AND g.principal_id = ?) OR "
        "(g.principal_type = 'group' AND g.principal_id IN (SELECT value FROM json_each(?))))))",
        ['bob', 'bob', '["team-x", "team-y"]'],
        id='write-sqlite-context_entries',
    ),
    pytest.param(
        AccessMode.WRITE, 'sqlite', 'ce', 5,
        "(ce.owner_id = ? OR EXISTS (SELECT 1 FROM context_entry_grants g WHERE g.context_entry_id = ce.id AND "
        "g.permission = 'write' AND ((g.principal_type = 'user' AND g.principal_id = ?) OR (g.principal_type = "
        "'group' AND g.principal_id IN (SELECT value FROM json_each(?))))))",
        ['bob', 'bob', '["team-x", "team-y"]'],
        id='write-sqlite-ce',
    ),
    pytest.param(
        AccessMode.WRITE, 'postgresql', 'context_entries', 1,
        "(context_entries.owner_id = $1 OR EXISTS (SELECT 1 FROM context_entry_grants g WHERE g.context_entry_id = "
        "context_entries.id AND g.permission = 'write' AND ((g.principal_type = 'user' AND g.principal_id = $2) OR "
        "(g.principal_type = 'group' AND g.principal_id = ANY($3::text[])))))",
        ['bob', 'bob', ['team-x', 'team-y']],
        id='write-postgresql-context_entries',
    ),
    pytest.param(
        AccessMode.WRITE, 'postgresql', 'ce', 5,
        "(ce.owner_id = $5 OR EXISTS (SELECT 1 FROM context_entry_grants g WHERE g.context_entry_id = ce.id AND "
        "g.permission = 'write' AND ((g.principal_type = 'user' AND g.principal_id = $6) OR (g.principal_type = "
        "'group' AND g.principal_id = ANY($7::text[])))))",
        ['bob', 'bob', ['team-x', 'team-y']],
        id='write-postgresql-ce',
    ),
    pytest.param(
        AccessMode.OWNER, 'sqlite', 'context_entries', 1, 'context_entries.owner_id = ?', ['bob'],
        id='owner-sqlite-context_entries',
    ),
    pytest.param(AccessMode.OWNER, 'sqlite', 'ce', 5, 'ce.owner_id = ?', ['bob'], id='owner-sqlite-ce'),
    pytest.param(
        AccessMode.OWNER, 'postgresql', 'context_entries', 1, 'context_entries.owner_id = $1', ['bob'],
        id='owner-postgresql-context_entries',
    ),
    pytest.param(AccessMode.OWNER, 'postgresql', 'ce', 5, 'ce.owner_id = $5', ['bob'], id='owner-postgresql-ce'),
]


@pytest.mark.parametrize(('mode', 'backend_type', 'outer', 'start', 'expected_sql', 'expected_params'), _EXACT_CASES)
def test_exact_predicate_text(
    mode: AccessMode,
    backend_type: str,
    outer: str,
    start: int,
    expected_sql: str,
    expected_params: list[str | list[str]],
) -> None:
    """Each (mode, backend, outer) renders the documented SQL and binds in placeholder order."""
    predicate = build_access_predicate(BOB_TEAMS, mode=mode, backend_type=backend_type, outer=outer, start=start)

    assert predicate.sql == expected_sql
    assert predicate.params == expected_params
    assert predicate.and_clause() == f' AND {expected_sql}'
    assert predicate.where_clause() == f' WHERE {expected_sql}'


# ============================================================================
# Parameter-shape invariants over every combination
# ============================================================================

_GROUP_SETS = {
    0: frozenset[str](),
    1: frozenset({'team-x'}),
    2: frozenset({'team-y', 'team-x'}),
    500: frozenset(f'group-{index:03d}' for index in range(500)),
}
_EXPECTED_BINDS = {AccessMode.READ: 3, AccessMode.WRITE: 3, AccessMode.OWNER: 1}
_SHAPE_CASES = [
    pytest.param(mode, backend_type, group_count, start, outer, id=f'{mode}-{backend_type}-g{group_count}-s{start}-{outer}')
    for mode, backend_type, group_count, start, outer in itertools.product(
        ['read', 'write', 'owner', 'system'],
        ['sqlite', 'postgresql'],
        sorted(_GROUP_SETS),
        [1, 5],
        ['context_entries', 'ce'],
    )
]


def _assert_system_predicate(predicate: AccessPredicate) -> None:
    """Assert the predicate restricts nothing and binds nothing."""
    assert predicate.sql == ''
    assert predicate.params == []
    assert predicate.bind_count == 0
    assert predicate.and_clause() == ''
    assert predicate.where_clause() == ''


@pytest.mark.parametrize(('mode_name', 'backend_type', 'group_count', 'start', 'outer'), _SHAPE_CASES)
def test_predicate_parameter_shape(mode_name: str, backend_type: str, group_count: int, start: int, outer: str) -> None:
    """Bind count, placeholder numbering and the single group bind hold for every combination."""
    groups = _GROUP_SETS[group_count]

    if mode_name == 'system':
        for mode in AccessMode:
            predicate = build_access_predicate(SYSTEM_SCOPE, mode=mode, backend_type=backend_type, outer=outer, start=start)
            _assert_system_predicate(predicate)
        return

    mode = AccessMode(mode_name)
    scope = AccessScope('principal-p', groups)
    predicate = build_access_predicate(scope, mode=mode, backend_type=backend_type, outer=outer, start=start)
    expected_binds = _EXPECTED_BINDS[mode]

    assert predicate.bind_count == len(predicate.params) == expected_binds
    if backend_type == 'postgresql':
        assert '?' not in predicate.sql
        numbers = [int(number) for number in re.findall(r'\$(\d+)', predicate.sql)]
        assert numbers == list(range(start, start + expected_binds))
    else:
        assert '$' not in predicate.sql
        assert predicate.sql.count('?') == expected_binds

    assert predicate.params[0] == 'principal-p'
    assert f'{outer}.owner_id = ' in predicate.sql
    assert 'shared' not in predicate.sql
    assert predicate.and_clause() == f' AND {predicate.sql}'
    assert predicate.where_clause() == f' WHERE {predicate.sql}'

    if mode is AccessMode.OWNER:
        assert 'context_entry_grants' not in predicate.sql
        return

    assert predicate.params[1] == 'principal-p'
    assert f'g.context_entry_id = {outer}.id' in predicate.sql
    group_bind = predicate.params[2]
    if backend_type == 'postgresql':
        assert predicate.sql.count('= ANY(') == 1
        assert group_bind == sorted(groups)
    else:
        assert predicate.sql.count('json_each(') == 1
        assert isinstance(group_bind, str)
        assert json.loads(group_bind) == sorted(groups)
    assert ' IN ()' not in predicate.sql


# ============================================================================
# Readable-parent predicate
# ============================================================================


def test_readable_parent_predicate_exact_text() -> None:
    """The child key is tested for membership in the keys of the entries the READ predicate admits."""
    predicate = build_readable_parent_predicate(
        BOB_TEAMS, child_key='t.context_entry_id', parent_key='id', backend_type='sqlite',
    )

    assert predicate.sql == (
        "t.context_entry_id IN (SELECT ce.id FROM context_entries ce WHERE (ce.owner_id = ? OR ce.visibility = 'public' "
        "OR EXISTS (SELECT 1 FROM context_entry_grants g WHERE g.context_entry_id = ce.id AND g.permission IN ('read', "
        "'write') AND ((g.principal_type = 'user' AND g.principal_id = ?) OR (g.principal_type = 'group' AND "
        'g.principal_id IN (SELECT value FROM json_each(?)))))))'
    )
    assert predicate.params == ['bob', 'bob', '["team-x", "team-y"]']
    assert predicate.where_clause() == f' WHERE {predicate.sql}'
    assert predicate.and_clause() == f' AND {predicate.sql}'


@pytest.mark.parametrize(
    ('backend_type', 'start', 'child_key', 'parent_key'),
    [
        pytest.param('sqlite', 1, 'd.id', 'rowid_int', id='sqlite-rowid_int'),
        pytest.param('postgresql', 1, 'n.context_id', 'id', id='postgresql-start-1'),
        pytest.param('postgresql', 4, 'i.context_entry_id', 'id', id='postgresql-start-4'),
    ],
)
def test_readable_parent_predicate_wraps_the_read_predicate(
    backend_type: str, start: int, child_key: str, parent_key: str,
) -> None:
    """The subquery carries the READ predicate on ``ce`` unchanged, with its parameters and numbering."""
    read = build_access_predicate(BOB_TEAMS, mode=AccessMode.READ, backend_type=backend_type, outer='ce', start=start)

    predicate = build_readable_parent_predicate(
        BOB_TEAMS, child_key=child_key, parent_key=parent_key, backend_type=backend_type, start=start,
    )

    assert predicate.sql == f'{child_key} IN (SELECT ce.{parent_key} FROM context_entries ce WHERE {read.sql})'
    assert predicate.params == read.params
    assert predicate.bind_count == _EXPECTED_BINDS[AccessMode.READ]


@pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
def test_readable_parent_predicate_is_empty_for_the_system_scope(backend_type: str) -> None:
    """The system scope admits every child row, so the predicate restricts nothing."""
    predicate = build_readable_parent_predicate(
        SYSTEM_SCOPE, child_key='t.context_entry_id', parent_key='id', backend_type=backend_type,
    )

    _assert_system_predicate(predicate)


# ============================================================================
# Scope value objects
# ============================================================================


class TestAccessScope:
    """Tests for the AccessScope value object."""

    def test_groups_list_is_sorted(self) -> None:
        """groups_list returns the groups in sorted order."""
        scope = AccessScope('alice', frozenset({'team-c', 'team-a', 'team-b'}))
        assert scope.groups_list() == ['team-a', 'team-b', 'team-c']

    def test_groups_json_is_sorted_and_stable(self) -> None:
        """groups_json is the sorted JSON array, identical whatever order the groups were built in."""
        forward = AccessScope('alice', frozenset(f'group-{index:03d}' for index in range(500)))
        backward = AccessScope('alice', frozenset(f'group-{index:03d}' for index in reversed(range(500))))

        assert forward.groups_json() == json.dumps(sorted(forward.groups))
        assert forward.groups_json() == backward.groups_json()
        assert forward.groups_json() == forward.groups_json()

    def test_empty_groups_bind_an_empty_array(self) -> None:
        """An empty group set encodes as an empty JSON array, which matches no grant row."""
        scope = AccessScope('alice', frozenset())
        assert scope.groups_list() == []
        assert scope.groups_json() == '[]'

    def test_equality_and_hashing(self) -> None:
        """Scopes compare and hash by principal and group set."""
        first = AccessScope('alice', frozenset({'team-x', 'team-y'}))
        same = AccessScope('alice', frozenset({'team-y', 'team-x'}))

        assert first == same
        assert hash(first) == hash(same)
        assert len({first, same}) == 1
        assert first != AccessScope('bob', frozenset({'team-x', 'team-y'}))
        assert first != AccessScope('alice', frozenset({'team-x'}))

    def test_scope_is_immutable(self) -> None:
        """AccessScope is a frozen value object."""
        scope = AccessScope('alice', frozenset())
        field_name = 'principal_id'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(scope, field_name, 'mallory')


# ============================================================================
# Row admission against a real SQLite database
# ============================================================================


async def _insert_entry(backend: StorageBackend, label: str, owner_id: str, visibility: str) -> str:
    """Insert one context entry through raw SQL and return its id."""
    context_id = generate_id()

    def _insert(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
            "VALUES (?, 'scope-thread', 'agent', 'text', ?, ?, ?)",
            (context_id, label, owner_id, visibility),
        )

    await backend.execute_write(_insert)
    return context_id


async def _seed(backend: StorageBackend) -> dict[str, str]:
    """Seed the owner, visibility and grant combinations; return label to id."""
    ids = {
        'alice-private': await _insert_entry(backend, 'alice-private', 'alice', 'private'),
        'alice-public': await _insert_entry(backend, 'alice-public', 'alice', 'public'),
        'alice-private-read-bob': await _insert_entry(backend, 'alice-private-read-bob', 'alice', 'private'),
        'alice-private-read-team-x': await _insert_entry(backend, 'alice-private-read-team-x', 'alice', 'private'),
        'alice-private-write-bob': await _insert_entry(backend, 'alice-private-write-bob', 'alice', 'private'),
        'alice-public-write-carol': await _insert_entry(backend, 'alice-public-write-carol', 'alice', 'public'),
        'alice-public-read-dave': await _insert_entry(backend, 'alice-public-read-dave', 'alice', 'public'),
        'bob-private': await _insert_entry(backend, 'bob-private', 'bob', 'private'),
    }
    await insert_grant(backend, ids['alice-private-read-bob'], 'user', 'bob', 'read', 'alice')
    await insert_grant(backend, ids['alice-private-read-team-x'], 'group', 'team-x', 'read', 'alice')
    await insert_grant(backend, ids['alice-private-write-bob'], 'user', 'bob', 'write', 'alice')
    await insert_grant(backend, ids['alice-public-write-carol'], 'user', 'carol', 'write', 'alice')
    await insert_grant(backend, ids['alice-public-read-dave'], 'user', 'dave', 'read', 'alice')
    return ids


async def _admitted(backend: StorageBackend, scope: Scope, mode: AccessMode, outer: str) -> set[str]:
    """Return the labels of the rows the predicate admits, composed after a leading bind."""
    predicate = build_access_predicate(scope, mode=mode, backend_type='sqlite', outer=outer)
    alias = '' if outer == 'context_entries' else f' {outer}'
    query = f'SELECT {outer}.text_content FROM context_entries{alias} WHERE {outer}.thread_id = ?{predicate.and_clause()}'

    def _select(conn: sqlite3.Connection) -> set[str]:
        return {row[0] for row in conn.execute(query, ('scope-thread', *predicate.params)).fetchall()}

    return await backend.execute_read(_select)


_ALL_LABELS = {
    'alice-private',
    'alice-public',
    'alice-private-read-bob',
    'alice-private-read-team-x',
    'alice-private-write-bob',
    'alice-public-write-carol',
    'alice-public-read-dave',
    'bob-private',
}
_ALICE_LABELS = _ALL_LABELS - {'bob-private'}
_PUBLIC_LABELS = {'alice-public', 'alice-public-write-carol', 'alice-public-read-dave'}

_ADMISSION_CASES = [
    pytest.param(AccessScope('alice', frozenset()), AccessMode.READ, _ALICE_LABELS, id='alice-read'),
    pytest.param(AccessScope('alice', frozenset()), AccessMode.WRITE, _ALICE_LABELS, id='alice-write'),
    pytest.param(AccessScope('alice', frozenset()), AccessMode.OWNER, _ALICE_LABELS, id='alice-owner'),
    pytest.param(
        AccessScope('bob', frozenset()), AccessMode.READ,
        _PUBLIC_LABELS | {'alice-private-read-bob', 'alice-private-write-bob', 'bob-private'},
        id='bob-read',
    ),
    pytest.param(
        AccessScope('bob', frozenset()), AccessMode.WRITE, {'alice-private-write-bob', 'bob-private'}, id='bob-write',
    ),
    pytest.param(AccessScope('bob', frozenset()), AccessMode.OWNER, {'bob-private'}, id='bob-owner'),
    pytest.param(
        AccessScope('carol', frozenset({'team-x'})), AccessMode.READ, _PUBLIC_LABELS | {'alice-private-read-team-x'},
        id='carol-read',
    ),
    pytest.param(
        AccessScope('carol', frozenset({'team-x'})), AccessMode.WRITE, {'alice-public-write-carol'}, id='carol-write',
    ),
    pytest.param(AccessScope('carol', frozenset({'team-x'})), AccessMode.OWNER, set[str](), id='carol-owner'),
    pytest.param(AccessScope('dave', frozenset()), AccessMode.READ, _PUBLIC_LABELS, id='dave-read'),
    pytest.param(AccessScope('dave', frozenset()), AccessMode.WRITE, set[str](), id='dave-write'),
    pytest.param(SYSTEM_SCOPE, AccessMode.READ, _ALL_LABELS, id='system-read'),
    pytest.param(SYSTEM_SCOPE, AccessMode.OWNER, _ALL_LABELS, id='system-owner'),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('scope', 'mode', 'expected'), _ADMISSION_CASES)
@pytest.mark.parametrize('outer', ['context_entries', 'ce'])
async def test_predicate_admits_rows_on_sqlite(
    async_db_initialized: StorageBackend,
    scope: Scope,
    mode: AccessMode,
    expected: set[str],
    outer: str,
) -> None:
    """Owner, public, user-grant and group-grant arms admit exactly the rows the access model allows."""
    await _seed(async_db_initialized)

    assert await _admitted(async_db_initialized, scope, mode, outer) == expected


_READ_ADMISSION_CASES = [
    pytest.param(case.values[0], case.values[2], id=case.id) for case in _ADMISSION_CASES if case.values[1] is AccessMode.READ
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('scope', 'expected'), _READ_ADMISSION_CASES)
@pytest.mark.parametrize(
    ('child_select', 'child_key', 'parent_key'),
    [
        pytest.param('SELECT t.tag FROM tags t', 't.context_entry_id', 'id', id='tags-by-id'),
        pytest.param('SELECT x.text_content FROM context_entries x', 'x.rowid_int', 'rowid_int', id='rows-by-rowid_int'),
    ],
)
async def test_readable_parent_predicate_admits_children_of_readable_rows(
    async_db_initialized: StorageBackend,
    scope: Scope,
    expected: set[str],
    child_select: str,
    child_key: str,
    parent_key: str,
) -> None:
    """A child row is admitted exactly when the READ predicate admits the entry its key names."""
    ids = await _seed(async_db_initialized)

    def _tag_every_entry(conn: sqlite3.Connection) -> None:
        conn.executemany(
            'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', [(entry_id, label) for label, entry_id in ids.items()],
        )

    await async_db_initialized.execute_write(_tag_every_entry)
    predicate = build_readable_parent_predicate(scope, child_key=child_key, parent_key=parent_key, backend_type='sqlite')

    def _select(conn: sqlite3.Connection) -> set[str]:
        return {row[0] for row in conn.execute(f'{child_select}{predicate.where_clause()}', predicate.params).fetchall()}

    assert await async_db_initialized.execute_read(_select) == expected


@pytest.mark.asyncio
async def test_grant_helpers_round_trip(async_db_initialized: StorageBackend) -> None:
    """insert_grant writes grant rows that read_grants returns in a deterministic order."""
    ids = await _seed(async_db_initialized)
    await insert_grant(async_db_initialized, ids['alice-private'], 'user', 'dave', 'write', 'alice')
    await insert_grant(async_db_initialized, ids['alice-private'], 'group', 'team-x', 'read', 'alice')

    assert await read_grants(async_db_initialized, ids['alice-private']) == [
        ('group', 'team-x', 'read', 'alice'),
        ('user', 'dave', 'write', 'alice'),
    ]
    assert await read_grants(async_db_initialized, ids['bob-private']) == []


# ============================================================================
# Module boundary
# ============================================================================


def test_module_imports_only_the_standard_library() -> None:
    """app.access_scope stays a leaf: repositories import it without pulling in settings or fastmcp."""
    module_file = app.access_scope.__file__
    assert module_file is not None
    tree = ast.parse(Path(module_file).read_text(encoding='utf-8'))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert node.level == 0
            imported.add(node.module.partition('.')[0])

    assert imported
    assert imported <= sys.stdlib_module_names
