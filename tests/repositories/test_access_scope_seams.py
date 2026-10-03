"""Two-principal access-scope cases on SQLite, and the checks of the case scaffold itself.

Every case of :mod:`tests.repositories._access_scope_cases` runs once per scope it names, on
a freshly seeded SQLite database of the case's layout; the fp32 layout needs sqlite-vec for
its vector table. The scaffold checks prove the seed on both layouts, the expected visibility
tables against the access predicate, the case registry and runner, and the variable cap of
the ``sqlite_999_variables`` fixture.
"""

import sqlite3
from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.access_scope import SYSTEM_SCOPE
from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import Scope
from app.backends.base import StorageBackend
from tests.conftest import requires_sqlite_vec
from tests.repositories._access_scope_cases import CASES
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SCOPES
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import case_params
from tests.repositories._access_scope_cases import expected_visibility
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import observe_predicate_visibility
from tests.repositories._access_scope_cases import registry_problems
from tests.repositories._access_scope_cases import run_case
from tests.repositories._access_scope_layouts import SQLITE_TEST_VARIABLE_LIMIT
from tests.repositories._access_scope_layouts import assert_seeded_layout
from tests.repositories._access_scope_layouts import limit_sqlite_variables
from tests.repositories._access_scope_layouts import sqlite_scoped_db


@pytest_asyncio.fixture
async def scoped_db_fp32(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[ScopedDb]:
    """Seeded SQLite database with compression off and the fp32 vector table.

    Yields:
        The seeded database.
    """
    async with sqlite_scoped_db(async_db_initialized, 'fp32', monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def scoped_db_compressed(
    async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
) -> AsyncIterator[ScopedDb]:
    """Seeded SQLite database with compression on from the first start and no fp32 table.

    Yields:
        The seeded database.
    """
    async with sqlite_scoped_db(async_db_initialized, 'compressed', monkeypatch) as db:
        yield db


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(CASES, 'fp32', marks=(requires_sqlite_vec,)))
async def test_access_case_fp32(scoped_db_fp32: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each fp32-layout case yields the expected observable for the scope."""
    await run_case(scoped_db_fp32, case, scope_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(CASES, 'compressed'))
async def test_access_case_compressed(scoped_db_compressed: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each compressed-layout case yields the expected observable for the scope."""
    await run_case(scoped_db_compressed, case, scope_name)


class TestSeededLayouts:
    """Both layouts hold the seed rows, their children and the layout's vector storage."""

    @requires_sqlite_vec
    @pytest.mark.asyncio
    async def test_fp32_layout_holds_the_seed(self, scoped_db_fp32: ScopedDb) -> None:
        """The fp32 layout stores every seed row with fp32 vectors and no compressed table."""
        await assert_seeded_layout(scoped_db_fp32)

    @pytest.mark.asyncio
    async def test_compressed_layout_holds_the_seed(self, scoped_db_compressed: ScopedDb) -> None:
        """The compressed layout stores every seed row with compressed payloads and no fp32 table."""
        await assert_seeded_layout(scoped_db_compressed)

    @pytest.mark.asyncio
    async def test_expected_tables_match_the_access_predicate(self, scoped_db_compressed: ScopedDb) -> None:
        """READABLE, WRITABLE and OWNED are what the access predicate selects for every scope."""
        assert await observe_predicate_visibility(scoped_db_compressed) == expected_visibility()

    @pytest.mark.asyncio
    async def test_raw_write_changes_what_the_predicate_admits(self, scoped_db_compressed: ScopedDb) -> None:
        """A row published through the raw-write helper becomes readable to bob."""
        db = scoped_db_compressed

        await db.execute('UPDATE context_entries SET visibility = ? WHERE id = ?', ('public', db.ids['alice_private']))

        assert await db.labels_selected_by(SCOPES['bob'], AccessMode.READ) == in_seed_order(
            (*READABLE['bob'], 'alice_private'),
        )


async def _principal_of(db: ScopedDb, scope: Scope) -> str:
    """Return the scope's principal id, or ``system`` for the system scope."""
    del db
    return scope.principal_id if isinstance(scope, AccessScope) else 'system'


async def _readable_labels(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
    """Return the labels the scope reads, in seed order."""
    return await db.labels_selected_by(scope, AccessMode.READ)


async def _add_alice_draft(db: ScopedDb) -> None:
    """Add one private alice row beyond the seed."""
    await db.add_entry('alice_draft', owner='alice', visibility='private')


class TestCaseRegistry:
    """The registry names known case ids and the runner checks one scope per test."""

    def test_registered_cases_are_well_formed(self) -> None:
        """Every registered case uses a known case id, known scope names and a unique test id."""
        assert registry_problems(CASES) == []

    def test_registry_problems_name_each_defect(self) -> None:
        """Unknown case ids, unknown scope names and duplicate test ids are all reported."""
        cases = (
            AccessCase('S1', _principal_of, {'bob': 'bob'}),
            AccessCase('S1', _principal_of, {'bob': 'bob', 'eve': 'eve'}),
            AccessCase('Q9', _principal_of, {'alice': 'alice'}),
        )

        assert registry_problems(cases) == [
            "S1: unknown scope names ['eve']",
            'S1: duplicate test id S1-bob',
            'Q9: unknown case id',
        ]

    def test_case_params_name_the_case_variant_filter_and_scope(self) -> None:
        """Test ids start with the case id, so -k selects every variant of a case."""
        cases = (
            AccessCase('S1', _principal_of, {'alice': 'alice', 'bob': 'bob'}),
            AccessCase('S1', _principal_of, {'carol': 'carol'}, filtered=True, variant='page-fill'),
            AccessCase('R1', _principal_of, {'system': 'system'}, layout='fp32'),
        )

        compressed = case_params(cases, 'compressed')
        fp32 = case_params(cases, 'fp32', marks=(requires_sqlite_vec,))

        assert [param.id for param in compressed] == ['S1-alice', 'S1-bob', 'S1-page-fill-filtered-carol']
        assert [param.values for param in compressed] == [
            (cases[0], 'alice'), (cases[0], 'bob'), (cases[1], 'carol'),
        ]
        assert [param.id for param in fp32] == ['R1-system']
        assert [mark.name for param in fp32 for mark in param.marks] == ['skipif']

    @pytest.mark.asyncio
    async def test_run_case_runs_setup_before_the_invoker(self, scoped_db_compressed: ScopedDb) -> None:
        """A case's setup rows exist when its invoker runs."""
        case = AccessCase(
            'S1', _readable_labels, {'alice': (*READABLE['alice'], 'alice_draft')}, setup=_add_alice_draft,
        )

        await run_case(scoped_db_compressed, case, 'alice')

    @pytest.mark.asyncio
    async def test_run_case_fails_on_an_unexpected_observable(self, scoped_db_compressed: ScopedDb) -> None:
        """A scope whose observable differs from the expected one fails the case."""
        case = AccessCase('S1', _readable_labels, {'bob': READABLE['alice']})

        with pytest.raises(AssertionError, match='S1 as bob'):
            await run_case(scoped_db_compressed, case, 'bob')

    @pytest.mark.asyncio
    async def test_system_scope_reads_every_seed_row(self, scoped_db_compressed: ScopedDb) -> None:
        """The system scope's empty predicate admits the whole seed."""
        assert await scoped_db_compressed.labels_selected_by(SYSTEM_SCOPE, AccessMode.READ) == READABLE['system']


def _select_binds(count: int) -> str:
    """Return a SELECT statement with ``count`` bind variables."""
    return f'SELECT {", ".join("?" * count)}'


async def _bind_on_reader(backend: StorageBackend, count: int) -> None:
    """Run a statement with ``count`` bind variables on a reader connection."""

    def _read(conn: sqlite3.Connection) -> None:
        conn.execute(_select_binds(count), [0] * count).fetchone()

    await backend.execute_read(_read)


async def _bind_on_writer(backend: StorageBackend, count: int) -> None:
    """Run a statement with ``count`` bind variables on the writer connection."""

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(_select_binds(count), [0] * count).fetchone()

    await backend.execute_write(_write)


async def _assert_capped(backend: StorageBackend) -> None:
    """Assert both connection kinds bind up to the cap and refuse one variable more."""
    await _bind_on_reader(backend, SQLITE_TEST_VARIABLE_LIMIT)
    await _bind_on_writer(backend, SQLITE_TEST_VARIABLE_LIMIT)
    with pytest.raises(sqlite3.OperationalError, match='too many SQL variables'):
        await _bind_on_reader(backend, SQLITE_TEST_VARIABLE_LIMIT + 1)
    with pytest.raises(sqlite3.OperationalError, match='too many SQL variables'):
        await _bind_on_writer(backend, SQLITE_TEST_VARIABLE_LIMIT + 1)


class TestSqliteVariableCap:
    """The 999-variable cap reaches every connection of a SQLite backend."""

    @pytest.mark.asyncio
    async def test_uncapped_backend_binds_beyond_the_cap(self, async_db_initialized: StorageBackend) -> None:
        """Without the cap the bundled SQLite binds more than 999 variables, so the cap is what bites."""
        await _bind_on_reader(async_db_initialized, SQLITE_TEST_VARIABLE_LIMIT + 1)
        await _bind_on_writer(async_db_initialized, SQLITE_TEST_VARIABLE_LIMIT + 1)

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('sqlite_999_variables')
    async def test_fixture_caps_reader_and_writer(self, async_db_initialized: StorageBackend) -> None:
        """Under the fixture both the writer and every reader refuse a 1,000th variable."""
        await _assert_capped(async_db_initialized)

    @pytest.mark.asyncio
    async def test_cap_reaches_connections_opened_before_it(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A backend whose writer is already open is capped as soon as the cap is installed."""
        await _bind_on_writer(async_db_initialized, 2 * SQLITE_TEST_VARIABLE_LIMIT)

        limit_sqlite_variables(monkeypatch, SQLITE_TEST_VARIABLE_LIMIT)

        await _assert_capped(async_db_initialized)
