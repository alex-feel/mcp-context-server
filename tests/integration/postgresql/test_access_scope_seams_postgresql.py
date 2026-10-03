"""Two-principal access-scope cases on PostgreSQL, and the checks of the case scaffold itself.

Every case of :mod:`tests.repositories._access_scope_cases` runs once per scope it names, on
a freshly seeded, isolated database of the case's layout on the pgvector container of the
``pg_test_url`` fixture (``@requires_docker_postgres``, skipped cleanly without Docker). The
scaffold checks prove the seed on both layouts and the expected visibility tables against the
access predicate, including the one-parameter ``text[]`` group bind.
"""

from collections.abc import AsyncIterator

import pytest
import pytest_asyncio

from app.access_scope import AccessMode
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SCOPES
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import case_params
from tests.repositories._access_scope_cases import expected_visibility
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import observe_predicate_visibility
from tests.repositories._access_scope_cases import run_case
from tests.repositories._access_scope_layouts import assert_seeded_layout
from tests.repositories._access_scope_layouts import postgresql_scoped_db
from tests.repositories._access_scope_registry import CASES

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


@pytest_asyncio.fixture
async def scoped_db_fp32(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """Seeded isolated database with compression off and the fp32 vector table.

    Yields:
        The seeded database.
    """
    async with postgresql_scoped_db(pg_test_url, 'fp32', monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def scoped_db_compressed(pg_test_url: str, monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[ScopedDb]:
    """Seeded isolated database with compression on from the first start and no fp32 table.

    Yields:
        The seeded database.
    """
    async with postgresql_scoped_db(pg_test_url, 'compressed', monkeypatch) as db:
        yield db


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(CASES, 'fp32', backend='postgresql'))
async def test_access_case_fp32(scoped_db_fp32: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each fp32-layout case yields the expected observable for the scope."""
    await run_case(scoped_db_fp32, case, scope_name)


@pytest.mark.asyncio
@pytest.mark.parametrize(('case', 'scope_name'), case_params(CASES, 'compressed', backend='postgresql'))
async def test_access_case_compressed(scoped_db_compressed: ScopedDb, case: AccessCase, scope_name: str) -> None:
    """Each compressed-layout case yields the expected observable for the scope."""
    await run_case(scoped_db_compressed, case, scope_name)


class TestSeededLayoutsPostgreSQL:
    """Both layouts hold the seed rows, their children and the layout's vector storage."""

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
