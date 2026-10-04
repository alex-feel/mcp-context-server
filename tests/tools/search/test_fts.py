"""fts_search_context validation-error responses, their stats block, and scoping to readable entries."""

from typing import Literal
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio

import app.startup
import app.tools
from app.access_scope import AccessScope
from tests.helpers import as_principal


class TestFtsValidationErrorStats:
    """The FTS validation-error response carries a full stats dict when explain_query=True.

    A FtsValidationError (invalid boolean query or invalid metadata filter) returns
    a structured error response instead of raising. When explain_query is requested,
    that response's ``stats`` dict must include the ``backend`` key, matching every
    other stats path so the shape is uniform for the client.
    """

    @pytest.mark.asyncio
    async def test_validation_error_stats_include_backend(self) -> None:
        """The error-path stats dict includes backend (the active storage backend type)."""
        from app.repositories.fts_repository.faults import FtsValidationError
        from app.tools.search.fts import fts_search_context

        with (
            patch('app.tools.search.fts.get_reranking_provider', return_value=None),
            patch('app.tools.search.fts.ensure_repositories', new=AsyncMock(return_value=MagicMock())),
            patch(
                'app.tools.search.fts.fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', ['bad operator: nope'])),
            ),
        ):
            response = await fts_search_context(
                query='python AND (',
                mode='boolean',
                explain_query=True,
            )

        assert response['count'] == 0
        assert response['results'] == []
        assert response['error'] == 'Invalid filters'
        assert response['validation_errors'] == ['bad operator: nope']
        assert 'stats' in response
        stats = response['stats']
        # The backend key must be present and match the backend the tool actually
        # resolves (the module-level settings binding the production code reads),
        # so the error-path stats shape matches every other stats path.
        import app.tools.search.limits as search_limits

        assert 'backend' in stats
        assert stats['backend'] == search_limits.settings.storage.backend_type
        # The other documented error-path stat keys accompany it.
        assert stats['execution_time_ms'] == 0.0
        assert stats['filters_applied'] == 0
        assert stats['rows_returned'] == 0
        # query_plan is present as an explicit null rather than omitted: the docs let a
        # client read stats['query_plan'] unconditionally under explain_query, so dropping
        # the key on the rejection path would make that documented access raise KeyError.
        assert 'query_plan' in stats
        assert stats['query_plan'] is None

    @pytest.mark.asyncio
    async def test_validation_error_omits_stats_without_explain_query(self) -> None:
        """Without explain_query the validation-error response carries no stats block."""
        from app.repositories.fts_repository.faults import FtsValidationError
        from app.tools.search.fts import fts_search_context

        with (
            patch('app.tools.search.fts.get_reranking_provider', return_value=None),
            patch('app.tools.search.fts.ensure_repositories', new=AsyncMock(return_value=MagicMock())),
            patch(
                'app.tools.search.fts.fts_search_raw',
                AsyncMock(side_effect=FtsValidationError('Invalid filters', ['bad operator: nope'])),
            ),
        ):
            response = await fts_search_context(
                query='python AND (',
                mode='boolean',
                explain_query=False,
            )

        assert response['count'] == 0
        assert response['error'] == 'Invalid filters'
        assert 'stats' not in response


@pytest_asyncio.fixture
async def fts_index(initialized_server: None) -> None:
    """Provision the full-text index on the initialized server's database."""
    del initialized_server
    from app.migrations import apply_fts_migration

    backend = app.startup.get_backend()
    assert backend is not None
    await apply_fts_migration(backend, force=True)


@pytest.mark.usefixtures('fts_index')
class TestFtsSearchScoping:
    """fts_search_context matches only the entries the caller may read."""

    @staticmethod
    async def _store_as(principal_id: str, text: str, visibility: Literal['private', 'public']) -> str:
        """Store one entry in the scoped thread as the principal and return its id."""
        with as_principal(principal_id):
            result = await app.tools.store_context(
                thread_id='scoped-fts', source='agent', text=text, visibility=visibility, metadata={'project': 'p'},
            )
        return result['context_id']

    @pytest.mark.asyncio
    async def test_unreadable_entries_are_absent(self) -> None:
        """Bob matches alice's public entry and nothing of her private one; the count matches."""
        from app.tools.search.fts import fts_search_context

        await self._store_as('alice', 'alice private lattice target', 'private')
        public_id = await self._store_as('alice', 'alice public lattice target', 'public')

        with as_principal('bob'):
            results = await fts_search_context(query='lattice', thread_id='scoped-fts', limit=50)

        assert [entry['id'] for entry in results['results']] == [public_id]
        assert results['count'] == 1

    @pytest.mark.asyncio
    async def test_owner_matches_their_private_entry(self) -> None:
        """The owner matches both of their entries."""
        from app.tools.search.fts import fts_search_context

        private_id = await self._store_as('alice', 'alice private lattice own', 'private')
        public_id = await self._store_as('alice', 'alice public lattice own', 'public')

        with as_principal('alice'):
            results = await fts_search_context(query='lattice', thread_id='scoped-fts', limit=50)

        assert sorted(entry['id'] for entry in results['results']) == sorted([private_id, public_id])

    @pytest.mark.asyncio
    async def test_scope_reaches_the_repository(self) -> None:
        """The caller's principal and groups reach the full-text search as its scope."""
        from app.tools.search.fts import fts_search_context

        repos = await app.startup.ensure_repositories()

        with (
            as_principal('bob', groups=['team-x']),
            patch.object(repos.fts, 'search', AsyncMock(return_value=([], {}))) as spy,
        ):
            await fts_search_context(query='lattice', thread_id='scoped-fts', limit=10)

        assert spy.await_args is not None
        assert spy.await_args.kwargs['scope'] == AccessScope('bob', frozenset({'team-x'}))

    @pytest.mark.asyncio
    async def test_filters_applied_is_the_same_for_every_caller(self) -> None:
        """The explain stats count the client filters only, whoever runs them."""
        from app.tools.search.fts import fts_search_context

        await self._store_as('alice', 'alice private lattice stats', 'private')

        filters_applied: dict[str, int] = {}
        for principal_id in ('alice', 'bob'):
            with as_principal(principal_id):
                results = await fts_search_context(
                    query='lattice', thread_id='scoped-fts', source='agent', metadata={'project': 'p'},
                    explain_query=True, limit=10,
                )
            filters_applied[principal_id] = results['stats']['filters_applied']
            assert results['stats']['query_plan']

        assert filters_applied == {'alice': 3, 'bob': 3}
