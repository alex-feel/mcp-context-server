"""fts_search_context validation-error responses and their stats block."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


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
