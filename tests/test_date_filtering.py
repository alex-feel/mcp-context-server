"""Tests for date range filtering in search_context and semantic_search_context.

This module contains comprehensive tests for the date filtering feature,
including validation tests, integration tests with both search tools,
and edge case handling.
"""

from datetime import UTC
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.server
from app.repositories import RepositoryContainer
from app.server import validate_date_param
from app.server import validate_date_range

# Get the actual async functions - they are no longer wrapped by @mcp.tool() at import time
store_context = app.server.store_context
search_context = app.server.search_context


class TestDateValidation:
    """Test date parameter validation functions."""

    def test_valid_date_only_start_date(self) -> None:
        """Test date-only format (YYYY-MM-DD) for start_date - unchanged."""
        result = validate_date_param('2025-11-29', 'start_date')
        assert result == '2025-11-29'

    def test_valid_date_only_end_date_expands_to_end_of_day(self) -> None:
        """Test date-only format for end_date expands to end-of-day (T23:59:59.999999).

        This follows Elasticsearch precedent where missing time components are replaced
        with max values for 'lte' operations, matching user expectations that
        end_date='2025-11-29' should include ALL entries on November 29th.

        Uses microsecond precision (.999999) for PostgreSQL compatibility where
        CURRENT_TIMESTAMP stores microseconds (e.g., 23:59:59.500000).
        """
        result = validate_date_param('2025-11-29', 'end_date')
        assert result == '2025-11-29T23:59:59.999999'

    def test_end_date_with_datetime_not_expanded(self) -> None:
        """Test end_date with full datetime format is NOT expanded.

        Only date-only end_date values should be expanded to end-of-day.
        Explicit datetime values should be preserved as-is.
        """
        # With T separator
        result = validate_date_param('2025-11-29T14:00:00', 'end_date')
        assert result == '2025-11-29T14:00:00'

        # With timezone
        result = validate_date_param('2025-11-29T14:00:00Z', 'end_date')
        assert result == '2025-11-29T14:00:00Z'

        # With timezone offset
        result = validate_date_param('2025-11-29T14:00:00+02:00', 'end_date')
        assert result == '2025-11-29T14:00:00+02:00'

    def test_valid_datetime(self) -> None:
        """Test full datetime format without timezone."""
        result = validate_date_param('2025-11-29T10:00:00', 'start_date')
        assert result == '2025-11-29T10:00:00'

    def test_valid_datetime_with_timezone_offset(self) -> None:
        """Test datetime with timezone offset."""
        result = validate_date_param('2025-11-29T10:00:00+02:00', 'start_date')
        assert result == '2025-11-29T10:00:00+02:00'

    def test_valid_datetime_with_negative_timezone(self) -> None:
        """Test datetime with negative timezone offset."""
        result = validate_date_param('2025-11-29T10:00:00-05:00', 'start_date')
        assert result == '2025-11-29T10:00:00-05:00'

    def test_valid_datetime_utc_z_suffix(self) -> None:
        """Test datetime with Z suffix for UTC."""
        result = validate_date_param('2025-11-29T10:00:00Z', 'start_date')
        assert result == '2025-11-29T10:00:00Z'

    def test_valid_datetime_with_microseconds(self) -> None:
        """Test datetime with microseconds."""
        result = validate_date_param('2025-11-29T10:00:00.123456', 'start_date')
        assert result == '2025-11-29T10:00:00.123456'

    def test_none_passthrough(self) -> None:
        """Test None value passes through unchanged."""
        result = validate_date_param(None, 'start_date')
        assert result is None

    def test_invalid_format_day_month_year(self) -> None:
        """Test invalid DD-MM-YYYY format raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('29-11-2025', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)
        assert 'ISO 8601' in str(exc_info.value)

    def test_invalid_format_slash_separator(self) -> None:
        """Test invalid YYYY/MM/DD format raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('2025/11/29', 'end_date')
        assert 'Invalid end_date format' in str(exc_info.value)

    def test_invalid_date_values(self) -> None:
        """Test invalid date values raise ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('2025-13-45', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)

    def test_invalid_month_out_of_range(self) -> None:
        """Test month out of range raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('2025-00-15', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)

    def test_invalid_empty_string(self) -> None:
        """Test empty string raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('', 'start_date')
        assert 'Invalid start_date format' in str(exc_info.value)

    def test_invalid_random_text(self) -> None:
        """Test random text raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_param('not-a-date', 'end_date')
        assert 'Invalid end_date format' in str(exc_info.value)


class TestDateRangeValidation:
    """Test date range validation (start_date <= end_date)."""

    def test_valid_range_same_date(self) -> None:
        """Test same date for start and end is valid."""
        # Should not raise
        validate_date_range('2025-11-29', '2025-11-29')

    def test_valid_range_start_before_end(self) -> None:
        """Test start_date before end_date is valid."""
        # Should not raise
        validate_date_range('2025-11-01', '2025-11-30')

    def test_valid_range_with_datetimes(self) -> None:
        """Test datetime range is valid."""
        # Should not raise
        validate_date_range('2025-11-29T00:00:00', '2025-11-29T23:59:59')

    def test_invalid_range_start_after_end(self) -> None:
        """Test start_date after end_date raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_range('2025-12-01', '2025-11-01')
        assert 'Invalid date range' in str(exc_info.value)
        assert 'start_date' in str(exc_info.value)
        assert 'after' in str(exc_info.value)

    def test_invalid_range_with_time(self) -> None:
        """Test datetime range where start is after end."""
        with pytest.raises(ToolError) as exc_info:
            validate_date_range('2025-11-29T23:00:00', '2025-11-29T10:00:00')
        assert 'Invalid date range' in str(exc_info.value)

    def test_none_start_date_valid(self) -> None:
        """Test None start_date with valid end_date."""
        # Should not raise
        validate_date_range(None, '2025-11-29')

    def test_none_end_date_valid(self) -> None:
        """Test valid start_date with None end_date."""
        # Should not raise
        validate_date_range('2025-11-29', None)

    def test_both_none_valid(self) -> None:
        """Test both dates None is valid."""
        # Should not raise
        validate_date_range(None, None)


@pytest.mark.usefixtures('mock_server_dependencies')
class TestSearchContextDateFiltering:
    """Test date filtering in search_context tool."""

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.mock_repos = MagicMock(spec=RepositoryContainer)
        self.mock_repos.context = AsyncMock()
        self.mock_repos.tags = AsyncMock()
        self.mock_repos.images = AsyncMock()

    @pytest.mark.asyncio
    async def test_filter_by_start_date_future(self) -> None:
        """Test filtering with future start_date calls repository correctly."""
        # Mock search_contexts to return empty results
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        future_date = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')

        with patch('app.server._ensure_repositories', return_value=self.mock_repos):
            result = await search_context(
                thread_id='date-test-1',
                start_date=future_date,
            limit=50,
            )

        assert len(result['results']) == 0
        # Verify start_date was passed to repository
        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] == future_date

    @pytest.mark.asyncio
    async def test_filter_by_end_date_past(self) -> None:
        """Test filtering with past end_date calls repository correctly.

        Note: Date-only end_date is expanded to end-of-day (T23:59:59.999999) by validate_date_param().
        """
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        past_date = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
        # Expected expanded value includes end-of-day time with microsecond precision
        expected_end_date = f'{past_date}T23:59:59.999999'

        with patch('app.server._ensure_repositories', return_value=self.mock_repos):
            result = await search_context(
                thread_id='date-test-2',
                end_date=past_date,
            limit=50,
            )

        assert len(result['results']) == 0
        call_args = self.mock_repos.context.search_contexts.call_args
        # Verify end_date was expanded to end-of-day
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_filter_by_date_range(self) -> None:
        """Test filtering with both start and end dates.

        Note: Date-only end_date is expanded to end-of-day (T23:59:59.999999) by validate_date_param().
        """
        mock_entry = {
            'id': 1,
            'thread_id': 'date-test-3',
            'source': 'user',
            'content_type': 'text',
            'text_content': 'Test entry',
            'metadata': None,
            'created_at': '2025-11-29 10:00:00',
            'updated_at': '2025-11-29 10:00:00',
        }
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([mock_entry], {}))
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).date().strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).strftime('%Y-%m-%d')
        # Expected expanded end_date includes end-of-day time with microsecond precision
        expected_end_date = f'{tomorrow}T23:59:59.999999'

        with patch('app.server._ensure_repositories', return_value=self.mock_repos):
            result = await search_context(
                thread_id='date-test-3',
                start_date=today,
                end_date=tomorrow,
            limit=50,
            )

        assert len(result['results']) == 1
        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] == today
        # Verify end_date was expanded to end-of-day
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_invalid_date_format_raises_error(self) -> None:
        """Test invalid date format raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            await search_context(
                thread_id='any-thread',
                start_date='invalid-date',
            limit=50,
            )
        assert 'Invalid start_date format' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_invalid_date_range_raises_error(self) -> None:
        """Test start_date > end_date raises ToolError."""
        with pytest.raises(ToolError) as exc_info:
            await search_context(
                thread_id='any-thread',
                start_date='2025-12-01',
                end_date='2025-11-01',
            limit=50,
            )
        assert 'Invalid date range' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_filter_with_datetime_format(self) -> None:
        """Test filtering with full datetime format."""
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        now = datetime.now(UTC)
        start = (now - timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')
        end = (now + timedelta(hours=1)).strftime('%Y-%m-%dT%H:%M:%S')

        with patch('app.server._ensure_repositories', return_value=self.mock_repos):
            await search_context(
                thread_id='date-test-6',
                start_date=start,
                end_date=end,
            limit=50,
            )

        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] == start
        assert call_args[1]['end_date'] == end

    @pytest.mark.asyncio
    async def test_no_date_filter_passes_none(self) -> None:
        """Test that no date filter passes None to repository."""
        self.mock_repos.context.search_contexts = AsyncMock(return_value=([], {}))

        with patch('app.server._ensure_repositories', return_value=self.mock_repos):
            await search_context(
                thread_id='date-test-7',
            limit=50,
            )

        call_args = self.mock_repos.context.search_contexts.call_args
        assert call_args[1]['start_date'] is None
        assert call_args[1]['end_date'] is None


@pytest.mark.usefixtures('mock_server_dependencies', 'initialized_server')
class TestSearchContextDateIntegration:
    """Integration tests for date filtering with actual database."""

    @pytest.mark.asyncio
    async def test_date_only_end_date_includes_entire_day(self) -> None:
        """Test that date-only end_date includes ALL entries on that day.

        This is the core UX fix: When a user specifies end_date='2025-11-29',
        they expect to include ALL entries created on November 29th, not just
        entries before midnight.

        Before fix: end_date='2025-11-29' was interpreted as <= 2025-11-29 00:00:00
        After fix: end_date='2025-11-29' is expanded to <= 2025-11-29T23:59:59.999999

        This follows Elasticsearch precedent where missing time components are
        replaced with max values for 'lte' operations.
        """
        # Store an entry - it will be created at the current time (e.g., 18:56:45)
        await store_context(
            thread_id='end-date-ux-fix-test',
            source='user',
            text='UX fix test entry',
        )

        # Use date-only end_date for TODAY
        # Before fix: This would FAIL to find the entry because:
        #   end_date='2025-11-29' -> <= '2025-11-29 00:00:00' (midnight)
        #   Entry at 18:56:45 > 00:00:00 -> NOT included
        # After fix: This should FIND the entry because:
        #   end_date='2025-11-29' -> <= '2025-11-29T23:59:59' (end of day)
        #   Entry at 18:56:45 <= 23:59:59 -> INCLUDED
        today = datetime.now(UTC).date().strftime('%Y-%m-%d')

        result = await search_context(
            thread_id='end-date-ux-fix-test',
            end_date=today,
        limit=50,
        )

        # Entry should be found because end_date is expanded to end-of-day
        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'UX fix test entry'

    @pytest.mark.asyncio
    async def test_date_filter_with_real_database(self) -> None:
        """Test date filtering with real database operations."""
        # Store a test entry
        await store_context(
            thread_id='date-integration-1',
            source='user',
            text='Integration test entry',
        )

        # Get today and tomorrow dates
        today = datetime.now(UTC).date().strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).strftime('%Y-%m-%d')

        # Search with date range including today - should find the entry
        result = await search_context(
            thread_id='date-integration-1',
            start_date=today,
            end_date=tomorrow,
        limit=50,
        )

        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_future_start_date_returns_empty(self) -> None:
        """Test that future start_date returns empty results."""
        # Store a test entry
        await store_context(
            thread_id='date-integration-2',
            source='agent',
            text='Another test entry',
        )

        # Future date should return empty
        future_date = (datetime.now(UTC).date() + timedelta(days=10)).strftime('%Y-%m-%d')

        result = await search_context(
            thread_id='date-integration-2',
            start_date=future_date,
        limit=50,
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_past_end_date_returns_empty(self) -> None:
        """Test that past end_date returns empty results."""
        # Store a test entry
        await store_context(
            thread_id='date-integration-3',
            source='user',
            text='Yet another test entry',
        )

        # Past date should return empty
        past_date = (datetime.now(UTC).date() - timedelta(days=10)).strftime('%Y-%m-%d')

        result = await search_context(
            thread_id='date-integration-3',
            end_date=past_date,
        limit=50,
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_date_filter_combined_with_source(self) -> None:
        """Test date filtering combined with source filter."""
        # Store entries with different sources
        await store_context(
            thread_id='date-integration-4',
            source='user',
            text='User entry',
        )
        await store_context(
            thread_id='date-integration-4',
            source='agent',
            text='Agent entry',
        )

        today = datetime.now(UTC).date().strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC).date() + timedelta(days=1)).strftime('%Y-%m-%d')

        # Filter by source and date
        result = await search_context(
            thread_id='date-integration-4',
            source='user',
            start_date=today,
            end_date=tomorrow,
        limit=50,
        )

        assert len(result['results']) == 1
        assert result['results'][0]['source'] == 'user'


@pytest.mark.usefixtures('mock_server_dependencies')
class TestSemanticSearchDateFiltering:
    """Test date filtering in semantic_search_context tool.

    Tests verify that date parameters are correctly validated and passed
    to the embedding repository for semantic search operations.
    """

    @pytest.fixture(autouse=True)
    def setup(self) -> None:
        """Set up test fixtures."""
        self.mock_repos = MagicMock(spec=RepositoryContainer)
        self.mock_repos.context = AsyncMock()
        self.mock_repos.tags = AsyncMock()
        self.mock_repos.embeddings = AsyncMock()

    @pytest.mark.asyncio
    async def test_semantic_search_with_start_date(self) -> None:
        """Test semantic_search_context with start_date filter."""
        # Mock embedding service
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)

        # Mock search results (now returns tuple with stats)
        self.mock_repos.embeddings.search = AsyncMock(return_value=([
            {
                'id': 1,
                'thread_id': 'test-thread',
                'source': 'user',
                'text_content': 'Test entry',
                'distance': 0.5,
            },
        ], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 1}))
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).strftime('%Y-%m-%d')

        with (
            patch('app.server._ensure_repositories', return_value=self.mock_repos),
            patch('app.server._embedding_service', mock_embedding_service),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_semantic_search = True
            mock_settings.embedding_model = 'test-model'

            # Import and get the actual function
            import app.server
            semantic_search = app.server.semantic_search_context

            result = await semantic_search(
                query='test query',
                start_date=today,
            limit=20,
            )

        # Verify search was called with start_date
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] == today
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_semantic_search_with_end_date(self) -> None:
        """Test semantic_search_context with end_date filter.

        Note: Date-only end_date is expanded to end-of-day (T23:59:59.999999).
        """
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).strftime('%Y-%m-%d')
        expected_end_date = f'{today}T23:59:59.999999'

        with (
            patch('app.server._ensure_repositories', return_value=self.mock_repos),
            patch('app.server._embedding_service', mock_embedding_service),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_semantic_search = True
            mock_settings.embedding_model = 'test-model'

            import app.server
            semantic_search = app.server.semantic_search_context

            await semantic_search(
                query='test query',
                end_date=today,
            limit=20,
            )

        # Verify end_date was expanded to end-of-day
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_semantic_search_with_date_range(self) -> None:
        """Test semantic_search_context with both start_date and end_date."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        today = datetime.now(UTC).strftime('%Y-%m-%d')
        tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
        expected_end_date = f'{tomorrow}T23:59:59.999999'

        with (
            patch('app.server._ensure_repositories', return_value=self.mock_repos),
            patch('app.server._embedding_service', mock_embedding_service),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_semantic_search = True
            mock_settings.embedding_model = 'test-model'

            import app.server
            semantic_search = app.server.semantic_search_context

            await semantic_search(
                query='test query',
                start_date=today,
                end_date=tomorrow,
            limit=20,
            )

        # Verify both dates were passed correctly
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] == today
        assert call_args[1]['end_date'] == expected_end_date

    @pytest.mark.asyncio
    async def test_semantic_search_invalid_date_format_raises_error(self) -> None:
        """Test semantic_search_context with invalid date format raises ToolError."""
        mock_embedding_service = MagicMock()

        with (
            patch('app.server._embedding_service', mock_embedding_service),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_semantic_search = True

            import app.server
            semantic_search = app.server.semantic_search_context

            with pytest.raises(ToolError) as exc_info:
                await semantic_search(
                    query='test query',
                    start_date='invalid-date',
                limit=20,
                )
            assert 'Invalid start_date format' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_semantic_search_invalid_date_range_raises_error(self) -> None:
        """Test semantic_search_context with start_date > end_date raises ToolError."""
        mock_embedding_service = MagicMock()

        with (
            patch('app.server._embedding_service', mock_embedding_service),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_semantic_search = True

            import app.server
            semantic_search = app.server.semantic_search_context

            with pytest.raises(ToolError) as exc_info:
                await semantic_search(
                    query='test query',
                    start_date='2025-12-01',
                    end_date='2025-11-01',
                limit=20,
                )
            assert 'Invalid date range' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_semantic_search_with_datetime_format(self) -> None:
        """Test semantic_search_context with full datetime format."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        start = '2025-11-29T10:00:00'
        end = '2025-11-29T18:00:00'

        with (
            patch('app.server._ensure_repositories', return_value=self.mock_repos),
            patch('app.server._embedding_service', mock_embedding_service),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_semantic_search = True
            mock_settings.embedding_model = 'test-model'

            import app.server
            semantic_search = app.server.semantic_search_context

            await semantic_search(
                query='test query',
                start_date=start,
                end_date=end,
            limit=20,
            )

        # Verify datetime format is preserved (not expanded)
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] == start
        assert call_args[1]['end_date'] == end

    @pytest.mark.asyncio
    async def test_semantic_search_no_date_filter_passes_none(self) -> None:
        """Test semantic_search_context without date filter passes None to repository."""
        mock_embedding_service = MagicMock()
        mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.1] * 768)

        self.mock_repos.embeddings.search = AsyncMock(
            return_value=([], {'execution_time_ms': 1.0, 'filters_applied': 0, 'rows_returned': 0}),
        )
        self.mock_repos.tags.get_tags_for_context = AsyncMock(return_value=[])

        with (
            patch('app.server._ensure_repositories', return_value=self.mock_repos),
            patch('app.server._embedding_service', mock_embedding_service),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_semantic_search = True
            mock_settings.embedding_model = 'test-model'

            import app.server
            semantic_search = app.server.semantic_search_context

            await semantic_search(
                query='test query',
            limit=20,
            )

        # Verify None dates were passed
        call_args = self.mock_repos.embeddings.search.call_args
        assert call_args[1]['start_date'] is None
        assert call_args[1]['end_date'] is None


class TestParseDateForPostgresql:
    """Test _parse_date_for_postgresql helper for asyncpg datetime conversion.

    asyncpg requires Python datetime objects for TIMESTAMPTZ parameters.
    This helper converts ISO 8601 date strings to datetime objects.
    """

    def test_none_returns_none(self) -> None:
        """Test None input returns None."""
        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql(None)
        assert result is None

    def test_date_only_returns_datetime_utc(self) -> None:
        """Test date-only format returns datetime at start of day UTC."""

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29')
        assert result is not None
        assert result.year == 2025
        assert result.month == 11
        assert result.day == 29
        assert result.hour == 0
        assert result.minute == 0
        assert result.second == 0
        assert result.tzinfo == UTC

    def test_datetime_without_timezone_interpreted_as_utc(self) -> None:
        """Test naive datetime (without timezone) is interpreted as UTC.

        Industry standard: Naive datetime is interpreted as UTC to match
        Elasticsearch, MongoDB, DynamoDB, and Firestore behavior.
        This ensures deterministic behavior regardless of server timezone.
        """
        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45')
        assert result is not None
        assert result.year == 2025
        assert result.month == 11
        assert result.day == 29
        assert result.hour == 10
        assert result.minute == 30
        assert result.second == 45
        # Naive datetime is now interpreted as UTC (industry standard)
        assert result.tzinfo == UTC

    def test_datetime_with_z_suffix(self) -> None:
        """Test datetime with Z suffix returns UTC datetime."""

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45Z')
        assert result is not None
        assert result.year == 2025
        assert result.hour == 10
        assert result.tzinfo == UTC

    def test_datetime_with_positive_offset(self) -> None:
        """Test datetime with positive timezone offset."""
        from datetime import timezone

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45+02:00')
        assert result is not None
        assert result.year == 2025
        assert result.hour == 10
        expected_tz = timezone(timedelta(hours=2))
        assert result.tzinfo == expected_tz

    def test_datetime_with_negative_offset(self) -> None:
        """Test datetime with negative timezone offset."""
        from datetime import timezone

        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45-05:00')
        assert result is not None
        assert result.year == 2025
        assert result.hour == 10
        expected_tz = timezone(timedelta(hours=-5))
        assert result.tzinfo == expected_tz

    def test_datetime_with_microseconds(self) -> None:
        """Test datetime with microseconds."""
        from app.repositories.base import BaseRepository

        result = BaseRepository._parse_date_for_postgresql('2025-11-29T10:30:45.123456')
        assert result is not None
        assert result.microsecond == 123456

    def test_result_is_datetime_type(self) -> None:
        """Test that result is always a datetime object (not date)."""
        from datetime import datetime as dt

        from app.repositories.base import BaseRepository

        # Date-only input should still return datetime
        result = BaseRepository._parse_date_for_postgresql('2025-11-29')
        assert isinstance(result, dt)


@pytest.mark.usefixtures('mock_server_dependencies', 'initialized_server')
class TestSQLiteDatetimeNormalization:
    """Test SQLite datetime() normalization for ISO 8601 formats.

    SQLite stores timestamps as TEXT in 'YYYY-MM-DD HH:MM:SS' format (space separator).
    ISO 8601 uses 'T' separator (e.g., '2025-11-29T10:00:00').

    Without datetime() normalization, TEXT comparison fails:
    - Space character (ASCII 0x20) < T character (ASCII 0x54)
    - Therefore: '2025-11-29 18:07:40' < '2025-11-29T00:00:00' (incorrect!)

    SQLite's datetime() function normalizes all ISO 8601 formats to space-separated format:
    - datetime('2025-11-29T10:00:00')       -> '2025-11-29 10:00:00'
    - datetime('2025-11-29T10:00:00Z')      -> '2025-11-29 10:00:00'
    - datetime('2025-11-29T10:00:00+02:00') -> '2025-11-29 08:00:00' (UTC converted)
    """

    @pytest.mark.asyncio
    async def test_sqlite_datetime_t_separator(self) -> None:
        """Test ISO 8601 with T separator works in SQLite.

        This is the primary fix - ensuring T-separator format works correctly.
        Before the fix, this would fail because 'T' > ' ' in ASCII ordering,
        causing entries stored with space separator to not match.
        """
        # Store entry (will use CURRENT_TIMESTAMP which has space separator)
        result = await store_context(
            thread_id='sqlite-iso8601-t-test',
            source='user',
            text='Entry with space separator in timestamp',
        )
        assert result['success']

        # Get today's date in T-separator format (before stored time)
        today_t = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00')

        # Search using T-separator format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-t-test',
            start_date=today_t,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry with space separator in timestamp'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_z_suffix(self) -> None:
        """Test ISO 8601 with Z suffix (UTC) works in SQLite.

        SQLite datetime() treats Z suffix as UTC (no-op, already UTC).
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-iso8601-z-test',
            source='agent',
            text='Entry for Z suffix test',
        )
        assert result['success']

        # Get today in Z-suffix format
        today_z = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00Z')

        # Search using Z-suffix format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-z-test',
            start_date=today_z,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for Z suffix test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_positive_timezone_offset(self) -> None:
        """Test ISO 8601 with positive timezone offset (+HH:MM) works in SQLite.

        SQLite datetime() converts timezone offsets to UTC.
        For example: '2025-11-29T10:00:00+02:00' -> '2025-11-29 08:00:00' (UTC)
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-iso8601-tz-positive-test',
            source='user',
            text='Entry for positive timezone test',
        )
        assert result['success']

        # Use positive offset (e.g., Eastern European Time +02:00)
        # Use start of day in a positive timezone to ensure we catch the entry
        today_tz = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00+02:00')

        # Search using timezone offset format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-tz-positive-test',
            start_date=today_tz,
        limit=50,
        )

        # Entry should be found (datetime() normalizes +02:00 to UTC, which is 2 hours earlier)
        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for positive timezone test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_negative_timezone_offset(self) -> None:
        """Test ISO 8601 with negative timezone offset (-HH:MM) works in SQLite.

        SQLite datetime() converts timezone offsets to UTC.
        For example: '2025-11-29T10:00:00-05:00' -> '2025-11-29 15:00:00' (UTC)
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-iso8601-tz-negative-test',
            source='agent',
            text='Entry for negative timezone test',
        )
        assert result['success']

        # Use negative offset (e.g., EST -05:00)
        # Use yesterday at midnight in negative timezone to ensure we catch today's entry
        yesterday = datetime.now(UTC) - timedelta(days=1)
        yesterday_tz = yesterday.strftime('%Y-%m-%dT00:00:00-05:00')

        # Search using timezone offset format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-iso8601-tz-negative-test',
            start_date=yesterday_tz,
        limit=50,
        )

        # Entry should be found (datetime() normalizes -05:00 to UTC, which is 5 hours later)
        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for negative timezone test'

    @pytest.mark.asyncio
    async def test_sqlite_date_only_still_works(self) -> None:
        """Test that date-only format continues to work after datetime() change.

        Date-only format (YYYY-MM-DD) should still work correctly.
        datetime('2025-11-29') -> '2025-11-29 00:00:00'
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-date-only-test',
            source='user',
            text='Entry for date-only test',
        )
        assert result['success']

        # Use date-only format
        today = datetime.now(UTC).strftime('%Y-%m-%d')

        # Search using date-only format - should find the entry
        search_result = await search_context(
            thread_id='sqlite-date-only-test',
            start_date=today,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for date-only test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_end_date_with_t_separator(self) -> None:
        """Test end_date with T separator works correctly.

        Ensures both start_date and end_date handle T-separator properly.
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-end-date-t-test',
            source='user',
            text='Entry for end_date T separator test',
        )
        assert result['success']

        # Use date range with T-separator
        today_t_start = datetime.now(UTC).strftime('%Y-%m-%dT00:00:00')
        tomorrow_t_end = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59')

        # Search using T-separator for both dates
        search_result = await search_context(
            thread_id='sqlite-end-date-t-test',
            start_date=today_t_start,
            end_date=tomorrow_t_end,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for end_date T separator test'

    @pytest.mark.asyncio
    async def test_sqlite_datetime_mixed_formats(self) -> None:
        """Test mixing date-only start with T-separator end date.

        Ensures different formats can be mixed in the same query.
        """
        # Store entry
        result = await store_context(
            thread_id='sqlite-mixed-format-test',
            source='agent',
            text='Entry for mixed format test',
        )
        assert result['success']

        # Use date-only for start, T-separator for end
        today_date_only = datetime.now(UTC).strftime('%Y-%m-%d')
        tomorrow_t = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%dT23:59:59Z')

        # Search using mixed formats
        search_result = await search_context(
            thread_id='sqlite-mixed-format-test',
            start_date=today_date_only,
            end_date=tomorrow_t,
        limit=50,
        )

        assert len(search_result['results']) == 1
        assert search_result['results'][0]['text_content'] == 'Entry for mixed format test'
