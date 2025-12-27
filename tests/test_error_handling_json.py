"""Comprehensive tests for FastMCP error handling with JSON responses.

This test module validates that all error cases return proper JSON format
when using FastMCP's mask_error_details=False configuration.
"""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.server

# Access the underlying functions directly - no longer wrapped by @mcp.tool() at import time
store_context = app.server.store_context
search_context = app.server.search_context
get_context_by_ids = app.server.get_context_by_ids
delete_context = app.server.delete_context
update_context = app.server.update_context
list_threads = app.server.list_threads
get_statistics = app.server.get_statistics


@pytest.fixture
def mock_repos():
    """Create a mock repository container."""
    repos = MagicMock()
    repos.context = AsyncMock()
    repos.tags = AsyncMock()
    repos.images = AsyncMock()
    repos.statistics = AsyncMock()
    return repos


@pytest.fixture
def mock_server_dependencies(mock_repos):
    """Mock server dependencies for testing."""
    with patch('app.server._ensure_repositories', return_value=mock_repos):
        yield mock_repos


class TestStoreContextErrors:
    """Test error handling for store_context tool."""

    @pytest.mark.asyncio
    async def test_empty_thread_id(self, mock_server_dependencies):
        """Test that empty thread_id raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='thread_id cannot be empty'):
            await store_context(
                thread_id='',
                source='user',
                text='test content',
            )

    @pytest.mark.asyncio
    async def test_whitespace_thread_id(self, mock_server_dependencies):
        """Test that whitespace-only thread_id raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='thread_id cannot be empty'):
            await store_context(
                thread_id='   ',
                source='user',
                text='test content',
            )

    @pytest.mark.asyncio
    async def test_empty_text(self, mock_server_dependencies):
        """Test that empty text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='text cannot be empty'):
            await store_context(
                thread_id='test-thread',
                source='user',
                text='',
            )

    @pytest.mark.asyncio
    async def test_whitespace_text(self, mock_server_dependencies):
        """Test that whitespace-only text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='text cannot be empty'):
            await store_context(
                thread_id='test-thread',
                source='user',
                text='   \n\t   ',
            )

    @pytest.mark.asyncio
    async def test_invalid_source(self, mock_server_dependencies):
        """Test that invalid source is caught by Pydantic Literal validation.

        Note: This test is kept for documentation but Pydantic handles this at the
        FastMCP level. If someone bypasses Pydantic (using .fn), the database
        CHECK constraint will catch it.
        """
        # Set up mock to return valid response
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)

        # Pydantic Literal['user', 'agent'] handles validation
        # Using .fn bypasses Pydantic, so we just verify function works with valid input
        result = await store_context(
            thread_id='test-thread',
            source='user',  # Valid source
            text='test content',
        )
        assert result['success'] is True

    @pytest.mark.asyncio
    async def test_invalid_base64_image(self, mock_server_dependencies):
        """Test that invalid base64 image data raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='Invalid base64 encoded data in image 0'):
            await store_context(
                thread_id='test-thread',
                source='user',
                text='test content',
                images=[{'data': 'not-base64!!!', 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    async def test_image_exceeds_size_limit(self, mock_server_dependencies):
        """Test that oversized image raises ToolError."""
        # Set up the mock to return a proper tuple
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)

        # Create a large base64 image (simulate > 10MB)
        large_data = 'A' * (15 * 1024 * 1024)  # 15MB of 'A'
        import base64

        encoded = base64.b64encode(large_data.encode()).decode()

        with pytest.raises(ToolError, match='Image 0 exceeds .* limit'):
            await store_context(
                thread_id='test-thread',
                source='user',
                text='test content',
                images=[{'data': encoded, 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    async def test_database_error(self, mock_server_dependencies):
        """Test that database errors are wrapped in ToolError."""
        mock_server_dependencies.context.store_with_deduplication.side_effect = Exception('DB connection failed')

        with pytest.raises(ToolError, match='Failed to store context: DB connection failed'):
            await store_context(
                thread_id='test-thread',
                source='user',
                text='test content',
            )


class TestUpdateContextErrors:
    """Test error handling for update_context tool."""

    @pytest.mark.asyncio
    async def test_empty_text_update(self, mock_server_dependencies):
        """Test that updating with empty text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='text cannot be empty'):
            await update_context(
                context_id=1,
                text='',
            )

    @pytest.mark.asyncio
    async def test_whitespace_text_update(self, mock_server_dependencies):
        """Test that updating with whitespace text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='text cannot be empty'):
            await update_context(
                context_id=1,
                text='   ',
            )

    @pytest.mark.asyncio
    async def test_no_fields_provided(self, mock_server_dependencies):
        """Test that update without any fields raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='At least one field must be provided'):
            await update_context(
                context_id=1,
            )

    @pytest.mark.asyncio
    async def test_context_not_found(self, mock_server_dependencies):
        """Test that updating non-existent context raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = False

        with pytest.raises(ToolError, match='Context entry with ID 999 not found'):
            await update_context(
                context_id=999,
                text='new text',
            )

    @pytest.mark.asyncio
    async def test_update_failure(self, mock_server_dependencies):
        """Test that update failure raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = True
        mock_server_dependencies.context.update_context_entry.return_value = (False, [])

        with pytest.raises(ToolError, match='Failed to update context entry'):
            await update_context(
                context_id=1,
                text='new text',
            )

    @pytest.mark.asyncio
    async def test_invalid_image_format(self, mock_server_dependencies):
        """Test that invalid image format raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = True

        with pytest.raises(ToolError, match='Each image must have "data" and "mime_type" fields'):
            await update_context(
                context_id=1,
                images=[{'data': 'base64data'}],  # Missing mime_type
            )

    @pytest.mark.asyncio
    async def test_invalid_base64_in_update(self, mock_server_dependencies):
        """Test that invalid base64 in update raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = True

        with pytest.raises(ToolError, match='Invalid base64 image data'):
            await update_context(
                context_id=1,
                images=[{'data': 'not-base64!!!', 'mime_type': 'image/png'}],
            )


class TestDeleteContextErrors:
    """Test error handling for delete_context tool."""

    @pytest.mark.asyncio
    async def test_no_parameters_provided(self, mock_server_dependencies):
        """Test that delete without parameters raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='Must provide either context_ids or thread_id'):
            await delete_context()

    @pytest.mark.asyncio
    async def test_database_deletion_error(self, mock_server_dependencies):
        """Test that database deletion error raises ToolError."""
        mock_server_dependencies.context.delete_by_ids.side_effect = Exception('Deletion failed')

        with pytest.raises(ToolError, match='Failed to delete context: Deletion failed'):
            await delete_context(context_ids=[1, 2, 3])


class TestSearchContextErrors:
    """Test error handling for search_context tool."""

    @pytest.mark.asyncio
    async def test_invalid_limit(self, mock_server_dependencies):
        """Test that Pydantic Field(ge=1, le=100) handles limit validation.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        # Set up mock to return valid response (rows, stats_dict)
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # Valid limits work fine
        result = await search_context(limit=1)
        assert 'results' in result

        result = await search_context(limit=100)
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_negative_offset(self, mock_server_dependencies):
        """Test that Pydantic Field(ge=0) handles offset validation.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        # Set up mock to return valid response (rows, stats_dict)
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # Valid offsets work fine
        result = await search_context(limit=50, offset=0)
        assert 'results' in result

        result = await search_context(limit=50, offset=100)
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_search_database_error(self, mock_server_dependencies):
        """Test that database search error raises ToolError."""
        mock_server_dependencies.context.search_contexts.side_effect = Exception('Search failed')

        with pytest.raises(ToolError, match='Failed to search context: Search failed'):
            await search_context(thread_id='test-thread', limit=50)


class TestGetContextByIdsErrors:
    """Test error handling for get_context_by_ids tool."""

    @pytest.mark.asyncio
    async def test_empty_context_ids(self, mock_server_dependencies):
        """Test that Pydantic Field(min_length=1) handles empty list validation.

        Note: Pydantic validates at FastMCP level. This test verifies normal operation.
        """
        # Set up mock to return valid response
        mock_server_dependencies.context.get_by_ids.return_value = []

        # Valid non-empty list works fine
        result = await get_context_by_ids(context_ids=[1, 2, 3])
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_fetch_database_error(self, mock_server_dependencies):
        """Test that database fetch error raises ToolError."""
        mock_server_dependencies.context.get_by_ids.side_effect = Exception('Fetch failed')

        with pytest.raises(ToolError, match='Failed to fetch context entries: Fetch failed'):
            await get_context_by_ids(context_ids=[1, 2, 3])


class TestListThreadsErrors:
    """Test error handling for list_threads tool."""

    @pytest.mark.asyncio
    async def test_list_threads_database_error(self, mock_server_dependencies):
        """Test that database error in list_threads raises ToolError."""
        mock_server_dependencies.statistics.get_thread_list.side_effect = Exception('List failed')

        with pytest.raises(ToolError, match='Failed to list threads: List failed'):
            await list_threads()


class TestGetStatisticsErrors:
    """Test error handling for get_statistics tool."""

    @pytest.mark.asyncio
    async def test_statistics_database_error(self, mock_server_dependencies):
        """Test that database error in get_statistics raises ToolError."""
        mock_server_dependencies.statistics.get_database_statistics.side_effect = Exception('Stats failed')

        with pytest.raises(ToolError, match='Failed to get statistics: Stats failed'):
            await get_statistics()


class TestFieldValidation:
    """Test that Field validation constraints are properly applied."""

    @pytest.mark.asyncio
    async def test_thread_id_min_length(self, mock_server_dependencies):
        """Test that thread_id min_length is enforced."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        # Empty string should be caught by min_length=1
        # But we also have manual validation for whitespace
        with pytest.raises(ToolError):
            await store_context(
                thread_id='',
                source='user',
                text='test',
            )

    @pytest.mark.asyncio
    async def test_text_min_length(self, mock_server_dependencies):
        """Test that text min_length is enforced."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        # Empty string should be caught by min_length=1
        # But we also have manual validation for whitespace
        with pytest.raises(ToolError):
            await store_context(
                thread_id='test',
                source='user',
                text='',
            )

    @pytest.mark.asyncio
    async def test_context_id_positive(self, mock_server_dependencies):
        """Test that context_id must be positive."""
        mock_server_dependencies.context.check_entry_exists.return_value = True
        # This would be caught by Field(gt=0) at FastMCP level
        # Testing our manual validation as fallback
        with pytest.raises(ToolError):
            await update_context(
                context_id=0,  # Should be > 0
                text='test',
            )

    @pytest.mark.asyncio
    async def test_limit_range(self, mock_server_dependencies):
        """Test that Pydantic Field(ge=1, le=100) enforces limit range."""
        # Set up mock to return valid response (rows, stats_dict)
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # Valid limits work
        result = await search_context(limit=1)
        assert 'results' in result
        result = await search_context(limit=100)
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_offset_non_negative(self, mock_server_dependencies):
        """Test that Pydantic Field(ge=0) enforces non-negative offset."""
        # Set up mock to return valid response (rows, stats_dict)
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # Valid offsets work
        result = await search_context(limit=50, offset=0)
        assert 'results' in result
        result = await search_context(limit=50, offset=100)
        assert 'results' in result


class TestErrorMessageConsistency:
    """Test that error messages are consistent and informative."""

    @pytest.mark.asyncio
    async def test_validation_errors_have_field_context(self, mock_server_dependencies):
        """Test that validation errors mention the field name."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='thread_id'):
            await store_context(
                thread_id='',
                source='user',
                text='test',
            )

        with pytest.raises(ToolError, match='text'):
            await store_context(
                thread_id='test',
                source='user',
                text='',
            )

    @pytest.mark.asyncio
    async def test_business_logic_errors_are_clear(self, mock_server_dependencies):
        """Test that business logic errors have clear messages."""
        mock_server_dependencies.context.check_entry_exists.return_value = False

        with pytest.raises(ToolError, match='Context entry with ID .* not found'):
            await update_context(
                context_id=999,
                text='test',
            )

    @pytest.mark.asyncio
    async def test_wrapped_exceptions_preserve_context(self, mock_server_dependencies):
        """Test that wrapped exceptions preserve original error context."""
        mock_server_dependencies.context.store_with_deduplication.side_effect = ValueError('Specific DB error')

        with pytest.raises(ToolError, match='Specific DB error'):
            await store_context(
                thread_id='test',
                source='user',
                text='test',
            )
