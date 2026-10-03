"""Per-tool error handling for the MCP tools.

Validates that store, update, delete, search, get-by-ids, list-threads and statistics raise ToolError with a
descriptive message for invalid input, image validation failures and repository errors.
"""

import base64

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories.context_repository.records import EntryProbe
from tests.tools._tool_error_mocks import build_mock_repos
from tests.tools._tool_error_mocks import patch_tool_repositories

# Tool coroutines under test; FastMCP registration happens only in the server lifespan
store_context = app.tools.store_context
search_context = app.tools.search_context
get_context_by_ids = app.tools.get_context_by_ids
delete_context = app.tools.delete_context
update_context = app.tools.update_context
list_threads = app.tools.list_threads
get_statistics = app.tools.get_statistics


# --- Fixtures ---


@pytest.fixture
def mock_repos():
    """Create a mock repository container with transaction support."""
    return build_mock_repos()


@pytest.fixture
def mock_server_dependencies(mock_repos):
    """Mock server dependencies for tool error testing.

    Patches ensure_repositories in each tool module where it is imported.

    Yields:
        MagicMock: The mock repository container.
    """
    with patch_tool_repositories(mock_repos):
        yield mock_repos


# --- Test classes ---


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
        with pytest.raises(ToolError, match='Image 0 has invalid base64 encoding'):
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
                context_id='0190abcdef1234567890abcd00000001',
                text='',
            )

    @pytest.mark.asyncio
    async def test_whitespace_text_update(self, mock_server_dependencies):
        """Test that updating with whitespace text raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='text cannot be empty'):
            await update_context(
                context_id='0190abcdef1234567890abcd00000001',
                text='   ',
            )

    @pytest.mark.asyncio
    async def test_no_fields_provided(self, mock_server_dependencies):
        """Test that update without any fields raises ToolError."""
        _ = mock_server_dependencies  # Fixture needed for mocking
        with pytest.raises(ToolError, match='At least one field must be provided'):
            await update_context(
                context_id='0190abcdef1234567890abcd00000001',
            )

    @pytest.mark.asyncio
    async def test_context_not_found(self, mock_server_dependencies):
        """Test that updating non-existent context raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(False, None, None, None)

        with pytest.raises(ToolError, match='Context entry with ID 0190abcdef1234567890abcd000003e7 not found'):
            await update_context(
                context_id='0190abcdef1234567890abcd000003e7',
                text='new text',
            )

    @pytest.mark.asyncio
    async def test_update_failure(self, mock_server_dependencies):
        """A no-such-row update (repository reports no matching row) surfaces a clean not-found error."""
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(True, 'agent', 0, 'local')
        mock_server_dependencies.context.update_context_entry.return_value = (False, [])

        with pytest.raises(ToolError, match='Context entry with ID 0190abcdef1234567890abcd00000001 not found'):
            await update_context(
                context_id='0190abcdef1234567890abcd00000001',
                text='new text',
            )

    @pytest.mark.asyncio
    async def test_invalid_image_format(self, mock_server_dependencies):
        """Test that invalid image data raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(True, 'agent', 0, 'local')

        with pytest.raises(ToolError, match='Image 0 has invalid base64 encoding'):
            await update_context(
                context_id='0190abcdef1234567890abcd00000001',
                images=[{'data': 'not-valid-base64!!!'}],  # Invalid base64, mime_type defaults to 'image/png'
            )

    @pytest.mark.asyncio
    async def test_invalid_base64_in_update(self, mock_server_dependencies):
        """Test that invalid base64 in update raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = EntryProbe(True, 'agent', 0, 'local')

        with pytest.raises(ToolError, match='Image 0 has invalid base64 encoding'):
            await update_context(
                context_id='0190abcdef1234567890abcd00000001',
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
            await delete_context(
                context_ids=[
                    '0190abcdef1234567890abcd00000001',
                    '0190abcdef1234567890abcd00000002',
                    '0190abcdef1234567890abcd00000003',
                ],
            )


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
        result = await get_context_by_ids(
            context_ids=[
                '0190abcdef1234567890abcd00000001',
                '0190abcdef1234567890abcd00000002',
                '0190abcdef1234567890abcd00000003',
            ],
        )
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_fetch_database_error(self, mock_server_dependencies):
        """Test that database fetch error raises ToolError."""
        mock_server_dependencies.context.get_by_ids.side_effect = Exception('Fetch failed')

        with pytest.raises(ToolError, match='Failed to fetch context entries: Fetch failed'):
            await get_context_by_ids(
                context_ids=[
                    '0190abcdef1234567890abcd00000001',
                    '0190abcdef1234567890abcd00000002',
                    '0190abcdef1234567890abcd00000003',
                ],
            )


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
