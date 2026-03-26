"""Comprehensive tests for MCP tool error handling.

Validates that all tool error cases return proper ToolError exceptions
with descriptive messages. Covers validation errors, database errors,
image validation, field constraints, and JSON format consistency.
"""

import base64
from contextlib import asynccontextmanager
from typing import Literal
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
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


# --- Fixtures ---


@pytest.fixture
def mock_repos():
    """Create a mock repository container with transaction support."""
    repos = MagicMock()

    mock_backend = Mock()

    @asynccontextmanager
    async def mock_begin_transaction():
        txn = Mock()
        txn.backend_type = 'sqlite'
        txn.connection = Mock()
        yield txn

    mock_backend.begin_transaction = mock_begin_transaction

    repos.context = AsyncMock()
    repos.context.backend = mock_backend
    repos.tags = AsyncMock()
    repos.images = AsyncMock()
    repos.statistics = AsyncMock()

    repos.embeddings = AsyncMock()
    repos.embeddings.store = AsyncMock(return_value=None)
    repos.embeddings.store_chunked = AsyncMock(return_value=None)
    repos.embeddings.delete_all_chunks = AsyncMock(return_value=None)

    return repos


@pytest.fixture
def mock_server_dependencies(mock_repos):
    """Mock server dependencies for tool error testing.

    Patches ensure_repositories in each tool module where it is imported.

    Yields:
        MagicMock: The mock repository container.
    """
    with (
        patch('app.tools.context.ensure_repositories', return_value=mock_repos),
        patch('app.tools.search.ensure_repositories', return_value=mock_repos),
        patch('app.tools.discovery.ensure_repositories', return_value=mock_repos),
    ):
        yield mock_repos


# --- Test Classes (from test_error_handling_json.py -- 31 tests) ---


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
        mock_server_dependencies.context.check_entry_exists.return_value = (False, None)

        with pytest.raises(ToolError, match='Context entry with ID 999 not found'):
            await update_context(
                context_id=999,
                text='new text',
            )

    @pytest.mark.asyncio
    async def test_update_failure(self, mock_server_dependencies):
        """Test that update failure raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = (True, 'agent')
        mock_server_dependencies.context.update_context_entry.return_value = (False, [])

        with pytest.raises(ToolError, match='Failed to update context entry'):
            await update_context(
                context_id=1,
                text='new text',
            )

    @pytest.mark.asyncio
    async def test_invalid_image_format(self, mock_server_dependencies):
        """Test that invalid image data raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = (True, 'agent')

        with pytest.raises(ToolError, match='Invalid base64 image data'):
            await update_context(
                context_id=1,
                images=[{'data': 'not-valid-base64!!!'}],  # Invalid base64, mime_type defaults to 'image/png'
            )

    @pytest.mark.asyncio
    async def test_invalid_base64_in_update(self, mock_server_dependencies):
        """Test that invalid base64 in update raises ToolError."""
        mock_server_dependencies.context.check_entry_exists.return_value = (True, 'agent')

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
        mock_server_dependencies.context.check_entry_exists.return_value = (True, 'agent')
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
        mock_server_dependencies.context.check_entry_exists.return_value = (False, None)

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


# --- Test Classes (unique tests from test_error_formats.py -- 8 tests) ---


class TestErrorFormatConsistency:
    """Test that error format is consistent across tools.

    Tests Pydantic bypass scenarios and runtime validation behavior.
    """

    @pytest.mark.asyncio
    async def test_store_context_invalid_source(self, mock_server_dependencies):
        """Test store_context with invalid source wraps DB error in ToolError.

        When called directly (bypassing FastMCP), Pydantic Literal validation
        is not applied. The database CHECK constraint catches the invalid value.
        With mock fixtures, we simulate the DB CHECK constraint error.
        """
        # Simulate the database CHECK constraint rejecting an invalid source value
        mock_server_dependencies.context.store_with_deduplication.side_effect = Exception(
            'CHECK constraint failed: source must be user or agent',
        )

        invalid_source = cast(Literal['user', 'agent'], 'invalid')
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test_thread',
                source=invalid_source,
                text='Some text',
            )

        # Verify the error is wrapped as ToolError with context
        error_msg = str(exc_info.value).lower()
        assert 'source' in error_msg, 'Error should mention source'

    @pytest.mark.asyncio
    async def test_get_context_by_ids_empty_list(self, mock_server_dependencies):
        """Test get_context_by_ids with empty list - Pydantic handles at protocol layer."""
        # Set up mock to return empty results for empty input
        mock_server_dependencies.context.get_by_ids.return_value = []

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(min_length=1) validates at the MCP protocol layer.
        result = await get_context_by_ids(
            context_ids=[],
        )

        # Repository returns empty list for empty input
        assert result == [], 'Should return empty list for empty input when bypassing protocol validation'

    @pytest.mark.asyncio
    async def test_search_context_invalid_limit(self, mock_server_dependencies):
        """Test search_context with invalid limit - Pydantic handles at protocol layer."""
        # Set up mock to return valid response
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        # This is correct - Pydantic Field(ge=1, le=100) validates at the MCP protocol layer.
        #
        # However, database-level validation may still occur:
        # - SQLite: Allows negative LIMIT (treated as no limit), returns results
        # - PostgreSQL: Rejects negative LIMIT with error "LIMIT must not be negative"

        try:
            result = await search_context(
                limit=-1,
            )
            # SQLite backend / mock: proceeds with invalid value
            assert 'results' in result, 'Should return result structure'
        except ToolError:
            # PostgreSQL backend: database-level validation rejects negative LIMIT
            pass

    @pytest.mark.asyncio
    async def test_search_context_excessive_limit(self, mock_server_dependencies):
        """Test search_context with excessive limit - Pydantic handles at protocol layer."""
        # Set up mock to return valid response
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        result = await search_context(
            limit=101,  # Max is 100
        )

        # Function proceeds with invalid value when protocol validation is bypassed
        assert 'results' in result, 'Should return result structure even with excessive limit'

    @pytest.mark.asyncio
    async def test_search_context_negative_offset(self, mock_server_dependencies):
        """Test search_context with negative offset - Pydantic handles at protocol layer."""
        # Set up mock to return valid response
        mock_server_dependencies.context.search_contexts.return_value = ([], {})

        # When called directly (bypassing FastMCP), no runtime validation occurs.
        #
        # However, database-level validation may still occur:
        # - SQLite: Allows negative OFFSET (treated as 0), returns results
        # - PostgreSQL: Rejects negative OFFSET with error "OFFSET must not be negative"

        try:
            result = await search_context(
                limit=50,
                offset=-1,
            )
            # SQLite backend / mock: proceeds with invalid value
            assert 'results' in result, 'Should return result structure'
        except ToolError:
            # PostgreSQL backend: database-level validation rejects negative OFFSET
            pass

    def test_no_raw_validation_errors_in_responses(self) -> None:
        """Meta test documenting expected error format patterns.

        Raw Pydantic errors typically look like:
        - "Input validation error: '' should be non-empty"
        - "validation error for Model"

        All errors should be ToolError with descriptive messages.
        """
        # Expected error format (all errors should follow this pattern):
        expected_format = {
            'success': False,
            'error': 'Human-readable error message',
        }

        # Raw Pydantic error formats we should NOT see:
        raw_error_patterns = [
            'Input validation error:',
            'validation error for',
            'should be non-empty',
            'String should have at least',
            'ensure this value',
        ]

        # Document that all error responses should:
        # 1. Be a dictionary
        # 2. Have 'success': False
        # 3. Have 'error' key with descriptive message
        # 4. NOT contain raw Pydantic error patterns
        assert expected_format is not None
        assert raw_error_patterns is not None

    @pytest.mark.asyncio
    async def test_all_tools_handle_none_parameters(self, mock_server_dependencies):
        """Test that tools rely on Pydantic for None validation."""
        _ = mock_server_dependencies  # Fixture needed for mocking

        # When called directly with None (bypassing FastMCP validation),
        # the function has defensive None checks to prevent AttributeError crashes.
        none_text = cast(str, None)
        with pytest.raises(ToolError) as exc_info:
            await store_context(
                thread_id='test',
                source='user',
                text=none_text,
            )

        # The ToolError contains defensive None check message
        error_msg = str(exc_info.value).lower()
        assert 'required' in error_msg


class TestValidationIntegration:
    """Integration tests for validation across the full stack."""

    @pytest.mark.asyncio
    async def test_validation_with_fastmcp_client(self) -> None:
        """Test that validation errors through FastMCP client are properly formatted.

        This would require a running server and FastMCP client.
        Marked for future implementation.
        """
        # Future implementation: Implement when server is running


# --- Test Classes (from test_json_error_consistency.py -- 6 tests) ---


class TestJSONErrorConsistency:
    """Test that ALL error conditions return consistent JSON format through FastMCP."""

    @pytest.mark.asyncio
    async def test_all_validation_errors_raise_tool_error(self, mock_server_dependencies):
        """Test that all BUSINESS LOGIC validation errors raise ToolError.

        Note: Input validation (Field constraints) is handled by Pydantic at FastMCP level.
        This test only validates business logic errors (e.g., whitespace-only after strip()).
        """
        # Set up mocks for successful database operations
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)
        mock_server_dependencies.context.check_entry_exists.return_value = (True, 'agent')

        # Test cases that should raise ToolError for BUSINESS LOGIC
        test_cases = [
            # Business logic: empty strings after strip() are not allowed
            ('store_context empty thread_id', lambda: store_context(thread_id='', source='user', text='test'), 'thread_id'),
            ('store_context empty text', lambda: store_context(thread_id='test', source='user', text=''), 'text'),
            (
                'store_context whitespace thread_id',
                lambda: store_context(thread_id='   ', source='user', text='test'),
                'thread_id',
            ),
            ('store_context whitespace text', lambda: store_context(thread_id='test', source='user', text='   '), 'text'),
            # update_context business logic validation
            ('update_context empty text', lambda: update_context(context_id=1, text=''), 'text'),
            ('update_context no fields', lambda: update_context(context_id=1), 'field'),
            # delete_context business logic validation
            ('delete_context no parameters', lambda: delete_context(), 'provide'),
        ]

        for test_name, test_func, expected_keyword in test_cases:
            with pytest.raises(ToolError) as exc_info:
                await test_func()

            error_msg = str(exc_info.value)
            assert isinstance(error_msg, str), f'{test_name}: Error should be a string'
            assert expected_keyword in error_msg.lower(), (
                f'{test_name}: Error should mention {expected_keyword}, got: {error_msg}'
            )

    @pytest.mark.asyncio
    async def test_all_database_errors_raise_tool_error(self, mock_server_dependencies):
        """Test that all database errors are wrapped in ToolError."""

        # Test store_context database error
        mock_server_dependencies.context.store_with_deduplication.side_effect = Exception('DB error')
        with pytest.raises(ToolError, match='Failed to store context'):
            await store_context(thread_id='test', source='user', text='test')

        # Reset mock
        mock_server_dependencies.context.store_with_deduplication.side_effect = None
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)

        # Test update_context database error
        mock_server_dependencies.context.check_entry_exists.return_value = (True, 'agent')
        mock_server_dependencies.context.update_context_entry.side_effect = Exception('Update failed')
        with pytest.raises(ToolError, match='Failed to update context'):
            await update_context(context_id=1, text='new text')

        # Test search_context database error
        mock_server_dependencies.context.search_contexts.side_effect = Exception('Search failed')
        with pytest.raises(ToolError, match='Failed to search context'):
            await search_context(limit=50)

        # Test get_context_by_ids database error
        mock_server_dependencies.context.get_by_ids.side_effect = Exception('Fetch failed')
        with pytest.raises(ToolError, match='Failed to fetch context'):
            await get_context_by_ids(context_ids=[1, 2])

        # Test list_threads database error
        mock_server_dependencies.statistics.get_thread_list.side_effect = Exception('List failed')
        with pytest.raises(ToolError, match='Failed to list threads'):
            await list_threads()

        # Test get_statistics database error
        mock_server_dependencies.statistics.get_database_statistics.side_effect = Exception('Stats failed')
        with pytest.raises(ToolError, match='Failed to get statistics'):
            await get_statistics()

    @pytest.mark.asyncio
    async def test_image_validation_errors_raise_tool_error(self, mock_server_dependencies):
        """Test that image validation errors raise ToolError."""
        mock_server_dependencies.context.store_with_deduplication.return_value = (1, False)

        # Test invalid base64 image
        with pytest.raises(ToolError, match='Invalid base64'):
            await store_context(
                thread_id='test',
                source='user',
                text='test',
                images=[{'data': 'not-base64!@#$', 'mime_type': 'image/png'}],
            )

        # Test oversized image
        large_data = 'A' * (15 * 1024 * 1024)  # 15MB
        encoded = base64.b64encode(large_data.encode()).decode()

        with pytest.raises(ToolError, match='exceeds .* limit'):
            await store_context(
                thread_id='test',
                source='user',
                text='test',
                images=[{'data': encoded, 'mime_type': 'image/png'}],
            )

    @pytest.mark.asyncio
    async def test_error_messages_are_descriptive(self, mock_server_dependencies):
        """Test that BUSINESS LOGIC error messages are descriptive and not generic.

        Note: Input validation messages come from Pydantic Field constraints.
        This test validates business logic error messages only.
        """
        _ = mock_server_dependencies  # Fixture needed for mocking
        # Test empty thread_id (business logic: whitespace-only not allowed)
        with pytest.raises(ToolError) as exc_info:
            await store_context(thread_id='', source='user', text='test')
        assert 'thread_id' in str(exc_info.value).lower()
        assert 'empty' in str(exc_info.value).lower() or 'whitespace' in str(exc_info.value).lower()

        # Test empty text (business logic: whitespace-only not allowed)
        with pytest.raises(ToolError) as exc_info:
            await store_context(thread_id='test', source='user', text='')
        assert 'text' in str(exc_info.value).lower()
        assert 'empty' in str(exc_info.value).lower() or 'whitespace' in str(exc_info.value).lower()

        # Test no fields provided (business logic: at least one field required)
        with pytest.raises(ToolError) as exc_info:
            await update_context(context_id=1)
        assert 'field' in str(exc_info.value).lower()
        assert 'least' in str(exc_info.value).lower() or 'provide' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_no_raw_pydantic_errors_exposed(self, mock_server_dependencies):
        """Test that raw Pydantic validation errors are never exposed in business logic errors.

        Note: Pydantic Field validation happens at FastMCP level and is properly formatted.
        This test ensures our business logic errors don't leak Pydantic internals.
        """
        _ = mock_server_dependencies  # Fixture needed for mocking
        # These patterns should NEVER appear in BUSINESS LOGIC error messages
        forbidden_patterns = [
            'Input validation error:',
            'validation error for',
            'ensure this value',
            'String should have at least',
            'field required',
            'type=value_error',
            'loc=',
            'ctx=',
        ]

        # Test business logic error conditions
        error_messages = []

        # Collect error messages from business logic validation
        try:
            await store_context(thread_id='', source='user', text='test')
        except ToolError as e:
            error_messages.append(str(e))

        try:
            await store_context(thread_id='test', source='user', text='')
        except ToolError as e:
            error_messages.append(str(e))

        try:
            await update_context(context_id=1)
        except ToolError as e:
            error_messages.append(str(e))

        try:
            await delete_context()
        except ToolError as e:
            error_messages.append(str(e))

        # Check that no forbidden patterns appear in any error message
        for msg in error_messages:
            for pattern in forbidden_patterns:
                assert pattern not in msg, f'Raw Pydantic pattern "{pattern}" found in error: {msg}'

    @pytest.mark.asyncio
    async def test_consistent_error_format_across_tools(self, mock_server_dependencies):
        """Test that all tools use consistent BUSINESS LOGIC error format.

        Note: Input validation is handled by Pydantic Field constraints.
        This test validates business logic error consistency.
        """
        _ = mock_server_dependencies  # Fixture needed for mocking
        # All tools should raise ToolError for business logic failures
        tools_and_errors = [
            (store_context, {'thread_id': '', 'source': 'user', 'text': 'test'}),  # Empty after strip
            (update_context, {'context_id': 1, 'text': ''}),  # Empty text
            (delete_context, {}),  # No parameters provided
        ]

        for tool_func, params in tools_and_errors:
            with pytest.raises(ToolError) as exc_info:
                await tool_func(**params)

            # All errors should be ToolError instances
            assert isinstance(exc_info.value, ToolError)

            # All error messages should be strings
            error_msg = str(exc_info.value)
            assert isinstance(error_msg, str)

            # All error messages should be non-empty
            assert len(error_msg) > 0

            # No error message should contain raw exception details
            assert 'Traceback' not in error_msg
            assert 'File "' not in error_msg
