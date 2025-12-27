"""
Comprehensive test to verify ALL errors return consistent JSON format.

This test validates that when errors occur, FastMCP with mask_error_details=False
converts all ToolError exceptions into consistent JSON error responses.
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
        mock_server_dependencies.context.check_entry_exists.return_value = True

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
        mock_server_dependencies.context.check_entry_exists.return_value = True
        mock_server_dependencies.context.update_context_entry.side_effect = Exception('Update failed')
        with pytest.raises(ToolError, match='Update operation failed'):
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
        import base64

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
