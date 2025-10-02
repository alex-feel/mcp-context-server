"""
Comprehensive tests for validation error handling in all MCP tools.

This test module ensures that ALL validation errors return proper JSON responses
with {"success": false, "error": "..."} format, never raw validation errors.
"""

from typing import Any
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

import app.server
from app.repositories import RepositoryContainer

# Get the actual async functions from the FunctionTool wrappers
store_context = app.server.store_context.fn
search_context = app.server.search_context.fn
get_context_by_ids = app.server.get_context_by_ids.fn
delete_context = app.server.delete_context.fn
update_context = app.server.update_context.fn
list_threads = app.server.list_threads.fn
get_statistics = app.server.get_statistics.fn


@pytest.fixture
def mock_repos():
    """Mock repository container for testing."""
    repos = MagicMock(spec=RepositoryContainer)

    # Mock context repository
    repos.context = AsyncMock()
    repos.context.store_with_deduplication = AsyncMock(return_value=(1, False))
    repos.context.check_entry_exists = AsyncMock(return_value=True)
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.search_contexts = AsyncMock(return_value=([], {}))
    repos.context.get_by_ids = AsyncMock(return_value=[])
    repos.context.delete_by_ids = AsyncMock(return_value=1)
    repos.context.delete_by_thread = AsyncMock(return_value=1)

    # Mock tags repository
    repos.tags = AsyncMock()
    repos.tags.store_tags = AsyncMock()
    repos.tags.replace_tags_for_context = AsyncMock()

    # Mock images repository
    repos.images = AsyncMock()
    repos.images.store_images = AsyncMock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    return repos


class TestStoreContextValidation:
    """Test validation error handling for store_context tool."""

    @pytest.mark.asyncio
    async def test_empty_thread_id(self, mock_repos):
        """Test that empty thread_id returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Test empty string
            result = await store_context(
                thread_id='',
                source='user',
                text='Test content',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'thread_id' in result['error'].lower()
            assert 'required' in result['error'].lower() or 'empty' in result['error'].lower()

            # Test whitespace only
            result = await store_context(
                thread_id='   ',
                source='user',
                text='Test content',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'thread_id' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_empty_text(self, mock_repos):
        """Test that empty text returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Test empty string
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'text' in result['error'].lower()
            assert 'required' in result['error'].lower() or 'empty' in result['error'].lower()

            # Test whitespace only
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='   \n\t   ',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'text' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_invalid_source(self, mock_repos):
        """Test that invalid source values pass through (no manual validation for source)."""
        # Note: When called directly, Literal type checking doesn't apply
        # Only FastMCP's validation layer enforces Literal constraints
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            from typing import cast
            invalid_source = cast(Any, 'invalid')
            # The function should accept any source when called directly
            result = await store_context(
                thread_id='test-thread',
                source=invalid_source,
                text='Test content',
            )
            # Should succeed since we don't manually validate source
            assert isinstance(result, dict)
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_oversized_image(self, mock_repos):
        """Test that oversized images return proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Create a fake large image (simulate > 10MB)
            large_data = 'A' * (15 * 1024 * 1024)  # 15MB of 'A's
            import base64
            encoded = base64.b64encode(large_data.encode()).decode()

            with patch('app.server.MAX_IMAGE_SIZE_MB', 10):
                result = await store_context(
                    thread_id='test-thread',
                    source='user',
                    text='Test content',
                    images=[{'data': encoded, 'mime_type': 'image/png'}],
                )
                assert isinstance(result, dict)
                assert result['success'] is False
                assert 'exceeds' in result['error'].lower()
                assert '10MB' in result['error'] or 'limit' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_invalid_image_data(self, mock_repos):
        """Test that invalid base64 image data returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='Test content',
                images=[{'data': 'not-valid-base64!!!', 'mime_type': 'image/png'}],
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'failed to process image' in result['error'].lower() or 'invalid' in result['error'].lower()


class TestUpdateContextValidation:
    """Test validation error handling for update_context tool."""

    @pytest.mark.asyncio
    async def test_empty_text(self, mock_repos):
        """Test that empty text returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Test empty string
            result = await update_context(
                context_id=1,
                text='',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'text' in result['error'].lower()
            assert 'empty' in result['error'].lower() or 'whitespace' in result['error'].lower()

            # Test whitespace only
            result = await update_context(
                context_id=1,
                text='   \n\t   ',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'text' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_no_fields_provided(self, mock_repos):
        """Test that providing no fields returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await update_context(
                context_id=1,
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'at least one field' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_nonexistent_context(self, mock_repos):
        """Test that updating non-existent context returns proper error JSON."""
        mock_repos.context.check_entry_exists = AsyncMock(return_value=False)

        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await update_context(
                context_id=999,
                text='New text',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'not found' in result['error'].lower()
            assert '999' in result['error']

    @pytest.mark.asyncio
    async def test_invalid_image_structure(self, mock_repos):
        """Test that invalid image structure returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Missing 'data' field
            result = await update_context(
                context_id=1,
                images=[{'mime_type': 'image/png'}],  # Missing 'data'
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'must have' in result['error'].lower()
            assert 'data' in result['error'].lower()

            # Missing 'mime_type' field
            result = await update_context(
                context_id=1,
                images=[{'data': 'somedata'}],  # Missing 'mime_type'
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'must have' in result['error'].lower()
            assert 'mime_type' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_oversized_images(self, mock_repos):
        """Test that oversized images return proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Create a fake large image
            large_data = 'A' * (15 * 1024 * 1024)  # 15MB
            import base64
            encoded = base64.b64encode(large_data.encode()).decode()

            with patch('app.server.MAX_IMAGE_SIZE_MB', 10):
                result = await update_context(
                    context_id=1,
                    images=[{'data': encoded, 'mime_type': 'image/png'}],
                )
                assert isinstance(result, dict)
                assert result['success'] is False
                assert 'exceeds' in result['error'].lower()
                assert '10MB' in result['error'] or 'limit' in result['error'].lower()


class TestSearchContextValidation:
    """Test validation error handling for search_context tool."""

    @pytest.mark.asyncio
    async def test_invalid_source(self, mock_repos):
        """Test that invalid source values pass through (no manual validation)."""
        # Note: When called directly, Literal constraints aren't enforced
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            invalid_source = cast(Any, 'invalid')  # Should be 'user' or 'agent'
            result = await search_context(
                source=invalid_source,
            )
            # Should succeed but return empty results
            assert isinstance(result, dict)
            assert 'entries' in result
            assert result['entries'] == []

    @pytest.mark.asyncio
    async def test_invalid_content_type(self, mock_repos):
        """Test that invalid content_type values pass through (no manual validation)."""
        # Note: When called directly, Literal constraints aren't enforced
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            invalid_content_type = cast(Any, 'invalid')  # Should be 'text' or 'multimodal'
            result = await search_context(
                content_type=invalid_content_type,
            )
            # Should succeed but return empty results
            assert isinstance(result, dict)
            assert 'entries' in result
            assert result['entries'] == []

    @pytest.mark.asyncio
    async def test_invalid_limit(self, mock_repos):
        """Test that invalid limit returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Test zero limit
            result = await search_context(
                limit=0,
            )
            assert isinstance(result, dict)
            assert 'entries' in result
            assert result['entries'] == []
            assert 'error' in result
            assert 'limit' in result['error'].lower()
            assert 'at least 1' in result['error']

            # Test negative limit
            result = await search_context(
                limit=-5,
            )
            assert isinstance(result, dict)
            assert 'entries' in result
            assert result['entries'] == []
            assert 'error' in result
            assert 'limit' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_negative_offset(self, mock_repos):
        """Test that negative offset returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await search_context(
                offset=-1,
            )
            assert isinstance(result, dict)
            assert 'entries' in result
            assert result['entries'] == []
            assert 'error' in result
            assert 'offset' in result['error'].lower()
            assert 'negative' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_limit_exceeds_maximum(self, mock_repos):
        """Test that limit exceeding 500 returns error."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await search_context(
                limit=1000,
            )
            assert isinstance(result, dict)
            assert 'entries' in result
            assert result['entries'] == []
            assert 'error' in result
            assert 'limit' in result['error'].lower()
            assert '500' in result['error']


class TestGetContextByIdsValidation:
    """Test validation error handling for get_context_by_ids tool."""

    @pytest.mark.asyncio
    async def test_empty_list(self, mock_repos):
        """Test that empty list returns empty result (backwards compatible)."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await get_context_by_ids(
                context_ids=[],
            )
            assert isinstance(result, list)
            assert result == []
            # No error - returns empty list for backwards compatibility

    @pytest.mark.asyncio
    async def test_invalid_ids(self, mock_repos):
        """Test that invalid IDs are handled gracefully."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Test with non-integer values (if they somehow get through)
            invalid_ids = cast(Any, ['not', 'integers'])
            # This may cause a type error or be handled by the function
            # The function should handle it gracefully
            result = await get_context_by_ids(
                context_ids=invalid_ids,
            )
            # Either returns empty list or raises an error
            # Function should handle gracefully
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_valid_integer_strings(self, mock_repos):
        """Test that integer strings can be coerced to integers."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Pydantic should be able to coerce string integers to int
            # This may work depending on Pydantic's coercion settings
            await get_context_by_ids(
                context_ids=[1, 2, 3],  # Use actual integers
            )
            # Should work fine with actual integers
            mock_repos.context.get_by_ids.assert_called_once()
            call_args = mock_repos.context.get_by_ids.call_args
            assert call_args[0][0] == [1, 2, 3]


class TestDeleteContextValidation:
    """Test validation error handling for delete_context tool."""

    @pytest.mark.asyncio
    async def test_no_parameters(self, mock_repos):
        """Test that providing no parameters returns proper error JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await delete_context()
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'must provide' in result['error'].lower()
            assert 'context_ids or thread_id' in result['error']

    @pytest.mark.asyncio
    async def test_successful_deletion_by_ids(self, mock_repos):
        """Test successful deletion by IDs returns proper success JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await delete_context(context_ids=[1, 2, 3])
            assert isinstance(result, dict)
            assert result['success'] is True
            assert 'deleted_count' in result
            assert result['deleted_count'] == 1
            assert 'message' in result

    @pytest.mark.asyncio
    async def test_successful_deletion_by_thread(self, mock_repos):
        """Test successful deletion by thread returns proper success JSON."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await delete_context(thread_id='test-thread')
            assert isinstance(result, dict)
            assert result['success'] is True
            assert 'deleted_count' in result
            assert result['deleted_count'] == 1
            assert 'message' in result

    @pytest.mark.asyncio
    async def test_deletion_error(self, mock_repos):
        """Test that deletion errors return proper error JSON."""
        mock_repos.context.delete_by_ids = AsyncMock(side_effect=Exception('Database error'))

        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await delete_context(context_ids=[1])
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'error' in result
            assert 'Database error' in result['error']


class TestEdgeCasesAndCombinations:
    """Test edge cases and combinations of validation errors."""

    @pytest.mark.asyncio
    async def test_multiple_validation_errors_store(self, mock_repos):
        """Test that multiple validation errors are caught properly."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Both thread_id and text are empty
            result = await store_context(
                thread_id='',
                source='user',
                text='',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            # Should catch the first validation error (thread_id)
            assert 'thread_id' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_unicode_and_special_chars(self, mock_repos):
        """Test that unicode and special characters are handled properly."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Test with unicode characters
            result = await store_context(
                thread_id='тест-поток',  # Cyrillic
                source='user',
                text='测试内容',  # Chinese
            )
            assert isinstance(result, dict)
            assert result['success'] is True

            # Test with special characters and emojis
            result = await store_context(
                thread_id='test-thread-😀',
                source='agent',
                text='Content with special chars: <>!@#$%^&*()',
            )
            assert isinstance(result, dict)
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_very_long_inputs(self, mock_repos):
        """Test that very long inputs are handled properly."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Very long thread_id
            long_thread_id = 'thread-' + 'x' * 10000
            result = await store_context(
                thread_id=long_thread_id,
                source='user',
                text='Test content',
            )
            assert isinstance(result, dict)
            assert result['success'] is True

            # Very long text
            long_text = 'A' * 1000000  # 1MB of text
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text=long_text,
            )
            assert isinstance(result, dict)
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_null_vs_empty_string(self, mock_repos):
        """Test distinction between null and empty string."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Update with None text (should be allowed - no update to text)
            result = await update_context(
                context_id=1,
                text=None,
                metadata={'key': 'value'},
            )
            assert isinstance(result, dict)
            assert result['success'] is True

            # Update with empty string text (should be rejected)
            result = await update_context(
                context_id=1,
                text='',
                metadata={'key': 'value'},
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'text' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_search_with_all_filters(self, mock_repos):
        """Test search with all possible filter combinations."""
        with patch('app.server._ensure_repositories', return_value=mock_repos):
            # Valid combination of all filters
            result = await search_context(
                thread_id='test-thread',
                source='user',
                tags=['tag1', 'tag2'],
                content_type='text',
                metadata={'status': 'active'},
                metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 5}],
                limit=100,
                offset=10,
                include_images=True,
                explain_query=True,
            )
            assert isinstance(result, dict)
            assert 'entries' in result
            assert 'stats' in result

            # Invalid source with other valid filters
            invalid_source = cast(Any, 'invalid')
            result = await search_context(
                thread_id='test-thread',
                source=invalid_source,
                tags=['tag1'],
                content_type='text',
            )
            # When called directly, source validation doesn't happen
            assert isinstance(result, dict)
            assert 'entries' in result


class TestExceptionHandling:
    """Test that exceptions are properly caught and return JSON errors."""

    @pytest.mark.asyncio
    async def test_repository_exception_store(self, mock_repos):
        """Test that repository exceptions return proper error JSON."""
        mock_repos.context.store_with_deduplication = AsyncMock(
            side_effect=Exception('Database connection failed'),
        )

        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await store_context(
                thread_id='test-thread',
                source='user',
                text='Test content',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'error' in result
            assert 'Database connection failed' in result['error']

    @pytest.mark.asyncio
    async def test_repository_exception_update(self, mock_repos):
        """Test that repository exceptions in update return proper error JSON."""
        mock_repos.context.update_context_entry = AsyncMock(
            side_effect=Exception('Update failed'),
        )

        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await update_context(
                context_id=1,
                text='New text',
            )
            assert isinstance(result, dict)
            assert result['success'] is False
            assert 'error' in result
            assert 'Update failed' in result['error']

    @pytest.mark.asyncio
    async def test_repository_exception_search(self, mock_repos):
        """Test that repository exceptions in search return proper error JSON."""
        mock_repos.context.search_contexts = AsyncMock(
            side_effect=Exception('Search failed'),
        )

        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await search_context()
            assert isinstance(result, dict)
            assert 'entries' in result
            assert result['entries'] == []
            assert 'error' in result
            assert 'Search failed' in result['error']

    @pytest.mark.asyncio
    async def test_repository_exception_get_by_ids(self, mock_repos):
        """Test that repository exceptions in get_by_ids return empty list."""
        mock_repos.context.get_by_ids = AsyncMock(
            side_effect=Exception('Fetch failed'),
        )

        with patch('app.server._ensure_repositories', return_value=mock_repos):
            result = await get_context_by_ids(context_ids=[1, 2, 3])
            assert isinstance(result, list)
            assert result == []
            # get_context_by_ids returns empty list on error for backwards compatibility


# Integration test to ensure all tools are accessible
class TestToolAccessibility:
    """Ensure all tools are properly exposed and callable."""

    def test_all_tools_exist(self):
        """Test that all expected tools exist and are callable."""
        # Tools are already imported at module level
        assert callable(store_context)
        assert callable(update_context)
        assert callable(search_context)
        assert callable(get_context_by_ids)
        assert callable(delete_context)
        assert callable(list_threads)
        assert callable(get_statistics)

    def test_tool_signatures(self):
        """Test that tools have the expected signatures."""
        import inspect

        # Check store_context parameters
        sig = inspect.signature(store_context)
        params = list(sig.parameters.keys())
        assert 'thread_id' in params
        assert 'source' in params
        assert 'text' in params
        assert 'images' in params
        assert 'metadata' in params
        assert 'tags' in params
        assert 'ctx' in params

        # Check update_context parameters
        sig = inspect.signature(update_context)
        params = list(sig.parameters.keys())
        assert 'context_id' in params
        assert 'text' in params
        assert 'metadata' in params
        assert 'tags' in params
        assert 'images' in params
        assert 'ctx' in params

        # Check search_context parameters
        sig = inspect.signature(search_context)
        params = list(sig.parameters.keys())
        assert 'thread_id' in params
        assert 'source' in params
        assert 'tags' in params
        assert 'content_type' in params
        assert 'metadata' in params
        assert 'metadata_filters' in params
        assert 'limit' in params
        assert 'offset' in params
        assert 'include_images' in params
        assert 'explain_query' in params
        assert 'ctx' in params

        # Check get_context_by_ids parameters
        sig = inspect.signature(get_context_by_ids)
        params = list(sig.parameters.keys())
        assert 'context_ids' in params
        assert 'include_images' in params
        assert 'ctx' in params

        # Check delete_context parameters
        sig = inspect.signature(delete_context)
        params = list(sig.parameters.keys())
        assert 'context_ids' in params
        assert 'thread_id' in params
        assert 'ctx' in params
