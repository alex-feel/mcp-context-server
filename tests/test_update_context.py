"""
Comprehensive tests for the update_context tool.

Tests cover individual field updates, combined updates, validation,
error handling, and transaction safety.
"""

import json
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import app.server
import pytest
from fastmcp import Context

# Get the actual async function from the FunctionTool wrapper
# FastMCP wraps our function in a FunctionTool object, but we need the original function for testing
update_context = app.server.update_context.fn


@pytest.fixture
def mock_context():
    """Create a mock FastMCP context for testing."""
    ctx = Mock(spec=Context)
    ctx.info = AsyncMock()
    return ctx


@pytest.fixture
def mock_repositories():
    """Create mock repository container with all necessary repositories."""
    repos = Mock()

    # Mock context repository
    repos.context = Mock()
    repos.context.check_entry_exists = AsyncMock(return_value=True)
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)

    # Mock tags repository
    repos.tags = Mock()
    repos.tags.replace_tags_for_context = AsyncMock()

    # Mock images repository
    repos.images = Mock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    return repos


class TestUpdateContext:
    """Test suite for update_context tool."""

    @pytest.mark.asyncio
    async def test_update_text_content_only(self, mock_context, mock_repositories):
        """Test updating only text content."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=123,
                text='Updated text content',
                metadata=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert result['context_id'] == 123
            assert 'text_content' in result['updated_fields']
            assert result['message'] == 'Successfully updated 1 field(s)'

            # Verify repository calls
            mock_repositories.context.check_entry_exists.assert_called_once_with(123)
            mock_repositories.context.update_context_entry.assert_called_once_with(
                context_id=123,
                text_content='Updated text content',
                metadata=None,
            )

    @pytest.mark.asyncio
    async def test_update_metadata_only(self, mock_context, mock_repositories):
        """Test updating only metadata."""
        metadata = {'status': 'completed', 'priority': 5}
        mock_repositories.context.update_context_entry.return_value = (True, ['metadata'])

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=456,
                text=None,
                metadata=metadata,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert result['context_id'] == 456
            assert 'metadata' in result['updated_fields']

            # Verify metadata was JSON-encoded
            expected_metadata_str = json.dumps(metadata)
            mock_repositories.context.update_context_entry.assert_called_once_with(
                context_id=456,
                text_content=None,
                metadata=expected_metadata_str,
            )

    @pytest.mark.asyncio
    async def test_update_tags_only(self, mock_context, mock_repositories):
        """Test replacing tags."""
        tags = ['python', 'testing', 'async']

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=789,
                text=None,
                metadata=None,
                tags=tags,
                images=None,
                ctx=mock_context,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'tags' in result['updated_fields']

            # Verify tags were replaced
            mock_repositories.tags.replace_tags_for_context.assert_called_once_with(789, tags)

    @pytest.mark.asyncio
    async def test_update_images_with_content_type_change(self, mock_context, mock_repositories):
        """Test replacing images and updating content_type to multimodal."""
        images = [
            {
                'data': 'aGVsbG8gd29ybGQ=',  # base64 encoded "hello world"
                'mime_type': 'image/png',
            },
        ]

        with patch('app.server._ensure_repositories', return_value=mock_repositories), \
             patch('app.server.MAX_IMAGE_SIZE_MB', 10), \
             patch('app.server.MAX_TOTAL_SIZE_MB', 100):
            result = await update_context(
                context_id=111,
                text=None,
                metadata=None,
                tags=None,
                images=images,
                ctx=mock_context,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'images' in result['updated_fields']
            assert 'content_type' in result['updated_fields']

            # Verify images were replaced and content_type updated
            mock_repositories.images.replace_images_for_context.assert_called_once_with(111, images)
            mock_repositories.context.update_content_type.assert_called_once_with(111, 'multimodal')

    @pytest.mark.asyncio
    async def test_remove_all_images_updates_content_type(self, mock_context, mock_repositories):
        """Test that providing empty images list removes images and sets content_type to text."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=222,
                text=None,
                metadata=None,
                tags=None,
                images=[],  # Empty list removes all images
                ctx=mock_context,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'images' in result['updated_fields']
            assert 'content_type' in result['updated_fields']

            # Verify images were cleared and content_type set to text
            mock_repositories.images.replace_images_for_context.assert_called_once_with(222, [])
            mock_repositories.context.update_content_type.assert_called_once_with(222, 'text')

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, mock_context, mock_repositories):
        """Test updating multiple fields in one call."""
        mock_repositories.context.update_context_entry.return_value = (True, ['text_content', 'metadata'])

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=333,
                text='New text',
                metadata={'key': 'value'},
                tags=['tag1', 'tag2'],
                images=None,
                ctx=mock_context,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'text_content' in result['updated_fields']
            assert 'metadata' in result['updated_fields']
            assert 'tags' in result['updated_fields']
            assert len(result['updated_fields']) == 3

    @pytest.mark.asyncio
    async def test_no_fields_provided_error(self, mock_context, mock_repositories):
        """Test error when no fields are provided for update."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=444,
                text=None,
                metadata=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert result['context_id'] == 444
            assert 'at least one field' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_context_not_found_error(self, mock_context, mock_repositories):
        """Test error when context entry doesn't exist."""
        mock_repositories.context.check_entry_exists.return_value = False

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=999,
                text='Some text',
                metadata=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert result['context_id'] == 999
            assert 'not found' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_invalid_image_data(self, mock_context, mock_repositories):
        """Test error with invalid image data."""
        images = [
            {
                'data': 'not-valid-base64!!!',
                'mime_type': 'image/png',
            },
        ]

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=555,
                text=None,
                metadata=None,
                tags=None,
                images=images,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert 'invalid base64' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_missing_image_fields(self, mock_context, mock_repositories):
        """Test error when image is missing required fields."""
        images = [
            {
                'data': 'aGVsbG8=',
                # Missing mime_type
            },
        ]

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=666,
                text=None,
                metadata=None,
                tags=None,
                images=images,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert 'must have "data" and "mime_type"' in result['error']

    @pytest.mark.asyncio
    async def test_image_size_limit_exceeded(self, mock_context, mock_repositories):
        """Test error when individual image exceeds size limit."""
        # Create a large base64 string (simulating a large image)
        large_data = 'A' * (15 * 1024 * 1024)  # 15MB of 'A's
        images = [
            {
                'data': large_data,
                'mime_type': 'image/png',
            },
        ]

        with patch('app.server._ensure_repositories', return_value=mock_repositories), \
             patch('app.server.MAX_IMAGE_SIZE_MB', 10):  # 10MB limit
            result = await update_context(
                context_id=777,
                text=None,
                metadata=None,
                tags=None,
                images=images,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert 'exceeds size limit' in result['error']

    @pytest.mark.asyncio
    async def test_total_image_size_limit_exceeded(self, mock_context, mock_repositories):
        """Test error when total image size exceeds limit."""
        # Create multiple images that together exceed total limit
        # Create actual binary data and encode it to base64
        import base64
        binary_data = b'X' * (30 * 1024 * 1024)  # 30MB of binary data
        image_data = base64.b64encode(binary_data).decode('utf-8')
        images = [
            {'data': image_data, 'mime_type': 'image/png'},
            {'data': image_data, 'mime_type': 'image/png'},
            {'data': image_data, 'mime_type': 'image/png'},
            {'data': image_data, 'mime_type': 'image/png'},
        ]

        with patch('app.server._ensure_repositories', return_value=mock_repositories), \
             patch('app.server.MAX_IMAGE_SIZE_MB', 50), \
             patch('app.server.MAX_TOTAL_SIZE_MB', 100):  # Each image OK, Total exceeds
            result = await update_context(
                context_id=888,
                text=None,
                metadata=None,
                tags=None,
                images=images,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert 'total image size' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_repository_update_failure(self, mock_context, mock_repositories):
        """Test handling of repository update failure."""
        mock_repositories.context.update_context_entry.return_value = (False, [])

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=999,
                text='Some text',
                metadata=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert 'failed to update' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_auto_content_type_management_with_existing_images(self, mock_context, mock_repositories):
        """Test that content_type is properly managed when updating text with existing images."""
        # Simulate existing images in the context
        mock_repositories.images.count_images_for_context.return_value = 2
        mock_repositories.context.get_content_type.return_value = 'text'  # Wrong content type

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=1111,
                text='Updated text',
                metadata=None,
                tags=None,
                images=None,  # Not updating images
                ctx=mock_context,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'content_type' in result['updated_fields']

            # Verify content_type was corrected to multimodal
            mock_repositories.context.update_content_type.assert_called_once_with(1111, 'multimodal')

    @pytest.mark.asyncio
    async def test_exception_handling_during_update(self, mock_context, mock_repositories):
        """Test handling of unexpected exceptions during update."""
        mock_repositories.tags.replace_tags_for_context.side_effect = Exception('Database error')

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=2222,
                text=None,
                metadata=None,
                tags=['tag1'],
                images=None,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False
            assert 'update operation failed' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_context_logging(self, mock_context, mock_repositories):
        """Test that context logging is called appropriately."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            await update_context(
                context_id=3333,
                text='Test',
                metadata=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            # Verify context info was logged
            mock_context.info.assert_called_once_with('Updating context entry 3333')

    @pytest.mark.asyncio
    async def test_transaction_rollback_simulation(self, mock_context, mock_repositories):
        """Test that operations are properly sequenced for transaction safety."""
        call_order = []

        async def track_update(*_args, **_kwargs):
            call_order.append('update_context_entry')
            return True, ['text_content']

        async def track_tags(*_args, **_kwargs):
            call_order.append('replace_tags')
            raise Exception('Simulated failure')

        mock_repositories.context.update_context_entry.side_effect = track_update
        mock_repositories.tags.replace_tags_for_context.side_effect = track_tags

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=4444,
                text='Text',
                metadata=None,
                tags=['tag'],
                images=None,
                ctx=mock_context,
            )

            # Check that result is an error response
            assert result['success'] is False

            # Verify operations were attempted in order
            assert call_order == ['update_context_entry', 'replace_tags']
