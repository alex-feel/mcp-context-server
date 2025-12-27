"""
Comprehensive tests for the update_context tool.

Tests cover individual field updates, combined updates, validation,
error handling, and transaction safety.
"""

import json
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

import app.server

# Get the actual async function - no longer wrapped by @mcp.tool() at import time
# Tools are registered dynamically in lifespan(), so we can access the functions directly
update_context = app.server.update_context


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
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))

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

        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            patch('app.server.MAX_IMAGE_SIZE_MB', 10),
            patch('app.server.MAX_TOTAL_SIZE_MB', 100),
        ):
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
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=444,
                    text=None,
                    metadata=None,
                    tags=None,
                    images=None,
                    ctx=mock_context,
                )
            assert 'at least one field' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_context_not_found_error(self, mock_context, mock_repositories):
        """Test error when context entry doesn't exist."""
        mock_repositories.context.check_entry_exists.return_value = False

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=999,
                    text='Some text',
                    metadata=None,
                    tags=None,
                    images=None,
                    ctx=mock_context,
                )
            assert '999' in str(exc_info.value) or 'not found' in str(exc_info.value).lower()

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
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=555,
                    text=None,
                    metadata=None,
                    tags=None,
                    images=images,
                    ctx=mock_context,
                )
            assert 'invalid base64' in str(exc_info.value).lower()

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
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=666,
                    text=None,
                    metadata=None,
                    tags=None,
                    images=images,
                    ctx=mock_context,
                )
            assert 'must have "data" and "mime_type"' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_image_size_limit_exceeded(self, mock_context, mock_repositories):
        """Test error when individual image exceeds size limit."""
        # Create actual large binary data and encode it to base64
        import base64

        large_binary = b'\x00' * (15 * 1024 * 1024)  # 15MB of binary data
        large_data = base64.b64encode(large_binary).decode('ascii')
        images = [
            {
                'data': large_data,
                'mime_type': 'image/png',
            },
        ]

        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            patch('app.server.MAX_IMAGE_SIZE_MB', 10),
        ):  # 10MB limit
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=777,
                    text=None,
                    metadata=None,
                    tags=None,
                    images=images,
                    ctx=mock_context,
                )
            assert 'exceeds size limit' in str(exc_info.value) or 'exceeds' in str(exc_info.value).lower()

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

        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            patch('app.server.MAX_IMAGE_SIZE_MB', 50),
            patch('app.server.MAX_TOTAL_SIZE_MB', 100),  # Each image OK, Total exceeds
            pytest.raises(ToolError, match='[Tt]otal.*size.*exceeds'),
        ):
            await update_context(
                context_id=888,
                text=None,
                metadata=None,
                tags=None,
                images=images,
                ctx=mock_context,
            )

    @pytest.mark.asyncio
    async def test_repository_update_failure(self, mock_context, mock_repositories):
        """Test handling of repository update failure."""
        mock_repositories.context.update_context_entry.return_value = (False, [])

        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            pytest.raises(ToolError, match='Failed to update context entry'),
        ):
            await update_context(
                context_id=999,
                text='Some text',
                metadata=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

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

        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            pytest.raises(ToolError, match='Update operation failed'),
        ):
            await update_context(
                context_id=2222,
                text=None,
                metadata=None,
                tags=['tag1'],
                images=None,
                ctx=mock_context,
            )

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

        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            pytest.raises(ToolError, match='Update operation failed'),
        ):
            await update_context(
                context_id=4444,
                text='Text',
                metadata=None,
                tags=['tag'],
                images=None,
                ctx=mock_context,
            )

        # Verify operations were attempted in order
        assert call_order == ['update_context_entry', 'replace_tags']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_context')
    async def test_empty_text_validation_error(self, mock_repositories):
        """Test that empty text is properly validated in the function body."""
        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            # Empty string is now validated in the function body, not by Pydantic
            pytest.raises(ToolError, match='text cannot be empty'),
        ):
            await update_context(
                context_id=123,
                text='',  # Empty string should fail in function validation
            )

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('mock_context')
    async def test_whitespace_only_text_validation_error(self, mock_repositories):
        """Test that whitespace-only text is rejected by business logic validation.

        Note: Since we removed Pydantic min_length constraint, validation is now done
        in the function body which properly checks for non-whitespace content.
        """
        with (
            patch('app.server._ensure_repositories', return_value=mock_repositories),
            # Whitespace-only strings are now caught in function validation
            pytest.raises(ToolError, match='text cannot be empty or contain only whitespace'),
        ):
            await update_context(
                context_id=456,
                text='   \t\n  ',  # Whitespace only should fail
            )

    @pytest.mark.asyncio
    async def test_valid_single_character_text(self, mock_context, mock_repositories):
        """Test that single character text is valid."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=789,
                text='x',  # Single character should pass
                metadata=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert result['context_id'] == 789
            assert 'text_content' in result['updated_fields']


class TestMetadataPatchIntegration:
    """Integration tests for metadata_patch parameter in update_context.

    These tests verify the integration between the update_context tool
    and the underlying patch_metadata repository method.
    """

    @pytest.mark.asyncio
    async def test_metadata_patch_basic_integration(self, mock_context, mock_repositories):
        """Test basic metadata_patch integration with repository."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=100,
                text=None,
                metadata=None,
                metadata_patch={'status': 'updated'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            # Verify patch_metadata was called with correct parameters
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=100,
                patch={'status': 'updated'},
            )

    @pytest.mark.asyncio
    async def test_metadata_patch_with_text_update(self, mock_context, mock_repositories):
        """Test metadata_patch combined with text content update."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=200,
                text='New content',
                metadata=None,
                metadata_patch={'priority': 10},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'text_content' in result['updated_fields']
            assert 'metadata' in result['updated_fields']

            # Verify both operations were called
            mock_repositories.context.update_context_entry.assert_called_once()
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_patch_mutual_exclusivity_error(self, mock_context, mock_repositories):
        """Test error when both metadata and metadata_patch are provided."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=300,
                    text=None,
                    metadata={'full': 'replacement'},
                    metadata_patch={'partial': 'update'},
                    tags=None,
                    images=None,
                    ctx=mock_context,
                )

            error_msg = str(exc_info.value).lower()
            assert 'metadata' in error_msg
            assert 'metadata_patch' in error_msg

    @pytest.mark.asyncio
    async def test_metadata_patch_counts_as_valid_update(self, mock_context, mock_repositories):
        """Test that metadata_patch alone is a valid update (no 'no fields provided' error)."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            # Should NOT raise 'At least one field must be provided' error
            result = await update_context(
                context_id=400,
                text=None,
                metadata=None,
                metadata_patch={'only_field': 'value'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_metadata_patch_failure_handling(self, mock_context, mock_repositories):
        """Test handling of patch_metadata repository failure."""
        mock_repositories.context.patch_metadata.return_value = (False, [])

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=500,
                    text=None,
                    metadata=None,
                    metadata_patch={'field': 'value'},
                    tags=None,
                    images=None,
                    ctx=mock_context,
                )

            assert 'failed' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_metadata_patch_with_tags(self, mock_context, mock_repositories):
        """Test metadata_patch combined with tags update."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=600,
                text=None,
                metadata=None,
                metadata_patch={'agent_name': 'test-agent'},
                tags=['new-tag'],
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']
            assert 'tags' in result['updated_fields']

    @pytest.mark.asyncio
    async def test_metadata_patch_preserves_full_metadata_behavior(self, mock_context, mock_repositories):
        """Test that full metadata replacement still works when metadata_patch is not used."""
        metadata = {'full': 'replacement', 'all_fields': True}
        mock_repositories.context.update_context_entry.return_value = (True, ['metadata'])

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=700,
                text=None,
                metadata=metadata,
                metadata_patch=None,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            # Verify update_context_entry was called (not patch_metadata)
            mock_repositories.context.update_context_entry.assert_called_once()
            mock_repositories.context.patch_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_patch_empty_dict(self, mock_context, mock_repositories):
        """Test metadata_patch with empty dict (should still be valid update)."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=800,
                text=None,
                metadata=None,
                metadata_patch={},  # Empty patch - RFC 7396: no-op but updates timestamp
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=800,
                patch={},
            )


class TestMetadataPatchRFC7396Semantics:
    """Test RFC 7396 deep merge semantics for metadata_patch.

    These tests verify correct behavior documentation for RFC 7396 compliant
    operations. While using mocks, they document the expected patch data
    that should be passed to achieve RFC 7396 compliance.

    IMPORTANT: Actual RFC 7396 semantics are tested in test_real_server.py
    with a real database. These tests verify the tool correctly delegates
    to the repository layer.

    RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396
    """

    @pytest.mark.asyncio
    async def test_rfc7396_case7_nested_merge_with_deletion(self, mock_context, mock_repositories):
        """RFC 7396 Test Case #7: Nested object merge with deletion.

        Verifies that nested patch with null value is correctly passed to repository.
        The actual merge semantics are handled by the database layer.
        """
        nested_patch = {'a': {'b': 'd', 'c': None}}

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7007,
                text=None,
                metadata=None,
                metadata_patch=nested_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7007,
                patch=nested_patch,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case13_null_preservation(self, mock_context, mock_repositories):
        """RFC 7396 Test Case #13: Existing null value preserved.

        Verifies that adding new keys does not affect existing null values in target.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7013,
                text=None,
                metadata=None,
                metadata_patch={'a': 1},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7013,
                patch={'a': 1},
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case15_deeply_nested_null(self, mock_context, mock_repositories):
        """RFC 7396 Test Case #15: Deeply nested null deletion.

        Verifies that deeply nested null patch is correctly passed to repository.
        """
        deep_patch = {'a': {'bb': {'ccc': None}}}

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7015,
                text=None,
                metadata=None,
                metadata_patch=deep_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7015,
                patch=deep_patch,
            )

    @pytest.mark.asyncio
    async def test_deep_merge_preserves_sibling_nested_keys(self, mock_context, mock_repositories):
        """Verify patch for deep merge with sibling key preservation.

        When patching {"a": {"b": "updated"}}, sibling keys in the nested object
        should be preserved. This test verifies the correct patch is passed.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7100,
                text=None,
                metadata=None,
                metadata_patch={'a': {'b': 'updated'}},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7100,
                patch={'a': {'b': 'updated'}},
            )
