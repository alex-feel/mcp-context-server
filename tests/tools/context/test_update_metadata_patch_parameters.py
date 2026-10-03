"""Tests for how metadata_patch combines with the other update_context parameters: mutual exclusivity with
metadata, combination with text and tags, repository delegation, and not-found handling."""

from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories.context_repository.records import EntryProbe
from app.types import MetadataDict
from tests.helpers import LOCAL_SCOPE

# Tools are plain async functions registered at server startup, so tests call them directly.
update_context = app.tools.update_context


class TestMetadataPatchValidation:
    """Test validation and error handling for metadata_patch."""

    @pytest.mark.asyncio
    async def test_mutual_exclusivity_error(self, mock_repositories):
        """Test error when both metadata and metadata_patch are provided.

        These parameters are mutually exclusive - use metadata for full replacement
        or metadata_patch for partial updates.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd000003e7',
                    text=None,
                    metadata={'full': 'replacement'},
                    metadata_patch={'partial': 'update'},
                    tags=None,
                    images=None,
                )

            error_message = str(exc_info.value).lower()
            assert 'metadata' in error_message
            assert 'metadata_patch' in error_message or 'both' in error_message or 'mutually exclusive' in error_message

    @pytest.mark.asyncio
    async def test_context_not_found_error(self, mock_repositories):
        """Test error when context entry doesn't exist."""
        mock_repositories.context.check_entry_exists.return_value = EntryProbe(False, None, None, None, False)

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00003039',
                    text=None,
                    metadata=None,
                    metadata_patch={'field': 'value'},
                    tags=None,
                    images=None,
                )

            assert '12345' in str(exc_info.value) or 'not found' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_patch_metadata_failure(self, mock_repositories):
        """A patch against a missing row surfaces a clean not-found error.

        patch_metadata returns success=False only when no row matches its
        WHERE id=? (the entry was deleted concurrently, or the id is stale),
        so the tool converts that outcome into a not-found ToolError naming
        the context_id rather than a generic failure message.
        """
        mock_repositories.context.patch_metadata.return_value = (False, [])

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00000457',
                    text=None,
                    metadata=None,
                    metadata_patch={'field': 'value'},
                    tags=None,
                    images=None,
                )

            error_message = str(exc_info.value)
            assert 'not found' in error_message.lower()
            assert '0190abcdef1234567890abcd00000457' in error_message


class TestMetadataPatchWithOtherFields:
    """Test metadata_patch combined with other field updates."""

    @pytest.mark.asyncio
    async def test_patch_with_text_update(self, mock_repositories):
        """Test metadata_patch combined with text content update."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000008ae',
                text='Updated text content',
                metadata=None,
                metadata_patch={'status': 'updated'},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'text_content' in result['updated_fields']
            assert 'metadata' in result['updated_fields']

            # Verify both operations were called
            mock_repositories.context.update_context_entry.assert_called_once()
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_with_tags_update(self, mock_repositories):
        """Test metadata_patch combined with tags update."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000d05',
                text=None,
                metadata=None,
                metadata_patch={'status': 'tagged'},
                tags=['new-tag'],
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']
            assert 'tags' in result['updated_fields']

    @pytest.mark.asyncio
    async def test_patch_alone_is_valid_update(self, mock_repositories):
        """Test that metadata_patch alone constitutes a valid update.

        Unlike the error when no fields provided, metadata_patch alone should work.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000115c',
                text=None,
                metadata=None,
                metadata_patch={'only_field': 'only_value'},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']


class TestMetadataPatchIntegration:
    """Integration tests for metadata_patch parameter in update_context.

    These tests verify the integration between the update_context tool
    and the underlying patch_metadata repository method.
    """

    @pytest.mark.asyncio
    async def test_metadata_patch_basic_integration(self, mock_repositories):
        """Test basic metadata_patch integration with repository."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000064',
                text=None,
                metadata=None,
                metadata_patch={'status': 'updated'},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            # Verify patch_metadata was called with correct parameters
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00000064',
                patch={'status': 'updated'},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_metadata_patch_with_text_update(self, mock_repositories):
        """Test metadata_patch combined with text content update."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000000c8',
                text='New content',
                metadata=None,
                metadata_patch={'priority': 10},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'text_content' in result['updated_fields']
            assert 'metadata' in result['updated_fields']

            # Verify both operations were called
            mock_repositories.context.update_context_entry.assert_called_once()
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_patch_mutual_exclusivity_error(self, mock_repositories):
        """Test error when both metadata and metadata_patch are provided."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd0000012c',
                    text=None,
                    metadata={'full': 'replacement'},
                    metadata_patch={'partial': 'update'},
                    tags=None,
                    images=None,
                )

            error_msg = str(exc_info.value).lower()
            assert 'metadata' in error_msg
            assert 'metadata_patch' in error_msg

    @pytest.mark.asyncio
    async def test_metadata_patch_counts_as_valid_update(self, mock_repositories):
        """Test that metadata_patch alone is a valid update (no 'no fields provided' error)."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            # Should NOT raise 'At least one field must be provided' error
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000190',
                text=None,
                metadata=None,
                metadata_patch={'only_field': 'value'},
                tags=None,
                images=None,
            )

            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_metadata_patch_failure_handling(self, mock_repositories):
        """A metadata_patch against a missing entry surfaces a clean not-found error."""
        mock_repositories.context.patch_metadata.return_value = (False, [])

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd000001f4',
                    text=None,
                    metadata=None,
                    metadata_patch={'field': 'value'},
                    tags=None,
                    images=None,
                )

            assert 'not found' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_metadata_patch_with_tags(self, mock_repositories):
        """Test metadata_patch combined with tags update."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000258',
                text=None,
                metadata=None,
                metadata_patch={'agent_name': 'test-agent'},
                tags=['new-tag'],
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']
            assert 'tags' in result['updated_fields']

    @pytest.mark.asyncio
    async def test_metadata_patch_preserves_full_metadata_behavior(self, mock_repositories):
        """Test that full metadata replacement still works when metadata_patch is not used."""
        metadata: MetadataDict = {'full': 'replacement', 'all_fields': True}
        mock_repositories.context.update_context_entry.return_value = (True, ['metadata'])

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000002bc',
                text=None,
                metadata=metadata,
                metadata_patch=None,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            # Verify update_context_entry was called (not patch_metadata)
            mock_repositories.context.update_context_entry.assert_called_once()
            mock_repositories.context.patch_metadata.assert_not_called()

    @pytest.mark.asyncio
    async def test_metadata_patch_empty_dict(self, mock_repositories):
        """Test metadata_patch with empty dict (should still be valid update)."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000320',
                text=None,
                metadata=None,
                metadata_patch={},  # Empty patch - RFC 7396: no-op but updates timestamp
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00000320',
                patch={},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )
