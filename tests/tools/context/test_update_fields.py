"""Tests for update_context field updates: text, metadata, tags and images, content_type management, and
updated_at stamping."""

import json
from unittest.mock import patch

import pytest

import app.tools
from app.types import MetadataDict
from tests.helpers import LOCAL_SCOPE

# Tools are plain async functions registered at server startup, so tests call them directly.
update_context = app.tools.update_context
get_context_by_ids = app.tools.get_context_by_ids


class TestUpdateContext:
    """Test suite for update_context tool."""

    @pytest.mark.asyncio
    async def test_update_text_content_only(self, mock_repositories):
        """Test updating only text content.

        With no summary provider configured at update time, a text change clears the
        (now-stale) summary instead of leaving one that describes the replaced text.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='Updated text content',
                metadata=None,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert result['context_id'] == '0190abcdef1234567890abcd0000007b'
            assert 'text_content' in result['updated_fields']
            assert result['message'] == 'Successfully updated 1 field(s) (summary cleared)'

            # Verify repository calls
            mock_repositories.context.check_entry_exists.assert_called_once_with(
                '0190abcdef1234567890abcd0000007b', scope=LOCAL_SCOPE,
            )
            from unittest.mock import ANY
            mock_repositories.context.update_context_entry.assert_called_once_with(
                context_id='0190abcdef1234567890abcd0000007b',
                text_content='Updated text content',
                metadata=None,
                summary=None,
                clear_summary=True,
                visibility=None,
                expected_version=0,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_update_metadata_only(self, mock_repositories):
        """Test updating only metadata."""
        metadata: MetadataDict = {'status': 'completed', 'priority': 5}
        mock_repositories.context.update_context_entry.return_value = (True, ['metadata'])

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000001c8',
                text=None,
                metadata=metadata,
                tags=None,
                images=None,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert result['context_id'] == '0190abcdef1234567890abcd000001c8'
            assert 'metadata' in result['updated_fields']

            # Verify metadata was JSON-encoded
            from unittest.mock import ANY
            expected_metadata_str = json.dumps(metadata)
            mock_repositories.context.update_context_entry.assert_called_once_with(
                context_id='0190abcdef1234567890abcd000001c8',
                text_content=None,
                metadata=expected_metadata_str,
                summary=None,
                clear_summary=False,
                visibility=None,
                expected_version=0,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_update_tags_only(self, mock_repositories):
        """Test replacing tags."""
        tags = ['python', 'testing', 'async']

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000315',
                text=None,
                metadata=None,
                tags=tags,
                images=None,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'tags' in result['updated_fields']

            # Verify tags were replaced
            from unittest.mock import ANY
            mock_repositories.tags.replace_tags_for_context.assert_called_once_with(
                '0190abcdef1234567890abcd00000315', tags, txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_tags_only_update_advances_updated_at(self, mock_repositories):
        """A tags-only update must still advance the entry's public mutation timestamp.

        ``updated_at`` is documented as auto-managed and is the only mutation
        timestamp the API exposes, so clients key incremental sync and cache
        invalidation on it. It rides on writes to context_entries itself; a
        tags-only update writes only the child `tags` table, so the auto-managed
        block issues the context_entries write that carries the stamp; without it
        the update would report success while the timestamp stayed unchanged and
        such a client would never observe the change.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000316',
                tags=['alpha'],
            )

        assert result['success'] is True
        assert 'tags' in result['updated_fields']
        # No other branch wrote context_entries, so the auto-managed block must.
        mock_repositories.context.update_context_entry.assert_not_called()
        mock_repositories.context.patch_metadata.assert_not_called()
        # content_type is already correct, so the timestamp is stamped explicitly
        # rather than by rewriting content_type back to its own value.
        mock_repositories.context.update_content_type.assert_not_called()
        mock_repositories.context.touch_updated_at.assert_called_once()
        assert 'content_type' not in result['updated_fields']

    @pytest.mark.asyncio
    async def test_metadata_patch_only_update_does_not_double_stamp(self, mock_repositories):
        """patch_metadata already writes context_entries, so no extra stamping write."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000317',
                metadata_patch={'k': 'v'},
            )

        assert result['success'] is True
        mock_repositories.context.patch_metadata.assert_called_once()
        mock_repositories.context.update_content_type.assert_not_called()
        mock_repositories.context.touch_updated_at.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_images_with_content_type_change(self, mock_repositories):
        """Test replacing images and updating content_type to multimodal."""
        images = [
            {
                'data': 'aGVsbG8gd29ybGQ=',  # base64 encoded "hello world"
                'mime_type': 'image/png',
            },
        ]

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools._validation.MAX_IMAGE_SIZE_MB', 10),
            patch('app.tools._validation.MAX_TOTAL_SIZE_MB', 100),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000006f',
                text=None,
                metadata=None,
                tags=None,
                images=images,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'images' in result['updated_fields']
            assert 'content_type' in result['updated_fields']

            # Verify images were replaced and content_type updated
            from unittest.mock import ANY
            mock_repositories.images.replace_images_for_context.assert_called_once_with(
                '0190abcdef1234567890abcd0000006f', images, txn=ANY,
            )
            mock_repositories.context.update_content_type.assert_called_once_with(
                '0190abcdef1234567890abcd0000006f', 'multimodal', scope=LOCAL_SCOPE, txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_remove_all_images_updates_content_type(self, mock_repositories):
        """Test that providing empty images list removes images and sets content_type to text."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000000de',
                text=None,
                metadata=None,
                tags=None,
                images=[],  # Empty list removes all images
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'images' in result['updated_fields']
            assert 'content_type' in result['updated_fields']

            # Verify images were cleared and content_type set to text
            from unittest.mock import ANY
            mock_repositories.images.replace_images_for_context.assert_called_once_with(
                '0190abcdef1234567890abcd000000de', [], txn=ANY,
            )
            mock_repositories.context.update_content_type.assert_called_once_with(
                '0190abcdef1234567890abcd000000de', 'text', scope=LOCAL_SCOPE, txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_update_multiple_fields(self, mock_repositories):
        """Test updating multiple fields in one call."""
        mock_repositories.context.update_context_entry.return_value = (True, ['text_content', 'metadata'])

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000014d',
                text='New text',
                metadata={'key': 'value'},
                tags=['tag1', 'tag2'],
                images=None,
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'text_content' in result['updated_fields']
            assert 'metadata' in result['updated_fields']
            assert 'tags' in result['updated_fields']
            assert len(result['updated_fields']) == 3

    @pytest.mark.asyncio
    async def test_auto_content_type_management_with_existing_images(self, mock_repositories):
        """Test that content_type is properly managed when updating text with existing images."""
        # Simulate existing images in the context
        mock_repositories.images.count_images_for_context.return_value = 2
        mock_repositories.context.get_content_type.return_value = 'text'  # Wrong content type

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000457',
                text='Updated text',
                metadata=None,
                tags=None,
                images=None,  # Not updating images
            )

            # Check that result is a successful response
            assert result['success'] is True
            assert 'content_type' in result['updated_fields']

            # Verify content_type was corrected to multimodal
            from unittest.mock import ANY
            mock_repositories.context.update_content_type.assert_called_once_with(
                '0190abcdef1234567890abcd00000457', 'multimodal', scope=LOCAL_SCOPE, txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_update_context_image_without_mime_type_defaults_png(self, mock_repositories):
        """Image without mime_type defaults to 'image/png' in update_context."""
        import base64

        img_data = base64.b64encode(b'test image data').decode('utf-8')
        images = [{'data': img_data}]  # No mime_type key

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text=None,
                metadata=None,
                tags=None,
                images=images,
            )

        assert result['success'] is True
        # Verify images were stored (replace_images_for_context was called)
        mock_repositories.images.replace_images_for_context.assert_called_once()
        stored_images = mock_repositories.images.replace_images_for_context.call_args[0][1]
        assert stored_images[0]['mime_type'] == 'image/png'

    @pytest.mark.asyncio
    async def test_update_context_image_with_explicit_mime_type(self, mock_repositories):
        """Image with explicit mime_type preserves the provided value."""
        import base64

        img_data = base64.b64encode(b'test jpeg data').decode('utf-8')
        images = [{'data': img_data, 'mime_type': 'image/jpeg'}]

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools.context.update.get_embedding_provider', return_value=None),
            patch('app.tools._generation.get_embedding_provider', return_value=None),
            patch('app.tools.context.update.get_summary_provider', return_value=None),
            patch('app.tools._generation.get_summary_provider', return_value=None),
        ):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text=None,
                metadata=None,
                tags=None,
                images=images,
            )

        assert result['success'] is True
        stored_images = mock_repositories.images.replace_images_for_context.call_args[0][1]
        assert stored_images[0]['mime_type'] == 'image/jpeg'


@pytest.mark.usefixtures('initialized_server')
class TestUpdatedAtAutoManagement:
    """updated_at advances for every update variant, against a real database."""

    @pytest.mark.asyncio
    async def test_tags_only_update_advances_stored_updated_at(self) -> None:
        """End-to-end: a tags-only update moves the stored updated_at forward.

        SQLite CURRENT_TIMESTAMP has second granularity, so the test waits past a
        second boundary before the update; without the auto-managed stamping write
        the value would stay at its original second no matter how long it waited.
        """
        import asyncio

        from app.tools.context.store import store_context

        stored = await store_context(
            thread_id='updated-at-tags-only',
            source='user',
            text='Original body',
        )
        context_id = stored['context_id']

        before = (await get_context_by_ids(context_ids=[context_id]))[0].get('updated_at')
        assert before is not None

        # Cross a whole-second boundary so a genuine re-stamp is observable.
        await asyncio.sleep(1.1)

        result = await update_context(context_id=context_id, tags=['fresh-tag'])
        assert result['success'] is True

        after_entry = (await get_context_by_ids(context_ids=[context_id]))[0]
        assert after_entry.get('tags') == ['fresh-tag']
        after_updated = after_entry.get('updated_at')
        assert after_updated is not None
        assert after_updated > before
