"""Tests for update_context input validation and error handling: missing fields, unknown entries, image
validation, repository failures, rollback, and context_id normalization."""

import base64
from typing import Any
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError

import app.tools
from app.repositories.context_repository.records import EntryProbe
from tests.helpers import LOCAL_WRITE_SCOPE

# Tools are plain async functions registered at server startup, so tests call them directly.
update_context = app.tools.update_context


class TestUpdateContext:
    """Test suite for update_context tool."""

    @pytest.mark.asyncio
    async def test_no_fields_provided_error(self, mock_repositories):
        """Test error when no fields are provided for update."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd000001bc',
                    text=None,
                    metadata=None,
                    tags=None,
                    images=None,
                )
            assert 'at least one field' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_context_not_found_error(self, mock_repositories):
        """Test error when context entry doesn't exist."""
        mock_repositories.context.check_entry_exists.return_value = EntryProbe(False, None, None, None, False)

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd000003e7',
                    text='Some text',
                    metadata=None,
                    tags=None,
                    images=None,
                )
            assert '999' in str(exc_info.value) or 'not found' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_invalid_image_data(self, mock_repositories):
        """Test error with invalid image data."""
        images = [
            {
                'data': 'not-valid-base64!!!',
                'mime_type': 'image/png',
            },
        ]

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd0000022b',
                    text=None,
                    metadata=None,
                    tags=None,
                    images=images,
                )
            assert 'invalid base64' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_missing_image_data_field(self, mock_repositories):
        """Test error when image is missing required data field."""
        images = [
            {
                'mime_type': 'image/png',
                # Missing data field
            },
        ]

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd0000029a',
                    text=None,
                    metadata=None,
                    tags=None,
                    images=images,
                )
            assert 'missing required "data" field' in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_image_size_limit_exceeded(self, mock_repositories):
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
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools._validation.MAX_IMAGE_SIZE_MB', 10),
        ):  # 10MB limit
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id='0190abcdef1234567890abcd00000309',
                    text=None,
                    metadata=None,
                    tags=None,
                    images=images,
                )
            assert 'exceeds size limit' in str(exc_info.value) or 'exceeds' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_total_image_size_limit_exceeded(self, mock_repositories):
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
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            patch('app.tools._validation.MAX_IMAGE_SIZE_MB', 50),
            patch('app.tools._validation.MAX_TOTAL_SIZE_MB', 100),  # Each image OK, Total exceeds
            pytest.raises(ToolError, match='[Tt]otal.*size.*exceeds'),
        ):
            await update_context(
                context_id='0190abcdef1234567890abcd00000378',
                text=None,
                metadata=None,
                tags=None,
                images=images,
            )

    @pytest.mark.asyncio
    async def test_repository_update_failure(self, mock_repositories):
        """A no-such-row update (repository reports no matching row) surfaces a clean not-found error."""
        mock_repositories.context.update_context_entry.return_value = (False, [])

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            pytest.raises(ToolError, match='Context entry with ID 0190abcdef1234567890abcd000003e7 not found'),
        ):
            await update_context(
                context_id='0190abcdef1234567890abcd000003e7',
                text='Some text',
                metadata=None,
                tags=None,
                images=None,
            )

    @pytest.mark.asyncio
    async def test_exception_handling_during_update(self, mock_repositories):
        """Test handling of unexpected exceptions during update."""
        mock_repositories.tags.replace_tags_for_context.side_effect = Exception('Database error')

        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            pytest.raises(ToolError, match='Failed to update context'),
        ):
            await update_context(
                context_id='0190abcdef1234567890abcd000008ae',
                text=None,
                metadata=None,
                tags=['tag1'],
                images=None,
            )

    @pytest.mark.asyncio
    async def test_context_id_is_normalized_before_lookup(self, mock_repositories):
        """Whitespace and uppercase in context_id are folded to canonical form before any repository call."""
        canonical = '0190abcdef1234567890abcd00000d05'

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='  0190ABCDEF1234567890ABCD00000D05  ',
                text='Test',
                metadata=None,
                tags=None,
                images=None,
            )

        assert result['context_id'] == canonical
        mock_repositories.context.check_entry_exists.assert_awaited_once_with(canonical, scope=LOCAL_WRITE_SCOPE)

    @pytest.mark.asyncio
    async def test_transaction_rollback_simulation(self, mock_repositories):
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
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            pytest.raises(ToolError, match='Failed to update context'),
        ):
            await update_context(
                context_id='0190abcdef1234567890abcd0000115c',
                text='Text',
                metadata=None,
                tags=['tag'],
                images=None,
            )

        # Verify operations were attempted in order
        assert call_order == ['update_context_entry', 'replace_tags']

    @pytest.mark.asyncio
    async def test_empty_text_validation_error(self, mock_repositories):
        """Test that empty text is properly validated in the function body."""
        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            # Empty string is validated in the function body, not by Pydantic
            pytest.raises(ToolError, match='text cannot be empty'),
        ):
            await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text='',  # Empty string should fail in function validation
            )

    @pytest.mark.asyncio
    async def test_whitespace_only_text_validation_error(self, mock_repositories):
        """Test that whitespace-only text is rejected by business logic validation.

        Whitespace-only text satisfies the parameter's Pydantic ``min_length=1``
        constraint, so the function body strips it and rejects the empty result.
        """
        with (
            patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories),
            # Whitespace-only strings are caught in function validation
            pytest.raises(ToolError, match='text cannot be empty or contain only whitespace'),
        ):
            await update_context(
                context_id='0190abcdef1234567890abcd000001c8',
                text='   \t\n  ',  # Whitespace only should fail
            )

    @pytest.mark.asyncio
    async def test_valid_single_character_text(self, mock_repositories):
        """Test that single character text is valid."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000315',
                text='x',  # Single character should pass
                metadata=None,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert result['context_id'] == '0190abcdef1234567890abcd00000315'
            assert 'text_content' in result['updated_fields']


_INGEST_BLOCK: dict[str, Any] = {
    'scope': 'kb:handbook',
    'source_id': 'guides/setup.md',
    'meta_digest': 'ab' * 32,
    'synced_at': '2026-10-18T12:00:00Z',
}
_PATCH_SHAPE_ERROR = (
    'metadata_patch.ingest must be null or an object with exactly the keys meta_digest, scope, source_id, synced_at'
)


class TestUpdateIngestBlock:
    """update_context validates the reserved ingest block before any generation or database work."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('arguments', 'message'),
        [
            (
                {'metadata': {'ingest': {**_INGEST_BLOCK, 'scope': ''}}},
                'metadata.ingest.scope cannot be empty',
            ),
            (
                {'metadata': {'ingest': None}},
                'metadata.ingest must be an object with exactly the keys meta_digest, scope, source_id, synced_at',
            ),
            (
                {'metadata_patch': {'ingest': {**_INGEST_BLOCK, 'meta_digest': 'abc'}}},
                'metadata_patch.ingest.meta_digest must be 64 lowercase hexadecimal characters',
            ),
            ({'metadata_patch': {'ingest': {'synced_at': '2026-10-19T08:00:00Z'}}}, _PATCH_SHAPE_ERROR),
            ({'metadata_patch': {'ingest': {**_INGEST_BLOCK, 'extra': 1}}}, _PATCH_SHAPE_ERROR),
            ({'text': 'New body', 'metadata_patch': {'ingest': 'kb:handbook'}}, _PATCH_SHAPE_ERROR),
        ],
        ids=['metadata-empty-scope', 'metadata-null', 'patch-bad-digest', 'patch-partial', 'patch-extra-key', 'patch-string'],
    )
    async def test_malformed_block_refused_before_generation(self, arguments: dict[str, Any], message: str) -> None:
        """A malformed block, or a partial block in a patch, raises its fixed message before any work."""
        embedding_provider = AsyncMock()
        summary_provider = AsyncMock()
        with (
            patch('app.tools.context.update.ensure_repositories', new_callable=AsyncMock) as ensure,
            patch('app.tools.context.update.get_embedding_provider', return_value=embedding_provider),
            patch('app.tools.context.update.get_summary_provider', return_value=summary_provider),
            pytest.raises(ToolError) as exc_info,
        ):
            await update_context(context_id='0190abcdef1234567890abcd0000007b', **arguments)

        assert str(exc_info.value) == message
        ensure.assert_not_awaited()
        assert embedding_provider.mock_calls == []
        assert summary_provider.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'arguments',
        [{'metadata': {'title': 'Setup', 'ingest': _INGEST_BLOCK}}, {'metadata_patch': {'ingest': None}}],
        ids=['complete-block', 'patch-null'],
    )
    async def test_valid_forms_pass_the_boundary(self, mock_repositories: Mock, arguments: dict[str, Any]) -> None:
        """A complete block in full metadata and a null block in a patch both reach the update."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(context_id='0190abcdef1234567890abcd0000007b', **arguments)

        assert result['success'] is True


@pytest.mark.usefixtures('initialized_server')
class TestUpdateContextImageValidation:
    """Empty data check and per-image index in update_context."""

    @pytest.mark.asyncio
    async def test_update_context_rejects_empty_image_data(self):
        """update_context rejects images with empty data field."""
        from app.tools.batch.store import store_context_batch
        from app.tools.context.update import update_context

        store_result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hello'}],
        )
        cid = store_result['results'][0]['context_id']
        assert cid is not None

        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await update_context(context_id=cid, images=[{'data': ''}])

    @pytest.mark.asyncio
    async def test_update_context_rejects_whitespace_image_data(self):
        """update_context rejects images with whitespace-only data."""
        from app.tools.batch.store import store_context_batch
        from app.tools.context.update import update_context

        store_result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hello'}],
        )
        cid = store_result['results'][0]['context_id']
        assert cid is not None

        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            await update_context(context_id=cid, images=[{'data': '   '}])

    @pytest.mark.asyncio
    async def test_update_context_image_errors_include_index(self):
        """Error messages include per-image index."""
        from app.tools.batch.store import store_context_batch
        from app.tools.context.update import update_context

        store_result = await store_context_batch(
            entries=[{'thread_id': 't', 'source': 'user', 'text': 'hello'}],
        )
        cid = store_result['results'][0]['context_id']
        assert cid is not None

        valid_image = base64.b64encode(b'\x89PNG\r\n').decode()
        with pytest.raises(ToolError, match='Image 1') as exc_info:
            await update_context(
                context_id=cid,
                images=[
                    {'data': valid_image},
                    {'data': 'not-valid-base64!!!'},
                ],
            )
        assert '1' in str(exc_info.value)
