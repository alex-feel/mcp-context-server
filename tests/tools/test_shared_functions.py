"""Tests for app.tools._shared module.

Contains:
- Relocated tests from tests/backends/test_postgresql_backend.py
  (TestTransactionHeartbeat, TestConnectionErrorClassification)
- New tests for validate_and_normalize_images
- New tests for build_store_response_message, build_update_response_message
- New tests for build_batch_store_response_message, build_batch_update_response_message
"""

import base64
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock

import asyncpg
import pytest
from fastmcp.exceptions import ToolError

from app.repositories.embedding_repository import ChunkEmbedding
from app.tools._shared import EmbeddingsReconcileRequiredError
from app.tools._shared import build_batch_store_response_message
from app.tools._shared import build_batch_update_response_message
from app.tools._shared import build_store_response_message
from app.tools._shared import build_update_response_message
from app.tools._shared import execute_store_in_transaction
from app.tools._shared import execute_update_in_transaction
from app.tools._shared import is_connection_error
from app.tools._shared import transaction_heartbeat
from app.tools._shared import validate_and_normalize_images

_ChunkEmbeddingList = list[ChunkEmbedding]

# ---------------------------------------------------------------------------
# Relocated: TestTransactionHeartbeat (from tests/backends/test_postgresql_backend.py)
# ---------------------------------------------------------------------------


class TestTransactionHeartbeat:
    """Test in-transaction heartbeat helper."""

    @pytest.mark.asyncio
    async def test_heartbeat_executes_select_1(self) -> None:
        """Verify transaction_heartbeat sends SELECT 1 for PostgreSQL transactions."""
        mock_conn = AsyncMock()
        mock_conn.execute = AsyncMock()

        mock_txn = AsyncMock()
        type(mock_txn).backend_type = PropertyMock(return_value='postgresql')
        type(mock_txn).connection = PropertyMock(return_value=mock_conn)

        await transaction_heartbeat(mock_txn)

        mock_conn.execute.assert_called_once_with('SELECT 1')

    @pytest.mark.asyncio
    async def test_heartbeat_noop_for_sqlite(self) -> None:
        """Verify transaction_heartbeat is a no-op for SQLite transactions."""
        mock_conn = MagicMock()
        mock_txn = MagicMock()
        type(mock_txn).backend_type = PropertyMock(return_value='sqlite')
        type(mock_txn).connection = PropertyMock(return_value=mock_conn)

        await transaction_heartbeat(mock_txn)

        mock_conn.execute.assert_not_called()


# ---------------------------------------------------------------------------
# Relocated: TestConnectionErrorClassification (from tests/backends/test_postgresql_backend.py)
# ---------------------------------------------------------------------------


class TestConnectionErrorClassification:
    """Test connection error classification for retry logic."""

    def test_connection_errors_classified_correctly(self) -> None:
        """Verify is_connection_error identifies retryable connection errors."""
        assert is_connection_error(asyncpg.InterfaceError('connection closed'))
        assert is_connection_error(ConnectionResetError('reset'))
        assert is_connection_error(OSError('network unreachable'))

    def test_non_connection_errors_not_retried(self) -> None:
        """Verify non-connection errors are not classified as retryable."""
        assert not is_connection_error(ValueError('bad value'))
        assert not is_connection_error(TypeError('wrong type'))
        assert not is_connection_error(RuntimeError('logic error'))

    def test_query_canceled_error_is_retryable(self) -> None:
        """statement_timeout cancel (SQLSTATE 57014) is classified retryable.

        QueryCanceledError is raised when PostgreSQL cancels a statement that
        exceeded statement_timeout. It is a transient lock-wait/timeout error,
        safe to retry because the DB write is idempotent and generation already
        completed outside the transaction.
        """
        assert is_connection_error(asyncpg.exceptions.QueryCanceledError('canceling statement due to statement timeout'))

    def test_query_canceled_error_sqlstate_is_57014(self) -> None:
        """Document the SQLSTATE this classifier now treats as retryable."""
        assert asyncpg.exceptions.QueryCanceledError.sqlstate == '57014'


# ---------------------------------------------------------------------------
# New: TestValidateAndNormalizeImages
# ---------------------------------------------------------------------------

# Valid base64 PNG (1x1 transparent pixel)
VALID_BASE64_PNG = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 50).decode()


class TestValidateAndNormalizeImages:
    """Test validate_and_normalize_images shared function."""

    def test_none_images_returns_text(self) -> None:
        """None images returns empty list, 'text' content type, no errors."""
        images, content_type, errors = validate_and_normalize_images(None)
        assert images == []
        assert content_type == 'text'
        assert errors == []

    def test_empty_list_returns_text(self) -> None:
        """Empty list returns empty list, 'text' content type, no errors."""
        images, content_type, errors = validate_and_normalize_images([])
        assert images == []
        assert content_type == 'text'
        assert errors == []

    def test_valid_image_returns_multimodal(self) -> None:
        """Valid base64 image returns multimodal content type."""
        img = {'data': VALID_BASE64_PNG, 'mime_type': 'image/png'}
        images, content_type, errors = validate_and_normalize_images([img])
        assert content_type == 'multimodal'
        assert errors == []
        assert len(images) == 1

    def test_defaults_mime_type(self) -> None:
        """Image without mime_type gets 'image/png' default."""
        img = {'data': VALID_BASE64_PNG}
        images, content_type, errors = validate_and_normalize_images([img])
        assert images[0]['mime_type'] == 'image/png'
        assert content_type == 'multimodal'
        assert errors == []

    def test_json_string_image_metadata_passes(self) -> None:
        """A JSON-encoded-string metadata value (the string-valued tool contract) passes.

        Per-image dicts are dict[str, str]: callers pass structured image metadata as a
        JSON-encoded string, which carries no bare float and is not rejected.
        """
        img = {'data': VALID_BASE64_PNG, 'mime_type': 'image/png', 'metadata': '{"position": 1}'}
        _, content_type, errors = validate_and_normalize_images([img])
        assert content_type == 'multimodal'
        assert errors == []

    def test_raise_mode_non_finite_image_metadata(self) -> None:
        """A dict-valued image metadata (untyped batch path) with NaN/Infinity is rejected.

        json.dumps emits the invalid-JSON tokens NaN/Infinity, which PostgreSQL's jsonb
        image_metadata column rejects while SQLite stores them -- a cross-backend parity
        divergence caught in Phase 1, before the generation pass and the transaction.
        """
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': {'score': float('nan')}}])
        with pytest.raises(ToolError, match='Image 0 metadata:'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_collect_mode_non_finite_image_metadata_nested(self) -> None:
        """collect mode reports a non-finite image-metadata error nested at any depth."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': {'deep': {'x': float('inf')}}}])
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'Image 0 metadata:' in errors[0]

    def test_raise_mode_missing_data(self) -> None:
        """error_mode='raise' raises ToolError for missing data field."""
        with pytest.raises(ToolError, match='Image 0 is missing required "data" field'):
            validate_and_normalize_images([{'mime_type': 'image/png'}], error_mode='raise')

    def test_raise_mode_empty_data(self) -> None:
        """error_mode='raise' raises ToolError for empty data field."""
        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            validate_and_normalize_images([{'data': '   '}], error_mode='raise')

    def test_raise_mode_invalid_base64(self) -> None:
        """error_mode='raise' raises ToolError for invalid base64 encoding."""
        with pytest.raises(ToolError, match='invalid base64 encoding'):
            validate_and_normalize_images([{'data': '!!!not-base64!!!'}], error_mode='raise')

    def test_raise_mode_oversized(self) -> None:
        """error_mode='raise' raises ToolError for oversized image."""
        # Create data that exceeds MAX_IMAGE_SIZE_MB (10MB default)
        large_data = base64.b64encode(b'\x00' * (11 * 1024 * 1024)).decode()
        with pytest.raises(ToolError, match='exceeds.*MB limit'):
            validate_and_normalize_images([{'data': large_data}], error_mode='raise')

    def test_collect_mode_missing_data(self) -> None:
        """error_mode='collect' returns errors list for missing data."""
        images, content_type, errors = validate_and_normalize_images(
            [{'mime_type': 'image/png'}], error_mode='collect',
        )
        assert len(errors) == 1
        assert 'Image 0 is missing required "data" field' in errors[0]

    def test_collect_mode_empty_data(self) -> None:
        """error_mode='collect' returns errors list, does not raise."""
        images, content_type, errors = validate_and_normalize_images(
            [{'data': ''}], error_mode='collect',
        )
        assert len(errors) == 1
        assert 'Image 0 has empty "data" field' in errors[0]

    def test_collect_mode_multiple_errors(self) -> None:
        """First error returned in collect mode (early return)."""
        imgs = [
            {'mime_type': 'image/png'},  # missing data
            {'data': ''},  # empty data
        ]
        _, _, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert len(errors) == 1
        assert 'Image 0' in errors[0]

    def test_enumerate_index_in_errors(self) -> None:
        """Error messages include correct image index."""
        imgs = [
            {'data': VALID_BASE64_PNG, 'mime_type': 'image/png'},
            {'mime_type': 'image/png'},  # missing data at index 1
        ]
        _, _, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert len(errors) == 1
        assert 'Image 1' in errors[0]

    def test_raise_mode_non_string_mime_type(self) -> None:
        """A present non-string mime_type (untyped batch input) is rejected in raise mode."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'mime_type': 123}])
        with pytest.raises(ToolError, match='Image 0 has a non-string "mime_type" field'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_raise_mode_null_mime_type(self) -> None:
        """A present null mime_type is rejected, not bound into the NOT NULL column."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'mime_type': None}])
        with pytest.raises(ToolError, match='Image 0 has a non-string "mime_type" field'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_collect_mode_non_string_mime_type(self) -> None:
        """collect mode records a non-string mime_type error instead of raising."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'mime_type': None}])
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'Image 0 has a non-string "mime_type" field' in errors[0]

    def test_raise_mode_non_string_data(self) -> None:
        """A present non-string data value is rejected before .strip()/base64 decode."""
        imgs = cast('list[dict[str, str]]', [{'data': 123, 'mime_type': 'image/png'}])
        with pytest.raises(ToolError, match='Image 0 has a non-string "data" field'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_collect_mode_non_string_data(self) -> None:
        """collect mode records a non-string data error instead of crashing with AttributeError."""
        imgs = cast('list[dict[str, str]]', [{'data': 123, 'mime_type': 'image/png'}])
        _, _, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert len(errors) == 1
        assert 'Image 0 has a non-string "data" field' in errors[0]

    def test_raise_mode_garbage_decodes_to_zero_bytes(self) -> None:
        """Garbage base64 (all non-alphabet, length a multiple of 4) decodes to 0 bytes and is rejected.

        base64.b64decode is lenient (validate omitted), so a value like '!!!!' silently
        discards every non-alphabet character and decodes to b''. Without an explicit
        zero-length guard it would be stored as a 0-byte image, defeating the validator's
        documented protection against silent 0-byte storage.
        """
        with pytest.raises(ToolError, match='Image 0 "data" decodes to zero bytes'):
            validate_and_normalize_images([{'data': '!!!!'}], error_mode='raise')

    def test_collect_mode_garbage_decodes_to_zero_bytes(self) -> None:
        """collect mode records the zero-byte error instead of raising or storing a 0-byte image."""
        _, content_type, errors = validate_and_normalize_images(
            [{'data': '@#$%'}], error_mode='collect',
        )
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'decodes to zero bytes' in errors[0]

    def test_whitespace_wrapped_base64_still_accepted(self) -> None:
        """Newline-wrapped base64 still decodes (the decode stays lenient; only a 0-byte result is rejected)."""
        wrapped = VALID_BASE64_PNG[:4] + '\n' + VALID_BASE64_PNG[4:]
        _, content_type, errors = validate_and_normalize_images(
            [{'data': wrapped, 'mime_type': 'image/png'}], error_mode='collect',
        )
        assert content_type == 'multimodal'
        assert errors == []


# ---------------------------------------------------------------------------
# New: TestBuildStoreResponseMessage
# ---------------------------------------------------------------------------


class TestBuildStoreResponseMessage:
    """Test build_store_response_message shared function."""

    def test_basic_stored(self) -> None:
        """Basic store with no extras produces simple message."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert msg == 'Context stored'

    def test_stored_with_images(self) -> None:
        """Store with images includes image count."""
        msg = build_store_response_message(
            action='stored', image_count=3,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert msg == 'Context stored with 3 images'

    def test_embedding_generated_and_stored(self) -> None:
        """Embedding generated and stored shows 'embedding generated'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=True, embedding_stored=True,
            summary_generated=False, summary_preserved=False,
        )
        assert 'embedding generated' in msg
        assert 'not stored' not in msg

    def test_embedding_generated_not_stored(self) -> None:
        """Embedding generated but not stored shows duplicate message."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=True, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert 'embedding generated but not stored - duplicate' in msg

    def test_summary_generated(self) -> None:
        """Summary generated shows 'summary generated'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=True, summary_preserved=False,
        )
        assert 'summary generated' in msg

    def test_summary_preserved(self) -> None:
        """Summary preserved shows 'summary preserved'."""
        msg = build_store_response_message(
            action='stored', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=True,
        )
        assert 'summary preserved' in msg

    def test_all_parts(self) -> None:
        """All flags set produces message with all parts."""
        msg = build_store_response_message(
            action='stored', image_count=2,
            embedding_generated=True, embedding_stored=True,
            summary_generated=True, summary_preserved=False,
        )
        assert 'Context stored with 2 images' in msg
        assert 'embedding generated' in msg
        assert 'summary generated' in msg

    def test_no_parts(self) -> None:
        """No flags set produces no parenthetical."""
        msg = build_store_response_message(
            action='updated', image_count=0,
            embedding_generated=False, embedding_stored=False,
            summary_generated=False, summary_preserved=False,
        )
        assert msg == 'Context updated'
        assert '(' not in msg


# ---------------------------------------------------------------------------
# New: TestBuildUpdateResponseMessage
# ---------------------------------------------------------------------------


class TestBuildUpdateResponseMessage:
    """Test build_update_response_message shared function."""

    def test_basic_update(self) -> None:
        """Basic update with no extras."""
        msg = build_update_response_message(
            updated_fields_count=3,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=False,
        )
        assert msg == 'Successfully updated 3 field(s)'

    def test_embedding_regenerated(self) -> None:
        """Embedding regenerated shows in message."""
        msg = build_update_response_message(
            updated_fields_count=2,
            embedding_generated=True,
            summary_generated=False,
            summary_cleared=False,
        )
        assert 'embedding regenerated' in msg

    def test_summary_regenerated(self) -> None:
        """Summary regenerated shows in message."""
        msg = build_update_response_message(
            updated_fields_count=2,
            embedding_generated=False,
            summary_generated=True,
            summary_cleared=False,
        )
        assert 'summary regenerated' in msg

    def test_summary_cleared(self) -> None:
        """Summary cleared shows in message."""
        msg = build_update_response_message(
            updated_fields_count=1,
            embedding_generated=False,
            summary_generated=False,
            summary_cleared=True,
        )
        assert 'summary cleared' in msg

    def test_all_parts(self) -> None:
        """All flags set produces all parts."""
        msg = build_update_response_message(
            updated_fields_count=5,
            embedding_generated=True,
            summary_generated=True,
            summary_cleared=False,
        )
        assert 'embedding regenerated' in msg
        assert 'summary regenerated' in msg
        assert '5 field(s)' in msg


# ---------------------------------------------------------------------------
# New: TestBuildBatchStoreResponseMessage
# ---------------------------------------------------------------------------


class TestBuildBatchStoreResponseMessage:
    """Test build_batch_store_response_message shared function."""

    def test_basic_batch_message(self) -> None:
        """Basic batch store message."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0, embeddings_stored_count=0,
            summaries_generated_count=0, summaries_preserved_count=0,
        )
        assert msg == 'Stored 3/3 entries successfully'

    def test_with_embeddings_not_stored(self) -> None:
        """Batch with some embeddings not stored shows duplicate count."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=3, embeddings_stored_count=1,
            summaries_generated_count=0, summaries_preserved_count=0,
        )
        assert 'embeddings generated (2 not stored - duplicates)' in msg

    def test_with_summaries_preserved(self) -> None:
        """Batch with preserved summaries shows count."""
        msg = build_batch_store_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0, embeddings_stored_count=0,
            summaries_generated_count=2, summaries_preserved_count=1,
        )
        assert 'summaries generated' in msg
        assert 'summaries preserved' in msg


# ---------------------------------------------------------------------------
# New: TestBuildBatchUpdateResponseMessage
# ---------------------------------------------------------------------------


class TestBuildBatchUpdateResponseMessage:
    """Test build_batch_update_response_message shared function."""

    def test_basic_batch_update(self) -> None:
        """Basic batch update message."""
        msg = build_batch_update_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0,
            summaries_generated_count=0,
            summaries_cleared_count=0,
        )
        assert msg == 'Updated 3/3 entries successfully'

    def test_with_summaries_cleared(self) -> None:
        """Batch with cleared summaries shows count."""
        msg = build_batch_update_response_message(
            succeeded=3, total=3,
            embeddings_generated_count=0,
            summaries_generated_count=0,
            summaries_cleared_count=2,
        )
        assert 'summaries cleared' in msg


class TestExecuteStoreInTransaction:
    """Test execute_store_in_transaction shared function."""

    @pytest.fixture
    def mock_repos(self) -> MagicMock:
        """Create a mock RepositoryContainer with all required sub-repositories."""
        repos = MagicMock()
        repos.context.store_with_deduplication = AsyncMock(return_value=('42', False))
        repos.tags.store_tags = AsyncMock()
        repos.tags.replace_tags_for_context = AsyncMock()
        repos.images.store_images = AsyncMock()
        repos.images.replace_images_for_context = AsyncMock()
        repos.embeddings.exists = AsyncMock(return_value=False)
        repos.embeddings.store_chunked = AsyncMock()
        return repos

    @pytest.fixture
    def mock_txn(self) -> MagicMock:
        """Create a mock transaction context."""
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        return txn

    @pytest.mark.asyncio
    async def test_basic_store_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Store a new entry with no tags, images, or embeddings."""
        context_id, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='test-thread',
            source='user',
            content_type='text',
            text_content='Hello world',
            metadata_str=None,
            summary=None,
            tags=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='test-model',
        )
        assert context_id == '42'
        assert was_updated is False
        assert embedding_stored is False
        mock_repos.context.store_with_deduplication.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_with_tags_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """New entry stores tags via store_tags (not replace)."""
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=['tag1', 'tag2'], validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.store_tags.assert_called_once_with('42', ['tag1', 'tag2'], txn=mock_txn)
        mock_repos.tags.replace_tags_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_tags_dedup_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Deduplicated entry replaces tags via replace_tags_for_context."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=(42, True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=['tag1'], validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.replace_tags_for_context.assert_called_once_with(42, ['tag1'], txn=mock_txn)
        mock_repos.tags.store_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_empty_tags_dedup_entry_clears(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """An explicitly provided empty tags list CLEARS tags on a dedup UPDATE.

        The documented replacement contract distinguishes provided from None:
        [] is a provided value and must replace (clear), matching update_context
        semantics; only None preserves existing tags.
        """
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=[], validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.replace_tags_for_context.assert_called_once_with('42', [], txn=mock_txn)
        mock_repos.tags.store_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_none_tags_dedup_entry_preserves(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """tags=None preserves existing tags on a dedup UPDATE (no tag write)."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.tags.replace_tags_for_context.assert_not_called()
        mock_repos.tags.store_tags.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_provided_empty_images_dedup_entry_clears(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """images provided as an empty list CLEARS images on a dedup UPDATE."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[], images_provided=True,
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.images.replace_images_for_context.assert_called_once_with('42', [], txn=mock_txn)
        mock_repos.images.store_images.assert_not_called()
        # Providing images (even []) means content_type is NOT preserved.
        dedup_kwargs = mock_repos.context.store_with_deduplication.call_args.kwargs
        assert dedup_kwargs['preserve_content_type_on_dedup'] is False

    @pytest.mark.asyncio
    async def test_store_with_absent_images_dedup_entry_preserves(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """images not provided (None from the caller) preserves existing images."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[], images_provided=False,
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.images.replace_images_for_context.assert_not_called()
        mock_repos.images.store_images.assert_not_called()
        dedup_kwargs = mock_repos.context.store_with_deduplication.call_args.kwargs
        assert dedup_kwargs['preserve_content_type_on_dedup'] is True

    @pytest.mark.asyncio
    async def test_store_summary_pending_divergence_raises_reconcile(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A divergence INSERT with a reused (summary_pending) summary aborts.

        The reused summary was read from a candidate that has since diverged and
        may describe different text; the transaction must abort via the
        reconcile signal so the caller regenerates it for THIS text.
        """
        from app.tools._shared import EmbeddingsReconcileRequiredError

        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', False))
        with pytest.raises(EmbeddingsReconcileRequiredError):
            await execute_store_in_transaction(
                mock_repos, mock_txn,
                thread_id='t', source='user', content_type='text',
                text_content='text', metadata_str=None, summary='reused summary',
                tags=None, validated_images=[],
                chunk_embeddings=None, embedding_model='m',
                summary_pending=True,
            )

    @pytest.mark.asyncio
    async def test_store_with_embeddings_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """New entry stores embeddings and returns embedding_stored=True."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        context_id, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=chunk_embeddings, embedding_model='m',
        )
        assert embedding_stored is True
        mock_repos.embeddings.store_chunked.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_embeddings_skipped_for_dedup_with_existing(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Deduplicated entry with existing embeddings skips storage."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=(42, True))
        mock_repos.embeddings.exists = AsyncMock(return_value=True)
        chunk_embeddings = cast(_ChunkEmbeddingList, [MagicMock()])
        context_id, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=chunk_embeddings, embedding_model='m',
        )
        assert embedding_stored is False
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_raises_on_failed_dedup(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises ToolError when store_with_deduplication returns falsy context_id."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=(0, False))
        with pytest.raises(ToolError, match='Failed to store context'):
            await execute_store_in_transaction(
                mock_repos, mock_txn,
                thread_id='t', source='user', content_type='text',
                text_content='text', metadata_str=None, summary=None,
                tags=None, validated_images=[],
                chunk_embeddings=None, embedding_model='m',
            )

    @pytest.mark.asyncio
    async def test_store_raises_reconcile_when_insert_skipped_embeddings(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """INSERT with skipped embeddings + generation enabled raises reconcile signal.

        Models the dedup pre-check / transaction divergence: the caller skipped
        embedding generation expecting an UPDATE, but store_with_deduplication
        inserted a new entry. The transaction must abort so the caller can
        regenerate embeddings outside the transaction and retry.
        """
        # Default fixture returns ('42', False) -- a genuine INSERT.
        with pytest.raises(EmbeddingsReconcileRequiredError) as exc_info:
            await execute_store_in_transaction(
                mock_repos, mock_txn,
                thread_id='t', source='user', content_type='text',
                text_content='reconcile me', metadata_str=None, summary=None,
                tags=None, validated_images=[],
                chunk_embeddings=None, embedding_model='m',
                embedding_generation_enabled=True,
            )
        assert exc_info.value.text_content == 'reconcile me'
        # Transaction aborted before any embedding write.
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_no_reconcile_on_dedup_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A dedup UPDATE with skipped embeddings does NOT trigger reconciliation."""
        mock_repos.context.store_with_deduplication = AsyncMock(return_value=('42', True))
        _, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
            embedding_generation_enabled=True,
        )
        assert was_updated is True
        assert embedding_stored is False

    @pytest.mark.asyncio
    async def test_store_no_reconcile_when_embeddings_present(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A new INSERT that already carries embeddings does NOT reconcile."""
        chunk_embeddings = cast(list[ChunkEmbedding], [MagicMock()])
        _, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=chunk_embeddings, embedding_model='m',
            embedding_generation_enabled=True,
        )
        assert was_updated is False
        assert embedding_stored is True
        mock_repos.embeddings.store_chunked.assert_called_once()

    @pytest.mark.asyncio
    async def test_store_no_reconcile_when_generation_disabled(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """With generation disabled (default), a new INSERT with no embeddings is allowed."""
        _, was_updated, embedding_stored = await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='text',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=[],
            chunk_embeddings=None, embedding_model='m',
            embedding_generation_enabled=False,
        )
        assert was_updated is False
        assert embedding_stored is False
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_store_with_images_new_entry(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """New entry stores images via store_images."""
        images = [{'data': 'abc', 'mime_type': 'image/png'}]
        await execute_store_in_transaction(
            mock_repos, mock_txn,
            thread_id='t', source='user', content_type='multimodal',
            text_content='text', metadata_str=None, summary=None,
            tags=None, validated_images=images,
            chunk_embeddings=None, embedding_model='m',
        )
        mock_repos.images.store_images.assert_called_once_with('42', images, txn=mock_txn)
        mock_repos.images.replace_images_for_context.assert_not_called()


class TestExecuteUpdateInTransaction:
    """Test execute_update_in_transaction shared function."""

    @pytest.fixture
    def mock_repos(self) -> MagicMock:
        """Create a mock RepositoryContainer with all required sub-repositories."""
        repos = MagicMock()
        repos.context.update_context_entry = AsyncMock(return_value=(True, ['text']))
        repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))
        repos.context.update_content_type = AsyncMock()
        repos.context.get_content_type = AsyncMock(return_value='text')
        repos.tags.replace_tags_for_context = AsyncMock()
        repos.images.replace_images_for_context = AsyncMock()
        repos.images.count_images_for_context = AsyncMock(return_value=0)
        repos.embeddings.delete_all_chunks = AsyncMock()
        repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)
        repos.embeddings.store_chunked = AsyncMock()
        return repos

    @pytest.fixture
    def mock_txn(self) -> MagicMock:
        """Create a mock transaction context."""
        txn = MagicMock()
        txn.backend_type = 'sqlite'
        return txn

    @pytest.mark.asyncio
    async def test_basic_text_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Update text field returns updated_fields with text."""
        updated_fields, summary_cleared = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='New text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'text' in updated_fields
        assert summary_cleared is False
        mock_repos.context.update_context_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_metadata_patch_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Apply metadata_patch calls patch_metadata."""
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text=None,
            metadata=None,
            metadata_patch={'key': 'value'},
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'metadata' in updated_fields
        mock_repos.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_tags_replacement(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Tags provided triggers replace_tags_for_context."""
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text=None,
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=['new-tag'],
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'tags' in updated_fields
        mock_repos.tags.replace_tags_for_context.assert_called_once()

    @pytest.mark.asyncio
    async def test_images_removal(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Empty images list removes all images and sets content_type to text."""
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text=None,
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=[],
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert 'images' in updated_fields
        assert 'content_type' in updated_fields
        mock_repos.images.replace_images_for_context.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', [], txn=mock_txn,
        )
        mock_repos.context.update_content_type.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', 'text', txn=mock_txn,
        )

    @pytest.mark.asyncio
    async def test_embeddings_regeneration(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Chunk embeddings provided triggers delete+store cycle."""
        chunk_embeddings = cast(_ChunkEmbeddingList, [MagicMock()])
        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='New text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=chunk_embeddings,
            embedding_model='m',
        )
        assert 'embedding' in updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', txn=mock_txn,
        )
        mock_repos.embeddings.store_chunked.assert_called_once()

    @pytest.mark.asyncio
    async def test_text_change_without_provider_clears_stale_embeddings(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Text changed, no new embeddings, tables exist -> stale chunks deleted.

        When an update changes text but no embedding provider regenerates vectors,
        the stored chunks describe the REPLACED text and must be DELETEd so semantic
        search cannot match the old content. Guarded by embedding_tables_exist.
        """
        mock_repos.images.count_images_for_context = AsyncMock(return_value=0)
        mock_repos.context.get_content_type = AsyncMock(return_value='text')
        mock_repos.context.update_content_type = AsyncMock()
        mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=True)
        mock_repos.embeddings.delete_all_chunks = AsyncMock(return_value=True)

        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='Replaced text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )

        assert 'embedding' in updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_called_once_with(
            '0190abcdef1234567890abcd00000001', txn=mock_txn,
        )
        mock_repos.embeddings.store_chunked.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_change_without_provider_tables_absent_is_noop(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Same text-change-without-provider path but embeddings were never
        provisioned -> safe no-op (no delete, 'embedding' not in updated_fields)."""
        mock_repos.images.count_images_for_context = AsyncMock(return_value=0)
        mock_repos.context.get_content_type = AsyncMock(return_value='text')
        mock_repos.context.update_content_type = AsyncMock()
        mock_repos.embeddings.embedding_tables_exist = AsyncMock(return_value=False)
        mock_repos.embeddings.delete_all_chunks = AsyncMock(return_value=True)

        updated_fields, _ = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='Replaced text',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=False,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )

        assert 'embedding' not in updated_fields
        mock_repos.embeddings.delete_all_chunks.assert_not_called()

    @pytest.mark.asyncio
    async def test_update_raises_on_failed_entry_update(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises ToolError when update_context_entry returns success=False."""
        mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))
        with pytest.raises(ToolError, match='Failed to update context entry'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                text='New text',
                metadata=None,
                metadata_patch=None,
                summary=None,
                clear_summary=False,
                tags=None,
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
            )

    @pytest.mark.asyncio
    async def test_update_raises_on_failed_metadata_patch(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """Raises ToolError when patch_metadata returns success=False."""
        mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))
        with pytest.raises(ToolError, match='Failed to patch metadata'):
            await execute_update_in_transaction(
                mock_repos, mock_txn,
                context_id='0190abcdef1234567890abcd00000001',
                text=None,
                metadata=None,
                metadata_patch={'key': 'value'},
                summary=None,
                clear_summary=False,
                tags=None,
                images=None,
                validated_images=[],
                chunk_embeddings=None,
                embedding_model='m',
            )

    @pytest.mark.asyncio
    async def test_summary_cleared_flag(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """clear_summary=True is returned as summary_cleared."""
        _, summary_cleared = await execute_update_in_transaction(
            mock_repos, mock_txn,
            context_id='0190abcdef1234567890abcd00000001',
            text='Short',
            metadata=None,
            metadata_patch=None,
            summary=None,
            clear_summary=True,
            tags=None,
            images=None,
            validated_images=[],
            chunk_embeddings=None,
            embedding_model='m',
        )
        assert summary_cleared is True
