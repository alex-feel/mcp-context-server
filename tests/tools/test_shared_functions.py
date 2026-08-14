"""Tests for app.tools._shared module.

Contains:
- Relocated tests from tests/backends/test_postgresql_backend.py
  (TestTransactionHeartbeat, TestConnectionErrorClassification)
- New tests for validate_and_normalize_images
- New tests for build_store_response_message, build_update_response_message
- New tests for build_batch_store_response_message, build_batch_update_response_message
"""

import asyncio
import base64
import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from typing import get_type_hints
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import PropertyMock
from unittest.mock import patch

import asyncpg
import pytest
from fastmcp.exceptions import ToolError
from pydantic import TypeAdapter

import app.tools._shared as shared_module
from app.models import MAX_IMAGES_PER_ENTRY
from app.repositories.embedding_repository import ChunkEmbedding
from app.settings import get_settings
from app.summary.retry import SummaryRetryExhaustedError
from app.summary.retry import SummaryTimeoutError
from app.tools._shared import EmbeddingsReconcileRequiredError
from app.tools._shared import EntryNotFoundError
from app.tools._shared import build_batch_store_response_message
from app.tools._shared import build_batch_update_response_message
from app.tools._shared import build_store_response_message
from app.tools._shared import build_update_response_message
from app.tools._shared import entry_boundary_error
from app.tools._shared import execute_store_in_transaction
from app.tools._shared import execute_update_in_transaction
from app.tools._shared import generate_summary_with_timeout
from app.tools._shared import indexed_value_error
from app.tools._shared import is_connection_error
from app.tools._shared import reject_invalid_indexed_values
from app.tools._shared import reread_entry_version
from app.tools._shared import transaction_heartbeat
from app.tools._shared import validate_and_normalize_images
from app.tools.context import store_context
from app.tools.context import update_context

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

    def test_transaction_rollback_errors_are_retryable(self) -> None:
        """Server-initiated transaction rollbacks (SQLSTATE class 40) are retryable.

        PostgreSQL aborts one transaction to break a deadlock (40P01) or a
        serialization cycle (40001); the loser is expected to retry and succeeds
        once the competing transaction commits. Classifying the class-40 base
        as a connection-style transient makes the tool layer re-run the
        transaction instead of surfacing routine lock contention to the client.
        """
        assert is_connection_error(asyncpg.exceptions.TransactionRollbackError('rollback'))
        assert is_connection_error(asyncpg.exceptions.DeadlockDetectedError('deadlock detected'))
        assert is_connection_error(asyncpg.exceptions.SerializationError('could not serialize access'))

    def test_transaction_rollback_sqlstates_are_class_40(self) -> None:
        """Document the SQLSTATEs this classifier treats as retryable rollbacks."""
        assert asyncpg.exceptions.TransactionRollbackError.sqlstate == '40000'
        assert asyncpg.exceptions.DeadlockDetectedError.sqlstate == '40P01'
        assert asyncpg.exceptions.SerializationError.sqlstate == '40001'

    def test_sqlite_locked_family_is_retryable(self) -> None:
        """SQLite write contention (SQLITE_BUSY / SQLITE_LOCKED family) is retryable.

        begin_transaction -- the path every store/update transaction site uses --
        bypasses the SQLite write queue and performs no backend-level retry, so
        the tool-layer retry loops must classify a cross-process lock collision
        as transient, mirroring the PostgreSQL class-40 rollback treatment.
        """
        assert is_connection_error(sqlite3.OperationalError('database is locked'))
        assert is_connection_error(sqlite3.OperationalError('database table is locked'))

    def test_generic_sqlite_operational_error_not_retryable(self) -> None:
        """A non-contention sqlite3.OperationalError is NOT classified retryable."""
        assert not is_connection_error(sqlite3.OperationalError('no such table: context_entries'))
        assert not is_connection_error(sqlite3.OperationalError('malformed database schema'))

    def test_pool_saturation_timeout_error_not_retryable(self) -> None:
        """A pool-acquire TimeoutError is NOT retried despite subclassing OSError.

        TimeoutError is an OSError subclass on Python 3.12, so without an explicit
        carve-out it would ride the bare-OSError arm and be retried. The pool-acquire
        TimeoutError begin_transaction re-raises signals a SATURATED connection pool,
        and retrying it re-runs the full acquire wait each time. It must fail fast at
        the tool layer, matching execute_write's handling of the same signal.
        """
        assert isinstance(TimeoutError(), OSError)
        # asyncio.TimeoutError is an alias for the builtin TimeoutError on Python
        # 3.11+, and that builtin is exactly what asyncpg's pool.acquire timeout
        # surfaces, so excluding TimeoutError covers the pool-saturation shape.
        assert not is_connection_error(TimeoutError('pool acquire timed out'))

    def test_connection_reset_error_still_retryable_after_timeout_carveout(self) -> None:
        """The TimeoutError carve-out does not disturb the other OSError members.

        ConnectionResetError is an OSError subclass but not a TimeoutError, so a lost
        connection stays retryable after the saturation-timeout exclusion.
        """
        assert not isinstance(ConnectionResetError('reset'), TimeoutError)
        assert is_connection_error(ConnectionResetError('connection reset by peer'))
        assert is_connection_error(OSError('network unreachable'))


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

    def test_raise_mode_rejects_dict_image_metadata(self) -> None:
        """A dict-valued image metadata (untyped batch path only) is rejected.

        Per-image metadata crosses the boundary as a JSON-ENCODED STRING: the typed
        single-entry tools declare images as list[dict[str, str]], so a dict there is
        a Pydantic error. The untyped batch path bypassed that and stored the dict,
        which the strict get_context_by_ids output schema then refused to serialize --
        making the entry permanently unreadable. Rejecting the shape here also
        subsumes the NaN/Infinity parity hazard: only a bare float serializes to the
        invalid-JSON tokens PostgreSQL's jsonb column rejects, and a JSON-encoded
        string carries none.
        """
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': {'score': float('nan')}}])
        with pytest.raises(ToolError, match='Image 0 metadata must be a JSON-encoded string'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_collect_mode_rejects_non_string_image_metadata(self) -> None:
        """collect mode reports the same shape rejection as a per-entry error."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': {'deep': {'x': 1}}}])
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'Image 0 metadata must be a JSON-encoded string' in errors[0]

    def test_null_image_metadata_is_treated_as_absent(self) -> None:
        """An explicit null metadata means "no metadata" and is accepted.

        It stores as SQL NULL exactly like an omitted key, so refusing it would add
        friction without preventing anything.
        """
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': None}])
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'multimodal'
        assert errors == []

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

    def test_raise_mode_garbage_rejected_by_strict_decode(self) -> None:
        """Garbage input (all non-alphabet characters) fails loudly under the strict decode.

        The lenient decode used previously silently discarded every character outside
        the alphabet, so a value like '!!!!' decoded to b'' and was caught only by the
        zero-byte guard. The strict decode (validate=True) rejects it directly with a
        clear, deterministic per-image error.
        """
        with pytest.raises(ToolError, match='Image 0 has invalid base64 encoding: Invalid base64 data'):
            validate_and_normalize_images([{'data': '!!!!'}], error_mode='raise')

    def test_collect_mode_garbage_rejected_by_strict_decode(self) -> None:
        """collect mode records the strict-decode failure instead of raising."""
        _, content_type, errors = validate_and_normalize_images(
            [{'data': '@#$%'}], error_mode='collect',
        )
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'invalid base64 encoding' in errors[0]

    def test_raise_mode_empty_data_uri_payload_decodes_to_zero_bytes(self) -> None:
        """A data-URI prefix with an empty payload normalizes to '' and is rejected as zero bytes."""
        with pytest.raises(ToolError, match='Image 0 "data" decodes to zero bytes'):
            validate_and_normalize_images([{'data': 'data:image/png;base64,'}], error_mode='raise')

    def test_whitespace_wrapped_base64_still_accepted(self) -> None:
        """Newline-wrapped base64 is accepted: whitespace is removed by normalization before the strict decode."""
        wrapped = VALID_BASE64_PNG[:4] + '\n' + VALID_BASE64_PNG[4:]
        images, content_type, errors = validate_and_normalize_images(
            [{'data': wrapped, 'mime_type': 'image/png'}], error_mode='collect',
        )
        assert content_type == 'multimodal'
        assert errors == []
        assert images[0]['data'] == VALID_BASE64_PNG

    def test_data_uri_prefix_stripped_decodes_to_same_bytes(self) -> None:
        """A data-URI-prefixed payload decodes to the same bytes as the bare payload.

        Under the lenient decode, a prefix whose base64-alphabet character count is a
        multiple of 4 (e.g. some jpeg data-URIs) decoded as garbage bytes silently
        prepended to the image. Normalization strips the prefix so the stored bytes
        are exactly the intended image.
        """
        prefixed = {'data': 'data:image/jpeg;base64,' + VALID_BASE64_PNG}
        images, content_type, errors = validate_and_normalize_images([prefixed])
        assert errors == []
        assert content_type == 'multimodal'
        assert images[0]['data'] == VALID_BASE64_PNG
        assert base64.b64decode(images[0]['data'], validate=True) == base64.b64decode(VALID_BASE64_PNG, validate=True)

    def test_url_safe_alphabet_normalized_to_standard(self) -> None:
        """A URL-safe-alphabet payload is translated to the standard alphabet and decodes correctly."""
        raw = bytes(range(251, 256)) * 6
        standard = base64.b64encode(raw).decode()
        url_safe = base64.urlsafe_b64encode(raw).decode()
        assert url_safe != standard  # fixture sanity: the translation is actually exercised
        images, content_type, errors = validate_and_normalize_images([{'data': url_safe}])
        assert errors == []
        assert content_type == 'multimodal'
        assert images[0]['data'] == standard
        assert base64.b64decode(images[0]['data'], validate=True) == raw

    def test_url_safe_payload_without_padding_restored(self) -> None:
        """A URL-safe payload with stripped '=' padding is repadded and decodes to the original bytes."""
        raw = b'\xfb\xef\x01\x02'
        stripped = base64.urlsafe_b64encode(raw).decode().rstrip('=')
        images, _, errors = validate_and_normalize_images([{'data': stripped}])
        assert errors == []
        assert images[0]['data'] == base64.b64encode(raw).decode()
        assert base64.b64decode(images[0]['data'], validate=True) == raw

    def test_normalization_mutates_input_dict_in_place(self) -> None:
        """The canonical payload is written back into the caller's dict.

        The batch tools rely on this: they discard the returned list and later pass
        the same dict objects to the transaction helpers, so the repository re-decode
        must see the normalized payload through the original dict.
        """
        img = {'data': 'data:image/png;base64,' + VALID_BASE64_PNG}
        validate_and_normalize_images([img])
        assert img['data'] == VALID_BASE64_PNG

    def test_count_over_limit_rejected_raise_mode(self) -> None:
        """More than MAX_IMAGES_PER_ENTRY images is rejected at the shared chokepoint."""
        imgs = [{'data': VALID_BASE64_PNG} for _ in range(MAX_IMAGES_PER_ENTRY + 1)]
        expected = (
            f'Too many images: {MAX_IMAGES_PER_ENTRY + 1} provided, '
            f'maximum is {MAX_IMAGES_PER_ENTRY} per entry'
        )
        with pytest.raises(ToolError, match=expected):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_count_over_limit_rejected_collect_mode(self) -> None:
        """collect mode records the count-limit error (covers the batch tools' untyped path)."""
        imgs = [{'data': VALID_BASE64_PNG} for _ in range(MAX_IMAGES_PER_ENTRY + 1)]
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'Too many images' in errors[0]

    def test_count_at_limit_accepted(self) -> None:
        """Exactly MAX_IMAGES_PER_ENTRY images passes validation."""
        imgs = [{'data': VALID_BASE64_PNG} for _ in range(MAX_IMAGES_PER_ENTRY)]
        images, content_type, errors = validate_and_normalize_images(imgs)
        assert errors == []
        assert content_type == 'multimodal'
        assert len(images) == MAX_IMAGES_PER_ENTRY


# ---------------------------------------------------------------------------
# New: TestImageCountLimitToolSchema
# ---------------------------------------------------------------------------


class TestImageCountLimitToolSchema:
    """The MCP wire schema advertises the image-count bound on the live tool params."""

    @pytest.mark.parametrize('tool_fn', [store_context, update_context])
    def test_images_param_advertises_max_items(self, tool_fn: Callable[..., Awaitable[object]]) -> None:
        """The images parameter declares maxItems=MAX_IMAGES_PER_ENTRY in its JSON schema."""
        hints = get_type_hints(tool_fn, include_extras=True)
        schema = TypeAdapter(hints['images']).json_schema()
        branches = schema.get('anyOf', [schema])
        array_branches = [b for b in branches if isinstance(b, dict) and b.get('type') == 'array']
        assert array_branches, f'no array branch in images schema: {schema}'
        assert array_branches[0].get('maxItems') == MAX_IMAGES_PER_ENTRY


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
        repos.context.touch_updated_at = AsyncMock(return_value=True)
        repos.context.get_content_type = AsyncMock(return_value='text')
        repos.context.entry_exists = AsyncMock(return_value=True)
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
        """Raises EntryNotFoundError when update_context_entry reports no such row."""
        mock_repos.context.update_context_entry = AsyncMock(return_value=(False, []))
        with pytest.raises(EntryNotFoundError, match='not found'):
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
        """Raises EntryNotFoundError when patch_metadata reports no such row."""
        mock_repos.context.patch_metadata = AsyncMock(return_value=(False, []))
        with pytest.raises(EntryNotFoundError, match='not found'):
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
    async def test_tags_only_update_missing_parent_raises_not_found(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """A tags-only update against a vanished parent row raises EntryNotFoundError.

        The parent can disappear between the pre-generation existence check and this
        transaction (concurrent delete). Without the guard the tags write would fire a
        foreign-key insert against a missing parent, charging the circuit breaker.
        """
        mock_repos.context.entry_exists = AsyncMock(return_value=False)
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
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
        mock_repos.tags.replace_tags_for_context.assert_not_called()

    @pytest.mark.asyncio
    async def test_images_only_update_missing_parent_raises_not_found(
        self, mock_repos: MagicMock, mock_txn: MagicMock,
    ) -> None:
        """An images-only update against a vanished parent row raises EntryNotFoundError.

        Same concurrent-delete race as the tags-only path; the guard stops the image
        write from touching a missing parent and charging the circuit breaker.
        """
        mock_repos.context.entry_exists = AsyncMock(return_value=False)
        with pytest.raises(EntryNotFoundError, match='not found'):
            await execute_update_in_transaction(
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
        mock_repos.images.replace_images_for_context.assert_not_called()

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


class TestGenerateSummaryErrorContract:
    """generate_summary_with_timeout normalizes EVERY failure to ToolError.

    Its sibling abort-mandatory leg (generate_embeddings_with_timeout) already
    converts any provider failure into a ToolError, and call sites that isolate a
    per-entry failure rely on that shared contract with ``except ToolError``. The
    summary helper used to convert only its own outer timeout, so the retry layer's
    SummaryTimeoutError / SummaryRetryExhaustedError and the providers' bare
    RuntimeError / ValueError escaped raw and slipped past those guards.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        'failure',
        [
            SummaryRetryExhaustedError('retries exhausted'),
            SummaryTimeoutError('provider timed out'),
            RuntimeError('ollama unreachable'),
            ValueError('text too long for the model context'),
        ],
        ids=['retries-exhausted', 'provider-timeout', 'runtime-error', 'value-error'],
    )
    async def test_provider_failures_become_tool_error(self, failure: Exception) -> None:
        """Every provider failure class arrives as a ToolError naming the leg."""
        provider = MagicMock()
        provider.summarize = AsyncMock(side_effect=failure)
        with (
            patch('app.tools._shared.get_summary_provider', return_value=provider),
            pytest.raises(ToolError, match='Summary generation failed'),
        ):
            await generate_summary_with_timeout('body text', 'agent')

    @pytest.mark.asyncio
    async def test_outer_timeout_keeps_its_dedicated_message(self) -> None:
        """The total-timeout message stays distinct from the generic failure message."""

        async def _never_returns(text: str, source: str) -> str:
            _ = (text, source)
            await asyncio.sleep(10)
            return 'unreachable'

        provider = MagicMock()
        provider.summarize = _never_returns
        with (
            patch('app.tools._shared.get_summary_provider', return_value=provider),
            patch('app.tools._shared.compute_summary_total_timeout', return_value=0.01),
            pytest.raises(ToolError, match='exceeded total timeout'),
        ):
            await generate_summary_with_timeout('body text', 'agent')

    @pytest.mark.asyncio
    async def test_cancellation_still_propagates(self) -> None:
        """Cancellation is NOT swallowed: run_generation cancels this leg on abort."""
        started = asyncio.Event()

        async def _block(text: str, source: str) -> str:
            _ = (text, source)
            started.set()
            await asyncio.sleep(10)
            return 'unreachable'

        provider = MagicMock()
        provider.summarize = _block
        with patch('app.tools._shared.get_summary_provider', return_value=provider):
            task = asyncio.create_task(generate_summary_with_timeout('body text', 'agent'))
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task


class TestRereadEntryVersion:
    """The version refresh after a compare-and-set conflict retries the READ.

    ``version`` is monotonic, so re-entering the write with the token whose
    compare-and-set just failed matches zero rows by construction. A transient fault
    during the refresh must therefore be retried on the read itself, leaving the
    caller to re-enter the write only once a fresh token is in hand.
    """

    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_and_returns_fresh_version(self) -> None:
        """A dropped connection during the refresh retries and yields the new version."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(
            side_effect=[asyncpg.InterfaceError('connection recycled'), (True, 'agent', 7)],
        )
        with patch('app.tools._shared.asyncio.sleep', new_callable=AsyncMock):
            exists, version = await reread_entry_version(repos, '0190abcdef1234567890abcdef123456')
        assert exists is True
        assert version == 7
        assert repos.context.check_entry_exists.await_count == 2

    @pytest.mark.asyncio
    async def test_missing_entry_is_reported_not_retried(self) -> None:
        """A deleted row is a clean answer, not a fault to retry."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(return_value=(False, None, None))
        exists, version = await reread_entry_version(repos, '0190abcdef1234567890abcdef123456')
        assert exists is False
        assert version is None
        assert repos.context.check_entry_exists.await_count == 1

    @pytest.mark.asyncio
    async def test_exhausted_retries_propagate(self) -> None:
        """The refresh is bounded; a persistent fault surfaces to the caller."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(
            side_effect=asyncpg.InterfaceError('connection recycled'),
        )
        with (
            patch('app.tools._shared.asyncio.sleep', new_callable=AsyncMock),
            pytest.raises(asyncpg.InterfaceError),
        ):
            await reread_entry_version(repos, '0190abcdef1234567890abcdef123456', max_retries=1)
        assert repos.context.check_entry_exists.await_count == 2

    @pytest.mark.asyncio
    async def test_logical_error_is_not_retried(self) -> None:
        """A non-transient error fails immediately instead of burning retries."""
        repos = MagicMock()
        repos.context.check_entry_exists = AsyncMock(side_effect=ValueError('bad id'))
        with pytest.raises(ValueError, match='bad id'):
            await reread_entry_version(repos, '0190abcdef1234567890abcdef123456')
        assert repos.context.check_entry_exists.await_count == 1


class TestIndexedValueCastCompatibility:
    """A typed METADATA_INDEXED_FIELDS entry is validated at the write boundary.

    An ``integer``/``boolean``/``float`` type hint becomes a hard SQL cast inside
    the PostgreSQL expression index, which PostgreSQL evaluates on INSERT. Without a
    boundary check a cast-incompatible value passed every guard, paid the full
    generation pass, and then aborted the transaction with a raw driver error --
    while storing happily on SQLite, whose json_extract index applies no cast.
    """

    @pytest.fixture
    def typed_indexed_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configure one field per castable type hint for the module under test."""
        monkeypatch.setenv(
            'METADATA_INDEXED_FIELDS',
            'status,priority:integer,completed:boolean,score:float',
        )
        get_settings.cache_clear()
        monkeypatch.setattr(shared_module, 'settings', get_settings())

    @pytest.mark.usefixtures('typed_indexed_fields')
    @pytest.mark.parametrize(
        ('metadata', 'expected'),
        [
            ({'priority': 5}, None),
            ({'priority': '5'}, None),
            ({'priority': ' -12 '}, None),
            # PostgreSQL 16 added non-decimal literals and '_' digit separators to the
            # integer and numeric input functions, so these cast fine and refusing them
            # would block a store both backends accept.
            ({'priority': '0x10'}, None),
            ({'priority': '0o17'}, None),
            ({'priority': '0b101'}, None),
            ({'priority': '1_000'}, None),
            ({'priority': '0x_10'}, None),
            ({'priority': '1__0'}, 'not a valid integer'),
            ({'priority': '_1'}, 'not a valid integer'),
            ({'priority': '1_'}, 'not a valid integer'),
            # PostgreSQL trims ASCII whitespace only: its scanners run under the C locale
            # and test single bytes, so a Unicode space is ordinary content the cast
            # chokes on. Python's argument-less strip() would remove it and accept these.
            ({'priority': '\xa05'}, 'not a valid integer'),
            ({'priority': '5　'}, 'not a valid integer'),
            ({'priority': ' 5'}, 'not a valid integer'),
            ({'completed': '\xa0true'}, 'not a valid boolean'),
            ({'score': '\x1c1.5'}, 'not a valid float'),
            ({'priority': 'high'}, 'not a valid integer'),
            ({'priority': 5.5}, 'not a valid integer'),
            ({'priority': True}, 'not a valid integer'),
            ({'priority': [1, 2]}, 'cannot be stored'),
            ({'priority': 99999999999}, 'out of range'),
            ({'priority': -99999999999}, 'out of range'),
            ({'priority': None}, None),
            ({'completed': True}, None),
            ({'completed': 'yes'}, None),
            # 'of' is an accepted prefix of 'off'; a lone 'o' is ambiguous and rejected.
            ({'completed': 'of'}, None),
            ({'completed': 'o'}, 'not a valid boolean'),
            ({'completed': 'maybe'}, 'not a valid boolean'),
            ({'completed': 7}, 'not a valid boolean'),
            ({'score': 1.5}, None),
            ({'score': '2e3'}, None),
            ({'score': 'NaN'}, None),
            ({'score': 'high'}, 'not a valid float'),
            ({'status': 'anything at all'}, None),
            ({'unindexed': 'anything at all'}, None),
        ],
    )
    def test_cast_compatibility_matches_postgresql(
        self, metadata: dict[str, object], expected: str | None,
    ) -> None:
        """Values PostgreSQL would accept pass; values it would reject are refused."""
        result = indexed_value_error(metadata=metadata)
        if expected is None:
            assert result is None
        else:
            assert result is not None
            assert expected in result

    @pytest.mark.usefixtures('typed_indexed_fields')
    def test_raising_wrapper_reports_the_field(self) -> None:
        """The single-entry wrapper names the offending field, as the length cap does."""
        with pytest.raises(ToolError, match="metadata field 'priority'"):
            reject_invalid_indexed_values(metadata={'priority': 'high'})

    @pytest.mark.usefixtures('typed_indexed_fields')
    def test_batch_chokepoint_covers_metadata_and_patch(self) -> None:
        """The untyped batch chokepoint checks both the replacement and the patch form."""
        assert entry_boundary_error(metadata={'priority': 'high'}) is not None
        assert entry_boundary_error(metadata_patch={'completed': 'maybe'}) is not None

    def test_default_configuration_has_no_castable_hints(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The shipped defaults declare only string/array/object, so nothing casts.

        Array and object fields build no expression index at all (the always-present
        GIN index serves them), so their values must not be type-checked here.
        """
        get_settings.cache_clear()
        monkeypatch.setattr(shared_module, 'settings', get_settings())
        assert indexed_value_error(metadata={'status': 'anything', 'technologies': ['a', 'b']}) is None


class TestIndexedValueLengthMeasuresTheIndexedText:
    """The length cap measures the text ``metadata->>'<field>'`` actually yields.

    ``->>`` renders a JSON string unquoted and every other JSON value as its serialized
    form, so a list or object stored under a string-typed indexed field is indexed at
    its full serialized width. A cap that inspected only ``str`` values let such a
    container through to abort the PostgreSQL INSERT inside the store transaction --
    after a full generation pass, while charging the circuit breaker -- for a value
    SQLite stored happily.
    """

    @pytest.fixture
    def string_indexed_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configure one string-typed and one array-typed indexed field."""
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'project,technologies:array')
        get_settings.cache_clear()
        monkeypatch.setattr(shared_module, 'settings', get_settings())

    @pytest.mark.usefixtures('string_indexed_field')
    def test_oversized_container_under_a_string_field_is_refused(self) -> None:
        """A list whose serialized text exceeds the cap is refused, like a long string."""
        oversized = ['x' * 100] * 40

        message = indexed_value_error(metadata={'project': oversized})

        assert message is not None
        assert 'too long' in message

    @pytest.mark.usefixtures('string_indexed_field')
    def test_small_container_under_a_string_field_is_accepted(self) -> None:
        """A container whose serialized text fits is stored, not blanket-refused."""
        assert indexed_value_error(metadata={'project': ['alpha', 'beta']}) is None

    @pytest.mark.usefixtures('string_indexed_field')
    def test_array_typed_field_is_exempt_from_the_cap(self) -> None:
        """An array-typed field builds no expression index, so its width is unbounded.

        It is served by the always-present GIN index, which hashes its entries.
        """
        assert indexed_value_error(metadata={'technologies': ['x' * 100] * 40}) is None

    @pytest.mark.usefixtures('string_indexed_field')
    def test_batch_chokepoint_sees_the_same_breach(self) -> None:
        """The untyped batch path refuses it too, so both write paths agree."""
        assert entry_boundary_error(metadata={'project': ['x' * 100] * 40}) is not None
