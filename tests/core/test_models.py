"""
Test suite for Pydantic models used in the MCP Context Storage Server.

Tests model validation, field validators, content type detection,
tag normalization, and base64 image validation.
"""

import base64
from datetime import UTC
from datetime import datetime
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic import ValidationError

from app.models import MAX_IMAGES_PER_ENTRY
from app.models import ContentType
from app.models import ContextEntry
from app.models import DeleteContextRequest
from app.models import ImageAttachment
from app.models import SearchFilters
from app.models import SourceType
from app.models import StoreContextRequest
from app.models import normalize_base64_image_data


def _declared_max_length(model: type[BaseModel], field: str) -> int | None:
    """Extract the max_length constraint declared on a model field."""
    for meta in model.model_fields[field].metadata:
        value = getattr(meta, 'max_length', None)
        if value is not None:
            return int(value)
    return None


class TestNormalizeBase64ImageData:
    """Test the shared base64 payload normalizer."""

    def test_canonical_payload_is_identity(self) -> None:
        """Clean standard-alphabet base64 passes through unchanged."""
        payload = base64.b64encode(b'canonical bytes').decode()
        assert normalize_base64_image_data(payload) == payload

    def test_data_uri_prefix_stripped(self) -> None:
        """One leading RFC 2397 data-URI prefix is stripped."""
        payload = base64.b64encode(b'image bytes').decode()
        assert normalize_base64_image_data(f'data:image/png;base64,{payload}') == payload
        assert normalize_base64_image_data(f'data:image/jpeg;base64,{payload}') == payload

    def test_url_safe_alphabet_translated(self) -> None:
        """URL-safe '-'/'_' characters are translated to standard '+'/'/'."""
        raw = bytes(range(251, 256)) * 3
        url_safe = base64.urlsafe_b64encode(raw).decode()
        standard = base64.b64encode(raw).decode()
        assert url_safe != standard  # fixture sanity: the translation is actually exercised
        assert normalize_base64_image_data(url_safe) == standard

    def test_ascii_whitespace_removed(self) -> None:
        """Embedded ASCII whitespace (newlines, tabs, spaces) is removed."""
        payload = base64.b64encode(b'wrapped payload bytes').decode()
        wrapped = payload[:4] + '\r\n' + payload[4:8] + ' \t' + payload[8:]
        assert normalize_base64_image_data(wrapped) == payload

    def test_missing_padding_restored(self) -> None:
        """Stripped '=' padding is restored to a multiple of 4."""
        raw = b'\xfb\xef\x01\x02'
        payload = base64.b64encode(raw).decode()
        assert payload.endswith('=')
        assert normalize_base64_image_data(payload.rstrip('=')) == payload

    def test_non_base64_input_stays_strictly_rejectable(self) -> None:
        """Genuinely non-base64 input still fails a strict decode after normalization."""
        normalized = normalize_base64_image_data('!!!not base64!!!')
        with pytest.raises(Exception, match='(?i)base64'):
            base64.b64decode(normalized, validate=True)


class TestSourceType:
    """Test SourceType enum."""

    def test_source_type_values(self) -> None:
        """Test enum values are correct."""
        assert SourceType.USER.value == 'user'
        assert SourceType.AGENT.value == 'agent'

    def test_source_type_from_string(self) -> None:
        """Test creating enum from string."""
        assert SourceType('user') == SourceType.USER
        assert SourceType('agent') == SourceType.AGENT

    def test_invalid_source_type(self) -> None:
        """Test invalid source type raises error."""
        with pytest.raises(ValueError, match="'invalid' is not a valid SourceType"):
            SourceType('invalid')


class TestContentType:
    """Test ContentType enum."""

    def test_content_type_values(self) -> None:
        """Test enum values are correct."""
        assert ContentType.TEXT.value == 'text'
        assert ContentType.MULTIMODAL.value == 'multimodal'

    def test_content_type_from_string(self) -> None:
        """Test creating enum from string."""
        assert ContentType('text') == ContentType.TEXT
        assert ContentType('multimodal') == ContentType.MULTIMODAL


class TestImageAttachment:
    """Test ImageAttachment model."""

    def test_valid_image_attachment(self, sample_image_data: dict[str, Any]) -> None:
        """Test creating valid image attachment."""
        img = ImageAttachment(**sample_image_data)
        assert img.data == sample_image_data['data']
        assert img.mime_type == 'image/png'
        assert img.metadata is None  # No metadata in the sample
        assert img.position == 0

    def test_default_mime_type(self) -> None:
        """Test default mime type is image/png."""
        img = ImageAttachment(data=base64.b64encode(b'test').decode('utf-8'), position=0)
        assert img.mime_type == 'image/png'

    def test_invalid_base64(self) -> None:
        """Test invalid base64 data raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageAttachment(data='not-valid-base64!', position=0)
        assert 'Invalid base64 encoded data' in str(exc_info.value)

    def test_data_uri_payload_normalized_to_bare_base64(self) -> None:
        """A data-URI-prefixed payload is normalized to the canonical bare payload."""
        payload = base64.b64encode(b'attachment bytes').decode()
        img = ImageAttachment(data=f'data:image/png;base64,{payload}', position=0)
        assert img.data == payload

    def test_url_safe_payload_normalized_to_standard_alphabet(self) -> None:
        """A URL-safe payload is stored in the canonical standard alphabet."""
        raw = bytes(range(251, 256)) * 3
        img = ImageAttachment(data=base64.urlsafe_b64encode(raw).decode(), position=0)
        assert img.data == base64.b64encode(raw).decode()
        assert base64.b64decode(img.data, validate=True) == raw

    def test_zero_byte_payload_rejected(self) -> None:
        """A payload that decodes to zero bytes (empty data-URI payload) is rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ImageAttachment(data='data:image/png;base64,', position=0)
        assert 'decodes to zero bytes' in str(exc_info.value)

    def test_mime_type_is_advisory_free_form(self) -> None:
        """mime_type is an advisory, client-supplied label -- not an allowlist.

        Image bytes are opaque to the server (base64-validated and size-capped, never
        decoded or rendered), so a non-image label is stored verbatim rather than rejected.
        """
        img = ImageAttachment(
            data=base64.b64encode(b'test').decode('utf-8'),
            mime_type='image/svg+xml',
            position=0,
        )
        assert img.mime_type == 'image/svg+xml'

    @pytest.mark.parametrize(
        'mime_type',
        [
            'image/png',
            'image/jpeg',
            'image/jpg',
            'image/gif',
            'image/webp',
        ],
    )
    def test_valid_mime_types(self, mime_type: str) -> None:
        """Common image mime types are accepted and stored verbatim."""
        img = ImageAttachment(
            data=base64.b64encode(b'test').decode('utf-8'),
            mime_type=mime_type,
            position=0,
        )
        assert img.mime_type == mime_type

    def test_negative_position(self) -> None:
        """Test negative position raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageAttachment(
                data=base64.b64encode(b'test').decode('utf-8'),
                position=-1,
            )
        assert 'greater than or equal to 0' in str(exc_info.value)


class TestContextEntry:
    """Test ContextEntry model."""

    def test_minimal_context_entry(self) -> None:
        """Test creating context entry with minimal required fields."""
        entry = ContextEntry(
            thread_id='test_thread',
            source=SourceType.USER,
        )
        assert entry.thread_id == 'test_thread'
        assert entry.source == SourceType.USER
        assert entry.content_type == ContentType.TEXT
        assert entry.text_content is None
        assert entry.images == []
        assert entry.metadata is None
        assert entry.tags == []

    def test_full_context_entry(self) -> None:
        """Test creating context entry with all fields."""
        now = datetime.now(tz=UTC)
        canonical_id = '0190abcdef1234567890abcdef123456'
        entry = ContextEntry(
            id=canonical_id,
            thread_id='test_thread',
            source=SourceType.AGENT,
            content_type=ContentType.MULTIMODAL,
            text_content='Test content',
            images=[ImageAttachment(data=base64.b64encode(b'img').decode('utf-8'), position=0)],
            metadata={'key': 'value'},
            tags=['tag1', 'tag2'],
            created_at=now,
            updated_at=now,
        )
        assert entry.id == canonical_id
        assert entry.thread_id == 'test_thread'
        assert entry.source == SourceType.AGENT
        assert entry.content_type == ContentType.MULTIMODAL
        assert entry.text_content == 'Test content'
        assert len(entry.images) == 1
        assert entry.metadata == {'key': 'value'}
        assert entry.tags == ['tag1', 'tag2']
        assert entry.created_at == now
        assert entry.updated_at == now

    def test_tag_normalization(self) -> None:
        """Test tags are normalized to lowercase and stripped."""
        entry = ContextEntry(
            thread_id='test',
            source=SourceType.USER,
            tags=['  TAG1  ', 'Tag2', '  mixed-CASE  ', ''],
        )
        assert entry.tags == ['tag1', 'tag2', 'mixed-case']

    def test_auto_content_type_detection(self) -> None:
        """Test content type is automatically set to multimodal with images."""
        entry = ContextEntry(
            thread_id='test',
            source=SourceType.USER,
            content_type=ContentType.TEXT,  # Explicitly set to TEXT
            images=[ImageAttachment(data=base64.b64encode(b'img').decode('utf-8'), position=0)],
        )
        assert entry.content_type == ContentType.MULTIMODAL

    def test_empty_text_content_validation(self) -> None:
        """Test empty string in text_content is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            ContextEntry(
                thread_id='test',
                source=SourceType.USER,
                text_content='',  # Empty string should fail min_length=1
            )
        assert 'Text content cannot be empty or contain only whitespace' in str(exc_info.value)

    def test_too_many_images(self) -> None:
        """Test exceeding max images limit."""
        images = [
            ImageAttachment(data=base64.b64encode(b'img').decode('utf-8'), position=0)
            for _ in range(MAX_IMAGES_PER_ENTRY + 1)
        ]
        with pytest.raises(ValidationError) as exc_info:
            ContextEntry(
                thread_id='test',
                source=SourceType.USER,
                images=images,
            )
        assert f'List should have at most {MAX_IMAGES_PER_ENTRY} items' in str(exc_info.value)

    def test_images_max_length_matches_canonical_constant(self) -> None:
        """Both image-carrying models declare the single canonical count limit."""
        assert _declared_max_length(ContextEntry, 'images') == MAX_IMAGES_PER_ENTRY
        assert _declared_max_length(StoreContextRequest, 'images') == MAX_IMAGES_PER_ENTRY


class TestSearchFilters:
    """Test SearchFilters model."""

    def test_default_search_filters(self) -> None:
        """Test search filters with default values."""
        filters = SearchFilters(limit=50, offset=0)
        assert filters.thread_id is None
        assert filters.source is None
        assert filters.tags is None
        assert filters.start_date is None
        assert filters.end_date is None
        assert filters.content_type is None
        assert filters.limit == 50
        assert filters.offset == 0
        assert filters.include_images is False

    def test_search_filters_with_values(self) -> None:
        """Test search filters with all values set."""
        now = datetime.now(tz=UTC)
        filters = SearchFilters(
            thread_id='test_thread',
            source=SourceType.AGENT,
            tags=['tag1', 'tag2'],
            start_date=now,
            end_date=now,
            content_type=ContentType.MULTIMODAL,
            limit=100,
            offset=20,
            include_images=True,
        )
        assert filters.thread_id == 'test_thread'
        assert filters.source == SourceType.AGENT
        assert filters.tags == ['tag1', 'tag2']
        assert filters.start_date == now
        assert filters.end_date == now
        assert filters.content_type == ContentType.MULTIMODAL
        assert filters.limit == 100
        assert filters.offset == 20
        assert filters.include_images is True

    def test_limit_bounds(self) -> None:
        """Test limit parameter bounds validation."""
        # Test minimum
        filters = SearchFilters(limit=1, offset=0)
        assert filters.limit == 1

        # Test maximum
        filters = SearchFilters(limit=100, offset=0)
        assert filters.limit == 100

        # Test below minimum
        with pytest.raises(ValidationError) as exc_info:
            SearchFilters(limit=0, offset=0)
        assert 'greater than or equal to 1' in str(exc_info.value)

        # Test above maximum
        with pytest.raises(ValidationError) as exc_info:
            SearchFilters(limit=101, offset=0)
        assert 'less than or equal to 100' in str(exc_info.value)

    def test_negative_offset(self) -> None:
        """Test negative offset raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            SearchFilters(limit=50, offset=-1)
        assert 'greater than or equal to 0' in str(exc_info.value)


class TestStoreContextRequest:
    """Test StoreContextRequest model."""

    def test_minimal_store_request(self) -> None:
        """Test minimal valid store request."""
        request = StoreContextRequest(
            thread_id='test',
            source=SourceType.USER,
        )
        assert request.thread_id == 'test'
        assert request.source == SourceType.USER
        assert request.text is None
        assert request.images is None
        assert request.metadata is None
        assert request.tags is None

    def test_full_store_request(self, sample_image_data: dict[str, Any]) -> None:
        """Test store request with all fields."""
        request = StoreContextRequest(
            thread_id='test',
            source=SourceType.AGENT,
            text='Test content',
            images=[ImageAttachment(**sample_image_data)],
            metadata={'key': 'value'},
            tags=['tag1', 'tag2'],
        )
        assert request.thread_id == 'test'
        assert request.source == SourceType.AGENT
        assert request.text == 'Test content'
        assert request.images is not None
        assert len(request.images) == 1
        assert request.metadata == {'key': 'value'}
        assert request.tags == ['tag1', 'tag2']

    def test_empty_text_validation(self) -> None:
        """Test empty text string is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            StoreContextRequest(
                thread_id='test',
                source=SourceType.USER,
                text='',
            )
        assert 'text is required and cannot be empty' in str(exc_info.value)

    def test_too_many_images_in_request(self) -> None:
        """Test exceeding max images in store request."""
        images = [
            ImageAttachment(data=base64.b64encode(b'img').decode('utf-8'), position=0)
            for _ in range(MAX_IMAGES_PER_ENTRY + 1)
        ]
        with pytest.raises(ValidationError) as exc_info:
            StoreContextRequest(
                thread_id='test',
                source=SourceType.USER,
                images=images,
            )
        assert f'at most {MAX_IMAGES_PER_ENTRY} items' in str(exc_info.value)


class TestDeleteContextRequest:
    """Test DeleteContextRequest model."""

    def test_delete_by_ids(self) -> None:
        """Test delete request with context IDs."""
        ids = [
            '0190abcdef1234567890abcdef111111',
            '0190abcdef1234567890abcdef222222',
            '0190abcdef1234567890abcdef333333',
        ]
        request = DeleteContextRequest(context_ids=ids)
        assert request.context_ids == ids
        assert request.thread_id is None

    def test_delete_by_thread(self) -> None:
        """Test delete request with thread ID."""
        request = DeleteContextRequest(thread_id='test_thread')
        assert request.context_ids is None
        assert request.thread_id == 'test_thread'

    def test_delete_with_both_fields(self) -> None:
        """Test delete request can have both IDs and thread."""
        ids = [
            '0190abcdef1234567890abcdef444444',
            '0190abcdef1234567890abcdef555555',
        ]
        request = DeleteContextRequest(
            context_ids=ids,
            thread_id='test_thread',
        )
        assert request.context_ids == ids
        assert request.thread_id == 'test_thread'

    def test_delete_without_any_field(self) -> None:
        """Test delete request without any field raises error."""
        with pytest.raises(ValidationError) as exc_info:
            DeleteContextRequest()
        assert 'Must provide either context_ids or thread_id' in str(exc_info.value)

    def test_empty_context_ids_list(self) -> None:
        """Test empty context IDs list is invalid."""
        with pytest.raises(ValidationError) as exc_info:
            DeleteContextRequest(context_ids=[])
        assert 'context_ids cannot be an empty list' in str(exc_info.value)
