"""
Test suite for Pydantic models used in the MCP Context Storage Server.

Tests model validation, field validators, content type detection,
tag normalization, and base64 image validation.
"""

from __future__ import annotations

import base64
from datetime import UTC
from datetime import datetime
from typing import Any

import pytest
from app.models import ContentType
from app.models import ContextEntry
from app.models import DeleteContextRequest
from app.models import ImageAttachment
from app.models import SearchFilters
from app.models import SourceType
from app.models import StoreContextRequest
from pydantic import ValidationError


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

    def test_invalid_mime_type(self) -> None:
        """Test invalid mime type raises validation error."""
        with pytest.raises(ValidationError) as exc_info:
            ImageAttachment(
                data=base64.b64encode(b'test').decode('utf-8'),
                mime_type='text/plain',
                position=0,
            )
        assert 'String should match pattern' in str(exc_info.value)

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
        """Test all valid mime types are accepted."""
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
        entry = ContextEntry(
            id=1,
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
        assert entry.id == 1
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
        assert 'at least 1 character' in str(exc_info.value)

    def test_too_many_images(self) -> None:
        """Test exceeding max images limit."""
        images = [
            ImageAttachment(data=base64.b64encode(b'img').decode('utf-8'), position=0)
            for _ in range(11)  # 11 images, max is 10
        ]
        with pytest.raises(ValidationError) as exc_info:
            ContextEntry(
                thread_id='test',
                source=SourceType.USER,
                images=images,
            )
        assert 'List should have at most 10 items' in str(exc_info.value)


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
        filters = SearchFilters(limit=500, offset=0)
        assert filters.limit == 500

        # Test below minimum
        with pytest.raises(ValidationError) as exc_info:
            SearchFilters(limit=0, offset=0)
        assert 'greater than or equal to 1' in str(exc_info.value)

        # Test above maximum
        with pytest.raises(ValidationError) as exc_info:
            SearchFilters(limit=501, offset=0)
        assert 'less than or equal to 500' in str(exc_info.value)

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
        assert 'at least 1 character' in str(exc_info.value)

    def test_too_many_images_in_request(self) -> None:
        """Test exceeding max images in store request."""
        images = [ImageAttachment(data=base64.b64encode(b'img').decode('utf-8'), position=0) for _ in range(11)]
        with pytest.raises(ValidationError) as exc_info:
            StoreContextRequest(
                thread_id='test',
                source=SourceType.USER,
                images=images,
            )
        assert 'at most 10 items' in str(exc_info.value)


class TestDeleteContextRequest:
    """Test DeleteContextRequest model."""

    def test_delete_by_ids(self) -> None:
        """Test delete request with context IDs."""
        request = DeleteContextRequest(context_ids=[1, 2, 3])
        assert request.context_ids == [1, 2, 3]
        assert request.thread_id is None

    def test_delete_by_thread(self) -> None:
        """Test delete request with thread ID."""
        request = DeleteContextRequest(thread_id='test_thread')
        assert request.context_ids is None
        assert request.thread_id == 'test_thread'

    def test_delete_with_both_fields(self) -> None:
        """Test delete request can have both IDs and thread."""
        request = DeleteContextRequest(
            context_ids=[1, 2],
            thread_id='test_thread',
        )
        assert request.context_ids == [1, 2]
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
        assert 'List should have at least 1 item' in str(exc_info.value)
