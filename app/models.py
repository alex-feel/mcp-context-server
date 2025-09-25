from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

# Define MetadataDict directly to avoid import issues
type MetadataDict = dict[str, str | int | float | bool | None]


class SourceType(StrEnum):
    """Allowed source types for context entries"""

    USER = 'user'
    AGENT = 'agent'


class ContentType(StrEnum):
    """Content type classification"""

    TEXT = 'text'
    MULTIMODAL = 'multimodal'


class ImageAttachment(BaseModel):
    """Image attachment model for transport"""

    data: str = Field(..., description='Base64 encoded image data')
    mime_type: str = Field(default='image/png', pattern='^image/(png|jpeg|jpg|gif|webp)$')
    metadata: MetadataDict | None = Field(default=None)
    position: Annotated[int, Field(default=0, ge=0)]

    @field_validator('data')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Validate base64 format"""
        import base64

        try:
            base64.b64decode(v)
            return v
        except Exception as e:
            raise ValueError('Invalid base64 encoded data') from e


class ContextEntry(BaseModel):
    """Core context entry model"""

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'thread_id': 'task_123',
                    'source': 'user',
                    'text_content': 'Analyze this data',
                    'tags': ['analysis', 'priority-high'],
                },
            ],
        },
    }

    id: int | None = Field(default=None, description='Auto-generated ID')
    thread_id: str = Field(..., description='Thread identifier for context scoping')
    source: SourceType = Field(..., description='Origin of the context')
    content_type: ContentType = Field(default=ContentType.TEXT)
    text_content: str | None = Field(default=None, min_length=1)
    images: list[Any] = Field(default_factory=list, max_length=10)  # Pyright false positive - ImageAttachment is defined
    metadata: MetadataDict | None = Field(default=None)
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)

    @field_validator('tags')
    @classmethod
    def validate_tags(cls, v: list[str]) -> list[str]:
        """Ensure tags are properly formatted with robust character support.

        Handles:
        - Unicode characters including forward slashes (/)
        - Edge cases like double-encoding
        - Non-string types that can be converted
        - Whitespace normalization

        Returns:
            List of validated and normalized tag strings.
        """
        validated_tags: list[str] = []
        for tag in v:
            # The tag should always be a string at this point due to type hints
            # Normalize whitespace: replace multiple spaces/tabs/newlines with single space
            normalized_tag = ' '.join(tag.split()).strip().lower()
            if normalized_tag:
                validated_tags.append(normalized_tag)
        return validated_tags

    @model_validator(mode='after')
    def set_content_type(self) -> ContextEntry:
        """Auto-set content type to MULTIMODAL when images are present"""
        if self.images:
            self.content_type = ContentType.MULTIMODAL
        return self


class SearchFilters(BaseModel):
    """Search filter parameters with efficient indexing support"""

    model_config = {
        'json_schema_extra': {
            'examples': [
                {
                    'thread_id': 'task_123',
                    'source': 'agent',
                    'limit': 10,
                },
            ],
        },
    }

    thread_id: str | None = Field(default=None, description='Filter by thread')
    source: SourceType | None = Field(default=None, description='Filter by source')
    tags: list[str] | None = Field(default=None, description='Filter by tags (OR logic)')
    start_date: datetime | None = Field(default=None)
    end_date: datetime | None = Field(default=None)
    content_type: ContentType | None = Field(default=None)
    limit: Annotated[int, Field(default=50, le=500, ge=1)]
    offset: Annotated[int, Field(default=0, ge=0)]
    include_images: bool = Field(default=False, description='Include image data in response')


class StoreContextRequest(BaseModel):
    """Request model for storing context"""

    thread_id: str = Field(..., description='Thread ID for context scoping')
    source: SourceType = Field(..., description="Must be 'user' or 'agent'")
    text: str | None = Field(default=None, min_length=1)
    images: list[ImageAttachment] | None = Field(default=None, max_length=10)
    metadata: MetadataDict | None = Field(default=None)
    tags: list[str] | None = Field(default=None)


class DeleteContextRequest(BaseModel):
    """Request model for deleting context"""

    context_ids: list[int] | None = Field(default=None, min_length=1)
    thread_id: str | None = Field(default=None)

    @model_validator(mode='after')
    def validate_has_fields(self) -> DeleteContextRequest:
        """Ensure at least one field is provided for deletion"""
        if not self.context_ids and not self.thread_id:
            raise ValueError('Must provide either context_ids or thread_id')
        return self
