
import base64
import re
import string
from datetime import datetime
from enum import StrEnum
from typing import Annotated
from typing import Any

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from app.types import MetadataDict

# Single source of truth for the per-entry image attachment count limit.
# Enforced in three aligned places: the Pydantic models below (max_length),
# the shared validation chokepoint (validate_and_normalize_images in
# app/tools/_shared.py, covering store_context, update_context, and both
# batch tools), and the tool-boundary Field declarations in
# app/tools/context.py (so the MCP wire schema advertises the bound as
# maxItems). It lives here because app.models imports nothing from app.tools,
# so the tools layer can import it without an import cycle.
MAX_IMAGES_PER_ENTRY = 10

# One leading RFC 2397 data-URI prefix (any 'data:<mediatype>;base64,' form,
# e.g. 'data:image/png;base64,'). Pure base64 never contains ':' or ',', so
# stripping this prefix can never eat real payload characters.
_DATA_URI_BASE64_PREFIX = re.compile(r'^data:[^,]*;base64,', re.IGNORECASE)

# Translate the URL-safe base64 alphabet ('-'/'_') to the standard one
# ('+'/'/') and delete ASCII whitespace in a single pass.
_BASE64_CANONICAL_TABLE = str.maketrans('-_', '+/', string.whitespace)


def normalize_base64_image_data(data: str) -> str:
    """Normalize a base64 image payload to canonical standard-alphabet form.

    Client payloads arrive in several near-base64 shapes that a lenient
    ``base64.b64decode`` silently corrupts instead of rejecting: an RFC 2397
    data URI (whenever the prefix's base64-alphabet character count is a
    multiple of 4, the prefix decodes as garbage bytes prepended to the
    image), the URL-safe alphabet (``-``/``_`` are discarded, shifting every
    following byte), whitespace-wrapped transfer encodings, and payloads with
    stripped ``=`` padding. This helper reduces all of them to one canonical
    form so a subsequent STRICT decode (``validate=True``) either yields
    exactly the intended bytes or fails loudly:

    1. Strip one leading data-URI prefix (any ``data:...;base64,`` form).
    2. Remove ASCII whitespace and translate the URL-safe alphabet to the
       standard alphabet.
    3. Restore ``=`` padding to a multiple of 4.

    Args:
        data: The raw client-supplied base64 payload.

    Returns:
        The canonical standard-alphabet base64 string. Genuinely non-base64
        input is returned in a form that a strict decode rejects loudly.
    """
    normalized = _DATA_URI_BASE64_PREFIX.sub('', data, count=1)
    normalized = normalized.translate(_BASE64_CANONICAL_TABLE)
    remainder = len(normalized) % 4
    if remainder:
        normalized += '=' * (4 - remainder)
    return normalized


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
    # Advisory, client-supplied label stored verbatim -- intentionally NOT an allowlist. Image
    # bytes are opaque to the server (base64-validated and size-capped, never decoded or
    # rendered), so the mime_type is not restricted; escaping/rendering is the consuming
    # client's responsibility. Strict ingest, if ever needed, belongs in content sniffing on the
    # store path, not in a model pattern (the store/update tools take raw dicts and never built
    # this model, so a pattern here only ever constrained tests, never live input).
    mime_type: str = Field(default='image/png')
    metadata: MetadataDict | None = Field(default=None)
    position: Annotated[int, Field(default=0, ge=0)]

    @field_validator('data')
    @classmethod
    def validate_base64(cls, v: str) -> str:
        """Normalize to canonical base64 and reject non-base64 data.

        Mirrors the tool-boundary chokepoint (validate_and_normalize_images in
        app/tools/_shared.py): normalize first, then decode STRICTLY, so a
        data-URI prefix, URL-safe alphabet, or stripped padding is repaired
        instead of silently decoding to corrupted bytes, and genuinely
        non-base64 input fails loudly.

        Returns:
            The canonical standard-alphabet base64 string.

        Raises:
            ValueError: If the normalized payload is not valid base64 or
                decodes to zero bytes.
        """
        normalized = normalize_base64_image_data(v)
        try:
            decoded = base64.b64decode(normalized, validate=True)
        except Exception as e:
            raise ValueError('Invalid base64 encoded data') from e
        if not decoded:
            raise ValueError('Base64 data decodes to zero bytes (not valid image content)')
        return normalized


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

    id: str | None = Field(default=None, description='Auto-generated UUIDv7 hex (32-char lowercase) ID')
    thread_id: str = Field(..., description='Thread identifier for context scoping')
    source: SourceType = Field(..., description='Origin of the context')
    content_type: ContentType = Field(default=ContentType.TEXT)
    text_content: str | None = Field(default=None)
    images: list[Any] = Field(default_factory=list, max_length=MAX_IMAGES_PER_ENTRY)
    metadata: MetadataDict | None = Field(default=None)
    summary: str | None = Field(default=None, description='LLM-generated summary for search result display')
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = Field(default=None)
    updated_at: datetime | None = Field(default=None)

    @field_validator('text_content')
    @classmethod
    def validate_text_content(cls, v: str | None) -> str | None:
        """Validate text content is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError('Text content cannot be empty or contain only whitespace')
        return v

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
    def set_content_type(self) -> 'ContextEntry':
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
    limit: Annotated[int, Field(default=50, le=100, ge=1)]
    offset: Annotated[int, Field(default=0, ge=0)]
    include_images: bool = Field(default=False, description='Include image data in response')


class StoreContextRequest(BaseModel):
    """Request model for storing context"""

    thread_id: str = Field(..., description='Thread ID for context scoping')
    source: SourceType = Field(..., description="Must be 'user' or 'agent'")
    text: str | None = Field(default=None)
    images: list[ImageAttachment] | None = Field(default=None, max_length=MAX_IMAGES_PER_ENTRY)
    metadata: MetadataDict | None = Field(default=None)
    tags: list[str] | None = Field(default=None)

    @field_validator('thread_id')
    @classmethod
    def validate_thread_id(cls, v: str) -> str:
        """Validate thread_id is not empty or whitespace."""
        if not v.strip():
            raise ValueError('thread_id is required and cannot be empty')
        return v

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str | None) -> str | None:
        """Validate text is not empty if provided."""
        if v is not None and not v.strip():
            raise ValueError('text is required and cannot be empty')
        return v


class DeleteContextRequest(BaseModel):
    """Request model for deleting context"""

    context_ids: list[str] | None = Field(default=None)
    thread_id: str | None = Field(default=None)

    @field_validator('context_ids')
    @classmethod
    def validate_context_ids(cls, v: list[str] | None) -> list[str] | None:
        """Validate context_ids is not empty if provided."""
        if v is not None and len(v) == 0:
            raise ValueError('context_ids cannot be an empty list')
        return v

    @model_validator(mode='after')
    def validate_has_fields(self) -> 'DeleteContextRequest':
        """Ensure at least one field is provided for deletion"""
        if not self.context_ids and not self.thread_id:
            raise ValueError('Must provide either context_ids or thread_id')
        return self
