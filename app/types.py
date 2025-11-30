"""Type definitions for the MCP context server.

This module provides type definitions to replace explicit Any usage
and ensure strict type safety throughout the codebase.
"""

from typing import TypedDict

# JSON value types - recursive union for JSON-like data structures
type JsonValue = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']

# Metadata value types - simpler non-recursive type for metadata fields
type MetadataValue = str | int | float | bool | None

# Metadata dictionary type for use in models - supports nested JSON structures
type MetadataDict = dict[str, JsonValue]


# API Response TypedDicts for proper return type annotations
class ImageAttachmentDict(TypedDict):
    """Type definition for image attachment responses."""

    image_id: int
    context_id: int
    mime_type: str
    size_bytes: int


class ContextEntryDict(TypedDict, total=False):
    """Type definition for context entry responses.

    Uses total=False to handle optional fields properly.
    """

    id: int
    thread_id: str
    source: str
    content_type: str
    text_content: str | None
    metadata: MetadataDict | None
    created_at: str
    updated_at: str
    tags: list[str]
    images: list[ImageAttachmentDict] | list[dict[str, str]] | None
    is_truncated: bool | None


class StoreContextSuccessDict(TypedDict):
    """Type definition for successful store context response."""

    success: bool
    context_id: int
    thread_id: str
    message: str


class ThreadInfoDict(TypedDict):
    """Type definition for individual thread info."""

    thread_id: str
    entry_count: int
    source_types: int
    multimodal_count: int
    first_entry: str
    last_entry: str
    last_id: int


class ThreadListDict(TypedDict):
    """Type definition for thread list response."""

    threads: list[ThreadInfoDict]
    total_threads: int


class ImageDict(TypedDict, total=False):
    """Type definition for image data in API responses."""

    data: str
    mime_type: str
    metadata: dict[str, str] | None


class UpdateContextSuccessDict(TypedDict):
    """Type definition for successful update context response."""

    success: bool
    context_id: int
    updated_fields: list[str]
    message: str


# Bulk operation TypedDicts


class BulkStoreItemDict(TypedDict, total=False):
    """Type definition for a single item in bulk store request.

    Required fields: thread_id, source, text
    Optional fields: metadata, tags, images
    """

    thread_id: str
    source: str
    text: str
    metadata: MetadataDict | None
    tags: list[str] | None
    images: list[dict[str, str]] | None


class BulkStoreResultItemDict(TypedDict):
    """Type definition for a single result in bulk store response."""

    index: int
    success: bool
    context_id: int | None
    error: str | None


class BulkStoreResponseDict(TypedDict):
    """Type definition for bulk store response."""

    success: bool
    total: int
    succeeded: int
    failed: int
    results: list[BulkStoreResultItemDict]
    message: str


class BulkUpdateItemDict(TypedDict, total=False):
    """Type definition for a single item in bulk update request.

    Required field: context_id
    Optional fields: text, metadata, metadata_patch, tags, images
    Note: metadata and metadata_patch are mutually exclusive per entry.
    """

    context_id: int
    text: str | None
    metadata: MetadataDict | None
    metadata_patch: MetadataDict | None
    tags: list[str] | None
    images: list[dict[str, str]] | None


class BulkUpdateResultItemDict(TypedDict):
    """Type definition for a single result in bulk update response."""

    index: int
    context_id: int
    success: bool
    updated_fields: list[str] | None
    error: str | None


class BulkUpdateResponseDict(TypedDict):
    """Type definition for bulk update response."""

    success: bool
    total: int
    succeeded: int
    failed: int
    results: list[BulkUpdateResultItemDict]
    message: str


class BulkDeleteResponseDict(TypedDict):
    """Type definition for bulk delete response."""

    success: bool
    deleted_count: int
    criteria_used: list[str]
    message: str
