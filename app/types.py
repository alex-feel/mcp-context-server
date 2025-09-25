"""Type definitions for the MCP context server.

This module provides type definitions to replace explicit Any usage
and ensure strict type safety throughout the codebase.
"""

from typing import TypedDict

# JSON value types - recursive union for JSON-like data structures
type JsonValue = str | int | float | bool | None | list['JsonValue'] | dict[str, 'JsonValue']

# Metadata value types - simpler non-recursive type for metadata fields
type MetadataValue = str | int | float | bool | None

# Metadata dictionary type for use in models
type MetadataDict = dict[str, MetadataValue]


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


class StoreContextErrorDict(TypedDict):
    """Type definition for error store context response."""

    success: bool
    error: str


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
