"""
MCP Context Server implementation using FastMCP.

This server provides persistent multimodal context storage capabilities for LLM agents,
enabling shared memory across different conversation threads with support for text and images.
"""

import base64
import contextlib
import json
import logging
import sqlite3
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

from fastmcp import Context
from fastmcp import FastMCP
from pydantic import Field

from app.db_manager import DatabaseConnectionManager
from app.db_manager import get_connection_manager
from app.logger_config import config_logger
from app.repositories import RepositoryContainer
from app.settings import get_settings
from app.types import ContextEntryDict
from app.types import JsonValue
from app.types import MetadataDict
from app.types import StoreContextErrorDict
from app.types import StoreContextSuccessDict
from app.types import ThreadListDict

# Get setting
settings = get_settings()
# Configure logging
config_logger(settings.log_level)
logger = logging.getLogger(__name__)

# Database configuration
DB_PATH = settings.storage.db_path
MAX_IMAGE_SIZE_MB = settings.storage.max_image_size_mb
MAX_TOTAL_SIZE_MB = settings.storage.max_total_size_mb
SCHEMA_PATH = Path(__file__).parent / 'schema.sql'

# Global connection manager and repositories
_db_manager: DatabaseConnectionManager | None = None
_repositories: RepositoryContainer | None = None


# Lifespan context manager for FastMCP
@asynccontextmanager
async def lifespan(_: FastMCP[None]) -> AsyncGenerator[None, None]:
    """Manage server lifecycle - initialize on startup, cleanup on shutdown.

    This ensures that the database manager's background tasks run in the
    same event loop as FastMCP, preventing the hanging issue.

    Args:
        _: The FastMCP server instance (unused but required by signature)

    Yields:
        None: Control is yielded back to FastMCP during server operation
    """
    global _db_manager, _repositories

    # Startup
    try:
        await _ensure_db_manager()
        # 1) Ensure schema exists using a short-lived manager
        await init_database()
        # 2) Start long-lived manager for server runtime
        global _db_manager, _repositories
        _db_manager = get_connection_manager(DB_PATH)
        await _db_manager.initialize()
        # 3) Initialize repositories
        _repositories = RepositoryContainer(_db_manager)
        logger.info(f'MCP Context Server initialized with database: {DB_PATH}')
    except Exception as e:
        logger.error(f'Failed to initialize server: {e}')
        if _db_manager:
            await _db_manager.shutdown()
        raise

    # Yield control to FastMCP
    yield

    # Shutdown
    logger.info('Shutting down MCP Context Server')
    # At this point, startup succeeded and _db_manager must be set
    assert _db_manager is not None
    try:
        await _db_manager.shutdown()
    except Exception as e:
        logger.error(f'Error during shutdown: {e}')
    finally:
        _db_manager = None
        _repositories = None
    logger.info('MCP Context Server shutdown complete')


# Initialize FastMCP server with lifespan management
mcp = FastMCP(name='mcp-context-server', lifespan=lifespan)


async def init_database() -> None:
    """Initialize database schema only using a short-lived manager.

    This avoids leaving background tasks running when tests call this function directly.
    """
    try:
        await _ensure_db_manager()
        # Ensure database path exists
        if DB_PATH:
            DB_PATH.parent.mkdir(parents=True, exist_ok=True)
            if not DB_PATH.exists():
                DB_PATH.touch()

        # Read schema from file or fallback
        if SCHEMA_PATH.exists():
            schema_sql = SCHEMA_PATH.read_text(encoding='utf-8')
        else:
            logger.warning(f'Schema file not found at {SCHEMA_PATH}, using embedded schema')
            # Fallback to embedded schema
            schema_sql = '''
-- Main context storage table
CREATE TABLE IF NOT EXISTS context_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT NOT NULL,
    source TEXT NOT NULL CHECK(source IN ('user', 'agent')),
    content_type TEXT NOT NULL CHECK(content_type IN ('text', 'multimodal')),
    text_content TEXT,
    metadata JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_thread_id ON context_entries(thread_id);
CREATE INDEX IF NOT EXISTS idx_source ON context_entries(source);
CREATE INDEX IF NOT EXISTS idx_created_at ON context_entries(created_at);
CREATE INDEX IF NOT EXISTS idx_thread_source ON context_entries(thread_id, source);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
    tag TEXT NOT NULL,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_tags_entry ON tags(context_entry_id);
CREATE INDEX IF NOT EXISTS idx_tags_tag ON tags(tag);

CREATE TABLE IF NOT EXISTS image_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    context_entry_id INTEGER NOT NULL,
    image_data BLOB NOT NULL,
    mime_type TEXT NOT NULL,
    image_metadata JSON,
    position INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (context_entry_id) REFERENCES context_entries(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_image_context ON image_attachments(context_entry_id);
            '''

        # Apply schema using a short-lived manager
        temp_manager = get_connection_manager(DB_PATH, force_new=True)
        await temp_manager.initialize()
        try:
            def _init_schema(conn: sqlite3.Connection) -> None:
                # Single executescript to create all objects atomically
                conn.executescript(schema_sql)
            await temp_manager.execute_write(_init_schema)
            logger.info('Database schema initialized successfully')
        finally:
            # Always shutdown to stop background tasks and close connections
            await temp_manager.shutdown()
    except Exception as e:
        logger.error(f'Failed to initialize database: {e}')
        raise


# Utility functions


async def _ensure_db_manager() -> DatabaseConnectionManager:
    """Ensure a connection manager exists and is initialized.

    In tests, FastMCP lifespan isn't running, so tools need a lazy
    initializer to operate directly.

    Returns:
        Initialized `DatabaseConnectionManager` singleton to use for DB ops.
    """
    global _db_manager
    if _db_manager is None:
        manager = get_connection_manager(DB_PATH)
        await manager.initialize()
        _db_manager = manager
    return _db_manager


async def _ensure_repositories() -> RepositoryContainer:
    """Ensure repositories are initialized.

    Returns:
        Initialized repository container.
    """
    global _repositories
    if _repositories is None:
        manager = await _ensure_db_manager()
        _repositories = RepositoryContainer(manager)
    return _repositories


def deserialize_json_param(
    value: JsonValue | None,
) -> JsonValue | None:
    """Deserialize JSON string parameters if needed with enhanced safety checks.

    COMPATIBILITY NOTE: This function works around a known issue where some MCP clients
    (including Claude Code) send complex parameters as JSON strings instead of native
    Python objects. This is documented in multiple GitHub issues:
    - FastMCP #932: JSON Arguments Encapsulated as String Cause Validation Failure
    - Claude Code #5504: JSON objects converted to quoted strings
    - Claude Code #4192: Consecutive parameter calls fail
    - Claude Code #3084: Pydantic model parameters cause validation errors

    Enhanced to handle:
    - Double-encoding issues (JSON within JSON)
    - Single string values that should be treated as tags
    - Edge cases with special characters like forward slashes

    This function can be removed when the upstream issues are resolved.

    Args:
        value: The parameter value which might be a JSON string

    Returns:
        The deserialized value if it was a JSON string, or the original value
    """
    if isinstance(value, str):
        try:
            result = json.loads(value)
            # Check for double-encoding (JSON string within JSON)
            if isinstance(result, str):
                with contextlib.suppress(json.JSONDecodeError, ValueError):
                    # Try to decode again in case of double-encoding
                    result = json.loads(result)
            return cast(JsonValue | None, result)
        except (json.JSONDecodeError, ValueError):
            # Not valid JSON - check if it's meant to be a single tag
            if value.strip():
                # For tags parameter, a single string should become a list
                # This helps handle edge cases where a single tag is passed as string
                # The caller will need to handle this appropriately
                pass
            return value
    return value


def truncate_text(text: str | None, max_length: int = 150) -> tuple[str | None, bool]:
    """
    Truncate text at word boundary when possible.

    Args:
        text: The text to truncate
        max_length: Maximum character length (default: 150)

    Returns:
        tuple: (truncated_text, is_truncated)
    """
    if not text or len(text) <= max_length:
        return text, False

    # Try to truncate at word boundary
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')

    if last_space > max_length * 0.7:  # Only use word boundary if it's not too short
        truncated = truncated[:last_space]

    return truncated + '...', True


# MCP Tools


@mcp.tool()
async def store_context(
    thread_id: Annotated[str, Field(description='Unique identifier for the conversation/task thread')],
    source: Annotated[Literal['user', 'agent'], Field(description="Either 'user' or 'agent'")],
    text: Annotated[str, Field(description='Text content to store', min_length=1)],
    images: Annotated[list[dict[str, str]] | None, Field(description='List of base64 encoded images with mime_type')] = None,
    metadata: Annotated[MetadataDict | None, Field(description='Additional structured data')] = None,
    tags: Annotated[list[str] | None, Field(description='List of tags (will be normalized and stored separately)')] = None,
    ctx: Context | None = None,
) -> StoreContextSuccessDict | StoreContextErrorDict:
    """
    Store a context entry with optional images.

    Thread_id is critical for context scoping - all agents working on the same task
    should use the same thread_id to share context.

    Args:
        thread_id: Unique identifier for the conversation/task thread
        source: Either 'user' or 'agent'
        text: Text content
        images: List of base64 encoded images with mime_type
        metadata: Additional structured data
        tags: List of tags (will be normalized and stored separately)
        ctx: FastMCP context object

    Returns:
        dict: Success status with context_id if successful, error message if failed.

    Raises:
        ValueError: If context insertion fails.
    """
    try:
        # Log info if context is available
        if ctx:
            await ctx.info(f'Storing context for thread: {thread_id}')

        # Deserialize JSON parameters if needed
        images_raw = deserialize_json_param(cast(JsonValue | None, images))
        images = cast(list[dict[str, str]] | None, images_raw)
        tags_raw = deserialize_json_param(cast(JsonValue | None, tags))
        tags = cast(list[str] | None, tags_raw)
        metadata_raw = deserialize_json_param(cast(JsonValue | None, metadata))
        metadata = cast(MetadataDict | None, metadata_raw)

        # Validate input - text is required
        if not text:
            text_error_response: StoreContextErrorDict = {
                'success': False,
                'error': 'Text content is required',
            }
            return text_error_response

        # Validate source
        if source not in ['user', 'agent']:
            source_error_response: StoreContextErrorDict = {
                'success': False,
                'error': "Source must be 'user' or 'agent'",
            }
            return source_error_response

        # Get repositories
        repos = await _ensure_repositories()

        # Determine content type
        content_type = 'multimodal' if images else 'text'

        # Store context entry with deduplication
        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            text_content=text,
            metadata=json.dumps(metadata) if metadata else None,
        )

        # Ensure we got a valid ID (not None or 0)
        if not context_id:
            raise ValueError('Failed to store context')

        # Store normalized tags
        if tags:
            await repos.tags.store_tags(context_id, tags)

        # Store images if provided
        total_size: float = 0.0
        if images:
            # Validate image sizes first
            for idx, img in enumerate(images):
                img_data_str = img.get('data', '')
                if not img_data_str:
                    logger.warning(f'Image {idx} has no data, skipping')
                    continue

                try:
                    image_binary = base64.b64decode(img_data_str)
                    image_size_mb = len(image_binary) / (1024 * 1024)

                    if image_size_mb > MAX_IMAGE_SIZE_MB:
                        image_size_error_response: StoreContextErrorDict = {
                            'success': False,
                            'error': f'Image {idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit',
                        }
                        return image_size_error_response

                    total_size += image_size_mb
                    if total_size > MAX_TOTAL_SIZE_MB:
                        total_size_error_response: StoreContextErrorDict = {
                            'success': False,
                            'error': f'Total size exceeds {MAX_TOTAL_SIZE_MB}MB limit',
                        }
                        return total_size_error_response
                except Exception as e:
                    image_processing_error_response: StoreContextErrorDict = {
                        'success': False,
                        'error': f'Failed to process image {idx}: {str(e)}',
                    }
                    return image_processing_error_response

            # All validations passed, store the images
            try:
                await repos.images.store_images(context_id, images)
            except Exception as e:
                image_storage_error_response: StoreContextErrorDict = {
                    'success': False,
                    'error': f'Failed to store images: {str(e)}',
                }
                return image_storage_error_response

        action = 'updated' if was_updated else 'stored'
        logger.info(f'{action.capitalize()} context {context_id} in thread {thread_id}')

        success_response: StoreContextSuccessDict = {
            'success': True,
            'context_id': context_id,
            'thread_id': thread_id,
            'message': f'Context {action} with {len(images) if images else 0} images',
        }
        return success_response
    except Exception as e:
        logger.error(f'Error storing context: {e}')
        storage_error_response: StoreContextErrorDict = {
            'success': False,
            'error': f'Failed to store context: {str(e)}',
        }
        return storage_error_response


@mcp.tool()
async def search_context(
    thread_id: Annotated[str | None, Field(description='Filter by thread (uses index)')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Filter by source type (uses index)')] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags')] = None,
    content_type: Annotated[Literal['text', 'multimodal'] | None, Field(description='Filter by content type')] = None,
    limit: Annotated[int, Field(description='Maximum results (max 500)', ge=1, le=500)] = 50,
    offset: Annotated[int, Field(description='Pagination offset', ge=0)] = 0,
    include_images: Annotated[bool, Field(description='Whether to include image data')] = False,
    ctx: Context | None = None,
) -> list[ContextEntryDict]:
    """
    Search context entries with efficient filtering.

    Uses database indexes for optimal performance on thread_id and source filters.
    Tag filtering uses OR logic (matches any of the provided tags).

    Args:
        thread_id: Filter by thread (uses index)
        source: Filter by source type (uses index)
        tags: Filter by any of these tags
        content_type: Filter by content type
        limit: Maximum results (max 500)
        offset: Pagination offset
        include_images: Whether to include image data
        ctx: FastMCP context object

    Returns:
        list: List of matching context entries with their metadata.
    """
    try:
        if ctx:
            await ctx.info(f'Searching context with filters: thread_id={thread_id}, source={source}')

        # Validate limit
        limit = min(limit, 500)

        # Get repositories
        repos = await _ensure_repositories()

        # Search for context entries using repository
        rows = await repos.context.search_contexts(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            limit=limit,
            offset=offset,
        )
        entries: list[ContextEntryDict] = []

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))

            # Parse JSON metadata - database stores as JSON string
            metadata_raw = entry.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    entry['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    entry['metadata'] = None

            # Get normalized tags
            entry_id = int(entry.get('id', 0))
            tags_result = await repos.tags.get_tags_for_context(entry_id)
            entry['tags'] = tags_result

            # Apply text truncation for search_context
            text_content = entry.get('text_content', '')
            truncated_text, is_truncated = truncate_text(text_content)
            entry['text_content'] = truncated_text
            entry['is_truncated'] = is_truncated

            # Fetch images if requested and applicable
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id = int(entry.get('id', 0))
                images_result = await repos.images.get_images_for_context(entry_id, include_data=True)
                entry['images'] = cast(list[dict[str, str]], images_result)

            entries.append(entry)

        return entries
    except Exception as e:
        logger.error(f'Error searching context: {e}')
        return []


@mcp.tool()
async def get_context_by_ids(
    context_ids: Annotated[list[int], Field(description='List of context entry IDs to retrieve', min_length=1)],
    include_images: Annotated[bool, Field(description='Whether to include image data')] = True,
    ctx: Context | None = None,
) -> list[ContextEntryDict]:
    """
    Fetch specific context entries by their IDs.

    Useful when agents need to reference specific pieces of context.
    Always includes full content for completeness.

    Args:
        context_ids: List of context entry IDs
        include_images: Whether to include image data
        ctx: FastMCP context object

    Returns:
        list: List of context entries with full content.
    """
    try:
        if not context_ids:
            return []

        if ctx:
            await ctx.info(f'Fetching context entries: {context_ids}')

        # Get repositories
        repos = await _ensure_repositories()

        # Fetch context entries using repository
        rows = await repos.context.get_by_ids(context_ids)
        entries: list[ContextEntryDict] = []

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))

            # Parse JSON metadata - database stores as JSON string
            metadata_raw = entry.get('metadata')
            # Database can return string that needs parsing
            # Using hasattr to check for string-like object avoids unreachable code warning
            if metadata_raw is not None and hasattr(metadata_raw, 'strip'):  # String-like object from DB
                try:
                    entry['metadata'] = json.loads(str(metadata_raw))
                except (json.JSONDecodeError, ValueError, AttributeError):
                    entry['metadata'] = None

            # Get normalized tags
            entry_id = int(entry.get('id', 0))
            tags_result = await repos.tags.get_tags_for_context(entry_id)
            entry['tags'] = tags_result

            # Fetch images
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id = int(entry.get('id', 0))
                images_result = await repos.images.get_images_for_context(entry_id, include_data=True)
                entry['images'] = cast(list[dict[str, str]], images_result)

            entries.append(entry)

        return entries
    except Exception as e:
        logger.error(f'Error fetching context by IDs: {e}')
        return []


@mcp.tool()
async def delete_context(
    context_ids: Annotated[list[int] | None, Field(description='Specific context entry IDs to delete')] = None,
    thread_id: Annotated[str | None, Field(description='Delete all entries in a thread')] = None,
    ctx: Context | None = None,
) -> dict[str, bool | int | str]:
    """
    Delete context entries by IDs or thread.

    Cascading deletes ensure tags and images are also removed.
    Use with caution as this operation cannot be undone.

    Args:
        context_ids: Specific IDs to delete
        thread_id: Delete all entries in a thread
        ctx: FastMCP context object

    Returns:
        dict: Success status with deletion count or error message.
    """
    try:
        if not context_ids and not thread_id:
            return {
                'success': False,
                'error': 'Must provide either context_ids or thread_id',
            }

        if ctx:
            await ctx.info(f'Deleting context: ids={context_ids}, thread={thread_id}')

        # Get repositories
        repos = await _ensure_repositories()

        deleted = 0

        if context_ids:
            deleted = await repos.context.delete_by_ids(context_ids)
            logger.info(f'Deleted {deleted} context entries by IDs')

        elif thread_id:
            deleted = await repos.context.delete_by_thread(thread_id)
            logger.info(f'Deleted {deleted} entries from thread {thread_id}')

        return {
            'success': True,
            'deleted_count': deleted,
            'message': f'Successfully deleted {deleted} context entries',
        }
    except Exception as e:
        logger.error(f'Error deleting context: {e}')
        return {
            'success': False,
            'error': f'Failed to delete context: {str(e)}',
        }


# MCP Resources for read-only access


@mcp.tool()
async def list_threads(ctx: Context | None = None) -> ThreadListDict:
    """
    List all active threads with statistics.
    Read-only resource for thread discovery.

    Returns:
        dict: Dictionary containing list of threads and total count.
    """
    try:
        if ctx:
            await ctx.info('Listing all threads')

        # Get repositories
        repos = await _ensure_repositories()

        # Use statistics repository to get thread list
        threads = await repos.statistics.get_thread_list()

        return {
            'threads': threads,
            'total_threads': len(threads),
        }
    except Exception as e:
        logger.error(f'Error listing threads: {e}')
        return {'threads': [], 'total_threads': 0}


@mcp.tool()
async def get_statistics(ctx: Context | None = None) -> dict[str, Any]:
    """
    Database statistics and usage metrics.
    Useful for monitoring and debugging.

    Returns:
        dict: Database statistics including counts and size metrics.
    """
    try:
        if ctx:
            await ctx.info('Getting database statistics')

        # Get repositories
        repos = await _ensure_repositories()

        # Use statistics repository to get database stats
        stats = await repos.statistics.get_database_statistics(DB_PATH)

        # Ensure db_manager for metrics
        manager = await _ensure_db_manager()

        # Add connection manager metrics for monitoring
        stats['connection_metrics'] = manager.get_metrics()

        return stats
    except Exception as e:
        logger.error(f'Error getting statistics: {e}')
        return {}


# Main entry point
def main() -> None:
    """Main entry point for the MCP Context Server.

    Simply runs the FastMCP server. Initialization and shutdown
    are handled by the @mcp.startup and @mcp.shutdown decorators.
    """
    try:
        # Run the FastMCP server - this manages its own event loop
        # and will call our startup/shutdown hooks
        mcp.run()
    except KeyboardInterrupt:
        logger.info('Server shutdown requested')
    except Exception as e:
        logger.error(f'Server error: {e}')
        raise


if __name__ == '__main__':
    main()
