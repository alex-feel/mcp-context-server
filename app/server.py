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
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.db_manager import DatabaseConnectionManager
from app.db_manager import get_connection_manager
from app.logger_config import config_logger
from app.repositories import RepositoryContainer
from app.settings import get_settings
from app.types import ContextEntryDict
from app.types import JsonValue
from app.types import MetadataDict
from app.types import StoreContextSuccessDict
from app.types import ThreadListDict
from app.types import UpdateContextSuccessDict

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

# Global embedding service (only if semantic search enabled and available)
_embedding_service: Any | None = None


# Dependency check functions for semantic search
async def check_semantic_search_dependencies() -> bool:
    """Check all semantic search dependencies.

    Performs comprehensive checks for:
    - Python packages (ollama, numpy, sqlite_vec)
    - Ollama service availability
    - EmbeddingGemma model availability
    - sqlite-vec extension loading

    Returns:
        True if all dependencies are available, False otherwise
    """
    logger.info('Checking semantic search dependencies...')

    # Check ollama package
    try:
        import ollama

        logger.debug('✓ ollama package available')
    except ImportError as e:
        logger.warning(f'✗ ollama package not available: {e}')
        return False

    # Check numpy package
    try:
        import importlib.util

        if importlib.util.find_spec('numpy') is None:
            logger.warning('✗ numpy package not available')
            return False
        logger.debug('✓ numpy package available')
    except ImportError as e:
        logger.warning(f'✗ numpy package not available: {e}')
        return False

    # Check sqlite_vec package
    try:
        import sqlite_vec

        logger.debug('✓ sqlite_vec package available')
    except ImportError as e:
        logger.warning(f'✗ sqlite_vec package not available: {e}')
        return False

    # Check Ollama service
    try:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(settings.ollama_host, timeout=1.0)
            if response.status_code == 200:
                logger.debug('✓ Ollama service running')
            else:
                logger.warning(f'✗ Ollama service returned status {response.status_code}')
                return False
    except Exception as e:
        logger.warning(f'✗ Ollama service not accessible: {e}')
        return False

    # Check EmbeddingGemma model
    try:
        ollama_client = ollama.Client(host=settings.ollama_host)
        ollama_client.show(settings.embedding_model)
        logger.debug(f'✓ EmbeddingGemma model "{settings.embedding_model}" available')
    except Exception as e:
        logger.warning(f'✗ EmbeddingGemma model not available: {e}')
        logger.warning(f'  Run: ollama pull {settings.embedding_model}')
        return False

    # Check sqlite-vec extension loading
    try:
        import sqlite3

        test_conn = sqlite3.connect(':memory:')
        test_conn.enable_load_extension(True)
        sqlite_vec.load(test_conn)
        test_conn.enable_load_extension(False)
        test_conn.close()
        logger.debug('✓ sqlite-vec extension loads successfully')
    except Exception as e:
        logger.warning(f'✗ sqlite-vec extension failed to load: {e}')
        return False

    logger.info('✓ All semantic search dependencies available')
    return True


async def apply_semantic_search_migration() -> None:
    """Apply semantic search migration if enabled.

    This function:
    1. Checks if vector table already exists with embeddings
    2. Validates dimension compatibility (existing vs configured)
    3. Templates the migration SQL with configured embedding dimension
    4. Applies the migration if safe to proceed

    Raises:
        RuntimeError: If migration fails or dimension mismatch detected
    """
    if not settings.enable_semantic_search:
        return

    migration_path = Path(__file__).parent / 'migrations' / 'add_semantic_search.sql'

    if not migration_path.exists():
        logger.warning(f'Semantic search migration file not found: {migration_path}')
        return

    try:
        # Read migration SQL template
        migration_sql_template = migration_path.read_text(encoding='utf-8')

        # Apply migration using a short-lived manager
        temp_manager = get_connection_manager(DB_PATH, force_new=True)
        await temp_manager.initialize()
        try:

            # Check for existing table and dimension compatibility
            def _check_existing_dimension(conn: sqlite3.Connection) -> tuple[bool, int | None]:
                # Check if vector table exists
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='vec_context_embeddings'",
                )
                table_exists = cursor.fetchone() is not None

                if not table_exists:
                    return False, None

                # Get existing dimension from any embedding metadata
                cursor = conn.execute('SELECT dimensions FROM embedding_metadata LIMIT 1')
                row = cursor.fetchone()
                existing_dim = row[0] if row else None

                return True, existing_dim

            table_exists, existing_dim = await temp_manager.execute_read(_check_existing_dimension)

            # Validate dimension compatibility
            if table_exists and existing_dim is not None and existing_dim != settings.embedding_dim:
                db_path = str(DB_PATH).replace('\\', '/')
                raise RuntimeError(
                    f'Embedding dimension mismatch detected!\n'
                    f'  Existing database dimension: {existing_dim}\n'
                    f'  Configured EMBEDDING_DIM: {settings.embedding_dim}\n\n'
                    f'To change embedding dimensions, you must:\n'
                    f'  1. Back up your database: {db_path}\n'
                    f'  2. Delete or rename the database file\n'
                    f'  3. Restart the server to create new tables with dimension {settings.embedding_dim}\n'
                    f'  4. Re-import your context data (embeddings will be regenerated)\n\n'
                    f'Note: Changing dimensions will lose all existing embeddings.',
                )

            # Template the migration SQL with configured dimension
            migration_sql = migration_sql_template.replace(
                '{EMBEDDING_DIM}', str(settings.embedding_dim),
            )

            # Apply migration
            def _apply_migration(conn: sqlite3.Connection) -> None:
                conn.executescript(migration_sql)

            await temp_manager.execute_write(_apply_migration)

            if existing_dim is None:
                logger.info(
                    f'Semantic search migration applied successfully with dimension: {settings.embedding_dim}',
                )
            else:
                logger.info('Semantic search migration: tables already exist, skipping')

        finally:
            await temp_manager.shutdown()
            await temp_manager.wait_for_shutdown_complete(
                timeout_seconds=get_settings().storage.shutdown_timeout_test_s,
            )
    except Exception as e:
        logger.error(f'Failed to apply semantic search migration: {e}')
        raise RuntimeError(f'Semantic search migration failed: {e}') from e


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
    global _db_manager, _repositories, _embedding_service

    # Startup
    try:
        await _ensure_db_manager()
        # 1) Ensure schema exists using a short-lived manager
        await init_database()
        # 2) Apply semantic search migration if enabled
        await apply_semantic_search_migration()
        # 3) Start long-lived manager for server runtime
        _db_manager = get_connection_manager(DB_PATH)
        await _db_manager.initialize()
        # 4) Initialize repositories
        _repositories = RepositoryContainer(_db_manager)

        # 5) Initialize semantic search if enabled
        if settings.enable_semantic_search:
            semantic_available = await check_semantic_search_dependencies()

            if semantic_available:
                try:
                    from app.services.embedding_service import EmbeddingService

                    _embedding_service = EmbeddingService()
                    logger.info('✓ Semantic search enabled and available')

                    # Conditionally register semantic_search_tool
                    # Apply @mcp.tool() decorator programmatically to convert function to Tool object
                    mcp.add_tool(mcp.tool()(semantic_search_tool))
                    logger.info('✓ semantic_search_tool registered and exposed')
                except Exception as e:
                    logger.error(f'Failed to initialize embedding service: {e}')
                    _embedding_service = None
                    logger.warning('⚠ Semantic search enabled but initialization failed - feature disabled')
                    logger.info('⚠ semantic_search_tool not registered (initialization failed)')
            else:
                _embedding_service = None
                logger.warning('⚠ Semantic search enabled but dependencies not met - feature disabled')
                logger.warning('  Install dependencies: uv sync --extra semantic-search')
                logger.warning(f'  Download model: ollama pull {settings.embedding_model}')
                logger.info('⚠ semantic_search_tool not registered (dependencies not met)')
        else:
            _embedding_service = None
            logger.info('Semantic search disabled (ENABLE_SEMANTIC_SEARCH=false)')
            logger.info('⚠ semantic_search_tool not registered (feature disabled)')

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
        _embedding_service = None
    logger.info('MCP Context Server shutdown complete')


# Initialize FastMCP server with lifespan management
# mask_error_details=False exposes validation errors for LLM autocorrection
mcp = FastMCP(name='mcp-context-server', lifespan=lifespan, mask_error_details=False)


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

-- Functional indexes for common metadata patterns (improves metadata filtering performance)
-- These indexes extract specific JSON fields for faster querying

-- Status-based filtering (most common use case)
CREATE INDEX IF NOT EXISTS idx_metadata_status
ON context_entries(json_extract(metadata, '$.status'))
WHERE json_extract(metadata, '$.status') IS NOT NULL;

-- Priority-based filtering (numeric comparisons)
CREATE INDEX IF NOT EXISTS idx_metadata_priority
ON context_entries(json_extract(metadata, '$.priority'))
WHERE json_extract(metadata, '$.priority') IS NOT NULL;

-- Agent name filtering (identify specific agents)
CREATE INDEX IF NOT EXISTS idx_metadata_agent_name
ON context_entries(json_extract(metadata, '$.agent_name'))
WHERE json_extract(metadata, '$.agent_name') IS NOT NULL;

-- Task name filtering (search by task title/name)
CREATE INDEX IF NOT EXISTS idx_metadata_task_name
ON context_entries(json_extract(metadata, '$.task_name'))
WHERE json_extract(metadata, '$.task_name') IS NOT NULL;

-- Composite indexes for common filter combinations
CREATE INDEX IF NOT EXISTS idx_thread_metadata_status
ON context_entries(thread_id, json_extract(metadata, '$.status'));

CREATE INDEX IF NOT EXISTS idx_thread_metadata_priority
ON context_entries(thread_id, json_extract(metadata, '$.priority'));

-- Boolean flag indexes
CREATE INDEX IF NOT EXISTS idx_metadata_completed
ON context_entries(json_extract(metadata, '$.completed'))
WHERE json_extract(metadata, '$.completed') IS NOT NULL;
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
            # Wait for shutdown to complete with timeout to prevent race conditions
            # This ensures background tasks are fully stopped before proceeding
            await temp_manager.wait_for_shutdown_complete(
                timeout_seconds=get_settings().storage.shutdown_timeout_test_s,
            )
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
    thread_id: Annotated[str, Field(min_length=1, description='Unique identifier for the conversation/task thread')],
    source: Annotated[Literal['user', 'agent'], Field(description="Either 'user' or 'agent'")],
    text: Annotated[str, Field(min_length=1, description='Text content to store')],
    images: Annotated[list[dict[str, str]] | None, Field(description='List of base64 encoded images with mime_type')] = None,
    metadata: Annotated[
        MetadataDict | None,
        Field(
            description='Additional structured data. For optimal performance, consider using indexed field names: '
            'status (state information), priority (numeric value for range queries), '
            'agent_name (specific agent identifier), task_name (task title for string searches), '
            'completed (boolean flag for completion state). '
            'These fields are indexed for faster filtering but not required.',
        ),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='List of tags (will be normalized and stored separately)')] = None,
    ctx: Context | None = None,
) -> StoreContextSuccessDict:
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
        dict: Success status with context_id if successful.

    Raises:
        ToolError: If validation fails or context insertion fails.
    """
    try:
        # Clean input strings - defensive try/except handles edge cases where Pydantic validation bypassed
        try:
            thread_id = thread_id.strip()
        except AttributeError:
            raise ToolError('thread_id is required') from None
        try:
            text = text.strip()
        except AttributeError:
            raise ToolError('text is required') from None

        # Business logic: empty strings after stripping are not allowed
        if not thread_id:
            raise ToolError('thread_id cannot be empty or whitespace')
        if not text:
            raise ToolError('text cannot be empty or whitespace')

        # Validate images if provided
        if images:
            for idx, img in enumerate(images):
                if 'data' in img:
                    try:
                        base64.b64decode(img['data'])
                    except Exception:
                        raise ToolError(f'Invalid base64 encoded data in image {idx}') from None

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
            metadata=json.dumps(metadata, ensure_ascii=False) if metadata else None,
        )

        # Ensure we got a valid ID (not None or 0)
        if not context_id:
            raise ToolError('Failed to store context')

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
                except Exception:
                    raise ToolError(f'Invalid base64 encoded data in image {idx}') from None

                image_size_mb = len(image_binary) / (1024 * 1024)

                if image_size_mb > MAX_IMAGE_SIZE_MB:
                    raise ToolError(f'Image {idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit')

                total_size += image_size_mb
                if total_size > MAX_TOTAL_SIZE_MB:
                    raise ToolError(f'Total size exceeds {MAX_TOTAL_SIZE_MB}MB limit')

            # All validations passed, store the images
            try:
                await repos.images.store_images(context_id, images)
            except Exception as e:
                raise ToolError(f'Failed to store images: {str(e)}') from e

        # Generate embedding if semantic search is available (non-blocking)
        embedding_generated = False
        if _embedding_service is not None:
            try:
                embedding = await _embedding_service.generate_embedding(text)
                await repos.embeddings.store(
                    context_id=context_id,
                    embedding=embedding,
                    model=settings.embedding_model,
                )
                embedding_generated = True
                logger.debug(f'Generated embedding for context {context_id}')
            except Exception as e:
                logger.warning(f'Failed to generate/store embedding for context {context_id}: {e}')
                # Non-blocking: continue even if embedding fails

        action = 'updated' if was_updated else 'stored'
        logger.info(f'{action.capitalize()} context {context_id} in thread {thread_id}')

        return StoreContextSuccessDict(
            success=True,
            context_id=context_id,
            thread_id=thread_id,
            message=f'Context {action} with {len(images) if images else 0} images'
            + (' (embedding generated)' if embedding_generated else ''),
        )
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error storing context: {e}')
        raise ToolError(f'Failed to store context: {str(e)}') from e


@mcp.tool()
async def search_context(
    thread_id: Annotated[str | None, Field(min_length=1, description='Filter by thread (uses index)')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Filter by source type (uses index)')] = None,
    tags: Annotated[list[str] | None, Field(description='Filter by any of these tags')] = None,
    content_type: Annotated[Literal['text', 'multimodal'] | None, Field(description='Filter by content type')] = None,
    metadata: Annotated[
        dict[str, str | int | float | bool] | None,
        Field(description='Simple metadata filters (key=value equality)'),
    ] = None,
    metadata_filters: Annotated[
        list[dict[str, Any]] | None,
        Field(description='Advanced metadata filters with operators [{"key": "priority", "operator": "gt", "value": 5}]'),
    ] = None,
    limit: Annotated[int, Field(ge=1, le=500, description='Maximum results (max 500)')] = 50,
    offset: Annotated[int, Field(ge=0, description='Pagination offset')] = 0,
    include_images: Annotated[bool, Field(description='Whether to include image data')] = False,
    explain_query: Annotated[bool, Field(description='Include query execution statistics')] = False,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Search context entries with efficient filtering including metadata.

    Uses database indexes for optimal performance on thread_id and source filters.
    Tag filtering uses OR logic (matches any of the provided tags).
    Supports both simple metadata filtering (key=value) and advanced filtering with operators.

    Args:
        thread_id: Filter by thread (uses index)
        source: Filter by source type (uses index)
        tags: Filter by any of these tags
        content_type: Filter by content type
        metadata: Simple metadata filters (key=value equality)
        metadata_filters: Advanced metadata filters with operators
        limit: Maximum results (max 500)
        offset: Pagination offset
        include_images: Whether to include image data
        explain_query: Include query execution statistics
        ctx: FastMCP context object

    Returns:
        dict: Contains 'entries' list and optional 'stats' if explain_query is True.

    Raises:
        ToolError: If validation fails or search operation fails.
    """
    try:
        if ctx:
            await ctx.info(f'Searching context with filters: thread_id={thread_id}, source={source}')

        # Get repositories
        repos = await _ensure_repositories()

        # Use the improved search_contexts method that now supports metadata
        result = await repos.context.search_contexts(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            metadata=metadata,
            metadata_filters=metadata_filters,
            limit=limit,
            offset=offset,
            explain_query=explain_query,
        )

        # Always expect tuple from repository
        rows, stats = result

        # Check for validation errors in stats
        if 'error' in stats:
            # Return the error response with validation details
            error_response: dict[str, Any] = {
                'entries': [],
                'error': stats.get('error', 'Unknown error'),
            }
            if 'validation_errors' in stats:
                error_response['validation_errors'] = stats['validation_errors']
            return error_response

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
            entry_id_raw = entry.get('id')
            if entry_id_raw is not None:
                entry_id = int(entry_id_raw)
                tags_result = await repos.tags.get_tags_for_context(entry_id)
                entry['tags'] = tags_result
            else:
                entry['tags'] = []

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

        # Always return dict with entries and stats
        response: dict[str, Any] = {'entries': entries, 'stats': stats}
        return response
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error searching context: {e}')
        raise ToolError(f'Failed to search context: {str(e)}') from e


@mcp.tool()
async def get_context_by_ids(
    context_ids: Annotated[list[int], Field(min_length=1, description='List of context entry IDs to retrieve')],
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

    Raises:
        ToolError: If fetching entries fails.
    """
    try:
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
            entry_id_raw = entry.get('id')
            if entry_id_raw is not None:
                entry_id = int(entry_id_raw)
                tags_result = await repos.tags.get_tags_for_context(entry_id)
                entry['tags'] = tags_result
            else:
                entry['tags'] = []

            # Fetch images
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id_img = entry.get('id')
                if entry_id_img is not None:
                    images_result = await repos.images.get_images_for_context(int(entry_id_img), include_data=True)
                    entry['images'] = cast(list[dict[str, str]], images_result)
                else:
                    entry['images'] = []

            entries.append(entry)

        return entries
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error fetching context by IDs: {e}')
        raise ToolError(f'Failed to fetch context entries: {str(e)}') from e


@mcp.tool()
async def delete_context(
    context_ids: Annotated[list[int] | None, Field(min_length=1, description='Specific context entry IDs to delete')] = None,
    thread_id: Annotated[str | None, Field(min_length=1, description='Delete all entries in a thread')] = None,
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

    Raises:
        ToolError: If validation fails or deletion fails.
    """
    try:
        # Ensure at least one parameter is provided (business logic validation)
        if not context_ids and not thread_id:
            raise ToolError('Must provide either context_ids or thread_id')

        if ctx:
            await ctx.info(f'Deleting context: ids={context_ids}, thread={thread_id}')

        # Get repositories
        repos = await _ensure_repositories()

        deleted = 0

        if context_ids:
            # Delete embeddings first (explicit cleanup)
            if settings.enable_semantic_search:
                for context_id in context_ids:
                    try:
                        await repos.embeddings.delete(context_id)
                    except Exception as e:
                        logger.warning(f'Failed to delete embedding for context {context_id}: {e}')
                        # Non-blocking: continue even if embedding deletion fails

            deleted = await repos.context.delete_by_ids(context_ids)
            logger.info(f'Deleted {deleted} context entries by IDs')

        elif thread_id:
            # Get all context IDs in thread for embedding cleanup
            if settings.enable_semantic_search:
                try:
                    # Get all context IDs in this thread
                    results = await repos.context.search_contexts(
                        thread_id=thread_id,
                        limit=10000,  # Large limit to get all
                        offset=0,
                        explain_query=False,
                    )
                    rows, _ = results

                    # Delete embeddings for all contexts in thread
                    for row in rows:
                        context_id = row['id']  # sqlite3.Row supports __getitem__
                        if context_id:
                            try:
                                await repos.embeddings.delete(int(context_id))
                            except Exception as e:
                                logger.warning(f'Failed to delete embedding for context {context_id}: {e}')
                except Exception as e:
                    logger.warning(f'Failed to cleanup embeddings for thread {thread_id}: {e}')
                    # Non-blocking: continue with context deletion

            deleted = await repos.context.delete_by_thread(thread_id)
            logger.info(f'Deleted {deleted} entries from thread {thread_id}')

        return {
            'success': True,
            'deleted_count': deleted,
            'message': f'Successfully deleted {deleted} context entries',
        }
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error deleting context: {e}')
        raise ToolError(f'Failed to delete context: {str(e)}') from e


@mcp.tool()
async def update_context(
    context_id: Annotated[int, Field(gt=0, description='ID of the context entry to update')],
    text: Annotated[str | None, Field(min_length=1, description='New text content (replaces existing)')] = None,
    metadata: Annotated[MetadataDict | None, Field(description='New metadata object (replaces existing)')] = None,
    tags: Annotated[list[str] | None, Field(description='New tags list (replaces all existing tags)')] = None,
    images: Annotated[
        list[dict[str, str]] | None,
        Field(description='New images list with base64 data and mime_type (replaces all existing images)'),
    ] = None,
    ctx: Context | None = None,
) -> UpdateContextSuccessDict:
    """
    Update an existing context entry with selective field updates.

    Allows updating text content, metadata, tags, and images while preserving
    immutable fields like ID, thread_id, source, and created_at. The content_type
    field is automatically recalculated based on the presence of images.

    Args:
        context_id: ID of the context entry to update
        text: New text content (replaces existing if provided)
        metadata: New metadata object (replaces existing if provided)
        tags: New tags list (replaces all existing tags if provided)
        images: New images list (replaces all existing images if provided)
        ctx: FastMCP context object

    Returns:
        UpdateContextSuccessDict on success.

    Raises:
        ToolError: If validation fails or update operation fails.
    """
    try:
        # Clean text input if provided
        if text is not None:
            text = text.strip()
            # Business logic: if text provided, it cannot be empty after stripping
            if not text:
                raise ToolError('text cannot be empty or contain only whitespace')

        # Validate that at least one field is provided for update
        if text is None and metadata is None and tags is None and images is None:
            raise ToolError('At least one field must be provided for update')

        if ctx:
            await ctx.info(f'Updating context entry {context_id}')

        # Get repositories
        repos = await _ensure_repositories()

        # Check if entry exists
        exists = await repos.context.check_entry_exists(context_id)
        if not exists:
            raise ToolError(f'Context entry with ID {context_id} not found')

        updated_fields: list[str] = []

        # Start transaction-like operations
        try:
            # Update text content and/or metadata if provided
            if text is not None or metadata is not None:
                # Prepare metadata JSON string if provided
                metadata_str: str | None = None
                if metadata is not None:
                    metadata_str = json.dumps(metadata, ensure_ascii=False)

                # Update context entry
                success, fields = await repos.context.update_context_entry(
                    context_id=context_id,
                    text_content=text,
                    metadata=metadata_str,
                )

                if not success:
                    raise ToolError('Failed to update context entry')

                updated_fields.extend(fields)

            # Replace tags if provided
            if tags is not None:
                await repos.tags.replace_tags_for_context(context_id, tags)
                updated_fields.append('tags')
                logger.debug(f'Replaced tags for context {context_id}')

            # Replace images if provided
            if images is not None:
                # If images list is empty (removing all images), update content_type to text
                if len(images) == 0:
                    await repos.images.replace_images_for_context(context_id, [])
                    await repos.context.update_content_type(context_id, 'text')
                    updated_fields.extend(['images', 'content_type'])
                    logger.debug(f'Removed all images from context {context_id}')
                else:
                    # Validate image data first
                    total_size = 0.0
                    for img in images:
                        if 'data' not in img or 'mime_type' not in img:
                            raise ToolError('Each image must have "data" and "mime_type" fields')

                        # Check individual image size
                        try:
                            img_data = base64.b64decode(img['data'])
                        except Exception:
                            raise ToolError('Invalid base64 image data') from None

                        img_size_mb = len(img_data) / (1024 * 1024)
                        total_size += img_size_mb

                        if img_size_mb > MAX_IMAGE_SIZE_MB:
                            raise ToolError(f'Image exceeds size limit of {MAX_IMAGE_SIZE_MB}MB')

                    # Check total size
                    if total_size > MAX_TOTAL_SIZE_MB:
                        raise ToolError(f'Total image size {total_size:.2f}MB exceeds limit of {MAX_TOTAL_SIZE_MB}MB')

                    # Replace images
                    await repos.images.replace_images_for_context(context_id, images)
                    updated_fields.append('images')

                    # Update content_type to multimodal if images were added
                    await repos.context.update_content_type(context_id, 'multimodal')
                    updated_fields.append('content_type')
                    logger.debug(f'Replaced images for context {context_id}')

            # Check if we need to update content_type based on current state
            if images is None and (text is not None or metadata is not None):
                # Check if there are existing images to determine content_type
                image_count = await repos.images.count_images_for_context(context_id)
                current_content_type = 'multimodal' if image_count > 0 else 'text'

                # Get the stored content type
                stored_content_type = await repos.context.get_content_type(context_id)

                # Update if different
                if stored_content_type != current_content_type:
                    await repos.context.update_content_type(context_id, current_content_type)
                    updated_fields.append('content_type')

            # Regenerate embedding if text was changed and semantic search is available (non-blocking)
            if text is not None and _embedding_service is not None:
                try:
                    new_embedding = await _embedding_service.generate_embedding(text)

                    # Check if embedding exists
                    embedding_exists = await repos.embeddings.exists(context_id)

                    if embedding_exists:
                        await repos.embeddings.update(
                            context_id=context_id,
                            embedding=new_embedding,
                        )
                        logger.debug(f'Updated embedding for context {context_id}')
                    else:
                        await repos.embeddings.store(
                            context_id=context_id,
                            embedding=new_embedding,
                            model=settings.embedding_model,
                        )
                        logger.debug(f'Created embedding for context {context_id}')

                    updated_fields.append('embedding')
                except Exception as e:
                    logger.warning(f'Failed to update embedding for context {context_id}: {e}')
                    # Non-blocking: continue even if embedding update fails

            logger.info(f'Successfully updated context {context_id}, fields: {updated_fields}')

            return UpdateContextSuccessDict(
                success=True,
                context_id=context_id,
                updated_fields=updated_fields,
                message=f'Successfully updated {len(updated_fields)} field(s)',
            )

        except ToolError:
            raise  # Re-raise ToolError as-is
        except Exception as update_error:
            logger.error(f'Error during context update: {update_error}')
            raise ToolError(f'Update operation failed: {str(update_error)}') from update_error

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error updating context: {e}')
        raise ToolError(f'Unexpected error: {str(e)}') from e


# MCP Resources for read-only access


@mcp.tool()
async def list_threads(ctx: Context | None = None) -> ThreadListDict:
    """
    List all active threads with statistics.
    Read-only resource for thread discovery.

    Returns:
        dict: Dictionary containing list of threads and total count.

    Raises:
        ToolError: If listing threads fails.
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
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error listing threads: {e}')
        raise ToolError(f'Failed to list threads: {str(e)}') from e


@mcp.tool()
async def get_statistics(ctx: Context | None = None) -> dict[str, Any]:
    """
    Database statistics and usage metrics.
    Useful for monitoring and debugging.

    Returns:
        dict: Database statistics including counts and size metrics.

    Raises:
        ToolError: If getting statistics fails.
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

        # Add semantic search metrics if available
        if settings.enable_semantic_search:
            if _embedding_service is not None:
                embedding_stats = await repos.embeddings.get_statistics()
                stats['semantic_search'] = {
                    'enabled': True,
                    'available': True,
                    'model': settings.embedding_model,
                    'dimensions': settings.embedding_dim,
                    'embedding_count': embedding_stats['total_embeddings'],
                    'coverage_percentage': embedding_stats['coverage_percentage'],
                }
            else:
                stats['semantic_search'] = {
                    'enabled': True,
                    'available': False,
                    'message': 'Dependencies not met or initialization failed',
                }
        else:
            stats['semantic_search'] = {
                'enabled': False,
                'available': False,
            }

        return stats
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error getting statistics: {e}')
        raise ToolError(f'Failed to get statistics: {str(e)}') from e


async def semantic_search_tool(
    query: Annotated[str, Field(min_length=1, description='Natural language search query')],
    top_k: Annotated[int, Field(ge=1, le=100, description='Number of results to return')] = 20,
    thread_id: Annotated[str | None, Field(min_length=1, description='Filter by thread')] = None,
    source: Annotated[Literal['user', 'agent'] | None, Field(description='Filter by source type')] = None,
    ctx: Context | None = None,
) -> dict[str, Any]:
    """
    Semantic similarity search using vector embeddings.

    This tool performs semantic search on stored context using vector similarity.
    It finds contexts that are semantically similar to the query, even if they
    don't share exact keywords.

    Note: This tool is only available when semantic search is enabled and all
    dependencies are met (ollama, numpy, sqlite-vec, EmbeddingGemma model).

    Args:
        query: Natural language search query
        top_k: Number of results to return (1-100)
        thread_id: Optional filter by thread
        source: Optional filter by source type
        ctx: FastMCP context object

    Returns:
        dict: Search results with context entries and similarity scores

    Raises:
        ToolError: If semantic search is not available or search fails
    """
    # Check if semantic search is available
    if _embedding_service is None:
        raise ToolError(
            'Semantic search is not available. '
            'Ensure ENABLE_SEMANTIC_SEARCH=true and all dependencies are installed. '
            f'Run: uv sync --extra semantic-search && ollama pull {settings.embedding_model}',
        )

    try:
        if ctx:
            await ctx.info(f'Performing semantic search: "{query[:50]}..."')

        # Get repositories
        repos = await _ensure_repositories()

        # Generate embedding for query
        try:
            query_embedding = await _embedding_service.generate_embedding(query)
        except Exception as e:
            logger.error(f'Failed to generate query embedding: {e}')
            raise ToolError(f'Failed to generate embedding for query: {str(e)}') from e

        # Perform similarity search
        try:
            results = await repos.embeddings.search(
                query_embedding=query_embedding,
                limit=top_k,
                thread_id=thread_id,
                source=source,
            )
        except Exception as e:
            logger.error(f'Semantic search failed: {e}')
            raise ToolError(f'Semantic search failed: {str(e)}') from e

        # Enrich results with tags
        for result in results:
            context_id = result.get('id')
            if context_id:
                tags_result = await repos.tags.get_tags_for_context(int(context_id))
                result['tags'] = tags_result
            else:
                result['tags'] = []

        logger.info(f'Semantic search found {len(results)} results for query: "{query[:50]}..."')

        return {
            'query': query,
            'results': results,
            'count': len(results),
            'model': settings.embedding_model,
        }

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error in semantic search: {e}')
        raise ToolError(f'Semantic search failed: {str(e)}') from e


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
