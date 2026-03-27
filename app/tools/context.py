"""
Context CRUD operations for MCP tools.

This module contains the core context management tools:
- store_context: Store a new context entry
- get_context_by_ids: Retrieve specific context entries by ID
- update_context: Update an existing context entry
- delete_context: Delete context entries

Generation-First Transactional Integrity:
This module implements atomic generation + data storage. When embedding or summary
generation is enabled:
1. Embeddings and summaries are generated in PARALLEL via asyncio.gather(return_exceptions=True)
   OUTSIDE any database transaction
2. Each result is independently inspected -- if ANY generation fails, NO data is saved
3. Retry budgets are fully managed by app/embeddings/retry.py and app/summary/retry.py;
   no re-invocation occurs at the gather level
4. If all generation succeeds, ALL database operations occur in a SINGLE atomic transaction

Infrastructure functions (embedding/summary generation, transaction heartbeat,
connection error classification, image validation, response message builders) are
in app.tools._shared -- the single source of truth for logic shared with batch.py.
"""

import asyncio
import json
import logging
from collections.abc import Awaitable
from typing import Annotated
from typing import Literal
from typing import cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.errors import format_exception_message
from app.repositories.embedding_repository import ChunkEmbedding
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.tools._shared import build_store_response_message
from app.tools._shared import build_update_response_message
from app.tools._shared import execute_store_in_transaction
from app.tools._shared import execute_update_in_transaction
from app.tools._shared import generate_embeddings_with_timeout
from app.tools._shared import generate_summary_with_timeout
from app.tools._shared import is_connection_error
from app.tools._shared import validate_and_normalize_images
from app.types import ContextEntryDict
from app.types import MetadataDict
from app.types import StoreContextSuccessDict
from app.types import UpdateContextSuccessDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def store_context(
    thread_id: Annotated[str, Field(min_length=1, description='Unique identifier for the conversation/task thread')],
    source: Annotated[Literal['user', 'agent'], Field(description="Either 'user' or 'agent'")],
    text: Annotated[str, Field(min_length=1, description='Text content to store')],
    images: Annotated[
        list[dict[str, str]] | None,
        Field(description='List of base64 encoded images with mime_type. Each image max 10MB, total max 100MB'),
    ] = None,
    metadata: Annotated[
        MetadataDict | None,
        Field(
            description='Additional structured data. For optimal performance, consider using indexed field names: '
            'status (state information), agent_name (specific agent identifier), '
            'task_name (task title for string searches), project (project name), '
            'report_type (type of report). '
            'These fields are indexed for faster filtering but not required.',
        ),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='List of tags (normalized to lowercase)')] = None,
    ctx: Context | None = None,
) -> StoreContextSuccessDict:
    """Store a context entry.

    All agents working on the same task should use the same thread_id to share context.

    Deduplication: if an entry with identical thread_id, source, and text already exists,
    the existing entry is updated instead of creating a duplicate:
    - metadata: New values override existing; omitting metadata preserves current values
    - tags: REPLACED with new list if provided; preserved if tags=None
    - images: REPLACED with new list if provided; preserved if images=None

    Deduplication is suppressed when opposite-source entries (e.g., agent entries
    for a user store) exist after the candidate duplicate. This preserves
    chronological ordering for repeated identical messages in conversations.

    Notes:
        - Tags are normalized to lowercase
        - Use indexed metadata fields for faster filtering:
          status, agent_name, task_name, project, report_type

    Returns:
        StoreContextSuccessDict with success, context_id, thread_id, message fields.

    Raises:
        ToolError: If validation fails, embedding generation fails, or storage operation fails.
    """
    try:
        # === PHASE 1: Input Validation (no DB operations) ===

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

        # Log info if context is available
        if ctx:
            await ctx.info(f'Storing context for thread: {thread_id}')

        # Determine content type and validate images
        validated_images, content_type, _ = validate_and_normalize_images(images, error_mode='raise')

        # === PHASE 2: Generate Summary + Embedding in PARALLEL (Outside Transaction) ===
        # CRITICAL: Both generation steps happen BEFORE any database operations.
        # If either fails, NO data is saved - this is the core of transactional integrity.

        repos = await ensure_repositories()
        chunk_embeddings: list[ChunkEmbedding] | None = None
        embedding_generated = False
        summary_text: str | None = None
        summary_generated = False

        # Performance optimization: pre-check for likely duplicates (read-only)
        likely_duplicate_id: int | None = None

        if get_embedding_provider() is not None or get_summary_provider() is not None:
            likely_duplicate_id = await repos.context.check_latest_is_duplicate(
                thread_id=thread_id,
                source=source,
                text_content=text,
            )

        # Build parallel tasks
        tasks_to_run: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
        task_names: list[str] = []

        # Embedding task (existing logic with pre-check optimization)
        if get_embedding_provider() is not None:
            if likely_duplicate_id is not None:
                has_embeddings = await repos.embeddings.exists(likely_duplicate_id)
                if has_embeddings:
                    logger.debug(
                        'Pre-check: skipping embedding generation for likely duplicate '
                        'of context %d in thread %s',
                        likely_duplicate_id, thread_id,
                    )
                else:
                    logger.debug(
                        'Pre-check: duplicate detected but no embeddings exist for context %d '
                        'in thread %s, generating embeddings',
                        likely_duplicate_id, thread_id,
                    )
                    tasks_to_run.append(generate_embeddings_with_timeout(text))
                    task_names.append('embedding')
            else:
                tasks_to_run.append(generate_embeddings_with_timeout(text))
                task_names.append('embedding')

        # Summary task (mirrors embedding pre-check pattern)
        if get_summary_provider() is not None:
            min_content_length = settings.summary.min_content_length
            if min_content_length > 0 and len(text) < min_content_length:
                logger.info(
                    'Skipping summary generation: text length %d < min_content_length %d',
                    len(text), min_content_length,
                )
            elif likely_duplicate_id is not None:
                existing_summary = await repos.context.get_summary(likely_duplicate_id)
                if existing_summary is not None:
                    summary_text = existing_summary
                    logger.debug(
                        'Pre-check: reusing existing summary for likely duplicate '
                        'of context %d in thread %s',
                        likely_duplicate_id, thread_id,
                    )
                else:
                    tasks_to_run.append(generate_summary_with_timeout(text, source))
                    task_names.append('summary')
            else:
                tasks_to_run.append(generate_summary_with_timeout(text, source))
                task_names.append('summary')

        # Execute all tasks in parallel
        if tasks_to_run:
            results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            errors: list[tuple[str, BaseException]] = []
            for name, result in zip(task_names, results, strict=True):
                if isinstance(result, BaseException):
                    errors.append((name, result))
                    logger.error('Generation failed for %s (retries exhausted): %s', name, result)
                elif name == 'embedding':
                    chunk_embeddings = cast(list[ChunkEmbedding] | None, result)
                    embedding_generated = chunk_embeddings is not None
                elif name == 'summary':
                    summary_text = cast(str | None, result)
                    summary_generated = bool(summary_text)

            if errors:
                error_details = '; '.join(
                    f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
                )
                raise ToolError(f'Generation failed after exhausting configured retries: {error_details}')

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend
        metadata_str = json.dumps(metadata, ensure_ascii=False) if metadata else None

        max_retries = 2
        context_id = 0
        was_updated = False
        embedding_stored = False

        for attempt in range(max_retries + 1):
            try:
                async with backend.begin_transaction() as txn:
                    context_id, was_updated, embedding_stored = await execute_store_in_transaction(
                        repos, txn,
                        thread_id=thread_id,
                        source=source,
                        content_type=content_type,
                        text_content=text,
                        metadata_str=metadata_str,
                        summary=summary_text,
                        tags=tags,
                        validated_images=validated_images,
                        chunk_embeddings=chunk_embeddings,
                        embedding_model=settings.embedding.model,
                    )

                # Transaction committed successfully -- break retry loop
                break

            except ToolError:
                raise  # ToolError is a logical error, not connection error -- do not retry
            except Exception as e:
                if is_connection_error(e) and attempt < max_retries:
                    delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                    logger.warning(
                        'Transaction failed with connection error, retrying in %.1fs '
                        '(attempt %d/%d): %s',
                        delay, attempt + 1, max_retries, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise  # Non-connection error or max retries exceeded

        action = 'updated' if was_updated else 'stored'
        logger.info(f'{action.capitalize()} context {context_id} in thread {thread_id}')

        message = build_store_response_message(
            action=action,
            image_count=len(validated_images),
            embedding_generated=embedding_generated,
            embedding_stored=embedding_stored,
            summary_generated=summary_generated,
            summary_preserved=summary_text is not None and not summary_generated,
        )

        return StoreContextSuccessDict(
            success=True,
            context_id=context_id,
            thread_id=thread_id,
            message=message,
        )
    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error storing context: {e}')
        raise ToolError(f'Failed to store context: {format_exception_message(e)}') from e


async def get_context_by_ids(
    context_ids: Annotated[list[int], Field(min_length=1, description='List of context entry IDs to retrieve')],
    include_images: Annotated[bool, Field(description='Whether to include image data')] = True,
    ctx: Context | None = None,
) -> list[ContextEntryDict]:
    """Fetch specific context entries by their IDs with FULL (non-truncated) text content.

    Use this when you have specific context IDs from previous operations
    and need the complete, untruncated content.

    Non-existent IDs are silently skipped; only found entries are returned.

    Returns:
        List of ContextEntryDict with id, thread_id, source, text_content, metadata,
        tags, images, created_at, updated_at fields.

    Raises:
        ToolError: If fetching context entries fails.
    """
    try:
        if ctx:
            await ctx.info(f'Fetching context entries: {context_ids}')

        # Get repositories
        repos = await ensure_repositories()

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
        raise ToolError(f'Failed to fetch context entries: {format_exception_message(e)}') from e


async def delete_context(
    context_ids: Annotated[
        list[int] | None,
        Field(min_length=1, description='Specific context entry IDs to delete (mutually exclusive with thread_id)'),
    ] = None,
    thread_id: Annotated[
        str | None,
        Field(min_length=1, description='Delete ALL entries in thread (mutually exclusive with context_ids)'),
    ] = None,
    ctx: Context | None = None,
) -> dict[str, bool | int | str]:
    """Delete context entries by specific IDs or by entire thread. IRREVERSIBLE.

    Provide EITHER context_ids OR thread_id (not both). All associated data
    (tags, images) is also removed.

    WARNING: This operation cannot be undone. Verify IDs/thread before deletion.

    Returns:
        Dict with success (bool), deleted_count (int), and message (str) fields.

    Raises:
        ToolError: If neither context_ids nor thread_id provided, or deletion fails.
    """
    try:
        # Ensure at least one parameter is provided (business logic validation)
        if not context_ids and not thread_id:
            raise ToolError('Must provide either context_ids or thread_id')

        if ctx:
            await ctx.info(f'Deleting context: ids={context_ids}, thread={thread_id}')

        # Get repositories
        repos = await ensure_repositories()

        deleted = 0

        if context_ids:
            # Delete embeddings first (explicit cleanup)
            if settings.semantic_search.enabled:
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
            if settings.semantic_search.enabled:
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
        raise ToolError(f'Failed to delete context: {format_exception_message(e)}') from e


async def update_context(
    context_id: Annotated[int, Field(gt=0, description='ID of the context entry to update')],
    text: Annotated[str | None, Field(min_length=1, description='New text content (replaces existing)')] = None,
    metadata: Annotated[MetadataDict | None, Field(description='New metadata (FULL REPLACEMENT)')] = None,
    metadata_patch: Annotated[
        MetadataDict | None,
        Field(
            description='Partial metadata update (RFC 7396 JSON Merge Patch): new keys added, '
            'existing updated, null values DELETE keys. MUTUALLY EXCLUSIVE with metadata.',
        ),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='New tags list (REPLACES all existing)')] = None,
    images: Annotated[
        list[dict[str, str]] | None,
        Field(description='New images with base64 data and mime_type (REPLACES all existing)'),
    ] = None,
    ctx: Context | None = None,
) -> UpdateContextSuccessDict:
    """Update an existing context entry.

    Immutable fields: id, thread_id, source, created_at (cannot be changed)
    Auto-managed: content_type (recalculated based on images), updated_at

    Metadata options (MUTUALLY EXCLUSIVE):
    - metadata: FULL REPLACEMENT of entire metadata object
    - metadata_patch: RFC 7396 JSON Merge Patch - merge with existing
      - New keys added, existing keys updated, null values DELETE keys
      - Limitation: Cannot store null values (use full replacement instead)
      - Limitation: Arrays replaced entirely (no element-wise merge)

    Tags and images use REPLACEMENT semantics (not merge).

    Returns:
        UpdateContextSuccessDict with success, context_id, updated_fields, message fields.

    Raises:
        ToolError: If validation fails, embedding generation fails, entry not found, or update fails.
    """
    try:
        # === PHASE 1: Input Validation (no DB operations) ===

        # Clean text input if provided
        if text is not None:
            text = text.strip()
            # Business logic: if text provided, it cannot be empty after stripping
            if not text:
                raise ToolError('text cannot be empty or contain only whitespace')

        # Validate mutual exclusivity: metadata and metadata_patch cannot be used together
        # RFC 7396 Note: metadata_patch is for partial updates (merge), metadata is for full replacement
        if metadata is not None and metadata_patch is not None:
            raise ToolError(
                'Cannot use both metadata and metadata_patch parameters together. '
                'Use metadata for full replacement or metadata_patch for partial updates.',
            )

        # Validate that at least one field is provided for update
        # Note: metadata_patch is also a valid update field
        if text is None and metadata is None and metadata_patch is None and tags is None and images is None:
            raise ToolError('At least one field must be provided for update')

        if ctx:
            await ctx.info(f'Updating context entry {context_id}')

        # Validate images early (before any operations)
        validated_images, _, _ = validate_and_normalize_images(images, error_mode='raise')

        # Get repositories
        repos = await ensure_repositories()

        # Check if entry exists and retrieve source (read-only, outside transaction)
        exists, entry_source = await repos.context.check_entry_exists(context_id)
        if not exists:
            raise ToolError(f'Context entry with ID {context_id} not found')
        assert entry_source is not None  # guaranteed by exists=True

        # === PHASE 2: Generate Summary + Embedding FIRST (Outside Transaction) ===
        # CRITICAL: Both generation steps happen BEFORE any database modifications.
        # If either fails, NO data is modified - original data is preserved.
        chunk_embeddings: list[ChunkEmbedding] | None = None
        embedding_generated = False
        summary_text: str | None = None
        summary_generated = False
        clear_summary = False

        if text is not None:
            # Text changed - regenerate both embedding and summary in parallel
            tasks: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
            task_names: list[str] = []

            if get_embedding_provider() is not None:
                tasks.append(generate_embeddings_with_timeout(text))
                task_names.append('embedding')

            if get_summary_provider() is not None:
                min_content_length = settings.summary.min_content_length
                if min_content_length > 0 and len(text) < min_content_length:
                    clear_summary = True
                    logger.info(
                        'Skipping summary generation for update: text length %d < '
                        'min_content_length %d. Existing summary will be cleared.',
                        len(text), min_content_length,
                    )
                else:
                    tasks.append(generate_summary_with_timeout(text, entry_source))
                    task_names.append('summary')

            results = await asyncio.gather(*tasks, return_exceptions=True)

            errors: list[tuple[str, BaseException]] = []
            for name, result in zip(task_names, results, strict=True):
                if isinstance(result, BaseException):
                    errors.append((name, result))
                    logger.error('Generation failed for %s (retries exhausted): %s', name, result)
                elif name == 'embedding':
                    chunk_embeddings = cast(list[ChunkEmbedding] | None, result)
                    embedding_generated = chunk_embeddings is not None
                elif name == 'summary':
                    summary_text = cast(str | None, result)
                    summary_generated = bool(summary_text)

            if errors:
                error_details = '; '.join(
                    f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
                )
                raise ToolError(f'Generation failed after exhausting configured retries: {error_details}')

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend
        updated_fields: list[str] = []

        max_retries = 2

        for attempt in range(max_retries + 1):
            try:
                async with backend.begin_transaction() as txn:
                    updated_fields, _ = await execute_update_in_transaction(
                        repos, txn,
                        context_id=context_id,
                        text=text,
                        metadata=metadata,
                        metadata_patch=metadata_patch,
                        summary=summary_text,
                        clear_summary=clear_summary,
                        tags=tags,
                        images=images,
                        validated_images=validated_images,
                        chunk_embeddings=chunk_embeddings,
                        embedding_model=settings.embedding.model,
                    )

                # Transaction committed -- break retry loop
                break

            except ToolError:
                raise  # ToolError is a logical error, not connection error -- do not retry
            except Exception as e:
                if is_connection_error(e) and attempt < max_retries:
                    delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                    logger.warning(
                        'Transaction failed with connection error, retrying in %.1fs '
                        '(attempt %d/%d): %s',
                        delay, attempt + 1, max_retries, e,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise  # Non-connection error or max retries exceeded

        logger.info(f'Successfully updated context {context_id}, fields: {updated_fields}')

        message = build_update_response_message(
            updated_fields_count=len(updated_fields),
            embedding_generated=embedding_generated,
            summary_generated=summary_generated,
            summary_cleared=clear_summary,
        )

        return UpdateContextSuccessDict(
            success=True,
            context_id=context_id,
            updated_fields=updated_fields,
            message=message,
        )

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error updating context: {e}')
        raise ToolError(f'Failed to update context: {format_exception_message(e)}') from e
