"""
Context CRUD operations for MCP tools.

This module contains the core context management tools:
- store_context: Store a new context entry
- get_context_by_ids: Retrieve specific context entries by ID
- update_context: Update an existing context entry
- delete_context: Delete context entries

Embedding-First Pattern:
This module implements atomic embedding + data storage. When embeddings are enabled:
1. Embeddings are generated FIRST (outside any database transaction)
2. If embedding generation fails, NO data is saved (returns error immediately)
3. If embedding generation succeeds, ALL database operations occur in a SINGLE atomic transaction
"""

import base64
import json
import logging
from typing import Annotated
from typing import Literal
from typing import cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.migrations import format_exception_message
from app.repositories.embedding_repository import ChunkEmbedding
from app.settings import get_settings
from app.startup import MAX_IMAGE_SIZE_MB
from app.startup import MAX_TOTAL_SIZE_MB
from app.startup import ensure_repositories
from app.startup import get_chunking_service
from app.startup import get_embedding_provider
from app.startup.validation import deserialize_json_param
from app.types import ContextEntryDict
from app.types import JsonValue
from app.types import MetadataDict
from app.types import StoreContextSuccessDict
from app.types import UpdateContextSuccessDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def _generate_embeddings_for_text(text: str) -> list[ChunkEmbedding] | None:
    """Generate embeddings for text using configured provider.

    This function implements the 'embedding-first' pattern by generating
    embeddings BEFORE any database transaction is started. If embedding
    generation fails, no data should be saved.

    Args:
        text: Text content to embed

    Returns:
        List of ChunkEmbedding objects with embedding vectors and boundaries,
        or None if embedding generation is not enabled.

    Raises:
        ToolError: If embedding generation is enabled but fails.
    """
    embedding_provider = get_embedding_provider()
    if embedding_provider is None:
        return None

    try:
        chunking_service = get_chunking_service()
        logger.debug(
            f'[EMBEDDING-FIRST] Chunking service state: service={chunking_service}, '
            f'enabled={chunking_service.is_enabled if chunking_service else "N/A"}',
        )

        if chunking_service is not None and chunking_service.is_enabled:
            # Chunked embedding for long documents
            chunks = chunking_service.split_text(text)
            chunk_texts = [chunk.text for chunk in chunks]
            logger.info(f'[EMBEDDING-FIRST] Generating embeddings: text_len={len(text)}, chunks={len(chunks)}')
            embeddings = await embedding_provider.embed_documents(chunk_texts)
            logger.info(f'[EMBEDDING-FIRST] Embeddings generated: chunks={len(chunk_texts)}, embeddings={len(embeddings)}')

            return [
                ChunkEmbedding(
                    embedding=emb,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                )
                for emb, chunk in zip(embeddings, chunks, strict=True)
            ]
        # Single embedding (chunking disabled)
        logger.info(f'[EMBEDDING-FIRST] Generating single embedding: text_len={len(text)}')
        embedding = await embedding_provider.embed_query(text)
        logger.info('[EMBEDDING-FIRST] Single embedding generated')
        return [ChunkEmbedding(embedding=embedding, start_index=0, end_index=len(text))]

    except Exception as e:
        # CRITICAL: Embedding generation failed - this error must be raised
        # to prevent any data from being saved
        raise ToolError(f'Embedding generation failed: {format_exception_message(e)}') from e


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
    """Store a context entry with atomic embedding + data storage.

    EMBEDDING-FIRST PATTERN:
    1. Embeddings are generated FIRST (outside any database transaction)
    2. If embedding generation fails, NO data is saved (returns error immediately)
    3. If embedding generation succeeds, ALL database operations occur in a SINGLE atomic transaction

    All agents working on the same task should use the same thread_id to share context.
    If an entry with identical thread_id, source, and text already exists, it will be
    updated instead of creating a duplicate (deduplication).

    Notes:
        - Tags are normalized to lowercase and stored separately for efficient filtering
        - If semantic search is enabled, an embedding is automatically generated
        - Use indexed metadata fields for faster filtering in search_context:
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

        # Deserialize JSON parameters if needed
        images_raw = deserialize_json_param(cast(JsonValue | None, images))
        images = cast(list[dict[str, str]] | None, images_raw)
        tags_raw = deserialize_json_param(cast(JsonValue | None, tags))
        tags = cast(list[str] | None, tags_raw)
        metadata_raw = deserialize_json_param(cast(JsonValue | None, metadata))
        metadata = cast(MetadataDict | None, metadata_raw)

        # Determine content type and validate images
        content_type: Literal['text', 'multimodal'] = 'text'
        validated_images: list[dict[str, str]] = []

        if images:
            total_size: float = 0.0
            for idx, img in enumerate(images):
                # Validate required data field
                if 'data' not in img:
                    raise ToolError(f'Image {idx} is missing required "data" field')

                img_data_str = img.get('data', '')
                if not img_data_str or not img_data_str.strip():
                    raise ToolError(f'Image {idx} has empty "data" field')

                # mime_type is optional - defaults to 'image/png' if not provided
                if 'mime_type' not in img:
                    img['mime_type'] = 'image/png'

                # Validate base64 encoding
                try:
                    image_binary = base64.b64decode(img_data_str)
                except Exception as e:
                    raise ToolError(f'Image {idx} has invalid base64 encoding: {str(e)}') from None

                # Validate image size
                image_size_mb = len(image_binary) / (1024 * 1024)

                if image_size_mb > MAX_IMAGE_SIZE_MB:
                    raise ToolError(f'Image {idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit')

                total_size += image_size_mb
                if total_size > MAX_TOTAL_SIZE_MB:
                    raise ToolError(f'Total size exceeds {MAX_TOTAL_SIZE_MB}MB limit')

            content_type = 'multimodal'
            validated_images = images
            logger.debug(f'Pre-validation passed for {len(validated_images)} images, total size: {total_size:.2f}MB')

        # === PHASE 2: Generate Embedding FIRST (Outside Transaction) ===
        # CRITICAL: Embedding generation happens BEFORE any database operations.
        # If it fails, NO data is saved - this is the core of transactional integrity.
        chunk_embeddings = await _generate_embeddings_for_text(text)
        # If we get here, embedding either succeeded or is disabled (None)
        embedding_generated = chunk_embeddings is not None

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        repos = await ensure_repositories()
        backend = repos.context.backend
        metadata_str = json.dumps(metadata, ensure_ascii=False) if metadata else None

        async with backend.begin_transaction() as txn:
            # Store context entry with deduplication
            context_id, was_updated = await repos.context.store_with_deduplication(
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                text_content=text,
                metadata=metadata_str,
                txn=txn,
            )

            # Ensure we got a valid ID (not None or 0)
            if not context_id:
                raise ToolError('Failed to store context')

            # Store normalized tags
            if tags:
                await repos.tags.store_tags(context_id, tags, txn=txn)

            # Store images
            if validated_images:
                await repos.images.store_images(context_id, validated_images, txn=txn)

            # Store embeddings only if:
            # 1. New entry (not was_updated) - always store embeddings, OR
            # 2. Deduplicated entry (was_updated) but no embeddings exist yet - store embeddings
            # Skip if: Deduplicated entry AND embeddings already exist
            if chunk_embeddings is not None:
                should_store_embedding = True
                if was_updated:
                    embedding_exists = await repos.embeddings.exists(context_id)
                    should_store_embedding = not embedding_exists
                    if not should_store_embedding:
                        logger.debug(
                            f'Skipping embedding storage for deduplicated context {context_id} '
                            f'(embeddings already exist)',
                        )

                if should_store_embedding:
                    await repos.embeddings.store_chunked(
                        context_id=context_id,
                        chunk_embeddings=chunk_embeddings,
                        model=settings.embedding.model,
                        txn=txn,
                        upsert=was_updated,
                    )

            # COMMIT happens here - all or nothing

        action = 'updated' if was_updated else 'stored'
        logger.info(f'{action.capitalize()} context {context_id} in thread {thread_id}')

        return StoreContextSuccessDict(
            success=True,
            context_id=context_id,
            thread_id=thread_id,
            message=f'Context {action} with {len(validated_images)} images'
            + (' (embedding generated)' if embedding_generated else ''),
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

    Use this after search_context to retrieve complete content for entries of interest,
    or when you have specific context IDs from previous operations.

    Workflow: search_context (browse, truncated) -> get_context_by_ids (retrieve full content)

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
        raise ToolError(f'Failed to fetch context entries: {str(e)}') from e


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

    Provide EITHER context_ids OR thread_id (not both). Cascading delete removes
    associated tags, images, and embeddings.

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
        raise ToolError(f'Failed to delete context: {str(e)}') from e


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
    """Update an existing context entry with atomic embedding + data storage.

    EMBEDDING-FIRST PATTERN:
    1. Embeddings are generated FIRST (outside any database transaction) if text changed
    2. If embedding generation fails, NO data is saved (original data preserved)
    3. If embedding generation succeeds, ALL database operations occur in a SINGLE atomic transaction

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
        validated_images: list[dict[str, str]] = []
        if images is not None and len(images) > 0:
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

            validated_images = images

        # Get repositories
        repos = await ensure_repositories()

        # Check if entry exists (read-only, outside transaction)
        exists = await repos.context.check_entry_exists(context_id)
        if not exists:
            raise ToolError(f'Context entry with ID {context_id} not found')

        # === PHASE 2: Generate Embedding FIRST (Outside Transaction) ===
        # CRITICAL: Embedding generation happens BEFORE any database modifications.
        # If it fails, NO data is modified - original data is preserved.
        chunk_embeddings: list[ChunkEmbedding] | None = None
        embedding_generated = False

        if text is not None:
            # Only generate embedding if text is being changed
            chunk_embeddings = await _generate_embeddings_for_text(text)
            embedding_generated = chunk_embeddings is not None

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend
        updated_fields: list[str] = []

        async with backend.begin_transaction() as txn:
            # Update text content and/or metadata (full replacement) if provided
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
                    txn=txn,
                )

                if not success:
                    raise ToolError('Failed to update context entry')

                updated_fields.extend(fields)

            # Apply metadata patch (partial update) if provided
            # RFC 7396 JSON Merge Patch: merges with existing metadata
            # - New keys are added
            # - Existing keys are replaced with new values
            # - null values DELETE keys (cannot store null values with patch)
            if metadata_patch is not None:
                success, fields = await repos.context.patch_metadata(
                    context_id=context_id,
                    patch=metadata_patch,
                    txn=txn,
                )

                if not success:
                    raise ToolError('Failed to patch metadata')

                updated_fields.extend(fields)
                logger.debug(f'Applied metadata patch to context {context_id}')

            # Replace tags if provided
            if tags is not None:
                await repos.tags.replace_tags_for_context(context_id, tags, txn=txn)
                updated_fields.append('tags')
                logger.debug(f'Replaced tags for context {context_id}')

            # Replace images if provided
            if images is not None:
                # If images list is empty (removing all images), update content_type to text
                if len(images) == 0:
                    await repos.images.replace_images_for_context(context_id, [], txn=txn)
                    await repos.context.update_content_type(context_id, 'text', txn=txn)
                    updated_fields.extend(['images', 'content_type'])
                    logger.debug(f'Removed all images from context {context_id}')
                else:
                    # Replace images with validated data
                    await repos.images.replace_images_for_context(context_id, validated_images, txn=txn)
                    updated_fields.append('images')

                    # Update content_type to multimodal if images were added
                    await repos.context.update_content_type(context_id, 'multimodal', txn=txn)
                    updated_fields.append('content_type')
                    logger.debug(f'Replaced images for context {context_id}')

            # Check if we need to update content_type based on current state
            # Note: image count check is still read-only, safe to do inside transaction
            if images is None and (text is not None or metadata is not None):
                # Check if there are existing images to determine content_type
                image_count = await repos.images.count_images_for_context(context_id)
                current_content_type = 'multimodal' if image_count > 0 else 'text'

                # Get the stored content type
                stored_content_type = await repos.context.get_content_type(context_id)

                # Update if different
                if stored_content_type != current_content_type:
                    await repos.context.update_content_type(context_id, current_content_type, txn=txn)
                    updated_fields.append('content_type')

            # Store embeddings (guaranteed to exist if text changed and embedding is enabled)
            if chunk_embeddings is not None:
                # Delete existing chunks first (within the same transaction)
                await repos.embeddings.delete_all_chunks(context_id, txn=txn)

                # Store new embeddings
                await repos.embeddings.store_chunked(
                    context_id=context_id,
                    chunk_embeddings=chunk_embeddings,
                    model=settings.embedding.model,
                    txn=txn,
                )
                updated_fields.append('embedding')
                logger.debug(f'Updated {len(chunk_embeddings)} chunk embeddings for context {context_id}')

            # COMMIT happens here - all or nothing

        logger.info(f'Successfully updated context {context_id}, fields: {updated_fields}')

        return UpdateContextSuccessDict(
            success=True,
            context_id=context_id,
            updated_fields=updated_fields,
            message=f'Successfully updated {len(updated_fields)} field(s)'
            + (' (embedding regenerated)' if embedding_generated else ''),
        )

    except ToolError:
        raise  # Re-raise ToolError as-is for FastMCP to handle
    except Exception as e:
        logger.error(f'Error updating context: {e}')
        raise ToolError(f'Failed to update context: {format_exception_message(e)}') from e
