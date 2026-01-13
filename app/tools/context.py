"""
Context CRUD operations for MCP tools.

This module contains the core context management tools:
- store_context: Store a new context entry
- get_context_by_ids: Retrieve specific context entries by ID
- update_context: Update an existing context entry
- delete_context: Delete context entries
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

from app.settings import get_settings
from app.startup import MAX_IMAGE_SIZE_MB
from app.startup import MAX_TOTAL_SIZE_MB
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup.validation import deserialize_json_param
from app.types import ContextEntryDict
from app.types import JsonValue
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
            'status (state information), priority (numeric value for range queries), '
            'agent_name (specific agent identifier), task_name (task title for string searches), '
            'completed (boolean flag for completion state). '
            'These fields are indexed for faster filtering but not required.',
        ),
    ] = None,
    tags: Annotated[list[str] | None, Field(description='List of tags (normalized to lowercase)')] = None,
    ctx: Context | None = None,
) -> StoreContextSuccessDict:
    """Store a context entry with optional images and metadata.

    All agents working on the same task should use the same thread_id to share context.
    If an entry with identical thread_id, source, and text already exists, it will be
    updated instead of creating a duplicate (deduplication).

    Notes:
        - Tags are normalized to lowercase and stored separately for efficient filtering
        - If semantic search is enabled, an embedding is automatically generated
        - Use indexed metadata fields (status, priority, agent_name, task_name, completed)
          for faster filtering in search_context

    Returns:
        StoreContextSuccessDict with success, context_id, thread_id, message fields.

    Raises:
        ToolError: If validation fails or storage operation fails.
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
        repos = await ensure_repositories()

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
        valid_image_count = 0
        if images:
            # Pre-validate ALL images before storing any
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

                valid_image_count += 1

            # All validations passed, store the images
            logger.debug(f'Pre-validation passed for {valid_image_count} images, total size: {total_size:.2f}MB')
            try:
                await repos.images.store_images(context_id, images)
            except Exception as e:
                raise ToolError(f'Failed to store images: {str(e)}') from e

        # Generate embedding if embedding generation is enabled (BLOCKING when enabled)
        # If ENABLE_EMBEDDING_GENERATION=true (provider is not None), embedding failures are errors
        # This ensures no silent embedding gaps when user expects embeddings to be created
        embedding_generated = False
        store_embedding_provider = get_embedding_provider()
        if store_embedding_provider is not None:
            try:
                # Import lazily to avoid linter removing unused import at module level
                from app.startup import get_chunking_service

                chunking_service = get_chunking_service()
                logger.debug(
                    f'[EMBEDDING] Chunking service state: service={chunking_service}, '
                    f'enabled={chunking_service.is_enabled if chunking_service else "N/A"}')
                if chunking_service is not None and chunking_service.is_enabled:
                    # Chunked embedding generation for long documents
                    chunks = chunking_service.split_text(text)
                    chunk_texts = [chunk.text for chunk in chunks]
                    logger.info(f'[EMBEDDING] Generating embeddings: text_len={len(text)}, chunks={len(chunks)}')
                    embeddings = await store_embedding_provider.embed_documents(chunk_texts)
                    logger.info(f'[EMBEDDING] Embeddings generated: chunks={len(chunk_texts)}, embeddings={len(embeddings)}')
                    await repos.embeddings.store_chunked(
                        context_id=context_id,
                        embeddings=embeddings,
                        model=settings.embedding.model,
                    )
                    embedding_generated = True
                    logger.debug(f'Generated {len(chunks)} chunk embeddings for context {context_id}')
                else:
                    # Single embedding (chunking disabled or not configured)
                    embedding = await store_embedding_provider.embed_query(text)
                    await repos.embeddings.store(
                        context_id=context_id,
                        embedding=embedding,
                        model=settings.embedding.model,
                    )
                    embedding_generated = True
                    logger.debug(f'Generated embedding for context {context_id}')
            except Exception as e:
                logger.error(f'Failed to generate/store embedding for context {context_id}: {e}')
                # BLOCKING: If embedding generation is enabled, failure is an error
                # The context was stored, but embedding failed - this is a data integrity issue
                raise ToolError(
                    f'Context stored (id={context_id}) but embedding generation failed: {str(e)}. '
                    f'The entry exists but cannot be found via semantic search.',
                ) from e

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
    """Update an existing context entry. Only provided fields are modified.

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
        ToolError: If validation fails, entry not found, or update operation fails.
    """
    try:
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

        # Get repositories
        repos = await ensure_repositories()

        # Check if entry exists
        exists = await repos.context.check_entry_exists(context_id)
        if not exists:
            raise ToolError(f'Context entry with ID {context_id} not found')

        updated_fields: list[str] = []

        # Start transaction-like operations
        try:
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
                )

                if not success:
                    raise ToolError('Failed to patch metadata')

                updated_fields.extend(fields)
                logger.debug(f'Applied metadata patch to context {context_id}')

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

            # Regenerate embedding if text was changed and embedding generation is enabled (BLOCKING)
            # If ENABLE_EMBEDDING_GENERATION=true (provider is not None), embedding failures are errors
            update_embedding_provider = get_embedding_provider()
            if text is not None and update_embedding_provider is not None:
                try:
                    # Import lazily to avoid linter removing unused import at module level
                    from app.startup import get_chunking_service

                    chunking_service = get_chunking_service()
                    if chunking_service is not None and chunking_service.is_enabled:
                        # Chunked embedding regeneration for long documents
                        # Delete existing chunks first
                        await repos.embeddings.delete_all_chunks(context_id)
                        # Generate and store new chunks
                        chunks = chunking_service.split_text(text)
                        chunk_texts = [chunk.text for chunk in chunks]
                        embeddings = await update_embedding_provider.embed_documents(chunk_texts)
                        await repos.embeddings.store_chunked(
                            context_id=context_id,
                            embeddings=embeddings,
                            model=settings.embedding.model,
                        )
                        logger.debug(f'Regenerated {len(chunks)} chunk embeddings for context {context_id}')
                    else:
                        # Single embedding (chunking disabled or not configured)
                        new_embedding = await update_embedding_provider.embed_query(text)

                        # Check if embedding exists
                        embedding_exists = await repos.embeddings.exists(context_id)

                        if embedding_exists:
                            await repos.embeddings.update(
                                context_id=context_id,
                                embeddings=[new_embedding],
                                model=settings.embedding.model,
                            )
                            logger.debug(f'Updated embedding for context {context_id}')
                        else:
                            await repos.embeddings.store(
                                context_id=context_id,
                                embedding=new_embedding,
                                model=settings.embedding.model,
                            )
                            logger.debug(f'Created embedding for context {context_id}')

                    updated_fields.append('embedding')
                except Exception as e:
                    logger.error(f'Failed to update embedding for context {context_id}: {e}')
                    # BLOCKING: If embedding generation is enabled, failure is an error
                    raise ToolError(
                        f'Context updated but embedding regeneration failed: {str(e)}. '
                        f'The text was updated but semantic search may return stale results.',
                    ) from e

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
