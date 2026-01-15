"""
Batch operations for MCP tools.

This module contains tools for bulk context management:
- store_context_batch: Store multiple entries in one operation
- update_context_batch: Update multiple entries in one operation
- delete_context_batch: Delete entries by various criteria
"""

import base64
import json
import logging
import operator
from typing import Annotated
from typing import Any
from typing import Literal

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.migrations import format_exception_message
from app.settings import get_settings
from app.startup import MAX_IMAGE_SIZE_MB
from app.startup import MAX_TOTAL_SIZE_MB
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.types import BulkDeleteResponseDict
from app.types import BulkStoreResponseDict
from app.types import BulkStoreResultItemDict
from app.types import BulkUpdateResponseDict
from app.types import BulkUpdateResultItemDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def store_context_batch(
    entries: Annotated[
        list[dict[str, Any]],
        Field(
            description='List of context entries to store. Each entry must have: '
            'thread_id (str), source ("user" or "agent"), text (str). '
            'Optional: metadata (dict), tags (list[str]), images (list[dict]).',
            min_length=1,
            max_length=100,
        ),
    ],
    atomic: Annotated[
        bool,
        Field(
            description='If true, ALL entries must succeed or NONE are stored (transaction rollback). '
            'If false, partial success is allowed with per-item error reporting.',
        ),
    ] = True,
    ctx: Context | None = None,
) -> BulkStoreResponseDict:
    """Store multiple context entries in a single batch operation.

    Batch processing is significantly faster than individual store_context calls
    when storing many entries. Use for migrations, imports, or bulk operations.

    Atomicity modes:
    - atomic=True (default): All-or-nothing. If ANY entry fails, ALL are rolled back.
    - atomic=False: Best-effort. Each entry processed independently; partial success possible.

    Size limits:
    - Maximum 100 entries per batch
    - Image limits per entry: 10MB each, 100MB total
    - Standard tag normalization (lowercase)

    Returns:
        BulkStoreResponseDict with success (bool), total (int), succeeded (int),
        failed (int), results (list of index, success, context_id, error), message (str).

    Raises:
        ToolError: If validation fails in atomic mode or batch operation fails.
    """
    try:
        if ctx:
            await ctx.info(f'Batch storing {len(entries)} context entries (atomic={atomic})')

        repos = await ensure_repositories()

        # Phase 1: Validate all entries before processing
        validated_entries: list[dict[str, Any]] = []
        validation_errors: list[tuple[int, str]] = []

        for idx, entry in enumerate(entries):
            # Validate required fields
            if 'thread_id' not in entry or not entry.get('thread_id'):
                validation_errors.append((idx, 'Missing required field: thread_id'))
                continue
            if 'source' not in entry or entry.get('source') not in ('user', 'agent'):
                validation_errors.append((idx, 'Missing or invalid source (must be "user" or "agent")'))
                continue
            if 'text' not in entry or not entry.get('text'):
                validation_errors.append((idx, 'Missing required field: text'))
                continue

            # Clean input strings
            thread_id = str(entry['thread_id']).strip()
            text = str(entry['text']).strip()

            if not thread_id:
                validation_errors.append((idx, 'thread_id cannot be empty or whitespace'))
                continue
            if not text:
                validation_errors.append((idx, 'text cannot be empty or whitespace'))
                continue

            # Validate images if present
            images = entry.get('images', [])
            content_type = 'multimodal' if images else 'text'

            if images:
                total_size = 0.0
                for img_idx, img in enumerate(images):
                    if 'data' not in img:
                        validation_errors.append((idx, f'Image {img_idx} is missing required "data" field'))
                        break
                    try:
                        img_data = base64.b64decode(img['data'])
                        img_size_mb = len(img_data) / (1024 * 1024)
                        if img_size_mb > MAX_IMAGE_SIZE_MB:
                            validation_errors.append((idx, f'Image {img_idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit'))
                            break
                        total_size += img_size_mb
                        if total_size > MAX_TOTAL_SIZE_MB:
                            validation_errors.append((idx, f'Total image size exceeds {MAX_TOTAL_SIZE_MB}MB limit'))
                            break
                    except Exception:
                        validation_errors.append((idx, f'Image {img_idx} has invalid base64 encoding'))
                        break
                else:
                    # All images valid for this entry
                    pass

            # Check if entry already had validation errors from images
            if any(idx == err[0] for err in validation_errors):
                continue

            # Prepare validated entry
            metadata = entry.get('metadata')
            validated_entries.append({
                'index': idx,
                'thread_id': thread_id,
                'source': entry['source'],
                'text_content': text,
                'metadata': json.dumps(metadata, ensure_ascii=False) if metadata else None,
                'content_type': content_type,
                'tags': entry.get('tags', []),
                'images': images,
            })

        # Phase 2: In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at index {first_error[0]}: {first_error[1]}',
            )

        # Phase 3: Process validated entries through repository
        # Build results list including validation errors
        results: list[BulkStoreResultItemDict] = []

        # Add validation errors to results
        for idx, error in validation_errors:
            results.append(BulkStoreResultItemDict(
                index=idx,
                success=False,
                context_id=None,
                error=error,
            ))

        if validated_entries:
            # Prepare entries for repository batch operation
            repo_entries = [
                {
                    'thread_id': e['thread_id'],
                    'source': e['source'],
                    'text_content': e['text_content'],
                    'metadata': e['metadata'],
                    'content_type': e['content_type'],
                }
                for e in validated_entries
            ]

            # Execute batch store
            batch_results = await repos.context.store_contexts_batch(repo_entries)

            # Process repository results and store tags/images
            for repo_idx, ctx_id, repo_error in batch_results:
                original_entry = validated_entries[repo_idx]
                original_idx = original_entry['index']

                if ctx_id is not None and repo_error is None:
                    # Store tags if provided
                    if original_entry.get('tags'):
                        await repos.tags.store_tags(ctx_id, original_entry['tags'])

                    # Store images if provided
                    if original_entry.get('images'):
                        await repos.images.store_images(ctx_id, original_entry['images'])

                    # Generate embedding if embedding generation enabled
                    # Behavior depends on atomic flag:
                    # - atomic=True: Embedding failure fails the ENTIRE batch
                    # - atomic=False: Embedding failure is reported per-item, context is stored
                    batch_store_embedding_provider = get_embedding_provider()
                    if batch_store_embedding_provider is not None:
                        try:
                            # Import lazily to avoid linter removing unused import at module level
                            from app.startup import get_chunking_service

                            chunking_service = get_chunking_service()
                            if chunking_service is not None and chunking_service.is_enabled:
                                # Chunked embedding generation for long documents
                                # Import ChunkEmbedding for boundary-aware storage
                                from app.repositories.embedding_repository import ChunkEmbedding

                                text_content = original_entry['text_content']
                                chunks = chunking_service.split_text(text_content)
                                chunk_texts = [chunk.text for chunk in chunks]
                                embeddings = await batch_store_embedding_provider.embed_documents(chunk_texts)

                                # Create ChunkEmbedding objects with boundary information
                                chunk_embeddings = [
                                    ChunkEmbedding(
                                        embedding=emb,
                                        start_index=chunk.start_index,
                                        end_index=chunk.end_index,
                                    )
                                    for emb, chunk in zip(embeddings, chunks, strict=True)
                                ]

                                await repos.embeddings.store_chunked(
                                    context_id=ctx_id,
                                    chunk_embeddings=chunk_embeddings,
                                    model=settings.embedding.model,
                                )
                            else:
                                # Single embedding (chunking disabled or not configured)
                                text_content = original_entry['text_content']
                                embedding = await batch_store_embedding_provider.embed_query(text_content)
                                await repos.embeddings.store(
                                    context_id=ctx_id,
                                    embedding=embedding,
                                    model=settings.embedding.model,
                                    start_index=0,
                                    end_index=len(text_content),
                                )
                        except Exception as emb_err:
                            logger.error(f'Failed to generate embedding for context {ctx_id}: {emb_err}')
                            if atomic:
                                # Atomic mode: Embedding failure fails the entire batch
                                # Note: Context was already stored - in production, this would need
                                # transaction support to rollback. For now, raise error with partial info.
                                raise ToolError(
                                    f'Batch operation failed at index {original_idx}: embedding generation failed '
                                    f'for context {ctx_id}: {emb_err}. '
                                    f'Some entries may have been stored without embeddings.',
                                ) from emb_err
                            # Non-atomic mode: Report per-item error, context was stored but no embedding
                            results.append(BulkStoreResultItemDict(
                                index=original_idx,
                                success=False,
                                context_id=ctx_id,
                                error=f'Stored but embedding failed: {str(emb_err)}',
                            ))
                            continue  # Skip the success result below

                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=True,
                        context_id=ctx_id,
                        error=None,
                    ))
                else:
                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=False,
                        context_id=None,
                        error=repo_error or 'Unknown error',
                    ))

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(entries) - succeeded

        logger.info(f'Batch store completed: {succeeded}/{len(entries)} succeeded')

        return BulkStoreResponseDict(
            success=failed == 0,
            total=len(entries),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=f'Stored {succeeded}/{len(entries)} entries successfully',
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch store: {e}')
        raise ToolError(f'Batch store failed: {format_exception_message(e)}') from e


async def update_context_batch(
    updates: Annotated[
        list[dict[str, Any]],
        Field(
            description='List of update operations. Each must have context_id (int). '
            'Optional: text (str), metadata (dict - full replace), '
            'metadata_patch (dict - RFC 7396 merge), tags (list[str]), images (list[dict]).',
            min_length=1,
            max_length=100,
        ),
    ],
    atomic: Annotated[
        bool,
        Field(
            description='If true, ALL updates succeed or NONE are applied. '
            'If false, partial success allowed.',
        ),
    ] = True,
    ctx: Context | None = None,
) -> BulkUpdateResponseDict:
    """Update multiple context entries in a single batch operation.

    Similar semantics to update_context but for multiple entries:
    - Each update is identified by context_id
    - Only provided fields are modified
    - metadata vs metadata_patch are mutually exclusive per entry
    - Tags and images use replacement semantics

    Atomicity modes:
    - atomic=True (default): All-or-nothing transaction
    - atomic=False: Best-effort with per-item error reporting

    Returns:
        BulkUpdateResponseDict with success (bool), total (int), succeeded (int),
        failed (int), results (list of index, context_id, success, updated_fields, error),
        message (str).

    Raises:
        ToolError: If validation fails in atomic mode or batch operation fails.
    """
    try:
        if ctx:
            await ctx.info(f'Batch updating {len(updates)} context entries (atomic={atomic})')

        repos = await ensure_repositories()

        # Phase 1: Validate all updates before processing
        validated_updates: list[dict[str, Any]] = []
        validation_errors: list[tuple[int, int, str]] = []  # (index, context_id, error)

        for idx, update in enumerate(updates):
            # Validate required context_id
            if 'context_id' not in update:
                validation_errors.append((idx, 0, 'Missing required field: context_id'))
                continue

            context_id = update['context_id']
            if not isinstance(context_id, int) or context_id <= 0:
                validation_errors.append((idx, 0, 'context_id must be a positive integer'))
                continue

            # Validate mutual exclusivity of metadata and metadata_patch
            if update.get('metadata') is not None and update.get('metadata_patch') is not None:
                validation_errors.append((
                    idx,
                    context_id,
                    'Cannot use both metadata and metadata_patch. Use one or the other.',
                ))
                continue

            # Validate text if provided
            text = update.get('text')
            if text is not None:
                text = str(text).strip()
                if not text:
                    validation_errors.append((idx, context_id, 'text cannot be empty or whitespace'))
                    continue

            # Check that at least one field is provided for update
            has_update = any(
                update.get(field) is not None
                for field in ['text', 'metadata', 'metadata_patch', 'tags', 'images']
            )
            if not has_update:
                validation_errors.append((idx, context_id, 'At least one field must be provided for update'))
                continue

            # Validate images if provided
            images = update.get('images')
            if images is not None and len(images) > 0:
                total_size = 0.0
                for img_idx, img in enumerate(images):
                    if 'data' not in img:
                        validation_errors.append((idx, context_id, f'Image {img_idx} missing "data" field'))
                        break
                    try:
                        img_data = base64.b64decode(img['data'])
                        img_size_mb = len(img_data) / (1024 * 1024)
                        if img_size_mb > MAX_IMAGE_SIZE_MB:
                            validation_errors.append((
                                idx,
                                context_id,
                                f'Image {img_idx} exceeds {MAX_IMAGE_SIZE_MB}MB',
                            ))
                            break
                        total_size += img_size_mb
                        if total_size > MAX_TOTAL_SIZE_MB:
                            validation_errors.append((
                                idx,
                                context_id,
                                f'Total size exceeds {MAX_TOTAL_SIZE_MB}MB',
                            ))
                            break
                    except Exception:
                        validation_errors.append((idx, context_id, f'Image {img_idx} has invalid base64'))
                        break

            # Check if entry already had validation errors from images
            if any(idx == err[0] for err in validation_errors):
                continue

            # Prepare validated update
            validated_updates.append({
                'index': idx,
                'context_id': context_id,
                'text': text,
                'metadata': update.get('metadata'),
                'metadata_patch': update.get('metadata_patch'),
                'tags': update.get('tags'),
                'images': images,
            })

        # Phase 2: In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at context_id {first_error[1]}: {first_error[2]}',
            )

        # Phase 3: Process validated updates
        results: list[BulkUpdateResultItemDict] = []

        # Add validation errors to results
        for idx, context_id, error in validation_errors:
            results.append(BulkUpdateResultItemDict(
                index=idx,
                context_id=context_id,
                success=False,
                updated_fields=None,
                error=error,
            ))

        # Process each validated update
        for update in validated_updates:
            original_idx = update['index']
            context_id = update['context_id']
            updated_fields: list[str] = []

            try:
                # Check if entry exists
                exists = await repos.context.check_entry_exists(context_id)
                if not exists:
                    results.append(BulkUpdateResultItemDict(
                        index=original_idx,
                        context_id=context_id,
                        success=False,
                        updated_fields=None,
                        error=f'Context entry {context_id} not found',
                    ))
                    continue

                # Update text and/or metadata (full replacement)
                if update.get('text') is not None or update.get('metadata') is not None:
                    metadata_str = None
                    if update.get('metadata') is not None:
                        metadata_str = json.dumps(update['metadata'], ensure_ascii=False)

                    success, fields = await repos.context.update_context_entry(
                        context_id=context_id,
                        text_content=update.get('text'),
                        metadata=metadata_str,
                    )
                    if success:
                        updated_fields.extend(fields)

                # Apply metadata patch if provided
                if update.get('metadata_patch') is not None:
                    success, fields = await repos.context.patch_metadata(
                        context_id=context_id,
                        patch=update['metadata_patch'],
                    )
                    if success:
                        updated_fields.extend(fields)

                # Replace tags if provided
                if update.get('tags') is not None:
                    await repos.tags.replace_tags_for_context(context_id, update['tags'])
                    updated_fields.append('tags')

                # Replace images if provided
                if update.get('images') is not None:
                    images = update['images']
                    if len(images) == 0:
                        await repos.images.replace_images_for_context(context_id, [])
                        await repos.context.update_content_type(context_id, 'text')
                        updated_fields.extend(['images', 'content_type'])
                    else:
                        await repos.images.replace_images_for_context(context_id, images)
                        await repos.context.update_content_type(context_id, 'multimodal')
                        updated_fields.extend(['images', 'content_type'])

                # Regenerate embedding if text changed and embedding generation enabled
                # Behavior depends on atomic flag (same as store_context_batch)
                batch_update_embedding_provider = get_embedding_provider()
                if update.get('text') is not None and batch_update_embedding_provider is not None:
                    try:
                        # Import lazily to avoid linter removing unused import at module level
                        from app.startup import get_chunking_service

                        chunking_service = get_chunking_service()
                        if chunking_service is not None and chunking_service.is_enabled:
                            # Chunked embedding regeneration for long documents
                            # Import ChunkEmbedding for boundary-aware storage
                            from app.repositories.embedding_repository import ChunkEmbedding

                            # Delete existing chunks first
                            await repos.embeddings.delete_all_chunks(context_id)
                            # Generate and store new chunks with boundary info
                            text_content = update['text']
                            chunks = chunking_service.split_text(text_content)
                            chunk_texts = [chunk.text for chunk in chunks]
                            embeddings = await batch_update_embedding_provider.embed_documents(chunk_texts)

                            # Create ChunkEmbedding objects with boundary information
                            chunk_embeddings = [
                                ChunkEmbedding(
                                    embedding=emb,
                                    start_index=chunk.start_index,
                                    end_index=chunk.end_index,
                                )
                                for emb, chunk in zip(embeddings, chunks, strict=True)
                            ]

                            await repos.embeddings.store_chunked(
                                context_id=context_id,
                                chunk_embeddings=chunk_embeddings,
                                model=settings.embedding.model,
                            )
                        else:
                            # Single embedding (chunking disabled or not configured)
                            # Import ChunkEmbedding for boundary-aware storage
                            from app.repositories.embedding_repository import ChunkEmbedding

                            text_content = update['text']
                            new_embedding = await batch_update_embedding_provider.embed_query(text_content)
                            embedding_exists = await repos.embeddings.exists(context_id)

                            # Create single ChunkEmbedding with full document boundaries
                            single_chunk = ChunkEmbedding(
                                embedding=new_embedding,
                                start_index=0,
                                end_index=len(text_content),
                            )

                            if embedding_exists:
                                await repos.embeddings.update(
                                    context_id=context_id,
                                    chunk_embeddings=[single_chunk],
                                    model=settings.embedding.model,
                                )
                            else:
                                await repos.embeddings.store(
                                    context_id=context_id,
                                    embedding=new_embedding,
                                    model=settings.embedding.model,
                                    start_index=0,
                                    end_index=len(text_content),
                                )
                        updated_fields.append('embedding')
                    except Exception as emb_err:
                        logger.error(f'Failed to update embedding for context {context_id}: {emb_err}')
                        if atomic:
                            # Atomic mode: Embedding failure fails the entire batch
                            raise ToolError(
                                f'Batch update failed at index {original_idx}: embedding regeneration failed '
                                f'for context {context_id}: {emb_err}. '
                                f'Some entries may have been updated without embedding changes.',
                            ) from emb_err
                        # Non-atomic mode: Report per-item error
                        results.append(BulkUpdateResultItemDict(
                            index=original_idx,
                            context_id=context_id,
                            success=False,
                            updated_fields=updated_fields,  # Partial update succeeded
                            error=f'Updated but embedding failed: {str(emb_err)}',
                        ))
                        continue  # Skip the success result below

                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=True,
                    updated_fields=updated_fields,
                    error=None,
                ))

            except Exception as e:
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=str(e),
                ))

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(updates) - succeeded

        logger.info(f'Batch update completed: {succeeded}/{len(updates)} succeeded')

        return BulkUpdateResponseDict(
            success=failed == 0,
            total=len(updates),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=f'Updated {succeeded}/{len(updates)} entries successfully',
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch update: {e}')
        raise ToolError(f'Batch update failed: {format_exception_message(e)}') from e


async def delete_context_batch(
    context_ids: Annotated[
        list[int] | None,
        Field(description='Specific context IDs to delete'),
    ] = None,
    thread_ids: Annotated[
        list[str] | None,
        Field(description='Delete ALL entries in these threads'),
    ] = None,
    source: Annotated[
        Literal['user', 'agent'] | None,
        Field(description='Delete only entries from this source (combine with other criteria)'),
    ] = None,
    older_than_days: Annotated[
        int | None,
        Field(description='Delete entries older than N days', gt=0),
    ] = None,
    ctx: Context | None = None,
) -> BulkDeleteResponseDict:
    """Delete multiple context entries by various criteria. IRREVERSIBLE.

    Criteria can be combined for targeted deletion:
    - context_ids: Delete specific entries by ID
    - thread_ids: Delete all entries in specified threads
    - source: Filter by source ('user' or 'agent')
    - older_than_days: Delete entries created more than N days ago

    At least one criterion must be provided.
    Cascading delete removes associated tags, images, and embeddings.

    WARNING: This operation cannot be undone. Verify criteria before deletion.

    Returns:
        BulkDeleteResponseDict with success (bool), deleted_count (int),
        criteria_used (list of str), message (str).

    Raises:
        ToolError: If no criteria provided or deletion fails.
    """
    try:
        # Validate at least one criterion is provided
        if not any([context_ids, thread_ids, source, older_than_days]):
            raise ToolError(
                'At least one deletion criterion must be provided: '
                'context_ids, thread_ids, source, or older_than_days',
            )

        # Validate source if provided alone
        if source and not any([context_ids, thread_ids, older_than_days]):
            raise ToolError(
                'source filter must be combined with another criterion '
                '(context_ids, thread_ids, or older_than_days)',
            )

        if ctx:
            criteria_summary: list[str] = []
            if context_ids:
                criteria_summary.append(f'{len(context_ids)} IDs')
            if thread_ids:
                criteria_summary.append(f'{len(thread_ids)} threads')
            if source:
                criteria_summary.append(f'source={source}')
            if older_than_days:
                criteria_summary.append(f'older_than={older_than_days}d')
            await ctx.info(f'Batch delete with criteria: {", ".join(criteria_summary)}')

        repos = await ensure_repositories()

        # Delete embeddings first if context_ids are specified
        if settings.enable_semantic_search and context_ids:
            for cid in context_ids:
                try:
                    await repos.embeddings.delete(cid)
                except Exception as e:
                    logger.warning(f'Failed to delete embedding for context {cid}: {e}')

        # Execute batch delete through repository
        deleted_count, criteria_used = await repos.context.delete_contexts_batch(
            context_ids=context_ids,
            thread_ids=thread_ids,
            source=source,
            older_than_days=older_than_days,
        )

        logger.info(f'Batch delete completed: {deleted_count} entries removed')

        return BulkDeleteResponseDict(
            success=True,
            deleted_count=deleted_count,
            criteria_used=criteria_used,
            message=f'Successfully deleted {deleted_count} context entries',
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch delete: {e}')
        raise ToolError(f'Batch delete failed: {format_exception_message(e)}') from e
