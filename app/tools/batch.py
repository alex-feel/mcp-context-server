"""
Batch operations for MCP tools.

This module contains tools for bulk context management:
- store_context_batch: Store multiple entries in one operation
- update_context_batch: Update multiple entries in one operation
- delete_context_batch: Delete entries by various criteria

Embedding-First Pattern:
This module implements atomic embedding + data storage for batch operations.
When embeddings are enabled:
1. ALL embeddings are generated FIRST (outside any database transaction)
2. If ANY embedding generation fails in atomic mode, NO data is saved
3. If ALL embeddings succeed, ALL database operations occur in a SINGLE atomic transaction
"""

import asyncio
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
from app.tools.context import is_connection_error
from app.tools.context import transaction_heartbeat
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
    """Store multiple context entries with atomic embedding + data storage.

    EMBEDDING-FIRST PATTERN:
    - atomic=True: ALL embeddings generated FIRST. If ANY fails, NO data is saved.
                   ALL database operations occur in a SINGLE atomic transaction.
    - atomic=False: Each entry processed independently with its own embedding-first pattern.

    Batch processing is significantly faster than individual store_context calls
    when storing many entries. Use for migrations, imports, or bulk operations.

    Size limits:
    - Maximum 100 entries per batch
    - Image limits per entry: 10MB each, 100MB total
    - Standard tag normalization (lowercase)

    Returns:
        BulkStoreResponseDict with success (bool), total (int), succeeded (int),
        failed (int), results (list of index, success, context_id, error), message (str).

    Raises:
        ToolError: If validation fails, embedding generation fails (atomic), or batch operation fails.
    """
    # Import types at function level to avoid linter removing unused module-level imports
    from app.repositories.embedding_repository import ChunkEmbedding
    from app.startup import get_chunking_service

    try:
        if ctx:
            await ctx.info(f'Batch storing {len(entries)} context entries (atomic={atomic})')

        repos = await ensure_repositories()

        # === PHASE 1: Validate all entries before processing ===
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

        # In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at index {first_error[0]}: {first_error[1]}',
            )

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

        if not validated_entries:
            # All entries failed validation
            return BulkStoreResponseDict(
                success=False,
                total=len(entries),
                succeeded=0,
                failed=len(entries),
                results=results,
                message='All entries failed validation',
            )

        # === PHASE 2: Generate ALL Embeddings FIRST (Outside Transaction) ===
        # CRITICAL: Embedding generation happens BEFORE any database modifications.
        # If it fails in atomic mode, NO data is saved.
        embedding_provider = get_embedding_provider()
        chunking_service = get_chunking_service()

        # Mapping: validated_entry_index -> list[ChunkEmbedding] | None
        entry_embeddings: dict[int, list[ChunkEmbedding] | None] = {}
        embedding_errors: list[tuple[int, str]] = []  # (original_idx, error_message)

        if embedding_provider is not None:
            for ve_idx, entry in enumerate(validated_entries):
                original_idx = entry['index']
                text_content = entry['text_content']

                try:
                    if chunking_service is not None and chunking_service.is_enabled:
                        # Chunked embedding for long documents
                        chunks = chunking_service.split_text(text_content)
                        chunk_texts = [chunk.text for chunk in chunks]
                        embeddings = await embedding_provider.embed_documents(chunk_texts)

                        chunk_embeddings = [
                            ChunkEmbedding(
                                embedding=emb,
                                start_index=chunk.start_index,
                                end_index=chunk.end_index,
                            )
                            for emb, chunk in zip(embeddings, chunks, strict=True)
                        ]
                        entry_embeddings[ve_idx] = chunk_embeddings
                    else:
                        # Single embedding (chunking disabled)
                        embedding = await embedding_provider.embed_query(text_content)
                        entry_embeddings[ve_idx] = [
                            ChunkEmbedding(
                                embedding=embedding,
                                start_index=0,
                                end_index=len(text_content),
                            ),
                        ]
                except Exception as emb_err:
                    logger.error(f'Embedding generation failed at index {original_idx}: {emb_err}')
                    if atomic:
                        # CRITICAL: In atomic mode, embedding failure fails the ENTIRE batch
                        # NO data has been saved yet, so we can safely return error
                        raise ToolError(
                            f'Embedding generation failed at index {original_idx}: '
                            f'{format_exception_message(emb_err)}. No data was saved.',
                        ) from emb_err
                    # Non-atomic mode: record error, skip this entry
                    embedding_errors.append((original_idx, f'Embedding generation failed: {str(emb_err)}'))
                    entry_embeddings[ve_idx] = None  # Mark as failed
        else:
            # No embedding provider - all entries get None
            for ve_idx in range(len(validated_entries)):
                entry_embeddings[ve_idx] = None

        # In non-atomic mode, add embedding errors to results
        if not atomic:
            for original_idx, error in embedding_errors:
                results.append(BulkStoreResultItemDict(
                    index=original_idx,
                    success=False,
                    context_id=None,
                    error=error,
                ))

        # Filter out entries with embedding failures in non-atomic mode
        if not atomic and embedding_errors:
            failed_indices = {idx for idx, _ in embedding_errors}
            validated_entries_filtered = [
                (ve_idx, e) for ve_idx, e in enumerate(validated_entries)
                if e['index'] not in failed_indices
            ]
        else:
            validated_entries_filtered = list(enumerate(validated_entries))

        if not validated_entries_filtered:
            # All entries failed (validation or embedding)
            results.sort(key=operator.itemgetter('index'))
            return BulkStoreResponseDict(
                success=False,
                total=len(entries),
                succeeded=0,
                failed=len(entries),
                results=results,
                message='All entries failed validation or embedding generation',
            )

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend

        if atomic:
            # ATOMIC MODE: All entries in a single transaction with retry
            max_retries = 2

            for attempt in range(max_retries + 1):
                try:
                    results_attempt: list[BulkStoreResultItemDict] = []

                    async with backend.begin_transaction() as txn:
                        for idx, (ve_idx, entry) in enumerate(validated_entries_filtered):
                            original_idx = entry['index']

                            # Heartbeat between entries (skip first)
                            if idx > 0:
                                await transaction_heartbeat(txn)

                            # Store context entry with deduplication
                            context_id, was_updated = await repos.context.store_with_deduplication(
                                thread_id=entry['thread_id'],
                                source=entry['source'],
                                content_type=entry['content_type'],
                                text_content=entry['text_content'],
                                metadata=entry['metadata'],
                                txn=txn,
                            )

                            if not context_id:
                                raise ToolError(f'Failed to store context at index {original_idx}')

                            # Store tags if provided
                            if entry.get('tags'):
                                await repos.tags.store_tags(context_id, entry['tags'], txn=txn)

                            # Store images if provided
                            if entry.get('images'):
                                await repos.images.store_images(context_id, entry['images'], txn=txn)

                            # Store embeddings only if needed (skip for deduplicated entries with existing embeddings)
                            entry_chunk_embeddings = entry_embeddings.get(ve_idx)
                            if entry_chunk_embeddings is not None:
                                should_store_embedding = True
                                if was_updated:
                                    embedding_exists = await repos.embeddings.exists(context_id)
                                    should_store_embedding = not embedding_exists
                                    if not should_store_embedding:
                                        logger.debug(
                                            f'Skipping embedding storage for deduplicated context {context_id} '
                                            f'at index {original_idx} (embeddings already exist)',
                                        )

                                if should_store_embedding:
                                    await repos.embeddings.store_chunked(
                                        context_id=context_id,
                                        chunk_embeddings=entry_chunk_embeddings,
                                        model=settings.embedding.model,
                                        txn=txn,
                                        upsert=was_updated,
                                    )

                            results_attempt.append(BulkStoreResultItemDict(
                                index=original_idx,
                                success=True,
                                context_id=context_id,
                                error=None,
                            ))

                        # COMMIT happens here - all or nothing

                    # Transaction committed successfully
                    results.extend(results_attempt)
                    break

                except ToolError:
                    raise  # Logical error — do not retry
                except Exception as e:
                    if is_connection_error(e) and attempt < max_retries:
                        delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                        logger.warning(
                            'Atomic batch store transaction failed, retrying in %.1fs '
                            '(attempt %d/%d): %s',
                            delay, attempt + 1, max_retries, e,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise  # Non-connection error or max retries exceeded
        else:
            # NON-ATOMIC MODE: Each entry in its own transaction
            for ve_idx, entry in validated_entries_filtered:
                original_idx = entry['index']

                try:
                    async with backend.begin_transaction() as txn:
                        # Store context entry with deduplication
                        context_id, was_updated = await repos.context.store_with_deduplication(
                            thread_id=entry['thread_id'],
                            source=entry['source'],
                            content_type=entry['content_type'],
                            text_content=entry['text_content'],
                            metadata=entry['metadata'],
                            txn=txn,
                        )

                        if not context_id:
                            raise ToolError(f'Failed to store context at index {original_idx}')

                        # Heartbeat between sub-operations
                        await transaction_heartbeat(txn)

                        # Store tags if provided
                        if entry.get('tags'):
                            await repos.tags.store_tags(context_id, entry['tags'], txn=txn)

                        # Store images if provided
                        if entry.get('images'):
                            await repos.images.store_images(context_id, entry['images'], txn=txn)

                        # Store embeddings only if needed (skip for deduplicated entries with existing embeddings)
                        entry_chunk_embeddings = entry_embeddings.get(ve_idx)
                        if entry_chunk_embeddings is not None:
                            # Heartbeat before potentially long embedding storage
                            await transaction_heartbeat(txn)
                            should_store_embedding = True
                            if was_updated:
                                embedding_exists = await repos.embeddings.exists(context_id)
                                should_store_embedding = not embedding_exists
                                if not should_store_embedding:
                                    logger.debug(
                                        f'Skipping embedding storage for deduplicated context {context_id} '
                                        f'at index {original_idx} (embeddings already exist)',
                                    )

                            if should_store_embedding:
                                await repos.embeddings.store_chunked(
                                    context_id=context_id,
                                    chunk_embeddings=entry_chunk_embeddings,
                                    model=settings.embedding.model,
                                    txn=txn,
                                    upsert=was_updated,
                                )

                        # COMMIT happens here for this entry

                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=True,
                        context_id=context_id,
                        error=None,
                    ))
                except Exception as e:
                    logger.error(f'Failed to store entry at index {original_idx}: {e}')
                    results.append(BulkStoreResultItemDict(
                        index=original_idx,
                        success=False,
                        context_id=None,
                        error=str(e),
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
            message=f'Stored {succeeded}/{len(entries)} entries successfully'
            + (' (embeddings generated)' if embedding_provider is not None else ''),
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
    """Update multiple context entries with atomic embedding + data storage.

    EMBEDDING-FIRST PATTERN:
    - atomic=True: ALL embeddings generated FIRST for text changes. If ANY fails, NO data is modified.
                   ALL database operations occur in a SINGLE atomic transaction.
    - atomic=False: Each update processed independently with its own embedding-first pattern.

    Similar semantics to update_context but for multiple entries:
    - Each update is identified by context_id
    - Only provided fields are modified
    - metadata vs metadata_patch are mutually exclusive per entry
    - Tags and images use replacement semantics

    Returns:
        BulkUpdateResponseDict with success (bool), total (int), succeeded (int),
        failed (int), results (list of index, context_id, success, updated_fields, error),
        message (str).

    Raises:
        ToolError: If validation fails, embedding generation fails (atomic), or batch operation fails.
    """
    # Import types at function level to avoid linter removing unused module-level imports
    from app.repositories.embedding_repository import ChunkEmbedding
    from app.startup import get_chunking_service

    try:
        if ctx:
            await ctx.info(f'Batch updating {len(updates)} context entries (atomic={atomic})')

        repos = await ensure_repositories()

        # === PHASE 1: Validate all updates before processing ===
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

        # In atomic mode, fail fast if any validation errors
        if atomic and validation_errors:
            first_error = validation_errors[0]
            raise ToolError(
                f'Validation failed for {len(validation_errors)} entries. '
                f'First error at context_id {first_error[1]}: {first_error[2]}',
            )

        # Build results list including validation errors
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

        if not validated_updates:
            # All updates failed validation
            return BulkUpdateResponseDict(
                success=False,
                total=len(updates),
                succeeded=0,
                failed=len(updates),
                results=results,
                message='All updates failed validation',
            )

        # === PHASE 1.5: Check all entries exist (fail fast in atomic mode) ===
        existence_errors: list[tuple[int, int, str]] = []  # (index, context_id, error)

        for update in validated_updates:
            original_idx = update['index']
            context_id = update['context_id']

            exists = await repos.context.check_entry_exists(context_id)
            if not exists:
                if atomic:
                    raise ToolError(f'Context entry {context_id} not found at index {original_idx}')
                existence_errors.append((original_idx, context_id, f'Context entry {context_id} not found'))

        # In non-atomic mode, add existence errors to results
        if not atomic:
            for original_idx, context_id, error in existence_errors:
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=error,
                ))

        # Filter out non-existent entries in non-atomic mode
        if not atomic and existence_errors:
            missing_ctx_ids = {ctx_id for _, ctx_id, _ in existence_errors}
            validated_updates_filtered = [
                (vu_idx, u) for vu_idx, u in enumerate(validated_updates)
                if u['context_id'] not in missing_ctx_ids
            ]
        else:
            validated_updates_filtered = list(enumerate(validated_updates))

        if not validated_updates_filtered:
            # All entries failed (validation or existence)
            results.sort(key=operator.itemgetter('index'))
            return BulkUpdateResponseDict(
                success=False,
                total=len(updates),
                succeeded=0,
                failed=len(updates),
                results=results,
                message='All updates failed validation or entry not found',
            )

        # === PHASE 2: Generate ALL Embeddings FIRST (Outside Transaction) ===
        # CRITICAL: Embedding generation happens BEFORE any database modifications.
        # If it fails in atomic mode, NO data is modified - original data is preserved.
        embedding_provider = get_embedding_provider()
        chunking_service = get_chunking_service()

        # Mapping: validated_update_index -> list[ChunkEmbedding] | None
        update_embeddings: dict[int, list[ChunkEmbedding] | None] = {}
        embedding_errors: list[tuple[int, int, str]] = []  # (original_idx, context_id, error_message)

        if embedding_provider is not None:
            for vu_idx, update in validated_updates_filtered:
                original_idx = update['index']
                context_id = update['context_id']
                text_content = update.get('text')

                # Only generate embedding if text is being changed
                if text_content is None:
                    update_embeddings[vu_idx] = None  # No text change, no embedding needed
                    continue

                try:
                    if chunking_service is not None and chunking_service.is_enabled:
                        # Chunked embedding for long documents
                        chunks = chunking_service.split_text(text_content)
                        chunk_texts = [chunk.text for chunk in chunks]
                        embeddings = await embedding_provider.embed_documents(chunk_texts)

                        chunk_embeddings = [
                            ChunkEmbedding(
                                embedding=emb,
                                start_index=chunk.start_index,
                                end_index=chunk.end_index,
                            )
                            for emb, chunk in zip(embeddings, chunks, strict=True)
                        ]
                        update_embeddings[vu_idx] = chunk_embeddings
                    else:
                        # Single embedding (chunking disabled)
                        embedding = await embedding_provider.embed_query(text_content)
                        update_embeddings[vu_idx] = [
                            ChunkEmbedding(
                                embedding=embedding,
                                start_index=0,
                                end_index=len(text_content),
                            ),
                        ]
                except Exception as emb_err:
                    logger.error(f'Embedding generation failed for context {context_id} at index {original_idx}: {emb_err}')
                    if atomic:
                        # CRITICAL: In atomic mode, embedding failure fails the ENTIRE batch
                        # NO data has been modified yet, so we can safely return error
                        raise ToolError(
                            f'Embedding generation failed for context {context_id} at index {original_idx}: '
                            f'{format_exception_message(emb_err)}. No data was modified.',
                        ) from emb_err
                    # Non-atomic mode: record error, skip this entry
                    embedding_errors.append((original_idx, context_id, f'Embedding generation failed: {str(emb_err)}'))
                    update_embeddings[vu_idx] = None  # Mark as failed
        else:
            # No embedding provider - all entries get None
            for vu_idx, _ in validated_updates_filtered:
                update_embeddings[vu_idx] = None

        # In non-atomic mode, add embedding errors to results
        if not atomic:
            for original_idx, context_id, error in embedding_errors:
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=error,
                ))

        # Filter out entries with embedding failures in non-atomic mode
        if not atomic and embedding_errors:
            failed_ctx_ids = {ctx_id for _, ctx_id, _ in embedding_errors}
            validated_updates_final = [
                (vu_idx, u) for vu_idx, u in validated_updates_filtered
                if u['context_id'] not in failed_ctx_ids
            ]
        else:
            validated_updates_final = validated_updates_filtered

        if not validated_updates_final:
            # All entries failed (validation, existence, or embedding)
            results.sort(key=operator.itemgetter('index'))
            return BulkUpdateResponseDict(
                success=False,
                total=len(updates),
                succeeded=0,
                failed=len(updates),
                results=results,
                message='All updates failed validation, entry not found, or embedding generation',
            )

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend

        if atomic:
            # ATOMIC MODE: All updates in a single transaction with retry
            max_retries = 2

            for attempt in range(max_retries + 1):
                try:
                    results_attempt: list[BulkUpdateResultItemDict] = []

                    async with backend.begin_transaction() as txn:
                        for idx, (vu_idx, update) in enumerate(validated_updates_final):
                            original_idx = update['index']
                            context_id = update['context_id']
                            updated_fields: list[str] = []

                            # Heartbeat between entries (skip first)
                            if idx > 0:
                                await transaction_heartbeat(txn)

                            # Update text and/or metadata (full replacement)
                            if update.get('text') is not None or update.get('metadata') is not None:
                                metadata_str = None
                                if update.get('metadata') is not None:
                                    metadata_str = json.dumps(update['metadata'], ensure_ascii=False)

                                success, fields = await repos.context.update_context_entry(
                                    context_id=context_id,
                                    text_content=update.get('text'),
                                    metadata=metadata_str,
                                    txn=txn,
                                )
                                if success:
                                    updated_fields.extend(fields)

                            # Apply metadata patch if provided
                            if update.get('metadata_patch') is not None:
                                success, fields = await repos.context.patch_metadata(
                                    context_id=context_id,
                                    patch=update['metadata_patch'],
                                    txn=txn,
                                )
                                if success:
                                    updated_fields.extend(fields)

                            # Replace tags if provided
                            if update.get('tags') is not None:
                                await repos.tags.replace_tags_for_context(context_id, update['tags'], txn=txn)
                                updated_fields.append('tags')

                            # Replace images if provided
                            if update.get('images') is not None:
                                update_images = update['images']
                                if len(update_images) == 0:
                                    await repos.images.replace_images_for_context(context_id, [], txn=txn)
                                    await repos.context.update_content_type(context_id, 'text', txn=txn)
                                    updated_fields.extend(['images', 'content_type'])
                                else:
                                    await repos.images.replace_images_for_context(context_id, update_images, txn=txn)
                                    await repos.context.update_content_type(context_id, 'multimodal', txn=txn)
                                    updated_fields.extend(['images', 'content_type'])

                            # Store embeddings (only if text was changed and embedding exists)
                            entry_chunk_embeddings = update_embeddings.get(vu_idx)
                            if entry_chunk_embeddings is not None:
                                # Heartbeat before embedding operations
                                await transaction_heartbeat(txn)

                                # Delete existing chunks first (within the same transaction)
                                await repos.embeddings.delete_all_chunks(context_id, txn=txn)

                                # Store new embeddings
                                await repos.embeddings.store_chunked(
                                    context_id=context_id,
                                    chunk_embeddings=entry_chunk_embeddings,
                                    model=settings.embedding.model,
                                    txn=txn,
                                )
                                updated_fields.append('embedding')

                            results_attempt.append(BulkUpdateResultItemDict(
                                index=original_idx,
                                context_id=context_id,
                                success=True,
                                updated_fields=updated_fields,
                                error=None,
                            ))

                        # COMMIT happens here - all or nothing

                    # Transaction committed successfully
                    results.extend(results_attempt)
                    break

                except ToolError:
                    raise  # Logical error — do not retry
                except Exception as e:
                    if is_connection_error(e) and attempt < max_retries:
                        delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                        logger.warning(
                            'Atomic batch update transaction failed, retrying in %.1fs '
                            '(attempt %d/%d): %s',
                            delay, attempt + 1, max_retries, e,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise  # Non-connection error or max retries exceeded
        else:
            # NON-ATOMIC MODE: Each update in its own transaction
            for vu_idx, update in validated_updates_final:
                original_idx = update['index']
                context_id = update['context_id']

                try:
                    updated_fields_list: list[str] = []

                    async with backend.begin_transaction() as txn:
                        # Update text and/or metadata (full replacement)
                        if update.get('text') is not None or update.get('metadata') is not None:
                            metadata_str = None
                            if update.get('metadata') is not None:
                                metadata_str = json.dumps(update['metadata'], ensure_ascii=False)

                            success, fields = await repos.context.update_context_entry(
                                context_id=context_id,
                                text_content=update.get('text'),
                                metadata=metadata_str,
                                txn=txn,
                            )
                            if success:
                                updated_fields_list.extend(fields)

                        # Apply metadata patch if provided
                        if update.get('metadata_patch') is not None:
                            success, fields = await repos.context.patch_metadata(
                                context_id=context_id,
                                patch=update['metadata_patch'],
                                txn=txn,
                            )
                            if success:
                                updated_fields_list.extend(fields)

                        # Replace tags if provided
                        if update.get('tags') is not None:
                            await repos.tags.replace_tags_for_context(context_id, update['tags'], txn=txn)
                            updated_fields_list.append('tags')

                        # Replace images if provided
                        if update.get('images') is not None:
                            update_images = update['images']
                            if len(update_images) == 0:
                                await repos.images.replace_images_for_context(context_id, [], txn=txn)
                                await repos.context.update_content_type(context_id, 'text', txn=txn)
                                updated_fields_list.extend(['images', 'content_type'])
                            else:
                                await repos.images.replace_images_for_context(context_id, update_images, txn=txn)
                                await repos.context.update_content_type(context_id, 'multimodal', txn=txn)
                                updated_fields_list.extend(['images', 'content_type'])

                        # Store embeddings
                        entry_chunk_embeddings = update_embeddings.get(vu_idx)
                        if entry_chunk_embeddings is not None:
                            # Heartbeat before embedding operations
                            await transaction_heartbeat(txn)

                            # Delete existing chunks first (within the same transaction)
                            await repos.embeddings.delete_all_chunks(context_id, txn=txn)

                            # Store new embeddings
                            await repos.embeddings.store_chunked(
                                context_id=context_id,
                                chunk_embeddings=entry_chunk_embeddings,
                                model=settings.embedding.model,
                                txn=txn,
                            )
                            updated_fields_list.append('embedding')

                        # COMMIT happens here for this update

                    results.append(BulkUpdateResultItemDict(
                        index=original_idx,
                        context_id=context_id,
                        success=True,
                        updated_fields=updated_fields_list,
                        error=None,
                    ))
                except Exception as e:
                    logger.error(f'Failed to update entry at index {original_idx}: {e}')
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
            message=f'Updated {succeeded}/{len(updates)} entries successfully'
            + (' (embeddings regenerated)' if embedding_provider is not None else ''),
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
        if settings.semantic_search.enabled and context_ids:
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
