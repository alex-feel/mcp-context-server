"""
Shared infrastructure and per-entry processing logic for MCP tools.

This module is INTERNAL to the app.tools package (underscore prefix).
It must NOT be re-exported via app.tools.__init__.py.

Contains:
- Connection error classification and transaction heartbeat utilities
- Concurrency-limited embedding and summary generation with timeout
- Image validation and normalization
- Store/update transaction execution helpers
- Response message builders for store and update operations

All functions in this module are consumed by:
- app.tools.context (non-batch CRUD operations)
- app.tools.batch (batch CRUD operations)
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer

import asyncpg
from fastmcp.exceptions import ToolError

from app.embeddings.retry import compute_embedding_total_timeout
from app.errors import format_exception_message
from app.repositories.embedding_repository import ChunkEmbedding
from app.settings import get_settings
from app.startup import MAX_IMAGE_SIZE_MB
from app.startup import MAX_TOTAL_SIZE_MB
from app.startup import get_chunking_service
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.summary.retry import compute_summary_total_timeout

logger = logging.getLogger(__name__)
settings = get_settings()

# ---------------------------------------------------------------------------
# Concurrency limiters for embedding/summary generation
# ---------------------------------------------------------------------------

# Concurrency limiter for embedding generation to prevent provider overload.
# Initialized lazily on first use to ensure correct event loop binding.
_embedding_semaphore: asyncio.Semaphore | None = None

# Concurrency limiter for summary generation to prevent provider overload.
# Initialized lazily on first use to ensure correct event loop binding.
_summary_semaphore: asyncio.Semaphore | None = None


def _get_embedding_semaphore() -> asyncio.Semaphore:
    """Get or create the embedding concurrency semaphore.

    Returns:
        asyncio.Semaphore configured with EMBEDDING_MAX_CONCURRENT limit.
    """
    global _embedding_semaphore
    if _embedding_semaphore is None:
        _embedding_semaphore = asyncio.Semaphore(settings.embedding.max_concurrent)
    return _embedding_semaphore


def _get_summary_semaphore() -> asyncio.Semaphore:
    """Get or create the summary concurrency semaphore."""
    global _summary_semaphore
    if _summary_semaphore is None:
        _summary_semaphore = asyncio.Semaphore(settings.summary.max_concurrent)
    return _summary_semaphore


# ---------------------------------------------------------------------------
# Embedding and summary generation with timeout
# ---------------------------------------------------------------------------


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
            f'Chunking service state: service={chunking_service}, '
            f'enabled={chunking_service.is_enabled if chunking_service else "N/A"}',
        )

        if chunking_service is not None and chunking_service.is_enabled:
            # Chunked embedding for long documents
            chunks = chunking_service.split_text(text)
            chunk_texts = [chunk.text for chunk in chunks]
            logger.info(f'Generating embeddings: text_len={len(text)}, chunks={len(chunks)}')
            embeddings = await embedding_provider.embed_documents(chunk_texts)
            logger.info(f'Embeddings generated: chunks={len(chunk_texts)}, embeddings={len(embeddings)}')

            return [
                ChunkEmbedding(
                    embedding=emb,
                    start_index=chunk.start_index,
                    end_index=chunk.end_index,
                )
                for emb, chunk in zip(embeddings, chunks, strict=True)
            ]
        # Single embedding (chunking disabled)
        logger.info(f'Generating single embedding: text_len={len(text)}')
        embedding = await embedding_provider.embed_query(text)
        logger.info('Single embedding generated')
        return [ChunkEmbedding(embedding=embedding, start_index=0, end_index=len(text))]

    except Exception as e:
        # CRITICAL: Embedding generation failed - this error must be raised
        # to prevent any data from being saved
        raise ToolError(f'Embedding generation failed: {format_exception_message(e)}') from e


async def generate_embeddings_with_timeout(text: str) -> list[ChunkEmbedding] | None:
    """Generate embeddings with concurrency limiting and total timeout.

    Wraps _generate_embeddings_for_text with:
    - Concurrency-limited access via embedding semaphore
    - Total timeout computed from retry settings
    - ToolError on timeout for clear client feedback

    Used by all four tools: store_context, update_context, store_context_batch,
    and update_context_batch.

    Args:
        text: Text content to generate embeddings for.

    Returns:
        List of ChunkEmbedding objects, or None if embedding provider
        is not configured.

    Raises:
        ToolError: If embedding generation times out or fails.
    """
    if get_embedding_provider() is None:
        return None

    total_timeout = compute_embedding_total_timeout()
    try:
        async with _get_embedding_semaphore():
            return await asyncio.wait_for(
                _generate_embeddings_for_text(text),
                timeout=total_timeout,
            )
    except TimeoutError:
        raise ToolError(
            f'Embedding generation exceeded total timeout ({total_timeout:.0f}s). '
            f'This may indicate the embedding provider is overloaded or unreachable.',
        ) from None


async def generate_summary_with_timeout(text: str, source: str) -> str | None:
    """Generate summary with concurrency limiting and total timeout.

    Wraps summary_provider.summarize() with:
    - Concurrency-limited access via summary semaphore
    - Total timeout computed from retry settings
    - ToolError on timeout for clear client feedback

    Used by all four tools: store_context, update_context,
    store_context_batch, and update_context_batch.

    Args:
        text: Text content to generate summary for.
        source: Source type ('user' or 'agent').

    Returns:
        Summary string, or None if summary provider is not configured.

    Raises:
        ToolError: If summary generation times out or fails.
    """
    summary_provider = get_summary_provider()
    if summary_provider is None:
        return None

    total_timeout = compute_summary_total_timeout()
    try:
        logger.info('Generating summary: text_len=%d', len(text))
        async with _get_summary_semaphore():
            result = await asyncio.wait_for(
                summary_provider.summarize(text, source),
                timeout=total_timeout,
            )
        # Normalize empty/whitespace-only summaries to None
        if not result.strip():
            logger.warning('Summary provider returned empty/whitespace-only response, treating as None')
            return None
        logger.info('Summary generated: text_len=%d, summary_len=%d', len(text), len(result))
        return result
    except TimeoutError:
        raise ToolError(
            f'Summary generation exceeded total timeout ({total_timeout:.0f}s). '
            f'This may indicate the summary provider is overloaded or unreachable.',
        ) from None


# ---------------------------------------------------------------------------
# Transaction utilities
# ---------------------------------------------------------------------------


async def transaction_heartbeat(txn: object) -> None:
    """Send lightweight heartbeat to prevent network intermediary idle timeout.

    Executes SELECT 1 on the connection to generate wire-protocol traffic,
    preventing NAT/firewall/proxy from classifying the connection as idle
    and closing it during long-running transactions.

    This is a defense-in-depth measure complementing TCP keepalive:
    - TCP keepalive operates at kernel level (probes every ~15s)
    - Heartbeat operates at application level (between sequential DB operations)
    - Together they provide maximum protection against intermediary timeouts

    For SQLite connections this is a no-op since SQLite does not use network
    connections and is not subject to intermediary idle timeouts.

    Args:
        txn: Transaction context (TransactionContext) providing connection and backend_type.
             Accepts object type for compatibility across backends.
    """
    backend_type = getattr(txn, 'backend_type', None)
    if backend_type != 'postgresql':
        return
    conn = getattr(txn, 'connection', None)
    if conn is None:
        return
    pg_conn = cast(asyncpg.Connection, conn)
    await pg_conn.execute('SELECT 1')


def is_connection_error(exc: Exception) -> bool:
    """Check if an exception indicates a connection-level failure.

    These errors are safe to retry because they indicate the connection
    was lost (not a logical/data error). Operations using deduplication
    (store_context) or idempotent updates are safe to retry.

    Args:
        exc: The exception to classify

    Returns:
        True if the exception is a connection error safe for retry
    """
    return isinstance(exc, (
        asyncpg.InterfaceError,
        asyncpg.ConnectionDoesNotExistError,
        ConnectionResetError,
        OSError,
    ))


# ---------------------------------------------------------------------------
# Image validation and normalization
# ---------------------------------------------------------------------------


def validate_and_normalize_images(
    images: list[dict[str, str]] | None,
    *,
    error_mode: Literal['raise', 'collect'] = 'raise',
) -> tuple[list[dict[str, str]], Literal['text', 'multimodal'], list[str]]:
    """Validate and normalize image attachments.

    Performs all image validation steps:
    - Checks required 'data' field presence
    - Rejects empty/whitespace data (prevents silent 0-byte storage)
    - Defaults mime_type to 'image/png' when not provided
    - Validates base64 encoding
    - Enforces per-image size limit (MAX_IMAGE_SIZE_MB)
    - Enforces total size limit (MAX_TOTAL_SIZE_MB)
    - Uses enumerate() for indexed error messages

    Args:
        images: List of image dicts with 'data' and optional 'mime_type' keys.
            None or empty list means no images.
        error_mode: 'raise' raises ToolError on first validation failure
            (for non-batch single-entry operations).
            'collect' accumulates errors and returns them
            (for batch operations where per-entry error reporting is needed).

    Returns:
        Tuple of (validated_images, content_type, errors):
        - validated_images: The validated image list (may have mime_type added)
        - content_type: 'multimodal' if images present, 'text' otherwise
        - errors: Empty list in 'raise' mode; list of error strings in 'collect' mode

    Raises:
        ToolError: In 'raise' mode, on the first validation failure.
    """
    if not images:
        return [], 'text', []

    errors: list[str] = []
    total_size: float = 0.0

    for idx, img in enumerate(images):
        # Validate required data field
        if 'data' not in img:
            msg = f'Image {idx} is missing required "data" field'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        img_data_str = img.get('data', '')
        if not img_data_str or not img_data_str.strip():
            msg = f'Image {idx} has empty "data" field'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # mime_type is optional - defaults to 'image/png' if not provided
        if 'mime_type' not in img:
            img['mime_type'] = 'image/png'

        # Validate base64 encoding
        try:
            image_binary = base64.b64decode(img_data_str)
        except Exception as e:
            if error_mode == 'raise':
                raise ToolError(f'Image {idx} has invalid base64 encoding: {format_exception_message(e)}') from None
            errors.append(f'Image {idx} has invalid base64 encoding')
            return images, 'text', errors

        # Validate image size
        image_size_mb = len(image_binary) / (1024 * 1024)

        if image_size_mb > MAX_IMAGE_SIZE_MB:
            msg = f'Image {idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        total_size += image_size_mb
        if total_size > MAX_TOTAL_SIZE_MB:
            msg = f'Total image size exceeds {MAX_TOTAL_SIZE_MB}MB limit'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

    logger.debug(f'Pre-validation passed for {len(images)} images, total size: {total_size:.2f}MB')
    return images, 'multimodal', []


# ---------------------------------------------------------------------------
# Response message builders
# ---------------------------------------------------------------------------


def build_store_response_message(
    *,
    action: str,
    image_count: int,
    embedding_generated: bool,
    embedding_stored: bool,
    summary_generated: bool,
    summary_preserved: bool,
) -> str:
    """Build a response message for a store operation.

    Constructs a human-readable message with parenthetical detail parts
    covering embedding status, summary status, and image count.

    Args:
        action: 'stored' or 'updated' (deduplication outcome)
        image_count: Number of validated images (0 suppresses image mention)
        embedding_generated: Whether embeddings were generated
        embedding_stored: Whether generated embeddings were stored to DB
        summary_generated: Whether a new summary was generated
        summary_preserved: Whether an existing summary was reused

    Returns:
        Formatted message string like 'Context stored (embedding generated, summary generated)'.
    """
    parts: list[str] = []

    if embedding_generated and not embedding_stored:
        parts.append('embedding generated but not stored - duplicate')
    elif embedding_stored:
        parts.append('embedding generated')

    if summary_generated:
        parts.append('summary generated')
    elif summary_preserved:
        parts.append('summary preserved')

    # Suppress "with 0 images" when no images
    base = f'Context {action} with {image_count} images' if image_count > 0 else f'Context {action}'

    # Single consolidated parenthetical
    return f'{base} ({", ".join(parts)})' if parts else base


def build_update_response_message(
    *,
    updated_fields_count: int,
    embedding_generated: bool,
    summary_generated: bool,
    summary_cleared: bool,
) -> str:
    """Build a response message for an update operation.

    Args:
        updated_fields_count: Number of fields updated
        embedding_generated: Whether embeddings were regenerated
        summary_generated: Whether summary was regenerated
        summary_cleared: Whether existing summary was cleared

    Returns:
        Formatted message string.
    """
    parts: list[str] = []
    if embedding_generated:
        parts.append('embedding regenerated')
    if summary_generated:
        parts.append('summary regenerated')
    elif summary_cleared:
        parts.append('summary cleared')

    base = f'Successfully updated {updated_fields_count} field(s)'
    return f'{base} ({", ".join(parts)})' if parts else base


def build_batch_store_response_message(
    *,
    succeeded: int,
    total: int,
    embeddings_generated_count: int,
    embeddings_stored_count: int,
    summaries_generated_count: int,
    summaries_preserved_count: int,
) -> str:
    """Build a response message for a batch store operation.

    Args:
        succeeded: Number of successfully stored entries
        total: Total number of entries in the batch
        embeddings_generated_count: Number of entries with generated embeddings
        embeddings_stored_count: Number of entries where embeddings were stored
        summaries_generated_count: Number of entries with generated summaries
        summaries_preserved_count: Number of entries with preserved summaries

    Returns:
        Formatted batch message string.
    """
    parts: list[str] = []
    if embeddings_generated_count > 0:
        not_stored = embeddings_generated_count - embeddings_stored_count
        if not_stored > 0:
            parts.append(f'embeddings generated ({not_stored} not stored - duplicates)')
        else:
            parts.append('embeddings generated')
    if summaries_generated_count > 0:
        parts.append('summaries generated')
    if summaries_preserved_count > 0:
        parts.append('summaries preserved')
    base = f'Stored {succeeded}/{total} entries successfully'
    return f'{base} ({", ".join(parts)})' if parts else base


def build_batch_update_response_message(
    *,
    succeeded: int,
    total: int,
    embeddings_generated_count: int,
    summaries_generated_count: int,
    summaries_cleared_count: int,
) -> str:
    """Build a response message for a batch update operation.

    Args:
        succeeded: Number of successfully updated entries
        total: Total number of entries in the batch
        embeddings_generated_count: Number of entries with regenerated embeddings
        summaries_generated_count: Number of entries with regenerated summaries
        summaries_cleared_count: Number of entries with cleared summaries

    Returns:
        Formatted batch message string.
    """
    parts: list[str] = []
    if embeddings_generated_count > 0:
        parts.append('embeddings regenerated')
    if summaries_generated_count > 0:
        parts.append('summaries regenerated')
    if summaries_cleared_count > 0:
        parts.append('summaries cleared')
    base = f'Updated {succeeded}/{total} entries successfully'
    return f'{base} ({", ".join(parts)})' if parts else base


# ---------------------------------------------------------------------------
# Transaction execution helpers for store and update operations
# ---------------------------------------------------------------------------


async def execute_store_in_transaction(
    repos: RepositoryContainer,
    txn: TransactionContext,
    *,
    thread_id: str,
    source: str,
    content_type: str,
    text_content: str,
    metadata_str: str | None,
    summary: str | None,
    tags: list[str] | None,
    validated_images: list[dict[str, str]],
    chunk_embeddings: list[ChunkEmbedding] | None,
    embedding_model: str,
) -> tuple[int, bool, bool]:
    """Execute all store operations within an existing transaction.

    Performs deduplication-aware storage of a single context entry:
    1. Store entry with deduplication (store_with_deduplication)
    2. Store/replace tags based on dedup outcome
    3. Store/replace images based on dedup outcome
    4. Store embeddings (skip if dedup + embeddings already exist)
    5. Track embedding_stored flag for response message parity

    Args:
        repos: Repository container with context, tags, images, embeddings repos.
        txn: Active transaction context.
        thread_id: Thread identifier.
        source: 'user' or 'agent'.
        content_type: 'text' or 'multimodal'.
        text_content: The text content to store.
        metadata_str: JSON-serialized metadata or None.
        summary: Generated/preserved summary or None.
        tags: Tag list or None.
        validated_images: Validated image list (may be empty).
        chunk_embeddings: Generated embeddings or None.
        embedding_model: Model name for embedding storage.

    Returns:
        Tuple of (context_id, was_updated, embedding_stored):
        - context_id: ID of stored/updated entry
        - was_updated: True if deduplication updated existing entry
        - embedding_stored: True if embeddings were written to DB

    Raises:
        ToolError: If store_with_deduplication fails (returns falsy context_id).
    """
    # Store context entry with deduplication
    context_id, was_updated = await repos.context.store_with_deduplication(
        thread_id=thread_id,
        source=source,
        content_type=content_type,
        text_content=text_content,
        metadata=metadata_str,
        summary=summary,
        txn=txn,
    )

    if not context_id:
        raise ToolError('Failed to store context')

    # Heartbeat: keep connection alive between sequential operations
    await transaction_heartbeat(txn)

    # Store or replace tags depending on deduplication outcome
    if tags:
        if was_updated:
            await repos.tags.replace_tags_for_context(context_id, tags, txn=txn)
        else:
            await repos.tags.store_tags(context_id, tags, txn=txn)

    # Store or replace images depending on deduplication outcome
    if validated_images:
        if was_updated:
            await repos.images.replace_images_for_context(
                context_id, validated_images, txn=txn,
            )
        else:
            await repos.images.store_images(context_id, validated_images, txn=txn)

    # Store embeddings only if:
    # 1. New entry (not was_updated) - always store, OR
    # 2. Deduplicated entry (was_updated) but no embeddings exist yet
    # Skip if: Deduplicated entry AND embeddings already exist
    embedding_stored = False
    if chunk_embeddings is not None:
        # Heartbeat before potentially long embedding storage
        await transaction_heartbeat(txn)

        should_store = True
        if was_updated:
            embedding_exists = await repos.embeddings.exists(context_id)
            should_store = not embedding_exists
            if not should_store:
                logger.debug(
                    'Skipping embedding storage for deduplicated context %d '
                    '(embeddings already exist)',
                    context_id,
                )

        if should_store:
            await repos.embeddings.store_chunked(
                context_id=context_id,
                chunk_embeddings=chunk_embeddings,
                model=embedding_model,
                txn=txn,
                upsert=was_updated,
            )
            embedding_stored = True

    return context_id, was_updated, embedding_stored


async def execute_update_in_transaction(
    repos: RepositoryContainer,
    txn: TransactionContext,
    *,
    context_id: int,
    text: str | None,
    metadata: dict[str, Any] | None,
    metadata_patch: dict[str, Any] | None,
    summary: str | None,
    clear_summary: bool,
    tags: list[str] | None,
    images: list[dict[str, str]] | None,
    validated_images: list[dict[str, str]],
    chunk_embeddings: list[ChunkEmbedding] | None,
    embedding_model: str,
) -> tuple[list[str], bool]:
    """Execute all update operations within an existing transaction.

    Performs a complete update of a single context entry:
    1. Update text/metadata/summary via update_context_entry (CHECK success)
    2. Apply metadata_patch via patch_metadata (CHECK success)
    3. Replace tags if provided
    4. Replace images if provided (update content_type accordingly)
    5. Auto-correct content_type based on actual image presence
    6. Delete old + store new embeddings if text changed

    Args:
        repos: Repository container.
        txn: Active transaction context.
        context_id: ID of entry to update.
        text: New text content or None.
        metadata: Full metadata replacement or None.
        metadata_patch: Metadata merge patch or None.
        summary: New summary or None.
        clear_summary: Whether to clear existing summary.
        tags: New tags or None.
        images: Raw images parameter from caller (for None vs empty detection).
        validated_images: Validated image list (empty if images is None).
        chunk_embeddings: Regenerated embeddings or None.
        embedding_model: Model name for embedding storage.

    Returns:
        Tuple of (updated_fields, summary_cleared):
        - updated_fields: List of field names that were updated
        - summary_cleared: True if summary was cleared (for response message)

    Raises:
        ToolError: If update_context_entry or patch_metadata returns success=False.
    """
    updated_fields: list[str] = []

    # Update text content and/or metadata (full replacement) if provided
    if text is not None or metadata is not None:
        metadata_str: str | None = None
        if metadata is not None:
            metadata_str = json.dumps(metadata, ensure_ascii=False)

        success, fields = await repos.context.update_context_entry(
            context_id=context_id,
            text_content=text,
            metadata=metadata_str,
            summary=summary,
            clear_summary=clear_summary,
            txn=txn,
        )

        if not success:
            raise ToolError(f'Failed to update context entry {context_id}')

        updated_fields.extend(fields)

    # Apply metadata patch (partial update) if provided
    if metadata_patch is not None:
        success, fields = await repos.context.patch_metadata(
            context_id=context_id,
            patch=metadata_patch,
            txn=txn,
        )

        if not success:
            raise ToolError(f'Failed to patch metadata for context {context_id}')

        updated_fields.extend(fields)

    # Heartbeat between operation groups
    await transaction_heartbeat(txn)

    # Replace tags if provided
    if tags is not None:
        await repos.tags.replace_tags_for_context(context_id, tags, txn=txn)
        updated_fields.append('tags')

    # Replace images if provided
    if images is not None:
        if len(images) == 0:
            await repos.images.replace_images_for_context(context_id, [], txn=txn)
            await repos.context.update_content_type(context_id, 'text', txn=txn)
            updated_fields.extend(['images', 'content_type'])
        else:
            await repos.images.replace_images_for_context(
                context_id, validated_images, txn=txn,
            )
            await repos.context.update_content_type(
                context_id, 'multimodal', txn=txn,
            )
            updated_fields.extend(['images', 'content_type'])

    # Auto-correct content_type when images not explicitly changed
    if images is None and (text is not None or metadata is not None):
        image_count = await repos.images.count_images_for_context(context_id)
        current_content_type = 'multimodal' if image_count > 0 else 'text'
        stored_content_type = await repos.context.get_content_type(context_id)
        if stored_content_type != current_content_type:
            await repos.context.update_content_type(
                context_id, current_content_type, txn=txn,
            )
            updated_fields.append('content_type')

    # Store embeddings if text changed and new embeddings were generated
    if chunk_embeddings is not None:
        await transaction_heartbeat(txn)
        await repos.embeddings.delete_all_chunks(context_id, txn=txn)
        await repos.embeddings.store_chunked(
            context_id=context_id,
            chunk_embeddings=chunk_embeddings,
            model=embedding_model,
            txn=txn,
        )
        updated_fields.append('embedding')

    return updated_fields, clear_summary
