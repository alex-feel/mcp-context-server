"""
Context CRUD operations for MCP tools.

This module contains the core context management tools:
- store_context: Store a new context entry
- get_context_by_ids: Retrieve specific context entries by ID
- update_context: Update an existing context entry
- delete_context: Delete context entries

Generation-First Transactional Integrity:
This module implements atomic generation + data storage. When embedding or summary
generation is enabled, all generation runs OUTSIDE any database transaction via
run_generation() (app.tools._shared), which drives three concurrent legs:
1. embed_then_compress -- embedding followed by TurboQuant compression (abort-mandatory)
2. the flat document summary (abort-mandatory)
3. the index_tree per-node summaries (never-raise), started after the flat summary
   completes and overlapped with the embedding leg
If either abort-mandatory leg fails after exhausting its retries (managed by
app/embeddings/retry.py and app/summary/retry.py), the failures are collected into a
single deterministic ToolError and NO data is saved; the never-raise node leg is then
cancelled and awaited before the transaction opens. A node-summary failure or timeout
never aborts the store. Only when both abort-mandatory legs succeed do ALL database
operations occur in a SINGLE atomic transaction.

Infrastructure functions (embedding/summary generation, transaction heartbeat,
connection error classification, image validation, response message builders) are
in app.tools._shared -- the single source of truth for logic shared with batch.py.
"""

import asyncio
import json
import logging
from typing import Annotated
from typing import Literal
from typing import cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.errors import format_exception_message
from app.ids import resolve_or_normalize_id
from app.ids import resolve_or_normalize_ids
from app.repositories.base import canonical_timestamp
from app.repositories.context_repository import VersionConflictError
from app.repositories.embedding_repository import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.tools._shared import EmbeddingsReconcileRequiredError
from app.tools._shared import build_store_response_message
from app.tools._shared import build_update_response_message
from app.tools._shared import embed_then_compress
from app.tools._shared import execute_store_in_transaction
from app.tools._shared import execute_update_in_transaction
from app.tools._shared import generate_index_nodes_with_timeout
from app.tools._shared import is_connection_error
from app.tools._shared import node_layer_active
from app.tools._shared import run_generation
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

        # === PHASE 2: Generate embeddings, summary, and index_tree node summaries ===
        # All generation happens BEFORE any database operation: if an
        # abort-mandatory step (embedding, compression, or the flat summary)
        # fails, NO data is saved. The embedding->compression leg and the
        # summary->nodes leg run CONCURRENTLY (disjoint resources); the flat
        # summary precedes the never-raise node summaries on the shared
        # summary-model budget, and a failure cancels the other leg cleanly
        # (see run_generation).

        repos = await ensure_repositories()
        summary_text: str | None = None
        summary_generated = False

        # Performance optimization: pre-check for likely duplicates (read-only)
        likely_duplicate_id: str | None = None

        if get_embedding_provider() is not None or get_summary_provider() is not None:
            likely_duplicate_id = await repos.context.check_latest_is_duplicate(
                thread_id=thread_id,
                source=source,
                text_content=text,
            )

        # Decide which generation legs to run. The dedup pre-check lets a likely
        # retransmit skip work it would only discard (embeddings already stored,
        # summary reusable, node rows left untouched).
        run_embedding = False
        if get_embedding_provider() is not None:
            if likely_duplicate_id is not None:
                run_embedding = not await repos.embeddings.exists(likely_duplicate_id)
                if not run_embedding:
                    logger.debug(
                        'Pre-check: skipping embedding generation for likely duplicate '
                        'of context %s in thread %s', likely_duplicate_id, thread_id,
                    )
            else:
                run_embedding = True

        run_summary = False
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
                    summary_text = existing_summary  # reuse; no model call
                    logger.debug(
                        'Pre-check: reusing existing summary for likely duplicate '
                        'of context %s in thread %s', likely_duplicate_id, thread_id,
                    )
                else:
                    run_summary = True
            else:
                run_summary = True

        # Node summaries are gated symmetrically with embeddings/summary: a likely
        # retransmit re-issues none and leaves stored rows untouched (index_nodes
        # stays None). On a reconcile-divergence INSERT below they are regenerated
        # for the diverged text.
        run_nodes = likely_duplicate_id is None

        chunk_embeddings, generated_summary, index_nodes = await run_generation(
            text, source,
            run_embedding=run_embedding,
            run_summary=run_summary,
            run_nodes=run_nodes,
        )
        if generated_summary is not None:
            summary_text = generated_summary
            summary_generated = True
        embedding_generated = chunk_embeddings is not None

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend
        metadata_str = json.dumps(metadata, ensure_ascii=False) if metadata is not None else None

        max_retries = 2
        context_id = ''
        was_updated = False
        embedding_stored = False
        reconciled = False
        attempt = 0

        while True:
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
                        embedding_generation_enabled=get_embedding_provider() is not None,
                        index_nodes=index_nodes,
                        nodes_pending=node_layer_active() and likely_duplicate_id is not None,
                    )

                # Transaction committed successfully -- break retry loop
                break

            except EmbeddingsReconcileRequiredError:
                # The read-only pre-check skipped embedding generation expecting a
                # deduplication UPDATE, but the transaction inserted a new entry
                # (a concurrent interleaving write flipped the decision).
                # Regenerate embeddings OUTSIDE the transaction (preserving the
                # generation-first invariant) and retry once.
                if reconciled:
                    raise ToolError(
                        'Failed to reconcile embeddings after deduplication divergence',
                    ) from None
                reconciled = True
                logger.info(
                    'Deduplication pre-check/transaction divergence in thread %s; '
                    'regenerating skipped embeddings before retry',
                    thread_id,
                )
                # Only regenerate embeddings if they were actually skipped. A
                # node-only reconcile (embeddings already present, nodes pending)
                # must NOT re-run the provider: that would discard valid
                # embeddings and let a transient provider failure abort the store.
                if chunk_embeddings is None:
                    chunk_embeddings = await embed_then_compress(text)
                    embedding_generated = chunk_embeddings is not None
                # The divergence INSERTed a new entry, so its node summaries were
                # never computed (gated off above as a likely duplicate). Regenerate
                # for the diverged text so the new entry gets its index_tree nodes.
                if index_nodes is None:
                    # Total node-summary degradation returns None; coerce to []
                    # so the reconcile gate clears on retry. The node layer is
                    # never-raise: degradation must NOT abort the store, and a
                    # divergence INSERT has no stale node rows to preserve, so []
                    # (write no node rows now) is the correct value.
                    index_nodes = await generate_index_nodes_with_timeout(text) or []
                continue

            except ToolError:
                raise  # ToolError is a logical error, not connection error -- do not retry
            except Exception as e:
                if is_connection_error(e) and attempt < max_retries:
                    delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                    attempt += 1
                    logger.warning(
                        'Transaction failed with connection error, retrying in %.1fs '
                        '(attempt %d/%d): %s',
                        delay, attempt, max_retries, e,
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
    context_ids: Annotated[list[str], Field(min_length=1, description='List of context entry IDs to retrieve')],
    include_images: Annotated[bool, Field(description='Whether to include image data')] = True,
    ctx: Context | None = None,
) -> list[ContextEntryDict]:
    """Fetch specific context entries by their IDs with FULL (non-truncated) text content.

    Use this when you have specific context IDs from previous operations
    and need the complete, untruncated content.

    Non-existent IDs are silently skipped; only found entries are returned.

    Returns:
        List of ContextEntryDict with id, thread_id, source, text_content, metadata,
        tags, images, created_at, updated_at fields. The summary field follows a
        tri-state contract controlled by GET_CONTEXT_BY_IDS_INCLUDE_SUMMARY:

        - When disabled (the default), the summary key is omitted entirely; consumers
          reading entry.get('summary') will receive None, which is the conventional
          Python signal for "feature disabled, no value to surface".
        - When enabled and the stored summary is a non-empty string, the value is
          returned verbatim.
        - When enabled but the stored summary is NULL or empty (e.g., generation was
          skipped because text was shorter than SUMMARY_MIN_CONTENT_LENGTH, or no
          provider is configured), the value is normalized to an empty string ''.
          This mirrors the search-tool contract (search tools always emit summary as
          a string, never None) and provides an explicit "feature on, no data yet"
          signal distinct from the "feature disabled" None.

    Raises:
        ToolError: If fetching context entries fails.
    """
    try:
        # Get repositories first; prefix resolution below needs the context repo.
        repos = await ensure_repositories()

        # Resolve incoming IDs at the boundary: accept full 32/36-char IDs or
        # 8-31 char hex prefixes (uniform with update_context/delete_context).
        try:
            context_ids = await resolve_or_normalize_ids(context_ids, repos.context)
        except ValueError as e:
            raise ToolError(f'Invalid context ID: {e}') from e

        if ctx:
            await ctx.info(f'Fetching context entries: {context_ids}')

        # Fetch context entries using repository
        rows = await repos.context.get_by_ids(context_ids)
        entries: list[ContextEntryDict] = []
        include_summary = settings.retrieval.include_summary

        for row in rows:
            # Create entry dict with proper typing for dynamic fields
            entry = cast(ContextEntryDict, dict(row))

            # Canonical timestamp wire format: identical across SQLite and PostgreSQL
            # and to the search tools (text_content stays FULL here -- get_context_by_ids
            # is never truncated). See app.repositories.base.canonical_timestamp.
            created_at_val = entry.get('created_at')
            if created_at_val is not None:
                entry['created_at'] = cast(str, canonical_timestamp(created_at_val))
            updated_at_val = entry.get('updated_at')
            if updated_at_val is not None:
                entry['updated_at'] = cast(str, canonical_timestamp(updated_at_val))

            if include_summary:
                # Mirror search-tool normalization (app/tools/search.py:140-145):
                # surface an empty string for the "feature ON but no data yet" state.
                # Tri-state contract:
                #   include_summary=False (default)              -> key omitted (consumers see entry.get('summary') == None)
                #   include_summary=True  + stored non-empty str -> verbatim stored string
                #   include_summary=True  + DB NULL/empty        -> '' (explicit signal "feature on, no data yet")
                summary = entry.get('summary')
                if isinstance(summary, str) and summary.strip():
                    entry['summary'] = summary
                else:
                    entry['summary'] = ''
            else:
                entry.pop('summary', None)

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
                entry_id = str(entry_id_raw)
                tags_result = await repos.tags.get_tags_for_context(entry_id)
                entry['tags'] = tags_result
            else:
                entry['tags'] = []

            # Fetch images
            if include_images and entry.get('content_type') == 'multimodal':
                entry_id_img = entry.get('id')
                if entry_id_img is not None:
                    images_result = await repos.images.get_images_for_context(str(entry_id_img), include_data=True)
                    entry['images'] = images_result
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
        list[str] | None,
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

        # Get repositories first; prefix resolution below needs the context repo.
        repos = await ensure_repositories()

        # Resolve incoming IDs at the boundary: accept full 32/36-char IDs or
        # 8-31 char hex prefixes (uniform with get_context_by_ids/update_context).
        if context_ids:
            try:
                context_ids = await resolve_or_normalize_ids(context_ids, repos.context)
            except ValueError as e:
                raise ToolError(f'Invalid context ID: {e}') from e

        if ctx:
            await ctx.info(f'Deleting context: ids={context_ids}, thread={thread_id}')

        deleted = 0

        if context_ids:
            # Delete embeddings first (explicit cleanup). Gate on whether the
            # embedding tables were ever PROVISIONED (embedding_tables_exist),
            # NOT on the runtime ENABLE_EMBEDDING_GENERATION/COMPRESSION toggles:
            # a prior session may have written embeddings that a now-disabled
            # toggle would skip cleaning. On SQLite the vec0 table has no FK and
            # is reachable only through the embedding_chunks bridge, so once that
            # bridge cascades away with the context row the vec0 vectors are
            # orphaned permanently. Mirrors the stale-embedding cleanup on the
            # update path (app/tools/_shared.py), which gates on the same signal.
            if await repos.embeddings.embedding_tables_exist():
                for context_id in context_ids:
                    try:
                        await repos.embeddings.delete(context_id)
                    except Exception as e:
                        logger.warning(f'Failed to delete embedding for context {context_id}: {e}')
                        # Non-blocking: continue even if embedding deletion fails

            deleted = await repos.context.delete_by_ids(context_ids)
            logger.info(f'Deleted {deleted} context entries by IDs')

        elif thread_id:
            # Embedding cleanup for the whole thread. On PostgreSQL the embedding
            # rows cascade-delete with the context rows, so explicit cleanup is
            # SQLite-only (vec0 virtual tables lack CASCADE). Use the UNBOUNDED
            # criteria query rather than a capped search so a thread with more than
            # 10000 entries does not leave orphaned embeddings.
            # Gate on table PRESENCE (embedding_tables_exist), not the runtime
            # toggles: a prior session's embeddings must still be cleaned even
            # after generation/compression are turned off, or the FK-less vec0
            # rows orphan. Mirrors the context_ids branch above.
            if await repos.embeddings.embedding_tables_exist():
                backend = repos.context.backend
                if backend.backend_type == 'sqlite':
                    try:
                        thread_ids_to_clean = await repos.context.get_ids_matching_batch_criteria(
                            thread_ids=[thread_id],
                        )
                        for cid in thread_ids_to_clean:
                            try:
                                await repos.embeddings.delete(cid)
                            except Exception as e:
                                logger.warning(f'Failed to delete embedding for context {cid}: {e}')
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
    context_id: Annotated[
        str,
        Field(
            min_length=8,
            description='ID (full 32-char hex, 36-char hyphenated, or 8-31 char hex prefix) '
            'of the context entry to update',
        ),
    ],
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

        # Validate images early (before any operations)
        validated_images, _, _ = validate_and_normalize_images(images, error_mode='raise')

        # Get repositories
        repos = await ensure_repositories()

        # Boundary normalization: accept full hex (32 or 36 chars) or 8-31 char hex prefix
        try:
            context_id = await resolve_or_normalize_id(context_id, repos.context)
        except ValueError as e:
            raise ToolError(f'Invalid context ID: {e}') from e

        if ctx:
            await ctx.info(f'Updating context entry {context_id}')

        # Check if entry exists; capture source and the optimistic-concurrency
        # version BEFORE generation so a concurrent writer that commits during
        # our (slow) generation is caught by the conditional write below.
        exists, entry_source, expected_version = await repos.context.check_entry_exists(context_id)
        if not exists:
            raise ToolError(f'Context entry with ID {context_id} not found')
        assert entry_source is not None  # guaranteed by exists=True

        # === PHASE 2: Generate Summary + Embedding FIRST (Outside Transaction) ===
        # CRITICAL: All generation happens BEFORE any database modification.
        # If an abort-mandatory step fails, NO data is modified.
        chunk_embeddings: list[ChunkEmbedding] | None = None
        embedding_generated = False
        summary_text: str | None = None
        summary_generated = False
        clear_summary = False
        # None leaves the index_tree node table untouched (feature off, or text
        # unchanged); recomputed only when text changes (below).
        index_nodes: list[IndexNodeRow] | None = None

        if text is not None:
            # Text changed -> regenerate embeddings, summary, and node summaries.
            # The embedding->compression leg overlaps the flat-summary->nodes leg
            # (see run_generation); a node-summary failure never aborts the update.
            run_embedding = get_embedding_provider() is not None
            run_summary = False
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
                    run_summary = True
            else:
                # No summary provider at update time (summary generation disabled/absent).
                # The stored summary describes the REPLACED text, so CLEAR it instead of
                # leaving a stale summary (mirrors the too-short / empty-output branches
                # and the stale-embedding cleanup on the same text-change path).
                clear_summary = True

            chunk_embeddings, summary_text, index_nodes = await run_generation(
                text, entry_source,
                run_embedding=run_embedding,
                run_summary=run_summary,
                run_nodes=True,
            )
            embedding_generated = chunk_embeddings is not None
            summary_generated = bool(summary_text)
            # If summary regeneration ran but produced nothing (empty/whitespace
            # provider output normalized to None), the OLD summary describes the
            # REPLACED text, so CLEAR it -- mirroring the too-short branch above and
            # the node-layer clear below -- instead of preserving a stale summary.
            if run_summary and summary_text is None:
                clear_summary = True
            # On a text change, a None node result must CLEAR the stored rows ([] replaces)
            # whenever the per-node layer is ENABLED, because those rows describe the OLD
            # text and navigate_context surfaces them gated ONLY on
            # settings.index_tree.node_summaries_enabled (it does NOT require a provider).
            # This covers BOTH total degradation (every section summary failed) AND the
            # provider-removed-but-feature-on case (generation returns None for lack of a
            # provider): in either case preserving the stale rows would let
            # navigate_context mis-attach an old section summary to a new section by reused
            # heading slug. Gating on node_summaries_enabled (the reader's gate) rather than
            # node_layer_active() (which also requires a provider) mirrors the
            # provider-independent stale-clear the embedding and flat-summary legs already
            # perform on this same text-change path. When the feature is OFF the reader is
            # also off, so a None legitimately leaves the table untouched.
            if index_nodes is None and settings.index_tree.node_summaries_enabled:
                index_nodes = []

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend
        updated_fields: list[str] = []

        max_retries = 2
        attempt = 0
        version_conflicts = 0
        max_version_conflicts = 5

        while True:
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
                        index_nodes=index_nodes,
                        expected_version=expected_version,
                    )

                # Transaction committed -- break retry loop
                break

            except VersionConflictError:
                # A concurrent writer committed a newer version of this entry
                # during our generation. Re-read the current version and retry
                # the write with the SAME generated artifacts (they describe the
                # text THIS call requested), so our update applies on top of the
                # latest row instead of silently overwriting it.
                if version_conflicts >= max_version_conflicts:
                    raise ToolError(
                        f'Concurrent modification of context {context_id}: the entry kept '
                        f'changing during the update. Retry the request.',
                    ) from None
                version_conflicts += 1
                try:
                    exists, _src, current_version = await repos.context.check_entry_exists(context_id)
                except Exception as reread_error:
                    # A transient connection error during the conflict re-read must
                    # get the same bounded backoff/retry as the rest of the loop,
                    # not abort the whole update.
                    if is_connection_error(reread_error) and attempt < max_retries:
                        delay = 0.5 * (2 ** attempt)
                        attempt += 1
                        logger.warning(
                            'Connection error during version-conflict re-read of %s; '
                            'retrying in %.1fs (attempt %d/%d): %s',
                            context_id, delay, attempt, max_retries, reread_error,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise
                if not exists:
                    raise ToolError(f'Context entry with ID {context_id} not found') from None
                expected_version = current_version
                logger.info(
                    'Version conflict updating context %s; retrying (%d/%d)',
                    context_id, version_conflicts, max_version_conflicts,
                )
                continue

            except ToolError:
                raise  # ToolError is a logical error, not connection error -- do not retry
            except Exception as e:
                if is_connection_error(e) and attempt < max_retries:
                    delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                    attempt += 1
                    logger.warning(
                        'Transaction failed with connection error, retrying in %.1fs '
                        '(attempt %d/%d): %s',
                        delay, attempt, max_retries, e,
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
