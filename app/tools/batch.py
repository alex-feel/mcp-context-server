"""
Batch operations for MCP tools.

This module contains tools for bulk context management:
- store_context_batch: Store multiple entries in one operation
- update_context_batch: Update multiple entries in one operation
- delete_context_batch: Delete entries by various criteria

Generation-First Transactional Integrity:
This module implements atomic generation + data storage for batch operations.
Each entry's embedding + summary are generated in PARALLEL via asyncio.gather(return_exceptions=True),
reusing generate_embeddings_with_timeout and generate_summary_with_timeout from app.tools._shared.
Entries within a batch are processed SEQUENTIALLY. When generation is enabled:
1. Per-entry: embeddings and summaries run in parallel OUTSIDE any database transaction
2. If ANY generation fails in atomic mode, NO data is saved
3. If all generation succeeds, ALL database operations occur in a SINGLE atomic transaction

Infrastructure functions (embedding/summary generation, transaction heartbeat,
connection error classification, image validation, response message builders) are
in app.tools._shared -- the single source of truth for logic shared with context.py.
"""

import asyncio
import json
import logging
import operator
from collections.abc import Awaitable
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from typing import Annotated
from typing import Any
from typing import Literal
from typing import cast

from fastmcp import Context
from fastmcp.exceptions import ToolError
from pydantic import Field

from app.errors import format_exception_message
from app.ids import resolve_or_normalize_id
from app.ids import resolve_or_normalize_ids
from app.repositories.context_repository import VersionConflictError
from app.repositories.embedding_repository import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.tools._shared import EmbeddingsReconcileRequiredError
from app.tools._shared import build_batch_store_response_message
from app.tools._shared import build_batch_update_response_message
from app.tools._shared import execute_store_in_transaction
from app.tools._shared import execute_update_in_transaction
from app.tools._shared import generate_compression_with_timeout
from app.tools._shared import generate_embeddings_with_timeout
from app.tools._shared import generate_index_nodes_with_timeout
from app.tools._shared import generate_summary_with_timeout
from app.tools._shared import is_connection_error
from app.tools._shared import node_layer_active
from app.tools._shared import transaction_heartbeat
from app.tools._shared import validate_and_normalize_images
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
    """Store multiple context entries in a batch.

    Batch processing is significantly faster than individual calls when storing many entries.
    Use for migrations, imports, or bulk operations.

    Atomicity modes:
    - atomic=True (default): ALL entries must succeed or NONE are stored (transaction rollback).
    - atomic=False: Each entry processed independently with per-item error reporting.

    Deduplication: if an entry with identical thread_id, source, and text already exists,
    the existing entry is updated. Deduplication is suppressed when opposite-source
    entries exist after the candidate, preserving chronological ordering.
    Pre-check optimization skips embedding/summary generation for likely duplicates.

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
            if images is not None and (
                not isinstance(images, list) or not all(isinstance(i, dict) for i in images)
            ):
                # Parity with the single-entry, Pydantic-typed store_context (which rejects a
                # non-list / non-object images). Without this, a dict-as-images or a list with a
                # non-dict element would reach validate_and_normalize_images and raise a raw
                # AttributeError/TypeError that aborts the whole non-atomic batch instead of
                # recording a per-entry error.
                validation_errors.append((idx, 'images must be a list of objects'))
                continue
            if images:
                _, content_type_from_images, img_errors = validate_and_normalize_images(
                    cast(list[dict[str, str]], images), error_mode='collect',
                )
                if img_errors:
                    validation_errors.append((idx, img_errors[0]))
                    continue
                content_type = content_type_from_images
            else:
                content_type = 'text'

            # Validate metadata is a JSON object and tags is a list of strings.
            # The single-entry store_context is Pydantic-typed and rejects these;
            # the batch path takes untyped dicts, so check here for parity: a
            # non-dict metadata breaks search/metadata_filters, and a bare-string
            # tags would be stored one character per tag.
            metadata = entry.get('metadata')
            if metadata is not None and not isinstance(metadata, dict):
                validation_errors.append((idx, 'metadata must be a JSON object'))
                continue
            tags = entry.get('tags')
            if tags is not None and (
                not isinstance(tags, list) or not all(isinstance(t, str) for t in tags)
            ):
                validation_errors.append((idx, 'tags must be a list of strings'))
                continue

            # Prepare validated entry
            validated_entries.append({
                'index': idx,
                'thread_id': thread_id,
                'source': entry['source'],
                'text_content': text,
                'metadata': json.dumps(metadata, ensure_ascii=False) if metadata is not None else None,
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

        # === PHASE 2: Generate Embeddings + Summaries per entry (Outside Transaction) ===
        # CRITICAL: Generation happens BEFORE any database modifications.
        # Each entry runs embedding+summary in parallel via asyncio.gather.
        # Entries are processed sequentially to limit provider load.
        entry_embeddings: dict[int, list[ChunkEmbedding] | None] = {}
        entry_summaries: dict[int, str | None] = {}
        # Per-entry dedup pre-check decision, captured here so the index_tree node
        # pass below gates symmetrically with embeddings/summary.
        entry_likely_duplicate: dict[int, str | None] = {}
        generation_errors: list[tuple[int, str]] = []  # (original_idx, error_message)

        embedding_provider = get_embedding_provider()
        summary_provider = get_summary_provider()
        min_content_length = settings.summary.min_content_length
        embeddings_generated_count = 0
        embeddings_stored_count = 0
        summaries_generated_count = 0
        summaries_preserved_count = 0
        # ve_idx values whose summary was reused from a likely duplicate (pre-check),
        # so a non-atomic compression failure that discards the entry can decrement the
        # preserved count it provisionally bumped -- mirroring the generated counts,
        # which are now bumped only after compression succeeds.
        preserved_summary_indices: set[int] = set()

        for ve_idx, entry in enumerate(validated_entries):
            original_idx = entry['index']
            text_content = entry['text_content']

            tasks_to_run: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
            task_names: list[str] = []

            # Performance optimization: pre-check for likely duplicates (read-only)
            likely_duplicate_id: str | None = None
            if embedding_provider is not None or summary_provider is not None:
                likely_duplicate_id = await repos.context.check_latest_is_duplicate(
                    thread_id=entry['thread_id'],
                    source=entry['source'],
                    text_content=text_content,
                )
            entry_likely_duplicate[ve_idx] = likely_duplicate_id

            # Embedding task (with pre-check optimization)
            if embedding_provider is not None:
                if likely_duplicate_id is not None:
                    has_embeddings = await repos.embeddings.exists(likely_duplicate_id)
                    if has_embeddings:
                        logger.debug(
                            'Pre-check: skipping embedding generation for likely duplicate '
                            'of context %s at index %d',
                            likely_duplicate_id, original_idx,
                        )
                    else:
                        tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                        task_names.append('embedding')
                else:
                    tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                    task_names.append('embedding')

            # Summary task (with pre-check optimization)
            if summary_provider is not None:
                if min_content_length > 0 and len(text_content) < min_content_length:
                    logger.info(
                        'Skipping summary generation at index %d: '
                        'text length %d < min_content_length %d',
                        original_idx, len(text_content), min_content_length,
                    )
                    entry_summaries[ve_idx] = None
                elif likely_duplicate_id is not None:
                    existing_summary = await repos.context.get_summary(likely_duplicate_id)
                    if existing_summary is not None:
                        entry_summaries[ve_idx] = existing_summary
                        summaries_preserved_count += 1
                        preserved_summary_indices.add(ve_idx)
                        logger.debug(
                            'Pre-check: reusing existing summary for likely duplicate '
                            'of context %s at index %d',
                            likely_duplicate_id, original_idx,
                        )
                    else:
                        tasks_to_run.append(generate_summary_with_timeout(text_content, entry['source']))
                        task_names.append('summary')
                else:
                    tasks_to_run.append(generate_summary_with_timeout(text_content, entry['source']))
                    task_names.append('summary')

            if not tasks_to_run:
                entry_embeddings[ve_idx] = None
                if ve_idx not in entry_summaries:
                    entry_summaries[ve_idx] = None
                continue

            gather_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            errors: list[tuple[str, BaseException]] = []
            for name, result in zip(task_names, gather_results, strict=True):
                if isinstance(result, BaseException):
                    errors.append((name, result))
                    logger.error(
                        'Generation failed for %s at index %d (retries exhausted): %s',
                        name, original_idx, result,
                    )
                elif name == 'embedding':
                    entry_embeddings[ve_idx] = cast(list[ChunkEmbedding] | None, result)
                elif name == 'summary':
                    entry_summaries[ve_idx] = cast(str | None, result)

            if errors:
                error_details = '; '.join(
                    f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
                )
                if atomic:
                    raise ToolError(
                        f'Generation failed at index {original_idx} after exhausting '
                        f'configured retries: {error_details}. No data was saved.',
                    )
                generation_errors.append((original_idx, f'Generation failed: {error_details}'))
                entry_embeddings.setdefault(ve_idx, None)
                entry_summaries.setdefault(ve_idx, None)
            else:
                entry_embeddings.setdefault(ve_idx, None)
                if ve_idx not in entry_summaries:
                    entry_summaries[ve_idx] = None

                # Compress embeddings OUTSIDE the DB transaction (mirrors
                # the generation-first invariant). No-op when
                # ENABLE_EMBEDDING_COMPRESSION=false. In atomic mode a
                # failure aborts the whole batch; in non-atomic mode this
                # entry is marked failed and processing continues.
                try:
                    entry_embeddings[ve_idx] = await generate_compression_with_timeout(
                        entry_embeddings.get(ve_idx),
                    )
                except ToolError as compress_err:
                    if atomic:
                        raise ToolError(
                            f'Compression failed at index {original_idx}: '
                            f'{compress_err}. No data was saved.',
                        ) from compress_err
                    generation_errors.append(
                        (original_idx, f'Compression failed: {compress_err}'),
                    )
                    entry_embeddings[ve_idx] = None
                    entry_summaries[ve_idx] = None
                    # This entry is discarded below; undo the preserved-summary count it
                    # provisionally bumped in the pre-check, mirroring the generated counts
                    # which are bumped only on compression success.
                    if ve_idx in preserved_summary_indices:
                        summaries_preserved_count -= 1
                else:
                    # Count generated artifacts only AFTER compression succeeds, so a
                    # non-atomic compression failure (which discards this entry below)
                    # cannot inflate the generated counts in the response diagnostic.
                    if entry_embeddings.get(ve_idx) is not None:
                        embeddings_generated_count += 1
                    # Only count a summary as GENERATED when a summary model call
                    # actually ran for this entry; a reused/preserved summary (no
                    # 'summary' task queued, accounted by summaries_preserved_count)
                    # must not be double-counted here, mirroring the single-store
                    # generated-vs-preserved split.
                    if (
                        'summary' in task_names
                        and entry_summaries.get(ve_idx)
                        and isinstance(entry_summaries[ve_idx], str)
                    ):
                        summaries_generated_count += 1

        # In non-atomic mode, add generation errors to results and filter
        if not atomic and generation_errors:
            for original_idx, error in generation_errors:
                results.append(BulkStoreResultItemDict(
                    index=original_idx,
                    success=False,
                    context_id=None,
                    error=error,
                ))
            failed_indices = {idx for idx, _ in generation_errors}
            validated_entries_filtered = [
                (ve_idx, e) for ve_idx, e in enumerate(validated_entries)
                if e['index'] not in failed_indices
            ]
        else:
            validated_entries_filtered = list(enumerate(validated_entries))

        if not validated_entries_filtered:
            results.sort(key=operator.itemgetter('index'))
            return BulkStoreResponseDict(
                success=False,
                total=len(entries),
                succeeded=0,
                failed=len(entries),
                results=results,
                message='All entries failed validation or generation',
            )

        # Build index_tree per-node summaries for each surviving entry in a
        # SEPARATE fenced never-raise pass, isolated from the abort-mandatory
        # embedding/summary/compression above: a node-summary failure never fails
        # an entry. None per entry when the feature is disabled OR the entry is a
        # likely deduplication retransmit (gated symmetrically with
        # embeddings/summary). Computed once and reused across reconcile/connection
        # retries below.
        entry_index_nodes: dict[int, list[IndexNodeRow] | None] = {}
        for ve_idx, entry in validated_entries_filtered:
            # Skip node regeneration for a likely deduplication retransmit (None
            # leaves stored rows untouched); reconcile-divergence INSERTs below
            # regenerate for the diverged text.
            entry_index_nodes[ve_idx] = (
                None
                if entry_likely_duplicate.get(ve_idx) is not None
                else await generate_index_nodes_with_timeout(entry['text_content'])
            )

        # === PHASE 3: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend

        if atomic:
            # ATOMIC MODE: All entries in a single transaction with retry
            max_retries = 2
            attempt = 0
            # A pre-check-skipped embedding can diverge once per distinct entry,
            # so bound reconciliation passes by the number of entries.
            reconcile_passes = 0
            max_reconcile_passes = len(validated_entries_filtered)

            while True:
                try:
                    results_attempt: list[BulkStoreResultItemDict] = []
                    # Count stored embeddings per attempt; only fold into the
                    # outer total once the transaction commits, so connection
                    # retries and reconciliation passes never double-count.
                    stored_count_attempt = 0

                    async with backend.begin_transaction() as txn:
                        for idx, (ve_idx, entry) in enumerate(validated_entries_filtered):
                            original_idx = entry['index']

                            # Heartbeat between entries (skip first)
                            if idx > 0:
                                await transaction_heartbeat(txn)

                            context_id, was_updated, embedding_stored = (
                                await execute_store_in_transaction(
                                    repos, txn,
                                    thread_id=entry['thread_id'],
                                    source=entry['source'],
                                    content_type=entry['content_type'],
                                    text_content=entry['text_content'],
                                    metadata_str=entry['metadata'],
                                    summary=entry_summaries.get(ve_idx),
                                    tags=entry.get('tags'),
                                    validated_images=entry.get('images', []),
                                    chunk_embeddings=entry_embeddings.get(ve_idx),
                                    embedding_model=settings.embedding.model,
                                    embedding_generation_enabled=embedding_provider is not None,
                                    index_nodes=entry_index_nodes.get(ve_idx),
                                    nodes_pending=(
                                        node_layer_active()
                                        and entry_likely_duplicate.get(ve_idx) is not None
                                    ),
                                )
                            )

                            if embedding_stored:
                                stored_count_attempt += 1

                            results_attempt.append(BulkStoreResultItemDict(
                                index=original_idx,
                                success=True,
                                context_id=context_id,
                                error=None,
                            ))

                        # COMMIT happens here - all or nothing

                    # Transaction committed successfully
                    results.extend(results_attempt)
                    embeddings_stored_count += stored_count_attempt
                    break

                except EmbeddingsReconcileRequiredError as reconcile:
                    # The read-only pre-check skipped embedding generation for a
                    # likely duplicate, but the transaction inserted a new entry.
                    # Regenerate OUTSIDE the transaction (generation-first) for the
                    # diverging text -- and any other filtered entry sharing that
                    # exact text whose embeddings were also skipped -- then re-run
                    # the whole atomic transaction.
                    if reconcile_passes >= max_reconcile_passes:
                        raise ToolError(
                            'Failed to reconcile embeddings after deduplication '
                            'divergence (atomic batch store)',
                        ) from None
                    reconcile_passes += 1
                    # Only re-run the embedding provider when at least one diverged
                    # entry actually lacks embeddings. A node-only reconcile (the
                    # nodes_pending gate fired while embeddings are present) must not
                    # re-invoke the provider: that wastes a call and a transient
                    # failure would spuriously abort the whole atomic batch.
                    needs_embeddings = embedding_provider is not None and any(
                        entry_embeddings.get(r_ve_idx) is None
                        and r_entry['text_content'] == reconcile.text_content
                        for r_ve_idx, r_entry in validated_entries_filtered
                    )
                    if needs_embeddings:
                        regenerated = await generate_compression_with_timeout(
                            await generate_embeddings_with_timeout(reconcile.text_content),
                        )
                        for r_ve_idx, r_entry in validated_entries_filtered:
                            if (entry_embeddings.get(r_ve_idx) is None
                                    and r_entry['text_content'] == reconcile.text_content):
                                entry_embeddings[r_ve_idx] = regenerated
                                # Count only embeddings actually produced. With
                                # generation disabled the no-op provider returns
                                # None, so a node-only reconcile must not inflate
                                # the generated count (parity with the single store).
                                if regenerated is not None:
                                    embeddings_generated_count += 1
                    # Node summaries for the diverged text were gated off as a
                    # likely duplicate; this divergence INSERTs, so regenerate them.
                    # Coerce total degradation (None) to [] so the reconcile gate
                    # clears on retry: the node layer is never-raise and must not
                    # abort the store; a fresh INSERT has no stale node rows.
                    regenerated_nodes = await generate_index_nodes_with_timeout(reconcile.text_content) or []
                    for r_ve_idx, r_entry in validated_entries_filtered:
                        if (entry_index_nodes.get(r_ve_idx) is None
                                and r_entry['text_content'] == reconcile.text_content):
                            entry_index_nodes[r_ve_idx] = regenerated_nodes
                    continue

                except ToolError:
                    raise  # Logical error -- do not retry
                except Exception as e:
                    if is_connection_error(e) and attempt < max_retries:
                        delay = 0.5 * (2 ** attempt)  # 0.5s, 1.0s
                        attempt += 1
                        logger.warning(
                            'Atomic batch store transaction failed, retrying in %.1fs '
                            '(attempt %d/%d): %s',
                            delay, attempt, max_retries, e,
                        )
                        await asyncio.sleep(delay)
                        continue
                    raise  # Non-connection error or max retries exceeded
        else:
            # NON-ATOMIC MODE: Each entry in its own transaction (with retry)
            for ve_idx, entry in validated_entries_filtered:
                original_idx = entry['index']
                max_retries = 2
                attempt = 0
                reconciled = False

                while True:
                    try:
                        async with backend.begin_transaction() as txn:
                            context_id, was_updated, embedding_stored = (
                                await execute_store_in_transaction(
                                    repos, txn,
                                    thread_id=entry['thread_id'],
                                    source=entry['source'],
                                    content_type=entry['content_type'],
                                    text_content=entry['text_content'],
                                    metadata_str=entry['metadata'],
                                    summary=entry_summaries.get(ve_idx),
                                    tags=entry.get('tags'),
                                    validated_images=entry.get('images', []),
                                    chunk_embeddings=entry_embeddings.get(ve_idx),
                                    embedding_model=settings.embedding.model,
                                    embedding_generation_enabled=embedding_provider is not None,
                                    index_nodes=entry_index_nodes.get(ve_idx),
                                    nodes_pending=(
                                        node_layer_active()
                                        and entry_likely_duplicate.get(ve_idx) is not None
                                    ),
                                )
                            )

                            # COMMIT happens here for this entry

                        if embedding_stored:
                            embeddings_stored_count += 1

                        results.append(BulkStoreResultItemDict(
                            index=original_idx,
                            success=True,
                            context_id=context_id,
                            error=None,
                        ))
                        break  # Success -- exit retry loop

                    except EmbeddingsReconcileRequiredError as reconcile:
                        # Pre-check skipped embeddings for a likely duplicate, but
                        # this store inserted a new entry. Regenerate OUTSIDE the
                        # transaction (generation-first) and retry this entry once.
                        if reconciled:
                            logger.error(
                                'Failed to reconcile embeddings for entry at index %d '
                                'after deduplication divergence', original_idx,
                            )
                            results.append(BulkStoreResultItemDict(
                                index=original_idx,
                                success=False,
                                context_id=None,
                                error='Failed to reconcile embeddings after deduplication divergence',
                            ))
                            break
                        reconciled = True
                        # Only re-run the embedding provider when this entry's
                        # embeddings were actually skipped; a node-only reconcile
                        # must not re-invoke the provider (wasted call + a transient
                        # failure would spuriously fail an otherwise-good entry).
                        #
                        # This regeneration is abort-mandatory and raises ToolError on
                        # provider failure/timeout. In non-atomic mode that ToolError
                        # must fail ONLY this entry: without a local guard it would
                        # escape the per-entry loop to the function-level handler and
                        # abort the whole batch, discarding sibling results already
                        # collected -- breaking the documented per-entry isolation of
                        # atomic=false. Record a per-entry failure and stop this entry,
                        # mirroring the per-entry ToolError branch below. (The atomic
                        # branch deliberately lets it abort: all-or-nothing.)
                        try:
                            if embedding_provider is not None and entry_embeddings.get(ve_idx) is None:
                                entry_embeddings[ve_idx] = await generate_compression_with_timeout(
                                    await generate_embeddings_with_timeout(reconcile.text_content),
                                )
                                # Count only embeddings actually produced (the no-op
                                # provider returns None when generation is disabled).
                                if entry_embeddings[ve_idx] is not None:
                                    embeddings_generated_count += 1
                        except ToolError as e:
                            logger.error(
                                'Failed to regenerate embeddings for entry at index %d '
                                'after deduplication divergence: %s', original_idx, e,
                            )
                            results.append(BulkStoreResultItemDict(
                                index=original_idx,
                                success=False,
                                context_id=None,
                                error=format_exception_message(e),
                            ))
                            break
                        # Node summaries were gated off as a likely duplicate; this
                        # entry actually INSERTs, so regenerate its index_tree nodes.
                        # Coerce total degradation (None) to [] so the reconcile gate
                        # clears on retry: the node layer is never-raise and must not
                        # abort this entry's store; a fresh INSERT has no stale rows.
                        if entry_index_nodes.get(ve_idx) is None:
                            entry_index_nodes[ve_idx] = await generate_index_nodes_with_timeout(
                                reconcile.text_content,
                            ) or []
                        continue

                    except ToolError as e:
                        # Logical error -- do not retry, record as failure
                        logger.error(f'Failed to store entry at index {original_idx}: {e}')
                        results.append(BulkStoreResultItemDict(
                            index=original_idx,
                            success=False,
                            context_id=None,
                            error=format_exception_message(e),
                        ))
                        break
                    except Exception as e:
                        if is_connection_error(e) and attempt < max_retries:
                            delay = 0.5 * (2 ** attempt)
                            attempt += 1
                            logger.warning(
                                'Non-atomic batch store entry %d failed with connection error, '
                                'retrying in %.1fs (attempt %d/%d): %s',
                                original_idx, delay, attempt, max_retries, e,
                            )
                            await asyncio.sleep(delay)
                            continue
                        # Non-connection error or max retries exceeded
                        logger.error(f'Failed to store entry at index {original_idx}: {e}')
                        results.append(BulkStoreResultItemDict(
                            index=original_idx,
                            success=False,
                            context_id=None,
                            error=format_exception_message(e),
                        ))
                        break

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(entries) - succeeded

        logger.info(f'Batch store completed: {succeeded}/{len(entries)} succeeded')

        message = build_batch_store_response_message(
            succeeded=succeeded,
            total=len(entries),
            embeddings_generated_count=embeddings_generated_count,
            embeddings_stored_count=embeddings_stored_count,
            summaries_generated_count=summaries_generated_count,
            summaries_preserved_count=summaries_preserved_count,
        )

        return BulkStoreResponseDict(
            success=failed == 0,
            total=len(entries),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=message,
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
            description='List of update operations. Each must have context_id (str, accepts 32-char hex, '
            '36-char hyphenated UUID, or 8-31 char hex prefix). '
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
    """Update multiple context entries in a batch.

    Atomicity modes:
    - atomic=True (default): ALL updates succeed or NONE are applied (transaction rollback).
    - atomic=False: Each update processed independently with per-item error reporting.

    Update semantics per entry:
    - Each update is identified by context_id
    - Only provided fields are modified
    - Immutable fields (cannot be changed): id, thread_id, source, created_at
    - Auto-managed fields: content_type (recalculated based on images), updated_at
    - Metadata options (MUTUALLY EXCLUSIVE per entry):
      - metadata: FULL REPLACEMENT of entire metadata object
      - metadata_patch: RFC 7396 JSON Merge Patch (new keys added, existing updated,
        null values DELETE keys; cannot store null values; arrays replaced entirely)
    - Tags and images use REPLACEMENT semantics (not merge)

    Size limits:
    - Maximum 100 entries per batch
    - Image limits per entry: 10MB each, 100MB total

    Returns:
        BulkUpdateResponseDict with success (bool), total (int), succeeded (int),
        failed (int), results (list of index, context_id, success, updated_fields, error),
        message (str).

    Raises:
        ToolError: If validation fails, embedding generation fails (atomic), or batch operation fails.
    """
    try:
        if ctx:
            await ctx.info(f'Batch updating {len(updates)} context entries (atomic={atomic})')

        repos = await ensure_repositories()

        # === PHASE 1: Validate all updates before processing ===
        validated_updates: list[dict[str, Any]] = []
        validation_errors: list[tuple[int, str, str]] = []  # (index, context_id, error)

        for idx, update in enumerate(updates):
            # Validate required context_id
            if 'context_id' not in update:
                validation_errors.append((idx, '', 'Missing required field: context_id'))
                continue

            context_id_raw = update['context_id']
            if not isinstance(context_id_raw, str) or not context_id_raw.strip():
                validation_errors.append((idx, '', 'context_id must be a non-empty string'))
                continue

            # Resolve to canonical 32-char hex (accept full or prefix)
            try:
                context_id = await resolve_or_normalize_id(context_id_raw, repos.context)
            except ValueError as e:
                validation_errors.append((idx, context_id_raw, f'Invalid context_id: {e}'))
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

            # Validate metadata / metadata_patch are JSON objects and tags is a
            # list of strings (parity with the single-entry, Pydantic-typed
            # update_context, which rejects these).
            metadata_field = update.get('metadata')
            if metadata_field is not None and not isinstance(metadata_field, dict):
                validation_errors.append((idx, context_id, 'metadata must be a JSON object'))
                continue
            metadata_patch_field = update.get('metadata_patch')
            if metadata_patch_field is not None and not isinstance(metadata_patch_field, dict):
                validation_errors.append((idx, context_id, 'metadata_patch must be a JSON object'))
                continue
            tags_field = update.get('tags')
            if tags_field is not None and (
                not isinstance(tags_field, list) or not all(isinstance(t, str) for t in tags_field)
            ):
                validation_errors.append((idx, context_id, 'tags must be a list of strings'))
                continue

            # Validate images if provided
            images = update.get('images')
            if images is not None and (
                not isinstance(images, list) or not all(isinstance(i, dict) for i in images)
            ):
                # Parity with the single-entry, Pydantic-typed update_context (which rejects a
                # non-list / non-object images). Without this, a dict-as-images or a list with a
                # non-dict element would reach validate_and_normalize_images and raise a raw
                # AttributeError/TypeError that aborts the whole non-atomic batch instead of
                # recording a per-entry error.
                validation_errors.append((idx, context_id, 'images must be a list of objects'))
                continue
            if images:
                _, _, img_errors = validate_and_normalize_images(
                    cast(list[dict[str, str]], images), error_mode='collect',
                )
                if img_errors:
                    validation_errors.append((idx, context_id, img_errors[0]))
                    continue

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

        # === PHASE 2: Check all entries exist (fail fast in atomic mode) ===
        existence_errors: list[tuple[int, str, str]] = []  # (index, context_id, error)
        entry_sources: dict[str, str] = {}  # context_id -> source
        # context_id -> optimistic-concurrency version captured BEFORE generation;
        # passed to execute_update_in_transaction as the compare-and-set guard so a
        # concurrent writer that commits during generation is detected.
        entry_versions: dict[str, int] = {}

        for update in validated_updates:
            original_idx = update['index']
            context_id = update['context_id']

            exists, entry_source, entry_version = await repos.context.check_entry_exists(context_id)
            if not exists:
                if atomic:
                    raise ToolError(f'Context entry {context_id} not found at index {original_idx}')
                existence_errors.append((original_idx, context_id, f'Context entry {context_id} not found'))
            else:
                assert entry_source is not None
                assert entry_version is not None
                entry_sources[context_id] = entry_source
                entry_versions[context_id] = entry_version

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

        # === PHASE 3: Generate Embeddings + Summaries per entry (Outside Transaction) ===
        # CRITICAL: Generation happens BEFORE any database modifications.
        # Each entry runs embedding+summary in parallel via asyncio.gather.
        # Entries are processed sequentially to limit provider load.
        # Only entries with text changes need generation.
        update_embeddings: dict[int, list[ChunkEmbedding] | None] = {}
        update_summaries: dict[int, str | None] = {}
        update_clear_summaries: set[int] = set()
        generation_errors: list[tuple[int, str, str]] = []  # (original_idx, context_id, error)

        embedding_provider = get_embedding_provider()
        summary_provider = get_summary_provider()
        min_content_length = settings.summary.min_content_length
        embeddings_generated_count = 0
        summaries_generated_count = 0
        summaries_cleared_count = 0

        for vu_idx, update in validated_updates_filtered:
            original_idx = update['index']
            context_id = update['context_id']
            text_content = update.get('text')

            # No text change -- skip generation entirely
            if text_content is None:
                update_embeddings[vu_idx] = None
                update_summaries[vu_idx] = None
                continue

            tasks_to_run: list[Awaitable[list[ChunkEmbedding] | str | None]] = []
            task_names: list[str] = []

            # Embedding task
            if embedding_provider is not None:
                tasks_to_run.append(generate_embeddings_with_timeout(text_content))
                task_names.append('embedding')

            # Summary task (min_content_length pre-check OUTSIDE wrapper)
            if summary_provider is not None:
                if min_content_length > 0 and len(text_content) < min_content_length:
                    update_summaries[vu_idx] = None
                    update_clear_summaries.add(vu_idx)
                    logger.info(
                        'Skipping summary for context %s at index %d: '
                        'text length %d < min_content_length %d. Existing summary will be cleared.',
                        context_id, original_idx, len(text_content), min_content_length,
                    )
                else:
                    tasks_to_run.append(generate_summary_with_timeout(text_content, entry_sources[context_id]))
                    task_names.append('summary')
            else:
                # No summary provider at update time (summary generation disabled/absent).
                # The stored summary describes the REPLACED text, so CLEAR it instead of
                # leaving a stale summary -- mirroring the too-short branch above, the
                # single-update path (context.py), and the stale-embedding / index_tree
                # node-row clears that already run on this same text-change path.
                update_summaries[vu_idx] = None
                update_clear_summaries.add(vu_idx)

            if not tasks_to_run:
                update_embeddings.setdefault(vu_idx, None)
                if vu_idx not in update_summaries:
                    update_summaries[vu_idx] = None
                continue

            gather_results = await asyncio.gather(*tasks_to_run, return_exceptions=True)

            errors: list[tuple[str, BaseException]] = []
            for name, result in zip(task_names, gather_results, strict=True):
                if isinstance(result, BaseException):
                    errors.append((name, result))
                    logger.error(
                        'Generation failed for %s on context %s at index %d (retries exhausted): %s',
                        name, context_id, original_idx, result,
                    )
                elif name == 'embedding':
                    update_embeddings[vu_idx] = cast(list[ChunkEmbedding] | None, result)
                elif name == 'summary':
                    update_summaries[vu_idx] = cast(str | None, result)

            # Summary regeneration that ran but produced nothing (empty -> None)
            # on a text change must CLEAR the now-stale summary (it describes the
            # OLD text), mirroring the too-short branch above and the single-update
            # path. Only when the summary task itself did not error.
            summary_errored = any(name == 'summary' for name, _ in errors)
            if 'summary' in task_names and not summary_errored and update_summaries.get(vu_idx) is None:
                update_clear_summaries.add(vu_idx)

            if errors:
                error_details = '; '.join(
                    f'{name}: {type(exc).__name__}: {exc}' for name, exc in errors
                )
                if atomic:
                    raise ToolError(
                        f'Generation failed for context {context_id} at index {original_idx} '
                        f'after exhausting configured retries: {error_details}. No data was modified.',
                    )
                generation_errors.append((original_idx, context_id, f'Generation failed: {error_details}'))
                update_embeddings.setdefault(vu_idx, None)
                update_summaries.setdefault(vu_idx, None)
            else:
                update_embeddings.setdefault(vu_idx, None)
                if vu_idx not in update_summaries:
                    update_summaries[vu_idx] = None

                # Compress regenerated embeddings OUTSIDE the DB transaction
                # (generation-first invariant). No-op when compression is
                # disabled. Atomic mode aborts the batch; non-atomic marks
                # this entry failed and continues.
                try:
                    update_embeddings[vu_idx] = await generate_compression_with_timeout(
                        update_embeddings.get(vu_idx),
                    )
                except ToolError as compress_err:
                    if atomic:
                        raise ToolError(
                            f'Compression failed for context {context_id} at '
                            f'index {original_idx}: {compress_err}. '
                            f'No data was modified.',
                        ) from compress_err
                    generation_errors.append(
                        (original_idx, context_id,
                         f'Compression failed: {compress_err}'),
                    )
                    update_embeddings[vu_idx] = None
                    update_summaries[vu_idx] = None
                else:
                    # Count generated artifacts only AFTER compression succeeds, so a
                    # non-atomic compression failure (which discards this entry below)
                    # cannot inflate the generated counts in the response diagnostic.
                    if update_embeddings.get(vu_idx) is not None:
                        embeddings_generated_count += 1
                    if update_summaries.get(vu_idx) and isinstance(update_summaries[vu_idx], str):
                        summaries_generated_count += 1

        # In non-atomic mode, add generation errors to results and filter
        if not atomic and generation_errors:
            for original_idx, context_id, error in generation_errors:
                results.append(BulkUpdateResultItemDict(
                    index=original_idx,
                    context_id=context_id,
                    success=False,
                    updated_fields=None,
                    error=error,
                ))
            # Filter by ORIGINAL INDEX, not by context_id: two updates may target
            # the same context_id in one non-atomic batch, and filtering by a
            # context_id set would also drop the sibling that did NOT fail
            # (silently losing a successful update). Mirrors store_context_batch.
            failed_indices = {original_idx for original_idx, _, _ in generation_errors}
            validated_updates_final = [
                (vu_idx, u) for vu_idx, u in validated_updates_filtered
                if u['index'] not in failed_indices
            ]
        else:
            validated_updates_final = validated_updates_filtered

        if not validated_updates_final:
            results.sort(key=operator.itemgetter('index'))
            return BulkUpdateResponseDict(
                success=False,
                total=len(updates),
                succeeded=0,
                failed=len(updates),
                results=results,
                message='All updates failed validation, entry not found, or generation',
            )

        # Rebuild index_tree per-node summaries for each update that changes text,
        # in a SEPARATE fenced never-raise pass (isolated from the abort-mandatory
        # generation above). Updates without a text change are absent from the map,
        # so .get() yields None and leaves the node table untouched.
        update_index_nodes: dict[int, list[IndexNodeRow] | None] = {}
        for vu_idx, update in validated_updates_final:
            new_text = update.get('text')
            if new_text is not None:
                rebuilt = await generate_index_nodes_with_timeout(new_text)
                # Text changed: a None result must CLEAR the rows describing the old text
                # ([]) whenever the per-node layer is ENABLED, because navigate_context
                # surfaces those rows gated only on settings.index_tree.node_summaries_enabled
                # (no provider required). Gating the clear on node_summaries_enabled rather
                # than node_layer_active() covers the provider-removed-but-feature-on case
                # (None for lack of a provider) as well as total degradation, mirroring the
                # single-update path and the provider-independent embedding/summary clears.
                # When the feature is OFF the reader is also off, so None leaves them alone.
                update_index_nodes[vu_idx] = (
                    [] if rebuilt is None and settings.index_tree.node_summaries_enabled else rebuilt
                )

        # === PHASE 4: Single Atomic Transaction for ALL Database Operations ===
        backend = repos.context.backend

        if atomic:
            # ATOMIC MODE: All updates in a single transaction with retry
            max_retries = 2

            for attempt in range(max_retries + 1):
                try:
                    results_attempt: list[BulkUpdateResultItemDict] = []
                    cleared_attempt = 0
                    # Running version per id within THIS attempt: a same-id update
                    # later in the batch must present the version the earlier same-id
                    # update bumped to (the CAS is against the in-transaction row).
                    # Reset each attempt because a rolled-back attempt did not commit.
                    live_versions = dict(entry_versions)

                    async with backend.begin_transaction() as txn:
                        for idx, (vu_idx, update) in enumerate(validated_updates_final):
                            original_idx = update['index']
                            context_id = update['context_id']

                            # Heartbeat between entries (skip first)
                            if idx > 0:
                                await transaction_heartbeat(txn)

                            update_images = update.get('images')
                            updated_fields, summary_cleared = (
                                await execute_update_in_transaction(
                                    repos, txn,
                                    context_id=context_id,
                                    text=update.get('text'),
                                    metadata=update.get('metadata'),
                                    metadata_patch=update.get('metadata_patch'),
                                    summary=update_summaries.get(vu_idx),
                                    clear_summary=vu_idx in update_clear_summaries,
                                    tags=update.get('tags'),
                                    images=update_images,
                                    validated_images=update_images or [],
                                    chunk_embeddings=update_embeddings.get(vu_idx),
                                    embedding_model=settings.embedding.model,
                                    index_nodes=update_index_nodes.get(vu_idx),
                                    expected_version=live_versions.get(context_id),
                                )
                            )
                            if summary_cleared:
                                cleared_attempt += 1
                            bumps_version = update.get('text') is not None or update.get('metadata') is not None
                            if bumps_version and context_id in live_versions:
                                # update_context_entry bumped version by 1; a later
                                # same-id update in this batch must see the new value.
                                live_versions[context_id] += 1

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
                    # Accumulate the authoritative per-entry cleared count from this
                    # committed attempt (a rolled-back/retried attempt contributes nothing).
                    summaries_cleared_count += cleared_attempt
                    break

                except VersionConflictError as e:
                    # A concurrent writer modified one of these entries during
                    # generation. Atomic mode is all-or-nothing, so abort the whole
                    # batch with a clear error; the client can retry the request.
                    raise ToolError(
                        f'Concurrent modification during atomic batch update: {e}. Retry the request.',
                    ) from None
                except ToolError:
                    raise  # Logical error -- do not retry
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
            # NON-ATOMIC MODE: Each update in its own transaction (with retry).
            # Running version per id, persisting ACROSS entries (processed
            # sequentially): a same-id update later in the batch must see the
            # version the earlier same-id update committed.
            live_versions = dict(entry_versions)
            for vu_idx, update in validated_updates_final:
                original_idx = update['index']
                context_id = update['context_id']
                max_retries = 2
                attempt = 0
                version_conflicts = 0
                max_version_conflicts = 5

                while True:
                    try:
                        async with backend.begin_transaction() as txn:
                            update_images = update.get('images')
                            updated_fields_list, summary_cleared = (
                                await execute_update_in_transaction(
                                    repos, txn,
                                    context_id=context_id,
                                    text=update.get('text'),
                                    metadata=update.get('metadata'),
                                    metadata_patch=update.get('metadata_patch'),
                                    summary=update_summaries.get(vu_idx),
                                    clear_summary=vu_idx in update_clear_summaries,
                                    tags=update.get('tags'),
                                    images=update_images,
                                    validated_images=update_images or [],
                                    chunk_embeddings=update_embeddings.get(vu_idx),
                                    embedding_model=settings.embedding.model,
                                    index_nodes=update_index_nodes.get(vu_idx),
                                    expected_version=live_versions.get(context_id),
                                )
                            )

                            # COMMIT happens here for this update

                        results.append(BulkUpdateResultItemDict(
                            index=original_idx,
                            context_id=context_id,
                            success=True,
                            updated_fields=updated_fields_list,
                            error=None,
                        ))
                        if summary_cleared:
                            summaries_cleared_count += 1
                        bumps_version = update.get('text') is not None or update.get('metadata') is not None
                        if bumps_version and context_id in live_versions:
                            live_versions[context_id] += 1
                        break  # Success -- exit retry loop
                    except VersionConflictError as e:
                        # The row changed since we captured its version (a same-id
                        # update earlier in this batch, a concurrent external writer,
                        # or a commit whose ack was lost). Re-read the current version
                        # and retry so this entry self-heals into a success instead of
                        # a spurious failure (mirrors the single update_context path).
                        if version_conflicts >= max_version_conflicts:
                            logger.warning('Version conflict updating entry at index %d: %s', original_idx, e)
                            results.append(BulkUpdateResultItemDict(
                                index=original_idx,
                                context_id=context_id,
                                success=False,
                                updated_fields=None,
                                error=format_exception_message(e),
                            ))
                            break
                        version_conflicts += 1
                        try:
                            exists, _src, current_version = await repos.context.check_entry_exists(context_id)
                        except Exception as reread_error:
                            # A transient connection error during the conflict re-read
                            # gets the same bounded backoff/retry as the rest of the
                            # loop, then records a per-entry failure if exhausted.
                            if is_connection_error(reread_error) and attempt < max_retries:
                                delay = 0.5 * (2 ** attempt)
                                attempt += 1
                                logger.warning(
                                    'Connection error during version-conflict re-read of entry %d; '
                                    'retrying in %.1fs (attempt %d/%d): %s',
                                    original_idx, delay, attempt, max_retries, reread_error,
                                )
                                await asyncio.sleep(delay)
                                continue
                            logger.error(f'Failed to update entry at index {original_idx}: {reread_error}')
                            results.append(BulkUpdateResultItemDict(
                                index=original_idx,
                                context_id=context_id,
                                success=False,
                                updated_fields=None,
                                error=format_exception_message(reread_error),
                            ))
                            break
                        if not exists:
                            results.append(BulkUpdateResultItemDict(
                                index=original_idx,
                                context_id=context_id,
                                success=False,
                                updated_fields=None,
                                error=f'Context entry {context_id} not found',
                            ))
                            break
                        assert current_version is not None  # exists=True guarantees a version
                        live_versions[context_id] = current_version
                        continue
                    except ToolError as e:
                        # Logical error -- do not retry, record as failure
                        logger.error(f'Failed to update entry at index {original_idx}: {e}')
                        results.append(BulkUpdateResultItemDict(
                            index=original_idx,
                            context_id=context_id,
                            success=False,
                            updated_fields=None,
                            error=format_exception_message(e),
                        ))
                        break
                    except Exception as e:
                        if is_connection_error(e) and attempt < max_retries:
                            delay = 0.5 * (2 ** attempt)
                            attempt += 1
                            logger.warning(
                                'Non-atomic batch update entry %d failed with connection error, '
                                'retrying in %.1fs (attempt %d/%d): %s',
                                original_idx, delay, attempt, max_retries, e,
                            )
                            await asyncio.sleep(delay)
                            continue
                        # Non-connection error or max retries exceeded
                        logger.error(f'Failed to update entry at index {original_idx}: {e}')
                        results.append(BulkUpdateResultItemDict(
                            index=original_idx,
                            context_id=context_id,
                            success=False,
                            updated_fields=None,
                            error=format_exception_message(e),
                        ))
                        break

        # Sort results by index for consistent ordering
        results.sort(key=operator.itemgetter('index'))

        # Calculate summary
        succeeded = sum(1 for r in results if r['success'])
        failed = len(updates) - succeeded

        logger.info(f'Batch update completed: {succeeded}/{len(updates)} succeeded')

        # Both modes now accumulate summaries_cleared_count inline from the authoritative
        # per-entry summary_cleared returned by execute_update_in_transaction (atomic in the
        # committed-attempt loop, non-atomic per entry). The earlier post-hoc cross-product
        # over results x validated_updates_final matched by context_id only and miscounted
        # when two updates targeted the same context_id.

        message = build_batch_update_response_message(
            succeeded=succeeded,
            total=len(updates),
            embeddings_generated_count=embeddings_generated_count,
            summaries_generated_count=summaries_generated_count,
            summaries_cleared_count=summaries_cleared_count,
        )

        return BulkUpdateResponseDict(
            success=failed == 0,
            total=len(updates),
            succeeded=succeeded,
            failed=failed,
            results=results,
            message=message,
        )

    except ToolError:
        raise
    except Exception as e:
        logger.error(f'Error in batch update: {e}')
        raise ToolError(f'Batch update failed: {format_exception_message(e)}') from e


async def delete_context_batch(
    context_ids: Annotated[
        list[str] | None,
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
    Note: source filter alone is insufficient; it must be combined with another criterion.
    All associated data (tags, images) is also removed.

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

        # Resolve incoming IDs at the boundary: accept full 32/36-char IDs or
        # 8-31 char hex prefixes (uniform with delete_context).
        if context_ids:
            try:
                context_ids = await resolve_or_normalize_ids(context_ids, repos.context)
            except ValueError as e:
                raise ToolError(f'Invalid context ID: {e}') from e

        # Delete embeddings for the contexts the COMBINED criteria will actually
        # delete. On PostgreSQL the embedding rows cascade-delete with the context
        # row (ON DELETE CASCADE on the surviving vec table for the active
        # compression mode), so explicit cleanup is SQLite-only (vec0 virtual tables
        # lack CASCADE). The cleanup pre-queries the exact to-be-deleted subset, so
        # combining context_ids with source/older_than_days never deletes embeddings
        # for a context_id those filters exclude (which would orphan a surviving
        # entry). Gate on table PRESENCE (embedding_tables_exist), NOT the runtime
        # generation/compression toggles: a prior session's embeddings must still
        # be cleaned after the toggles flip off, or the FK-less vec0 rows orphan.
        # Mirrors the stale-embedding cleanup on the update path.
        backend = repos.context.backend
        # Resolve older_than_days to ONE absolute UTC cutoff for SQLite so the
        # embedding-cleanup pre-query and the row DELETE below share an identical age
        # boundary. SQLite's datetime('now') re-evaluates per statement, so without a
        # shared cutoff a row crossing the boundary between the pre-query and the delete
        # would be deleted while its vec0 embeddings (no CASCADE on the virtual table)
        # are left behind -- an orphan. created_at is stored as canonical UTC
        # "YYYY-MM-DD HH:MM:SS"; matching that format keeps the TEXT comparison correct.
        older_than_cutoff: str | None = None
        if older_than_days is not None and backend.backend_type == 'sqlite':
            older_than_cutoff = (
                datetime.now(UTC) - timedelta(days=older_than_days)
            ).strftime('%Y-%m-%d %H:%M:%S')

        if backend.backend_type == 'sqlite' and await repos.embeddings.embedding_tables_exist():
            try:
                affected_ids = await repos.context.get_ids_matching_batch_criteria(
                    context_ids=context_ids,
                    thread_ids=thread_ids,
                    source=source,
                    older_than_days=older_than_days,
                    older_than_cutoff=older_than_cutoff,
                )
                for cid in affected_ids:
                    try:
                        await repos.embeddings.delete(cid)
                    except Exception as e:
                        logger.warning(f'Failed to delete embedding for context {cid}: {e}')
            except Exception as e:
                logger.warning(f'Failed to pre-query context IDs for embedding cleanup: {e}')

        # Execute batch delete through repository (shares older_than_cutoff with the
        # cleanup pre-query above so SQLite cannot orphan vec0 rows).
        deleted_count, criteria_used = await repos.context.delete_contexts_batch(
            context_ids=context_ids,
            thread_ids=thread_ids,
            source=source,
            older_than_days=older_than_days,
            older_than_cutoff=older_than_cutoff,
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
