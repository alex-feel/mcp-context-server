"""The update_context tool: update one existing context entry."""

import asyncio
import logging
from typing import Annotated
from typing import Literal

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_effective_principal
from app.auth import visibility_denied_reason
from app.errors import format_exception_message
from app.ids import resolve_or_normalize_id
from app.metadata_types import non_finite_metadata_error
from app.models import MAX_IMAGES_PER_ENTRY
from app.models import MAX_TAG_LENGTH
from app.models import MAX_TAGS_PER_ENTRY
from app.repositories.context_repository.records import VersionConflictError
from app.repositories.embedding_repository.records import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.tools._generation import run_generation
from app.tools._responses import build_update_response_message
from app.tools._transactions import EntryNotFoundError
from app.tools._transactions import execute_update_in_transaction
from app.tools._transactions import is_connection_error
from app.tools._transactions import reread_entry_version
from app.tools._validation import reject_invalid_indexed_values
from app.tools._validation import reject_oversized_tags
from app.tools._validation import reject_unstorable_input
from app.tools._validation import validate_and_normalize_images
from app.types import MetadataDict
from app.types import UpdateContextSuccessDict

logger = logging.getLogger(__name__)
settings = get_settings()


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
    tags: Annotated[
        list[Annotated[str, Field(max_length=MAX_TAG_LENGTH)]] | None,
        Field(
            max_length=MAX_TAGS_PER_ENTRY,
            description=f'New tags list (REPLACES all existing). Max {MAX_TAGS_PER_ENTRY} tags, '
            f'each at most {MAX_TAG_LENGTH} characters',
        ),
    ] = None,
    images: Annotated[
        list[dict[str, str]] | None,
        Field(
            max_length=MAX_IMAGES_PER_ENTRY,
            description=f'New images with base64 data and mime_type (REPLACES all existing). '
            f'Max {MAX_IMAGES_PER_ENTRY} images',
        ),
    ] = None,
    visibility: Annotated[
        Literal['private', 'shared', 'public'] | None,
        Field(
            description='New access visibility: private (owner only), shared (owner + '
            'explicit grants), public (any principal). Only the entry owner may change '
            'visibility, and publishing as public may require a configured role.',
        ),
    ] = None,
) -> UpdateContextSuccessDict:
    """Update an existing context entry.

    Immutable fields: id, thread_id, source, created_at, ownership (cannot be changed)
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
        # Note: metadata_patch and visibility are also valid update fields
        if (
            text is None
            and metadata is None
            and metadata_patch is None
            and tags is None
            and images is None
            and visibility is None
        ):
            raise ToolError('At least one field must be provided for update')

        # Validate images early (before any operations)
        validated_images, _, _ = validate_and_normalize_images(images, error_mode='raise')

        # Reject an oversized tag list here in the input-validation phase, for the
        # same cross-backend-parity and breaker reason as the store path: an
        # over-long tag stores on SQLite and aborts the PostgreSQL INSERT on
        # idx_tags_tag inside the transaction, after a wasted generation pass.
        reject_oversized_tags(tags)

        # Reject an over-long or cast-incompatible value under an INDEXED metadata key,
        # in both the replacement and the merge-patch form: each lands in that field's
        # expression btree index (idx_metadata_<field>), whose index-tuple ceiling --
        # and, for a typed field, whose SQL cast -- aborts the UPDATE inside the
        # transaction where SQLite stores it. thread_id is immutable on this path, so
        # only metadata is checked. Runs in the input-validation phase, alongside the
        # tag caps and before any database work.
        for meta_value in (metadata, metadata_patch):
            reject_invalid_indexed_values(metadata=meta_value)

        # Get repositories
        repos = await ensure_repositories()

        # Boundary normalization: accept full hex (32 or 36 chars) or 8-31 char hex prefix
        try:
            context_id = await resolve_or_normalize_id(context_id, repos.context)
        except ValueError as e:
            raise ToolError(f'Invalid context ID: {e}') from e

        # Check if entry exists; capture source and the optimistic-concurrency
        # version BEFORE generation so a concurrent writer that commits during
        # our (slow) generation is caught by the conditional write below. The
        # same probe returns the immutable owner_id backing the visibility gate.
        probe = await repos.context.check_entry_exists(context_id)
        if not probe.exists:
            raise ToolError(f'Context entry with ID {context_id} not found')
        entry_source = probe.source
        expected_version = probe.version
        assert entry_source is not None  # guaranteed by exists=True

        # Visibility changes are owner-only, and publishing as 'public' may
        # additionally require the configured publish role. owner_id is
        # immutable, so this pre-generation read cannot go stale.
        if visibility is not None:
            principal = resolve_effective_principal()
            if probe.owner_id != principal.principal_id:
                raise ToolError(
                    f'Only the owner may change the visibility of context {context_id}',
                )
            denial = visibility_denied_reason(visibility, principal)
            if denial is not None:
                raise ToolError(denial)

        # Reject non-finite floats in metadata (full or patch) BEFORE generation:
        # they serialize to invalid JSON that PostgreSQL rejects, so the update
        # would succeed on SQLite but fail on PostgreSQL after a wasted pass.
        for meta_value in (metadata, metadata_patch):
            if meta_value is not None:
                metadata_error = non_finite_metadata_error(meta_value)
                if metadata_error is not None:
                    raise ToolError(metadata_error)

        # Reject an embedded NUL or unpaired UTF-16 surrogate in any user-supplied
        # string (text, tags, metadata, metadata_patch) BEFORE generation, for the
        # same cross-backend-parity and circuit-breaker reason as the store path.
        reject_unstorable_input(text=text, tags=tags, metadata=metadata, metadata_patch=metadata_patch)

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
            # because those rows describe the OLD text: any heading whose slug the edit
            # retains would otherwise have its pre-edit summary mis-attached to the new
            # section (node_id is a pure function of heading path). This covers total
            # degradation (every section summary failed) AND the provider-removed case
            # (generation returns None for lack of a provider). The clear is UNCONDITIONAL
            # (not gated on node_summaries_enabled) so a disable/edit/re-enable cycle cannot
            # resurface stale rows: while the feature is off the reader is also off, but the
            # rows must not survive the edit and reappear when the toggle is turned back on.
            # This mirrors the provider-independent stale-clear the embedding and flat-summary
            # legs already perform on this same text-change path; replace_nodes_for_context
            # pre-checks table existence, so clearing when the table is absent is a safe no-op.
            if index_nodes is None:
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
                        visibility=visibility,
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
                # Refresh the token before re-entering the write. A transient fault
                # during the refresh is retried inside the helper, on the READ alone:
                # re-running the write transaction with the token whose compare-and-set
                # just failed is doomed by construction (version is monotonic) and would
                # burn a conflict slot on what was only a connection blip.
                exists, current_version = await reread_entry_version(repos, context_id)
                if not exists:
                    raise ToolError(f'Context entry with ID {context_id} not found') from None
                expected_version = current_version
                logger.info(
                    'Version conflict updating context %s; retrying (%d/%d)',
                    context_id, version_conflicts, max_version_conflicts,
                )
                continue

            except EntryNotFoundError:
                # The entry was deleted concurrently between the pre-generation
                # existence check and this transaction (or the id is stale). Surface
                # a clean not-found error; EntryNotFoundError is a ControlFlowError,
                # so the failed write never charged the circuit breaker, and it is
                # terminal -- no retry can resurrect a deleted row.
                raise ToolError(f'Context entry with ID {context_id} not found') from None

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
