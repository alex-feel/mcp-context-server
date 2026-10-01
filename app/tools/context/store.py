"""The store_context tool: store one context entry, deduplicating client retransmits."""

import asyncio
import json
import logging
from typing import Annotated
from typing import Literal

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_effective_principal
from app.auth import visibility_denied_reason
from app.errors import format_exception_message
from app.metadata_types import non_finite_metadata_error
from app.models import MAX_IMAGES_PER_ENTRY
from app.models import MAX_TAG_LENGTH
from app.models import MAX_TAGS_PER_ENTRY
from app.models import MAX_THREAD_ID_LENGTH
from app.settings import get_settings
from app.startup import ensure_repositories
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.tools._generation import embed_then_compress
from app.tools._generation import generate_index_nodes_with_timeout
from app.tools._generation import generate_summary_with_timeout
from app.tools._generation import node_layer_active
from app.tools._generation import run_generation
from app.tools._responses import build_store_response_message
from app.tools._transactions import EmbeddingsReconcileRequiredError
from app.tools._transactions import execute_store_in_transaction
from app.tools._transactions import is_connection_error
from app.tools._validation import reject_invalid_indexed_values
from app.tools._validation import reject_oversized_tags
from app.tools._validation import reject_unstorable_input
from app.tools._validation import validate_and_normalize_images
from app.types import MetadataDict
from app.types import StoreContextSuccessDict

logger = logging.getLogger(__name__)
settings = get_settings()


async def store_context(
    thread_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_THREAD_ID_LENGTH,
            description='Unique identifier for the conversation/task thread '
            f'(at most {MAX_THREAD_ID_LENGTH} characters)',
        ),
    ],
    source: Annotated[Literal['user', 'agent'], Field(description="Either 'user' or 'agent'")],
    text: Annotated[str, Field(min_length=1, description='Text content to store')],
    images: Annotated[
        list[dict[str, str]] | None,
        Field(
            max_length=MAX_IMAGES_PER_ENTRY,
            description=f'List of base64 encoded images with mime_type. Max {MAX_IMAGES_PER_ENTRY} images, '
            'each image max 10MB, total max 100MB',
        ),
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
    tags: Annotated[
        list[Annotated[str, Field(max_length=MAX_TAG_LENGTH)]] | None,
        Field(
            max_length=MAX_TAGS_PER_ENTRY,
            description=f'List of tags (normalized to lowercase). Max {MAX_TAGS_PER_ENTRY} tags, '
            f'each at most {MAX_TAG_LENGTH} characters',
        ),
    ] = None,
    visibility: Annotated[
        Literal['private', 'shared', 'public'] | None,
        Field(
            description='Access visibility for a newly stored entry: private (owner only), '
            'shared (owner + explicit grants), public (any principal). Omitted uses the '
            "server's configured default. Publishing as public may require a configured "
            'role. Ignored when deduplication updates an existing entry.',
        ),
    ] = None,
) -> StoreContextSuccessDict:
    """Store a context entry.

    All agents working on the same task should use the same thread_id to share context.

    Deduplication: if an entry with identical thread_id, source, and text already exists,
    the existing entry is updated instead of creating a duplicate:
    - metadata: New values override existing; omitting metadata preserves current values
    - tags: REPLACED with new list if provided; preserved if tags=None
    - images: REPLACED with new list if provided; preserved if images=None
    - visibility/ownership: NEVER changed by a deduplication update (a retransmit
      must not re-own or re-publish the existing row)

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

        # Resolve the effective principal (verified token, or the configured
        # default principal) and the visibility to stamp, then enforce the
        # publish gate on the EFFECTIVE value: a caller-omitted visibility that
        # defaults to 'public' is still a publish and still needs the role.
        principal = resolve_effective_principal()
        effective_visibility: str = (
            visibility if visibility is not None else settings.access_control.default_visibility
        )
        denial = visibility_denied_reason(effective_visibility, principal)
        if denial is not None:
            raise ToolError(denial)
        author_group_grants: frozenset[str] = (
            principal.groups
            if settings.access_control.default_group_grants == 'author_groups'
            else frozenset()
        )

        # Determine content type and validate images
        validated_images, content_type, _ = validate_and_normalize_images(images, error_mode='raise')

        # Reject non-finite floats in metadata BEFORE generation: they serialize
        # to invalid JSON that PostgreSQL's jsonb parser rejects, so the same
        # entry would store on SQLite but fail on PostgreSQL -- and the failure
        # would otherwise surface only after a wasted generation pass.
        if metadata is not None:
            metadata_error = non_finite_metadata_error(metadata)
            if metadata_error is not None:
                raise ToolError(metadata_error)

        # Reject an embedded NUL or unpaired UTF-16 surrogate in any user-supplied
        # string (thread_id, text, tags, metadata) BEFORE generation: PostgreSQL
        # cannot store it, so the same entry would store on SQLite but hard-fail on
        # PostgreSQL after a wasted generation pass, inside the transaction where a
        # non-ControlFlowError charges the circuit breaker.
        reject_unstorable_input(thread_id=thread_id, text=text, tags=tags, metadata=metadata)

        # Reject an oversized tag list BEFORE generation. The wire schema already
        # advertises both bounds, but the shared chokepoint is what makes the four
        # write tools agree: without the per-tag length cap an over-long tag stores
        # on SQLite and aborts the PostgreSQL INSERT on idx_tags_tag inside the
        # transaction, after a wasted generation pass and while charging the breaker.
        reject_oversized_tags(tags)

        # Reject an over-long thread_id or indexed metadata value for the same reason,
        # and at the same point: each also lands in a PostgreSQL btree index whose
        # index-tuple ceiling aborts the INSERT inside the transaction where SQLite
        # stores the identical value. The same guard rejects a value that cannot
        # survive the SQL cast a typed METADATA_INDEXED_FIELDS entry puts inside its
        # expression index, which PostgreSQL evaluates on every INSERT.
        reject_invalid_indexed_values(thread_id=thread_id, metadata=metadata)

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

        # Performance optimization: pre-check for likely duplicates (read-only).
        # The candidate id and its stored summary come from ONE statement-level
        # snapshot (see DuplicateCandidate): a separate later summary read could
        # observe a row version a concurrent update committed in between,
        # pairing a reused summary with text it does not describe.
        likely_duplicate_id: str | None = None
        duplicate_summary: str | None = None

        if get_embedding_provider() is not None or get_summary_provider() is not None:
            duplicate_candidate = await repos.context.check_latest_is_duplicate(
                thread_id=thread_id,
                source=source,
                text_content=text,
            )
            if duplicate_candidate is not None:
                likely_duplicate_id = duplicate_candidate.context_id
                duplicate_summary = duplicate_candidate.summary

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
        summary_reused = False
        if get_summary_provider() is not None:
            min_content_length = settings.summary.min_content_length
            if min_content_length > 0 and len(text) < min_content_length:
                logger.info(
                    'Skipping summary generation: text length %d < min_content_length %d',
                    len(text), min_content_length,
                )
            elif likely_duplicate_id is not None:
                if duplicate_summary is not None:
                    summary_text = duplicate_summary  # reuse; no model call
                    summary_reused = True
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
                        owner_id=principal.principal_id,
                        visibility=effective_visibility,
                        author_group_grants=author_group_grants,
                        metadata_str=metadata_str,
                        summary=summary_text,
                        tags=tags,
                        validated_images=validated_images,
                        images_provided=images is not None,
                        chunk_embeddings=chunk_embeddings,
                        embedding_model=settings.embedding.model,
                        embedding_generation_enabled=get_embedding_provider() is not None,
                        index_nodes=index_nodes,
                        nodes_pending=node_layer_active() and likely_duplicate_id is not None,
                        summary_pending=summary_reused and not summary_generated,
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
                # A REUSED summary was read from the since-diverged candidate and
                # may describe different text; regenerate it for THIS text so the
                # divergence INSERT cannot persist a mismatched summary.
                if summary_reused and not summary_generated:
                    summary_text = await generate_summary_with_timeout(text, source)
                    summary_generated = summary_text is not None
                    summary_reused = False
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
