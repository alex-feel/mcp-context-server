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
from app.errors import ControlFlowError
from app.errors import format_exception_message
from app.repositories.embedding_repository import ChunkEmbedding
from app.repositories.index_node_repository import IndexNodeRow
from app.services.outline_service import OutlineNode
from app.services.outline_service import parse_outline
from app.settings import get_settings
from app.startup import MAX_IMAGE_SIZE_MB
from app.startup import MAX_TOTAL_SIZE_MB
from app.startup import get_chunking_service
from app.startup import get_embedding_provider
from app.startup import get_summary_provider
from app.summary.instructions import resolve_index_tree_node_summary_prompt
from app.summary.retry import compute_summary_total_timeout

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingsReconcileRequiredError(ControlFlowError):
    """Internal control-flow signal raised inside ``execute_store_in_transaction``.

    The read-only deduplication pre-check (performed by the caller OUTSIDE the
    transaction) skips embedding generation when a likely duplicate already has
    embeddings, on the assumption that this store will deduplicate into an
    UPDATE. If a concurrent same-thread write commits in the window between the
    pre-check and the transaction, ``store_with_deduplication`` can instead
    INSERT a brand-new entry. Committing that entry would leave a row with no
    embeddings while embedding generation is enabled, silently violating the
    generation-first guarantee.

    Raising this exception rolls the open transaction back and instructs the
    caller to regenerate embeddings OUTSIDE the transaction and retry the store.
    It is deliberately NOT a ``ToolError`` (so the ``except ToolError`` fast-path
    does not swallow it) and NOT a connection error (so it is not treated as a
    transient retry). ``text_content`` lets the caller regenerate embeddings for
    the exact entry that diverged.
    """

    def __init__(self, text_content: str) -> None:
        super().__init__('Embedding reconciliation required after deduplication divergence')
        self.text_content = text_content


# ---------------------------------------------------------------------------
# Concurrency limiters for embedding / summary / compression generation
# ---------------------------------------------------------------------------
#
# All three semaphores are constructed at module import time. asyncio.Semaphore
# has been parameterless (no ``loop`` argument) since Python 3.10, so its
# construction does NOT require a running event loop. Constructing at module
# scope is simpler than the prior lazy-init helpers and gives every caller a
# stable reference for the lifetime of the process.
#
# The semaphores are intentionally separate, one per physical resource:
#   * ``_embedding_semaphore``: bounds outbound HTTP concurrency to the
#     embedding provider.
#   * ``_summary_model_semaphore``: bounds outbound concurrency to the single
#     physical SUMMARY model. It is acquired by BOTH the flat document summary
#     (``generate_summary_with_timeout``) AND every per-node index_tree summary
#     (``_summarize_node``) -- both hit the same model, so ONE shared budget
#     caps global summary-model concurrency at ``SUMMARY_MAX_CONCURRENT`` no
#     matter how the flat and node passes overlap. That is what protects a small
#     local model (e.g. Ollama) from the overload the 3->2 de-tune fixed.
#   * ``_compression_semaphore``: bounds CPU-bound encoding offloaded via
#     ``asyncio.to_thread``; contention is for the GIL / CPU, not the event
#     loop, so a separate budget applies.
#   * ``_node_summary_semaphore``: a node-task LAUNCH / fan-out cap (NOT a second
#     model budget). It bounds how many ``_summarize_node`` coroutines are
#     in-flight at once so a many-heading document cannot create an unbounded
#     fan-out; the inner ``_summary_model_semaphore`` is what actually gates the
#     model call. Sized by ``INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT``.

_embedding_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.embedding.max_concurrent,
)
_summary_model_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.summary.max_concurrent,
)
_compression_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.compression.max_concurrent,
)
_node_summary_semaphore: asyncio.Semaphore = asyncio.Semaphore(
    settings.index_tree.max_concurrent,
)


def _reset_embedding_semaphore() -> None:
    """Rebind the embedding semaphore against the current settings value.

    Test fixtures that mutate ``settings.embedding.max_concurrent`` between
    cases call this to ensure the next ``async with _embedding_semaphore``
    block uses the freshly configured limit.
    """
    global _embedding_semaphore
    _embedding_semaphore = asyncio.Semaphore(settings.embedding.max_concurrent)


def _reset_summary_model_semaphore() -> None:
    """Rebind the shared summary-model semaphore against current settings.

    Test fixtures that mutate ``settings.summary.max_concurrent`` between cases
    call this so the next ``async with _summary_model_semaphore`` block -- used
    by BOTH the flat document summary and every per-node index_tree summary --
    uses the freshly configured limit.
    """
    global _summary_model_semaphore
    _summary_model_semaphore = asyncio.Semaphore(settings.summary.max_concurrent)


def _reset_compression_semaphore() -> None:
    """Rebind the compression semaphore against the current settings value.

    Test fixtures that mutate ``settings.compression.max_concurrent``
    between cases call this to ensure the next
    ``async with _compression_semaphore`` block uses the freshly
    configured limit.
    """
    global _compression_semaphore
    _compression_semaphore = asyncio.Semaphore(settings.compression.max_concurrent)


def _reset_node_summary_semaphore() -> None:
    """Rebind the node-summary semaphore against the current settings value.

    Test fixtures that mutate ``settings.index_tree.max_concurrent`` between
    cases call this to ensure the next ``async with _node_summary_semaphore``
    block uses the freshly configured limit.
    """
    global _node_summary_semaphore
    _node_summary_semaphore = asyncio.Semaphore(settings.index_tree.max_concurrent)


# The code-derived outline parse (parse_outline) is CPU-bound and O(text) over
# UNBOUNDED stored entry text, and the index_tree node leg runs it on the store/
# update (and batch) write path. A large entry is offloaded to a worker thread so
# the parse cannot pin the event loop and starve every other concurrent MCP
# request -- the same discipline the read paths apply (navigation._OFFLOAD_MIN_CHARS,
# grep_service._OFFLOAD_MIN_CHARS). Small entries stay inline to avoid a per-call
# thread hop. Unicode code points, not bytes.
_OFFLOAD_MIN_CHARS = 1_000_000


# Explicit re-export so type checkers do NOT flag the reset helpers as unused.
# These are called only by test fixtures that need to rebind the module-level
# semaphores against patched ``settings.*.max_concurrent`` values between
# cases; production code uses the semaphores directly without rebinding.
_RESET_HELPERS_EXPORT = (
    _reset_embedding_semaphore,
    _reset_summary_model_semaphore,
    _reset_compression_semaphore,
    _reset_node_summary_semaphore,
)


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
            # Chunked embedding for long documents. split_text (->
            # RecursiveCharacterTextSplitter.create_documents) is O(text) pure CPU
            # over unbounded entry text; offload a large entry to a worker thread so
            # it cannot pin the event loop on the store/update embedding leg (see
            # _OFFLOAD_MIN_CHARS), matching the read-path and index_tree node-leg
            # offloads. split_text's own len<=chunk_size fast path keeps small
            # entries inline regardless, so the threshold only spares the thread hop.
            if len(text) > _OFFLOAD_MIN_CHARS:
                chunks = await asyncio.to_thread(chunking_service.split_text, text)
            else:
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
        async with _embedding_semaphore:
            return await asyncio.wait_for(
                _generate_embeddings_for_text(text),
                timeout=total_timeout,
            )
    except TimeoutError:
        raise ToolError(
            f'Embedding generation exceeded total timeout ({total_timeout:.0f}s). '
            f'This may indicate the embedding provider is overloaded or unreachable.',
        ) from None


async def generate_compression_with_timeout(
    chunk_embeddings: list[ChunkEmbedding] | None,
) -> list[ChunkEmbedding] | None:
    """Compress each chunk's embedding into a bytes payload.

    Runs OUTSIDE any DB transaction, preserving the generation-first
    transactional-integrity invariant: when compression fails the storage
    write does not happen and the entry is not persisted.

    When ENABLE_EMBEDDING_COMPRESSION is false this is a no-op that returns
    the input unchanged so callers can wire the helper unconditionally.
    When true it calls the active provider's ``encode_sync`` for each chunk
    inside a worker thread (``asyncio.to_thread``) bounded by the
    compression semaphore, returning a fresh ``ChunkEmbedding`` list with
    the ``payload`` field populated.

    Args:
        chunk_embeddings: Embeddings returned by
            :func:`generate_embeddings_with_timeout`. ``None`` is passed
            through unchanged (no embeddings to compress).

    Returns:
        The same list of ``ChunkEmbedding`` objects when compression is
        disabled or ``chunk_embeddings is None``; a fresh list with
        ``payload`` populated otherwise.

    Raises:
        ToolError: If compression provider construction or any encode call
            fails. The transactional write is aborted by the propagating
            exception.
    """
    if not settings.compression.enabled:
        return chunk_embeddings

    if chunk_embeddings is None:
        return None

    # Defer provider import until enabled to keep numpy out of the import
    # graph for installations that skipped the compression extra. The
    # cached helper unifies provider construction across read (search)
    # and write (encode) paths: both reuse the same rotation matrix and
    # codebook arrays per process.
    from app.compression import get_cached_compression_provider

    try:
        provider = await get_cached_compression_provider()
    except Exception as e:
        raise ToolError(
            f'Compression provider initialization failed: '
            f'{format_exception_message(e)}',
        ) from e

    async def _encode_one(chunk: ChunkEmbedding) -> ChunkEmbedding:
        # Local import keeps numpy out of the hot import graph; this branch
        # only executes when compression is enabled (extra installed).
        import numpy as np

        vector = np.asarray([chunk.embedding], dtype=np.float32)
        # Acquire one semaphore permit per encode call so the configured
        # COMPRESSION_MAX_CONCURRENT limit governs in-flight CPU work
        # accurately. Wrapping the outer asyncio.gather() would let an
        # N-chunk batch run all N encodes under one permit, bypassing
        # the bound. Mirrors the established embedding/summary semaphore
        # pattern.
        async with _compression_semaphore:
            try:
                payload_bytes = await asyncio.to_thread(provider.encode_sync, vector)
            except Exception as e:
                raise ToolError(
                    f'Compression encode failed: {format_exception_message(e)}',
                ) from e
        return ChunkEmbedding(
            embedding=chunk.embedding,
            start_index=chunk.start_index,
            end_index=chunk.end_index,
            payload=payload_bytes,
        )

    return await asyncio.gather(*[_encode_one(c) for c in chunk_embeddings])


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
        async with _summary_model_semaphore:
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


async def generate_index_nodes_with_timeout(text: str) -> list[IndexNodeRow] | None:
    """Build index_tree node rows with per-node LLM summaries (NEVER raises).

    The code-derived outline is always parsed (pure CPU). Each heading section
    long enough to warrant one is summarized via the existing summary provider's
    ``summarize_with_prompt`` with a dedicated short prompt, bounded by a per-node
    timeout and the node-summary semaphore. This is the additive, fenced layer: a
    provider failure or timeout omits that node's summary and NEVER aborts the
    store -- the deliberate contrast with the abort-mandatory
    embedding/summary/compression helpers above.

    Args:
        text: The entry's full text content.

    Returns:
        ``None`` -- meaning "leave the node table untouched" -- when per-node
        summaries are disabled, no summary provider is configured, OR every
        attempted per-node summary failed/timed out (TOTAL degradation: a
        transient provider outage must NOT wipe previously-good stored rows on
        replace). Otherwise the list of node rows that received a summary --
        possibly empty when no section qualified (no headings, or all sections
        below the minimum length), which legitimately clears stale rows on replace.
    """
    if not settings.index_tree.node_summaries_enabled:
        return None

    provider = get_summary_provider()
    if provider is None:
        # Per-node summaries reuse the summary provider; with none configured the
        # feature is inert, so leave the node table untouched (None = no write).
        return None

    try:
        # parse_outline is O(text) pure CPU over unbounded entry text; offload a
        # large entry to a worker thread so it cannot pin the event loop (see
        # _OFFLOAD_MIN_CHARS), matching the read-path offload discipline.
        if len(text) > _OFFLOAD_MIN_CHARS:
            root = await asyncio.to_thread(parse_outline, text)
        else:
            root = parse_outline(text)
    except Exception as e:  # defensive: parsing is pure CPU and should not fail
        logger.warning('Index-tree outline parse failed; skipping node summaries: %s', e)
        return []

    nodes: list[OutlineNode] = []
    stack = list(root.children)
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(node.children)

    if not nodes:
        return []

    min_len = settings.index_tree.min_content_length
    timeout = settings.index_tree.timeout_s
    prompt = resolve_index_tree_node_summary_prompt()

    # A summary is ATTEMPTED only for sections that clear the minimum length;
    # shorter sections are deliberately skipped (not a failure). Counting attempts
    # lets us tell "nothing qualified" (attempted == 0 -> return [], legitimately
    # clearing stale rows) apart from "every attempt failed" (attempted > 0,
    # zero rows -> return None), so a transient provider outage cannot wipe
    # previously-good stored rows. len(section) == char_end - char_start (offsets
    # are code points), so this mirrors _summarize_node's own eligibility check.
    attempted = sum(1 for node in nodes if (node.char_end - node.char_start) >= min_len)

    async def _summarize_node(node: OutlineNode) -> IndexNodeRow | None:
        section = text[node.char_start:node.char_end]
        if len(section) < min_len:
            return None
        try:
            # Outer acquire = node-task fan-out cap (bounds how many node
            # coroutines run at once). Inner acquire = the SHARED summary-model
            # budget, so per-node calls and the flat document summary together
            # never exceed SUMMARY_MAX_CONCURRENT on the one physical model.
            async with _node_summary_semaphore, _summary_model_semaphore:
                result = await asyncio.wait_for(
                    provider.summarize_with_prompt(section, prompt),
                    timeout=timeout,
                )
        except Exception as e:
            logger.warning('Index-tree node summary failed for %s (skipped): %s', node.node_id, e)
            return None
        summary = result.strip()
        if not summary:
            return None
        return IndexNodeRow(
            node_id=node.node_id,
            level=node.level,
            ordinal=node.ordinal,
            title=node.title,
            node_summary=summary,
            char_start=node.char_start,
            char_end=node.char_end,
        )

    # _summarize_node never raises; return_exceptions is a defensive backstop so a
    # surprise (e.g. cancellation of a child) cannot turn into a store-aborting raise.
    results = await asyncio.gather(*[_summarize_node(node) for node in nodes], return_exceptions=True)
    rows = [result for result in results if isinstance(result, IndexNodeRow)]

    # TOTAL degradation: sections were eligible and summaries attempted, but every
    # one failed/timed out. Return None so callers PRESERVE existing stored rows
    # (replace_nodes_for_context with None is a no-op) instead of wiping them.
    if attempted > 0 and not rows:
        logger.warning(
            'Index-tree node summaries: all %d attempted section(s) failed; '
            'preserving any existing stored rows (skipping replace).',
            attempted,
        )
        return None

    return rows


def node_layer_active() -> bool:
    """Whether the index_tree per-node summary layer is ACTIVE (would attempt work).

    Mirrors the activation gate at the top of
    :func:`generate_index_nodes_with_timeout`: the feature toggle is on AND a
    summary provider is configured. Used on the STORE / DEDUPLICATION pre-check
    paths to set the ``nodes_pending`` reconcile flag: the read-only pre-check
    skips generation, so ``nodes_pending`` must record whether node work WOULD
    have been attempted -- which requires both the feature toggle on AND a
    provider present (no provider means no node call was made, hence nothing to
    reconcile).

    NOTE: the TEXT-CHANGE update stale-node clear does NOT use this helper. It
    gates on ``settings.index_tree.node_summaries_enabled`` directly (the
    provider-independent reader's gate that ``navigate_context`` consults), so a
    provider-removed-but-feature-on update still CLEARS stale rows that would
    otherwise let ``navigate_context`` mis-attach an old section summary to a new
    section sharing a reused heading slug.

    Returns:
        True when per-node summaries are enabled and a summary provider exists.
    """
    return settings.index_tree.node_summaries_enabled and get_summary_provider() is not None


async def embed_then_compress(text: str) -> list[ChunkEmbedding] | None:
    """Generate embeddings then compress them, as ONE abort-mandatory leg.

    Compression has a hard data dependency on the embeddings, so chaining keeps
    that dependency while letting the whole (embedding -> compression) leg
    overlap the concurrently-running summary/node leg in store/update instead of
    serializing after it. Both steps are generation-first: a failure propagates
    so nothing is saved. Compression is a no-op passthrough when
    ENABLE_EMBEDDING_COMPRESSION is false.

    Returns:
        The compressed ``ChunkEmbedding`` list, or ``None`` when no embedding
        provider is configured.
    """
    chunk_embeddings = await generate_embeddings_with_timeout(text)
    return await generate_compression_with_timeout(chunk_embeddings)


async def _nodes_after_summary(
    summary_task: asyncio.Task[str | None] | None,
    text: str,
) -> list[IndexNodeRow] | None:
    """Generate index_tree node summaries AFTER the flat summary completes.

    Awaiting the flat-summary task first gives the ABORT-MANDATORY flat summary
    strict precedence on the shared summary-model budget, so the never-raise node
    summaries can never starve it (no latency inversion). If the flat summary
    failed there is no store to enrich, so node generation is skipped. Never
    raises on a provider error (mirrors ``generate_index_nodes_with_timeout``); a
    cancellation still propagates.

    Returns:
        The node rows, or ``None`` when nodes are skipped/disabled or the flat
        summary failed.

    Raises:
        asyncio.CancelledError: Propagated (not swallowed) if this leg is cancelled.
    """
    if summary_task is not None:
        try:
            await summary_task
        except asyncio.CancelledError:
            raise
        except Exception:
            return None
    return await generate_index_nodes_with_timeout(text)


async def run_generation(
    text: str,
    source: str,
    *,
    run_embedding: bool,
    run_summary: bool,
    run_nodes: bool,
) -> tuple[list[ChunkEmbedding] | None, str | None, list[IndexNodeRow] | None]:
    """Run the embedding->compression, flat-summary, and node-summary legs concurrently.

    The embedding->compression leg (embedding model + CPU) and the summary legs
    (summary model) use disjoint resources, so they overlap genuinely -- this is
    what removes the node-summary serial tail and the post-gather compression wait
    from store/update latency. The node-summary leg starts only AFTER the flat
    summary finishes, keeping the abort-mandatory flat summary's precedence on the
    shared summary-model budget (no latency inversion).

    The embedding leg and the flat summary are ABORT-MANDATORY: both are awaited
    and ALL their errors are collected, so a failure reports every abort-mandatory
    leg deterministically. On EVERY exit path -- a normal return, the combined
    abort ToolError, OR an outer cancellation (MCP client disconnect / request
    timeout) landing on the abort-legs gather -- the ``finally`` cancels and awaits
    every created task that is not yet done. So no in-flight summary-model or
    embedding call outlives the request: in particular the never-raise node leg,
    which when ``run_summary=False`` does NOT transitively cancel via the flat
    summary, can never be orphaned holding the shared summary-model permit.

    Returns:
        ``(chunk_embeddings, summary_text, index_nodes)``; any leg that was not
        requested yields ``None``.

    Raises:
        ToolError: If an abort-mandatory leg (embeddings, compression, or the
            flat summary) fails after exhausting its configured retries; the
            message names every failed leg.
    """
    embed_task: asyncio.Task[list[ChunkEmbedding] | None] | None = None
    summary_task: asyncio.Task[str | None] | None = None
    node_task: asyncio.Task[list[IndexNodeRow] | None] | None = None

    try:
        if run_embedding:
            embed_task = asyncio.create_task(embed_then_compress(text))
        if run_summary:
            summary_task = asyncio.create_task(generate_summary_with_timeout(text, source))
        if run_nodes:
            node_task = asyncio.create_task(_nodes_after_summary(summary_task, text))

        # Await the ABORT-MANDATORY legs, collecting every error (return_exceptions so
        # one failure does not hide another); the never-raise node leg is NOT awaited
        # here so its timeouts can never delay an abort.
        abort_legs: list[tuple[str, asyncio.Task[Any]]] = []
        if embed_task is not None:
            abort_legs.append(('embedding', embed_task))
        if summary_task is not None:
            abort_legs.append(('summary', summary_task))

        errors: list[str] = []
        if abort_legs:
            await asyncio.gather(*(task for _, task in abort_legs), return_exceptions=True)
            for name, task in abort_legs:
                exc = task.exception()
                if exc is not None and not isinstance(exc, asyncio.CancelledError):
                    errors.append(f'{name}: {type(exc).__name__}: {exc}')

        if errors:
            # Abort-mandatory failure: surface a combined, deterministic error
            # naming every failed leg. The never-raise node leg (and any other
            # in-flight leg) is cancelled and awaited by the ``finally`` below, so
            # no in-flight summary-model call outlives the failed request.
            raise ToolError(
                'Generation failed after exhausting configured retries: ' + '; '.join(errors),
            )

        chunk_embeddings = embed_task.result() if embed_task is not None else None
        summary_text = summary_task.result() if summary_task is not None else None
        # The index_tree node leg is contractually NEVER-RAISE: a node-summary
        # failure or timeout must never abort a store (None preserves existing
        # node rows). _nodes_after_summary already swallows its own non-Cancelled
        # exceptions, so this guard is defense-in-depth that keeps the structural
        # generation-first guarantee intact against any future regression in the
        # node helpers -- a surprise node-leg exception is coerced to None rather
        # than aborting an otherwise-successful store.
        # ``except Exception`` deliberately does NOT catch ``CancelledError``
        # (it subclasses ``BaseException``), so an inner/outer cancellation still
        # propagates and is cleaned up by the ``finally`` below.
        index_nodes: list[IndexNodeRow] | None = None
        if node_task is not None:
            try:
                index_nodes = await node_task
            except Exception:
                logger.warning(
                    'Index-tree node leg raised unexpectedly; preserving existing '
                    'node rows (None).',
                    exc_info=True,
                )
                index_nodes = None
        return chunk_embeddings, summary_text, index_nodes
    finally:
        # Guarantee NO created task outlives this coroutine on ANY exit path. On a
        # normal return every task is already done (no-op). On the abort ToolError
        # the node leg may still be running. On OUTER cancellation, CancelledError
        # propagates straight out of the abort-legs gather BEFORE the final
        # ``await node_task`` runs, so without this cleanup the node leg -- when
        # run_summary=False it never transitively cancels via the flat summary --
        # would be orphaned and keep holding its shared summary-model permit,
        # progressively starving all summary generation. Cancelling and awaiting
        # every not-done task here always releases the embedding/summary-model
        # permits and ensures no orphaned model call survives the request.
        pending: list[asyncio.Task[Any]] = [
            task for task in (embed_task, summary_task, node_task) if task is not None and not task.done()
        ]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


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
    """Check if an exception is a transient DB error that is safe to retry.

    Despite the historical name, this classifier covers two transient
    families, both safe to retry because the database write that follows is
    idempotent (store_context deduplicates; update_context is a keyed
    partial update) and ALL embedding/summary/compression generation has
    already completed OUTSIDE the transaction (generation-first invariant) --
    so a retry re-runs only the rolled-back DB write and never regenerates or
    skips generation:

    1. Connection-level failures (the connection was lost, not a logical/data
       error): asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError,
       ConnectionResetError, OSError.
    2. Statement / lock-wait timeouts: asyncpg.exceptions.QueryCanceledError
       (SQLSTATE 57014). PostgreSQL cancels the statement when it exceeds the
       connection's statement_timeout (set to ~0.9 * POSTGRESQL_COMMAND_TIMEOUT_S
       in PostgreSQLBackend._setup_connection). Retrying with the SAME ceiling
       only helps a TRANSIENT lock-WAIT (the write was blocked behind a
       concurrent writer and the contention has since cleared); it does NOT
       help a write that is fundamentally slower than the ceiling -- for that
       case (notably fp32 mode, ENABLE_EMBEDDING_COMPRESSION=false, where each
       per-chunk INSERT performs in-transaction HNSW maintenance) the operator
       must also raise POSTGRESQL_COMMAND_TIMEOUT_S or keep compression ON. See
       docs/database-backends.md.

    QueryCanceledError is PostgreSQL-only; the isinstance check is harmless on
    SQLite, which never raises it (SQLite write contention surfaces as
    sqlite3.OperationalError 'database is locked', handled by the SQLite
    backend's own write-queue retry).

    Args:
        exc: The exception to classify

    Returns:
        True if the exception is a transient DB error safe for retry
    """
    return isinstance(exc, (
        asyncpg.InterfaceError,
        asyncpg.ConnectionDoesNotExistError,
        asyncpg.exceptions.QueryCanceledError,
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
    - Rejects a non-string 'data' or 'mime_type' value (the batch path is untyped)
    - Rejects empty/whitespace data (prevents silent 0-byte storage)
    - Defaults mime_type to 'image/png' when the key is absent
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

        # The batch tools accept untyped list[dict[str, Any]] entries, so a value
        # that would be rejected by the single-entry Pydantic list[dict[str, str]]
        # schema (a JSON null or number) can reach here. Reject a non-string "data"
        # before .strip()/base64 decode rather than crashing with an opaque
        # AttributeError, keeping the two paths consistent.
        data_val = cast(object, img['data'])
        if not isinstance(data_val, str):
            msg = f'Image {idx} has a non-string "data" field'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors
        img_data_str = data_val
        if not img_data_str or not img_data_str.strip():
            msg = f'Image {idx} has empty "data" field'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # mime_type is optional and defaults to 'image/png' only when the key is
        # ABSENT. A PRESENT but non-string value (a JSON null/number from the
        # untyped batch path) must be rejected, not bound into the mime_type
        # TEXT NOT NULL column: SQLite would silently coerce a number to text
        # while PostgreSQL raises a DataError, and a null trips the NOT NULL
        # constraint and aborts an atomic batch. This mirrors the single-entry
        # Pydantic list[dict[str, str]] contract, which already rejects a
        # non-string mime_type at the tool boundary.
        if 'mime_type' not in img:
            img['mime_type'] = 'image/png'
        else:
            mime_val = cast(object, img['mime_type'])
            if not isinstance(mime_val, str):
                msg = f'Image {idx} has a non-string "mime_type" field'
                if error_mode == 'raise':
                    raise ToolError(msg)
                errors.append(msg)
                return images, 'text', errors

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
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
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
    embedding_generation_enabled: bool = False,
    index_nodes: list[IndexNodeRow] | None = None,
    nodes_pending: bool = False,
) -> tuple[str, bool, bool]:
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
        embedding_generation_enabled: True when an embedding provider is
            configured. When True and this store INSERTs a new entry
            (was_updated False) while chunk_embeddings is None -- which only
            happens when the caller's read-only pre-check skipped generation
            expecting a deduplication UPDATE -- the transaction is aborted via
            EmbeddingsReconcileRequiredError so the caller can regenerate
            embeddings outside the transaction and retry. Defaults to False so
            callers unaware of the pre-check optimization keep prior behavior.
        nodes_pending: True when the index_tree node layer is active and the
            caller's pre-check skipped node generation for a likely duplicate.
            When True and this store INSERTs a new entry (was_updated False)
            while index_nodes is None, the transaction aborts via
            EmbeddingsReconcileRequiredError so the caller regenerates node
            summaries outside the transaction and retries -- even when embedding
            generation is disabled. Defaults to False.

    Returns:
        Tuple of (context_id, was_updated, embedding_stored):
        - context_id: ID of stored/updated entry
        - was_updated: True if deduplication updated existing entry
        - embedding_stored: True if embeddings were written to DB

    Raises:
        ToolError: If store_with_deduplication fails (returns falsy context_id).
        EmbeddingsReconcileRequiredError: If the store inserted a new entry while
            the caller's pre-check had skipped embedding generation, or (when
            nodes_pending) node-summary generation; signals the caller to
            regenerate the skipped legs outside the transaction and retry.
    """
    # Store context entry with deduplication
    context_id, was_updated = await repos.context.store_with_deduplication(
        thread_id=thread_id,
        source=source,
        content_type=content_type,
        text_content=text_content,
        metadata=metadata_str,
        summary=summary,
        # Preserve the existing content_type on a dedup UPDATE when no images are provided
        # this call (images are preserved, not replaced). Overwriting it would flip a
        # multimodal entry to 'text' while its image rows remain, making them
        # unretrievable. When images ARE provided, content_type is overwritten to match.
        preserve_content_type_on_dedup=not validated_images,
        txn=txn,
    )

    if not context_id:
        raise ToolError('Failed to store context')

    # Generation-first reconciliation: the caller's read-only pre-check skips
    # embedding AND node-summary generation when a likely duplicate already has
    # them, expecting this store to deduplicate into an UPDATE. If a concurrent
    # same-thread write committed in the meantime, store_with_deduplication can
    # instead INSERT a brand-new entry (was_updated False). Committing now would
    # persist a row missing its embeddings (when generation is enabled) or its
    # per-node summaries (when the node layer is active), violating the
    # generation-first guarantee. Abort so the caller regenerates the skipped
    # legs OUTSIDE the transaction and retries. The node reconcile is decoupled
    # from embeddings so the node layer is repaired even when embeddings are off.
    needs_embedding_reconcile = embedding_generation_enabled and chunk_embeddings is None
    needs_node_reconcile = nodes_pending and index_nodes is None
    if not was_updated and (needs_embedding_reconcile or needs_node_reconcile):
        raise EmbeddingsReconcileRequiredError(text_content)

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
            embedding_exists = await repos.embeddings.exists(context_id, txn=txn)
            should_store = not embedding_exists
            if not should_store:
                logger.debug(
                    'Skipping embedding storage for deduplicated context %s '
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

    # Replace index_tree node summaries atomically. None means the per-node
    # summary feature is off, so the node table is left untouched. An empty list
    # clears stale rows, but only on a fresh INSERT: on a dedup UPDATE a
    # post-reconcile [] (coerced from total node-summary degradation) must NOT
    # wipe an existing entry's node rows, so an empty list is suppressed when
    # was_updated is True.
    if index_nodes is not None and (not was_updated or index_nodes):
        await repos.index_nodes.replace_nodes_for_context(context_id, index_nodes, txn=txn)

    return context_id, was_updated, embedding_stored


async def execute_update_in_transaction(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    *,
    context_id: str,
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
    index_nodes: list[IndexNodeRow] | None = None,
    expected_version: int | None = None,
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
        context_id: ID (32-char canonical hex) of entry to update.
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
            expected_version=expected_version,
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
        image_count = await repos.images.count_images_for_context(context_id, txn=txn)
        current_content_type = 'multimodal' if image_count > 0 else 'text'
        stored_content_type = await repos.context.get_content_type(context_id, txn=txn)
        if stored_content_type != current_content_type:
            await repos.context.update_content_type(
                context_id, current_content_type, txn=txn,
            )
            updated_fields.append('content_type')

    # Embeddings describe text_content, so a text change invalidates the stored vectors.
    if chunk_embeddings is not None:
        # New embeddings were generated -> replace the old chunks.
        await transaction_heartbeat(txn)
        await repos.embeddings.delete_all_chunks(context_id, txn=txn)
        await repos.embeddings.store_chunked(
            context_id=context_id,
            chunk_embeddings=chunk_embeddings,
            model=embedding_model,
            txn=txn,
        )
        updated_fields.append('embedding')
    elif text is not None and await repos.embeddings.embedding_tables_exist(txn=txn):
        # Text changed but embeddings were NOT regenerated (no embedding provider at
        # update time -- generation disabled/absent). The stored chunks describe the
        # REPLACED text, so DELETE them rather than leave stale vectors that semantic
        # search would match against the old content. Guarded by embedding_tables_exist
        # so a database that never provisioned embeddings is a safe no-op. (Mirrors the
        # stale-summary clear on this same text-change path.)
        await transaction_heartbeat(txn)
        if await repos.embeddings.delete_all_chunks(context_id, txn=txn):
            updated_fields.append('embedding')

    # Replace index_tree node summaries atomically. None means leave the node
    # table untouched (feature off, or text unchanged so the caller did not
    # recompute); an empty list clears stale rows when text shrank below the
    # summary thresholds.
    if index_nodes is not None:
        await repos.index_nodes.replace_nodes_for_context(context_id, index_nodes, txn=txn)

    return updated_fields, clear_summary
