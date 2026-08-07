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
import time
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

if TYPE_CHECKING:
    from app.backends.base import TransactionContext
    from app.repositories import RepositoryContainer

import asyncpg
from fastmcp.exceptions import ToolError

from app.backends.sqlite_backend import is_sqlite_locked_error
from app.embeddings.retry import compute_embedding_total_timeout
from app.errors import ControlFlowError
from app.errors import format_exception_message
from app.metadata_types import sanitize_pg_unstorable_text
from app.metadata_types import unstorable_string_error
from app.models import MAX_IMAGES_PER_ENTRY
from app.models import MAX_INDEXED_METADATA_VALUE_LENGTH
from app.models import MAX_TAG_LENGTH
from app.models import MAX_TAGS_PER_ENTRY
from app.models import MAX_THREAD_ID_LENGTH
from app.models import normalize_base64_image_data
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


def reject_unstorable_input(**fields: object) -> None:
    """Raise ``ToolError`` if any user-supplied field carries a PostgreSQL-unstorable string.

    A ``thread_id``, ``text``, tag, or metadata string carrying an embedded NUL
    (U+0000) or an unpaired UTF-16 surrogate stores on SQLite but is rejected by
    PostgreSQL (TEXT bind or jsonb parse), so the same request diverges across
    backends and -- on the store/update write path -- the failure surfaces only
    after a wasted generation pass, inside the transaction where a
    non-ControlFlowError charges the circuit breaker. Rejecting at the tool
    boundary in the input-validation phase (before any generation or connection
    scope) fails both backends fast and identically with a clean client error and
    zero breaker charge, mirroring the ``non_finite_metadata_error`` guard.
    ``unstorable_string_error`` walks metadata dict keys/values and tag lists, so
    a scalar string and a nested structure are both validated.

    Args:
        **fields: Named user-supplied values to validate; the name is surfaced in
            the error message so the caller learns which field is at fault.

    Raises:
        ToolError: On the first field containing an unstorable string.
    """
    for name, value in fields.items():
        if value is None:
            continue
        message = unstorable_string_error(value)
        if message is not None:
            raise ToolError(f'{name}: {message}')


def tag_limits_error(tags: list[str] | None) -> str | None:
    """Return an error message when a tag list breaches a per-entry write cap.

    Two dimensions are checked, both of which the write path previously left
    unbounded:

    * COUNT -- every tag is a separate INSERT issued inside the open store /
      update transaction (on SQLite that is the single writer), and every later
      search response re-hydrates the entry's full tag list, so one oversized
      list stalls the write path once and inflates every subsequent read
      forever.
    * PER-TAG LENGTH -- ``idx_tags_tag`` is a PostgreSQL btree index whose
      index-tuple ceiling (~2704 bytes) rejects an oversized tag inside the
      transaction, AFTER a full generation pass and while charging the circuit
      breaker, while SQLite stores the same value happily. Capping at the tool
      boundary makes both backends accept and reject identically, and keeps the
      value migratable between them.

    This is the single source of truth shared by the typed single-entry tools
    (whose ``Field`` declarations advertise the same bounds in the MCP wire
    schema) and the untyped batch tools (whose per-entry lists never pass
    through a Pydantic model), mirroring how ``MAX_IMAGES_PER_ENTRY`` is
    enforced by :func:`validate_and_normalize_images`.

    Args:
        tags: The client-supplied tag list, or ``None`` when absent.

    Returns:
        An error message describing the first breach, or ``None`` when the list
        is within both caps.
    """
    if tags is None:
        return None
    if len(tags) > MAX_TAGS_PER_ENTRY:
        return f'Too many tags: {len(tags)} provided, maximum is {MAX_TAGS_PER_ENTRY} per entry'
    for idx, tag in enumerate(tags):
        if len(tag) > MAX_TAG_LENGTH:
            return (
                f'Tag {idx} is too long: {len(tag)} characters, maximum is '
                f'{MAX_TAG_LENGTH} characters per tag'
            )
    return None


def reject_oversized_tags(tags: list[str] | None) -> None:
    """Raise ``ToolError`` when a tag list breaches a per-entry write cap.

    The raising wrapper used by the single-entry tools; the batch tools call
    :func:`tag_limits_error` directly so they can record a per-entry failure
    instead of aborting the whole request.

    Args:
        tags: The client-supplied tag list, or ``None`` when absent.

    Raises:
        ToolError: When the list exceeds the count or per-tag length cap.
    """
    message = tag_limits_error(tags)
    if message is not None:
        raise ToolError(message)


def indexed_value_limits_error(
    thread_id: str | None = None,
    metadata: object = None,
) -> str | None:
    """Return an error message when a client value that lands in a btree index is too long.

    ``tags`` is not the only write-path value PostgreSQL indexes with a btree whose
    index-tuple ceiling (~2704 bytes) rejects an oversized entry INSIDE the store
    transaction -- after a full generation pass, and while charging the circuit
    breaker -- where SQLite stores it happily:

    * ``thread_id`` feeds ``idx_thread_id``, ``idx_thread_source``,
      ``idx_context_entries_dedup_hash`` and ``idx_thread_created``.
    * every value stored under a ``METADATA_INDEXED_FIELDS`` key feeds that field's
      expression index ``idx_metadata_<field>`` on ``metadata->>'<field>'``.

    Values under NON-indexed metadata keys are deliberately not capped: jsonb imposes
    no such limit and the always-present GIN index uses ``jsonb_path_ops``, which
    hashes its entries. Only the top level of ``metadata`` is inspected, because
    ``metadata->>'<field>'`` addresses top-level keys only.

    Args:
        thread_id: The client-supplied thread identifier, or None when absent.
        metadata: The client-supplied metadata mapping, or None when absent.

    Returns:
        An error message describing the first breach, or None when every indexed
        value is within its cap.
    """
    if thread_id is not None and len(thread_id) > MAX_THREAD_ID_LENGTH:
        return (
            f'thread_id is too long: {len(thread_id)} characters, maximum is '
            f'{MAX_THREAD_ID_LENGTH} characters'
        )
    if isinstance(metadata, dict):
        indexed_fields = settings.storage.metadata_indexed_fields
        for key, value in cast('dict[object, object]', metadata).items():
            if not isinstance(key, str) or key not in indexed_fields:
                continue
            if isinstance(value, str) and len(value) > MAX_INDEXED_METADATA_VALUE_LENGTH:
                return (
                    f'metadata field {key!r} is indexed and its value is too long: '
                    f'{len(value)} characters, maximum is '
                    f'{MAX_INDEXED_METADATA_VALUE_LENGTH} characters'
                )
    return None


def entry_boundary_error(
    *,
    thread_id: str | None = None,
    text: str | None = None,
    tags: object = None,
    metadata: object = None,
    metadata_patch: object = None,
) -> str | None:
    """Return the first cross-backend boundary error for one client-supplied entry.

    The single chokepoint the UNTYPED batch paths use where the typed single-entry
    tools rely on their wire schema plus the individual raising guards. It bundles
    both families of "SQLite accepts it, PostgreSQL rejects it" input so the batch
    loops carry one call instead of a long boolean chain: PostgreSQL-unstorable
    strings (embedded NUL, unpaired UTF-16 surrogate) and the length caps on the
    values that land in a PostgreSQL btree index.

    Every argument is optional so a call site passes only the fields that shape
    exists (``update`` has no ``thread_id``; ``store`` has no ``metadata_patch``).
    Absent values are skipped, and a non-string / non-container value is ignored by
    the underlying checks rather than raising.

    Args:
        thread_id: The client-supplied thread identifier, when the shape has one.
        text: The client-supplied text content, when provided.
        tags: The client-supplied tag list, when provided.
        metadata: The client-supplied metadata mapping (full replacement).
        metadata_patch: The client-supplied merge-patch mapping, when provided.

    Returns:
        The first error message found, or None when the entry is acceptable on both
        backends.
    """
    return (
        unstorable_string_error(thread_id)
        or unstorable_string_error(text)
        or unstorable_string_error(tags)
        or unstorable_string_error(metadata)
        or unstorable_string_error(metadata_patch)
        or indexed_value_limits_error(thread_id=thread_id, metadata=metadata)
        or indexed_value_limits_error(metadata=metadata_patch)
    )


def reject_oversized_indexed_values(
    thread_id: str | None = None,
    metadata: object = None,
) -> None:
    """Raise ``ToolError`` when an indexed client value breaches its write cap.

    The raising wrapper used by the single-entry tools; the batch tools call
    :func:`indexed_value_limits_error` directly so they can record a per-entry
    failure instead of aborting the whole request.

    Args:
        thread_id: The client-supplied thread identifier, or None when absent.
        metadata: The client-supplied metadata mapping, or None when absent.

    Raises:
        ToolError: When an indexed value exceeds its length cap.
    """
    message = indexed_value_limits_error(thread_id=thread_id, metadata=metadata)
    if message is not None:
        raise ToolError(message)


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


class EntryNotFoundError(ControlFlowError):
    """Internal control-flow signal: the target context entry does not exist.

    Raised INSIDE an update transaction when a write targets a row that is gone
    (deleted concurrently, or a stale/wrong id): update_context_entry or
    patch_metadata reports no such row, or the tags-only / images-only path finds
    the parent missing before replacing its children. It is deliberately a
    ``ControlFlowError`` -- NOT a ``ToolError`` (so the ``except ToolError``
    fast-path does not swallow it) and NOT a connection error -- so the failed
    write is a clean client-input outcome that is NOT charged to the circuit
    breaker (a missing row is not a backend fault). The update tools catch it
    OUTSIDE the transaction and convert it to a not-found ``ToolError``.
    """

    def __init__(self, context_id: str) -> None:
        super().__init__(f'Context entry with ID {context_id} not found')
        self.context_id = context_id


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


# Size half of the line-scan offload predicate below: a text this large is offloaded
# regardless of how few lines it has, because even the cheapest per-character pass
# over it is no longer negligible. The line-count half carries the rest of the
# decision. Small entries stay inline to avoid a per-call thread hop. Unicode code
# points, not bytes. Mirrors grep_service._OFFLOAD_MIN_CHARS, where size genuinely
# IS the whole cost driver.
_OFFLOAD_MIN_CHARS = 1_000_000

# Line-oriented scans over stored text -- outline parsing (parse_outline /
# resolve_node_span, several regexes per line) and line splitting
# (split_lines_with_offsets, one slice and one list append per line) -- cost time
# proportional to LINE COUNT, not character count. A million characters on ONE line
# parses in about two milliseconds; the same million characters split into short
# heading lines takes about two seconds to parse and about a tenth of a second to
# split. A size-only threshold is blind to that three-order-of-magnitude spread,
# which is how a dense sub-threshold entry ends up processed inline and pins the
# event loop on every navigate/read call. Counting newlines is a single C-level scan
# (microseconds even for megabytes), so the extra signal is effectively free. At
# roughly five microseconds per heading line for the costlier of the two workloads,
# this bound keeps an inline pass in the single-digit-millisecond range.
_OFFLOAD_MIN_LINES = 1_000


def should_offload_line_scan(text: str) -> bool:
    """Whether a line-oriented scan over ``text`` must run in a worker thread.

    Covers every CPU-bound pass whose cost tracks line count: the outline parse
    (``parse_outline`` / ``resolve_node_span``) and the offset-preserving line
    split (``split_lines_with_offsets``).

    Args:
        text: The entry text about to be scanned.

    Returns:
        True when the text is large enough OR line-dense enough that scanning it
        inline would block the event loop noticeably.
    """
    return len(text) > _OFFLOAD_MIN_CHARS or text.count('\n') >= _OFFLOAD_MIN_LINES


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
            # RecursiveCharacterTextSplitter.create_documents) is pure CPU over
            # unbounded entry text and is EXPENSIVE per character -- roughly a
            # microsecond each, so a plain 1MB document costs the better part of a
            # second. Gating on a large fixed character threshold left exactly that
            # case running inline on the event loop. The real cost switch is the
            # splitter's own fast path: at or below chunk_size split_text returns
            # immediately (one chunk, no recursion), and above it the recursive
            # split runs in full. Offload precisely when that recursion happens --
            # the thread hop is negligible next to the provider round trip this leg
            # is about to await anyway.
            if len(text) > chunking_service.chunk_size:
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

    # return_exceptions=True keeps the fan-out structured: a bare gather
    # would propagate the first encode failure while the sibling tasks kept
    # running detached past the request (run_generation's finally cancels
    # only the three top-level legs and cannot reach these children).
    # Awaiting every child first, then raising, matches the abort loop in
    # run_generation and every other fan-out on the store paths.
    results = await asyncio.gather(
        *[_encode_one(c) for c in chunk_embeddings],
        return_exceptions=True,
    )
    encoded: list[ChunkEmbedding] = []
    for result in results:
        if isinstance(result, BaseException):
            raise result
        encoded.append(result)
    return encoded


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
        # The summary is model-generated, not client-supplied: a stray NUL or unpaired
        # surrogate in the provider's output would store on SQLite yet abort the
        # PostgreSQL bind inside the (abort-mandatory) transaction, charging the breaker.
        # Repair rather than reject -- the client's own text is valid and must not be
        # refused for a provider quirk.
        sanitized = sanitize_pg_unstorable_text(result)
        logger.info('Summary generated: text_len=%d, summary_len=%d', len(text), len(sanitized))
        return sanitized
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

    TOTAL work is bounded, not just concurrency: at most
    INDEX_TREE_NODE_SUMMARY_MAX_NODES sections are summarized (the shallowest and
    longest first, so the outline degrades gracefully), they are processed in
    bounded chunks rather than one unbounded gather, and the whole pass runs under
    the INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S aggregate budget.

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
        # parse_outline is pure CPU whose cost tracks LINE count over unbounded entry
        # text; offload a large OR line-dense entry to a worker thread so it cannot pin
        # the event loop (see should_offload_line_scan), matching the read-path discipline.
        if should_offload_line_scan(text):
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
    # shorter sections are deliberately skipped (not a failure). Filtering here
    # rather than inside the worker means an entry with hundreds of thousands of
    # tiny headings never materializes a task per heading just to return None.
    # len(section) == char_end - char_start (offsets are code points).
    eligible = [node for node in nodes if (node.char_end - node.char_start) >= min_len]
    if not eligible:
        # Nothing qualified: return [] so a replace legitimately clears stale rows.
        return []

    # Bound TOTAL work, not just concurrency. The semaphores cap how many calls run
    # at once and asyncio.wait_for caps each one, but neither bounds HOW MANY happen:
    # a heading-dense entry would otherwise hold the single summary model for one
    # store_context request for as long as its section count demands. When more
    # sections qualify than the cap, prefer the shallowest (most structurally
    # significant) and, within a level, the longest sections, then restore the
    # original traversal order so the stored rows stay deterministic.
    max_nodes = settings.index_tree.max_nodes
    if len(eligible) > max_nodes:
        logger.info(
            'Index-tree node summaries: %d eligible section(s) exceeds the cap of %d; '
            'summarizing the shallowest and longest sections only.',
            len(eligible), max_nodes,
        )
        # Rank by POSITION, never by node identity: OutlineNode is a frozen
        # dataclass holding its children, so hashing one hashes the whole subtree
        # and a set membership test over a large outline would be quadratic.
        ranked = sorted(
            range(len(eligible)),
            key=lambda i: (eligible[i].level, -(eligible[i].char_end - eligible[i].char_start)),
        )
        eligible = [eligible[i] for i in sorted(ranked[:max_nodes])]

    attempted = len(eligible)

    async def _summarize_node(node: OutlineNode) -> IndexNodeRow | None:
        # Eligibility (section length >= min_content_length) is pre-filtered above.
        section = text[node.char_start:node.char_end]
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
        # Repair a model-emitted NUL/unpaired surrogate before it binds into
        # context_index_nodes.node_summary (same abort-mandatory PostgreSQL bind as
        # the flat summary); an all-NUL result sanitizes to empty and is skipped.
        summary = sanitize_pg_unstorable_text(result.strip())
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

    # Process in bounded chunks under an aggregate wall-clock deadline instead of one
    # mega-gather: the chunking keeps the number of live tasks proportional to the
    # fan-out cap rather than to the section count, and the deadline stops a
    # pathological entry from stretching a single store indefinitely (each per-node
    # wait_for bounds ONE call, and their sum is unbounded without this). Whatever was
    # produced before the deadline is kept -- this leg never aborts a store.
    # _summarize_node never raises; return_exceptions is a defensive backstop so a
    # surprise (e.g. cancellation of a child) cannot turn into a store-aborting raise.
    chunk_size = max(settings.index_tree.max_concurrent * 4, 16)
    deadline = time.monotonic() + settings.index_tree.total_timeout_s
    rows: list[IndexNodeRow] = []
    for start in range(0, attempted, chunk_size):
        if time.monotonic() >= deadline:
            logger.warning(
                'Index-tree node summaries: aggregate budget of %.0fs expired after %d of '
                '%d section(s); keeping the summaries produced so far.',
                settings.index_tree.total_timeout_s, start, attempted,
            )
            break
        chunk = eligible[start:start + chunk_size]
        results = await asyncio.gather(
            *[_summarize_node(node) for node in chunk], return_exceptions=True,
        )
        rows.extend(result for result in results if isinstance(result, IndexNodeRow))

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
    is UNCONDITIONAL (a None node result on a text change becomes an empty
    list regardless of any toggle or provider), so stale rows describing the
    old text can never survive the edit -- not even through a
    disable/edit/re-enable cycle -- and ``navigate_context`` can never
    mis-attach an old section summary to a new section sharing a reused
    heading slug. ``replace_nodes_for_context`` pre-checks table existence,
    making the clear a safe no-op when the node table is absent.

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
            # Inspect the GATHER RESULTS, not Task.exception(): a task whose
            # coroutine ended with CancelledError is marked cancelled, and
            # Task.exception() then RAISES CancelledError instead of returning
            # it -- so an exception()-based loop can never skip a cancelled leg
            # (the tolerance the CancelledError check intends). gather with
            # return_exceptions=True hands back the CancelledError as a value,
            # which is safe to type-check.
            leg_outcomes = await asyncio.gather(
                *(task for _, task in abort_legs), return_exceptions=True,
            )
            for (name, _task), outcome in zip(abort_legs, leg_outcomes, strict=True):
                if isinstance(outcome, BaseException) and not isinstance(outcome, asyncio.CancelledError):
                    errors.append(f'{name}: {type(outcome).__name__}: {outcome}')

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
# Delete-path embedding cleanup
# ---------------------------------------------------------------------------


def sqlite_embedding_cleanup_required() -> bool:
    """Whether a SQLite delete still needs the explicit per-entry embedding cleanup.

    The cleanup exists for ONE reason: the fp32 ``vec_context_embeddings`` vec0
    VIRTUAL table carries no foreign key and is reachable only through the
    ``embedding_chunks`` bridge, so once that bridge cascades away with the
    context row its vectors orphan permanently.

    With embedding compression enabled that table does not exist at all -- the
    compression migration drops it and the payload table
    ``vec_context_embeddings_compressed`` is an ordinary table with
    ``ON DELETE CASCADE`` on ``context_id``, exactly like ``embedding_metadata``.
    A single ``DELETE FROM context_entries`` then removes every embedding row
    atomically, and the per-entry loop degenerates into one redundant write round
    trip per deleted entry -- which on a thread-wide or criteria-wide delete is
    unbounded and holds the single SQLite writer while every other client stalls.

    Cascade only fires while ``PRAGMA foreign_keys`` is ON, so an operator who
    set ``SQLITE_FOREIGN_KEYS=false`` keeps the explicit cleanup.

    Returns:
        True when the explicit per-entry cleanup is still required on SQLite.
    """
    return not (settings.compression.enabled and settings.storage.sqlite_foreign_keys)


async def cleanup_embeddings_for_delete(
    repos: 'RepositoryContainer',
    txn: 'TransactionContext',
    context_ids: list[str],
) -> None:
    """Delete FK-less SQLite embedding rows for entries about to be removed.

    Runs on the caller's transaction connection so the cleanup and the row
    delete commit or roll back together: without that, a failure between them
    (a client disconnect cancelling the request, a lock-wait timeout, a dropped
    connection) leaves entries stripped of their vectors while their rows
    survive, silently absent from semantic and hybrid search with nothing to
    regenerate them.

    A no-op on PostgreSQL, where ``ON DELETE CASCADE`` removes the embedding
    rows inside the same statement, and a no-op on SQLite whenever cascade
    already covers them (see :func:`sqlite_embedding_cleanup_required`).

    The ids are cleaned in bounded MULTI-ROW statements
    (``delete_all_chunks_bulk``), not one write per entry: the thread-wide and
    criteria-wide delete paths take an id list no client parameter caps, and a
    per-entry loop would hold the single SQLite writer for one round trip per
    matched row while every other client's writes stall behind it.

    A cleanup failure is logged and skipped rather than aborting: deleting an
    entry must stay possible even when its embedding rows are unreadable (a
    missing vec0 module, a corrupted row). On SQLite an error inside a statement
    does not poison the open transaction, so the row delete proceeds normally.
    Because the statements are batched, such a failure leaves the batch's
    remaining vectors orphaned instead of only the offending entry's -- the same
    fail-open direction (never a blocked delete, never lost user data), traded
    for a write path that no longer scales with the number of matched rows.

    Args:
        repos: Repository container.
        txn: The open transaction that will also issue the row delete.
        context_ids: The exact ids the delete will remove.
    """
    if not context_ids or txn.backend_type != 'sqlite':
        return
    if not sqlite_embedding_cleanup_required():
        return
    # Gate on whether the embedding tables were ever PROVISIONED, NOT on the
    # runtime ENABLE_EMBEDDING_GENERATION toggle: a prior session may have
    # written embeddings a now-disabled toggle would skip cleaning.
    if not await repos.embeddings.embedding_tables_exist(txn=txn):
        return
    try:
        await repos.embeddings.delete_all_chunks_bulk(context_ids, txn=txn)
    except Exception as exc:
        logger.warning('Failed to delete embeddings for %d contexts: %s', len(context_ids), exc)


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

    Despite the historical name, this classifier covers four transient
    families, all safe to retry because the database write that follows is
    idempotent (store_context deduplicates; update_context is a keyed
    partial update) and ALL embedding/summary/compression generation has
    already completed OUTSIDE the transaction (generation-first invariant) --
    so a retry re-runs only the rolled-back DB write and never regenerates or
    skips generation:

    1. Connection-level failures (the connection was lost, not a logical/data
       error): asyncpg.InterfaceError, asyncpg.ConnectionDoesNotExistError,
       ConnectionResetError, OSError -- EXCLUDING TimeoutError. TimeoutError is
       an OSError subclass on Python 3.12, but the pool-acquire TimeoutError that
       begin_transaction re-raises uncharged signals a SATURATED connection pool,
       not a lost connection: retrying it re-runs the full POSTGRESQL_POOL_TIMEOUT_S
       acquire wait each time, multiplying one saturation stall into several. So a
       saturated pool must fail fast at the tool layer after one bounded wait,
       matching execute_write's fail-fast handling of the identical signal.
    2. Statement / lock-wait timeouts: asyncpg.exceptions.QueryCanceledError
       (SQLSTATE 57014). PostgreSQL cancels the statement when it exceeds the
       connection's statement_timeout (set to ~0.9 * POSTGRESQL_COMMAND_TIMEOUT_S
       in app.backends.postgresql_backend._setup_pool_connection). Retrying with the SAME ceiling
       only helps a TRANSIENT lock-WAIT (the write was blocked behind a
       concurrent writer and the contention has since cleared); it does NOT
       help a write that is fundamentally slower than the ceiling -- for that
       case (notably fp32 mode, ENABLE_EMBEDDING_COMPRESSION=false, where each
       per-chunk INSERT performs in-transaction HNSW maintenance) the operator
       must also raise POSTGRESQL_COMMAND_TIMEOUT_S or keep compression ON. See
       docs/database-backends.md.
    3. Transaction-rollback failures: asyncpg.exceptions.TransactionRollbackError
       (SQLSTATE class 40 -- deadlock_detected 40P01, serialization_failure 40001,
       and siblings). PostgreSQL aborts one transaction to break a deadlock or a
       serialization cycle; by definition the loser is expected to retry, and the
       retry succeeds once the competing transaction has committed. Without this
       class a deadlock (e.g. two atomic update batches that lock the same rows in
       opposite order) is neither a ControlFlowError nor a connection error, so it
       would charge the circuit breaker instead of retrying -- turning a routine,
       self-clearing lock cycle into an outage.

    4. SQLite write contention: sqlite3.OperationalError in the SQLITE_BUSY /
       SQLITE_LOCKED family ('database is locked'), classified by the shared
       is_sqlite_locked_error predicate. The backend's write-queue path
       (execute_write) retries this family internally, but begin_transaction --
       the path every store/update transaction site uses -- bypasses the write
       queue and performs NO backend-level retry, so a cross-process lock
       collision (e.g. two MCP server processes sharing one SQLite database
       file) must be retried by these tool-layer loops, mirroring the
       PostgreSQL class-40 treatment above. The backend re-raises the family
       without charging the circuit breaker for the same reason.

    QueryCanceledError is PostgreSQL-only; the isinstance check is harmless on
    SQLite, which never raises it.

    Args:
        exc: The exception to classify

    Returns:
        True if the exception is a transient DB error safe for retry
    """
    # TimeoutError is an OSError subclass; exclude it so a saturated-pool acquire
    # timeout is NOT retried (see the family list above). asyncpg.InterfaceError,
    # ConnectionDoesNotExistError, and ConnectionResetError are not TimeoutError
    # subclasses, so they still match.
    if isinstance(exc, TimeoutError):
        return False
    return is_sqlite_locked_error(exc) or isinstance(exc, (
        asyncpg.InterfaceError,
        asyncpg.ConnectionDoesNotExistError,
        asyncpg.exceptions.QueryCanceledError,
        asyncpg.exceptions.TransactionRollbackError,
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
    - Enforces the per-entry image count limit (MAX_IMAGES_PER_ENTRY)
    - Checks required 'data' field presence
    - Rejects a non-string 'data' or 'mime_type' value (the batch path is untyped)
    - Rejects empty/whitespace data (prevents silent 0-byte storage)
    - Defaults mime_type to 'image/png' when the key is absent
    - Normalizes each base64 payload to canonical standard-alphabet form
      (data-URI prefix stripped, ASCII whitespace removed, URL-safe alphabet
      translated, '=' padding restored) and REWRITES img['data'] in place to
      that canonical string, so every later re-decode (ImageRepository write
      paths) operates on the exact payload validated here
    - Decodes STRICTLY (base64.b64decode with validate=True), so genuinely
      non-base64 input fails loudly instead of being silently mangled
    - Enforces per-image size limit (MAX_IMAGE_SIZE_MB)
    - Enforces total size limit (MAX_TOTAL_SIZE_MB)
    - Rejects a per-image 'metadata' value that is neither absent/null nor a
      JSON-encoded string (the canonical per-image metadata wire shape). This
      subsumes the non-finite-float check entry metadata needs: only a bare
      float can serialize to the invalid-JSON NaN/Infinity tokens PostgreSQL's
      jsonb parser rejects, and a JSON-encoded string carries no bare float
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

    # Enforce the documented per-entry count limit here at the single shared
    # chokepoint so it covers store_context, update_context, AND both batch
    # tools (whose per-entry image lists never pass through the Pydantic
    # models). The tool-boundary Field declarations in app/tools/context.py
    # advertise the same bound as maxItems in the MCP wire schema.
    if len(images) > MAX_IMAGES_PER_ENTRY:
        msg = f'Too many images: {len(images)} provided, maximum is {MAX_IMAGES_PER_ENTRY} per entry'
        if error_mode == 'raise':
            raise ToolError(msg)
        errors.append(msg)
        return images, 'text', errors

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
            # A mime_type STRING carrying an embedded NUL (U+0000) or an unpaired
            # UTF-16 surrogate binds into the image_attachments.mime_type TEXT NOT
            # NULL column inside the transaction -- SQLite stores it while
            # PostgreSQL raises a DataError AFTER a full generation pass, charging
            # the circuit breaker. The Pydantic list[dict[str, str]] contract on the
            # single-entry path enforces str TYPE but not this byte content, so the
            # guard must live here (shared by both paths), mirroring the per-image
            # metadata check below.
            mime_unstorable = unstorable_string_error(mime_val)
            if mime_unstorable is not None:
                msg = f'Image {idx} mime_type: {mime_unstorable}'
                if error_mode == 'raise':
                    raise ToolError(msg)
                errors.append(msg)
                return images, 'text', errors

        # Per-image 'metadata' crosses the boundary as a JSON-ENCODED STRING: the
        # typed single-entry tools declare images as list[dict[str, str]], the write
        # path json.dumps that already-stringified value, and the read path json.loads
        # it back to the same string. The untyped batch path (list[dict[str, Any]])
        # bypasses that contract, so a dict/number/list slips through and is stored in
        # a shape the single-entry tool would have refused -- and which then fails the
        # strict get_context_by_ids output schema, making the entry permanently
        # unreadable. Enforce the same shape here, at the chokepoint both paths share.
        # A JSON null is treated as "no metadata" (it stores as SQL NULL either way).
        metadata_value = cast(object, img.get('metadata'))
        if metadata_value is not None and not isinstance(metadata_value, str):
            msg = (
                f'Image {idx} metadata must be a JSON-encoded string '
                f'(got {type(metadata_value).__name__})'
            )
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # A per-image 'metadata' key or value carrying an embedded NUL (U+0000) or an
        # unpaired UTF-16 surrogate stores on SQLite but is rejected by PostgreSQL's jsonb
        # image_metadata column -- the same cross-backend divergence and breaker-charging
        # failure unstorable_string_error guards for entry metadata, applied here for the
        # untyped batch path's per-image metadata.
        unstorable_error = unstorable_string_error(cast(object, img.get('metadata')))
        if unstorable_error is not None:
            msg = f'Image {idx} metadata: {unstorable_error}'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # Normalize the payload to canonical standard-alphabet base64 BEFORE
        # decoding: strip one RFC 2397 data-URI prefix, remove ASCII whitespace,
        # translate the URL-safe alphabet, restore '=' padding. A lenient
        # b64decode previously accepted these shapes and silently corrupted the
        # bytes (a data-URI prefix whose base64-alphabet length is a multiple of
        # 4 decodes as garbage prepended to the image; '-'/'_' were discarded,
        # shifting every following byte). The STRICT decode (validate=True) then
        # either yields exactly the intended bytes or fails loudly per image.
        normalized_data = normalize_base64_image_data(img_data_str)
        try:
            image_binary = base64.b64decode(normalized_data, validate=True)
        except Exception as e:
            if error_mode == 'raise':
                raise ToolError(
                    f'Image {idx} has invalid base64 encoding: Invalid base64 data ({format_exception_message(e)})',
                ) from None
            errors.append(f'Image {idx} has invalid base64 encoding')
            return images, 'text', errors

        # Rewrite the payload in place to the canonical string so the write-path
        # re-decode in ImageRepository (strict as well) operates on the exact
        # payload validated here. The batch tools rely on this in-place mutation:
        # they discard the returned list and later pass the same dict objects to
        # the transaction helpers.
        img['data'] = normalized_data

        # A payload that normalizes to the empty string (e.g. a bare data-URI
        # prefix with nothing after the comma) decodes to zero bytes and is not
        # a real image; reject it rather than storing a 0-byte attachment.
        # Non-alphabet garbage no longer reaches this guard -- the strict decode
        # above rejects it loudly.
        if not image_binary:
            msg = f'Image {idx} "data" decodes to zero bytes (not valid base64 image content)'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
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
    images_provided: bool | None = None,
    chunk_embeddings: list[ChunkEmbedding] | None,
    embedding_model: str,
    embedding_generation_enabled: bool = False,
    index_nodes: list[IndexNodeRow] | None = None,
    nodes_pending: bool = False,
    summary_pending: bool = False,
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
        tags: Tag list or None. None PRESERVES existing tags on a dedup UPDATE;
            a provided list (including []) REPLACES them, matching the
            documented replacement contract and update_context semantics.
        validated_images: Validated image list (may be empty).
        images_provided: Whether the CALLER passed an images value. None (the
            default) falls back to ``bool(validated_images)`` for callers that
            predate the flag; True with an empty validated_images clears
            existing images on a dedup UPDATE instead of preserving them.
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
        summary_pending: True when the caller's pre-check REUSED the likely
            duplicate's stored summary instead of generating one. When True and
            this store INSERTs a new entry (was_updated False), the reused
            summary was read from a candidate that has since diverged and may
            describe different text, so the transaction aborts via
            EmbeddingsReconcileRequiredError for the caller to regenerate the
            summary outside the transaction and retry. Defaults to False.

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
    # Resolve whether the CALLER passed an images value before any use: an
    # explicitly provided empty list must behave as a REPLACEMENT (clear) on a
    # dedup UPDATE, not as absent. Falls back to list truthiness for callers
    # that predate the flag.
    if images_provided is None:
        images_provided = bool(validated_images)

    # Store context entry with deduplication
    context_id, was_updated = await repos.context.store_with_deduplication(
        thread_id=thread_id,
        source=source,
        content_type=content_type,
        text_content=text_content,
        metadata=metadata_str,
        summary=summary,
        # Preserve the existing content_type on a dedup UPDATE only when no images
        # value was provided this call (images are preserved, not replaced).
        # Overwriting it then would flip a multimodal entry to 'text' while its
        # image rows remain, making them unretrievable. When images ARE provided
        # -- including an explicit empty list, which clears them below --
        # content_type is overwritten to match the request.
        preserve_content_type_on_dedup=not images_provided,
        txn=txn,
    )

    if not context_id:
        raise ToolError('Failed to store context')

    # Generation-first reconciliation: the caller's read-only pre-check skips
    # embedding AND node-summary generation (and may REUSE the candidate's
    # summary) when a likely duplicate already has them, expecting this store to
    # deduplicate into an UPDATE. If a concurrent same-thread write committed in
    # the meantime, store_with_deduplication can instead INSERT a brand-new
    # entry (was_updated False). Committing now would persist a row missing its
    # embeddings (when generation is enabled) or its per-node summaries (when
    # the node layer is active) -- or carrying a REUSED summary read from the
    # since-diverged candidate, which may describe DIFFERENT text (the summary
    # read happens after the hash check, so a commit between them poisons it).
    # Abort so the caller regenerates the skipped/reused legs OUTSIDE the
    # transaction and retries. The three reconcile triggers are decoupled so
    # each leg is repaired regardless of which others are active.
    needs_embedding_reconcile = embedding_generation_enabled and chunk_embeddings is None
    needs_node_reconcile = nodes_pending and index_nodes is None
    needs_summary_reconcile = summary_pending
    if not was_updated and (needs_embedding_reconcile or needs_node_reconcile or needs_summary_reconcile):
        raise EmbeddingsReconcileRequiredError(text_content)

    # Heartbeat: keep connection alive between sequential operations
    await transaction_heartbeat(txn)

    # Store or replace tags depending on deduplication outcome. The documented
    # contract distinguishes PROVIDED from None: an explicitly provided empty
    # list REPLACES (clears) existing tags on a dedup UPDATE, exactly like
    # update_context's `if tags is not None` semantics; only None preserves.
    # On a fresh INSERT an empty list stores nothing, so the write is skipped.
    if tags is not None:
        if was_updated:
            await repos.tags.replace_tags_for_context(context_id, tags, txn=txn)
        elif tags:
            await repos.tags.store_tags(context_id, tags, txn=txn)

    # Store or replace images depending on deduplication outcome, with the same
    # provided-vs-None distinction. validated_images is always a list (the
    # validator normalizes None to []), so images_provided (resolved above)
    # carries whether the CALLER passed an images value; when it did, an empty
    # list clears existing images on a dedup UPDATE instead of preserving them.
    if images_provided:
        if was_updated:
            await repos.images.replace_images_for_context(
                context_id, validated_images, txn=txn,
            )
        elif validated_images:
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
    5. Maintain the auto-managed fields: recompute content_type from actual image
       presence, and guarantee updated_at advanced for this update (a tags-only
       change writes no context_entries row of its own)
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
        index_nodes: Replacement index_tree node rows; None leaves the stored
            rows untouched, an empty list clears them.
        expected_version: Optimistic-concurrency token captured before
            generation; None skips the compare-and-set.

    Returns:
        Tuple of (updated_fields, summary_cleared):
        - updated_fields: List of field names that were updated
        - summary_cleared: True if summary was cleared (for response message)

    Raises:
        EntryNotFoundError: If the target entry does not exist -- update_context_entry
            or patch_metadata reports no matching row, or the tags-only / images-only
            path finds the parent missing. A ControlFlowError, so the failed write is
            not charged to the circuit breaker; the caller catches it outside the
            transaction and converts it to a not-found ToolError.
    """
    updated_fields: list[str] = []

    # ``updated_at`` is an auto-managed PUBLIC field: get_context_by_ids and every
    # search tool return it, and it is the only mutation timestamp the API exposes,
    # so clients key incremental sync and cache invalidation on it. It is stamped
    # ONLY by a write to context_entries itself (update_context_entry,
    # patch_metadata, update_content_type) -- a branch that touches just a child
    # table leaves it stale. Track whether such a write happened so the auto-managed
    # block below can stamp it exactly once for every update variant.
    entry_row_stamped = False

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
            # update_context_entry returns success=False only when no row matched
            # its WHERE id=? (a version mismatch is raised, not returned), i.e. the
            # entry was deleted concurrently or the id is stale.
            raise EntryNotFoundError(context_id)

        updated_fields.extend(fields)
        entry_row_stamped = True

    # Apply metadata patch (partial update) if provided
    if metadata_patch is not None:
        success, fields = await repos.context.patch_metadata(
            context_id=context_id,
            patch=metadata_patch,
            txn=txn,
        )

        if not success:
            # patch_metadata returns success=False only when the row is gone.
            raise EntryNotFoundError(context_id)

        updated_fields.extend(fields)
        entry_row_stamped = True

    # A tags-only or images-only update issues no write that first confirms the
    # parent exists: the text/metadata and metadata_patch branches each SELECT the
    # row (and raise EntryNotFoundError above when it is gone), but neither ran
    # here. Without that guard, replacing tags/images against a deleted parent
    # violates the child foreign key -- a non-ControlFlowError that charges the
    # circuit breaker -- or, with FK enforcement off, orphans the replacement
    # rows. Confirm the parent explicitly and raise the same breaker-exempt signal.
    if (
        text is None
        and metadata is None
        and metadata_patch is None
        and (tags is not None or images is not None)
        and not await repos.context.entry_exists(context_id, txn=txn)
    ):
        raise EntryNotFoundError(context_id)

    # Heartbeat between operation groups
    await transaction_heartbeat(txn)

    # Replace tags and/or images if provided. Both write child rows keyed on the
    # parent context_entries id. The entry_exists guard above (locking the parent
    # with FOR KEY SHARE on PostgreSQL) already confirms and holds the parent for
    # the tags-only / images-only path, so a concurrent delete cannot race these
    # writes. This catch is defense in depth for any residual foreign-key
    # violation, mapping it to the breaker-exempt EntryNotFoundError so a missing
    # parent surfaces as a clean not-found outcome rather than a raw asyncpg error
    # that charges the circuit breaker.
    try:
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
            # update_content_type writes context_entries, so it carried the stamp.
            entry_row_stamped = True
    except asyncpg.exceptions.ForeignKeyViolationError as exc:
        raise EntryNotFoundError(context_id) from exc

    # Enforce the two auto-managed fields centrally, for EVERY update variant that
    # changed something, instead of relying on whichever data write a branch happens
    # to issue:
    #   * content_type is recomputed from the entry's ACTUAL image rows (the explicit
    #     images branch above already wrote the matching value, so this only runs when
    #     the caller left images untouched);
    #   * updated_at is advanced whenever no branch has written context_entries yet --
    #     the tags-only variant writes only the child `tags` table, so without this it
    #     would report success while leaving the entry's public mutation timestamp at
    #     its previous value, and a client syncing or invalidating caches on
    #     updated_at would never observe the change.
    # The recomputed content_type is derived from the image rows inside this same
    # transaction, so it is correct by construction rather than a read-modify-write
    # of a field a concurrent writer could have moved. When it is already correct,
    # the timestamp is stamped EXPLICITLY (touch_updated_at) instead of rewriting an
    # unrelated column back to its own value just to carry the stamp along.
    if images is None and updated_fields:
        image_count = await repos.images.count_images_for_context(context_id, txn=txn)
        current_content_type = 'multimodal' if image_count > 0 else 'text'
        stored_content_type = await repos.context.get_content_type(context_id, txn=txn)
        if stored_content_type != current_content_type:
            await repos.context.update_content_type(
                context_id, current_content_type, txn=txn,
            )
            entry_row_stamped = True
            updated_fields.append('content_type')
        elif not entry_row_stamped:
            await repos.context.touch_updated_at(context_id, txn=txn)
            entry_row_stamped = True

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
