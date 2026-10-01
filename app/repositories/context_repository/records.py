"""Record types and the column list shared by the context repository and its callers.

Holds the optimistic-concurrency ``VersionConflictError``, the explicit
``CONTEXT_ENTRY_COLUMNS`` select list, and the ``EntryProbe`` and
``DuplicateCandidate`` result tuples.
"""

from typing import NamedTuple

from app.errors import ControlFlowError


class VersionConflictError(ControlFlowError):
    """Raised by ``update_context_entry`` when its optimistic-concurrency guard
    finds the row's ``version`` changed since the caller read it.

    A concurrent writer committed a newer update first, so applying this one
    would silently overwrite it (and leave index_tree node rows describing stale
    text). The caller re-reads the current version and retries the transaction so
    the update applies against the latest row instead of clobbering it. It is
    deliberately NOT a ``ToolError`` (so an ``except ToolError`` fast-path does
    not swallow it) and NOT a connection error (so it is not treated as a
    transient retry).
    """

    def __init__(self, context_id: str) -> None:
        super().__init__(f'Version conflict updating context {context_id}')
        self.context_id = context_id


# Explicit column list to avoid exposing internal database columns (e.g., text_search_vector)
# This constant is used in all SELECT queries that return context entries to ensure
# only the expected columns are returned, preventing internal PostgreSQL columns from
# leaking into API responses.
CONTEXT_ENTRY_COLUMNS = 'id, thread_id, source, content_type, text_content, metadata, summary, created_at, updated_at'


class EntryProbe(NamedTuple):
    """Existence probe result for one context entry (see ``check_entry_exists``).

    ``source``, ``version``, and ``owner_id`` are None when the entry does not
    exist. ``version`` is the optimistic-concurrency token the update paths
    capture BEFORE generation as their compare-and-set guard; ``owner_id`` backs
    the owner-only visibility-change authorization (it is immutable, so a
    pre-generation read of it cannot go stale).
    """

    exists: bool
    source: str | None
    version: int | None
    owner_id: str | None


class DuplicateCandidate(NamedTuple):
    """Statement-level snapshot of a likely-duplicate entry found by the pre-check.

    ``context_id`` and ``summary`` come from the SAME row read that matched the
    content hash, so the summary is guaranteed to describe the text the hash
    was computed from. Reading the summary in a separate later statement could
    observe a row version whose summary describes DIFFERENT text (a concurrent
    update landing between the two reads) -- and the dedup UPDATE's
    content-hash predicate cannot distinguish a revision-consistent row from a
    restored one, so the mismatched summary would persist via COALESCE.
    """

    context_id: str
    summary: str | None
