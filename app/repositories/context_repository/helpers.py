"""Pure helpers shared by the context repository mixins and the tool layer.

Bounded chunking of id lists for per-statement ``IN (...)`` binds, the
human-readable description of batch-delete criteria, and content hashing for
deduplication.
"""

import hashlib

# Maximum number of ids bound into a single IN (...) statement. An id list can be
# arbitrarily long (e.g. every entry in a large thread, or a client-supplied bulk list), so
# id-list SELECTs and DELETEs are issued in bounded chunks to stay under each backend's
# per-statement bound-parameter limit -- SQLite's SQLITE_MAX_VARIABLE_NUMBER (historically
# as low as 999) and PostgreSQL's 65535 parameters. 900 clears the lowest historical SQLite
# ceiling with margin. Shared by get_by_ids, delete_by_ids, and the batch-criteria
# statements (via _criteria_chunk_pairs).
_ID_CHUNK_SIZE = 900


def chunk_ids(ids: list[str]) -> list[list[str]]:
    """Split an id list into bounded chunks for per-statement IN (...) binds.

    Args:
        ids: The id list to split.

    Returns:
        Consecutive slices of ``ids``, each at most ``_ID_CHUNK_SIZE`` long.
    """
    return [ids[start : start + _ID_CHUNK_SIZE] for start in range(0, len(ids), _ID_CHUNK_SIZE)]


def describe_batch_delete_criteria(
    context_ids: list[str] | None = None,
    thread_ids: list[str] | None = None,
    source: str | None = None,
    older_than_days: int | None = None,
) -> list[str]:
    """Describe batch-delete criteria as human-readable ``criteria_used`` strings.

    Single source of truth for the strings returned in batch-delete tool
    responses. ``delete_contexts_batch`` builds them here for its criteria-based
    delete, and the SQLite branch of the ``delete_context_batch`` tool builds
    them here too, because it deletes by the pre-queried snapshot ids (see
    ``get_ids_matching_batch_criteria``) and never reaches the criteria-building
    closures.

    Args:
        context_ids: Specific context entry IDs targeted by the delete
        thread_ids: Threads whose entries are targeted by the delete
        source: Source filter ('user' or 'agent')
        older_than_days: Age filter in days

    Returns:
        List of criteria descriptions in the order the filters are applied.
    """
    criteria_used: list[str] = []
    if context_ids:
        criteria_used.append(f'context_ids: {len(context_ids)} IDs')
    if thread_ids:
        criteria_used.append(f'thread_ids: {len(thread_ids)} threads')
    if source:
        criteria_used.append(f'source: {source}')
    if older_than_days is not None:
        criteria_used.append(f'older_than_days: {older_than_days}')
    return criteria_used


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content for deduplication.

    Used to avoid transferring full text_content over the network when
    checking for duplicates. The hash is stored alongside text_content
    and compared instead of the full text.

    Args:
        text: The text content to hash.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()
