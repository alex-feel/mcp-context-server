"""Filter accounting shared by the context, embedding and FTS repositories.

Every search path reports the ``filters_applied`` statistic through
``count_applied_filters``, so identical filter arguments always report the
identical count whichever repository served the search.
"""


def count_applied_filters(
    *,
    thread_id: str | None = None,
    source: str | None = None,
    content_type: str | None = None,
    tags: list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    metadata_filter_count: int = 0,
) -> int:
    """Count the filters a search actually applied, for the ``filters_applied`` statistic.

    Single source of truth for the tally, shared by the browse path
    (:meth:`ContextRepository.search_contexts` via
    :meth:`ContextRepository._build_context_filter_clause`), the FTS repository and both
    semantic paths (fp32 and compressed), so the identical arguments always report the
    identical count. Re-implementing the sum per repository is what let ``search_context``
    report only the metadata filters while its sibling tools reported all of them for the
    same request, which misleads an operator using ``explain_query`` to verify that a
    filter took effect.

    Args:
        thread_id: Thread filter, counted when set.
        source: Source filter, counted when set.
        content_type: Content-type filter, counted when set.
        tags: Tag filter, counted as ONE filter when non-empty (a single OR-ed subquery).
        start_date: Lower date bound, counted when set.
        end_date: Upper date bound, counted when set.
        metadata_filter_count: Number of metadata conditions the query builder emitted
            (simple equality filters plus advanced operator filters).

    Returns:
        Total number of applied filter conditions.
    """
    return sum([
        1 if thread_id else 0,
        1 if source else 0,
        1 if content_type else 0,
        1 if tags else 0,
        1 if start_date else 0,
        1 if end_date else 0,
    ]) + metadata_filter_count
