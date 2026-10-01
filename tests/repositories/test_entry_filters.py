"""Tests for the shared ``filters_applied`` tally in app/repositories/entry_filters.py."""


class TestCountAppliedFilters:
    """The shared filters_applied tally used by every search repository.

    search_context, fts_search_context, semantic_search_context and the compressed semantic
    path all publish this number as stats.filters_applied, so it must be computed in exactly
    one place: re-deriving it per repository is what let the browse tool report a different
    count than its siblings for identical arguments.
    """

    def test_counts_each_dimension_once(self) -> None:
        """Every emitted condition contributes exactly one, tags counting as one subquery."""
        from app.repositories.entry_filters import count_applied_filters

        assert count_applied_filters() == 0
        assert count_applied_filters(thread_id='t') == 1
        assert count_applied_filters(source='agent') == 1
        assert count_applied_filters(content_type='text') == 1
        assert count_applied_filters(tags=['a', 'b', 'c']) == 1
        assert count_applied_filters(start_date='2020-01-01') == 1
        assert count_applied_filters(end_date='2020-01-01') == 1
        assert count_applied_filters(metadata_filter_count=3) == 3

    def test_ignores_empty_values(self) -> None:
        """Empty strings and empty lists are not applied conditions."""
        from app.repositories.entry_filters import count_applied_filters

        assert count_applied_filters(thread_id='', source='', content_type='', tags=[]) == 0

    def test_sums_all_dimensions(self) -> None:
        """A fully specified request counts every condition plus the metadata filters."""
        from app.repositories.entry_filters import count_applied_filters

        assert count_applied_filters(
            thread_id='t',
            source='agent',
            content_type='text',
            tags=['x'],
            start_date='2020-01-01',
            end_date='2999-01-01',
            metadata_filter_count=2,
        ) == 8
