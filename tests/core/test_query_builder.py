"""MetadataQueryBuilder SQL generation for simple filters, each operator, and metadata key validation."""

import math

import pytest

from app.metadata_sql import is_safe_key
from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


class TestMetadataQueryBuilder:
    """Test the MetadataQueryBuilder class."""

    def test_simple_filter(self) -> None:
        """Test simple key=value filtering."""
        builder = MetadataQueryBuilder()
        builder.add_simple_filter('status', 'active')

        where_clause, params = builder.build_where_clause()
        # A string value matches a JSON-string-typed stored value ONLY (text guard).
        assert "json_type(metadata, '$.status') = 'text'" in where_clause
        assert "CAST(json_extract(metadata, '$.status') AS TEXT) = ?" in where_clause
        assert params == ['active']

    def test_multiple_simple_filters(self) -> None:
        """Test multiple simple filters combined with AND."""
        builder = MetadataQueryBuilder()
        builder.add_simple_filter('status', 'active')
        builder.add_simple_filter('priority', 5)

        where_clause, params = builder.build_where_clause()
        assert 'json_extract' in where_clause
        assert len(params) == 2
        assert 'active' in params
        assert 5 in params

    def test_operator_eq(self) -> None:
        """Test equality operator."""
        builder = MetadataQueryBuilder()
        # Test with case_sensitive=True for exact matching
        filter_spec = MetadataFilter(key='status', operator=MetadataOperator.EQ, value='active', case_sensitive=True)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # String EQ matches a JSON-string-typed stored value ONLY (text guard).
        assert "json_type(metadata, '$.status') = 'text'" in where_clause
        assert "CAST(json_extract(metadata, '$.status') AS TEXT) = ?" in where_clause
        assert params == ['active']

        # Test default case-insensitive behavior
        builder2 = MetadataQueryBuilder()
        filter_spec2 = MetadataFilter(key='status', operator=MetadataOperator.EQ, value='active')
        builder2.add_advanced_filter(filter_spec2)

        where_clause2, params2 = builder2.build_where_clause()
        assert 'LOWER' in where_clause2
        assert params2 == ['active']

    def test_operator_ne(self) -> None:
        """Test not-equal operator."""
        builder = MetadataQueryBuilder()
        # Use case_sensitive=True to avoid LOWER function
        filter_spec = MetadataFilter(key='status', operator=MetadataOperator.NE, value='inactive', case_sensitive=True)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '!=' in where_clause
        assert params == ['inactive']

    def test_operator_gt(self) -> None:
        """Test greater-than operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'CAST' in where_clause
        assert '>' in where_clause
        assert params == [5]

    def test_operator_in(self) -> None:
        """Test IN operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.IN,
            value=['active', 'pending', 'review'],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 3
        assert 'active' in params

    def test_operator_in_with_integers(self) -> None:
        """Test IN operator with integer array values.

        Numeric IN members are matched NUMERICALLY (type-aware): a number guard restricts
        the match to JSON-number-typed stored values and the raw ints are bound (no text
        comparison), so out-of-int64 / high-precision numbers cannot diverge across backends.
        """
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='priority',
            operator=MetadataOperator.IN,
            value=[5, 9],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 2
        # Numeric members bind as raw ints (not stringified for TEXT comparison).
        assert all(isinstance(p, int) for p in params)
        assert 5 in params
        assert 9 in params

    def test_operator_in_with_floats(self) -> None:
        """Test IN operator with float array values."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='score',
            operator=MetadataOperator.IN,
            value=[math.pi, math.e, 1.41],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 3
        # Numeric members bind as raw floats (matched numerically, not as text).
        assert all(isinstance(p, float) for p in params)

    def test_operator_in_with_mixed_types(self) -> None:
        """Test IN operator with mixed string and integer array values."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='value',
            operator=MetadataOperator.IN,
            value=['active', 5, 'pending', 10],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert len(params) == 4
        # Type-aware: string members bind as text (string-typed match), numeric members
        # bind as raw numbers (number-typed match).
        assert 'active' in params
        assert 'pending' in params
        assert 5 in params
        assert 10 in params

    def test_numeric_path_segment_rejected(self) -> None:
        """A numeric path segment AFTER the first (e.g. 'items.0', 'a.-1') is rejected on BOTH
        backends: it array-indexes on PostgreSQL but resolves to a literal object key on SQLite,
        a silent divergence. A single numeric key ('0') and non-numeric nested paths stay valid."""
        for bad in ('items.0', 'a.-1', 'a.0.b', 'items.01'):
            assert is_safe_key(bad) is False
            with pytest.raises(ValueError, match='Numeric path segments'):
                MetadataFilter(key=bad, operator=MetadataOperator.EQ, value='x')
        # Allowed: a single numeric key (consistent object-key on both backends) and
        # non-numeric nested paths (a numeric-suffixed segment like 'b0' is not all-digits).
        for good in ('0', 'a.b', 'items.foo', 'user.preferences.theme', 'a.b0'):
            assert is_safe_key(good) is True
            MetadataFilter(key=good, operator=MetadataOperator.EQ, value='x')  # must not raise

    def test_empty_path_segment_rejected(self) -> None:
        """A dotted key with an empty segment -- leading '.x', trailing 'x.', consecutive
        'a..b', or the degenerate '.'/'..' -- is rejected on BOTH validators. An empty
        segment builds a malformed PostgreSQL array literal like '{a,,b}' (a raw parser
        error) while SQLite silently mismatches, a cross-backend divergence; this mirrors
        the numeric-segment rejection. Valid keys (single segment, dotted, a numeric key,
        hyphenated, underscored) still pass unchanged -- no over-restriction. ('a.0' is
        intentionally absent: it is rejected by the separate numeric-path-segment guard,
        not the empty-segment guard under test here.)"""
        for bad in ('.x', 'x.', 'a..b', '.', '..'):
            assert is_safe_key(bad) is False
            with pytest.raises(ValueError, match='Empty path segments'):
                MetadataFilter(key=bad, operator=MetadataOperator.EQ, value=1)
        # Allowed: keys whose every dot-separated segment is non-empty stay valid.
        for good in ('a', 'a.b', '0', 'metadata_version', 'a-b', 'user.preferences.theme'):
            assert is_safe_key(good) is True
            MetadataFilter(key=good, operator=MetadataOperator.EQ, value=1)  # must not raise

    def test_trailing_newline_key_rejected(self) -> None:
        """A key ending in a newline is rejected on BOTH validators. Python's `$` also matches
        immediately before a single trailing '\\n', so a `re.match(r'^...$')` gate would have
        passed 'status\\n'; the un-stripped simple-filter path then diverged (SQLite
        json_extract('$.a.status\\n') misses while PostgreSQL's #>> array-literal parse trims
        the newline and matches). fullmatch closes the parity gap. Clean keys stay valid."""
        for bad in ('status\n', 'a.status\n', 'status\n\n', 'a\nb'):
            assert is_safe_key(bad) is False
            with pytest.raises(ValueError, match='Invalid metadata key'):
                MetadataFilter(key=bad, operator=MetadataOperator.EQ, value='x')
        for good in ('status', 'a.status', 'user.preferences.theme'):
            assert is_safe_key(good) is True
            MetadataFilter(key=good, operator=MetadataOperator.EQ, value='x')  # must not raise

    def test_string_operator_matches_string_typed_only(self) -> None:
        """String operators match JSON-string-typed stored values ONLY (text guard), so a stored
        number is never compared as text (which diverges across backends for out-of-int64 /
        high-precision numbers). Numeric IN members still match stored numbers numerically."""
        for backend, guard in (
            ('sqlite', "json_type(metadata, '$.k') = 'text'"),
            ('postgresql', "jsonb_typeof(metadata->'k') = 'string'"),
        ):
            b = MetadataQueryBuilder(backend_type=backend)
            b.add_advanced_filter(
                MetadataFilter(key='k', operator=MetadataOperator.EQ, value='5', case_sensitive=True),
            )
            clause, _ = b.build_where_clause()
            assert guard in clause

        # A numeric IN member matches a JSON-number-typed stored value numerically (number
        # guard + raw numeric bind), NOT as text.
        bi = MetadataQueryBuilder(backend_type='sqlite')
        bi.add_advanced_filter(MetadataFilter(key='n', operator=MetadataOperator.IN, value=[5]))
        clause_i, params_i = bi.build_where_clause()
        assert "json_type(metadata, '$.n') IN ('integer', 'real')" in clause_i
        assert params_i == [5]

    def test_operator_not_in(self) -> None:
        """Test NOT IN operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.NOT_IN,
            value=['deleted', 'archived'],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NOT NULL AND NOT (' in where_clause  # NOT_IN = presence guard + negated membership
        assert 'IN (' in where_clause
        assert len(params) == 2

    def test_operator_not_in_with_integers(self) -> None:
        """Test NOT IN operator with integer array values.

        Numeric members are matched NUMERICALLY (number guard + raw numeric binds), so the
        backends agree without any text comparison.
        """
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='priority',
            operator=MetadataOperator.NOT_IN,
            value=[1, 2, 3],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NOT NULL AND NOT (' in where_clause  # NOT_IN = presence guard + negated membership
        assert 'IN (' in where_clause
        assert len(params) == 3
        # Numeric members bind as raw ints (matched numerically, not stringified).
        assert all(isinstance(p, int) for p in params)
        assert 1 in params
        assert 2 in params
        assert 3 in params

    def test_operator_not_in_with_mixed_types(self) -> None:
        """Test NOT IN operator with mixed string and integer array values."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.NOT_IN,
            value=['archived', 100, 'deleted', 200],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NOT NULL AND NOT (' in where_clause  # NOT_IN = presence guard + negated membership
        assert 'IN (' in where_clause
        assert len(params) == 4
        # Type-aware: string members bind as text, numeric members bind as raw numbers.
        assert 'archived' in params
        assert 'deleted' in params
        assert 100 in params
        assert 200 in params

    def test_operator_exists(self) -> None:
        """Test EXISTS operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.EXISTS)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NOT NULL' in where_clause
        assert len(params) == 0

    def test_operator_not_exists(self) -> None:
        """Test NOT EXISTS operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='optional_field', operator=MetadataOperator.NOT_EXISTS)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NULL' in where_clause
        assert len(params) == 0

    def test_operator_contains(self) -> None:
        """Test CONTAINS operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='description',
            operator=MetadataOperator.CONTAINS,
            value='important',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause
        assert "'%' ||" in where_clause
        assert params == ['important']

    def test_operator_starts_with(self) -> None:
        """Test STARTS_WITH operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='name',
            operator=MetadataOperator.STARTS_WITH,
            value='test_',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause
        assert "|| '%'" in where_clause
        # The '_' in the value is escaped so starts_with matches it literally,
        # not as a single-char wildcard (paired with the ESCAPE clause).
        assert params == ['test\\_']

    def test_operator_ends_with(self) -> None:
        """Test ENDS_WITH operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='filename',
            operator=MetadataOperator.ENDS_WITH,
            value='.txt',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause
        assert "'%' ||" in where_clause
        assert params == ['.txt']

    def test_operator_is_null(self) -> None:
        """Test IS_NULL operator."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(key='deleted_at', operator=MetadataOperator.IS_NULL)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'json_type' in where_clause
        assert "= 'null'" in where_clause
        assert len(params) == 0

    def test_case_insensitive_string_comparison(self) -> None:
        """Test case-insensitive string operations."""
        builder = MetadataQueryBuilder()
        filter_spec = MetadataFilter(
            key='name',
            operator=MetadataOperator.EQ,
            value='TEST',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, _ = builder.build_where_clause()
        assert 'LOWER' in where_clause

    def test_nested_json_path(self) -> None:
        """Test nested JSON path support."""
        builder = MetadataQueryBuilder()
        builder.add_simple_filter('user.preferences.theme', 'dark')

        where_clause, params = builder.build_where_clause()
        assert '$.user.preferences.theme' in where_clause
        assert params == ['dark']

    def test_sql_injection_prevention(self) -> None:
        """Test that SQL injection attempts are prevented."""
        builder = MetadataQueryBuilder()

        # Attempt SQL injection in key
        with pytest.raises(ValueError, match='Invalid metadata key'):
            builder.add_simple_filter("status'; DROP TABLE context_entries; --", 'active')

        # Valid key with special characters should work
        builder.add_simple_filter('valid_key-123.nested', 'value')
        where_clause, _ = builder.build_where_clause()
        assert where_clause is not None

    def test_empty_filters(self) -> None:
        """Test behavior with no filters."""
        builder = MetadataQueryBuilder()
        where_clause, params = builder.build_where_clause()
        assert where_clause == ''
        assert params == []

    def test_filter_count(self) -> None:
        """Test filter counting."""
        builder = MetadataQueryBuilder()
        assert builder.get_filter_count() == 0

        builder.add_simple_filter('status', 'active')
        assert builder.get_filter_count() == 1

        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        builder.add_advanced_filter(filter_spec)
        assert builder.get_filter_count() == 2
