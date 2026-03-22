"""Tests for PostgreSQL-specific query builder functionality.

This module tests the MetadataQueryBuilder class with PostgreSQL backend type
to ensure proper SQL generation for PostgreSQL syntax.
"""

from __future__ import annotations

import pytest

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


class TestMetadataQueryBuilderPostgresql:
    """Test the MetadataQueryBuilder class with PostgreSQL backend."""

    def test_simple_filter_postgresql(self) -> None:
        """Test simple key=value filtering for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('status', 'active')

        where_clause, params = builder.build_where_clause()
        # PostgreSQL uses ->> operator and $1, $2... placeholders
        assert '->>' in where_clause
        assert '$1' in where_clause
        assert params == ['active']

    def test_multiple_simple_filters_postgresql(self) -> None:
        """Test multiple simple filters for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('status', 'active')
        builder.add_simple_filter('priority', 5)

        where_clause, params = builder.build_where_clause()
        assert '->>' in where_clause
        assert '$1' in where_clause
        assert '$2' in where_clause
        assert len(params) == 2
        assert 'active' in params
        assert 5 in params

    def test_operator_eq_postgresql(self) -> None:
        """Test equality operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.EQ,
            value='active',
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '->>' in where_clause
        assert '=' in where_clause
        assert params == ['active']

    def test_operator_eq_case_insensitive_postgresql(self) -> None:
        """Test case-insensitive equality operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.EQ,
            value='ACTIVE',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LOWER' in where_clause
        # PostgreSQL uses ->> or #>> for JSON access
        assert '->>' in where_clause or '#>>' in where_clause
        # Value is passed as-is, LOWER is applied in SQL to both sides
        assert params == ['ACTIVE']

    def test_operator_ne_postgresql(self) -> None:
        """Test not-equal operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.NE,
            value='inactive',
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '!=' in where_clause or '<>' in where_clause
        assert '->>' in where_clause
        assert params == ['inactive']

    def test_operator_gt_postgresql(self) -> None:
        """Test greater-than operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # PostgreSQL uses ::numeric for type casting
        assert '::' in where_clause or 'CAST' in where_clause
        assert '>' in where_clause
        assert params == [5]

    def test_operator_gte_postgresql(self) -> None:
        """Test greater-than-or-equal operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.GTE, value=5)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '>=' in where_clause
        assert params == [5]

    def test_operator_lt_postgresql(self) -> None:
        """Test less-than operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.LT, value=10)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # Should be just < not <=
        assert '<' in where_clause
        assert params == [10]

    def test_operator_lte_postgresql(self) -> None:
        """Test less-than-or-equal operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.LTE, value=10)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '<=' in where_clause
        assert params == [10]

    def test_operator_in_postgresql(self) -> None:
        """Test IN operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.IN,
            value=['active', 'pending', 'review'],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IN (' in where_clause
        assert '$1' in where_clause
        assert '$2' in where_clause
        assert '$3' in where_clause
        assert len(params) == 3
        assert 'active' in params

    def test_operator_not_in_postgresql(self) -> None:
        """Test NOT IN operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='status',
            operator=MetadataOperator.NOT_IN,
            value=['deleted', 'archived'],
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'NOT IN (' in where_clause
        assert len(params) == 2

    def test_operator_exists_postgresql(self) -> None:
        """Test EXISTS operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.EXISTS)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NOT NULL' in where_clause
        assert len(params) == 0

    def test_operator_not_exists_postgresql(self) -> None:
        """Test NOT EXISTS operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='optional', operator=MetadataOperator.NOT_EXISTS)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'IS NULL' in where_clause
        assert len(params) == 0

    def test_operator_contains_postgresql(self) -> None:
        """Test CONTAINS operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='description',
            operator=MetadataOperator.CONTAINS,
            value='important',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause or 'ILIKE' in where_clause
        assert params == ['important']

    def test_operator_starts_with_postgresql(self) -> None:
        """Test STARTS_WITH operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='name',
            operator=MetadataOperator.STARTS_WITH,
            value='test_',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause or 'ILIKE' in where_clause
        assert params == ['test_']

    def test_operator_ends_with_postgresql(self) -> None:
        """Test ENDS_WITH operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='filename',
            operator=MetadataOperator.ENDS_WITH,
            value='.txt',
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LIKE' in where_clause or 'ILIKE' in where_clause
        assert params == ['.txt']

    def test_operator_is_null_postgresql(self) -> None:
        """Test IS_NULL operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='deleted_at', operator=MetadataOperator.IS_NULL)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # PostgreSQL uses jsonb_typeof for null checks
        assert 'jsonb_typeof' in where_clause or 'null' in where_clause.lower()
        assert len(params) == 0

    def test_operator_is_not_null_postgresql(self) -> None:
        """Test IS_NOT_NULL operator for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(key='created_at', operator=MetadataOperator.IS_NOT_NULL)
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'NOT' in where_clause or 'IS NOT NULL' in where_clause
        assert len(params) == 0

    def test_nested_json_path_postgresql(self) -> None:
        """Test nested JSON path support for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('user.preferences.theme', 'dark')

        where_clause, params = builder.build_where_clause()
        # PostgreSQL uses #>> with path array syntax for nested paths
        assert '#>>' in where_clause or '->>' in where_clause
        assert 'user' in where_clause
        assert 'preferences' in where_clause
        assert 'theme' in where_clause
        assert params == ['dark']

    def test_deeply_nested_path_postgresql(self) -> None:
        """Test deeply nested JSON path for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('a.b.c.d.e', 'value')

        where_clause, params = builder.build_where_clause()
        # PostgreSQL uses #>> with path array syntax like '{a,b,c,d,e}'
        assert '#>>' in where_clause or '->>' in where_clause
        # All path components should be in the query
        for comp in ['a', 'b', 'c', 'd', 'e']:
            assert comp in where_clause
        assert params == ['value']

    def test_sql_injection_prevention_postgresql(self) -> None:
        """Test that SQL injection attempts are prevented for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')

        with pytest.raises(ValueError, match='Invalid metadata key'):
            builder.add_simple_filter("status'; DROP TABLE context_entries; --", 'active')

    def test_empty_filters_postgresql(self) -> None:
        """Test behavior with no filters for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        where_clause, params = builder.build_where_clause()
        assert where_clause == ''
        assert params == []

    def test_filter_count_postgresql(self) -> None:
        """Test filter counting for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        assert builder.get_filter_count() == 0

        builder.add_simple_filter('status', 'active')
        assert builder.get_filter_count() == 1

        filter_spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        builder.add_advanced_filter(filter_spec)
        assert builder.get_filter_count() == 2

    def test_combined_filters_postgresql(self) -> None:
        """Test combining multiple filter types for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')

        # Add simple filter
        builder.add_simple_filter('status', 'active')

        # Add advanced filters
        builder.add_advanced_filter(
            MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5),
        )
        builder.add_advanced_filter(
            MetadataFilter(key='agent_name', operator=MetadataOperator.EXISTS),
        )

        where_clause, params = builder.build_where_clause()

        # Should have all conditions combined with AND
        assert 'AND' in where_clause
        assert builder.get_filter_count() == 3
        # Should have 2 params (status=active and priority>5, EXISTS has no param)
        assert len(params) == 2

    def test_boolean_true_value_postgresql(self) -> None:
        """Test boolean True value handling for PostgreSQL.

        PostgreSQL JSONB ->> operator extracts boolean values as TEXT strings
        ('true' or 'false'), not as numeric values. The query builder must:
        1. Check isinstance(value, bool) BEFORE isinstance(value, (int, float))
           because bool is a subclass of int in Python
        2. Use TEXT comparison (not NUMERIC cast) for boolean values
        3. Normalize True to 'true' string for parameter binding
        """
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='is_active',
            operator=MetadataOperator.EQ,
            value=True,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # Should use TEXT comparison, not NUMERIC cast
        assert '::TEXT' in where_clause
        assert '::NUMERIC' not in where_clause
        # Boolean True should be normalized to 'true' string for PostgreSQL
        assert params == ['true']

    def test_boolean_false_value_postgresql(self) -> None:
        """Test boolean False value handling for PostgreSQL.

        Same as True, but verifies False is normalized to 'false' string.
        """
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='is_active',
            operator=MetadataOperator.EQ,
            value=False,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # Should use TEXT comparison, not NUMERIC cast
        assert '::TEXT' in where_clause
        assert '::NUMERIC' not in where_clause
        # Boolean False should be normalized to 'false' string for PostgreSQL
        assert params == ['false']

    def test_boolean_not_equal_postgresql(self) -> None:
        """Test boolean not-equal operator for PostgreSQL.

        Verifies that NE operator with boolean values also uses TEXT comparison.
        """
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='is_completed',
            operator=MetadataOperator.NE,
            value=True,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # Should use != with TEXT comparison
        assert '!=' in where_clause
        assert '::TEXT' in where_clause
        assert '::NUMERIC' not in where_clause
        assert params == ['true']

    def test_boolean_simple_filter_postgresql(self) -> None:
        """Test boolean value in simple filter for PostgreSQL.

        Simple filters (add_simple_filter) should also handle booleans correctly.
        """
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('completed', True)

        where_clause, params = builder.build_where_clause()
        # Should use TEXT comparison, not NUMERIC cast
        assert '::TEXT' in where_clause
        assert '::NUMERIC' not in where_clause
        assert params == ['true']

    def test_boolean_nested_path_postgresql(self) -> None:
        """Test boolean value with nested JSON path for PostgreSQL.

        Verifies boolean handling works correctly with nested paths (#>> operator).
        """
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='user.settings.notifications_enabled',
            operator=MetadataOperator.EQ,
            value=True,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # Should use #>> for nested path with TEXT comparison
        assert '#>>' in where_clause
        assert '::TEXT' in where_clause
        assert '::NUMERIC' not in where_clause
        assert params == ['true']

    def test_numeric_value_postgresql(self) -> None:
        """Test numeric value handling for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='score',
            operator=MetadataOperator.EQ,
            value=42,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '=' in where_clause
        assert params == [42]

    def test_float_value_postgresql(self) -> None:
        """Test float value handling for PostgreSQL."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='rating',
            operator=MetadataOperator.GTE,
            value=3.5,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '>=' in where_clause
        assert params == [3.5]


class TestQueryBuilderBackendDetection:
    """Test backend type detection in query builder."""

    def test_default_backend_is_sqlite(self) -> None:
        """Test that default backend is SQLite."""
        builder = MetadataQueryBuilder()
        builder.add_simple_filter('status', 'active')

        where_clause, _ = builder.build_where_clause()
        # SQLite uses json_extract
        assert 'json_extract' in where_clause

    def test_explicit_sqlite_backend(self) -> None:
        """Test explicit SQLite backend."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_simple_filter('status', 'active')

        where_clause, _ = builder.build_where_clause()
        assert 'json_extract' in where_clause

    def test_explicit_postgresql_backend(self) -> None:
        """Test explicit PostgreSQL backend."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('status', 'active')

        where_clause, _ = builder.build_where_clause()
        assert '->>' in where_clause

    def test_placeholder_difference(self) -> None:
        """Test placeholder syntax difference between backends."""
        # SQLite uses ?
        sqlite_builder = MetadataQueryBuilder(backend_type='sqlite')
        sqlite_builder.add_simple_filter('status', 'active')
        sqlite_clause, _ = sqlite_builder.build_where_clause()
        assert '?' in sqlite_clause

        # PostgreSQL uses $1, $2, etc.
        pg_builder = MetadataQueryBuilder(backend_type='postgresql')
        pg_builder.add_simple_filter('status', 'active')
        pg_clause, _ = pg_builder.build_where_clause()
        assert '$1' in pg_clause


class TestSqliteBooleanBackwardCompatibility:
    """Test SQLite boolean handling to ensure backward compatibility.

    SQLite stores JSON booleans as integers (0/1), which differs from PostgreSQL's
    TEXT storage ('true'/'false'). These tests verify that the boolean fix for
    PostgreSQL did not break SQLite functionality.
    """

    def test_boolean_true_value_sqlite(self) -> None:
        """Test boolean True value handling for SQLite.

        SQLite stores JSON booleans as integers (1 for true, 0 for false).
        """
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='is_active',
            operator=MetadataOperator.EQ,
            value=True,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        # Should use json_extract for SQLite
        assert 'json_extract' in where_clause
        # Boolean True should be normalized to integer 1 for SQLite
        assert params == [1]

    def test_boolean_false_value_sqlite(self) -> None:
        """Test boolean False value handling for SQLite."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='is_active',
            operator=MetadataOperator.EQ,
            value=False,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'json_extract' in where_clause
        # Boolean False should be normalized to integer 0 for SQLite
        assert params == [0]

    def test_boolean_not_equal_sqlite(self) -> None:
        """Test boolean not-equal operator for SQLite."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='is_completed',
            operator=MetadataOperator.NE,
            value=True,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '!=' in where_clause
        assert 'json_extract' in where_clause
        assert params == [1]

    def test_boolean_simple_filter_sqlite(self) -> None:
        """Test boolean value in simple filter for SQLite."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_simple_filter('completed', True)

        where_clause, params = builder.build_where_clause()
        assert 'json_extract' in where_clause
        # Boolean should be normalized to integer for SQLite
        assert params == [1]

    def test_boolean_false_simple_filter_sqlite(self) -> None:
        """Test boolean False in simple filter for SQLite."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_simple_filter('completed', False)

        where_clause, params = builder.build_where_clause()
        assert 'json_extract' in where_clause
        assert params == [0]
