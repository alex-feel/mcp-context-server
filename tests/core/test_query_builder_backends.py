"""MetadataQueryBuilder backend selection and SQLite boolean comparisons."""

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


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


class TestSqliteBooleanIntegerStorage:
    """SQLite boolean filters compare JSON booleans as stored integers.

    SQLite stores JSON booleans as integers (0/1), unlike PostgreSQL's TEXT
    ('true'/'false'); these tests pin the SQLite comparison form that pairs with
    the PostgreSQL boolean handling.
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
