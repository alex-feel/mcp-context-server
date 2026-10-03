"""MetadataQueryBuilder ARRAY_CONTAINS clauses on SQLite and PostgreSQL, including the non-array type check."""

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


class TestArrayContainsQueryBuilder:
    """Tests for the ARRAY_CONTAINS operator in MetadataQueryBuilder."""

    def test_sqlite_array_contains_string(self) -> None:
        """Test SQLite array_contains with string value."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='python',
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'EXISTS' in where_clause
        assert 'json_each' in where_clause
        assert '$.technologies' in where_clause
        assert params == ['python']

    def test_sqlite_array_contains_case_insensitive(self) -> None:
        """Test SQLite array_contains with case-insensitive string."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='PYTHON',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'LOWER' in where_clause
        assert 'json_each' in where_clause
        assert params == ['PYTHON']

    def test_sqlite_array_contains_integer(self) -> None:
        """Test SQLite array_contains with integer value."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='priority_levels',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=5,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'EXISTS' in where_clause
        assert 'json_each' in where_clause
        assert params == [5]

    def test_sqlite_array_contains_boolean(self) -> None:
        """Test SQLite array_contains with boolean value."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='flags',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'EXISTS' in where_clause
        assert 'json_each' in where_clause
        # Boolean should be converted to 1
        assert params == [1]

    def test_postgresql_array_contains_string(self) -> None:
        """Test PostgreSQL array_contains with string value."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='python',
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '@>' in where_clause
        # Uses ::jsonb cast instead of to_jsonb() to avoid asyncpg type resolution issues
        assert '::jsonb' in where_clause
        # Value is JSON-stringified for ::jsonb cast
        assert params == ['"python"']

    def test_postgresql_array_contains_case_insensitive(self) -> None:
        """Test PostgreSQL array_contains with case-insensitive string."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='PYTHON',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'EXISTS' in where_clause
        assert 'jsonb_array_elements(' in where_clause  # iterate as jsonb (string-typed only), not _text
        assert "jsonb_typeof(elem) = 'string'" in where_clause
        # Case-insensitive fold is ASCII-only via translate() (parity with SQLite LOWER).
        assert 'translate' in where_clause
        assert params == ['PYTHON']

    def test_postgresql_array_contains_nested_path(self) -> None:
        """Test PostgreSQL array_contains with nested path."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='references.context_ids',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=200,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert '@>' in where_clause
        # Uses ::jsonb cast instead of to_jsonb() to avoid asyncpg type resolution issues
        assert '::jsonb' in where_clause
        assert '{"references","context_ids"}' in where_clause
        # Value is JSON-stringified for ::jsonb cast
        assert params == ['200']

    def test_sqlite_array_contains_nested_path(self) -> None:
        """Test SQLite array_contains with nested path."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='references.context_ids',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=200,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert 'EXISTS' in where_clause
        assert 'json_each' in where_clause
        assert '$.references.context_ids' in where_clause
        assert params == [200]


class TestArrayContainsNonArrayHandling:
    """Tests for array_contains graceful handling of non-array fields.

    These tests verify that the SQL includes type checks to prevent errors
    when array_contains is used on non-array fields.
    """

    def test_sqlite_array_contains_includes_type_check(self) -> None:
        """Test SQLite array_contains SQL includes json_type check."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='category',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='backend',
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert "json_type(metadata, '$.category') = 'array'" in where_clause
        assert 'json_each' in where_clause
        assert params == ['backend']

    def test_sqlite_array_contains_case_insensitive_includes_type_check(self) -> None:
        """Test SQLite case-insensitive array_contains SQL includes json_type check."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='PYTHON',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert "json_type(metadata, '$.technologies') = 'array'" in where_clause
        assert "json_each.type = 'text'" in where_clause  # string member matches string elements only
        assert 'LOWER' in where_clause
        assert 'json_each' in where_clause
        assert params == ['PYTHON']

    def test_sqlite_array_contains_boolean_includes_type_check(self) -> None:
        """Test SQLite array_contains with boolean includes json_type check."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        filter_spec = MetadataFilter(
            key='flags',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert "json_type(metadata, '$.flags') = 'array'" in where_clause
        assert 'json_each' in where_clause
        assert params == [1]

    def test_postgresql_array_contains_includes_type_check(self) -> None:
        """Test PostgreSQL array_contains SQL includes jsonb_typeof check."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='category',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='backend',
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert "jsonb_typeof(metadata->'category') = 'array'" in where_clause
        assert 'CASE WHEN' in where_clause
        assert 'ELSE FALSE END' in where_clause
        assert '@>' in where_clause

    def test_postgresql_array_contains_case_insensitive_includes_type_check(self) -> None:
        """Test PostgreSQL case-insensitive array_contains SQL includes jsonb_typeof check."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='PYTHON',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        assert "jsonb_typeof(metadata->'technologies') = 'array'" in where_clause
        assert 'jsonb_array_elements(' in where_clause  # iterate as jsonb (string-typed only), not _text
        assert "jsonb_typeof(elem) = 'string'" in where_clause
        assert 'CASE WHEN' in where_clause
        assert 'ELSE FALSE END' in where_clause
        assert 'translate' in where_clause  # ASCII-only ci fold (parity with SQLite LOWER)

    def test_postgresql_nested_path_includes_type_check(self) -> None:
        """Test PostgreSQL nested path array_contains SQL includes jsonb_typeof check."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='references.context_ids',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=200,
            case_sensitive=True,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        nested = '{"references","context_ids"}'  # quoted segments: a 'null' segment stays a literal key
        assert f"jsonb_typeof(metadata#>'{nested}') = 'array'" in where_clause
        assert 'CASE WHEN' in where_clause
        assert 'ELSE FALSE END' in where_clause

    def test_postgresql_nested_case_insensitive_includes_type_check(self) -> None:
        """Test PostgreSQL nested case-insensitive array_contains SQL includes jsonb_typeof check."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        filter_spec = MetadataFilter(
            key='references.youtrack',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='AI-100',
            case_sensitive=False,
        )
        builder.add_advanced_filter(filter_spec)

        where_clause, params = builder.build_where_clause()
        nested = '{"references","youtrack"}'
        assert f"jsonb_typeof(metadata#>'{nested}') = 'array'" in where_clause
        assert 'jsonb_array_elements(' in where_clause  # iterate as jsonb (string-typed only), not _text
        assert "jsonb_typeof(elem) = 'string'" in where_clause
        assert 'CASE WHEN' in where_clause
        assert 'ELSE FALSE END' in where_clause
        assert 'translate' in where_clause  # ASCII-only ci fold (parity with SQLite LOWER)
