"""Nested JSON path traversal and PostgreSQL path-segment quoting in MetadataQueryBuilder."""

import pytest

from app.metadata_sql import _pg_path_literal
from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder

_NESTED_KEY = 'user.preferences.theme'
# Every path segment is double-quoted so a segment spelled 'null' stays a literal key
# instead of parsing as a SQL NULL array element (which would make the whole accessor NULL).
_NESTED_ARRAY = '{"user","preferences","theme"}'


def _pg_nested_operator_cases(key: str = _NESTED_KEY) -> list[MetadataFilter]:
    """One MetadataFilter per advanced operator on a NESTED key.

    Each operator emits structurally distinct SQL that must TRAVERSE the nested
    path via PostgreSQL ``#>>``/``#>`` array notation. The list/None values are
    literals inferred in-context against the field type.

    Args:
        key: The nested metadata key every filter is bound to.

    Returns:
        One MetadataFilter per supported operator, all bound to a nested key.
    """
    k = key
    return [
        MetadataFilter(key=k, operator=MetadataOperator.EQ, value='dark'),
        MetadataFilter(key=k, operator=MetadataOperator.NE, value='dark'),
        MetadataFilter(key=k, operator=MetadataOperator.GT, value=5),
        MetadataFilter(key=k, operator=MetadataOperator.GTE, value=5),
        MetadataFilter(key=k, operator=MetadataOperator.LT, value=5),
        MetadataFilter(key=k, operator=MetadataOperator.LTE, value=5),
        MetadataFilter(key=k, operator=MetadataOperator.IN, value=['a', 'b']),
        MetadataFilter(key=k, operator=MetadataOperator.NOT_IN, value=['a', 'b']),
        MetadataFilter(key=k, operator=MetadataOperator.EXISTS, value=None),
        MetadataFilter(key=k, operator=MetadataOperator.NOT_EXISTS, value=None),
        MetadataFilter(key=k, operator=MetadataOperator.CONTAINS, value='x'),
        MetadataFilter(key=k, operator=MetadataOperator.STARTS_WITH, value='x'),
        MetadataFilter(key=k, operator=MetadataOperator.ENDS_WITH, value='x'),
        MetadataFilter(key=k, operator=MetadataOperator.IS_NULL, value=None),
        MetadataFilter(key=k, operator=MetadataOperator.IS_NOT_NULL, value=None),
        MetadataFilter(key=k, operator=MetadataOperator.ARRAY_CONTAINS, value='x'),
    ]


_PG_NESTED_OPERATOR_CASES = _pg_nested_operator_cases()


class TestMetadataQueryBuilderPostgresqlNestedPathTraversal:
    """Every PostgreSQL advanced operator TRAVERSES a nested key via #>>/#>.

    Each operator must read ``metadata#>>'{a,b,c}'``, never ``metadata->>'a.b.c'``, a
    literal top-level key named ``a.b.c`` that never traverses on PostgreSQL. SQLite
    always traverses (``json_extract`` with ``$.a.b.c``), so a non-traversing PostgreSQL
    operator would return wrong or empty results on PostgreSQL only. These tests pin
    every operator.
    """

    @pytest.mark.parametrize('filter_spec', _PG_NESTED_OPERATOR_CASES, ids=lambda f: f.operator.value)
    def test_pg_nested_operator_uses_array_notation(self, filter_spec: MetadataFilter) -> None:
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_advanced_filter(filter_spec)
        clause, _ = builder.build_where_clause()
        op = filter_spec.operator.value
        assert clause, f'{op} produced no clause'
        # Must traverse via the array-notation accessor...
        assert _NESTED_ARRAY in clause, f'{op} did not use #>>/#> array notation: {clause}'
        # ...and NEVER read a literal top-level key whose name contains the dots.
        assert _NESTED_KEY not in clause, f'{op} read a literal dotted key: {clause}'

    @pytest.mark.parametrize('filter_spec', _PG_NESTED_OPERATOR_CASES, ids=lambda f: f.operator.value)
    def test_sqlite_nested_operator_traverses(self, filter_spec: MetadataFilter) -> None:
        # Counterpart guard: SQLite traverses for every operator via the full
        # JSONPath ($.user.preferences.theme), so the cross-backend result agrees.
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(filter_spec)
        clause, _ = builder.build_where_clause()
        op = filter_spec.operator.value
        assert clause, f'{op} produced no clause'
        assert f'$.{_NESTED_KEY}' in clause, f'{op} did not traverse on SQLite: {clause}'


_NULL_SEGMENT_KEY = 'a.null'
_QUOTED_NULL_PATH = '{"a","null"}'
_UNQUOTED_NULL_PATH = '{a,null}'
_PG_NULL_SEGMENT_CASES = _pg_nested_operator_cases(_NULL_SEGMENT_KEY)


class TestMetadataQueryBuilderPostgresqlPathSegmentQuoting:
    """Every PostgreSQL path segment is DOUBLE-QUOTED inside the ``text[]`` accessor literal.

    PostgreSQL's array-literal parser reads an unquoted, case-insensitive bareword ``null``
    as a genuine SQL NULL element, and ``#>>``/``#>`` return NULL as soon as any path element
    is NULL. An unquoted literal would therefore collapse the whole accessor to NULL for a
    legitimate object key spelled ``null``: ``a.null eq x`` would match nothing on PostgreSQL
    while SQLite matches the row, and ``a.null not_exists`` would return the very entry that
    DOES carry the key. The key validators accept such a key (only empty and post-first numeric
    segments are rejected), so quoting is what makes the two backends traverse the same path.
    """

    @pytest.mark.parametrize('filter_spec', _PG_NULL_SEGMENT_CASES, ids=lambda f: f.operator.value)
    def test_null_segment_is_quoted_for_every_operator(self, filter_spec: MetadataFilter) -> None:
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_advanced_filter(filter_spec)
        clause, _ = builder.build_where_clause()
        op = filter_spec.operator.value
        assert clause, f'{op} produced no clause'
        assert _QUOTED_NULL_PATH in clause, f'{op} did not quote the path segments: {clause}'
        assert _UNQUOTED_NULL_PATH not in clause, f'{op} emitted an unquoted NULL segment: {clause}'

    def test_null_segment_is_quoted_on_the_simple_equality_path(self) -> None:
        # The simple metadata={} dict routes into the same accessor, so it needs the same quoting.
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter(_NULL_SEGMENT_KEY, 'x')
        clause, params = builder.build_where_clause()
        assert _QUOTED_NULL_PATH in clause
        assert _UNQUOTED_NULL_PATH not in clause
        assert params == ['x']

    def test_null_segment_traverses_the_same_path_on_sqlite(self) -> None:
        # Parity control: SQLite reads the segment as a literal object key with no quoting,
        # which is the behavior PostgreSQL must match.
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_simple_filter(_NULL_SEGMENT_KEY, 'x')
        clause, _ = builder.build_where_clause()
        assert '$.a.null' in clause

    @pytest.mark.parametrize(
        ('key', 'expected'),
        [
            ('a.b', '{"a","b"}'),
            ('a.null', '{"a","null"}'),
            ('a.NULL', '{"a","NULL"}'),
            ('null.Null', '{"null","Null"}'),
            ('a.b-c.d_e', '{"a","b-c","d_e"}'),
            ('single', '{"single"}'),
        ],
    )
    def test_path_literal_quotes_every_segment(self, key: str, expected: str) -> None:
        # The validators restrict segments to [A-Za-z0-9_-], so quoting needs no escaping.
        assert _pg_path_literal(key) == expected

    def test_flat_key_named_null_needs_no_array_literal(self) -> None:
        # A flat key uses ->>'null', a plain key name rather than an array literal, so array-literal
        # NULL parsing never applies to it -- and it must stay unquoted or it would look up the key '"null"'.
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('null', 'x')
        clause, _ = builder.build_where_clause()
        assert "metadata->>'null'" in clause
        assert '"null"' not in clause

    def test_table_alias_still_qualifies_the_column_under_quoting(self) -> None:
        # The alias rewrite keys on the column token followed by ->/#>, so quoted segments
        # (including one literally named 'metadata') are still never mistaken for the column.
        builder = MetadataQueryBuilder(backend_type='postgresql', table_alias='ce')
        builder.add_simple_filter('metadata.null', 'x')
        clause, _ = builder.build_where_clause()
        assert 'ce.metadata#>>' in clause
        assert '{"metadata","null"}' in clause
        assert '{"ce.metadata"' not in clause
