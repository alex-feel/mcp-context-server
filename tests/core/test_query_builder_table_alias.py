"""MetadataQueryBuilder table_alias qualification of the metadata column on both backends."""

import re

import pytest

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


def _alias_filter_cases(key: str) -> list[MetadataFilter]:
    """One MetadataFilter per operator on ``key``, honoring each value contract.

    Each of the 16 metadata operators emits structurally different SQL that the
    table_alias column-qualification regex (query_builder.build_where_clause) must
    handle. The list/None values are literals passed directly to the model, so they
    are inferred in-context against the field type (no cast needed).

    Returns:
        One MetadataFilter per supported operator, all bound to ``key``.
    """
    return [
        MetadataFilter(key=key, operator=MetadataOperator.EQ, value=5),
        MetadataFilter(key=key, operator=MetadataOperator.NE, value=5),
        MetadataFilter(key=key, operator=MetadataOperator.GT, value=5),
        MetadataFilter(key=key, operator=MetadataOperator.GTE, value=5),
        MetadataFilter(key=key, operator=MetadataOperator.LT, value=5),
        MetadataFilter(key=key, operator=MetadataOperator.LTE, value=5),
        MetadataFilter(key=key, operator=MetadataOperator.IN, value=[1, 2]),
        MetadataFilter(key=key, operator=MetadataOperator.NOT_IN, value=[1, 2]),
        MetadataFilter(key=key, operator=MetadataOperator.EXISTS, value=None),
        MetadataFilter(key=key, operator=MetadataOperator.NOT_EXISTS, value=None),
        MetadataFilter(key=key, operator=MetadataOperator.CONTAINS, value='x'),
        MetadataFilter(key=key, operator=MetadataOperator.STARTS_WITH, value='x'),
        MetadataFilter(key=key, operator=MetadataOperator.ENDS_WITH, value='x'),
        MetadataFilter(key=key, operator=MetadataOperator.IS_NULL, value=None),
        MetadataFilter(key=key, operator=MetadataOperator.IS_NOT_NULL, value=None),
        MetadataFilter(key=key, operator=MetadataOperator.ARRAY_CONTAINS, value='x'),
    ]


# Built over two keys whose NAME contains the substring 'metadata' (top-level and
# nested) -- exactly the keys a substring replace of 'metadata' would corrupt.
_ALIAS_FILTER_CASES = _alias_filter_cases('metadata_version') + _alias_filter_cases('a.metadata_b')

# A metadata COLUMN position NOT already qualified with the 'ce.' alias (a bare
# leak the FTS/semantic JOIN could mis-resolve). SQLite passes the column as the
# first json_extract/json_type/json_each argument (immediately followed by a
# comma); PostgreSQL accesses it via ->/#>. A JSON key never matches either form
# (a key never precedes ->/#> nor sits in the column-comma position).
_SQLITE_COLUMN_LEAK_RE = re.compile(r'(?<!ce\.)\b(?:json_extract|json_type|json_each)\(\s*metadata\s*,')
_PG_COLUMN_LEAK_RE = re.compile(r'(?<![\w.])metadata(?=->|#>)')


class TestMetadataQueryBuilderTableAlias:
    """table_alias qualifies the metadata COLUMN, never a JSON key.

    A substring replace of 'metadata' with 'ce.metadata' would corrupt any JSON key
    containing that substring (a filter on 'metadata_version' would read
    'ce.metadata_version', so PostgreSQL semantic search would match no rows). The
    builder qualifies only column positions.
    """

    def test_no_alias_emits_bare_column(self) -> None:
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_simple_filter('status', 'active')
        where_clause, _ = builder.build_where_clause()
        assert "metadata->>'status'" in where_clause
        assert 'ce.metadata' not in where_clause

    def test_alias_qualifies_column(self) -> None:
        builder = MetadataQueryBuilder(backend_type='postgresql', table_alias='ce')
        builder.add_simple_filter('status', 'active')
        where_clause, _ = builder.build_where_clause()
        assert "ce.metadata->>'status'" in where_clause

    def test_alias_does_not_corrupt_metadata_substring_key(self) -> None:
        # A substring replace would rewrite the 'metadata_version' key to 'ce.metadata_version'.
        builder = MetadataQueryBuilder(backend_type='postgresql', table_alias='ce')
        builder.add_simple_filter('metadata_version', '1')
        where_clause, _ = builder.build_where_clause()
        assert "ce.metadata->>'metadata_version'" in where_clause
        assert "'ce.metadata_version'" not in where_clause

    def test_alias_does_not_corrupt_key_named_exactly_metadata(self) -> None:
        builder = MetadataQueryBuilder(backend_type='postgresql', table_alias='ce')
        builder.add_simple_filter('metadata', 'x')
        where_clause, _ = builder.build_where_clause()
        assert "ce.metadata->>'metadata'" in where_clause
        assert "->>'ce.metadata'" not in where_clause

    def test_alias_does_not_corrupt_nested_metadata_segment(self) -> None:
        # A nested array-path segment named 'metadata' (followed by a comma) must
        # NOT be mistaken for the column.
        builder = MetadataQueryBuilder(backend_type='postgresql', table_alias='ce')
        builder.add_simple_filter('metadata.version', 'x')
        where_clause, _ = builder.build_where_clause()
        nested = '{"metadata","version"}'
        assert f"ce.metadata#>>'{nested}'" in where_clause
        assert '{"ce.metadata","version"}' not in where_clause

    def test_alias_qualifies_sqlite_json_extract_column(self) -> None:
        builder = MetadataQueryBuilder(backend_type='sqlite', table_alias='ce')
        builder.add_simple_filter('metadata_version', '1')
        where_clause, _ = builder.build_where_clause()
        assert "json_extract(ce.metadata, '$.metadata_version')" in where_clause
        assert '$.ce.metadata_version' not in where_clause

    @pytest.mark.parametrize('backend', ['sqlite', 'postgresql'])
    @pytest.mark.parametrize('filter_spec', _ALIAS_FILTER_CASES, ids=lambda f: f'{f.operator.value}-{f.key}')
    def test_alias_qualifies_every_operator_without_key_corruption(
        self,
        backend: str,
        filter_spec: MetadataFilter,
    ) -> None:
        """Every operator: column qualified to ce.metadata, 'metadata'-substring key intact.

        The simple-filter cases above pin only ``eq``. These pin the structurally
        distinct branches the column-qualification regex must also handle -- the
        multi-column-position SQLite ``array_contains`` (json_type + json_each) and
        the ``->``-form PostgreSQL ``is_null``/``exists``/``array_contains`` -- so a
        regex or operator-SQL change cannot silently introduce JSON-key corruption
        (e.g. ``metadata_version`` -> ``ce.metadata_version``) on an operator no test
        exercises. The parametrization covers every operator on both backends.
        """
        builder = MetadataQueryBuilder(backend_type=backend, table_alias='ce')
        builder.add_advanced_filter(filter_spec)
        clause, _ = builder.build_where_clause()
        op_name = filter_spec.operator.value
        assert clause, f'{op_name} on {backend} produced no clause'
        leak_re = _SQLITE_COLUMN_LEAK_RE if backend == 'sqlite' else _PG_COLUMN_LEAK_RE
        assert leak_re.search(clause) is None, f'unqualified metadata column leaked ({op_name}/{backend}): {clause}'
        key_segment = filter_spec.key.split('.')[-1]
        assert key_segment in clause, f'key {key_segment} missing ({op_name}/{backend}): {clause}'
        assert f'ce.{key_segment}' not in clause, f'JSON key corrupted ({op_name}/{backend}): {clause}'

    def test_alias_sqlite_array_contains_qualifies_both_column_positions(self) -> None:
        # SQLite array_contains is the only form with TWO column positions
        # (json_type(...) AND json_each(...)); both must be qualified, key intact.
        builder = MetadataQueryBuilder(backend_type='sqlite', table_alias='ce')
        builder.add_advanced_filter(
            MetadataFilter(key='metadata_version', operator=MetadataOperator.ARRAY_CONTAINS, value='x'),
        )
        clause, _ = builder.build_where_clause()
        assert 'json_type(ce.metadata,' in clause
        assert 'json_each(ce.metadata,' in clause
        assert "'$.ce.metadata_version'" not in clause

    def test_alias_pg_is_null_qualifies_arrow_form_column(self) -> None:
        # PostgreSQL is_null uses the '->' (not '->>') accessor via jsonb_typeof.
        builder = MetadataQueryBuilder(backend_type='postgresql', table_alias='ce')
        builder.add_advanced_filter(
            MetadataFilter(key='metadata_version', operator=MetadataOperator.IS_NULL, value=None),
        )
        clause, _ = builder.build_where_clause()
        assert 'jsonb_typeof(ce.metadata->' in clause
        assert "'ce.metadata_version'" not in clause

    def test_alias_pg_array_contains_qualifies_arrow_form_column(self) -> None:
        # PostgreSQL array_contains uses '->' via jsonb_array_elements / jsonb_typeof.
        builder = MetadataQueryBuilder(backend_type='postgresql', table_alias='ce')
        builder.add_advanced_filter(
            MetadataFilter(key='metadata_version', operator=MetadataOperator.ARRAY_CONTAINS, value='x'),
        )
        clause, _ = builder.build_where_clause()
        assert 'jsonb_array_elements(ce.metadata->' in clause
        assert "'ce.metadata_version'" not in clause
