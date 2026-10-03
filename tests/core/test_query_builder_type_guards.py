"""MetadataQueryBuilder JSON type guards: boolean and string values match only same-typed stored values."""

import pytest

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


class TestMetadataQueryBuilder:
    """Test the MetadataQueryBuilder class."""

    @pytest.mark.parametrize(
        ('backend', 'expected'),
        [('sqlite', ['1', '0']), ('postgresql', ['true', 'false'])],
    )
    def test_in_operator_normalizes_booleans(self, backend: str, expected: list[str]) -> None:
        """IN with boolean members binds the per-backend stored boolean text.

        A blanket str(bool) would bind 'True'/'False', matching neither backend.
        """
        builder = MetadataQueryBuilder(backend_type=backend)
        builder.add_advanced_filter(
            MetadataFilter(key='flag', operator=MetadataOperator.IN, value=[True, False]),
        )
        _clause, params = builder.build_where_clause()
        assert params == expected

    @pytest.mark.parametrize(
        ('backend', 'expected'),
        [('sqlite', ['1']), ('postgresql', ['true'])],
    )
    def test_not_in_operator_normalizes_booleans(self, backend: str, expected: list[str]) -> None:
        """NOT_IN with a boolean member binds the per-backend stored boolean text."""
        builder = MetadataQueryBuilder(backend_type=backend)
        builder.add_advanced_filter(
            MetadataFilter(key='flag', operator=MetadataOperator.NOT_IN, value=[True]),
        )
        _clause, params = builder.build_where_clause()
        assert params == expected

    @pytest.mark.parametrize('operator', [MetadataOperator.EQ, MetadataOperator.NE])
    def test_boolean_guards_boolean_type(self, operator: MetadataOperator) -> None:
        """Boolean EQ/NE are JSON-boolean-typed-only on both backends: SQLite guards on
        ``json_type(...) IN ('true', 'false')`` and PostgreSQL on
        ``jsonb_typeof(...) = 'boolean'`` so a numeric 0/1 or a string 'true'/'false' never
        matches a boolean param -- the parity counterpart to the number-only contract."""
        sbuilder = MetadataQueryBuilder(backend_type='sqlite')
        sbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value=True))
        sclause, _ = sbuilder.build_where_clause()
        assert "json_type(metadata, '$.flag') IN ('true', 'false')" in sclause

        pbuilder = MetadataQueryBuilder(backend_type='postgresql')
        pbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value=True))
        pclause, _ = pbuilder.build_where_clause()
        assert "jsonb_typeof(metadata->'flag') = 'boolean'" in pclause

    @pytest.mark.parametrize('operator', [MetadataOperator.IN, MetadataOperator.NOT_IN])
    def test_in_boolean_member_is_type_guarded(self, operator: MetadataOperator) -> None:
        """A boolean member of IN/NOT_IN is matched JSON-boolean-only on both backends so
        it cannot collide with a same-text non-boolean (SQLite renders a JSON boolean and a
        JSON integer 1/0 BOTH as '1'/'0'; PostgreSQL renders a JSON boolean and a JSON string
        'true'/'false' BOTH as that text). The guard mirrors the EQ/NE boolean contract."""
        sbuilder = MetadataQueryBuilder(backend_type='sqlite')
        sbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value=[True]))
        sclause, _ = sbuilder.build_where_clause()
        assert "json_type(metadata, '$.flag') IN ('true', 'false')" in sclause

        pbuilder = MetadataQueryBuilder(backend_type='postgresql')
        pbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value=[True]))
        pclause, _ = pbuilder.build_where_clause()
        assert "jsonb_typeof(metadata->'flag') = 'boolean'" in pclause

    @pytest.mark.parametrize('operator', [MetadataOperator.EQ, MetadataOperator.NE])
    def test_string_value_excludes_stored_boolean(self, operator: MetadataOperator) -> None:
        """A STRING EQ/NE value matches a JSON-string-typed stored value ONLY, so a stored
        JSON boolean (and a stored number) is excluded. PostgreSQL ``->>`` renders a boolean as
        'true'/'false' text, so without the text guard a string 'true' would match a stored
        boolean on PostgreSQL but not SQLite. The text-typed guard makes both backends exclude
        a non-string -- parity by construction."""
        sbuilder = MetadataQueryBuilder(backend_type='sqlite')
        sbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value='true'))
        sclause, _ = sbuilder.build_where_clause()
        assert "json_type(metadata, '$.flag') = 'text'" in sclause

        pbuilder = MetadataQueryBuilder(backend_type='postgresql')
        pbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value='true'))
        pclause, _ = pbuilder.build_where_clause()
        assert "jsonb_typeof(metadata->'flag') = 'string'" in pclause

    @pytest.mark.parametrize(
        'operator',
        [
            MetadataOperator.CONTAINS, MetadataOperator.STARTS_WITH, MetadataOperator.ENDS_WITH,
            MetadataOperator.GT, MetadataOperator.GTE, MetadataOperator.LT, MetadataOperator.LTE,
        ],
    )
    def test_string_comparing_operators_exclude_stored_boolean(self, operator: MetadataOperator) -> None:
        """contains/starts_with/ends_with and a STRING-valued gt/gte/lt/lte match a
        JSON-string-typed stored value ONLY, so a stored JSON boolean (and a stored number) is
        excluded on both backends. The text-typed guard is applied for parity by construction."""
        sbuilder = MetadataQueryBuilder(backend_type='sqlite')
        sbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value='tru'))
        sclause, _ = sbuilder.build_where_clause()
        assert "json_type(metadata, '$.flag') = 'text'" in sclause

        pbuilder = MetadataQueryBuilder(backend_type='postgresql')
        pbuilder.add_advanced_filter(MetadataFilter(key='flag', operator=operator, value='tru'))
        pclause, _ = pbuilder.build_where_clause()
        assert "jsonb_typeof(metadata->'flag') = 'string'" in pclause

    def test_simple_filter_string_excludes_stored_boolean(self) -> None:
        """add_simple_filter (the ``metadata`` dict path) restricts a string value to a
        JSON-string-typed stored value too (excluding a stored boolean or number), matching the
        advanced EQ contract on both backends."""
        sbuilder = MetadataQueryBuilder(backend_type='sqlite')
        sbuilder.add_simple_filter('flag', 'true')
        sclause, _ = sbuilder.build_where_clause()
        assert "json_type(metadata, '$.flag') = 'text'" in sclause

        pbuilder = MetadataQueryBuilder(backend_type='postgresql')
        pbuilder.add_simple_filter('flag', 'true')
        pclause, _ = pbuilder.build_where_clause()
        assert "jsonb_typeof(metadata->'flag') = 'string'" in pclause

    def test_case_insensitive_fold_is_ascii_only(self) -> None:
        """Case-insensitive matching folds ASCII A-Z ONLY on both backends so they agree on
        non-ASCII data: PostgreSQL uses translate() (not its full-Unicode LOWER()), SQLite keeps
        its ASCII-only LOWER(), and IN/NOT_IN members are ASCII-folded in Python (not str.lower)."""
        pbuilder = MetadataQueryBuilder(backend_type='postgresql')
        pbuilder.add_advanced_filter(MetadataFilter(key='k', operator=MetadataOperator.EQ, value='ACTIVE'))
        pclause, _ = pbuilder.build_where_clause()
        assert 'translate(' in pclause
        assert 'LOWER(' not in pclause  # PG must NOT use full-Unicode LOWER()

        sbuilder = MetadataQueryBuilder(backend_type='sqlite')
        sbuilder.add_advanced_filter(MetadataFilter(key='k', operator=MetadataOperator.EQ, value='ACTIVE'))
        sclause, _ = sbuilder.build_where_clause()
        assert 'LOWER(' in sclause  # SQLite's built-in LOWER() is ASCII-only (the reference)

        # IN string members are ASCII-folded, NOT str.lower() (full-Unicode): 'CAFÉ' (CAFE
        # with uppercase E-acute) folds to 'cafÉ' -- C/A/F lowered, the accent untouched,
        # matching the SQL accessor fold so both backends agree.
        ibuilder = MetadataQueryBuilder(backend_type='postgresql')
        ibuilder.add_advanced_filter(MetadataFilter(key='k', operator=MetadataOperator.IN, value=['CAFÉ']))
        _clause, iparams = ibuilder.build_where_clause()
        assert iparams == ['cafÉ']

    def test_array_contains_boolean_is_type_guarded_sqlite(self) -> None:
        """array_contains with a boolean member guards json_each.type on SQLite so a JSON
        boolean array element (whose json_each.value is 1/0) is not confused with a numeric
        1/0 element -- matching PostgreSQL's type-exact ``@> '<json bool>'::jsonb``."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(MetadataFilter(key='flags', operator=MetadataOperator.ARRAY_CONTAINS, value=True))
        clause, _ = builder.build_where_clause()
        assert "json_each.type IN ('true', 'false')" in clause

    def test_array_contains_case_sensitive_string_is_type_guarded_sqlite(self) -> None:
        """array_contains with a case-sensitive string member guards json_each.type='text'.

        SQLite's json_each.value renders a NESTED array/object element as its minified
        JSON text, which a string member CAN equal (value '["x","y"]' against element
        ["x","y"]), while PostgreSQL's type-exact ``@> '"<str>"'::jsonb`` containment
        never matches a string against a container element. The text guard closes that
        cross-backend divergence and mirrors the case-insensitive branch.
        """
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(
            MetadataFilter(key='refs', operator=MetadataOperator.ARRAY_CONTAINS, value='x', case_sensitive=True),
        )
        clause, params = builder.build_where_clause()
        assert "json_each.type = 'text'" in clause
        assert params == ['x']

    def test_array_contains_string_does_not_match_container_element_sqlite(self) -> None:
        """A string equal to a container's minified JSON text does not match on SQLite.

        Functional end-to-end check of the text guard against a real SQLite session:
        the entry's array holds a nested array element whose json_each.value text is
        exactly the filter value, and it still must not match (PostgreSQL parity).
        """
        import json
        import sqlite3

        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(
            MetadataFilter(
                key='refs',
                operator=MetadataOperator.ARRAY_CONTAINS,
                value='["x","y"]',
                case_sensitive=True,
            ),
        )
        clause, params = builder.build_where_clause()

        with sqlite3.connect(':memory:') as conn:
            conn.execute('CREATE TABLE t (metadata TEXT)')
            conn.execute(
                'INSERT INTO t (metadata) VALUES (?)',
                (json.dumps({'refs': ['a', ['x', 'y']]}),),
            )
            matches = conn.execute(f'SELECT COUNT(*) FROM t WHERE {clause}', params).fetchone()[0]
            assert matches == 0

            # A genuine string element with the same characters still matches.
            conn.execute(
                'INSERT INTO t (metadata) VALUES (?)',
                (json.dumps({'refs': ['a', '["x","y"]']}),),
            )
            matches = conn.execute(f'SELECT COUNT(*) FROM t WHERE {clause}', params).fetchone()[0]
            assert matches == 1
