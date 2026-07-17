"""Comprehensive tests for metadata filtering functionality."""

import math
import time
from typing import TYPE_CHECKING

import pytest

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder
from app.server import search_context
from app.server import store_context
from app.types import JsonValue

if TYPE_CHECKING:
    pass


_TestData = list[dict[str, JsonValue]]


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

    @pytest.mark.parametrize(
        'operator',
        [
            MetadataOperator.EQ, MetadataOperator.NE,
            MetadataOperator.GT, MetadataOperator.GTE, MetadataOperator.LT, MetadataOperator.LTE,
        ],
    )
    def test_pg_numeric_guards_number_type(self, operator: MetadataOperator) -> None:
        """Numeric operators are number-typed-only on PostgreSQL: an explicit
        ``jsonb_typeof(...) = 'number'`` AND-guard excludes every non-number value
        (text/bool/json-null/absent), so the query never aborts and a non-number never
        matches. There is no ``ELSE 0`` coercion (that diverged from SQLite for booleans
        and numeric-prefix strings); both backends use the same number-only contract."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_advanced_filter(
            MetadataFilter(key='priority', operator=operator, value=5),
        )
        clause, _params = builder.build_where_clause()
        assert "jsonb_typeof(metadata->'priority') = 'number'" in clause
        assert '::NUMERIC' in clause
        assert 'ELSE 0' not in clause
        assert '::DOUBLE PRECISION' not in clause

    @pytest.mark.parametrize(
        'operator',
        [
            MetadataOperator.EQ, MetadataOperator.NE,
            MetadataOperator.GT, MetadataOperator.GTE, MetadataOperator.LT, MetadataOperator.LTE,
        ],
    )
    def test_sqlite_numeric_guards_number_type(self, operator: MetadataOperator) -> None:
        """Numeric operators are number-typed-only on SQLite too: each is guarded by
        ``json_type(metadata, '$.priority') IN ('integer', 'real')`` so a non-number
        value (text/bool/json-null/absent) never matches -- the parity counterpart to
        the PostgreSQL ``jsonb_typeof(...) = 'number'`` guard."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(
            MetadataFilter(key='priority', operator=operator, value=5),
        )
        clause, _params = builder.build_where_clause()
        assert "json_type(metadata, '$.priority') IN ('integer', 'real')" in clause

    @pytest.mark.parametrize(
        'operator',
        [
            MetadataOperator.EQ, MetadataOperator.NE,
            MetadataOperator.GT, MetadataOperator.GTE, MetadataOperator.LT, MetadataOperator.LTE,
        ],
    )
    def test_pg_float_param_matches_sqlite_exact_semantics(self, operator: MetadataOperator) -> None:
        """On PostgreSQL a FLOAT param reproduces SQLite's exact integer/double comparison:
        the exact-NUMERIC compare is kept only for stored values that are integral AND provably
        int-origin (not equal to their nearest double's shortest round-trip decimal, probed via
        ``(stored::float8)::NUMERIC``); everything else -- fractional, or integral values that ARE
        the canonical form of some double (every float stores as its shortest repr, so above 2**53
        the stored NUMERIC differs from the double's exact value) -- compares double-vs-double
        (both snapped to float8). The stored side is never down-cast through DOUBLE PRECISION
        (which truncated a stored integer > 2**53) and the param is never run through
        float8::numeric (which rounds the PARAM to ~15 significant digits); an INTEGER param
        compares exact NUMERIC for every stored value."""
        fbuilder = MetadataQueryBuilder(backend_type='postgresql')
        fbuilder.add_advanced_filter(MetadataFilter(key='score', operator=operator, value=0.3))
        fclause, _ = fbuilder.build_where_clause()
        # Float param: integrality + int-origin CASE (trunc + text-routed canonical-form
        # probe -- float8out is shortest-repr while a direct float8::NUMERIC cast rounds
        # to ~15 digits), double-vs-double branch (::float8), exact NUMERIC stored side;
        # never DOUBLE PRECISION, never float8::numeric anywhere.
        assert 'trunc(' in fclause
        assert '::float8' in fclause
        assert '::float8)::text::NUMERIC' in fclause
        assert ')::NUMERIC' in fclause
        assert '::DOUBLE PRECISION' not in fclause
        assert '::float8::numeric' not in fclause
        assert '::float8)::NUMERIC' not in fclause

        ibuilder = MetadataQueryBuilder(backend_type='postgresql')
        ibuilder.add_advanced_filter(MetadataFilter(key='score', operator=operator, value=5))
        iclause, _ = ibuilder.build_where_clause()
        # Integer param: exact NUMERIC, no integrality CASE, no float8, no DOUBLE PRECISION.
        assert ')::NUMERIC' in iclause
        assert 'trunc(' not in iclause
        assert '::float8' not in iclause
        assert '::DOUBLE PRECISION' not in iclause

    def test_pg_high_magnitude_int_not_truncated_by_float_param(self) -> None:
        """A float param must not truncate a stored high-magnitude integer on PostgreSQL.

        Regression for the cross-backend divergence where a stored integer > 2**53 was down-cast
        through DOUBLE PRECISION (losing its low bits) -- and where folding the param via
        float8::numeric rounds to ~15 significant digits -- so the same metadata_filter returned
        different rows than SQLite. The fix reads the stored value as exact NUMERIC and branches on
        its integrality, across the advanced EQ path, the IN per-member path, and the
        simple-equality path."""
        # Advanced EQ with a float param: integrality CASE, exact NUMERIC stored, no truncation.
        eq = MetadataQueryBuilder(backend_type='postgresql')
        eq.add_advanced_filter(MetadataFilter(key='n', operator=MetadataOperator.EQ, value=9007199254740992.0))
        eq_clause, _ = eq.build_where_clause()
        assert '::DOUBLE PRECISION' not in eq_clause
        assert '::float8::numeric' not in eq_clause
        assert 'trunc(' in eq_clause
        assert ')::NUMERIC' in eq_clause

        # IN with a float member: same per-member integrality CASE, no stored-side truncation.
        in_b = MetadataQueryBuilder(backend_type='postgresql')
        in_b.add_advanced_filter(MetadataFilter(key='n', operator=MetadataOperator.IN, value=[9007199254740992.0]))
        in_clause, _ = in_b.build_where_clause()
        assert '::DOUBLE PRECISION' not in in_clause
        assert '::float8::numeric' not in in_clause
        assert 'trunc(' in in_clause

        # Simple metadata={} equality with a float value uses the same guard + body.
        simple = MetadataQueryBuilder(backend_type='postgresql')
        simple.add_simple_filter('n', 9007199254740992.0)
        simple_clause, _ = simple.build_where_clause()
        assert '::DOUBLE PRECISION' not in simple_clause
        assert '::float8::numeric' not in simple_clause
        assert 'trunc(' in simple_clause
        assert ')::NUMERIC' in simple_clause

    @pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_filter_value_rejected(self, bad: float) -> None:
        """A NaN/Infinity filter value is rejected on both the advanced and simple paths.

        SQLite binds a non-finite float as NULL (matching nothing) while
        PostgreSQL orders NaN above all numbers (matching everything), so the
        same filter diverges; rejecting it uniformly is the parity-correct
        contract (mirroring the int64 guard).
        """
        with pytest.raises(ValueError, match='[Nn]on-finite'):
            MetadataFilter(key='v', operator=MetadataOperator.LT, value=bad)

        with pytest.raises(ValueError, match='[Nn]on-finite'):
            MetadataQueryBuilder(backend_type='sqlite').add_simple_filter('v', bad)

        # A non-finite member inside an IN list is rejected too.
        with pytest.raises(ValueError, match='[Nn]on-finite'):
            MetadataFilter(key='v', operator=MetadataOperator.IN, value=[1.0, bad])

    def test_non_finite_metadata_error_walks_nested_structures(self) -> None:
        """Stored metadata is scanned recursively for non-finite floats.

        A non-finite float at any depth serializes to invalid JSON that
        PostgreSQL's jsonb parser rejects, so the store must fail fast before
        generation; finite floats and non-float values return no error.
        """
        from app.metadata_types import non_finite_metadata_error

        assert non_finite_metadata_error({'a': 1, 'b': [0.5, {'c': 'ok'}], 'd': True}) is None
        for bad_meta in (
            {'v': float('nan')},
            {'nested': [{'deep': float('inf')}]},
            [1, 2, float('-inf')],
            {'a': {'b': {'c': float('nan')}}},
        ):
            message = non_finite_metadata_error(bad_meta)
            assert message is not None
            assert 'on-finite' in message

    def test_pg_float_discriminator_guards_int64_and_float8_range(self) -> None:
        """The float discriminator guards the int64 boundary and never overflows float8.

        A float param routes through a nested CASE that (a) keeps the exact-NUMERIC
        compare only for integral stored values WITHIN int64 -- an out-of-int64 integer
        SQLite reads as REAL must take the double branch -- and (b) maps an
        out-of-float8-range stored value to +/-inf instead of casting it to float8,
        which would raise 22003 and abort the whole query on a legal 309-digit stored
        integer.
        """
        b = MetadataQueryBuilder(backend_type='postgresql')
        b.add_advanced_filter(MetadataFilter(key='n', operator=MetadataOperator.GT, value=1.5))
        clause, _ = b.build_where_clause()
        # int64-boundary guard: the exact branch only fires within int64.
        assert 'BETWEEN -9223372036854775808 AND 9223372036854775807' in clause
        # safe-float8 maps out-of-range magnitudes to +/-inf (no 22003 abort).
        assert "'infinity'::float8" in clause
        assert "'-infinity'::float8" in clause
        # The infinity threshold is the TRUE float8 overflow boundary (2**1024 - 2**970),
        # crossed with >= / <= -- NOT DBL_MAX's shortest-repr decimal, which is strictly
        # smaller and would misclassify the finite band (DBL_MAX, overflow) as Infinity.
        overflow = str(2**1024 - 2**970)
        assert overflow == MetadataQueryBuilder._FLOAT8_OVERFLOW
        assert f'>= {overflow}' in clause
        assert f'<= -{overflow}' in clause
        assert '1.7976931348623157e308' not in clause
        # This asserts SQL TEXT only, never cross-engine runtime agreement at the boundary.
        # The midpoint is PostgreSQL's exact float8 overflow point and agrees with SQLite
        # builds that flip there (<= 3.40.x), but newer SQLite (observed 3.47.x/3.49.x) reads
        # a stored integer in a narrow band at/above the midpoint as a FINITE DBL_MAX while
        # this guard maps it to Infinity -- a version-dependent, irreducible residual (no
        # single literal tracks SQLite's flip point), so no cross-engine behavior is asserted.
        # The exact-form probe still routes through ::text (Ryu shortest-repr).
        assert '::float8)::text::NUMERIC' in clause

    def test_pg_float_discriminator_clamps_underflow_to_zero(self) -> None:
        """safe_float8 clamps the symmetric low-magnitude underflow band to 0.

        A nonzero stored NUMERIC of magnitude <= 2**-1075 (the IEEE round-to-zero boundary)
        underflows to 0.0 when cast to float8; PostgreSQL raises 22003 and aborts the WHOLE
        query, while SQLite reads the same stored value as 0.0. The float discriminator's
        double branch maps that band to 0 so the query neither aborts nor diverges -- the
        symmetric twin of the high-magnitude overflow clamp. The threshold MUST be exact: the
        smallest denormal (2**-1074) just above it is a legal float8 both engines keep, so an
        approximate boundary would either clamp a value PostgreSQL and SQLite both preserve or
        leave a residual abort gap just below the boundary.
        """
        b = MetadataQueryBuilder(backend_type='postgresql')
        b.add_advanced_filter(MetadataFilter(key='n', operator=MetadataOperator.LT, value=0.5))
        clause, _ = b.build_where_clause()
        tiny = MetadataQueryBuilder._FLOAT8_TINY
        # The underflow clamp is present, keyed on the exact boundary and excluding a genuine 0.
        assert f'BETWEEN -{tiny} AND {tiny}' in clause
        assert '<> 0' in clause
        assert 'THEN (0)::float8' in clause
        # _FLOAT8_TINY is EXACTLY 2**-1075 = 5**1075 / 10**1075 (a fixed-point string with 1075
        # fractional digits), NOT the Python float 2**-1075 (which underflows to 0.0) nor a
        # context-rounded Decimal.
        assert tiny.startswith('0.')
        assert len(tiny[2:]) == 1075
        assert int(tiny[2:]) == 5**1075

    def test_pg_array_contains_float_member_uses_shared_discriminator(self) -> None:
        """A float array_contains member matches numeric elements via the shared discriminator.

        A bare ``@>`` containment matches only the float's canonical decimal form and
        diverges from SQLite's exact int-vs-double element comparison above 2**53. The
        float-member path iterates numeric elements and reuses _pg_numeric_compare, so a
        genuinely int-origin element now matches on both backends; ints/strings keep @>.
        """
        fb = MetadataQueryBuilder(backend_type='postgresql')
        fb.add_advanced_filter(
            MetadataFilter(key='vals', operator=MetadataOperator.ARRAY_CONTAINS, value=3.602879701896397e16),
        )
        fclause, fparams = fb.build_where_clause()
        assert 'jsonb_array_elements' in fclause
        assert "jsonb_typeof(elem) = 'number'" in fclause
        assert "(elem #>> '{}')::NUMERIC" in fclause
        assert '@>' not in fclause
        # The numeric member is bound as the raw float, not json.dumps text.
        assert fparams == [3.602879701896397e16]

        # An INTEGER member still uses exact @> containment (json.dumps is exact).
        ib = MetadataQueryBuilder(backend_type='postgresql')
        ib.add_advanced_filter(
            MetadataFilter(key='vals', operator=MetadataOperator.ARRAY_CONTAINS, value=7),
        )
        iclause, _ = ib.build_where_clause()
        assert '@>' in iclause
        assert 'jsonb_array_elements' not in iclause

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
        # Numeric members bind as raw ints now (not stringified for TEXT comparison).
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
        # Numeric members bind as raw floats now (matched numerically, not as text).
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
        builder = MetadataQueryBuilder()
        for bad in ('items.0', 'a.-1', 'a.0.b', 'items.01'):
            assert builder._is_safe_key(bad) is False
            with pytest.raises(ValueError, match='Numeric path segments'):
                MetadataFilter(key=bad, operator=MetadataOperator.EQ, value='x')
        # Allowed: a single numeric key (consistent object-key on both backends) and
        # non-numeric nested paths (a numeric-suffixed segment like 'b0' is not all-digits).
        for good in ('0', 'a.b', 'items.foo', 'user.preferences.theme', 'a.b0'):
            assert builder._is_safe_key(good) is True
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
        builder = MetadataQueryBuilder()
        for bad in ('.x', 'x.', 'a..b', '.', '..'):
            assert builder._is_safe_key(bad) is False
            with pytest.raises(ValueError, match='Empty path segments'):
                MetadataFilter(key=bad, operator=MetadataOperator.EQ, value=1)
        # Allowed: keys whose every dot-separated segment is non-empty stay valid.
        for good in ('a', 'a.b', '0', 'metadata_version', 'a-b', 'user.preferences.theme'):
            assert builder._is_safe_key(good) is True
            MetadataFilter(key=good, operator=MetadataOperator.EQ, value=1)  # must not raise

    def test_trailing_newline_key_rejected(self) -> None:
        """A key ending in a newline is rejected on BOTH validators. Python's `$` also matches
        immediately before a single trailing '\\n', so a `re.match(r'^...$')` gate would have
        passed 'status\\n'; the un-stripped simple-filter path then diverged (SQLite
        json_extract('$.a.status\\n') misses while PostgreSQL's #>> array-literal parse trims
        the newline and matches). fullmatch closes the parity gap. Clean keys stay valid."""
        builder = MetadataQueryBuilder()
        for bad in ('status\n', 'a.status\n', 'status\n\n', 'a\nb'):
            assert builder._is_safe_key(bad) is False
            with pytest.raises(ValueError, match='Invalid metadata key'):
                MetadataFilter(key=bad, operator=MetadataOperator.EQ, value='x')
        for good in ('status', 'a.status', 'user.preferences.theme'):
            assert builder._is_safe_key(good) is True
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
        # Numeric members bind as raw ints now (matched numerically, not stringified).
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


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestMetadataFilteringIntegration:
    """Integration tests for metadata filtering with the full stack."""

    async def _setup_test_data(self) -> None:
        """Helper method to set up test data."""
        # Use a unique thread_id for each test run
        import time
        from typing import Any

        self.test_thread_id = f'test_metadata_{int(time.time() * 1000)}'

        # Create fresh test data
        test_data: list[dict[str, Any]] = [
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 1',
                'metadata': {'status': 'active', 'priority': 5, 'agent_name': 'planner'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 2',
                'metadata': {'status': 'pending', 'priority': 3, 'agent_name': 'executor'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'user',
                'text': 'Task 3',
                'metadata': {'status': 'active', 'priority': 8, 'agent_name': 'reviewer'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 4',
                'metadata': {'status': 'completed', 'priority': 1},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 5',
                'metadata': {'status': 'error', 'priority': 10, 'error_message': 'timeout'},
            },
            {
                'thread_id': self.test_thread_id,
                'source': 'agent',
                'text': 'Task 6 - no metadata',
                'metadata': None,
            },
        ]

        for data in test_data:
            await store_context(**data, ctx=None)

    @pytest.mark.asyncio
    async def test_simple_metadata_filter(self) -> None:
        """Test simple metadata filtering."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 2
        for entry in result['results']:
            assert entry['metadata']['status'] == 'active'

    @pytest.mark.asyncio
    async def test_multiple_simple_filters(self) -> None:
        """Test multiple simple metadata filters."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active', 'priority': 5},
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'Task 1'

    @pytest.mark.asyncio
    async def test_advanced_gt_operator(self) -> None:
        """Test greater-than operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[{'key': 'priority', 'operator': 'gt', 'value': 5}],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 2  # priority 8 and 10
        priorities = [e['metadata']['priority'] for e in result['results']]
        assert all(p > 5 for p in priorities)

    @pytest.mark.asyncio
    async def test_advanced_in_operator(self) -> None:
        """Test IN operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'status',
                    'operator': 'in',
                    'value': ['active', 'pending'],
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 3
        statuses = [e['metadata']['status'] for e in result['results']]
        assert all(s in ['active', 'pending'] for s in statuses)

    @pytest.mark.asyncio
    async def test_advanced_in_operator_with_integer_array(self) -> None:
        """Test IN operator with integer array values.

        Regression test: Integer arrays caused silent failures on SQLite
        (type mismatch with json_extract TEXT result) and explicit errors
        on PostgreSQL (asyncpg type mismatch with TEXT cast).
        """
        await self._setup_test_data()

        # Test IN with integer array [5, 10]
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'priority',
                    'operator': 'in',
                    'value': [5, 10],  # Integer array
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        # Should find entries with priority 5 and 10
        assert len(result['results']) == 2
        priorities = [e['metadata']['priority'] for e in result['results']]
        assert all(p in [5, 10] for p in priorities)

    @pytest.mark.asyncio
    async def test_advanced_not_in_operator_with_integer_array(self) -> None:
        """Test NOT IN operator with integer array values.

        Regression test: Integer arrays caused failures in NOT IN operator.
        """
        await self._setup_test_data()

        # Test NOT IN with integer array - should exclude entries with priority 1, 3, 5
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'priority',
                    'operator': 'not_in',
                    'value': [1, 3, 5],  # Integer array
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        # Should find entries with priority 8 and 10 (excluding 1, 3, 5)
        assert len(result['results']) == 2
        priorities = [e['metadata']['priority'] for e in result['results']]
        assert all(p not in [1, 3, 5] for p in priorities)

    @pytest.mark.asyncio
    async def test_advanced_exists_operator(self) -> None:
        """Test EXISTS operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[{'key': 'agent_name', 'operator': 'exists'}],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 3
        for entry in result['results']:
            assert 'agent_name' in entry['metadata']

    @pytest.mark.asyncio
    async def test_advanced_contains_operator(self) -> None:
        """Test CONTAINS operator."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[
                {
                    'key': 'agent_name',
                    'operator': 'contains',
                    'value': 'plan',
                },
            ],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 1
        assert result['results'][0]['metadata']['agent_name'] == 'planner'

    @pytest.mark.asyncio
    async def test_combined_filters(self) -> None:
        """Test combining simple and advanced filters."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            source='agent',
            metadata={'status': 'active'},
            metadata_filters=[{'key': 'priority', 'operator': 'gte', 'value': 5}],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 1
        entry = result['results'][0]
        assert entry['metadata']['status'] == 'active'
        assert entry['metadata']['priority'] >= 5
        assert entry['source'] == 'agent'

    @pytest.mark.asyncio
    async def test_explain_query(self) -> None:
        """Test query explanation feature."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            explain_query=True,
            ctx=None,
        )

        assert 'results' in result
        assert 'stats' in result
        stats = result['stats']
        assert 'execution_time_ms' in stats
        assert 'filters_applied' in stats
        assert 'rows_returned' in stats
        # The implementation counts filters differently - accept either 1 or 2
        assert stats['filters_applied'] in [1, 2]  # Could be just metadata filter or thread_id + metadata
        assert stats['rows_returned'] == 2

    @pytest.mark.asyncio
    async def test_empty_result_set(self) -> None:
        """Test filtering that returns no results."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'nonexistent'},
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_null_metadata_handling(self) -> None:
        """Test handling of entries with null metadata."""
        await self._setup_test_data()

        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata_filters=[{'key': 'status', 'operator': 'not_exists'}],
            ctx=None,
        )

        assert 'results' in result
        # Should find the entry with null metadata
        found_null = any('no metadata' in e['text_content'] for e in result['results'])
        assert found_null

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_metadata_filter_performance(self) -> None:
        """Test that metadata filtering meets performance targets."""
        await self._setup_test_data()

        # Simple filter performance test
        start_time = time.time()
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            ctx=None,
        )
        simple_time = (time.time() - start_time) * 1000

        assert simple_time < 200  # Should be under 200ms (relaxed for CI variability)
        assert 'results' in result

        # Complex filter performance test
        start_time = time.time()
        result = await search_context(
            limit=50,
            thread_id=self.test_thread_id,
            metadata={'status': 'active'},
            metadata_filters=[
                {'key': 'priority', 'operator': 'gt', 'value': 3},
                {'key': 'agent_name', 'operator': 'exists'},
            ],
            ctx=None,
        )
        complex_time = (time.time() - start_time) * 1000

        assert complex_time < 500  # Should be under 500ms (relaxed for CI variability)
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_in_operator_over_member_cap_returns_structured_validation_error(self) -> None:
        """An IN filter above MAX_IN_LIST_MEMBERS short-circuits through the
        metadata-filter validation channel (a structured, breaker-exempt error
        response) instead of expanding into an oversized single-statement SQL bind."""
        from app.metadata_types import MAX_IN_LIST_MEMBERS

        oversized: list[str | int | float | bool] = [f'v{i}' for i in range(MAX_IN_LIST_MEMBERS + 1)]
        result = await search_context(
            limit=50,
            thread_id='in_member_cap_thread',
            metadata_filters=[{'key': 'status', 'operator': 'in', 'value': oversized}],
            ctx=None,
        )

        assert result['results'] == []
        assert result['count'] == 0
        assert 'error' in result
        assert any('at most' in message for message in result['validation_errors'])


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
@pytest.mark.parametrize(
    ('operator', 'value', 'expected_count'),
    [
        (MetadataOperator.EQ, 'active', 2),
        (MetadataOperator.NE, 'active', 3),
        (MetadataOperator.GT, 5, 2),
        (MetadataOperator.GTE, 5, 3),
        (MetadataOperator.LT, 5, 2),
        (MetadataOperator.LTE, 5, 3),
        (MetadataOperator.IN, ['active', 'pending'], 3),
        (MetadataOperator.NOT_IN, ['active', 'pending'], 2),
        (MetadataOperator.EXISTS, None, 3),
        (MetadataOperator.NOT_EXISTS, None, 2),
    ],
)
async def test_all_operators(
    operator: MetadataOperator,
    value: str | int | list[str] | None,
    expected_count: int,
) -> None:
    """Parameterized test for all metadata operators."""
    # Create test data
    test_data: list[dict[str, JsonValue]] = [
        {'status': 'active', 'priority': 5, 'agent_name': 'planner'},
        {'status': 'pending', 'priority': 3, 'agent_name': 'executor'},
        {'status': 'active', 'priority': 8, 'agent_name': 'reviewer'},
        {'status': 'completed', 'priority': 1},
        {'status': 'error', 'priority': 10},
    ]

    for i, metadata in enumerate(test_data):
        await store_context(
            thread_id='test_operators',
            source='agent',
            text=f'Task {i + 1}',
            metadata=metadata,
            ctx=None,
        )

    # Determine which field to filter on
    if operator in (MetadataOperator.EXISTS, MetadataOperator.NOT_EXISTS):
        key = 'agent_name'
    elif operator in (MetadataOperator.GT, MetadataOperator.GTE, MetadataOperator.LT, MetadataOperator.LTE):
        key = 'priority'
    else:
        key = 'status'

    # Apply filter
    result = await search_context(
        limit=50,
        thread_id='test_operators',
        metadata_filters=[{'key': key, 'operator': operator.value, 'value': value}],
        ctx=None,
    )

    assert 'results' in result
    assert len(result['results']) == expected_count


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestMetadataFilterErrorHandling:
    """Test error handling for invalid metadata filters."""

    @pytest.mark.asyncio
    async def test_invalid_operator_returns_validation_error(self) -> None:
        """Test that invalid operator returns explicit validation error."""
        result = await search_context(
            limit=50,
            metadata_filters=[{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}],
            ctx=None,
        )

        assert 'error' in result
        assert result['error'] == 'Metadata filter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 1
        error_msg = result['validation_errors'][0].lower()
        assert 'invalid_operator' in error_msg or 'invalid' in error_msg

    @pytest.mark.asyncio
    async def test_multiple_invalid_filters_returns_all_errors(self) -> None:
        """Test that multiple invalid filters return all validation errors."""
        result = await search_context(
            limit=50,
            metadata_filters=[
                {'key': 'status', 'operator': 'invalid_op1', 'value': 'test'},
                {'key': 'priority', 'operator': 'invalid_op2', 'value': 123},
            ],
            ctx=None,
        )

        assert 'error' in result
        assert result['error'] == 'Metadata filter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 2

    @pytest.mark.asyncio
    async def test_invalid_key_with_sql_injection_returns_error(self) -> None:
        """Test that invalid keys (SQL injection attempts) return validation error."""
        result = await search_context(
            limit=50,
            metadata_filters=[{'key': 'DROP TABLE;--', 'operator': 'eq', 'value': 'test'}],
            ctx=None,
        )

        assert 'error' in result
        assert result['error'] == 'Metadata filter validation failed'
        assert 'validation_errors' in result
        assert len(result['validation_errors']) == 1


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestNestedJSONMetadata:
    """Test nested JSON structures in metadata."""

    @pytest.mark.asyncio
    async def test_store_nested_objects(self) -> None:
        """Test storing nested JSON objects in metadata."""
        complex_metadata: dict[str, JsonValue] = {
            'status': 'active',
            'config': {
                'database': {
                    'connection': {
                        'pool': {'size': 10, 'timeout': 30},
                        'retry': {'max_attempts': 3, 'backoff': 2.5},
                    },
                },
                'cache': {'enabled': True, 'ttl': 300},
            },
            'user': {'id': 123, 'name': 'Alice Johnson', 'preferences': {'theme': 'dark', 'language': 'en'}},
        }

        result = await store_context(
            thread_id='test_nested_json',
            source='agent',
            text='Test nested metadata storage',
            metadata=complex_metadata,
            ctx=None,
        )

        assert result['success'] is True
        assert 'context_id' in result

        # Retrieve and verify the metadata is preserved
        search_result = await search_context(limit=50, thread_id='test_nested_json', ctx=None)
        assert len(search_result['results']) == 1

        stored_metadata = search_result['results'][0]['metadata']
        assert stored_metadata['status'] == 'active'
        assert stored_metadata['config']['database']['connection']['pool']['size'] == 10
        assert stored_metadata['config']['database']['connection']['pool']['timeout'] == 30
        assert stored_metadata['config']['database']['connection']['retry']['max_attempts'] == 3
        assert stored_metadata['config']['database']['connection']['retry']['backoff'] == 2.5
        assert stored_metadata['config']['cache']['enabled'] is True
        assert stored_metadata['user']['preferences']['theme'] == 'dark'
        assert stored_metadata['user']['preferences']['language'] == 'en'

    @pytest.mark.asyncio
    async def test_store_arrays_in_metadata(self) -> None:
        """Test storing arrays in metadata."""
        metadata_with_arrays: dict[str, JsonValue] = {
            'tags': ['urgent', 'backend', 'production'],
            'priority_levels': [1, 2, 3, 4, 5],
            'mixed_array': ['string', 42, math.pi, True, None],
            'nested_arrays': [[1, 2], [3, 4], [5, 6]],
        }

        result = await store_context(
            thread_id='test_arrays',
            source='agent',
            text='Test array metadata',
            metadata=metadata_with_arrays,
            ctx=None,
        )

        assert result['success'] is True

        # Retrieve and verify arrays are preserved
        search_result = await search_context(limit=50, thread_id='test_arrays', ctx=None)
        stored_metadata = search_result['results'][0]['metadata']

        assert stored_metadata['tags'] == ['urgent', 'backend', 'production']
        assert stored_metadata['priority_levels'] == [1, 2, 3, 4, 5]
        assert stored_metadata['mixed_array'] == ['string', 42, math.pi, True, None]
        assert stored_metadata['nested_arrays'] == [[1, 2], [3, 4], [5, 6]]

    @pytest.mark.asyncio
    async def test_query_nested_paths(self) -> None:
        """Test querying nested JSON paths."""
        # Store multiple entries with nested metadata
        await store_context(
            thread_id='test_nested_paths',
            source='agent',
            text='Entry 1',
            metadata={'user': {'preferences': {'theme': 'dark', 'notifications': {'email': True}}}},
            ctx=None,
        )

        await store_context(
            thread_id='test_nested_paths',
            source='agent',
            text='Entry 2',
            metadata={'user': {'preferences': {'theme': 'light', 'notifications': {'email': False}}}},
            ctx=None,
        )

        # Query using nested path
        result = await search_context(
            limit=50,
            thread_id='test_nested_paths',
            metadata={'user.preferences.theme': 'dark'},
            ctx=None,
        )

        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'Entry 1'
        assert result['results'][0]['metadata']['user']['preferences']['theme'] == 'dark'

    @pytest.mark.asyncio
    async def test_complex_nested_structure(self) -> None:
        """Test very complex nested structure with multiple levels."""
        complex_structure: dict[str, JsonValue] = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'value': 'deeply_nested',
                            'number': 42,
                            'array': [1, 2, 3],
                            'object': {'key': 'value'},
                        },
                    },
                },
            },
            'metrics': {
                'cpu': 45.5,
                'memory': 512,
                'disk': {'used': 80.5, 'total': 100.0, 'partitions': ['/dev/sda1', '/dev/sda2']},
            },
            'features': {
                'enabled': ['feature_a', 'feature_b', 'feature_c'],
                'disabled': [],
                'experimental': {'count': 3, 'names': ['exp_1', 'exp_2', 'exp_3']},
            },
        }

        result = await store_context(
            thread_id='test_complex',
            source='agent',
            text='Complex nested structure test',
            metadata=complex_structure,
            ctx=None,
        )

        assert result['success'] is True

        # Verify structure is preserved
        search_result = await search_context(limit=50, thread_id='test_complex', ctx=None)
        stored_metadata = search_result['results'][0]['metadata']

        # Verify deep nesting
        assert stored_metadata['level1']['level2']['level3']['level4']['value'] == 'deeply_nested'
        assert stored_metadata['level1']['level2']['level3']['level4']['number'] == 42
        assert stored_metadata['level1']['level2']['level3']['level4']['array'] == [1, 2, 3]
        assert stored_metadata['level1']['level2']['level3']['level4']['object']['key'] == 'value'

        # Verify metrics
        assert stored_metadata['metrics']['cpu'] == 45.5
        assert stored_metadata['metrics']['disk']['used'] == 80.5
        assert stored_metadata['metrics']['disk']['partitions'] == ['/dev/sda1', '/dev/sda2']

        # Verify features
        assert stored_metadata['features']['enabled'] == ['feature_a', 'feature_b', 'feature_c']
        assert stored_metadata['features']['disabled'] == []
        assert stored_metadata['features']['experimental']['count'] == 3

    @pytest.mark.asyncio
    async def test_mixed_flat_and_nested(self) -> None:
        """Test mixing flat and nested metadata structures."""
        mixed_metadata: dict[str, JsonValue] = {
            'simple_string': 'value',
            'simple_int': 42,
            'simple_bool': True,
            'nested': {'level1': {'level2': 'deep_value'}},
            'array': [1, 2, 3],
        }

        result = await store_context(
            thread_id='test_mixed',
            source='agent',
            text='Mixed flat and nested',
            metadata=mixed_metadata,
            ctx=None,
        )

        assert result['success'] is True

        # Query using both flat and nested paths
        search_result = await search_context(
            limit=50, thread_id='test_mixed', metadata={'simple_string': 'value'}, ctx=None,
        )
        assert len(search_result['results']) == 1

        # Verify all types are preserved
        stored_metadata = search_result['results'][0]['metadata']
        assert stored_metadata['simple_string'] == 'value'
        assert stored_metadata['simple_int'] == 42
        assert stored_metadata['simple_bool'] is True
        assert stored_metadata['nested']['level1']['level2'] == 'deep_value'
        assert stored_metadata['array'] == [1, 2, 3]


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestArrayContainsOperator:
    """Tests for the ARRAY_CONTAINS operator."""

    test_thread_id: str

    async def _setup_test_data(self) -> None:
        """Set up test data with array metadata fields."""
        self.test_thread_id = f'test_array_contains_{int(time.time() * 1000)}'

        # Entry with string array
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='Python and FastAPI project',
            metadata={
                'technologies': ['python', 'fastapi', 'postgresql'],
                'tags': ['backend', 'api', 'production'],
            },
            ctx=None,
        )

        # Entry with different technologies
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='JavaScript frontend',
            metadata={
                'technologies': ['javascript', 'react', 'typescript'],
                'tags': ['frontend', 'ui'],
            },
            ctx=None,
        )

        # Entry with numeric array
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='Priority levels test',
            metadata={
                'priority_levels': [1, 3, 5, 7, 9],
                'scores': [85.5, 90.0, 78.3],
            },
            ctx=None,
        )

        # Entry with nested array
        await store_context(
            thread_id=self.test_thread_id,
            source='agent',
            text='Nested references',
            metadata={
                'references': {
                    'context_ids': [100, 200, 300],
                    'youtrack': ['AI-100', 'AI-200'],
                },
            },
            ctx=None,
        )

    @pytest.mark.asyncio
    async def test_array_contains_string_value(self) -> None:
        """Test array_contains with string value."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'python'},
            ],
            ctx=None,
        )

        assert len(result['results']) == 1
        assert 'Python and FastAPI' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_case_insensitive(self) -> None:
        """Test array_contains with case-insensitive string matching."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'PYTHON', 'case_sensitive': False},
            ],
            ctx=None,
        )

        assert len(result['results']) == 1
        assert 'Python and FastAPI' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_case_sensitive_no_match(self) -> None:
        """Test array_contains with case-sensitive string (no match expected)."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'PYTHON', 'case_sensitive': True},
            ],
            ctx=None,
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_integer_value(self) -> None:
        """Test array_contains with integer value."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'priority_levels', 'operator': 'array_contains', 'value': 5},
            ],
            ctx=None,
        )

        assert len(result['results']) == 1
        assert 'Priority levels' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_float_value(self) -> None:
        """Test array_contains with float value."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'scores', 'operator': 'array_contains', 'value': 90.0},
            ],
            ctx=None,
        )

        assert len(result['results']) == 1
        assert 'Priority levels' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_nested_path(self) -> None:
        """Test array_contains with nested JSON path."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'references.context_ids', 'operator': 'array_contains', 'value': 200},
            ],
            ctx=None,
        )

        assert len(result['results']) == 1
        assert 'Nested references' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_nested_string_array(self) -> None:
        """Test array_contains with nested string array."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'references.youtrack', 'operator': 'array_contains', 'value': 'AI-100'},
            ],
            ctx=None,
        )

        assert len(result['results']) == 1
        assert 'Nested references' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_no_match(self) -> None:
        """Test array_contains returns empty when element not found."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'rust'},
            ],
            ctx=None,
        )

        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_combined_with_other_filters(self) -> None:
        """Test array_contains combined with other metadata filters."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'python'},
                {'key': 'tags', 'operator': 'array_contains', 'value': 'production'},
            ],
            ctx=None,
        )

        assert len(result['results']) == 1
        assert 'Python and FastAPI' in result['results'][0]['text_content']

    @pytest.mark.asyncio
    async def test_array_contains_non_existent_field_returns_empty(self) -> None:
        """Test array_contains on non-existent field returns empty (graceful handling)."""
        await self._setup_test_data()

        result = await search_context(
            thread_id=self.test_thread_id,
            metadata_filters=[
                {'key': 'nonexistent', 'operator': 'array_contains', 'value': 'test'},
            ],
            ctx=None,
        )

        # Should return empty, not error
        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_scalar_field_returns_empty(self) -> None:
        """Test array_contains on scalar string field returns empty (graceful handling, not error).

        Regression test: PostgreSQL jsonb_array_elements_text() throws
        "cannot extract elements from a scalar" on non-array fields.
        The documented behavior is to return empty results gracefully.
        """
        test_thread_id = f'test_array_contains_scalar_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with scalar category',
            metadata={
                'category': 'backend',  # Scalar string, NOT an array
                'technologies': ['python', 'fastapi'],  # This IS an array
            },
            ctx=None,
        )

        # This should return empty results, NOT throw an error
        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'category', 'operator': 'array_contains', 'value': 'backend'},
            ],
            ctx=None,
        )

        # Should return empty results, not error
        assert 'results' in result
        assert len(result['results']) == 0

        # Verify the array field still works correctly
        result2 = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'technologies', 'operator': 'array_contains', 'value': 'python'},
            ],
            ctx=None,
        )
        assert len(result2['results']) == 1

    @pytest.mark.asyncio
    async def test_array_contains_object_field_returns_empty(self) -> None:
        """Test array_contains on object field returns empty (graceful handling)."""
        test_thread_id = f'test_array_contains_object_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with object config field',
            metadata={
                'config': {'timeout': 30, 'retries': 3},  # Object, NOT an array
            },
            ctx=None,
        )

        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'config', 'operator': 'array_contains', 'value': 30},
            ],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_number_field_returns_empty(self) -> None:
        """Test array_contains on number field returns empty (graceful handling)."""
        test_thread_id = f'test_array_contains_number_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with number priority field',
            metadata={
                'priority': 5,  # Number scalar, NOT an array
            },
            ctx=None,
        )

        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'priority', 'operator': 'array_contains', 'value': 5},
            ],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 0

    @pytest.mark.asyncio
    async def test_array_contains_null_field_returns_empty(self) -> None:
        """Test array_contains on null field returns empty (graceful handling)."""
        test_thread_id = f'test_array_contains_null_{int(time.time() * 1000)}'
        await store_context(
            thread_id=test_thread_id,
            source='agent',
            text='Entry with null field',
            metadata={
                'tags': None,  # Explicit null, NOT an array
            },
            ctx=None,
        )

        result = await search_context(
            thread_id=test_thread_id,
            metadata_filters=[
                {'key': 'tags', 'operator': 'array_contains', 'value': 'test'},
            ],
            ctx=None,
        )

        assert 'results' in result
        assert len(result['results']) == 0


class TestArrayContainsValidation:
    """Tests for ARRAY_CONTAINS operator validation."""

    def test_array_contains_rejects_list_value(self) -> None:
        """Test that array_contains rejects list values."""
        with pytest.raises(ValueError, match='requires a single value'):
            MetadataFilter(
                key='technologies',
                operator=MetadataOperator.ARRAY_CONTAINS,
                value=['python', 'fastapi'],
            )

    def test_array_contains_rejects_none_value(self) -> None:
        """Test that array_contains rejects None value."""
        with pytest.raises(ValueError, match='requires a non-null value'):
            MetadataFilter(
                key='technologies',
                operator=MetadataOperator.ARRAY_CONTAINS,
                value=None,
            )

    def test_array_contains_accepts_string_value(self) -> None:
        """Test that array_contains accepts string value."""
        f = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='python',
        )
        assert f.value == 'python'

    def test_array_contains_accepts_integer_value(self) -> None:
        """Test that array_contains accepts integer value."""
        f = MetadataFilter(
            key='priority_levels',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=5,
        )
        assert f.value == 5

    def test_array_contains_accepts_float_value(self) -> None:
        """Test that array_contains accepts float value."""
        f = MetadataFilter(
            key='scores',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=85.5,
        )
        assert f.value == 85.5

    def test_array_contains_accepts_boolean_value(self) -> None:
        """Test that array_contains accepts boolean value."""
        f = MetadataFilter(
            key='flags',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=True,
        )
        assert f.value is True


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
        assert '{references,context_ids}' in where_clause
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
        assert "jsonb_typeof(metadata#>'{references,context_ids}') = 'array'" in where_clause
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
        assert "jsonb_typeof(metadata#>'{references,youtrack}') = 'array'" in where_clause
        assert 'jsonb_array_elements(' in where_clause  # iterate as jsonb (string-typed only), not _text
        assert "jsonb_typeof(elem) = 'string'" in where_clause
        assert 'CASE WHEN' in where_clause
        assert 'ELSE FALSE END' in where_clause
        assert 'translate' in where_clause  # ASCII-only ci fold (parity with SQLite LOWER)


class TestOutOfInt64FilterRejection:
    """Integer filter values outside the signed 64-bit range are rejected
    uniformly on BOTH backends.

    SQLite binds a Python int as a 64-bit column value and raises OverflowError for
    anything outside [-2**63, 2**63-1], aborting the whole search, while PostgreSQL
    binds it into an arbitrary-precision NUMERIC and matches -- a cross-backend
    divergence (and, for IN/NOT_IN, a v3.0.0 regression from the baseline str()-bind).
    The validator forbids the divergent case on both backends instead.
    """

    @pytest.mark.parametrize('value', [10**20, 2**63, -(2**63) - 1, 99999999999999999999999999])
    def test_scalar_out_of_int64_rejected(self, value: int) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.EQ, value=value)
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.GT, value=value)

    def test_list_member_out_of_int64_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.IN, value=[7, 10**20])
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.NOT_IN, value=[10**20])

    def test_array_contains_out_of_int64_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.ARRAY_CONTAINS, value=2**63)

    @pytest.mark.parametrize('value', [2**63 - 1, -(2**63), 0, 9999, True, False])
    def test_in_range_and_bool_accepted(self, value: int | bool) -> None:
        # bool is an int subclass but always in range; in-range ints (incl. the
        # int64 boundaries) are accepted unchanged.
        MetadataFilter(key='v', operator=MetadataOperator.EQ, value=value)

    @pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
    @pytest.mark.parametrize('value', [10**20, 2**63, -(2**63) - 1])
    def test_simple_filter_out_of_int64_rejected(self, backend_type: str, value: int) -> None:
        # The simple metadata={} equality path goes through add_simple_filter, which
        # bypasses MetadataFilter validation; it must reject an out-of-int64 integer
        # the same way on BOTH backends (ValueError, before any backend-specific bind)
        # instead of aborting the search on SQLite (OverflowError) while PostgreSQL
        # matches. The guard fires before the backend branch, so a single raise covers
        # both backend_type values.
        builder = MetadataQueryBuilder(backend_type=backend_type)
        with pytest.raises(ValueError, match='64-bit'):
            builder.add_simple_filter('k', value)

    @pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
    @pytest.mark.parametrize('value', [2**63 - 1, -(2**63), 0, 9999, True, False, 'x'])
    def test_simple_filter_in_range_and_non_int_accepted(
        self, backend_type: str, value: str | int | bool,
    ) -> None:
        # In-range ints (incl. the int64 boundaries), bools, and strings pass the
        # guard and build a condition normally on both backends.
        builder = MetadataQueryBuilder(backend_type=backend_type)
        builder.add_simple_filter('k', value)
        assert builder.conditions  # a condition was built, no rejection


class TestNotInNumericNonNumberParity:
    """NOT_IN with numeric members keeps a present non-number row identically on both backends.

    The PostgreSQL numeric disjunct guards on ``jsonb_typeof = 'number'`` (mirroring
    SQLite's ``json_type`` guard) so a present non-number value yields a deterministic
    FALSE match. Without the guard the numeric accessor's ``CASE ... ELSE NULL`` made
    the match NULL, and under NOT_IN's ``present AND NOT match`` the row was silently
    dropped on PostgreSQL (``NOT NULL`` -> NULL) while SQLite kept it (``NOT FALSE`` ->
    TRUE) -- a v3.0.0 regression from the baseline str()-bind that compared every member
    as text. Positive IN never diverged (NULL is falsy in a WHERE clause); only NOT_IN's
    negation exposed the NULL-vs-FALSE asymmetry.
    """

    def test_postgresql_not_in_numeric_has_typeof_number_guard(self) -> None:
        """The PG NOT_IN numeric disjunct is wrapped in an explicit jsonb_typeof='number' guard."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_advanced_filter(
            MetadataFilter(key='k', operator=MetadataOperator.NOT_IN, value=[1, 2]),
        )
        where_clause, _ = builder.build_where_clause()
        assert "jsonb_typeof(metadata->'k') = 'number'" in where_clause
        # NOT_IN is the negation of the match under an explicit presence guard.
        assert 'IS NOT NULL AND NOT' in where_clause

    def test_sqlite_not_in_numeric_keeps_present_nonnumber_rows(self) -> None:
        """End-to-end on in-memory SQLite: NOT_IN [1, 2] over a key keeps present string/boolean
        rows and the non-matching number row, excludes the matching number and the missing key.

        This is the row-result parity the PostgreSQL ``jsonb_typeof='number'`` guard restores:
        before the guard, PostgreSQL dropped the string/boolean rows that SQLite (shown here)
        keeps.
        """
        import json
        import sqlite3

        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(
            MetadataFilter(key='k', operator=MetadataOperator.NOT_IN, value=[1, 2]),
        )
        where_clause, params = builder.build_where_clause()

        db = sqlite3.connect(':memory:')
        db.execute('CREATE TABLE context_entries (id INTEGER PRIMARY KEY, metadata TEXT)')
        db.executemany(
            'INSERT INTO context_entries (id, metadata) VALUES (?, ?)',
            [
                (1, json.dumps({'k': 'abc'})),    # present string  -> kept
                (2, json.dumps({'k': True})),     # present boolean -> kept
                (3, json.dumps({'k': 1})),        # number in [1, 2] -> excluded
                (4, json.dumps({'k': 99})),       # number not in list -> kept
                (5, json.dumps({'other': 'x'})),  # key missing -> excluded by presence guard
            ],
        )
        sql = f'SELECT id FROM context_entries WHERE {where_clause} ORDER BY id'
        kept = [row[0] for row in db.execute(sql, params).fetchall()]
        assert kept == [1, 2, 4]


class TestNulAndSurrogateRejection:
    """A NUL (U+0000) or unpaired UTF-16 surrogate in a string is rejected uniformly.

    Both sequences store and match on SQLite but abort the query on PostgreSQL
    (asyncpg rejects the bind or the jsonb parser rejects the escape), and the
    driver error -- not a ControlFlowError -- charges the circuit breaker. Rejecting
    them at the filter-value guards (reject_nul) and the store/update walker
    (unstorable_string_error) makes both backends fail fast and identically,
    mirroring the reject_non_finite / reject_out_of_int64 parity guards.
    """

    def test_pg_bind_reject_reason_detects_nul_and_surrogate(self) -> None:
        """The low-level predicate flags a NUL and a lone surrogate, passes clean text."""
        from app.metadata_types import pg_bind_reject_reason

        assert pg_bind_reject_reason('clean text') is None
        assert pg_bind_reject_reason('') is None
        nul_reason = pg_bind_reject_reason('a\x00b')
        assert nul_reason is not None
        assert 'NUL' in nul_reason
        surrogate_reason = pg_bind_reject_reason('a\ud800b')
        assert surrogate_reason is not None
        assert 'surrogate' in surrogate_reason

    @pytest.mark.parametrize('bad', ['done\x00', 'x\ud800'])
    def test_reject_nul_filter_value_rejected_both_paths(self, bad: str) -> None:
        """A NUL/surrogate filter value is rejected on the advanced and simple paths.

        The value binds and matches on SQLite but aborts the query and charges the
        circuit breaker on PostgreSQL, so it is rejected on both for parity.
        """
        with pytest.raises(ValueError, match='NUL|surrogate'):
            MetadataFilter(key='status', operator=MetadataOperator.EQ, value=bad)

        with pytest.raises(ValueError, match='NUL|surrogate'):
            MetadataQueryBuilder(backend_type='sqlite').add_simple_filter('status', bad)

        # A NUL/surrogate member inside an IN list is rejected too.
        with pytest.raises(ValueError, match='NUL|surrogate'):
            MetadataFilter(key='status', operator=MetadataOperator.IN, value=['ok', bad])

    def test_reject_nul_allows_clean_values_and_non_strings(self) -> None:
        """reject_nul is a no-op for clean strings, numbers, booleans, None, and clean lists."""
        from app.metadata_types import reject_nul

        reject_nul('clean')
        reject_nul('unicode-é中')
        reject_nul(2.5)
        reject_nul(True)
        reject_nul(None)
        clean_list: list[str | int | float | bool] = ['a', 'b']
        reject_nul(clean_list)

    def test_unstorable_string_error_walks_keys_values_and_lists(self) -> None:
        """The store/update walker catches a NUL/surrogate in a scalar, a value, a KEY, or a list."""
        from app.metadata_types import unstorable_string_error

        assert unstorable_string_error('plain text') is None
        assert unstorable_string_error({'a': 1, 'b': ['ok', {'c': 'fine'}], 'd': True}) is None
        assert unstorable_string_error(['tag-one', 'tag-two']) is None

        assert unstorable_string_error('bad\x00text') is not None
        assert unstorable_string_error('bad\ud800text') is not None
        # NUL in a nested value.
        assert unstorable_string_error({'note': {'deep': 'x\x00y'}}) is not None
        # NUL in a metadata KEY (not only values) -- PostgreSQL jsonb rejects it too.
        key_message = unstorable_string_error({'k\x00ey': 'value'})
        assert key_message is not None
        assert 'key' in key_message
        # NUL in a list member (e.g. a tag list).
        assert unstorable_string_error(['ok', 'ta\x00g']) is not None

    def test_sqlite_binds_nul_string_documenting_the_divergence(self) -> None:
        """SQLite binds and round-trips a NUL-bearing string (the exact divergence guarded).

        PostgreSQL's asyncpg rejects the same bind, so without the guards the two
        backends diverge; this pins the SQLite half of the divergence the guards close.
        """
        import sqlite3

        db = sqlite3.connect(':memory:')
        db.execute('CREATE TABLE t (v TEXT)')
        db.execute('INSERT INTO t (v) VALUES (?)', ('a\x00b',))
        stored = db.execute('SELECT v FROM t').fetchone()[0]
        assert stored == 'a\x00b'


class TestBindParameterBudget:
    """The builder rejects a clause whose accumulated binds exceed the shared budget.

    The per-dimension caps bound each input list individually, but capped dimensions
    multiply (filters times IN-list members), so MAX_METADATA_BIND_PARAMS is the
    aggregate backstop guaranteeing no combination of filter dimensions can grow a
    single statement past the backend's per-statement bind limit. The raised
    ValueError rides the same structured validation channel as every other filter
    validation failure, so it never charges the circuit breaker.
    """

    def test_simple_filter_over_budget_rejected(self) -> None:
        """add_simple_filter raises once the accumulated binds exceed the budget.

        param_offset counts the enclosing statement's earlier placeholders, so
        seeding it just below the budget exercises the overflow without building
        tens of thousands of filters.
        """
        from app.metadata_types import MAX_METADATA_BIND_PARAMS

        builder = MetadataQueryBuilder(backend_type='postgresql', param_offset=MAX_METADATA_BIND_PARAMS)
        with pytest.raises(ValueError, match='bind parameters'):
            builder.add_simple_filter('status', 'active')

    def test_advanced_filter_over_budget_rejected(self) -> None:
        """add_advanced_filter raises once the accumulated binds exceed the budget."""
        from app.metadata_types import MAX_METADATA_BIND_PARAMS

        builder = MetadataQueryBuilder(backend_type='sqlite', param_offset=MAX_METADATA_BIND_PARAMS)
        spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        with pytest.raises(ValueError, match='bind parameters'):
            builder.add_advanced_filter(spec)

    def test_within_budget_accepted(self) -> None:
        """A clause within the budget builds normally (control)."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_simple_filter('status', 'active')
        builder.add_advanced_filter(MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5))
        where_clause, params = builder.build_where_clause()
        assert where_clause
        assert len(params) == 2
