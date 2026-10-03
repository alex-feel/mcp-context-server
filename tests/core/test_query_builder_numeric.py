"""MetadataQueryBuilder numeric semantics: number type guards, exact PostgreSQL int/float comparison, non-finite rejection."""

import pytest

from app.metadata_sql import _FLOAT8_OVERFLOW
from app.metadata_sql import _FLOAT8_TINY
from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


class TestMetadataQueryBuilder:
    """Test the MetadataQueryBuilder class."""

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

        Down-casting a stored integer > 2**53 through DOUBLE PRECISION loses its low bits, and
        folding the param via float8::numeric rounds to ~15 significant digits; either would make
        the same metadata_filter return different rows than SQLite. The builder reads the stored
        value as exact NUMERIC and branches on its integrality, across the advanced EQ path, the
        IN per-member path, and the simple-equality path."""
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
        assert overflow == _FLOAT8_OVERFLOW
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
        tiny = _FLOAT8_TINY
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
        float-member path iterates numeric elements and reuses pg_numeric_compare, so a
        genuinely int-origin element matches on both backends; ints/strings keep @>.
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


class TestNotInNumericNonNumberParity:
    """NOT_IN with numeric members keeps a present non-number row identically on both backends.

    The PostgreSQL numeric disjunct guards on ``jsonb_typeof = 'number'`` (mirroring
    SQLite's ``json_type`` guard) so a present non-number value yields a deterministic
    FALSE match. Without the guard the numeric accessor's ``CASE ... ELSE NULL`` would
    make the match NULL, and under NOT_IN's ``present AND NOT match`` the row would be
    silently dropped on PostgreSQL (``NOT NULL`` -> NULL) while SQLite keeps it
    (``NOT FALSE`` -> TRUE). Positive IN does not diverge (NULL is falsy in a WHERE
    clause); only NOT_IN's negation exposes the NULL-vs-FALSE asymmetry.
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

        This is the row-result parity the PostgreSQL ``jsonb_typeof='number'`` guard provides:
        without it, PostgreSQL would drop the string/boolean rows that SQLite (shown here) keeps.
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
