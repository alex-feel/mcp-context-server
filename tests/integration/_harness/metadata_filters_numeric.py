"""Real-server checks for numeric and boolean metadata filter parity.

Numeric operators match JSON numbers only and boolean operators JSON booleans
only; non-dyadic floats, integers above 2**53 filtered by a float, floats
stored above 2**53, and numbers beyond the int64 and float8 ranges filter to
the same counts on SQLite and PostgreSQL.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class MetadataFiltersNumericMixin(HarnessCore):
    """Checks for numeric and boolean metadata filter parity across backends."""

    async def test_metadata_filter_numeric_type_parity(self) -> bool:
        """Numeric operators match JSON-number-typed values only -- identically on
        both backends.

        SQLite's ``CAST(text AS NUMERIC)`` coerces a JSON boolean to 1/0 and a
        numeric-prefix string like ``'12abc'`` to 12, while a bare PostgreSQL
        ``::NUMERIC`` either aborts or diverges on those same values. Numeric
        EQ/NE/GT/GTE/LT/LTE are therefore number-typed-only on BOTH
        backends (SQLite ``json_type(...) IN ('integer','real')`` / PostgreSQL
        ``jsonb_typeof(...) = 'number'``), so a non-number value -- text, boolean,
        json-null, or an absent key -- never matches any numeric operator. Running the
        SAME expected counts on both backends proves parity by construction.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter Numeric Type Parity'
        print('Testing numeric-type metadata-filter parity...')
        thread = f'{self.test_thread_id}_num_parity'
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'num seven', 'metadata': {'score': 7}},
            {'thread_id': thread, 'source': 'agent', 'text': 'num five', 'metadata': {'score': 5}},
            {'thread_id': thread, 'source': 'agent', 'text': 'num three-five', 'metadata': {'score': 3.5}},
            {'thread_id': thread, 'source': 'agent', 'text': 'str prefix', 'metadata': {'score': '12abc'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'str plain', 'metadata': {'score': 'abc'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'bool true', 'metadata': {'score': True}},
            {'thread_id': thread, 'source': 'agent', 'text': 'missing key', 'metadata': {'other': 1}},
        ]

        async def _count(filters: list[dict[str, Any]]) -> int:
            assert self.client is not None
            res = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': filters},
            )
            return len(self._extract_content(res).get('results', []))

        try:
            assert self.client is not None  # Type guard for Pyright
            for entry in entries:
                result = await self.client.call_tool('store_context', entry)
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store numeric-parity entries'))
                    return False

            # Only the three JSON numbers (7, 5, 3.5) ever participate; the
            # numeric-prefix string '12abc', the plain string 'abc', the boolean,
            # and the missing-key row never match any numeric operator on either backend.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('eq 5', [{'key': 'score', 'operator': 'eq', 'value': 5}], 1),       # {5}
                ('ne 5', [{'key': 'score', 'operator': 'ne', 'value': 5}], 2),       # {7, 3.5}
                ('gt 5', [{'key': 'score', 'operator': 'gt', 'value': 5}], 1),       # {7}
                ('gte 5', [{'key': 'score', 'operator': 'gte', 'value': 5}], 2),     # {7, 5}
                ('lt 5', [{'key': 'score', 'operator': 'lt', 'value': 5}], 1),       # {3.5}
                ('lte 5', [{'key': 'score', 'operator': 'lte', 'value': 5}], 2),     # {5, 3.5}
                ('gt 0', [{'key': 'score', 'operator': 'gt', 'value': 0}], 3),       # {7, 5, 3.5}; NOT bool/'12abc'
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'numeric-parity {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All numeric-type metadata-filter parity tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_float_precision_parity(self) -> bool:
        """Numeric EQ/NE/comparison on non-dyadic floats agree across backends.

        A Python float bound in a PostgreSQL ``::NUMERIC`` context encodes as its FULL
        Decimal expansion (0.3 -> 0.2999999999999999888...), so a plain
        ``'0.3'::NUMERIC = $1`` is False (and ``> 0.3`` on a stored 0.3 wrongly True) while
        SQLite -- comparing the bit-identical IEEE double on both sides -- matches.
        ``pg_numeric_body`` in ``app/metadata_sql.py`` therefore branches on the STORED
        value: a provably int-origin integral stored value compares exact ``NUMERIC``
        against the param (see the high-magnitude round-trip check), and a fractional stored
        value compares double-vs-double via ``(stored)::float8 <op> (<ph>)::float8``. It
        deliberately does NOT fold the param via
        ``(<ph>)::float8::numeric`` -- that cast rounds to ~15 significant digits and collapses
        large integers, and the unit tests assert the pattern is absent from the generated SQL.
        Uses non-dyadic fractions (0.1, 0.3) that are NOT exactly representable, which the
        integer/3.5 dataset misses.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter Float Precision Parity'
        print('Testing non-dyadic float metadata-filter parity...')
        thread = f'{self.test_thread_id}_float_parity'
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'f 0.3', 'metadata': {'score': 0.3}},
            {'thread_id': thread, 'source': 'agent', 'text': 'f 0.1', 'metadata': {'score': 0.1}},
            {'thread_id': thread, 'source': 'agent', 'text': 'f 0.5', 'metadata': {'score': 0.5}},
            {'thread_id': thread, 'source': 'agent', 'text': 'f 7', 'metadata': {'score': 7}},
        ]

        async def _count(filters: list[dict[str, Any]]) -> int:
            assert self.client is not None
            res = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': filters},
            )
            return len(self._extract_content(res).get('results', []))

        try:
            assert self.client is not None  # Type guard for Pyright
            for entry in entries:
                result = await self.client.call_tool('store_context', entry)
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store float-parity entries'))
                    return False

            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('eq 0.3', [{'key': 'score', 'operator': 'eq', 'value': 0.3}], 1),    # {0.3}
                ('eq 0.1', [{'key': 'score', 'operator': 'eq', 'value': 0.1}], 1),    # {0.1}
                ('ne 0.3', [{'key': 'score', 'operator': 'ne', 'value': 0.3}], 3),    # {0.1, 0.5, 7}
                ('gt 0.3', [{'key': 'score', 'operator': 'gt', 'value': 0.3}], 2),    # {0.5, 7}; 0.3 excluded
                ('gte 0.3', [{'key': 'score', 'operator': 'gte', 'value': 0.3}], 3),  # {0.3, 0.5, 7}
                ('lt 0.5', [{'key': 'score', 'operator': 'lt', 'value': 0.5}], 2),    # {0.3, 0.1}
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'float-parity {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All non-dyadic float metadata-filter parity tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_high_magnitude_int_float_param_parity(self) -> bool:
        """A high-magnitude stored integer (> 2**53) filtered by a FLOAT param agrees across backends.

        Casting the STORED value through ``DOUBLE PRECISION`` for a float param would drop the low
        bits of a stored integer like 2**53+1 on PostgreSQL (folding it to 2**53), so ``eq 2**53.0``
        would match on PostgreSQL but NOT on SQLite (which compares the exact integer), and ``gt``
        would be the mirror image. ``pg_numeric_body`` therefore keeps a provably int-origin stored
        integer such as 2**53+1 (one NOT equal to ``(stored::float8)::text::NUMERIC``) on the exact
        ``NUMERIC`` branch against the float param, so it is never truncated, and casts only a
        canonical-double-form value such as 2**53-1, which a double represents exactly, for a
        double-vs-double compare. Running the SAME expected counts on both backends proves parity by
        construction; a stored-side DOUBLE PRECISION cast gives PostgreSQL different counts than
        SQLite for eq/gt/in here.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter High-Magnitude Int Float-Param Parity'
        print('Testing high-magnitude-int vs float-param metadata-filter parity...')
        thread = f'{self.test_thread_id}_bigint_parity'
        # BIG = 2**53 + 1 (NOT representable as a double -> would fold to 2**53 under a stored-side
        # DOUBLE PRECISION cast); SMALL = 2**53 - 1 (exactly representable); float param = 2**53.
        big = 9007199254740993
        small = 9007199254740991
        fparam = 9007199254740992.0
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'big 2**53+1', 'metadata': {'n': big}},
            {'thread_id': thread, 'source': 'agent', 'text': 'small 2**53-1', 'metadata': {'n': small}},
        ]

        async def _count(filters: list[dict[str, Any]]) -> int:
            assert self.client is not None
            res = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': filters},
            )
            return len(self._extract_content(res).get('results', []))

        try:
            assert self.client is not None  # Type guard for Pyright
            for entry in entries:
                result = await self.client.call_tool('store_context', entry)
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store high-magnitude-int entries'))
                    return False

            # SQLite (exact int-vs-real) defines the expected counts, and PostgreSQL must match.
            # A stored-side DOUBLE PRECISION cast would make PG 'eq fparam' match BIG (1) and
            # 'gt fparam' miss it (0) -- the opposite of SQLite.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('eq float 2**53', [{'key': 'n', 'operator': 'eq', 'value': fparam}], 0),    # neither equals 2**53
                ('gt float 2**53', [{'key': 'n', 'operator': 'gt', 'value': fparam}], 1),    # {BIG}
                ('gte float 2**53', [{'key': 'n', 'operator': 'gte', 'value': fparam}], 1),  # {BIG}
                ('lt float 2**53', [{'key': 'n', 'operator': 'lt', 'value': fparam}], 1),    # {SMALL}
                ('lte float 2**53', [{'key': 'n', 'operator': 'lte', 'value': fparam}], 1),  # {SMALL}
                ('in float 2**53', [{'key': 'n', 'operator': 'in', 'value': [fparam]}], 0),  # neither equals 2**53
                ('eq int BIG', [{'key': 'n', 'operator': 'eq', 'value': big}], 1),           # int param stays exact
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'bigint-parity {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All high-magnitude-int float-param metadata-filter parity tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_high_magnitude_float_roundtrip_parity(self) -> bool:
        """A high-magnitude stored FLOAT (> 2**53) filtered by the same float agrees across backends.

        A Python float stores as its ``repr`` -- the SHORTEST decimal that round-trips -- so
        above 2**53 the jsonb NUMERIC differs from the double's exact value (``float(2**55)``
        stores as 36028797018963970 while the double is ...968). asyncpg binds a float param as
        its EXACT double expansion, so an exact-NUMERIC compare of every integral stored value
        would test 36028797018963970 = 36028797018963968 and fail ``eq`` against the very value
        the user stored, while SQLite (double vs double) matches. ``pg_numeric_body`` therefore
        keeps the exact compare only for provably int-origin stored values -- those NOT equal to
        ``(stored::float8)::text::NUMERIC`` -- and compares canonical-double-form values
        double-vs-double. A stored INT equal to the double's exact value (2**55 itself,
        non-canonical) stays on the exact branch and still matches the float param exactly.
        Running the SAME expected counts on both backends proves parity; an exact branch for
        every integral value gives PostgreSQL different counts than SQLite for every check
        with a float-stored value.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter High-Magnitude Float Roundtrip Parity'
        print('Testing high-magnitude-float roundtrip metadata-filter parity...')
        thread = f'{self.test_thread_id}_bigfloat_parity'
        # fstored = float(2**55): exactly 36028797018963968.0, whose shortest repr
        # ('3.602879701896397e+16') parses to jsonb NUMERIC 36028797018963970.
        # istored = 2**55 as an exact int. tsfloat = a nanosecond-epoch-scale float
        # (a realistic shape for a stored timestamp).
        fstored = float(2 ** 55)
        istored = 2 ** 55
        tsfloat = float(1719939368123456789)
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'float 2**55', 'metadata': {'v': fstored}},
            {'thread_id': thread, 'source': 'agent', 'text': 'int 2**55', 'metadata': {'v': istored}},
            {'thread_id': thread, 'source': 'agent', 'text': 'ns-epoch float', 'metadata': {'v': tsfloat}},
        ]

        async def _count(filters: list[dict[str, Any]]) -> int:
            assert self.client is not None
            res = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': filters},
            )
            return len(self._extract_content(res).get('results', []))

        try:
            assert self.client is not None  # Type guard for Pyright
            for entry in entries:
                result = await self.client.call_tool('store_context', entry)
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store high-magnitude-float entries'))
                    return False

            # SQLite (REAL vs REAL / INTEGER vs REAL exact) defines the expected counts,
            # and PostgreSQL must match. An exact branch for every integral value would
            # make PG 'eq fstored' miss the float-stored row (exact 36028797018963970 !=
            # ...968) and 'gt fstored' match it. NOTE: an INT param against the
            # float-stored row is the documented irreducible residual (jsonb keeps no
            # provenance) and is deliberately NOT asserted here.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('eq float 2**55', [{'key': 'v', 'operator': 'eq', 'value': fstored}], 2),   # {float, int}
                ('gt float 2**55', [{'key': 'v', 'operator': 'gt', 'value': fstored}], 1),   # {ns-epoch}
                ('gte float 2**55', [{'key': 'v', 'operator': 'gte', 'value': fstored}], 3),
                ('lt float 2**55', [{'key': 'v', 'operator': 'lt', 'value': fstored}], 0),
                ('ne float 2**55', [{'key': 'v', 'operator': 'ne', 'value': fstored}], 1),   # {ns-epoch}
                ('in [float 2**55]', [{'key': 'v', 'operator': 'in', 'value': [fstored]}], 2),
                ('eq ns-epoch float', [{'key': 'v', 'operator': 'eq', 'value': tsfloat}], 1),
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'bigfloat-parity {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All high-magnitude-float roundtrip metadata-filter parity tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_out_of_range_numeric_parity(self) -> bool:
        """Out-of-float8 and out-of-int64 stored numbers filter identically on both backends.

        The float discriminator guards two boundaries: (1) a stored integer beyond float8
        range (309+ digits, legal through jsonb) would make the exact-form probe's float8
        cast raise 22003 and abort the WHOLE PostgreSQL query while SQLite answers normally;
        (2) a stored integer beyond int64 must not be classified int-origin (exact NUMERIC),
        because SQLite reads it as REAL and the same float filter would diverge. The nested
        CASE guards the int64 boundary and maps out-of-float8 magnitudes to +/-inf (mirroring
        SQLite's REAL read), so the same expected counts hold on both backends by
        construction; without those guards PostgreSQL either errors or returns different counts.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter Out-of-Range Numeric Parity'
        print('Testing out-of-float8 / out-of-int64 numeric metadata-filter parity...')
        thread = f'{self.test_thread_id}_outofrange_parity'
        # huge = 10**400 (beyond DBL_MAX -> SQLite REAL inf); u64 = 2**64+1 (beyond int64,
        # SQLite reads as REAL 2**64); small = 5 (a plain in-range integer).
        huge = 10 ** 400
        u64 = 2 ** 64 + 1
        f64 = float(2 ** 64)
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'huge int', 'metadata': {'v': huge}},
            {'thread_id': thread, 'source': 'agent', 'text': 'u64+1 int', 'metadata': {'v': u64}},
            {'thread_id': thread, 'source': 'agent', 'text': 'small int', 'metadata': {'v': 5}},
        ]

        async def _count(filters: list[dict[str, Any]]) -> int:
            assert self.client is not None
            res = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': filters},
            )
            return len(self._extract_content(res).get('results', []))

        try:
            assert self.client is not None  # Type guard for Pyright
            for entry in entries:
                result = await self.client.call_tool('store_context', entry)
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store out-of-range entries'))
                    return False

            # SQLite (REAL inf for huge, REAL 2**64 for u64) defines the expected counts;
            # PostgreSQL must match instead of aborting / diverging. A float
            # param is used throughout so the discriminator's float branch is exercised.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('gt 5.0 float', [{'key': 'v', 'operator': 'gt', 'value': 5.0}], 2),   # {huge, u64}
                ('eq 5.0 float', [{'key': 'v', 'operator': 'eq', 'value': 5.0}], 1),   # {5}
                ('lt 5.0 float', [{'key': 'v', 'operator': 'lt', 'value': 5.0}], 0),   # none
                ('eq f64', [{'key': 'v', 'operator': 'eq', 'value': f64}], 1),         # {u64}=2**64
                ('gt f64', [{'key': 'v', 'operator': 'gt', 'value': f64}], 1),         # {huge}; u64 rounds to 2**64
                ('lt f64', [{'key': 'v', 'operator': 'lt', 'value': f64}], 1),         # {5}
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'out-of-range parity {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All out-of-range numeric metadata-filter parity tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_boolean_type_parity(self) -> bool:
        """Boolean EQ/NE match JSON-boolean-typed values only -- identically on both backends.

        Unguarded, SQLite binds a boolean as 0/1 and compares the typed ``json_extract`` (so
        a stored numeric 0/1 would match), while PostgreSQL compares ``->>`` text
        'true'/'false' (so a stored string 'true'/'false' would match). Both backends are
        therefore guarded -- SQLite ``json_type(...) IN ('true','false')``
        and PostgreSQL ``jsonb_typeof(...) = 'boolean'`` -- so a numeric 0/1 or a string
        'true'/'false' never matches a boolean operator on either backend, mirroring the
        number-only contract.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter Boolean Type Parity'
        print('Testing boolean-type metadata-filter parity...')
        thread = f'{self.test_thread_id}_bool_parity'
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'b true', 'metadata': {'flag': True}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b false', 'metadata': {'flag': False}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b num1', 'metadata': {'flag': 1}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b num0', 'metadata': {'flag': 0}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b strtrue', 'metadata': {'flag': 'true'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b strfalse', 'metadata': {'flag': 'false'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b missing', 'metadata': {'other': 1}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b arr bool', 'metadata': {'tags': [True, 2]}},
            {'thread_id': thread, 'source': 'agent', 'text': 'b arr int', 'metadata': {'tags': [1, 2]}},
        ]

        async def _count(filters: list[dict[str, Any]]) -> int:
            assert self.client is not None
            res = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': filters},
            )
            return len(self._extract_content(res).get('results', []))

        try:
            assert self.client is not None  # Type guard for Pyright
            for entry in entries:
                result = await self.client.call_tool('store_context', entry)
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store boolean-parity entries'))
                    return False

            # Only the two JSON booleans participate; numeric 0/1, string 'true', and the
            # missing-key row never match a boolean operator (eq/ne/in/not_in/array_contains)
            # on either backend, which keeps mixed-type metadata from colliding across backends.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('eq true', [{'key': 'flag', 'operator': 'eq', 'value': True}], 1),    # {true}
                ('eq false', [{'key': 'flag', 'operator': 'eq', 'value': False}], 1),  # {false}
                ('ne true', [{'key': 'flag', 'operator': 'ne', 'value': True}], 1),    # {false}
                ('ne false', [{'key': 'flag', 'operator': 'ne', 'value': False}], 1),  # {true}
                ('in [true]', [{'key': 'flag', 'operator': 'in', 'value': [True]}], 1),  # {true}; not num1/str'true'
                # present non-(JSON true): {false(bool), num1, num0, strtrue, strfalse}; missing/array excluded
                ('not_in [true]', [{'key': 'flag', 'operator': 'not_in', 'value': [True]}], 5),
                # SYMMETRIC direction: a numeric/string member must NOT match the stored JSON
                # boolean (SQLite renders bool true as '1', PostgreSQL renders it as 'true').
                ('in [1] (not bool)', [{'key': 'flag', 'operator': 'in', 'value': [1]}], 1),       # {num1} only
                ('in ["true"] (not bool)', [{'key': 'flag', 'operator': 'in', 'value': ['true']}], 1),  # {strtrue} only
                ('array_contains true', [{'key': 'tags', 'operator': 'array_contains', 'value': True}], 1),  # {[true,2]}
                ('array_contains 2', [{'key': 'tags', 'operator': 'array_contains', 'value': 2}], 2),        # {[true,2],[1,2]}
                ('array_contains 1 (not bool)', [{'key': 'tags', 'operator': 'array_contains', 'value': 1}], 1),  # {[1,2]}
                # case-insensitive STRING member matches ONLY JSON-string array elements
                # (the type-aware string-only contract): the numeric int 1 element
                # is NOT text-matched and there is no string "1" element, so the result is 0 on
                # BOTH backends -- eliminating the prior divergence where a string member matched
                # a numeric element via SQLite's double-render of out-of-int64 / high-precision ints.
                ('array_contains "1" (ci string-only, no string elem)',
                 [{'key': 'tags', 'operator': 'array_contains', 'value': '1'}], 0),
                # STRING eq/ne match a JSON-STRING-typed stored value ONLY (type-aware string-only
                # contract): a stored JSON boolean (PG ->> renders it 'true'/'false') AND a stored
                # NUMBER (whose text form diverges across backends for out-of-int64 values) are
                # both excluded. eq 'true' -> only the string 'true' row {strtrue}.
                ('eq "true" (str)', [{'key': 'flag', 'operator': 'eq', 'value': 'true'}], 1),
                # eq 'false' (str) -> only the string 'false' row {strfalse}, NOT the JSON bool false.
                ('eq "false" (str)', [{'key': 'flag', 'operator': 'eq', 'value': 'false'}], 1),
                # ne 'true' (str): the JSON booleans and the numeric 1/0 are EXCLUDED by type; the
                # only string flag != 'true' is {strfalse} -> 1 on both backends. (Text-matching the
                # numeric rows would give 2 here, a count that diverges across backends.)
                ('ne "true" (str)', [{'key': 'flag', 'operator': 'ne', 'value': 'true'}], 1),
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'boolean-parity {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All boolean-type metadata-filter parity tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
