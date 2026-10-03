"""Real-server checks for the membership metadata filters.

``array_contains`` on array and non-array fields and with a high-magnitude
float member, and ``in``/``not_in`` lists that mix integer and float members
or meet a row storing a non-number at the key, with the same counts on both
backends.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class MetadataFiltersMembershipMixin(HarnessCore):
    """Checks for the array_contains, in and not_in metadata filters."""

    async def test_array_contains_operator(self) -> bool:
        """Test the array_contains operator for metadata filtering.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Array Contains Operator'
        print('Testing array_contains operator...')

        # Store test context entries with array metadata
        test_entries = [
            {
                'thread_id': f'{self.test_thread_id}_array_contains',
                'source': 'agent',
                'text': 'Python and FastAPI project',
                'metadata': {
                    'technologies': ['python', 'fastapi', 'postgresql'],
                    'priority_levels': [1, 3, 5],
                },
            },
            {
                'thread_id': f'{self.test_thread_id}_array_contains',
                'source': 'agent',
                'text': 'JavaScript frontend',
                'metadata': {
                    'technologies': ['javascript', 'react', 'typescript'],
                    'priority_levels': [2, 4, 6],
                },
            },
            {
                'thread_id': f'{self.test_thread_id}_array_contains',
                'source': 'agent',
                'text': 'Full stack project',
                'metadata': {
                    'technologies': ['python', 'javascript', 'docker'],
                    'priority_levels': [1, 5, 10],
                },
            },
        ]

        try:
            # Store all test entries
            assert self.client is not None  # Type guard for Pyright
            for entry in test_entries:
                result = await self.client.call_tool('store_context', entry)
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    print(f'Failed to store test entry: {result_data}')
                    self.test_results.append((test_name, False, 'Failed to store test entries'))
                    return False

            # Test 1: array_contains with string value
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': f'{self.test_thread_id}_array_contains',
                    'metadata_filters': [{'key': 'technologies', 'operator': 'array_contains', 'value': 'python'}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 2:
                print(f"array_contains string failed: expected 2, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'array_contains string filter failed'))
                return False

            # Test 2: array_contains with integer value
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': f'{self.test_thread_id}_array_contains',
                    'metadata_filters': [{'key': 'priority_levels', 'operator': 'array_contains', 'value': 5}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 2:
                print(f"array_contains integer failed: expected 2, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'array_contains integer filter failed'))
                return False

            # Test 3: array_contains with no match
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': f'{self.test_thread_id}_array_contains',
                    'metadata_filters': [{'key': 'technologies', 'operator': 'array_contains', 'value': 'rust'}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 0:
                print(f"array_contains no match failed: expected 0, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'array_contains no match test failed'))
                return False

            # Test 4: Combined array_contains filters
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': f'{self.test_thread_id}_array_contains',
                    'metadata_filters': [
                        {'key': 'technologies', 'operator': 'array_contains', 'value': 'python'},
                        {'key': 'technologies', 'operator': 'array_contains', 'value': 'javascript'},
                    ],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 1:
                print(f"Combined array_contains failed: expected 1, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Combined array_contains filter failed'))
                return False

            print('[OK] All array_contains operator tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_array_contains_non_array_field(self) -> bool:
        """Test array_contains gracefully handles non-array fields (returns empty, not error).

        PostgreSQL jsonb_array_elements_text() throws "cannot extract elements
        from a scalar" on non-array fields, so both backends check that the field
        is an array first and a non-array field returns empty results gracefully.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Array Contains Non-Array Field Handling'
        print('Testing array_contains on non-array fields...')

        # Store test context entries with SCALAR metadata (not arrays)
        test_thread_id = f'{self.test_thread_id}_array_contains_scalar'
        test_entries = [
            {
                'thread_id': test_thread_id,
                'source': 'agent',
                'text': 'Entry with scalar category',
                'metadata': {
                    'category': 'backend',  # Scalar string, NOT an array
                    'technologies': ['python', 'fastapi'],  # This IS an array
                },
            },
            {
                'thread_id': test_thread_id,
                'source': 'agent',
                'text': 'Entry with object config',
                'metadata': {
                    'config': {'timeout': 30, 'retries': 3},  # Object, NOT an array
                },
            },
            {
                'thread_id': test_thread_id,
                'source': 'agent',
                'text': 'Entry with number priority',
                'metadata': {
                    'priority': 5,  # Number scalar, NOT an array
                },
            },
        ]

        try:
            # Store all test entries
            assert self.client is not None  # Type guard for Pyright
            for entry in test_entries:
                result = await self.client.call_tool('store_context', entry)
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    print(f'Failed to store test entry: {result_data}')
                    self.test_results.append((test_name, False, 'Failed to store test entries'))
                    return False

            # Test 1: array_contains on SCALAR string field should return empty (not error)
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': test_thread_id,
                    'metadata_filters': [{'key': 'category', 'operator': 'array_contains', 'value': 'backend'}],
                },
            )
            result_data = self._extract_content(result)
            # Should return empty results, NOT an error
            if 'error' in result_data:
                print(f"array_contains on scalar field threw error: {result_data.get('error')}")
                self.test_results.append(
                    (test_name, False, 'array_contains on scalar field threw error instead of returning empty'),
                )
                return False
            if len(result_data.get('results', [])) != 0:
                print(f"array_contains on scalar field failed: expected 0, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'array_contains on scalar field should return empty'))
                return False

            # Test 2: array_contains on OBJECT field should return empty (not error)
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': test_thread_id,
                    'metadata_filters': [{'key': 'config', 'operator': 'array_contains', 'value': 30}],
                },
            )
            result_data = self._extract_content(result)
            if 'error' in result_data:
                print(f"array_contains on object field threw error: {result_data.get('error')}")
                self.test_results.append(
                    (test_name, False, 'array_contains on object field threw error instead of returning empty'),
                )
                return False
            if len(result_data.get('results', [])) != 0:
                print(f"array_contains on object field failed: expected 0, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'array_contains on object field should return empty'))
                return False

            # Test 3: array_contains on NUMBER field should return empty (not error)
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': test_thread_id,
                    'metadata_filters': [{'key': 'priority', 'operator': 'array_contains', 'value': 5}],
                },
            )
            result_data = self._extract_content(result)
            if 'error' in result_data:
                print(f"array_contains on number field threw error: {result_data.get('error')}")
                self.test_results.append(
                    (test_name, False, 'array_contains on number field threw error instead of returning empty'),
                )
                return False
            if len(result_data.get('results', [])) != 0:
                print(f"array_contains on number field failed: expected 0, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'array_contains on number field should return empty'))
                return False

            # Test 4: Verify array field STILL works correctly
            result = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50,
                    'thread_id': test_thread_id,
                    'metadata_filters': [{'key': 'technologies', 'operator': 'array_contains', 'value': 'python'}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 1:
                print(f"array_contains on array field failed: expected 1, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'array_contains on array field should still work'))
                return False

            print('[OK] All array_contains non-array field tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_array_contains_high_magnitude_float_parity(self) -> bool:
        """array_contains with a high-magnitude float member matches identically on both backends.

        A bare ``@>`` containment matches only the float's canonical decimal form, so a
        genuinely int-origin array element (e.g. the exact integer 2**55) would match on
        SQLite but not PostgreSQL above 2**53. The float-member path therefore iterates numeric
        elements through the shared exact/double discriminator, so a clean int-origin
        element matches on both backends and a clearly-different element matches on
        neither.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Array Contains High-Magnitude Float Parity'
        print('Testing high-magnitude-float array_contains parity...')
        thread = f'{self.test_thread_id}_arrfloat_parity'
        i55 = 2 ** 55            # exact integer element (36028797018963968)
        other = 2 ** 55 + 8      # the next representable double up; must NOT match float(2**55)
        fparam = float(2 ** 55)
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'has 2**55', 'metadata': {'vals': [1, i55, 3]}},
            {'thread_id': thread, 'source': 'agent', 'text': 'has 2**55+8', 'metadata': {'vals': [other]}},
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
                    self.test_results.append((test_name, False, 'Failed to store array_contains entries'))
                    return False

            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                # The int-origin 2**55 element matches float(2**55) on BOTH backends.
                ('contains float 2**55', [{'key': 'vals', 'operator': 'array_contains', 'value': fparam}], 1),
                # A plain int member matches its exact element through the @> containment path.
                ('contains int 2**55', [{'key': 'vals', 'operator': 'array_contains', 'value': i55}], 1),
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'array_contains parity {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All high-magnitude-float array_contains parity tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_not_in_numeric_over_nonnumber_row_parity(self) -> bool:
        """NOT_IN with numeric members keeps a present non-number row identically on both backends.

        A row storing a JSON STRING or boolean at the key, filtered by NOT_IN with
        numeric members, must be INCLUDED on BOTH SQLite and PostgreSQL. The PostgreSQL
        numeric disjunct guards on jsonb_typeof = 'number' (mirroring SQLite's json_type
        guard) so a present non-number yields a deterministic FALSE match -> NOT FALSE ->
        the row is kept, instead of a NULL match that PostgreSQL would silently drop
        under NOT_IN's three-valued logic.

        Returns:
            bool: True if test passed.
        """
        test_name = 'metadata_not_in_numeric_over_nonnumber_row_parity'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_not_in_numeric'
            entries = [
                ('not_in numeric string row', {'k': 'abc'}),       # present string -> kept
                ('not_in numeric boolean row', {'k': True}),       # present boolean -> kept
                ('not_in numeric matching number row', {'k': 1}),  # in [1, 2] -> excluded
                ('not_in numeric other number row', {'k': 99}),    # number not in list -> kept
            ]
            for text, metadata in entries:
                store = await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent', 'text': text, 'metadata': metadata,
                })
                if not self._extract_content(store).get('success'):
                    self.test_results.append((test_name, False, f'Store failed for {text!r}'))
                    return False

            result = await self.client.call_tool('search_context', {
                'thread_id': thread,
                'metadata_filters': [{'key': 'k', 'operator': 'not_in', 'value': [1, 2]}],
                'limit': 30,
            })
            data = self._extract_content(result)
            results = data.get('results', [])
            texts = {r.get('text_content', '') for r in results}
            # Identical on BOTH backends: the string, boolean, and 99 rows are kept; only the
            # matching-number row (k=1) is excluded. Without the number guard PostgreSQL drops
            # the string and boolean rows, so a count of 2 here would mark a parity break.
            if len(results) != 3:
                self.test_results.append((test_name, False,
                    (f'NOT_IN [1,2] over key k returned {len(results)} rows, expected 3 '
                     '(string + boolean + non-matching number kept; matching number excluded)')))
                return False
            if any('matching number row' in t for t in texts):
                self.test_results.append((test_name, False,
                    'NOT_IN [1,2] wrongly kept the matching-number row'))
                return False
            if not (any('string row' in t for t in texts) and any('boolean row' in t for t in texts)):
                self.test_results.append((test_name, False,
                    'NOT_IN [1,2] dropped a present non-number row (cross-backend parity)'))
                return False
            self.test_results.append((test_name, True,
                'NOT_IN with numeric members keeps present non-number rows identically on both backends'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_numeric_in_mixed_members_parity(self) -> bool:
        """Multi-member IN / NOT_IN lists mixing ints and floats agree across backends.

        The PostgreSQL numeric membership group emits the exact/double discriminator
        ONCE per filter, with an IN list on each arm, instead of rebuilding it per member
        (which would grow the generated statement by kilobytes per float member). The
        per-member SEMANTICS are those of a per-member comparison: an int member compares
        as exact NUMERIC for every stored value, a float member takes the discriminator,
        and a present NON-number stored value is a deterministic FALSE match so NOT_IN
        keeps it. Running the same
        expected counts on both backends proves the hoisted form preserves them; the
        single-member cases elsewhere in this harness cannot detect a per-arm mistake.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'metadata_filter_numeric_in_mixed_members_parity'
        assert self.client is not None
        thread = f'{self.test_thread_id}_in_mixed'
        # float(2**55) and the exact int 2**55 sit on opposite sides of the
        # int-origin discriminator; 0.3 is an ordinary fractional double; 7 and 42
        # are plain ints; 'abc' is the present non-number the NOT_IN guard must keep.
        fstored = float(2 ** 55)
        istored = 2 ** 55
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'in-mixed float 2**55', 'metadata': {'v': fstored}},
            {'thread_id': thread, 'source': 'agent', 'text': 'in-mixed int 2**55', 'metadata': {'v': istored}},
            {'thread_id': thread, 'source': 'agent', 'text': 'in-mixed fraction', 'metadata': {'v': 0.3}},
            {'thread_id': thread, 'source': 'agent', 'text': 'in-mixed seven', 'metadata': {'v': 7}},
            {'thread_id': thread, 'source': 'agent', 'text': 'in-mixed forty-two', 'metadata': {'v': 42}},
            {'thread_id': thread, 'source': 'agent', 'text': 'in-mixed string', 'metadata': {'v': 'abc'}},
        ]

        async def _count(filters: list[dict[str, Any]]) -> int:
            assert self.client is not None
            res = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': filters},
            )
            return len(self._extract_content(res).get('results', []))

        try:
            for entry in entries:
                stored = self._extract_content(await self.client.call_tool('store_context', entry))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store in-mixed entries: {stored}'))
                    return False

            mixed: list[float | int] = [fstored, 0.3, 7]
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                # float 2**55 matches BOTH the float-stored and the int-stored row
                # (the documented canonical-double-form equivalence), 0.3 matches the
                # fraction row, and the int member 7 matches the seven row.
                ('in mixed int+float members', [{'key': 'v', 'operator': 'in', 'value': mixed}], 4),
                # Complement over the same list: the unmatched number row plus the
                # present non-number row, which the jsonb_typeof guard keeps.
                ('not_in mixed int+float members', [{'key': 'v', 'operator': 'not_in', 'value': mixed}], 2),
                ('in float-only members', [{'key': 'v', 'operator': 'in', 'value': [fstored, 0.3]}], 3),
                ('in int-only members', [{'key': 'v', 'operator': 'in', 'value': [7, 42]}], 2),
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'in-mixed {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            self.test_results.append((test_name, True, 'Multi-member mixed int/float IN and NOT_IN agree on both backends'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
