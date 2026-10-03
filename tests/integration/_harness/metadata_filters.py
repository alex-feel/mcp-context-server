"""Real-server checks for metadata filter operators and JSON paths.

The advanced ``metadata_filters`` operators end to end, nested JSON paths and
a key spelled ``null`` traversed the same way on both backends, and LIKE and
GLOB special characters in ``contains``/``starts_with``/``ends_with`` values
matched literally.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class MetadataFiltersMixin(HarnessCore):
    """Checks for metadata filter operators, nested paths and literal matching."""

    async def test_metadata_filtering(self) -> bool:
        """Test advanced metadata filtering functionality.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filtering'
        print('Testing metadata filtering...')

        # Store test context entries with various metadata
        test_entries = [
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'High priority task',
                'metadata': {'status': 'active', 'priority': 10, 'agent_name': 'analyzer'},
            },
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'Medium priority task',
                'metadata': {'status': 'active', 'priority': 5, 'agent_name': 'coordinator'},
            },
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'Low priority completed',
                'metadata': {'status': 'completed', 'priority': 1, 'completed': True},
            },
            {
                'thread_id': f'{self.test_thread_id}_metadata',
                'source': 'agent',
                'text': 'Failed task',
                'metadata': {'status': 'failed', 'priority': 8},
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

            # Test 1: Simple metadata filtering
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata': {'status': 'active'},
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 2:
                print(f"Simple filter failed: expected 2, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Simple metadata filter failed'))
                return False

            # Test 2: Advanced metadata filtering with gte operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata_filters': [{'key': 'priority', 'operator': 'gte', 'value': 5}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 3:
                print(f"Advanced gte filter failed: expected 3, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Advanced gte filter failed'))
                return False

            # Test 3: Combined metadata filters
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata': {'status': 'active'},
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 7}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 1:
                print(f"Combined filter failed: expected 1, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Combined filter failed'))
                return False

            # Test 4: Exists operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata_filters': [{'key': 'completed', 'operator': 'exists', 'value': None}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 1:
                print(f"Exists filter failed: expected 1, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'Exists operator filter failed'))
                return False

            # Test 5: In operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': f'{self.test_thread_id}_metadata',
                    'metadata_filters': [{'key': 'agent_name', 'operator': 'in', 'value': ['analyzer', 'coordinator']}],
                },
            )
            result_data = self._extract_content(result)
            if len(result_data.get('results', [])) != 2:
                print(f"In operator filter failed: expected 2, got {len(result_data.get('results', []))}")
                self.test_results.append((test_name, False, 'In operator filter failed'))
                return False

            print('[OK] All metadata filtering tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_nested_path(self) -> bool:
        """Nested-JSON-path metadata operators traverse on BOTH backends.

        PostgreSQL ``metadata->>'a.b.c'`` would read a literal top-level key named
        ``a.b.c`` instead of traversing, so every PostgreSQL operator accesses a dotted
        key through ``metadata#>>'{"a","b","c"}'``, which traverses the path the way
        SQLite's ``json_extract`` does. A nested ``ne``/``gt``/``contains``/``exists``/
        ``is_not_null`` filter therefore returns the same count on both backends; the
        check runs on both so the counts agree.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter Nested Path'
        print('Testing nested-path metadata filtering...')
        thread = f'{self.test_thread_id}_nested_meta'
        entries = [
            {
                'thread_id': thread, 'source': 'agent', 'text': 'nested A',
                'metadata': {'user': {'preferences': {'theme': 'dark'}}, 'settings': {'level': 10}},
            },
            {
                'thread_id': thread, 'source': 'agent', 'text': 'nested B',
                'metadata': {'user': {'preferences': {'theme': 'light'}}, 'settings': {'level': 5}},
            },
            {
                'thread_id': thread, 'source': 'agent', 'text': 'nested C',
                'metadata': {'user': {'preferences': {'theme': 'dark'}}, 'settings': {'level': 1}},
            },
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
                    self.test_results.append((test_name, False, 'Failed to store nested entries'))
                    return False

            # Each operator on a NESTED key must return the SQLite count on
            # PostgreSQL too.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('eq dark', [{'key': 'user.preferences.theme', 'operator': 'eq', 'value': 'dark'}], 2),
                ('ne dark', [{'key': 'user.preferences.theme', 'operator': 'ne', 'value': 'dark'}], 1),
                ('gt level 4', [{'key': 'settings.level', 'operator': 'gt', 'value': 4}], 2),
                ('contains ar', [{'key': 'user.preferences.theme', 'operator': 'contains', 'value': 'ar'}], 2),
                ('exists theme', [{'key': 'user.preferences.theme', 'operator': 'exists', 'value': None}], 3),
                ('is_not_null theme', [{'key': 'user.preferences.theme', 'operator': 'is_not_null', 'value': None}], 3),
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'nested {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            # The returned metadata must be a TRAVERSABLE nested dict, not a string.
            res = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50, 'thread_id': thread,
                    'metadata_filters': [{'key': 'settings.level', 'operator': 'eq', 'value': 10}],
                },
            )
            rows = self._extract_content(res).get('results', [])
            if len(rows) != 1 or rows[0].get('metadata', {}).get('user', {}).get('preferences', {}).get('theme') != 'dark':
                self.test_results.append((test_name, False, 'nested metadata not returned as a dict'))
                return False

            print('[OK] All nested-path metadata filtering tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_null_path_segment_parity(self) -> bool:
        """A metadata key literally spelled ``null`` traverses identically on both backends.

        PostgreSQL's array-literal parser reads an unquoted, case-insensitive bareword
        ``null`` inside ``metadata#>>'{a,null}'`` as a genuine SQL NULL path element, and
        the accessor yields NULL as soon as ANY element is NULL. Unquoted, an object key
        spelled ``null`` would collapse the whole accessor on PostgreSQL only: ``eq``
        would match nothing while SQLite matches, ``exists`` would find nothing, and
        ``not_exists`` would return the very entry that DOES carry the key. Quoting every
        path segment makes both backends traverse the same path, so the same counts hold
        here by construction.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'metadata_filter_null_path_segment_parity'
        assert self.client is not None
        thread = f'{self.test_thread_id}_null_segment'
        entries = [
            {
                'thread_id': thread, 'source': 'agent', 'text': 'entry carrying a null-named key',
                'metadata': {'a': {'null': 'x', 'b': 'y'}},
            },
            {
                'thread_id': thread, 'source': 'agent', 'text': 'entry without a null-named key',
                'metadata': {'a': {'other': 'z'}},
            },
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
                    self.test_results.append((test_name, False, f'Failed to store null-segment entries: {stored}'))
                    return False

            # SQLite defines the expected counts; the quoted path literal makes
            # PostgreSQL match. An unquoted ``null`` segment would make PostgreSQL
            # answer 0 / 2 / 0 for the first three.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('eq a.null', [{'key': 'a.null', 'operator': 'eq', 'value': 'x'}], 1),
                ('not_exists a.null', [{'key': 'a.null', 'operator': 'not_exists', 'value': None}], 1),
                ('exists a.null', [{'key': 'a.null', 'operator': 'exists', 'value': None}], 1),
                ('eq a.b control', [{'key': 'a.b', 'operator': 'eq', 'value': 'y'}], 1),
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'null-segment {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            self.test_results.append((test_name, True, 'A null-named metadata key traverses identically on both backends'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_operators_comprehensive(self) -> bool:
        """Verify the 9 metadata operators not covered elsewhere, end-to-end.

        The harness already exercises eq/gt/gte/in/exists/array_contains. This
        method validates the remaining advertised operators against a live
        engine (SQLite json_extract vs PostgreSQL ->>/jsonb): ne, lte, not_in,
        not_exists, contains, starts_with, ends_with, is_null, is_not_null.
        Result semantics are identical across backends (only the generated SQL
        differs), so one dataset yields one set of expected counts on both.

        Returns:
            bool: True if test passed.
        """
        test_name = 'metadata_filter_operators_comprehensive'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_meta_ops'
            entries = [
                ('Operators dataset entry one',
                 {'status': 'open', 'priority': 5, 'label': 'hello world', 'flag': 'active'}),
                ('Operators dataset entry two',
                 {'status': 'closed', 'priority': 10, 'label': 'hello there'}),
                ('Operators dataset entry three',
                 {'status': 'open', 'priority': 15, 'label': 'goodbye world', 'flag': None}),
                ('Operators dataset entry four',
                 {'status': 'archived', 'priority': 20, 'label': 'world peace'}),
            ]
            for text, metadata in entries:
                store = await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent', 'text': text, 'metadata': metadata,
                })
                if not self._extract_content(store).get('success'):
                    self.test_results.append((test_name, False, f'Store failed for {text!r}'))
                    return False

            # (label, metadata_filter, expected_match_count). Counts derive from
            # the dataset above and the documented operator semantics: missing
            # keys never match ne/lte/not_in/contains/starts_with/ends_with;
            # not_exists matches missing-key OR JSON-null; is_null matches only a
            # present JSON null; is_not_null matches only a present non-null value.
            cases: list[tuple[str, dict[str, Any], int]] = [
                ('ne', {'key': 'status', 'operator': 'ne', 'value': 'open'}, 2),
                ('lte', {'key': 'priority', 'operator': 'lte', 'value': 10}, 2),
                ('not_in', {'key': 'status', 'operator': 'not_in', 'value': ['open', 'closed']}, 1),
                ('not_exists', {'key': 'flag', 'operator': 'not_exists'}, 3),
                ('contains', {'key': 'label', 'operator': 'contains', 'value': 'world'}, 3),
                ('starts_with', {'key': 'label', 'operator': 'starts_with', 'value': 'hello'}, 2),
                ('ends_with', {'key': 'label', 'operator': 'ends_with', 'value': 'world'}, 2),
                ('is_null', {'key': 'flag', 'operator': 'is_null'}, 1),
                ('is_not_null', {'key': 'flag', 'operator': 'is_not_null'}, 1),
            ]
            for label, filter_spec, expected in cases:
                result = await self.client.call_tool('search_context', {
                    'thread_id': thread, 'metadata_filters': [filter_spec], 'limit': 30,
                })
                data = self._extract_content(result)
                got = len(data.get('results', []))
                if got != expected:
                    self.test_results.append((test_name, False,
                        f"operator '{label}' returned {got}, expected {expected} (filter={filter_spec})"))
                    return False

            self.test_results.append((test_name, True,
                ('All 9 metadata operators (ne, lte, not_in, not_exists, contains, '
                 'starts_with, ends_with, is_null, is_not_null) returned correct rows')))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_like_wildcard_literal(self) -> bool:
        """contains/starts_with/ends_with treat %/_ in the value as LITERAL chars.

        An unescaped value like ``50%`` would be a LIKE pattern, so CONTAINS
        ``"50%"`` would match any value with ``"50"`` followed by anything
        (over-broad). The value is escaped and compared with an ESCAPE clause, so
        it matches literally on BOTH backends.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter LIKE Wildcard Literal'
        print('Testing LIKE-wildcard-literal metadata filtering...')
        thread = f'{self.test_thread_id}_like_lit'
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'pct A', 'metadata': {'note': 'discount 50% off'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'pct B', 'metadata': {'note': 'discount 5000 dollars'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'und C', 'metadata': {'note': 'file_name.txt'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'und D', 'metadata': {'note': 'fileXname.txt'}},
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
                    self.test_results.append((test_name, False, 'Failed to store like-literal entries'))
                    return False

            # '50%' must match ONLY the literal '50% off' (A), NOT '5000' (B);
            # a wildcard reading would match both.
            checks: list[tuple[str, list[dict[str, Any]], int]] = [
                ('contains 50% literal', [{'key': 'note', 'operator': 'contains', 'value': '50%'}], 1),
                # '_' must be literal: 'file_name' matches C only, not 'fileXname' (D).
                ('contains file_name literal', [{'key': 'note', 'operator': 'contains', 'value': 'file_name'}], 1),
                # sanity: a wildcard-free substring still matches both percent rows.
                ('contains discount', [{'key': 'note', 'operator': 'contains', 'value': 'discount'}], 2),
            ]
            for label, filters, expected in checks:
                got = await _count(filters)
                if got != expected:
                    msg = f'like-literal {label}: expected {expected}, got {got}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] All LIKE-wildcard-literal metadata filtering tests passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_filter_glob_special_literal(self) -> bool:
        """Case-sensitive starts_with/ends_with treat GLOB specials (* ? [) literally.

        On the SQLite case-sensitive GLOB branch, a value containing a GLOB
        metacharacter is bracket-escaped so it matches literally on SQLite
        (PostgreSQL uses LIKE+ESCAPE), agreeing across backends.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Metadata Filter GLOB Special Literal'
        print('Testing case-sensitive GLOB-literal metadata filtering...')
        thread = f'{self.test_thread_id}_glob_lit'
        entries = [
            {'thread_id': thread, 'source': 'agent', 'text': 'glob A', 'metadata': {'code': 'a*b token'}},
            {'thread_id': thread, 'source': 'agent', 'text': 'glob B', 'metadata': {'code': 'aXXb token'}},
        ]
        try:
            assert self.client is not None  # Type guard for Pyright
            for entry in entries:
                result = await self.client.call_tool('store_context', entry)
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store glob entries'))
                    return False

            # Case-sensitive starts_with 'a*b' must match ONLY 'a*b token' (A),
            # not 'aXXb token' (B). Backslash-escaping would return 0 on SQLite,
            # whose GLOB has no ESCAPE clause and reads a backslash literally.
            res = await self.client.call_tool(
                'search_context',
                {
                    'limit': 50, 'thread_id': thread,
                    'metadata_filters': [
                        {'key': 'code', 'operator': 'starts_with', 'value': 'a*b', 'case_sensitive': True},
                    ],
                },
            )
            rows = self._extract_content(res).get('results', [])
            codes = sorted(r.get('metadata', {}).get('code', '') for r in rows)
            if codes != ['a*b token']:
                msg = f'glob starts_with a*b literal: expected [a*b token], got {codes}'
                print(f'[FAIL] {msg}')
                self.test_results.append((test_name, False, msg))
                return False

            print('[OK] Case-sensitive GLOB-literal metadata filtering test passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
