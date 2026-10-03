"""Real-server checks for the ``fts_search_context`` filters.

``start_date``/``end_date`` ranges, simple metadata equality, a metadata key
whose name contains ``metadata``, and the advanced ``metadata_filters``
operators applied to full-text search.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class SearchFtsFiltersMixin(HarnessCore):
    """Checks for date and metadata filters on full-text search."""

    async def test_fts_date_range_filter(self) -> bool:
        """Test FTS date range filtering with start_date and end_date.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_date_range_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for date filter tests
            date_thread = f'{self.test_thread_id}_fts_date'

            # Store test contexts
            test_texts = [
                'Database optimization techniques for large datasets',
                'Query performance tuning and indexing strategies',
            ]

            for text in test_texts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': date_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Get dates in ISO format for filtering using UTC timezone
            from datetime import UTC
            from datetime import datetime
            from datetime import timedelta

            now = datetime.now(tz=UTC)
            yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            tomorrow = (now + timedelta(days=1)).strftime('%Y-%m-%d')

            # Test 1: Search with start_date (should include today's entries)
            start_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'database',
                    'mode': 'match',
                    'thread_id': date_thread,
                    'start_date': yesterday,
                    'limit': 10,
                },
            )

            start_data = self._extract_content(start_result)

            if 'results' not in start_data:
                self.test_results.append((test_name, False, f'Start date filter search failed: {start_data}'))
                return False

            start_results = start_data.get('results', [])
            if len(start_results) < 1:
                self.test_results.append(
                    (test_name, False, f'Expected at least 1 result with start_date filter, got {len(start_results)}'),
                )
                return False

            # Test 2: Search with both start_date and end_date
            range_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'database',
                    'mode': 'match',
                    'thread_id': date_thread,
                    'start_date': yesterday,
                    'end_date': tomorrow,
                    'limit': 10,
                },
            )

            range_data = self._extract_content(range_result)

            if 'results' not in range_data:
                self.test_results.append((test_name, False, f'Date range filter search failed: {range_data}'))
                return False

            range_results = range_data.get('results', [])
            if len(range_results) < 1:
                self.test_results.append(
                    (test_name, False, f'Expected at least 1 result with date range filter, got {len(range_results)}'),
                )
                return False

            # Test 3: Search with future start_date (should return no results)
            future_start = (datetime.now(tz=UTC) + timedelta(days=10)).strftime('%Y-%m-%d')
            future_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'database',
                    'mode': 'match',
                    'thread_id': date_thread,
                    'start_date': future_start,
                    'limit': 10,
                },
            )

            future_data = self._extract_content(future_result)

            if 'results' not in future_data:
                self.test_results.append((test_name, False, f'Future date filter search failed: {future_data}'))
                return False

            future_results = future_data.get('results', [])
            if len(future_results) != 0:
                self.test_results.append(
                    (test_name, False, f'Expected 0 results for future start_date, got {len(future_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Date range filtering working (start_date, end_date)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_metadata_filter(self) -> bool:
        """Test FTS simple metadata equality filtering.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_metadata_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for metadata filter tests
            meta_thread = f'{self.test_thread_id}_fts_meta'

            # Store test contexts with different metadata
            test_entries = [
                {
                    'text': 'API design patterns for RESTful services',
                    'metadata': {'category': 'backend', 'priority': 1},
                },
                {
                    'text': 'Frontend component design with React',
                    'metadata': {'category': 'frontend', 'priority': 2},
                },
                {
                    'text': 'Backend database design principles',
                    'metadata': {'category': 'backend', 'priority': 3},
                },
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': meta_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'metadata': entry['metadata'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Filter by category='backend'
            backend_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'design',
                    'mode': 'match',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'backend'},
                    'limit': 10,
                },
            )

            backend_data = self._extract_content(backend_result)

            if 'results' not in backend_data:
                self.test_results.append((test_name, False, f'Metadata filter search failed: {backend_data}'))
                return False

            backend_results = backend_data.get('results', [])
            # Should find 2 entries with category='backend'
            if len(backend_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 backend results, got {len(backend_results)}'),
                )
                return False

            # Verify all results have the correct metadata
            for r in backend_results:
                meta = r.get('metadata', {})
                if meta.get('category') != 'backend':
                    self.test_results.append(
                        (test_name, False, f"Result has wrong category: {meta.get('category')}"),
                    )
                    return False

            # Test 2: Filter by category='frontend'
            frontend_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'design',
                    'mode': 'match',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'frontend'},
                    'limit': 10,
                },
            )

            frontend_data = self._extract_content(frontend_result)

            if 'results' not in frontend_data:
                self.test_results.append((test_name, False, f'Frontend filter search failed: {frontend_data}'))
                return False

            frontend_results = frontend_data.get('results', [])
            # Should find exactly 1 entry with category='frontend'
            if len(frontend_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 frontend result, got {len(frontend_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Simple metadata filtering working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_metadata_filter_key_substring(self) -> bool:
        """FTS metadata filter on a key whose NAME contains 'metadata'.

        The FTS query aliases ``context_entries`` as ``ce``, so its metadata conditions
        must reference ``ce.metadata``. A global ``str.replace('metadata', 'ce.metadata')``
        would also rewrite every 'metadata' substring inside a JSON key such as
        ``metadata_version`` (into ``ce.metadata_version``), so the filter would match
        a non-existent key and return nothing on BOTH backends.
        ``MetadataQueryBuilder(table_alias='ce')`` qualifies only the column positions
        and leaves the key intact. Runs on SQLite and PostgreSQL through the shared
        harness.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_metadata_filter_key_substring'
        assert self.client is not None  # Type guard for Pyright
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            sub_thread = f'{self.test_thread_id}_fts_meta_substr'
            # Keys 'metadata_version' / 'metadata_tags' (both contain the substring
            # 'metadata') and the ordinary 'status' key; all share the FTS term
            # 'versioning'. The 'metadata_tags' array key exercises array_contains,
            # whose qualified SQL has the structurally hardest forms (SQLite's TWO
            # column positions json_type+json_each, PostgreSQL's '->' accessor).
            test_entries = [
                {'text': 'metadata versioning alpha record',
                 'metadata': {'metadata_version': 1, 'status': 'active', 'metadata_tags': ['python', 'rust']}},
                {'text': 'metadata versioning beta record',
                 'metadata': {'metadata_version': 2, 'status': 'active', 'metadata_tags': ['go']}},
                {'text': 'metadata versioning gamma record',
                 'metadata': {'metadata_version': 1, 'status': 'archived', 'metadata_tags': ['python']}},
            ]
            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {'thread_id': sub_thread, 'source': 'agent', 'text': entry['text'], 'metadata': entry['metadata']},
                )
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, f'Failed to store: {self._extract_content(result)}'))
                    return False

            async def _count(filters: list[dict[str, Any]] | None = None, simple: dict[str, Any] | None = None) -> int:
                assert self.client is not None  # Type guard for Pyright/mypy (narrowing lost in closure)
                args: dict[str, Any] = {'query': 'versioning', 'mode': 'match', 'thread_id': sub_thread, 'limit': 10}
                if filters is not None:
                    args['metadata_filters'] = filters
                if simple is not None:
                    args['metadata'] = simple
                data = self._extract_content(await self.client.call_tool('fts_search_context', args))
                if 'results' not in data:
                    raise AssertionError(f'fts_search_context failed: {data}')
                return len(data.get('results', []))

            # eq on the 'metadata'-substring key: a corrupted key would return 0.
            n_v1 = await _count(filters=[{'key': 'metadata_version', 'operator': 'eq', 'value': 1}])
            if n_v1 != 2:
                self.test_results.append((test_name, False, f'metadata_version=1 expected 2, got {n_v1}'))
                return False
            n_v2 = await _count(filters=[{'key': 'metadata_version', 'operator': 'eq', 'value': 2}])
            if n_v2 != 1:
                self.test_results.append((test_name, False, f'metadata_version=2 expected 1, got {n_v2}'))
                return False
            n_gt = await _count(filters=[{'key': 'metadata_version', 'operator': 'gt', 'value': 1}])
            if n_gt != 1:
                self.test_results.append((test_name, False, f'metadata_version>1 expected 1, got {n_gt}'))
                return False
            # Ordinary key path stays correct.
            n_active = await _count(simple={'status': 'active'})
            if n_active != 2:
                self.test_results.append((test_name, False, f'status=active expected 2, got {n_active}'))
                return False
            # array_contains on a 'metadata'-substring array key: exercises the
            # multi-column-position (SQLite) / '->'-form (PostgreSQL) qualification
            # end-to-end through the real FTS JOIN.
            n_py = await _count(filters=[{'key': 'metadata_tags', 'operator': 'array_contains', 'value': 'python'}])
            if n_py != 2:
                self.test_results.append((test_name, False, f"array_contains 'python' expected 2, got {n_py}"))
                return False
            n_java = await _count(filters=[{'key': 'metadata_tags', 'operator': 'array_contains', 'value': 'java'}])
            if n_java != 0:
                self.test_results.append((test_name, False, f"array_contains 'java' expected 0, got {n_java}"))
                return False

            self.test_results.append((test_name, True, "FTS 'metadata'-substring key filter working (eq/gt/array_contains)"))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_advanced_metadata_filters(self) -> bool:
        """Test FTS advanced metadata filters with operators (gt, lt, contains, etc.).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_advanced_metadata_filters'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if FTS is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            is_enabled = fts_info.get('enabled', False)
            is_available = fts_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for advanced metadata filter tests
            adv_thread = f'{self.test_thread_id}_fts_adv_meta'

            # Store test contexts with priority metadata for numeric comparison
            test_entries = [
                {
                    'text': 'Critical security vulnerability fix',
                    'metadata': {'priority': 1, 'status': 'resolved'},
                },
                {
                    'text': 'Performance optimization for security module',
                    'metadata': {'priority': 5, 'status': 'pending'},
                },
                {
                    'text': 'Security audit documentation update',
                    'metadata': {'priority': 10, 'status': 'completed'},
                },
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': adv_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'metadata': entry['metadata'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Filter with 'gt' (greater than) operator - priority > 3
            gt_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'security',
                    'mode': 'match',
                    'thread_id': adv_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 3}],
                    'limit': 10,
                },
            )

            gt_data = self._extract_content(gt_result)

            if 'results' not in gt_data:
                self.test_results.append((test_name, False, f'gt operator search failed: {gt_data}'))
                return False

            gt_results = gt_data.get('results', [])
            # Should find 2 entries with priority > 3 (priority 5 and 10)
            if len(gt_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 results for priority > 3, got {len(gt_results)}'),
                )
                return False

            # Verify all results have priority > 3
            for r in gt_results:
                meta = r.get('metadata', {})
                if meta.get('priority', 0) <= 3:
                    self.test_results.append(
                        (test_name, False, f"Result has priority <= 3: {meta.get('priority')}"),
                    )
                    return False

            # Test 2: Filter with 'lt' (less than) operator - priority < 5
            lt_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'security',
                    'mode': 'match',
                    'thread_id': adv_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'lt', 'value': 5}],
                    'limit': 10,
                },
            )

            lt_data = self._extract_content(lt_result)

            if 'results' not in lt_data:
                self.test_results.append((test_name, False, f'lt operator search failed: {lt_data}'))
                return False

            lt_results = lt_data.get('results', [])
            # Should find 1 entry with priority < 5 (priority 1)
            if len(lt_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result for priority < 5, got {len(lt_results)}'),
                )
                return False

            # Test 3: Filter with 'eq' (equals) operator - status = 'pending'
            eq_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'security',
                    'mode': 'match',
                    'thread_id': adv_thread,
                    'metadata_filters': [{'key': 'status', 'operator': 'eq', 'value': 'pending'}],
                    'limit': 10,
                },
            )

            eq_data = self._extract_content(eq_result)

            if 'results' not in eq_data:
                self.test_results.append((test_name, False, f'eq operator search failed: {eq_data}'))
                return False

            eq_results = eq_data.get('results', [])
            # Should find 1 entry with status='pending'
            if len(eq_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result for status=pending, got {len(eq_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Advanced metadata filters (gt, lt, eq) working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
