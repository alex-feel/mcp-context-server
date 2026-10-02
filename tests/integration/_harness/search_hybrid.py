"""Real-server checks for the ``hybrid_search_context`` tool.

RRF fusion of full-text and semantic results ordered by score, the adaptive
switch to OR boolean mode for long queries, fused results whenever full-text
search is available, and ``offset`` pagination.
"""

import asyncio

from tests.integration._harness.core import HarnessCore


class SearchHybridMixin(HarnessCore):
    """Checks for hybrid search fusion, adaptive FTS mode, degradation and pagination."""

    async def test_hybrid_search_context(self) -> bool:
        """Test hybrid search combining FTS and semantic search with RRF fusion.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if ENABLE_HYBRID_SEARCH environment variable is set
            if 'hybrid_search_context' not in self.registered_tools:
                self.test_results.append(
                    (test_name, True, 'Skipped (hybrid_search_context not registered)'),
                )
                return True

            # Check if hybrid search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            # Check both FTS and semantic search availability
            fts_info = stats_data.get('fts', {})
            semantic_info = stats_data.get('semantic_search', {})

            fts_enabled = fts_info.get('enabled', False)
            fts_available = fts_info.get('available', False)
            semantic_enabled = semantic_info.get('enabled', False)
            semantic_available = semantic_info.get('available', False)

            # Hybrid search requires at least one of FTS or semantic to be available
            has_fts = fts_enabled and fts_available
            has_semantic = semantic_enabled and semantic_available

            # Skip gracefully if neither search type is available
            if not has_fts and not has_semantic:
                self.test_results.append(
                    (
                        test_name,
                        True,
                        f'Skipped (fts={has_fts}, semantic={has_semantic})',
                    ),
                )
                return True

            # Create a separate thread for hybrid search tests
            hybrid_thread = f'{self.test_thread_id}_hybrid'

            # Store test contexts with diverse content
            test_contexts = [
                'Python machine learning algorithms for data science applications',
                'Advanced database indexing and query optimization techniques',
                'Neural networks and deep learning frameworks in Python',
                'JavaScript frontend development with modern frameworks',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': hybrid_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Basic hybrid search with default settings
            hybrid_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'python machine learning',
                    'thread_id': hybrid_thread,
                    'limit': 10,
                },
            )

            hybrid_data = self._extract_content(hybrid_result)

            # Check for results
            if 'results' not in hybrid_data:
                self.test_results.append((test_name, False, f'Hybrid search failed: {hybrid_data}'))
                return False

            # Verify response structure
            if 'fusion_method' not in hybrid_data:
                self.test_results.append((test_name, False, 'Response missing fusion_method field'))
                return False

            if 'search_modes_used' not in hybrid_data:
                self.test_results.append((test_name, False, 'Response missing search_modes_used field'))
                return False

            # Verify search_modes_used reflects execution, not results.
            # If FTS ran (no warning about failure) but returned 0 results,
            # it should still appear in search_modes_used.
            fts_count = hybrid_data.get('fts_count', 0)
            has_fts_warning = any(
                'FTS sub-search failed' in w for w in hybrid_data.get('warnings', [])
            )
            modes_used = hybrid_data.get('search_modes_used', [])
            if has_fts and not has_fts_warning and fts_count == 0 and 'fts' not in modes_used:
                msg = (
                    f'search_modes_used={modes_used} excludes fts despite '
                    f'successful execution (fts_count=0, no error)'
                )
                self.test_results.append((test_name, False, msg))
                return False

            if 'fts_count' not in hybrid_data or 'semantic_count' not in hybrid_data:
                self.test_results.append((test_name, False, 'Response missing source counts'))
                return False

            hybrid_results = hybrid_data.get('results', [])
            if len(hybrid_results) < 1:
                self.test_results.append((test_name, False, 'No results from hybrid search'))
                return False

            # Test 2: Verify results have RRF scores structure
            first_result = hybrid_results[0]
            if 'scores' not in first_result:
                self.test_results.append((test_name, False, 'Result missing scores field'))
                return False

            scores = first_result.get('scores', {})
            if 'rrf' not in scores:
                self.test_results.append((test_name, False, 'Scores missing rrf field'))
                return False

            # Test 3: Test source filtering
            source_filter_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'python',
                    'thread_id': hybrid_thread,
                    'source': 'agent',
                    'limit': 10,
                },
            )

            source_data = self._extract_content(source_filter_result)
            if 'results' not in source_data:
                self.test_results.append((test_name, False, f'Source filter search failed: {source_data}'))
                return False

            # All our test entries are from 'agent', should find results
            source_results = source_data.get('results', [])
            # Verify all results have source='agent'
            for r in source_results:
                if r.get('source') != 'agent':
                    self.test_results.append(
                        (test_name, False, f"Expected source='agent', got '{r.get('source')}'"),
                    )
                    return False

            # Test 4: Verify fusion method in response
            if hybrid_data.get('fusion_method') != 'rrf':
                self.test_results.append(
                    (test_name, False, f"Expected fusion_method='rrf', got '{hybrid_data.get('fusion_method')}'"),
                )
                return False

            self.test_results.append((test_name, True, 'Hybrid search with RRF fusion working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_adaptive_fts_mode(self) -> bool:
        """Test that hybrid search uses adaptive FTS mode for long queries.

        Verifies that:
        1. Long queries (4+ terms) switch to boolean mode with OR logic
        2. The explain_query stats include adaptive_fts_mode field
        3. Short queries continue to use match mode

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_adaptive_fts_mode'
        assert self.client is not None  # Type guard for Pyright
        try:
            if 'hybrid_search_context' not in self.registered_tools:
                self.test_results.append(
                    (test_name, True, 'Skipped (hybrid_search_context not registered)'),
                )
                return True

            if 'fts_search_context' not in self.registered_tools:
                self.test_results.append(
                    (test_name, True, 'Skipped (fts_search_context not registered)'),
                )
                return True

            # Test 1: Long query with explain_query to verify adaptive_fts_mode
            long_query = 'DRY extraction embedding helper timeout semaphore pattern implementation'
            long_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': long_query,
                    'explain_query': True,
                    'limit': 5,
                },
            )

            long_data = self._extract_content(long_result)

            # Verify stats include adaptive_fts_mode
            stats = long_data.get('stats')
            if stats is None:
                self.test_results.append(
                    (test_name, False, 'Missing stats with explain_query=True'),
                )
                return False

            adaptive_mode = stats.get('adaptive_fts_mode')
            if adaptive_mode is None:
                self.test_results.append(
                    (test_name, False, 'Missing adaptive_fts_mode in stats'),
                )
                return False

            if adaptive_mode != 'boolean':
                self.test_results.append(
                    (test_name, False, f'Expected boolean mode for long query, got {adaptive_mode}'),
                )
                return False

            # Test 2: Short query should use match mode
            short_query = 'python async'
            short_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': short_query,
                    'explain_query': True,
                    'limit': 5,
                },
            )

            short_data = self._extract_content(short_result)

            short_stats = short_data.get('stats')
            if short_stats is None:
                self.test_results.append(
                    (test_name, False, 'Missing stats for short query with explain_query=True'),
                )
                return False

            short_mode = short_stats.get('adaptive_fts_mode')
            if short_mode != 'match':
                self.test_results.append(
                    (test_name, False, f'Expected match mode for short query, got {short_mode}'),
                )
                return False

            self.test_results.append(
                (test_name, True, 'Adaptive FTS mode working: long=boolean, short=match'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Error: {e}'))
            return False

    async def test_hybrid_search_graceful_degradation_fts_only(self) -> bool:
        """Verify hybrid search works when only FTS is available.

        Returns:
            bool: True if test passed.
        """
        test_name = 'hybrid_search_graceful_degradation_fts_only'
        assert self.client is not None
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            hybrid_enabled = 'hybrid_search_context' in self.registered_tools

            if not has_fts or not hybrid_enabled:
                self.test_results.append((test_name, True,
                    f'Skipped (fts={has_fts}, hybrid={hybrid_enabled})'))
                return True

            degrade_thread = f'{self.test_thread_id}_hybrid_degrade'
            await self.client.call_tool('store_context', {
                'thread_id': degrade_thread, 'source': 'agent',
                'text': 'Database optimization and query performance tuning',
            })

            result = await self.client.call_tool('hybrid_search_context', {
                'query': 'database optimization',
                'thread_id': degrade_thread,
                'limit': 10,
            })
            data = self._extract_content(result)

            if 'results' not in data:
                self.test_results.append((test_name, False, f'Hybrid search failed: {data}'))
                return False

            modes = data.get('search_modes_used', [])
            if not modes:
                self.test_results.append((test_name, False, 'No search modes used'))
                return False

            if len(data.get('results', [])) < 1:
                self.test_results.append((test_name, False, 'No results from hybrid (degraded) search'))
                return False

            if 'fusion_method' not in data:
                self.test_results.append((test_name, False, 'Missing fusion_method'))
                return False

            semantic_count = data.get('semantic_count', 0)
            fts_count = data.get('fts_count', 0)

            self.test_results.append((test_name, True,
                f'Hybrid degradation OK: modes={modes}, fts={fts_count}, semantic={semantic_count}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_rrf_scores_ordering(self) -> bool:
        """Verify hybrid search results are ordered by RRF score (highest first).

        Returns:
            bool: True if test passed.
        """
        test_name = 'hybrid_search_rrf_scores_ordering'
        assert self.client is not None
        try:
            if 'hybrid_search_context' not in self.registered_tools:
                self.test_results.append((test_name, True, 'Skipped (hybrid search disabled)'))
                return True

            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)

            if not has_fts:
                self.test_results.append((test_name, True, 'Skipped (FTS unavailable)'))
                return True

            rrf_thread = f'{self.test_thread_id}_rrf_ordering'

            for text in [
                'Advanced Python machine learning with deep neural networks',
                'Simple Python tutorial for absolute beginners',
                'Unrelated topic about cooking recipes and ingredients',
            ]:
                await self.client.call_tool('store_context', {
                    'thread_id': rrf_thread, 'source': 'agent', 'text': text,
                })

            await asyncio.sleep(0.5)

            result = await self.client.call_tool('hybrid_search_context', {
                'query': 'Python machine learning neural networks',
                'thread_id': rrf_thread, 'limit': 10,
            })
            data = self._extract_content(result)

            if 'results' not in data:
                self.test_results.append((test_name, False, f'Hybrid search failed: {data}'))
                return False

            results = data.get('results', [])
            if len(results) < 2:
                self.test_results.append((test_name, True,
                    f'Only {len(results)} results, ordering check inconclusive'))
                return True

            rrf_scores = [r.get('scores', {}).get('rrf', 0) for r in results]
            is_ordered = all(rrf_scores[i] >= rrf_scores[i + 1] for i in range(len(rrf_scores) - 1))

            if not is_ordered:
                self.test_results.append((test_name, False,
                    f'RRF scores not in descending order: {rrf_scores}'))
                return False

            self.test_results.append((test_name, True,
                f'RRF scores correctly ordered: {rrf_scores[:3]}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_offset_pagination(self) -> bool:
        """Test offset pagination for hybrid_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_offset_pagination'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if hybrid search is available
            if 'hybrid_search_context' not in self.registered_tools:
                self.test_results.append((test_name, True, 'Skipped (hybrid_search_context not registered)'))
                return True

            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            semantic_info = stats_data.get('semantic_search', {})

            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not has_fts and not has_semantic:
                self.test_results.append((test_name, True, f'Skipped (fts={has_fts}, semantic={has_semantic})'))
                return True

            # Create a separate thread for pagination tests
            page_thread = f'{self.test_thread_id}_hybrid_offset'

            # Store 5 entries for pagination testing
            # NOTE: All entries MUST contain both 'Python' AND 'programming' because
            # FTS 'match' mode interprets "Python programming" as "Python AND programming"
            test_texts = [
                'First Python programming tutorial for beginners learning to code',
                'Second Python programming advanced concepts for software experts',
                'Third Python programming web development with Django framework',
                'Fourth Python programming data science and machine learning apps',
                'Fifth Python programming automation and scripting guide for DevOps',
            ]

            for text in test_texts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': page_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: First page (offset=0, limit=2)
            page1_result = await self.client.call_tool(
                'hybrid_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 0, 'limit': 2},
            )
            page1_data = self._extract_content(page1_result)
            if 'results' not in page1_data:
                self.test_results.append((test_name, False, f'Page 1 search failed: {page1_data}'))
                return False

            page1_results = page1_data.get('results', [])
            if len(page1_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 results for page 1, got {len(page1_results)}'))
                return False

            page1_ids = [r.get('id') for r in page1_results]

            # Test 2: Second page (offset=2, limit=2)
            page2_result = await self.client.call_tool(
                'hybrid_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 2, 'limit': 2},
            )
            page2_data = self._extract_content(page2_result)
            if 'results' not in page2_data:
                self.test_results.append((test_name, False, f'Page 2 search failed: {page2_data}'))
                return False

            page2_results = page2_data.get('results', [])
            if len(page2_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 results for page 2, got {len(page2_results)}'))
                return False

            page2_ids = [r.get('id') for r in page2_results]

            # Verify no overlap between pages
            overlap = set(page1_ids) & set(page2_ids)
            if overlap:
                self.test_results.append((test_name, False, f'Overlap found between pages: {overlap}'))
                return False

            # Test 3: Third page (offset=4, limit=2) - should get remaining results
            page3_result = await self.client.call_tool(
                'hybrid_search_context',
                {'query': 'Python programming', 'thread_id': page_thread, 'offset': 4, 'limit': 2},
            )
            page3_data = self._extract_content(page3_result)
            if 'results' not in page3_data:
                self.test_results.append((test_name, False, f'Page 3 search failed: {page3_data}'))
                return False

            page3_results = page3_data.get('results', [])
            if len(page3_results) != 1:
                self.test_results.append((test_name, False, f'Expected 1 result for page 3, got {len(page3_results)}'))
                return False

            page3_ids = [r.get('id') for r in page3_results]

            # Verify no overlap with previous pages
            overlap23 = set(page2_ids) & set(page3_ids)
            overlap13 = set(page1_ids) & set(page3_ids)
            if overlap23 or overlap13:
                self.test_results.append((test_name, False, f'Overlap found with page 3: {overlap23 | overlap13}'))
                return False

            # Verify pagination worked (different IDs across pages)
            all_ids = set(page1_ids) | set(page2_ids) | set(page3_ids)
            msg = f'Offset pagination working - {len(all_ids)} unique results across 3 pages'
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
