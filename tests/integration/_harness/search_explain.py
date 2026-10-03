"""Real-server checks for ``explain_query``.

``explain_query=True`` adds execution statistics to the ``search_context``,
FTS and hybrid search responses, and ``explain_query=False`` or an omitted
flag leaves them out.
"""

import asyncio

from tests.integration._harness.core import HarnessCore


class SearchExplainMixin(HarnessCore):
    """Checks for explain_query execution statistics."""

    async def test_explain_query_statistics(self) -> bool:
        """Test explain_query parameter for search_context, fts_search, and hybrid_search.

        Verifies that explain_query=True returns execution statistics.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'explain_query_statistics'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            fts_info = stats_data.get('fts', {})
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_semantic = stats_data.get('semantic_search', {}).get('available', False)
            hybrid_enabled = 'hybrid_search_context' in self.registered_tools
            has_hybrid = (has_fts or has_semantic) and hybrid_enabled

            # Create a separate thread for explain_query tests
            explain_thread = f'{self.test_thread_id}_explain_query'

            # Store test entry
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': explain_thread,
                    'source': 'agent',
                    'text': 'Test content for explain query statistics verification',
                },
            )
            result_data = self._extract_content(result)
            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: search_context with explain_query=True
            search_result = await self.client.call_tool(
                'search_context',
                {'thread_id': explain_thread, 'explain_query': True, 'limit': 10},
            )
            search_data = self._extract_content(search_result)
            if not search_data.get('success'):
                self.test_results.append((test_name, False, f'search_context with explain_query failed: {search_data}'))
                return False

            # Verify stats are included
            if 'stats' not in search_data:
                self.test_results.append((test_name, False, 'search_context: Missing stats with explain_query=True'))
                return False

            search_stats = search_data.get('stats', {})
            if 'execution_time_ms' not in search_stats:
                self.test_results.append((test_name, False, 'search_context: Missing execution_time_ms in stats'))
                return False

            # Test 2: search_context with explain_query=False (default)
            no_explain_result = await self.client.call_tool(
                'search_context',
                {'thread_id': explain_thread, 'explain_query': False, 'limit': 10},
            )
            # With explain_query=False, stats should NOT be included
            no_explain_data = self._extract_content(no_explain_result)

            # Stats should NOT exist when explain_query=False
            if 'stats' in no_explain_data:
                self.test_results.append(
                    (test_name, False, 'search_context: stats should not be included when explain_query=False'),
                )
                return False

            # Test 3: fts_search with explain_query=True (if available)
            if has_fts:
                fts_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'test content',
                        'mode': 'match',
                        'thread_id': explain_thread,
                        'explain_query': True,
                        'limit': 10,
                    },
                )
                fts_data = self._extract_content(fts_result)
                if 'results' not in fts_data:
                    self.test_results.append((test_name, False, f'fts_search with explain_query failed: {fts_data}'))
                    return False

                # Verify stats are included for FTS
                if 'stats' not in fts_data:
                    self.test_results.append((test_name, False, 'fts_search: Missing stats with explain_query=True'))
                    return False

                fts_stats = fts_data.get('stats', {})
                if 'execution_time_ms' not in fts_stats:
                    self.test_results.append((test_name, False, 'fts_search: Missing execution_time_ms in stats'))
                    return False

            # Test 4: hybrid_search with explain_query=True (if available)
            if has_hybrid:
                hybrid_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {'query': 'test content', 'thread_id': explain_thread, 'explain_query': True, 'limit': 10},
                )
                hybrid_data = self._extract_content(hybrid_result)
                if 'results' not in hybrid_data:
                    self.test_results.append((test_name, False, f'hybrid_search with explain_query failed: {hybrid_data}'))
                    return False

                # Verify stats are included for hybrid
                if 'stats' not in hybrid_data:
                    self.test_results.append((test_name, False, 'hybrid_search: Missing stats with explain_query=True'))
                    return False

                hybrid_stats = hybrid_data.get('stats', {})
                if 'execution_time_ms' not in hybrid_stats:
                    self.test_results.append((test_name, False, 'hybrid_search: Missing execution_time_ms in stats'))
                    return False

            self.test_results.append((test_name, True, f'explain_query working (fts={has_fts}, hybrid={has_hybrid})'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_explain_query_false_no_stats(self) -> bool:
        """Verify that explain_query=False (or omitted) does NOT include stats.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_explain_query_false_no_stats'
        assert self.client is not None
        try:
            no_stats_thread = f'{self.test_thread_id}_no_stats'

            await self.client.call_tool('store_context', {
                'thread_id': no_stats_thread, 'source': 'agent',
                'text': 'Test entry for explain_query=False verification',
            })

            result = await self.client.call_tool('search_context', {
                'thread_id': no_stats_thread, 'limit': 10,
            })
            data = self._extract_content(result)

            if 'stats' in data:
                self.test_results.append((test_name, False,
                    'search_context: stats present without explain_query'))
                return False

            result2 = await self.client.call_tool('search_context', {
                'thread_id': no_stats_thread, 'limit': 10, 'explain_query': False,
            })
            data2 = self._extract_content(result2)

            if 'stats' in data2:
                self.test_results.append((test_name, False,
                    'search_context: stats present with explain_query=False'))
                return False

            self.test_results.append((test_name, True, 'explain_query=False correctly omits stats'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
