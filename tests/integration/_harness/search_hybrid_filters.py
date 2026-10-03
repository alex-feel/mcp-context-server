"""Real-server checks for the ``hybrid_search_context`` filters.

The ``metadata`` and ``metadata_filters`` arguments and ``start_date``/
``end_date`` ranges applied to hybrid search.
"""

import asyncio
from datetime import UTC
from typing import Any

from tests.integration._harness.core import HarnessCore


class SearchHybridFiltersMixin(HarnessCore):
    """Checks for metadata and date filters on hybrid search."""

    async def test_hybrid_search_metadata_filtering(self) -> bool:
        """Test metadata and metadata_filters parameters for hybrid_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_metadata_filtering'
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

            # Create a separate thread for metadata tests
            meta_thread = f'{self.test_thread_id}_hybrid_metadata'

            # Store entries with different metadata
            test_entries = [
                {'text': 'High priority backend task for API development', 'metadata': {'priority': 9, 'category': 'backend'}},
                {'text': 'Low priority frontend task for UI updates', 'metadata': {'priority': 3, 'category': 'frontend'}},
                {'text': 'High priority database optimization task', 'metadata': {'priority': 8, 'category': 'backend'}},
                {'text': 'Medium priority testing task', 'metadata': {'priority': 5, 'category': 'testing'}},
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
                    self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: Simple metadata filter (category=backend)
            simple_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development task',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'backend'},
                    'limit': 10,
                },
            )
            simple_data = self._extract_content(simple_result)
            if 'results' not in simple_data:
                self.test_results.append((test_name, False, f'Simple metadata filter failed: {simple_data}'))
                return False

            simple_results = simple_data.get('results', [])
            # Should find 2 backend entries
            if len(simple_results) < 1:
                self.test_results.append((test_name, False, 'No results with category=backend'))
                return False

            # Helper to get metadata (may be dict or JSON string)
            def get_meta(result: dict[str, Any]) -> dict[str, Any]:
                meta = result.get('metadata', {})
                if isinstance(meta, str):
                    import json
                    try:
                        return json.loads(meta)
                    except (json.JSONDecodeError, TypeError):
                        return {}
                return meta if isinstance(meta, dict) else {}

            # Verify all results have category=backend
            for r in simple_results:
                meta = get_meta(r)
                if meta.get('category') != 'backend':
                    cat = meta.get('category')
                    self.test_results.append((test_name, False, f"Expected category='backend', got '{cat}'"))
                    return False

            # Test 2: Advanced metadata filter (priority > 5)
            adv_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'task',
                    'thread_id': meta_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 5}],
                    'limit': 10,
                },
            )
            adv_data = self._extract_content(adv_result)
            if 'results' not in adv_data:
                self.test_results.append((test_name, False, f'Advanced metadata filter failed: {adv_data}'))
                return False

            adv_results = adv_data.get('results', [])
            # Should find entries with priority > 5 (9, 8 = 2)
            if len(adv_results) < 1:
                self.test_results.append((test_name, False, 'No results with priority > 5'))
                return False

            # Verify all results have priority > 5
            for r in adv_results:
                meta = get_meta(r)
                if meta.get('priority', 0) <= 5:
                    self.test_results.append((test_name, False, f"Expected priority > 5, got {meta.get('priority')}"))
                    return False

            # Test 3: Combined metadata filter (category=backend AND priority >= 8)
            combined_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'task',
                    'thread_id': meta_thread,
                    'metadata': {'category': 'backend'},
                    'metadata_filters': [{'key': 'priority', 'operator': 'gte', 'value': 8}],
                    'limit': 10,
                },
            )
            combined_data = self._extract_content(combined_result)
            if 'results' not in combined_data:
                self.test_results.append((test_name, False, f'Combined metadata filter failed: {combined_data}'))
                return False

            combined_results = combined_data.get('results', [])
            # Should find entries with category=backend AND priority >= 8 (9, 8 = 2)
            for r in combined_results:
                meta = get_meta(r)
                if meta.get('category') != 'backend' or meta.get('priority', 0) < 8:
                    self.test_results.append((test_name, False, f'Expected backend+priority>=8, got {meta}'))
                    return False

            self.test_results.append((test_name, True, 'metadata and metadata_filters working for hybrid_search'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_hybrid_search_date_range_filtering(self) -> bool:
        """Test start_date and end_date parameters for hybrid_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'hybrid_search_date_range_filtering'
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

            # Create a separate thread for date range tests
            date_thread = f'{self.test_thread_id}_hybrid_date'

            # Store test entries (will all have current timestamp)
            test_texts = [
                'Python machine learning algorithms for AI development',
                'Database query optimization techniques',
                'Frontend web development with modern frameworks',
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
                    self.test_results.append((test_name, False, f'Failed to store context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Get current date for testing
            from datetime import datetime
            from datetime import timedelta

            now = datetime.now(UTC)
            today = now.strftime('%Y-%m-%d')
            yesterday = (now - timedelta(days=1)).strftime('%Y-%m-%d')
            tomorrow = (now + timedelta(days=1)).strftime('%Y-%m-%d')

            # Test 1: Filter with start_date (should find entries from today)
            start_date_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'start_date': today,
                    'limit': 10,
                },
            )
            start_date_data = self._extract_content(start_date_result)
            if 'results' not in start_date_data:
                self.test_results.append((test_name, False, f'start_date filter failed: {start_date_data}'))
                return False

            start_results = start_date_data.get('results', [])
            # Should find all entries (created today)
            if len(start_results) < 1:
                self.test_results.append((test_name, False, 'No results with start_date filter'))
                return False

            # Test 2: Filter with end_date (should find entries up to today)
            end_date_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'end_date': today,
                    'limit': 10,
                },
            )
            end_date_data = self._extract_content(end_date_result)
            if 'results' not in end_date_data:
                self.test_results.append((test_name, False, f'end_date filter failed: {end_date_data}'))
                return False

            end_results = end_date_data.get('results', [])
            if len(end_results) < 1:
                self.test_results.append((test_name, False, 'No results with end_date filter'))
                return False

            # Test 3: Filter with date range (yesterday to tomorrow)
            range_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'start_date': yesterday,
                    'end_date': tomorrow,
                    'limit': 10,
                },
            )
            range_data = self._extract_content(range_result)
            if 'results' not in range_data:
                self.test_results.append((test_name, False, f'Date range filter failed: {range_data}'))
                return False

            range_results = range_data.get('results', [])
            if len(range_results) < 1:
                self.test_results.append((test_name, False, 'No results with date range filter'))
                return False

            # Test 4: Filter with future start_date (should find no entries)
            future_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'development',
                    'thread_id': date_thread,
                    'start_date': tomorrow,
                    'limit': 10,
                },
            )
            future_data = self._extract_content(future_result)
            if 'results' not in future_data:
                self.test_results.append((test_name, False, f'Future date filter failed: {future_data}'))
                return False

            future_results = future_data.get('results', [])
            if len(future_results) > 0:
                self.test_results.append((test_name, False, f'Expected 0 results for future date, got {len(future_results)}'))
                return False

            self.test_results.append((test_name, True, 'start_date and end_date working for hybrid_search'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
