"""Real-server checks for the ``semantic_search_context`` tool.

Vector-similarity search when an embedding provider is available, with
``start_date``/``end_date`` and metadata filters and ``offset`` pagination,
and metadata returned as a dict by both the semantic and the hybrid search
tools.
"""

import asyncio

from tests.integration._harness.core import HarnessCore


class SearchSemanticMixin(HarnessCore):
    """Checks for semantic search results, filters and pagination."""

    async def test_semantic_search_context(self) -> bool:
        """Test semantic search functionality (conditional on availability).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for semantic search tests
            semantic_thread = f'{self.test_thread_id}_semantic'

            # Store semantically diverse test contexts
            test_contexts = [
                'Machine learning models require large datasets for training and validation',
                'Python is a popular programming language for data science and AI applications',
                'The weather today is sunny with a high of 25 degrees celsius',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': semantic_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Allow time for embedding generation (non-blocking operation)
            await asyncio.sleep(0.5)

            # Test 1: Semantic search for ML-related content
            ml_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'artificial intelligence and deep learning',
                    'limit': 5,
                },
            )

            ml_search_data = self._extract_content(ml_search_result)

            # semantic_search_context returns results directly without 'success' field
            # Check for 'results' key instead
            if 'results' not in ml_search_data:
                self.test_results.append((test_name, False, f'ML semantic search failed: {ml_search_data}'))
                return False

            # Verify results contain distance/similarity scores in scores object
            ml_results = ml_search_data.get('results', [])
            if not ml_results or 'scores' not in ml_results[0] or 'semantic_distance' not in ml_results[0].get('scores', {}):
                self.test_results.append((test_name, False, 'Missing scores or semantic_distance in results'))
                return False

            # Test 2: Search with thread_id filter
            thread_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'programming languages',
                    'thread_id': semantic_thread,
                    'limit': 3,
                },
            )

            thread_search_data = self._extract_content(thread_search_result)

            if 'results' not in thread_search_data:
                self.test_results.append((test_name, False, f'Thread-filtered search failed: {thread_search_data}'))
                return False

            # Test 3: Search with source filter
            source_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'data science',
                    'source': 'agent',
                    'limit': 5,
                },
            )

            source_search_data = self._extract_content(source_search_result)

            if 'results' not in source_search_data:
                self.test_results.append((test_name, False, f'Source-filtered search failed: {source_search_data}'))
                return False

            # Get model name for success message
            model_name = semantic_info.get('model', 'unknown')

            self.test_results.append((test_name, True, f'Semantic search working (model: {model_name})'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_context_with_date_filtering(self) -> bool:
        """Test semantic_search_context with date filtering parameters.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_date_filtering'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for semantic search date filtering tests
            semantic_date_thread = f'{self.test_thread_id}_semantic_date'

            # Store semantically meaningful test content
            test_contexts = [
                'Machine learning algorithms process data to make predictions',
                'Database indexing improves query performance significantly',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': semantic_date_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Allow time for embedding generation
            import asyncio

            await asyncio.sleep(0.5)

            # Get date information for filtering
            from datetime import UTC
            from datetime import datetime
            from datetime import timedelta

            today = datetime.now(UTC).strftime('%Y-%m-%d')
            tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
            future_date = (datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')
            past_date = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')

            # Test 1: Semantic search with valid date range - should find results
            valid_range_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'machine learning artificial intelligence',
                    'thread_id': semantic_date_thread,
                    'start_date': today,
                    'end_date': tomorrow,
                    'limit': 5,
                },
            )
            valid_range_data = self._extract_content(valid_range_result)
            if 'results' not in valid_range_data or len(valid_range_data.get('results', [])) == 0:
                self.test_results.append(
                    (test_name, False, f'Valid date range semantic search failed: {valid_range_data}'),
                )
                return False

            # Test 2: Semantic search with future start_date - should return empty
            future_start_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'machine learning',
                    'thread_id': semantic_date_thread,
                    'start_date': future_date,
                    'limit': 5,
                },
            )
            future_start_data = self._extract_content(future_start_result)
            if 'results' not in future_start_data or len(future_start_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Future start_date returned results unexpectedly: {future_start_data}'),
                )
                return False

            # Test 3: Semantic search with past end_date - should return empty
            past_end_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'database indexing',
                    'thread_id': semantic_date_thread,
                    'end_date': past_date,
                    'limit': 5,
                },
            )
            past_end_data = self._extract_content(past_end_result)
            if 'results' not in past_end_data or len(past_end_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Past end_date returned results unexpectedly: {past_end_data}'),
                )
                return False

            # Test 4: Combined filters (date + source)
            combined_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'algorithms data processing',
                    'thread_id': semantic_date_thread,
                    'source': 'agent',
                    'start_date': today,
                    'end_date': tomorrow,
                    'limit': 5,
                },
            )
            combined_data = self._extract_content(combined_result)
            if 'results' not in combined_data or len(combined_data.get('results', [])) == 0:
                self.test_results.append(
                    (test_name, False, f'Combined date+source filter failed: {combined_data}'),
                )
                return False

            self.test_results.append((test_name, True, 'All semantic search date filtering tests passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_context_with_metadata_filters(self) -> bool:
        """Test semantic search with metadata filtering (conditional on availability).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_context_with_metadata_filters'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled via get_statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            # Skip gracefully if not enabled or available
            if not is_enabled or not is_available:
                self.test_results.append(
                    (test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'),
                )
                return True

            # Create a separate thread for metadata filter tests
            metadata_thread = f'{self.test_thread_id}_semantic_metadata'

            # Store test contexts with different metadata
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
                        'thread_id': metadata_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'metadata': entry['metadata'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: Semantic search with simple metadata filter (category=backend)
            metadata_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'development tasks',
                    'thread_id': metadata_thread,
                    'metadata': {'category': 'backend'},
                    'limit': 10,
                },
            )

            metadata_search_data = self._extract_content(metadata_search_result)

            if 'results' not in metadata_search_data:
                self.test_results.append((test_name, False, f'Metadata filter search failed: {metadata_search_data}'))
                return False

            # Should return only backend entries (2 entries)
            metadata_results = metadata_search_data.get('results', [])
            if len(metadata_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 backend entries, got {len(metadata_results)}'),
                )
                return False

            # Test 2: Semantic search with advanced metadata filter (priority > 5)
            advanced_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'tasks',
                    'thread_id': metadata_thread,
                    'metadata_filters': [{'key': 'priority', 'operator': 'gt', 'value': 5}],
                    'limit': 10,
                },
            )

            advanced_search_data = self._extract_content(advanced_search_result)

            if 'results' not in advanced_search_data:
                self.test_results.append((test_name, False, f'Advanced filter search failed: {advanced_search_data}'))
                return False

            # Should return entries with priority > 5 (priority 8 and 9)
            advanced_results = advanced_search_data.get('results', [])
            if len(advanced_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 high priority entries, got {len(advanced_results)}'),
                )
                return False

            # Test 3: Combined metadata + other filters
            combined_search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'backend',
                    'thread_id': metadata_thread,
                    'source': 'agent',
                    'metadata': {'category': 'backend'},
                    'metadata_filters': [{'key': 'priority', 'operator': 'gte', 'value': 8}],
                    'limit': 10,
                },
            )

            combined_search_data = self._extract_content(combined_search_result)

            if 'results' not in combined_search_data:
                self.test_results.append((test_name, False, f'Combined filter search failed: {combined_search_data}'))
                return False

            # Should return only high priority backend entries (priority >= 8 and category=backend)
            combined_results = combined_search_data.get('results', [])
            if len(combined_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 combined filter entries, got {len(combined_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Semantic search with metadata filtering working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_offset_pagination(self) -> bool:
        """Test offset pagination for semantic_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_offset_pagination'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check if semantic search is enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            is_enabled = semantic_info.get('enabled', False)
            is_available = semantic_info.get('available', False)

            if not is_enabled or not is_available:
                self.test_results.append((test_name, True, f'Skipped (enabled={is_enabled}, available={is_available})'))
                return True

            # Create a separate thread for pagination tests
            page_thread = f'{self.test_thread_id}_semantic_offset'

            # Store 5 entries for pagination testing
            test_texts = [
                'First Python programming tutorial for beginners',
                'Second Python advanced programming concepts',
                'Third Python web development with Django',
                'Fourth Python data science and machine learning',
                'Fifth Python automation and scripting guide',
            ]

            stored_ids = []
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
                stored_ids.append(result_data.get('context_id'))

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: First page (offset=0, limit=2)
            page1_result = await self.client.call_tool(
                'semantic_search_context',
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
                'semantic_search_context',
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

            # Test 3: Third page (offset=4, limit=2) - should get 1 result
            page3_result = await self.client.call_tool(
                'semantic_search_context',
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

            self.test_results.append((test_name, True, 'Offset pagination working for semantic_search'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_hybrid_metadata_is_dict(self) -> bool:
        """semantic_search_context and hybrid_search_context return metadata as a dict.

        Guards the metadata parse in semantic_search_raw (app/tools/search/legs.py):
        the repository returns metadata as a JSON string (SQLite TEXT; PostgreSQL
        JSONB-as-str via asyncpg), and the leg must surface it as a dict. Asserts the
        type AND a nested read, so a missing parse fails on BOTH backends.

        Returns:
            bool: True if all tests pass.
        """
        test_name = 'Semantic Hybrid Metadata Dict'
        print('Testing semantic/hybrid metadata-is-dict...')
        thread = f'{self.test_thread_id}_meta_dict'
        try:
            assert self.client is not None  # Type guard for Pyright
            text = (
                'Vector embeddings map text into a high dimensional space for semantic '
                'similarity search across stored context entries. '
            ) * 6
            store = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': thread, 'source': 'agent', 'text': text,
                    'metadata': {'project': 'meta-dict', 'nested': {'level': 3, 'label': 'deep'}},
                },
            )
            cid = self._extract_content(store).get('context_id')
            if not cid:
                self.test_results.append((test_name, False, 'store failed'))
                return False

            for tool in ('semantic_search_context', 'hybrid_search_context'):
                res = await self.client.call_tool(
                    tool,
                    {'query': 'semantic similarity vector embeddings', 'thread_id': thread, 'limit': 5},
                )
                rows = self._extract_content(res).get('results', [])
                hit = next((x for x in rows if x.get('id') == cid), None)
                if hit is None:
                    self.test_results.append((test_name, False, f'{tool} did not find entry'))
                    return False
                meta = hit.get('metadata')
                if not isinstance(meta, dict) or meta.get('nested', {}).get('label') != 'deep':
                    msg = f'{tool} metadata not a traversable dict: {type(meta).__name__}'
                    print(f'[FAIL] {msg}')
                    self.test_results.append((test_name, False, msg))
                    return False

            print('[OK] semantic/hybrid metadata-is-dict test passed')
            self.test_results.append((test_name, True, 'All tests passed'))
            return True

        except Exception as e:
            print(f'Test failed with exception: {e}')
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
