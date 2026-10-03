"""Real-server checks for the ``fts_search_context`` tool.

Full-text search results, ``offset`` pagination, highlighted snippets, and
stemming in match mode.
"""

from tests.integration._harness.core import HarnessCore


class SearchFtsMixin(HarnessCore):
    """Checks for full-text search results, pagination, snippets and stemming."""

    async def test_fts_search_context(self) -> bool:
        """Test full-text search functionality (conditional on availability).

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_search_context'
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

            # Create a separate thread for FTS tests
            fts_thread = f'{self.test_thread_id}_fts'

            # Store test contexts for full-text search
            test_contexts = [
                'Python programming language tutorial for beginners',
                'Advanced machine learning with Python frameworks',
                'JavaScript and TypeScript web development guide',
                'Database indexing and query optimization techniques',
            ]

            for text in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': fts_thread,
                        'source': 'agent',
                        'text': text,
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Basic match mode search for 'Python'
            match_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'python',
                    'mode': 'match',
                    'thread_id': fts_thread,
                    'limit': 10,
                },
            )

            match_data = self._extract_content(match_result)

            # Check for results
            if 'results' not in match_data:
                self.test_results.append((test_name, False, f'Match mode search failed: {match_data}'))
                return False

            match_results = match_data.get('results', [])
            if len(match_results) != 2:  # Should find 2 Python entries
                self.test_results.append(
                    (test_name, False, f'Expected 2 Python results, got {len(match_results)}'),
                )
                return False

            # Verify results have scores object with fts_score
            if not all('scores' in r and 'fts_score' in r.get('scores', {}) for r in match_results):
                self.test_results.append((test_name, False, 'Missing scores or fts_score in results'))
                return False

            # Test 2: Prefix mode search for 'prog*'
            prefix_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'prog',
                    'mode': 'prefix',
                    'thread_id': fts_thread,
                    'limit': 10,
                },
            )

            prefix_data = self._extract_content(prefix_result)

            if 'results' not in prefix_data:
                self.test_results.append((test_name, False, f'Prefix mode search failed: {prefix_data}'))
                return False

            prefix_results = prefix_data.get('results', [])
            if len(prefix_results) < 1:  # Should find at least 1 entry with 'programming'
                self.test_results.append(
                    (test_name, False, f'Expected results for prefix "prog*", got {len(prefix_results)}'),
                )
                return False

            # Test 3: Phrase mode search
            phrase_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'machine learning',
                    'mode': 'phrase',
                    'thread_id': fts_thread,
                    'limit': 10,
                },
            )

            phrase_data = self._extract_content(phrase_result)

            if 'results' not in phrase_data:
                self.test_results.append((test_name, False, f'Phrase mode search failed: {phrase_data}'))
                return False

            phrase_results = phrase_data.get('results', [])
            if len(phrase_results) != 1:  # Should find exactly 1 entry with 'machine learning'
                self.test_results.append(
                    (test_name, False, f'Expected 1 phrase result, got {len(phrase_results)}'),
                )
                return False

            # Test 4: Search with source filter
            source_filter_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'python',
                    'mode': 'match',
                    'thread_id': fts_thread,
                    'source': 'agent',
                    'limit': 10,
                },
            )

            source_data = self._extract_content(source_filter_result)

            if 'results' not in source_data:
                self.test_results.append((test_name, False, f'Source filter search failed: {source_data}'))
                return False

            # All our test entries are from 'agent', should still find 2
            source_results = source_data.get('results', [])
            if len(source_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 agent results, got {len(source_results)}'),
                )
                return False

            # Test 5: Verify response structure includes required fields
            if match_data.get('mode') != 'match':
                self.test_results.append((test_name, False, 'Response missing mode field'))
                return False

            if 'count' not in match_data:
                self.test_results.append((test_name, False, 'Response missing count field'))
                return False

            if 'language' not in match_data:
                self.test_results.append((test_name, False, 'Response missing language field'))
                return False

            self.test_results.append((test_name, True, 'FTS search modes and filters working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_pagination_offset(self) -> bool:
        """Test FTS pagination with offset parameter.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_pagination_offset'
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

            # Create a separate thread for pagination tests
            page_thread = f'{self.test_thread_id}_fts_page'

            # Store multiple test contexts for pagination
            test_texts = [
                'Testing pagination feature one',
                'Testing pagination feature two',
                'Testing pagination feature three',
                'Testing pagination feature four',
                'Testing pagination feature five',
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
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: Get first page (offset=0, limit=2)
            page1_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'pagination',
                    'mode': 'match',
                    'thread_id': page_thread,
                    'offset': 0,
                    'limit': 2,
                },
            )

            page1_data = self._extract_content(page1_result)

            if 'results' not in page1_data:
                self.test_results.append((test_name, False, f'First page search failed: {page1_data}'))
                return False

            page1_results = page1_data.get('results', [])
            if len(page1_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 results on first page, got {len(page1_results)}'),
                )
                return False

            # Get IDs from first page
            page1_ids = {r.get('id') for r in page1_results}

            # Test 2: Get second page (offset=2, limit=2)
            page2_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'pagination',
                    'mode': 'match',
                    'thread_id': page_thread,
                    'offset': 2,
                    'limit': 2,
                },
            )

            page2_data = self._extract_content(page2_result)

            if 'results' not in page2_data:
                self.test_results.append((test_name, False, f'Second page search failed: {page2_data}'))
                return False

            page2_results = page2_data.get('results', [])
            if len(page2_results) != 2:
                self.test_results.append(
                    (test_name, False, f'Expected 2 results on second page, got {len(page2_results)}'),
                )
                return False

            # Get IDs from second page
            page2_ids = {r.get('id') for r in page2_results}

            # Verify no overlap between pages
            if page1_ids & page2_ids:
                self.test_results.append(
                    (test_name, False, f'Pages overlap: {page1_ids & page2_ids}'),
                )
                return False

            # Test 3: Get third page (offset=4, limit=2) - should get 1 result
            page3_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'pagination',
                    'mode': 'match',
                    'thread_id': page_thread,
                    'offset': 4,
                    'limit': 2,
                },
            )

            page3_data = self._extract_content(page3_result)

            if 'results' not in page3_data:
                self.test_results.append((test_name, False, f'Third page search failed: {page3_data}'))
                return False

            page3_results = page3_data.get('results', [])
            if len(page3_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result on third page, got {len(page3_results)}'),
                )
                return False

            self.test_results.append((test_name, True, 'Pagination with offset working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_highlight_snippets(self) -> bool:
        """Test FTS highlight parameter returns highlighted text with markers.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_highlight_snippets'
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

            # Create a separate thread for highlight tests
            hl_thread = f'{self.test_thread_id}_fts_highlight'

            # Store test context with specific searchable terms
            test_text = 'Advanced algorithms for sorting and searching in databases'
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': hl_thread,
                    'source': 'agent',
                    'text': test_text,
                },
            )
            result_data = self._extract_content(result)
            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                return False

            # Test 1: Search without highlight (default)
            no_hl_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'algorithms',
                    'mode': 'match',
                    'thread_id': hl_thread,
                    'limit': 10,
                },
            )

            no_hl_data = self._extract_content(no_hl_result)

            if 'results' not in no_hl_data:
                self.test_results.append((test_name, False, f'No-highlight search failed: {no_hl_data}'))
                return False

            no_hl_results = no_hl_data.get('results', [])
            if len(no_hl_results) < 1:
                self.test_results.append((test_name, False, 'No results found'))
                return False

            # Verify 'highlighted' value is None when highlight=False (default)
            # The field is always present in results but should be None when not requested
            if no_hl_results[0].get('highlighted') is not None:
                self.test_results.append((test_name, False, 'Highlighted value should be None when highlight=False'))
                return False

            # Test 2: Search with highlight=True
            hl_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'algorithms',
                    'mode': 'match',
                    'thread_id': hl_thread,
                    'highlight': True,
                    'limit': 10,
                },
            )

            hl_data = self._extract_content(hl_result)

            if 'results' not in hl_data:
                self.test_results.append((test_name, False, f'Highlight search failed: {hl_data}'))
                return False

            hl_results = hl_data.get('results', [])
            if len(hl_results) < 1:
                self.test_results.append((test_name, False, 'No results found with highlight=True'))
                return False

            # Verify 'highlighted' value is not None when highlight=True
            if hl_results[0].get('highlighted') is None:
                self.test_results.append((test_name, False, 'Highlighted value should not be None when highlight=True'))
                return False

            highlighted_text = hl_results[0].get('highlighted', '')

            # Verify <mark> tags are present in highlighted text
            if '<mark>' not in highlighted_text or '</mark>' not in highlighted_text:
                self.test_results.append(
                    (test_name, False, f'Highlighted text missing <mark> tags: {highlighted_text}'),
                )
                return False

            self.test_results.append((test_name, True, 'Highlight snippets with <mark> tags working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_search_match_mode_stemming(self) -> bool:
        """Verify FTS match mode applies stemming (e.g., 'running' matches 'run').

        Returns:
            bool: True if test passed.
        """
        test_name = 'fts_search_match_mode_stemming'
        assert self.client is not None
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS unavailable)'))
                return True

            stem_thread = f'{self.test_thread_id}_fts_stemming'

            await self.client.call_tool('store_context', {
                'thread_id': stem_thread, 'source': 'agent',
                'text': 'The programmer was running several optimization algorithms',
            })

            result = await self.client.call_tool('fts_search_context', {
                'query': 'run', 'mode': 'match',
                'thread_id': stem_thread, 'limit': 10,
            })
            data = self._extract_content(result)

            if 'results' not in data:
                self.test_results.append((test_name, False, f'FTS search failed: {data}'))
                return False

            results = data.get('results', [])
            if len(results) < 1:
                self.test_results.append((test_name, False,
                    'Stemming failed: "run" did not match "running"'))
                return False

            self.test_results.append((test_name, True, 'FTS stemming verified: "run" matched "running"'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
