"""Real-server checks for the ``search_context`` browse tool.

Thread, source and tag filters, ``start_date``/``end_date`` filtering, an
empty result, ``limit`` clamping with its hint, ``offset`` pagination, and the
``content_type`` filter for multimodal entries.
"""

from tests.integration._harness.core import HarnessCore


class SearchBrowseMixin(HarnessCore):
    """Checks for search_context filtering and pagination."""

    async def test_search_context(self) -> bool:
        """Test searching with various filters.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # First store some test data
            await self.client.call_tool(
                'store_context',
                {
                    'thread_id': self.test_thread_id,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Message for search testing',
                    'tags': ['searchable', 'test'],
                },
            )

            # Test search by thread
            thread_results = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': self.test_thread_id},
            )

            thread_data = self._extract_content(thread_results)

            # search_context returns success with results
            if not thread_data.get('success'):
                self.test_results.append((test_name, False, f'Thread search failed: {thread_data}'))
                return False

            # Test search by source
            source_results = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'source': 'user'},
            )

            source_data = self._extract_content(source_results)

            # search_context returns success with results
            if not source_data.get('success'):
                self.test_results.append((test_name, False, f'Source search failed: {source_data}'))
                return False

            # Test search by tags
            tag_results = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'tags': ['searchable']},
            )

            tag_data = self._extract_content(tag_results)

            # search_context returns success with results
            if not tag_data.get('success'):
                self.test_results.append((test_name, False, f'Tag search failed: {tag_data}'))
                return False

            # Test pagination
            paginated_results = await self.client.call_tool(
                'search_context',
                {
                    'thread_id': self.test_thread_id,
                    'limit': 1,
                    'offset': 0,
                },
            )

            paginated_data = self._extract_content(paginated_results)

            # search_context returns success with results
            if not paginated_data.get('success'):
                self.test_results.append((test_name, False, f'Pagination failed: {paginated_data}'))
                return False

            # Verify all searches returned results
            all_have_results = all([
                len(thread_data.get('results', [])) > 0,
                len(source_data.get('results', [])) > 0,
                len(tag_data.get('results', [])) > 0,
                len(paginated_data.get('results', [])) > 0,
            ])

            if all_have_results:
                self.test_results.append((test_name, True, 'All search filters working'))
                return True
            self.test_results.append((test_name, False, 'Some searches returned no results'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_with_date_filtering(self) -> bool:
        """Test search_context with start_date and end_date parameters.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_date_filtering'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for date filtering tests
            date_filter_thread = f'{self.test_thread_id}_date_filter'

            # Store a test entry (will be created at current time)
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': date_filter_thread,
                    'source': 'user',
                    'text': 'Entry for date filtering test',
                    'tags': ['date-filter', 'test'],
                },
            )

            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store test entry: {store_data}'))
                return False

            # Get current date information for testing
            from datetime import UTC
            from datetime import datetime
            from datetime import timedelta

            today = datetime.now(UTC).strftime('%Y-%m-%d')
            tomorrow = (datetime.now(UTC) + timedelta(days=1)).strftime('%Y-%m-%d')
            yesterday = (datetime.now(UTC) - timedelta(days=1)).strftime('%Y-%m-%d')
            future_date = (datetime.now(UTC) + timedelta(days=30)).strftime('%Y-%m-%d')
            past_date = (datetime.now(UTC) - timedelta(days=30)).strftime('%Y-%m-%d')

            # Test 1: Search with valid date range (today to tomorrow) - should find entry
            valid_range_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'start_date': today,
                    'end_date': tomorrow,
                },
            )
            valid_range_data = self._extract_content(valid_range_result)
            if not valid_range_data.get('success') or len(valid_range_data.get('results', [])) != 1:
                self.test_results.append(
                    (test_name, False, f'Valid date range search failed: {valid_range_data}'),
                )
                return False

            # Test 2: Search with future start_date - should NOT find entry
            future_start_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'start_date': future_date,
                },
            )
            future_start_data = self._extract_content(future_start_result)
            if not future_start_data.get('success') or len(future_start_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Future start_date returned results unexpectedly: {future_start_data}'),
                )
                return False

            # Test 3: Search with past end_date - should NOT find entry
            past_end_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'end_date': past_date,
                },
            )
            past_end_data = self._extract_content(past_end_result)
            if not past_end_data.get('success') or len(past_end_data.get('results', [])) != 0:
                self.test_results.append(
                    (test_name, False, f'Past end_date returned results unexpectedly: {past_end_data}'),
                )
                return False

            # Test 4: Search with date-only end_date for today - should find entry
            # This verifies that a date-only end_date expands to end-of-day
            today_end_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'end_date': today,
                },
            )
            today_end_data = self._extract_content(today_end_result)
            if not today_end_data.get('success') or len(today_end_data.get('results', [])) != 1:
                self.test_results.append(
                    (test_name, False, f'Date-only end_date failed to find today entry: {today_end_data}'),
                )
                return False

            # Test 5: Combined filters (date + source)
            combined_result = await self.client.call_tool(
                'search_context',
                {'limit': 50,
                    'thread_id': date_filter_thread,
                    'source': 'user',
                    'start_date': yesterday,
                    'end_date': tomorrow,
                },
            )
            combined_data = self._extract_content(combined_result)
            if not combined_data.get('success') or len(combined_data.get('results', [])) != 1:
                self.test_results.append(
                    (test_name, False, f'Combined date+source filter failed: {combined_data}'),
                )
                return False

            self.test_results.append((test_name, True, 'All date filtering tests passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_no_results(self) -> bool:
        """Test search with no matching results returns empty array.

        Returns:
            bool: True if test passed.
        """
        test_name = 'Search Context No Results'
        assert self.client is not None
        try:
            # Search for a non-existent thread
            result = await self.client.call_tool(
                'search_context',
                {
                    'thread_id': 'nonexistent_thread_xyz_123456789',
                    'limit': 50,
                },
            )

            data = self._extract_content(result)

            # Should succeed with empty results
            if data.get('success') and len(data.get('results', [])) == 0:
                self.test_results.append((test_name, True, 'No results returned correctly'))
                return True

            self.test_results.append((test_name, False, f'Expected empty results: {data}'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_limit_clamping(self) -> bool:
        """Search with limit > 100 should clamp and include clamped_limit hint.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_limit_clamping'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Search with limit > 100 (should be clamped)
            results = await self.client.call_tool(
                'search_context',
                {'limit': 200},
            )

            data = self._extract_content(results)

            if not data.get('success'):
                self.test_results.append((test_name, False, f'Search failed: {data}'))
                return False

            # Verify clamped_limit hint is present
            if 'clamped_limit' not in data:
                self.test_results.append((
                    test_name, False,
                    'Missing clamped_limit hint for limit=200',
                ))
                return False

            clamped = data['clamped_limit']
            if clamped != {'requested': 200, 'applied': 100}:
                self.test_results.append((
                    test_name, False,
                    f'Wrong clamped_limit values: {clamped}',
                ))
                return False

            # Verify normal limit does NOT include clamped_limit
            normal_results = await self.client.call_tool(
                'search_context',
                {'limit': 50},
            )

            normal_data = self._extract_content(normal_results)

            if not normal_data.get('success'):
                self.test_results.append((
                    test_name, False,
                    f'Normal search failed: {normal_data}',
                ))
                return False

            if 'clamped_limit' in normal_data:
                self.test_results.append((
                    test_name, False,
                    'clamped_limit should not appear for limit=50',
                ))
                return False

            self.test_results.append((test_name, True, 'Limit clamping works correctly'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_offset_pagination(self) -> bool:
        """Verify search_context offset pagination partitions results cleanly.

        FTS/semantic/hybrid offset pagination are covered, but search_context
        (the keyword browse tool) is only ever called at offset 0 elsewhere.
        This stores 5 entries in one thread and pages with limit=2 at offsets
        0/2/4, asserting no overlap and full, gapless coverage of all 5.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_offset_pagination'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_search_pag'
            for i in range(5):
                store = await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'Pagination browse entry number {i}',
                })
                if not self._extract_content(store).get('success'):
                    self.test_results.append((test_name, False, f'Store {i} failed'))
                    return False

            seen: list[str] = []
            for offset in (0, 2, 4):
                page = await self.client.call_tool('search_context', {
                    'thread_id': thread, 'limit': 2, 'offset': offset,
                })
                page_ids = [r['id'] for r in self._extract_content(page).get('results', [])]
                seen.extend(page_ids)

            if len(seen) != 5:
                self.test_results.append((test_name, False, f'Expected 5 ids across pages, got {len(seen)}'))
                return False
            if len(set(seen)) != 5:
                self.test_results.append((test_name, False, f'Pages overlap: {seen}'))
                return False

            self.test_results.append((test_name, True, 'search_context offset pagination: 5 unique ids, no overlap'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_content_type_filter_multimodal(self) -> bool:
        """Verify content_type filter works in search_context for multimodal entries.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_content_type_filter_multimodal'
        assert self.client is not None
        try:
            ct_filter_thread = f'{self.test_thread_id}_ct_filter'

            await self.client.call_tool('store_context', {
                'thread_id': ct_filter_thread, 'source': 'agent',
                'text': 'Text only entry for filter test',
            })

            await self.client.call_tool('store_context', {
                'thread_id': ct_filter_thread, 'source': 'agent',
                'text': 'Multimodal entry for filter test',
                'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
            })

            result = await self.client.call_tool('search_context', {
                'thread_id': ct_filter_thread, 'content_type': 'multimodal', 'limit': 50,
            })
            data = self._extract_content(result)

            results = data.get('results', [])
            if len(results) != 1:
                self.test_results.append((test_name, False,
                    f'Expected 1 multimodal entry, got {len(results)}'))
                return False

            if results[0].get('content_type') != 'multimodal':
                self.test_results.append((test_name, False, 'Returned entry is not multimodal'))
                return False

            text_result = await self.client.call_tool('search_context', {
                'thread_id': ct_filter_thread, 'content_type': 'text', 'limit': 50,
            })
            text_data = self._extract_content(text_result)
            text_results = text_data.get('results', [])

            if len(text_results) != 1:
                self.test_results.append((test_name, False,
                    f'Expected 1 text entry, got {len(text_results)}'))
                return False

            self.test_results.append((test_name, True, 'Content type filter correctly separates entries'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
