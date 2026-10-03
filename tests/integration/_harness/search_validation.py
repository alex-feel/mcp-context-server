"""Real-server checks for search argument validation.

Invalid metadata filters, whitespace-only tags, over-cap ``metadata_filters``
lists and out-of-int64 simple metadata values return structured validation
errors from the search tools on both backends, and identical filter arguments
report the same ``filters_applied`` from every search tool.
"""

from typing import Any

from tests.integration._harness.core import HarnessCore


class SearchValidationMixin(HarnessCore):
    """Checks for search argument validation and filter reporting."""

    async def test_search_context_invalid_filter_returns_error(self) -> bool:
        """Test that search_context returns explicit error for invalid metadata filter.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_invalid_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Test with invalid operator
            result = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'metadata_filters': [{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}]},
            )

            result_data = self._extract_content(result)

            # Should return error response
            if 'error' not in result_data:
                self.test_results.append((test_name, False, f'Expected error response, got: {result_data}'))
                return False

            if result_data['error'] != 'Metadata filter validation failed':
                self.test_results.append((test_name, False, f"Wrong error message: {result_data['error']}"))
                return False

            if 'validation_errors' not in result_data:
                self.test_results.append((test_name, False, 'Missing validation_errors in response'))
                return False

            self.test_results.append((test_name, True, 'Invalid filter returns error as expected'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_blank_tags_returns_error(self) -> bool:
        """A whitespace-only tags filter is a validation error, not a widened result set.

        A non-empty tags list that normalizes to empty (every tag blank after
        trimming, e.g. ['   ']) must not drop the criterion silently and return
        every entry in scope. It surfaces the structured, breaker-exempt
        validation error on BOTH backends instead, while a genuinely
        empty tags list (tags=[]) still means "no tag filter" and returns the
        matching entries.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_blank_tags'
        assert self.client is not None  # Type guard for Pyright
        thread = f'{self.test_thread_id}_blank_tags'
        try:
            for i in range(2):
                await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': thread,
                        'source': 'agent',
                        'text': f'blank-tags entry {i}',
                        'tags': ['real'],
                    },
                )

            # A whitespace-only tags filter must be rejected, NOT silently dropped
            # (which would return both entries in scope).
            blank_result = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'tags': ['   ']},
            )
            blank_data = self._extract_content(blank_result)
            if 'error' not in blank_data or not blank_data.get('validation_errors'):
                self.test_results.append(
                    (test_name, False, f'Blank tags not rejected as a validation error: {blank_data}'),
                )
                return False
            if blank_data.get('results'):
                self.test_results.append(
                    (test_name, False, f'Blank tags returned a widened result set: {blank_data}'),
                )
                return False

            # tags=[] means "no tag filter": the two entries are returned.
            empty_result = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'tags': []},
            )
            empty_data = self._extract_content(empty_result)
            if not empty_data.get('success') or len(empty_data.get('results', [])) != 2:
                self.test_results.append(
                    (test_name, False, f'Empty tags list should return all entries, got: {empty_data}'),
                )
                return False

            self.test_results.append((test_name, True, 'Blank tags rejected; empty tags list unfiltered'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_context_oversized_metadata_filters_rejected(self) -> bool:
        """An over-cap metadata_filters list fails wire-schema validation, never reaching the tool body or SQL.

        The metadata_filters wire schema carries a max_length cap of 100, enforced during argument validation.

        A 101-filter list therefore raises a tool-level validation error naming metadata_filters; the tool body never runs.

        The in-body structured cap response is defense in depth and is unreachable through the real server.

        An at-cap list of exactly 100 filters still executes normally, pinning the boundary on BOTH backends.

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_context_oversized_metadata_filters'
        assert self.client is not None  # Type guard for Pyright
        thread = f'{self.test_thread_id}_filters_cap'
        try:
            await self.client.call_tool(
                'store_context',
                {
                    'thread_id': thread,
                    'source': 'agent',
                    'text': 'filters-cap entry',
                    'metadata': {'marker': True},
                },
            )

            # One filter over the cap of 100: the served schema's maxItems bound
            # must reject the list during argument validation, so the call raises
            # instead of returning a structured result.
            oversized = [{'key': 'marker', 'operator': 'exists'}] * 101
            try:
                leaked = await self.client.call_tool(
                    'search_context',
                    {'limit': 50, 'thread_id': thread, 'metadata_filters': oversized},
                )
                leaked_data = self._extract_content(leaked)
                self.test_results.append(
                    (test_name, False, f'Oversized metadata_filters not rejected at the wire schema: {leaked_data}'),
                )
                return False
            except Exception as exc:
                message = str(exc)
                if 'metadata_filters' not in message or ('too_long' not in message and 'at most 100' not in message):
                    self.test_results.append(
                        (test_name, False, f'Unexpected rejection message for oversized metadata_filters: {message}'),
                    )
                    return False

            # Exactly 100 filters is at the cap: the call passes the wire schema,
            # reaches the repository, and returns the stored entry, proving the
            # rejection above targets only the over-cap list.
            at_cap = [{'key': 'marker', 'operator': 'exists'}] * 100
            capped_result = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': thread, 'metadata_filters': at_cap},
            )
            capped_data = self._extract_content(capped_result)
            if not capped_data.get('success') or len(capped_data.get('results', [])) != 1:
                self.test_results.append(
                    (test_name, False, f'At-cap metadata_filters list should return the entry, got: {capped_data}'),
                )
                return False

            self.test_results.append(
                (test_name, True, 'Oversized metadata_filters rejected at the wire schema; at-cap list executes'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_simple_metadata_out_of_int64_rejected(self) -> bool:
        """search_context with a simple metadata={} integer beyond the signed 64-bit
        range returns a structured validation error on BOTH backends, never aborting
        the search (SQLite would otherwise raise OverflowError while PostgreSQL matches).

        Returns:
            bool: True if test passed.
        """
        test_name = 'search_simple_metadata_out_of_int64'
        assert self.client is not None  # Type guard for Pyright
        try:
            result = await self.client.call_tool(
                'search_context',
                {'metadata': {'huge': 10**20}},
            )
            result_data = self._extract_content(result)

            if 'error' not in result_data:
                self.test_results.append(
                    (test_name, False, f'Expected validation error, got: {result_data}'),
                )
                return False
            if result_data.get('count') != 0 or result_data.get('results') != []:
                self.test_results.append(
                    (test_name, False, f'Expected count=0 / empty results on error, got: {result_data}'),
                )
                return False

            self.test_results.append(
                (test_name, True, 'Out-of-int64 simple metadata filter rejected uniformly'),
            )
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_semantic_search_invalid_filter_returns_error(self) -> bool:
        """Test that semantic_search_context returns explicit error for invalid metadata filter.

        This test verifies unified error handling between search_context and semantic_search_context.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'semantic_search_invalid_filter'
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

            # Test with invalid operator
            result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'test query',
                    'metadata_filters': [{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}],
                },
            )

            result_data = self._extract_content(result)

            # Should return error response (unified with search_context behavior)
            if 'error' not in result_data:
                self.test_results.append((test_name, False, f'Expected error response, got: {result_data}'))
                return False

            if result_data['error'] != 'Metadata filter validation failed':
                self.test_results.append((test_name, False, f"Wrong error message: {result_data['error']}"))
                return False

            if 'validation_errors' not in result_data:
                self.test_results.append((test_name, False, 'Missing validation_errors in response'))
                return False

            # Verify response structure includes expected fields
            if result_data.get('count') != 0:
                self.test_results.append((test_name, False, 'Expected count=0 on error'))
                return False

            if result_data.get('results') != []:
                self.test_results.append((test_name, False, 'Expected empty results on error'))
                return False

            self.test_results.append((test_name, True, 'Invalid filter returns error (unified with search_context)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_search_invalid_filter_returns_error(self) -> bool:
        """Test that fts_search_context returns explicit error for invalid metadata filter.

        This test verifies unified error handling between fts_search_context and other search tools.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_search_invalid_filter'
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

            # Test with invalid operator
            result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'test query',
                    'metadata_filters': [{'key': 'status', 'operator': 'invalid_operator', 'value': 'test'}],
                },
            )

            result_data = self._extract_content(result)

            # Should return error response (unified with search_context behavior)
            if 'error' not in result_data:
                self.test_results.append((test_name, False, f'Expected error response, got: {result_data}'))
                return False

            if result_data['error'] != 'Metadata filter validation failed':
                self.test_results.append((test_name, False, f"Wrong error message: {result_data['error']}"))
                return False

            if 'validation_errors' not in result_data:
                self.test_results.append((test_name, False, 'Missing validation_errors in response'))
                return False

            # Verify response structure includes expected fields
            if result_data.get('count') != 0:
                self.test_results.append((test_name, False, 'Expected count=0 on error'))
                return False

            if result_data.get('results') != []:
                self.test_results.append((test_name, False, 'Expected empty results on error'))
                return False

            self.test_results.append((test_name, True, 'Invalid filter returns error (unified with search_context)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_all_stopword_query_with_invalid_filter_returns_error(self) -> bool:
        """An all-operator FTS query (transforms to an empty FTS query) combined with an
        invalid metadata filter must STILL raise the validation error on BOTH backends:
        metadata validation runs before the empty-query short-circuit, so the invalid
        filter is never silently swallowed into an empty result.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_all_operator_query_invalid_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            if not fts_info.get('enabled', False) or not fts_info.get('available', False):
                self.test_results.append(
                    (test_name, True, f"Skipped (enabled={fts_info.get('enabled')}, available={fts_info.get('available')})"),
                )
                return True

            result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'AND OR NOT',  # transforms to an empty FTS query
                    'metadata_filters': [{'key': 'status', 'operator': 'invalid_operator', 'value': 'x'}],
                },
            )
            result_data = self._extract_content(result)

            if 'error' not in result_data:
                self.test_results.append(
                    (test_name, False, f'Expected validation error, got: {result_data}'),
                )
                return False

            self.test_results.append(
                (test_name, True, 'Empty-FTS-query + invalid filter raises (validation precedes short-circuit)'),
            )
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_filters_applied_agreement_across_search_tools(self) -> bool:
        """Identical filter arguments report the same ``filters_applied`` from every search tool.

        ``filters_applied`` is what an operator reads back under ``explain_query`` to
        confirm a filter took effect. Counting only the METADATA conditions on the browse
        path while its sibling tools count every applied condition would make the same
        thread + source + tags + metadata request report 1 from ``search_context`` and 4
        from the FTS and semantic tools. One shared tally feeds all of them, so the number
        is identical across tools and equals the four conditions requested.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'filters_applied_agreement_across_search_tools'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_filters_applied'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Filter tally parity entry mentioning stanchion hardware',
                'tags': ['tallyparity'],
                'metadata': {'project': 'filters-parity'},
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False

            shared: dict[str, Any] = {
                'thread_id': thread,
                'source': 'agent',
                'tags': ['tallyparity'],
                'metadata': {'project': 'filters-parity'},
                'explain_query': True,
                'limit': 10,
            }
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            fts_info = stats_data.get('fts', {})
            tool_names = {t.name for t in await self.client.list_tools()}

            calls: list[tuple[str, dict[str, Any]]] = [('search_context', dict(shared))]
            if fts_info.get('enabled') and fts_info.get('available') and 'fts_search_context' in tool_names:
                calls.append(('fts_search_context', {**shared, 'query': 'stanchion', 'mode': 'match'}))
            if stats_data.get('semantic_search', {}).get('available') and 'semantic_search_context' in tool_names:
                calls.append(('semantic_search_context', {**shared, 'query': 'stanchion hardware'}))
            if len(calls) < 2:
                self.test_results.append((test_name, True, 'Skipped (no sibling search tool available to compare)'))
                return True

            applied: dict[str, Any] = {}
            for tool, args in calls:
                data = self._extract_content(await self.client.call_tool(tool, args))
                tool_stats = data.get('stats')
                if not isinstance(tool_stats, dict) or 'filters_applied' not in tool_stats:
                    self.test_results.append((test_name, False, f'{tool} returned no filters_applied: {data}'))
                    return False
                applied[tool] = tool_stats['filters_applied']

            # thread_id + source + tags + one metadata equality = four conditions.
            expected = 4
            if set(applied.values()) != {expected}:
                self.test_results.append((
                    test_name, False, f'filters_applied disagrees across tools (expected {expected} each): {applied}',
                ))
                return False

            self.test_results.append((
                test_name, True, f'All {len(applied)} search tools report filters_applied={expected}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
