"""Real-server checks for the discovery tools.

``list_threads`` on empty and populated databases and with ``limit`` and
``offset`` pagination, and ``get_statistics`` with its chunking, reranking
and summary configuration.
"""

import asyncio

from tests.integration._harness.core import HarnessCore


class DiscoveryMixin(HarnessCore):
    """Checks for list_threads and get_statistics."""

    async def test_list_threads(self) -> bool:
        """Test thread listing resource.

        Returns:
            bool: True if test passed.
        """
        test_name = 'list_threads'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create multiple threads with contexts
            threads = [
                f'{self.test_thread_id}_list_1',
                f'{self.test_thread_id}_list_2',
                f'{self.test_thread_id}_list_3',
            ]

            for thread in threads:
                # Store multiple contexts per thread
                for i in range(3):
                    result = await self.client.call_tool(
                        'store_context',
                        {
                            'thread_id': thread,
                            'source': 'agent' if i % 2 == 0 else 'user',  # Alternate sources
                            'text': f'Message {i} in {thread}',
                        },
                    )
                    data = self._extract_content(result)
                    if not data.get('success'):
                        self.test_results.append((test_name, False, f'Failed to store context for {thread}: {data}'))
                        return False

            # List threads
            thread_list = await self.client.call_tool('list_threads', {})

            list_data = self._extract_content(thread_list)

            # list_threads returns a dict with threads array (no success flag needed)
            if 'threads' not in list_data:
                self.test_results.append((test_name, False, f'Failed to list threads: {list_data}'))
                return False

            # Verify threads are in the list
            listed_threads = list_data['threads']
            thread_ids = [t['thread_id'] for t in listed_threads]

            all_present = all(thread in thread_ids for thread in threads)

            if all_present:
                # Check that threads have correct statistics
                for thread_info in listed_threads:
                    if thread_info['thread_id'] in threads and thread_info.get('entry_count', 0) != 3:
                        error_msg = f"Thread {thread_info['thread_id']} has wrong count: {thread_info.get('entry_count', 0)}"
                        self.test_results.append((test_name, False, error_msg))
                        return False

                self.test_results.append((test_name, True, f'Listed {len(threads)} test threads with correct counts'))
                return True
            self.test_results.append((test_name, False, 'Not all threads present in list'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_list_threads_empty_database(self) -> bool:
        """Test listing threads when no data exists for a specific thread pattern.

        Returns:
            bool: True if test passed.
        """
        test_name = 'List Threads With Filter'
        assert self.client is not None
        try:
            # List threads - no parameters needed (list_threads has no limit/filter params)
            result = await self.client.call_tool(
                'list_threads',
                {},
            )

            data = self._extract_content(result)

            # Should have threads array (may or may not have explicit success flag)
            if 'threads' in data:
                threads = data.get('threads', [])
                total = data.get('total_threads', len(threads))
                self.test_results.append((test_name, True, f'Listed {len(threads)} threads (total: {total})'))
                return True

            # If no threads key, check if there's an error
            if 'error' in data:
                self.test_results.append((test_name, False, f'Error listing threads: {data}'))
                return False

            self.test_results.append((test_name, False, f'Unexpected response format: {data}'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_list_threads_with_populated_database(self) -> bool:
        """Verify list_threads returns accurate thread metadata after storing entries across multiple threads.

        Returns:
            bool: True if test passed.
        """
        test_name = 'list_threads_with_populated_database'
        assert self.client is not None
        try:
            thread_a = f'{self.test_thread_id}_pop_threads_a'
            thread_b = f'{self.test_thread_id}_pop_threads_b'
            thread_c = f'{self.test_thread_id}_pop_threads_c'

            for thread_id in [thread_a, thread_b, thread_c]:
                await self.client.call_tool('store_context', {
                    'thread_id': thread_id, 'source': 'agent',
                    'text': f'Entry for {thread_id}',
                })
            # Store a second entry in thread_a from a different source
            await self.client.call_tool('store_context', {
                'thread_id': thread_a, 'source': 'user',
                'text': 'Second entry in thread A',
            })

            result = await self.client.call_tool('list_threads', {})
            data = self._extract_content(result)

            if 'threads' not in data:
                self.test_results.append((test_name, False, f'Missing threads key: {data}'))
                return False

            threads = data['threads']
            thread_ids_found = {t.get('thread_id') for t in threads}

            for expected in [thread_a, thread_b, thread_c]:
                if expected not in thread_ids_found:
                    self.test_results.append((test_name, False, f'Missing thread: {expected}'))
                    return False

            thread_a_info = next((t for t in threads if t.get('thread_id') == thread_a), None)
            if not thread_a_info or thread_a_info.get('entry_count', 0) < 2:
                self.test_results.append((test_name, False,
                    f'Thread A entry_count wrong: {thread_a_info}'))
                return False

            if thread_a_info.get('source_types', 0) < 2:
                self.test_results.append((test_name, False,
                    f'Thread A source_types should be 2: {thread_a_info}'))
                return False

            self.test_results.append((test_name, True,
                f'Found {len(threads)} threads with correct metadata'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_list_threads_pagination(self) -> bool:
        """Verify list_threads optional limit/offset pagination is bounded, ordered, and backward-compatible.

        Returns:
            bool: True if test passed.
        """
        test_name = 'list_threads_pagination'
        assert self.client is not None
        try:
            # Store entries in four distinct threads with increasing created_at so
            # the most-recent-first ordering is deterministic: _page_d is newest.
            page_threads = [
                f'{self.test_thread_id}_page_a',
                f'{self.test_thread_id}_page_b',
                f'{self.test_thread_id}_page_c',
                f'{self.test_thread_id}_page_d',
            ]
            for thread_id in page_threads:
                store = await self.client.call_tool('store_context', {
                    'thread_id': thread_id, 'source': 'agent',
                    'text': f'Entry for {thread_id}',
                })
                store_data = self._extract_content(store)
                if not store_data.get('success'):
                    self.test_results.append((test_name, False, f'Store failed for {thread_id}: {store_data}'))
                    return False
                # Ensure distinct created_at timestamps for deterministic ordering.
                await asyncio.sleep(0.01)

            # 1) No-arg call returns ALL threads (backward compatible).
            all_data = self._extract_content(await self.client.call_tool('list_threads', {}))
            if 'threads' not in all_data:
                self.test_results.append((test_name, False, f'Missing threads key (no-arg): {all_data}'))
                return False
            all_ids = [t.get('thread_id') for t in all_data['threads']]
            if not all(t in all_ids for t in page_threads):
                self.test_results.append((test_name, False, f'No-arg call did not return all stored threads: {all_ids}'))
                return False

            # 2) limit=2 returns exactly two threads; total_threads is the page count.
            limit_data = self._extract_content(await self.client.call_tool('list_threads', {'limit': 2}))
            limit_threads = limit_data.get('threads', [])
            if len(limit_threads) != 2:
                self.test_results.append((test_name, False, f'limit=2 returned {len(limit_threads)} threads'))
                return False
            if limit_data.get('total_threads') != 2:
                self.test_results.append((test_name, False,
                    f'total_threads should be 2: {limit_data.get("total_threads")}'))
                return False

            # 3) offset paginates without overlap.
            page1 = self._extract_content(
                await self.client.call_tool('list_threads', {'limit': 2, 'offset': 0}),
            ).get('threads', [])
            page2 = self._extract_content(
                await self.client.call_tool('list_threads', {'limit': 2, 'offset': 2}),
            ).get('threads', [])
            page1_ids = {t.get('thread_id') for t in page1}
            page2_ids = {t.get('thread_id') for t in page2}
            if page1_ids & page2_ids:
                self.test_results.append((test_name, False, f'Pages overlap: {page1_ids & page2_ids}'))
                return False

            # 4) Ordering: among the four stored threads the global newest-first list
            #    must yield d, c, b, a (newest created_at first).
            ordered_full = [t.get('thread_id') for t in all_data['threads']]
            our_in_order = [tid for tid in ordered_full if tid in page_threads]
            expected_order = list(reversed(page_threads))
            if our_in_order != expected_order:
                self.test_results.append((test_name, False,
                    f'Ordering wrong. Expected {expected_order}, got {our_in_order}'))
                return False

            self.test_results.append((test_name, True,
                'list_threads pagination bounded, ordered, and backward-compatible'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_get_statistics(self) -> bool:
        """Test statistics resource.

        Returns:
            bool: True if test passed.
        """
        test_name = 'get_statistics'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Get current statistics
            stats = await self.client.call_tool('get_statistics', {})

            stats_data = self._extract_content(stats)

            # Check if we have the statistics fields (no success field needed)
            if 'total_entries' not in stats_data:
                self.test_results.append((test_name, False, f'Failed to get statistics: {stats_data}'))
                return False

            # avg_entries_per_thread MUST serialize as a JSON number, never a
            # string. A regression here would surface as the PostgreSQL Decimal
            # serialization crash ("'X.00' is not of type 'number'").
            avg_entries = stats_data.get('avg_entries_per_thread')
            if not isinstance(avg_entries, (int, float)) or isinstance(avg_entries, bool):
                self.test_results.append(
                    (test_name, False,
                     f'avg_entries_per_thread must be a JSON number, got {avg_entries!r}'),
                )
                return False

            # When embeddings_size_mb is reported it MUST be a non-negative
            # number, carry the estimated flag, and appear alongside
            # database_size_mb (it is reported immediately after it).
            if 'embeddings_size_mb' in stats_data:
                emb_size = stats_data['embeddings_size_mb']
                if not isinstance(emb_size, (int, float)) or isinstance(emb_size, bool) or emb_size < 0:
                    self.test_results.append(
                        (test_name, False,
                         f'embeddings_size_mb must be a non-negative number, got {emb_size!r}'),
                    )
                    return False
                if not isinstance(stats_data.get('embeddings_size_estimated'), bool):
                    self.test_results.append(
                        (test_name, False, 'embeddings_size_estimated must be present and boolean'),
                    )
                    return False

            # The response MUST include the compression sub-block alongside
            # semantic_search/fts/chunking/reranking/summary so MCP clients
            # can verify the active compression configuration at runtime.
            if 'compression' not in stats_data:
                self.test_results.append(
                    (test_name, False, 'missing compression sub-block in stats response'),
                )
                return False
            compression_block = stats_data['compression']
            if 'enabled' not in compression_block:
                self.test_results.append(
                    (test_name, False,
                     f'compression block missing enabled key: {compression_block}'),
                )
                return False

            # Store a new context
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_stats',
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Context for statistics test',
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                },
            )

            result_data = self._extract_content(result)

            if not result_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to store test context'))
                return False

            # Get updated statistics
            new_stats = await self.client.call_tool('get_statistics', {})

            new_stats_data = self._extract_content(new_stats)

            # Check if we have the statistics fields (no success field needed)
            if 'total_entries' not in new_stats_data:
                self.test_results.append((test_name, False, f'Failed to get updated statistics: {new_stats_data}'))
                return False

            # Verify statistics increased
            old_count = stats_data.get('total_entries', 0)
            new_count = new_stats_data.get('total_entries', 0)
            old_images = stats_data.get('total_images', 0)
            new_images = new_stats_data.get('total_images', 0)

            # When compression is enabled AND a context was just stored that
            # produced embeddings, the semantic_search counts must reflect
            # the stored rows. The chunk total comes from
            # embedding_metadata.chunk_count -- the single source-of-truth
            # populated by every write path (fp32 + compressed) on both
            # backends. The compressed write path does NOT populate
            # embedding_chunks (SQLite) and PostgreSQL drops
            # vec_context_embeddings during the compression migration; any
            # nonzero count here proves embedding_metadata is the source.
            new_compression = new_stats_data.get('compression', {})
            new_semantic = new_stats_data.get('semantic_search', {})
            if (
                new_compression.get('enabled')
                and new_compression.get('available')
                and new_semantic.get('available')
            ):
                semantic_embedding_count = new_semantic.get('embedding_count', 0)
                semantic_avg_chunks = new_semantic.get('average_chunks_per_entry', 0.0)
                if semantic_embedding_count <= 0 or semantic_avg_chunks <= 0.0:
                    self.test_results.append(
                        (test_name, False,
                         ('Under compression the stats response shows '
                          f'embedding_count={semantic_embedding_count}, '
                          f'average_chunks_per_entry={semantic_avg_chunks} -- '
                          'embedding_metadata.chunk_count was not the source')),
                    )
                    return False

            if new_count > old_count and new_images > old_images:
                self.test_results.append(
                    (test_name, True, f'Stats updated: entries {old_count}->{new_count}, images {old_images}->{new_images}'),
                )
                return True
            self.test_results.append((test_name, False, 'Statistics not updated correctly'))
            return False

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_statistics_chunking_reranking_info(self) -> bool:
        """Test that get_statistics returns chunking and reranking configuration.

        Returns:
            bool: True if test passed.
        """
        test_name = 'Statistics Chunking Reranking Info'
        assert self.client is not None
        try:
            # Get statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            # Verify chunking section exists
            if 'chunking' not in stats_data:
                self.test_results.append((test_name, False, 'Missing chunking section in statistics'))
                return False

            chunking_info = stats_data['chunking']

            # Verify chunking fields (including the 'available' field for runtime state)
            required_chunking_fields = ['enabled', 'available', 'chunk_size', 'chunk_overlap', 'aggregation']
            for field in required_chunking_fields:
                if field not in chunking_info:
                    self.test_results.append((test_name, False, f'Missing chunking field: {field}'))
                    return False

            # Verify reranking section exists
            if 'reranking' not in stats_data:
                self.test_results.append((test_name, False, 'Missing reranking section in statistics'))
                return False

            reranking_info = stats_data['reranking']

            # Verify reranking fields
            required_reranking_fields = ['enabled', 'available']
            for field in required_reranking_fields:
                if field not in reranking_info:
                    self.test_results.append((test_name, False, f'Missing reranking field: {field}'))
                    return False

            # If reranking is enabled and available, verify provider and model
            is_reranking_active = reranking_info.get('enabled') and reranking_info.get('available')
            if is_reranking_active and ('provider' not in reranking_info or 'model' not in reranking_info):
                self.test_results.append((test_name, False, 'Missing provider/model in enabled reranking'))
                return False

            chunking_status = 'enabled' if chunking_info.get('enabled') else 'disabled'
            reranking_status = 'available' if reranking_info.get('available') else 'unavailable'
            self.test_results.append(
                (test_name, True, f'chunking={chunking_status}, reranking={reranking_status}'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_statistics_summary_info(self) -> bool:
        """Test that get_statistics returns summary generation configuration.

        Returns:
            bool: True if test passed.
        """
        test_name = 'Statistics Summary Info'
        assert self.client is not None
        try:
            # Get statistics
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            # Verify summary section exists
            if 'summary' not in stats_data:
                self.test_results.append((test_name, False, 'Missing summary section in statistics'))
                return False

            summary_info = stats_data['summary']

            # Verify required fields
            required_fields = ['enabled', 'available']
            for field in required_fields:
                if field not in summary_info:
                    self.test_results.append((test_name, False, f'Missing summary field: {field}'))
                    return False

            # If summary is enabled and available, verify additional fields
            is_summary_active = summary_info.get('enabled') and summary_info.get('available')
            if is_summary_active:
                active_fields = ['provider', 'model', 'summary_count', 'coverage_percentage', 'min_content_length']
                for field in active_fields:
                    if field not in summary_info:
                        self.test_results.append(
                            (test_name, False, f'Missing field in active summary: {field}'),
                        )
                        return False

            summary_status = 'available' if summary_info.get('available') else 'unavailable'
            self.test_results.append(
                (test_name, True, f'summary={summary_status}'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
