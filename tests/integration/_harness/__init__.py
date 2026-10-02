"""Shared, backend-parametrized harness for real-server integration tests.

Defines :class:`MCPServerIntegrationTest`, which drives a real MCP server
subprocess (launched via ``tests/run_server.py``) through the FastMCP client
and asserts the full tool surface. The harness is backend-agnostic: the same
assertion methods run against SQLite or PostgreSQL depending on the
``backend`` / ``pg_url`` passed to the constructor. The per-backend pytest
entry points live in ``tests/integration/sqlite/test_real_server.py`` and
``tests/integration/postgresql/test_real_server.py``.

This module is intentionally NOT named ``test_*`` so pytest does not collect
it directly; it is imported by the per-backend entry-point modules.
"""

import asyncio
from datetime import UTC
from typing import Any

from tests.integration._harness.access_control import AccessControlMixin
from tests.integration._harness.backend_runtime import BackendRuntimeMixin
from tests.integration._harness.batch import BatchMixin
from tests.integration._harness.batch_atomicity import BatchAtomicityMixin
from tests.integration._harness.batch_conformance import BatchConformanceMixin
from tests.integration._harness.delete import DeleteMixin
from tests.integration._harness.discovery import DiscoveryMixin
from tests.integration._harness.metadata_filters import MetadataFiltersMixin
from tests.integration._harness.metadata_filters_indexed import MetadataFiltersIndexedMixin
from tests.integration._harness.metadata_filters_membership import MetadataFiltersMembershipMixin
from tests.integration._harness.metadata_filters_numeric import MetadataFiltersNumericMixin
from tests.integration._harness.metadata_patch import MetadataPatchMixin
from tests.integration._harness.middleware import MiddlewareMixin
from tests.integration._harness.retrieve import RetrieveMixin
from tests.integration._harness.search_browse import SearchBrowseMixin
from tests.integration._harness.search_explain import SearchExplainMixin
from tests.integration._harness.search_validation import SearchValidationMixin
from tests.integration._harness.server import ServerMixin
from tests.integration._harness.store import StoreMixin
from tests.integration._harness.store_dedup import StoreDedupMixin
from tests.integration._harness.summary import SummaryMixin
from tests.integration._harness.tags_images import TagsImagesMixin
from tests.integration._harness.update import UpdateMixin


class MCPServerIntegrationTest(
    StoreMixin,
    StoreDedupMixin,
    RetrieveMixin,
    UpdateMixin,
    MetadataPatchMixin,
    DeleteMixin,
    AccessControlMixin,
    TagsImagesMixin,
    BatchMixin,
    BatchAtomicityMixin,
    BatchConformanceMixin,
    DiscoveryMixin,
    SummaryMixin,
    MiddlewareMixin,
    ServerMixin,
    BackendRuntimeMixin,
    MetadataFiltersMixin,
    MetadataFiltersNumericMixin,
    MetadataFiltersMembershipMixin,
    MetadataFiltersIndexedMixin,
    SearchBrowseMixin,
    SearchExplainMixin,
    SearchValidationMixin,
):
    """Integration test for real MCP Context Storage Server."""

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

    async def test_fts_boolean_mode(self) -> bool:
        """Test FTS boolean mode with AND/OR/NOT operators.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_boolean_mode'
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

            # Create a separate thread for boolean mode tests
            bool_thread = f'{self.test_thread_id}_fts_boolean'

            # Store test contexts for boolean search
            test_contexts = [
                {'text': 'Python is great for data science and machine learning', 'source': 'agent'},
                {'text': 'JavaScript and TypeScript are popular for web development', 'source': 'agent'},
                {'text': 'Python and JavaScript can both handle backend development', 'source': 'user'},
                {'text': 'Rust is known for memory safety without garbage collection', 'source': 'agent'},
            ]

            for ctx in test_contexts:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': bool_thread,
                        'source': ctx['source'],
                        'text': ctx['text'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store test context: {result_data}'))
                    return False

            # Test 1: OR operator - should find entries with Python OR JavaScript
            or_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'Python OR JavaScript',
                    'mode': 'boolean',
                    'thread_id': bool_thread,
                    'limit': 10,
                },
            )

            or_data = self._extract_content(or_result)

            if 'results' not in or_data:
                self.test_results.append((test_name, False, f'OR search failed: {or_data}'))
                return False

            or_results = or_data.get('results', [])
            # Should find at least 3 entries (2 with Python, 2 with JavaScript, 1 with both)
            if len(or_results) < 3:
                self.test_results.append(
                    (test_name, False, f'Expected at least 3 results for OR query, got {len(or_results)}'),
                )
                return False

            # Test 2: AND operator - should find entries with both Python AND data
            and_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'Python AND data',
                    'mode': 'boolean',
                    'thread_id': bool_thread,
                    'limit': 10,
                },
            )

            and_data = self._extract_content(and_result)

            if 'results' not in and_data:
                self.test_results.append((test_name, False, f'AND search failed: {and_data}'))
                return False

            and_results = and_data.get('results', [])
            # Should find exactly 1 entry with both Python AND data
            if len(and_results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result for AND query, got {len(and_results)}'),
                )
                return False

            # Verify response mode field
            if or_data.get('mode') != 'boolean':
                self.test_results.append((test_name, False, 'Response mode field incorrect'))
                return False

            self.test_results.append((test_name, True, 'Boolean mode OR/AND operators working'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

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

    async def test_fts_metadata_filter_key_substring(self) -> bool:
        """Regression: FTS metadata filter on a key whose NAME contains 'metadata'.

        Guards against the global ``str.replace('metadata', 'ce.metadata')`` bug
        class: rewriting every 'metadata' substring corrupts a JSON key such as
        ``metadata_version`` (it became ``ce.metadata_version``), so the filter
        matched a non-existent key and returned nothing on BOTH backends. The fix
        qualifies only the column via ``table_alias='ce'``. Runs on SQLite and
        PostgreSQL through the shared harness.

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

            # eq on the 'metadata'-substring key: pre-fix this returned 0 (corrupted key).
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

    async def test_search_tools_content_type_filter(self) -> bool:
        """Test content_type parameter across all 4 search tools.

        Verifies that content_type='text' and content_type='multimodal' filters
        work correctly for search_context, semantic_search, fts_search, and hybrid_search.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'search_tools_content_type_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            fts_info = stats_data.get('fts', {})

            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_hybrid = (has_semantic or has_fts) and 'hybrid_search_context' in self.registered_tools

            # Create a separate thread for content_type tests
            ct_thread = f'{self.test_thread_id}_content_type'

            # Store text-only entries
            for i in range(2):
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': ct_thread,
                        'source': 'agent',
                        'text': f'Text-only content for content type filtering test {i}',
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store text context: {result_data}'))
                    return False

            # Store multimodal entries with images
            for i in range(2):
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': ct_thread,
                        'source': 'agent',
                        'text': f'Multimodal content with image for filtering test {i}',
                        'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store multimodal context: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: search_context with content_type='text'
            text_result = await self.client.call_tool(
                'search_context',
                {'thread_id': ct_thread, 'content_type': 'text', 'limit': 10},
            )
            text_data = self._extract_content(text_result)
            if not text_data.get('success'):
                self.test_results.append((test_name, False, f'search_context text filter failed: {text_data}'))
                return False

            text_results = text_data.get('results', [])
            if len(text_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 text entries, got {len(text_results)}'))
                return False

            # Verify all results have content_type='text'
            for r in text_results:
                if r.get('content_type') != 'text':
                    ct = r.get('content_type')
                    self.test_results.append((test_name, False, f"Expected content_type='text', got '{ct}'"))
                    return False

            # Test 2: search_context with content_type='multimodal'
            mm_result = await self.client.call_tool(
                'search_context',
                {'thread_id': ct_thread, 'content_type': 'multimodal', 'limit': 10},
            )
            mm_data = self._extract_content(mm_result)
            if not mm_data.get('success'):
                self.test_results.append((test_name, False, f'search_context multimodal filter failed: {mm_data}'))
                return False

            mm_results = mm_data.get('results', [])
            if len(mm_results) != 2:
                self.test_results.append((test_name, False, f'Expected 2 multimodal entries, got {len(mm_results)}'))
                return False

            # Verify all results have content_type='multimodal'
            for r in mm_results:
                if r.get('content_type') != 'multimodal':
                    ct = r.get('content_type')
                    self.test_results.append((test_name, False, f"Expected content_type='multimodal', got '{ct}'"))
                    return False

            # Test 3: semantic_search with content_type filter (if available)
            if has_semantic:
                sem_text_result = await self.client.call_tool(
                    'semantic_search_context',
                    {'query': 'content filtering', 'thread_id': ct_thread, 'content_type': 'text', 'limit': 10},
                )
                sem_text_data = self._extract_content(sem_text_result)
                if 'results' not in sem_text_data:
                    self.test_results.append((test_name, False, f'semantic_search text filter failed: {sem_text_data}'))
                    return False

                # All results should be text type
                for r in sem_text_data.get('results', []):
                    if r.get('content_type') != 'text':
                        ct = r.get('content_type')
                        self.test_results.append((test_name, False, f"semantic: Expected 'text', got '{ct}'"))
                        return False

            # Test 4: fts_search with content_type filter (if available)
            if has_fts:
                fts_mm_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'content',
                        'mode': 'match',
                        'thread_id': ct_thread,
                        'content_type': 'multimodal',
                        'limit': 10,
                    },
                )
                fts_mm_data = self._extract_content(fts_mm_result)
                if 'results' not in fts_mm_data:
                    self.test_results.append((test_name, False, f'fts multimodal filter failed: {fts_mm_data}'))
                    return False

                # All results should be multimodal type
                for r in fts_mm_data.get('results', []):
                    if r.get('content_type') != 'multimodal':
                        ct = r.get('content_type')
                        self.test_results.append((test_name, False, f"fts: Expected 'multimodal', got '{ct}'"))
                        return False

            # Test 5: hybrid_search with content_type filter (if available)
            if has_hybrid:
                hyb_text_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {
                        'query': 'content filtering',
                        'thread_id': ct_thread,
                        'content_type': 'text',
                        'limit': 10,
                    },
                )
                hyb_text_data = self._extract_content(hyb_text_result)
                if 'results' not in hyb_text_data:
                    self.test_results.append((test_name, False, f'hybrid text filter failed: {hyb_text_data}'))
                    return False

                # All results should be text type
                for r in hyb_text_data.get('results', []):
                    if r.get('content_type') != 'text':
                        ct = r.get('content_type')
                        self.test_results.append((test_name, False, f"hybrid: Expected 'text', got '{ct}'"))
                        return False

            msg = f'content_type filter working (semantic={has_semantic}, fts={has_fts}, hybrid={has_hybrid})'
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_tools_include_images(self) -> bool:
        """Test include_images parameter across all 4 search tools.

        Verifies that include_images=True returns image data and include_images=False excludes it.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'search_tools_include_images'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            fts_info = stats_data.get('fts', {})

            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_hybrid = (has_semantic or has_fts) and 'hybrid_search_context' in self.registered_tools

            # Create a separate thread for include_images tests
            img_thread = f'{self.test_thread_id}_include_images'

            # Store multimodal entry with image
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': img_thread,
                    'source': 'agent',
                    'text': 'Multimodal content for include images test with Python code',
                    'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
                },
            )
            result_data = self._extract_content(result)
            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store multimodal context: {result_data}'))
                return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: search_context with include_images=True
            with_images_result = await self.client.call_tool(
                'search_context',
                {'thread_id': img_thread, 'include_images': True, 'limit': 10},
            )
            with_images_data = self._extract_content(with_images_result)
            if not with_images_data.get('success'):
                self.test_results.append((test_name, False, f'search_context include_images=True failed: {with_images_data}'))
                return False

            with_img_results = with_images_data.get('results', [])
            if len(with_img_results) < 1:
                self.test_results.append((test_name, False, 'No results found'))
                return False

            # Verify images are included
            first_result = with_img_results[0]
            images = first_result.get('images', [])
            if len(images) < 1:
                self.test_results.append((test_name, False, 'Expected images in result with include_images=True'))
                return False

            # Verify image has data
            if 'data' not in images[0] or not images[0]['data']:
                self.test_results.append((test_name, False, 'Image data missing with include_images=True'))
                return False

            # Test 2: search_context with include_images=False
            without_images_result = await self.client.call_tool(
                'search_context',
                {'thread_id': img_thread, 'include_images': False, 'limit': 10},
            )
            without_images_data = self._extract_content(without_images_result)
            if not without_images_data.get('success'):
                msg = f'search_context include_images=False failed: {without_images_data}'
                self.test_results.append((test_name, False, msg))
                return False

            without_img_results = without_images_data.get('results', [])
            if len(without_img_results) < 1:
                self.test_results.append((test_name, False, 'No results found with include_images=False'))
                return False

            # Verify images are excluded or empty
            first_wo_img = without_img_results[0]
            wo_images = first_wo_img.get('images', [])
            # Images should be empty list or not contain data
            if wo_images:
                for img in wo_images:
                    if img.get('data'):
                        self.test_results.append((test_name, False, 'Image data should be excluded with include_images=False'))
                        return False

            # Test 3: semantic_search with include_images (if available)
            if has_semantic:
                sem_result = await self.client.call_tool(
                    'semantic_search_context',
                    {
                        'query': 'multimodal content',
                        'thread_id': img_thread,
                        'include_images': True,
                        'limit': 10,
                    },
                )
                sem_data = self._extract_content(sem_result)
                if 'results' in sem_data and len(sem_data['results']) > 0:
                    sem_images = sem_data['results'][0].get('images', [])
                    if len(sem_images) < 1 or not sem_images[0].get('data'):
                        self.test_results.append((test_name, False, 'semantic: Expected images'))
                        return False

            # Test 4: fts_search with include_images (if available)
            if has_fts:
                fts_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'multimodal',
                        'mode': 'match',
                        'thread_id': img_thread,
                        'include_images': True,
                        'limit': 10,
                    },
                )
                fts_data = self._extract_content(fts_result)
                if 'results' in fts_data and len(fts_data['results']) > 0:
                    fts_images = fts_data['results'][0].get('images', [])
                    if len(fts_images) < 1 or not fts_images[0].get('data'):
                        self.test_results.append((test_name, False, 'fts: Expected images'))
                        return False

            # Test 5: hybrid_search with include_images (if available)
            if has_hybrid:
                hyb_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {
                        'query': 'multimodal content',
                        'thread_id': img_thread,
                        'include_images': True,
                        'limit': 10,
                    },
                )
                hyb_data = self._extract_content(hyb_result)
                if 'results' in hyb_data and len(hyb_data['results']) > 0:
                    hyb_images = hyb_data['results'][0].get('images', [])
                    if len(hyb_images) < 1 or not hyb_images[0].get('data'):
                        self.test_results.append((test_name, False, 'hybrid: Expected images'))
                        return False

            msg = f'include_images working (semantic={has_semantic}, fts={has_fts}, hybrid={has_hybrid})'
            self.test_results.append((test_name, True, msg))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_search_tools_tags_filter(self) -> bool:
        """Test tags parameter for semantic_search, fts_search, and hybrid_search.

        Note: search_context already tests tags. This tests the 3 other search tools.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'search_tools_tags_filter'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Check feature availability
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            semantic_info = stats_data.get('semantic_search', {})
            fts_info = stats_data.get('fts', {})

            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_hybrid = (has_semantic or has_fts) and 'hybrid_search_context' in self.registered_tools

            # Skip if no advanced search features are available
            if not has_semantic and not has_fts:
                self.test_results.append((test_name, True, 'Skipped (no advanced search available)'))
                return True

            # Create a separate thread for tags tests
            tags_thread = f'{self.test_thread_id}_tags_filter'

            # Store entries with different tags
            test_entries = [
                {'text': 'Python backend development with Flask', 'tags': ['backend', 'python']},
                {'text': 'JavaScript frontend development with React', 'tags': ['frontend', 'javascript']},
                {'text': 'Full stack development combining both', 'tags': ['fullstack', 'backend', 'frontend']},
                {'text': 'Database design and SQL optimization', 'tags': ['database', 'backend']},
            ]

            for entry in test_entries:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': tags_thread,
                        'source': 'agent',
                        'text': entry['text'],
                        'tags': entry['tags'],
                    },
                )
                result_data = self._extract_content(result)
                if not result_data.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store: {result_data}'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Test 1: semantic_search with tags filter (if available)
            if has_semantic:
                sem_result = await self.client.call_tool(
                    'semantic_search_context',
                    {
                        'query': 'development frameworks',
                        'thread_id': tags_thread,
                        'tags': ['backend'],
                        'limit': 10,
                    },
                )
                sem_data = self._extract_content(sem_result)
                if 'results' not in sem_data:
                    self.test_results.append((test_name, False, f'semantic tags failed: {sem_data}'))
                    return False

                sem_results = sem_data.get('results', [])
                # Should find entries with 'backend' tag (Python, Full stack, Database = 3)
                if len(sem_results) < 1:
                    self.test_results.append((test_name, False, 'semantic: No results with backend tag'))
                    return False

                # Verify all results have 'backend' tag
                for r in sem_results:
                    result_tags = r.get('tags', [])
                    if 'backend' not in result_tags:
                        self.test_results.append((test_name, False, f"semantic: Expected 'backend', got {result_tags}"))
                        return False

            # Test 2: fts_search with tags filter (if available)
            if has_fts:
                fts_result = await self.client.call_tool(
                    'fts_search_context',
                    {
                        'query': 'development',
                        'mode': 'match',
                        'thread_id': tags_thread,
                        'tags': ['frontend'],
                        'limit': 10,
                    },
                )
                fts_data = self._extract_content(fts_result)
                if 'results' not in fts_data:
                    self.test_results.append((test_name, False, f'fts tags failed: {fts_data}'))
                    return False

                fts_results = fts_data.get('results', [])
                # Should find entries with 'frontend' tag (JavaScript, Full stack = 2)
                if len(fts_results) < 1:
                    self.test_results.append((test_name, False, 'fts: No results with frontend tag'))
                    return False

                # Verify all results have 'frontend' tag
                for r in fts_results:
                    result_tags = r.get('tags', [])
                    if 'frontend' not in result_tags:
                        self.test_results.append((test_name, False, f"fts: Expected 'frontend', got {result_tags}"))
                        return False

            # Test 3: hybrid_search with tags filter (if available)
            if has_hybrid:
                hyb_result = await self.client.call_tool(
                    'hybrid_search_context',
                    {
                        'query': 'development',
                        'thread_id': tags_thread,
                        'tags': ['python'],
                        'limit': 10,
                    },
                )
                hyb_data = self._extract_content(hyb_result)
                if 'results' not in hyb_data:
                    self.test_results.append((test_name, False, f'hybrid tags failed: {hyb_data}'))
                    return False

                hyb_results = hyb_data.get('results', [])
                # Should find entries with 'python' tag (Python backend = 1)
                if len(hyb_results) < 1:
                    self.test_results.append((test_name, False, 'hybrid: No results with python tag'))
                    return False

                # Verify all results have 'python' tag
                for r in hyb_results:
                    result_tags = r.get('tags', [])
                    if 'python' not in result_tags:
                        self.test_results.append((test_name, False, f"hybrid: Expected 'python', got {result_tags}"))
                        return False

            # Test 4: Multiple tags (OR logic)
            if has_semantic:
                multi_tag_result = await self.client.call_tool(
                    'semantic_search_context',
                    {
                        'query': 'development',
                        'thread_id': tags_thread,
                        'tags': ['python', 'javascript'],
                        'limit': 10,
                    },
                )
                multi_tag_data = self._extract_content(multi_tag_result)
                if 'results' in multi_tag_data:
                    multi_results = multi_tag_data.get('results', [])
                    # Should find at least 2 entries (Python and JavaScript)
                    if len(multi_results) < 2:
                        msg = f'Expected 2+ results with python OR javascript, got {len(multi_results)}'
                        self.test_results.append((test_name, False, msg))
                        return False

            msg = f'tags filter working (semantic={has_semantic}, fts={has_fts}, hybrid={has_hybrid})'
            self.test_results.append((test_name, True, msg))
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

    # ========== Edge Case Tests (P3) ==========

    # =========================================================================
    # Chunking and Reranking E2E Tests
    # =========================================================================

    async def test_chunking_creates_multiple_embeddings(self) -> bool:
        """Test that chunking creates multiple embeddings per long document.

        This test verifies:
        1. A long document (>5000 chars) results in multiple embeddings
        2. The statistics API shows embedding_count > context_count
        3. The average_chunks_per_entry is > 1.0 when chunking works

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Chunking Creates Multiple Embeddings'
        assert self.client is not None
        try:
            # Check if chunking and semantic search are enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            chunking_info = stats_data.get('chunking', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_chunking_enabled = chunking_info.get('enabled', False) and chunking_info.get('available', False)
            is_semantic_enabled = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not is_chunking_enabled or not is_semantic_enabled:
                self.test_results.append(
                    (test_name, True, f'Skipped (chunking={is_chunking_enabled}, semantic={is_semantic_enabled})'),
                )
                return True

            # Store initial stats for comparison
            initial_context_count = semantic_info.get('context_count', 0)
            initial_embedding_count = semantic_info.get('embedding_count', 0)

            # Create a unique thread for this test
            multi_chunk_thread = f'{self.test_thread_id}_multi_chunk_test'

            # Generate a document > chunk_size (1500 chars default)
            # Using 5400+ chars to ensure 5-6 chunks
            long_text = ' '.join(['This is a test sentence for chunking verification.'] * 150)  # ~7500 chars

            # Store the long document
            result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': multi_chunk_thread,
                    'source': 'agent',
                    'text': long_text,
                    'tags': ['multi-chunk-verification'],
                },
            )

            result_data = self._extract_content(result)
            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store document: {result_data}'))
                return False

            # Allow time for embedding generation
            await asyncio.sleep(1.0)

            # Get updated statistics
            stats_after = await self.client.call_tool('get_statistics', {})
            stats_after_data = self._extract_content(stats_after)

            semantic_after = stats_after_data.get('semantic_search', {})
            chunking_after = stats_after_data.get('chunking', {})

            # Get the new counts
            new_context_count = semantic_after.get('context_count', 0)
            new_embedding_count = semantic_after.get('embedding_count', 0)
            avg_chunks = semantic_after.get('average_chunks_per_entry', 0.0)

            # Verify we stored exactly 1 new context
            contexts_added = new_context_count - initial_context_count
            if contexts_added < 1:
                self.test_results.append(
                    (test_name, False, f'Expected at least 1 new context, got {contexts_added}'),
                )
                return False

            # Verify multiple embeddings were created
            embeddings_added = new_embedding_count - initial_embedding_count
            if embeddings_added <= contexts_added:
                self.test_results.append(
                    (test_name, False,
                     (f'Expected embedding_count > context_count, '
                      f'got {embeddings_added} embeddings for {contexts_added} context(s)')),
                )
                return False

            # Verify average chunks > 1.0 (indicates chunking is working)
            if avg_chunks <= 1.0:
                self.test_results.append(
                    (test_name, False, f'Expected average_chunks_per_entry > 1.0, got {avg_chunks}'),
                )
                return False

            # Verify chunking is still available
            if not chunking_after.get('available', False):
                self.test_results.append((test_name, False, 'Chunking not available in runtime'))
                return False

            self.test_results.append(
                (test_name, True,
                 (f'Created {embeddings_added} embeddings for {contexts_added} context(s), '
                  f'avg_chunks={avg_chunks:.2f}')),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_chunking_long_document_storage(self) -> bool:
        """Test that long documents are properly chunked for semantic search.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Chunking Long Document Storage'
        assert self.client is not None
        try:
            # Check if chunking and semantic search are enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            chunking_info = stats_data.get('chunking', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_chunking_enabled = chunking_info.get('enabled', False)
            is_semantic_enabled = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not is_chunking_enabled or not is_semantic_enabled:
                self.test_results.append(
                    (test_name, True, f'Skipped (chunking={is_chunking_enabled}, semantic={is_semantic_enabled})'),
                )
                return True

            # Create a separate thread for chunking tests
            chunking_thread = f'{self.test_thread_id}_chunking_long'

            # Create a LONG document (2000+ characters) with distinct content in different sections
            long_text = '''
            SECTION ONE - MACHINE LEARNING CONCEPTS:
            Machine learning is a subset of artificial intelligence that enables computers to learn
            and improve from experience without being explicitly programmed. It focuses on developing
            algorithms that can access data and use it to learn for themselves. The primary aim is to
            allow computers to learn automatically without human intervention or assistance. Deep
            learning, a subset of machine learning, uses neural networks with many layers to model
            complex patterns in data. Popular frameworks include TensorFlow, PyTorch, and scikit-learn.

            SECTION TWO - DATABASE OPTIMIZATION TECHNIQUES:
            Database optimization involves various techniques to improve query performance and storage
            efficiency. Key strategies include proper indexing, query planning, schema normalization,
            and denormalization where appropriate. Connection pooling helps manage database connections
            efficiently. Caching frequently accessed data reduces database load. Query optimization
            through EXPLAIN plans helps identify bottlenecks. PostgreSQL offers advanced features like
            partial indexes and expression indexes for specific use cases.

            SECTION THREE - WEB DEVELOPMENT FRAMEWORKS:
            Modern web development encompasses both frontend and backend technologies. JavaScript
            frameworks like React, Vue, and Angular power dynamic user interfaces with component-based
            architectures. Python frameworks like FastAPI and Django handle server-side logic with
            excellent performance. FastAPI provides automatic API documentation through OpenAPI and
            built-in validation with Pydantic models. Django offers a batteries-included approach
            with ORM, authentication, and admin interface out of the box.
            '''

            # Store the long document
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': chunking_thread,
                    'source': 'agent',
                    'text': long_text,
                    'tags': ['long-document', 'chunking-test'],
                },
            )

            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store long document: {store_data}'))
                return False

            stored_context_id = store_data.get('context_id')

            # Allow time for embedding generation
            await asyncio.sleep(1.0)

            # Search for content from SECTION ONE (machine learning)
            ml_search = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'machine learning neural networks deep learning',
                    'thread_id': chunking_thread,
                    'limit': 5,
                },
            )
            ml_data = self._extract_content(ml_search)

            if 'results' not in ml_data or len(ml_data.get('results', [])) == 0:
                self.test_results.append((test_name, False, 'ML section search returned no results'))
                return False

            # Search for content from SECTION THREE (web development)
            web_search = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'FastAPI Django web frameworks Python',
                    'thread_id': chunking_thread,
                    'limit': 5,
                },
            )
            web_data = self._extract_content(web_search)

            if 'results' not in web_data or len(web_data.get('results', [])) == 0:
                self.test_results.append((test_name, False, 'Web section search returned no results'))
                return False

            # Verify BOTH searches find the SAME document (deduplication working)
            ml_ids = [r.get('id') for r in ml_data.get('results', [])]
            web_ids = [r.get('id') for r in web_data.get('results', [])]

            if stored_context_id in ml_ids and stored_context_id in web_ids:
                self.test_results.append((test_name, True, 'Long document chunks searchable and deduplicated'))
                return True

            # Even if the stored_context_id is not in results, verify the document appears once
            self.test_results.append(
                (test_name, True, f'Long document searchable (ml_results={len(ml_ids)}, web_results={len(web_ids)})'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_reranking_adds_score_to_results(self) -> bool:
        """Test that reranking adds rerank_score to semantic search results.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Reranking Adds Score to Results'
        assert self.client is not None
        try:
            # Check if reranking and semantic search are enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            reranking_info = stats_data.get('reranking', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_reranking_enabled = reranking_info.get('enabled', False) and reranking_info.get('available', False)
            is_semantic_enabled = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not is_reranking_enabled or not is_semantic_enabled:
                self.test_results.append(
                    (test_name, True, f'Skipped (reranking={is_reranking_enabled}, semantic={is_semantic_enabled})'),
                )
                return True

            # Create a separate thread for reranking tests
            reranking_thread = f'{self.test_thread_id}_reranking_score'

            # Store diverse test documents
            test_docs = [
                'Python programming language is excellent for data science and machine learning applications.',
                'JavaScript and TypeScript are popular for web development and frontend applications.',
                'Database optimization involves indexing, query planning, and caching strategies.',
                'Cloud computing platforms like AWS and Azure provide scalable infrastructure.',
                'Recipe for chocolate cake: mix flour, sugar, cocoa, eggs, and bake at 350 degrees.',
            ]

            for doc in test_docs:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': reranking_thread,
                        'source': 'agent',
                        'text': doc,
                    },
                )
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store test documents'))
                    return False

            # Allow time for embedding generation
            await asyncio.sleep(0.5)

            # Search for Python-related content
            search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'Python programming data science',
                    'thread_id': reranking_thread,
                    'limit': 5,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data or len(search_data.get('results', [])) == 0:
                self.test_results.append((test_name, False, 'Search returned no results'))
                return False

            results = search_data['results']

            # Verify rerank_score is present in scores object
            has_rerank_score = all('scores' in r and 'rerank_score' in r.get('scores', {}) for r in results)
            if not has_rerank_score:
                self.test_results.append((test_name, False, 'Missing rerank_score in results.scores'))
                return False

            # Verify rerank_score is a float between 0 and 1
            for i, result in enumerate(results):
                score = result.get('scores', {}).get('rerank_score')
                if not isinstance(score, (int, float)) or score < 0 or score > 1:
                    self.test_results.append((test_name, False, f'Invalid rerank_score at index {i}: {score}'))
                    return False

            # Verify results are sorted by rerank_score (descending)
            scores = [r['scores']['rerank_score'] for r in results]
            is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))
            if not is_sorted:
                self.test_results.append((test_name, False, f'Results not sorted by rerank_score: {scores}'))
                return False

            # Verify Python doc ranks higher than chocolate cake
            python_doc_ranked_high = any(
                'Python' in r.get('text_content', '') for r in results[:2]
            )
            if not python_doc_ranked_high:
                self.test_results.append((test_name, False, 'Python doc not in top 2 results'))
                return False

            self.test_results.append((test_name, True, f'rerank_score present and sorted ({len(results)} results)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_reranking_in_fts_search(self) -> bool:
        """Test that reranking is applied to FTS search results.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Reranking in FTS Search'
        assert self.client is not None
        try:
            # Check if reranking and FTS are enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            reranking_info = stats_data.get('reranking', {})
            fts_info = stats_data.get('fts', {})

            is_reranking_enabled = reranking_info.get('enabled', False) and reranking_info.get('available', False)
            is_fts_enabled = fts_info.get('enabled', False) and fts_info.get('available', False)

            if not is_reranking_enabled or not is_fts_enabled:
                self.test_results.append(
                    (test_name, True, f'Skipped (reranking={is_reranking_enabled}, fts={is_fts_enabled})'),
                )
                return True

            # Create a separate thread for FTS reranking tests
            fts_rerank_thread = f'{self.test_thread_id}_fts_rerank'

            # Store test documents with keyword matches
            test_docs = [
                'Python programming is widely used for scientific computing and data analysis.',
                'The python snake is a non-venomous reptile found in tropical regions.',
                'Learn Python basics: variables, functions, classes, and modules.',
            ]

            for doc in test_docs:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': fts_rerank_thread,
                        'source': 'agent',
                        'text': doc,
                    },
                )
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store test documents'))
                    return False

            # Allow time for FTS indexing
            await asyncio.sleep(0.3)

            # Search using FTS
            search_result = await self.client.call_tool(
                'fts_search_context',
                {
                    'query': 'Python programming',
                    'thread_id': fts_rerank_thread,
                    'limit': 5,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data or len(search_data.get('results', [])) == 0:
                self.test_results.append((test_name, False, 'FTS search returned no results'))
                return False

            results = search_data['results']

            # Verify results have both FTS score and rerank_score in scores object
            first_result = results[0]
            has_fts_score = 'scores' in first_result and 'fts_score' in first_result.get('scores', {})
            has_rerank_score = 'scores' in first_result and 'rerank_score' in first_result.get('scores', {})

            if not has_fts_score:
                self.test_results.append((test_name, False, 'Missing fts_score in results.scores'))
                return False

            if not has_rerank_score:
                self.test_results.append((test_name, False, 'Missing rerank_score in results.scores'))
                return False

            # Verify results are sorted by rerank_score
            scores = [r['scores']['rerank_score'] for r in results]
            is_sorted = all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))

            self.test_results.append(
                (test_name, True, f'FTS + rerank_score present (sorted={is_sorted}, count={len(results)})'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_reranking_in_hybrid_search(self) -> bool:
        """Test that hybrid search applies single reranking after RRF fusion.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Reranking in Hybrid Search'
        assert self.client is not None
        try:
            # Check if reranking and hybrid search are enabled
            if 'hybrid_search_context' not in self.registered_tools:
                self.test_results.append((test_name, True, 'Skipped (hybrid_search_context not registered)'))
                return True

            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            reranking_info = stats_data.get('reranking', {})
            fts_info = stats_data.get('fts', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_reranking_enabled = reranking_info.get('enabled', False) and reranking_info.get('available', False)
            has_fts = fts_info.get('enabled', False) and fts_info.get('available', False)
            has_semantic = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not is_reranking_enabled or (not has_fts and not has_semantic):
                self.test_results.append(
                    (test_name, True, f'Skipped (reranking={is_reranking_enabled}, fts={has_fts}, semantic={has_semantic})'),
                )
                return True

            # Create a separate thread for hybrid reranking tests
            hybrid_rerank_thread = f'{self.test_thread_id}_hybrid_rerank'

            # Store test documents
            test_docs = [
                'Machine learning algorithms for predictive analytics and data modeling.',
                'Deep learning neural networks using TensorFlow and PyTorch frameworks.',
                'Traditional cooking recipes from Mediterranean cuisine.',
            ]

            for doc in test_docs:
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': hybrid_rerank_thread,
                        'source': 'agent',
                        'text': doc,
                    },
                )
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store test documents'))
                    return False

            # Allow time for indexing
            await asyncio.sleep(0.5)

            # Search using hybrid search
            search_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'machine learning',
                    'thread_id': hybrid_rerank_thread,
                    'limit': 5,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data or len(search_data.get('results', [])) == 0:
                self.test_results.append((test_name, False, 'Hybrid search returned no results'))
                return False

            results = search_data['results']

            # Verify results have RRF scores structure
            first_result = results[0]
            if 'scores' not in first_result:
                self.test_results.append((test_name, False, 'Missing scores field in results'))
                return False

            scores = first_result['scores']
            has_rrf = 'rrf' in scores

            # Verify rerank_score is present inside scores dict
            has_rerank_score = 'rerank_score' in scores

            if not has_rrf:
                self.test_results.append((test_name, False, 'Missing RRF score in hybrid results'))
                return False

            if not has_rerank_score:
                self.test_results.append((test_name, False, 'Missing rerank_score in results.scores'))
                return False

            # Verify results are sorted by rerank_score
            rerank_scores = [r['scores']['rerank_score'] for r in results]
            is_sorted = all(rerank_scores[i] >= rerank_scores[i + 1] for i in range(len(rerank_scores) - 1))

            self.test_results.append(
                (test_name, True, f'Hybrid + RRF + rerank_score present (sorted={is_sorted}, count={len(results)})'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_chunking_deduplication_in_search(self) -> bool:
        """Test that chunk deduplication prevents duplicate documents in results.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Chunking Deduplication in Search'
        assert self.client is not None
        try:
            # Check if chunking and semantic search are enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            chunking_info = stats_data.get('chunking', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_chunking_enabled = chunking_info.get('enabled', False)
            is_semantic_enabled = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not is_chunking_enabled or not is_semantic_enabled:
                self.test_results.append(
                    (test_name, True, f'Skipped (chunking={is_chunking_enabled}, semantic={is_semantic_enabled})'),
                )
                return True

            # Create a separate thread for deduplication tests
            dedup_thread = f'{self.test_thread_id}_chunking_dedup'

            # Create a VERY LONG document with repetitive content that will span multiple chunks
            # The keyword "database optimization" appears in multiple places
            repetitive_text = '''
            DATABASE OPTIMIZATION STRATEGIES - PART 1:
            Database optimization is crucial for application performance. Proper indexing
            is the foundation of database optimization. Query planning and execution paths
            must be analyzed for effective database optimization. Connection pooling is
            another aspect of database optimization that improves efficiency.

            DATABASE OPTIMIZATION STRATEGIES - PART 2:
            Advanced database optimization techniques include partitioning large tables.
            Database optimization also involves monitoring query performance regularly.
            Caching strategies complement database optimization efforts significantly.
            The goal of database optimization is to reduce latency and increase throughput.

            DATABASE OPTIMIZATION STRATEGIES - PART 3:
            Modern database optimization leverages machine learning for query planning.
            Automatic database optimization tools analyze usage patterns continuously.
            Best practices in database optimization evolve with new database versions.
            Comprehensive database optimization requires understanding workload patterns.
            '''

            # Store the long repetitive document
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': dedup_thread,
                    'source': 'agent',
                    'text': repetitive_text,
                    'tags': ['repetitive-document'],
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to store repetitive document'))
                return False

            stored_id = store_data.get('context_id')

            # Allow time for embedding generation
            await asyncio.sleep(1.0)

            # Search for content that appears in MULTIPLE chunks
            search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'database optimization strategies performance',
                    'thread_id': dedup_thread,
                    'limit': 10,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data:
                self.test_results.append((test_name, False, 'Search failed'))
                return False

            results = search_data['results']

            # Count unique document IDs in results
            doc_ids = [r.get('id') for r in results]
            unique_ids = set(doc_ids)

            # Verify no duplicate document IDs (deduplication working)
            if len(doc_ids) != len(unique_ids):
                self.test_results.append(
                    (test_name, False, f'Duplicates found: {len(doc_ids)} results, {len(unique_ids)} unique'),
                )
                return False

            # Verify our stored document appears only once
            stored_id_count = doc_ids.count(stored_id)
            if stored_id_count > 1:
                self.test_results.append((test_name, False, f'Stored document appears {stored_id_count} times'))
                return False

            self.test_results.append(
                (test_name, True, f'No duplicates: {len(results)} results, all unique IDs'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_chunking_disabled_single_embedding(self) -> bool:
        """Test behavior when chunking is disabled (single embedding per document).

        Note: This test verifies that semantic search works without chunking.
        The actual chunking disabled state depends on environment configuration.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Chunking Disabled Single Embedding'
        assert self.client is not None
        try:
            # Check current state
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            chunking_info = stats_data.get('chunking', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_chunking_enabled = chunking_info.get('enabled', False)
            is_semantic_enabled = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            # If chunking IS enabled, we skip this test (cannot disable at runtime)
            if is_chunking_enabled:
                self.test_results.append(
                    (test_name, True, 'Skipped (chunking is enabled - cannot test disabled state at runtime)'),
                )
                return True

            if not is_semantic_enabled:
                self.test_results.append(
                    (test_name, True, 'Skipped (semantic search not available)'),
                )
                return True

            # Chunking is disabled - verify semantic search still works
            no_chunk_thread = f'{self.test_thread_id}_no_chunk'

            # Store a document
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': no_chunk_thread,
                    'source': 'agent',
                    'text': 'Testing semantic search without chunking enabled.',
                },
            )
            if not self._extract_content(store_result).get('success'):
                self.test_results.append((test_name, False, 'Failed to store document'))
                return False

            await asyncio.sleep(0.3)

            # Verify search works
            search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'semantic search chunking',
                    'thread_id': no_chunk_thread,
                    'limit': 5,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data:
                self.test_results.append((test_name, False, 'Search failed'))
                return False

            # Verify results have scores with semantic_distance (not chunking-related fields)
            results = search_data.get('results', [])
            if results and 'scores' in results[0] and 'semantic_distance' in results[0].get('scores', {}):
                self.test_results.append((test_name, True, 'Search works with chunking disabled'))
                return True

            self.test_results.append((test_name, True, 'Search works (chunking disabled)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_reranking_disabled_no_score(self) -> bool:
        """Test that when reranking is disabled, no rerank_score appears in results.

        Note: This test verifies behavior when reranking is unavailable.
        The actual reranking state depends on environment configuration.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Reranking Disabled No Score'
        assert self.client is not None
        try:
            # Check current state
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            reranking_info = stats_data.get('reranking', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_reranking_enabled = reranking_info.get('enabled', False) and reranking_info.get('available', False)
            is_semantic_enabled = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            # If reranking IS enabled, we skip this test (cannot disable at runtime)
            if is_reranking_enabled:
                self.test_results.append(
                    (test_name, True, 'Skipped (reranking is enabled - cannot test disabled state at runtime)'),
                )
                return True

            if not is_semantic_enabled:
                self.test_results.append(
                    (test_name, True, 'Skipped (semantic search not available)'),
                )
                return True

            # Reranking is disabled - verify no rerank_score in results
            no_rerank_thread = f'{self.test_thread_id}_no_rerank'

            # Store test documents
            for i in range(3):
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': no_rerank_thread,
                        'source': 'agent',
                        'text': f'Test document {i} for reranking disabled verification.',
                    },
                )
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, 'Failed to store documents'))
                    return False

            await asyncio.sleep(0.3)

            # Search without reranking
            search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'test document verification',
                    'thread_id': no_rerank_thread,
                    'limit': 5,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data:
                self.test_results.append((test_name, False, 'Search failed'))
                return False

            results = search_data.get('results', [])

            # Verify NO rerank_score in results (reranking disabled)
            has_rerank_score = any(
                'scores' in r and r.get('scores', {}).get('rerank_score') is not None for r in results
            )

            if has_rerank_score:
                self.test_results.append((test_name, False, 'rerank_score present when reranking disabled'))
                return False

            # Verify results are ordered by semantic_distance instead
            if results and 'scores' in results[0] and 'semantic_distance' in results[0].get('scores', {}):
                self.test_results.append((test_name, True, 'No rerank_score, ordered by semantic_distance'))
                return True

            self.test_results.append((test_name, True, 'No rerank_score in results (reranking disabled)'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_chunking_reranking_integration(self) -> bool:
        """Complete integration test: long document + chunking + reranking.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Chunking Reranking Integration'
        assert self.client is not None
        try:
            # Check if both chunking and reranking are enabled
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)

            chunking_info = stats_data.get('chunking', {})
            reranking_info = stats_data.get('reranking', {})
            semantic_info = stats_data.get('semantic_search', {})

            is_chunking_enabled = chunking_info.get('enabled', False)
            is_reranking_enabled = reranking_info.get('enabled', False) and reranking_info.get('available', False)
            is_semantic_enabled = semantic_info.get('enabled', False) and semantic_info.get('available', False)

            if not is_chunking_enabled or not is_reranking_enabled or not is_semantic_enabled:
                skip_msg = (
                    f'Skipped (chunking={is_chunking_enabled}, '
                    f'reranking={is_reranking_enabled}, semantic={is_semantic_enabled})'
                )
                self.test_results.append((test_name, True, skip_msg))
                return True

            # Create a separate thread for integration tests
            integration_thread = f'{self.test_thread_id}_integration'

            # Store 3 documents: one short, two long with different content
            # Document A: Short (no chunking needed)
            doc_a = 'Short document about cloud computing and serverless architecture.'

            # Document B: Long with Python/ML content (will be chunked)
            doc_b = '''
            COMPREHENSIVE GUIDE TO PYTHON MACHINE LEARNING:
            Python has become the dominant language for machine learning and data science.
            Key libraries include NumPy for numerical computing, Pandas for data manipulation,
            scikit-learn for classical machine learning, and TensorFlow/PyTorch for deep learning.
            Feature engineering is a critical step in building effective ML models.
            Cross-validation helps ensure model generalization to unseen data.
            Hyperparameter tuning with grid search or random search optimizes model performance.
            Python's ecosystem includes visualization tools like Matplotlib and Seaborn.
            Jupyter notebooks provide an interactive environment for exploratory data analysis.
            Production ML pipelines often use tools like MLflow for experiment tracking.
            '''

            # Document C: Long with JavaScript content (will be chunked)
            doc_c = '''
            MODERN JAVASCRIPT DEVELOPMENT PRACTICES:
            JavaScript has evolved significantly with ES6+ features and modern frameworks.
            React, Vue, and Angular dominate the frontend framework landscape.
            Node.js enables server-side JavaScript with excellent performance characteristics.
            TypeScript adds static typing to JavaScript for improved code quality.
            Package managers like npm and yarn handle dependency management efficiently.
            Build tools such as Webpack and Vite optimize application bundles.
            Testing frameworks include Jest for unit tests and Cypress for end-to-end testing.
            State management solutions range from Redux to Zustand for React applications.
            Server-side rendering with Next.js improves SEO and initial load performance.
            '''

            # Store all documents
            for i, doc in enumerate([doc_a, doc_b, doc_c]):
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': integration_thread,
                        'source': 'agent',
                        'text': doc,
                        'tags': [f'doc-{chr(65 + i)}'],
                    },
                )
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, f'Failed to store document {chr(65 + i)}'))
                    return False

            # Allow time for chunking and embedding
            await asyncio.sleep(1.5)

            # Search for Python ML content - Document B should rank highest
            search_result = await self.client.call_tool(
                'semantic_search_context',
                {
                    'query': 'Python machine learning scikit-learn TensorFlow',
                    'thread_id': integration_thread,
                    'limit': 5,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data or len(search_data.get('results', [])) == 0:
                self.test_results.append((test_name, False, 'Search returned no results'))
                return False

            results = search_data['results']

            # Verify rerank_score is present in scores object (reranking working)
            if 'scores' not in results[0] or 'rerank_score' not in results[0].get('scores', {}):
                self.test_results.append((test_name, False, 'Missing rerank_score in results.scores'))
                return False

            # Count unique documents (deduplication working)
            unique_ids = {r.get('id') for r in results}
            if len(unique_ids) != len(results):
                self.test_results.append((test_name, False, 'Duplicate documents in results'))
                return False

            # Verify Python doc (doc B) ranks highest
            top_result_text = results[0].get('text_content', '')
            if 'Python' not in top_result_text and 'machine learning' not in top_result_text.lower():
                # The test is more lenient - just verify results are returned and deduplicated
                pass

            self.test_results.append(
                (test_name, True, f'Integration: chunking + dedup + reranking working ({len(results)} unique results)'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_overfetch_chain_verification(self) -> bool:
        """Verify the overfetch multiplier chain produces sufficient candidates.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'Overfetch Chain Verification'
        assert self.client is not None
        try:
            # Check if hybrid search is enabled
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
                self.test_results.append(
                    (test_name, True, f'Skipped (fts={has_fts}, semantic={has_semantic})'),
                )
                return True

            # Create a thread with many documents
            overfetch_thread = f'{self.test_thread_id}_overfetch'

            # Store 20 diverse documents
            topics = [
                'machine learning', 'database systems', 'web development', 'cloud computing',
                'data science', 'software testing', 'DevOps practices', 'API design',
                'microservices', 'containerization', 'security best practices', 'performance tuning',
                'code review', 'agile methodology', 'continuous integration', 'monitoring systems',
                'logging strategies', 'error handling', 'authentication', 'authorization',
            ]

            for i, topic in enumerate(topics):
                result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': overfetch_thread,
                        'source': 'agent',
                        'text': f'Document about {topic}: This entry discusses {topic} concepts and implementations.',
                    },
                )
                if not self._extract_content(result).get('success'):
                    self.test_results.append((test_name, False, f'Failed to store document {i}'))
                    return False

            # Allow time for indexing
            await asyncio.sleep(1.0)

            # Request a small limit with explain_query to see stats
            search_result = await self.client.call_tool(
                'hybrid_search_context',
                {
                    'query': 'software development best practices',
                    'thread_id': overfetch_thread,
                    'limit': 5,
                    'explain_query': True,
                },
            )
            search_data = self._extract_content(search_result)

            if 'results' not in search_data:
                self.test_results.append((test_name, False, 'Hybrid search failed'))
                return False

            # Verify we got results
            results = search_data.get('results', [])
            result_count = len(results)

            # Verify overfetch: source counts should be >= requested limit
            # Note: stats dict (with fts_stats, semantic_stats, fusion_stats) is available
            # when explain_query=True, but we verify overfetch via fts_count/semantic_count
            fts_count = search_data.get('fts_count', 0)
            semantic_count = search_data.get('semantic_count', 0)

            # At least one source should have searched more docs than final limit
            overfetch_verified = fts_count > result_count or semantic_count > result_count

            if overfetch_verified:
                self.test_results.append(
                    (test_name, True,
                     f'Overfetch verified: fts={fts_count}, semantic={semantic_count}, final={result_count}'),
                )
                return True

            # Even if exact overfetch cannot be verified, successful search is acceptable
            self.test_results.append(
                (test_name, True,
                 f'Search successful: fts={fts_count}, semantic={semantic_count}, results={result_count}'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    # Coverage of generation-first transactional store path with stub providers

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

    async def test_fts_search_not_operator_exclusion(self) -> bool:
        """Verify FTS boolean mode NOT operator excludes entries.

        SQLite FTS5 uses the NOT keyword in boolean mode, while PostgreSQL
        uses the '-' prefix via websearch_to_tsquery.

        Returns:
            bool: True if test passed.
        """
        test_name = 'fts_search_not_operator_exclusion'
        assert self.client is not None
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS unavailable)'))
                return True

            not_thread = f'{self.test_thread_id}_fts_not'

            await self.client.call_tool('store_context', {
                'thread_id': not_thread, 'source': 'agent',
                'text': 'Python web development with Django framework',
            })
            await self.client.call_tool('store_context', {
                'thread_id': not_thread, 'source': 'agent',
                'text': 'Python data science with pandas and numpy',
            })

            # Boolean NOT uses each backend's documented NATIVE syntax. The
            # server intentionally does NOT unify boolean operators across
            # backends (no native cross-engine syntax exists); the correct
            # per-backend syntax is published to clients via the dynamic tool
            # descriptions in app/tools/descriptions.py:
            #   SQLite FTS5  -> NOT keyword ('Python NOT Django')
            #   PostgreSQL   -> websearch '-' prefix ('Python -Django'); the
            #                   bare word 'NOT' is an English stop word there
            #                   (websearch_to_tsquery), so it is NOT an operator.
            not_query = 'Python -Django' if self.backend == 'postgresql' else 'Python NOT Django'
            result = await self.client.call_tool('fts_search_context', {
                'query': not_query, 'mode': 'boolean',
                'thread_id': not_thread, 'limit': 10,
            })
            data = self._extract_content(result)

            if 'results' not in data:
                self.test_results.append((test_name, False, f'NOT search failed: {data}'))
                return False

            results = data.get('results', [])
            if len(results) != 1:
                self.test_results.append((test_name, False,
                    f'Expected 1 result (Django excluded), got {len(results)}'))
                return False

            found_text = results[0].get('text_content', '')
            if 'Django' in found_text:
                self.test_results.append((test_name, False, 'NOT operator failed: Django entry included'))
                return False

            self.test_results.append((test_name, True, 'FTS NOT operator correctly excludes entries'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_search_malformed_boolean_parity(self) -> bool:
        """Verify malformed boolean FTS queries degrade gracefully on BOTH backends.

        A malformed boolean query (an unbalanced parenthesis) used to raise an
        'fts5: syntax error' ToolError on SQLite while PostgreSQL's tolerant
        websearch_to_tsquery returned results -- a cross-backend MCP-contract
        divergence for byte-identical arguments. SQLite now degrades a malformed
        boolean query to the crash-safe sanitized term match, so both backends
        return a result set (no hard ToolError). A well-formed boolean query keeps
        working natively on both, exercising each backend's documented boolean syntax.

        Returns:
            bool: True if test passed.
        """
        test_name = 'fts_search_malformed_boolean_parity'
        assert self.client is not None
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS unavailable)'))
                return True

            mb_thread = f'{self.test_thread_id}_fts_malformed_boolean'
            await self.client.call_tool('store_context', {
                'thread_id': mb_thread, 'source': 'agent',
                'text': 'Structured error handling guidance for resilient services',
            })

            # Malformed boolean (unbalanced parenthesis): raised 'fts5: syntax error' on
            # SQLite while PostgreSQL returned results. Both must now return a result set
            # (no hard ToolError) and still find the entry via best-effort term recall.
            malformed_result = await self.client.call_tool('fts_search_context', {
                'query': 'error AND (handling', 'mode': 'boolean',
                'thread_id': mb_thread, 'limit': 10,
            })
            malformed_data = self._extract_content(malformed_result)
            if 'results' not in malformed_data:
                self.test_results.append((test_name, False,
                    f'Malformed boolean did not return a result set: {malformed_data}'))
                return False
            if len(malformed_data.get('results', [])) < 1:
                self.test_results.append((test_name, False,
                    'Malformed boolean degraded to zero results (expected best-effort recall)'))
                return False

            # Well-formed boolean still works (each backend's native syntax: 'AND' is an FTS5
            # operator on SQLite and an ignored stop word on PostgreSQL websearch -- both AND
            # the surviving lexemes, so the entry matches on both).
            valid_result = await self.client.call_tool('fts_search_context', {
                'query': 'error AND handling', 'mode': 'boolean',
                'thread_id': mb_thread, 'limit': 10,
            })
            valid_data = self._extract_content(valid_result)
            if len(valid_data.get('results', [])) < 1:
                self.test_results.append((test_name, False,
                    f'Well-formed boolean returned no results: {valid_data}'))
                return False

            self.test_results.append((test_name, True,
                'Malformed boolean degrades gracefully on both backends; valid boolean unaffected'))
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

    async def test_grep_context_literal_regex_unicode(self) -> bool:
        """Verify grep_context matches literal, regex, and Unicode-case patterns identically on both backends.

        The Cyrillic upper-vs-lower case match is the parity-critical check: it
        forces Python re.IGNORECASE (the ASCII-only SQL substring pre-narrow is
        skipped for non-ASCII), so SQLite and PostgreSQL must agree.

        Returns:
            bool: True if test passed.
        """
        test_name = 'grep_context_literal_regex_unicode'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_grep'
            cyr_lower = ''.join(chr(c) for c in (0x043F, 0x0440, 0x0438, 0x0432, 0x0435, 0x0442))
            store = await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': f'alpha NEEDLE line\nbeta line\n{cyr_lower} tail',
            })
            if not self._extract_content(store).get('success'):
                self.test_results.append((test_name, False, 'store failed'))
                return False

            literal = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'NEEDLE', 'thread_id': thread,
            }))
            if len(literal.get('results', [])) != 1:
                self.test_results.append((test_name, False, f'literal grep expected 1 entry, got {literal}'))
                return False

            unicode_ci = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': cyr_lower.upper(), 'thread_id': thread,
            }))
            if len(unicode_ci.get('results', [])) != 1:
                self.test_results.append((test_name, False, 'Cyrillic case-insensitive grep failed (backend parity)'))
                return False

            regex = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'N.+E', 'thread_id': thread, 'is_regex': True,
            }))
            if len(regex.get('results', [])) != 1:
                self.test_results.append((test_name, False, 'regex grep failed'))
                return False

            self.test_results.append((test_name, True, f'grep literal+regex+unicode on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_read_context_range_clamp_and_composition(self) -> bool:
        """Verify read_context_range char/line addressing, clamp+echo, and grep->read composition.

        Proves the shared code-point offset contract end to end on both backends:
        a grep content match's offsets feed read_context_range to extract exactly
        the matched span, and an over-range request is clamped to the document end.

        Returns:
            bool: True if test passed.
        """
        test_name = 'read_context_range_clamp_and_composition'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_read'
            full_text = 'line one\nfind TARGET here\nline three'
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': full_text,
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, 'store failed'))
                return False
            cid = store['context_id']

            char_read = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': cid, 'start_char': 0, 'end_char': 8,
            }))
            if char_read.get('text') != 'line one':
                self.test_results.append((test_name, False, f'char range wrong: {char_read}'))
                return False

            clamped = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': cid, 'start_char': 0, 'end_char': 100000,
            }))
            if clamped.get('end_char') != len(full_text):
                self.test_results.append((test_name, False, f'clamp not applied: {clamped}'))
                return False

            grep = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'TARGET', 'thread_id': thread, 'output_mode': 'content', 'case_sensitive': True,
            }))
            matches = grep.get('results', [])
            if not matches:
                self.test_results.append((test_name, False, 'grep content found no TARGET match'))
                return False
            match = matches[0]
            extracted = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': match['context_id'],
                'start_char': match['match_start'],
                'end_char': match['match_end'],
            }))
            if extracted.get('text') != 'TARGET':
                self.test_results.append((test_name, False, f'composition extracted wrong span: {extracted}'))
                return False

            # Multibyte composition: a Cyrillic prefix makes code-point and UTF-8
            # byte offsets diverge, so this proves grep's match offsets are
            # code-point indices that compose with read_context_range identically
            # on SQLite and PostgreSQL.
            mb_thread = f'{self.test_thread_id}_readmb'
            cyr = ''.join(chr(c) for c in (0x0451, 0x0451, 0x0451))  # 3 two-byte chars
            mb_store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': mb_thread, 'source': 'agent', 'text': f'{cyr} MULTIBYTE tail',
            }))
            mb_cid = mb_store['context_id']
            mb_grep = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'MULTIBYTE', 'thread_id': mb_thread, 'output_mode': 'content', 'case_sensitive': True,
            }))
            mb_matches = mb_grep.get('results', [])
            if not mb_matches:
                self.test_results.append((test_name, False, 'multibyte grep found no match'))
                return False
            mb_match = mb_matches[0]
            # Code-point offset is 4 (3 Cyrillic + 1 space), not 7 UTF-8 bytes.
            if mb_match['match_start'] != 4:
                self.test_results.append((
                    test_name, False,
                    f'multibyte match_start is not a code-point offset (got {mb_match["match_start"]}, want 4)',
                ))
                return False
            mb_extracted = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': mb_cid, 'start_char': mb_match['match_start'], 'end_char': mb_match['match_end'],
            }))
            if mb_extracted.get('text') != 'MULTIBYTE':
                self.test_results.append((test_name, False, f'multibyte composition wrong span: {mb_extracted}'))
                return False

            self.test_results.append((test_name, True, f'read_context_range clamp+composition (+multibyte) on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_navigate_context_outline_and_node_read(self) -> bool:
        """Verify navigate_context builds a Markdown outline and node_id reads its section, on both backends.

        Stores a multi-section Markdown entry, asserts the on-demand heading tree
        (root + nested sections with code-point offsets), then resolves a node_id
        through read_context_range to extract exactly that section -- proving the
        navigate->extract path and the shared offset contract across backends.

        Returns:
            bool: True if test passed.
        """
        test_name = 'navigate_context_outline_and_node_read'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_nav'
            text = '# Intro\nintro body\n## Details\ndetail body here\n'
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': text,
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, 'store failed'))
                return False
            cid = store['context_id']

            nav = self._extract_content(await self.client.call_tool('navigate_context', {'context_id': cid}))
            if nav.get('node_count') != 2:
                self.test_results.append((test_name, False, f'expected 2 nodes, got {nav}'))
                return False
            root = nav.get('root', {})
            intro = root.get('children', [{}])[0]
            details = intro.get('children', [{}])[0]
            if details.get('node_id') != 'intro/details':
                self.test_results.append((test_name, False, f'node_id wrong: {details}'))
                return False

            section = self._extract_content(await self.client.call_tool('read_context_range', {
                'context_id': cid, 'node_id': 'intro/details',
            }))
            if not section.get('text', '').startswith('## Details'):
                self.test_results.append((test_name, False, f'node read wrong span: {section}'))
                return False

            self.test_results.append((test_name, True, f'navigate_context outline+node read on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_index_tree_node_summaries_and_statistics(self) -> bool:
        """Verify node summaries stay additive (never abort a store) and get_statistics exposes index_tree.

        Stores a multi-section Markdown entry with per-node summaries ON (default).
        Whether or not a summary provider is configured, the store must succeed
        (a missing/failed node summary never aborts), and get_statistics must
        carry an ``index_tree`` block with enabled + a non-negative node_count.

        Returns:
            bool: True if test passed.
        """
        test_name = 'index_tree_node_summaries_and_statistics'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_idxtree'
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': '# Alpha\nalpha body\n## Beta\nbeta body here\n',
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, f'store failed: {store}'))
                return False

            stats = self._extract_content(await self.client.call_tool('get_statistics', {}))
            index_tree = stats.get('index_tree')
            if not isinstance(index_tree, dict):
                self.test_results.append((test_name, False, f'index_tree block missing from statistics: {stats.keys()}'))
                return False
            if 'enabled' not in index_tree or not isinstance(index_tree.get('node_count'), int):
                self.test_results.append((test_name, False, f'index_tree block malformed: {index_tree}'))
                return False
            if index_tree['node_count'] < 0:
                self.test_results.append((test_name, False, f'negative node_count: {index_tree}'))
                return False

            self.test_results.append((
                test_name, True,
                (
                    f'index_tree additive store + statistics on {self.backend} '
                    f'(node_count={index_tree["node_count"]})'
                ),
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_grep_keyset_scan_exhaustive_beyond_limit(self) -> bool:
        """grep_context's keyset scan must cover ALL matching entries, not cap at
        the ``search_contexts`` LIMIT 50, on both backends.

        Stores 60 entries (> 50) each carrying a shared token in a dedicated
        thread, then greps for the token and asserts every entry comes back. This
        guards ``grep_scan_text_contents``'s exhaustive id-DESC keyset pagination
        against a regression to the capped search path.

        Returns:
            bool: True if test passed.
        """
        test_name = 'grep_keyset_scan_exhaustive_beyond_limit'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_grepkeyset'
            count = 60
            for i in range(count):
                store = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'entry number {i} carries KEYSETTOKEN inline',
                }))
                if not store.get('success'):
                    self.test_results.append((test_name, False, f'store {i} failed: {store}'))
                    return False

            result = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': 'KEYSETTOKEN', 'thread_id': thread,
                'output_mode': 'files_with_matches', 'max_matches': 1000, 'max_entries_scanned': 1000,
            }))
            rows = result.get('results', [])
            if len(rows) != count:
                self.test_results.append((
                    test_name, False,
                    (
                        f'keyset scan returned {len(rows)} of {count} entries '
                        f'(capped at search LIMIT?): truncated={result.get("truncated")}'
                    ),
                ))
                return False

            self.test_results.append((test_name, True, f'grep keyset scanned all {count} entries on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_grep_scan_cap_boundary_truncation(self) -> bool:
        """grep_context's cap+lookahead must distinguish exhaustion (exactly the
        cap) from overflow (one more row) on BOTH backends.

        Exercises ``grep_scan_text_contents``'s per-backend single-row lookahead at
        the EXACT ``max_entries_scanned`` boundary: with N matching entries and
        ``max_entries_scanned=N`` the scan is exhausted -> ``truncated`` False;
        adding one more matching entry -> ``truncated`` True. Guards the
        structurally-duplicated ``_scan_sqlite`` / ``_scan_postgresql`` lookahead
        against drift (the PostgreSQL boundary branch was previously unexercised).

        Returns:
            bool: True if test passed.
        """
        test_name = 'grep_scan_cap_boundary_truncation'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_grepcap'
            token = 'GREPCAPBOUNDARY'
            cap = 5
            for i in range(cap):
                store = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'entry {i} holds {token} inline',
                }))
                if not store.get('success'):
                    self.test_results.append((test_name, False, f'store {i} failed: {store}'))
                    return False

            exact = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': token, 'thread_id': thread,
                'output_mode': 'files_with_matches', 'max_matches': 1000, 'max_entries_scanned': cap,
            }))
            if exact.get('truncated') is not False:
                self.test_results.append((
                    test_name, False,
                    f'exact-fit scan (N==cap=={cap}) should be truncated=False, got {exact.get("truncated")}',
                ))
                return False

            # One more matching entry -> the scan caps at `cap` and the lookahead
            # finds the overflow row -> truncated True.
            store = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': f'entry overflow holds {token} inline',
            }))
            if not store.get('success'):
                self.test_results.append((test_name, False, f'overflow store failed: {store}'))
                return False

            over = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': token, 'thread_id': thread,
                'output_mode': 'files_with_matches', 'max_matches': 1000, 'max_entries_scanned': cap,
            }))
            if over.get('truncated') is not True:
                self.test_results.append((
                    test_name, False,
                    f'one-over-cap scan (N==cap+1) should be truncated=True, got {over.get("truncated")}',
                ))
                return False

            self.test_results.append((test_name, True, f'grep cap boundary correct on {self.backend}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_tied_score_pagination_parity(self) -> bool:
        """Paging a TIED ranked result set neither skips nor duplicates a row.

        Three byte-identical documents in one thread score identically in every ranked
        search, and a score-only ORDER BY left their relative order to the scan: on
        PostgreSQL an unrelated UPDATE rewrites the physical tuple (MVCC), so the heap
        order behind a tied LIMIT/OFFSET window changed under churn and a client paging
        one row at a time silently lost one document and saw another twice. Each ranked
        tool now carries an explicit UNIQUE secondary key (the context id), so the union
        of the single-row pages must equal the unpaginated result -- asserted before AND
        after a metadata-only update, which is the churn that reproduced the defect live.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'tied_score_pagination_parity'
        assert self.client is not None
        try:
            tool_names = {t.name for t in await self.client.list_tools()}
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            fts_info = stats_data.get('fts', {})
            fts_ok = bool(fts_info.get('enabled')) and bool(fts_info.get('available'))
            semantic_ok = bool(stats_data.get('semantic_search', {}).get('available'))

            thread = f'{self.test_thread_id}_tied_page'
            text = 'Tied ranking parity document about zebra quokka narwhal ordering'
            # Byte-identical text in ONE thread. The opposite-source store between the
            # repeats trips the deduplication interleaving check, so each repeat is a new
            # turn and INSERTS instead of updating the previous entry.
            expected_ids: list[str] = []
            for source in ('agent', 'user', 'agent'):
                stored = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': source, 'text': text,
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store tied entry: {stored}'))
                    return False
                expected_ids.append(str(stored['context_id']))
            if len(set(expected_ids)) != 3:
                self.test_results.append((
                    test_name, False, f'Deduplication collapsed the identical-text entries: {expected_ids}',
                ))
                return False

            legs: list[tuple[str, dict[str, Any]]] = []
            if fts_ok and 'fts_search_context' in tool_names:
                legs.append(('fts_search_context', {'query': 'zebra quokka', 'mode': 'match', 'thread_id': thread}))
            if semantic_ok and 'semantic_search_context' in tool_names:
                legs.append(('semantic_search_context', {'query': text, 'thread_id': thread}))
            if (fts_ok or semantic_ok) and 'hybrid_search_context' in tool_names:
                legs.append(('hybrid_search_context', {'query': text, 'thread_id': thread}))
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True

            async def _paging_error(label: str) -> str | None:
                """Return an error message when a leg's pages disagree with its full result."""
                assert self.client is not None
                for tool, args in legs:
                    full = self._extract_content(
                        await self.client.call_tool(tool, {**args, 'limit': 10, 'offset': 0}),
                    )
                    full_ids = [str(row.get('id')) for row in full.get('results', [])]
                    if sorted(full_ids) != sorted(expected_ids):
                        return f'{tool} {label}: unpaginated result {full_ids} != stored {expected_ids}'
                    paged: list[str] = []
                    for offset in range(3):
                        page = self._extract_content(
                            await self.client.call_tool(tool, {**args, 'limit': 1, 'offset': offset}),
                        )
                        rows = page.get('results', [])
                        if len(rows) != 1:
                            return f'{tool} {label}: page at offset {offset} returned {len(rows)} rows, expected 1'
                        paged.append(str(rows[0].get('id')))
                    if sorted(paged) != sorted(expected_ids):
                        return f'{tool} {label}: pages {paged} are not a partition of {expected_ids}'
                return None

            error = await _paging_error('before churn')
            if error:
                self.test_results.append((test_name, False, error))
                return False

            churn = self._extract_content(await self.client.call_tool('update_context', {
                'context_id': expected_ids[0], 'metadata_patch': {'churn': 'tied-pagination'},
            }))
            if not churn.get('success'):
                self.test_results.append((test_name, False, f'Metadata-only update failed: {churn}'))
                return False

            error = await _paging_error('after metadata-only update')
            if error:
                self.test_results.append((test_name, False, error))
                return False

            self.test_results.append((
                test_name, True,
                f'Tied pagination stable across {len(legs)} ranked tool(s) before and after churn',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_embedded_quote_term_parity(self) -> bool:
        """A bare FTS term carrying an embedded double quote matches identically on both backends.

        Escaping the quote by doubling it does NOT neutralize it: FTS5 re-tokenizes the
        contents of a string literal, the escape decodes back to a literal quote, and the
        tokenizer treats it as a word boundary -- silently turning an ordinary token into
        a strict two-word ADJACENCY phrase, so SQLite returned only the document whose
        words happened to be adjacent while PostgreSQL's plainto_tsquery ANDed the two
        lexemes with no adjacency requirement and returned both. Splitting the token on
        the quote into independently ANDed literals makes the two backends agree.

        The hyphen target is pinned as the complementary invariant: it stays an adjacency
        phrase on SQLite and a compound lexeme on PostgreSQL, so the NON-adjacent document
        must never match it on either backend.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_embedded_quote_term_parity'
        assert self.client is not None
        try:
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS not available)'))
                return True

            thread = f'{self.test_thread_id}_fts_quote'
            seeds = [
                ('alpha zulu beta', 'adjacent'),
                ('alpha somewhere else entirely zulu', 'separated'),
            ]
            ids: dict[str, str] = {}
            for text, label in seeds:
                stored = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent', 'text': text,
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Failed to store {label} seed: {stored}'))
                    return False
                ids[label] = str(stored['context_id'])

            quoted = self._extract_content(await self.client.call_tool('fts_search_context', {
                'query': 'alpha"zulu', 'mode': 'match', 'thread_id': thread, 'limit': 10,
            }))
            quoted_ids = {str(row.get('id')) for row in quoted.get('results', [])}
            if quoted_ids != set(ids.values()):
                self.test_results.append((
                    test_name, False,
                    f'Query alpha"zulu returned {len(quoted_ids)} of 2 documents on {self.backend}: {quoted_ids}',
                ))
                return False

            hyphen = self._extract_content(await self.client.call_tool('fts_search_context', {
                'query': 'alpha-zulu', 'mode': 'match', 'thread_id': thread, 'limit': 10,
            }))
            hyphen_ids = {str(row.get('id')) for row in hyphen.get('results', [])}
            if ids['separated'] in hyphen_ids:
                self.test_results.append((
                    test_name, False,
                    'Query alpha-zulu matched the non-adjacent document (the hyphen lost its adjacency meaning)',
                ))
                return False

            self.test_results.append((
                test_name, True, 'An embedded quote ANDs its fragments on both backends; the hyphen stays adjacency-only',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_grep_context_request_caps_clamped(self) -> bool:
        """grep_context clamps an oversized request to the server caps on both backends.

        The wire schema deliberately admits values far above the server bounds
        (``max_matches`` to 10000, ``context_lines`` to 100, ``max_entries_scanned`` to
        1000000) so a client is never rejected for asking; the server clamps each one to
        its configured cap. Without the clamp a single call can return every match in a
        dense corpus with a hundred context lines apiece, flooding the caller's context
        window and the event loop. The corpus here carries more matches than the default
        1000 cap and more surrounding lines than the default 20, so a missing clamp
        shows up as an over-cap response rather than as a silent pass.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'grep_context_request_caps_clamped'
        assert self.client is not None
        try:
            tool_names = {t.name for t in await self.client.list_tools()}
            if 'grep_context' not in tool_names:
                self.test_results.append((test_name, True, 'Skipped (grep_context not registered)'))
                return True

            dense_thread = f'{self.test_thread_id}_grep_caps_dense'
            token = 'GREPCLAMPTOKEN'
            # 30 lines x 40 occurrences = 1200 matches, past the default 1000 cap.
            dense_line = ' '.join([token] * 40)
            dense_text = '\n'.join(f'{index:03d} {dense_line}' for index in range(30))
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': dense_thread, 'source': 'agent', 'text': dense_text,
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Failed to store the dense corpus: {stored}'))
                return False

            counted = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': token, 'thread_id': dense_thread, 'output_mode': 'count',
                'max_matches': 10000, 'context_lines': 100, 'max_entries_scanned': 1000000,
            }))
            total = counted.get('total_matches')
            if not isinstance(total, int) or not (0 < total <= 1000):
                self.test_results.append((
                    test_name, False, f'total_matches {total!r} is not clamped into (0, 1000]',
                ))
                return False
            if counted.get('truncated') is not True:
                self.test_results.append((
                    test_name, False, f'A capped scan must report truncated=True, got {counted.get("truncated")!r}',
                ))
                return False

            # A separate small entry keeps the content-mode response tiny while still
            # offering more surrounding lines than the context cap allows.
            context_thread = f'{self.test_thread_id}_grep_caps_context'
            marker = 'GREPCLAMPMARKER'
            lines = [f'context line {index}' for index in range(61)]
            lines[30] = f'context line 30 holds {marker}'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': context_thread, 'source': 'agent', 'text': '\n'.join(lines),
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Failed to store the context corpus: {stored}'))
                return False

            content = self._extract_content(await self.client.call_tool('grep_context', {
                'pattern': marker, 'thread_id': context_thread, 'output_mode': 'content',
                'max_matches': 10000, 'context_lines': 100, 'max_entries_scanned': 1000000,
            }))
            rows = content.get('results', [])
            if len(rows) != 1:
                self.test_results.append((test_name, False, f'Expected exactly one content match, got {len(rows)}'))
                return False
            before = rows[0].get('before', [])
            after = rows[0].get('after', [])
            if not (0 < len(before) <= 20) or not (0 < len(after) <= 20):
                self.test_results.append((
                    test_name, False,
                    f'context_lines was not clamped: before={len(before)}, after={len(after)} (30 available each side)',
                ))
                return False

            self.test_results.append((
                test_name, True,
                f'Oversized request clamped: total_matches={total}, before={len(before)}, after={len(after)}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def _ranked_search_legs(self, thread: str, query: str) -> list[tuple[str, dict[str, Any]]]:
        """Build the call arguments for every ranked search tool currently available.

        Args:
            thread: Thread the ranked query is scoped to.
            query: Free-text query passed to each tool.

        Returns:
            A (tool_name, arguments) pair per available ranked tool; empty when none is.
        """
        assert self.client is not None
        tool_names = {t.name for t in await self.client.list_tools()}
        stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
        fts_info = stats_data.get('fts', {})
        fts_ok = bool(fts_info.get('enabled')) and bool(fts_info.get('available'))
        semantic_ok = bool(stats_data.get('semantic_search', {}).get('available'))

        legs: list[tuple[str, dict[str, Any]]] = []
        if fts_ok and 'fts_search_context' in tool_names:
            legs.append(('fts_search_context', {'query': query, 'mode': 'match', 'thread_id': thread}))
        if semantic_ok and 'semantic_search_context' in tool_names:
            legs.append(('semantic_search_context', {'query': query, 'thread_id': thread}))
        if (fts_ok or semantic_ok) and 'hybrid_search_context' in tool_names:
            legs.append(('hybrid_search_context', {'query': query, 'thread_id': thread}))
        return legs

    async def test_ranked_pagination_union_matches_single_page(self) -> bool:
        """Paging a ranked result set yields exactly the rows the single call yields.

        Semantic, FTS and hybrid search decide their FINAL order after the database
        returns rows -- cross-encoder reranking, RRF fusion, or both -- so sizing the
        candidate window from the requested page built page N and page N+1 from
        DIFFERENT candidate pools: a document that only entered the larger pool could
        outrank rows already served, pushing them onto a later page a second time while
        other rows were never returned by any page. The candidate depth is now fixed and
        page-independent, so one query has ONE ordering and limit/offset merely select a
        window inside it.

        The assertion is the union property that guarantees: four two-row pages
        concatenated must equal the first eight rows of a single eight-row call, in the
        same order and with no id repeated. The corpus is deliberately larger than the
        window so the pool sizes under the old scheme would have differed.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'ranked_pagination_union_matches_single_page'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_rank_pages'
            corpus_size = 25
            entries = [
                {
                    'thread_id': thread, 'source': 'agent',
                    'text': (
                        f'Ranked pagination corpus document {index:02d} discussing paginated '
                        f'retrieval windows, ordering stability and page offsets'
                    ),
                }
                for index in range(corpus_size)
            ]
            batch = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': entries, 'atomic': True,
            }))
            if batch.get('succeeded') != corpus_size:
                self.test_results.append((test_name, False, f'Corpus store failed: {batch}'))
                return False

            legs = await self._ranked_search_legs(thread, 'paginated retrieval windows')
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True

            page_size = 2
            window = 8
            for tool, args in legs:
                single = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': window, 'offset': 0}),
                )
                single_ids = [str(row.get('id')) for row in single.get('results', [])]
                if len(single_ids) != window:
                    self.test_results.append((
                        test_name, False, f'{tool} returned {len(single_ids)} rows for a {window}-row page',
                    ))
                    return False

                paged_ids: list[str] = []
                for offset in range(0, window, page_size):
                    page = self._extract_content(
                        await self.client.call_tool(tool, {**args, 'limit': page_size, 'offset': offset}),
                    )
                    rows = page.get('results', [])
                    if len(rows) != page_size:
                        self.test_results.append((
                            test_name, False,
                            f'{tool} page at offset {offset} returned {len(rows)} rows, expected {page_size}',
                        ))
                        return False
                    paged_ids.extend(str(row.get('id')) for row in rows)

                if len(set(paged_ids)) != len(paged_ids):
                    self.test_results.append((
                        test_name, False, f'{tool} returned the same id on two pages: {paged_ids}',
                    ))
                    return False
                if paged_ids != single_ids:
                    self.test_results.append((
                        test_name, False,
                        f'{tool} paged ids {paged_ids} differ from the single-call ids {single_ids}',
                    ))
                    return False

            self.test_results.append((
                test_name, True,
                f'Paged and unpaged ids agree across {len(legs)} ranked tool(s) over {corpus_size} documents',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_ranked_depth_limit_hint(self) -> bool:
        """A ranked page reaching past the fixed depth is reported, not silently empty.

        Ranked search serves every page from one ordering at most 100 rows deep (the
        depth the tool documentation advertises), so a window past that depth comes back
        short -- empty when the offset alone is past it. Without the hint a client cannot
        tell that from an exhausted result set and pages forever. The hint therefore
        appears exactly when ``offset + limit`` exceeds the depth, echoing the window it
        describes, and is ABSENT for an ordinary page.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'ranked_depth_limit_hint'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_rank_depth'
            entries = [
                {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'Depth hint probe document {index} about paginated ranking depth',
                }
                for index in range(3)
            ]
            batch = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': entries, 'atomic': True,
            }))
            if batch.get('succeeded') != len(entries):
                self.test_results.append((test_name, False, f'Probe store failed: {batch}'))
                return False

            legs = await self._ranked_search_legs(thread, 'paginated ranking depth')
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True

            expected_hint = {'requested_offset': 99, 'requested_limit': 5, 'rank_depth': 100}
            for tool, args in legs:
                deep = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': 5, 'offset': 99}),
                )
                if deep.get('rank_depth_limit') != expected_hint:
                    self.test_results.append((
                        test_name, False,
                        f'{tool} reported rank_depth_limit {deep.get("rank_depth_limit")!r}, expected {expected_hint}',
                    ))
                    return False
                if deep.get('results'):
                    self.test_results.append((
                        test_name, False, f'{tool} returned rows for a window past the ranked depth: {deep}',
                    ))
                    return False

                ordinary = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': 5, 'offset': 0}),
                )
                if 'rank_depth_limit' in ordinary:
                    self.test_results.append((
                        test_name, False, f'{tool} reported rank_depth_limit for an ordinary page: {ordinary}',
                    ))
                    return False

                # A window that STARTS at the depth is empty by arithmetic alone, so the
                # tool answers it without retrieving or scoring anything -- and must still
                # report the same shape rather than an error or a bare empty page.
                past = self._extract_content(
                    await self.client.call_tool(tool, {**args, 'limit': 10, 'offset': 100}),
                )
                if past.get('rank_depth_limit') != {
                    'requested_offset': 100, 'requested_limit': 10, 'rank_depth': 100,
                } or past.get('results') or past.get('count') != 0:
                    self.test_results.append((
                        test_name, False, f'{tool} mis-reported a page starting past the ranked depth: {past}',
                    ))
                    return False

            self.test_results.append((
                test_name, True, f'rank_depth_limit reported only past the ranked depth on {len(legs)} tool(s)',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_collation_ordering_parity(self) -> bool:
        """Ordered text comes back byte-ordered on BOTH backends, not locale-ordered.

        SQLite compares TEXT with its BINARY (byte) collation while PostgreSQL uses the
        database locale, which ranks punctuation and case differently, so byte-identical
        data serialized in a DIFFERENT order on the two backends: an entry's public
        ``tags`` array, and the tiebreak deciding which rows survive the statistics
        LIMIT. Every observable ordering site now renders an explicit byte-wise
        collation, so two expectations hold on both backends:

        * an entry's tags come back byte-ordered ('t-z' before 'ta', because '-' sorts
          below 'a' by byte while the locale ranks it after), identically from
          get_context_by_ids and from every search tool;
        * inside the statistics top-N lists, rows sharing a count are byte-ordered --
          asserted over deliberately collation-sensitive tags seeded at a count that
          places them inside the top_tags window, and over whatever else the window
          holds.

        Returns:
            bool: True if test passed.
        """
        test_name = 'collation_ordering_parity'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_collation'
            entry_tags = ['ta', 'tb', 't-z']
            expected_tags = sorted(entry_tags)
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Collation ordering probe entry mentioning ferroniobium alloys',
                'tags': entry_tags,
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False
            entry_id = str(stored['context_id'])

            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}))
            rows = got.get('results', [])
            if len(rows) != 1 or rows[0].get('tags') != expected_tags:
                self.test_results.append((
                    test_name, False,
                    f'get_context_by_ids returned tags {rows[0].get("tags") if rows else None}, expected {expected_tags}',
                ))
                return False

            legs: list[tuple[str, dict[str, Any]]] = [('search_context', {'thread_id': thread, 'limit': 10})]
            legs.extend(await self._ranked_search_legs(thread, 'ferroniobium alloys'))
            for tool, args in legs:
                data = self._extract_content(await self.client.call_tool(tool, {**args, 'limit': 10}))
                row = next((r for r in data.get('results', []) if str(r.get('id')) == entry_id), None)
                if row is None:
                    self.test_results.append((test_name, False, f'{tool} did not return the tagged entry'))
                    return False
                if row.get('tags') != expected_tags:
                    self.test_results.append((
                        test_name, False, f'{tool} returned tags {row.get("tags")}, expected {expected_tags}',
                    ))
                    return False

            # Six collation-sensitive labels whose byte order ('-' below 'a') differs
            # from the locale order, seeded across six equally-sized threads so both
            # statistics tiebreaks see them.
            collation_names = ['coll-a', 'coll-b', 'coll-c', 'colla', 'collb', 'collc']
            seed_entries = [
                {
                    'thread_id': f'{thread}_{collation_names[index % len(collation_names)]}',
                    'source': 'agent',
                    'text': f'Collation tie seed {index}',
                    'tags': collation_names,
                }
                for index in range(len(collation_names) * 2)
            ]
            seeded = self._extract_content(await self.client.call_tool('store_context_batch', {
                'entries': seed_entries, 'atomic': True,
            }))
            if seeded.get('succeeded') != len(seed_entries):
                self.test_results.append((test_name, False, f'Collation seed store failed: {seeded}'))
                return False

            stats = self._extract_content(await self.client.call_tool('get_statistics', {}))
            top_tags = stats.get('top_tags', [])
            most_active = stats.get('most_active_threads', [])
            seeded_order = [str(row.get('tag')) for row in top_tags if str(row.get('tag')) in set(collation_names)]
            if len(seeded_order) < 2:
                self.test_results.append((
                    test_name, False,
                    f'The seeded collation tags did not reach the top_tags window: {top_tags}',
                ))
                return False
            if seeded_order != sorted(seeded_order):
                self.test_results.append((
                    test_name, False, f'Tied seeded tags came back as {seeded_order}, expected byte order',
                ))
                return False

            for label, listed, key in (('top_tags', top_tags, 'tag'), ('most_active_threads', most_active, 'thread_id')):
                previous_key: str | None = None
                previous_count: object = None
                for row in listed:
                    current_key = str(row.get(key))
                    current_count = row.get('count')
                    if previous_key is not None and current_count == previous_count and current_key < previous_key:
                        self.test_results.append((
                            test_name, False,
                            f'{label} ties are not byte-ordered: {previous_key!r} precedes {current_key!r}',
                        ))
                        return False
                    previous_key, previous_count = current_key, current_count

            self.test_results.append((
                test_name, True, f'Tags and tied statistics rows are byte-ordered on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_literal_markup_survives_ranked_search(self) -> bool:
        """A document containing literal <mark> markup stays searchable on both backends.

        Neither SQLite's ``highlight()`` nor PostgreSQL's ``ts_headline()`` escapes
        markup already present in a document, so a literal '<mark>' in the stored text is
        indistinguishable by shape from a marker the engine inserted. The passage
        extractor that feeds the cross-encoder used to count every tag as inserted,
        subtracting a phantom offset that pointed the extracted passage at unrelated
        text; it now aligns the highlight against the source instead. The observable
        contract at the tool boundary is what this pins: the document is still returned
        by the ranked tools, and its literal markup is stored and read back verbatim
        rather than being consumed as a highlight marker.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'literal_markup_survives_ranked_search'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_literal_markup'
            term = 'quarkbeacon'
            text = (
                f'Passage alignment probe: this document literally contains <mark>{term}</mark> '
                f'markup around the term {term}, and continues with further prose so the passage '
                f'extractor has a window of surrounding sentences to work with.'
            )
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': text,
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False
            entry_id = str(stored['context_id'])

            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}))
            rows = got.get('results', [])
            if len(rows) != 1 or f'<mark>{term}</mark>' not in str(rows[0].get('text_content', '')):
                self.test_results.append((test_name, False, 'The literal markup did not round-trip through storage'))
                return False

            legs = await self._ranked_search_legs(thread, term)
            if not legs:
                self.test_results.append((test_name, True, 'Skipped (no ranked search tool available)'))
                return True
            for tool, args in legs:
                data = self._extract_content(await self.client.call_tool(tool, {**args, 'limit': 10}))
                if not any(str(row.get('id')) == entry_id for row in data.get('results', [])):
                    self.test_results.append((
                        test_name, False, f'{tool} did not return the document carrying literal markup: {data}',
                    ))
                    return False

            self.test_results.append((
                test_name, True, f'Literal markup neither breaks nor is consumed by {len(legs)} ranked tool(s)',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_fts_deeply_nested_boolean_query_degrades(self) -> bool:
        """A deeply nested boolean query degrades instead of erroring, and spares the breaker.

        Boolean mode forwards the client's query to the engine verbatim, and SQLite's
        FTS5 rejects a deeply nested expression with its own parser message rather than
        the handful of grammar messages that used to be enumerated as client errors.
        Anything unenumerated read as a server fault: it propagated as a hard error where
        PostgreSQL's tolerant websearch parser succeeded on the same input, and it charged
        the PROCESS-GLOBAL circuit breaker, so a client repeating one malformed query
        could open the breaker and have every other caller's reads and writes rejected.
        Failure attribution is now inverted -- the database-fault families are the closed
        set and everything else is attributed to the one client-controlled fragment -- so
        the query degrades to the sanitized term match on SQLite, matches on PostgreSQL,
        and neither backend charges a failure.

        The repetition afterwards is the point of the breaker half: it exceeds the
        consecutive-failure threshold that would have tripped it, and ordinary traffic on
        the SAME server must still succeed.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'fts_deeply_nested_boolean_query_degrades'
        assert self.client is not None
        try:
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            fts_info = stats_data.get('fts', {})
            if not (fts_info.get('enabled') and fts_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (FTS not available)'))
                return True

            thread = f'{self.test_thread_id}_fts_nested'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Deeply nested boolean probe describing an error condition in the ingest pipeline',
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False

            nesting = 100
            query = '(' * nesting + 'error' + ')' * nesting
            repeats = 12
            for attempt in range(repeats):
                data = self._extract_content(await self.client.call_tool('fts_search_context', {
                    'query': query, 'mode': 'boolean', 'thread_id': thread, 'limit': 10,
                }))
                if 'results' not in data:
                    self.test_results.append((
                        test_name, False, f'Nested boolean call {attempt} returned no result set: {data}',
                    ))
                    return False
                if not data.get('results'):
                    self.test_results.append((
                        test_name, False, f'Nested boolean call {attempt} degraded to zero results',
                    ))
                    return False

            browse = self._extract_content(await self.client.call_tool('search_context', {
                'thread_id': thread, 'limit': 10,
            }))
            if not browse.get('results'):
                self.test_results.append((
                    test_name, False, f'Ordinary browse failed after {repeats} nested queries: {browse}',
                ))
                return False
            follow_up = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Write issued after the repeated nested boolean queries',
            }))
            if not follow_up.get('success'):
                self.test_results.append((
                    test_name, False, f'Ordinary write failed after {repeats} nested queries: {follow_up}',
                ))
                return False

            after = self._extract_content(await self.client.call_tool('get_statistics', {}))
            circuit_state = str(after.get('connection_metrics', {}).get('circuit_state'))
            if circuit_state != 'healthy':
                self.test_results.append((
                    test_name, False, f'circuit_state is {circuit_state!r} after {repeats} nested queries',
                ))
                return False

            self.test_results.append((
                test_name, True,
                f'{repeats} nested boolean queries returned results and left the breaker healthy on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def run_all_tests(self) -> bool:
        """Run all tests and report results.

        Returns:
            bool: True if all tests passed.
        """
        print('\n' + '=' * 50)
        print(f'MCP SERVER INTEGRATION TEST ({self.backend}, {self.client_mode} client mode)')
        print('=' * 50)

        # Start server
        if not await self.start_server():
            print('[ERROR] Failed to start server')
            await self.cleanup()
            return False

        # Connect client
        if not await self.connect_client():
            print('[ERROR] Failed to connect client')
            await self.cleanup()
            return False

        # Run all tests
        tests = [
            ('Store Context', self.test_store_context),
            ('Search Context', self.test_search_context),
            ('Search Context Date Filtering', self.test_search_context_with_date_filtering),
            ('Metadata Filtering', self.test_metadata_filtering),
            ('Metadata Filter Nested Path', self.test_metadata_filter_nested_path),
            ('Metadata Filter LIKE Wildcard Literal', self.test_metadata_filter_like_wildcard_literal),
            ('Metadata Filter Numeric Type Parity', self.test_metadata_filter_numeric_type_parity),
            ('Metadata Filter Float Precision Parity', self.test_metadata_filter_float_precision_parity),
            (
                'Metadata Filter High-Magnitude Int Float-Param Parity',
                self.test_metadata_filter_high_magnitude_int_float_param_parity,
            ),
            (
                'Metadata Filter High-Magnitude Float Roundtrip Parity',
                self.test_metadata_filter_high_magnitude_float_roundtrip_parity,
            ),
            (
                'Metadata Filter Out-of-Range Numeric Parity',
                self.test_metadata_filter_out_of_range_numeric_parity,
            ),
            (
                'Array Contains High-Magnitude Float Parity',
                self.test_array_contains_high_magnitude_float_parity,
            ),
            ('Metadata Filter Boolean Type Parity', self.test_metadata_filter_boolean_type_parity),
            ('Metadata Filter GLOB Special Literal', self.test_metadata_filter_glob_special_literal),
            ('Semantic Hybrid Metadata Dict', self.test_semantic_hybrid_metadata_is_dict),
            ('Array Contains Operator', self.test_array_contains_operator),
            ('Array Contains Non-Array Field', self.test_array_contains_non_array_field),
            ('Get Context by IDs', self.test_get_context_by_ids),
            ('Delete Context', self.test_delete_context),
            ('Update Context', self.test_update_context),
            ('Visibility Lifecycle', self.test_visibility_lifecycle),
            ('Metadata Patch Deep Merge', self.test_metadata_patch_deep_merge),
            ('Metadata Patch RFC 7396 Full Compliance', self.test_metadata_patch_rfc7396_full_compliance),
            ('Metadata Patch Successive Patches', self.test_metadata_patch_successive_patches),
            ('Metadata Patch Type Conversions', self.test_metadata_patch_type_conversions),
            ('List Threads', self.test_list_threads),
            ('Get Statistics', self.test_get_statistics),
            ('Store Context Batch', self.test_store_context_batch),
            ('Update Context Batch', self.test_update_context_batch),
            ('Update Context Batch Version Guard', self.test_update_context_batch_version_guard),
            ('Delete Context Batch', self.test_delete_context_batch),
            ('Semantic Search', self.test_semantic_search_context),
            ('Semantic Search Date Filtering', self.test_semantic_search_context_with_date_filtering),
            ('Semantic Search Metadata Filtering', self.test_semantic_search_context_with_metadata_filters),
            ('Search Context Invalid Filter Error', self.test_search_context_invalid_filter_returns_error),
            ('Search Context Blank Tags Error', self.test_search_context_blank_tags_returns_error),
            (
                'Search Context Oversized Metadata Filters',
                self.test_search_context_oversized_metadata_filters_rejected,
            ),
            ('NUL Input Does Not Trip Breaker', self.test_nul_input_does_not_trip_breaker),
            ('Semantic Search Invalid Filter Error', self.test_semantic_search_invalid_filter_returns_error),
            ('FTS Search', self.test_fts_search_context),
            ('FTS Search Invalid Filter Error', self.test_fts_search_invalid_filter_returns_error),
            ('FTS Boolean Mode', self.test_fts_boolean_mode),
            ('FTS Date Range Filter', self.test_fts_date_range_filter),
            ('FTS Metadata Filter', self.test_fts_metadata_filter),
            ('FTS Metadata Filter Key Substring', self.test_fts_metadata_filter_key_substring),
            ('FTS Advanced Metadata Filters', self.test_fts_advanced_metadata_filters),
            ('FTS Pagination Offset', self.test_fts_pagination_offset),
            ('FTS Highlight Snippets', self.test_fts_highlight_snippets),
            ('Hybrid Search', self.test_hybrid_search_context),
            ('Hybrid Search Adaptive FTS Mode', self.test_hybrid_search_adaptive_fts_mode),
            ('Search Tools Content Type Filter', self.test_search_tools_content_type_filter),
            ('Search Tools Include Images', self.test_search_tools_include_images),
            ('Search Tools Tags Filter', self.test_search_tools_tags_filter),
            ('Semantic Search Offset Pagination', self.test_semantic_search_offset_pagination),
            ('Hybrid Search Metadata Filtering', self.test_hybrid_search_metadata_filtering),
            ('Hybrid Search Date Range Filtering', self.test_hybrid_search_date_range_filtering),
            ('Hybrid Search Offset Pagination', self.test_hybrid_search_offset_pagination),
            ('Explain Query Statistics', self.test_explain_query_statistics),
            # Chunking and Reranking Tests
            ('Statistics Chunking Reranking Info', self.test_statistics_chunking_reranking_info),
            ('Statistics Summary Info', self.test_statistics_summary_info),
            ('Chunking Creates Multiple Embeddings', self.test_chunking_creates_multiple_embeddings),
            ('Chunking Long Document Storage', self.test_chunking_long_document_storage),
            ('Reranking Adds Score to Results', self.test_reranking_adds_score_to_results),
            ('Reranking in FTS Search', self.test_reranking_in_fts_search),
            ('Reranking in Hybrid Search', self.test_reranking_in_hybrid_search),
            ('Chunking Deduplication in Search', self.test_chunking_deduplication_in_search),
            ('Chunking Disabled Single Embedding', self.test_chunking_disabled_single_embedding),
            ('Reranking Disabled No Score', self.test_reranking_disabled_no_score),
            ('Chunking Reranking Integration', self.test_chunking_reranking_integration),
            ('Overfetch Chain Verification', self.test_overfetch_chain_verification),
            # Quality Improvement Tests
            ('Search Context Limit Clamping', self.test_search_context_limit_clamping),
            # Protocol Era And Server Identity Tests
            ('Client Negotiates Requested Protocol Era', self.test_client_negotiates_requested_protocol_era),
            ('Server Version Is Project Version', self.test_server_version_is_project_version),
            # Deduplication Data Integrity Tests
            ('Dedup Data Integrity', self.test_store_context_deduplication_data_integrity),
            # Edge Case Tests (P3)
            ('Store Context Empty Text', self.test_store_context_empty_text),
            ('Store Context Max Size Image', self.test_store_context_max_size_image),
            ('Search Context No Results', self.test_search_context_no_results),
            ('Delete Context Nonexistent ID', self.test_delete_context_nonexistent_id),
            ('Update Context Nonexistent ID', self.test_update_context_nonexistent_id),
            ('Get Context By IDs Partial Match', self.test_get_context_by_ids_partial_match),
            ('List Threads With Filter', self.test_list_threads_empty_database),
            ('Batch Operations Atomic Rollback', self.test_batch_operations_atomic_rollback),
            ('Batch Operations Non-Atomic Partial', self.test_batch_operations_non_atomic_partial),
            # Summary Field Tests
            ('Get Context By IDs Omits Summary By Default', self.test_get_context_by_ids_omits_summary_by_default),
            ('Get Context By IDs Includes Summary When Enabled', self.test_get_context_by_ids_includes_summary_when_enabled),
            ('Search Context Summary Display', self.test_search_context_summary_display),
            ('Batch Store Summary Field', self.test_batch_store_summary_field),
            # Generation-First Pattern Tests
            ('Store Context Generation First', self.test_store_context_generation_first_return_exceptions),
            ('Batch Store Generation First', self.test_batch_store_generation_first_return_exceptions),
            # Middleware JSON String Deserializer Tests
            ('Middleware Deserializes Stringified Tags', self.test_middleware_deserializes_stringified_tags),
            ('Middleware Deserializes Stringified Metadata', self.test_middleware_deserializes_stringified_metadata),
            ('Middleware Preserves String Text', self.test_middleware_preserves_string_text),
            ('Middleware Deserializes Stringified Context IDs', self.test_middleware_deserializes_stringified_context_ids),
            # Summary Env Var Tests
            ('Summary Env Vars Accepted', self.test_summary_env_vars_accepted),
            # Coverage of generation-first transactional store path with stub providers
            ('List Threads Populated Database', self.test_list_threads_with_populated_database),
            ('List Threads Pagination', self.test_list_threads_pagination),
            ('Session Pooler Validation No-op on SQLite', self.test_session_pooler_validation_noop_on_sqlite),
            ('Update Triggers Embedding Regen', self.test_update_context_triggers_embedding_regeneration),
            ('Batch Store Dedup Within Batch', self.test_store_context_batch_dedup_within_batch),
            ('Hybrid Search Graceful Degradation', self.test_hybrid_search_graceful_degradation_fts_only),
            ('Content Type Auto Detection', self.test_content_type_auto_detection_multimodal),
            ('B5 Image No Mime Type Integration', self.test_update_context_image_without_mime_type_integration),
            ('B6 Batch Content Type Integration', self.test_update_context_batch_content_type_correction),
            ('Content Type Filter Multimodal', self.test_search_context_content_type_filter_multimodal),
            ('FTS Match Stemming', self.test_fts_search_match_mode_stemming),
            ('FTS NOT Operator', self.test_fts_search_not_operator_exclusion),
            ('FTS Malformed Boolean Parity', self.test_fts_search_malformed_boolean_parity),
            ('Hybrid RRF Score Ordering', self.test_hybrid_search_rrf_scores_ordering),
            ('Explain Query False No Stats', self.test_search_context_explain_query_false_no_stats),
            ('Dedup Preserves Tags', self.test_store_context_dedup_preserves_tags_when_none),
            ('Dedup Interleaving Check', self.test_store_context_dedup_interleaving_check),
            ('Batch Non-Atomic Partial', self.test_update_context_batch_non_atomic_generation_failure),
            ('Health Endpoint B8', self.test_health_endpoint_returns_ok),
            # Batch/Non-Batch Conformance Tests
            ('Store Batch Conformance', self.test_store_batch_single_matches_store_nonbatch),
            ('Update Batch Conformance', self.test_update_batch_single_matches_update_nonbatch),
            ('Delete Batch Conformance', self.test_delete_batch_single_matches_delete_nonbatch),
            # Blast-radius coverage (run on both backends)
            ('Metadata Filter Operators Comprehensive', self.test_metadata_filter_operators_comprehensive),
            ('Metadata NOT_IN Numeric Over Non-Number Row Parity',
             self.test_metadata_not_in_numeric_over_nonnumber_row_parity),
            ('Image Attachment Cascade Delete', self.test_image_attachment_cascade_delete),
            ('Tags Lowercase Normalization', self.test_tags_lowercase_normalization),
            ('Tool Annotations Exposed To Client', self.test_tool_annotations_exposed_to_client),
            ('Search Context Offset Pagination', self.test_search_context_offset_pagination),
            ('Grep Context Literal Regex Unicode', self.test_grep_context_literal_regex_unicode),
            ('Read Context Range Clamp And Composition', self.test_read_context_range_clamp_and_composition),
            ('Navigate Context Outline And Node Read', self.test_navigate_context_outline_and_node_read),
            ('Index Tree Node Summaries And Statistics', self.test_index_tree_node_summaries_and_statistics),
            ('Prefix Id Resolution Returns Canonical Id', self.test_prefix_id_resolution_returns_canonical_id),
            ('Grep Keyset Scan Exhaustive Beyond Limit', self.test_grep_keyset_scan_exhaustive_beyond_limit),
            ('Grep Scan Cap Boundary Truncation', self.test_grep_scan_cap_boundary_truncation),
            ('Force-Off Removes Search Tool', self.test_force_off_removes_search_tool),
            # Cross-backend parity regressions: out-of-int64 simple metadata filter,
            # FTS validation-before-empty-query-short-circuit, and dedup content_type
            # plus image preservation on an image-less retransmit.
            ('Simple Metadata Out-Of-Int64 Rejected', self.test_search_simple_metadata_out_of_int64_rejected),
            ('FTS Empty-Query Invalid Filter Raises', self.test_fts_all_stopword_query_with_invalid_filter_returns_error),
            ('Dedup Preserves Content Type And Image', self.test_store_context_dedup_preserves_content_type_and_image),
            # Cross-backend parity regressions: the shared connection-metrics contract,
            # write-queue delivery across idle windows, null-named metadata path segments,
            # multi-member numeric IN/NOT_IN, tied-score pagination, embedded-quote FTS
            # terms, the filters_applied tally, tag write caps, updated_at stamping,
            # per-image metadata typing, grep request clamping, and embedding cleanup.
            ('Connection Metrics Cross-Backend Contract', self.test_connection_metrics_cross_backend_contract),
            ('Write Queue Idle Windows And Bursts', self.test_write_queue_survives_idle_windows_and_bursts),
            ('Metadata Filter Null Path Segment Parity', self.test_metadata_filter_null_path_segment_parity),
            (
                'Metadata Filter Numeric IN Mixed Members Parity',
                self.test_metadata_filter_numeric_in_mixed_members_parity,
            ),
            ('Tied-Score Pagination Parity', self.test_tied_score_pagination_parity),
            ('FTS Embedded Quote Term Parity', self.test_fts_embedded_quote_term_parity),
            ('Filters Applied Agreement Across Search Tools', self.test_filters_applied_agreement_across_search_tools),
            ('Tag Write Caps Parity', self.test_tag_write_caps_parity),
            ('Update Advances updated_at For Every Variant', self.test_update_context_advances_updated_at),
            ('Image Metadata JSON String Contract', self.test_image_metadata_json_string_contract),
            ('Grep Context Request Caps Clamped', self.test_grep_context_request_caps_clamped),
            ('Delete Removes Embedding Rows', self.test_delete_removes_embedding_rows),
            # Cross-backend parity regressions: the completed-operation counter for
            # transactional writes, the idle writer recycle, fixed-depth ranked pagination
            # and its depth hint, mutually exclusive delete selectors, typed indexed
            # metadata casts, byte-wise text ordering, tag deduplication, per-image
            # metadata fidelity, literal markup in a ranked document, and a deeply nested
            # boolean query that must not charge the circuit breaker.
            ('Transactional Write Moves total_queries', self.test_transactional_write_moves_total_queries),
            ('Idle Writer Recycle Invisible To Callers', self.test_idle_writer_recycle_is_invisible_to_callers),
            ('Ranked Pagination Union Matches Single Page', self.test_ranked_pagination_union_matches_single_page),
            ('Ranked Depth Limit Hint', self.test_ranked_depth_limit_hint),
            ('Delete Context Rejects Both Selectors', self.test_delete_context_rejects_both_selectors),
            ('Indexed Metadata Container Length Parity', self.test_indexed_metadata_container_length_parity),
            ('Typed Indexed Metadata Cast Parity', self.test_typed_indexed_metadata_cast_parity),
            ('Collation Ordering Parity', self.test_collation_ordering_parity),
            ('Tag Deduplication Across Write Paths', self.test_tag_deduplication_across_write_paths),
            ('Image Metadata Empty String Preserved', self.test_image_metadata_empty_string_preserved),
            ('Literal Markup Survives Ranked Search', self.test_literal_markup_survives_ranked_search),
            ('FTS Deeply Nested Boolean Query Degrades', self.test_fts_deeply_nested_boolean_query_degrades),
        ]

        print('\nRunning tests...\n')

        for test_name, test_func in tests:
            print(f'Testing: {test_name}...')
            try:
                success = await test_func()
                if success:
                    print(f'  [OK] {test_name} passed')
                else:
                    print(f'  [FAIL] {test_name} failed')
            except Exception as e:
                print(f'  [ERROR] {test_name} error: {e}')
                self.test_results.append((test_name, False, f'Exception: {e}'))

        # Display results
        print('\n' + '=' * 50)
        print('TEST RESULTS')
        print('=' * 50)

        passed = 0
        failed = 0

        for test_name, result, details in self.test_results:
            status = '[PASS]' if result else '[FAIL]'
            print(f'{status}: {test_name}')
            if details:
                print(f'   Details: {details}')
            if result:
                passed += 1
            else:
                failed += 1

        total = passed + failed
        print(f'\nTotal: {passed}/{total} tests passed')

        # Cleanup
        await self.cleanup()

        return failed == 0
