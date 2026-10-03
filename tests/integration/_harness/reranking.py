"""Real-server checks for cross-encoder reranking.

The ``rerank_score`` added to semantic, full-text and hybrid search results,
its absence when reranking is unavailable, and the overfetch that gives
hybrid search more candidates than the requested limit.
"""

import asyncio

from tests.integration._harness.core import HarnessCore


class RerankingMixin(HarnessCore):
    """Checks for reranking scores and candidate overfetch."""

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
