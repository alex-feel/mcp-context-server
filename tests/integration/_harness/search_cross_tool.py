"""Real-server checks for the arguments every search tool shares.

The ``content_type`` filter and the ``include_images`` flag across
``search_context``, ``semantic_search_context``, ``fts_search_context`` and
``hybrid_search_context``, and the ``tags`` filter across the semantic,
full-text and hybrid search tools.
"""

import asyncio

from tests.integration._harness.core import HarnessCore


class SearchCrossToolMixin(HarnessCore):
    """Checks for the content_type, include_images and tags arguments across the search tools."""

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
