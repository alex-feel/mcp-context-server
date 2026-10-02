"""Real-server checks for the JSON-string deserializer middleware.

List and dict parameters sent as JSON-encoded strings (``tags``,
``metadata``, ``context_ids``) reach the tools deserialized, while a
string parameter whose value looks like JSON stays a string.
"""

from tests.integration._harness.core import HarnessCore


class MiddlewareMixin(HarnessCore):
    """Checks for JSON-string parameter deserialization."""

    async def test_middleware_deserializes_stringified_tags(self) -> bool:
        """Test that the middleware deserializes JSON-stringified tags.

        Returns:
            bool: True if test passed.
        """
        test_name = 'middleware_deserializes_stringified_tags'
        assert self.client is not None  # Type guard for Pyright
        try:
            mw_thread = f'{self.test_thread_id}_middleware_tags'

            # Send tags as a JSON string (simulating client serialization bug)
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': mw_thread,
                    'source': 'agent',
                    'text': 'Middleware tag deserialization test',
                    'tags': '["tag1","tag2","tag3"]',
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Failed to store context: {store_data}'),
                )
                return False

            context_id = store_data['context_id']

            # Retrieve and verify tags were stored as a list
            search_result = await self.client.call_tool(
                'search_context',
                {'thread_id': mw_thread, 'tags': ['tag1']},
            )
            search_data = self._extract_content(search_result)
            if search_data.get('count', 0) < 1:
                self.test_results.append(
                    (test_name, False,
                     f'Tag search returned no results (tags not deserialized): {search_data}'),
                )
                return False

            # Verify all 3 tags are present via get_context_by_ids
            get_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])
            if len(results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result, got {len(results)}'),
                )
                return False

            entry_tags = sorted(results[0].get('tags', []))
            if entry_tags != ['tag1', 'tag2', 'tag3']:
                self.test_results.append(
                    (test_name, False,
                     f'Expected tags [tag1, tag2, tag3], got {entry_tags}'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'Stringified tags deserialized and stored correctly'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_middleware_deserializes_stringified_metadata(self) -> bool:
        """Test that the middleware deserializes JSON-stringified metadata.

        Returns:
            bool: True if test passed.
        """
        test_name = 'middleware_deserializes_stringified_metadata'
        assert self.client is not None  # Type guard for Pyright
        try:
            mw_thread = f'{self.test_thread_id}_middleware_metadata'

            # Send metadata as a JSON string
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': mw_thread,
                    'source': 'agent',
                    'text': 'Middleware metadata deserialization test',
                    'metadata': '{"agent_name":"test_agent","status":"done"}',
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Failed to store context: {store_data}'),
                )
                return False

            context_id = store_data['context_id']

            # Retrieve and verify metadata was stored as a dict
            get_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])
            if len(results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result, got {len(results)}'),
                )
                return False

            metadata = results[0].get('metadata', {})
            if metadata.get('agent_name') != 'test_agent' or metadata.get('status') != 'done':
                self.test_results.append(
                    (test_name, False,
                     f'Metadata not deserialized correctly: {metadata}'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'Stringified metadata deserialized and stored correctly'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_middleware_preserves_string_text(self) -> bool:
        """Test that the middleware does NOT deserialize string text content.

        Returns:
            bool: True if test passed.
        """
        test_name = 'middleware_preserves_string_text'
        assert self.client is not None  # Type guard for Pyright
        try:
            mw_thread = f'{self.test_thread_id}_middleware_text'

            # Store text that looks like JSON but should remain as-is
            json_like_text = '["this","is","text","not","a","list"]'
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': mw_thread,
                    'source': 'agent',
                    'text': json_like_text,
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Failed to store context: {store_data}'),
                )
                return False

            context_id = store_data['context_id']

            # Retrieve and verify text was NOT deserialized
            get_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])
            if len(results) != 1:
                self.test_results.append(
                    (test_name, False, f'Expected 1 result, got {len(results)}'),
                )
                return False

            stored_text = results[0].get('text_content', '')
            if stored_text != json_like_text:
                self.test_results.append(
                    (test_name, False,
                     f'Text was modified (over-deserialized): {stored_text!r}'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'String text preserved as-is (not over-deserialized)'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_middleware_deserializes_stringified_context_ids(self) -> bool:
        """Test that the middleware deserializes JSON-stringified context_ids.

        Returns:
            bool: True if test passed.
        """
        test_name = 'middleware_deserializes_stringified_context_ids'
        assert self.client is not None  # Type guard for Pyright
        try:
            mw_thread = f'{self.test_thread_id}_middleware_ids'

            # First store an entry to get a valid context_id
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': mw_thread,
                    'source': 'agent',
                    'text': 'Entry for context_ids deserialization test',
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append(
                    (test_name, False, f'Failed to store context: {store_data}'),
                )
                return False

            context_id = store_data['context_id']

            # Send context_ids as a JSON string (simulating client bug)
            get_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': f'["{context_id}"]'},
            )
            get_data = self._extract_content(get_result)
            results = get_data.get('results', [])
            if len(results) != 1:
                self.test_results.append(
                    (test_name, False,
                     f'Expected 1 result from stringified context_ids, got {len(results)}'),
                )
                return False

            if results[0].get('text_content') != 'Entry for context_ids deserialization test':
                self.test_results.append(
                    (test_name, False,
                     f'Text mismatch: {results[0].get("text_content")!r}'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'Stringified context_ids deserialized and entry retrieved'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
