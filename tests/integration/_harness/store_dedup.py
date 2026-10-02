"""Real-server checks for ``store_context`` deduplication.

A retransmitted entry updates the stored one: metadata, tags, content type
and image attachments follow the deduplication merge rules, and an
opposite-source entry in between suppresses deduplication.
"""

import asyncio

from tests.integration._harness.core import HarnessCore


class StoreDedupMixin(HarnessCore):
    """Checks for store_context deduplication."""

    async def test_store_context_dedup_preserves_content_type_and_image(self) -> bool:
        """A dedup retransmit WITHOUT images preserves the stored multimodal content_type
        and the existing image attachment on BOTH backends (content_type resolves via
        COALESCE(NULL, existing) when no images are provided on the duplicate).

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_dedup_preserves_content_type_and_image'
        assert self.client is not None  # Type guard for Pyright
        try:
            dedup_ct_thread = f'{self.test_thread_id}_dedup_content_type'
            text = 'Multimodal entry preserved across an image-less dedup retransmit'

            store1 = await self.client.call_tool('store_context', {
                'thread_id': dedup_ct_thread, 'source': 'agent', 'text': text,
                'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
            })
            data1 = self._extract_content(store1)
            if not data1.get('success'):
                self.test_results.append((test_name, False, f'First store failed: {data1}'))
                return False
            context_id = data1.get('context_id')

            # Retransmit the identical text with NO images (network-retry shape).
            store2 = await self.client.call_tool('store_context', {
                'thread_id': dedup_ct_thread, 'source': 'agent', 'text': text,
            })
            data2 = self._extract_content(store2)
            if not data2.get('success'):
                self.test_results.append((test_name, False, f'Dedup store failed: {data2}'))
                return False

            get_result = await self.client.call_tool('get_context_by_ids', {
                'context_ids': [context_id], 'include_images': True,
            })
            get_data = self._extract_content(get_result)
            entry = get_data.get('results', [{}])[0]

            if entry.get('content_type') != 'multimodal':
                self.test_results.append(
                    (test_name, False, f"content_type not preserved across dedup: {entry.get('content_type')}"),
                )
                return False
            if not entry.get('images'):
                self.test_results.append((test_name, False, 'Image attachment lost across dedup'))
                return False

            self.test_results.append(
                (test_name, True, 'Dedup preserved multimodal content_type and image'),
            )
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_deduplication_data_integrity(self) -> bool:
        """Test that deduplication correctly handles metadata, tags, and timestamps.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_deduplication_data_integrity'
        assert self.client is not None  # Type guard for Pyright
        try:
            dedup_thread = f'{self.test_thread_id}_dedup_integrity'

            # 1. Store first entry with metadata and tags
            result1 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': dedup_thread,
                    'source': 'agent',
                    'text': 'Dedup integrity test content',
                    'metadata': {'key': 'original'},
                    'tags': ['a', 'b'],
                },
            )
            data1 = self._extract_content(result1)
            if not data1.get('success'):
                self.test_results.append(
                    (test_name, False, f'First store failed: {data1}'),
                )
                return False

            context_id = data1['context_id']

            # 2. Wait for timestamp separation (SQLite second precision)
            await asyncio.sleep(1.1)

            # 3. Store duplicate with updated metadata and tags
            result2 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': dedup_thread,
                    'source': 'agent',
                    'text': 'Dedup integrity test content',
                    'metadata': {'key': 'updated', 'extra': 'value'},
                    'tags': ['c', 'd'],
                },
            )
            data2 = self._extract_content(result2)
            if not data2.get('success'):
                self.test_results.append(
                    (test_name, False, f'Dedup store failed: {data2}'),
                )
                return False

            # Verify same context_id (dedup occurred)
            if data2['context_id'] != context_id:
                self.test_results.append(
                    (test_name, False,
                     f'Expected same context_id {context_id}, got {data2["context_id"]}'),
                )
                return False

            # 4. Retrieve and verify via get_context_by_ids
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

            entry = results[0]

            # Verify metadata was updated (not the original)
            entry_metadata = entry.get('metadata', {})
            if entry_metadata.get('key') != 'updated' or entry_metadata.get('extra') != 'value':
                self.test_results.append(
                    (test_name, False,
                     f'Metadata not updated correctly: {entry_metadata}'),
                )
                return False

            # Verify tags were replaced (not accumulated)
            entry_tags = sorted(entry.get('tags', []))
            if entry_tags != ['c', 'd']:
                self.test_results.append(
                    (test_name, False,
                     f'Tags not replaced correctly, expected [c, d], got {entry_tags}'),
                )
                return False

            # Verify updated_at differs from created_at
            if entry.get('created_at') == entry.get('updated_at'):
                self.test_results.append(
                    (test_name, False,
                     'updated_at should differ from created_at after dedup'),
                )
                return False

            self.test_results.append(
                (test_name, True,
                 'Dedup correctly updates metadata, replaces tags, and updates timestamp'),
            )
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_dedup_preserves_tags_when_none(self) -> bool:
        """Verify dedup PRESERVES existing tags when new entry provides tags=None.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_dedup_preserves_tags_when_none'
        assert self.client is not None
        try:
            dedup_tags_thread = f'{self.test_thread_id}_dedup_tags'

            store1 = await self.client.call_tool('store_context', {
                'thread_id': dedup_tags_thread, 'source': 'agent',
                'text': 'Entry with important tags for dedup test',
                'tags': ['important', 'preserve-me'],
            })
            data1 = self._extract_content(store1)
            if not data1.get('success'):
                self.test_results.append((test_name, False, f'First store failed: {data1}'))
                return False
            context_id = data1.get('context_id')

            store2 = await self.client.call_tool('store_context', {
                'thread_id': dedup_tags_thread, 'source': 'agent',
                'text': 'Entry with important tags for dedup test',
            })
            data2 = self._extract_content(store2)
            if not data2.get('success'):
                self.test_results.append((test_name, False, f'Dedup store failed: {data2}'))
                return False

            get_result = await self.client.call_tool('get_context_by_ids', {
                'context_ids': [context_id],
            })
            get_data = self._extract_content(get_result)
            entry = get_data.get('results', [{}])[0]
            tags = entry.get('tags', [])

            if 'important' not in tags or 'preserve-me' not in tags:
                self.test_results.append((test_name, False,
                    f'Tags not preserved during dedup: {tags}'))
                return False

            self.test_results.append((test_name, True, f'Dedup preserved tags: {tags}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_dedup_interleaving_check(self) -> bool:
        """Verify dedup is suppressed when opposite-source entries interleave.

        Stores user "Proceed", then agent "Working...", then user "Proceed" again.
        The second user "Proceed" must create a NEW entry (not update the first one)
        to preserve chronological ordering.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_dedup_interleaving_check'
        assert self.client is not None
        try:
            interleave_thread = f'{self.test_thread_id}_interleave'

            # 1. Store user "Proceed"
            result1 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': interleave_thread,
                    'source': 'user',
                    'text': 'Proceed',
                },
            )
            data1 = self._extract_content(result1)
            if not data1.get('success'):
                self.test_results.append(
                    (test_name, False, f'First store failed: {data1}'),
                )
                return False
            first_user_id = data1['context_id']

            # 2. Store agent "Working..."
            result2 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': interleave_thread,
                    'source': 'agent',
                    'text': 'Working...',
                },
            )
            data2 = self._extract_content(result2)
            if not data2.get('success'):
                self.test_results.append(
                    (test_name, False, f'Agent store failed: {data2}'),
                )
                return False

            # 3. Store user "Proceed" again (should be NEW entry, not dedup)
            result3 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': interleave_thread,
                    'source': 'user',
                    'text': 'Proceed',
                },
            )
            data3 = self._extract_content(result3)
            if not data3.get('success'):
                self.test_results.append(
                    (test_name, False, f'Second user store failed: {data3}'),
                )
                return False
            second_user_id = data3['context_id']

            # 4. Verify the two user entries have different IDs
            if second_user_id == first_user_id:
                self.test_results.append(
                    (test_name, False,
                     f'Interleaving check failed: both user entries have same ID {first_user_id}'),
                )
                return False

            # 5. Verify 3 distinct entries exist in chronological order
            search_result = await self.client.call_tool(
                'search_context',
                {
                    'thread_id': interleave_thread,
                    'limit': 100,
                },
            )
            search_data = self._extract_content(search_result)
            results = search_data.get('results', [])
            if len(results) != 3:
                self.test_results.append(
                    (test_name, False,
                     f'Expected 3 entries, got {len(results)}'),
                )
                return False

            self.test_results.append((test_name, True,
                f'Interleaving check works: user IDs {first_user_id} != {second_user_id}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
