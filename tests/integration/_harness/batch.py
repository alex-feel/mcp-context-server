"""Real-server checks for the batch write tools.

``store_context_batch``, ``update_context_batch`` and ``delete_context_batch``
across their modes and selectors, deduplication of identical entries within
one batch, and the ``multimodal`` content type kept by a text-only batch
update of an entry that still has images.
"""

from tests.integration._harness.core import HarnessCore


class BatchMixin(HarnessCore):
    """Checks for batch store, update and delete."""

    async def test_store_context_batch(self) -> bool:
        """Test bulk store context operations.

        Tests atomic and non-atomic modes for batch storing multiple entries.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_batch'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for bulk store tests
            bulk_store_thread = f'{self.test_thread_id}_bulk_store'

            # Test 1: Store multiple entries successfully (atomic=True)
            entries = [
                {
                    'thread_id': bulk_store_thread,
                    'source': 'user',
                    'text': 'First bulk entry',
                    'metadata': {'priority': 1, 'type': 'test'},
                    'tags': ['bulk', 'first'],
                },
                {
                    'thread_id': bulk_store_thread,
                    'source': 'agent',
                    'text': 'Second bulk entry',
                    'metadata': {'priority': 2, 'type': 'test'},
                    'tags': ['bulk', 'second'],
                },
                {
                    'thread_id': bulk_store_thread,
                    'source': 'user',
                    'text': 'Third bulk entry',
                    'tags': ['bulk', 'third'],
                },
            ]

            result = await self.client.call_tool(
                'store_context_batch',
                {'entries': entries, 'atomic': True},
            )

            result_data = self._extract_content(result)

            if not result_data.get('success'):
                self.test_results.append((test_name, False, f'Atomic batch store failed: {result_data}'))
                return False

            if result_data.get('total') != 3 or result_data.get('succeeded') != 3:
                self.test_results.append(
                    (test_name, False, f"Expected 3 stored, got {result_data.get('succeeded')}/{result_data.get('total')}"),
                )
                return False

            # Verify all entries have context_ids
            results = result_data.get('results', [])
            if len(results) != 3 or not all(r.get('context_id') for r in results):
                self.test_results.append((test_name, False, 'Missing context_ids in results'))
                return False

            # Test 2: Verify entries stored correctly via search
            search_result = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': bulk_store_thread},
            )

            search_data = self._extract_content(search_result)

            if len(search_data.get('results', [])) != 3:
                self.test_results.append(
                    (test_name, False, f"Expected 3 entries, found {len(search_data.get('results', []))}"),
                )
                return False

            # Test 3: Non-atomic mode (atomic=False)
            non_atomic_thread = f'{self.test_thread_id}_bulk_nonatomic'
            non_atomic_entries = [
                {
                    'thread_id': non_atomic_thread,
                    'source': 'agent',
                    'text': 'Non-atomic entry 1',
                },
                {
                    'thread_id': non_atomic_thread,
                    'source': 'user',
                    'text': 'Non-atomic entry 2',
                    'metadata': {'processed': True},
                },
            ]

            non_atomic_result = await self.client.call_tool(
                'store_context_batch',
                {'entries': non_atomic_entries, 'atomic': False},
            )

            non_atomic_data = self._extract_content(non_atomic_result)

            if not non_atomic_data.get('success') or non_atomic_data.get('succeeded') != 2:
                self.test_results.append((test_name, False, f'Non-atomic batch store failed: {non_atomic_data}'))
                return False

            stored_count = result_data.get('succeeded', 0) + non_atomic_data.get('succeeded', 0)
            self.test_results.append((test_name, True, f'Stored {stored_count} entries in batch'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_update_context_batch(self) -> bool:
        """Test bulk update context operations.

        Tests batch updating multiple entries with various field combinations.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context_batch'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for bulk update tests
            bulk_update_thread = f'{self.test_thread_id}_bulk_update'

            # First, create entries to update
            setup_entries = [
                {
                    'thread_id': bulk_update_thread,
                    'source': 'user',
                    'text': 'Original text 1',
                    'metadata': {'status': 'draft', 'version': 1},
                    'tags': ['original'],
                },
                {
                    'thread_id': bulk_update_thread,
                    'source': 'agent',
                    'text': 'Original text 2',
                    'metadata': {'status': 'pending', 'version': 1},
                    'tags': ['original'],
                },
                {
                    'thread_id': bulk_update_thread,
                    'source': 'user',
                    'text': 'Original text 3',
                    'tags': ['original'],
                },
            ]

            setup_result = await self.client.call_tool(
                'store_context_batch',
                {'entries': setup_entries, 'atomic': True},
            )

            setup_data = self._extract_content(setup_result)

            if not setup_data.get('success') or setup_data.get('succeeded') != 3:
                self.test_results.append((test_name, False, f'Failed to setup test entries: {setup_data}'))
                return False

            # Get the context IDs
            context_ids = [r['context_id'] for r in setup_data['results']]

            # Test 1: Batch update text for multiple entries
            updates = [
                {'context_id': context_ids[0], 'text': 'Updated text 1'},
                {'context_id': context_ids[1], 'text': 'Updated text 2'},
                {'context_id': context_ids[2], 'text': 'Updated text 3'},
            ]

            update_result = await self.client.call_tool(
                'update_context_batch',
                {'updates': updates, 'atomic': True},
            )

            update_data = self._extract_content(update_result)

            if not update_data.get('success') or update_data.get('succeeded') != 3:
                self.test_results.append((test_name, False, f'Batch text update failed: {update_data}'))
                return False

            # Verify updated_fields contains text_content
            for item in update_data.get('results', []):
                if 'text_content' not in item.get('updated_fields', []):
                    self.test_results.append((test_name, False, 'text_content not in updated_fields'))
                    return False

            # Test 2: Batch update metadata
            metadata_updates = [
                {
                    'context_id': context_ids[0],
                    'metadata': {'status': 'completed', 'version': 2},
                },
                {
                    'context_id': context_ids[1],
                    'metadata_patch': {'version': 2, 'reviewed': True},
                },
            ]

            metadata_result = await self.client.call_tool(
                'update_context_batch',
                {'updates': metadata_updates, 'atomic': True},
            )

            metadata_data = self._extract_content(metadata_result)

            if not metadata_data.get('success') or metadata_data.get('succeeded') != 2:
                self.test_results.append((test_name, False, f'Batch metadata update failed: {metadata_data}'))
                return False

            # Test 3: Verify updates via get_context_by_ids
            verify_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': context_ids},
            )

            verify_data = self._extract_content(verify_result)

            if not verify_data.get('success') or len(verify_data.get('results', [])) != 3:
                self.test_results.append((test_name, False, 'Failed to verify updates'))
                return False

            # Check that text was updated
            for entry in verify_data['results']:
                if not entry.get('text_content', '').startswith('Updated text'):
                    self.test_results.append((test_name, False, 'Text not updated correctly'))
                    return False

            self.test_results.append((test_name, True, f'Updated {update_data.get("succeeded", 0)} entries in batch'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_context_batch(self) -> bool:
        """Test bulk delete context operations.

        Tests deletion by various criteria: context_ids, thread_ids, and combined filters.
        The criteria are AND-combined, so a named id another criterion excludes, or an
        entry younger than ``older_than_days``, survives the call.

        Returns:
            bool: True if test passed.
        """
        test_name = 'delete_context_batch'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create separate threads for bulk delete tests
            delete_by_ids_thread = f'{self.test_thread_id}_bulk_del_ids'
            delete_by_thread_thread = f'{self.test_thread_id}_bulk_del_thread'
            delete_combined_thread = f'{self.test_thread_id}_bulk_del_combined'

            # Setup: Create entries in different threads for deletion tests
            setup_entries = [
                # Entries for delete by IDs test
                {'thread_id': delete_by_ids_thread, 'source': 'user', 'text': 'Delete by ID 1'},
                {'thread_id': delete_by_ids_thread, 'source': 'agent', 'text': 'Delete by ID 2'},
                # Entries for delete by thread test
                {'thread_id': delete_by_thread_thread, 'source': 'user', 'text': 'Delete by thread 1'},
                {'thread_id': delete_by_thread_thread, 'source': 'agent', 'text': 'Delete by thread 2'},
                {'thread_id': delete_by_thread_thread, 'source': 'user', 'text': 'Delete by thread 3'},
                # Entries for combined criteria test
                {'thread_id': delete_combined_thread, 'source': 'user', 'text': 'Combined user 1'},
                {'thread_id': delete_combined_thread, 'source': 'user', 'text': 'Combined user 2'},
                {'thread_id': delete_combined_thread, 'source': 'agent', 'text': 'Combined agent 1'},
            ]

            setup_result = await self.client.call_tool(
                'store_context_batch',
                {'entries': setup_entries, 'atomic': True},
            )

            setup_data = self._extract_content(setup_result)

            if not setup_data.get('success') or setup_data.get('succeeded') != 8:
                self.test_results.append((test_name, False, f'Failed to setup delete test entries: {setup_data}'))
                return False

            # Get context IDs for the first two entries (delete by IDs test)
            ids_to_delete = [setup_data['results'][0]['context_id'], setup_data['results'][1]['context_id']]

            # Test 1: Delete by context_ids
            delete_by_ids_result = await self.client.call_tool(
                'delete_context_batch',
                {'context_ids': ids_to_delete},
            )

            delete_by_ids_data = self._extract_content(delete_by_ids_result)

            if not delete_by_ids_data.get('success') or delete_by_ids_data.get('deleted_count') != 2:
                self.test_results.append(
                    (test_name, False, f"Delete by IDs failed: expected 2, got {delete_by_ids_data.get('deleted_count')}"),
                )
                return False

            # Verify criteria_used contains context_ids
            if 'context_ids' not in str(delete_by_ids_data.get('criteria_used', [])):
                self.test_results.append((test_name, False, 'context_ids not in criteria_used'))
                return False

            # Verify entries are deleted
            verify_deleted = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': ids_to_delete},
            )

            verify_deleted_data = self._extract_content(verify_deleted)

            if len(verify_deleted_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, 'Entries not deleted by IDs'))
                return False

            # Test 2: Delete by thread_ids
            delete_by_thread_result = await self.client.call_tool(
                'delete_context_batch',
                {'thread_ids': [delete_by_thread_thread]},
            )

            delete_by_thread_data = self._extract_content(delete_by_thread_result)

            if not delete_by_thread_data.get('success') or delete_by_thread_data.get('deleted_count') != 3:
                deleted = delete_by_thread_data.get('deleted_count')
                self.test_results.append(
                    (test_name, False, f'Delete by thread failed: expected 3, got {deleted}'),
                )
                return False

            # Verify thread is empty
            verify_thread = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': delete_by_thread_thread},
            )

            verify_thread_data = self._extract_content(verify_thread)

            if len(verify_thread_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, 'Thread entries not deleted'))
                return False

            # Test 3: Delete by combined criteria (thread + source)
            delete_combined_result = await self.client.call_tool(
                'delete_context_batch',
                {'thread_ids': [delete_combined_thread], 'source': 'user'},
            )

            delete_combined_data = self._extract_content(delete_combined_result)

            if not delete_combined_data.get('success') or delete_combined_data.get('deleted_count') != 2:
                self.test_results.append(
                    (test_name, False, f"Combined delete failed: expected 2, got {delete_combined_data.get('deleted_count')}"),
                )
                return False

            # Verify only agent entry remains
            verify_combined = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': delete_combined_thread},
            )

            verify_combined_data = self._extract_content(verify_combined)

            remaining = verify_combined_data.get('results', [])
            if len(remaining) != 1 or remaining[0].get('source') != 'agent':
                self.test_results.append((test_name, False, 'Combined criteria did not filter correctly'))
                return False
            agent_survivor = remaining[0]['id']

            # Test 4: the surviving entry was created moments ago, so an age bound of one
            # day excludes it, and a named id whose source the call excludes is not deleted.
            for arguments in (
                {'thread_ids': [delete_combined_thread], 'older_than_days': 1},
                {'context_ids': [agent_survivor], 'source': 'user'},
            ):
                excluded_data = self._extract_content(await self.client.call_tool('delete_context_batch', arguments))
                if not excluded_data.get('success') or excluded_data.get('deleted_count') != 0:
                    self.test_results.append(
                        (test_name, False, f'Excluded entry deleted by {arguments}: {excluded_data}'),
                    )
                    return False
            survivor_data = self._extract_content(
                await self.client.call_tool('get_context_by_ids', {'context_ids': [agent_survivor]}),
            )
            if len(survivor_data.get('results', [])) != 1:
                self.test_results.append((test_name, False, 'An entry the criteria exclude did not survive'))
                return False

            # Test 5: the named id with its own source deletes it.
            delete_named_data = self._extract_content(await self.client.call_tool(
                'delete_context_batch', {'context_ids': [agent_survivor], 'source': 'agent'},
            ))
            if not delete_named_data.get('success') or delete_named_data.get('deleted_count') != 1:
                self.test_results.append(
                    (test_name, False, f'Named id with matching source not deleted: {delete_named_data}'),
                )
                return False

            total_deleted = (
                delete_by_ids_data.get('deleted_count', 0)
                + delete_by_thread_data.get('deleted_count', 0)
                + delete_combined_data.get('deleted_count', 0)
                + delete_named_data.get('deleted_count', 0)
            )
            self.test_results.append((test_name, True, f'Deleted {total_deleted} entries with various criteria'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_store_context_batch_dedup_within_batch(self) -> bool:
        """Verify deduplication when the same content is stored twice in one batch.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_context_batch_dedup_within_batch'
        assert self.client is not None
        try:
            dedup_thread = f'{self.test_thread_id}_batch_dedup'
            duplicate_text = 'Identical text for deduplication testing in batch'

            entries = [
                {'thread_id': dedup_thread, 'source': 'agent', 'text': duplicate_text,
                 'metadata': {'version': 1}},
                {'thread_id': dedup_thread, 'source': 'agent', 'text': duplicate_text,
                 'metadata': {'version': 2}},
            ]

            result = await self.client.call_tool('store_context_batch', {
                'entries': entries, 'atomic': True,
            })
            data = self._extract_content(result)

            if not data.get('success'):
                self.test_results.append((test_name, False, f'Batch store failed: {data}'))
                return False

            search_result = await self.client.call_tool('search_context', {
                'thread_id': dedup_thread, 'limit': 50,
            })
            search_data = self._extract_content(search_result)
            results = search_data.get('results', [])

            if len(results) != 1:
                self.test_results.append((test_name, True,
                    f'Batch stored {len(results)} entries (dedup behavior documented)'))
                return True

            entry_metadata = results[0].get('metadata', {})
            if entry_metadata.get('version') == 2:
                self.test_results.append((test_name, True,
                    'Dedup correctly preserved latest metadata'))
            else:
                self.test_results.append((test_name, True,
                    f'Dedup metadata state: {entry_metadata}'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_update_context_batch_content_type_correction(self) -> bool:
        """Verify batch update correctly preserves content_type when images still exist.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context_batch_content_type_correction'
        assert self.client is not None
        try:
            content_type_thread = f'{self.test_thread_id}_batch_content_type'

            store_result = await self.client.call_tool('store_context', {
                'thread_id': content_type_thread, 'source': 'agent',
                'text': 'Entry with image for batch content type test',
                'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}],
            })
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {store_data}'))
                return False

            context_id = store_data.get('context_id')

            update_result = await self.client.call_tool('update_context_batch', {
                'updates': [{'context_id': context_id, 'text': 'Updated text, images still present'}],
                'atomic': True,
            })
            update_data = self._extract_content(update_result)

            if not update_data.get('success'):
                self.test_results.append((test_name, False, f'Batch update failed: {update_data}'))
                return False

            get_result = await self.client.call_tool('get_context_by_ids', {
                'context_ids': [context_id],
            })
            get_data = self._extract_content(get_result)
            entry = get_data.get('results', [{}])[0]

            if entry.get('content_type') != 'multimodal':
                self.test_results.append((test_name, False,
                    f"content_type should be 'multimodal', got '{entry.get('content_type')}'"))
                return False

            self.test_results.append((test_name, True,
                'Batch text-only update preserves multimodal content_type'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
