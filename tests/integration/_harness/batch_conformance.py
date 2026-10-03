"""Real-server checks that a one-entry batch matches its single-entry tool.

``store_context_batch``, ``update_context_batch`` and ``delete_context_batch``
called with one entry leave the same database state and return the same
result as ``store_context``, ``update_context`` and ``delete_context``.
"""

from tests.integration._harness.core import HarnessCore


class BatchConformanceMixin(HarnessCore):
    """Checks for batch and single-entry tool conformance."""

    async def test_store_batch_single_matches_store_nonbatch(self) -> bool:
        """Conformance: store_context_batch([single]) produces same DB state as store_context.

        Returns:
            bool: True if test passed.
        """
        test_name = 'store_batch_single_matches_store_nonbatch'
        try:
            assert self.client is not None
            # Store via non-batch
            nonbatch_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_conform_store_nb',
                    'source': 'user',
                    'text': 'Conformance test entry for store',
                    'metadata': {'conformance': True, 'value': 42},
                    'tags': ['conformance', 'store'],
                },
            )
            nb_data = self._extract_content(nonbatch_result)
            if not nb_data.get('success'):
                self.test_results.append((test_name, False, f'Non-batch store failed: {nb_data}'))
                return False

            # Store via batch (single entry)
            batch_result = await self.client.call_tool(
                'store_context_batch',
                {
                    'entries': [{
                        'thread_id': f'{self.test_thread_id}_conform_store_b',
                        'source': 'user',
                        'text': 'Conformance test entry for store',
                        'metadata': {'conformance': True, 'value': 42},
                        'tags': ['conformance', 'store'],
                    }],
                    'atomic': True,
                },
            )
            b_data = self._extract_content(batch_result)
            if not b_data.get('success') or b_data.get('succeeded') != 1:
                self.test_results.append((test_name, False, f'Batch store failed: {b_data}'))
                return False

            # Read back both entries via get_context_by_ids
            nb_id = nb_data['context_id']
            b_id = b_data['results'][0]['context_id']

            nb_entry_result = await self.client.call_tool(
                'get_context_by_ids', {'context_ids': [nb_id]},
            )
            b_entry_result = await self.client.call_tool(
                'get_context_by_ids', {'context_ids': [b_id]},
            )

            nb_entries = self._extract_content(nb_entry_result).get('results', [])
            b_entries = self._extract_content(b_entry_result).get('results', [])

            if len(nb_entries) != 1 or len(b_entries) != 1:
                self.test_results.append((test_name, False, 'Failed to read back entries'))
                return False

            nb_entry = nb_entries[0]
            b_entry = b_entries[0]

            # Compare DB state (excluding id, thread_id, timestamps)
            for field in ['source', 'content_type', 'text_content']:
                if nb_entry.get(field) != b_entry.get(field):
                    self.test_results.append((
                        test_name, False,
                        f'Field {field} mismatch: nb={nb_entry.get(field)} vs b={b_entry.get(field)}',
                    ))
                    return False

            # Compare metadata
            nb_meta = nb_entry.get('metadata', {})
            b_meta = b_entry.get('metadata', {})
            if nb_meta != b_meta:
                self.test_results.append((
                    test_name, False,
                    f'Metadata mismatch: nb={nb_meta} vs b={b_meta}',
                ))
                return False

            # Compare tags
            nb_tags = sorted(nb_entry.get('tags', []))
            b_tags = sorted(b_entry.get('tags', []))
            if nb_tags != b_tags:
                self.test_results.append((
                    test_name, False,
                    f'Tags mismatch: nb={nb_tags} vs b={b_tags}',
                ))
                return False

            self.test_results.append((test_name, True, 'Store conformance verified'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_update_batch_single_matches_update_nonbatch(self) -> bool:
        """Conformance: update_context_batch([single]) produces same DB state as update_context.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_batch_single_matches_update_nonbatch'
        try:
            assert self.client is not None
            # Create two entries (one for each path)
            nb_store = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_conform_upd_nb',
                    'source': 'user',
                    'text': 'Original text for update conformance',
                    'tags': ['original'],
                },
            )
            b_store = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_conform_upd_b',
                    'source': 'user',
                    'text': 'Original text for update conformance',
                    'tags': ['original'],
                },
            )

            nb_id = self._extract_content(nb_store)['context_id']
            b_id = self._extract_content(b_store)['context_id']

            # Update both with identical parameters
            await self.client.call_tool(
                'update_context',
                {
                    'context_id': nb_id,
                    'text': 'Updated conformance text',
                    'tags': ['updated', 'conformance'],
                },
            )
            await self.client.call_tool(
                'update_context_batch',
                {
                    'updates': [{
                        'context_id': b_id,
                        'text': 'Updated conformance text',
                        'tags': ['updated', 'conformance'],
                    }],
                    'atomic': True,
                },
            )

            # Read back both
            nb_read = await self.client.call_tool(
                'get_context_by_ids', {'context_ids': [nb_id]},
            )
            b_read = await self.client.call_tool(
                'get_context_by_ids', {'context_ids': [b_id]},
            )

            nb_entry = self._extract_content(nb_read).get('results', [])[0]
            b_entry = self._extract_content(b_read).get('results', [])[0]

            # Compare updated fields
            for field in ['source', 'content_type', 'text_content']:
                if nb_entry.get(field) != b_entry.get(field):
                    self.test_results.append((
                        test_name, False,
                        f'Field {field} mismatch after update: nb={nb_entry.get(field)} vs b={b_entry.get(field)}',
                    ))
                    return False

            nb_tags = sorted(nb_entry.get('tags', []))
            b_tags = sorted(b_entry.get('tags', []))
            if nb_tags != b_tags:
                self.test_results.append((
                    test_name, False,
                    f'Tags mismatch after update: nb={nb_tags} vs b={b_tags}',
                ))
                return False

            self.test_results.append((test_name, True, 'Update conformance verified'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_batch_single_matches_delete_nonbatch(self) -> bool:
        """Conformance: delete_context_batch([single]) produces same result as delete_context.

        Returns:
            bool: True if test passed.
        """
        test_name = 'delete_batch_single_matches_delete_nonbatch'
        try:
            assert self.client is not None
            # Create two entries
            nb_store = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_conform_del_nb',
                    'source': 'user',
                    'text': 'Delete conformance entry',
                },
            )
            b_store = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': f'{self.test_thread_id}_conform_del_b',
                    'source': 'user',
                    'text': 'Delete conformance entry',
                },
            )

            nb_id = self._extract_content(nb_store)['context_id']
            b_id = self._extract_content(b_store)['context_id']

            # Delete via non-batch
            nb_del = await self.client.call_tool(
                'delete_context', {'context_ids': [nb_id]},
            )
            nb_del_data = self._extract_content(nb_del)

            # Delete via batch
            b_del = await self.client.call_tool(
                'delete_context_batch', {'context_ids': [b_id]},
            )
            b_del_data = self._extract_content(b_del)

            # Both should report deleted_count == 1
            nb_count = nb_del_data.get('deleted_count', 0)
            b_count = b_del_data.get('deleted_count', 0)

            if nb_count != 1:
                self.test_results.append((test_name, False,
                    f'Non-batch deleted_count={nb_count}, expected 1'))
                return False
            if b_count != 1:
                self.test_results.append((test_name, False,
                    f'Batch deleted_count={b_count}, expected 1'))
                return False

            # Verify both entries are gone
            nb_check = await self.client.call_tool(
                'get_context_by_ids', {'context_ids': [nb_id]},
            )
            b_check = await self.client.call_tool(
                'get_context_by_ids', {'context_ids': [b_id]},
            )

            nb_results = self._extract_content(nb_check).get('results', [])
            b_results = self._extract_content(b_check).get('results', [])

            if len(nb_results) != 0:
                self.test_results.append((test_name, False,
                    'Non-batch entry still exists after delete'))
                return False
            if len(b_results) != 0:
                self.test_results.append((test_name, False,
                    'Batch entry still exists after delete'))
                return False

            self.test_results.append((test_name, True, 'Delete conformance verified'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
