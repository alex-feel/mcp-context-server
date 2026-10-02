"""Real-server checks for ``update_context``.

Text, metadata, tag and image updates, an unknown id, embedding regeneration
after a text change, and the ``updated_at`` advance on every update variant.
"""

import asyncio
from typing import Any

from tests.integration._harness.core import HarnessCore


class UpdateMixin(HarnessCore):
    """Checks for updating single context entries."""

    async def test_update_context(self) -> bool:
        """Test updating existing context entries.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for update tests
            update_thread = f'{self.test_thread_id}_update'

            # Store initial context
            initial_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': update_thread,
                    'source': 'agent',
                    'text': 'Initial text content',
                    'metadata': {'status': 'draft', 'priority': 1},
                    'tags': ['initial', 'test'],
                },
            )

            initial_data = self._extract_content(initial_result)

            if not initial_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to store initial context: {initial_data}'))
                return False

            context_id = initial_data.get('context_id')

            # Test 1: Update text only
            update_text_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'text': 'Updated text content',
                },
            )

            update_text_data = self._extract_content(update_text_result)

            if not update_text_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to update text: {update_text_data}'))
                return False

            # Verify text was updated
            verify_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_data = self._extract_content(verify_result)

            if not verify_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify text update'))
                return False

            updated_entry = verify_data['results'][0]

            if updated_entry.get('text_content') != 'Updated text content':
                self.test_results.append((test_name, False, 'Text not updated correctly'))
                return False

            # Test 2: Update metadata only
            update_metadata_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata': {'status': 'completed', 'priority': 10, 'reviewed': True},
                },
            )

            update_metadata_data = self._extract_content(update_metadata_result)

            if not update_metadata_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to update metadata: {update_metadata_data}'))
                return False

            # Test 3: Update tags (replacement)
            update_tags_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'tags': ['updated', 'final'],
                },
            )

            update_tags_data = self._extract_content(update_tags_result)

            if not update_tags_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to update tags: {update_tags_data}'))
                return False

            # Test 4: Add images (verify content_type changes to multimodal)
            update_images_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'images': [
                        {
                            'data': self._create_test_image(),
                            'mime_type': 'image/png',
                        },
                    ],
                },
            )

            update_images_data = self._extract_content(update_images_result)

            if not update_images_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to add images: {update_images_data}'))
                return False

            # Verify content_type changed to multimodal
            verify_multimodal = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id], 'include_images': True},
            )

            verify_multimodal_data = self._extract_content(verify_multimodal)

            if not verify_multimodal_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify multimodal update'))
                return False

            multimodal_entry = verify_multimodal_data['results'][0]

            if multimodal_entry.get('content_type') != 'multimodal':
                self.test_results.append((test_name, False, 'Content type not changed to multimodal'))
                return False

            # Test 5: Remove images (verify content_type changes back to text)
            remove_images_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'images': [],
                },
            )

            remove_images_data = self._extract_content(remove_images_result)

            if not remove_images_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to remove images: {remove_images_data}'))
                return False

            # Verify content_type changed back to text
            verify_text_type = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_text_type_data = self._extract_content(verify_text_type)

            if not verify_text_type_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify text-only update'))
                return False

            text_only_entry = verify_text_type_data['results'][0]

            if text_only_entry.get('content_type') != 'text':
                self.test_results.append((test_name, False, 'Content type not changed back to text'))
                return False

            # Test 6: Metadata patch - add new field to existing metadata
            # First restore metadata for patch testing
            restore_metadata_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata': {'status': 'active', 'priority': 5},
                },
            )

            restore_metadata_data = self._extract_content(restore_metadata_result)

            if not restore_metadata_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to restore metadata: {restore_metadata_data}'))
                return False

            # Now patch to add new field
            patch_add_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'new_field': 'added_value'},
                },
            )

            patch_add_data = self._extract_content(patch_add_result)

            if not patch_add_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch-add new field: {patch_add_data}'))
                return False

            # Verify patch added new field while preserving existing ones
            verify_patch_add = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_patch_add_data = self._extract_content(verify_patch_add)

            if not verify_patch_add_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch-add'))
                return False

            patched_metadata = verify_patch_add_data['results'][0].get('metadata', {})

            # Check that existing fields are preserved and new field was added
            if patched_metadata.get('status') != 'active':
                self.test_results.append((test_name, False, 'Patch-add did not preserve existing status field'))
                return False

            if patched_metadata.get('priority') != 5:
                self.test_results.append((test_name, False, 'Patch-add did not preserve existing priority field'))
                return False

            if patched_metadata.get('new_field') != 'added_value':
                self.test_results.append((test_name, False, 'Patch-add did not add new field'))
                return False

            # Test 7: Metadata patch - update existing field
            patch_update_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'priority': 10},
                },
            )

            patch_update_data = self._extract_content(patch_update_result)

            if not patch_update_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch-update existing field: {patch_update_data}'))
                return False

            # Verify patch updated field while preserving others
            verify_patch_update = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_patch_update_data = self._extract_content(verify_patch_update)

            if not verify_patch_update_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch-update'))
                return False

            updated_metadata = verify_patch_update_data['results'][0].get('metadata', {})

            if updated_metadata.get('priority') != 10:
                self.test_results.append((test_name, False, 'Patch-update did not change priority'))
                return False

            if updated_metadata.get('status') != 'active':
                self.test_results.append((test_name, False, 'Patch-update did not preserve status field'))
                return False

            if updated_metadata.get('new_field') != 'added_value':
                self.test_results.append((test_name, False, 'Patch-update did not preserve new_field'))
                return False

            # Test 8: Metadata patch - delete field with null value (RFC 7396 semantics)
            patch_delete_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'new_field': None},
                },
            )

            patch_delete_data = self._extract_content(patch_delete_result)

            if not patch_delete_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch-delete field: {patch_delete_data}'))
                return False

            # Verify patch deleted the field
            verify_patch_delete = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_patch_delete_data = self._extract_content(verify_patch_delete)

            if not verify_patch_delete_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch-delete'))
                return False

            deleted_metadata = verify_patch_delete_data['results'][0].get('metadata', {})

            if 'new_field' in deleted_metadata:
                self.test_results.append((test_name, False, 'Patch-delete did not remove field (RFC 7396 null semantics)'))
                return False

            if deleted_metadata.get('status') != 'active' or deleted_metadata.get('priority') != 10:
                self.test_results.append((test_name, False, 'Patch-delete modified other fields'))
                return False

            # Test 9: Mutual exclusivity - providing both metadata and metadata_patch should fail
            # The server raises ToolError which may propagate as an exception to the client
            mutual_exclusivity_validated = False
            try:
                mutual_exclusion_result = await self.client.call_tool(
                    'update_context',
                    {
                        'context_id': context_id,
                        'metadata': {'full': 'replacement'},
                        'metadata_patch': {'partial': 'update'},
                    },
                )

                mutual_exclusion_data = self._extract_content(mutual_exclusion_result)

                # If we get here without exception, check the response
                if mutual_exclusion_data.get('success'):
                    self.test_results.append(
                        (test_name, False, 'Mutual exclusivity check failed - both metadata and metadata_patch accepted'),
                    )
                    return False

                # Check error message in response
                error_msg = mutual_exclusion_data.get('error', '')
                error_mentions_mutual_exclusivity = 'mutual' in error_msg.lower() or 'exclusive' in error_msg.lower()
                error_mentions_metadata_params = 'metadata' in error_msg.lower() and 'patch' in error_msg.lower()
                if error_mentions_mutual_exclusivity or error_mentions_metadata_params:
                    mutual_exclusivity_validated = True
                else:
                    self.test_results.append(
                        (test_name, False, f'Mutual exclusivity error message unclear: {error_msg}'),
                    )
                    return False

            except Exception as mutual_exc:
                # ToolError is expected - verify the error message mentions mutual exclusivity
                error_msg = str(mutual_exc)
                error_mentions_mutual_exclusivity = 'mutual' in error_msg.lower() or 'exclusive' in error_msg.lower()
                error_mentions_metadata_params = 'metadata' in error_msg.lower() and 'patch' in error_msg.lower()
                if error_mentions_mutual_exclusivity or error_mentions_metadata_params:
                    mutual_exclusivity_validated = True
                else:
                    self.test_results.append(
                        (test_name, False, f'Unexpected exception during mutual exclusivity test: {mutual_exc}'),
                    )
                    return False

            if not mutual_exclusivity_validated:
                self.test_results.append((test_name, False, 'Mutual exclusivity validation did not complete'))
                return False

            # Test 10: Metadata patch on context with no existing metadata
            # Create new context without metadata
            no_metadata_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': update_thread,
                    'source': 'agent',
                    'text': 'Context without initial metadata',
                },
            )

            no_metadata_data = self._extract_content(no_metadata_result)

            if not no_metadata_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to create context without metadata: {no_metadata_data}'))
                return False

            no_metadata_context_id = no_metadata_data.get('context_id')

            # Apply patch to context with no metadata
            patch_empty_result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': no_metadata_context_id,
                    'metadata_patch': {'created_via': 'patch', 'version': 1},
                },
            )

            patch_empty_data = self._extract_content(patch_empty_result)

            if not patch_empty_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to patch empty metadata: {patch_empty_data}'))
                return False

            # Verify metadata was created from scratch
            verify_patch_empty = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [no_metadata_context_id]},
            )

            verify_patch_empty_data = self._extract_content(verify_patch_empty)

            if not verify_patch_empty_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify patch on empty metadata'))
                return False

            created_metadata = verify_patch_empty_data['results'][0].get('metadata', {})

            if created_metadata.get('created_via') != 'patch' or created_metadata.get('version') != 1:
                self.test_results.append((test_name, False, 'Patch on empty metadata did not create expected fields'))
                return False

            # Verify immutable fields remain unchanged
            if text_only_entry.get('thread_id') != update_thread or text_only_entry.get('source') != 'agent':
                self.test_results.append((test_name, False, 'Immutable fields were modified'))
                return False

            self.test_results.append((test_name, True, 'All update operations passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_update_context_nonexistent_id(self) -> bool:
        """Test updating non-existent context returns error.

        Returns:
            bool: True if test passed (error is returned).
        """
        test_name = 'Update Context Nonexistent ID'
        assert self.client is not None
        try:
            # Try to update a non-existent context ID
            result = await self.client.call_tool(
                'update_context',
                {
                    'context_id': '0' * 32,  # 32-char hex, never generated
                    'text': 'Updated text',
                },
            )

            data = self._extract_content(result)

            # Should fail with error about not found
            if data.get('success') is False or 'error' in data or 'not found' in str(data).lower():
                self.test_results.append((test_name, True, 'Update non-existent correctly rejected'))
                return True

            self.test_results.append((test_name, False, f'Expected error for non-existent ID: {data}'))
            return False

        except Exception as e:
            # Exception is expected for non-existent ID
            if 'not found' in str(e).lower():
                self.test_results.append((test_name, True, f'Update non-existent correctly rejected: {e}'))
                return True
            self.test_results.append((test_name, False, f'Unexpected exception: {e}'))
            return False

    async def test_update_context_triggers_embedding_regeneration(self) -> bool:
        """Verify that updating text content triggers embedding regeneration.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context_triggers_embedding_regeneration'
        assert self.client is not None
        try:
            stats = await self.client.call_tool('get_statistics', {})
            stats_data = self._extract_content(stats)
            has_semantic = stats_data.get('semantic_search', {}).get('available', False)

            if not has_semantic:
                self.test_results.append((test_name, True, 'Skipped (semantic search unavailable)'))
                return True

            update_thread = f'{self.test_thread_id}_embed_regen'

            store_result = await self.client.call_tool('store_context', {
                'thread_id': update_thread, 'source': 'agent',
                'text': 'Python machine learning frameworks TensorFlow and PyTorch',
            })
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {store_data}'))
                return False

            context_id = store_data.get('context_id')
            await asyncio.sleep(0.5)

            search1 = await self.client.call_tool('semantic_search_context', {
                'query': 'Python machine learning', 'thread_id': update_thread, 'limit': 5,
            })
            search1_data = self._extract_content(search1)
            if len(search1_data.get('results', [])) < 1:
                self.test_results.append((test_name, False, 'Original entry not found via semantic search'))
                return False

            update_result = await self.client.call_tool('update_context', {
                'context_id': context_id,
                'text': 'JavaScript React frontend web development with TypeScript',
            })
            update_data = self._extract_content(update_result)
            if not update_data.get('success'):
                self.test_results.append((test_name, False, f'Update failed: {update_data}'))
                return False

            await asyncio.sleep(0.5)

            search2 = await self.client.call_tool('semantic_search_context', {
                'query': 'JavaScript React frontend', 'thread_id': update_thread, 'limit': 5,
            })
            search2_data = self._extract_content(search2)
            if len(search2_data.get('results', [])) < 1:
                self.test_results.append((test_name, False, 'Updated entry not found via semantic search'))
                return False

            found_text = search2_data['results'][0].get('text_content', '')
            if 'JavaScript' not in found_text and 'React' not in found_text:
                self.test_results.append((test_name, False,
                    f'Search found wrong content: {found_text[:100]}'))
                return False

            self.test_results.append((test_name, True, 'Embedding regeneration verified via semantic search'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_update_context_advances_updated_at(self) -> bool:
        """Every update variant advances the entry's public ``updated_at`` timestamp.

        ``updated_at`` is the only mutation timestamp the API exposes, so clients key
        incremental sync and cache invalidation on it, and it is stamped only by a write
        to ``context_entries`` itself. A tags-only update writes just the child ``tags``
        table, so before the central stamp it reported success while leaving the timestamp
        at its previous value and a syncing client never observed the change. All four
        variants run in turn. SQLite's CURRENT_TIMESTAMP has SECOND granularity, so the
        updates are spaced past a whole-second boundary and the canonical
        ``YYYY-MM-DDTHH:MM:SSZ`` strings (identical wire format on both backends) compare
        directly.

        Returns:
            bool: True if test passed.
        """
        test_name = 'update_context_advances_updated_at'
        assert self.client is not None

        async def _updated_at(entry_id: str) -> str:
            """Read one entry's canonical updated_at string."""
            assert self.client is not None
            got = self._extract_content(
                await self.client.call_tool('get_context_by_ids', {'context_ids': [entry_id]}),
            )
            rows = got.get('results', [])
            return str(rows[0].get('updated_at', '')) if rows else ''

        try:
            thread = f'{self.test_thread_id}_updated_at'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent', 'text': 'Entry for the updated_at contract',
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False
            entry_id = str(stored['context_id'])

            previous = await _updated_at(entry_id)
            if not previous:
                self.test_results.append((test_name, False, 'Stored entry reported no updated_at'))
                return False

            variants: list[tuple[str, dict[str, Any]]] = [
                ('text', {'text': 'Updated text for the updated_at contract'}),
                ('metadata_patch', {'metadata_patch': {'stage': 'patched'}}),
                ('tags', {'tags': ['updated-at-tag']}),
                ('images', {'images': [{'data': self._create_test_image(), 'mime_type': 'image/png'}]}),
            ]
            for label, payload in variants:
                # Cross a whole-second boundary so an advance is observable rather than
                # rounded away by SQLite's second-granularity CURRENT_TIMESTAMP.
                await asyncio.sleep(1.1)
                result = self._extract_content(await self.client.call_tool('update_context', {
                    'context_id': entry_id, **payload,
                }))
                if not result.get('success'):
                    self.test_results.append((test_name, False, f'{label}-only update failed: {result}'))
                    return False
                current = await _updated_at(entry_id)
                if current <= previous:
                    self.test_results.append((
                        test_name, False,
                        f'{label}-only update left updated_at at {previous} (read back {current})',
                    ))
                    return False
                previous = current

            self.test_results.append((
                test_name, True, 'text, metadata_patch, tags and images updates each advanced updated_at',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
