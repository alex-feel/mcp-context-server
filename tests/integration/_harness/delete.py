"""Real-server checks for ``delete_context``.

Deletion by id and by thread, an unknown id, rejection of a request naming
both selectors, and removal of the deleted entries' embedding rows and
image attachments.
"""

from tests.integration._harness.core import HarnessCore


class DeleteMixin(HarnessCore):
    """Checks for deleting context entries."""

    async def test_delete_context(self) -> bool:
        """Test deletion operations.

        Returns:
            bool: True if test passed.
        """
        test_name = 'delete_context'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for deletion tests
            delete_thread = f'{self.test_thread_id}_delete'

            # Store multiple contexts
            result1 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': delete_thread,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Context to delete by ID',
                },
            )

            result2 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': delete_thread,
                    'source': 'agent',  # Must be 'user' or 'agent'
                    'text': 'Context to delete with thread',
                },
            )

            result3 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': delete_thread,
                    'source': 'user',  # Must be 'user' or 'agent'
                    'text': 'Another context in thread',
                },
            )

            data1 = self._extract_content(result1)
            data2 = self._extract_content(result2)
            data3 = self._extract_content(result3)

            if not all([
                data1.get('success'),
                data2.get('success'),
                data3.get('success'),
            ]):
                self.test_results.append((test_name, False, f'Failed to store test contexts: {data1}, {data2}, {data3}'))
                return False

            # Test delete by ID
            delete_by_id = await self.client.call_tool(
                'delete_context',
                {'context_ids': [data1['context_id']]},
            )

            delete_data = self._extract_content(delete_by_id)

            if not delete_data.get('success') or delete_data.get('deleted_count') != 1:
                self.test_results.append((test_name, False, f'Failed to delete by ID: {delete_data}'))
                return False

            # Verify deletion by trying to retrieve
            check_deleted = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [data1['context_id']]},
            )

            check_data = self._extract_content(check_deleted)

            # get_context_by_ids returns success with results
            if len(check_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, f'Context not deleted by ID: {check_data}'))
                return False

            # Test delete by thread
            delete_by_thread = await self.client.call_tool(
                'delete_context',
                {'thread_id': delete_thread},
            )

            thread_delete_data = self._extract_content(delete_by_thread)

            # The thread still holds the two entries the delete by id left in place.
            if not thread_delete_data.get('success') or thread_delete_data.get('deleted_count') != 2:
                self.test_results.append((test_name, False, f'Failed to delete by thread: {thread_delete_data}'))
                return False

            # Verify thread deletion
            check_thread = await self.client.call_tool(
                'search_context',
                {'limit': 50, 'thread_id': delete_thread},
            )

            check_thread_data = self._extract_content(check_thread)

            # search_context returns success with results
            if len(check_thread_data.get('results', [])) > 0:
                self.test_results.append((test_name, False, f'Thread contexts not deleted: {check_thread_data}'))
                return False

            deleted_count = delete_data.get('deleted_count', 0) + thread_delete_data.get('deleted_count', 0)
            self.test_results.append((test_name, True, f'Deleted {deleted_count} contexts'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_context_nonexistent_id(self) -> bool:
        """Deleting an unknown id or an unknown thread succeeds with nothing deleted.

        Returns:
            bool: True if test passed.
        """
        test_name = 'Delete Context Nonexistent ID'
        assert self.client is not None
        try:
            for arguments in (
                {'context_ids': ['0190abcdef1234567890abcdef0fffff']},
                {'thread_id': 'nonexistent_thread_for_delete_xyz'},
            ):
                data = self._extract_content(await self.client.call_tool('delete_context', arguments))
                if not data.get('success') or data.get('deleted_count', -1) != 0:
                    self.test_results.append((test_name, False, f'Unexpected result for {arguments}: {data}'))
                    return False

            self.test_results.append((test_name, True, 'Unknown id and unknown thread each deleted nothing'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_context_rejects_both_selectors(self) -> bool:
        """delete_context refuses a request naming BOTH context_ids and thread_id.

        The two parameters are documented as mutually exclusive and the dispatch is
        if/elif, so accepting both would delete only the listed ids while the response
        read as full success -- a partially executed irreversible delete the caller has
        no way to detect, with the rest of the named thread quietly surviving. The
        combination is refused outright, and nothing is deleted.

        Returns:
            bool: True if test passed.
        """
        test_name = 'delete_context_rejects_both_selectors'
        assert self.client is not None
        try:
            thread = f'{self.test_thread_id}_delete_exclusive'
            stored_ids: list[str] = []
            for index in range(2):
                stored = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'Entry {index} that must survive a refused delete',
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Store {index} failed: {stored}'))
                    return False
                stored_ids.append(str(stored['context_id']))

            refused = False
            try:
                response = self._extract_content(await self.client.call_tool('delete_context', {
                    'context_ids': [stored_ids[0]], 'thread_id': thread,
                }))
            except Exception:
                refused = True
            else:
                refused = response.get('success') is not True
            if not refused:
                self.test_results.append((
                    test_name, False, 'delete_context accepted both context_ids and thread_id',
                ))
                return False

            got = self._extract_content(await self.client.call_tool('get_context_by_ids', {
                'context_ids': stored_ids,
            }))
            survivors = {str(row.get('id')) for row in got.get('results', [])}
            if survivors != set(stored_ids):
                self.test_results.append((
                    test_name, False,
                    f'A refused delete still removed entries: {sorted(set(stored_ids) - survivors)}',
                ))
                return False

            self.test_results.append((
                test_name, True, 'Both selectors together are refused and every entry survives',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_delete_removes_embedding_rows(self) -> bool:
        """Deleting entries takes their embedding rows with them, by id and by thread.

        Both deletes run the embedding cleanup and the row delete inside ONE transaction,
        and the thread-wide delete constrains itself to exactly the id snapshot it took --
        while PostgreSQL carries no explicit per-entry cleanup at all and relies entirely
        on the ON DELETE CASCADE from the embedding tables. The global
        embedding count in ``get_statistics`` is the observable that pins that reliance:
        it counts ``embedding_metadata`` rows WITHOUT joining ``context_entries``, so a
        cascade that stopped firing (or a delete that committed the row removal while its
        cleanup rolled back) leaves the count above its pre-store value on either backend.

        Returns:
            bool: True if test passed or skipped gracefully.
        """
        test_name = 'delete_removes_embedding_rows'
        assert self.client is not None

        async def _embedding_rows() -> int:
            """Count the embedding rows get_statistics reports across the whole database."""
            assert self.client is not None
            stats = self._extract_content(await self.client.call_tool('get_statistics', {}))
            return int(stats.get('semantic_search', {}).get('context_count', 0))

        try:
            stats_data = self._extract_content(await self.client.call_tool('get_statistics', {}))
            semantic_info = stats_data.get('semantic_search', {})
            if not (semantic_info.get('enabled') and semantic_info.get('available')):
                self.test_results.append((test_name, True, 'Skipped (embeddings not available)'))
                return True
            baseline = int(semantic_info.get('context_count', 0))

            single_thread = f'{self.test_thread_id}_del_emb_single'
            stored = self._extract_content(await self.client.call_tool('store_context', {
                'thread_id': single_thread, 'source': 'agent',
                'text': 'Entry whose embedding rows must not outlive an id-scoped delete',
            }))
            if not stored.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {stored}'))
                return False
            entry_id = str(stored['context_id'])

            after_store = await _embedding_rows()
            if after_store != baseline + 1:
                self.test_results.append((
                    test_name, False,
                    (
                        f'Expected {baseline + 1} embedding rows after the store, found {after_store} '
                        '(the orphan check would be vacuous)'
                    ),
                ))
                return False

            deleted = self._extract_content(await self.client.call_tool('delete_context', {
                'context_ids': [entry_id],
            }))
            if not deleted.get('success'):
                self.test_results.append((test_name, False, f'delete_context by id failed: {deleted}'))
                return False
            after_delete = await _embedding_rows()
            if after_delete != baseline:
                self.test_results.append((
                    test_name, False,
                    f'{after_delete - baseline} embedding row(s) outlived the id-deleted entry on {self.backend}',
                ))
                return False

            thread = f'{self.test_thread_id}_del_emb_thread'
            for index in range(2):
                stored = self._extract_content(await self.client.call_tool('store_context', {
                    'thread_id': thread, 'source': 'agent',
                    'text': f'Thread-wide delete entry {index} whose embedding rows must vanish with it',
                }))
                if not stored.get('success'):
                    self.test_results.append((test_name, False, f'Store {index} failed: {stored}'))
                    return False
            if await _embedding_rows() != baseline + 2:
                self.test_results.append((test_name, False, 'The thread-wide fixture did not produce two embedding rows'))
                return False

            deleted = self._extract_content(await self.client.call_tool('delete_context', {'thread_id': thread}))
            if not deleted.get('success'):
                self.test_results.append((test_name, False, f'delete_context by thread failed: {deleted}'))
                return False
            after_thread_delete = await _embedding_rows()
            if after_thread_delete != baseline:
                self.test_results.append((
                    test_name, False,
                    f'{after_thread_delete - baseline} embedding row(s) outlived the thread delete on {self.backend}',
                ))
                return False

            remaining = self._extract_content(await self.client.call_tool('search_context', {
                'thread_id': thread, 'limit': 10,
            }))
            if remaining.get('results'):
                self.test_results.append((test_name, False, 'Entries survived the thread-wide delete'))
                return False

            self.test_results.append((
                test_name, True, f'Id-scoped and thread-wide deletes left no embedding rows behind on {self.backend}',
            ))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_image_attachment_cascade_delete(self) -> bool:
        """Verify deleting a multimodal entry removes it and its image attachments.

        Validates the user-facing guarantee on BOTH backends: after
        delete_context the entry is gone from get_context_by_ids and absent from
        a multimodal content_type search. On PostgreSQL it
        ADDITIONALLY proves the actual ON DELETE CASCADE via a direct
        image_attachments row count (orphaned image rows are not observable
        through any tool, so a tool-level check alone cannot distinguish a real
        cascade from leaked orphan rows). The PG count is asserted
        >= 1 before delete (so the probe is non-vacuous) and == 0 after. The
        SQLite server manages its own temp database that the harness does not
        address directly, so the SQLite cascade is asserted at the tool surface.

        Returns:
            bool: True if test passed.
        """
        test_name = 'image_attachment_cascade_delete'
        assert self.client is not None

        async def _pg_image_row_count(entry_id: str) -> int:
            """Count image_attachments rows for entry_id on the live PG database."""
            import asyncpg

            conn = await asyncpg.connect(self.pg_url or '')
            try:
                count = await conn.fetchval(
                    'SELECT COUNT(*) FROM image_attachments WHERE context_entry_id = $1::uuid',
                    entry_id,
                )
                return int(count or 0)
            finally:
                await conn.close()

        try:
            thread = f'{self.test_thread_id}_img_cascade'
            image_b64 = self._create_test_image()
            store = await self.client.call_tool('store_context', {
                'thread_id': thread, 'source': 'agent',
                'text': 'Entry with an image attachment for cascade-delete check',
                'images': [{'data': image_b64, 'mime_type': 'image/png'}],
            })
            store_data = self._extract_content(store)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Store failed: {store_data}'))
                return False
            context_id = store_data['context_id']

            # Confirm stored as multimodal with image data retrievable.
            got = await self.client.call_tool('get_context_by_ids', {
                'context_ids': [context_id], 'include_images': True,
            })
            results = self._extract_content(got).get('results', [])
            if len(results) != 1 or not results[0].get('images'):
                self.test_results.append((test_name, False, 'Stored entry missing image before delete'))
                return False

            # PostgreSQL: confirm the image row exists before delete so the
            # post-delete == 0 assertion cannot pass vacuously.
            if self.backend == 'postgresql':
                before = await _pg_image_row_count(context_id)
                if before < 1:
                    self.test_results.append((test_name, False,
                        f'Expected >= 1 image_attachments row before delete, found {before}'))
                    return False

            await self.client.call_tool('delete_context', {'context_ids': [context_id]})

            after = await self.client.call_tool('get_context_by_ids', {
                'context_ids': [context_id], 'include_images': True,
            })
            if len(self._extract_content(after).get('results', [])) != 0:
                self.test_results.append((test_name, False, 'Entry still present after delete'))
                return False

            # No orphaned multimodal entry resurfaces via search.
            search = await self.client.call_tool('search_context', {
                'thread_id': thread, 'content_type': 'multimodal', 'include_images': True, 'limit': 10,
            })
            if len(self._extract_content(search).get('results', [])) != 0:
                self.test_results.append((test_name, False, 'Orphaned multimodal entry/image remains after delete'))
                return False

            # PostgreSQL: prove the FK ON DELETE CASCADE actually removed the
            # image_attachments rows (not just the parent context_entries row).
            if self.backend == 'postgresql':
                remaining = await _pg_image_row_count(context_id)
                if remaining != 0:
                    self.test_results.append((test_name, False,
                        f'FK cascade left {remaining} orphaned image_attachments row(s) on PostgreSQL'))
                    return False

            self.test_results.append((test_name, True, 'Multimodal entry and its image attachments cascade-deleted'))
            return True
        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
