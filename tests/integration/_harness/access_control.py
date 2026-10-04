"""Real-server checks for entry visibility.

The ``visibility`` parameter on store and update, rejection of an invalid
value, and the internal access-control columns staying out of read
responses.
"""

from tests.integration._harness.core import HarnessCore


class AccessControlMixin(HarnessCore):
    """Checks for entry visibility and ownership."""

    async def test_visibility_lifecycle(self) -> bool:
        """Test the visibility parameter on store and update, and owner privacy.

        Stores an entry with an explicit visibility, changes it via
        update_context (the connected client IS the owner: without a verified
        token every request maps to the configured default principal), rejects
        an invalid visibility value, and confirms the internal access-control
        columns never appear in read responses.

        Returns:
            bool: True if test passed.
        """
        test_name = 'visibility_lifecycle'
        assert self.client is not None  # Type guard for Pyright
        try:
            visibility_thread = f'{self.test_thread_id}_visibility'

            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': visibility_thread,
                    'source': 'agent',
                    'text': 'Visibility lifecycle entry: stored private, published public.',
                    'visibility': 'private',
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, f'Store with visibility failed: {store_data}'))
                return False
            context_id = store_data.get('context_id')

            update_result = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'visibility': 'public'},
            )
            update_data = self._extract_content(update_result)
            if not update_data.get('success'):
                self.test_results.append((test_name, False, f'Visibility update failed: {update_data}'))
                return False
            if 'visibility' not in (update_data.get('updated_fields') or []):
                self.test_results.append((
                    test_name, False,
                    f'updated_fields missing visibility: {update_data.get("updated_fields")}',
                ))
                return False

            # An invalid visibility value is rejected at the tool boundary.
            try:
                await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': visibility_thread,
                        'source': 'agent',
                        'text': 'This entry must be rejected.',
                        'visibility': 'everyone',
                    },
                )
                self.test_results.append((test_name, False, 'Invalid visibility value was accepted'))
                return False
            except Exception:
                pass  # Expected rejection

            # The internal access-control columns never leak into read responses.
            fetch_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            fetch_data = self._extract_content(fetch_result)
            entries = fetch_data.get('results') or []
            if not fetch_data.get('success') or not entries:
                self.test_results.append((test_name, False, f'Stored entry not retrievable: {fetch_data}'))
                return False
            leaked = {'owner_id', 'visibility'} & set(entries[0].keys())
            if leaked:
                self.test_results.append((test_name, False, f'Internal columns leaked: {leaked}'))
                return False

            self.test_results.append((test_name, True, 'Visibility store/update/reject/privacy passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
