"""Real-server checks for ``update_context`` with ``metadata_patch``.

RFC 7396 JSON Merge Patch semantics: deep merge, the Appendix A cases,
successive patches to one entry, and value type changes.
"""

from tests.integration._harness.core import HarnessCore


class MetadataPatchMixin(HarnessCore):
    """Checks for metadata_patch merge semantics."""

    async def test_metadata_patch_deep_merge(self) -> bool:
        """Test RFC 7396 deep merge semantics for metadata_patch.

        This test verifies that nested objects are correctly merged according to
        RFC 7396 JSON Merge Patch specification, including deep merge and nested
        null deletion.

        RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396

        Returns:
            bool: True if all deep merge tests passed.
        """
        test_name = 'metadata_patch_deep_merge'
        assert self.client is not None  # Type guard for Pyright
        try:
            # Create a separate thread for deep merge tests
            deep_merge_thread = f'{self.test_thread_id}_deep_merge'

            # Test 1: Setup - Create context with nested metadata
            setup_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': deep_merge_thread,
                    'source': 'agent',
                    'text': 'Context for RFC 7396 deep merge testing',
                    'metadata': {
                        'a': {
                            'b': 'original_b',
                            'd': 'original_d',
                        },
                        'top_level': 'preserved',
                    },
                },
            )

            setup_data = self._extract_content(setup_result)
            if not setup_data.get('success'):
                self.test_results.append((test_name, False, f'Failed to create test context: {setup_data}'))
                return False

            context_id = setup_data.get('context_id')

            # Test 2: RFC 7396 Case #7 - Deep merge with nested update (preserves siblings)
            # Patch: {"a": {"b": "updated"}} should preserve "d" in nested object
            patch_deep_merge = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'a': {'b': 'updated_b'}},
                },
            )

            patch_deep_data = self._extract_content(patch_deep_merge)
            if not patch_deep_data.get('success'):
                self.test_results.append((test_name, False, f'Failed deep merge patch: {patch_deep_data}'))
                return False

            # Verify deep merge preserved sibling key
            verify_deep = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_deep_data = self._extract_content(verify_deep)
            if not verify_deep_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify deep merge'))
                return False

            deep_metadata = verify_deep_data['results'][0].get('metadata', {})

            # RFC 7396: Nested sibling key "d" MUST be preserved
            if deep_metadata.get('a', {}).get('b') != 'updated_b':
                self.test_results.append((test_name, False, 'Deep merge did not update nested key "b"'))
                return False

            if deep_metadata.get('a', {}).get('d') != 'original_d':
                error_msg = f'RFC 7396 VIOLATION: Deep merge did not preserve sibling key "d". Got: {deep_metadata}'
                self.test_results.append((test_name, False, error_msg))
                return False

            if deep_metadata.get('top_level') != 'preserved':
                self.test_results.append((test_name, False, 'Deep merge did not preserve top-level key'))
                return False

            # Test 3: Nested null deletion (RFC 7396 Case #7 variant)
            # Patch: {"a": {"b": null}} should delete "b" but preserve "d"
            patch_nested_delete = await self.client.call_tool(
                'update_context',
                {
                    'context_id': context_id,
                    'metadata_patch': {'a': {'b': None}},  # RFC 7396: null means delete
                },
            )

            patch_delete_data = self._extract_content(patch_nested_delete)
            if not patch_delete_data.get('success'):
                self.test_results.append((test_name, False, f'Failed nested deletion patch: {patch_delete_data}'))
                return False

            # Verify nested deletion preserved sibling
            verify_delete = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )

            verify_delete_data = self._extract_content(verify_delete)
            if not verify_delete_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify nested deletion'))
                return False

            delete_metadata = verify_delete_data['results'][0].get('metadata', {})

            # RFC 7396: Key "b" should be deleted
            if 'b' in delete_metadata.get('a', {}):
                self.test_results.append(
                    (test_name, False, f'RFC 7396 VIOLATION: Nested null did not delete key "b". Got: {delete_metadata}'),
                )
                return False

            # RFC 7396: Key "d" MUST be preserved
            if delete_metadata.get('a', {}).get('d') != 'original_d':
                error_msg = f'RFC 7396 VIOLATION: Nested deletion did not preserve "d". Got: {delete_metadata}'
                self.test_results.append((test_name, False, error_msg))
                return False

            # Test 4: Deeply nested null deletion (RFC 7396 Case #15)
            # Create new context for this test
            deep_nested_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': deep_merge_thread,
                    'source': 'agent',
                    'text': 'Context for deeply nested null test',
                    'metadata': {},  # Start empty
                },
            )

            deep_nested_data = self._extract_content(deep_nested_result)
            if not deep_nested_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to create deeply nested test context'))
                return False

            deep_context_id = deep_nested_data.get('context_id')

            # RFC 7396 Case #15: {"a": {"bb": {"ccc": null}}} should result in {"a": {"bb": {}}}
            patch_deep_null = await self.client.call_tool(
                'update_context',
                {
                    'context_id': deep_context_id,
                    'metadata_patch': {'a': {'bb': {'ccc': None}}},
                },
            )

            patch_deep_null_data = self._extract_content(patch_deep_null)
            if not patch_deep_null_data.get('success'):
                self.test_results.append((test_name, False, f'Failed deeply nested null patch: {patch_deep_null_data}'))
                return False

            # Verify deeply nested structure
            verify_deep_null = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [deep_context_id]},
            )

            verify_deep_null_data = self._extract_content(verify_deep_null)
            if not verify_deep_null_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to verify deeply nested null'))
                return False

            deep_null_metadata = verify_deep_null_data['results'][0].get('metadata', {})

            # RFC 7396 Case #15: Expected {"a": {"bb": {}}}
            # The deeply nested null should create empty nested objects, not include null
            if 'a' not in deep_null_metadata:
                self.test_results.append(
                    (test_name, False, f'RFC 7396 Case #15 VIOLATION: Missing top-level "a". Got: {deep_null_metadata}'),
                )
                return False

            if 'bb' not in deep_null_metadata.get('a', {}):
                self.test_results.append(
                    (test_name, False, f'RFC 7396 Case #15 VIOLATION: Missing nested "bb". Got: {deep_null_metadata}'),
                )
                return False

            # The key "ccc" should NOT exist (deleted by null)
            if 'ccc' in deep_null_metadata.get('a', {}).get('bb', {}):
                self.test_results.append(
                    (test_name, False, f'RFC 7396 Case #15 VIOLATION: Key "ccc" should be deleted. Got: {deep_null_metadata}'),
                )
                return False

            self.test_results.append((test_name, True, 'All RFC 7396 deep merge tests passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_patch_rfc7396_full_compliance(self) -> bool:
        """Comprehensive RFC 7396 JSON Merge Patch compliance tests.

        This test validates the FULL RFC 7396 Appendix A test cases against a
        real server on both backends: SQLite applies the patch with json_patch()
        and PostgreSQL with the jsonb_merge_patch() function.

        RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396

        Returns:
            bool: True if all RFC 7396 tests passed.
        """
        test_name = 'metadata_patch_rfc7396_full_compliance'
        assert self.client is not None  # Type guard for Pyright
        try:
            rfc_thread = f'{self.test_thread_id}_rfc7396'

            # RFC 7396 Test Cases from Appendix A
            # (name, initial_metadata, patch, expected_result)
            test_cases: list[tuple[str, dict[str, object], dict[str, object], dict[str, object]]] = [
                ('Case1_simple_replace', {'a': 'b'}, {'a': 'c'}, {'a': 'c'}),
                ('Case2_add_new_key', {'a': 'b'}, {'b': 'c'}, {'a': 'b', 'b': 'c'}),
                ('Case3_delete_key', {'a': 'b'}, {'a': None}, {}),
                ('Case4_delete_preserve', {'a': 'b', 'b': 'c'}, {'a': None}, {'b': 'c'}),
                ('Case5_array_replace', {'a': ['b']}, {'a': 'c'}, {'a': 'c'}),
                ('Case6_value_to_array', {'a': 'c'}, {'a': ['b']}, {'a': ['b']}),
                ('Case7_nested_merge', {'a': {'b': 'c'}}, {'a': {'b': 'd', 'c': None}}, {'a': {'b': 'd'}}),
                ('Case8_array_objects', {'a': [{'b': 'c'}]}, {'a': [1]}, {'a': [1]}),
                ('Case13_preserve_null', {'e': None}, {'a': 1}, {'a': 1, 'e': None}),
                ('Case15_deep_nested', {}, {'a': {'bb': {'ccc': None}}}, {'a': {'bb': {}}}),
            ]

            for case_name, initial, patch, expected in test_cases:
                # Create context with initial metadata
                store_result = await self.client.call_tool(
                    'store_context',
                    {
                        'thread_id': rfc_thread,
                        'source': 'agent',
                        'text': f'RFC 7396 test: {case_name}',
                        'metadata': initial,
                    },
                )
                store_data = self._extract_content(store_result)
                if not store_data.get('success'):
                    self.test_results.append((test_name, False, f'{case_name}: Store failed'))
                    return False

                context_id = store_data.get('context_id')

                # Apply patch
                patch_result = await self.client.call_tool(
                    'update_context',
                    {
                        'context_id': context_id,
                        'metadata_patch': patch,
                    },
                )
                patch_data = self._extract_content(patch_result)
                if not patch_data.get('success'):
                    self.test_results.append((test_name, False, f'{case_name}: Patch failed'))
                    return False

                # Verify result
                verify_result = await self.client.call_tool(
                    'get_context_by_ids',
                    {'context_ids': [context_id]},
                )
                verify_data = self._extract_content(verify_result)
                result_metadata = verify_data['results'][0].get('metadata', {})

                if result_metadata != expected:
                    error_msg = f'{case_name}: Expected {expected}, got {result_metadata}'
                    self.test_results.append((test_name, False, error_msg))
                    return False

            self.test_results.append((test_name, True, 'All RFC 7396 test cases passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_patch_successive_patches(self) -> bool:
        """Test applying multiple successive patches to the same entry.

        Verifies that patches accumulate correctly and don't interfere with
        each other when applied in sequence.

        Returns:
            bool: True if all successive patch tests passed.
        """
        test_name = 'metadata_patch_successive_patches'
        assert self.client is not None
        try:
            successive_thread = f'{self.test_thread_id}_successive'

            # Create initial entry with some metadata
            store_result = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': successive_thread,
                    'source': 'agent',
                    'text': 'Entry for successive patch testing',
                    'metadata': {'version': 1, 'status': 'created'},
                },
            )
            store_data = self._extract_content(store_result)
            if not store_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to create test entry'))
                return False

            context_id = store_data.get('context_id')

            # Patch 1: Add new field
            patch1_result = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'author': 'agent-1'}},
            )
            if not self._extract_content(patch1_result).get('success'):
                self.test_results.append((test_name, False, 'Patch 1 failed'))
                return False

            # Patch 2: Update existing field
            patch2_result = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'status': 'updated'}},
            )
            if not self._extract_content(patch2_result).get('success'):
                self.test_results.append((test_name, False, 'Patch 2 failed'))
                return False

            # Patch 3: Increment version
            patch3_result = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'version': 2}},
            )
            if not self._extract_content(patch3_result).get('success'):
                self.test_results.append((test_name, False, 'Patch 3 failed'))
                return False

            # Patch 4: Delete a field
            patch4_result = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'status': None}},
            )
            if not self._extract_content(patch4_result).get('success'):
                self.test_results.append((test_name, False, 'Patch 4 failed'))
                return False

            # Verify final state
            verify_result = await self.client.call_tool(
                'get_context_by_ids',
                {'context_ids': [context_id]},
            )
            verify_data = self._extract_content(verify_result)
            final_metadata = verify_data['results'][0].get('metadata', {})

            expected = {'version': 2, 'author': 'agent-1'}  # status was deleted
            if final_metadata != expected:
                self.test_results.append(
                    (test_name, False, f'Final state mismatch. Expected {expected}, got {final_metadata}'),
                )
                return False

            self.test_results.append((test_name, True, 'All successive patches accumulated correctly'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False

    async def test_metadata_patch_type_conversions(self) -> bool:
        """Test type conversion scenarios in metadata_patch.

        RFC 7396 allows values to change types - objects can become arrays,
        scalars can become objects, etc.

        Returns:
            bool: True if all type conversion tests passed.
        """
        test_name = 'metadata_patch_type_conversions'
        assert self.client is not None
        try:
            type_thread = f'{self.test_thread_id}_types'

            # Test: Object to scalar
            store1 = await self.client.call_tool(
                'store_context',
                {
                    'thread_id': type_thread,
                    'source': 'agent',
                    'text': 'Object to scalar test',
                    'metadata': {'config': {'nested': 'value', 'deep': {'key': 1}}},
                },
            )
            store1_data = self._extract_content(store1)
            if not store1_data.get('success'):
                self.test_results.append((test_name, False, 'Failed to store object'))
                return False

            context_id = store1_data.get('context_id')

            # Replace object with scalar
            patch1 = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'config': 'simple_string'}},
            )
            if not self._extract_content(patch1).get('success'):
                self.test_results.append((test_name, False, 'Object to scalar patch failed'))
                return False

            verify1 = await self.client.call_tool('get_context_by_ids', {'context_ids': [context_id]})
            verify1_data = self._extract_content(verify1)
            if verify1_data['results'][0].get('metadata', {}).get('config') != 'simple_string':
                self.test_results.append((test_name, False, 'Object to scalar conversion failed'))
                return False

            # Test: Scalar to object
            patch2 = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'config': {'new_nested': 'value'}}},
            )
            if not self._extract_content(patch2).get('success'):
                self.test_results.append((test_name, False, 'Scalar to object patch failed'))
                return False

            verify2 = await self.client.call_tool('get_context_by_ids', {'context_ids': [context_id]})
            verify2_data = self._extract_content(verify2)
            if verify2_data['results'][0].get('metadata', {}).get('config') != {'new_nested': 'value'}:
                self.test_results.append((test_name, False, 'Scalar to object conversion failed'))
                return False

            # Test: Object to array
            patch3 = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'config': ['item1', 'item2']}},
            )
            if not self._extract_content(patch3).get('success'):
                self.test_results.append((test_name, False, 'Object to array patch failed'))
                return False

            verify3 = await self.client.call_tool('get_context_by_ids', {'context_ids': [context_id]})
            verify3_data = self._extract_content(verify3)
            if verify3_data['results'][0].get('metadata', {}).get('config') != ['item1', 'item2']:
                self.test_results.append((test_name, False, 'Object to array conversion failed'))
                return False

            # Test: Array to object
            patch4 = await self.client.call_tool(
                'update_context',
                {'context_id': context_id, 'metadata_patch': {'config': {'back_to': 'object'}}},
            )
            if not self._extract_content(patch4).get('success'):
                self.test_results.append((test_name, False, 'Array to object patch failed'))
                return False

            verify4 = await self.client.call_tool('get_context_by_ids', {'context_ids': [context_id]})
            verify4_data = self._extract_content(verify4)
            if verify4_data['results'][0].get('metadata', {}).get('config') != {'back_to': 'object'}:
                self.test_results.append((test_name, False, 'Array to object conversion failed'))
                return False

            self.test_results.append((test_name, True, 'All type conversion tests passed'))
            return True

        except Exception as e:
            self.test_results.append((test_name, False, f'Exception: {e}'))
            return False
