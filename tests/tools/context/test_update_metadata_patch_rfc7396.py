"""Tests that update_context hands RFC 7396 merge-patch payloads (the Appendix A cases and the nested deep-merge
cases) unchanged to the repository; the merge itself is verified against real databases in
``tests.integration._harness``."""

from unittest.mock import patch

import pytest

import app.tools
from app.types import MetadataDict
from tests.helpers import LOCAL_SCOPE

# Tools are plain async functions registered at server startup, so tests call them directly.
update_context = app.tools.update_context


class TestRFC7396DeepMergeSemantics:
    """Test RFC 7396 deep merge semantics for metadata_patch.

    These tests verify that the correct patch data is passed to the repository
    for RFC 7396 compliant operations. The actual deep merge logic is tested
    at the integration level in ``tests.integration._harness``.

    RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396
    """

    @pytest.mark.asyncio
    async def test_rfc7396_nested_object_merge_case7(self, mock_repositories):
        """RFC 7396 Test Case #7: Nested object merge with deletion.

        Target: {"a": {"b": "c"}}
        Patch:  {"a": {"b": "d", "c": null}}
        Expected: {"a": {"b": "d"}}

        The null value in the nested patch should delete that key from the
        nested object, not from the top-level object.
        """
        nested_patch: MetadataDict = {
            'a': {
                'b': 'd',
                'c': None,  # RFC 7396: null means delete
            },
        }

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001ce4',
                text=None,
                metadata=None,
                metadata_patch=nested_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001ce4',
                patch=nested_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_existing_null_preserved_case13(self, mock_repositories):
        """RFC 7396 Test Case #13: Existing null value preserved.

        Target: {"e": null}
        Patch:  {"a": 1}
        Expected: {"e": null, "a": 1}

        CRITICAL: A null value in the TARGET is preserved (it's actual data).
        Only null values in the PATCH cause deletion.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001cf5',
                text=None,
                metadata=None,
                metadata_patch={'a': 1},  # Does not affect existing null in target
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001cf5',
                patch={'a': 1},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_deeply_nested_null_deletion_case15(self, mock_repositories):
        """RFC 7396 Test Case #15: Deeply nested null deletion.

        Target: {}
        Patch:  {"a": {"bb": {"ccc": null}}}
        Expected: {"a": {"bb": {}}}

        The null value at depth 3 causes deletion at that level,
        but the containing objects are created/preserved.
        """
        deep_patch: MetadataDict = {
            'a': {
                'bb': {
                    'ccc': None,  # RFC 7396: null deletes at depth 3
                },
            },
        }

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001cf7',
                text=None,
                metadata=None,
                metadata_patch=deep_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001cf7',
                patch=deep_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_deep_merge_preserves_sibling_keys(self, mock_repositories):
        """Test that deep merge preserves sibling keys in nested objects.

        Target: {"a": {"b": "c", "d": "e"}}
        Patch:  {"a": {"b": "updated"}}
        Expected: {"a": {"b": "updated", "d": "e"}}

        Key "d" should be preserved because it's not mentioned in the patch.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001ce8',
                text=None,
                metadata=None,
                metadata_patch={'a': {'b': 'updated'}},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001ce8',
                patch={'a': {'b': 'updated'}},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_nested_key_deletion_preserves_siblings(self, mock_repositories):
        """Test that nested key deletion preserves sibling keys.

        Target: {"a": {"b": "c", "d": "e"}}
        Patch:  {"a": {"b": null}}
        Expected: {"a": {"d": "e"}}

        Key "b" should be deleted, but key "d" should be preserved.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001ce9',
                text=None,
                metadata=None,
                metadata_patch={'a': {'b': None}},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001ce9',
                patch={'a': {'b': None}},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )


class TestMetadataPatchRFC7396AppendixA:
    """Test RFC 7396 Appendix A official test cases at unit level.

    These tests verify that the correct patch data is passed to the repository.
    Actual RFC 7396 semantics are tested at integration level in ``tests.integration._harness``.

    RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396#appendix-A
    """

    @pytest.mark.asyncio
    async def test_rfc7396_case1_simple_value_replacement(self, mock_repositories):
        """RFC 7396 Case #1: Simple value replacement {"a":"b"} + {"a":"c"} = {"a":"c"}."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001c85',
                text=None,
                metadata=None,
                metadata_patch={'a': 'c'},
                tags=None,
                images=None,
            )
            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001c85',
                patch={'a': 'c'},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case2_add_new_key(self, mock_repositories):
        """RFC 7396 Case #2: Add new key {"a":"b"} + {"b":"c"} = {"a":"b","b":"c"}."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001c86',
                text=None,
                metadata=None,
                metadata_patch={'b': 'c'},
                tags=None,
                images=None,
            )
            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001c86',
                patch={'b': 'c'},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case3_delete_key_with_null(self, mock_repositories):
        """RFC 7396 Case #3: Delete key with null {"a":"b"} + {"a":null} = {}."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001c87',
                text=None,
                metadata=None,
                metadata_patch={'a': None},
                tags=None,
                images=None,
            )
            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001c87',
                patch={'a': None},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case4_delete_one_preserve_other(self, mock_repositories):
        """RFC 7396 Case #4: Delete one key, preserve another {"a":"b","b":"c"} + {"a":null} = {"b":"c"}."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001c88',
                text=None,
                metadata=None,
                metadata_patch={'a': None},
                tags=None,
                images=None,
            )
            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001c88',
                patch={'a': None},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case5_array_replacement(self, mock_repositories):
        """RFC 7396 Case #5: Array replacement {"a":["b"]} + {"a":"c"} = {"a":"c"}."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001c89',
                text=None,
                metadata=None,
                metadata_patch={'a': 'c'},
                tags=None,
                images=None,
            )
            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001c89',
                patch={'a': 'c'},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case6_replace_value_with_array(self, mock_repositories):
        """RFC 7396 Case #6: Replace value with array {"a":"c"} + {"a":["b"]} = {"a":["b"]}."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001c8a',
                text=None,
                metadata=None,
                metadata_patch={'a': ['b']},
                tags=None,
                images=None,
            )
            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001c8a',
                patch={'a': ['b']},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case8_array_of_objects_replacement(self, mock_repositories):
        """RFC 7396 Case #8: Array of objects replacement {"a":[{"b":"c"}]} + {"a":[1]} = {"a":[1]}."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001c8c',
                text=None,
                metadata=None,
                metadata_patch={'a': [1]},
                tags=None,
                images=None,
            )
            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001c8c',
                patch={'a': [1]},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )


class TestMetadataPatchRFC7396Semantics:
    """Test RFC 7396 deep merge semantics for metadata_patch.

    These tests verify correct behavior documentation for RFC 7396 compliant
    operations. While using mocks, they document the expected patch data
    that should be passed to achieve RFC 7396 compliance.

    IMPORTANT: Actual RFC 7396 semantics are tested against real databases in
    the real-server harness (``tests.integration._harness``). These tests verify
    the tool correctly delegates to the repository layer.

    RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396
    """

    @pytest.mark.asyncio
    async def test_rfc7396_case7_nested_merge_with_deletion(self, mock_repositories):
        """RFC 7396 Test Case #7: Nested object merge with deletion.

        Verifies that nested patch with null value is correctly passed to repository.
        The actual merge semantics are handled by the database layer.
        """
        nested_patch: MetadataDict = {'a': {'b': 'd', 'c': None}}

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001b5f',
                text=None,
                metadata=None,
                metadata_patch=nested_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001b5f',
                patch=nested_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case13_null_preservation(self, mock_repositories):
        """RFC 7396 Test Case #13: Existing null value preserved.

        Verifies that adding new keys does not affect existing null values in target.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001b65',
                text=None,
                metadata=None,
                metadata_patch={'a': 1},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001b65',
                patch={'a': 1},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_case15_deeply_nested_null(self, mock_repositories):
        """RFC 7396 Test Case #15: Deeply nested null deletion.

        Verifies that deeply nested null patch is correctly passed to repository.
        """
        deep_patch: MetadataDict = {'a': {'bb': {'ccc': None}}}

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001b67',
                text=None,
                metadata=None,
                metadata_patch=deep_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001b67',
                patch=deep_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_deep_merge_preserves_sibling_nested_keys(self, mock_repositories):
        """Verify patch for deep merge with sibling key preservation.

        When patching {"a": {"b": "updated"}}, sibling keys in the nested object
        should be preserved. This test verifies the correct patch is passed.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001bbc',
                text=None,
                metadata=None,
                metadata_patch={'a': {'b': 'updated'}},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001bbc',
                patch={'a': {'b': 'updated'}},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )
