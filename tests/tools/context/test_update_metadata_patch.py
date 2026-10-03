"""Tests for the metadata_patch parameter of update_context (RFC 7396 JSON Merge Patch): adding, updating and
deleting fields, nested and multi-field patches, empty patches, special values, type conversions, and timestamp
stamping. RFC 7396 limitations covered: a null value deletes its key (a value cannot be set to null), and arrays
are replaced, never merged element-wise."""

from unittest.mock import patch

import pytest

import app.tools
from app.types import MetadataDict
from tests.helpers import LOCAL_SCOPE

# Tools are plain async functions registered at server startup, so tests call them directly.
update_context = app.tools.update_context


class TestMetadataPatchBasicOperations:
    """Test basic metadata patch operations: add, update, delete."""

    @pytest.mark.asyncio
    async def test_patch_add_new_field(self, mock_repositories):
        """Test adding a new field to existing metadata using patch.

        RFC 7396: New keys in patch object are added to target.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000007b',
                text=None,
                metadata=None,
                metadata_patch={'new_field': 'new_value'},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert result['context_id'] == '0190abcdef1234567890abcd0000007b'
            assert 'metadata' in result['updated_fields']

            # Verify patch_metadata was called with correct arguments
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd0000007b',
                patch={'new_field': 'new_value'},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_patch_update_existing_field(self, mock_repositories):
        """Test updating an existing field value using patch.

        RFC 7396: Existing keys in target are replaced with patch values.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000001c8',
                text=None,
                metadata=None,
                metadata_patch={'status': 'completed'},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd000001c8',
                patch={'status': 'completed'},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_patch_delete_field_with_null(self, mock_repositories):
        """Test deleting a field by setting it to null.

        RFC 7396: A null value in the patch removes the key from target.
        WARNING: This means you cannot set a value to null - null always means delete.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000315',
                text=None,
                metadata=None,
                metadata_patch={'field_to_delete': None},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00000315',
                patch={'field_to_delete': None},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )


class TestMetadataPatchNestedOperations:
    """Test nested metadata patching operations."""

    @pytest.mark.asyncio
    async def test_patch_nested_metadata(self, mock_repositories):
        """Test patching nested object fields.

        RFC 7396: Nested objects are recursively merged.
        """
        nested_patch: MetadataDict = {
            'user': {
                'preferences': {
                    'theme': 'dark',
                },
            },
        }

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000006f',
                text=None,
                metadata=None,
                metadata_patch=nested_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd0000006f',
                patch=nested_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_patch_deeply_nested_structure(self, mock_repositories):
        """Test patching deeply nested structures."""
        deep_patch: MetadataDict = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 42,
                    },
                },
            },
        }

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000000de',
                text=None,
                metadata=None,
                metadata_patch=deep_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd000000de',
                patch=deep_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )


class TestMetadataPatchMultipleFields:
    """Test patching multiple fields in a single operation."""

    @pytest.mark.asyncio
    async def test_patch_multiple_fields(self, mock_repositories):
        """Test patching multiple fields at once."""
        multi_patch: MetadataDict = {
            'status': 'in_progress',
            'priority': 10,
            'agent_name': 'test-agent',
            'completed': False,
        }

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000014d',
                text=None,
                metadata=None,
                metadata_patch=multi_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd0000014d',
                patch=multi_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_patch_mixed_operations(self, mock_repositories):
        """Test mixed operations: add, update, and delete in one patch.

        This tests the core RFC 7396 behavior where:
        - New keys are added
        - Existing keys are updated
        - null values delete keys
        """
        mixed_patch: MetadataDict = {
            'new_field': 'added',
            'existing_field': 'updated_value',
            'field_to_remove': None,  # RFC 7396: null means delete
        }

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000001bc',
                text=None,
                metadata=None,
                metadata_patch=mixed_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd000001bc',
                patch=mixed_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )


class TestMetadataPatchEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_patch_empty_patch(self, mock_repositories):
        """Test empty patch {} behavior.

        RFC 7396: Empty patch is a no-op for the data but still updates timestamp.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000022b',
                text=None,
                metadata=None,
                metadata_patch={},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd0000022b',
                patch={},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_patch_on_empty_metadata(self, mock_repositories):
        """Test patching when no existing metadata exists.

        The patch should create new metadata from scratch.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000029a',
                text=None,
                metadata=None,
                metadata_patch={'first_field': 'first_value'},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_preserves_unchanged_fields(self, mock_repositories):
        """Verify that unchanged fields remain after patch operation.

        This is tested at the repository level, but we verify the tool correctly
        delegates to the patch_metadata method.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000309',
                text=None,
                metadata=None,
                metadata_patch={'only_this_changes': 'new_value'},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            # The patch_metadata method is responsible for preserving other fields
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_array_replacement(self, mock_repositories):
        """Test that arrays are replaced entirely, not merged.

        RFC 7396 Limitation: Arrays cannot be patched element-wise.
        The entire array is replaced.
        """
        array_patch: MetadataDict = {
            'tags_list': ['new', 'array', 'values'],
        }

        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00000378',
                text=None,
                metadata=None,
                metadata_patch=array_patch,
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00000378',
                patch=array_patch,
                scope=LOCAL_SCOPE,
                txn=ANY,
            )


class TestMetadataPatchTimestamp:
    """Test that metadata_patch properly updates timestamp."""

    @pytest.mark.asyncio
    async def test_patch_updates_timestamp(self, mock_repositories):
        """Verify updated_at timestamp updates when using metadata_patch.

        The repository method should update the timestamp atomically with the patch.
        """
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000015b3',
                text=None,
                metadata=None,
                metadata_patch={'timestamp_test': True},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            # The patch_metadata method includes updated_at = CURRENT_TIMESTAMP
            mock_repositories.context.patch_metadata.assert_called_once()


class TestMetadataPatchSpecialValues:
    """Test metadata_patch with special value types."""

    @pytest.mark.asyncio
    async def test_patch_with_boolean_values(self, mock_repositories):
        """Test patching with boolean values."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001a0a',
                text=None,
                metadata=None,
                metadata_patch={'completed': True, 'active': False},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001a0a',
                patch={'completed': True, 'active': False},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_patch_with_numeric_values(self, mock_repositories):
        """Test patching with integer and float values."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00001e61',
                text=None,
                metadata=None,
                metadata_patch={'priority': 5, 'score': 98.6},
                tags=None,
                images=None,
            )

            assert result['success'] is True
            from unittest.mock import ANY
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id='0190abcdef1234567890abcd00001e61',
                patch={'priority': 5, 'score': 98.6},
                scope=LOCAL_SCOPE,
                txn=ANY,
            )

    @pytest.mark.asyncio
    async def test_patch_with_string_values(self, mock_repositories):
        """Test patching with various string values including special characters."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd000022b8',
                text=None,
                metadata=None,
                metadata_patch={
                    'name': 'Test Agent',
                    'description': 'Contains "quotes" and special chars: <>&',
                    'unicode': 'Hello World',
                },
                tags=None,
                images=None,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once()


class TestMetadataPatchTypeConversions:
    """Test type conversion scenarios for metadata_patch.

    RFC 7396 allows changing value types - objects can become arrays,
    strings can become objects, etc.
    """

    @pytest.mark.asyncio
    async def test_patch_object_to_scalar(self, mock_repositories):
        """Test replacing object value with scalar."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd00002329',
                text=None,
                metadata=None,
                metadata_patch={'config': 'simple_value'},
                tags=None,
                images=None,
            )
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_patch_scalar_to_object(self, mock_repositories):
        """Test replacing scalar value with object."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000232a',
                text=None,
                metadata=None,
                metadata_patch={'status': {'code': 200, 'message': 'OK'}},
                tags=None,
                images=None,
            )
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_patch_array_to_object(self, mock_repositories):
        """Test replacing array value with object."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000232b',
                text=None,
                metadata=None,
                metadata_patch={'items': {'count': 3, 'data': [1, 2, 3]}},
                tags=None,
                images=None,
            )
            assert result['success'] is True

    @pytest.mark.asyncio
    async def test_patch_object_to_array(self, mock_repositories):
        """Test replacing object value with array."""
        with patch('app.tools.context.update.ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id='0190abcdef1234567890abcd0000232c',
                text=None,
                metadata=None,
                metadata_patch={'items': ['a', 'b', 'c']},
                tags=None,
                images=None,
            )
            assert result['success'] is True
