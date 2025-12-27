"""
Comprehensive tests for the metadata_patch feature in update_context tool.

Tests cover RFC 7396 JSON Merge Patch semantics including:
- Adding new fields to existing metadata
- Updating existing field values
- Deleting fields using null values
- Nested metadata patching
- Multiple field operations in a single patch
- Empty patch behavior
- Patching when no existing metadata exists
- Mutual exclusivity validation (metadata vs metadata_patch)
- Preservation of unchanged fields
- Timestamp update verification

RFC 7396 Limitations Tested:
- null values DELETE keys (cannot set value to null)
- Array operations are replace-only (no append/remove individual elements)
"""

from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastmcp import Context
from fastmcp.exceptions import ToolError

import app.server

# Get the actual async function - no longer wrapped by @mcp.tool() at import time
update_context = app.server.update_context


@pytest.fixture
def mock_context():
    """Create a mock FastMCP context for testing."""
    ctx = Mock(spec=Context)
    ctx.info = AsyncMock()
    return ctx


@pytest.fixture
def mock_repositories():
    """Create mock repository container with all necessary repositories."""
    repos = Mock()

    # Mock context repository
    repos.context = Mock()
    repos.context.check_entry_exists = AsyncMock(return_value=True)
    repos.context.update_context_entry = AsyncMock(return_value=(True, ['text_content']))
    repos.context.get_content_type = AsyncMock(return_value='text')
    repos.context.update_content_type = AsyncMock(return_value=True)
    repos.context.patch_metadata = AsyncMock(return_value=(True, ['metadata']))

    # Mock tags repository
    repos.tags = Mock()
    repos.tags.replace_tags_for_context = AsyncMock()

    # Mock images repository
    repos.images = Mock()
    repos.images.replace_images_for_context = AsyncMock()
    repos.images.count_images_for_context = AsyncMock(return_value=0)

    return repos


class TestMetadataPatchBasicOperations:
    """Test basic metadata patch operations: add, update, delete."""

    @pytest.mark.asyncio
    async def test_patch_add_new_field(self, mock_context, mock_repositories):
        """Test adding a new field to existing metadata using patch.

        RFC 7396: New keys in patch object are added to target.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=123,
                text=None,
                metadata=None,
                metadata_patch={'new_field': 'new_value'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert result['context_id'] == 123
            assert 'metadata' in result['updated_fields']

            # Verify patch_metadata was called with correct arguments
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=123,
                patch={'new_field': 'new_value'},
            )

    @pytest.mark.asyncio
    async def test_patch_update_existing_field(self, mock_context, mock_repositories):
        """Test updating an existing field value using patch.

        RFC 7396: Existing keys in target are replaced with patch values.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=456,
                text=None,
                metadata=None,
                metadata_patch={'status': 'completed'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=456,
                patch={'status': 'completed'},
            )

    @pytest.mark.asyncio
    async def test_patch_delete_field_with_null(self, mock_context, mock_repositories):
        """Test deleting a field by setting it to null.

        RFC 7396: A null value in the patch removes the key from target.
        WARNING: This means you cannot set a value to null - null always means delete.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=789,
                text=None,
                metadata=None,
                metadata_patch={'field_to_delete': None},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=789,
                patch={'field_to_delete': None},
            )


class TestMetadataPatchNestedOperations:
    """Test nested metadata patching operations."""

    @pytest.mark.asyncio
    async def test_patch_nested_metadata(self, mock_context, mock_repositories):
        """Test patching nested object fields.

        RFC 7396: Nested objects are recursively merged.
        """
        nested_patch = {
            'user': {
                'preferences': {
                    'theme': 'dark',
                },
            },
        }

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=111,
                text=None,
                metadata=None,
                metadata_patch=nested_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']

            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=111,
                patch=nested_patch,
            )

    @pytest.mark.asyncio
    async def test_patch_deeply_nested_structure(self, mock_context, mock_repositories):
        """Test patching deeply nested structures."""
        deep_patch = {
            'level1': {
                'level2': {
                    'level3': {
                        'value': 42,
                    },
                },
            },
        }

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=222,
                text=None,
                metadata=None,
                metadata_patch=deep_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=222,
                patch=deep_patch,
            )


class TestMetadataPatchMultipleFields:
    """Test patching multiple fields in a single operation."""

    @pytest.mark.asyncio
    async def test_patch_multiple_fields(self, mock_context, mock_repositories):
        """Test patching multiple fields at once."""
        multi_patch = {
            'status': 'in_progress',
            'priority': 10,
            'agent_name': 'test-agent',
            'completed': False,
        }

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=333,
                text=None,
                metadata=None,
                metadata_patch=multi_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=333,
                patch=multi_patch,
            )

    @pytest.mark.asyncio
    async def test_patch_mixed_operations(self, mock_context, mock_repositories):
        """Test mixed operations: add, update, and delete in one patch.

        This tests the core RFC 7396 behavior where:
        - New keys are added
        - Existing keys are updated
        - null values delete keys
        """
        mixed_patch = {
            'new_field': 'added',
            'existing_field': 'updated_value',
            'field_to_remove': None,  # RFC 7396: null means delete
        }

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=444,
                text=None,
                metadata=None,
                metadata_patch=mixed_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=444,
                patch=mixed_patch,
            )


class TestMetadataPatchEdgeCases:
    """Test edge cases and special scenarios."""

    @pytest.mark.asyncio
    async def test_patch_empty_patch(self, mock_context, mock_repositories):
        """Test empty patch {} behavior.

        RFC 7396: Empty patch is a no-op for the data but still updates timestamp.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=555,
                text=None,
                metadata=None,
                metadata_patch={},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=555,
                patch={},
            )

    @pytest.mark.asyncio
    async def test_patch_on_empty_metadata(self, mock_context, mock_repositories):
        """Test patching when no existing metadata exists.

        The patch should create new metadata from scratch.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=666,
                text=None,
                metadata=None,
                metadata_patch={'first_field': 'first_value'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_preserves_unchanged_fields(self, mock_context, mock_repositories):
        """Verify that unchanged fields remain after patch operation.

        This is tested at the repository level, but we verify the tool correctly
        delegates to the patch_metadata method.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=777,
                text=None,
                metadata=None,
                metadata_patch={'only_this_changes': 'new_value'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            # The patch_metadata method is responsible for preserving other fields
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_array_replacement(self, mock_context, mock_repositories):
        """Test that arrays are replaced entirely, not merged.

        RFC 7396 Limitation: Arrays cannot be patched element-wise.
        The entire array is replaced.
        """
        array_patch = {
            'tags_list': ['new', 'array', 'values'],
        }

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=888,
                text=None,
                metadata=None,
                metadata_patch=array_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=888,
                patch=array_patch,
            )


class TestMetadataPatchValidation:
    """Test validation and error handling for metadata_patch."""

    @pytest.mark.asyncio
    async def test_mutual_exclusivity_error(self, mock_context, mock_repositories):
        """Test error when both metadata and metadata_patch are provided.

        These parameters are mutually exclusive - use metadata for full replacement
        or metadata_patch for partial updates.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=999,
                    text=None,
                    metadata={'full': 'replacement'},
                    metadata_patch={'partial': 'update'},
                    tags=None,
                    images=None,
                    ctx=mock_context,
                )

            error_message = str(exc_info.value).lower()
            assert 'metadata' in error_message
            assert 'metadata_patch' in error_message or 'both' in error_message or 'mutually exclusive' in error_message

    @pytest.mark.asyncio
    async def test_context_not_found_error(self, mock_context, mock_repositories):
        """Test error when context entry doesn't exist."""
        mock_repositories.context.check_entry_exists.return_value = False

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=12345,
                    text=None,
                    metadata=None,
                    metadata_patch={'field': 'value'},
                    tags=None,
                    images=None,
                    ctx=mock_context,
                )

            assert '12345' in str(exc_info.value) or 'not found' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_patch_metadata_failure(self, mock_context, mock_repositories):
        """Test handling of repository patch_metadata failure."""
        mock_repositories.context.patch_metadata.return_value = (False, [])

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            with pytest.raises(ToolError) as exc_info:
                await update_context(
                    context_id=1111,
                    text=None,
                    metadata=None,
                    metadata_patch={'field': 'value'},
                    tags=None,
                    images=None,
                    ctx=mock_context,
                )

            assert 'failed' in str(exc_info.value).lower()


class TestMetadataPatchWithOtherFields:
    """Test metadata_patch combined with other field updates."""

    @pytest.mark.asyncio
    async def test_patch_with_text_update(self, mock_context, mock_repositories):
        """Test metadata_patch combined with text content update."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=2222,
                text='Updated text content',
                metadata=None,
                metadata_patch={'status': 'updated'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'text_content' in result['updated_fields']
            assert 'metadata' in result['updated_fields']

            # Verify both operations were called
            mock_repositories.context.update_context_entry.assert_called_once()
            mock_repositories.context.patch_metadata.assert_called_once()

    @pytest.mark.asyncio
    async def test_patch_with_tags_update(self, mock_context, mock_repositories):
        """Test metadata_patch combined with tags update."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=3333,
                text=None,
                metadata=None,
                metadata_patch={'status': 'tagged'},
                tags=['new-tag'],
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']
            assert 'tags' in result['updated_fields']

    @pytest.mark.asyncio
    async def test_patch_alone_is_valid_update(self, mock_context, mock_repositories):
        """Test that metadata_patch alone constitutes a valid update.

        Unlike the error when no fields provided, metadata_patch alone should work.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=4444,
                text=None,
                metadata=None,
                metadata_patch={'only_field': 'only_value'},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            assert 'metadata' in result['updated_fields']


class TestMetadataPatchTimestamp:
    """Test that metadata_patch properly updates timestamp."""

    @pytest.mark.asyncio
    async def test_patch_updates_timestamp(self, mock_context, mock_repositories):
        """Verify updated_at timestamp updates when using metadata_patch.

        The repository method should update the timestamp atomically with the patch.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=5555,
                text=None,
                metadata=None,
                metadata_patch={'timestamp_test': True},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            # The patch_metadata method includes updated_at = CURRENT_TIMESTAMP
            mock_repositories.context.patch_metadata.assert_called_once()


class TestMetadataPatchSpecialValues:
    """Test metadata_patch with special value types."""

    @pytest.mark.asyncio
    async def test_patch_with_boolean_values(self, mock_context, mock_repositories):
        """Test patching with boolean values."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=6666,
                text=None,
                metadata=None,
                metadata_patch={'completed': True, 'active': False},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=6666,
                patch={'completed': True, 'active': False},
            )

    @pytest.mark.asyncio
    async def test_patch_with_numeric_values(self, mock_context, mock_repositories):
        """Test patching with integer and float values."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7777,
                text=None,
                metadata=None,
                metadata_patch={'priority': 5, 'score': 98.6},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7777,
                patch={'priority': 5, 'score': 98.6},
            )

    @pytest.mark.asyncio
    async def test_patch_with_string_values(self, mock_context, mock_repositories):
        """Test patching with various string values including special characters."""
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=8888,
                text=None,
                metadata=None,
                metadata_patch={
                    'name': 'Test Agent',
                    'description': 'Contains "quotes" and special chars: <>&',
                    'unicode': 'Hello World',
                },
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once()


class TestRFC7396DeepMergeSemantics:
    """Test RFC 7396 deep merge semantics for metadata_patch.

    These tests verify that the correct patch data is passed to the repository
    for RFC 7396 compliant operations. The actual deep merge logic is tested
    at the integration level in test_real_server.py.

    RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396
    """

    @pytest.mark.asyncio
    async def test_rfc7396_nested_object_merge_case7(self, mock_context, mock_repositories):
        """RFC 7396 Test Case #7: Nested object merge with deletion.

        Target: {"a": {"b": "c"}}
        Patch:  {"a": {"b": "d", "c": null}}
        Expected: {"a": {"b": "d"}}

        The null value in the nested patch should delete that key from the
        nested object, not from the top-level object.
        """
        nested_patch = {
            'a': {
                'b': 'd',
                'c': None,  # RFC 7396: null means delete
            },
        }

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7396,
                text=None,
                metadata=None,
                metadata_patch=nested_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7396,
                patch=nested_patch,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_existing_null_preserved_case13(self, mock_context, mock_repositories):
        """RFC 7396 Test Case #13: Existing null value preserved.

        Target: {"e": null}
        Patch:  {"a": 1}
        Expected: {"e": null, "a": 1}

        CRITICAL: A null value in the TARGET is preserved (it's actual data).
        Only null values in the PATCH cause deletion.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7413,
                text=None,
                metadata=None,
                metadata_patch={'a': 1},  # Does not affect existing null in target
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7413,
                patch={'a': 1},
            )

    @pytest.mark.asyncio
    async def test_rfc7396_deeply_nested_null_deletion_case15(self, mock_context, mock_repositories):
        """RFC 7396 Test Case #15: Deeply nested null deletion.

        Target: {}
        Patch:  {"a": {"bb": {"ccc": null}}}
        Expected: {"a": {"bb": {}}}

        The null value at depth 3 causes deletion at that level,
        but the containing objects are created/preserved.
        """
        deep_patch = {
            'a': {
                'bb': {
                    'ccc': None,  # RFC 7396: null deletes at depth 3
                },
            },
        }

        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7415,
                text=None,
                metadata=None,
                metadata_patch=deep_patch,
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7415,
                patch=deep_patch,
            )

    @pytest.mark.asyncio
    async def test_rfc7396_deep_merge_preserves_sibling_keys(self, mock_context, mock_repositories):
        """Test that deep merge preserves sibling keys in nested objects.

        Target: {"a": {"b": "c", "d": "e"}}
        Patch:  {"a": {"b": "updated"}}
        Expected: {"a": {"b": "updated", "d": "e"}}

        Key "d" should be preserved because it's not mentioned in the patch.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7400,
                text=None,
                metadata=None,
                metadata_patch={'a': {'b': 'updated'}},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7400,
                patch={'a': {'b': 'updated'}},
            )

    @pytest.mark.asyncio
    async def test_rfc7396_nested_key_deletion_preserves_siblings(self, mock_context, mock_repositories):
        """Test that nested key deletion preserves sibling keys.

        Target: {"a": {"b": "c", "d": "e"}}
        Patch:  {"a": {"b": null}}
        Expected: {"a": {"d": "e"}}

        Key "b" should be deleted, but key "d" should be preserved.
        """
        with patch('app.server._ensure_repositories', return_value=mock_repositories):
            result = await update_context(
                context_id=7401,
                text=None,
                metadata=None,
                metadata_patch={'a': {'b': None}},
                tags=None,
                images=None,
                ctx=mock_context,
            )

            assert result['success'] is True
            mock_repositories.context.patch_metadata.assert_called_once_with(
                context_id=7401,
                patch={'a': {'b': None}},
            )
