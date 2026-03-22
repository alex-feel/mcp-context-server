"""Comprehensive tests for bulk operations in MCP Context Server.

This module tests the three bulk operation MCP tools:
- store_context_batch: Batch insert with deduplication
- update_context_batch: Batch update with partial field support
- delete_context_batch: Criteria-based batch delete

Tests cover:
- Repository layer unit tests
- MCP tool integration tests
- Atomic and non-atomic mode tests
- Error handling and validation tests
"""

from __future__ import annotations

import pytest
from fastmcp.exceptions import ToolError

# Import the actual async functions from app.server, not the MCP-wrapped versions
# The FunctionTool objects store the original functions in their 'fn' attribute
import app.server

# Get the actual async functions - they are no longer wrapped by @mcp.tool() at import time
# Tools are registered dynamically in lifespan(), so we can access the functions directly
store_context = app.server.store_context
store_context_batch = app.server.store_context_batch
update_context_batch = app.server.update_context_batch
delete_context_batch = app.server.delete_context_batch


@pytest.mark.usefixtures('initialized_server')
class TestStoreContextBatch:
    """Tests for the store_context_batch MCP tool."""

    @pytest.mark.asyncio
    async def test_store_batch_success_atomic(self) -> None:
        """Test successful batch store with atomic mode."""
        entries = [
            {'thread_id': 'batch-test-1', 'source': 'user', 'text': 'First entry'},
            {'thread_id': 'batch-test-1', 'source': 'agent', 'text': 'Second entry'},
            {'thread_id': 'batch-test-2', 'source': 'user', 'text': 'Third entry'},
        ]

        result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert result['total'] == 3
        assert result['succeeded'] == 3
        assert result['failed'] == 0
        assert len(result['results']) == 3

        # Verify all results have context_ids
        for item in result['results']:
            assert item['success'] is True
            assert item['context_id'] is not None
            assert item['error'] is None

    @pytest.mark.asyncio
    async def test_store_batch_success_non_atomic(self) -> None:
        """Test successful batch store with non-atomic mode."""
        entries = [
            {'thread_id': 'batch-non-atomic-1', 'source': 'user', 'text': 'Entry one'},
            {'thread_id': 'batch-non-atomic-2', 'source': 'agent', 'text': 'Entry two'},
        ]

        result = await store_context_batch(entries=entries, atomic=False)

        assert result['success'] is True
        assert result['total'] == 2
        assert result['succeeded'] == 2
        assert result['failed'] == 0

    @pytest.mark.asyncio
    async def test_store_batch_with_metadata(self) -> None:
        """Test batch store with metadata."""
        entries = [
            {
                'thread_id': 'batch-meta-1',
                'source': 'user',
                'text': 'Entry with metadata',
                'metadata': {'priority': 1, 'status': 'pending'},
            },
            {
                'thread_id': 'batch-meta-1',
                'source': 'agent',
                'text': 'Another entry',
                'metadata': {'agent_name': 'test-agent'},
            },
        ]

        result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 2

    @pytest.mark.asyncio
    async def test_store_batch_with_tags(self) -> None:
        """Test batch store with tags."""
        entries = [
            {
                'thread_id': 'batch-tags-1',
                'source': 'user',
                'text': 'Entry with tags',
                'tags': ['important', 'review'],
            },
            {
                'thread_id': 'batch-tags-1',
                'source': 'agent',
                'text': 'Another entry',
                'tags': ['processed'],
            },
        ]

        result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 2

    @pytest.mark.asyncio
    async def test_store_batch_deduplication(self) -> None:
        """Test deduplication when storing duplicate entries."""
        entries = [
            {'thread_id': 'batch-dedup-1', 'source': 'user', 'text': 'Duplicate entry'},
            {'thread_id': 'batch-dedup-1', 'source': 'user', 'text': 'Duplicate entry'},
        ]

        result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 2

        # Both should return the same context_id due to deduplication
        ids = [r['context_id'] for r in result['results']]
        assert ids[0] == ids[1]

    @pytest.mark.asyncio
    async def test_store_batch_validation_error_atomic(self) -> None:
        """Test that atomic mode fails fast on validation error."""
        entries = [
            {'thread_id': 'batch-valid-1', 'source': 'user', 'text': 'Valid entry'},
            {'thread_id': 'batch-valid-1', 'source': 'invalid', 'text': 'Invalid source'},
            {'thread_id': 'batch-valid-1', 'source': 'user', 'text': 'Another valid'},
        ]

        with pytest.raises(ToolError) as exc_info:
            await store_context_batch(entries=entries, atomic=True)

        assert 'Validation failed' in str(exc_info.value)
        assert 'source' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_store_batch_validation_error_non_atomic(self) -> None:
        """Test non-atomic mode allows partial success with validation errors."""
        entries = [
            {'thread_id': 'batch-partial-1', 'source': 'user', 'text': 'Valid entry'},
            {'thread_id': 'batch-partial-1', 'source': 'invalid', 'text': 'Invalid source'},
            {'thread_id': 'batch-partial-1', 'source': 'agent', 'text': 'Another valid'},
        ]

        result = await store_context_batch(entries=entries, atomic=False)

        assert result['success'] is False
        assert result['total'] == 3
        assert result['succeeded'] == 2
        assert result['failed'] == 1

        # Check the failed entry
        failed = [r for r in result['results'] if not r['success']]
        assert len(failed) == 1
        assert failed[0]['index'] == 1
        error_msg = failed[0]['error']
        assert error_msg is not None
        assert 'source' in error_msg.lower()

    @pytest.mark.asyncio
    async def test_store_batch_missing_thread_id(self) -> None:
        """Test validation error for missing thread_id."""
        entries = [
            {'source': 'user', 'text': 'Missing thread_id'},
        ]

        result = await store_context_batch(entries=entries, atomic=False)

        assert result['success'] is False
        assert result['failed'] == 1
        error_msg = result['results'][0]['error']
        assert error_msg is not None
        assert 'thread_id' in error_msg.lower()

    @pytest.mark.asyncio
    async def test_store_batch_missing_text(self) -> None:
        """Test validation error for missing text."""
        entries = [
            {'thread_id': 'batch-no-text', 'source': 'user'},
        ]

        result = await store_context_batch(entries=entries, atomic=False)

        assert result['success'] is False
        assert result['failed'] == 1
        error_msg = result['results'][0]['error']
        assert error_msg is not None
        assert 'text' in error_msg.lower()

    @pytest.mark.asyncio
    async def test_store_batch_empty_text(self) -> None:
        """Test validation error for empty text."""
        entries = [
            {'thread_id': 'batch-empty-text', 'source': 'user', 'text': '   '},
        ]

        result = await store_context_batch(entries=entries, atomic=False)

        assert result['success'] is False
        assert result['failed'] == 1
        error_msg = result['results'][0]['error']
        assert error_msg is not None
        assert 'empty' in error_msg.lower()


@pytest.mark.usefixtures('initialized_server')
class TestUpdateContextBatch:
    """Tests for the update_context_batch MCP tool."""

    @pytest.mark.asyncio
    async def test_update_batch_text_success(self) -> None:
        """Test successful batch text update."""
        # First, create entries to update
        entry1 = await store_context(
            thread_id='update-batch-1',
            source='user',
            text='Original text 1',
        )
        entry2 = await store_context(
            thread_id='update-batch-1',
            source='agent',
            text='Original text 2',
        )

        updates = [
            {'context_id': entry1['context_id'], 'text': 'Updated text 1'},
            {'context_id': entry2['context_id'], 'text': 'Updated text 2'},
        ]

        result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert result['total'] == 2
        assert result['succeeded'] == 2

        # Verify updated_fields includes text_content
        for item in result['results']:
            assert item['success'] is True
            assert item['updated_fields'] is not None
            assert 'text_content' in item['updated_fields']

    @pytest.mark.asyncio
    async def test_update_batch_metadata_success(self) -> None:
        """Test successful batch metadata update."""
        entry = await store_context(
            thread_id='update-meta-1',
            source='user',
            text='Entry for metadata update',
            metadata={'original': True},
        )

        updates = [
            {
                'context_id': entry['context_id'],
                'metadata': {'updated': True, 'version': 2},
            },
        ]

        result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 1
        assert result['results'][0]['updated_fields'] is not None
        assert 'metadata' in result['results'][0]['updated_fields']

    @pytest.mark.asyncio
    async def test_update_batch_tags_success(self) -> None:
        """Test successful batch tags update."""
        entry = await store_context(
            thread_id='update-tags-1',
            source='user',
            text='Entry for tags update',
            tags=['old-tag'],
        )

        updates = [
            {
                'context_id': entry['context_id'],
                'tags': ['new-tag-1', 'new-tag-2'],
            },
        ]

        result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 1
        assert result['results'][0]['updated_fields'] is not None
        assert 'tags' in result['results'][0]['updated_fields']

    @pytest.mark.asyncio
    async def test_update_batch_not_found(self) -> None:
        """Test update of non-existent context entry."""
        updates = [
            {'context_id': 999999, 'text': 'This should fail'},
        ]

        result = await update_context_batch(updates=updates, atomic=False)

        assert result['success'] is False
        assert result['failed'] == 1
        error_msg = result['results'][0]['error']
        assert error_msg is not None
        assert 'not found' in error_msg.lower()

    @pytest.mark.asyncio
    async def test_update_batch_missing_context_id(self) -> None:
        """Test validation error for missing context_id."""
        updates = [
            {'text': 'No context_id provided'},
        ]

        result = await update_context_batch(updates=updates, atomic=False)

        assert result['success'] is False
        assert result['failed'] == 1
        error_msg = result['results'][0]['error']
        assert error_msg is not None
        assert 'context_id' in error_msg.lower()

    @pytest.mark.asyncio
    async def test_update_batch_metadata_and_patch_conflict(self) -> None:
        """Test validation error when both metadata and metadata_patch are provided."""
        entry = await store_context(
            thread_id='update-conflict-1',
            source='user',
            text='Entry for conflict test',
        )

        updates = [
            {
                'context_id': entry['context_id'],
                'metadata': {'full': True},
                'metadata_patch': {'partial': True},
            },
        ]

        result = await update_context_batch(updates=updates, atomic=False)

        assert result['success'] is False
        assert result['failed'] == 1
        error_msg = result['results'][0]['error']
        assert error_msg is not None
        assert 'both' in error_msg.lower()

    @pytest.mark.asyncio
    async def test_update_batch_no_fields_to_update(self) -> None:
        """Test validation error when no fields are provided for update."""
        entry = await store_context(
            thread_id='update-nofields-1',
            source='user',
            text='Entry for no-fields test',
        )

        updates = [
            {'context_id': entry['context_id']},
        ]

        result = await update_context_batch(updates=updates, atomic=False)

        assert result['success'] is False
        assert result['failed'] == 1
        error_msg = result['results'][0]['error']
        assert error_msg is not None
        assert 'field' in error_msg.lower()

    @pytest.mark.asyncio
    async def test_update_batch_partial_success(self) -> None:
        """Test non-atomic mode with partial success."""
        entry = await store_context(
            thread_id='update-partial-1',
            source='user',
            text='Valid entry',
        )

        updates = [
            {'context_id': entry['context_id'], 'text': 'Updated successfully'},
            {'context_id': 999999, 'text': 'This will fail'},
        ]

        result = await update_context_batch(updates=updates, atomic=False)

        assert result['success'] is False
        assert result['total'] == 2
        assert result['succeeded'] == 1
        assert result['failed'] == 1

    @pytest.mark.asyncio
    async def test_update_batch_atomic_rollback(self) -> None:
        """Test atomic mode fails fast on validation error."""
        updates = [
            {'context_id': 1, 'text': 'Valid update'},
            {'context_id': -1, 'text': 'Invalid context_id'},
        ]

        with pytest.raises(ToolError) as exc_info:
            await update_context_batch(updates=updates, atomic=True)

        assert 'Validation failed' in str(exc_info.value)


@pytest.mark.usefixtures('initialized_server')
class TestDeleteContextBatch:
    """Tests for the delete_context_batch MCP tool."""

    @pytest.mark.asyncio
    async def test_delete_batch_by_ids(self) -> None:
        """Test batch delete by context IDs."""
        entry1 = await store_context(
            thread_id='delete-batch-1',
            source='user',
            text='Entry to delete 1',
        )
        entry2 = await store_context(
            thread_id='delete-batch-1',
            source='agent',
            text='Entry to delete 2',
        )

        result = await delete_context_batch(
            context_ids=[entry1['context_id'], entry2['context_id']],
        )

        assert result['success'] is True
        assert result['deleted_count'] == 2
        assert any('context_ids' in c for c in result['criteria_used'])

    @pytest.mark.asyncio
    async def test_delete_batch_by_thread_ids(self) -> None:
        """Test batch delete by thread IDs."""
        await store_context(
            thread_id='delete-thread-1',
            source='user',
            text='Entry in thread 1',
        )
        await store_context(
            thread_id='delete-thread-1',
            source='agent',
            text='Another entry in thread 1',
        )
        await store_context(
            thread_id='delete-thread-2',
            source='user',
            text='Entry in thread 2',
        )

        result = await delete_context_batch(
            thread_ids=['delete-thread-1', 'delete-thread-2'],
        )

        assert result['success'] is True
        assert result['deleted_count'] == 3
        assert any('thread_ids' in c for c in result['criteria_used'])

    @pytest.mark.asyncio
    async def test_delete_batch_by_thread_and_source(self) -> None:
        """Test batch delete by thread with source filter."""
        await store_context(
            thread_id='delete-source-1',
            source='user',
            text='User entry',
        )
        await store_context(
            thread_id='delete-source-1',
            source='agent',
            text='Agent entry',
        )

        result = await delete_context_batch(
            thread_ids=['delete-source-1'],
            source='user',
        )

        assert result['success'] is True
        assert result['deleted_count'] == 1
        assert any('source' in c for c in result['criteria_used'])

    @pytest.mark.asyncio
    async def test_delete_batch_no_criteria_error(self) -> None:
        """Test that delete fails when no criteria provided."""
        with pytest.raises(ToolError) as exc_info:
            await delete_context_batch()

        assert 'criterion' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_delete_batch_source_only_error(self) -> None:
        """Test that source alone is not a valid criterion."""
        with pytest.raises(ToolError) as exc_info:
            await delete_context_batch(source='user')

        assert 'combined' in str(exc_info.value).lower()

    @pytest.mark.asyncio
    async def test_delete_batch_no_matches(self) -> None:
        """Test delete with no matching entries."""
        result = await delete_context_batch(
            context_ids=[999998, 999999],
        )

        assert result['success'] is True
        assert result['deleted_count'] == 0


@pytest.mark.usefixtures('initialized_server')
class TestBulkOperationsEdgeCases:
    """Edge case tests for bulk operations."""

    @pytest.mark.asyncio
    async def test_store_batch_large_batch(self) -> None:
        """Test batch store with larger batch (50 entries)."""
        entries = [
            {
                'thread_id': f'large-batch-{i % 5}',
                'source': 'user' if i % 2 == 0 else 'agent',
                'text': f'Entry number {i}',
            }
            for i in range(50)
        ]

        result = await store_context_batch(entries=entries, atomic=True)

        assert result['success'] is True
        assert result['total'] == 50
        assert result['succeeded'] == 50

    @pytest.mark.asyncio
    async def test_update_batch_multiple_fields(self) -> None:
        """Test updating multiple fields in single batch update."""
        entry = await store_context(
            thread_id='multi-field-1',
            source='user',
            text='Original text',
            metadata={'original': True},
            tags=['original'],
        )

        updates = [
            {
                'context_id': entry['context_id'],
                'text': 'Updated text',
                'metadata': {'updated': True},
                'tags': ['updated'],
            },
        ]

        result = await update_context_batch(updates=updates, atomic=True)

        assert result['success'] is True
        assert result['succeeded'] == 1
        updated_fields = result['results'][0]['updated_fields']
        assert updated_fields is not None
        assert 'text_content' in updated_fields
        assert 'metadata' in updated_fields
        assert 'tags' in updated_fields

    @pytest.mark.asyncio
    async def test_results_ordered_by_index(self) -> None:
        """Test that results are ordered by original index."""
        entries = [
            {'thread_id': 'order-test', 'source': 'user', 'text': f'Entry {i}'}
            for i in range(10)
        ]

        result = await store_context_batch(entries=entries, atomic=True)

        # Verify results are in order
        indices = [r['index'] for r in result['results']]
        assert indices == list(range(10))
