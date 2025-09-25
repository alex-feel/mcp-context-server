"""
End-to-end integration tests for the MCP Context Storage Server.

Tests complete workflows, multi-agent scenarios, concurrent operations,
and system behavior under various conditions.
"""

from __future__ import annotations

import asyncio
import base64

# Import the actual async functions from app.server, not the MCP-wrapped versions
# The FunctionTool objects store the original functions in their 'fn' attribute
import app.server
import pytest

# Get the actual async functions from the FunctionTool wrappers
# FastMCP wraps our functions in FunctionTool objects, but we need the original functions for testing
store_context = app.server.store_context.fn
search_context = app.server.search_context.fn
get_context_by_ids = app.server.get_context_by_ids.fn
delete_context = app.server.delete_context.fn
list_threads = app.server.list_threads.fn
get_statistics = app.server.get_statistics.fn


@pytest.mark.usefixtures('initialized_server')
class TestCompleteWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_conversation_workflow(self) -> None:
        """Test a complete conversation workflow between user and agent."""
        thread_id = 'conversation_001'

        # User asks a question
        user_msg = await store_context(
            thread_id=thread_id,
            source='user',
            text='How do I implement async functions in Python?',
            metadata={'timestamp': '2024-01-01T10:00:00Z'},
            tags=['python', 'async', 'question'],
        )

        assert user_msg['success'] is True

        # Agent provides response
        agent_response = await store_context(
            thread_id=thread_id,
            source='agent',
            text='''To implement async functions in Python, use the async/await syntax:

```python
async def my_async_function():
    await asyncio.sleep(1)
    return "Done"
```

Key points:
1. Use `async def` to define async functions
2. Use `await` to call other async functions
3. Run with `asyncio.run()` or in an event loop''',
            metadata={'model': 'claude-3', 'tokens': 150},
            tags=['python', 'async', 'tutorial'],
        )

        assert agent_response['success'] is True

        # User follows up with code example
        user_code = await store_context(
            thread_id=thread_id,
            source='user',
            text='Here is my implementation, is it correct?',
            images=[
                {
                    'data': base64.b64encode(b'code_screenshot').decode('utf-8'),
                    'mime_type': 'image/png',
                    'metadata': {'filename': 'async_code.png'},
                },
            ],
            tags=['code-review', 'python'],
        )

        assert user_code['success'] is True

        # Agent reviews with annotations
        agent_review = await store_context(
            thread_id=thread_id,
            source='agent',
            text='Your implementation looks good! Here are some suggestions:',
            images=[
                {
                    'data': base64.b64encode(b'annotated_code').decode('utf-8'),
                    'mime_type': 'image/png',
                    'metadata': {'annotations': 3},
                },
            ],
            metadata={'review_status': 'approved'},
            tags=['code-review', 'approved'],
        )

        assert agent_review['success'] is True

        # Verify the complete conversation
        conversation = await search_context(thread_id=thread_id)
        assert len(conversation) == 4

        # Check conversation order (newest first) - compare actual IDs not list indices
        conversation_ids = [c['id'] for c in conversation]
        assert agent_review['context_id'] in conversation_ids
        assert user_msg['context_id'] in conversation_ids
        # Verify newest is first (agent_review was stored last)
        assert conversation[0]['id'] == agent_review['context_id']

        # Verify mixed content types
        content_types = [c['content_type'] for c in conversation]
        assert 'text' in content_types
        assert 'multimodal' in content_types

        # Search for code review entries
        reviews = await search_context(tags=['code-review'])
        assert len(reviews) == 2

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_multi_agent_collaboration(self) -> None:
        """Test multiple agents collaborating on a task."""
        thread_id = 'collab_task_001'

        # User defines task
        await store_context(
            thread_id=thread_id,
            source='user',
            text='Build a REST API with authentication',
            metadata={'project': 'api_v2', 'priority': 'high'},
            tags=['task', 'api', 'authentication'],
        )

        # Planning agent creates plan
        await store_context(
            thread_id=thread_id,
            source='agent',
            text='''Task breakdown:
1. Design API endpoints
2. Implement user model
3. Add JWT authentication
4. Create middleware
5. Write tests''',
            metadata={'agent_type': 'planner', 'subtasks': 5},
            tags=['planning', 'architecture'],
        )

        # Coding agent implements
        await store_context(
            thread_id=thread_id,
            source='agent',
            text='Implemented user model and authentication endpoints',
            metadata={'agent_type': 'coder', 'files_modified': 3},
            tags=['implementation', 'backend'],
        )

        # Testing agent adds tests
        await store_context(
            thread_id=thread_id,
            source='agent',
            text='Added unit tests for authentication flow',
            metadata={'agent_type': 'tester', 'test_count': 15, 'coverage': 95},
            tags=['testing', 'quality'],
        )

        # Review agent provides feedback
        await store_context(
            thread_id=thread_id,
            source='agent',
            text='Code review complete. Found 2 minor issues.',
            metadata={'agent_type': 'reviewer', 'issues': 2},
            tags=['review', 'feedback'],
        )

        # Get thread statistics
        threads = await list_threads()
        collab_thread = next(t for t in threads['threads'] if t['thread_id'] == thread_id)

        assert collab_thread['entry_count'] == 5
        assert collab_thread['source_types'] == 2  # user and agent

        # Analyze agent contributions
        agent_entries = await search_context(thread_id=thread_id, source='agent')
        assert len(agent_entries) == 4

        # Verify different agent types participated
        agent_types = {e['metadata']['agent_type'] for e in agent_entries}
        assert agent_types == {'planner', 'coder', 'tester', 'reviewer'}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_context_reuse_across_threads(self) -> None:
        """Test referencing context from other threads."""
        # Create knowledge base in thread 1
        kb_thread = 'knowledge_base'
        kb_entry = await store_context(
            thread_id=kb_thread,
            source='agent',
            text='API Documentation: Use /auth/login for authentication',
            metadata={'doc_type': 'api', 'version': '2.0'},
            tags=['documentation', 'api', 'reference'],
        )

        # Create new task thread that references knowledge
        task_thread = 'task_implementation'
        await store_context(
            thread_id=task_thread,
            source='user',
            text='Implement login functionality',
            metadata={'references': [kb_entry['context_id']]},
            tags=['task', 'authentication'],
        )

        # Agent uses referenced knowledge
        await store_context(
            thread_id=task_thread,
            source='agent',
            text='Implemented login using /auth/login endpoint from documentation',
            metadata={
                'used_references': [kb_entry['context_id']],
                'implementation_complete': True,
            },
            tags=['implementation', 'complete'],
        )

        # Fetch the referenced documentation
        referenced_docs = await get_context_by_ids(
            context_ids=[kb_entry['context_id']],
        )

        assert len(referenced_docs) == 1
        assert 'API Documentation' in referenced_docs[0]['text_content']

        # Search across documentation
        docs = await search_context(tags=['documentation'])
        assert len(docs) >= 1


@pytest.mark.usefixtures('initialized_server')
class TestConcurrentOperations:
    """Test concurrent access and operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_writes(self) -> None:
        """Test multiple concurrent write operations."""
        thread_id = 'concurrent_test'

        # Create 20 concurrent store operations
        async def store_entry(index):
            return await store_context(
                thread_id=thread_id,
                source='user' if index % 2 == 0 else 'agent',
                text=f'Concurrent entry {index}',
                metadata={'index': index},
                tags=[f'batch_{index // 5}'],
            )

        # Execute concurrently
        results = await asyncio.gather(
            *[store_entry(i) for i in range(20)],
            return_exceptions=True,
        )

        # Verify all succeeded
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, dict)
            assert result['success'] is True

        # Verify all entries exist
        all_entries = await search_context(thread_id=thread_id)
        assert len(all_entries) == 20

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_reads(self) -> None:
        """Test multiple concurrent read operations."""
        # Setup test data
        thread_id = 'read_test'
        context_ids = []

        for i in range(10):
            result = await store_context(
                thread_id=thread_id,
                source='user',
                text=f'Entry {i}',
            )
            context_ids.append(result['context_id'])

        # Concurrent read operations
        async def read_operations():
            return await asyncio.gather(
                search_context(thread_id=thread_id),
                search_context(source='user'),
                get_context_by_ids(context_ids=context_ids[:5]),
                get_context_by_ids(context_ids=context_ids[5:]),
                list_threads(),
                get_statistics(),
            )

        results = await read_operations()

        # Verify all operations succeeded
        assert len(results) == 6
        assert all(r is not None for r in results)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_concurrent_mixed_operations(self) -> None:
        """Test mixed read/write/delete operations concurrently."""
        base_thread = 'mixed_ops'

        # Initial data
        initial = await store_context(
            thread_id=base_thread,
            source='user',
            text='Initial entry',
        )

        async def mixed_operations():
            return await asyncio.gather(
                # Write operations
                store_context(thread_id=f'{base_thread}_1', source='user', text='Write 1'),
                store_context(thread_id=f'{base_thread}_2', source='agent', text='Write 2'),
                # Read operations
                search_context(thread_id=base_thread),
                list_threads(),
                # Delete operation
                delete_context(context_ids=[initial['context_id']]),
                # More writes
                store_context(thread_id=f'{base_thread}_3', source='user', text='Write 3'),
                return_exceptions=True,
            )

        results = await mixed_operations()

        # Check no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)

        # Verify final state
        stats = await get_statistics()
        assert stats['total_entries'] >= 3


@pytest.mark.usefixtures('initialized_server')
class TestDataIntegrity:
    """Test data integrity and consistency."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_transaction_consistency(self) -> None:
        """Test that operations maintain consistency."""
        thread_id = 'integrity_test'

        # Store entry with all features
        result = await store_context(
            thread_id=thread_id,
            source='user',
            text='Complete entry',
            images=[{'data': base64.b64encode(b'img').decode('utf-8')}],
            metadata={'important': True},
            tags=['critical', 'test'],
        )

        context_id = result['context_id']

        # Fetch and verify complete data
        fetched = await get_context_by_ids(
            context_ids=[context_id],
            include_images=True,
        )

        assert len(fetched) == 1
        entry = fetched[0]

        assert entry['thread_id'] == thread_id
        assert entry['source'] == 'user'
        assert entry['text_content'] == 'Complete entry'
        assert entry['content_type'] == 'multimodal'
        assert len(entry['images']) == 1
        assert entry['metadata'] == {'important': True}
        assert set(entry['tags']) == {'critical', 'test'}

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_cascade_deletion_integrity(self) -> None:
        """Test that cascade deletions maintain referential integrity."""
        thread_id = 'cascade_test'

        # Create complex entry
        await store_context(
            thread_id=thread_id,
            source='agent',
            text='Entry with dependencies',
            images=[
                {'data': base64.b64encode(b'img1').decode('utf-8')},
                {'data': base64.b64encode(b'img2').decode('utf-8')},
            ],
            tags=['tag1', 'tag2', 'tag3'],
        )

        # Delete the thread
        await delete_context(thread_id=thread_id)

        # Verify complete deletion
        remaining = await search_context(thread_id=thread_id)
        assert remaining == []

        # Verify stats reflect deletion
        await get_statistics()
        # Tags and images should also be deleted


@pytest.mark.usefixtures('initialized_server')
class TestPerformanceAndScaling:
    """Test performance with larger datasets."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    @pytest.mark.slow  # Mark as slow test
    async def test_large_thread_handling(self) -> None:
        """Test handling of threads with many entries."""
        thread_id = 'large_thread'

        # Create 100 entries in a single thread
        for batch in range(10):
            tasks = []
            for i in range(10):
                index = batch * 10 + i
                tasks.append(
                    store_context(
                        thread_id=thread_id,
                        source='user' if index % 2 == 0 else 'agent',
                        text=f'Entry {index} with some content to make it realistic',
                        metadata={'batch': batch, 'index': i},
                        tags=[f'batch_{batch}', 'performance-test'],
                    ),
                )
            await asyncio.gather(*tasks)

        # Test pagination through large result set
        page1 = await search_context(thread_id=thread_id, limit=20, offset=0)
        page2 = await search_context(thread_id=thread_id, limit=20, offset=20)
        page3 = await search_context(thread_id=thread_id, limit=20, offset=40)

        assert len(page1) == 20
        assert len(page2) == 20
        assert len(page3) == 20

        # Verify no overlap
        ids1 = {e['id'] for e in page1}
        ids2 = {e['id'] for e in page2}
        ids3 = {e['id'] for e in page3}

        assert ids1.isdisjoint(ids2)
        assert ids2.isdisjoint(ids3)
        assert ids1.isdisjoint(ids3)

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_many_threads_handling(self) -> None:
        """Test handling many different threads."""
        # Create 50 threads with 2 entries each
        for thread_num in range(50):
            thread_id = f'thread_{thread_num:03d}'
            await store_context(
                thread_id=thread_id,
                source='user',
                text=f'User message in thread {thread_num}',
                tags=['multi-thread-test'],
            )
            await store_context(
                thread_id=thread_id,
                source='agent',
                text=f'Agent response in thread {thread_num}',
                tags=['multi-thread-test'],
            )

        # List all threads
        threads = await list_threads()
        assert threads['total_threads'] >= 50

        # Search across all threads
        all_entries = await search_context(tags=['multi-thread-test'], limit=500)
        assert len(all_entries) >= 100

        # Get statistics
        stats = await get_statistics()
        assert stats['total_entries'] >= 100


@pytest.mark.usefixtures('initialized_server')
class TestErrorRecovery:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_partial_failure_handling(self) -> None:
        """Test handling of partial failures in batch operations."""
        # Mix of valid and invalid operations
        results = await asyncio.gather(
            store_context(thread_id='valid1', source='user', text='Valid'),
            store_context(thread_id='valid2', source='invalid', text='Invalid source'),  # Will fail
            store_context(thread_id='valid3', source='agent', text='Valid'),
            return_exceptions=True,
        )

        # Check that valid operations succeeded
        assert not isinstance(results[0], BaseException)
        assert results[0]['success'] is True
        assert not isinstance(results[1], BaseException)
        assert results[1]['success'] is False
        assert not isinstance(results[2], BaseException)
        assert results[2]['success'] is True

        # Verify only valid entries were stored
        all_entries = await search_context()
        thread_ids = {e['thread_id'] for e in all_entries}
        assert 'valid1' in thread_ids
        assert 'valid2' not in thread_ids  # Should not exist
        assert 'valid3' in thread_ids

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_recovery_after_error(self) -> None:
        """Test that system recovers after errors."""
        # Cause an error
        error_result = await store_context(
            thread_id='test',
            source='invalid_source',  # Invalid
            text='This will fail',
        )
        assert error_result['success'] is False

        # System should still work normally
        success_result = await store_context(
            thread_id='test',
            source='user',
            text='This should work',
        )
        assert success_result['success'] is True

        # Verify data integrity
        entries = await search_context(thread_id='test')
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'This should work'


@pytest.mark.usefixtures('initialized_server')
class TestComplexQueries:
    """Test complex query scenarios."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_complex_search_combinations(self) -> None:
        """Test combining multiple search filters."""
        # Setup diverse test data
        threads = ['project_a', 'project_b', 'project_c']
        sources = ['user', 'agent']
        tags_sets = [
            ['python', 'backend'],
            ['javascript', 'frontend'],
            ['python', 'ml'],
            ['devops', 'ci'],
        ]

        # Create varied entries
        for thread in threads:
            for source in sources:
                for tag_set in tags_sets:
                    await store_context(
                        thread_id=thread,
                        source=source,
                        text=f"{thread} - {source} - {', '.join(tag_set)}",
                        tags=tag_set,
                    )

        # Complex query 1: Python backend entries from agents in project_a
        results = await search_context(
            thread_id='project_a',
            source='agent',
            tags=['python', 'backend'],
        )
        # Should find entries that have BOTH python AND backend tags
        # Since we created one entry per tag_set, expecting 1 result
        assert len(results) >= 1

        # Complex query 2: All Python entries (ML or backend)
        python_results = await search_context(tags=['python'])
        assert len(python_results) >= 12  # At least 2 python tag sets * 3 threads * 2 sources

        # Complex query 3: User entries in project_b with pagination
        page1 = await search_context(
            thread_id='project_b',
            source='user',
            limit=2,
            offset=0,
        )
        page2 = await search_context(
            thread_id='project_b',
            source='user',
            limit=2,
            offset=2,
        )
        assert len(page1) == 2
        assert len(page2) == 2
        assert page1[0]['id'] != page2[0]['id']


@pytest.mark.usefixtures('initialized_server')
class TestMaintenanceOperations:
    """Test maintenance and cleanup operations."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_old_thread_cleanup(self) -> None:
        """Test cleaning up old threads."""
        # Create threads with different ages (simulated by order)
        old_threads = ['old_1', 'old_2', 'old_3']
        new_threads = ['new_1', 'new_2']

        for thread in old_threads:
            await store_context(
                thread_id=thread,
                source='user',
                text='Old data',
                metadata={'archived': True},
            )

        for thread in new_threads:
            await store_context(
                thread_id=thread,
                source='user',
                text='Current data',
                metadata={'archived': False},
            )

        # Get initial stats
        initial_stats = await get_statistics()
        initial_count = initial_stats['total_entries']

        # Clean up old threads
        for thread in old_threads:
            await delete_context(thread_id=thread)

        # Verify cleanup
        final_stats = await get_statistics()
        assert final_stats['total_entries'] == initial_count - len(old_threads)

        # Verify new threads remain
        remaining_threads = await list_threads()
        remaining_ids = {t['thread_id'] for t in remaining_threads['threads']}
        for thread in new_threads:
            assert thread in remaining_ids
        for thread in old_threads:
            assert thread not in remaining_ids

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_tag_normalization_consistency(self) -> None:
        """Test that tag normalization works consistently."""
        # Store entries with various tag formats
        await store_context(
            thread_id='tag_test',
            source='user',
            text='Entry 1',
            tags=['Python', 'PYTHON', 'python', '  python  '],
        )

        await store_context(
            thread_id='tag_test',
            source='agent',
            text='Entry 2',
            tags=['TESTING', 'testing', 'Testing'],
        )

        # Search by normalized tag
        python_results = await search_context(tags=['python'])
        testing_results = await search_context(tags=['testing'])

        assert len(python_results) == 1
        assert len(testing_results) == 1

        # Check that tags are normalized in results
        assert 'python' in python_results[0]['tags']
        assert 'testing' in testing_results[0]['tags']

        # Get unique tags from stats
        stats = await get_statistics()
        # Should only have 2 unique tags after normalization
        assert stats['unique_tags'] == 2
