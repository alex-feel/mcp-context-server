"""
Test that nested JSON structures can be stored in metadata after the fix.

This test verifies that the metadata type definition fix allows complex
nested JSON structures to be stored and retrieved correctly.
"""

from __future__ import annotations

import math

import pytest

# Import the actual async functions from app.server, not the MCP-wrapped versions
# The FunctionTool objects store the original functions in their 'fn' attribute
import app.server

# Get the actual async functions - they are no longer wrapped by @mcp.tool() at import time
store_context = app.server.store_context
search_context = app.server.search_context


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
async def test_complex_nested_metadata() -> None:
    """Test that complex nested JSON structures can be stored in metadata."""
    complex_metadata = {
        'database': {
            'connection': {
                'pool': {
                    'size': 10,
                    'timeout': 30,
                    'retry': {
                        'max_attempts': 3,
                        'backoff_ms': 100,
                    },
                },
            },
            'config': {
                'read_only': False,
                'cache_enabled': True,
            },
        },
        'tags': ['urgent', 'backend', 'production'],
        'metrics': {
            'cpu': 45.5,
            'memory': 512,
            'active_connections': [1, 2, 3, 4, 5],
        },
        'user': {
            'preferences': {
                'theme': 'dark',
                'notifications': {
                    'email': True,
                    'sms': False,
                },
            },
        },
    }

    result = await store_context(
        thread_id='test_nested_json',
        source='agent',
        text='Testing nested JSON metadata',
        metadata=complex_metadata,
    )

    assert result['success'] is True
    assert result['context_id'] > 0

    # Verify retrieval
    search_result = await search_context(
        thread_id='test_nested_json',
        limit=1,
    )

    entries = search_result.get('results', [])
    assert len(entries) == 1
    retrieved_metadata = entries[0].get('metadata')
    assert retrieved_metadata is not None
    assert retrieved_metadata['database']['connection']['pool']['size'] == 10
    assert retrieved_metadata['user']['preferences']['theme'] == 'dark'
    assert retrieved_metadata['tags'] == ['urgent', 'backend', 'production']


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
async def test_array_metadata() -> None:
    """Test that arrays can be stored in metadata."""
    array_metadata = {
        'tags': ['tag1', 'tag2', 'tag3'],
        'numbers': [1, 2, 3, 4, 5],
        'mixed': ['string', 42, math.pi, True, None],
        'nested_arrays': [[1, 2], [3, 4], [5, 6]],
    }

    result = await store_context(
        thread_id='test_array_metadata',
        source='agent',
        text='Testing array metadata',
        metadata=array_metadata,
    )

    assert result['success'] is True
    assert result['context_id'] > 0

    # Verify retrieval
    search_result = await search_context(
        thread_id='test_array_metadata',
        limit=1,
    )

    entries = search_result.get('results', [])
    assert len(entries) == 1
    retrieved_metadata = entries[0].get('metadata')
    assert retrieved_metadata is not None
    assert retrieved_metadata['tags'] == ['tag1', 'tag2', 'tag3']
    assert retrieved_metadata['numbers'] == [1, 2, 3, 4, 5]
    assert retrieved_metadata['nested_arrays'] == [[1, 2], [3, 4], [5, 6]]


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
async def test_deeply_nested_metadata() -> None:
    """Test that deeply nested structures (7 levels) can be stored."""
    deeply_nested = {
        'level1': {
            'level2': {
                'level3': {
                    'level4': {
                        'level5': {
                            'level6': {
                                'level7': {
                                    'value': 'deep',
                                    'number': 42,
                                    'list': [1, 2, 3],
                                },
                            },
                        },
                    },
                },
            },
        },
    }

    result = await store_context(
        thread_id='test_deep_nesting',
        source='agent',
        text='Testing deep nesting',
        metadata=deeply_nested,
    )

    assert result['success'] is True

    # Verify retrieval
    search_result = await search_context(
        thread_id='test_deep_nesting',
        limit=1,
    )

    entries = search_result.get('results', [])
    assert len(entries) == 1
    retrieved_metadata = entries[0].get('metadata')
    assert retrieved_metadata is not None
    assert retrieved_metadata['level1']['level2']['level3']['level4']['level5']['level6']['level7']['value'] == 'deep'


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
async def test_mixed_nested_structures() -> None:
    """Test mixed nested structures with objects and arrays."""
    mixed_metadata = {
        'config': {
            'database': {
                'hosts': ['host1', 'host2', 'host3'],
                'port': 5432,
                'options': {
                    'ssl': True,
                    'timeout': 30,
                    'retry_policy': {
                        'max_retries': 3,
                        'delays': [100, 200, 400],
                    },
                },
            },
            'cache': {
                'enabled': True,
                'ttl': 3600,
                'backends': ['redis', 'memcached'],
            },
        },
        'metrics': {
            'counters': {
                'requests': 1000,
                'errors': 5,
            },
            'timings': [10, 20, 15, 25, 18],
        },
    }

    result = await store_context(
        thread_id='test_mixed_structures',
        source='agent',
        text='Testing mixed nested structures',
        metadata=mixed_metadata,
    )

    assert result['success'] is True

    # Verify retrieval and structure preservation
    search_result = await search_context(
        thread_id='test_mixed_structures',
        limit=1,
    )

    entries = search_result.get('results', [])
    assert len(entries) == 1
    retrieved_metadata = entries[0].get('metadata')
    assert retrieved_metadata is not None
    assert retrieved_metadata['config']['database']['hosts'] == ['host1', 'host2', 'host3']
    assert retrieved_metadata['config']['cache']['backends'] == ['redis', 'memcached']
    assert retrieved_metadata['metrics']['timings'] == [10, 20, 15, 25, 18]


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
async def test_backward_compatibility_flat_metadata() -> None:
    """Test that flat metadata still works (backward compatibility)."""
    flat_metadata = {
        'status': 'active',
        'priority': 8,
        'completed': False,
        'agent_name': 'test-agent',
    }

    result = await store_context(
        thread_id='test_flat_metadata',
        source='agent',
        text='Testing flat metadata',
        metadata=flat_metadata,
    )

    assert result['success'] is True

    # Verify retrieval
    search_result = await search_context(
        thread_id='test_flat_metadata',
        limit=1,
    )

    entries = search_result.get('results', [])
    assert len(entries) == 1
    retrieved_metadata = entries[0].get('metadata')
    assert retrieved_metadata is not None
    assert retrieved_metadata['status'] == 'active'
    assert retrieved_metadata['priority'] == 8
    assert retrieved_metadata['completed'] is False
