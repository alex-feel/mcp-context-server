"""Nested JSON metadata round-trips through store_context and search_context and is filterable by nested path."""

import math

import pytest

from app.tools import search_context
from app.tools import store_context
from app.types import JsonValue


@pytest.mark.asyncio
@pytest.mark.usefixtures('initialized_server')
async def test_complex_nested_metadata() -> None:
    """Test that complex nested JSON structures can be stored in metadata."""
    complex_metadata: dict[str, JsonValue] = {
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
    assert len(result['context_id']) == 32

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
    array_metadata: dict[str, JsonValue] = {
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
    assert len(result['context_id']) == 32

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
    deeply_nested: dict[str, JsonValue] = {
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
    mixed_metadata: dict[str, JsonValue] = {
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
    """Test that flat (non-nested) metadata round-trips unchanged."""
    flat_metadata: dict[str, JsonValue] = {
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


@pytest.mark.integration
@pytest.mark.usefixtures('initialized_server')
class TestNestedJSONMetadata:
    """Test nested JSON structures in metadata."""

    @pytest.mark.asyncio
    async def test_store_nested_objects(self) -> None:
        """Test storing nested JSON objects in metadata."""
        complex_metadata: dict[str, JsonValue] = {
            'status': 'active',
            'config': {
                'database': {
                    'connection': {
                        'pool': {'size': 10, 'timeout': 30},
                        'retry': {'max_attempts': 3, 'backoff': 2.5},
                    },
                },
                'cache': {'enabled': True, 'ttl': 300},
            },
            'user': {'id': 123, 'name': 'Alice Johnson', 'preferences': {'theme': 'dark', 'language': 'en'}},
        }

        result = await store_context(
            thread_id='test_nested_json',
            source='agent',
            text='Test nested metadata storage',
            metadata=complex_metadata,
        )

        assert result['success'] is True
        assert 'context_id' in result

        # Retrieve and verify the metadata is preserved
        search_result = await search_context(limit=50, thread_id='test_nested_json')
        assert len(search_result['results']) == 1

        stored_metadata = search_result['results'][0]['metadata']
        assert stored_metadata['status'] == 'active'
        assert stored_metadata['config']['database']['connection']['pool']['size'] == 10
        assert stored_metadata['config']['database']['connection']['pool']['timeout'] == 30
        assert stored_metadata['config']['database']['connection']['retry']['max_attempts'] == 3
        assert stored_metadata['config']['database']['connection']['retry']['backoff'] == 2.5
        assert stored_metadata['config']['cache']['enabled'] is True
        assert stored_metadata['user']['preferences']['theme'] == 'dark'
        assert stored_metadata['user']['preferences']['language'] == 'en'

    @pytest.mark.asyncio
    async def test_store_arrays_in_metadata(self) -> None:
        """Test storing arrays in metadata."""
        metadata_with_arrays: dict[str, JsonValue] = {
            'tags': ['urgent', 'backend', 'production'],
            'priority_levels': [1, 2, 3, 4, 5],
            'mixed_array': ['string', 42, math.pi, True, None],
            'nested_arrays': [[1, 2], [3, 4], [5, 6]],
        }

        result = await store_context(
            thread_id='test_arrays',
            source='agent',
            text='Test array metadata',
            metadata=metadata_with_arrays,
        )

        assert result['success'] is True

        # Retrieve and verify arrays are preserved
        search_result = await search_context(limit=50, thread_id='test_arrays')
        stored_metadata = search_result['results'][0]['metadata']

        assert stored_metadata['tags'] == ['urgent', 'backend', 'production']
        assert stored_metadata['priority_levels'] == [1, 2, 3, 4, 5]
        assert stored_metadata['mixed_array'] == ['string', 42, math.pi, True, None]
        assert stored_metadata['nested_arrays'] == [[1, 2], [3, 4], [5, 6]]

    @pytest.mark.asyncio
    async def test_query_nested_paths(self) -> None:
        """Test querying nested JSON paths."""
        # Store multiple entries with nested metadata
        await store_context(
            thread_id='test_nested_paths',
            source='agent',
            text='Entry 1',
            metadata={'user': {'preferences': {'theme': 'dark', 'notifications': {'email': True}}}},
        )

        await store_context(
            thread_id='test_nested_paths',
            source='agent',
            text='Entry 2',
            metadata={'user': {'preferences': {'theme': 'light', 'notifications': {'email': False}}}},
        )

        # Query using nested path
        result = await search_context(
            limit=50,
            thread_id='test_nested_paths',
            metadata={'user.preferences.theme': 'dark'},
        )

        assert len(result['results']) == 1
        assert result['results'][0]['text_content'] == 'Entry 1'
        assert result['results'][0]['metadata']['user']['preferences']['theme'] == 'dark'

    @pytest.mark.asyncio
    async def test_complex_nested_structure(self) -> None:
        """Test very complex nested structure with multiple levels."""
        complex_structure: dict[str, JsonValue] = {
            'level1': {
                'level2': {
                    'level3': {
                        'level4': {
                            'value': 'deeply_nested',
                            'number': 42,
                            'array': [1, 2, 3],
                            'object': {'key': 'value'},
                        },
                    },
                },
            },
            'metrics': {
                'cpu': 45.5,
                'memory': 512,
                'disk': {'used': 80.5, 'total': 100.0, 'partitions': ['/dev/sda1', '/dev/sda2']},
            },
            'features': {
                'enabled': ['feature_a', 'feature_b', 'feature_c'],
                'disabled': [],
                'experimental': {'count': 3, 'names': ['exp_1', 'exp_2', 'exp_3']},
            },
        }

        result = await store_context(
            thread_id='test_complex',
            source='agent',
            text='Complex nested structure test',
            metadata=complex_structure,
        )

        assert result['success'] is True

        # Verify structure is preserved
        search_result = await search_context(limit=50, thread_id='test_complex')
        stored_metadata = search_result['results'][0]['metadata']

        # Verify deep nesting
        assert stored_metadata['level1']['level2']['level3']['level4']['value'] == 'deeply_nested'
        assert stored_metadata['level1']['level2']['level3']['level4']['number'] == 42
        assert stored_metadata['level1']['level2']['level3']['level4']['array'] == [1, 2, 3]
        assert stored_metadata['level1']['level2']['level3']['level4']['object']['key'] == 'value'

        # Verify metrics
        assert stored_metadata['metrics']['cpu'] == 45.5
        assert stored_metadata['metrics']['disk']['used'] == 80.5
        assert stored_metadata['metrics']['disk']['partitions'] == ['/dev/sda1', '/dev/sda2']

        # Verify features
        assert stored_metadata['features']['enabled'] == ['feature_a', 'feature_b', 'feature_c']
        assert stored_metadata['features']['disabled'] == []
        assert stored_metadata['features']['experimental']['count'] == 3

    @pytest.mark.asyncio
    async def test_mixed_flat_and_nested(self) -> None:
        """Test mixing flat and nested metadata structures."""
        mixed_metadata: dict[str, JsonValue] = {
            'simple_string': 'value',
            'simple_int': 42,
            'simple_bool': True,
            'nested': {'level1': {'level2': 'deep_value'}},
            'array': [1, 2, 3],
        }

        result = await store_context(
            thread_id='test_mixed',
            source='agent',
            text='Mixed flat and nested',
            metadata=mixed_metadata,
        )

        assert result['success'] is True

        # Query using both flat and nested paths
        search_result = await search_context(
            limit=50, thread_id='test_mixed', metadata={'simple_string': 'value'},
        )
        assert len(search_result['results']) == 1

        # Verify all types are preserved
        stored_metadata = search_result['results'][0]['metadata']
        assert stored_metadata['simple_string'] == 'value'
        assert stored_metadata['simple_int'] == 42
        assert stored_metadata['simple_bool'] is True
        assert stored_metadata['nested']['level1']['level2'] == 'deep_value'
        assert stored_metadata['array'] == [1, 2, 3]
