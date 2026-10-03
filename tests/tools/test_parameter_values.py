"""Tests for parameter values at the core tools: source/limit/offset validation, tag normalization, metadata
serialization, Unicode, large and duplicate values, and tags with slashes and other special characters.
"""

from typing import Literal
from typing import cast

import pytest
from fastmcp.exceptions import ToolError

from app.tools.context.retrieve import get_context_by_ids
from app.tools.context.store import store_context
from app.tools.search.browse import search_context
from app.types import MetadataDict


@pytest.mark.usefixtures('initialized_server')
class TestParameterValidation:
    """Test that parameter validation rejects invalid values and accepts in-range ones."""

    @pytest.mark.asyncio
    async def test_invalid_source_type(self) -> None:
        """Test that Pydantic Literal handles invalid source and database CHECK constraint works.

        Note: Pydantic validates at FastMCP level. Using cast() bypasses it to test database.
        """
        with pytest.raises(ToolError, match='CHECK constraint failed|source'):
            await store_context(
                thread_id='invalid_source_test',
                source=cast(Literal['user', 'agent'], 'invalid'),
                text='This should fail',
            )

    @pytest.mark.asyncio
    async def test_limit_validation(self) -> None:
        """Test that Pydantic Field(ge=1, le=100) enforces limit range.

        Note: Pydantic validates at FastMCP level. This test verifies valid limits work.
        """
        # Valid limits work fine
        result = await search_context(limit=1)
        assert 'results' in result
        result = await search_context(limit=100)
        assert 'results' in result

    @pytest.mark.asyncio
    async def test_offset_validation(self) -> None:
        """Test that Pydantic Field(ge=0) enforces non-negative offset.

        Note: Pydantic validates at FastMCP level. This test verifies valid offsets work.
        """
        # Valid offsets work fine
        result = await search_context(limit=50, offset=0)
        assert 'results' in result
        result = await search_context(limit=50, offset=100)
        assert 'results' in result


@pytest.mark.usefixtures('initialized_server')
class TestParameterTypeCoercion:
    """Test that parameters handle type coercion correctly."""

    @pytest.mark.asyncio
    async def test_tags_normalization(self) -> None:
        """Test that tags are normalized to lowercase."""
        mixed_case_tags = ['Python', 'TESTING', 'MiXeD-CaSe']

        result = await store_context(
            thread_id='tags_normalization_test',
            source='user',
            text='Testing tag normalization',
            tags=mixed_case_tags,
        )

        assert result['success'] is True

        # Verify tags were normalized to lowercase
        search_result = await search_context(limit=50, thread_id='tags_normalization_test')
        normalized_tags = ['python', 'testing', 'mixed-case']
        assert set(search_result['results'][0]['tags']) == set(normalized_tags)

    @pytest.mark.asyncio
    async def test_metadata_json_serialization(self) -> None:
        """Test that metadata is properly JSON serialized/deserialized."""
        # Complex metadata with various types
        metadata: MetadataDict = {
            'string': 'value',
            'number': 123.45,
            'boolean': True,
            'null': None,
            'array': [1, 'two', None, {'nested': 'object'}],
            'object': {
                'deep': {
                    'nesting': {
                        'works': 'correctly',
                    },
                },
            },
        }

        result = await store_context(
            thread_id='metadata_serialization_test',
            source='agent',
            text='Testing metadata serialization',
            metadata=metadata,
        )

        assert result['success'] is True

        # Fetch and verify metadata round-trip
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(fetched[0])
        assert entry['metadata'] == metadata


@pytest.mark.usefixtures('initialized_server')
class TestEdgeCasesForParameters:
    """Test edge cases specific to parameter handling."""

    @pytest.mark.asyncio
    async def test_unicode_in_tags(self) -> None:
        """Test that Unicode characters in tags are handled correctly."""
        unicode_tags = ['python', '中文标签', 'عربي', 'тэг', '🏷️tag']

        result = await store_context(
            thread_id='unicode_tags_test',
            source='user',
            text='Testing Unicode tags',
            tags=unicode_tags,
        )

        assert result['success'] is True

        # Verify Unicode tags work in search
        search_results = await search_context(limit=50, tags=['中文标签'])
        found = [r for r in search_results['results'] if r['thread_id'] == 'unicode_tags_test']
        assert len(found) == 1

    @pytest.mark.asyncio
    async def test_large_list_of_tags(self) -> None:
        """Test handling of large number of tags."""
        # Create 100 unique tags
        large_tags_list = [f'tag_{i:03d}' for i in range(100)]

        result = await store_context(
            thread_id='large_tags_test',
            source='user',
            text='Testing large number of tags',
            tags=large_tags_list,
        )

        assert result['success'] is True

        # Verify all tags were stored
        search_result = await search_context(limit=50, thread_id='large_tags_test')
        assert len(search_result['results'][0]['tags']) == 100

    @pytest.mark.asyncio
    async def test_very_large_metadata(self) -> None:
        """Test handling of very large metadata dictionary."""
        # Create a large metadata structure
        large_metadata: MetadataDict = {
            f'key_{i}': {
                'data': 'x' * 1000,  # 1KB per entry
                'index': i,
                'nested': {
                    'level1': {
                        'level2': list(range(10)),
                    },
                },
            }
            for i in range(100)  # 100+ KB total
        }

        result = await store_context(
            thread_id='large_metadata_test',
            source='agent',
            text='Testing very large metadata',
            metadata=large_metadata,
        )

        assert result['success'] is True

        # Verify large metadata round-trips correctly
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(fetched[0])
        assert entry['metadata'] == large_metadata

    @pytest.mark.asyncio
    async def test_multiple_identical_tags(self) -> None:
        """A repeated tag is stored once, not once per occurrence.

        The stored list is compared AS A LIST: collapsing it into a set first would
        make the assertion hold for any number of duplicate rows, which is exactly the
        defect it is meant to catch -- every reader returns the rows verbatim, so a
        duplicate row shows up in every response and inflates the tag statistics.
        """
        duplicate_tags = ['python', 'python', 'test', 'test', 'python']

        result = await store_context(
            thread_id='duplicate_tags_test',
            source='user',
            text='Testing duplicate tags',
            tags=duplicate_tags,
        )

        assert result['success'] is True

        search_result = await search_context(limit=50, thread_id='duplicate_tags_test')
        assert sorted(search_result['results'][0]['tags']) == ['python', 'test']

    @pytest.mark.asyncio
    async def test_tags_differing_only_in_case_collapse_to_one(self) -> None:
        """Tags are lower-cased before storage, so case variants are ONE label.

        Normalization manufactures the collision -- ``Tag`` and ``tag`` are distinct on
        the wire and identical once folded -- so deduplication has to run after it, not
        on the caller's raw list.
        """
        result = await store_context(
            thread_id='case_collision_tags_test',
            source='user',
            text='Testing tag case collisions',
            tags=['Tag', 'tag', ' TAG '],
        )

        assert result['success'] is True

        search_result = await search_context(limit=50, thread_id='case_collision_tags_test')
        assert search_result['results'][0]['tags'] == ['tag']

    @pytest.mark.asyncio
    async def test_special_characters_in_metadata_keys(self) -> None:
        """Test metadata with special characters in keys."""
        special_metadata: MetadataDict = {
            'normal_key': 'value1',
            'key-with-dash': 'value2',
            'key.with.dots': 'value3',
            'key_with_underscore': 'value4',
            'key with spaces': 'value5',
            '123numeric': 'value6',
            'üñíçødé': 'value7',
        }

        result = await store_context(
            thread_id='special_metadata_test',
            source='user',
            text='Testing special metadata keys',
            metadata=special_metadata,
        )

        assert result['success'] is True

        # Verify special keys preserved
        fetched = await get_context_by_ids(context_ids=[result['context_id']])
        entry = dict(fetched[0])
        assert entry['metadata'] == special_metadata


@pytest.mark.usefixtures('initialized_server')
class TestForwardSlashAndSpecialCharacterTags:
    """Test forward slashes and special characters in tags."""

    @pytest.mark.asyncio
    async def test_forward_slash_tags(self) -> None:
        """Test that forward slashes in tags work correctly."""
        # Path-like tags separated by forward slashes
        forward_slash_tags = ['app/file', 'config/database', 'src/main.py']

        result = await store_context(
            thread_id='forward_slash_test',
            source='user',
            text='Testing forward slash tags',
            tags=forward_slash_tags,
        )

        assert result['success'] is True
        assert 'context_id' in result

        # Verify tags were stored correctly with forward slashes preserved
        search_result = await search_context(limit=50, thread_id='forward_slash_test')
        assert len(search_result['results']) == 1
        assert set(search_result['results'][0]['tags']) == set(forward_slash_tags)

        # Test searching by forward slash tags
        search_by_tag = await search_context(limit=50, tags=['app/file'])
        found = [r for r in search_by_tag['results'] if r['thread_id'] == 'forward_slash_test']
        assert len(found) == 1

    @pytest.mark.asyncio
    async def test_path_like_tags(self) -> None:
        """Test various path-like structures in tags."""
        path_tags = [
            'src/components/header.tsx',
            'lib/utils/helpers.py',
            'tests/unit/test_models.py',
            'docs/api/v2/endpoints',
            '/absolute/path/to/file',
            'relative/../path/to/file',
            'path\\with\\backslashes',  # Windows-style paths
        ]

        result = await store_context(
            thread_id='path_tags_test',
            source='agent',
            text='Testing path-like tags',
            tags=path_tags,
        )

        assert result['success'] is True

        # Verify all path tags were stored
        search_result = await search_context(limit=50, thread_id='path_tags_test')
        assert len(search_result['results']) == 1
        assert len(search_result['results'][0]['tags']) == len(path_tags)

    @pytest.mark.asyncio
    async def test_mixed_special_characters_in_tags(self) -> None:
        """Test various special characters in tags."""
        special_tags = [
            'feature/new-ui',  # forward slash with hyphen
            'bug#123',  # hash symbol
            'v1.2.3',  # periods
            'user@domain.com',  # at symbol
            'python:3.12',  # colon
            'high-priority!',  # exclamation
            'question?',  # question mark
            'task[urgent]',  # brackets
            'scope{global}',  # braces
            'item_with_underscore',  # underscore
            '100%complete',  # percent
            'a&b',  # ampersand
            'c++',  # plus signs
            'price=$99',  # dollar sign
            'temp~backup',  # tilde
            'item*wildcard',  # asterisk
        ]

        result = await store_context(
            thread_id='special_chars_test',
            source='user',
            text='Testing special characters in tags',
            tags=special_tags,
        )

        assert result['success'] is True

        # Verify all special character tags were stored (normalized to lowercase)
        search_result = await search_context(limit=50, thread_id='special_chars_test')
        assert len(search_result['results']) == 1
        assert len(search_result['results'][0]['tags']) == len(special_tags)
        # Check that lowercase normalization happened
        assert 'feature/new-ui' in search_result['results'][0]['tags']
        assert 'v1.2.3' in search_result['results'][0]['tags']

    @pytest.mark.asyncio
    async def test_empty_and_whitespace_tags_filtering(self) -> None:
        """Test that empty and whitespace-only tags are filtered out."""
        tags_with_empty = [
            'valid_tag',
            '',  # empty string
            '   ',  # spaces only
            '\t',  # tab only
            '\n',  # newline only
            '  \t\n  ',  # mixed whitespace
            'another_valid_tag',
            None,  # None should be handled gracefully if it somehow gets through
        ]

        # Filter out None for the actual call
        tags_to_send = [tag for tag in tags_with_empty if tag is not None]

        result = await store_context(
            thread_id='empty_tags_test',
            source='agent',
            text='Testing empty tag filtering',
            tags=tags_to_send,
        )

        assert result['success'] is True

        # Verify only valid tags were stored
        search_result = await search_context(limit=50, thread_id='empty_tags_test')
        assert len(search_result['results']) == 1
        assert set(search_result['results'][0]['tags']) == {'valid_tag', 'another_valid_tag'}

    @pytest.mark.asyncio
    async def test_tag_normalization_with_special_chars(self) -> None:
        """Test that tag normalization preserves special characters correctly."""
        # Tags with mixed case and special characters
        mixed_tags = [
            'Feature/NEW-UI',  # uppercase with slash and hyphen
            'CONFIG/Database',  # mixed case with slash
            'SRC/Main.py',  # mixed case with extension
            'Path\\TO\\File',  # backslashes with mixed case
            'User@DOMAIN.COM',  # email-like with uppercase
        ]

        result = await store_context(
            thread_id='normalization_special_test',
            source='user',
            text='Testing normalization with special chars',
            tags=mixed_tags,
        )

        assert result['success'] is True

        # Verify normalization to lowercase while preserving special chars
        search_result = await search_context(limit=50, thread_id='normalization_special_test')
        expected_tags = [
            'feature/new-ui',
            'config/database',
            'src/main.py',
            'path\\to\\file',
            'user@domain.com',
        ]
        assert set(search_result['results'][0]['tags']) == set(expected_tags)

    @pytest.mark.asyncio
    async def test_search_by_forward_slash_tags(self) -> None:
        """Test searching specifically using tags with forward slashes."""
        # Store multiple entries with different forward slash tags
        await store_context(
            thread_id='search_slash_test',
            source='user',
            text='Python file',
            tags=['src/python/main.py', 'type/script'],
        )
        await store_context(
            thread_id='search_slash_test',
            source='agent',
            text='Config file',
            tags=['config/settings.yaml', 'type/config'],
        )
        await store_context(
            thread_id='search_slash_test',
            source='user',
            text='Test file',
            tags=['tests/unit/test_main.py', 'type/test'],
        )

        # Search by forward slash tag
        results = await search_context(limit=50, tags=['src/python/main.py'])
        found = [r for r in results['results'] if r['thread_id'] == 'search_slash_test']
        assert len(found) == 1
        assert found[0]['text_content'] == 'Python file'

        # Search by multiple forward slash tags (OR logic)
        results = await search_context(limit=50, tags=['config/settings.yaml', 'tests/unit/test_main.py'])
        found = [r for r in results['results'] if r['thread_id'] == 'search_slash_test']
        assert len(found) == 2
        texts = {r['text_content'] for r in found}
        assert texts == {'Config file', 'Test file'}

    @pytest.mark.asyncio
    async def test_extreme_forward_slash_cases(self) -> None:
        """Test edge cases with forward slashes."""
        edge_case_tags = [
            '/',  # just a slash
            '//',  # double slash
            '///',  # triple slash
            '/leading/slash',  # leading slash
            'trailing/slash/',  # trailing slash
            '/both/slashes/',  # both ends
            'multiple//slashes///in////middle',  # multiple consecutive
            '.',  # just a period
            '..',  # double period
            '../../../relative',  # relative path
        ]

        result = await store_context(
            thread_id='extreme_slash_test',
            source='user',
            text='Testing extreme forward slash cases',
            tags=edge_case_tags,
        )

        assert result['success'] is True

        # Verify all edge cases were stored
        search_result = await search_context(limit=50, thread_id='extreme_slash_test')
        assert len(search_result['results']) == 1
        assert len(search_result['results'][0]['tags']) == len(edge_case_tags)
