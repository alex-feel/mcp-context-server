"""Tool-level tests for read_context_range (SQLite backend): char, line and node_id ranges, the mandatory
clamp and echo, and the locate-then-extract composition with grep_context offsets.
"""

import pytest
from fastmcp.exceptions import ToolError

from app.backends import StorageBackend
from app.tools.navigation import read_context_range
from tests.helpers import as_principal
from tests.tools._navigation import grep_as_dict
from tests.tools._navigation import navigate_as_dict
from tests.tools._navigation import store_entry


class TestReadContextRange:
    """Partial reads by char/line range with mandatory clamp + echo."""

    @pytest.mark.asyncio
    async def test_char_range(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'hello world')
        result = await read_context_range(context_id=cid, start_char=6, end_char=11)
        assert result['text'] == 'world'
        assert result['start_char'] == 6
        assert result['end_char'] == 11

    @pytest.mark.asyncio
    async def test_line_range(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'line1\nline2\nline3')
        result = await read_context_range(context_id=cid, start_line=2, end_line=2)
        assert result['text'] == 'line2'
        assert result['start_line'] == 2
        assert result['end_line'] == 2

    @pytest.mark.asyncio
    async def test_out_of_range_is_clamped_and_echoed(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'short')
        result = await read_context_range(context_id=cid, start_char=2, end_char=9999)
        assert result['text'] == 'ort'
        assert result['end_char'] == 5  # clamped to len('short')

    @pytest.mark.asyncio
    async def test_both_modes_rejected(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'data')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid, start_char=0, start_line=1)

    @pytest.mark.asyncio
    async def test_no_mode_rejected(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'data')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid)

    @pytest.mark.asyncio
    async def test_missing_entry_raises(self, nav_backend: StorageBackend) -> None:
        assert nav_backend is not None  # fixture provides the wired repositories
        with pytest.raises(ToolError):
            await read_context_range(context_id='0' * 32, start_char=0, end_char=1)


class TestReadContextRangeScoping:
    """read_context_range reads only entries the caller may read."""

    @pytest.mark.asyncio
    async def test_unreadable_entry_fails_like_a_missing_one(self, nav_backend: StorageBackend) -> None:
        """Another principal's private entry yields the same not-found error as an absent id."""
        hidden_id = await store_entry(nav_backend, 'alice private text', owner='alice')
        absent_id = '0' * 32

        with pytest.raises(ToolError) as hidden:
            await read_context_range(context_id=hidden_id, start_char=0, end_char=5)
        with pytest.raises(ToolError) as absent:
            await read_context_range(context_id=absent_id, start_char=0, end_char=5)

        assert str(hidden.value) == f'Context entry not found: {hidden_id}'
        assert str(absent.value) == f'Context entry not found: {absent_id}'

    @pytest.mark.asyncio
    async def test_owner_reads_their_private_entry(self, nav_backend: StorageBackend) -> None:
        """The owner of a private entry reads its range."""
        cid = await store_entry(nav_backend, 'alice private text', owner='alice')

        with as_principal('alice'):
            result = await read_context_range(context_id=cid, start_char=0, end_char=5)

        assert result['text'] == 'alice'

    @pytest.mark.asyncio
    async def test_prefix_of_an_unreadable_entry_matches_nothing(self, nav_backend: StorageBackend) -> None:
        """A prefix that matches only a hidden entry resolves to no entry at all."""
        hidden_id = await store_entry(nav_backend, 'alice private text', owner='alice')

        with pytest.raises(ToolError, match=f"No context entry matches prefix '{hidden_id[:12]}'"):
            await read_context_range(context_id=hidden_id[:12], start_char=0, end_char=5)


class TestLocateThenExtract:
    """grep content-mode offsets feed read_context_range to extract the hit."""

    @pytest.mark.asyncio
    async def test_grep_offsets_read_back_the_match(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'prefix text then TARGET then suffix')
        grep_result = await grep_as_dict(pattern='TARGET', thread_id='t', output_mode='content', case_sensitive=True)
        match = grep_result['results'][0]
        assert match['context_id'] == cid
        read_result = await read_context_range(
            context_id=match['context_id'],
            start_char=match['match_start'],
            end_char=match['match_end'],
        )
        assert read_result['text'] == 'TARGET'


class TestReadByNodeId:
    """read_context_range resolves a navigate_context node_id to its section."""

    @pytest.mark.asyncio
    async def test_navigate_then_read_node(self, nav_backend: StorageBackend) -> None:
        text = '# Intro\nintro body\n## Details\ndetail body here\n'
        cid = await store_entry(nav_backend, text)
        nav = await navigate_as_dict(context_id=cid)
        details = nav['root']['children'][0]['children'][0]
        assert details['node_id'] == 'intro/details'
        read_result = await read_context_range(context_id=cid, node_id='intro/details')
        # The Details section spans from its heading to end of document.
        assert read_result['text'].startswith('## Details')
        assert 'detail body here' in read_result['text']

    @pytest.mark.asyncio
    async def test_unknown_node_id_raises(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, '# Only\nbody')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid, node_id='nope/missing')

    @pytest.mark.asyncio
    async def test_node_id_and_char_range_mutually_exclusive(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, '# Only\nbody')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid, node_id='only', start_char=0)


class TestReadLineRangeEcho:
    """read_context_range echoes the resolved line range faithfully."""

    @pytest.mark.asyncio
    async def test_line_range_through_trailing_empty_line(self, nav_backend: StorageBackend) -> None:
        # Text ends with a newline, so the split yields a trailing empty line 3.
        # Requesting lines 1..3 must echo end_line=3 (not 2): recomputing the echo
        # from the exclusive end offset would map it back to the prior line.
        cid = await store_entry(nav_backend, 'a\nb\n')
        result = await read_context_range(context_id=cid, start_line=1, end_line=3)
        assert result['start_line'] == 1
        assert result['end_line'] == 3
        assert result['text'] == 'a\nb\n'

    @pytest.mark.asyncio
    async def test_line_range_clamps_beyond_eof(self, nav_backend: StorageBackend) -> None:
        cid = await store_entry(nav_backend, 'one\ntwo\nthree')
        result = await read_context_range(context_id=cid, start_line=2, end_line=99)
        assert result['start_line'] == 2
        assert result['end_line'] == 3
        assert result['text'] == 'two\nthree'
