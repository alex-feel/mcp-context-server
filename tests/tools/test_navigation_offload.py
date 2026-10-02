"""Tool-level tests for the worker-thread offload of outline and line parsing in navigate_context and
read_context_range.
"""

import threading
from unittest.mock import patch

import pytest

from app.backends import StorageBackend
from app.services.text_lines import _OFFLOAD_MIN_CHARS
from app.tools.navigation import navigate_context
from app.tools.navigation import read_context_range
from tests.tools._navigation import navigate_as_dict
from tests.tools._navigation import store_entry


class TestLargeEntryOffloadNonBlocking:
    """A large entry's O(text) outline/line parse runs OFF the event loop (in a
    worker thread) so a multi-megabyte navigate/read cannot pin the single event
    loop; a small entry stays inline to avoid a thread hop. Mirrors the grep
    matcher's _OFFLOAD_MIN_CHARS discipline
    (test_grep_matcher.py::test_large_literal_scan_is_offloaded_correct_and_non_blocking).
    """

    @pytest.mark.asyncio
    async def test_navigate_offloads_large_entry_parse(self, nav_backend: StorageBackend) -> None:
        from app.services.outline_service import OutlineNode
        from app.services.outline_service import parse_outline as real_parse
        big = 'a' * (_OFFLOAD_MIN_CHARS + 10)  # exceeds the offload threshold
        cid = await store_entry(nav_backend, big)
        seen: dict[str, bool] = {}

        def spy(text: str, max_depth: int = 6) -> OutlineNode:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_parse(text, max_depth=max_depth)

        with patch('app.tools.navigation.parse_outline', spy):
            result = await navigate_as_dict(context_id=cid)
        assert seen['on_main'] is False  # parsed on a worker thread, not the event loop
        assert result['total_chars'] == len(big)

    @pytest.mark.asyncio
    async def test_navigate_small_entry_runs_inline(self, nav_backend: StorageBackend) -> None:
        from app.services.outline_service import OutlineNode
        from app.services.outline_service import parse_outline as real_parse
        cid = await store_entry(nav_backend, '# Intro\nbody\n')
        seen: dict[str, bool] = {}

        def spy(text: str, max_depth: int = 6) -> OutlineNode:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_parse(text, max_depth=max_depth)

        with patch('app.tools.navigation.parse_outline', spy):
            await navigate_context(context_id=cid)
        assert seen['on_main'] is True  # small entry stays inline (no thread hop)

    @pytest.mark.asyncio
    async def test_read_range_offloads_large_entry_line_split(self, nav_backend: StorageBackend) -> None:
        from app.services.text_lines import split_lines_with_offsets as real_split
        big = 'a' * (_OFFLOAD_MIN_CHARS + 10)
        cid = await store_entry(nav_backend, big)
        seen: dict[str, bool] = {}

        def spy(text: str) -> tuple[list[str], list[int]]:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_split(text)

        with patch('app.tools.navigation.split_lines_with_offsets', spy):
            result = await read_context_range(context_id=cid, start_char=0, end_char=5)
        assert seen['on_main'] is False  # line split offloaded to a worker thread
        assert result['text'] == 'aaaaa'

    @pytest.mark.asyncio
    async def test_read_range_offloads_large_entry_node_resolution(self, nav_backend: StorageBackend) -> None:
        from app.services.outline_service import resolve_node_span as real_resolve
        big = '# Title\n' + 'a' * (_OFFLOAD_MIN_CHARS + 10)
        cid = await store_entry(nav_backend, big)
        seen: dict[str, bool] = {}

        def spy(text: str, node_id: str) -> tuple[int, int] | None:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_resolve(text, node_id)

        with patch('app.tools.navigation.resolve_node_span', spy):
            result = await read_context_range(context_id=cid, node_id='root')
        assert seen['on_main'] is False  # node-span re-parse offloaded to a worker thread
        assert result['text']  # root span returned


class TestOutlineOffloadKeysOnLineDensity:
    """The outline offload predicate must track PARSE COST, not input size.

    parse_outline walks the text line by line running several regexes per line,
    so its cost tracks LINE COUNT: a million characters on one line parses in
    about two milliseconds, while the same million characters split into short
    heading lines takes about two seconds. A size-only threshold is blind to that
    spread: a dense entry well under the size threshold would parse inline and
    pin the event loop for seconds on every navigate / read_context_range
    call -- even one returning a five-character slice.
    """

    def test_dense_sub_threshold_text_is_offloaded(self) -> None:
        """A heading-dense entry far below the size threshold still offloads."""
        from app.services.text_lines import should_offload_line_scan

        dense = '# h\n' * 5_000  # 20k characters, 5k lines
        assert len(dense) < _OFFLOAD_MIN_CHARS
        assert should_offload_line_scan(dense) is True

    def test_single_long_line_stays_inline(self) -> None:
        """A huge single line is cheap to parse and must not pay a thread hop."""
        from app.services.text_lines import should_offload_line_scan

        assert should_offload_line_scan('x' * (_OFFLOAD_MIN_CHARS - 1)) is False

    def test_oversized_text_is_offloaded_regardless_of_density(self) -> None:
        """The original size signal still applies on its own."""
        from app.services.text_lines import should_offload_line_scan

        assert should_offload_line_scan('x' * (_OFFLOAD_MIN_CHARS + 1)) is True

    @pytest.mark.asyncio
    async def test_navigate_offloads_dense_sub_threshold_entry(
        self, nav_backend: StorageBackend,
    ) -> None:
        """navigate_context offloads the parse for a dense, sub-threshold entry."""
        from app.services.outline_service import OutlineNode
        from app.services.outline_service import parse_outline as real_parse

        dense = '# h\n' * 5_000
        assert len(dense) < _OFFLOAD_MIN_CHARS
        cid = await store_entry(nav_backend, dense)
        seen: dict[str, bool] = {}

        def spy(text: str, max_depth: int = 6) -> OutlineNode:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_parse(text, max_depth=max_depth)

        with patch('app.tools.navigation.parse_outline', spy):
            await navigate_context(context_id=cid)
        assert seen['on_main'] is False

    @pytest.mark.asyncio
    async def test_read_node_range_offloads_dense_sub_threshold_entry(
        self, nav_backend: StorageBackend,
    ) -> None:
        """read_context_range(node_id=...) offloads the re-parse for a dense entry."""
        from app.services.outline_service import resolve_node_span as real_resolve

        dense = '# h\n' * 5_000
        cid = await store_entry(nav_backend, dense)
        seen: dict[str, bool] = {}

        def spy(text: str, node_id: str) -> tuple[int, int] | None:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_resolve(text, node_id)

        with patch('app.tools.navigation.resolve_node_span', spy):
            result = await read_context_range(context_id=cid, node_id='root')
        assert seen['on_main'] is False
        assert result['text']

    @pytest.mark.asyncio
    async def test_read_range_offloads_dense_sub_threshold_line_split(
        self, nav_backend: StorageBackend,
    ) -> None:
        """The line split is line-count-driven too, so it needs the same predicate.

        Every read_context_range call splits lines, in every addressing mode, and one
        slice plus one list append per line makes a dense sub-threshold entry cost a
        tenth of a second inline -- on each call, even one returning five characters.
        """
        from app.services.text_lines import split_lines_with_offsets as real_split

        dense = '# h\n' * 5_000
        assert len(dense) < _OFFLOAD_MIN_CHARS
        cid = await store_entry(nav_backend, dense)
        seen: dict[str, bool] = {}

        def spy(text: str) -> tuple[list[str], list[int]]:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_split(text)

        with patch('app.tools.navigation.split_lines_with_offsets', spy):
            result = await read_context_range(context_id=cid, start_char=0, end_char=4)
        assert seen['on_main'] is False
        assert result['text'] == '# h\n'

    @pytest.mark.asyncio
    async def test_read_range_single_long_line_stays_inline(
        self, nav_backend: StorageBackend,
    ) -> None:
        """One huge line splits into one entry, so it must not pay a thread hop."""
        from app.services.text_lines import split_lines_with_offsets as real_split

        single_line = 'a' * 50_000
        cid = await store_entry(nav_backend, single_line)
        seen: dict[str, bool] = {}

        def spy(text: str) -> tuple[list[str], list[int]]:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_split(text)

        with patch('app.tools.navigation.split_lines_with_offsets', spy):
            result = await read_context_range(context_id=cid, start_char=0, end_char=5)
        assert seen['on_main'] is True
        assert result['text'] == 'aaaaa'
