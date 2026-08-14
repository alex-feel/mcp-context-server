"""Tests for generate_index_nodes_with_timeout (the never-raise node-summary pass).

Verifies the additive contract: disabled -> None; no provider -> empty; sections
get summaries via summarize_with_prompt; a provider failure or timeout omits that
node WITHOUT raising (never aborts a store); short sections are skipped.
"""

import asyncio
import threading
import time
from typing import Any
from typing import cast
from unittest.mock import patch

import pytest

import app.startup
import app.tools._shared as shared_module
from app.settings import get_settings
from app.tools._shared import generate_index_nodes_with_timeout


class _FakeProvider:
    """Minimal summary provider exposing summarize_with_prompt."""

    def __init__(self, *, fail: bool = False, value: str = 'a node summary') -> None:
        self._fail = fail
        self._value = value
        self.calls = 0

    async def summarize_with_prompt(self, text: str, system_prompt: str) -> str:
        _ = (text, system_prompt)  # signature parity with the provider protocol
        self.calls += 1
        if self._fail:
            raise RuntimeError('provider boom')
        return self._value


class _SlowProvider:
    """Summary provider whose calls take measurable time and track cancellation.

    ``active`` returns to zero only when every started call has settled, so a test
    can prove the pass awaited the node summaries it cancelled instead of leaving
    them running with a shared summary-model permit held.
    """

    def __init__(self, *, delay: float) -> None:
        self.delay = delay
        self.started = 0
        self.active = 0
        self.cancelled = 0

    async def summarize_with_prompt(self, text: str, system_prompt: str) -> str:
        _ = (text, system_prompt)  # signature parity with the provider protocol
        self.started += 1
        self.active += 1
        try:
            await asyncio.sleep(self.delay)
            return 'a node summary'
        except asyncio.CancelledError:
            self.cancelled += 1
            raise
        finally:
            self.active -= 1


def _set_provider(provider: _FakeProvider | _SlowProvider | None) -> None:
    app.startup.set_summary_provider(cast(Any, provider))


def _refresh_shared_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setattr(shared_module, 'settings', get_settings())


_TEXT = '# Section One\n' + ('alpha ' * 30) + '\n# Section Two\n' + ('beta ' * 30) + '\n'


class TestGenerateIndexNodes:
    @pytest.mark.asyncio
    async def test_disabled_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'false')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider())
        try:
            assert await generate_index_nodes_with_timeout(_TEXT) is None
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_no_provider_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        _refresh_shared_settings(monkeypatch)
        _set_provider(None)
        try:
            # No summary provider -> feature inert -> leave the node table untouched.
            assert await generate_index_nodes_with_timeout(_TEXT) is None
        finally:
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_generates_rows_for_sections(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider(value='gist'))
        try:
            rows = await generate_index_nodes_with_timeout(_TEXT)
            assert rows is not None
            assert {row.node_id for row in rows} == {'section-one', 'section-two'}
            assert all(row.node_summary == 'gist' for row in rows)
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_provider_failure_never_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider(fail=True))
        try:
            # Must NOT raise. TOTAL degradation (every attempted node failed)
            # returns None so callers PRESERVE existing stored rows rather than
            # wiping them on replace -- distinct from the legitimate-empty [] cases
            # (see test_short_sections_skipped, where nothing was attempted).
            assert await generate_index_nodes_with_timeout(_TEXT) is None
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_short_sections_skipped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '100000')
        _refresh_shared_settings(monkeypatch)
        provider = _FakeProvider()
        _set_provider(provider)
        try:
            assert await generate_index_nodes_with_timeout(_TEXT) == []
            assert provider.calls == 0
        finally:
            _set_provider(None)
            get_settings.cache_clear()


class TestNodeLayerActive:
    """node_layer_active() reports whether the per-node layer would ATTEMPT work
    (feature enabled AND a summary provider configured).

    It gates the store-path node attempt and the dedup ``nodes_pending`` pre-check. NOTE:
    it no longer gates the text-change clear-stale remap -- that remap is gated on
    settings.index_tree.node_summaries_enabled (the SAME gate navigate_context reads), so a
    text-change update clears stale rows even when the feature is on but the provider was
    removed (see test_update_context / test_batch_summary). The tests below pin
    node_layer_active()'s own definition.
    """

    def test_active_when_enabled_with_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider())
        try:
            assert shared_module.node_layer_active() is True
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    def test_inert_without_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        _refresh_shared_settings(monkeypatch)
        _set_provider(None)
        try:
            assert shared_module.node_layer_active() is False
        finally:
            get_settings.cache_clear()

    def test_inert_when_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'false')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider())
        try:
            assert shared_module.node_layer_active() is False
        finally:
            _set_provider(None)
            get_settings.cache_clear()


class TestLargeEntryWritePathOffloadNonBlocking:
    """The write-path index_tree outline parse offloads a large entry off the loop.

    ``generate_index_nodes_with_timeout`` parses the code-derived outline
    (``parse_outline``), which is O(text) pure CPU over UNBOUNDED stored entry
    text and runs on the store/update (and batch) write path. A large entry is
    offloaded to a worker thread so a multi-megabyte store cannot pin the single
    event loop and starve concurrent MCP requests; a small entry stays inline to
    avoid a per-call thread hop. This mirrors the read-path discipline
    (test_navigation_tools.py::TestLargeEntryOffloadNonBlocking) and the grep
    matcher (test_grep_matcher.py::test_large_literal_scan_is_offloaded_correct_and_non_blocking).
    """

    @pytest.mark.asyncio
    async def test_large_entry_parse_offloaded(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.services.outline_service import OutlineNode
        from app.services.outline_service import parse_outline as real_parse
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider())
        big = 'a' * (shared_module._OFFLOAD_MIN_CHARS + 10)  # exceeds the offload threshold
        seen: dict[str, bool] = {}

        def spy(text: str) -> OutlineNode:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_parse(text)

        try:
            with patch('app.tools._shared.parse_outline', spy):
                await generate_index_nodes_with_timeout(big)
            assert seen['on_main'] is False  # parsed on a worker thread, not the event loop
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_small_entry_parse_inline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from app.services.outline_service import OutlineNode
        from app.services.outline_service import parse_outline as real_parse
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider())
        seen: dict[str, bool] = {}

        def spy(text: str) -> OutlineNode:
            seen['on_main'] = threading.current_thread() is threading.main_thread()
            return real_parse(text)

        try:
            with patch('app.tools._shared.parse_outline', spy):
                await generate_index_nodes_with_timeout('# Intro\nbody\n')
            assert seen['on_main'] is True  # small entry stays inline (no thread hop)
        finally:
            _set_provider(None)
            get_settings.cache_clear()


class TestTotalWorkBounds:
    """Total node-summary work per store is bounded, not just its concurrency.

    The semaphores cap how many calls run at once and each call carries its own
    timeout, but neither bounds HOW MANY calls happen. A heading-dense entry
    therefore issued one model call per qualifying section and held the single
    summary model for that one store_context request for as long as the section
    count demanded, with no aggregate deadline to stop it.
    """

    @pytest.mark.asyncio
    async def test_node_count_is_capped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """More qualifying sections than the cap means exactly cap-many calls."""
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MAX_NODES', '3')
        _refresh_shared_settings(monkeypatch)
        provider = _FakeProvider(value='gist')
        _set_provider(provider)
        text = ''.join(f'# Section {i}\nbody {i}\n' for i in range(20))
        try:
            rows = await generate_index_nodes_with_timeout(text)
            assert rows is not None
            assert len(rows) == 3
            assert provider.calls == 3
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_cap_prefers_shallowest_then_longest(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the cap bites, the structurally significant sections survive."""
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MAX_NODES', '1')
        _refresh_shared_settings(monkeypatch)
        _set_provider(_FakeProvider(value='gist'))
        # One level-1 section containing a level-2 subsection: the parent is both
        # shallower and longer, so it is the one that keeps its summary.
        text = '# Parent\nparent body text\n## Child\nchild body\n'
        try:
            rows = await generate_index_nodes_with_timeout(text)
            assert rows is not None
            assert len(rows) == 1
            assert rows[0].level == 1
            assert rows[0].title == 'Parent'
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_ineligible_sections_never_become_tasks(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sections below the minimum length are filtered out before the fan-out.

        Materializing a task per heading only to return None immediately is pure
        overhead on a document with very many tiny headings.
        """
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '10000')
        _refresh_shared_settings(monkeypatch)
        provider = _FakeProvider(value='gist')
        _set_provider(provider)
        text = ''.join(f'# Section {i}\nbody\n' for i in range(50))
        try:
            rows = await generate_index_nodes_with_timeout(text)
            # Nothing qualified: an empty list legitimately clears stale rows.
            assert rows == []
            assert provider.calls == 0
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_aggregate_deadline_stops_the_pass(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An expired aggregate budget stops further chunks and keeps what was produced.

        The per-node timeout bounds ONE call; their sum was unbounded, so a
        pathological entry could stretch a single store indefinitely.
        """
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT', '4')
        _refresh_shared_settings(monkeypatch)
        provider = _FakeProvider(value='gist')
        _set_provider(provider)
        text = ''.join(f'# Section {i}\nbody {i}\n' for i in range(80))

        # Chunk size is max_concurrent * 4 == 16, so the first chunk completes and
        # the clock then jumps past the budget before the second chunk starts.
        ticks = iter([0.0, 0.0, 10_000.0])

        def fake_monotonic() -> float:
            try:
                return next(ticks)
            except StopIteration:
                return 10_000.0

        try:
            with patch('app.tools._shared.time.monotonic', fake_monotonic):
                rows = await generate_index_nodes_with_timeout(text)
            assert rows is not None
            assert len(rows) == 16
            assert provider.calls == 16
        finally:
            _set_provider(None)
            get_settings.cache_clear()

    @pytest.mark.asyncio
    async def test_aggregate_deadline_bounds_the_work_in_flight(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The budget bounds the running chunk, not just the gaps between chunks.

        Checking the clock only before dispatching each chunk bounds nothing: the
        chunk holds more sections than the shared summary-model budget lets run at
        once, so it serializes into several waves of per-node timeouts and overruns
        the aggregate budget by that multiple. Worse, an entry with no more sections
        than one chunk holds yields a single iteration whose only clock check
        happens before any work, where it can never be true -- leaving the budget
        inert for the ordinary case. The pass must stop at the budget either way.
        """
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT', '4')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_TIMEOUT_S', '30')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S', '0.4')
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '1')
        _refresh_shared_settings(monkeypatch)
        shared_module._reset_summary_model_semaphore()
        shared_module._reset_node_summary_semaphore()
        provider = _SlowProvider(delay=0.25)
        _set_provider(provider)
        # Six sections is a SINGLE chunk (chunk size is at least 16), so the whole
        # pass is one gather -- the exact shape a between-chunks check never reaches.
        # Serialized on one model permit they would need about 1.5s against a 0.4s
        # budget.
        text = ''.join(f'# Section {i}\nbody {i}\n' for i in range(6))
        try:
            started = time.monotonic()
            rows = await generate_index_nodes_with_timeout(text)
            elapsed = time.monotonic() - started

            assert elapsed < 1.0, f'aggregate budget not enforced: took {elapsed:.2f}s'
            assert rows is None or len(rows) < 6
            # Every started call has settled: the pass cancels AND awaits the
            # outstanding node summaries instead of orphaning them holding the
            # shared summary-model permit.
            assert provider.active == 0
            assert provider.cancelled >= 1
        finally:
            _set_provider(None)
            get_settings.cache_clear()
            shared_module._reset_summary_model_semaphore()
            shared_module._reset_node_summary_semaphore()

    @pytest.mark.asyncio
    async def test_rows_produced_before_the_deadline_are_kept(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An expired budget degrades the outline; it never aborts the store.

        The node leg is contractually never-raise, so the sections summarized before
        the budget expired must come back as rows rather than being discarded with
        the cancelled ones.
        """
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'true')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '0')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT', '4')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_TIMEOUT_S', '30')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S', '0.5')
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '1')
        _refresh_shared_settings(monkeypatch)
        shared_module._reset_summary_model_semaphore()
        shared_module._reset_node_summary_semaphore()
        provider = _SlowProvider(delay=0.1)
        _set_provider(provider)
        text = ''.join(f'# Section {i}\nbody {i}\n' for i in range(12))
        try:
            rows = await generate_index_nodes_with_timeout(text)
            assert rows is not None
            assert 1 <= len(rows) < 12
            assert provider.active == 0
        finally:
            _set_provider(None)
            get_settings.cache_clear()
            shared_module._reset_summary_model_semaphore()
            shared_module._reset_node_summary_semaphore()
