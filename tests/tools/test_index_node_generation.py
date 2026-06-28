"""Tests for generate_index_nodes_with_timeout (the never-raise node-summary pass).

Verifies the additive contract: disabled -> None; no provider -> empty; sections
get summaries via summarize_with_prompt; a provider failure or timeout omits that
node WITHOUT raising (never aborts a store); short sections are skipped.
"""

import threading
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


def _set_provider(provider: _FakeProvider | None) -> None:
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
