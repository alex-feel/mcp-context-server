"""Tests for generate_index_nodes_with_timeout (the never-raise node-summary pass).

Verifies the additive contract: disabled -> None; no provider -> empty; sections
get summaries via summarize_with_prompt; a provider failure or timeout omits that
node WITHOUT raising (never aborts a store); short sections are skipped.
"""

from typing import Any
from typing import cast

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
    """node_layer_active() gates the text-change None->[] (clear-stale) remap.

    On a text-change update, a None node-generation result is ambiguous: it means
    TOTAL degradation when the layer is active (clear the stale rows) but "leave
    untouched" when the layer is inert. node_layer_active() resolves that, so the
    update paths only clear when the layer is genuinely active.
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
