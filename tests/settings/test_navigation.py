"""Tests for app/settings/navigation.py.

Covers the tri-state ENABLE_GREP_CONTEXT / ENABLE_CONTEXT_NAVIGATION /
ENABLE_CONTEXT_RANGE toggles (auto/true enable, false disables) and parsing of
the grep safety bounds.
"""

import pytest

from app.settings import get_settings


class TestGrepContextToggle:
    """ENABLE_GREP_CONTEXT is tri-state with an enabled property."""

    def test_default_is_auto_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ENABLE_GREP_CONTEXT', raising=False)
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.grep_context.mode == 'auto'
        assert settings.grep_context.enabled is True

    def test_true_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_GREP_CONTEXT', 'true')
        get_settings.cache_clear()
        assert get_settings().grep_context.enabled is True

    def test_false_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_GREP_CONTEXT', 'false')
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.grep_context.mode == 'false'
        assert settings.grep_context.enabled is False

    def test_boolean_spelling_one_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_GREP_CONTEXT', '1')
        get_settings.cache_clear()
        assert get_settings().grep_context.enabled is True

    def test_boolean_spelling_off_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_GREP_CONTEXT', 'off')
        get_settings.cache_clear()
        assert get_settings().grep_context.enabled is False


class TestGrepBounds:
    """The grep safety bounds parse from their env aliases with defaults."""

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            'GREP_MAX_MATCHES_CAP',
            'GREP_MAX_CONTEXT_LINES',
            'GREP_MAX_ENTRIES_SCANNED',
            'GREP_AGGREGATE_BYTES_BUDGET',
            'GREP_REGEX_TIMEOUT_S',
            'GREP_MAX_PATTERN_CHARS',
        ):
            monkeypatch.delenv(name, raising=False)
        get_settings.cache_clear()
        grep = get_settings().grep_context
        assert grep.max_matches_cap == 1000
        assert grep.max_context_lines == 20
        assert grep.max_entries_scanned == 1000
        assert grep.aggregate_bytes_budget == 67108864
        assert grep.regex_timeout_s == 5.0
        assert grep.max_pattern_chars == 32768

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('GREP_MAX_MATCHES_CAP', '250')
        monkeypatch.setenv('GREP_MAX_CONTEXT_LINES', '5')
        monkeypatch.setenv('GREP_MAX_ENTRIES_SCANNED', '2000')
        monkeypatch.setenv('GREP_REGEX_TIMEOUT_S', '1.5')
        monkeypatch.setenv('GREP_MAX_PATTERN_CHARS', '1024')
        get_settings.cache_clear()
        grep = get_settings().grep_context
        assert grep.max_matches_cap == 250
        assert grep.max_context_lines == 5
        assert grep.max_entries_scanned == 2000
        assert grep.regex_timeout_s == 1.5
        assert grep.max_pattern_chars == 1024


class TestContextRangeToggle:
    """ENABLE_CONTEXT_RANGE is tri-state with an enabled property."""

    def test_default_is_auto_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ENABLE_CONTEXT_RANGE', raising=False)
        get_settings.cache_clear()
        assert get_settings().context_range.enabled is True

    def test_false_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_RANGE', 'false')
        get_settings.cache_clear()
        assert get_settings().context_range.enabled is False


class TestContextNavigationToggle:
    """ENABLE_CONTEXT_NAVIGATION is tri-state with an enabled property."""

    def test_default_is_auto_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ENABLE_CONTEXT_NAVIGATION', raising=False)
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.context_navigation.mode == 'auto'
        assert settings.context_navigation.enabled is True

    def test_true_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_NAVIGATION', 'true')
        get_settings.cache_clear()
        assert get_settings().context_navigation.enabled is True

    def test_false_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_NAVIGATION', 'false')
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.context_navigation.mode == 'false'
        assert settings.context_navigation.enabled is False

    def test_boolean_spelling_no_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_NAVIGATION', 'no')
        get_settings.cache_clear()
        assert get_settings().context_navigation.enabled is False
