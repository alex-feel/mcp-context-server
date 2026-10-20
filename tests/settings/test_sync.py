"""Tests for app/settings/sync.py.

Covers the tri-state ENABLE_CONTEXT_SYNC toggle that registers plan_context_sync and
list_ingest_state: auto (the default) and true enable, false disables, and the
boolean spellings coerce to those modes.
"""

import pytest

from app.settings import get_settings


class TestContextSyncToggle:
    """ENABLE_CONTEXT_SYNC is tri-state with an enabled property."""

    def test_default_is_auto_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ENABLE_CONTEXT_SYNC', raising=False)
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.context_sync.mode == 'auto'
        assert settings.context_sync.enabled is True

    def test_true_enables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_SYNC', 'true')
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.context_sync.mode == 'true'
        assert settings.context_sync.enabled is True

    def test_false_disables(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_SYNC', 'false')
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.context_sync.mode == 'false'
        assert settings.context_sync.enabled is False

    @pytest.mark.parametrize(('spelling', 'mode'), [('1', 'true'), ('yes', 'true'), ('on', 'true'), ('0', 'false'),
                                                   ('no', 'false'), ('off', 'false')])
    def test_boolean_spellings_coerce(self, monkeypatch: pytest.MonkeyPatch, spelling: str, mode: str) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_SYNC', spelling)
        get_settings.cache_clear()
        settings = get_settings()
        assert settings.context_sync.mode == mode
        assert settings.context_sync.enabled is (mode == 'true')

    def test_unknown_value_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_CONTEXT_SYNC', 'sometimes')
        get_settings.cache_clear()
        with pytest.raises(ValueError, match='ENABLE_CONTEXT_SYNC'):
            get_settings()
