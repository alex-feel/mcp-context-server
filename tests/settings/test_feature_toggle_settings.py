"""Tests for the tri-state search feature toggles.

Covers the ``_normalize_feature_toggle`` helper, the shared
``FeatureToggleSettings`` base, and the three concrete search-toggle classes
(``SemanticSearchSettings``, ``FtsSettings``, ``HybridSearchSettings``): their
'auto'/'true'/'false' ``mode`` field, the derived read-only ``enabled``
property, their env aliases, and their composition on ``AppSettings``.

The toggles read their ``mode`` from environment-variable aliases, so the
parametrized cases drive values through ``monkeypatch.setenv`` rather than
constructor keywords. That keeps the tests type-safe (the constructors expose a
typed ``Literal['auto', 'true', 'false']`` signature) while exercising the real
alias-resolution path operators actually use.
"""

import pytest
from pydantic import ValidationError

from app.settings import AppSettings
from app.settings import FeatureToggleSettings
from app.settings import FtsSettings
from app.settings import HybridSearchSettings
from app.settings import SemanticSearchSettings
from app.settings import _normalize_feature_toggle


class TestNormalizeFeatureToggle:
    """Tests for the _normalize_feature_toggle module helper."""

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            # 'auto' family: literal token and empty string both map to 'auto'.
            ('auto', 'auto'),
            ('', 'auto'),
            ('AUTO', 'auto'),
            ('  auto  ', 'auto'),
            # 'true' family: assorted truthy spellings, case-insensitive, trimmed.
            ('true', 'true'),
            ('1', 'true'),
            ('yes', 'true'),
            ('on', 'true'),
            ('enabled', 'true'),
            ('TRUE', 'true'),
            (' True ', 'true'),
            # 'false' family: assorted falsy spellings, case-insensitive, trimmed.
            ('false', 'false'),
            ('0', 'false'),
            ('no', 'false'),
            ('off', 'false'),
            ('disabled', 'false'),
            ('False', 'false'),
            (' OFF ', 'false'),
        ],
    )
    def test_string_inputs_normalize(self, value: str, expected: str) -> None:
        """Recognized string spellings normalize to the tri-state tokens."""
        assert _normalize_feature_toggle(value) == expected

    def test_python_bool_true_maps_to_true(self) -> None:
        """A real Python bool True becomes the 'true' token."""
        assert _normalize_feature_toggle(True) == 'true'

    def test_python_bool_false_maps_to_false(self) -> None:
        """A real Python bool False becomes the 'false' token."""
        assert _normalize_feature_toggle(False) == 'false'

    def test_unknown_value_passes_through_lowercased(self) -> None:
        """An unrecognized value falls through lowercased for Literal rejection."""
        assert _normalize_feature_toggle('Maybe') == 'maybe'

    def test_unknown_value_is_trimmed(self) -> None:
        """Unrecognized values are still stripped before being returned."""
        assert _normalize_feature_toggle('  Bogus  ') == 'bogus'


# Each entry pairs a concrete toggle class with its environment-variable alias.
_TOGGLE_CLASSES = [
    (SemanticSearchSettings, 'ENABLE_SEMANTIC_SEARCH'),
    (FtsSettings, 'ENABLE_FTS'),
    (HybridSearchSettings, 'ENABLE_HYBRID_SEARCH'),
]


class TestFeatureToggleClasses:
    """Tests shared by all three concrete feature-toggle classes."""

    @pytest.mark.parametrize(('cls', 'alias'), _TOGGLE_CLASSES)
    def test_default_mode_is_auto(
        self,
        cls: type[SemanticSearchSettings | FtsSettings | HybridSearchSettings],
        alias: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default mode is 'auto' and the toggle reports enabled."""
        monkeypatch.delenv(alias, raising=False)
        settings = cls()
        assert settings.mode == 'auto'
        assert settings.enabled is True

    @pytest.mark.parametrize(('cls', 'alias'), _TOGGLE_CLASSES)
    def test_explicit_false_disables(
        self,
        cls: type[SemanticSearchSettings | FtsSettings | HybridSearchSettings],
        alias: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting the alias to 'false' yields mode='false', not enabled."""
        monkeypatch.setenv(alias, 'false')
        settings = cls()
        assert settings.mode == 'false'
        assert settings.enabled is False

    @pytest.mark.parametrize(('cls', 'alias'), _TOGGLE_CLASSES)
    def test_explicit_true_enables(
        self,
        cls: type[SemanticSearchSettings | FtsSettings | HybridSearchSettings],
        alias: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting the alias to 'true' yields mode='true' and reports enabled."""
        monkeypatch.setenv(alias, 'true')
        settings = cls()
        assert settings.mode == 'true'
        assert settings.enabled is True

    @pytest.mark.parametrize(('cls', 'alias'), _TOGGLE_CLASSES)
    def test_numeric_one_enables(
        self,
        cls: type[SemanticSearchSettings | FtsSettings | HybridSearchSettings],
        alias: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting the alias to '1' normalizes to mode='true' and reports enabled."""
        monkeypatch.setenv(alias, '1')
        settings = cls()
        assert settings.mode == 'true'
        assert settings.enabled is True

    @pytest.mark.parametrize(('cls', 'alias'), _TOGGLE_CLASSES)
    def test_invalid_value_raises(
        self,
        cls: type[SemanticSearchSettings | FtsSettings | HybridSearchSettings],
        alias: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An unrecognized value is rejected by Literal validation."""
        monkeypatch.setenv(alias, 'bogus')
        with pytest.raises(ValidationError):
            cls()


class TestEnabledProperty:
    """Tests for the derived FeatureToggleSettings.enabled property."""

    @pytest.mark.parametrize(
        ('env_value', 'expected_mode', 'expected_enabled'),
        [
            ('auto', 'auto', True),
            ('true', 'true', True),
            ('false', 'false', False),
        ],
    )
    def test_enabled_for_each_mode(
        self,
        env_value: str,
        expected_mode: str,
        expected_enabled: bool,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """enabled is True for 'auto' and 'true', False only for 'false'."""
        monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', env_value)
        settings = SemanticSearchSettings()
        assert settings.mode == expected_mode
        assert settings.enabled is expected_enabled

    def test_enabled_is_a_read_only_property(self) -> None:
        """enabled is a property descriptor with no setter (read-only)."""
        descriptor = FeatureToggleSettings.__dict__['enabled']
        assert isinstance(descriptor, property)
        assert descriptor.fset is None


class TestAppSettingsComposition:
    """Tests that the three toggles remain composed on AppSettings."""

    def test_defaults_all_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """AppSettings defaults give all three toggles mode='auto', enabled True."""
        monkeypatch.delenv('ENABLE_SEMANTIC_SEARCH', raising=False)
        monkeypatch.delenv('ENABLE_FTS', raising=False)
        monkeypatch.delenv('ENABLE_HYBRID_SEARCH', raising=False)
        settings = AppSettings()
        assert settings.semantic_search.mode == 'auto'
        assert settings.fts.mode == 'auto'
        assert settings.hybrid_search.mode == 'auto'
        assert settings.semantic_search.enabled is True
        assert settings.fts.enabled is True
        assert settings.hybrid_search.enabled is True

    def test_semantic_search_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_SEMANTIC_SEARCH flows through to the nested toggle."""
        monkeypatch.setenv('ENABLE_SEMANTIC_SEARCH', 'false')
        settings = AppSettings()
        assert settings.semantic_search.mode == 'false'
        assert settings.semantic_search.enabled is False

    def test_fts_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_FTS flows through to the nested toggle."""
        monkeypatch.setenv('ENABLE_FTS', 'true')
        settings = AppSettings()
        assert settings.fts.mode == 'true'
        assert settings.fts.enabled is True

    def test_hybrid_search_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """ENABLE_HYBRID_SEARCH flows through to the nested toggle."""
        monkeypatch.setenv('ENABLE_HYBRID_SEARCH', 'false')
        settings = AppSettings()
        assert settings.hybrid_search.mode == 'false'
        assert settings.hybrid_search.enabled is False

    def test_invalid_nested_value_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An invalid nested toggle value fails AppSettings construction."""
        monkeypatch.setenv('ENABLE_FTS', 'bogus')
        with pytest.raises(ValidationError):
            AppSettings()
