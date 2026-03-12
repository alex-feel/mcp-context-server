"""Tests for SummarySettings, field aliases, and summary prompt configuration.

Tests verify:
- SummarySettings field defaults, env var overrides, and validation ranges
- Field alias names match expected environment variable names
- Field constraints (ge, le, default) are correctly configured
- SUMMARY_MIN_CONTENT_LENGTH in SummarySettings
- DEFAULT_SUMMARY_PROMPT and resolve_summary_prompt() behavior
- SummarySettings integration with AppSettings
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from app.settings import AppSettings
from app.settings import SummarySettings
from app.summary.instructions import DEFAULT_SUMMARY_PROMPT
from app.summary.instructions import resolve_summary_prompt


class TestSummarySettings:
    """Tests for SummarySettings defaults and validation."""

    def test_summary_generation_enabled_by_default(self) -> None:
        """Verify ENABLE_SUMMARY_GENERATION defaults to True."""
        settings = SummarySettings()
        assert settings.generation_enabled is True

    def test_summary_provider_default_is_ollama(self) -> None:
        """Verify SUMMARY_PROVIDER defaults to 'ollama'."""
        settings = SummarySettings()
        assert settings.provider == 'ollama'

    def test_summary_model_default_is_qwen3_1_7b(self) -> None:
        """Verify SUMMARY_MODEL defaults to 'qwen3:1.7b'."""
        settings = SummarySettings()
        assert settings.model == 'qwen3:1.7b'

    def test_summary_max_tokens_default_is_2000(self) -> None:
        """Verify SUMMARY_MAX_TOKENS defaults to 2000."""
        settings = SummarySettings()
        assert settings.max_tokens == 2000

    def test_summary_max_tokens_minimum_50(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_TOKENS rejects values below 50."""
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '49')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_max_tokens_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_TOKENS accepts minimum value of 50."""
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '50')
        settings = SummarySettings()
        assert settings.max_tokens == 50

    def test_summary_max_tokens_maximum_5000(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_TOKENS rejects values above 5000."""
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '5001')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_max_tokens_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_TOKENS accepts maximum value of 5000."""
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '5000')
        settings = SummarySettings()
        assert settings.max_tokens == 5000

    def test_summary_timeout_default_30(self) -> None:
        """Verify SUMMARY_TIMEOUT_S defaults to 30.0."""
        settings = SummarySettings()
        assert settings.timeout_s == 30.0

    def test_summary_timeout_zero_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_TIMEOUT_S rejects zero."""
        monkeypatch.setenv('SUMMARY_TIMEOUT_S', '0')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_timeout_exceeds_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_TIMEOUT_S rejects values above 300."""
        monkeypatch.setenv('SUMMARY_TIMEOUT_S', '301')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_retry_max_attempts_default_3(self) -> None:
        """Verify SUMMARY_RETRY_MAX_ATTEMPTS defaults to 3."""
        settings = SummarySettings()
        assert settings.retry_max_attempts == 3

    def test_summary_retry_max_attempts_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_RETRY_MAX_ATTEMPTS accepts minimum value of 1."""
        monkeypatch.setenv('SUMMARY_RETRY_MAX_ATTEMPTS', '1')
        settings = SummarySettings()
        assert settings.retry_max_attempts == 1

    def test_summary_retry_max_attempts_below_minimum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_RETRY_MAX_ATTEMPTS rejects zero."""
        monkeypatch.setenv('SUMMARY_RETRY_MAX_ATTEMPTS', '0')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_retry_max_attempts_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_RETRY_MAX_ATTEMPTS rejects values above 10."""
        monkeypatch.setenv('SUMMARY_RETRY_MAX_ATTEMPTS', '11')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_retry_base_delay_default_1(self) -> None:
        """Verify SUMMARY_RETRY_BASE_DELAY_S defaults to 1.0."""
        settings = SummarySettings()
        assert settings.retry_base_delay_s == 1.0

    def test_summary_retry_base_delay_zero_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_RETRY_BASE_DELAY_S rejects zero."""
        monkeypatch.setenv('SUMMARY_RETRY_BASE_DELAY_S', '0')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_retry_base_delay_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_RETRY_BASE_DELAY_S rejects values above 30."""
        monkeypatch.setenv('SUMMARY_RETRY_BASE_DELAY_S', '31')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_max_concurrent_default_3(self) -> None:
        """Verify SUMMARY_MAX_CONCURRENT defaults to 3."""
        settings = SummarySettings()
        assert settings.max_concurrent == 3

    def test_summary_max_concurrent_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_CONCURRENT accepts minimum value of 1."""
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '1')
        settings = SummarySettings()
        assert settings.max_concurrent == 1

    def test_summary_max_concurrent_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_CONCURRENT accepts maximum value of 20."""
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '20')
        settings = SummarySettings()
        assert settings.max_concurrent == 20

    def test_summary_max_concurrent_below_minimum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_CONCURRENT rejects zero."""
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '0')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_max_concurrent_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_CONCURRENT rejects values above 20."""
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '21')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_prompt_default_none(self) -> None:
        """Verify SUMMARY_PROMPT defaults to None."""
        settings = SummarySettings()
        assert settings.prompt is None

    def test_summary_settings_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify all settings can be overridden via environment variables."""
        monkeypatch.setenv('ENABLE_SUMMARY_GENERATION', 'false')
        monkeypatch.setenv('SUMMARY_PROVIDER', 'openai')
        monkeypatch.setenv('SUMMARY_MODEL', 'gpt-5-nano')
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '500')
        monkeypatch.setenv('SUMMARY_TIMEOUT_S', '60')
        monkeypatch.setenv('SUMMARY_RETRY_MAX_ATTEMPTS', '5')
        monkeypatch.setenv('SUMMARY_RETRY_BASE_DELAY_S', '2.0')
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '10')
        monkeypatch.setenv('SUMMARY_PROMPT', 'Custom prompt text')

        settings = SummarySettings()
        assert settings.generation_enabled is False
        assert settings.provider == 'openai'
        assert settings.model == 'gpt-5-nano'
        assert settings.max_tokens == 500
        assert settings.timeout_s == 60.0
        assert settings.retry_max_attempts == 5
        assert settings.retry_base_delay_s == 2.0
        assert settings.max_concurrent == 10
        assert settings.prompt == 'Custom prompt text'

    def test_summary_provider_invalid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify invalid SUMMARY_PROVIDER raises validation error."""
        monkeypatch.setenv('SUMMARY_PROVIDER', 'invalid_provider')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_provider_anthropic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_PROVIDER accepts 'anthropic'."""
        monkeypatch.setenv('SUMMARY_PROVIDER', 'anthropic')
        settings = SummarySettings()
        assert settings.provider == 'anthropic'

    def test_min_content_length_default_is_300(self) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH defaults to 300."""
        settings = SummarySettings()
        assert settings.min_content_length == 300

    def test_min_content_length_minimum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH accepts 0 (ge=0)."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '0')
        settings = SummarySettings()
        assert settings.min_content_length == 0

    def test_min_content_length_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH accepts 10000 (le=10000)."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '10000')
        settings = SummarySettings()
        assert settings.min_content_length == 10000

    def test_min_content_length_below_minimum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH rejects -1."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '-1')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_min_content_length_above_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH rejects 10001."""
        monkeypatch.setenv('SUMMARY_MIN_CONTENT_LENGTH', '10001')
        with pytest.raises(ValidationError):
            SummarySettings()


class TestSummarySettingsFieldAliases:
    """Tests verifying field aliases match expected environment variable names."""

    def test_max_tokens_field_alias(self) -> None:
        """Verify max_tokens field has alias 'SUMMARY_MAX_TOKENS'."""
        field_info = SummarySettings.model_fields['max_tokens']
        assert field_info.alias == 'SUMMARY_MAX_TOKENS'

    def test_max_tokens_field_constraints(self) -> None:
        """Verify max_tokens field has default=2000, ge=50, le=5000."""
        field_info = SummarySettings.model_fields['max_tokens']
        assert field_info.default == 2000
        metadata = field_info.metadata
        ge_values = [m.ge for m in metadata if hasattr(m, 'ge')]
        le_values = [m.le for m in metadata if hasattr(m, 'le')]
        assert 50 in ge_values
        assert 5000 in le_values

    def test_min_content_length_field_alias(self) -> None:
        """Verify min_content_length field has alias 'SUMMARY_MIN_CONTENT_LENGTH'."""
        field_info = SummarySettings.model_fields['min_content_length']
        assert field_info.alias == 'SUMMARY_MIN_CONTENT_LENGTH'

    def test_min_content_length_field_constraints(self) -> None:
        """Verify min_content_length field has default=300, ge=0, le=10000."""
        field_info = SummarySettings.model_fields['min_content_length']
        assert field_info.default == 300
        metadata = field_info.metadata
        ge_values = [m.ge for m in metadata if hasattr(m, 'ge')]
        le_values = [m.le for m in metadata if hasattr(m, 'le')]
        assert 0 in ge_values
        assert 10000 in le_values

    def test_generation_enabled_field_alias(self) -> None:
        """Verify generation_enabled field has alias 'ENABLE_SUMMARY_GENERATION'."""
        field_info = SummarySettings.model_fields['generation_enabled']
        assert field_info.alias == 'ENABLE_SUMMARY_GENERATION'

    def test_provider_field_alias(self) -> None:
        """Verify provider field has alias 'SUMMARY_PROVIDER'."""
        field_info = SummarySettings.model_fields['provider']
        assert field_info.alias == 'SUMMARY_PROVIDER'

    def test_model_field_alias(self) -> None:
        """Verify model field has alias 'SUMMARY_MODEL'."""
        field_info = SummarySettings.model_fields['model']
        assert field_info.alias == 'SUMMARY_MODEL'

    def test_prompt_field_alias(self) -> None:
        """Verify prompt field has alias 'SUMMARY_PROMPT'."""
        field_info = SummarySettings.model_fields['prompt']
        assert field_info.alias == 'SUMMARY_PROMPT'


class TestSummaryPrompt:
    """Tests for DEFAULT_SUMMARY_PROMPT and resolve_summary_prompt()."""

    def test_default_prompt_exists_and_non_empty(self) -> None:
        """Verify DEFAULT_SUMMARY_PROMPT is defined and non-empty."""
        assert DEFAULT_SUMMARY_PROMPT
        assert len(DEFAULT_SUMMARY_PROMPT) > 100

    def test_default_prompt_contains_no_think(self) -> None:
        """Verify DEFAULT_SUMMARY_PROMPT starts with /no_think."""
        assert DEFAULT_SUMMARY_PROMPT.startswith('/no_think')

    def test_default_prompt_contains_key_constraints(self) -> None:
        """Verify prompt contains essential constraint phrases."""
        assert 'single' in DEFAULT_SUMMARY_PROMPT.lower()
        assert 'paragraph' in DEFAULT_SUMMARY_PROMPT.lower()
        assert 'do not add' in DEFAULT_SUMMARY_PROMPT.lower()
        assert 'Output ONLY' in DEFAULT_SUMMARY_PROMPT

    def test_resolve_returns_default_when_none(self) -> None:
        """Verify resolve_summary_prompt returns default when prompt is None."""
        settings = MagicMock()
        settings.prompt = None
        assert resolve_summary_prompt(settings) == DEFAULT_SUMMARY_PROMPT

    def test_resolve_returns_default_when_empty(self) -> None:
        """Verify resolve_summary_prompt returns default when prompt is empty string."""
        settings = MagicMock()
        settings.prompt = ''
        assert resolve_summary_prompt(settings) == DEFAULT_SUMMARY_PROMPT

    def test_resolve_returns_default_when_whitespace(self) -> None:
        """Verify resolve_summary_prompt returns default when prompt is whitespace."""
        settings = MagicMock()
        settings.prompt = '   '
        assert resolve_summary_prompt(settings) == DEFAULT_SUMMARY_PROMPT

    def test_resolve_returns_custom_when_set(self) -> None:
        """Verify resolve_summary_prompt returns custom prompt when set."""
        settings = MagicMock()
        settings.prompt = 'Custom prompt for testing'
        assert resolve_summary_prompt(settings) == 'Custom prompt for testing'

    def test_resolve_with_real_settings(self) -> None:
        """Verify resolve_summary_prompt works with real SummarySettings."""
        settings = SummarySettings()
        result = resolve_summary_prompt(settings)
        assert result == DEFAULT_SUMMARY_PROMPT


class TestAppSettingsSummaryIntegration:
    """Tests for SummarySettings integration with AppSettings."""

    def test_summary_settings_nested_in_app_settings(self) -> None:
        """Verify SummarySettings is accessible via AppSettings.summary."""
        settings = AppSettings()
        assert isinstance(settings.summary, SummarySettings)

    def test_summary_defaults_via_app_settings(self) -> None:
        """Verify default values are correct through AppSettings."""
        settings = AppSettings()
        assert settings.summary.generation_enabled is True
        assert settings.summary.provider == 'ollama'
        assert settings.summary.model == 'qwen3:1.7b'
        assert settings.summary.max_tokens == 2000
        assert settings.summary.timeout_s == 30.0
        assert settings.summary.retry_max_attempts == 3
        assert settings.summary.retry_base_delay_s == 1.0
        assert settings.summary.max_concurrent == 3
        assert settings.summary.prompt is None

    def test_summary_env_override_via_app_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify env vars propagate through AppSettings.summary."""
        monkeypatch.setenv('SUMMARY_MODEL', 'qwen3:4b')
        monkeypatch.setenv('ENABLE_SUMMARY_GENERATION', 'false')
        settings = AppSettings()
        assert settings.summary.model == 'qwen3:4b'
        assert settings.summary.generation_enabled is False
