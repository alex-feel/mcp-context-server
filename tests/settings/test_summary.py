"""Tests for app/settings/summary.py.

Tests verify:
- SummarySettings field defaults, env var overrides, and validation ranges
- Field alias names match expected environment variable names
- Field constraints (ge, le, default) are correctly configured
- SUMMARY_MIN_CONTENT_LENGTH in SummarySettings
- The Ollama-specific summary settings
- SummarySettings integration with AppSettings
- IndexTreeNodeSummarySettings defaults, overrides, and bounds
"""


import pytest
from pydantic import ValidationError

from app.settings import AppSettings
from app.settings import get_settings
from app.settings.summary import SummarySettings
from tests.helpers import env_vars


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
        """Verify SUMMARY_MODEL defaults to 'qwen3:0.6b'."""
        settings = SummarySettings()
        assert settings.model == 'qwen3:0.6b'

    def test_summary_max_tokens_default_is_4000(self) -> None:
        """Verify SUMMARY_MAX_TOKENS defaults to 4000."""
        settings = SummarySettings()
        assert settings.max_tokens == 4000

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

    def test_summary_max_tokens_maximum_16384(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_TOKENS rejects values above 16384."""
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '16385')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_max_tokens_maximum_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_MAX_TOKENS accepts maximum value of 16384."""
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '16384')
        settings = SummarySettings()
        assert settings.max_tokens == 16384

    def test_summary_timeout_default_240(self) -> None:
        """Verify SUMMARY_TIMEOUT_S defaults to 240.0."""
        settings = SummarySettings()
        assert settings.timeout_s == 240.0

    def test_summary_timeout_zero_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_TIMEOUT_S rejects zero."""
        monkeypatch.setenv('SUMMARY_TIMEOUT_S', '0')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_timeout_exceeds_maximum_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_TIMEOUT_S rejects values above 600."""
        monkeypatch.setenv('SUMMARY_TIMEOUT_S', '601')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_summary_retry_max_attempts_default_5(self) -> None:
        """Verify SUMMARY_RETRY_MAX_ATTEMPTS defaults to 5."""
        settings = SummarySettings()
        assert settings.retry_max_attempts == 5

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

    def test_summary_retry_base_delay_default_3(self) -> None:
        """Verify SUMMARY_RETRY_BASE_DELAY_S defaults to 3.0."""
        settings = SummarySettings()
        assert settings.retry_base_delay_s == 3.0

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

    def test_summary_max_concurrent_default_2(self) -> None:
        """Verify SUMMARY_MAX_CONCURRENT defaults to 2."""
        settings = SummarySettings()
        assert settings.max_concurrent == 2

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

    def test_openai_reasoning_effort_default_low(self) -> None:
        """Verify SUMMARY_OPENAI_REASONING_EFFORT defaults to 'low'."""
        settings = SummarySettings()
        assert settings.openai_reasoning_effort == 'low'

    def test_anthropic_effort_default_none(self) -> None:
        """Verify SUMMARY_ANTHROPIC_EFFORT defaults to None."""
        settings = SummarySettings()
        assert settings.anthropic_effort is None

    def test_anthropic_effort_accepts_valid_values(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_ANTHROPIC_EFFORT accepts valid Literal values."""
        for value in ('max', 'high', 'medium', 'low'):
            monkeypatch.setenv('SUMMARY_ANTHROPIC_EFFORT', value)
            settings = SummarySettings()
            assert settings.anthropic_effort == value

    def test_anthropic_effort_rejects_invalid_value(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_ANTHROPIC_EFFORT rejects invalid values."""
        monkeypatch.setenv('SUMMARY_ANTHROPIC_EFFORT', 'invalid')
        with pytest.raises(ValidationError):
            SummarySettings()

    def test_openai_reasoning_effort_env_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify SUMMARY_OPENAI_REASONING_EFFORT can be overridden via env."""
        monkeypatch.setenv('SUMMARY_OPENAI_REASONING_EFFORT', 'high')
        settings = SummarySettings()
        assert settings.openai_reasoning_effort == 'high'

    def test_openai_reasoning_effort_empty_env_coerces_to_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty SUMMARY_OPENAI_REASONING_EFFORT is folded to None, not ''.

        Regression: the documented "set to empty to omit" idiom must yield None
        so the provider omits reasoning_effort entirely. Without coercion the
        empty string reaches ChatOpenAI as reasoning_effort='' (rejected by the
        OpenAI API).
        """
        monkeypatch.setenv('SUMMARY_OPENAI_REASONING_EFFORT', '')
        settings = SummarySettings()
        assert settings.openai_reasoning_effort is None

    def test_openai_reasoning_effort_whitespace_env_coerces_to_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A whitespace-only SUMMARY_OPENAI_REASONING_EFFORT is folded to None."""
        monkeypatch.setenv('SUMMARY_OPENAI_REASONING_EFFORT', '   ')
        settings = SummarySettings()
        assert settings.openai_reasoning_effort is None

    def test_anthropic_effort_empty_env_coerces_to_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty SUMMARY_ANTHROPIC_EFFORT is folded to None instead of failing.

        Symmetric with the OpenAI field: the empty-omit idiom must produce None
        rather than tripping the Anthropic Literal validation at startup.
        """
        monkeypatch.setenv('SUMMARY_ANTHROPIC_EFFORT', '')
        settings = SummarySettings()
        assert settings.anthropic_effort is None

    def test_summary_prompt_default_none(self) -> None:
        """Verify SUMMARY_PROMPT defaults to None."""
        settings = SummarySettings()
        assert settings.prompt is None

    def test_summary_settings_from_env_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify all settings can be overridden via environment variables."""
        monkeypatch.setenv('ENABLE_SUMMARY_GENERATION', 'false')
        monkeypatch.setenv('SUMMARY_PROVIDER', 'openai')
        monkeypatch.setenv('SUMMARY_MODEL', 'gpt-5.4-nano')
        monkeypatch.setenv('SUMMARY_MAX_TOKENS', '500')
        monkeypatch.setenv('SUMMARY_TIMEOUT_S', '90')
        monkeypatch.setenv('SUMMARY_RETRY_MAX_ATTEMPTS', '7')
        monkeypatch.setenv('SUMMARY_RETRY_BASE_DELAY_S', '2.0')
        monkeypatch.setenv('SUMMARY_MAX_CONCURRENT', '10')
        monkeypatch.setenv('SUMMARY_PROMPT', 'Custom prompt text')

        settings = SummarySettings()
        assert settings.generation_enabled is False
        assert settings.provider == 'openai'
        assert settings.model == 'gpt-5.4-nano'
        assert settings.max_tokens == 500
        assert settings.timeout_s == 90.0
        assert settings.retry_max_attempts == 7
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

    def test_min_content_length_default_is_500(self) -> None:
        """Verify SUMMARY_MIN_CONTENT_LENGTH defaults to 500."""
        settings = SummarySettings()
        assert settings.min_content_length == 500

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
        """Verify max_tokens field has default=4000, ge=50, le=16384."""
        field_info = SummarySettings.model_fields['max_tokens']
        assert field_info.default == 4000
        metadata = field_info.metadata
        ge_values = [m.ge for m in metadata if hasattr(m, 'ge')]
        le_values = [m.le for m in metadata if hasattr(m, 'le')]
        assert 50 in ge_values
        assert 16384 in le_values

    def test_openai_reasoning_effort_field_alias(self) -> None:
        """Verify openai_reasoning_effort field has alias 'SUMMARY_OPENAI_REASONING_EFFORT'."""
        field_info = SummarySettings.model_fields['openai_reasoning_effort']
        assert field_info.alias == 'SUMMARY_OPENAI_REASONING_EFFORT'

    def test_anthropic_effort_field_alias(self) -> None:
        """Verify anthropic_effort field has alias 'SUMMARY_ANTHROPIC_EFFORT'."""
        field_info = SummarySettings.model_fields['anthropic_effort']
        assert field_info.alias == 'SUMMARY_ANTHROPIC_EFFORT'

    def test_min_content_length_field_alias(self) -> None:
        """Verify min_content_length field has alias 'SUMMARY_MIN_CONTENT_LENGTH'."""
        field_info = SummarySettings.model_fields['min_content_length']
        assert field_info.alias == 'SUMMARY_MIN_CONTENT_LENGTH'

    def test_min_content_length_field_constraints(self) -> None:
        """Verify min_content_length field has default=500, ge=0, le=10000."""
        field_info = SummarySettings.model_fields['min_content_length']
        assert field_info.default == 500
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
        assert settings.summary.model == 'qwen3:0.6b'
        assert settings.summary.max_tokens == 4000
        assert settings.summary.timeout_s == 240.0
        assert settings.summary.retry_max_attempts == 5
        assert settings.summary.retry_base_delay_s == 3.0
        assert settings.summary.max_concurrent == 2
        assert settings.summary.prompt is None

    def test_summary_env_override_via_app_settings(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify env vars propagate through AppSettings.summary."""
        monkeypatch.setenv('SUMMARY_MODEL', 'qwen3:4b')
        monkeypatch.setenv('ENABLE_SUMMARY_GENERATION', 'false')
        settings = AppSettings()
        assert settings.summary.model == 'qwen3:4b'
        assert settings.summary.generation_enabled is False


class TestSummaryOllamaSettings:
    """Test SUMMARY_OLLAMA_TRUNCATE and SUMMARY_OLLAMA_NUM_CTX settings."""

    def test_summary_ollama_truncate_default_is_false(self) -> None:
        """Verify SUMMARY_OLLAMA_TRUNCATE defaults to false."""
        with env_vars(SUMMARY_OLLAMA_TRUNCATE=None):
            settings = AppSettings()
            assert settings.summary.ollama_truncate is False

    def test_summary_ollama_truncate_can_be_set_true(self) -> None:
        """Verify SUMMARY_OLLAMA_TRUNCATE can be explicitly set to true."""
        with env_vars(SUMMARY_OLLAMA_TRUNCATE='true'):
            settings = AppSettings()
            assert settings.summary.ollama_truncate is True

    def test_summary_ollama_truncate_can_be_set_false(self) -> None:
        """Verify SUMMARY_OLLAMA_TRUNCATE can be explicitly set to false."""
        with env_vars(SUMMARY_OLLAMA_TRUNCATE='false'):
            settings = AppSettings()
            assert settings.summary.ollama_truncate is False

    def test_summary_ollama_num_ctx_default_is_32768(self) -> None:
        """Verify SUMMARY_OLLAMA_NUM_CTX defaults to 32768."""
        with env_vars(SUMMARY_OLLAMA_NUM_CTX=None):
            settings = AppSettings()
            assert settings.summary.ollama_num_ctx == 32768

    def test_summary_ollama_num_ctx_can_be_customized(self) -> None:
        """Verify SUMMARY_OLLAMA_NUM_CTX can be set to custom value."""
        with env_vars(SUMMARY_OLLAMA_NUM_CTX='8192'):
            settings = AppSettings()
            assert settings.summary.ollama_num_ctx == 8192

    def test_summary_ollama_num_ctx_minimum_validation(self) -> None:
        """Verify SUMMARY_OLLAMA_NUM_CTX validates minimum value (512)."""
        with env_vars(SUMMARY_OLLAMA_NUM_CTX='100'), pytest.raises(ValidationError):
            AppSettings()

    def test_summary_ollama_num_ctx_maximum_validation(self) -> None:
        """Verify SUMMARY_OLLAMA_NUM_CTX validates maximum value (2097152)."""
        with env_vars(SUMMARY_OLLAMA_NUM_CTX='3000000'), pytest.raises(ValidationError):
            AppSettings()


class TestIndexTreeNodeSummarySettings:
    """Per-node index_tree summary settings parse with the documented defaults."""

    def test_node_summaries_default_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', raising=False)
        get_settings.cache_clear()
        assert get_settings().index_tree.node_summaries_enabled is True

    def test_node_summaries_can_disable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('ENABLE_INDEX_TREE_NODE_SUMMARIES', 'false')
        get_settings.cache_clear()
        assert get_settings().index_tree.node_summaries_enabled is False

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for name in (
            'INDEX_TREE_NODE_SUMMARY_PROMPT',
            'INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH',
            'INDEX_TREE_NODE_SUMMARY_TIMEOUT_S',
            'INDEX_TREE_NODE_SUMMARY_MAX_NODES',
            'INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S',
        ):
            monkeypatch.delenv(name, raising=False)
        get_settings.cache_clear()
        index_tree = get_settings().index_tree
        assert index_tree.prompt is None
        assert index_tree.min_content_length == 500
        assert index_tree.timeout_s == 240.0
        assert index_tree.max_concurrent >= 1
        # Total-work bounds: the concurrency caps limit how much runs at once,
        # these limit how much runs in total for one entry.
        assert index_tree.max_nodes == 200
        assert index_tree.total_timeout_s == 600.0

    def test_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH', '50')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_TIMEOUT_S', '12.5')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT', '7')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_MAX_NODES', '25')
        monkeypatch.setenv('INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S', '90')
        get_settings.cache_clear()
        index_tree = get_settings().index_tree
        assert index_tree.min_content_length == 50
        assert index_tree.timeout_s == 12.5
        assert index_tree.max_concurrent == 7
        assert index_tree.max_nodes == 25
        assert index_tree.total_timeout_s == 90.0

    @pytest.mark.parametrize(
        ('name', 'value'),
        [
            ('INDEX_TREE_NODE_SUMMARY_MAX_NODES', '0'),
            ('INDEX_TREE_NODE_SUMMARY_MAX_NODES', '10001'),
            ('INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S', '0'),
            ('INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S', '3601'),
        ],
    )
    def test_total_work_bounds_are_range_checked(
        self, monkeypatch: pytest.MonkeyPatch, name: str, value: str,
    ) -> None:
        """Out-of-range values are refused rather than silently disabling the bound."""
        from pydantic import ValidationError

        monkeypatch.setenv(name, value)
        get_settings.cache_clear()
        with pytest.raises(ValidationError):
            get_settings()
