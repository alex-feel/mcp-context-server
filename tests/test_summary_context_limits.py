"""Tests for summary context limits module.

Tests verify:
- SummaryModelSpec dataclass creation
- get_summary_model_spec lookup for known and unknown models
- get_summary_provider_default_context for known and unknown providers
- Registry consistency (all entries have required fields)
"""

from __future__ import annotations

import pytest

from app.summary.context_limits import SUMMARY_MODEL_SPECS
from app.summary.context_limits import SummaryModelSpec
from app.summary.context_limits import get_summary_model_spec
from app.summary.context_limits import get_summary_provider_default_context


class TestSummaryModelSpec:
    """Tests for SummaryModelSpec dataclass."""

    def test_create_spec(self) -> None:
        """SummaryModelSpec can be created with all fields."""
        spec = SummaryModelSpec(
            provider='ollama',
            model='test-model',
            max_input_tokens=4096,
            truncation_behavior='configurable',
            notes='Test note',
        )
        assert spec.provider == 'ollama'
        assert spec.model == 'test-model'
        assert spec.max_input_tokens == 4096
        assert spec.truncation_behavior == 'configurable'
        assert spec.notes == 'Test note'

    def test_create_spec_without_notes(self) -> None:
        """SummaryModelSpec notes defaults to empty string."""
        spec = SummaryModelSpec(
            provider='openai',
            model='gpt-5',
            max_input_tokens=400000,
            truncation_behavior='error',
        )
        assert spec.notes == ''

    def test_spec_is_frozen(self) -> None:
        """SummaryModelSpec is immutable (frozen dataclass)."""
        spec = SummaryModelSpec(
            provider='ollama',
            model='test',
            max_input_tokens=4096,
            truncation_behavior='configurable',
        )
        with pytest.raises((AttributeError, TypeError)):
            exec('spec.max_input_tokens = 8192', {'spec': spec})


class TestGetSummaryModelSpec:
    """Tests for get_summary_model_spec function."""

    def test_returns_spec_for_known_model(self) -> None:
        """Returns SummaryModelSpec for a known model."""
        spec = get_summary_model_spec('qwen3:0.6b')
        assert spec is not None
        assert spec.provider == 'ollama'
        assert spec.max_input_tokens == 32768

    def test_returns_none_for_unknown_model(self) -> None:
        """Returns None for an unknown model."""
        spec = get_summary_model_spec('nonexistent-model')
        assert spec is None

    def test_returns_openai_spec(self) -> None:
        """Returns correct spec for OpenAI models."""
        spec = get_summary_model_spec('gpt-5-mini')
        assert spec is not None
        assert spec.provider == 'openai'
        assert spec.max_input_tokens == 400000
        assert spec.truncation_behavior == 'error'

    def test_returns_anthropic_spec(self) -> None:
        """Returns correct spec for Anthropic models."""
        spec = get_summary_model_spec('claude-haiku-4-5-20251001')
        assert spec is not None
        assert spec.provider == 'anthropic'
        assert spec.max_input_tokens == 200000


class TestGetSummaryProviderDefaultContext:
    """Tests for get_summary_provider_default_context function."""

    def test_ollama_default(self) -> None:
        """Ollama default context is 32768."""
        assert get_summary_provider_default_context('ollama') == 32768

    def test_openai_default(self) -> None:
        """OpenAI default context is 400000."""
        assert get_summary_provider_default_context('openai') == 400000

    def test_anthropic_default(self) -> None:
        """Anthropic default context is 200000."""
        assert get_summary_provider_default_context('anthropic') == 200000

    def test_unknown_provider_default(self) -> None:
        """Unknown provider returns conservative 32768."""
        assert get_summary_provider_default_context('unknown') == 32768


class TestSummaryModelSpecsRegistry:
    """Tests for SUMMARY_MODEL_SPECS registry consistency."""

    def test_all_entries_have_required_fields(self) -> None:
        """All entries in SUMMARY_MODEL_SPECS have valid required fields."""
        for name, spec in SUMMARY_MODEL_SPECS.items():
            assert spec.provider in ('ollama', 'openai', 'anthropic'), f'{name}: invalid provider'
            assert spec.model == name, f'{name}: model name mismatch'
            assert spec.max_input_tokens > 0, f'{name}: max_input_tokens must be positive'
            assert spec.truncation_behavior in ('error', 'silent', 'configurable'), (
                f'{name}: invalid truncation_behavior'
            )

    def test_ollama_models_are_configurable(self) -> None:
        """All Ollama models have configurable truncation."""
        for name, spec in SUMMARY_MODEL_SPECS.items():
            if spec.provider == 'ollama':
                assert spec.truncation_behavior == 'configurable', f'{name}: should be configurable'

    def test_openai_models_return_error(self) -> None:
        """All OpenAI models have error truncation behavior."""
        for name, spec in SUMMARY_MODEL_SPECS.items():
            if spec.provider == 'openai':
                assert spec.truncation_behavior == 'error', f'{name}: should be error'

    def test_default_model_exists_in_registry(self) -> None:
        """Default summary model (qwen3:0.6b) exists in registry."""
        assert 'qwen3:0.6b' in SUMMARY_MODEL_SPECS
