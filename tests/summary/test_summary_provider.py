"""Tests for the SummaryProvider Protocol compliance and factory.

Tests verify:
- SummaryProvider protocol is runtime_checkable
- Protocol compliance for conforming and non-conforming classes
- Factory creates correct provider instances
- Factory error handling for unsupported providers
- Global state set/get round-trip
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from app.summary.base import SummaryProvider
from app.summary.factory import PROVIDER_CLASSES
from app.summary.factory import PROVIDER_INSTALL_INSTRUCTIONS
from app.summary.factory import PROVIDER_MODULES
from app.summary.factory import create_summary_provider


class _CompleteSummaryProvider:
    """Minimal class implementing all SummaryProvider protocol methods."""

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def summarize(self, text: str, source: str) -> str:
        return f'summary of {len(text)} chars ({source})'

    async def is_available(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return 'test'


class _IncompleteSummaryProvider:
    """Class missing the summarize() method."""

    async def initialize(self) -> None:
        pass

    async def shutdown(self) -> None:
        pass

    async def is_available(self) -> bool:
        return True

    @property
    def provider_name(self) -> str:
        return 'test'


class TestSummaryProviderProtocol:
    """Tests for SummaryProvider protocol compliance."""

    def test_protocol_is_runtime_checkable(self) -> None:
        """SummaryProvider must be decorated with @runtime_checkable."""
        provider = _CompleteSummaryProvider()
        assert isinstance(provider, SummaryProvider)

    def test_complete_provider_satisfies_protocol(self) -> None:
        """A class implementing all methods satisfies isinstance check."""
        provider = _CompleteSummaryProvider()
        assert isinstance(provider, SummaryProvider)

    def test_incomplete_provider_does_not_satisfy_protocol(self) -> None:
        """A class missing summarize() does NOT satisfy the protocol."""
        provider = _IncompleteSummaryProvider()
        assert not isinstance(provider, SummaryProvider)


class TestSummaryProviderFactory:
    """Tests for create_summary_provider() factory."""

    def test_default_provider_is_ollama(self) -> None:
        """Factory with default settings returns OllamaSummaryProvider."""
        with patch('app.summary.factory.get_settings') as mock_settings:
            mock_settings.return_value.summary.provider = 'ollama'
            provider = create_summary_provider()
            assert provider.provider_name == 'ollama'

    def test_create_openai_provider(self) -> None:
        """Factory with explicit 'openai' returns OpenAISummaryProvider."""
        provider = create_summary_provider('openai')
        assert provider.provider_name == 'openai'

    def test_create_anthropic_provider(self) -> None:
        """Factory with explicit 'anthropic' returns AnthropicSummaryProvider."""
        provider = create_summary_provider('anthropic')
        assert provider.provider_name == 'anthropic'

    def test_unsupported_provider_raises_value_error(self) -> None:
        """Factory with unsupported provider raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported summary provider: 'invalid'"):
            create_summary_provider('invalid')

    def test_provider_modules_has_three_entries(self) -> None:
        """PROVIDER_MODULES must have exactly 3 entries."""
        assert len(PROVIDER_MODULES) == 3
        assert set(PROVIDER_MODULES.keys()) == {'ollama', 'openai', 'anthropic'}

    def test_provider_classes_has_three_entries(self) -> None:
        """PROVIDER_CLASSES must have exactly 3 entries."""
        assert len(PROVIDER_CLASSES) == 3
        assert set(PROVIDER_CLASSES.keys()) == {'ollama', 'openai', 'anthropic'}

    def test_provider_install_instructions_has_three_entries(self) -> None:
        """PROVIDER_INSTALL_INSTRUCTIONS must have exactly 3 entries."""
        assert len(PROVIDER_INSTALL_INSTRUCTIONS) == 3
        assert set(PROVIDER_INSTALL_INSTRUCTIONS.keys()) == {'ollama', 'openai', 'anthropic'}

    def test_provider_modules_and_classes_keys_match(self) -> None:
        """PROVIDER_MODULES and PROVIDER_CLASSES must have identical key sets."""
        assert set(PROVIDER_MODULES.keys()) == set(PROVIDER_CLASSES.keys())

    def test_all_created_providers_satisfy_protocol(self) -> None:
        """All providers created by the factory satisfy the SummaryProvider protocol."""
        for name in PROVIDER_MODULES:
            provider = create_summary_provider(name)
            assert isinstance(provider, SummaryProvider), f'{name} provider does not satisfy SummaryProvider protocol'


class TestSummaryProviderGlobalState:
    """Tests for set_/get_summary_provider global state."""

    def test_initial_state_is_none(self) -> None:
        """get_summary_provider() returns None before initialization."""
        from app.startup import get_summary_provider
        from app.startup import set_summary_provider

        # Save current state to restore after test
        original = get_summary_provider()
        try:
            set_summary_provider(None)
            assert get_summary_provider() is None
        finally:
            set_summary_provider(original)

    def test_set_get_round_trip(self) -> None:
        """set_summary_provider() / get_summary_provider() round-trip works."""
        from app.startup import get_summary_provider
        from app.startup import set_summary_provider

        original = get_summary_provider()
        try:
            mock_provider = _CompleteSummaryProvider()
            set_summary_provider(mock_provider)
            assert get_summary_provider() is mock_provider
        finally:
            set_summary_provider(original)

    def test_set_none_clears_provider(self) -> None:
        """set_summary_provider(None) clears the provider."""
        from app.startup import get_summary_provider
        from app.startup import set_summary_provider

        original = get_summary_provider()
        try:
            mock_provider = _CompleteSummaryProvider()
            set_summary_provider(mock_provider)
            assert get_summary_provider() is mock_provider

            set_summary_provider(None)
            assert get_summary_provider() is None
        finally:
            set_summary_provider(original)
