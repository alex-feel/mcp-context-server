"""Tests for the Anthropic summary provider with mocked ChatAnthropic.

Tests verify:
- __init__ reads settings correctly
- initialize() creates ChatAnthropic with correct params
- summarize() returns stripped content
- summarize() raises RuntimeError when not initialized
- summarize() logs WARNING on truncation (stop_reason=max_tokens)
- summarize() does NOT log warning on normal completion (stop_reason=end_turn)
- summarize() uses self._max_tokens in truncation warning (no hardcoded values)
- shutdown() sets _chat_model to None
- is_available() returns True when model responds
- is_available() returns False when not initialized
- provider_name returns 'anthropic'
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_anthropic_settings():
    """Mock settings for Anthropic summary provider tests."""
    with patch('app.summary.providers.langchain_anthropic.get_settings') as mock:
        mock.return_value.summary.model = 'claude-haiku-4-5-20251001'
        mock.return_value.summary.max_tokens = 1800
        mock.return_value.summary.prompt = None
        mock.return_value.summary.timeout_s = 240.0
        mock.return_value.summary.retry_max_attempts = 3
        mock.return_value.summary.retry_base_delay_s = 0.01
        yield mock


@pytest.fixture
def mock_anthropic_retry():
    """Mock the retry wrapper to bypass retry/timeout logic in unit tests."""
    with patch('app.summary.providers.langchain_anthropic.with_summary_retry_and_timeout') as mock:
        async def passthrough(func, _operation_name=''):
            return await func()

        mock.side_effect = passthrough
        yield mock


@pytest.mark.usefixtures('mock_anthropic_settings')
class TestAnthropicSummaryProviderInit:
    """Tests for AnthropicSummaryProvider.__init__."""

    def test_init_reads_model_from_settings(self) -> None:
        """__init__ reads model name from settings."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        assert provider._model == 'claude-haiku-4-5-20251001'

    def test_init_reads_max_tokens_from_settings(self) -> None:
        """__init__ reads max_tokens from settings."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        assert provider._max_tokens == 1800

    def test_init_chat_model_is_none(self) -> None:
        """__init__ does not create ChatAnthropic instance."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        assert provider._chat_model is None


@pytest.mark.usefixtures('mock_anthropic_settings')
class TestAnthropicSummaryProviderInitialize:
    """Tests for AnthropicSummaryProvider.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_creates_chat_model(self) -> None:
        """initialize() creates a ChatAnthropic instance with correct parameters."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        mock_chat_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatAnthropic = mock_chat_cls
        with patch.dict('sys.modules', {'langchain_anthropic': mock_module}):
            provider = AnthropicSummaryProvider()
            await provider.initialize()

            mock_chat_cls.assert_called_once_with(
                model='claude-haiku-4-5-20251001',
                temperature=0,
                max_tokens=1800,
            )
            assert provider._chat_model is not None

    @pytest.mark.asyncio
    async def test_initialize_raises_import_error(self) -> None:
        """initialize() raises ImportError when langchain-anthropic not installed."""
        import sys

        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()

        saved_modules = {}
        for key in list(sys.modules):
            if key.startswith('langchain_anthropic'):
                saved_modules[key] = sys.modules.pop(key)

        try:
            with (
                patch.dict('sys.modules', {'langchain_anthropic': None}),
                pytest.raises(ImportError, match='langchain-anthropic package required'),
            ):
                await provider.initialize()
        finally:
            sys.modules.update(saved_modules)


@pytest.mark.usefixtures('mock_anthropic_settings', 'mock_anthropic_retry')
class TestAnthropicSummaryProviderSummarize:
    """Tests for AnthropicSummaryProvider.summarize()."""

    @pytest.mark.asyncio
    async def test_summarize_returns_stripped_content(self) -> None:
        """summarize() returns stripped response content."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = '  Summary with whitespace  '
        mock_response.response_metadata = {'stop_reason': 'end_turn'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await provider.summarize('Some text to summarize.')

        assert result == 'Summary with whitespace'

    @pytest.mark.asyncio
    async def test_summarize_raises_when_not_initialized(self) -> None:
        """summarize() raises RuntimeError when provider not initialized."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        assert provider._chat_model is None

        with pytest.raises(RuntimeError, match='Provider not initialized'):
            await provider.summarize('Some text')

    @pytest.mark.asyncio
    async def test_summarize_logs_warning_on_truncation(self, caplog: pytest.LogCaptureFixture) -> None:
        """summarize() logs WARNING when stop_reason=max_tokens indicates truncation."""
        import logging

        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Truncated summary text'
        mock_response.response_metadata = {'stop_reason': 'max_tokens'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            result = await provider.summarize('Some long text')

        assert result == 'Truncated summary text'
        assert 'truncated by token limit' in caplog.text
        assert 'stop_reason=max_tokens' in caplog.text
        assert 'SUMMARY_MAX_TOKENS' in caplog.text
        # Verify runtime value appears (not a hardcoded number)
        assert str(provider._max_tokens) in caplog.text

    @pytest.mark.asyncio
    async def test_summarize_no_warning_on_normal_completion(self, caplog: pytest.LogCaptureFixture) -> None:
        """summarize() does NOT log truncation warning when stop_reason=end_turn."""
        import logging

        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Normal summary text'
        mock_response.response_metadata = {'stop_reason': 'end_turn'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            result = await provider.summarize('Some text')

        assert result == 'Normal summary text'
        assert 'truncated by token limit' not in caplog.text

    @pytest.mark.asyncio
    async def test_summarize_uses_runtime_max_tokens_in_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Truncation warning includes self._max_tokens, not a hardcoded value."""
        import logging

        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        # Verify the configured value is 1800 (from fixture)
        assert provider._max_tokens == 1800

        mock_response = MagicMock()
        mock_response.content = 'Truncated'
        mock_response.response_metadata = {'stop_reason': 'max_tokens'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            await provider.summarize('Text')

        # Must contain the runtime value 1800, not the default 2000
        assert '1800' in caplog.text

    @pytest.mark.asyncio
    async def test_summarize_uses_system_and_human_messages(self) -> None:
        """summarize() sends SystemMessage + HumanMessage to the chat model."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Summary'
        mock_response.response_metadata = {'stop_reason': 'end_turn'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        await provider.summarize('Input text content')

        call_args = provider._chat_model.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].__class__.__name__ == 'SystemMessage'
        assert messages[1].__class__.__name__ == 'HumanMessage'
        assert messages[1].content == 'Input text content'


@pytest.mark.usefixtures('mock_anthropic_settings')
class TestAnthropicSummaryProviderShutdown:
    """Tests for AnthropicSummaryProvider.shutdown()."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_chat_model_to_none(self) -> None:
        """shutdown() sets _chat_model to None."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        provider._chat_model = MagicMock()

        await provider.shutdown()

        assert provider._chat_model is None


@pytest.mark.usefixtures('mock_anthropic_settings')
class TestAnthropicSummaryProviderIsAvailable:
    """Tests for AnthropicSummaryProvider.is_available()."""

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_model_responds(self) -> None:
        """is_available() returns True when chat model responds successfully."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'ok'
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await provider.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_initialized(self) -> None:
        """is_available() returns False when _chat_model is None."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        assert provider._chat_model is None

        result = await provider.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_exception(self) -> None:
        """is_available() returns False when chat model raises an exception."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(side_effect=ConnectionError('Connection refused'))

        result = await provider.is_available()

        assert result is False


@pytest.mark.usefixtures('mock_anthropic_settings')
class TestAnthropicSummaryProviderName:
    """Tests for AnthropicSummaryProvider.provider_name property."""

    def test_provider_name_returns_anthropic(self) -> None:
        """provider_name returns 'anthropic'."""
        from app.summary.providers.langchain_anthropic import AnthropicSummaryProvider

        provider = AnthropicSummaryProvider()
        assert provider.provider_name == 'anthropic'
