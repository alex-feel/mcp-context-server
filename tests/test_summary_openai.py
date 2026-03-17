"""Tests for the OpenAI summary provider with mocked ChatOpenAI.

Tests verify:
- __init__ reads settings correctly
- initialize() creates ChatOpenAI with correct params
- summarize() returns stripped content
- summarize() raises RuntimeError when not initialized
- summarize() logs WARNING on truncation (finish_reason=length)
- summarize() does NOT log warning on normal completion (finish_reason=stop)
- summarize() uses self._max_tokens in truncation warning (no hardcoded values)
- shutdown() sets _chat_model to None
- is_available() returns True when model responds
- is_available() returns False when not initialized
- provider_name returns 'openai'
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_openai_settings():
    """Mock settings for OpenAI summary provider tests."""
    with patch('app.summary.providers.langchain_openai.get_settings') as mock:
        mock.return_value.summary.model = 'gpt-5-nano'
        mock.return_value.summary.max_tokens = 1500
        mock.return_value.summary.prompt = None
        mock.return_value.summary.timeout_s = 240.0
        mock.return_value.summary.retry_max_attempts = 5
        mock.return_value.summary.retry_base_delay_s = 0.01
        yield mock


@pytest.fixture
def mock_openai_retry():
    """Mock the retry wrapper to bypass retry/timeout logic in unit tests."""
    with patch('app.summary.providers.langchain_openai.with_summary_retry_and_timeout') as mock:
        async def passthrough(func, _operation_name=''):
            return await func()

        mock.side_effect = passthrough
        yield mock


@pytest.mark.usefixtures('mock_openai_settings')
class TestOpenAISummaryProviderInit:
    """Tests for OpenAISummaryProvider.__init__."""

    def test_init_reads_model_from_settings(self) -> None:
        """__init__ reads model name from settings."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        assert provider._model == 'gpt-5-nano'

    def test_init_reads_max_tokens_from_settings(self) -> None:
        """__init__ reads max_tokens from settings."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        assert provider._max_tokens == 1500

    def test_init_chat_model_is_none(self) -> None:
        """__init__ does not create ChatOpenAI instance."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        assert provider._chat_model is None


@pytest.mark.usefixtures('mock_openai_settings')
class TestOpenAISummaryProviderInitialize:
    """Tests for OpenAISummaryProvider.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_creates_chat_model(self) -> None:
        """initialize() creates a ChatOpenAI instance with correct parameters."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        mock_chat_cls = MagicMock()
        mock_module = MagicMock()
        mock_module.ChatOpenAI = mock_chat_cls
        with patch.dict('sys.modules', {'langchain_openai': mock_module}):
            provider = OpenAISummaryProvider()
            await provider.initialize()

            mock_chat_cls.assert_called_once_with(
                model='gpt-5-nano',
                temperature=0,
                max_tokens=1500,
            )
            assert provider._chat_model is not None

    @pytest.mark.asyncio
    async def test_initialize_raises_import_error(self) -> None:
        """initialize() raises ImportError when langchain-openai not installed."""
        import sys

        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()

        saved_modules = {}
        for key in list(sys.modules):
            if key.startswith('langchain_openai'):
                saved_modules[key] = sys.modules.pop(key)

        try:
            with (
                patch.dict('sys.modules', {'langchain_openai': None}),
                pytest.raises(ImportError, match='langchain-openai package required'),
            ):
                await provider.initialize()
        finally:
            sys.modules.update(saved_modules)


@pytest.mark.usefixtures('mock_openai_settings', 'mock_openai_retry')
class TestOpenAISummaryProviderSummarize:
    """Tests for OpenAISummaryProvider.summarize()."""

    @pytest.mark.asyncio
    async def test_summarize_returns_stripped_content(self) -> None:
        """summarize() returns stripped response content."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        mock_response = MagicMock()
        mock_response.content = '  Summary with whitespace  '
        mock_response.response_metadata = {'finish_reason': 'stop'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await provider.summarize('Some text to summarize.')

        assert result == 'Summary with whitespace'

    @pytest.mark.asyncio
    async def test_summarize_raises_when_not_initialized(self) -> None:
        """summarize() raises RuntimeError when provider not initialized."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        assert provider._chat_model is None

        with pytest.raises(RuntimeError, match='Provider not initialized'):
            await provider.summarize('Some text')

    @pytest.mark.asyncio
    async def test_summarize_logs_warning_on_truncation(self, caplog: pytest.LogCaptureFixture) -> None:
        """summarize() logs WARNING when finish_reason=length indicates truncation."""
        import logging

        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Truncated summary text'
        mock_response.response_metadata = {'finish_reason': 'length'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            result = await provider.summarize('Some long text')

        assert result == 'Truncated summary text'
        assert 'truncated by token limit' in caplog.text
        assert 'finish_reason=length' in caplog.text
        assert 'SUMMARY_MAX_TOKENS' in caplog.text
        # Verify runtime value appears (not a hardcoded number)
        assert str(provider._max_tokens) in caplog.text

    @pytest.mark.asyncio
    async def test_summarize_no_warning_on_normal_completion(self, caplog: pytest.LogCaptureFixture) -> None:
        """summarize() does NOT log truncation warning when finish_reason=stop."""
        import logging

        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Normal summary text'
        mock_response.response_metadata = {'finish_reason': 'stop'}
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

        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        # Verify the configured value is 1500 (from fixture)
        assert provider._max_tokens == 1500

        mock_response = MagicMock()
        mock_response.content = 'Truncated'
        mock_response.response_metadata = {'finish_reason': 'length'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            await provider.summarize('Text')

        # Must contain the runtime value 1500, not the default 2000
        assert '1500' in caplog.text

    @pytest.mark.asyncio
    async def test_summarize_uses_system_and_human_messages(self) -> None:
        """summarize() sends SystemMessage + HumanMessage to the chat model."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Summary'
        mock_response.response_metadata = {'finish_reason': 'stop'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        await provider.summarize('Input text content')

        call_args = provider._chat_model.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        assert messages[0].__class__.__name__ == 'SystemMessage'
        assert messages[1].__class__.__name__ == 'HumanMessage'
        assert messages[1].content == 'Input text content'


@pytest.mark.usefixtures('mock_openai_settings')
class TestOpenAISummaryProviderShutdown:
    """Tests for OpenAISummaryProvider.shutdown()."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_chat_model_to_none(self) -> None:
        """shutdown() sets _chat_model to None."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        provider._chat_model = MagicMock()

        await provider.shutdown()

        assert provider._chat_model is None


@pytest.mark.usefixtures('mock_openai_settings')
class TestOpenAISummaryProviderIsAvailable:
    """Tests for OpenAISummaryProvider.is_available()."""

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_model_responds(self) -> None:
        """is_available() returns True when chat model responds successfully."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'ok'
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await provider.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_initialized(self) -> None:
        """is_available() returns False when _chat_model is None."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        assert provider._chat_model is None

        result = await provider.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_exception(self) -> None:
        """is_available() returns False when chat model raises an exception."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(side_effect=ConnectionError('Connection refused'))

        result = await provider.is_available()

        assert result is False


@pytest.mark.usefixtures('mock_openai_settings')
class TestOpenAISummaryProviderName:
    """Tests for OpenAISummaryProvider.provider_name property."""

    def test_provider_name_returns_openai(self) -> None:
        """provider_name returns 'openai'."""
        from app.summary.providers.langchain_openai import OpenAISummaryProvider

        provider = OpenAISummaryProvider()
        assert provider.provider_name == 'openai'
