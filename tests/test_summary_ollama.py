"""Tests for the Ollama summary provider with mocked ChatOllama.

Tests verify:
- __init__ reads settings correctly
- initialize() creates ChatOllama with correct params
- summarize() returns stripped content
- summarize() raises RuntimeError when not initialized
- shutdown() sets _chat_model to None
- is_available() returns True when model responds
- is_available() returns False when not initialized
- provider_name returns 'ollama'
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_summary_settings():
    """Mock settings for Ollama summary provider tests."""
    with patch('app.summary.providers.langchain_ollama.get_settings') as mock:
        mock.return_value.summary.model = 'qwen3:0.6b'
        mock.return_value.summary.max_tokens = 2000
        mock.return_value.summary.prompt = None
        mock.return_value.summary.timeout_s = 120.0
        mock.return_value.summary.retry_max_attempts = 3
        mock.return_value.summary.retry_base_delay_s = 0.01
        mock.return_value.embedding.ollama_host = 'http://localhost:11434'
        yield mock


@pytest.fixture
def mock_retry():
    """Mock the retry wrapper to bypass retry/timeout logic in unit tests."""
    with patch('app.summary.providers.langchain_ollama.with_summary_retry_and_timeout') as mock:
        # Make the retry wrapper simply call the function directly
        async def passthrough(func, _operation_name=''):
            return await func()

        mock.side_effect = passthrough
        yield mock


@pytest.mark.usefixtures('mock_summary_settings')
class TestOllamaSummaryProviderInit:
    """Tests for OllamaSummaryProvider.__init__."""

    def test_init_reads_model_from_settings(self) -> None:
        """__init__ reads model name from settings."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._model == 'qwen3:0.6b'

    def test_init_reads_base_url_from_settings(self) -> None:
        """__init__ reads OLLAMA_HOST from embedding settings (shared Ollama instance)."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._base_url == 'http://localhost:11434'

    def test_init_reads_max_tokens_from_settings(self) -> None:
        """__init__ reads max_tokens from settings."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._max_tokens == 2000

    def test_init_chat_model_is_none(self) -> None:
        """__init__ does not create ChatOllama instance."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._chat_model is None


@pytest.mark.usefixtures('mock_summary_settings')
class TestOllamaSummaryProviderInitialize:
    """Tests for OllamaSummaryProvider.initialize()."""

    @pytest.mark.asyncio
    async def test_initialize_creates_chat_model(self) -> None:
        """initialize() creates a ChatOllama instance."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        with patch('app.summary.providers.langchain_ollama.ChatOllama', create=True) as mock_chat:
            # Patch the import inside initialize()
            mock_module = MagicMock()
            mock_module.ChatOllama = mock_chat
            with patch.dict('sys.modules', {'langchain_ollama': mock_module}):
                provider = OllamaSummaryProvider()
                await provider.initialize()

                mock_chat.assert_called_once_with(
                    model='qwen3:0.6b',
                    base_url='http://localhost:11434',
                    temperature=0,
                    num_predict=2000,
                )
                assert provider._chat_model is not None

    @pytest.mark.asyncio
    async def test_initialize_raises_import_error(self) -> None:
        """initialize() raises ImportError when langchain-ollama not installed."""
        import sys

        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()

        # Temporarily remove langchain_ollama from sys.modules to simulate missing package
        saved_modules = {}
        for key in list(sys.modules):
            if key.startswith('langchain_ollama'):
                saved_modules[key] = sys.modules.pop(key)

        try:
            with (
                patch.dict('sys.modules', {'langchain_ollama': None}),
                pytest.raises(ImportError, match='langchain-ollama package required'),
            ):
                await provider.initialize()
        finally:
            sys.modules.update(saved_modules)


@pytest.mark.usefixtures('mock_summary_settings', 'mock_retry')
class TestOllamaSummaryProviderSummarize:
    """Tests for OllamaSummaryProvider.summarize()."""

    @pytest.mark.asyncio
    async def test_summarize_returns_stripped_content(self) -> None:
        """summarize() returns stripped response content."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        # Simulate an initialized chat model
        mock_response = MagicMock()
        mock_response.content = '  Summary text with whitespace  '
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await provider.summarize('Some long text to summarize.')

        assert result == 'Summary text with whitespace'

    @pytest.mark.asyncio
    async def test_summarize_raises_when_not_initialized(self) -> None:
        """summarize() raises RuntimeError when provider not initialized."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._chat_model is None

        with pytest.raises(RuntimeError, match='Provider not initialized'):
            await provider.summarize('Some text')

    @pytest.mark.asyncio
    async def test_summarize_uses_system_and_human_messages(self) -> None:
        """summarize() sends SystemMessage + HumanMessage to the chat model."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Summary'
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        await provider.summarize('Input text content')

        # Verify ainvoke was called with a list of messages
        call_args = provider._chat_model.ainvoke.call_args
        messages = call_args[0][0]
        assert len(messages) == 2
        # First message is SystemMessage (prompt)
        assert messages[0].__class__.__name__ == 'SystemMessage'
        # Second message is HumanMessage (input text)
        assert messages[1].__class__.__name__ == 'HumanMessage'
        assert messages[1].content == 'Input text content'

    @pytest.mark.asyncio
    async def test_summarize_logs_warning_on_truncation(self, caplog: pytest.LogCaptureFixture) -> None:
        """summarize() logs WARNING when model response indicates token-limit truncation."""
        import logging

        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Truncated summary text'
        mock_response.response_metadata = {'done_reason': 'length'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            result = await provider.summarize('Some long text')

        assert result == 'Truncated summary text'
        assert 'truncated by token limit' in caplog.text
        assert 'done_reason=length' in caplog.text
        assert 'SUMMARY_MAX_TOKENS' in caplog.text
        # Verify runtime value appears (not a hardcoded number)
        assert str(provider._max_tokens) in caplog.text

    @pytest.mark.asyncio
    async def test_summarize_no_warning_on_normal_completion(self, caplog: pytest.LogCaptureFixture) -> None:
        """summarize() does NOT log truncation warning on normal completion."""
        import logging

        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Normal summary text'
        mock_response.response_metadata = {'done_reason': 'stop'}
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

        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        # Verify the configured value is 2000 (from fixture)
        assert provider._max_tokens == 2000

        mock_response = MagicMock()
        mock_response.content = 'Truncated'
        mock_response.response_metadata = {'done_reason': 'length'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with caplog.at_level(logging.WARNING):
            await provider.summarize('Text')

        # Must contain the runtime value
        assert '2000' in caplog.text
        assert 'SUMMARY_MAX_TOKENS' in caplog.text


@pytest.mark.usefixtures('mock_summary_settings')
class TestOllamaSummaryProviderShutdown:
    """Tests for OllamaSummaryProvider.shutdown()."""

    @pytest.mark.asyncio
    async def test_shutdown_sets_chat_model_to_none(self) -> None:
        """shutdown() sets _chat_model to None."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        provider._chat_model = MagicMock()  # Simulate initialized state

        await provider.shutdown()

        assert provider._chat_model is None


@pytest.mark.usefixtures('mock_summary_settings')
class TestOllamaSummaryProviderIsAvailable:
    """Tests for OllamaSummaryProvider.is_available()."""

    @pytest.mark.asyncio
    async def test_is_available_returns_true_when_model_responds(self) -> None:
        """is_available() returns True when chat model responds successfully."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'ok'
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        result = await provider.is_available()

        assert result is True

    @pytest.mark.asyncio
    async def test_is_available_returns_false_when_not_initialized(self) -> None:
        """is_available() returns False when _chat_model is None."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._chat_model is None

        result = await provider.is_available()

        assert result is False

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_exception(self) -> None:
        """is_available() returns False when chat model raises an exception."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(side_effect=ConnectionError('Connection refused'))

        result = await provider.is_available()

        assert result is False


@pytest.mark.usefixtures('mock_summary_settings')
class TestOllamaSummaryProviderName:
    """Tests for OllamaSummaryProvider.provider_name property."""

    def test_provider_name_returns_ollama(self) -> None:
        """provider_name returns 'ollama'."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider.provider_name == 'ollama'
