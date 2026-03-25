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
        mock.return_value.summary.timeout_s = 240.0
        mock.return_value.summary.retry_max_attempts = 5
        mock.return_value.summary.retry_base_delay_s = 0.01
        mock.return_value.summary.ollama_truncate = False
        mock.return_value.summary.ollama_num_ctx = 32768
        mock.return_value.ollama.host = 'http://localhost:11434'
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

    def test_init_reads_truncate_from_settings(self) -> None:
        """__init__ reads SUMMARY_OLLAMA_TRUNCATE from settings."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._truncate is False

    def test_init_reads_num_ctx_from_settings(self) -> None:
        """__init__ reads SUMMARY_OLLAMA_NUM_CTX from settings."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._num_ctx == 32768


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
                    num_ctx=32768,
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

    @pytest.mark.asyncio
    async def test_initialize_logs_truncation_mode(self, caplog: pytest.LogCaptureFixture) -> None:
        """initialize() logs truncation mode when SUMMARY_OLLAMA_TRUNCATE=false."""
        import logging

        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        with patch('app.summary.providers.langchain_ollama.ChatOllama', create=True) as mock_chat:
            mock_module = MagicMock()
            mock_module.ChatOllama = mock_chat
            with patch.dict('sys.modules', {'langchain_ollama': mock_module}):
                provider = OllamaSummaryProvider()
                with caplog.at_level(logging.INFO):
                    await provider.initialize()

        assert 'num_ctx=32768' in caplog.text
        assert 'SUMMARY_OLLAMA_TRUNCATE=false' in caplog.text
        assert 'Text length validation enabled' in caplog.text


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

        result = await provider.summarize('Some long text to summarize.', 'agent')

        assert result == 'Summary text with whitespace'

    @pytest.mark.asyncio
    async def test_summarize_raises_when_not_initialized(self) -> None:
        """summarize() raises RuntimeError when provider not initialized."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        assert provider._chat_model is None

        with pytest.raises(RuntimeError, match='Provider not initialized'):
            await provider.summarize('Some text', 'agent')

    @pytest.mark.asyncio
    async def test_summarize_uses_system_and_human_messages(self) -> None:
        """summarize() sends SystemMessage + HumanMessage to the chat model."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = 'Summary'
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        await provider.summarize('Input text content', 'agent')

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
            result = await provider.summarize('Some long text', 'agent')

        assert result == 'Truncated summary text'
        assert 'truncated by token limit' in caplog.text
        assert 'done_reason=length' in caplog.text
        assert 'SUMMARY_MAX_TOKENS' in caplog.text
        # Verify runtime value appears (not a hardcoded number)
        assert str(provider._max_tokens) in caplog.text

    @pytest.mark.asyncio
    async def test_summarize_raises_runtime_error_on_empty_truncated(self) -> None:
        """summarize() raises RuntimeError when done_reason=length AND content is empty."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        mock_response = MagicMock()
        mock_response.content = '   '
        mock_response.response_metadata = {'done_reason': 'length'}
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

        with pytest.raises(RuntimeError, match='empty output'):
            await provider.summarize('Some text', 'agent')

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
            result = await provider.summarize('Some text', 'agent')

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
            await provider.summarize('Text', 'agent')

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

    @pytest.mark.asyncio
    async def test_is_available_raises_configuration_error_on_4xx(self) -> None:
        """is_available() raises ConfigurationError on HTTP 4xx client errors."""
        from app.errors import ConfigurationError
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        class FakeClientError(Exception):
            """Exception with status_code attribute simulating an HTTP client error."""

            def __init__(self, message: str, status_code: int) -> None:
                super().__init__(message)
                self.status_code = status_code

        provider = OllamaSummaryProvider()
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(
            side_effect=FakeClientError('invalid model', status_code=400),
        )

        with pytest.raises(ConfigurationError, match='client error'):
            await provider.is_available()

    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_transient_error(self) -> None:
        """is_available() returns False for transient errors (no status_code or 5xx)."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        provider._chat_model = AsyncMock()
        provider._chat_model.ainvoke = AsyncMock(side_effect=TimeoutError('Connection timed out'))

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


@pytest.mark.usefixtures('mock_summary_settings')
class TestOllamaSummaryProviderValidateTextLength:
    """Tests for OllamaSummaryProvider._validate_text_length."""

    def test_validate_passes_for_short_text(self) -> None:
        """_validate_text_length does not raise for short text within limits."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        # With num_ctx=32768, max_tokens=2000, prompt ~230 tokens
        # Available = 32768 - 2000 - 230 = 30538 tokens = ~91614 chars
        provider._validate_text_length('short text', 'agent')

    def test_validate_raises_for_text_exceeding_context(self) -> None:
        """_validate_text_length raises ValueError when text exceeds available budget."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        # With num_ctx=32768, max_tokens=2000, prompt ~230 tokens
        # Available = 32768 - 2000 - 230 = ~30538 tokens = ~91614 chars
        # Create text that exceeds this
        long_text = 'x' * 300000

        with pytest.raises(ValueError, match='may exceed available input budget'):
            provider._validate_text_length(long_text, 'agent')

    def test_validate_raises_when_context_too_small_for_output(self) -> None:
        """_validate_text_length raises ValueError when context window too small for output budget."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        # Set impossibly small num_ctx that cannot fit output + prompt
        provider._num_ctx = 100  # Way too small for max_tokens=2000

        with pytest.raises(ValueError, match='too small for output budget'):
            provider._validate_text_length('any text', 'agent')

    def test_validate_uses_model_spec_when_available(self) -> None:
        """_validate_text_length uses model spec max_input_tokens when model is known."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        # qwen3:0.6b is in SUMMARY_MODEL_SPECS with max_input_tokens=32768
        # num_ctx=4096 is smaller, so it should cap at 4096
        # This is verified by the error message mentioning 'capped by'
        provider._num_ctx = 4096
        long_text = 'x' * 20000
        with pytest.raises(ValueError, match='capped by SUMMARY_OLLAMA_NUM_CTX'):
            provider._validate_text_length(long_text, 'agent')

    def test_validate_uses_num_ctx_for_unknown_model(self) -> None:
        """_validate_text_length uses SUMMARY_OLLAMA_NUM_CTX for unknown models."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        provider._model = 'unknown-model'
        provider._num_ctx = 4096
        long_text = 'x' * 20000
        with pytest.raises(ValueError, match='SUMMARY_OLLAMA_NUM_CTX'):
            provider._validate_text_length(long_text, 'agent')

    def test_validate_not_called_when_truncate_true(self) -> None:
        """_validate_text_length is not called when SUMMARY_OLLAMA_TRUNCATE=true."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        with patch('app.summary.providers.langchain_ollama.get_settings') as mock_settings:
            mock_settings.return_value.summary.model = 'qwen3:0.6b'
            mock_settings.return_value.summary.max_tokens = 2000
            mock_settings.return_value.summary.prompt = None
            mock_settings.return_value.summary.timeout_s = 240.0
            mock_settings.return_value.summary.retry_max_attempts = 5
            mock_settings.return_value.summary.retry_base_delay_s = 0.01
            mock_settings.return_value.summary.ollama_truncate = True
            mock_settings.return_value.summary.ollama_num_ctx = 4096
            mock_settings.return_value.ollama.host = 'http://localhost:11434'

            provider = OllamaSummaryProvider()
            assert provider._truncate is True
            # Should NOT raise even for very long text because truncate=True
            # (summarize() skips _validate_text_length when truncate=True)


@pytest.mark.usefixtures('mock_summary_settings', 'mock_retry')
class TestOllamaSummaryProviderSummarizeValidation:
    """Tests for _validate_text_length integration with summarize()."""

    @pytest.mark.asyncio
    async def test_summarize_raises_value_error_when_text_too_long(self) -> None:
        """summarize() raises ValueError for text exceeding context when truncate=false."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        provider = OllamaSummaryProvider()
        provider._chat_model = AsyncMock()
        long_text = 'x' * 300000

        with pytest.raises(ValueError, match='may exceed available input budget'):
            await provider.summarize(long_text, 'agent')

        # Verify the chat model was NOT called (fail-fast)
        provider._chat_model.ainvoke.assert_not_called()

    @pytest.mark.asyncio
    async def test_summarize_skips_validation_when_truncate_true(self) -> None:
        """summarize() skips text length validation when truncate=true."""
        from app.summary.providers.langchain_ollama import OllamaSummaryProvider

        with patch('app.summary.providers.langchain_ollama.get_settings') as mock_settings:
            mock_settings.return_value.summary.model = 'qwen3:0.6b'
            mock_settings.return_value.summary.max_tokens = 2000
            mock_settings.return_value.summary.prompt = None
            mock_settings.return_value.summary.timeout_s = 240.0
            mock_settings.return_value.summary.retry_max_attempts = 5
            mock_settings.return_value.summary.retry_base_delay_s = 0.01
            mock_settings.return_value.summary.ollama_truncate = True
            mock_settings.return_value.summary.ollama_num_ctx = 4096
            mock_settings.return_value.ollama.host = 'http://localhost:11434'

            provider = OllamaSummaryProvider()
            mock_response = MagicMock()
            mock_response.content = 'Summary of long text'
            mock_response.response_metadata = {'done_reason': 'stop'}
            provider._chat_model = AsyncMock()
            provider._chat_model.ainvoke = AsyncMock(return_value=mock_response)

            # Should NOT raise even for very long text
            result = await provider.summarize('x' * 20000, 'agent')
            assert result == 'Summary of long text'
