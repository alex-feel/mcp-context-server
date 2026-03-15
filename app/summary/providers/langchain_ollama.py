"""Ollama summary provider using LangChain ChatOllama integration.

Uses ChatOllama for LLM-based abstractive summarization.
Default model: qwen3:0.6b (configurable via SUMMARY_MODEL).

Qwen3 Family Model Recommendations:
    - qwen3:0.6b  - Default, lightweight and fast
    - qwen3:1.7b  - Higher quality, good balance of quality and speed
    - qwen3:4b    - Higher quality, proven stability, concurrent support
    - qwen3:8b    - Highest quality, requires more resources
    - qwen3:14b   - Near-frontier quality, significant resources
    - qwen3:32b   - Frontier-level, requires GPU with >20GB VRAM
"""

from __future__ import annotations

import logging
from typing import Any

from app.settings import get_settings
from app.summary.instructions import resolve_summary_prompt
from app.summary.retry import with_summary_retry_and_timeout

logger = logging.getLogger(__name__)


class OllamaSummaryProvider:
    """Ollama summary provider using LangChain ChatOllama integration.

    Implements the SummaryProvider protocol for Ollama models.

    Environment Variables:
        SUMMARY_PROVIDER: Must be 'ollama' (default)
        OLLAMA_HOST: Ollama server URL (default: http://localhost:11434)
        SUMMARY_MODEL: Model name (default: qwen3:0.6b)
        SUMMARY_MAX_TOKENS: Maximum output tokens for summary generation (default: 2000)
        SUMMARY_OLLAMA_NUM_CTX: Context length in tokens (default: 32768)
        SUMMARY_OLLAMA_TRUNCATE: Control truncation behavior (default: false = error on exceed)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.summary.model
        self._base_url = settings.ollama.host
        self._max_tokens = settings.summary.max_tokens
        self._truncate = settings.summary.ollama_truncate
        self._num_ctx = settings.summary.ollama_num_ctx
        self._prompt = resolve_summary_prompt(settings.summary)
        self._chat_model: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain ChatOllama client.

        Raises:
            ImportError: If langchain-ollama is not installed
        """
        try:
            from langchain_ollama import ChatOllama
        except ImportError as e:
            raise ImportError(
                'langchain-ollama package required for Ollama summary provider',
            ) from e

        self._chat_model = ChatOllama(
            model=self._model,
            base_url=self._base_url,
            temperature=0,
            num_predict=self._max_tokens,
            num_ctx=self._num_ctx,
        )
        logger.info(
            f'Initialized Ollama summary provider: {self._model} at {self._base_url}, '
            f'num_ctx={self._num_ctx}, max_tokens={self._max_tokens}',
        )
        if not self._truncate:
            logger.info(
                'SUMMARY_OLLAMA_TRUNCATE=false. Text length validation enabled. '
                'Texts exceeding context limit will raise error before summarization.',
            )

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._chat_model = None
        logger.info('Ollama summary provider shut down')

    async def summarize(self, text: str) -> str:
        """Generate summary for the given text.

        Uses SystemMessage (prompt) + HumanMessage (text content) pattern.

        Args:
            text: Text content to summarize

        Returns:
            Summary string

        Raises:
            RuntimeError: If provider not initialized
        """
        if self._chat_model is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        # Pre-validate text length when truncation disabled
        if not self._truncate:
            self._validate_text_length(text)

        from langchain_core.messages import HumanMessage
        from langchain_core.messages import SystemMessage

        messages = [
            SystemMessage(content=self._prompt),
            HumanMessage(content=text),
        ]

        async def _summarize() -> str:
            response = await self._chat_model.ainvoke(messages)
            result = str(response.content).strip()
            if not result:
                logger.warning('Ollama summary model returned empty response')
            # Detect token-limit truncation
            done_reason = response.response_metadata.get('done_reason')
            if done_reason == 'length':
                logger.warning(
                    'Summary was truncated by token limit (done_reason=length). '
                    'Consider increasing SUMMARY_MAX_TOKENS (current: %d)',
                    self._max_tokens,
                )
            return result

        return await with_summary_retry_and_timeout(
            _summarize, f'{self.provider_name}_summarize',
        )

    def _validate_text_length(self, text: str) -> None:
        """Validate text length against estimated context window.

        When SUMMARY_OLLAMA_TRUNCATE=false, provides fail-fast behavior by checking
        if text is likely to exceed the context window BEFORE calling the summary API.

        Unlike embedding validation, summary validation must account for:
        - SUMMARY_PROMPT system message token consumption
        - SUMMARY_MAX_TOKENS output token reservation

        Args:
            text: Text to validate

        Raises:
            ValueError: If text likely exceeds context window and truncation disabled
        """
        from app.summary.context_limits import get_summary_model_spec

        spec = get_summary_model_spec(self._model)
        if spec:
            max_tokens = min(spec.max_input_tokens, self._num_ctx)
            source = f'model spec ({spec.max_input_tokens}) capped by SUMMARY_OLLAMA_NUM_CTX ({self._num_ctx})'
        else:
            max_tokens = self._num_ctx
            source = f'SUMMARY_OLLAMA_NUM_CTX ({self._num_ctx})'

        # Reserve tokens for output and prompt overhead
        # SUMMARY_MAX_TOKENS is the output budget
        # Prompt overhead estimated at ~3 chars per token (conservative for English)
        prompt_overhead = len(self._prompt) // 3
        available_input_tokens = max_tokens - self._max_tokens - prompt_overhead

        if available_input_tokens <= 0:
            raise ValueError(
                f'Context window ({max_tokens} tokens from {source}) is too small '
                f'for output budget ({self._max_tokens} tokens) + prompt overhead (~{prompt_overhead} tokens). '
                f'Increase SUMMARY_OLLAMA_NUM_CTX or decrease SUMMARY_MAX_TOKENS.',
            )

        # Heuristic: 1 token ~ 3 characters for English
        estimated_tokens = len(text) / 3

        if estimated_tokens > available_input_tokens:
            raise ValueError(
                f'Text length ({len(text)} chars, ~{int(estimated_tokens)} estimated tokens) '
                f'may exceed available input budget ({available_input_tokens} tokens from {source}, '
                f'after reserving {self._max_tokens} output + ~{prompt_overhead} prompt tokens) '
                f'for model {self._model}. '
                f'Options: 1) Increase SUMMARY_OLLAMA_NUM_CTX, '
                f'2) Set SUMMARY_OLLAMA_TRUNCATE=true to allow silent truncation, '
                f'3) Use a larger-context model.',
            )

    async def is_available(self) -> bool:
        """Check if Ollama model is available.

        Returns:
            True if provider is ready to generate summaries
        """
        if self._chat_model is None:
            return False

        try:
            from langchain_core.messages import HumanMessage

            await self._chat_model.ainvoke([HumanMessage(content='test')])
        except Exception as e:
            logger.warning(f'Ollama summary provider not available: {e}')
            return False
        else:
            return True

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'ollama'
