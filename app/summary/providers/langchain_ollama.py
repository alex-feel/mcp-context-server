"""Ollama summary provider using LangChain ChatOllama integration.

Uses ChatOllama for LLM-based abstractive summarization.
Default model: qwen3:1.7b (configurable via SUMMARY_MODEL).

Qwen3 Family Model Recommendations:
    - qwen3:0.6b  - Minimal resources, basic summaries
    - qwen3:1.7b  - Default, good balance of quality and speed
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
        SUMMARY_MODEL: Model name (default: qwen3:1.7b)
        SUMMARY_MAX_TOKENS: Maximum output tokens for summary generation (default: 2000)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.summary.model
        self._base_url = settings.embedding.ollama_host
        self._max_tokens = settings.summary.max_tokens
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
        )
        logger.info(
            f'Initialized Ollama summary provider: model={self._model}, '
            f'base_url={self._base_url}, max_tokens={self._max_tokens}',
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
