"""OpenAI summary provider using LangChain ChatOpenAI integration.

Uses ChatOpenAI for LLM-based abstractive summarization via OpenAI API.

Requires OPENAI_API_KEY environment variable (auto-detected by ChatOpenAI).
"""

from __future__ import annotations

import logging
from typing import Any

from app.errors import ConfigurationError
from app.errors import is_client_error
from app.settings import get_settings
from app.summary.instructions import resolve_summary_prompt
from app.summary.retry import with_summary_retry_and_timeout

logger = logging.getLogger(__name__)


class OpenAISummaryProvider:
    """OpenAI summary provider using LangChain ChatOpenAI integration.

    Implements the SummaryProvider protocol for OpenAI models.

    Environment Variables:
        SUMMARY_PROVIDER: Must be 'openai'
        OPENAI_API_KEY: OpenAI API key (auto-detected by ChatOpenAI)
        SUMMARY_MODEL: Model name (e.g., 'gpt-5.4-nano', 'gpt-5.4')
        SUMMARY_MAX_TOKENS: Maximum output tokens for summary generation (default: 2000)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.summary.model
        self._max_tokens = settings.summary.max_tokens
        self._reasoning_effort = settings.summary.openai_reasoning_effort
        self._chat_model: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain ChatOpenAI client.

        ChatOpenAI auto-reads OPENAI_API_KEY from environment.

        Raises:
            ImportError: If langchain-openai is not installed
        """
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError(
                'langchain-openai package required for OpenAI summary provider',
            ) from e

        # Build kwargs to avoid pyright complaints about dynamically-loaded constructor
        kwargs: dict[str, Any] = {
            'model': self._model,
            'temperature': 0,
            'max_tokens': self._max_tokens,
        }
        if self._reasoning_effort is not None:
            kwargs['reasoning_effort'] = self._reasoning_effort
        self._chat_model = ChatOpenAI(**kwargs)
        logger.info(
            f'Initialized OpenAI summary provider: {self._model}, '
            f'max_tokens={self._max_tokens}, reasoning_effort={self._reasoning_effort}',
        )

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._chat_model = None
        logger.info('OpenAI summary provider shut down')

    async def summarize(self, text: str, source: str) -> str:
        """Generate summary for the given text.

        Args:
            text: Text content to summarize
            source: Source type ('user' or 'agent')

        Returns:
            Summary string

        Raises:
            RuntimeError: If provider not initialized, or if output is empty
                due to reasoning tokens consuming the entire budget
        """
        if self._chat_model is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        from langchain_core.messages import HumanMessage
        from langchain_core.messages import SystemMessage

        prompt = resolve_summary_prompt(source)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=text),
        ]

        async def _summarize() -> str:
            response = await self._chat_model.ainvoke(messages)
            result = str(response.content).strip()
            # Detect token-limit truncation
            finish_reason = response.response_metadata.get('finish_reason')
            if finish_reason == 'length':
                if not result:
                    raise RuntimeError(
                        'OpenAI summary generation produced empty output '
                        '(finish_reason=length). Reasoning tokens likely consumed '
                        f'the entire max_tokens budget ({self._max_tokens}). '
                        'Fix: reduce SUMMARY_OPENAI_REASONING_EFFORT (current: '
                        f'{self._reasoning_effort!r}) or increase SUMMARY_MAX_TOKENS.',
                    )
                logger.warning(
                    'Summary was truncated by token limit (finish_reason=length). '
                    'Consider increasing SUMMARY_MAX_TOKENS (current: %d)',
                    self._max_tokens,
                )
            return result

        return await with_summary_retry_and_timeout(
            _summarize, f'{self.provider_name}_summarize',
        )

    async def is_available(self) -> bool:
        """Check if OpenAI API is available.

        Returns:
            True if provider is ready to generate summaries

        Raises:
            ConfigurationError: If the API returns a client error (4xx) indicating
                a permanent configuration problem (e.g., invalid reasoning_effort value)
        """
        if self._chat_model is None:
            return False

        try:
            from langchain_core.messages import HumanMessage

            await self._chat_model.ainvoke([HumanMessage(content='test')])
        except Exception as e:
            if is_client_error(e):
                raise ConfigurationError(
                    f'OpenAI API returned a client error during availability check: {e}. '
                    'This indicates a permanent configuration problem (e.g., invalid '
                    'SUMMARY_OPENAI_REASONING_EFFORT or SUMMARY_MODEL). '
                    'Fix: Check the error message above and correct the configuration.',
                ) from e
            logger.warning(f'OpenAI summary provider not available: {e}')
            return False
        else:
            return True

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'openai'
