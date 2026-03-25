"""Anthropic summary provider using LangChain ChatAnthropic integration.

Uses ChatAnthropic for LLM-based abstractive summarization via Anthropic API.

Requires ANTHROPIC_API_KEY environment variable (auto-detected by ChatAnthropic).
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


class AnthropicSummaryProvider:
    """Anthropic summary provider using LangChain ChatAnthropic integration.

    Implements the SummaryProvider protocol for Anthropic (Claude) models.

    Environment Variables:
        SUMMARY_PROVIDER: Must be 'anthropic'
        ANTHROPIC_API_KEY: Anthropic API key (auto-detected by ChatAnthropic)
        SUMMARY_MODEL: Model name (e.g., 'claude-haiku-4-5-20251001')
        SUMMARY_MAX_TOKENS: Maximum output tokens for summary generation (default: 2000)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.summary.model
        self._max_tokens = settings.summary.max_tokens
        self._effort = settings.summary.anthropic_effort
        self._chat_model: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain ChatAnthropic client.

        ChatAnthropic auto-reads ANTHROPIC_API_KEY from environment.

        Raises:
            ImportError: If langchain-anthropic is not installed
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                'langchain-anthropic package required for Anthropic summary provider',
            ) from e

        # Build kwargs to avoid pyright complaints about dynamically-loaded constructor
        kwargs: dict[str, Any] = {
            'model': self._model,
            'temperature': 0,
            'max_tokens': self._max_tokens,
        }
        if self._effort is not None:
            kwargs['effort'] = self._effort
        self._chat_model = ChatAnthropic(**kwargs)
        logger.info(
            f'Initialized Anthropic summary provider: {self._model}, '
            f'max_tokens={self._max_tokens}, effort={self._effort}',
        )

    async def shutdown(self) -> None:
        """Cleanup resources."""
        self._chat_model = None
        logger.info('Anthropic summary provider shut down')

    async def summarize(self, text: str, source: str) -> str:
        """Generate summary for the given text.

        Args:
            text: Text content to summarize
            source: Source type ('user' or 'agent')

        Returns:
            Summary string

        Raises:
            RuntimeError: If provider not initialized, or if output is empty
                due to the entire max_tokens budget being consumed
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
            stop_reason = response.response_metadata.get('stop_reason')
            if stop_reason == 'max_tokens':
                if not result:
                    raise RuntimeError(
                        'Anthropic summary generation produced empty output '
                        '(stop_reason=max_tokens). '
                        f'The entire max_tokens budget ({self._max_tokens}) was consumed '
                        'without producing visible output. '
                        'Fix: increase SUMMARY_MAX_TOKENS'
                        + (f' or adjust SUMMARY_ANTHROPIC_EFFORT (current: {self._effort!r})'
                           if self._effort else '')
                        + '.',
                    )
                logger.warning(
                    'Summary was truncated by token limit (stop_reason=max_tokens). '
                    'Consider increasing SUMMARY_MAX_TOKENS (current: %d)',
                    self._max_tokens,
                )
            return result

        return await with_summary_retry_and_timeout(
            _summarize, f'{self.provider_name}_summarize',
        )

    async def is_available(self) -> bool:
        """Check if Anthropic API is available.

        Returns:
            True if provider is ready to generate summaries

        Raises:
            ConfigurationError: If the API returns a client error (4xx) indicating
                a permanent configuration problem (e.g., invalid effort value)
        """
        if self._chat_model is None:
            return False

        try:
            from langchain_core.messages import HumanMessage

            await self._chat_model.ainvoke([HumanMessage(content='test')])
        except Exception as e:
            if is_client_error(e):
                raise ConfigurationError(
                    f'Anthropic API returned a client error during availability check: {e}. '
                    'This indicates a permanent configuration problem (e.g., invalid '
                    'SUMMARY_ANTHROPIC_EFFORT or SUMMARY_MODEL). '
                    'Fix: Check the error message above and correct the configuration.',
                ) from e
            logger.warning(f'Anthropic summary provider not available: {e}')
            return False
        else:
            return True

    @property
    def provider_name(self) -> str:
        """Return provider identifier."""
        return 'anthropic'
