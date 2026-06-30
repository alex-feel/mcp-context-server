"""Anthropic summary provider using LangChain ChatAnthropic integration.

Uses ChatAnthropic for LLM-based abstractive summarization via Anthropic API.

Requires ANTHROPIC_API_KEY environment variable (read from SummarySettings, passed explicitly).
"""


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
        ANTHROPIC_API_KEY: Anthropic API key (read from SummarySettings, passed explicitly)
        SUMMARY_MODEL: Model name (e.g., 'claude-haiku-4-5-20251001')
        SUMMARY_MAX_TOKENS: Maximum output tokens for summary generation (default: 4000)
    """

    def __init__(self) -> None:
        """Initialize provider configuration from settings."""
        settings = get_settings()
        self._model = settings.summary.model
        self._max_tokens = settings.summary.max_tokens
        self._effort = settings.summary.anthropic_effort
        self._api_key = settings.summary.anthropic_api_key
        self._chat_model: Any = None

    async def initialize(self) -> None:
        """Initialize LangChain ChatAnthropic client.

        Reads ANTHROPIC_API_KEY from SummarySettings and passes it explicitly
        to the ChatAnthropic constructor.

        Raises:
            ImportError: If langchain-anthropic is not installed
            ValueError: If API key is not configured
        """
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError(
                'langchain-anthropic package required for Anthropic summary provider',
            ) from e

        if self._api_key is None:
            raise ValueError(
                'ANTHROPIC_API_KEY is required for Anthropic summary provider. '
                'Set the environment variable or use a different provider.',
            )

        # Build kwargs to avoid pyright complaints about dynamically-loaded constructor.
        # Disable internal SDK retry (max_retries=0) and timeout (timeout=None) so the
        # tenacity wrapper is the sole retry/timeout authority, matching the embedding
        # providers; otherwise each tenacity attempt would perform the SDK's own default
        # retries on top, multiplying call volume and inflating the per-attempt budget.
        kwargs: dict[str, Any] = {
            'model': self._model,
            'temperature': 0,
            'max_tokens': self._max_tokens,
            'api_key': self._api_key.get_secret_value(),
            'max_retries': 0,
            'timeout': None,
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

    async def summarize_with_prompt(self, text: str, system_prompt: str) -> str:
        """Generate a summary using an explicit system prompt (index_tree node abstracts).

        Reuses the same ChatAnthropic client and retry/timeout machinery as
        :meth:`summarize` but with a caller-supplied system prompt.

        Args:
            text: Text content to summarize.
            system_prompt: System prompt controlling the summary.

        Returns:
            Summary string (may be empty if the model returns nothing).

        Raises:
            RuntimeError: If the provider is not initialized or output is empty
                due to the max_tokens budget being fully consumed.
        """
        if self._chat_model is None:
            raise RuntimeError('Provider not initialized. Call initialize() first.')

        from langchain_core.messages import HumanMessage
        from langchain_core.messages import SystemMessage

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=text),
        ]

        async def _summarize() -> str:
            response = await self._chat_model.ainvoke(messages)
            result = str(response.content).strip()
            stop_reason = response.response_metadata.get('stop_reason')
            if stop_reason == 'max_tokens' and not result:
                raise RuntimeError(
                    'Anthropic node summary produced empty output (stop_reason=max_tokens); '
                    f'the max_tokens budget ({self._max_tokens}) was consumed. '
                    'Fix: increase SUMMARY_MAX_TOKENS.',
                )
            return result

        return await with_summary_retry_and_timeout(
            _summarize, f'{self.provider_name}_summarize_node',
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
