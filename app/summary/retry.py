"""Universal retry wrapper for summary providers using tenacity.

Mirrors app/embeddings/retry.py but reads from settings.summary.*
instead of settings.embedding.*. Same retry semantics, same exception
handling pattern.

Uses tenacity library for robust async retry logic with:
- Exponential backoff with jitter
- Configurable max attempts and timeouts
- Per-attempt timeout using asyncio.wait_for()
- Automatic logging before retry sleep
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable
from collections.abc import Callable
from typing import TYPE_CHECKING

import httpx
from tenacity import AsyncRetrying
from tenacity import RetryCallState
from tenacity import RetryError
from tenacity import retry_if_exception
from tenacity import stop_after_attempt
from tenacity import wait_exponential_jitter

from app.settings import get_settings

logger = logging.getLogger(__name__)


class SummaryTimeoutError(Exception):
    """Raised when summary generation times out."""


class SummaryRetryExhaustedError(Exception):
    """Raised when all summary retry attempts are exhausted."""


# Base exceptions that are always retryable (statically typed, always available)
_BASE_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    # Python built-in
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
    # Our custom exception
    SummaryTimeoutError,
    # httpx exceptions (always available since it's a core dependency)
    httpx.TimeoutException,
    httpx.NetworkError,
)

# Provider-specific exceptions identified by module and class name.
# This avoids importing optional packages at module load time.
_RETRYABLE_BY_MODULE: dict[str, frozenset[str]] = {
    # OpenAI SDK exceptions (used by ChatOpenAI)
    'openai': frozenset({
        'APITimeoutError',
        'APIConnectionError',
        'RateLimitError',
        'InternalServerError',
    }),
    # Ollama SDK exceptions (used by ChatOllama)
    'ollama': frozenset({
        'ResponseError',
    }),
    # Anthropic SDK exceptions (used by ChatAnthropic)
    'anthropic': frozenset({
        'APITimeoutError',
        'APIConnectionError',
        'RateLimitError',
        'InternalServerError',
    }),
}


def _is_retryable_exception(exc: BaseException) -> bool:
    """Check if an exception should trigger a retry.

    Args:
        exc: The exception to check.

    Returns:
        True if the exception should trigger a retry, False otherwise.
    """
    # Fast path: check base exceptions first (most common)
    if isinstance(exc, _BASE_RETRYABLE_EXCEPTIONS):
        return True

    # Check provider-specific exceptions by module and class name
    exc_type = type(exc)
    exc_module = exc_type.__module__
    exc_name = exc_type.__name__

    # Check if this exception's module has retryable exceptions
    if exc_module in _RETRYABLE_BY_MODULE and exc_name in _RETRYABLE_BY_MODULE[exc_module]:
        return True

    # Check parent module (e.g., 'openai._exceptions' -> 'openai')
    if '.' in exc_module:
        parent_module = exc_module.split('.')[0]
        if parent_module in _RETRYABLE_BY_MODULE and exc_name in _RETRYABLE_BY_MODULE[parent_module]:
            return True

    return False


# For type checking - expose what exceptions are being caught
if TYPE_CHECKING:
    RetryableExceptionType = (
        asyncio.TimeoutError
        | ConnectionError
        | OSError
        | SummaryTimeoutError
        | httpx.TimeoutException
        | httpx.NetworkError
    )


def _make_before_sleep_log(
    log: logging.Logger,
    log_level: int,
    operation_name: str,
) -> Callable[[RetryCallState], None]:
    """Create a before_sleep callback that logs the operation name.

    Tenacity's built-in ``before_sleep_log`` uses ``retry_state.fn`` to determine
    the function name, but ``fn`` is always ``None`` when using the
    ``AsyncRetrying`` iteration pattern (``async for attempt in retrying``).
    This custom callback uses the explicitly provided *operation_name* instead.

    Returns:
        Callback suitable for tenacity's ``before_sleep`` parameter.
    """
    def _log_retry(retry_state: RetryCallState) -> None:
        if retry_state.outcome is None or retry_state.next_action is None:
            return
        if retry_state.outcome.failed:
            ex = retry_state.outcome.exception()
            verb, value = 'raised', f'{ex.__class__.__name__}: {ex}'
        else:
            verb, value = 'returned', retry_state.outcome.result()
        log.log(
            log_level,
            'Retrying %s in %s seconds as it %s %s.',
            operation_name,
            retry_state.next_action.sleep,
            verb,
            value,
        )

    return _log_retry


async def with_summary_retry_and_timeout[T](
    func: Callable[[], Awaitable[T]],
    operation_name: str = 'summary',
) -> T:
    """Execute async function with retry and timeout for summary operations.

    Uses tenacity for retry logic with:
    - Exponential backoff with jitter
    - Configurable max attempts
    - Per-attempt timeout
    - Logging before each retry sleep

    Settings from SummarySettings:
    - SUMMARY_TIMEOUT_S: Timeout per attempt
    - SUMMARY_RETRY_MAX_ATTEMPTS: Maximum retry attempts
    - SUMMARY_RETRY_BASE_DELAY_S: Base delay for exponential backoff

    Args:
        func: Async function to execute (no arguments)
        operation_name: Description for logging

    Returns:
        Result from func

    Raises:
        SummaryTimeoutError: If final attempt times out
        SummaryRetryExhaustedError: If all retries exhausted due to errors
    """
    settings = get_settings()
    timeout_s = settings.summary.timeout_s
    max_attempts = settings.summary.retry_max_attempts
    base_delay_s = settings.summary.retry_base_delay_s

    retrying = AsyncRetrying(
        retry=retry_if_exception(_is_retryable_exception),
        wait=wait_exponential_jitter(
            initial=base_delay_s,
            max=60.0,  # Cap maximum wait at 60 seconds
            jitter=base_delay_s,  # Jitter up to base_delay
        ),
        stop=stop_after_attempt(max_attempts),
        before_sleep=_make_before_sleep_log(logger, logging.WARNING, operation_name),
        reraise=False,  # Wrap in RetryError so we can convert to our custom exception
    )

    try:
        async for attempt in retrying:
            with attempt:
                try:
                    return await asyncio.wait_for(func(), timeout=timeout_s)
                except TimeoutError:
                    raise SummaryTimeoutError(
                        f'{operation_name} timed out after {timeout_s}s',
                    ) from None
    except RetryError as e:
        last_exception = e.last_attempt.exception()
        if isinstance(last_exception, SummaryTimeoutError):
            raise last_exception from None
        raise SummaryRetryExhaustedError(
            f'{operation_name} failed after {max_attempts} attempts',
        ) from last_exception

    # This should never be reached, but satisfies type checker
    raise SummaryRetryExhaustedError(f'{operation_name} failed unexpectedly')


def compute_summary_total_timeout() -> float:
    """Compute worst-case wall-clock time for all retry attempts with safety margin.

    Formula: (max_attempts * timeout_s + total_backoff) * 1.1

    Returns:
        Total timeout in seconds for wrapping summary generation calls.
    """
    settings = get_settings()
    timeout_s = settings.summary.timeout_s
    max_attempts = settings.summary.retry_max_attempts
    base_delay_s = settings.summary.retry_base_delay_s

    attempt_time: float = max_attempts * timeout_s
    total_backoff: float = sum(
        min(base_delay_s * (2 ** i) + base_delay_s, 60.0)
        for i in range(max_attempts - 1)
    )
    return (attempt_time + total_backoff) * 1.1
