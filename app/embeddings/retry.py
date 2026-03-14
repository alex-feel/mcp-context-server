"""
Universal retry wrapper for embedding providers using tenacity.

All LangChain providers have their internal retry disabled (max_retries=0).
This wrapper provides the ONLY retry logic for embedding operations.

Uses tenacity library for robust async retry logic with:
- Exponential backoff with jitter
- Configurable max attempts and timeouts
- Per-attempt timeout using asyncio.wait_for()
- Automatic logging before retry sleep

Retryable exceptions include:
- Network errors: httpx.TimeoutException, httpx.NetworkError, ConnectionError, OSError
- OpenAI SDK: APITimeoutError, APIConnectionError, RateLimitError, InternalServerError
- Ollama SDK: ResponseError (server errors)
- HuggingFace Hub: HfHubHTTPError
- Custom: EmbeddingTimeoutError, asyncio.TimeoutError
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


class EmbeddingTimeoutError(Exception):
    """Raised when embedding generation times out."""


class EmbeddingRetryExhaustedError(Exception):
    """Raised when all retry attempts are exhausted."""


# Base exceptions that are always retryable (statically typed, always available)
_BASE_RETRYABLE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    # Python built-in
    asyncio.TimeoutError,
    ConnectionError,
    OSError,
    # Our custom exception
    EmbeddingTimeoutError,
    # httpx exceptions (always available since it's a core dependency)
    httpx.TimeoutException,
    httpx.NetworkError,
)

# Provider-specific exceptions identified by module and class name
# This avoids importing optional packages at module load time
_RETRYABLE_BY_MODULE: dict[str, frozenset[str]] = {
    # OpenAI SDK exceptions
    'openai': frozenset({
        'APITimeoutError',
        'APIConnectionError',
        'RateLimitError',
        'InternalServerError',
    }),
    # Ollama SDK exceptions
    'ollama': frozenset({
        'ResponseError',
    }),
    # HuggingFace Hub exceptions
    'huggingface_hub': frozenset({
        'HfHubHTTPError',
    }),
    'huggingface_hub.errors': frozenset({
        'HfHubHTTPError',
    }),
}


def _is_retryable_exception(exc: BaseException) -> bool:
    """Check if an exception should trigger a retry.

    This function checks both statically-typed base exceptions and
    dynamically-detected provider-specific exceptions by module/class name.

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
        | EmbeddingTimeoutError
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


async def with_retry_and_timeout[T](
    func: Callable[[], Awaitable[T]],
    operation_name: str = 'embedding',
) -> T:
    """Execute async function with retry and timeout.

    Uses tenacity for retry logic with:
    - Exponential backoff with jitter
    - Configurable max attempts
    - Per-attempt timeout
    - Logging before each retry sleep

    Settings from EmbeddingSettings:
    - EMBEDDING_TIMEOUT_S: Timeout per attempt
    - EMBEDDING_RETRY_MAX_ATTEMPTS: Maximum retry attempts
    - EMBEDDING_RETRY_BASE_DELAY_S: Base delay for exponential backoff

    Args:
        func: Async function to execute (no arguments)
        operation_name: Description for logging

    Returns:
        Result from func

    Raises:
        EmbeddingTimeoutError: If final attempt times out
        EmbeddingRetryExhaustedError: If all retries exhausted due to errors
    """
    settings = get_settings()
    timeout_s = settings.embedding.timeout_s
    max_attempts = settings.embedding.retry_max_attempts
    base_delay_s = settings.embedding.retry_base_delay_s

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
                    raise EmbeddingTimeoutError(
                        f'{operation_name} timed out after {timeout_s}s',
                    ) from None
    except RetryError as e:
        last_exception = e.last_attempt.exception()
        if isinstance(last_exception, EmbeddingTimeoutError):
            raise last_exception from None
        raise EmbeddingRetryExhaustedError(
            f'{operation_name} failed after {max_attempts} attempts',
        ) from last_exception

    # This should never be reached, but satisfies type checker
    raise EmbeddingRetryExhaustedError(f'{operation_name} failed unexpectedly')


def compute_embedding_total_timeout() -> float:
    """Compute worst-case wall-clock time for all retry attempts with safety margin.

    Formula: (max_attempts * timeout_s + total_backoff) * 1.1

    The total backoff accounts for exponential backoff with jitter,
    capped at 60 seconds per wait interval (matching tenacity's max=60.0).
    The 10% safety margin ensures tenacity's internal per-attempt timeout
    fires before the outer asyncio.wait_for, producing more informative
    error messages (EmbeddingTimeoutError vs generic TimeoutError).

    Returns:
        Total timeout in seconds for wrapping embedding generation calls.
    """
    settings = get_settings()
    timeout_s = settings.embedding.timeout_s
    max_attempts = settings.embedding.retry_max_attempts
    base_delay_s = settings.embedding.retry_base_delay_s

    attempt_time: float = max_attempts * timeout_s
    total_backoff: float = sum(
        min(base_delay_s * (2 ** i) + base_delay_s, 60.0)
        for i in range(max_attempts - 1)
    )
    return (attempt_time + total_backoff) * 1.1
