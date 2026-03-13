"""Tests for the summary retry wrapper using tenacity.

Tests verify:
- Successful first attempt (no retry needed)
- Retry on transient errors (ConnectionError, OSError)
- Timeout triggers retry
- All retries exhausted raises SummaryRetryExhaustedError
- All timeouts exhausted raises SummaryTimeoutError
- Non-retryable errors are not retried
- Operation name appears in error messages
- compute_summary_total_timeout() returns correct value
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from app.summary.retry import SummaryRetryExhaustedError
from app.summary.retry import SummaryTimeoutError
from app.summary.retry import compute_summary_total_timeout
from app.summary.retry import with_summary_retry_and_timeout


@pytest.fixture
def mock_summary_settings():
    """Mock summary settings for fast tests."""
    with patch('app.summary.retry.get_settings') as mock:
        mock.return_value.summary.timeout_s = 1.0
        mock.return_value.summary.retry_max_attempts = 3
        mock.return_value.summary.retry_base_delay_s = 0.01  # Fast retries for testing
        yield mock


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_successful_first_attempt() -> None:
    """Test successful execution on first attempt - no retry needed."""
    mock_func = AsyncMock(return_value='This is a summary.')

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'This is a summary.'
    assert mock_func.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_retry_on_connection_error() -> None:
    """Test retry on transient ConnectionError."""
    mock_func = AsyncMock(side_effect=[ConnectionError('Network error'), 'Summary text'])

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Summary text'
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_retry_on_os_error() -> None:
    """Test retry on transient OSError."""
    mock_func = AsyncMock(side_effect=[OSError('IO error'), 'Summary text'])

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Summary text'
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_timeout_triggers_retry() -> None:
    """Test that timeout triggers retry and eventually succeeds."""
    call_count = 0

    async def slow_then_fast() -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(10)  # Will timeout (settings have 1.0s timeout)
        return 'Summary text'

    result = await with_summary_retry_and_timeout(slow_then_fast, 'test_operation')

    assert result == 'Summary text'
    assert call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_exhausted_retries_raises_error() -> None:
    """Test all retries exhausted raises SummaryRetryExhaustedError."""
    mock_func = AsyncMock(side_effect=ConnectionError('Network error'))

    with pytest.raises(SummaryRetryExhaustedError) as exc_info:
        await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert 'failed after 3 attempts' in str(exc_info.value)
    assert mock_func.call_count == 3


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_timeout_exhausted_raises_timeout_error() -> None:
    """Test all timeouts exhausted raises SummaryTimeoutError."""

    async def always_slow() -> str:
        await asyncio.sleep(10)  # Will always timeout
        return 'Never reached'

    with pytest.raises(SummaryTimeoutError) as exc_info:
        await with_summary_retry_and_timeout(always_slow, 'test_operation')

    assert 'timed out after 1.0s' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_non_retryable_error_not_retried() -> None:
    """Test that non-retryable errors are not retried."""
    mock_func = AsyncMock(side_effect=ValueError('Bad input'))

    with pytest.raises(ValueError, match='Bad input'):
        await with_summary_retry_and_timeout(mock_func, 'test_operation')

    # ValueError is not in retry list, so only 1 attempt
    assert mock_func.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_operation_name_in_error_message() -> None:
    """Test that operation name appears in error messages."""
    mock_func = AsyncMock(side_effect=ConnectionError('Network error'))

    with pytest.raises(SummaryRetryExhaustedError) as exc_info:
        await with_summary_retry_and_timeout(mock_func, 'ollama_summarize')

    assert 'ollama_summarize' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_operation_name_in_timeout_message() -> None:
    """Test that operation name appears in timeout messages."""

    async def always_slow() -> str:
        await asyncio.sleep(10)
        return 'Never reached'

    with pytest.raises(SummaryTimeoutError) as exc_info:
        await with_summary_retry_and_timeout(always_slow, 'openai_summarize')

    assert 'openai_summarize' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_summary_settings')
async def test_multiple_failures_then_success() -> None:
    """Test recovery after multiple failures."""
    mock_func = AsyncMock(
        side_effect=[
            ConnectionError('Attempt 1 failed'),
            OSError('Attempt 2 failed'),
            'Final summary text',  # Attempt 3 succeeds
        ],
    )

    result = await with_summary_retry_and_timeout(mock_func, 'test_operation')

    assert result == 'Final summary text'
    assert mock_func.call_count == 3


def test_compute_summary_total_timeout() -> None:
    """Test compute_summary_total_timeout() returns a positive value."""
    with patch('app.summary.retry.get_settings') as mock:
        mock.return_value.summary.timeout_s = 30.0
        mock.return_value.summary.retry_max_attempts = 3
        mock.return_value.summary.retry_base_delay_s = 1.0

        total = compute_summary_total_timeout()

        # Formula: (max_attempts * timeout_s + total_backoff) * 1.1
        # = (3 * 30 + (min(1*1+1, 60) + min(1*2+1, 60))) * 1.1
        # = (90 + (2 + 3)) * 1.1
        # = 95 * 1.1
        # = 104.5
        assert total > 0
        assert total == pytest.approx(104.5)
