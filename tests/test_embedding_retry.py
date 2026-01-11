"""
Tests for the universal embedding retry wrapper using tenacity.

Tests verify:
- Successful first attempt (no retry needed)
- Retry on transient errors (ConnectionError, OSError)
- Timeout triggers retry
- All retries exhausted raises EmbeddingRetryExhaustedError
- All timeouts exhausted raises EmbeddingTimeoutError
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from app.embeddings.retry import EmbeddingRetryExhaustedError
from app.embeddings.retry import EmbeddingTimeoutError
from app.embeddings.retry import with_retry_and_timeout


@pytest.fixture
def mock_embedding_settings():
    """Mock embedding settings for fast tests."""
    with patch('app.embeddings.retry.get_settings') as mock:
        mock.return_value.embedding.timeout_s = 1.0
        mock.return_value.embedding.retry_max_attempts = 3
        mock.return_value.embedding.retry_base_delay_s = 0.01  # Fast retries for testing
        yield mock


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_successful_first_attempt() -> None:
    """Test successful execution on first attempt - no retry needed."""
    mock_func = AsyncMock(return_value=[0.1, 0.2, 0.3])

    result = await with_retry_and_timeout(mock_func, 'test_operation')

    assert result == [0.1, 0.2, 0.3]
    assert mock_func.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_retry_on_connection_error() -> None:
    """Test retry on transient ConnectionError."""
    mock_func = AsyncMock(side_effect=[ConnectionError('Network error'), [0.1, 0.2]])

    result = await with_retry_and_timeout(mock_func, 'test_operation')

    assert result == [0.1, 0.2]
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_retry_on_os_error() -> None:
    """Test retry on transient OSError."""
    mock_func = AsyncMock(side_effect=[OSError('IO error'), [0.1, 0.2]])

    result = await with_retry_and_timeout(mock_func, 'test_operation')

    assert result == [0.1, 0.2]
    assert mock_func.call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_timeout_triggers_retry() -> None:
    """Test that timeout triggers retry and eventually succeeds."""
    call_count = 0

    async def slow_then_fast() -> list[float]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            await asyncio.sleep(10)  # Will timeout (settings have 1.0s timeout)
        return [0.1, 0.2]

    result = await with_retry_and_timeout(slow_then_fast, 'test_operation')

    assert result == [0.1, 0.2]
    assert call_count == 2


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_exhausted_retries_raises_error() -> None:
    """Test all retries exhausted raises EmbeddingRetryExhaustedError."""
    mock_func = AsyncMock(side_effect=ConnectionError('Network error'))

    with pytest.raises(EmbeddingRetryExhaustedError) as exc_info:
        await with_retry_and_timeout(mock_func, 'test_operation')

    assert 'failed after 3 attempts' in str(exc_info.value)
    assert mock_func.call_count == 3


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_timeout_exhausted_raises_timeout_error() -> None:
    """Test all timeouts exhausted raises EmbeddingTimeoutError."""

    async def always_slow() -> list[float]:
        await asyncio.sleep(10)  # Will always timeout
        return [0.1]

    with pytest.raises(EmbeddingTimeoutError) as exc_info:
        await with_retry_and_timeout(always_slow, 'test_operation')

    assert 'timed out after 1.0s' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_non_retryable_error_not_retried() -> None:
    """Test that non-retryable errors are not retried."""
    mock_func = AsyncMock(side_effect=ValueError('Bad input'))

    with pytest.raises(ValueError, match='Bad input'):
        await with_retry_and_timeout(mock_func, 'test_operation')

    # ValueError is not in retry list, so only 1 attempt
    assert mock_func.call_count == 1


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_operation_name_in_error_message() -> None:
    """Test that operation name appears in error messages."""
    mock_func = AsyncMock(side_effect=ConnectionError('Network error'))

    with pytest.raises(EmbeddingRetryExhaustedError) as exc_info:
        await with_retry_and_timeout(mock_func, 'ollama_embed_query')

    assert 'ollama_embed_query' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_operation_name_in_timeout_message() -> None:
    """Test that operation name appears in timeout messages."""

    async def always_slow() -> list[float]:
        await asyncio.sleep(10)
        return [0.1]

    with pytest.raises(EmbeddingTimeoutError) as exc_info:
        await with_retry_and_timeout(always_slow, 'openai_embed_documents')

    assert 'openai_embed_documents' in str(exc_info.value)


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_embedding_settings')
async def test_multiple_failures_then_success() -> None:
    """Test recovery after multiple failures."""
    mock_func = AsyncMock(
        side_effect=[
            ConnectionError('Attempt 1 failed'),
            OSError('Attempt 2 failed'),
            [0.1, 0.2, 0.3],  # Attempt 3 succeeds
        ],
    )

    result = await with_retry_and_timeout(mock_func, 'test_operation')

    assert result == [0.1, 0.2, 0.3]
    assert mock_func.call_count == 3
