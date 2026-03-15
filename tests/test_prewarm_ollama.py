"""
Tests for prewarm_ollama_models() startup function.

Tests verify:
- No-op when no Ollama providers configured
- Embedding model pre-warming sends correct API request
- Summary model pre-warming sends correct API request
- Both models pre-warmed when both use Ollama
- Deduplication when embedding and summary use the same model
- Graceful failure handling (logs warning, does not raise)
- Correct timeout configuration (120s)
"""

from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


@pytest.fixture
def mock_settings_no_ollama():
    """Settings where neither provider uses Ollama."""
    mock = MagicMock()
    mock.embedding.generation_enabled = True
    mock.embedding.provider = 'openai'
    mock.embedding.model = 'text-embedding-3-small'
    mock.summary.generation_enabled = True
    mock.summary.provider = 'openai'
    mock.summary.model = 'gpt-4o-mini'
    mock.ollama.host = 'http://localhost:11434'
    return mock


@pytest.fixture
def mock_settings_embedding_ollama():
    """Settings where only embedding uses Ollama."""
    mock = MagicMock()
    mock.embedding.generation_enabled = True
    mock.embedding.provider = 'ollama'
    mock.embedding.model = 'qwen3-embedding:0.6b'
    mock.summary.generation_enabled = True
    mock.summary.provider = 'openai'
    mock.summary.model = 'gpt-4o-mini'
    mock.ollama.host = 'http://localhost:11434'
    return mock


@pytest.fixture
def mock_settings_summary_ollama():
    """Settings where only summary uses Ollama."""
    mock = MagicMock()
    mock.embedding.generation_enabled = True
    mock.embedding.provider = 'openai'
    mock.embedding.model = 'text-embedding-3-small'
    mock.summary.generation_enabled = True
    mock.summary.provider = 'ollama'
    mock.summary.model = 'qwen3:0.6b'
    mock.summary.ollama_num_ctx = 32768
    mock.ollama.host = 'http://localhost:11434'
    return mock


@pytest.fixture
def mock_settings_both_ollama():
    """Settings where both providers use Ollama with different models."""
    mock = MagicMock()
    mock.embedding.generation_enabled = True
    mock.embedding.provider = 'ollama'
    mock.embedding.model = 'qwen3-embedding:0.6b'
    mock.summary.generation_enabled = True
    mock.summary.provider = 'ollama'
    mock.summary.model = 'qwen3:0.6b'
    mock.summary.ollama_num_ctx = 32768
    mock.ollama.host = 'http://localhost:11434'
    return mock


@pytest.fixture
def mock_settings_same_model():
    """Settings where both providers use Ollama with the SAME model."""
    mock = MagicMock()
    mock.embedding.generation_enabled = True
    mock.embedding.provider = 'ollama'
    mock.embedding.model = 'shared-model:latest'
    mock.summary.generation_enabled = True
    mock.summary.provider = 'ollama'
    mock.summary.model = 'shared-model:latest'
    mock.summary.ollama_num_ctx = 32768
    mock.ollama.host = 'http://localhost:11434'
    return mock


@pytest.fixture
def mock_settings_disabled_generation():
    """Settings where generation is disabled."""
    mock = MagicMock()
    mock.embedding.generation_enabled = False
    mock.embedding.provider = 'ollama'
    mock.embedding.model = 'qwen3-embedding:0.6b'
    mock.summary.generation_enabled = False
    mock.summary.provider = 'ollama'
    mock.summary.model = 'qwen3:0.6b'
    mock.ollama.host = 'http://localhost:11434'
    return mock


def _make_mock_client(post_return=None, post_side_effect=None):
    """Create a mock httpx.AsyncClient with async context manager support."""
    mock_client = AsyncMock()
    if post_side_effect is not None:
        mock_client.post = AsyncMock(side_effect=post_side_effect)
    else:
        mock_client.post = AsyncMock(return_value=post_return)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _make_mock_response():
    """Create a mock httpx response with raise_for_status."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    return mock_response


class TestPrewarmOllamaModels:
    """Tests for prewarm_ollama_models() function."""

    @pytest.mark.asyncio
    async def test_noop_when_no_ollama_providers(self, mock_settings_no_ollama):
        """Test that function returns immediately when no Ollama providers configured."""
        with patch('app.startup.settings', mock_settings_no_ollama):
            import httpx

            with patch.object(httpx, 'AsyncClient') as mock_async_client:
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                # AsyncClient should never be created
                mock_async_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_noop_when_generation_disabled(self, mock_settings_disabled_generation):
        """Test that function returns immediately when generation is disabled."""
        with patch('app.startup.settings', mock_settings_disabled_generation):
            import httpx

            with patch.object(httpx, 'AsyncClient') as mock_async_client:
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                mock_async_client.assert_not_called()

    @pytest.mark.asyncio
    async def test_warms_embedding_model(self, mock_settings_embedding_ollama):
        """Test pre-warming an embedding model sends POST /api/embed."""
        mock_response = _make_mock_response()
        mock_client = _make_mock_client(post_return=mock_response)

        with patch('app.startup.settings', mock_settings_embedding_ollama):
            import httpx

            with patch.object(httpx, 'AsyncClient', return_value=mock_client):
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                mock_client.post.assert_called_once_with(
                    '/api/embed',
                    json={'model': 'qwen3-embedding:0.6b', 'input': 'warmup'},
                )
                mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_warms_summary_model(self, mock_settings_summary_ollama):
        """Test pre-warming a summary model sends POST /api/chat with options."""
        mock_response = _make_mock_response()
        mock_client = _make_mock_client(post_return=mock_response)

        with patch('app.startup.settings', mock_settings_summary_ollama):
            import httpx

            with patch.object(httpx, 'AsyncClient', return_value=mock_client):
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                mock_client.post.assert_called_once_with(
                    '/api/chat',
                    json={
                        'model': 'qwen3:0.6b',
                        'messages': [],
                        'stream': False,
                        'options': {'num_ctx': 32768},
                    },
                )
                mock_response.raise_for_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_warms_both_models(self, mock_settings_both_ollama):
        """Test pre-warming both embedding and summary models."""
        mock_response = _make_mock_response()
        mock_client = _make_mock_client(post_return=mock_response)

        with patch('app.startup.settings', mock_settings_both_ollama):
            import httpx

            with patch.object(httpx, 'AsyncClient', return_value=mock_client):
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                assert mock_client.post.call_count == 2

                # First call: embedding model
                first_call = mock_client.post.call_args_list[0]
                assert first_call[0][0] == '/api/embed'
                assert first_call[1]['json']['model'] == 'qwen3-embedding:0.6b'

                # Second call: summary model via /api/chat
                second_call = mock_client.post.call_args_list[1]
                assert second_call[0][0] == '/api/chat'
                assert second_call[1]['json']['model'] == 'qwen3:0.6b'
                assert second_call[1]['json']['messages'] == []
                assert second_call[1]['json']['options'] == {'num_ctx': 32768}

    @pytest.mark.asyncio
    async def test_deduplicates_same_model(self, mock_settings_same_model):
        """Test that same model used for both embedding and summary is only warmed once."""
        mock_response = _make_mock_response()
        mock_client = _make_mock_client(post_return=mock_response)

        with patch('app.startup.settings', mock_settings_same_model):
            import httpx

            with patch.object(httpx, 'AsyncClient', return_value=mock_client):
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                # Should only be called once due to deduplication
                mock_client.post.assert_called_once()

                # First occurrence is embedding, so it should use /api/embed
                call = mock_client.post.call_args_list[0]
                assert call[0][0] == '/api/embed'
                assert call[1]['json']['model'] == 'shared-model:latest'

    @pytest.mark.asyncio
    async def test_failure_does_not_raise(self, mock_settings_embedding_ollama):
        """Test that connection failure logs warning but does not raise."""
        mock_client = _make_mock_client(post_side_effect=Exception('Connection refused'))

        with (
            patch('app.startup.settings', mock_settings_embedding_ollama),
            patch('app.startup.logger') as mock_logger,
        ):
            import httpx

            with patch.object(httpx, 'AsyncClient', return_value=mock_client):
                from app.startup import prewarm_ollama_models

                # Should NOT raise
                await prewarm_ollama_models()

                # Should log warning about the failure
                mock_logger.warning.assert_called_once()
                warning_msg = mock_logger.warning.call_args[0][0]
                assert 'Failed to pre-warm model' in warning_msg
                assert 'Connection refused' in warning_msg

    @pytest.mark.asyncio
    async def test_http_error_does_not_raise(self, mock_settings_embedding_ollama):
        """Test that HTTP error response logs warning but does not raise."""
        import httpx

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock(
            side_effect=httpx.HTTPStatusError(
                'Server Error',
                request=MagicMock(),
                response=MagicMock(status_code=500),
            ),
        )
        mock_client = _make_mock_client(post_return=mock_response)

        with (
            patch('app.startup.settings', mock_settings_embedding_ollama),
            patch.object(httpx, 'AsyncClient', return_value=mock_client),
            patch('app.startup.logger') as mock_logger,
        ):
            from app.startup import prewarm_ollama_models

            # Should NOT raise
            await prewarm_ollama_models()

            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_timeout_configuration(self, mock_settings_embedding_ollama):
        """Test that httpx.AsyncClient is configured with 120s timeout."""
        mock_response = _make_mock_response()
        mock_client = _make_mock_client(post_return=mock_response)

        with patch('app.startup.settings', mock_settings_embedding_ollama):
            import httpx

            with patch.object(
                httpx, 'AsyncClient', return_value=mock_client,
            ) as mock_constructor:
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                mock_constructor.assert_called_once_with(
                    base_url='http://localhost:11434',
                    timeout=120.0,
                )

    @pytest.mark.asyncio
    async def test_partial_failure_continues(self, mock_settings_both_ollama):
        """Test that failure on first model does not prevent warming the second."""
        mock_response_success = _make_mock_response()

        # First call fails, second succeeds
        mock_client = _make_mock_client(
            post_side_effect=[
                Exception('First model failed'),
                mock_response_success,
            ],
        )

        with (
            patch('app.startup.settings', mock_settings_both_ollama),
            patch('app.startup.logger') as mock_logger,
        ):
            import httpx

            with patch.object(httpx, 'AsyncClient', return_value=mock_client):
                from app.startup import prewarm_ollama_models

                await prewarm_ollama_models()

                # Both models should have been attempted
                assert mock_client.post.call_count == 2

                # Should have warning for first model failure
                mock_logger.warning.assert_called_once()

                # Should have success log for second model
                info_calls = [c[0][0] for c in mock_logger.info.call_args_list]
                assert any('pre-warmed successfully' in msg for msg in info_calls)
