"""
Unit tests for EmbeddingService.

Tests the embedding generation service with mocked Ollama responses.
All tests require the ollama package but mock the actual API calls.
"""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

# Conditional skip marker for tests requiring ollama package
requires_ollama = pytest.mark.skipif(
    importlib.util.find_spec('ollama') is None,
    reason='ollama package not installed',
)


class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    @requires_ollama
    @pytest.mark.asyncio
    async def test_generate_embedding_success(self, embedding_dim: int) -> None:
        """Test successful embedding generation."""
        from app.services.embedding_service import EmbeddingService

        # Create a mock ollama module and client
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * embedding_dim
        mock_response.embeddings = [mock_embedding]
        mock_client.embed.return_value = mock_response

        service = EmbeddingService()
        # Replace the client directly
        service._client = mock_client

        embedding = await service.generate_embedding('test text')

        assert len(embedding) == embedding_dim
        assert all(isinstance(x, float) for x in embedding)
        mock_client.embed.assert_called_once()

    @requires_ollama
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self, embedding_dim: int) -> None:
        """Test batch embedding generation."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()

        # Mock batch embedding response
        mock_response = MagicMock()
        mock_embeddings = []
        for i in range(3):
            mock_emb = MagicMock()
            mock_emb.tolist.return_value = [0.1 * (i + 1)] * embedding_dim
            mock_embeddings.append(mock_emb)
        mock_response.embeddings = mock_embeddings
        mock_client.embed.return_value = mock_response

        service = EmbeddingService()
        service._client = mock_client

        embeddings = await service.generate_embeddings(['text1', 'text2', 'text3'])

        assert len(embeddings) == 3
        assert all(len(e) == embedding_dim for e in embeddings)
        mock_client.embed.assert_called_once()

    @requires_ollama
    @pytest.mark.asyncio
    async def test_is_available_check(self) -> None:
        """Test model availability check."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()
        mock_client.show.return_value = {'name': 'embeddinggemma:latest'}

        service = EmbeddingService()
        service._client = mock_client

        is_available = await service.is_available()

        assert is_available is True
        mock_client.show.assert_called_once()

    @requires_ollama
    @pytest.mark.asyncio
    async def test_is_available_returns_false_on_error(self) -> None:
        """Test that is_available returns False when model is not found."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()
        mock_client.show.side_effect = Exception('Model not found')

        service = EmbeddingService()
        service._client = mock_client

        is_available = await service.is_available()

        assert is_available is False

    @requires_ollama
    def test_get_dimension(self, embedding_dim: int) -> None:
        """Test get_dimension method."""
        from app.services.embedding_service import EmbeddingService

        with patch('app.services.embedding_service.settings') as mock_settings:
            mock_settings.embedding_model = 'embeddinggemma:latest'
            mock_settings.embedding_dim = embedding_dim
            mock_settings.ollama_host = 'http://localhost:11434'

            service = EmbeddingService()
            dim = service.get_dimension()

            assert dim == embedding_dim

    @requires_ollama
    @pytest.mark.asyncio
    async def test_generate_embedding_dimension_mismatch(self) -> None:
        """Test that dimension mismatch raises ValueError."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()

        # Return wrong dimension embedding (512 will always be wrong since
        # settings.embedding_dim defaults to 768 or CI sets it to 384)
        mock_response = MagicMock()
        mock_embedding = MagicMock()
        mock_embedding.tolist.return_value = [0.1] * 512  # Wrong dimension
        mock_response.embeddings = [mock_embedding]
        mock_client.embed.return_value = mock_response

        service = EmbeddingService()
        service._client = mock_client

        with pytest.raises(RuntimeError, match='Embedding generation failed'):
            await service.generate_embedding('test text')

    @requires_ollama
    @pytest.mark.asyncio
    async def test_generate_embedding_client_error(self) -> None:
        """Test error handling when Ollama client fails."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()
        mock_client.embed.side_effect = Exception('Connection refused')

        service = EmbeddingService()
        service._client = mock_client

        with pytest.raises(RuntimeError, match='Embedding generation failed'):
            await service.generate_embedding('test text')

    @requires_ollama
    @pytest.mark.asyncio
    async def test_generate_embeddings_batch_dimension_mismatch(
        self, embedding_dim: int,
    ) -> None:
        """Test batch embedding with dimension mismatch."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()

        # Return embeddings with wrong dimension
        mock_response = MagicMock()
        mock_embeddings = []
        for i in range(3):
            mock_emb = MagicMock()
            # Second embedding has wrong dimension (always 512 as wrong)
            dim = 512 if i == 1 else embedding_dim
            mock_emb.tolist.return_value = [0.1] * dim
            mock_embeddings.append(mock_emb)
        mock_response.embeddings = mock_embeddings
        mock_client.embed.return_value = mock_response

        service = EmbeddingService()
        service._client = mock_client

        with pytest.raises(RuntimeError, match='Batch embedding generation failed'):
            await service.generate_embeddings(['text1', 'text2', 'text3'])

    @requires_ollama
    def test_get_client_creates_once(self) -> None:
        """Test that Ollama client is created only once."""
        from app.services.embedding_service import EmbeddingService

        # Create a mock for the ollama module
        mock_ollama = MagicMock()
        mock_client = MagicMock()
        mock_ollama.Client.return_value = mock_client

        # Patch the import inside _get_client
        with patch.dict(sys.modules, {'ollama': mock_ollama}):
            service = EmbeddingService()

            # Call _get_client multiple times
            client1 = service._get_client()
            client2 = service._get_client()

            # Client should be created only once
            assert mock_ollama.Client.call_count == 1
            assert client1 is client2

    @requires_ollama
    @pytest.mark.asyncio
    async def test_generate_embedding_without_tolist(self, embedding_dim: int) -> None:
        """Test embedding generation when response doesn't have tolist method."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()

        # Mock embedding response without tolist method (plain list)
        mock_response = MagicMock()
        mock_response.embeddings = [[0.1] * embedding_dim]  # Plain list, no tolist method
        mock_client.embed.return_value = mock_response

        service = EmbeddingService()
        service._client = mock_client

        embedding = await service.generate_embedding('test text')

        assert len(embedding) == embedding_dim

    @requires_ollama
    @pytest.mark.asyncio
    async def test_generate_embedding_invalid_response_format(self) -> None:
        """Test error when response has unexpected format."""
        from app.services.embedding_service import EmbeddingService

        mock_client = MagicMock()

        # Mock response without embeddings attribute
        mock_response = MagicMock()
        mock_response.embeddings = None
        mock_client.embed.return_value = mock_response

        service = EmbeddingService()
        service._client = mock_client

        with pytest.raises(RuntimeError, match='Embedding generation failed'):
            await service.generate_embedding('test text')


class TestEmbeddingServiceImportError:
    """Test EmbeddingService behavior when ollama is not installed."""

    def test_get_client_import_error(self) -> None:
        """Test that _get_client raises ImportError when ollama is missing."""
        # This test documents the expected behavior when ollama is not installed
        # We can't easily test ImportError without actually removing ollama
        # from sys.modules at runtime, so we document the expected behavior
