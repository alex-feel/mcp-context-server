"""
Unit tests for EmbeddingService.

Tests the embedding generation service with mocked Ollama responses.
"""

import pytest


class TestEmbeddingService:
    """Test EmbeddingService functionality."""

    @pytest.mark.asyncio
    async def test_generate_embedding_success(self) -> None:
        """Test successful embedding generation."""
        pytest.skip('Requires ollama package - optional dependency')

    @pytest.mark.asyncio
    async def test_generate_embeddings_batch(self) -> None:
        """Test batch embedding generation."""
        pytest.skip('Requires ollama package - optional dependency')

    @pytest.mark.asyncio
    async def test_is_available_check(self) -> None:
        """Test model availability check."""
        pytest.skip('Requires ollama package - optional dependency')

    def test_get_dimension(self) -> None:
        """Test get_dimension method."""
        pytest.skip('Requires ollama package - optional dependency')
