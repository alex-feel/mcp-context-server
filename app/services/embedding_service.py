"""
Embedding service for semantic search using Ollama and EmbeddingGemma.

This module provides embedding generation capabilities for semantic search,
wrapping the Ollama API and providing async-compatible interfaces.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.logger_config import config_logger
from app.settings import get_settings

if TYPE_CHECKING:
    import ollama

# Get settings
settings = get_settings()
# Configure logging
config_logger(settings.log_level)
logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for embedding generation via Ollama.

    This service handles embedding generation for semantic search using the
    EmbeddingGemma model via Ollama. All operations are async-compatible.
    """

    def __init__(self) -> None:
        """Initialize the embedding service."""
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self.ollama_host = settings.ollama_host
        self._client: ollama.Client | None = None

    def _get_client(self) -> ollama.Client:
        """Get or create Ollama client.

        Returns:
            Ollama client instance

        Raises:
            ImportError: If ollama package is not installed
        """
        if self._client is None:
            try:
                import ollama
            except ImportError as e:
                raise ImportError(
                    'ollama package is required for semantic search. Install with: uv sync --extra semantic-search',
                ) from e

            self._client = ollama.Client(host=self.ollama_host)
        return self._client

    async def is_available(self) -> bool:
        """Check if embedding model is available (quick check, max 100ms).

        Returns:
            True if model is available, False otherwise
        """
        loop = asyncio.get_event_loop()

        def _check() -> bool:
            try:
                client = self._get_client()
                # Try to show model info to verify it exists
                client.show(self.model)
                return True
            except Exception as e:
                logger.debug(f'Embedding model not available: {e}')
                return False

        try:
            # Run with timeout to ensure quick check
            return await asyncio.wait_for(loop.run_in_executor(None, _check), timeout=0.1)
        except TimeoutError:
            logger.warning('Embedding model availability check timed out')
            return False

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate single embedding (50-150ms).

        Args:
            text: Text to embed

        Returns:
            768-dimensional embedding vector
        """
        loop = asyncio.get_event_loop()

        def _generate() -> list[float]:
            try:
                client = self._get_client()
                response = client.embed(model=self.model, input=text)

                # Extract embedding from response
                if hasattr(response, 'embeddings') and response.embeddings:
                    # Use .tolist() to convert numpy.float32 to Python float
                    # asyncpg with pgvector requires Python float, not numpy.float32
                    # Cast to Any for runtime type check - Ollama returns numpy array but types say Sequence
                    emb = response.embeddings[0]
                    embedding = cast(Any, emb).tolist() if hasattr(emb, 'tolist') else list(emb)
                else:
                    raise RuntimeError(f'Unexpected embedding response format: {type(response)}')

                # Validate dimensions
                if len(embedding) != self.dim:
                    error_msg = (
                        f'Embedding dimension mismatch: expected {self.dim}, got {len(embedding)}. '
                        f'This likely indicates a model mismatch. '
                        f'Ensure EMBEDDING_MODEL ({self.model}) produces {self.dim}-dimensional vectors, '
                        f'or update EMBEDDING_DIM to match your model output.'
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                return embedding

            except Exception as e:
                logger.error(f'Failed to generate embedding: {e}')
                raise RuntimeError(f'Embedding generation failed: {e}') from e

        return await loop.run_in_executor(None, _generate)

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings (200-500ms for 10 texts).

        Args:
            texts: List of texts to embed

        Returns:
            List of 768-dimensional embedding vectors
        """
        loop = asyncio.get_event_loop()

        def _generate_batch() -> list[list[float]]:
            try:
                client = self._get_client()
                response = client.embed(model=self.model, input=texts)

                # Extract embeddings from response
                if hasattr(response, 'embeddings'):
                    # Use .tolist() to convert numpy.float32 to Python float
                    # asyncpg with pgvector requires Python float, not numpy.float32
                    # Cast to Any for runtime type check - Ollama returns numpy array but types say Sequence
                    embeddings = [
                        cast(Any, emb).tolist() if hasattr(emb, 'tolist') else list(emb)
                        for emb in response.embeddings
                    ]
                else:
                    raise RuntimeError(f'Unexpected embedding response format: {type(response)}')

                # Validate dimensions
                for idx, embedding in enumerate(embeddings):
                    if len(embedding) != self.dim:
                        error_msg = (
                            f'Embedding {idx} dimension mismatch: expected {self.dim}, got {len(embedding)}. '
                            f'This likely indicates a model mismatch. '
                            f'Ensure EMBEDDING_MODEL ({self.model}) produces {self.dim}-dimensional vectors, '
                            f'or update EMBEDDING_DIM to match your model output.'
                        )
                        logger.error(error_msg)
                        raise ValueError(error_msg)

                return embeddings

            except Exception as e:
                logger.error(f'Failed to generate batch embeddings: {e}')
                raise RuntimeError(f'Batch embedding generation failed: {e}') from e

        return await loop.run_in_executor(None, _generate_batch)

    def get_dimension(self) -> int:
        """Get embedding dimensions.

        Returns:
            Embedding dimension (768 for EmbeddingGemma)
        """
        return self.dim
