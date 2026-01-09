"""
Embedding service for semantic search using Ollama.

This module provides embedding generation capabilities for semantic search,
wrapping the Ollama API and providing async-compatible interfaces with
configurable timeout and retry logic.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import Any
from typing import TypeVar
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

T = TypeVar('T')


class EmbeddingService:
    """Service for embedding generation via Ollama.

    This service handles embedding generation for semantic search using
    the configured embedding model via Ollama. All operations are async-compatible.

    Features:
    - Configurable timeout for API calls
    - Automatic retry with exponential backoff and jitter
    - Graceful degradation on transient failures
    """

    def __init__(self) -> None:
        """Initialize the embedding service."""
        self.model = settings.embedding_model
        self.dim = settings.embedding_dim
        self.ollama_host = settings.ollama_host
        self.timeout = settings.embedding_timeout_s
        self.max_retries = settings.embedding_retry_max_attempts
        self.base_delay = settings.embedding_retry_base_delay_s
        self._client: ollama.Client | None = None

    def _get_client(self) -> ollama.Client:
        """Get or create Ollama client with timeout configuration.

        Returns:
            Ollama client instance with configured timeout

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

            self._client = ollama.Client(
                host=self.ollama_host,
                timeout=self.timeout,
            )
        return self._client

    async def is_available(self) -> bool:
        """Check if embedding model is available (quick check, max 500ms).

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
            # Increased timeout for more reliable availability check
            return await asyncio.wait_for(loop.run_in_executor(None, _check), timeout=0.5)
        except TimeoutError:
            logger.warning('Embedding model availability check timed out')
            return False

    async def _execute_with_retry(
        self,
        operation: str,
        func: Callable[[], Awaitable[T]],
    ) -> T:
        """Execute operation with retry logic and exponential backoff.

        Args:
            operation: Description of the operation for logging
            func: The async function to execute

        Returns:
            The result of the function

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_exception: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                return await func()
            except TimeoutError as e:
                last_exception = e
                logger.warning(
                    f'{operation} timed out (attempt {attempt + 1}/{self.max_retries})',
                )
            except Exception as e:
                last_exception = e
                error_msg = str(e) or repr(e) or type(e).__name__
                logger.warning(
                    f'{operation} failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}',
                )

            if attempt < self.max_retries - 1:
                # Calculate delay with exponential backoff and jitter
                delay = self.base_delay * (2 ** attempt)
                jitter = random.uniform(0, delay * 0.1)
                total_delay = min(delay + jitter, 30.0)  # Cap at 30 seconds
                logger.debug(f'Retrying {operation} in {total_delay:.2f}s')
                await asyncio.sleep(total_delay)

        # All retries exhausted
        error_msg = str(last_exception) or repr(last_exception) or 'Unknown error'
        raise RuntimeError(
            f'{operation} failed after {self.max_retries} attempts: {error_msg}',
        ) from last_exception

    async def generate_embedding(self, text: str) -> list[float]:
        """Generate single embedding with timeout and retry.

        Args:
            text: Text to embed

        Returns:
            Embedding vector with configured dimensions
        """
        async def _generate_with_timeout() -> list[float]:
            loop = asyncio.get_event_loop()

            def _generate() -> list[float]:
                client = self._get_client()
                response = client.embed(model=self.model, input=text)

                # Extract embedding from response
                if hasattr(response, 'embeddings') and response.embeddings:
                    # Use .tolist() to convert numpy.float32 to Python float
                    # asyncpg with pgvector requires Python float, not numpy.float32
                    emb = response.embeddings[0]
                    embedding = cast(Any, emb).tolist() if hasattr(emb, 'tolist') else list(emb)
                else:
                    raise RuntimeError(f'Unexpected embedding response format: {type(response)}')

                # Validate dimensions
                if len(embedding) != self.dim:
                    error_msg = (
                        f'Embedding dimension mismatch: expected {self.dim}, got {len(embedding)}. '
                        f'Ensure EMBEDDING_MODEL ({self.model}) produces {self.dim}-dimensional vectors.'
                    )
                    raise ValueError(error_msg)

                return embedding

            # Apply timeout to the executor call
            return await asyncio.wait_for(
                loop.run_in_executor(None, _generate),
                timeout=self.timeout,
            )

        return await self._execute_with_retry(
            'Embedding generation',
            _generate_with_timeout,
        )

    async def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Generate batch embeddings with timeout and retry.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors with configured dimensions
        """
        async def _generate_batch_with_timeout() -> list[list[float]]:
            loop = asyncio.get_event_loop()

            def _generate_batch() -> list[list[float]]:
                client = self._get_client()
                response = client.embed(model=self.model, input=texts)

                # Extract embeddings from response
                if hasattr(response, 'embeddings'):
                    # Use .tolist() to convert numpy.float32 to Python float
                    # asyncpg with pgvector requires Python float, not numpy.float32
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
                            f'Embedding {idx} dimension mismatch: expected {self.dim}, '
                            f'got {len(embedding)}.'
                        )
                        raise ValueError(error_msg)

                return embeddings

            # Apply timeout to the executor call - longer timeout for batch operations
            return await asyncio.wait_for(
                loop.run_in_executor(None, _generate_batch),
                timeout=self.timeout * 2,
            )

        return await self._execute_with_retry(
            'Batch embedding generation',
            _generate_batch_with_timeout,
        )

    def get_dimension(self) -> int:
        """Get embedding dimensions.

        Returns:
            Embedding dimension (configured via EMBEDDING_DIM)
        """
        return self.dim
