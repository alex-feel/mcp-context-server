"""Shared test helper functions.

Provides utility functions used across test infrastructure files
(conftest.py, run_server.py) and test modules to avoid code duplication.
Application modules are imported inside the helpers, so importing this
module never loads the application or reads settings.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.repositories.embedding_repository import EmbeddingRepository


def is_ollama_model_available(
    model: str | None = None,
    host: str | None = None,
) -> bool:
    """Check if an Ollama model is available for testing.

    Performs two checks:
    1. Ollama service is running at the resolved host
    2. The specified model (or any candidate model) is installed

    Args:
        model: Model name to check. If None, checks candidate models
            in priority order: all-minilm, qwen3-embedding:0.6b.
        host: Ollama host URL. If None, resolves from settings
            (default: http://localhost:11434).

    Returns:
        True if the Ollama service is running and the model is available,
        False otherwise.
    """
    try:
        import httpx
        import ollama
    except ImportError:
        return False

    # Resolve host: explicit parameter > settings > hardcoded default
    if host is None:
        try:
            from app.settings import get_settings

            host = get_settings().ollama.host
        except Exception:
            host = 'http://localhost:11434'

    try:
        # Check 1: Service is running (short timeout)
        with httpx.Client(timeout=2.0) as client:
            response = client.get(host)
            if response.status_code != 200:
                return False

        # Check 2: Model is available
        ollama_client = ollama.Client(host=host, timeout=5.0)

        if model is not None:
            ollama_client.show(model)
            return True

        # No model specified -- check candidate models in priority order
        candidate_models = ['all-minilm', 'qwen3-embedding:0.6b']
        for candidate in candidate_models:
            try:
                ollama_client.show(candidate)
                return True
            except Exception:
                continue
        return False

    except Exception:
        return False


async def store_single_chunk_embedding(
    repo: 'EmbeddingRepository',
    context_id: str,
    embedding: list[float],
    model: str = 'test-model',
) -> None:
    """Store one embedding for a context entry as a single chunk.

    Seeds the embedding tables through ``EmbeddingRepository.store_chunked``
    with one chunk whose start and end offsets are both zero.

    Args:
        repo: Embedding repository to write through.
        context_id: ID of the context entry the embedding belongs to.
        embedding: Embedding vector.
        model: Model identifier recorded in ``embedding_metadata``.
    """
    from app.repositories.embedding_repository.records import ChunkEmbedding

    chunk = ChunkEmbedding(embedding=embedding, start_index=0, end_index=0)
    await repo.store_chunked(context_id, [chunk], model)
