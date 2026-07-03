"""The store/update embedding leg offloads a large entry's chunking off the loop.

``_generate_embeddings_for_text`` calls ``ChunkingService.split_text`` ->
``RecursiveCharacterTextSplitter.create_documents``, which is O(text) pure CPU
over UNBOUNDED stored entry text and runs on the embedding leg of
``run_generation`` (the store/update/batch write path). A large entry is offloaded
to a worker thread so a multi-megabyte store cannot pin the single event loop and
starve concurrent MCP requests; a small entry stays inline to avoid a thread hop.
Mirrors the read-path (test_navigation_tools.py::TestLargeEntryOffloadNonBlocking),
the index_tree node leg (test_index_node_generation.py::TestLargeEntryWritePathOffloadNonBlocking),
and the grep matcher (test_grep_matcher.py).
"""

import threading
from unittest.mock import patch

import pytest

import app.tools._shared as shared_module
from app.services.chunking_service import TextChunk
from app.tools._shared import _generate_embeddings_for_text


class _FakeEmbeddingProvider:
    """Minimal embedding provider returning a fixed vector per text."""

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    async def embed_query(self, text: str) -> list[float]:
        _ = text
        return [0.1, 0.2, 0.3]


class _SpyChunkingService:
    """Enabled chunking service whose split_text records its executing thread."""

    is_enabled = True

    def __init__(self) -> None:
        self.on_main: bool | None = None

    def split_text(self, text: str) -> list[TextChunk]:
        self.on_main = threading.current_thread() is threading.main_thread()
        return [TextChunk(text=text, chunk_index=0, start_index=0, end_index=len(text))]


@pytest.mark.asyncio
async def test_large_entry_chunking_offloaded() -> None:
    spy = _SpyChunkingService()
    big = 'a' * (shared_module._OFFLOAD_MIN_CHARS + 10)  # exceeds the offload threshold
    with (
        patch.object(shared_module, 'get_embedding_provider', lambda: _FakeEmbeddingProvider()),
        patch.object(shared_module, 'get_chunking_service', lambda: spy),
    ):
        result = await _generate_embeddings_for_text(big)
    assert spy.on_main is False  # chunking ran on a worker thread, not the event loop
    assert result is not None
    assert len(result) == 1


@pytest.mark.asyncio
async def test_small_entry_chunking_inline() -> None:
    spy = _SpyChunkingService()
    with (
        patch.object(shared_module, 'get_embedding_provider', lambda: _FakeEmbeddingProvider()),
        patch.object(shared_module, 'get_chunking_service', lambda: spy),
    ):
        await _generate_embeddings_for_text('short text body')
    assert spy.on_main is True  # small entry stays inline (no thread hop)
