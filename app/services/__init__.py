"""
Services package for mcp-context-server.

This package contains domain services that encapsulate business logic
separate from the repository layer.
"""

from app.services.chunking_service import ChunkingService
from app.services.chunking_service import TextChunk
from app.services.passage_extraction_service import HighlightRegion
from app.services.passage_extraction_service import extract_rerank_passage
from app.services.passage_extraction_service import parse_highlight_positions

__all__ = [
    'ChunkingService',
    'HighlightRegion',
    'TextChunk',
    'extract_rerank_passage',
    'parse_highlight_positions',
]
