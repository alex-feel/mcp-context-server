"""
Services package for mcp-context-server.

This package contains domain services that encapsulate business logic
separate from the repository layer.
"""

from app.services.chunking_service import ChunkingService
from app.services.chunking_service import TextChunk
from app.services.grep_service import GrepEntryResult
from app.services.grep_service import GrepMatch
from app.services.grep_service import compile_pattern
from app.services.grep_service import extract_ascii_literal
from app.services.grep_service import match_entry
from app.services.outline_service import OutlineNode
from app.services.outline_service import count_nodes
from app.services.outline_service import parse_outline
from app.services.outline_service import resolve_node_span
from app.services.outline_service import slugify
from app.services.passage_extraction_service import HighlightRegion
from app.services.passage_extraction_service import extract_rerank_passage
from app.services.passage_extraction_service import parse_highlight_positions
from app.services.text_lines import line_index_for_offset
from app.services.text_lines import split_lines_with_offsets

__all__ = [
    'ChunkingService',
    'GrepEntryResult',
    'GrepMatch',
    'HighlightRegion',
    'OutlineNode',
    'TextChunk',
    'compile_pattern',
    'count_nodes',
    'extract_ascii_literal',
    'extract_rerank_passage',
    'line_index_for_offset',
    'match_entry',
    'parse_highlight_positions',
    'parse_outline',
    'resolve_node_span',
    'slugify',
    'split_lines_with_offsets',
]
