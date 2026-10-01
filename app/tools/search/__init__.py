"""Search tools for context entries.

The package holds the four search tools and the pieces they share:

- browse: search_context, exact filter/browse with no free-text query
- semantic: semantic_search_context, vector similarity search
- fts: fts_search_context, full-text search with linguistic analysis
- hybrid: hybrid_search_context, FTS and semantic search fused with RRF
- legs: the raw semantic and FTS searches the standalone and hybrid tools share
- ranking: cross-encoder reranking and the display format applied to every result
- limits: argument bounds, filter caps, and the empty responses those bounds produce
"""
