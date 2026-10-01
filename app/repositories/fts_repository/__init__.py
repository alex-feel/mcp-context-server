"""
Repository for Full-Text Search (FTS) operations supporting both SQLite FTS5 and PostgreSQL tsvector.

This module provides data access for full-text search functionality,
handling search operations across both SQLite (FTS5) and PostgreSQL (tsvector) backends.
``FtsRepository`` is assembled here from the search mixin and the maintenance
mixin; query handling lives in ``query`` and failure classification in ``faults``.
"""

from app.repositories.fts_repository.maintenance import FtsMaintenanceMixin
from app.repositories.fts_repository.search import FtsSearchMixin


class FtsRepository(FtsSearchMixin, FtsMaintenanceMixin):
    """Repository for Full-Text Search operations supporting both FTS5 and tsvector.

    This repository handles all database operations for full-text search,
    using either SQLite FTS5 extension or PostgreSQL tsvector functionality
    depending on the configured storage backend.

    Supported backends:
    - SQLite: Uses FTS5 with BM25 ranking and a language-aware tokenizer.
      'english' (the default) uses 'porter unicode61' (Porter stemming, so
      "running" matches "run"); any other language uses plain 'unicode61'
      (multilingual tokenization, no stemming). The language parameter selects
      between those two tokenizers at table creation; per-language stemming for
      non-English values is not supported.
    - PostgreSQL: Uses tsvector with ts_rank_cd and language-specific stemming
      (supports 29 languages). Stemming means "running" WILL match "run".
    """
