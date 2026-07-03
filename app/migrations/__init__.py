"""
Database migration functions for mcp-context-server.

This package contains migration orchestration and all migration implementations:
- dependencies.py: Provider and vector storage dependency checking
- semantic.py: Semantic search migrations (vector tables, jsonb_merge_patch)
- fts.py: Full-text search migrations
- metadata.py: Metadata field index management
- chunking.py: 1:N embedding relationship migration
- summary.py: Summary column migration
- content_hash.py: Content hash column migration for deduplication optimization
- version.py: Optimistic-concurrency version column migration
- compression.py: Embedding compression migration (vec_context_embeddings_compressed + compression_metadata)

SQL Files (resources):
- add_semantic_search_*.sql: Vector table schemas
- add_fts_*.sql: FTS table schemas
- add_chunking_*.sql: 1:N embedding schema modifications
- add_compression_*.sql: Compressed-payload table schemas
- add_jsonb_merge_patch_postgresql.sql: PostgreSQL merge function
- fix_function_search_path_postgresql.sql: Security fix

PostgreSQL DDL convention: all TABLE and INDEX DDL in
``add_*_postgresql.sql`` uses BARE table names. Operators with a
non-default ``POSTGRESQL_SCHEMA`` value must configure ``search_path``
on every connection so the migration creates tables in the intended
schema. FUNCTION DDL (and ``ALTER FUNCTION`` targets, plus trigger
``EXECUTE FUNCTION`` references) remains schema-qualified for
CVE-2018-1058 mitigation; this is a distinct concern from table-name
resolution. Idempotency-check filters against ``information_schema``,
``pg_indexes``, and ``pg_namespace`` use ``current_schema()`` so the
check inspects whatever schema ``search_path`` resolves to. See
``docs/embedding-compression.md`` for the full operator contract.
"""

from app.migrations.chunking import apply_chunking_migration
from app.migrations.compression import apply_compression_migration
from app.migrations.content_hash import apply_content_hash_migration
from app.migrations.dependencies import ProviderCheckResult
from app.migrations.dependencies import check_provider_dependencies
from app.migrations.dependencies import check_summary_provider_dependencies
from app.migrations.dependencies import check_vector_storage_dependencies
from app.migrations.fts import FtsMigrationStatus
from app.migrations.fts import apply_fts_migration
from app.migrations.fts import estimate_migration_time
from app.migrations.fts import get_fts_migration_status
from app.migrations.fts import reset_fts_migration_status
from app.migrations.index_tree import apply_index_tree_migration
from app.migrations.metadata import handle_metadata_indexes
from app.migrations.semantic import apply_function_search_path_migration
from app.migrations.semantic import apply_jsonb_merge_patch_migration
from app.migrations.semantic import apply_semantic_search_migration
from app.migrations.summary import apply_summary_migration
from app.migrations.version import apply_version_migration

__all__ = [
    # Dependencies
    'ProviderCheckResult',
    'check_vector_storage_dependencies',
    'check_provider_dependencies',
    'check_summary_provider_dependencies',
    # Semantic
    'apply_semantic_search_migration',
    'apply_jsonb_merge_patch_migration',
    'apply_function_search_path_migration',
    # FTS
    'FtsMigrationStatus',
    'apply_fts_migration',
    'estimate_migration_time',
    'get_fts_migration_status',
    'reset_fts_migration_status',
    # Metadata
    'handle_metadata_indexes',
    # Chunking
    'apply_chunking_migration',
    # Index tree
    'apply_index_tree_migration',
    # Summary
    'apply_summary_migration',
    'check_summary_provider_dependencies',
    # Content hash
    'apply_content_hash_migration',
    # Version (optimistic concurrency)
    'apply_version_migration',
    # Compression
    'apply_compression_migration',
]
