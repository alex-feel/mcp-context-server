"""
Database migration functions for mcp-context-server.

This package contains migration orchestration and all migration implementations:
- utils.py: Shared utility functions (format_exception_message)
- dependencies.py: Provider and vector storage dependency checking
- semantic.py: Semantic search migrations (vector tables, jsonb_merge_patch)
- fts.py: Full-text search migrations
- metadata.py: Metadata field index management

SQL Files (resources):
- add_semantic_search_*.sql: Vector table schemas
- add_fts_*.sql: FTS table schemas
- add_jsonb_merge_patch_postgresql.sql: PostgreSQL merge function
- fix_function_search_path_postgresql.sql: Security fix
"""

from app.migrations.dependencies import ProviderCheckResult
from app.migrations.dependencies import check_provider_dependencies
from app.migrations.dependencies import check_vector_storage_dependencies
from app.migrations.fts import FtsMigrationStatus
from app.migrations.fts import apply_fts_migration
from app.migrations.fts import estimate_migration_time
from app.migrations.fts import get_fts_migration_status
from app.migrations.fts import reset_fts_migration_status
from app.migrations.metadata import handle_metadata_indexes
from app.migrations.semantic import apply_function_search_path_migration
from app.migrations.semantic import apply_jsonb_merge_patch_migration
from app.migrations.semantic import apply_semantic_search_migration
from app.migrations.utils import format_exception_message

__all__ = [
    # Utilities
    'format_exception_message',
    # Dependencies
    'ProviderCheckResult',
    'check_vector_storage_dependencies',
    'check_provider_dependencies',
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
]
