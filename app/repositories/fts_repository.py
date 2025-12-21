"""
Repository for Full-Text Search (FTS) operations supporting both SQLite FTS5 and PostgreSQL tsvector.

This module provides data access for full-text search functionality,
handling search operations across both SQLite (FTS5) and PostgreSQL (tsvector) backends.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

from app.backends.base import StorageBackend
from app.logger_config import config_logger
from app.repositories.base import BaseRepository
from app.settings import get_settings

# Regex pattern to match hyphenated words (e.g., "full-text", "pre-commit", "user-friendly")
# Matches word characters connected by one or more hyphens
HYPHENATED_WORD_PATTERN = re.compile(r'\b(\w+(?:-\w+)+)\b')

if TYPE_CHECKING:
    import asyncpg

# Get settings
settings = get_settings()
# Configure logging
config_logger(settings.log_level)
logger = logging.getLogger(__name__)


class FtsValidationError(Exception):
    """Exception raised when FTS query or filters fail validation.

    This exception enables unified error handling between fts_search_context
    and other search tools.
    """

    def __init__(self, message: str, validation_errors: list[str]) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            validation_errors: List of validation error messages
        """
        super().__init__(message)
        self.message = message
        self.validation_errors = validation_errors


class FtsRepository(BaseRepository):
    """Repository for Full-Text Search operations supporting both FTS5 and tsvector.

    This repository handles all database operations for full-text search,
    using either SQLite FTS5 extension or PostgreSQL tsvector functionality
    depending on the configured storage backend.

    Supported backends:
    - SQLite: Uses FTS5 with unicode61 tokenizer and BM25 ranking.
      Note: unicode61 provides multilingual tokenization but NO stemming,
      so "running" will NOT match "run". The language parameter is ignored.
    - PostgreSQL: Uses tsvector with ts_rank and language-specific stemming
      (supports 29 languages). Stemming means "running" WILL match "run".
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize the FTS repository.

        Args:
            backend: Storage backend for all database operations
        """
        super().__init__(backend)

    async def search(
        self,
        query: str,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'] = 'match',
        limit: int = 50,
        offset: int = 0,
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
        content_type: Literal['text', 'multimodal'] | None = None,
        tags: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        highlight: bool = False,
        language: str = 'english',
        explain_query: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Execute full-text search with optional filters.

        SQLite: Uses FTS5 MATCH with BM25 scoring
        PostgreSQL: Uses tsvector with ts_rank scoring

        Args:
            query: Full-text search query string
            mode: Search mode - 'match' (default), 'prefix' (wildcard), 'phrase' (exact), 'boolean' (AND/OR/NOT)
            limit: Maximum number of results to return
            offset: Number of results to skip (pagination)
            thread_id: Optional filter by thread
            source: Optional filter by source type
            content_type: Filter by content type (text or multimodal)
            tags: Filter by any of these tags (OR logic)
            start_date: Filter by created_at >= date (ISO 8601 format)
            end_date: Filter by created_at <= date (ISO 8601 format)
            metadata: Simple metadata filters (key=value equality)
            metadata_filters: Advanced metadata filters with operators
            highlight: Whether to include highlighted snippets in results
            language: Language for stemming (default: 'english').
                PostgreSQL: Supports 29 languages with full stemming.
                SQLite: This parameter is IGNORED - FTS5 uses unicode61 tokenizer
                which provides multilingual tokenization but no stemming.
            explain_query: If True, include query execution plan in stats

        Returns:
            Tuple of (search results list, statistics dictionary)
        """
        if self.backend.backend_type == 'sqlite':
            # Log warning if non-English language is requested with SQLite backend
            if language != 'english':
                logger.warning(
                    'SQLite FTS5 does not support language-specific stemming. '
                    "The language parameter '%s' is ignored. "
                    'SQLite uses unicode61 tokenizer which provides multilingual '
                    'tokenization but no stemming (e.g., "running" will NOT match "run"). '
                    'Use PostgreSQL backend for full language-specific stemming support.',
                    language,
                )
            return await self._search_sqlite(
                query=query,
                mode=mode,
                limit=limit,
                offset=offset,
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata=metadata,
                metadata_filters=metadata_filters,
                highlight=highlight,
                explain_query=explain_query,
            )
        # postgresql
        return await self._search_postgresql(
            query=query,
            mode=mode,
            limit=limit,
            offset=offset,
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            start_date=start_date,
            end_date=end_date,
            metadata=metadata,
            metadata_filters=metadata_filters,
            highlight=highlight,
            language=language,
            explain_query=explain_query,
        )

    async def _search_sqlite(
        self,
        query: str,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
        limit: int,
        offset: int,
        thread_id: str | None,
        source: str | None,
        content_type: str | None,
        tags: list[str] | None,
        start_date: str | None,
        end_date: str | None,
        metadata: dict[str, str | int | float | bool] | None,
        metadata_filters: list[dict[str, Any]] | None,
        highlight: bool,
        explain_query: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """SQLite FTS5 search implementation."""
        import time as time_module

        # Track metadata filter count for stats
        metadata_filter_count = 0

        def _search_sqlite_inner(conn: sqlite3.Connection) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            nonlocal metadata_filter_count
            start_time = time_module.time()
            # Transform query based on mode
            fts_query = self._transform_query_sqlite(query, mode)

            filter_conditions: list[str] = []
            filter_params: list[Any] = []

            if thread_id:
                filter_conditions.append('ce.thread_id = ?')
                filter_params.append(thread_id)

            if source:
                filter_conditions.append('ce.source = ?')
                filter_params.append(source)

            if content_type:
                filter_conditions.append('ce.content_type = ?')
                filter_params.append(content_type)

            # Tag filter (uses subquery with indexed tag table)
            if tags:
                normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
                if normalized_tags:
                    tag_placeholders = ','.join(['?' for _ in normalized_tags])
                    filter_conditions.append(f'''
                        ce.id IN (
                            SELECT DISTINCT context_entry_id
                            FROM tags
                            WHERE tag IN ({tag_placeholders})
                        )
                    ''')
                    filter_params.extend(normalized_tags)

            # Date range filtering - Use datetime() to normalize ISO 8601 input
            if start_date:
                filter_conditions.append('ce.created_at >= datetime(?)')
                filter_params.append(start_date)

            if end_date:
                filter_conditions.append('ce.created_at <= datetime(?)')
                filter_params.append(end_date)

            # Metadata filtering using MetadataQueryBuilder
            if metadata or metadata_filters:
                from pydantic import ValidationError

                from app.metadata_types import MetadataFilter
                from app.query_builder import MetadataQueryBuilder

                metadata_builder = MetadataQueryBuilder(backend_type='sqlite')

                # Simple metadata filters (key=value equality)
                if metadata:
                    for key, value in metadata.items():
                        try:
                            metadata_builder.add_simple_filter(key, value)
                        except ValueError as e:
                            logger.warning(f'Invalid simple metadata filter key={key}: {e}')

                # Advanced metadata filters with operators
                if metadata_filters:
                    validation_errors: list[str] = []
                    for filter_dict in metadata_filters:
                        try:
                            filter_spec = MetadataFilter(**filter_dict)
                            metadata_builder.add_advanced_filter(filter_spec)
                        except ValidationError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except ValueError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except Exception as e:
                            error_msg = f'Unexpected error in metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                            logger.error(f'Unexpected error processing metadata filter: {e}')

                    # Raise exception if validation fails
                    if validation_errors:
                        raise FtsValidationError(
                            'Metadata filter validation failed',
                            validation_errors,
                        )

                # Add metadata conditions to filter
                metadata_clause, metadata_params = metadata_builder.build_where_clause()
                if metadata_clause:
                    # Replace 'metadata' with 'ce.metadata' for table alias
                    metadata_clause_with_alias = metadata_clause.replace('metadata', 'ce.metadata')
                    filter_conditions.append(metadata_clause_with_alias)
                    filter_params.extend(metadata_params)

                # Track metadata filter count for stats
                metadata_filter_count = metadata_builder.get_filter_count()

            where_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ''

            # Build highlight expression if requested
            if highlight:
                highlight_expr = "highlight(context_entries_fts, 0, '<mark>', '</mark>') as highlighted"
            else:
                highlight_expr = 'NULL as highlighted'

            # Build main query with FTS5 join
            # bm25() returns negative scores where more negative = better match
            # We negate it to get positive scores where higher = better match
            sql_query = f'''
                SELECT
                    ce.id,
                    ce.thread_id,
                    ce.source,
                    ce.content_type,
                    ce.text_content,
                    ce.metadata,
                    ce.created_at,
                    ce.updated_at,
                    -bm25(context_entries_fts) as score,
                    {highlight_expr}
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                {where_clause}
                {'AND' if where_clause else 'WHERE'} fts.text_content MATCH ?
                ORDER BY score DESC
                LIMIT ? OFFSET ?
            '''

            # Combine params: filter_params + fts_query + limit + offset
            params = [*filter_params, fts_query, limit, offset]

            cursor = conn.execute(sql_query, params)
            rows = cursor.fetchall()

            results = [dict(row) for row in rows]

            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000

            # Count filters applied
            filter_count = sum([
                1 if thread_id else 0,
                1 if source else 0,
                1 if content_type else 0,
                1 if tags else 0,
                1 if start_date else 0,
                1 if end_date else 0,
            ])
            # Add metadata filter count
            filter_count += metadata_filter_count

            # Build statistics
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': filter_count,
                'rows_returned': len(results),
                'backend': 'sqlite',
            }

            # Get query plan if requested
            if explain_query:
                explain_cursor = conn.execute(f'EXPLAIN QUERY PLAN {sql_query}', params)
                plan_rows = explain_cursor.fetchall()
                plan_data: list[str] = []
                for row in plan_rows:
                    row_dict = dict(row)
                    id_val = row_dict.get('id', '?')
                    parent_val = row_dict.get('parent', '?')
                    notused_val = row_dict.get('notused', '?')
                    detail_val = row_dict.get('detail', '?')
                    formatted = f'id:{id_val} parent:{parent_val} notused:{notused_val} detail:{detail_val}'
                    plan_data.append(formatted)
                stats['query_plan'] = '\n'.join(plan_data)

            return results, stats

        return await self.backend.execute_read(_search_sqlite_inner)

    async def _search_postgresql(
        self,
        query: str,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
        limit: int,
        offset: int,
        thread_id: str | None,
        source: str | None,
        content_type: str | None,
        tags: list[str] | None,
        start_date: str | None,
        end_date: str | None,
        metadata: dict[str, str | int | float | bool] | None,
        metadata_filters: list[dict[str, Any]] | None,
        highlight: bool,
        language: str,
        explain_query: bool = False,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """PostgreSQL tsvector search implementation."""
        import time as time_module

        # Track metadata filter count for stats
        metadata_filter_count = 0

        async def _search_postgresql_inner(conn: asyncpg.Connection) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            nonlocal metadata_filter_count
            start_time = time_module.time()
            filter_conditions = ['1=1']  # Always true, makes building easier
            filter_params: list[Any] = []
            param_position = 1

            if thread_id:
                filter_conditions.append(f'ce.thread_id = {self._placeholder(param_position)}')
                filter_params.append(thread_id)
                param_position += 1

            if source:
                filter_conditions.append(f'ce.source = {self._placeholder(param_position)}')
                filter_params.append(source)
                param_position += 1

            if content_type:
                filter_conditions.append(f'ce.content_type = {self._placeholder(param_position)}')
                filter_params.append(content_type)
                param_position += 1

            # Tag filter (uses subquery with indexed tag table)
            if tags:
                normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
                if normalized_tags:
                    tag_placeholders = ','.join([
                        self._placeholder(param_position + i) for i in range(len(normalized_tags))
                    ])
                    filter_conditions.append(f'''
                        ce.id IN (
                            SELECT DISTINCT context_entry_id
                            FROM tags
                            WHERE tag IN ({tag_placeholders})
                        )
                    ''')
                    filter_params.extend(normalized_tags)
                    param_position += len(normalized_tags)

            # Date range filtering
            if start_date:
                filter_conditions.append(f'ce.created_at >= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(start_date))
                param_position += 1

            if end_date:
                filter_conditions.append(f'ce.created_at <= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(end_date))
                param_position += 1

            # Metadata filtering using MetadataQueryBuilder
            if metadata or metadata_filters:
                from pydantic import ValidationError

                from app.metadata_types import MetadataFilter
                from app.query_builder import MetadataQueryBuilder

                metadata_builder = MetadataQueryBuilder(
                    backend_type='postgresql',
                    param_offset=len(filter_params),
                )

                # Simple metadata filters
                if metadata:
                    for key, value in metadata.items():
                        try:
                            metadata_builder.add_simple_filter(key, value)
                        except ValueError as e:
                            logger.warning(f'Invalid simple metadata filter key={key}: {e}')

                # Advanced metadata filters
                if metadata_filters:
                    validation_errors: list[str] = []
                    for filter_dict in metadata_filters:
                        try:
                            filter_spec = MetadataFilter(**filter_dict)
                            metadata_builder.add_advanced_filter(filter_spec)
                        except ValidationError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except ValueError as e:
                            error_msg = f'Invalid metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                        except Exception as e:
                            error_msg = f'Unexpected error in metadata filter {filter_dict}: {e}'
                            validation_errors.append(error_msg)
                            logger.error(f'Unexpected error processing metadata filter: {e}')

                    if validation_errors:
                        raise FtsValidationError(
                            'Metadata filter validation failed',
                            validation_errors,
                        )

                metadata_clause, metadata_params = metadata_builder.build_where_clause()
                if metadata_clause:
                    metadata_clause_with_alias = metadata_clause.replace('metadata', 'ce.metadata')
                    filter_conditions.append(metadata_clause_with_alias)
                    filter_params.extend(metadata_params)
                    param_position += len(metadata_params)

                # Track metadata filter count for stats
                metadata_filter_count = metadata_builder.get_filter_count()

            where_clause = ' AND '.join(filter_conditions)

            # Transform query based on mode for PostgreSQL
            tsquery_func = self._get_tsquery_function(mode, language)

            # Query parameter position
            query_param_pos = param_position
            param_position += 1

            # Build highlight expression if requested
            if highlight:
                highlight_expr = f'''
                    ts_headline(
                        '{language}',
                        ce.text_content,
                        {tsquery_func}{self._placeholder(query_param_pos)}),
                        'StartSel=<mark>, StopSel=</mark>, MaxWords=50, MinWords=25'
                    ) as highlighted
                '''
            else:
                highlight_expr = 'NULL as highlighted'

            # Build main query with tsvector matching
            sql_query = f'''
                SELECT
                    ce.id,
                    ce.thread_id,
                    ce.source,
                    ce.content_type,
                    ce.text_content,
                    ce.metadata,
                    ce.created_at,
                    ce.updated_at,
                    ts_rank(ce.text_search_vector, {tsquery_func}{self._placeholder(query_param_pos)})) as score,
                    {highlight_expr}
                FROM context_entries ce
                WHERE {where_clause}
                AND ce.text_search_vector @@ {tsquery_func}{self._placeholder(query_param_pos)})
                ORDER BY score DESC
                LIMIT {self._placeholder(param_position)} OFFSET {self._placeholder(param_position + 1)}
            '''

            # Transform query based on mode (for prefix mode, adds :* suffix)
            transformed_query = self._transform_query_postgresql(query, mode)

            # Add transformed query, limit, offset to params
            filter_params.extend([transformed_query, limit, offset])

            rows = await conn.fetch(sql_query, *filter_params)

            results = [dict(row) for row in rows]

            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000

            # Count filters applied
            filter_count = sum([
                1 if thread_id else 0,
                1 if source else 0,
                1 if content_type else 0,
                1 if tags else 0,
                1 if start_date else 0,
                1 if end_date else 0,
            ])
            # Add metadata filter count
            filter_count += metadata_filter_count

            # Build statistics
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': filter_count,
                'rows_returned': len(results),
                'backend': 'postgresql',
            }

            # Get query plan if requested
            if explain_query:
                explain_result = await conn.fetch(f'EXPLAIN {sql_query}', *filter_params)
                plan_data: list[str] = [record['QUERY PLAN'] for record in explain_result]
                stats['query_plan'] = '\n'.join(plan_data)

            return results, stats

        return await self.backend.execute_read(cast(Any, _search_postgresql_inner))

    def _escape_double_quotes(self, text: str) -> str:
        """Escape double quotes for FTS5 phrase literals.

        FTS5 requires double quotes to be escaped by doubling them.

        Args:
            text: Text that may contain double quotes

        Returns:
            Text with double quotes escaped as ""
        """
        return text.replace('"', '""')

    def _quote_hyphenated_words_sqlite(self, query: str) -> str:
        """Wrap hyphenated words in double quotes for FTS5.

        Transforms queries like "full-text search" to '"full-text" search'
        so that FTS5 treats hyphens as part of words, not as NOT operator.

        Args:
            query: Original search query

        Returns:
            Query with hyphenated words wrapped in double quotes
        """

        def replace_hyphenated(match: re.Match[str]) -> str:
            word = match.group(1)
            escaped = self._escape_double_quotes(word)
            return f'"{escaped}"'

        return HYPHENATED_WORD_PATTERN.sub(replace_hyphenated, query)

    def _handle_hyphenated_prefix_postgresql(self, word: str) -> str:
        """Handle hyphenated words for PostgreSQL prefix mode.

        Splits hyphenated words into AND-ed prefix terms.
        "full-text" -> "full:* & text:*"

        Args:
            word: Single word that may contain hyphens

        Returns:
            PostgreSQL prefix query fragment
        """
        # Strip existing wildcards
        clean_word = word.rstrip('*').removesuffix(':')

        if '-' in clean_word:
            # Split and create AND-ed prefix terms
            parts = clean_word.split('-')
            return ' & '.join(f'{part}:*' for part in parts if part)

        return f'{clean_word}:*'

    def _transform_query_sqlite(
        self,
        query: str,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
    ) -> str:
        """Transform query string for SQLite FTS5 based on mode.

        Args:
            query: Original search query
            mode: Search mode

        Returns:
            Transformed query for FTS5 MATCH
        """
        # Clean the query
        query = query.strip()

        if mode == 'phrase':
            # Exact phrase matching - wrap in double quotes
            # Escape any existing double quotes first
            escaped = self._escape_double_quotes(query)
            return f'"{escaped}"'

        if mode == 'prefix':
            # Prefix matching - handle hyphenated words specially
            # First quote hyphenated words, then add wildcards
            quoted = self._quote_hyphenated_words_sqlite(query)
            words = quoted.split()
            result_words = []
            for word in words:
                if word.startswith('"') and word.endswith('"'):
                    # Hyphenated word already quoted, add wildcard after
                    result_words.append(f'{word}*')
                else:
                    # Regular word, add wildcard
                    result_words.append(f'{word.rstrip("*")}*')
            return ' '.join(result_words)

        if mode == 'boolean':
            # Boolean mode - pass through as-is (user provides AND/OR/NOT)
            # User is responsible for quoting hyphenated words
            return query

        # 'match' - default
        # Quote hyphenated words to prevent NOT operator interpretation
        return self._quote_hyphenated_words_sqlite(query)

    def _transform_query_postgresql(
        self,
        query: str,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
    ) -> str:
        """Transform query string for PostgreSQL tsquery based on mode.

        For prefix mode, transforms "hello world" to "hello:* & world:*"
        to work correctly with to_tsquery().

        Args:
            query: Original search query
            mode: Search mode

        Returns:
            Transformed query for PostgreSQL tsquery
        """
        # Clean the query
        query = query.strip()

        if mode == 'prefix':
            # Prefix matching - handle hyphenated words by splitting
            words = query.split()
            prefix_terms = [self._handle_hyphenated_prefix_postgresql(word) for word in words]
            return ' & '.join(prefix_terms)

        # For other modes, return query as-is
        # - match: plainto_tsquery discards punctuation (safe)
        # - phrase: phraseto_tsquery discards punctuation (safe)
        # - boolean: websearch_to_tsquery treats - as NOT (by design)
        return query

    def _get_tsquery_function(
        self,
        mode: Literal['match', 'prefix', 'phrase', 'boolean'],
        language: str,
    ) -> str:
        """Get the appropriate PostgreSQL tsquery function for the search mode.

        Args:
            mode: Search mode
            language: Language for text search

        Returns:
            SQL function call string for tsquery generation
        """
        if mode == 'phrase':
            return f"phraseto_tsquery('{language}', "
        if mode == 'prefix':
            # For prefix, we use to_tsquery which supports :* for prefix
            return f"to_tsquery('{language}', "
        if mode == 'boolean':
            # websearch supports Google-like syntax with OR, -, quotes
            return f"websearch_to_tsquery('{language}', "
        # 'match' - default
        # plainto_tsquery handles natural language input
        return f"plainto_tsquery('{language}', "

    async def rebuild_index(self) -> dict[str, Any]:
        """Rebuild the FTS index from scratch.

        Useful after bulk imports or to fix index corruption.

        Returns:
            Statistics about the rebuild operation
        """
        if self.backend.backend_type == 'sqlite':

            def _rebuild_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                # Count entries before rebuild
                cursor = conn.execute('SELECT COUNT(*) FROM context_entries')
                entry_count = cursor.fetchone()[0]

                # Rebuild FTS index
                conn.execute("INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')")

                return {
                    'success': True,
                    'entries_indexed': entry_count,
                    'backend': 'sqlite',
                }

            return await self.backend.execute_write(_rebuild_sqlite)

        # postgresql
        async def _rebuild_postgresql(conn: asyncpg.Connection) -> dict[str, Any]:
            # Count entries
            entry_count = await conn.fetchval('SELECT COUNT(*) FROM context_entries')

            # Reindex the GIN index
            await conn.execute('REINDEX INDEX idx_text_search_gin')

            return {
                'success': True,
                'entries_indexed': entry_count,
                'backend': 'postgresql',
            }

        return await self.backend.execute_write(cast(Any, _rebuild_postgresql))

    async def get_statistics(self, thread_id: str | None = None) -> dict[str, Any]:
        """Get FTS index statistics.

        Args:
            thread_id: Optional filter by thread

        Returns:
            Dictionary with statistics (entry count, index info)
        """
        if self.backend.backend_type == 'sqlite':

            def _get_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                # Count indexed entries
                if thread_id:
                    cursor = conn.execute(
                        '''
                        SELECT COUNT(*) FROM context_entries_fts fts
                        JOIN context_entries ce ON ce.id = fts.rowid
                        WHERE ce.thread_id = ?
                        ''',
                        (thread_id,),
                    )
                else:
                    cursor = conn.execute('SELECT COUNT(*) FROM context_entries_fts')

                indexed_count = cursor.fetchone()[0]

                # Get total entries
                if thread_id:
                    cursor = conn.execute(
                        'SELECT COUNT(*) FROM context_entries WHERE thread_id = ?',
                        (thread_id,),
                    )
                else:
                    cursor = conn.execute('SELECT COUNT(*) FROM context_entries')

                total_count = cursor.fetchone()[0]

                return {
                    'total_entries': total_count,
                    'indexed_entries': indexed_count,
                    'coverage_percentage': round((indexed_count / total_count * 100) if total_count > 0 else 0.0, 2),
                    'backend': 'sqlite',
                    'engine': 'fts5',
                }

            return await self.backend.execute_read(_get_stats_sqlite)

        # postgresql
        async def _get_stats_postgresql(conn: asyncpg.Connection) -> dict[str, Any]:
            # Count entries with tsvector populated
            if thread_id:
                indexed_count = await conn.fetchval(
                    '''
                    SELECT COUNT(*) FROM context_entries
                    WHERE thread_id = $1 AND text_search_vector IS NOT NULL
                    ''',
                    thread_id,
                )
                total_count = await conn.fetchval(
                    'SELECT COUNT(*) FROM context_entries WHERE thread_id = $1',
                    thread_id,
                )
            else:
                indexed_count = await conn.fetchval(
                    'SELECT COUNT(*) FROM context_entries WHERE text_search_vector IS NOT NULL',
                )
                total_count = await conn.fetchval('SELECT COUNT(*) FROM context_entries')

            return {
                'total_entries': total_count,
                'indexed_entries': indexed_count,
                'coverage_percentage': round((indexed_count / total_count * 100) if total_count > 0 else 0.0, 2),
                'backend': 'postgresql',
                'engine': 'tsvector',
            }

        return await self.backend.execute_read(cast(Any, _get_stats_postgresql))

    async def is_available(self) -> bool:
        """Check if FTS functionality is available.

        Returns:
            True if FTS is properly configured and available
        """
        if self.backend.backend_type == 'sqlite':

            def _check_sqlite(conn: sqlite3.Connection) -> bool:
                try:
                    # Check if FTS5 table exists
                    cursor = conn.execute(
                        "SELECT name FROM sqlite_master WHERE type='table' AND name='context_entries_fts'",
                    )
                    return cursor.fetchone() is not None
                except Exception:
                    return False

            return await self.backend.execute_read(_check_sqlite)

        # postgresql
        async def _check_postgresql(conn: asyncpg.Connection) -> bool:
            try:
                # Check if text_search_vector column exists
                result = await conn.fetchval(
                    '''
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'context_entries' AND column_name = 'text_search_vector'
                    )
                    ''',
                )
                return bool(result)
            except Exception:
                return False

        return await self.backend.execute_read(cast(Any, _check_postgresql))

    async def get_current_tokenizer(self) -> str | None:
        """Get the current FTS5 tokenizer from SQLite (SQLite only).

        Parses the sqlite_master table to extract the tokenizer definition
        from the FTS5 virtual table creation SQL.

        Returns:
            The tokenizer string (e.g., 'unicode61' or 'porter unicode61'),
            or None if FTS5 table doesn't exist or backend is not SQLite.
        """
        if self.backend.backend_type != 'sqlite':
            return None

        def _get_tokenizer(conn: sqlite3.Connection) -> str | None:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='context_entries_fts'",
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Parse the SQL to extract tokenizer
            # Example SQL: "CREATE VIRTUAL TABLE context_entries_fts USING fts5(..., tokenize='porter unicode61')"
            sql = row[0]
            if 'tokenize=' not in sql.lower():
                return 'unicode61'  # Default if not specified

            # Extract tokenizer value using string parsing
            # Find tokenize= and extract the quoted value
            import re

            # Pattern matches tokenize='...' or tokenize="..."
            pattern = r"tokenize\s*=\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                return match.group(1)

            return 'unicode61'  # Default fallback

        return await self.backend.execute_read(_get_tokenizer)

    async def get_current_language(self) -> str | None:
        """Get the current FTS language from PostgreSQL tsvector column (PostgreSQL only).

        Queries pg_attrdef to decompile the GENERATED ALWAYS AS expression
        and extracts the language parameter from to_tsvector().

        Returns:
            The language string (e.g., 'english', 'german'),
            or None if tsvector column doesn't exist or backend is not PostgreSQL.
        """
        if self.backend.backend_type != 'postgresql':
            return None

        async def _get_language(conn: asyncpg.Connection) -> str | None:
            # Query to get the generation expression for text_search_vector column
            result = await conn.fetchval(
                '''
                SELECT pg_get_expr(ad.adbin, ad.adrelid) AS generation_expression
                FROM pg_attribute a
                JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
                WHERE a.attrelid = 'context_entries'::regclass
                  AND a.attname = 'text_search_vector'
                  AND a.attgenerated = 's'
                ''',
            )
            if not result:
                return None

            # Parse the expression to extract language
            # Example: "to_tsvector('english'::regconfig, COALESCE(text_content, ''::text))"
            import re

            # Pattern matches to_tsvector('language'::regconfig, ...) or to_tsvector('language', ...)
            pattern = r"to_tsvector\s*\(\s*'([^']+)'"
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                return match.group(1)

            return 'english'  # Default fallback

        return await self.backend.execute_read(cast(Any, _get_language))

    async def get_desired_tokenizer(self, language: str) -> str:
        """Determine the desired SQLite FTS5 tokenizer based on language setting.

        Based on the research: English benefits from Porter stemmer, other languages
        should use unicode61 for proper multilingual tokenization.

        Args:
            language: The FTS_LANGUAGE setting value

        Returns:
            The tokenizer string to use ('porter unicode61' or 'unicode61')
        """
        # English (or not set) -> use Porter stemmer for English-language stemming
        if language.lower() == 'english':
            return 'porter unicode61'
        # Other languages -> use unicode61 only (no stemming, but proper Unicode tokenization)
        return 'unicode61'

    async def migrate_tokenizer(self, new_tokenizer: str) -> dict[str, Any]:
        """Migrate SQLite FTS5 to a new tokenizer (SQLite only).

        This operation drops the existing FTS5 virtual table and recreates it
        with the new tokenizer. The data is NOT lost because FTS5 uses external
        content mode (content='context_entries').

        Args:
            new_tokenizer: The new tokenizer to use (e.g., 'porter unicode61' or 'unicode61')

        Returns:
            Dictionary with migration results

        Raises:
            RuntimeError: If migration fails or backend is not SQLite
        """
        if self.backend.backend_type != 'sqlite':
            raise RuntimeError('migrate_tokenizer is only supported for SQLite backend')

        old_tokenizer = await self.get_current_tokenizer()

        def _migrate_tokenizer(conn: sqlite3.Connection) -> dict[str, Any]:
            # Count entries for statistics
            cursor = conn.execute('SELECT COUNT(*) FROM context_entries')
            entry_count = cursor.fetchone()[0]

            # Drop existing FTS5 table and triggers
            conn.execute('DROP TRIGGER IF EXISTS context_fts_insert')
            conn.execute('DROP TRIGGER IF EXISTS context_fts_delete')
            conn.execute('DROP TRIGGER IF EXISTS context_fts_update')
            conn.execute('DROP TABLE IF EXISTS context_entries_fts')

            # Recreate FTS5 table with new tokenizer
            create_sql = f'''
                CREATE VIRTUAL TABLE context_entries_fts USING fts5(
                    text_content,
                    content='context_entries',
                    content_rowid='id',
                    tokenize='{new_tokenizer}'
                )
            '''
            conn.execute(create_sql)

            # Recreate triggers
            conn.execute('''
                CREATE TRIGGER context_fts_insert AFTER INSERT ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(rowid, text_content)
                    VALUES (new.id, new.text_content);
                END
            ''')

            conn.execute('''
                CREATE TRIGGER context_fts_delete AFTER DELETE ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
                    VALUES('delete', old.id, old.text_content);
                END
            ''')

            conn.execute('''
                CREATE TRIGGER context_fts_update AFTER UPDATE OF text_content ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
                    VALUES('delete', old.id, old.text_content);
                    INSERT INTO context_entries_fts(rowid, text_content)
                    VALUES (new.id, new.text_content);
                END
            ''')

            # Rebuild the FTS index from existing data
            conn.execute("INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')")

            return {
                'success': True,
                'backend': 'sqlite',
                'old_tokenizer': old_tokenizer,
                'new_tokenizer': new_tokenizer,
                'entries_migrated': entry_count,
            }

        return await self.backend.execute_write(_migrate_tokenizer)

    async def migrate_language(self, new_language: str) -> dict[str, Any]:
        """Migrate PostgreSQL tsvector column to a new language (PostgreSQL only).

        This operation drops the existing text_search_vector column (and its GIN index)
        and recreates it with the new language. The GENERATED ALWAYS AS column is
        automatically populated from text_content on recreation.

        Args:
            new_language: The new language for tsvector (e.g., 'english', 'german')

        Returns:
            Dictionary with migration results

        Raises:
            RuntimeError: If migration fails or backend is not PostgreSQL
        """
        if self.backend.backend_type != 'postgresql':
            raise RuntimeError('migrate_language is only supported for PostgreSQL backend')

        old_language = await self.get_current_language()

        async def _migrate_language(conn: asyncpg.Connection) -> dict[str, Any]:
            # Count entries for statistics
            entry_count = await conn.fetchval('SELECT COUNT(*) FROM context_entries')

            # Drop existing column (also drops dependent GIN index)
            await conn.execute('ALTER TABLE context_entries DROP COLUMN IF EXISTS text_search_vector')

            # Recreate column with new language
            await conn.execute(f'''
                ALTER TABLE context_entries
                ADD COLUMN text_search_vector tsvector
                GENERATED ALWAYS AS (to_tsvector('{new_language}', COALESCE(text_content, ''))) STORED
            ''')

            # Recreate GIN index
            await conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_text_search_gin
                ON context_entries USING GIN(text_search_vector)
            ''')

            return {
                'success': True,
                'backend': 'postgresql',
                'old_language': old_language,
                'new_language': new_language,
                'entries_migrated': entry_count,
            }

        return await self.backend.execute_write(cast(Any, _migrate_language))
