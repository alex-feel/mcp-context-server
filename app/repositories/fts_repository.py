"""
Repository for Full-Text Search (FTS) operations supporting both SQLite FTS5 and PostgreSQL tsvector.

This module provides data access for full-text search functionality,
handling search operations across both SQLite (FTS5) and PostgreSQL (tsvector) backends.
"""


import logging
import re
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

from app.backends.base import StorageBackend
from app.repositories.base import BaseRepository

# Regex pattern to match hyphenated words (e.g., "full-text", "pre-commit", "user-friendly")
# Matches word characters connected by one or more hyphens
HYPHENATED_WORD_PATTERN = re.compile(r'\b(\w+(?:-\w+)+)\b')

# Tokenizer for FTS5 sanitization: a "quoted phrase" OR a run of non-whitespace.
_FTS_TOKEN_RE = re.compile(r'"[^"]*"|\S+')


def sanitize_sqlite_fts_terms(tokens: list[str]) -> list[str]:
    """Sanitize query tokens into crash-safe SQLite FTS5 terms.

    SQLite FTS5 MATCH treats AND/OR/NOT/NEAR as case-sensitive operators and rejects bare
    special characters, so an unsanitized token can raise ``fts5: syntax error`` in match,
    prefix, or boolean-OR contexts. Each token is made safe and aligned with PostgreSQL's
    plainto_tsquery (which drops operator stopwords and ANDs the remaining lexemes):

    - A quoted-phrase token (``"..."``) is preserved as-is.
    - An operator bareword (and/or/not, case-insensitive) is DROPPED.
    - Every other bare term is wrapped as an FTS5 string literal (embedded quotes doubled), so
      an operator-cased NEAR or a special char ( ( ) " : * ^ ) is a LITERAL term, never syntax.
      A quoted single word is still tokenized/stemmed, so ordinary-word recall is unchanged.

    Shared by the standalone FTS transform (``_transform_query_sqlite`` match/prefix modes) and
    the hybrid adaptive query builder so the two never diverge.

    Args:
        tokens: Raw whitespace/quote tokens from the query.

    Returns:
        The list of safe FTS5 term strings (operator barewords removed).
    """
    terms: list[str] = []
    for token in tokens:
        # Require >= 2 chars: a LONE '"' satisfies both startswith AND endswith (they test the
        # SAME character), so without the length check it would pass through as a "balanced
        # phrase" and produce an unterminated FTS5 string literal (MATCH syntax error). With the
        # check it falls through to the doubling path below -> '""""' (a balanced empty literal).
        if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
            terms.append(token)
            continue
        clean = token.replace('-', ' ').strip()
        if not clean or clean.lower() in ('and', 'or', 'not'):
            continue
        terms.append('"' + clean.replace('"', '""') + '"')
    return terms


_FTS5_GRAMMAR_ERROR_FRAGMENTS = (
    'fts5: syntax error',
    'unterminated',
    'no such column',
    'malformed match expression',
    'unknown special query',
    'expected integer',
)


def _is_fts5_grammar_error(exc: sqlite3.OperationalError) -> bool:
    """Return True if exc is an FTS5 MATCH query-grammar error (not an operational fault).

    Boolean mode forwards the user's raw query to FTS5 MATCH to expose native FTS5 boolean
    syntax. Malformed input (unbalanced parentheses, a leading/trailing/bare operator, a stray
    ':' column filter, an unterminated string, a leading '*' that FTS5 reads as an unknown
    special-query token, or a NEAR() clause whose distance argument is not an integer) makes
    FTS5 raise one of a small set of grammar errors; those are the only cases the boolean search
    path degrades to a safe term match. Any other OperationalError (locked database, disk I/O,
    missing table) is an operational fault and MUST propagate unchanged.

    Args:
        exc: The OperationalError raised while executing a MATCH query.

    Returns:
        True if the error message identifies an FTS5 query-grammar error.
    """
    message = str(exc).lower()
    return any(fragment in message for fragment in _FTS5_GRAMMAR_ERROR_FRAGMENTS)


def desired_sqlite_fts_tokenizer(language: str) -> str:
    """Map an FTS_LANGUAGE setting to the SQLite FTS5 tokenizer.

    The single source of truth for the language->tokenizer rule, shared by the FTS
    migration (server startup), FtsRepository.get_desired_tokenizer (the rebuild check),
    and the migration CLI's SQLite target initialization, so they cannot drift:
    English benefits from the Porter stemmer; other languages use plain unicode61
    (proper Unicode tokenization, no English stemming).

    Args:
        language: The FTS_LANGUAGE setting value.

    Returns:
        The FTS5 tokenizer string ('porter unicode61' for English, else 'unicode61').
    """
    if language.lower() == 'english':
        return 'porter unicode61'
    return 'unicode61'


if TYPE_CHECKING:
    import asyncpg

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
    - PostgreSQL: Uses tsvector with ts_rank_cd and language-specific stemming
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
        PostgreSQL: Uses tsvector with ts_rank_cd scoring

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
            # Transform query based on mode. An empty transformed query (every token
            # sanitized to an operator/stopword bareword) is short-circuited to an empty
            # result set BELOW, AFTER metadata validation -- see the note at that guard.
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

                metadata_builder = MetadataQueryBuilder(backend_type='sqlite', table_alias='ce')
                validation_errors: list[str] = []

                # Simple metadata filters (key=value equality). An invalid KEY is reported
                # as a structured validation error (NOT silently dropped, which would widen
                # the result set), consistent with search_context and the advanced filters.
                if metadata:
                    for key, value in metadata.items():
                        try:
                            metadata_builder.add_simple_filter(key, value)
                        except ValueError as e:
                            validation_errors.append(f'Invalid metadata key {key!r}: {e}')

                # Advanced metadata filters with operators
                if metadata_filters:
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

                # Raise if ANY (simple key or advanced filter) validation failed.
                if validation_errors:
                    raise FtsValidationError(
                        'Metadata filter validation failed',
                        validation_errors,
                    )

                # The builder emits the metadata conditions already qualified with
                # the 'ce.' table alias (table_alias='ce'), matching the
                # context_entries JOIN target without rewriting the built SQL (a
                # global str.replace would corrupt JSON keys that contain the
                # substring 'metadata', e.g. 'metadata_version').
                metadata_clause, metadata_params = metadata_builder.build_where_clause()
                if metadata_clause:
                    filter_conditions.append(metadata_clause)
                    filter_params.extend(metadata_params)

                # Track metadata filter count for stats
                metadata_filter_count = metadata_builder.get_filter_count()

            # An empty transformed query means every token sanitized away (operator/stopword
            # barewords). FTS5 `MATCH ''` is a syntax error and a literal-phrase fallback would
            # over-match (FTS5 keeps stopwords as tokens), so short-circuit to an empty result
            # set -- parity with PostgreSQL's empty tsquery, which @@-matches no rows. Only the
            # all-bareword match/prefix paths yield ''; phrase/boolean modes never do.
            #
            # This guard runs AFTER metadata validation on purpose: an invalid metadata key/
            # filter must raise FtsValidationError on BOTH backends (PostgreSQL has no early
            # return and always validates first), never be silently dropped by the SQLite
            # short-circuit. The empty stats also carry the same `filters_applied` count and,
            # under explain_query, a `query_plan` key the PostgreSQL path emits, so the stats
            # shape stays backend-consistent for the all-stopword case.
            def _empty_result(plan_note: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                empty_filter_count = sum([
                    1 if thread_id else 0,
                    1 if source else 0,
                    1 if content_type else 0,
                    1 if tags else 0,
                    1 if start_date else 0,
                    1 if end_date else 0,
                ]) + metadata_filter_count
                empty_stats: dict[str, Any] = {
                    'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
                    'filters_applied': empty_filter_count,
                    'rows_returned': 0,
                    'backend': 'sqlite',
                }
                if explain_query:
                    empty_stats['query_plan'] = plan_note
                return [], empty_stats

            if not fts_query.strip():
                return _empty_result(
                    'Query short-circuited: empty FTS query (all tokens were '
                    'operators/stopwords); 0 rows, no SQL executed.',
                )

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
                    ce.summary,
                    ce.created_at,
                    ce.updated_at,
                    -bm25(context_entries_fts) as score,
                    {highlight_expr}
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                {where_clause}
                {'AND' if where_clause else 'WHERE'} fts.text_content MATCH ?
                ORDER BY score DESC
                LIMIT ? OFFSET ?
            '''

            # Combine params: filter_params + fts_query + limit + offset
            params = [*filter_params, fts_query, limit, offset]

            try:
                cursor = conn.execute(sql_query, params)
                rows = cursor.fetchall()
            except sqlite3.OperationalError as exc:
                # Boolean mode forwards the raw query to FTS5 MATCH so native FTS5 boolean syntax
                # (AND/OR/NOT, parentheses, quoted phrases) reaches the engine intact -- it is the
                # one SQLite mode not pre-sanitized. A MALFORMED boolean query (unbalanced parens,
                # a dangling operator, a stray ':' column filter) makes FTS5 raise a grammar error;
                # PostgreSQL's websearch_to_tsquery never raises for the same input, so without
                # this guard the identical fts_search_context(mode='boolean') call hard-errors on
                # SQLite but succeeds on PostgreSQL and leaks the raw engine message to the client.
                # Degrade a malformed boolean query to the crash-safe sanitized term match (the
                # shared 'match' transform): a VALID boolean query executed above and never reaches
                # here, while a malformed one returns best-effort results instead of erroring --
                # matching PostgreSQL's tolerant contract without altering the documented native
                # syntax. Any non-grammar OperationalError (locked DB, disk I/O) propagates.
                if mode != 'boolean' or not _is_fts5_grammar_error(exc):
                    raise
                safe_query = self._transform_query_sqlite(query, 'match')
                if not safe_query.strip():
                    return _empty_result(
                        'Query short-circuited: malformed boolean query reduced to no '
                        'searchable terms; 0 rows.',
                    )
                params = [*filter_params, safe_query, limit, offset]
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

        async def _search_postgresql_inner(conn: 'asyncpg.Connection') -> tuple[list[dict[str, Any]], dict[str, Any]]:
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
                    table_alias='ce',
                )

                validation_errors: list[str] = []

                # Simple metadata filters (key=value equality). An invalid KEY is reported
                # as a structured validation error (NOT silently dropped, which would widen
                # the result set), consistent with search_context and the advanced filters.
                if metadata:
                    for key, value in metadata.items():
                        try:
                            metadata_builder.add_simple_filter(key, value)
                        except ValueError as e:
                            validation_errors.append(f'Invalid metadata key {key!r}: {e}')

                # Advanced metadata filters
                if metadata_filters:
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

                # Raise if ANY (simple key or advanced filter) validation failed.
                if validation_errors:
                    raise FtsValidationError(
                        'Metadata filter validation failed',
                        validation_errors,
                    )

                # The builder emits the metadata conditions already qualified with
                # the 'ce.' table alias (table_alias='ce'), matching the
                # context_entries JOIN target without rewriting the built SQL (a
                # global str.replace would corrupt JSON keys that contain the
                # substring 'metadata', e.g. 'metadata_version').
                metadata_clause, metadata_params = metadata_builder.build_where_clause()
                if metadata_clause:
                    filter_conditions.append(metadata_clause)
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

            # Build highlight expression for the outer query.
            # ts_headline is applied ONLY to the LIMIT'd result set (inner subquery)
            # to avoid O(N_matched) invocations on large result sets.
            if highlight:
                outer_highlight_expr = f'''
                    ts_headline(
                        '{language}',
                        sub.text_content,
                        {tsquery_func}{self._placeholder(query_param_pos)}),
                        'HighlightAll=true, StartSel=<mark>, StopSel=</mark>'
                    ) as highlighted
                '''
            else:
                outer_highlight_expr = 'NULL as highlighted'

            # Inner subquery: filter, rank, and LIMIT without ts_headline.
            # Outer query: apply ts_headline only to the final rows.
            sql_query = f'''
                SELECT
                    sub.id,
                    sub.thread_id,
                    sub.source,
                    sub.content_type,
                    sub.text_content,
                    sub.metadata,
                    sub.summary,
                    sub.created_at,
                    sub.updated_at,
                    sub.score,
                    {outer_highlight_expr}
                FROM (
                    SELECT
                        ce.id,
                        ce.thread_id,
                        ce.source,
                        ce.content_type,
                        ce.text_content,
                        ce.metadata,
                        ce.summary,
                        ce.created_at,
                        ce.updated_at,
                        ts_rank_cd(ce.text_search_vector, {tsquery_func}{self._placeholder(query_param_pos)})) as score
                    FROM context_entries ce
                    WHERE {where_clause}
                    AND ce.text_search_vector @@ {tsquery_func}{self._placeholder(query_param_pos)})
                    ORDER BY score DESC
                    LIMIT {self._placeholder(param_position)} OFFSET {self._placeholder(param_position + 1)}
                ) sub
                ORDER BY sub.score DESC
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
        # to_tsquery() is STRICT: a bare tsquery operator (&, |, !, parens), a stray ':'
        # or '*', or an unterminated quote raises "syntax error in tsquery" -- unlike
        # plainto/phraseto/websearch_to_tsquery, which never raise. Extract only word
        # characters (Unicode-aware) and emit each as an AND-ed prefix lexeme 'sub:*', so
        # an adversarial token ('cat(', 'foo:bar', '"x') degrades to safe literal
        # prefixes. A hyphenated/punctuated word splits into AND-ed prefixes
        # ("full-text" -> "full:* & text:*"); a token with no word characters yields ''
        # (dropped by the caller). A user trailing '*'/':' is naturally excluded.
        parts = re.findall(r'\w+', word)
        return ' & '.join(f'{part}:*' for part in parts)

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
            # Prefix (autocomplete): wrap each token as a safe FTS5 string literal, then add the
            # prefix wildcard ( "term"* matches terms starting with the token). Operator barewords
            # are dropped and special chars neutralized (so 'cat(' / 'foo:bar' are literal
            # prefixes, never an FTS5 syntax error); a user-supplied trailing '*' is stripped so
            # it is not doubled. A quoted-phrase token keeps its phrase and gets the wildcard.
            prefix_terms: list[str] = []
            for token in _FTS_TOKEN_RE.findall(query):
                # >= 2 chars so a lone '"' is not mistaken for a balanced phrase (see
                # sanitize_sqlite_fts_terms) -- otherwise it yields an unterminated FTS5 string.
                if len(token) >= 2 and token.startswith('"') and token.endswith('"'):
                    prefix_terms.append(f'{token}*')
                    continue
                clean = token.replace('-', ' ').rstrip('*').strip()
                if not clean or clean.lower() in ('and', 'or', 'not'):
                    continue
                prefix_terms.append('"' + clean.replace('"', '""') + '"*')
            if not prefix_terms:
                # Every token was an operator bareword: match NOTHING (empty result), in
                # parity with PostgreSQL's empty to_tsquery for the same input. '' is the
                # caller's "match nothing" sentinel (see _search_sqlite). The earlier
                # literal-phrase fallback ('"and"') was WRONG: FTS5's porter/unicode61
                # tokenizer keeps stopwords as tokens, so it matched every document containing
                # the dropped word while PostgreSQL returned zero -- a cross-backend divergence.
                return ''
            return ' '.join(prefix_terms)

        if mode == 'boolean':
            # Boolean mode passes the query through UNCHANGED so the user's native FTS5 boolean
            # syntax (AND/OR/NOT uppercase, parentheses, quoted phrases) reaches MATCH intact --
            # the one SQLite mode that is NOT pre-sanitized here. A malformed boolean query would
            # make FTS5 raise a grammar error; _search_sqlite catches that at execution and
            # degrades to the safe sanitized term match, so a valid query is unaffected while a
            # malformed one returns best-effort results instead of erroring (parity with
            # PostgreSQL's tolerant websearch_to_tsquery). The user is responsible for quoting
            # hyphenated words.
            return query

        # 'match' - default (AND logic). Sanitize each token to a safe FTS5 string literal so a
        # bare FTS5 operator (AND/OR/NOT, case-sensitive) or special char ((): "*^) becomes a
        # LITERAL term -- never 'fts5: syntax error'. This matches PostgreSQL's plainto_tsquery
        # (drops operator stopwords, ANDs the remaining lexemes); a quoted single word is still
        # stemmed, so ordinary-word recall is unchanged. Shared with the hybrid query builder.
        terms = sanitize_sqlite_fts_terms(_FTS_TOKEN_RE.findall(query))
        if not terms:
            # Every token was an operator/stopword bareword: match NOTHING (empty result), in
            # parity with PostgreSQL's empty plainto_tsquery for the same input. '' is the
            # caller's "match nothing" sentinel (see _search_sqlite). A literal-phrase fallback
            # ('"and"') would instead MATCH every document containing the dropped word (FTS5
            # porter/unicode61 keeps stopwords as tokens), diverging from PostgreSQL.
            return ''
        return ' '.join(terms)

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
            # Prefix matching: sanitize each whitespace token into safe AND-ed prefix
            # lexemes for the STRICT to_tsquery (the SQLite prefix path is already
            # sanitized; this keeps PostgreSQL from RAISING on adversarial input rather
            # than degrading gracefully). Drop tokens that sanitize to nothing; an
            # all-punctuation query yields '' -- an empty tsquery that simply matches
            # nothing, never a syntax error.
            words = query.split()
            prefix_terms = [frag for word in words if (frag := self._handle_hyphenated_prefix_postgresql(word))]
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
        # Import locally to avoid a repository->migrations import at module load.
        from app.migrations._pg_ddl import begin_migration
        from app.migrations._pg_ddl import execute_migration_ddl
        from app.migrations._pg_ddl import fetchval_migration
        from app.settings import get_settings

        migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s

        async def _rebuild_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            # Raise the transaction-scoped statement_timeout to the migration budget and
            # take the shared advisory lock BEFORE the rebuild. Reindexing the GIN index on
            # a large table (and the COUNT(*) probe) can run far longer than the pool's
            # command_timeout, so both must share the migration budget rather than be
            # cancelled client-side as a non-retryable error. SET LOCAL auto-reverts on
            # COMMIT/ROLLBACK.
            await begin_migration(conn, migration_timeout_s)

            # Count entries
            entry_count = await fetchval_migration(conn, 'SELECT COUNT(*) FROM context_entries', migration_timeout_s)

            # Reindex the GIN index
            await execute_migration_ddl(conn, 'REINDEX INDEX idx_text_search_gin', migration_timeout_s)

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
                        JOIN context_entries ce ON ce.rowid_int = fts.rowid
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
        async def _get_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
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
        async def _check_postgresql(conn: 'asyncpg.Connection') -> bool:
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

        async def _get_language(conn: 'asyncpg.Connection') -> str | None:
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
        return desired_sqlite_fts_tokenizer(language)

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

            # Recreate FTS5 table with new tokenizer.
            # The FTS5 ``content_rowid`` MUST point at an INTEGER PRIMARY KEY
            # column; ``context_entries.rowid_int`` is the SQLite private
            # surrogate used for this purpose, while ``context_entries.id`` is
            # the public UUIDv7 hex value exchanged across the MCP boundary.
            create_sql = f'''
                CREATE VIRTUAL TABLE context_entries_fts USING fts5(
                    text_content,
                    content='context_entries',
                    content_rowid='rowid_int',
                    tokenize='{new_tokenizer}'
                )
            '''
            conn.execute(create_sql)

            # Recreate triggers using ``rowid_int`` as the FTS5 rowid alias.
            conn.execute('''
                CREATE TRIGGER context_fts_insert AFTER INSERT ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(rowid, text_content)
                    VALUES (new.rowid_int, new.text_content);
                END
            ''')

            conn.execute('''
                CREATE TRIGGER context_fts_delete AFTER DELETE ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
                    VALUES('delete', old.rowid_int, old.text_content);
                END
            ''')

            conn.execute('''
                CREATE TRIGGER context_fts_update AFTER UPDATE OF text_content ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
                    VALUES('delete', old.rowid_int, old.text_content);
                    INSERT INTO context_entries_fts(rowid, text_content)
                    VALUES (new.rowid_int, new.text_content);
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

        # Import locally to avoid a repository->migrations import at module load.
        from app.migrations._pg_ddl import begin_migration
        from app.migrations._pg_ddl import execute_migration_ddl
        from app.migrations._pg_ddl import fetchval_migration
        from app.settings import get_settings

        migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s
        old_language = await self.get_current_language()

        async def _migrate_language(conn: 'asyncpg.Connection') -> dict[str, Any]:
            # Raise the transaction-scoped statement_timeout to the migration budget and
            # take the shared advisory lock under it BEFORE the rewrite. Recreating the
            # GENERATED ALWAYS AS (...) STORED tsvector column is the heaviest DDL of all
            # -- a full table rewrite plus a GIN index build -- so on a large existing
            # table it (and a lock wait on a peer pod's migration) must not be cancelled
            # at the pool's shorter command_timeout. SET LOCAL auto-reverts on
            # COMMIT/ROLLBACK, so no finally-restore (which would raise 25P02 in an
            # aborted transaction and mask the real DDL error) is used.
            await begin_migration(conn, migration_timeout_s)

            # Count entries for statistics
            entry_count = await fetchval_migration(conn, 'SELECT COUNT(*) FROM context_entries', migration_timeout_s)

            # Drop existing column (also drops dependent GIN index)
            await execute_migration_ddl(
                conn,
                'ALTER TABLE context_entries DROP COLUMN IF EXISTS text_search_vector',
                migration_timeout_s,
            )

            # Recreate column with new language
            await execute_migration_ddl(
                conn,
                f'''
                ALTER TABLE context_entries
                ADD COLUMN text_search_vector tsvector
                GENERATED ALWAYS AS (to_tsvector('{new_language}', COALESCE(text_content, ''))) STORED
            ''',
                migration_timeout_s,
            )

            # Recreate GIN index
            await execute_migration_ddl(
                conn,
                '''
                CREATE INDEX IF NOT EXISTS idx_text_search_gin
                ON context_entries USING GIN(text_search_vector)
            ''',
                migration_timeout_s,
            )

            return {
                'success': True,
                'backend': 'postgresql',
                'old_language': old_language,
                'new_language': new_language,
                'entries_migrated': entry_count,
            }

        return await self.backend.execute_write(cast(Any, _migrate_language))
