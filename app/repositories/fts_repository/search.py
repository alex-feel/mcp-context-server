"""Full-text search execution on SQLite FTS5 and PostgreSQL tsvector."""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

from app.repositories.base import BaseRepository
from app.repositories.entry_filters import count_applied_filters
from app.repositories.fts_repository.faults import FTS_UNPARSEABLE_QUERY_DETAIL
from app.repositories.fts_repository.faults import FtsValidationError
from app.repositories.fts_repository.faults import fts_relations_present
from app.repositories.fts_repository.faults import is_fts5_grammar_error
from app.repositories.fts_repository.faults import is_postgresql_query_failure
from app.repositories.fts_repository.query import fts_query_validation_errors
from app.repositories.fts_repository.query import get_tsquery_function
from app.repositories.fts_repository.query import transform_query_postgresql
from app.repositories.fts_repository.query import transform_query_sqlite

if TYPE_CHECKING:
    import asyncpg


logger = logging.getLogger(__name__)


class FtsSearchMixin(BaseRepository):
    """Full-text search over context entries on SQLite FTS5 and PostgreSQL tsvector.

    ``search`` validates the query and filters, then runs the backend-specific
    statement: FTS5 ``MATCH`` ranked by BM25 on SQLite, or a tsquery ranked by
    ``ts_rank_cd`` on PostgreSQL. Each branch classifies execution failures into
    client-query errors and genuine database faults, so a malformed query never
    counts against the circuit breaker.
    """

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
                SQLite: 'english' uses the 'porter unicode61' tokenizer (Porter
                stemming, so "running" matches "run"); any other value uses plain
                'unicode61' (multilingual tokenization, no stemming).
            explain_query: If True, include query execution plan in stats

        Returns:
            Tuple of (search results list, statistics dictionary)

        Raises:
            FtsValidationError: If the query contains an embedded NUL (U+0000) or an
                unpaired UTF-16 surrogate, which neither FTS5 nor the PostgreSQL wire
                protocol can parse.
        """
        # Reject strings PostgreSQL cannot bind at the shared boundary, BEFORE backend
        # dispatch. An embedded NUL (U+0000) and an unpaired UTF-16 surrogate both pass
        # JSON and Pydantic validation yet are fatal on the query path: FTS5's MATCH
        # parser reads the bound query as a NUL-terminated C string (truncating a quoted
        # literal into an 'unterminated string' grammar error that quoting cannot
        # neutralize), and asyncpg rejects any NUL-carrying or non-UTF-8-encodable text
        # bind parameter on PostgreSQL. The shared pg_bind_reject_reason probe catches
        # both sequences (a lone surrogate a bare '\x00' scan misses). Raising the
        # structured validation error here keeps the failure a CLIENT error on both
        # backends: the tools layer converts it to a clean error response, and its
        # ControlFlowError parentage keeps it out of circuit-breaker failure accounting
        # (an unparseable query can never succeed, so it says nothing about DB health).
        bind_errors = fts_query_validation_errors(query)
        if bind_errors is not None:
            raise FtsValidationError('FTS query validation failed', bind_errors)

        if self.backend.backend_type == 'sqlite':
            # Log warning if non-English language is requested with SQLite backend
            if language != 'english':
                logger.warning(
                    'SQLite FTS5 does not support language-specific stemming. '
                    "The language parameter '%s' does not enable stemming here (SQLite uses the "
                    'unicode61 tokenizer: multilingual tokenization, no stemming, so "running" '
                    'will NOT match "run"). It still governs operator-stopword parity with '
                    'PostgreSQL. Use the PostgreSQL backend for full language-specific stemming.',
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
                language=language,
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
        language: str = 'english',
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
            fts_query = transform_query_sqlite(query, mode, language)

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

            # Tag filter (uses subquery with indexed tag table). A non-empty
            # tags list that normalizes to empty (all-blank) raises through the
            # structured validation-error channel rather than silently widening
            # the result set.
            if tags:
                try:
                    normalized_tags = self.normalize_tag_filter(tags)
                except ValueError as e:
                    raise FtsValidationError(
                        'Metadata filter validation failed', [str(e)],
                    ) from e
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
                empty_filter_count = count_applied_filters(
                    thread_id=thread_id,
                    source=source,
                    content_type=content_type,
                    tags=tags,
                    start_date=start_date,
                    end_date=end_date,
                    metadata_filter_count=metadata_filter_count,
                )
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
            # ``ce.id DESC`` is an explicit, UNIQUE secondary sort key: equal-score rows are
            # common (identical or near-identical documents), and a score-only ORDER BY leaves
            # their relative order up to the scan, which is not stable across executions. The
            # LIMIT/OFFSET window would then be computed over an undefined ordering, so a client
            # paging a tied result set silently skips or duplicates rows. Matches the browse
            # path's ``created_at DESC, id DESC`` contract (newest-first on ties); the id is
            # lowercase hex on SQLite and a UUID on PostgreSQL, which order identically.
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
                ORDER BY score DESC, ce.id DESC
                LIMIT ? OFFSET ?
            '''

            # Combine params: filter_params + fts_query + limit + offset
            params = [*filter_params, fts_query, limit, offset]

            try:
                cursor = conn.execute(sql_query, params)
                rows = cursor.fetchall()
            except sqlite3.OperationalError as exc:
                # An OperationalError from one of the DATABASE-FAULT families (locked DB, disk
                # I/O, a missing FTS table) propagates unchanged for normal error handling. The
                # read-path backend wrapper charges the breaker for the disk-I/O subset but
                # EXEMPTS the SQLITE_BUSY/SQLITE_LOCKED contention family (routine self-clearing
                # cross-process contention that execute_read retries with bounded backoff on
                # fresh reader connections, re-raising uncharged only after the retry budget
                # is exhausted), matching the write path.
                # Everything else is attributed to the one client-controlled fragment of this
                # statement -- the MATCH expression -- and handled per mode below without ever
                # reaching the breaker, so no malformed query a client can send (however
                # unusual the engine's wording for it) can open the breaker on every caller.
                #
                # Boolean mode forwards the raw query to FTS5 MATCH so native FTS5 boolean
                # syntax (AND/OR/NOT, parentheses, quoted phrases) reaches the engine intact
                # -- it is the one SQLite mode not pre-sanitized. A MALFORMED boolean query
                # (unbalanced parens, a dangling operator, a stray ':' column filter) makes
                # FTS5 raise a grammar error; PostgreSQL's websearch_to_tsquery never raises
                # for the same input, so without the degradation below the identical
                # fts_search_context(mode='boolean') call hard-errors on SQLite but succeeds
                # on PostgreSQL and leaks the raw engine message to the client. Degrade a
                # malformed boolean query to the crash-safe sanitized term match (the shared
                # 'match' transform, with the configured language so the operator-bareword
                # drop keeps cross-backend parity): a VALID boolean query executed above and
                # never reaches here, while a malformed one returns best-effort results
                # instead of erroring -- matching PostgreSQL's tolerant contract without
                # altering the documented native syntax.
                if not is_fts5_grammar_error(exc, relations_present=fts_relations_present(conn)):
                    raise
                if mode != 'boolean':
                    # The match/prefix/phrase transforms sanitize every token into a
                    # quoted FTS5 literal, so a grammar error here means the input
                    # contains something quoting provably cannot neutralize. Retrying
                    # can never succeed, so classify it as a CLIENT validation error:
                    # the structured message replaces the raw engine text (no internal
                    # leak) and the ControlFlowError parentage keeps this repeatable
                    # client failure out of circuit-breaker failure accounting.
                    raise FtsValidationError(
                        'FTS query validation failed',
                        [FTS_UNPARSEABLE_QUERY_DETAIL],
                    ) from exc
                safe_query = transform_query_sqlite(query, 'match', language)
                if not safe_query.strip():
                    return _empty_result(
                        'Query short-circuited: malformed boolean query reduced to no '
                        'searchable terms; 0 rows.',
                    )
                params = [*filter_params, safe_query, limit, offset]
                try:
                    cursor = conn.execute(sql_query, params)
                    rows = cursor.fetchall()
                except sqlite3.OperationalError as retry_exc:
                    # The sanitized retry can fail too (an input the transform reduces to
                    # something FTS5 still rejects). Left bare, that second failure would
                    # propagate and charge the breaker for the very client input the
                    # degradation exists to keep off it, so it is classified exactly like
                    # the first and reported as a client validation error.
                    if not is_fts5_grammar_error(retry_exc, relations_present=fts_relations_present(conn)):
                        raise
                    raise FtsValidationError(
                        'FTS query validation failed',
                        [FTS_UNPARSEABLE_QUERY_DETAIL],
                    ) from retry_exc

            results = [dict(row) for row in rows]

            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000

            # Count filters applied (shared tally, so every search tool reports the same
            # number for the same arguments).
            filter_count = count_applied_filters(
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata_filter_count=metadata_filter_count,
            )

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

            # Tag filter (uses subquery with indexed tag table). A non-empty
            # tags list that normalizes to empty (all-blank) raises through the
            # structured validation-error channel rather than silently widening
            # the result set.
            if tags:
                try:
                    normalized_tags = self.normalize_tag_filter(tags)
                except ValueError as e:
                    raise FtsValidationError(
                        'Metadata filter validation failed', [str(e)],
                    ) from e
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
            tsquery_func = get_tsquery_function(mode, language)

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
            # BOTH ORDER BY clauses carry the unique ``id`` secondary key: ts_rank_cd ties are
            # common (identical or near-identical documents), and PostgreSQL MVCC writes a new
            # physical tuple on every UPDATE, so heap/scan order for equal-score rows changes
            # under unrelated concurrent writes. Without the tiebreak the LIMIT/OFFSET window is
            # computed over an undefined ordering and a client paging a tied result set silently
            # skips or duplicates rows; the outer re-sort would also be free to permute the page.
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
                    ORDER BY score DESC, ce.id DESC
                    LIMIT {self._placeholder(param_position)} OFFSET {self._placeholder(param_position + 1)}
                ) sub
                ORDER BY sub.score DESC, sub.id DESC
            '''

            # Transform query based on mode (for prefix mode, adds :* suffix)
            transformed_query = transform_query_postgresql(query, mode)

            # Add transformed query, limit, offset to params
            filter_params.extend([transformed_query, limit, offset])

            try:
                rows = await conn.fetch(sql_query, *filter_params)
            except Exception as exc:
                # PostgreSQL's tsquery constructors do not raise for malformed input the way
                # FTS5 does, but they DO fail at execution on an input that is merely too big
                # (a query of many thousands of terms aborts with 'invalid memory alloc request
                # size' while the tsquery is assembled). Unclassified, that failure charges the
                # process-global circuit breaker, so a client repeating one oversized query
                # rejects every other caller's reads and writes -- the same denial of service
                # the SQLite path closes, and the same reason both paths classify by provenance
                # rather than by the engine's wording. A statement-level rejection becomes a
                # structured, breaker-exempt validation error; a fault of the server or the
                # connection propagates unchanged.
                if not is_postgresql_query_failure(exc):
                    raise
                raise FtsValidationError(
                    'FTS query validation failed',
                    [FTS_UNPARSEABLE_QUERY_DETAIL],
                ) from exc

            results = [dict(row) for row in rows]

            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000

            # Count filters applied (shared tally, so every search tool reports the same
            # number for the same arguments).
            filter_count = count_applied_filters(
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                start_date=start_date,
                end_date=end_date,
                metadata_filter_count=metadata_filter_count,
            )

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
