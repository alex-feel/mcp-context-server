"""KNN search over the fp32 ``vec_context_embeddings`` table on both backends."""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.repositories.base import BaseRepository
from app.repositories.embedding_repository.records import MetadataFilterValidationError
from app.repositories.entry_filters import count_applied_filters

if TYPE_CHECKING:
    import asyncpg


logger = logging.getLogger(__name__)


class Fp32SearchMixin(BaseRepository):
    """KNN search over uncompressed fp32 embeddings.

    ``search_fp32`` filters ``context_entries`` by thread, source, content type,
    tags, dates and metadata, then by the caller's read access, ranks the
    matching chunks by L2 distance with sqlite-vec on SQLite and pgvector on
    PostgreSQL, and returns the best chunk of each entry together with the
    search statistics.
    """

    async def search_fp32(
        self,
        query_embedding: list[float],
        limit: int = 20,
        offset: int = 0,
        thread_id: str | None = None,
        source: Literal['user', 'agent'] | None = None,
        content_type: Literal['text', 'multimodal'] | None = None,
        tags: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        explain_query: bool = False,
        *,
        scope: Scope,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """KNN search over the fp32 embeddings of the entries the scope may read, with optional filters.

        SQLite: Uses CTE-based pre-filtering with vec_distance_l2() function
        PostgreSQL: Uses direct JOIN with <-> operator for L2 distance

        The read predicate joins the entry filters after every client filter and
        before the distance ranking, LIMIT and OFFSET, so an entry the scope may
        not read never takes a rank position and never counts as a filter.

        Args:
            query_embedding: Query vector for similarity search
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
            explain_query: If True, include query execution plan in stats
            scope: The caller's scope; only entries it may read are ranked.

        Returns:
            Tuple of (search results list, statistics dictionary)
        """
        if self.backend.backend_type == 'sqlite':

            def _search_sqlite(
                conn: sqlite3.Connection,
            ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
                import time as time_module

                start_time = time_module.time()

                try:
                    import sqlite_vec
                except ImportError as e:
                    raise RuntimeError(
                        'sqlite_vec package is required for semantic search. '
                        'Install: uv sync --extra embeddings-ollama (or other embeddings-* provider)',
                    ) from e

                query_blob: bytes = cast(Any, sqlite_vec).serialize_float32(query_embedding)

                filter_conditions: list[str] = []
                filter_params: list[Any] = []

                if thread_id:
                    filter_conditions.append('thread_id = ?')
                    filter_params.append(thread_id)

                if source:
                    filter_conditions.append('source = ?')
                    filter_params.append(source)

                if content_type:
                    filter_conditions.append('content_type = ?')
                    filter_params.append(content_type)

                # Tag filter (uses subquery with indexed tag table). A non-empty
                # tags list that normalizes to empty (all-blank) raises through
                # the structured validation-error channel rather than silently
                # widening the result set.
                if tags:
                    try:
                        normalized_tags = self.normalize_tag_filter(tags)
                    except ValueError as e:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed', [str(e)],
                        ) from e
                    tag_placeholders = ','.join(['?' for _ in normalized_tags])
                    filter_conditions.append(f'''
                        id IN (
                            SELECT DISTINCT context_entry_id
                            FROM tags
                            WHERE tag IN ({tag_placeholders})
                        )
                    ''')
                    filter_params.extend(normalized_tags)

                # Date range filtering - Use datetime() to normalize ISO 8601 input
                # datetime() converts all ISO 8601 formats (T separator, Z suffix, timezone offsets)
                # to SQLite's space-separated format 'YYYY-MM-DD HH:MM:SS' for proper comparison.
                # Without datetime(), TEXT comparison fails because 'T' > ' ' in ASCII ordering.
                if start_date:
                    filter_conditions.append('created_at >= datetime(?)')
                    filter_params.append(start_date)

                if end_date:
                    filter_conditions.append('created_at <= datetime(?)')
                    filter_params.append(end_date)

                # Metadata filtering using MetadataQueryBuilder
                metadata_filter_count = 0
                if metadata or metadata_filters:
                    from pydantic import ValidationError

                    from app.metadata_types import MetadataFilter
                    from app.query_builder import MetadataQueryBuilder

                    metadata_builder = MetadataQueryBuilder(backend_type='sqlite')
                    validation_errors: list[str] = []

                    # Simple metadata filters (key=value equality). An invalid KEY is
                    # reported as a structured validation error (NOT silently dropped, which
                    # would widen the result set), consistent with search_context and the
                    # advanced filters below.
                    if metadata:
                        for key, value in metadata.items():
                            try:
                                metadata_builder.add_simple_filter(key, value)
                                metadata_filter_count += 1
                            except ValueError as e:
                                validation_errors.append(f'Invalid metadata key {key!r}: {e}')

                    # Advanced metadata filters with operators
                    if metadata_filters:
                        for filter_dict in metadata_filters:
                            try:
                                filter_spec = MetadataFilter(**filter_dict)
                                metadata_builder.add_advanced_filter(filter_spec)
                                metadata_filter_count += 1
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

                    # Raise if ANY (simple key or advanced filter) validation failed,
                    # unified with search_context's structured short-circuit behavior.
                    if validation_errors:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed',
                            validation_errors,
                        )

                    # Add metadata conditions to filter
                    metadata_clause, metadata_params = metadata_builder.build_where_clause()
                    if metadata_clause:
                        filter_conditions.append(metadata_clause)
                        filter_params.extend(metadata_params)

                # The caller's read predicate follows every client filter, so it shifts no
                # filter placeholder and is never counted as a filter.
                read = build_access_predicate(
                    scope, mode=AccessMode.READ, backend_type='sqlite', outer='context_entries',
                )
                if read.sql:
                    filter_conditions.append(read.sql)
                    filter_params.extend(read.params)

                where_clause = f"WHERE {' AND '.join(filter_conditions)}" if filter_conditions else ''

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

                # Use CTE with deduplication by context_id - preserves best chunk boundaries
                # Uses subquery JOIN to identify which chunk had MIN(distance)
                query = f'''
                    WITH filtered_contexts AS (
                        SELECT id
                        FROM context_entries
                        {where_clause}
                    ),
                    chunk_distances AS (
                        SELECT
                            ec.context_id,
                            ec.start_index,
                            ec.end_index,
                            vec_distance_l2(?, ve.embedding) as distance
                        FROM filtered_contexts fc
                        JOIN embedding_chunks ec ON ec.context_id = fc.id
                        JOIN vec_context_embeddings ve ON ve.rowid = ec.vec_rowid
                    ),
                    best_chunks AS (
                        -- One row per context (the nearest chunk). ROW_NUMBER picks a
                        -- single chunk even when two chunks tie at the minimum
                        -- distance, matching the PostgreSQL DISTINCT ON path; a
                        -- MIN-equality join would emit duplicate rows for one context
                        -- on a tie. start_index is the unique per-context tiebreak, so
                        -- two equidistant chunks always resolve to the SAME reported
                        -- matched_chunk_start/end instead of whichever the scan
                        -- happened to visit first.
                        SELECT context_id, start_index, end_index, best_distance
                        FROM (
                            SELECT
                                cd.context_id,
                                cd.start_index,
                                cd.end_index,
                                cd.distance as best_distance,
                                ROW_NUMBER() OVER (
                                    PARTITION BY cd.context_id
                                    ORDER BY cd.distance, cd.start_index
                                ) as rn
                            FROM chunk_distances cd
                        )
                        WHERE rn = 1
                    )
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
                        bc.best_distance as distance,
                        bc.start_index as matched_chunk_start,
                        bc.end_index as matched_chunk_end
                    FROM best_chunks bc
                    JOIN context_entries ce ON ce.id = bc.context_id
                    -- context_id is the explicit UNIQUE secondary key: equal distances are
                    -- reachable (duplicate or near-duplicate text embeds identically), and a
                    -- distance-only ORDER BY leaves tied rows in scan order, which is not
                    -- stable across executions. The LIMIT/OFFSET window would then be computed
                    -- over an undefined ordering, so a client paging a tied result set silently
                    -- skips or duplicates rows.
                    ORDER BY bc.best_distance ASC, bc.context_id ASC
                    LIMIT ? OFFSET ?
                '''

                params = filter_params + [query_blob, limit, offset]

                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
                results = [dict(row) for row in rows]

                # Calculate execution time and build stats
                execution_time_ms = (time_module.time() - start_time) * 1000
                stats: dict[str, Any] = {
                    'execution_time_ms': round(execution_time_ms, 2),
                    'filters_applied': filter_count,
                    'rows_returned': len(results),
                    'backend': 'sqlite',
                }

                # Get query plan if requested
                if explain_query:
                    cursor = conn.execute(f'EXPLAIN QUERY PLAN {query}', params)
                    plan_rows = cursor.fetchall()
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

            return await self.backend.execute_read(_search_sqlite)

        # postgresql
        async def _search_postgresql(
            conn: 'asyncpg.Connection',
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            import time as time_module

            start_time = time_module.time()

            filter_conditions = ['1=1']  # Always true, makes building easier
            filter_params: list[Any] = [query_embedding]
            param_position = 2  # Start at 2 because $1 is embedding

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
                    raise MetadataFilterValidationError(
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

            # Date range filtering - PostgreSQL uses TIMESTAMPTZ comparison
            # asyncpg requires Python datetime objects, not strings, for TIMESTAMPTZ parameters
            if start_date:
                filter_conditions.append(f'ce.created_at >= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(start_date))
                param_position += 1

            if end_date:
                filter_conditions.append(f'ce.created_at <= {self._placeholder(param_position)}')
                filter_params.append(self._parse_date_for_postgresql(end_date))
                param_position += 1

            # Metadata filtering using MetadataQueryBuilder
            metadata_filter_count = 0
            if metadata or metadata_filters:
                from pydantic import ValidationError

                from app.metadata_types import MetadataFilter
                from app.query_builder import MetadataQueryBuilder

                # param_offset is the current number of params minus 1 because MetadataQueryBuilder
                # uses 1-based indexing and we need to continue from the current position
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
                            metadata_filter_count += 1
                        except ValueError as e:
                            validation_errors.append(f'Invalid metadata key {key!r}: {e}')

                # Advanced metadata filters with operators
                if metadata_filters:
                    for filter_dict in metadata_filters:
                        try:
                            filter_spec = MetadataFilter(**filter_dict)
                            metadata_builder.add_advanced_filter(filter_spec)
                            metadata_filter_count += 1
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

                # Raise if ANY (simple key or advanced filter) validation failed, unified
                # with search_context's structured short-circuit behavior.
                if validation_errors:
                    raise MetadataFilterValidationError(
                        'Metadata filter validation failed',
                        validation_errors,
                    )

                # The builder emits the metadata conditions already qualified with the
                # 'ce.' table alias (table_alias='ce'), matching the context_entries
                # JOIN target without rewriting the built SQL (a global str.replace
                # would corrupt JSON keys that contain the substring 'metadata').
                metadata_clause, metadata_params = metadata_builder.build_where_clause()
                if metadata_clause:
                    filter_conditions.append(metadata_clause)
                    filter_params.extend(metadata_params)
                    param_position += len(metadata_params)

            # The caller's read predicate follows every client filter, so it shifts no
            # filter placeholder and is never counted as a filter; LIMIT and OFFSET are
            # numbered after its binds.
            read = build_access_predicate(
                scope, mode=AccessMode.READ, backend_type='postgresql', outer='ce', start=param_position,
            )
            if read.sql:
                filter_conditions.append(read.sql)
                filter_params.extend(read.params)
                param_position += read.bind_count

            where_clause = ' AND '.join(filter_conditions)

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

            # Use CTE with DISTINCT ON to preserve best chunk boundaries
            # DISTINCT ON selects first row per context_id when ordered by distance;
            # start_index breaks a tie between two equidistant chunks of the same
            # context so the reported matched_chunk_start/end is deterministic.
            query = f'''
                    WITH chunk_distances AS (
                        SELECT
                            ve.context_id,
                            ve.start_index,
                            ve.end_index,
                            ve.embedding <-> {self._placeholder(1)} as distance
                        FROM vec_context_embeddings ve
                        JOIN context_entries ce ON ce.id = ve.context_id
                        WHERE {where_clause}
                    ),
                    best_chunks AS (
                        SELECT DISTINCT ON (context_id)
                            context_id,
                            start_index,
                            end_index,
                            distance as best_distance
                        FROM chunk_distances
                        ORDER BY context_id, distance, start_index
                    )
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
                        bc.best_distance as distance,
                        bc.start_index as matched_chunk_start,
                        bc.end_index as matched_chunk_end
                    FROM best_chunks bc
                    JOIN context_entries ce ON ce.id = bc.context_id
                    -- context_id is the explicit UNIQUE secondary key. Equal distances are
                    -- reachable (duplicate or near-duplicate text embeds identically) and
                    -- PostgreSQL MVCC writes a new physical tuple on every UPDATE, so heap/scan
                    -- order for tied rows changes under unrelated concurrent writes; without the
                    -- tiebreak the LIMIT/OFFSET window is computed over an undefined ordering and
                    -- a client paging a tied result set silently skips or duplicates rows.
                    ORDER BY bc.best_distance ASC, bc.context_id ASC
                    LIMIT {self._placeholder(param_position)} OFFSET {self._placeholder(param_position + 1)}
                '''

            filter_params.extend([limit, offset])

            rows = await conn.fetch(query, *filter_params)
            results = [dict(row) for row in rows]

            # Calculate execution time and build stats
            execution_time_ms = (time_module.time() - start_time) * 1000
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': filter_count,
                'rows_returned': len(results),
                'backend': 'postgresql',
            }

            # Get query plan if requested
            if explain_query:
                plan_result = await conn.fetch(f'EXPLAIN {query}', *filter_params)
                plan_data = [str(row[0]) for row in plan_result]
                stats['query_plan'] = '\n'.join(plan_data)

            return results, stats

        return await self.backend.execute_read(_search_postgresql)
