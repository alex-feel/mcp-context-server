"""Browse search and grep pre-filter scan over context entries.

Both read paths build their WHERE clause through one shared filter builder, so
the browse search and the server-side grep scan apply identical thread, source,
content-type, tag, date and metadata filters, followed by the caller's read
predicate, and match only the entries the caller may read.
"""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any

from pydantic import ValidationError

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.repositories.context_repository.records import CONTEXT_ENTRY_COLUMNS
from app.repositories.entry_filters import count_applied_filters

if TYPE_CHECKING:
    import asyncpg


logger = logging.getLogger(__name__)


def _escape_like(value: str) -> str:
    """Escape LIKE/ILIKE wildcards in a literal so it pre-filters exactly.

    Escapes the backslash escape character first, then the ``%`` (any run) and
    ``_`` (any single char) wildcards, for use with an explicit ``ESCAPE '\\'``
    clause on both SQLite and PostgreSQL. Keeps the optional substring pre-filter
    a tight superset of the authoritative Python match.

    Args:
        value: The literal substring to embed in a LIKE pattern.

    Returns:
        The escaped literal (still needs ``%`` sentinels added around it).
    """
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


class ContextSearchMixin(BaseRepository):
    """Browse search and grep pre-filter scan over ``context_entries``.

    ``_build_context_filter_clause`` turns the thread, source, content-type, tag,
    date and metadata arguments into one WHERE clause ending in the caller's read
    predicate, which ``search_contexts`` uses for the paginated browse search and
    ``grep_scan_text_contents`` for the exhaustive keyset scan feeding server-side
    grep.
    """

    def _build_context_filter_clause(
        self,
        *,
        scope: Scope,
        thread_id: str | None = None,
        source: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> tuple[str, list[Any], int, list[str]]:
        """Build the shared ``context_entries`` WHERE clause used by search and grep.

        Single source of truth for the portable filter surface (indexed
        thread/source/content_type, date range, metadata, and the tag subquery),
        so ``search_contexts`` and ``grep_scan_text_contents`` cannot drift. The
        caller is expected to have already emitted ``WHERE 1=1``; the returned SQL
        is a run of `` AND ...`` fragments (``''`` for the system scope with no
        filters), whose placeholders start at 1.

        The scope's read predicate is the LAST fragment, after every client filter:
        no filter placeholder moves, the predicate never counts toward
        ``filter_count`` or the metadata bind budget, and a caller appending
        ``LIMIT``/``OFFSET`` or keyset placeholders numbers them after its binds.

        Args:
            scope: The caller's scope; only entries it may read match.
            thread_id: Filter by thread id (indexed).
            source: Filter by source ('user' or 'agent', indexed).
            content_type: Filter by content type.
            tags: Filter by tags (OR logic, via the indexed tag table).
            metadata: Simple metadata equality filters.
            metadata_filters: Advanced metadata filters with operators.
            start_date: Filter by created_at >= date (ISO 8601).
            end_date: Filter by created_at <= date (ISO 8601).

        Returns:
            ``(where_sql, params, filter_count, validation_errors)``, where
            ``filter_count`` is the TOTAL number of conditions this clause emitted
            (indexed scalars + date bounds + the tag subquery + the metadata
            conditions) as counted by :func:`count_applied_filters` -- the same tally
            the FTS and semantic repositories publish, so ``search_context`` cannot
            report a different ``filters_applied`` than its sibling search tools for
            identical arguments. When ``validation_errors`` is non-empty the caller
            MUST short-circuit; ``where_sql``/``params`` are then empty.
        """
        from app.metadata_types import MetadataFilter
        from app.query_builder import MetadataQueryBuilder

        backend_type = self.backend.backend_type
        clauses: list[str] = []
        params: list[Any] = []
        validation_errors: list[str] = []

        def _next_ph() -> str:
            return self._placeholder(len(params) + 1)

        # Indexed scalar filters (thread_id + source use idx_thread_source).
        if thread_id:
            clauses.append(f' AND thread_id = {_next_ph()}')
            params.append(thread_id)
        if source:
            clauses.append(f' AND source = {_next_ph()}')
            params.append(source)
        if content_type:
            clauses.append(f' AND content_type = {_next_ph()}')
            params.append(content_type)

        # Date range. SQLite normalizes ISO 8601 via datetime(); PostgreSQL needs
        # Python datetime objects for TIMESTAMPTZ parameters.
        if start_date:
            if backend_type == 'sqlite':
                clauses.append(f' AND created_at >= datetime({_next_ph()})')
                params.append(start_date)
            else:
                clauses.append(f' AND created_at >= {_next_ph()}')
                params.append(self._parse_date_for_postgresql(start_date))
        if end_date:
            if backend_type == 'sqlite':
                clauses.append(f' AND created_at <= datetime({_next_ph()})')
                params.append(end_date)
            else:
                clauses.append(f' AND created_at <= {_next_ph()}')
                params.append(self._parse_date_for_postgresql(end_date))

        # Metadata filtering (backend-aware; PostgreSQL needs the current param offset).
        if backend_type == 'sqlite':
            metadata_builder = MetadataQueryBuilder(backend_type='sqlite')
        else:
            metadata_builder = MetadataQueryBuilder(
                backend_type='postgresql',
                param_offset=len(params),
            )

        if metadata:
            for key, value in metadata.items():
                # An invalid simple-metadata KEY (e.g. one with a space or ';' --
                # reachable because the MCP `metadata` param accepts arbitrary string keys)
                # is reported as a structured validation error and short-circuits the
                # search, exactly like an invalid advanced filter below. It must NOT raise
                # an unhandled ValueError, nor be silently dropped (which would widen the
                # result set) -- this keeps search_context, semantic, and fts consistent.
                try:
                    metadata_builder.add_simple_filter(key, value)
                except ValueError as e:
                    validation_errors.append(f'Invalid metadata key {key!r}: {e}')

        if metadata_filters:
            for filter_dict in metadata_filters:
                try:
                    filter_spec = MetadataFilter(**filter_dict)
                    metadata_builder.add_advanced_filter(filter_spec)
                except ValidationError as e:
                    validation_errors.append(f'Invalid metadata filter {filter_dict}: {e}')
                except ValueError as e:
                    validation_errors.append(f'Invalid metadata filter {filter_dict}: {e}')
                except Exception as e:
                    validation_errors.append(f'Unexpected error in metadata filter {filter_dict}: {e}')
                    logger.error(f'Unexpected error processing metadata filter: {e}')

        if validation_errors:
            return '', [], 0, validation_errors

        metadata_clause, metadata_params = metadata_builder.build_where_clause()
        if metadata_clause:
            clauses.append(f' AND {metadata_clause}')
            params.extend(metadata_params)

        # Tag filter via the indexed tag table (OR logic across normalized tags).
        # A non-empty tags list that normalizes to empty (all-blank) is a
        # structured validation error, not a dropped filter that would widen
        # results -- uniform with the metadata validators above.
        if tags:
            try:
                normalized_tags = self.normalize_tag_filter(tags)
            except ValueError as e:
                validation_errors.append(str(e))
                return '', [], 0, validation_errors
            tag_placeholders = ','.join([
                self._placeholder(len(params) + i + 1)
                for i in range(len(normalized_tags))
            ])
            clauses.append(
                f' AND id IN (SELECT DISTINCT context_entry_id FROM tags WHERE tag IN ({tag_placeholders}))',
            )
            params.extend(normalized_tags)

        filter_count = count_applied_filters(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            start_date=start_date,
            end_date=end_date,
            metadata_filter_count=metadata_builder.get_filter_count(),
        )

        # The caller's read predicate goes last, after every client filter, so it shifts
        # no filter placeholder and is never counted as a filter.
        read = build_access_predicate(
            scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries', start=len(params) + 1,
        )
        clauses.append(read.and_clause())
        params.extend(read.params)
        return ''.join(clauses), params, filter_count, validation_errors

    async def search_contexts(
        self,
        thread_id: str | None = None,
        source: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 50,
        offset: int = 0,
        explain_query: bool = False,
        *,
        scope: Scope,
    ) -> tuple[list[Any], dict[str, Any]]:
        """Search the context entries the scope may read, with metadata and date-range filtering.

        The read predicate applies in SQL before ordering and pagination, so every
        page is a window into the readable matches alone.

        Args:
            thread_id: Filter by thread ID
            source: Filter by source ('user' or 'agent')
            content_type: Filter by content type
            tags: Filter by tags (OR logic)
            metadata: Simple metadata filters (key=value)
            metadata_filters: Advanced metadata filters with operators
            start_date: Filter by created_at >= date (ISO 8601 format)
            end_date: Filter by created_at <= date (ISO 8601 format)
            limit: Maximum number of results
            offset: Pagination offset
            explain_query: If True, include query execution plan
            scope: The caller's scope; only entries it may read are returned.

        Returns:
            Tuple of (matching rows, query statistics)
            Note: Rows can be sqlite3.Row or asyncpg.Record depending on backend
        """
        import time as time_module

        if self.backend.backend_type == 'sqlite':

            def _search_sqlite(conn: sqlite3.Connection) -> tuple[list[Any], dict[str, Any]]:
                start_time = time_module.time()
                cursor = conn.cursor()

                # Build query with indexed fields first for optimization
                # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
                query = f'SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries WHERE 1=1'
                where_sql, params, filter_count, validation_errors = self._build_context_filter_clause(
                    scope=scope,
                    thread_id=thread_id,
                    source=source,
                    content_type=content_type,
                    tags=tags,
                    metadata=metadata,
                    metadata_filters=metadata_filters,
                    start_date=start_date,
                    end_date=end_date,
                )
                if validation_errors:
                    # Mirror the success-path stats shape (including 'backend' and an
                    # explicit null 'query_plan') so a validation rejection and a
                    # successful search expose the same keys.
                    return [], {
                        'error': 'Metadata filter validation failed',
                        'validation_errors': validation_errors,
                        'execution_time_ms': 0.0,
                        'filters_applied': 0,
                        'rows_returned': 0,
                        'backend': 'sqlite',
                        'query_plan': None,
                    }
                query += where_sql

                # Order and pagination - use id as secondary sort for consistency
                limit_placeholder = self._placeholder(len(params) + 1)
                offset_placeholder = self._placeholder(len(params) + 2)
                query += f' ORDER BY created_at DESC, id DESC LIMIT {limit_placeholder} OFFSET {offset_placeholder}'
                params.extend((limit, offset))

                cursor.execute(query, tuple(params))
                rows = cursor.fetchall()

                # Calculate execution time
                execution_time_ms = (time_module.time() - start_time) * 1000

                # Build statistics
                stats: dict[str, Any] = {
                    'execution_time_ms': round(execution_time_ms, 2),
                    'filters_applied': filter_count,
                    'rows_returned': len(rows),
                    'backend': 'sqlite',
                }

                # Get query plan if requested
                if explain_query:
                    cursor.execute(f'EXPLAIN QUERY PLAN {query}', tuple(params))
                    plan_rows = cursor.fetchall()
                    # Convert sqlite3.Row objects to readable format
                    plan_data: list[str] = []
                    for row in plan_rows:
                        # Convert sqlite3.Row to dict to avoid <Row object> repr
                        row_dict = dict(row)
                        # SQLite EXPLAIN QUERY PLAN columns: id, parent, notused, detail
                        id_val = row_dict.get('id', '?')
                        parent_val = row_dict.get('parent', '?')
                        notused_val = row_dict.get('notused', '?')
                        detail_val = row_dict.get('detail', '?')
                        formatted = f'id:{id_val} parent:{parent_val} notused:{notused_val} detail:{detail_val}'
                        plan_data.append(formatted)
                    stats['query_plan'] = '\n'.join(plan_data)

                # Return list of rows and statistics
                return list(rows), stats

            return await self.backend.execute_read(_search_sqlite)

        # PostgreSQL
        async def _search_postgresql(conn: 'asyncpg.Connection') -> tuple[list[Any], dict[str, Any]]:
            start_time = time_module.time()

            # Build query with indexed fields first for optimization
            # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
            query = f'SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries WHERE 1=1'
            where_sql, params, filter_count, validation_errors = self._build_context_filter_clause(
                scope=scope,
                thread_id=thread_id,
                source=source,
                content_type=content_type,
                tags=tags,
                metadata=metadata,
                metadata_filters=metadata_filters,
                start_date=start_date,
                end_date=end_date,
            )
            if validation_errors:
                # Mirror the success-path stats shape (including 'backend' and an
                # explicit null 'query_plan') so a validation rejection and a
                # successful search expose the same keys.
                return [], {
                    'error': 'Metadata filter validation failed',
                    'validation_errors': validation_errors,
                    'execution_time_ms': 0.0,
                    'filters_applied': 0,
                    'rows_returned': 0,
                    'backend': 'postgresql',
                    'query_plan': None,
                }
            query += where_sql

            # Order and pagination - use id as secondary sort for consistency
            limit_placeholder = self._placeholder(len(params) + 1)
            offset_placeholder = self._placeholder(len(params) + 2)
            query += f' ORDER BY created_at DESC, id DESC LIMIT {limit_placeholder} OFFSET {offset_placeholder}'
            params.extend((limit, offset))

            rows = await conn.fetch(query, *params)

            # Calculate execution time
            execution_time_ms = (time_module.time() - start_time) * 1000

            # Build statistics
            stats: dict[str, Any] = {
                'execution_time_ms': round(execution_time_ms, 2),
                'filters_applied': filter_count,
                'rows_returned': len(rows),
                'backend': 'postgresql',
            }

            # Get query plan if requested (PostgreSQL EXPLAIN format)
            if explain_query:
                explain_result = await conn.fetch(f'EXPLAIN {query}', *params)
                plan_data: list[str] = [record['QUERY PLAN'] for record in explain_result]
                stats['query_plan'] = '\n'.join(plan_data)

            # Return list of rows and statistics
            return list(rows), stats

        return await self.backend.execute_read(_search_postgresql)

    async def grep_scan_text_contents(
        self,
        *,
        scope: Scope,
        ascii_literal: str | None = None,
        thread_id: str | None = None,
        source: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        max_entries_scanned: int = 1000,
        aggregate_bytes_budget: int = 67108864,
        page_size: int = 200,
    ) -> tuple[list[tuple[str, str]], dict[str, Any]]:
        """Scan the ``text_content`` of the entries the scope may read, newest first, for server-side grep.

        Exhaustive keyset pagination ordered ``id DESC`` -- deliberately NOT
        ``search_contexts`` (whose ``LIMIT 50`` would cap results and make grep
        silently non-exhaustive). Returns ``(id, text_content)`` candidate rows
        (after the portable filters and an optional pure-ASCII substring
        pre-narrow); the authoritative regex/line/offset matching runs in Python
        in the tool layer. The scan is bounded by ``max_entries_scanned`` and an
        aggregate code-point budget so an unscoped thread cannot exhaust memory;
        the first entry that crosses the budget is still returned (so a single
        huge entry is never silently skipped). The read predicate is part of every
        page and lookahead statement, so an entry the scope may not read is never
        returned and never counts toward ``scanned`` or ``truncated``.

        Args:
            scope: The caller's scope; only entries it may read are scanned.
            ascii_literal: Optional pure-ASCII substring for an ``LIKE``/``ILIKE``
                pre-narrow (a superset of the Python match). None disables it.
            thread_id: Filter by thread id (indexed; bounds the scan).
            source: Filter by source ('user' or 'agent', indexed).
            content_type: Filter by content type.
            tags: Filter by tags (OR logic).
            metadata: Simple metadata equality filters.
            metadata_filters: Advanced metadata filters with operators.
            start_date: Filter by created_at >= date (ISO 8601).
            end_date: Filter by created_at <= date (ISO 8601).
            max_entries_scanned: Hard cap on candidate rows visited.
            aggregate_bytes_budget: Approximate resident-memory cap (summed
                code-point length of fetched text) before the scan stops.
            page_size: Rows fetched per keyset page.

        Returns:
            ``(rows, stats)`` where ``rows`` is a list of ``(context_id,
            text_content)`` with canonical 32-char hex ids, and ``stats`` carries
            ``scanned`` (int), ``truncated`` (bool), ``backend`` (str), and
            ``validation_errors`` (list, only when a metadata filter is invalid).
        """
        backend_type = self.backend.backend_type
        where_sql, base_params, _filter_count, validation_errors = self._build_context_filter_clause(
            scope=scope,
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            metadata=metadata,
            metadata_filters=metadata_filters,
            start_date=start_date,
            end_date=end_date,
        )
        if validation_errors:
            return [], {
                'scanned': 0,
                'truncated': False,
                'validation_errors': validation_errors,
                'backend': backend_type,
            }

        if backend_type == 'sqlite':

            def _scan_sqlite(conn: sqlite3.Connection) -> tuple[list[tuple[str, str]], dict[str, Any]]:
                cursor = conn.cursor()
                out: list[tuple[str, str]] = []
                scanned = 0
                total_chars = 0
                last_id: str | None = None
                truncated = False

                def _has_more_beyond(boundary_id: str | None) -> bool:
                    # Single-row lookahead beyond the last collected id, carrying the
                    # SAME base filters and optional ASCII pre-narrow, so a scan that
                    # stopped exactly on the final matching row is not falsely flagged
                    # truncated. Used by BOTH stop triggers (byte budget and entry cap).
                    look_params: list[Any] = list(base_params)
                    look_clause = where_sql
                    if ascii_literal is not None:
                        look_clause += f" AND text_content LIKE {self._placeholder(len(look_params) + 1)} ESCAPE '\\'"
                        look_params.append(f'%{_escape_like(ascii_literal)}%')
                    if boundary_id is not None:
                        look_clause += f' AND id < {self._placeholder(len(look_params) + 1)}'
                        look_params.append(boundary_id)
                    cursor.execute(
                        f'SELECT 1 FROM context_entries WHERE 1=1{look_clause} ORDER BY id DESC LIMIT 1',
                        tuple(look_params),
                    )
                    return cursor.fetchone() is not None

                while scanned < max_entries_scanned:
                    page_params: list[Any] = list(base_params)
                    clause = where_sql
                    if ascii_literal is not None:
                        clause += f" AND text_content LIKE {self._placeholder(len(page_params) + 1)} ESCAPE '\\'"
                        page_params.append(f'%{_escape_like(ascii_literal)}%')
                    if last_id is not None:
                        clause += f' AND id < {self._placeholder(len(page_params) + 1)}'
                        page_params.append(last_id)
                    page_limit = min(page_size, max_entries_scanned - scanned)
                    query = (
                        f'SELECT id, text_content FROM context_entries WHERE 1=1{clause} '
                        f'ORDER BY id DESC LIMIT {self._placeholder(len(page_params) + 1)}'
                    )
                    page_params.append(page_limit)
                    cursor.execute(query, tuple(page_params))
                    page = cursor.fetchall()
                    if not page:
                        break
                    budget_hit = False
                    for row in page:
                        raw_id = row['id']
                        text_value = row['text_content']
                        last_id = str(raw_id)
                        scanned += 1
                        text_str = text_value if text_value is not None else ''
                        out.append((normalize_id(str(raw_id)), text_str))
                        total_chars += len(text_str)
                        if total_chars >= aggregate_bytes_budget:
                            budget_hit = True
                            break
                    if budget_hit:
                        # Aggregate byte budget reached. A lookahead beyond the last
                        # collected row distinguishes a genuinely truncated scan from
                        # one that stopped exactly on the final matching row, mirroring
                        # the entry-cap path below.
                        truncated = _has_more_beyond(last_id)
                        break
                    if len(page) < page_limit:
                        break
                    if scanned >= max_entries_scanned:
                        # Cap reached on a full page. Distinguish exhaustion (the
                        # matching set is EXACTLY max_entries_scanned) from overflow
                        # (more remain) with the same single-row lookahead, so an
                        # exact-fit result is not falsely flagged truncated.
                        truncated = _has_more_beyond(last_id)
                        break
                return out, {'scanned': scanned, 'truncated': truncated, 'backend': 'sqlite'}

            return await self.backend.execute_read(_scan_sqlite)

        async def _scan_postgresql(conn: 'asyncpg.Connection') -> tuple[list[tuple[str, str]], dict[str, Any]]:
            out: list[tuple[str, str]] = []
            scanned = 0
            total_chars = 0
            last_id: Any | None = None
            truncated = False

            async def _has_more_beyond(boundary_id: object) -> bool:
                # Single-row lookahead beyond the last collected id, carrying the
                # SAME base filters and optional ASCII pre-narrow, so a scan that
                # stopped exactly on the final matching row is not falsely flagged
                # truncated. Used by BOTH stop triggers (byte budget and entry cap).
                look_params: list[Any] = list(base_params)
                look_clause = where_sql
                if ascii_literal is not None:
                    look_clause += f" AND text_content ILIKE {self._placeholder(len(look_params) + 1)} ESCAPE '\\'"
                    look_params.append(f'%{_escape_like(ascii_literal)}%')
                if boundary_id is not None:
                    look_clause += f' AND id < {self._placeholder(len(look_params) + 1)}'
                    look_params.append(boundary_id)
                look_hit = await conn.fetchval(
                    f'SELECT 1 FROM context_entries WHERE 1=1{look_clause} ORDER BY id DESC LIMIT 1',
                    *look_params,
                )
                return look_hit is not None

            while scanned < max_entries_scanned:
                page_params: list[Any] = list(base_params)
                clause = where_sql
                if ascii_literal is not None:
                    clause += f" AND text_content ILIKE {self._placeholder(len(page_params) + 1)} ESCAPE '\\'"
                    page_params.append(f'%{_escape_like(ascii_literal)}%')
                if last_id is not None:
                    clause += f' AND id < {self._placeholder(len(page_params) + 1)}'
                    page_params.append(last_id)
                page_limit = min(page_size, max_entries_scanned - scanned)
                query = (
                    f'SELECT id, text_content FROM context_entries WHERE 1=1{clause} '
                    f'ORDER BY id DESC LIMIT {self._placeholder(len(page_params) + 1)}'
                )
                page_params.append(page_limit)
                page = await conn.fetch(query, *page_params)
                if not page:
                    break
                budget_hit = False
                for row in page:
                    raw_id = row['id']
                    text_value = row['text_content']
                    last_id = raw_id
                    scanned += 1
                    text_str = text_value if text_value is not None else ''
                    out.append((normalize_id(str(raw_id)), text_str))
                    total_chars += len(text_str)
                    if total_chars >= aggregate_bytes_budget:
                        budget_hit = True
                        break
                if budget_hit:
                    # Aggregate byte budget reached. A lookahead beyond the last
                    # collected row distinguishes a genuinely truncated scan from
                    # one that stopped exactly on the final matching row, mirroring
                    # the entry-cap path below.
                    truncated = await _has_more_beyond(last_id)
                    break
                if len(page) < page_limit:
                    break
                if scanned >= max_entries_scanned:
                    # Cap reached on a full page. Single-row lookahead beyond
                    # last_id distinguishes exhaustion (exactly the cap) from
                    # overflow, so an exact-fit result is not falsely flagged.
                    truncated = await _has_more_beyond(last_id)
                    break
            return out, {'scanned': scanned, 'truncated': truncated, 'backend': 'postgresql'}

        return await self.backend.execute_read(_scan_postgresql)
