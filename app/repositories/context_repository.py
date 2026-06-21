"""
Context repository for managing context entries.

This module handles all database operations related to context entries,
including CRUD operations and deduplication logic.
"""


import hashlib
import json
import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from pydantic import ValidationError

from app.backends.base import StorageBackend
from app.ids import generate_id
from app.ids import normalize_id
from app.repositories.base import BaseRepository

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext

logger = logging.getLogger(__name__)

# Explicit column list to avoid exposing internal database columns (e.g., text_search_vector)
# This constant is used in all SELECT queries that return context entries to ensure
# only the expected columns are returned, preventing internal PostgreSQL columns from
# leaking into API responses.
CONTEXT_ENTRY_COLUMNS = 'id, thread_id, source, content_type, text_content, metadata, summary, created_at, updated_at'


def compute_content_hash(text: str) -> str:
    """Compute SHA-256 hash of text content for deduplication.

    Used to avoid transferring full text_content over the network when
    checking for duplicates. The hash is stored alongside text_content
    and compared instead of the full text.

    Args:
        text: The text content to hash.

    Returns:
        SHA-256 hex digest string (64 characters).
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


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


class ContextRepository(BaseRepository):
    """Repository for context entry operations.

    Handles storage, retrieval, search, and deletion of context entries
    with proper deduplication and transaction management.
    """

    def __init__(self, backend: StorageBackend) -> None:
        """Initialize context repository.

        Args:
            backend: Storage backend for executing database operations
        """
        super().__init__(backend)

    async def store_with_deduplication(
        self,
        thread_id: str,
        source: str,
        content_type: str,
        text_content: str,
        metadata: str | None = None,
        summary: str | None = None,
        txn: 'TransactionContext | None' = None,
    ) -> tuple[str, bool]:
        """Store context entry with deduplication logic.

        Checks if the latest entry has identical thread_id, source, and text_content.
        If found, updates metadata (via COALESCE), content_type, and updated_at.
        Otherwise, inserts new entry.

        Args:
            thread_id: Thread identifier
            source: 'user' or 'agent'
            content_type: 'text' or 'multimodal'
            text_content: The actual text content
            metadata: JSON metadata string or None
            summary: LLM-generated summary text or None
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Tuple of (context_id, was_updated) where was_updated=True means
            an existing entry was updated, False means new entry was inserted.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        # Normalize empty/whitespace summary to None for proper COALESCE behavior.
        # COALESCE(NULL, existing_value) preserves existing; COALESCE("", existing_value) overwrites.
        if summary is not None and not summary.strip():
            summary = None

        # Compute content hash for deduplication optimization.
        # Avoids transferring full text_content over the network for duplicate checks.
        content_hash = compute_content_hash(text_content)

        if backend_type == 'sqlite':

            def _store_sqlite(conn: sqlite3.Connection) -> tuple[str, bool]:
                cursor = conn.cursor()

                # Check if the LATEST entry (by id) for this thread_id and source is a duplicate.
                # Fetches content_hash instead of full text_content to reduce data transfer.
                # Falls back to text comparison when content_hash is NULL (pre-migration rows).
                cursor.execute(
                    f'''
                    SELECT id, content_hash, text_content FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                    ORDER BY id DESC
                    LIMIT 1
                    ''',
                    (thread_id, source),
                )

                latest_row = cursor.fetchone()

                is_duplicate = False
                if latest_row:
                    existing_hash = latest_row['content_hash']
                    if existing_hash is not None:
                        # Hash-based comparison (fast path)
                        is_duplicate = existing_hash == content_hash
                    else:
                        # Fallback for pre-migration rows without content_hash
                        is_duplicate = latest_row['text_content'] == text_content

                if is_duplicate and latest_row:
                    # Interleaving check: suppress dedup if opposite-source entries exist
                    # after the candidate. This preserves chronological ordering when identical
                    # text is sent as a new conversational turn rather than a retry.
                    # Intentionally duplicated across 4 blocks (SQLite/PostgreSQL x store/check)
                    # because sync/async closure patterns prevent clean extraction without
                    # losing type safety.
                    existing_id = latest_row['id']
                    opposite_source = 'agent' if source == 'user' else 'user'
                    cursor.execute(
                        f'''
                        SELECT 1 FROM context_entries
                        WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                        AND id > {self._placeholder(3)}
                        LIMIT 1
                        ''',
                        (thread_id, opposite_source, existing_id),
                    )
                    if cursor.fetchone() is not None:
                        is_duplicate = False

                if is_duplicate and latest_row:
                    # The latest entry has identical text - update metadata, content_type, and timestamp
                    existing_id = latest_row['id']
                    cursor.execute(
                        f'''
                        UPDATE context_entries
                        SET metadata = COALESCE({self._placeholder(1)}, metadata),
                            content_type = {self._placeholder(2)},
                            summary = COALESCE({self._placeholder(3)}, summary),
                            content_hash = {self._placeholder(4)},
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = {self._placeholder(5)}
                        ''',
                        (metadata, content_type, summary, content_hash, existing_id),
                    )
                    rows_affected = cursor.rowcount
                    if rows_affected == 0:
                        logger.warning(
                            'Deduplication UPDATE affected 0 rows for context %s in thread %s',
                            existing_id, thread_id,
                        )
                    logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                    return existing_id, True

                # No duplicate - insert new entry with pre-generated UUIDv7 hex id.
                new_id = generate_id()
                cursor.execute(
                    f'''
                    INSERT INTO context_entries
                    (id, thread_id, source, content_type, text_content, metadata, summary, content_hash)
                    VALUES ({self._placeholders(8)})
                    ''',
                    (new_id, thread_id, source, content_type, text_content, metadata, summary, content_hash),
                )
                logger.debug(f'Inserted new context entry {new_id} for thread {thread_id}')
                return new_id, False

            if txn:
                return _store_sqlite(cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_store_sqlite)

        # PostgreSQL
        # Note: TYPE_CHECKING ensures asyncpg.Connection type is only used during type checking
        async def _store_postgresql(conn: 'asyncpg.Connection') -> tuple[str, bool]:
            # Check latest entry - fetches content_hash instead of full text_content.
            # Falls back to text comparison when content_hash is NULL (pre-migration rows).
            latest_row = await conn.fetchrow(
                f'''
                    SELECT id, content_hash, text_content FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                    ORDER BY id DESC
                    LIMIT 1
                    ''',
                thread_id,
                source,
            )

            is_duplicate = False
            if latest_row:
                existing_hash = latest_row['content_hash']
                if existing_hash is not None:
                    is_duplicate = existing_hash == content_hash
                else:
                    is_duplicate = latest_row['text_content'] == text_content

            if is_duplicate and latest_row:
                # Interleaving check: suppress dedup if opposite-source entries exist
                # after the candidate. This preserves chronological ordering when identical
                # text is sent as a new conversational turn rather than a retry.
                # Intentionally duplicated across 4 blocks (SQLite/PostgreSQL x store/check)
                # because sync/async closure patterns prevent clean extraction without
                # losing type safety.
                existing_id = latest_row['id']
                opposite_source = 'agent' if source == 'user' else 'user'
                interleaving_row = await conn.fetchrow(
                    f'''
                        SELECT 1 FROM context_entries
                        WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                        AND id > {self._placeholder(3)}
                        LIMIT 1
                        ''',
                    thread_id,
                    opposite_source,
                    existing_id,
                )
                if interleaving_row is not None:
                    is_duplicate = False

            if is_duplicate and latest_row:
                # Update metadata, content_type, and timestamp
                existing_id = latest_row['id']
                result = await conn.execute(
                    f'''
                        UPDATE context_entries
                        SET metadata = COALESCE({self._placeholder(1)}, metadata),
                            content_type = {self._placeholder(2)},
                            summary = COALESCE({self._placeholder(3)}, summary),
                            content_hash = {self._placeholder(4)},
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = {self._placeholder(5)}
                        ''',
                    metadata,
                    content_type,
                    summary,
                    content_hash,
                    existing_id,
                )
                rows_affected = int(result.split()[-1]) if result else 0
                if rows_affected == 0:
                    logger.warning(
                        'Deduplication UPDATE affected 0 rows for context %s in thread %s',
                        existing_id, thread_id,
                    )
                logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                return existing_id, True

            # No duplicate - insert new entry with pre-generated UUIDv7 hex id.
            new_id = generate_id()
            await conn.execute(
                f'''
                    INSERT INTO context_entries
                    (id, thread_id, source, content_type, text_content, metadata, summary, content_hash)
                    VALUES ({self._placeholders(8)})
                    ''',
                new_id,
                thread_id,
                source,
                content_type,
                text_content,
                metadata,
                summary,
                content_hash,
            )
            logger.debug(f'Inserted new context entry {new_id} for thread {thread_id}')
            return new_id, False

        if txn:
            return await _store_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_store_postgresql)

    async def check_latest_is_duplicate(
        self,
        thread_id: str,
        source: str,
        text_content: str,
    ) -> str | None:
        """Check if the latest entry matches the given content (read-only pre-check).

        This is a performance optimization for the embedding-first pattern.
        It allows skipping expensive embedding generation when the content
        is identical to the latest entry. The in-transaction deduplication
        in store_with_deduplication remains as the authoritative safety net.

        Includes an interleaving check: if opposite-source entries (agent for
        user source, user for agent source) exist after the candidate duplicate,
        returns None to suppress deduplication. This preserves chronological
        ordering when identical text is sent as a new conversational turn
        rather than a retry.

        Args:
            thread_id: Thread identifier
            source: 'user' or 'agent'
            text_content: Text content to check for duplicates

        Returns:
            The context_id of the matching entry if duplicate found, None if no
            match or if interleaving entries suppress deduplication.
        """
        content_hash = compute_content_hash(text_content)

        if self.backend.backend_type == 'sqlite':

            def _check_sqlite(conn: sqlite3.Connection) -> str | None:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT id, content_hash, text_content FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                    ORDER BY id DESC
                    LIMIT 1
                    ''',
                    (thread_id, source),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                # Hash-based comparison; fall back to text for pre-migration rows (NULL hash)
                existing_hash = row['content_hash']
                is_match = (
                    existing_hash == content_hash
                    if existing_hash is not None
                    else row['text_content'] == text_content
                )
                if not is_match:
                    return None
                candidate_id = row['id']
                # Interleaving check: suppress dedup if opposite-source entries exist
                # after the candidate. This preserves chronological ordering when identical
                # text is sent as a new conversational turn rather than a retry.
                # Intentionally duplicated across 4 blocks (SQLite/PostgreSQL x store/check)
                # because sync/async closure patterns prevent clean extraction without
                # losing type safety.
                opposite_source = 'agent' if source == 'user' else 'user'
                cursor.execute(
                    f'''
                    SELECT 1 FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                    AND id > {self._placeholder(3)}
                    LIMIT 1
                    ''',
                    (thread_id, opposite_source, candidate_id),
                )
                if cursor.fetchone() is not None:
                    return None
                return cast(str, candidate_id)

            return await self.backend.execute_read(_check_sqlite)

        # PostgreSQL
        async def _check_postgresql(conn: 'asyncpg.Connection') -> str | None:
            row = await conn.fetchrow(
                f'''
                    SELECT id, content_hash, text_content FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                    ORDER BY id DESC
                    LIMIT 1
                    ''',
                thread_id,
                source,
            )
            if not row:
                return None
            # Hash-based comparison; fall back to text for pre-migration rows (NULL hash)
            existing_hash = row['content_hash']
            is_match = (
                existing_hash == content_hash
                if existing_hash is not None
                else row['text_content'] == text_content
            )
            if not is_match:
                return None
            candidate_id = row['id']
            # Interleaving check: suppress dedup if opposite-source entries exist
            # after the candidate. This preserves chronological ordering when identical
            # text is sent as a new conversational turn rather than a retry.
            # Intentionally duplicated across 4 blocks (SQLite/PostgreSQL x store/check)
            # because sync/async closure patterns prevent clean extraction without
            # losing type safety.
            opposite_source = 'agent' if source == 'user' else 'user'
            interleaving_row = await conn.fetchrow(
                f'''
                    SELECT 1 FROM context_entries
                    WHERE thread_id = {self._placeholder(1)} AND source = {self._placeholder(2)}
                    AND id > {self._placeholder(3)}
                    LIMIT 1
                    ''',
                thread_id,
                opposite_source,
                candidate_id,
            )
            if interleaving_row is not None:
                return None
            return cast(str, candidate_id)

        return await self.backend.execute_read(_check_postgresql)

    async def get_summary(self, context_id: str) -> str | None:
        """Get the summary for a context entry.

        Used during deduplication to check if a summary already exists,
        avoiding unnecessary LLM calls for duplicate entries.

        Args:
            context_id: ID of the context entry.

        Returns:
            Summary string if exists, None otherwise.
        """
        if self.backend.backend_type == 'sqlite':

            def _get_summary_sqlite(conn: sqlite3.Connection) -> str | None:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT summary FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                row = cursor.fetchone()
                if row:
                    value = cast(str | None, row['summary'])
                    # Normalize empty/whitespace to None for consistent behavior
                    if value is not None and not value.strip():
                        return None
                    return value
                return None

            return await self.backend.execute_read(_get_summary_sqlite)

        # PostgreSQL
        async def _get_summary_postgresql(conn: 'asyncpg.Connection') -> str | None:
            row = await conn.fetchrow(
                f'SELECT summary FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            if row:
                value = cast(str | None, row['summary'])
                # Normalize empty/whitespace to None for consistent behavior
                if value is not None and not value.strip():
                    return None
                return value
            return None

        return await self.backend.execute_read(_get_summary_postgresql)

    def _build_context_filter_clause(
        self,
        *,
        thread_id: str | None = None,
        source: str | None = None,
        content_type: str | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, str | int | float | bool] | None = None,
        metadata_filters: list[dict[str, Any]] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        params_start: int = 0,
    ) -> tuple[str, list[Any], int, list[str]]:
        """Build the shared ``context_entries`` WHERE clause used by search and grep.

        Single source of truth for the portable filter surface (indexed
        thread/source/content_type, date range, metadata, and the tag subquery),
        so ``search_contexts`` and ``grep_scan_text_contents`` cannot drift. The
        caller is expected to have already emitted ``WHERE 1=1``; the returned SQL
        is a run of `` AND ...`` fragments (or ``''`` when no filters apply).

        Args:
            thread_id: Filter by thread id (indexed).
            source: Filter by source ('user' or 'agent', indexed).
            content_type: Filter by content type.
            tags: Filter by tags (OR logic, via the indexed tag table).
            metadata: Simple metadata equality filters.
            metadata_filters: Advanced metadata filters with operators.
            start_date: Filter by created_at >= date (ISO 8601).
            end_date: Filter by created_at <= date (ISO 8601).
            params_start: Number of bind parameters already emitted before this
                clause, so PostgreSQL ``$n`` positions continue correctly.

        Returns:
            ``(where_sql, params, filter_count, validation_errors)``. When
            ``validation_errors`` is non-empty the caller MUST short-circuit;
            ``where_sql``/``params`` are then empty.
        """
        from app.metadata_types import MetadataFilter
        from app.query_builder import MetadataQueryBuilder

        backend_type = self.backend.backend_type
        clauses: list[str] = []
        params: list[Any] = []
        validation_errors: list[str] = []

        def _next_ph() -> str:
            return self._placeholder(params_start + len(params) + 1)

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
                param_offset=params_start + len(params),
            )

        if metadata:
            for key, value in metadata.items():
                metadata_builder.add_simple_filter(key, value)

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
        if tags:
            normalized_tags = [tag.strip().lower() for tag in tags if tag.strip()]
            if normalized_tags:
                tag_placeholders = ','.join([
                    self._placeholder(params_start + len(params) + i + 1)
                    for i in range(len(normalized_tags))
                ])
                clauses.append(
                    f' AND id IN (SELECT DISTINCT context_entry_id FROM tags WHERE tag IN ({tag_placeholders}))',
                )
                params.extend(normalized_tags)

        return ''.join(clauses), params, metadata_builder.get_filter_count(), validation_errors

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
    ) -> tuple[list[Any], dict[str, Any]]:
        """Search for context entries with filtering including metadata and date range.

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
                        'error': 'Metadata filter validation failed',
                        'validation_errors': validation_errors,
                        'execution_time_ms': 0.0,
                        'filters_applied': 0,
                        'rows_returned': 0,
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
                    'error': 'Metadata filter validation failed',
                    'validation_errors': validation_errors,
                    'execution_time_ms': 0.0,
                    'filters_applied': 0,
                    'rows_returned': 0,
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
        """Scan ``text_content`` newest-first for a server-side grep pre-filter.

        Exhaustive keyset pagination ordered ``id DESC`` -- deliberately NOT
        ``search_contexts`` (whose ``LIMIT 50`` would cap results and make grep
        silently non-exhaustive). Returns ``(id, text_content)`` candidate rows
        (after the portable filters and an optional pure-ASCII substring
        pre-narrow); the authoritative regex/line/offset matching runs in Python
        in the tool layer. The scan is bounded by ``max_entries_scanned`` and an
        aggregate code-point budget so an unscoped thread cannot exhaust memory;
        the first entry that crosses the budget is still returned (so a single
        huge entry is never silently skipped).

        Args:
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
                    for row in page:
                        raw_id = row['id']
                        text_value = row['text_content']
                        last_id = str(raw_id)
                        scanned += 1
                        text_str = text_value if text_value is not None else ''
                        out.append((normalize_id(str(raw_id)), text_str))
                        total_chars += len(text_str)
                        if total_chars >= aggregate_bytes_budget:
                            truncated = True
                            break
                    if truncated or len(page) < page_limit:
                        break
                    if scanned >= max_entries_scanned:
                        # Cap reached on a full page. Distinguish exhaustion (the
                        # matching set is EXACTLY max_entries_scanned) from overflow
                        # (more remain) with a single-row lookahead beyond last_id,
                        # so an exact-fit result is not falsely flagged truncated.
                        look_params: list[Any] = list(base_params)
                        look_clause = where_sql
                        if ascii_literal is not None:
                            look_clause += f" AND text_content LIKE {self._placeholder(len(look_params) + 1)} ESCAPE '\\'"
                            look_params.append(f'%{_escape_like(ascii_literal)}%')
                        if last_id is not None:
                            look_clause += f' AND id < {self._placeholder(len(look_params) + 1)}'
                            look_params.append(last_id)
                        cursor.execute(
                            f'SELECT 1 FROM context_entries WHERE 1=1{look_clause} ORDER BY id DESC LIMIT 1',
                            tuple(look_params),
                        )
                        truncated = cursor.fetchone() is not None
                        break
                return out, {'scanned': scanned, 'truncated': truncated, 'backend': 'sqlite'}

            return await self.backend.execute_read(_scan_sqlite)

        async def _scan_postgresql(conn: 'asyncpg.Connection') -> tuple[list[tuple[str, str]], dict[str, Any]]:
            out: list[tuple[str, str]] = []
            scanned = 0
            total_chars = 0
            last_id: Any | None = None
            truncated = False
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
                for row in page:
                    raw_id = row['id']
                    text_value = row['text_content']
                    last_id = raw_id
                    scanned += 1
                    text_str = text_value if text_value is not None else ''
                    out.append((normalize_id(str(raw_id)), text_str))
                    total_chars += len(text_str)
                    if total_chars >= aggregate_bytes_budget:
                        truncated = True
                        break
                if truncated or len(page) < page_limit:
                    break
                if scanned >= max_entries_scanned:
                    # Cap reached on a full page. Single-row lookahead beyond
                    # last_id distinguishes exhaustion (exactly the cap) from
                    # overflow, so an exact-fit result is not falsely flagged.
                    look_params: list[Any] = list(base_params)
                    look_clause = where_sql
                    if ascii_literal is not None:
                        look_clause += f" AND text_content ILIKE {self._placeholder(len(look_params) + 1)} ESCAPE '\\'"
                        look_params.append(f'%{_escape_like(ascii_literal)}%')
                    if last_id is not None:
                        look_clause += f' AND id < {self._placeholder(len(look_params) + 1)}'
                        look_params.append(last_id)
                    look_hit = await conn.fetchval(
                        f'SELECT 1 FROM context_entries WHERE 1=1{look_clause} ORDER BY id DESC LIMIT 1',
                        *look_params,
                    )
                    truncated = look_hit is not None
                    break
            return out, {'scanned': scanned, 'truncated': truncated, 'backend': 'postgresql'}

        return await self.backend.execute_read(_scan_postgresql)

    async def get_by_ids(self, context_ids: list[str]) -> list[Any]:
        """Get context entries by their IDs.

        Args:
            context_ids: List of context entry IDs

        Returns:
            List of context entry rows (sqlite3.Row or asyncpg.Record depending on backend)
        """
        # Defensive check: return empty list if no IDs provided
        # Prevents SQL syntax errors when constructing IN clauses
        if not context_ids:
            return []

        if self.backend.backend_type == 'sqlite':

            def _fetch_sqlite(conn: sqlite3.Connection) -> list[Any]:
                cursor = conn.cursor()
                placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
                # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
                query = f'''
                    SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries
                    WHERE id IN ({placeholders})
                    ORDER BY created_at DESC
                '''
                cursor.execute(query, tuple(context_ids))
                return list(cursor.fetchall())

            return await self.backend.execute_read(_fetch_sqlite)

        # PostgreSQL
        async def _fetch_postgresql(conn: 'asyncpg.Connection') -> list[Any]:
            placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
            # Use explicit column list to avoid exposing internal columns (e.g., text_search_vector)
            query = f'''
                SELECT {CONTEXT_ENTRY_COLUMNS} FROM context_entries
                WHERE id IN ({placeholders})
                ORDER BY created_at DESC
            '''
            rows = await conn.fetch(query, *context_ids)
            return list(rows)

        return await self.backend.execute_read(_fetch_postgresql)

    async def delete_by_ids(
        self,
        context_ids: list[str],
        txn: 'TransactionContext | None' = None,
    ) -> int:
        """Delete context entries by their IDs.

        Args:
            context_ids: List of context entry IDs to delete
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Number of deleted entries
        """
        # Defensive check: return 0 if no IDs provided
        # Prevents SQL syntax errors when constructing IN clauses
        if not context_ids:
            return 0

        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _delete_by_ids_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
                cursor.execute(
                    f'DELETE FROM context_entries WHERE id IN ({placeholders})',
                    tuple(context_ids),
                )
                return cursor.rowcount

            if txn:
                return _delete_by_ids_sqlite(cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_delete_by_ids_sqlite)

        # PostgreSQL
        async def _delete_by_ids_postgresql(conn: 'asyncpg.Connection') -> int:
            placeholders = ','.join([self._placeholder(i + 1) for i in range(len(context_ids))])
            result = await conn.execute(
                f'DELETE FROM context_entries WHERE id IN ({placeholders})',
                *context_ids,
            )
            # asyncpg returns "DELETE N" where N is the count
            return int(result.split()[-1]) if result else 0

        if txn:
            return await _delete_by_ids_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_delete_by_ids_postgresql)

    async def delete_by_thread(self, thread_id: str) -> int:
        """Delete all context entries in a thread.

        Args:
            thread_id: Thread ID to delete entries from

        Returns:
            Number of deleted entries
        """
        if self.backend.backend_type == 'sqlite':

            def _delete_by_thread_sqlite(conn: sqlite3.Connection) -> int:
                cursor = conn.cursor()
                cursor.execute(
                    f'DELETE FROM context_entries WHERE thread_id = {self._placeholder(1)}',
                    (thread_id,),
                )
                return cursor.rowcount

            return await self.backend.execute_write(_delete_by_thread_sqlite)

        # PostgreSQL
        async def _delete_by_thread_postgresql(conn: 'asyncpg.Connection') -> int:
            result = await conn.execute(
                f'DELETE FROM context_entries WHERE thread_id = {self._placeholder(1)}',
                thread_id,
            )
            # asyncpg returns "DELETE N" where N is the count
            return int(result.split()[-1]) if result else 0

        return await self.backend.execute_write(_delete_by_thread_postgresql)

    async def update_context_entry(
        self,
        context_id: str,
        text_content: str | None = None,
        metadata: str | None = None,
        summary: str | None = None,
        clear_summary: bool = False,
        txn: 'TransactionContext | None' = None,
    ) -> tuple[bool, list[str]]:
        """Update text content and/or metadata of a context entry.

        Args:
            context_id: ID of the context entry to update
            text_content: New text content (if provided)
            metadata: New metadata JSON string (if provided)
            summary: New LLM-generated summary text (if provided)
            clear_summary: If True, explicitly set summary to NULL in the database.
                Takes precedence over summary parameter.
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Tuple of (success, list_of_updated_fields)
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _update_entry_sqlite(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
                cursor = conn.cursor()
                updated_fields: list[str] = []

                # First, check if the entry exists
                cursor.execute(
                    f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                if not cursor.fetchone():
                    return False, []

                # Build update query dynamically based on provided fields
                update_parts: list[str] = []
                params: list[Any] = []

                if text_content is not None:
                    update_parts.extend([
                        f'text_content = {self._placeholder(len(params) + 1)}',
                        f'content_hash = {self._placeholder(len(params) + 2)}',
                    ])
                    params.extend([text_content, compute_content_hash(text_content)])
                    updated_fields.append('text_content')

                if metadata is not None:
                    update_parts.append(f'metadata = {self._placeholder(len(params) + 1)}')
                    params.append(metadata)
                    updated_fields.append('metadata')

                if clear_summary:
                    update_parts.append('summary = NULL')
                    updated_fields.append('summary')
                elif summary is not None:
                    update_parts.append(f'summary = {self._placeholder(len(params) + 1)}')
                    params.append(summary)
                    updated_fields.append('summary')

                # If no fields to update, return early
                if not update_parts:
                    return False, []

                # Always update the updated_at timestamp
                update_parts.append('updated_at = CURRENT_TIMESTAMP')

                # Execute update
                query = f"UPDATE context_entries SET {', '.join(update_parts)} WHERE id = {self._placeholder(len(params) + 1)}"
                params.append(context_id)
                cursor.execute(query, tuple(params))

                # Check if any rows were affected
                if cursor.rowcount > 0:
                    logger.debug(f'Updated context entry {context_id}, fields: {updated_fields}')
                    return True, updated_fields

                return False, []

            if txn:
                return _update_entry_sqlite(cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_update_entry_sqlite)

        # PostgreSQL
        async def _update_entry_postgresql(conn: 'asyncpg.Connection') -> tuple[bool, list[str]]:
            updated_fields: list[str] = []

            # First, check if the entry exists
            row = await conn.fetchrow(
                f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            if not row:
                return False, []

            # Build update query dynamically based on provided fields
            update_parts: list[str] = []
            params: list[Any] = []

            if text_content is not None:
                update_parts.extend([
                    f'text_content = {self._placeholder(len(params) + 1)}',
                    f'content_hash = {self._placeholder(len(params) + 2)}',
                ])
                params.extend([text_content, compute_content_hash(text_content)])
                updated_fields.append('text_content')

            if metadata is not None:
                update_parts.append(f'metadata = {self._placeholder(len(params) + 1)}')
                params.append(metadata)
                updated_fields.append('metadata')

            if clear_summary:
                update_parts.append('summary = NULL')
                updated_fields.append('summary')
            elif summary is not None:
                update_parts.append(f'summary = {self._placeholder(len(params) + 1)}')
                params.append(summary)
                updated_fields.append('summary')

            # If no fields to update, return early
            if not update_parts:
                return False, []

            # Always update the updated_at timestamp
            update_parts.append('updated_at = CURRENT_TIMESTAMP')

            # Execute update
            query = f"UPDATE context_entries SET {', '.join(update_parts)} WHERE id = {self._placeholder(len(params) + 1)}"
            params.append(context_id)
            result = await conn.execute(query, *params)

            # Check if any rows were affected (asyncpg returns "UPDATE N")
            rows_affected = int(result.split()[-1]) if result else 0
            if rows_affected > 0:
                logger.debug(f'Updated context entry {context_id}, fields: {updated_fields}')
                return True, updated_fields

            return False, []

        if txn:
            return await _update_entry_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_update_entry_postgresql)

    async def check_entry_exists(self, context_id: str) -> tuple[bool, str | None]:
        """Check if a context entry exists and return its source.

        Args:
            context_id: ID of the context entry

        Returns:
            Tuple of (exists, source). Source is None when entry does not exist.
            When exists=True, source is always 'user' or 'agent'.
        """
        if self.backend.backend_type == 'sqlite':

            def _check_exists_sqlite(conn: sqlite3.Connection) -> tuple[bool, str | None]:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT source FROM context_entries WHERE id = {self._placeholder(1)} LIMIT 1',
                    (context_id,),
                )
                row = cursor.fetchone()
                if row is None:
                    return False, None
                return True, cast(str, row['source'])

            return await self.backend.execute_read(_check_exists_sqlite)

        # PostgreSQL
        async def _check_exists_postgresql(conn: 'asyncpg.Connection') -> tuple[bool, str | None]:
            row = await conn.fetchrow(
                f'SELECT source FROM context_entries WHERE id = {self._placeholder(1)} LIMIT 1',
                context_id,
            )
            if row is None:
                return False, None
            return True, cast(str, row['source'])

        return await self.backend.execute_read(_check_exists_postgresql)

    async def get_content_type(self, context_id: str) -> str | None:
        """Get the content type of a context entry.

        Args:
            context_id: ID of the context entry

        Returns:
            Content type ('text' or 'multimodal') or None if entry doesn't exist
        """
        if self.backend.backend_type == 'sqlite':

            def _get_content_type_sqlite(conn: sqlite3.Connection) -> str | None:
                cursor = conn.cursor()
                cursor.execute(
                    f'SELECT content_type FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                row = cursor.fetchone()
                return row['content_type'] if row else None

            return await self.backend.execute_read(_get_content_type_sqlite)

        # PostgreSQL
        async def _get_content_type_postgresql(conn: 'asyncpg.Connection') -> str | None:
            row = await conn.fetchrow(
                f'SELECT content_type FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            return row['content_type'] if row else None

        return await self.backend.execute_read(_get_content_type_postgresql)

    async def update_content_type(
        self,
        context_id: str,
        content_type: str,
        txn: 'TransactionContext | None' = None,
    ) -> bool:
        """Update the content type of a context entry.

        Args:
            context_id: ID of the context entry
            content_type: New content type ('text' or 'multimodal')
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            True if updated successfully, False otherwise
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _update_content_type_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.cursor()
                content_type_placeholder = self._placeholder(1)
                id_placeholder = self._placeholder(2)
                query = (
                    f'UPDATE context_entries SET content_type = {content_type_placeholder}, '
                    f'updated_at = CURRENT_TIMESTAMP WHERE id = {id_placeholder}'
                )
                cursor.execute(query, (content_type, context_id))
                return cursor.rowcount > 0

            if txn:
                return _update_content_type_sqlite(cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_update_content_type_sqlite)

        # PostgreSQL
        async def _update_content_type_postgresql(conn: 'asyncpg.Connection') -> bool:
            content_type_placeholder = self._placeholder(1)
            id_placeholder = self._placeholder(2)
            query = (
                f'UPDATE context_entries SET content_type = {content_type_placeholder}, '
                f'updated_at = CURRENT_TIMESTAMP WHERE id = {id_placeholder}'
            )
            result = await conn.execute(query, content_type, context_id)
            # asyncpg returns "UPDATE N" where N is the count
            return int(result.split()[-1]) > 0 if result else False

        if txn:
            return await _update_content_type_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_update_content_type_postgresql)

    async def patch_metadata(
        self,
        context_id: str,
        patch: dict[str, Any],
        txn: 'TransactionContext | None' = None,
    ) -> tuple[bool, list[str]]:
        """Apply RFC 7396 JSON Merge Patch to metadata atomically.

        This method performs a partial update of the metadata field using database-native
        JSON patching functions for atomic, race-condition-free operations.

        RFC 7396 JSON Merge Patch Semantics:
        - New keys in patch are ADDED to existing metadata
        - Existing keys are REPLACED with new values
        - Keys with null values are DELETED from metadata

        IMPORTANT LIMITATIONS (RFC 7396):
        - Cannot set a value to null: null always means DELETE. If you need to store
          null values, use the full metadata replacement (metadata parameter) instead.
        - Array operations are replace-only: Arrays are replaced entirely, not merged.
          Individual array elements cannot be added, removed, or modified - the entire
          array is replaced with the new value.
        - Empty patch {} is a no-op for data but still updates the updated_at timestamp.

        Backend-specific implementation:
        - SQLite: Uses json_patch() function (available in SQLite 3.38.0+)
        - PostgreSQL: Uses custom jsonb_merge_patch() function for TRUE recursive deep merge.
          The function is created by migration app/migrations/add_jsonb_merge_patch_postgresql.sql
          and provides identical RFC 7396 semantics to SQLite's json_patch().

        Args:
            context_id: ID of the context entry to update
            patch: Dictionary containing the merge patch to apply
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Tuple of (success, list_of_updated_fields).
            Updated fields will include 'metadata' if successful.
        """
        # Convert patch dict to JSON string for database operations
        patch_json = json.dumps(patch, ensure_ascii=False)
        backend_type = txn.backend_type if txn else self.backend.backend_type

        if backend_type == 'sqlite':

            def _patch_metadata_sqlite(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
                cursor = conn.cursor()

                # Verify entry exists before attempting update
                cursor.execute(
                    f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                    (context_id,),
                )
                if not cursor.fetchone():
                    return False, []

                # Apply JSON Merge Patch using SQLite's json_patch() function
                # json_patch() implements RFC 7396 semantics:
                # - COALESCE ensures null metadata is treated as empty object '{}'
                # - json_patch(target, patch) merges patch into target
                # - null values in patch DELETE keys from result
                cursor.execute(
                    f'''
                    UPDATE context_entries
                    SET metadata = json_patch(COALESCE(metadata, '{{}}'), {self._placeholder(1)}),
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = {self._placeholder(2)}
                    ''',
                    (patch_json, context_id),
                )

                if cursor.rowcount > 0:
                    logger.debug(f'Patched metadata for context entry {context_id}')
                    return True, ['metadata']

                return False, []

            if txn:
                return _patch_metadata_sqlite(cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_patch_metadata_sqlite)

        # PostgreSQL implementation - RFC 7396 compliant using jsonb_merge_patch() function
        async def _patch_metadata_postgresql(conn: 'asyncpg.Connection') -> tuple[bool, list[str]]:
            # Import settings here to avoid circular import and ensure schema is retrieved at call time
            from app.settings import get_settings

            # Verify entry exists before attempting update
            row = await conn.fetchrow(
                f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}',
                context_id,
            )
            if not row:
                return False, []

            # RFC 7396 JSON Merge Patch Implementation for PostgreSQL
            #
            # Uses the custom jsonb_merge_patch() function that implements TRUE recursive
            # deep merge semantics as specified in RFC 7396:
            # - New keys in patch are ADDED to existing metadata
            # - Existing keys are REPLACED with new values from patch
            # - Keys with null values are DELETED from metadata
            # - Nested objects are RECURSIVELY merged (not replaced like || operator)
            #
            # The jsonb_merge_patch() function is created by the migration file:
            # app/migrations/add_jsonb_merge_patch_postgresql.sql
            #
            # This approach provides identical behavior to SQLite's json_patch() function,
            # ensuring consistent RFC 7396 semantics across both backends.
            #
            # IMPORTANT: Use schema-qualified function name to ensure the function is found
            # regardless of PostgreSQL search_path configuration (critical for Supabase).
            schema = get_settings().storage.postgresql_schema
            p1 = self._placeholder(1)
            p2 = self._placeholder(2)
            result = await conn.execute(
                f'''
                UPDATE context_entries
                SET metadata = {schema}.jsonb_merge_patch(COALESCE(metadata, '{{}}'::jsonb), {p1}::jsonb),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = {p2}
                ''',
                patch_json,
                context_id,
            )

            # asyncpg returns "UPDATE N" where N is the count
            rows_affected = int(result.split()[-1]) if result else 0
            if rows_affected > 0:
                logger.debug(f'Patched metadata for context entry {context_id}')
                return True, ['metadata']

            return False, []

        if txn:
            return await _patch_metadata_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_patch_metadata_postgresql)

    @staticmethod
    def row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        """Convert a database row to a dictionary.

        Args:
            row: SQLite Row object

        Returns:
            Dictionary representation of the row
        """
        entry = dict(row)

        # Parse JSON metadata if present
        metadata_raw = entry.get('metadata')
        if metadata_raw is not None and hasattr(metadata_raw, 'strip'):
            try:
                entry['metadata'] = json.loads(str(metadata_raw))
            except (json.JSONDecodeError, ValueError, AttributeError):
                entry['metadata'] = None

        return entry

    async def get_ids_matching_batch_criteria(
        self,
        thread_ids: list[str] | None = None,
        source: str | None = None,
        older_than_days: int | None = None,
    ) -> list[str]:
        """Return context entry IDs matching batch deletion criteria.

        Builds the same WHERE clause as delete_contexts_batch but executes
        a SELECT instead of DELETE. Used to pre-query affected IDs for
        embedding cleanup on SQLite (where vec0 virtual tables lack CASCADE).

        Args:
            thread_ids: Filter by these thread IDs
            source: Filter by source ('user' or 'agent')
            older_than_days: Filter entries older than N days

        Returns:
            List of matching context entry IDs.
        """
        if self.backend.backend_type == 'sqlite':

            def _select_ids_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                conditions: list[str] = []
                params: list[Any] = []

                if thread_ids:
                    placeholders = ','.join([
                        self._placeholder(len(params) + i + 1) for i in range(len(thread_ids))
                    ])
                    conditions.append(f'thread_id IN ({placeholders})')
                    params.extend(thread_ids)

                if source:
                    conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                    params.append(source)

                if older_than_days is not None:
                    conditions.append(
                        f"created_at < datetime('now', {self._placeholder(len(params) + 1)})",
                    )
                    params.append(f'-{older_than_days} days')

                if not conditions:
                    return []

                where_clause = ' AND '.join(conditions)
                query = f'SELECT id FROM context_entries WHERE {where_clause}'
                cursor.execute(query, tuple(params))
                return [row[0] for row in cursor.fetchall()]

            return await self.backend.execute_read(_select_ids_sqlite)

        # PostgreSQL: CASCADE handles embedding cleanup, so this method
        # returns an empty list (caller should not need it).
        return []

    async def delete_contexts_batch(
        self,
        context_ids: list[str] | None = None,
        thread_ids: list[str] | None = None,
        source: str | None = None,
        older_than_days: int | None = None,
    ) -> tuple[int, list[str]]:
        """Delete multiple context entries by various criteria.

        At least one criterion must be provided. Criteria can be combined
        for more targeted deletion. Cascading delete removes associated
        tags and images. On PostgreSQL, embedding rows are removed via
        ON DELETE CASCADE on the surviving embedding table for the active
        compression mode (fp32 ``vec_context_embeddings`` when compression
        is disabled; compressed ``vec_context_embeddings_compressed`` when
        enabled). On SQLite the vec0 virtual table is NOT covered by
        CASCADE and requires explicit cleanup via the embedding repository;
        the compressed ``vec_context_embeddings_compressed`` table IS covered
        by CASCADE on SQLite (it is a standard table, not a virtual one).

        Args:
            context_ids: Specific context entry IDs to delete
            thread_ids: Delete all entries in these threads
            source: Filter by source ('user' or 'agent') - combine with other criteria
            older_than_days: Delete entries older than N days

        Returns:
            Tuple of (deleted_count, list_of_criteria_used)
        """
        if self.backend.backend_type == 'sqlite':

            def _delete_batch_sqlite(conn: sqlite3.Connection) -> tuple[int, list[str]]:
                cursor = conn.cursor()
                conditions: list[str] = []
                params: list[Any] = []
                # Build per-invocation so transparent write-retries (e.g. on a
                # transient "database is locked" error) do not accumulate
                # duplicate criteria strings across attempts.
                criteria_used: list[str] = []

                if context_ids:
                    placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(context_ids))])
                    conditions.append(f'id IN ({placeholders})')
                    params.extend(context_ids)
                    criteria_used.append(f'context_ids: {len(context_ids)} IDs')

                if thread_ids:
                    placeholders = ','.join([
                        self._placeholder(len(params) + i + 1) for i in range(len(thread_ids))
                    ])
                    conditions.append(f'thread_id IN ({placeholders})')
                    params.extend(thread_ids)
                    criteria_used.append(f'thread_ids: {len(thread_ids)} threads')

                if source:
                    conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                    params.append(source)
                    criteria_used.append(f'source: {source}')

                if older_than_days is not None:
                    conditions.append(
                        f"created_at < datetime('now', {self._placeholder(len(params) + 1)})",
                    )
                    params.append(f'-{older_than_days} days')
                    criteria_used.append(f'older_than_days: {older_than_days}')

                if not conditions:
                    return 0, criteria_used

                where_clause = ' AND '.join(conditions)
                query = f'DELETE FROM context_entries WHERE {where_clause}'
                cursor.execute(query, tuple(params))

                deleted_count = cursor.rowcount
                logger.info(f'Batch delete: removed {deleted_count} entries using criteria: {criteria_used}')
                return deleted_count, criteria_used

            return await self.backend.execute_write(_delete_batch_sqlite)

        # PostgreSQL
        async def _delete_batch_postgresql(conn: 'asyncpg.Connection') -> tuple[int, list[str]]:
            conditions: list[str] = []
            params: list[Any] = []
            # Build per-invocation (see the SQLite closure) so retried writes do
            # not accumulate duplicate criteria strings across attempts.
            criteria_used: list[str] = []

            if context_ids:
                placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(context_ids))])
                conditions.append(f'id IN ({placeholders})')
                params.extend(context_ids)
                criteria_used.append(f'context_ids: {len(context_ids)} IDs')

            if thread_ids:
                placeholders = ','.join([self._placeholder(len(params) + i + 1) for i in range(len(thread_ids))])
                conditions.append(f'thread_id IN ({placeholders})')
                params.extend(thread_ids)
                criteria_used.append(f'thread_ids: {len(thread_ids)} threads')

            if source:
                conditions.append(f'source = {self._placeholder(len(params) + 1)}')
                params.append(source)
                criteria_used.append(f'source: {source}')

            if older_than_days is not None:
                conditions.append(
                    f"created_at < (NOW() - INTERVAL '{older_than_days} days')",
                )
                criteria_used.append(f'older_than_days: {older_than_days}')

            if not conditions:
                return 0, criteria_used

            where_clause = ' AND '.join(conditions)
            query = f'DELETE FROM context_entries WHERE {where_clause}'
            result = await conn.execute(query, *params)

            # asyncpg returns "DELETE N" where N is the count
            deleted_count = int(result.split()[-1]) if result else 0
            logger.info(f'Batch delete: removed {deleted_count} entries using criteria: {criteria_used}')
            return deleted_count, criteria_used

        return await self.backend.execute_write(_delete_batch_postgresql, validate_connection=True)

    async def find_ids_by_prefix(self, prefix: str, limit: int = 2) -> list[str]:
        """Find context entry IDs that begin with the given prefix.

        Used by the optional CLI prefix-resolution flow (`mcp-context-id-resolve`)
        to expand a short user-supplied prefix into a full id when ambiguity is
        unlikely. The caller decides what to do when ``limit`` rows are returned;
        this method's contract is "return up to N matches".

        Args:
            prefix: Lowercase hex prefix (3-32 characters). Caller is responsible
                for normalization.
            limit: Maximum number of IDs to return. Defaults to 2 so callers can
                detect ambiguity by checking ``len(result) > 1``.

        Returns:
            List of matching context_id strings (UUIDv7 hex, 32 chars), up to ``limit``.
        """
        if self.backend.backend_type == 'sqlite':

            def _find_sqlite(conn: sqlite3.Connection) -> list[str]:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT id FROM context_entries
                    WHERE id LIKE {self._placeholder(1)}
                    ORDER BY id
                    LIMIT {self._placeholder(2)}
                    ''',
                    (prefix + '%', limit),
                )
                return [row['id'] for row in cursor.fetchall()]

            return await self.backend.execute_read(_find_sqlite)

        # PostgreSQL: id is a uuid column; canonical text form contains hyphens
        # which are absent from the user-supplied hex prefix. REPLACE() removes
        # hyphens so LIKE matches against the 32-char hex representation.
        async def _find_postgresql(conn: 'asyncpg.Connection') -> list[str]:
            rows = await conn.fetch(
                f'''
                SELECT id FROM context_entries
                WHERE REPLACE(CAST(id AS TEXT), '-', '') LIKE {self._placeholder(1)}
                ORDER BY id
                LIMIT {self._placeholder(2)}
                ''',
                prefix + '%',
                limit,
            )
            # Canonicalize: asyncpg returns UUID columns as pgproto.UUID whose str()
            # is the 36-char hyphenated form. normalize_id yields the 32-char hex
            # the SQLite path already returns, so prefix resolution echoes a
            # canonical context_id on both backends (mirrors grep_scan_text_contents).
            return [normalize_id(str(row['id'])) for row in rows]

        return await self.backend.execute_read(_find_postgresql)
