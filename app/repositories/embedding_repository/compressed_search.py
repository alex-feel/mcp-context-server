"""KNN search over TurboQuant-compressed embedding payloads on both backends."""

import asyncio
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
from app.repositories.embedding_repository.compression_cache import get_cached_compression_metadata
from app.repositories.embedding_repository.records import SQLITE_IN_CLAUSE_BATCH
from app.repositories.embedding_repository.records import MetadataFilterValidationError
from app.repositories.entry_filters import count_applied_filters

if TYPE_CHECKING:
    import asyncpg


logger = logging.getLogger(__name__)


# Decoding the stored TurboQuant payloads (payload_from_bytes per chunk row, each
# doing struct.unpack + np.frombuffer().copy()) and concatenating them is O(N-chunks)
# pure-Python CPU on the compressed read path (search_compressed). For a large
# candidate set this would pin the asyncio event loop and starve other concurrent
# MCP requests, so it is offloaded to a worker thread above this row count -- the
# same discipline the read-path (navigation/grep) and write-leg (chunking/index_tree)
# offloads apply, here measured in chunk rows rather than characters. Small searches
# stay inline to avoid a per-call thread hop. The vectorized provider GEMM
# (estimate_inner_product / decode) is already offloaded inside the provider.
_COMPRESSED_OFFLOAD_MIN_ROWS = 2048


class CompressedSearchMixin(BaseRepository):
    """KNN search over compressed embeddings.

    ``search_compressed`` applies the same entry filters and read predicate as the
    fp32 search, scores the candidate payloads in ``vec_context_embeddings_compressed``
    with the active provider's inner-product estimator or decoded L2 distance,
    re-applies the read predicate when it hydrates the ranked page, and returns
    results in the same shape as the fp32 search.
    """

    async def search_compressed(
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
        """KNN search over the compressed embeddings (TurboQuant payloads) of the entries the scope may read.

        Algorithm:
            1. Resolve the singleton provenance row + cached provider.
            2. Filter ``context_entries`` exactly as :meth:`search_fp32` does,
               the scope's read predicate included, to narrow the candidate set;
               an entry the scope may not read is never scored.
            3. Read the matching rows from
               ``vec_context_embeddings_compressed``.
            4. For ``variant='ip'`` use the provider's unbiased
               inner-product estimator; convert to a distance via
               ``distance = -ip`` so smaller-is-better matches the existing
               result ordering.
            5. For ``variant='mse'`` decode each payload and compute the L2
               distance to the query.
            6. Aggregate per ``context_id`` (MIN distance per context, ties
               broken on the unique chunk ``start_index``, matching the
               best-chunk-per-context semantics of the fp32 read path).
            7. Sort ASC by ``(distance, context_id)`` -- the id is the unique
               secondary key that keeps a tied ordering reproducible across
               executions -- slice ``[offset : offset + limit]``, hydrate from
               ``context_entries`` under the read predicate again, so an entry
               that stopped being readable after its candidate was selected is
               dropped rather than returned.

        Return shape is IDENTICAL to :meth:`search_fp32` so the calling tool
        layer needs no compression-specific branching.

        Args:
            query_embedding: Query vector for similarity search.
            limit: Maximum number of results to return.
            offset: Number of results to skip (pagination).
            thread_id: Optional filter by thread.
            source: Optional filter by source type.
            content_type: Filter by content type (text or multimodal).
            tags: Filter by any of these tags (OR logic).
            start_date: Filter by created_at >= date (ISO 8601 format).
            end_date: Filter by created_at <= date (ISO 8601 format).
            metadata: Simple metadata filters (key=value equality).
            metadata_filters: Advanced metadata filters with operators.
            explain_query: If True, include query execution plan in stats.
            scope: The caller's scope; only entries it may read are ranked and hydrated.

        Returns:
            Tuple of (search results list, statistics dictionary). Each
            result carries ``distance``, ``matched_chunk_start``,
            ``matched_chunk_end`` plus the standard ``context_entries``
            columns.

        Raises:
            RuntimeError: If the singleton compression provenance row is
                missing or the query embedding length does not match the
                stored compression dimension.
            MetadataFilterValidationError: If a metadata filter fails validation
                or a non-empty tags filter normalizes to empty (all-blank).
        """
        import time as time_module

        import numpy as np

        from app.compression import get_cached_compression_provider

        provider = await get_cached_compression_provider()
        comp_meta = await get_cached_compression_metadata(self.backend)
        variant = comp_meta.variant

        if len(query_embedding) != comp_meta.dim:
            raise RuntimeError(
                f'query_embedding length {len(query_embedding)} does not '
                f'match compression metadata dim {comp_meta.dim}',
            )

        start_time = time_module.time()
        query_matrix = np.asarray([query_embedding], dtype=np.float32)

        normalized_tags: list[str] = []
        if tags:
            # A non-empty tags list that normalizes to empty (all-blank) raises
            # through the structured validation-error channel rather than
            # silently widening the result set.
            try:
                normalized_tags = self.normalize_tag_filter(tags)
            except ValueError as e:
                raise MetadataFilterValidationError(
                    'Metadata filter validation failed', [str(e)],
                ) from e

        # Compute the non-metadata filter count through the shared tally so this path
        # reports the same number as the fp32 search and every other search tool; the
        # metadata count is added below once the candidate query has built its filters.
        filter_count = count_applied_filters(
            thread_id=thread_id,
            source=source,
            content_type=content_type,
            tags=tags,
            start_date=start_date,
            end_date=end_date,
        )

        # Build the candidate-id query against context_entries reusing the
        # same building blocks as :meth:`search_fp32` (tags, dates, metadata).
        if self.backend.backend_type == 'sqlite':

            def _candidates_sqlite(
                conn: sqlite3.Connection,
            ) -> tuple[list[str], int, str | None]:
                conditions: list[str] = []
                params: list[Any] = []
                metadata_filter_count = 0

                if thread_id:
                    conditions.append('thread_id = ?')
                    params.append(thread_id)
                if source:
                    conditions.append('source = ?')
                    params.append(source)
                if content_type:
                    conditions.append('content_type = ?')
                    params.append(content_type)
                if normalized_tags:
                    placeholders = ','.join(['?' for _ in normalized_tags])
                    conditions.append(
                        'id IN (SELECT DISTINCT context_entry_id '
                        f'FROM tags WHERE tag IN ({placeholders}))',
                    )
                    params.extend(normalized_tags)
                if start_date:
                    conditions.append('created_at >= datetime(?)')
                    params.append(start_date)
                if end_date:
                    conditions.append('created_at <= datetime(?)')
                    params.append(end_date)

                if metadata or metadata_filters:
                    from pydantic import ValidationError

                    from app.metadata_types import MetadataFilter
                    from app.query_builder import MetadataQueryBuilder

                    builder = MetadataQueryBuilder(backend_type='sqlite')
                    errs: list[str] = []
                    # An invalid simple-metadata KEY is reported as a structured validation
                    # error (NOT silently dropped, which would widen the result set),
                    # consistent with search_context and the advanced filters below.
                    if metadata:
                        for key, value in metadata.items():
                            try:
                                builder.add_simple_filter(key, value)
                                metadata_filter_count += 1
                            except ValueError as e:
                                errs.append(f'Invalid metadata key {key!r}: {e}')
                    if metadata_filters:
                        for filter_dict in metadata_filters:
                            try:
                                spec = MetadataFilter(**filter_dict)
                                builder.add_advanced_filter(spec)
                                metadata_filter_count += 1
                            except (ValidationError, ValueError) as e:
                                errs.append(
                                    f'Invalid metadata filter {filter_dict}: {e}',
                                )
                            except Exception as e:
                                errs.append(
                                    f'Unexpected error in metadata filter '
                                    f'{filter_dict}: {e}',
                                )
                                logger.error(
                                    'Unexpected error processing metadata filter: %s',
                                    e,
                                )
                    if errs:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed', errs,
                        )

                    clause, mparams = builder.build_where_clause()
                    if clause:
                        conditions.append(clause)
                        params.extend(mparams)

                # The read predicate follows every client filter and is the only filter
                # applied before ranking, so the payloads of entries the scope may not
                # read are never scored.
                read = build_access_predicate(
                    scope, mode=AccessMode.READ, backend_type='sqlite', outer='context_entries',
                )
                if read.sql:
                    conditions.append(read.sql)
                    params.extend(read.params)

                where_clause = (
                    f'WHERE {" AND ".join(conditions)}' if conditions else ''
                )
                cand_sql = f'SELECT id FROM context_entries {where_clause}'
                cursor = conn.execute(cand_sql, params)
                cand_ids = [str(row[0]) for row in cursor.fetchall()]

                plan: str | None = None
                if explain_query:
                    cursor = conn.execute(
                        f'EXPLAIN QUERY PLAN {cand_sql}', params,
                    )
                    plan_rows = cursor.fetchall()
                    formatted: list[str] = []
                    for row in plan_rows:
                        row_dict = dict(row)
                        formatted.append(
                            f"id:{row_dict.get('id', '?')} "
                            f"parent:{row_dict.get('parent', '?')} "
                            f"notused:{row_dict.get('notused', '?')} "
                            f"detail:{row_dict.get('detail', '?')}",
                        )
                    plan = '\n'.join(formatted)
                return cand_ids, metadata_filter_count, plan

            candidate_ids, meta_count, query_plan = await self.backend.execute_read(
                _candidates_sqlite,
            )
        else:
            async def _candidates_pg(
                conn: 'asyncpg.Connection',
            ) -> tuple[list[str], int, str | None]:
                conditions: list[str] = ['1=1']
                params: list[Any] = []
                position = 1
                metadata_filter_count = 0

                if thread_id:
                    conditions.append(f'ce.thread_id = ${position}')
                    params.append(thread_id)
                    position += 1
                if source:
                    conditions.append(f'ce.source = ${position}')
                    params.append(source)
                    position += 1
                if content_type:
                    conditions.append(f'ce.content_type = ${position}')
                    params.append(content_type)
                    position += 1
                if normalized_tags:
                    placeholders = ','.join(
                        f'${position + i}' for i in range(len(normalized_tags))
                    )
                    conditions.append(
                        'ce.id IN (SELECT DISTINCT context_entry_id '
                        f'FROM tags WHERE tag IN ({placeholders}))',
                    )
                    params.extend(normalized_tags)
                    position += len(normalized_tags)
                if start_date:
                    conditions.append(f'ce.created_at >= ${position}')
                    params.append(self._parse_date_for_postgresql(start_date))
                    position += 1
                if end_date:
                    conditions.append(f'ce.created_at <= ${position}')
                    params.append(self._parse_date_for_postgresql(end_date))
                    position += 1

                if metadata or metadata_filters:
                    from pydantic import ValidationError

                    from app.metadata_types import MetadataFilter
                    from app.query_builder import MetadataQueryBuilder

                    builder = MetadataQueryBuilder(
                        backend_type='postgresql',
                        param_offset=len(params),
                        table_alias='ce',
                    )
                    errs: list[str] = []
                    # An invalid simple-metadata KEY is reported as a structured validation
                    # error (NOT silently dropped, which would widen the result set),
                    # consistent with search_context and the advanced filters below.
                    if metadata:
                        for key, value in metadata.items():
                            try:
                                builder.add_simple_filter(key, value)
                                metadata_filter_count += 1
                            except ValueError as e:
                                errs.append(f'Invalid metadata key {key!r}: {e}')
                    if metadata_filters:
                        for filter_dict in metadata_filters:
                            try:
                                spec = MetadataFilter(**filter_dict)
                                builder.add_advanced_filter(spec)
                                metadata_filter_count += 1
                            except (ValidationError, ValueError) as e:
                                errs.append(
                                    f'Invalid metadata filter {filter_dict}: {e}',
                                )
                            except Exception as e:
                                errs.append(
                                    f'Unexpected error in metadata filter '
                                    f'{filter_dict}: {e}',
                                )
                                logger.error(
                                    'Unexpected error processing metadata filter: %s',
                                    e,
                                )
                    if errs:
                        raise MetadataFilterValidationError(
                            'Metadata filter validation failed', errs,
                        )
                    clause, mparams = builder.build_where_clause()
                    if clause:
                        # The builder emits the clause already qualified with the 'ce.'
                        # alias (table_alias='ce'), matching the JOIN target -- no
                        # str.replace that would corrupt 'metadata'-containing keys.
                        conditions.append(clause)
                        params.extend(mparams)
                        position += len(mparams)

                # The read predicate follows every client filter, numbered after them.
                read = build_access_predicate(
                    scope, mode=AccessMode.READ, backend_type='postgresql', outer='ce', start=position,
                )
                if read.sql:
                    conditions.append(read.sql)
                    params.extend(read.params)
                    position += read.bind_count

                where_clause = ' AND '.join(conditions)
                cand_sql = (
                    f'SELECT ce.id FROM context_entries ce WHERE {where_clause}'
                )
                rows = await conn.fetch(cand_sql, *params)

                def _build_ids() -> list[str]:
                    return [str(row['id']) for row in rows]

                # O(N-contexts) id construction on the event loop for PostgreSQL;
                # offload a large candidate set, mirroring _read_compressed_pg.
                cand_ids = (
                    await asyncio.to_thread(_build_ids)
                    if len(rows) > _COMPRESSED_OFFLOAD_MIN_ROWS
                    else _build_ids()
                )

                plan: str | None = None
                if explain_query:
                    plan_rows = await conn.fetch(
                        f'EXPLAIN {cand_sql}', *params,
                    )
                    plan = '\n'.join(str(row[0]) for row in plan_rows)
                return cand_ids, metadata_filter_count, plan

            candidate_ids, meta_count, query_plan = await self.backend.execute_read(
                cast(Any, _candidates_pg),
            )

        filter_count += meta_count

        if not candidate_ids:
            stats: dict[str, Any] = {
                'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
                'filters_applied': filter_count,
                'rows_returned': 0,
                'backend': self.backend.backend_type,
            }
            if explain_query and query_plan is not None:
                stats['query_plan'] = query_plan
            return [], stats

        # Read compressed rows for the candidate context_ids.
        if self.backend.backend_type == 'sqlite':

            def _read_compressed_sqlite(
                conn: sqlite3.Connection,
            ) -> list[tuple[str, int, int, int, bytes]]:
                rows: list[tuple[str, int, int, int, bytes]] = []
                for start in range(0, len(candidate_ids), SQLITE_IN_CLAUSE_BATCH):
                    batch = candidate_ids[start:start + SQLITE_IN_CLAUSE_BATCH]
                    placeholders = ','.join('?' for _ in batch)
                    cursor = conn.execute(
                        'SELECT context_id, chunk_index, start_index, end_index, '
                        f'payload FROM vec_context_embeddings_compressed '
                        f'WHERE context_id IN ({placeholders})',
                        batch,
                    )
                    rows.extend(
                        (str(r[0]), int(r[1]), int(r[2]), int(r[3]), bytes(r[4]))
                        for r in cursor.fetchall()
                    )
                return rows

            payload_rows = await self.backend.execute_read(_read_compressed_sqlite)
        else:
            async def _read_compressed_pg(
                conn: 'asyncpg.Connection',
            ) -> list[tuple[str, int, int, int, bytes]]:
                rows = await conn.fetch(
                    'SELECT context_id, chunk_index, start_index, end_index, '
                    'payload FROM vec_context_embeddings_compressed '
                    'WHERE context_id = ANY($1::uuid[])',
                    candidate_ids,
                )

                def _build_rows() -> list[tuple[str, int, int, int, bytes]]:
                    return [
                        (
                            str(r['context_id']),
                            int(r['chunk_index']),
                            int(r['start_index']),
                            int(r['end_index']),
                            bytes(r['payload']),
                        )
                        for r in rows
                    ]

                # Building payload_rows copies each BYTEA payload (bytes(...)) and is
                # O(N-chunks) pure-Python over the unbounded candidate set; on
                # PostgreSQL this async read callable runs on the event loop (the
                # SQLite callable already runs inside execute_read's worker thread), so
                # a large candidate set is offloaded so it cannot pin the loop (see
                # _COMPRESSED_OFFLOAD_MIN_ROWS). The fetched asyncpg Records are fully
                # materialized and detached, so accessing them off-loop is safe.
                if len(rows) > _COMPRESSED_OFFLOAD_MIN_ROWS:
                    return await asyncio.to_thread(_build_rows)
                return _build_rows()

            payload_rows = await self.backend.execute_read(cast(Any, _read_compressed_pg))

        if not payload_rows:
            stats = {
                'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
                'filters_applied': filter_count,
                'rows_returned': 0,
                'backend': self.backend.backend_type,
            }
            if explain_query and query_plan is not None:
                stats['query_plan'] = query_plan
            return [], stats

        from app.compression.providers.turboquant._types import IPPayload
        from app.compression.providers.turboquant._types import MSEPayload
        from app.compression.providers.turboquant._types import payload_from_bytes

        def _decode_and_concat() -> bytes:
            # Decode every stored payload via the wire-format dispatcher and
            # concatenate same-variant payloads into ONE synthetic payload so the
            # provider's scoring call runs exactly ONCE per query (O(1) GEMM)
            # instead of once per candidate row. This decode+concat is O(N-chunks)
            # pure-Python CPU (payload_from_bytes does struct.unpack + frombuffer
            # copies per row); the caller offloads it for a large candidate set
            # (see _COMPRESSED_OFFLOAD_MIN_ROWS) so it cannot pin the event loop.
            decoded_payloads: list[MSEPayload | IPPayload] = [
                payload_from_bytes(payload_bytes)
                for _, _, _, _, payload_bytes in payload_rows
            ]
            if variant == 'ip':
                ip_subtypes: list[IPPayload] = []
                for p in decoded_payloads:
                    if not isinstance(p, IPPayload):
                        raise RuntimeError(
                            f'Compression metadata variant=ip but stored payload is '
                            f'{type(p).__name__}; storage corruption suspected',
                        )
                    ip_subtypes.append(p)
                return IPPayload.concat(ip_subtypes).to_bytes()
            mse_subtypes: list[MSEPayload] = []
            for p in decoded_payloads:
                if not isinstance(p, MSEPayload):
                    raise RuntimeError(
                        f'Compression metadata variant=mse but stored payload is '
                        f'{type(p).__name__}; storage corruption suspected',
                    )
                mse_subtypes.append(p)
            return MSEPayload.concat(mse_subtypes).to_bytes()

        if len(payload_rows) > _COMPRESSED_OFFLOAD_MIN_ROWS:
            combined_bytes = await asyncio.to_thread(_decode_and_concat)
        else:
            combined_bytes = _decode_and_concat()

        # Distance polarity convention (matches the fp32 read path):
        #   smaller distance = closer / more similar. The provider's async GEMM
        # wrapper offloads the matmul via asyncio.to_thread; the per-chunk distances
        # + per-context MIN aggregation + sort below are themselves O(N-chunks)
        # pure-Python/numpy work over the same unbounded candidate set, so they are
        # offloaded too (see _COMPRESSED_OFFLOAD_MIN_ROWS). Together with the decode
        # above, this leaves NO per-chunk work on the event loop for a large search;
        # only the bounded offset/limit page slice + hydration run inline.
        if variant == 'ip':
            # estimate_inner_product returns shape (nq, n_total); nq=1 slices to a
            # 1-D array of length n_total in payload_rows order (concat preserves it).
            gemm_output = await provider.estimate_inner_product(combined_bytes, query_matrix)
        else:  # variant == 'mse'
            # decode returns (n_total, d) for an L2 distance to the single query vector.
            gemm_output = await provider.decode(combined_bytes)

        def _rank_from_gemm() -> list[tuple[str, tuple[float, int, int]]]:
            if variant == 'ip':
                # IP negated so a larger IP (more similar) maps to a smaller distance.
                distances = [-float(x) for x in gemm_output[0].tolist()]
            else:
                diff = gemm_output - query_matrix[0]
                distances = [float(x) for x in np.linalg.norm(diff, axis=1).tolist()]
            best_by_context: dict[str, tuple[float, int, int]] = {}
            for (context_id, _chunk_index, start_index, end_index, _payload), dist in zip(
                payload_rows, distances, strict=True,
            ):
                current = best_by_context.get(context_id)
                # Tie between two equidistant chunks of the SAME context resolves on the
                # unique start_index, mirroring the fp32 paths' ROW_NUMBER/DISTINCT ON
                # tiebreak. A bare `dist < current[0]` would instead keep whichever chunk
                # the ORDER BY-less candidate read returned first, so the reported
                # matched_chunk_start/end could differ between two identical queries.
                if current is None or (dist, start_index) < (current[0], current[1]):
                    best_by_context[context_id] = (dist, start_index, end_index)
            # Sort on (distance, context_id): context_id is the explicit UNIQUE secondary
            # key, so equal-distance contexts get a reproducible order instead of
            # inheriting dict insertion order from an ORDER BY-less candidate fetch. The
            # page slice below is a LIMIT/OFFSET window over this list, so without the
            # tiebreak a client paging tied results silently skips or duplicates rows.
            # Ascending id matches the fp32 paths' ORDER BY best_distance ASC, context_id ASC.
            return sorted(best_by_context.items(), key=lambda kv: (kv[1][0], kv[0]))

        if len(payload_rows) > _COMPRESSED_OFFLOAD_MIN_ROWS:
            ranked = await asyncio.to_thread(_rank_from_gemm)
        else:
            ranked = _rank_from_gemm()

        page = ranked[offset : offset + limit]
        if not page:
            stats = {
                'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
                'filters_applied': filter_count,
                'rows_returned': 0,
                'backend': self.backend.backend_type,
            }
            if explain_query and query_plan is not None:
                stats['query_plan'] = query_plan
            return [], stats

        page_ids = [cid for cid, _ in page]

        # Hydrate the result rows from context_entries under the read predicate again:
        # an entry whose access changed after its candidate was selected is not
        # returned, and the skip-on-missing loop below drops it from the page.
        if self.backend.backend_type == 'sqlite':
            hydrate_read = build_access_predicate(
                scope, mode=AccessMode.READ, backend_type='sqlite', outer='context_entries',
            )

            def _hydrate_sqlite(
                conn: sqlite3.Connection,
            ) -> dict[str, dict[str, Any]]:
                # page_ids is bounded by the overfetch limit, not by the final
                # page size, so it can exceed SQLITE_MAX_VARIABLE_NUMBER; bind it
                # in bounded batches like the compressed candidate read (a batch
                # plus the predicate's binds stays below the variable limit).
                out: dict[str, dict[str, Any]] = {}
                for start in range(0, len(page_ids), SQLITE_IN_CLAUSE_BATCH):
                    batch = page_ids[start:start + SQLITE_IN_CLAUSE_BATCH]
                    placeholders = ','.join('?' for _ in batch)
                    cursor = conn.execute(
                        'SELECT id, thread_id, source, content_type, text_content, '
                        'metadata, summary, created_at, updated_at FROM context_entries '
                        f'WHERE id IN ({placeholders}){hydrate_read.and_clause()}',
                        [*batch, *hydrate_read.params],
                    )
                    out.update({str(dict(r)['id']): dict(r) for r in cursor.fetchall()})
                return out

            hydrated = await self.backend.execute_read(_hydrate_sqlite)
        else:
            hydrate_read = build_access_predicate(
                scope, mode=AccessMode.READ, backend_type='postgresql', outer='context_entries', start=2,
            )

            async def _hydrate_pg(
                conn: 'asyncpg.Connection',
            ) -> dict[str, dict[str, Any]]:
                rows = await conn.fetch(
                    'SELECT id, thread_id, source, content_type, text_content, '
                    'metadata, summary, created_at, updated_at FROM context_entries '
                    f'WHERE id = ANY($1::uuid[]){hydrate_read.and_clause()}',
                    page_ids,
                    *hydrate_read.params,
                )
                return {str(dict(r)['id']): dict(r) for r in rows}

            hydrated = await self.backend.execute_read(cast(Any, _hydrate_pg))

        results: list[dict[str, Any]] = []
        for context_id, (dist, start_index, end_index) in page:
            row = hydrated.get(context_id)
            if row is None:
                # The entry was deleted or stopped being readable between the
                # candidate scan and the hydration query. Skip rather than
                # fabricate a row.
                continue
            row['distance'] = dist
            row['matched_chunk_start'] = start_index
            row['matched_chunk_end'] = end_index
            results.append(row)

        stats = {
            'execution_time_ms': round((time_module.time() - start_time) * 1000, 2),
            'filters_applied': filter_count,
            'rows_returned': len(results),
            'backend': self.backend.backend_type,
        }
        if explain_query and query_plan is not None:
            stats['query_plan'] = query_plan
        return results, stats
