"""Updates of existing context entries.

The version-guarded entry update, the single-column ``updated_at`` and
``content_type`` writes, and the atomic metadata merge patch. Every statement
carries the caller's access predicate, so a write reaches only an entry the
caller may modify, and a visibility change only an entry the caller owns.
"""

import json
import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import compute_content_hash
from app.repositories.context_repository.records import VersionConflictError

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


logger = logging.getLogger(__name__)


class ContextUpdateMixin(BaseRepository):
    """Updates of existing ``context_entries`` rows.

    ``update_context_entry`` rewrites text, metadata, summary and visibility,
    compare-and-set guarded on ``version`` when the caller passes the version it
    read; ``touch_updated_at`` and ``update_content_type`` write one column each;
    ``patch_metadata`` applies an RFC 7396 merge patch atomically in the database.
    Each statement admits only an entry the caller's scope may modify (owner or
    write grant); a visibility change admits only an entry the caller owns. An
    entry the scope may not modify is reported exactly like a missing one.
    """

    async def update_context_entry(
        self,
        context_id: str,
        text_content: str | None = None,
        metadata: str | None = None,
        summary: str | None = None,
        clear_summary: bool = False,
        visibility: str | None = None,
        expected_version: int | None = None,
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> tuple[bool, list[str]]:
        """Update text content, metadata, and/or visibility of a context entry.

        ``owner_id`` is deliberately not updatable: ownership is stamped at
        INSERT and immutable (there is no ownership-transfer operation).

        The existence probe and the UPDATE both carry the caller's access
        predicate: OWNER when ``visibility`` is set, because a visibility change
        is owner-only, and WRITE otherwise. An entry the scope may not modify
        therefore returns ``(False, [])`` exactly like a missing one, and never
        surfaces as a version conflict, which would reveal that it exists.

        Args:
            context_id: ID of the context entry to update
            text_content: New text content (if provided)
            metadata: New metadata JSON string (if provided)
            summary: New LLM-generated summary text (if provided)
            clear_summary: If True, explicitly set summary to NULL in the database.
                Takes precedence over summary parameter.
            visibility: New visibility value ('private' or 'public') if
                provided. The caller validates the value before it reaches this
                method; the statement admits only an entry the scope owns.
            expected_version: Optimistic-concurrency token captured before
                generation; None skips the compare-and-set.
            scope: The caller's scope.
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            Tuple of (success, list_of_updated_fields). When the entry exists and
            the scope may modify it but its version no longer matches
            ``expected_version``, the statement raises ``VersionConflictError``
            instead of returning.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        mode = AccessMode.OWNER if visibility is not None else AccessMode.WRITE

        # Build the SET list from the provided fields.
        update_parts: list[str] = []
        params: list[Any] = []
        updated_fields: list[str] = []

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

        if visibility is not None:
            update_parts.append(f'visibility = {self._placeholder(len(params) + 1)}')
            params.append(visibility)
            updated_fields.append('visibility')

        if not update_parts:
            return False, []

        # Always update the updated_at timestamp
        update_parts.append('updated_at = CURRENT_TIMESTAMP')

        # Optimistic concurrency: bump the version and gate the write on the
        # version captured before generation. A concurrent writer that committed in
        # the meantime advanced the row's version, so the CAS matches 0 rows and
        # VersionConflictError is raised -- the caller re-reads and retries instead
        # of silently overwriting newer text.
        where_clause = f'id = {self._placeholder(len(params) + 1)}'
        params.append(context_id)
        if expected_version is not None:
            update_parts.append('version = version + 1')
            where_clause += f' AND version = {self._placeholder(len(params) + 1)}'
            params.append(expected_version)

        # The access predicate binds after the SET values, the id and the version, so
        # the UPDATE itself re-asserts the caller's access.
        update_access = build_access_predicate(
            scope, mode=mode, backend_type=backend_type, outer='context_entries', start=len(params) + 1,
        )
        params.extend(update_access.params)
        update_query = (
            f'UPDATE context_entries SET {", ".join(update_parts)} WHERE {where_clause}{update_access.and_clause()}'
        )

        # The existence probe: the id binds first, the access predicate next. It tells
        # a missing or inaccessible entry from a version mismatch, which a zero row
        # count of the guarded UPDATE cannot.
        probe_access = build_access_predicate(
            scope, mode=mode, backend_type=backend_type, outer='context_entries', start=2,
        )
        probe_query = f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}{probe_access.and_clause()}'
        probe_params: list[object] = [context_id, *probe_access.params]

        if backend_type == 'sqlite':

            def _update_entry_sqlite(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
                cursor = conn.cursor()
                cursor.execute(probe_query, tuple(probe_params))
                if not cursor.fetchone():
                    return False, []

                cursor.execute(update_query, tuple(params))
                if cursor.rowcount > 0:
                    logger.debug(f'Updated context entry {context_id}, fields: {updated_fields}')
                    return True, list(updated_fields)

                if expected_version is not None:
                    # The entry exists for the caller (probed above) but the version did not match.
                    raise VersionConflictError(context_id)
                return False, []

            if txn:
                return await self._run_sqlite_txn(_update_entry_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_update_entry_sqlite)

        # PostgreSQL
        async def _update_entry_postgresql(conn: 'asyncpg.Connection') -> tuple[bool, list[str]]:
            row = await conn.fetchrow(probe_query, *probe_params)
            if not row:
                return False, []

            result = await conn.execute(update_query, *params)

            # Check if any rows were affected (asyncpg returns "UPDATE N")
            rows_affected = int(result.split()[-1]) if result else 0
            if rows_affected > 0:
                logger.debug(f'Updated context entry {context_id}, fields: {updated_fields}')
                return True, list(updated_fields)

            if expected_version is not None:
                raise VersionConflictError(context_id)
            return False, []

        if txn:
            return await _update_entry_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_update_entry_postgresql)

    async def touch_updated_at(
        self,
        context_id: str,
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> bool:
        """Advance an entry's public mutation timestamp without changing any other field.

        ``updated_at`` is auto-managed and PUBLIC: ``get_context_by_ids`` and every
        search tool return it, and clients key incremental sync and cache
        invalidation on it. Most update variants advance it as a side effect of the
        ``context_entries`` write they happen to issue, but a variant that touches
        only a CHILD table (the tags-only and images-only paths) issues no such
        write. This is the explicit stamp for those cases, so the contract is
        enforced by intent rather than by re-writing an unrelated column back to its
        own value to carry the timestamp along.

        Args:
            context_id: ID of the context entry.
            scope: The caller's scope; only an entry it may modify is stamped.
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            True when a row was stamped, False when the entry does not exist or the
            scope may not modify it.
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        # The id binds first, the WRITE predicate next.
        write = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=2,
        )
        params: list[object] = [context_id, *write.params]

        if backend_type == 'sqlite':
            sqlite_query = (
                f'UPDATE context_entries SET updated_at = CURRENT_TIMESTAMP '
                f'WHERE id = {self._placeholder(1)}{write.and_clause()}'
            )

            def _touch_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.cursor()
                cursor.execute(sqlite_query, tuple(params))
                return cursor.rowcount > 0

            if txn:
                return await self._run_sqlite_txn(_touch_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_touch_sqlite)

        postgresql_query = (
            f'UPDATE context_entries SET updated_at = CURRENT_TIMESTAMP '
            f'WHERE id = {self._placeholder(1)}::uuid{write.and_clause()}'
        )

        async def _touch_postgresql(conn: 'asyncpg.Connection') -> bool:
            result = await conn.execute(postgresql_query, *params)
            return int(result.split()[-1]) > 0 if result else False

        if txn:
            return await _touch_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(cast(Any, _touch_postgresql))

    async def update_content_type(
        self,
        context_id: str,
        content_type: str,
        *,
        scope: Scope,
        txn: 'TransactionContext | None' = None,
    ) -> bool:
        """Update the content type of a context entry.

        Args:
            context_id: ID of the context entry
            content_type: New content type ('text' or 'multimodal')
            scope: The caller's scope; only an entry it may modify is updated.
            txn: Optional transaction context for atomic multi-repository operations.
                When provided, uses the transaction's connection directly.
                When None, uses execute_write() for standalone operation.

        Returns:
            True if updated successfully, False when the entry does not exist or the
            scope may not modify it
        """
        backend_type = txn.backend_type if txn else self.backend.backend_type
        # The content type binds first, the id second, the WRITE predicate next.
        write = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=3,
        )
        query = (
            f'UPDATE context_entries SET content_type = {self._placeholder(1)}, '
            f'updated_at = CURRENT_TIMESTAMP WHERE id = {self._placeholder(2)}{write.and_clause()}'
        )
        params: list[object] = [content_type, context_id, *write.params]

        if backend_type == 'sqlite':

            def _update_content_type_sqlite(conn: sqlite3.Connection) -> bool:
                cursor = conn.cursor()
                cursor.execute(query, tuple(params))
                return cursor.rowcount > 0

            if txn:
                return await self._run_sqlite_txn(_update_content_type_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_update_content_type_sqlite)

        # PostgreSQL
        async def _update_content_type_postgresql(conn: 'asyncpg.Connection') -> bool:
            result = await conn.execute(query, *params)
            # asyncpg returns "UPDATE N" where N is the count
            return int(result.split()[-1]) > 0 if result else False

        if txn:
            return await _update_content_type_postgresql(cast('asyncpg.Connection', txn.connection))
        return await self.backend.execute_write(_update_content_type_postgresql)

    async def patch_metadata(
        self,
        context_id: str,
        patch: dict[str, Any],
        *,
        scope: Scope,
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

        Both the existence check and the UPDATE carry the WRITE predicate, so an
        entry the scope may not modify is reported exactly like a missing one.

        Args:
            context_id: ID of the context entry to update
            patch: Dictionary containing the merge patch to apply
            scope: The caller's scope; only an entry it may modify is patched.
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

        # The existence check binds the id first and the WRITE predicate next; the
        # UPDATE binds the patch, then the id, then the WRITE predicate.
        check_access = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=2,
        )
        check_query = f'SELECT id FROM context_entries WHERE id = {self._placeholder(1)}{check_access.and_clause()}'
        check_params: list[object] = [context_id, *check_access.params]
        update_access = build_access_predicate(
            scope, mode=AccessMode.WRITE, backend_type=backend_type, outer='context_entries', start=3,
        )
        update_params: list[object] = [patch_json, context_id, *update_access.params]

        if backend_type == 'sqlite':

            def _patch_metadata_sqlite(conn: sqlite3.Connection) -> tuple[bool, list[str]]:
                cursor = conn.cursor()

                # Verify the entry exists for the caller before attempting the update
                cursor.execute(check_query, tuple(check_params))
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
                    WHERE id = {self._placeholder(2)}{update_access.and_clause()}
                    ''',
                    tuple(update_params),
                )

                if cursor.rowcount > 0:
                    logger.debug(f'Patched metadata for context entry {context_id}')
                    return True, ['metadata']

                return False, []

            if txn:
                return await self._run_sqlite_txn(_patch_metadata_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_patch_metadata_sqlite)

        # PostgreSQL implementation - RFC 7396 compliant using jsonb_merge_patch() function
        async def _patch_metadata_postgresql(conn: 'asyncpg.Connection') -> tuple[bool, list[str]]:
            # Import settings here to avoid circular import and ensure schema is retrieved at call time
            from app.backends.postgresql_backend.session import quote_pg_identifier
            from app.settings import get_settings

            # Verify the entry exists for the caller before attempting the update
            row = await conn.fetchrow(check_query, *check_params)
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
            # Quote the schema identically to the DDL that CREATEs the function (via
            # quote_pg_identifier) so a mixed-case, reserved-word, or hyphenated
            # POSTGRESQL_SCHEMA resolves to the SAME object: an unquoted mixed-case
            # qualifier case-folds to a schema that does not exist (SQLSTATE 3F000/42883)
            # and a reserved/hyphenated name is a syntax error.
            schema = quote_pg_identifier(get_settings().storage.postgresql_schema)
            p1 = self._placeholder(1)
            p2 = self._placeholder(2)
            result = await conn.execute(
                f'''
                UPDATE context_entries
                SET metadata = {schema}.jsonb_merge_patch(COALESCE(metadata, '{{}}'::jsonb), {p1}::jsonb),
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = {p2}{update_access.and_clause()}
                ''',
                *update_params,
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
