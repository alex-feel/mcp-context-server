"""Deduplicating store for context entries.

A store whose thread, source and text match the latest entry updates that
entry instead of inserting a duplicate, unless an opposite-source entry follows
it (a new conversational turn rather than a retransmitted request).
"""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import cast

from app.ids import generate_id
from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import compute_content_hash
from app.repositories.context_repository.records import DuplicateCandidate

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


logger = logging.getLogger(__name__)


class ContextDedupMixin(BaseRepository):
    """Deduplicating store over ``context_entries``.

    ``store_with_deduplication`` updates the latest entry of the same thread and
    source when its text is identical and no opposite-source entry follows it, and
    inserts a new entry otherwise; ``check_latest_is_duplicate`` runs the same
    match read-only so callers can skip generation for a retransmitted store.
    """

    async def store_with_deduplication(
        self,
        thread_id: str,
        source: str,
        content_type: str,
        text_content: str,
        *,
        owner_id: str,
        visibility: str,
        metadata: str | None = None,
        summary: str | None = None,
        preserve_content_type_on_dedup: bool = False,
        txn: 'TransactionContext | None' = None,
    ) -> tuple[str, bool]:
        """Store context entry with deduplication logic.

        Checks if the latest entry has identical thread_id, source, and text_content.
        If found, updates metadata and summary (via COALESCE), content_type, content_hash,
        version (bumped by one), and updated_at. Otherwise, inserts new entry.

        The access-control columns are stamped on INSERT only: a deduplication
        UPDATE deliberately never touches ``owner_id`` or ``visibility`` (a
        retransmit must not re-own or re-publish the existing row).

        Args:
            thread_id: Thread identifier
            source: 'user' or 'agent'
            content_type: 'text' or 'multimodal'
            text_content: The actual text content
            owner_id: Server-resolved principal stamped as the row owner on a
                fresh INSERT. Never caller-supplied at the tool boundary.
            visibility: 'private', 'shared', or 'public'; stamped on a fresh
                INSERT.
            metadata: JSON metadata string or None
            summary: LLM-generated summary text or None
            preserve_content_type_on_dedup: When True, a deduplication UPDATE keeps the
                existing content_type instead of overwriting it. The store path sets this
                when images are PRESERVED (none provided this call) so a multimodal entry
                does not flip to 'text' while its image rows remain (which would make the
                images unretrievable). The INSERT path always uses the concrete
                content_type. Defaults to False (overwrite, prior behavior).
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

        # On a dedup UPDATE, content_type must reflect the entry's FINAL image state.
        # When the store is preserving images (none provided this call), preserve the
        # existing content_type via COALESCE(NULL, content_type) -- overwriting it would
        # flip a multimodal entry to 'text' while its image rows remain, making those
        # images permanently unretrievable. The INSERT path still uses the concrete
        # content_type (a fresh row carries only the images supplied in this request).
        dedup_content_type = None if preserve_content_type_on_dedup else content_type

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
                # The hash value the dedup decision observed; re-asserted by the
                # UPDATE's null-safe predicate below (NULL for pre-migration rows).
                observed_hash: str | None = None
                if latest_row:
                    observed_hash = latest_row['content_hash']
                    if observed_hash is not None:
                        # Hash-based comparison (fast path)
                        is_duplicate = observed_hash == content_hash
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
                    # The latest entry has identical text - update metadata, content_type,
                    # and timestamp. The content_hash predicate (IS = SQLite's null-safe
                    # equality) re-asserts the dedup decision at write time: a concurrent
                    # committed writer that changed the candidate's text (its hash) or
                    # deleted the row between the SELECT above and this UPDATE makes the
                    # predicate miss, so the store falls through to INSERT instead of
                    # overwriting the newer text's hash/summary/metadata with values
                    # describing THIS request's text. SQLite's serialized single writer
                    # makes the race unreachable here, but the guard keeps the two
                    # backends behaviorally identical.
                    existing_id = latest_row['id']
                    cursor.execute(
                        f'''
                        UPDATE context_entries
                        SET metadata = COALESCE({self._placeholder(1)}, metadata),
                            content_type = COALESCE({self._placeholder(2)}, content_type),
                            summary = COALESCE({self._placeholder(3)}, summary),
                            content_hash = {self._placeholder(4)},
                            version = version + 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = {self._placeholder(5)}
                          AND content_hash IS {self._placeholder(6)}
                        ''',
                        (metadata, dedup_content_type, summary, content_hash, existing_id, observed_hash),
                    )
                    if cursor.rowcount > 0:
                        logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                        return existing_id, True
                    # A concurrent writer invalidated the dedup decision; fall through to
                    # the INSERT below (the caller's reconcile machinery regenerates any
                    # generation legs its pre-check skipped expecting an UPDATE).
                    logger.info(
                        'Deduplication UPDATE for context %s in thread %s matched 0 rows '
                        '(concurrent modification); inserting a new entry instead',
                        existing_id, thread_id,
                    )

                # No duplicate - insert new entry with pre-generated UUIDv7 hex id.
                new_id = generate_id()
                cursor.execute(
                    f'''
                    INSERT INTO context_entries
                    (id, thread_id, source, content_type, text_content, metadata, summary, content_hash,
                     owner_id, visibility)
                    VALUES ({self._placeholders(10)})
                    ''',
                    (
                        new_id, thread_id, source, content_type, text_content, metadata, summary,
                        content_hash, owner_id, visibility,
                    ),
                )
                logger.debug(f'Inserted new context entry {new_id} for thread {thread_id}')
                return new_id, False

            if txn:
                return await self._run_sqlite_txn(_store_sqlite, cast(sqlite3.Connection, txn.connection))
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
            # The hash value the dedup decision observed; re-asserted by the
            # UPDATE's null-safe predicate below (NULL for pre-migration rows).
            observed_hash: str | None = None
            if latest_row:
                observed_hash = latest_row['content_hash']
                if observed_hash is not None:
                    is_duplicate = observed_hash == content_hash
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
                # Update metadata, content_type, and timestamp. The content_hash
                # predicate (IS NOT DISTINCT FROM = null-safe equality) re-asserts
                # the dedup decision at write time: under READ COMMITTED a
                # concurrent writer can commit a text change (new hash) or a delete
                # between the SELECT above and this UPDATE, and the bare id
                # predicate would re-evaluate against the NEW row version
                # (EvalPlanQual) and overwrite it -- poisoning content_hash with
                # THIS request's hash and cross-attributing summary/metadata to
                # different text. A 0-row match instead falls through to INSERT.
                existing_id = latest_row['id']
                result = await conn.execute(
                    f'''
                        UPDATE context_entries
                        SET metadata = COALESCE({self._placeholder(1)}, metadata),
                            content_type = COALESCE({self._placeholder(2)}, content_type),
                            summary = COALESCE({self._placeholder(3)}, summary),
                            content_hash = {self._placeholder(4)},
                            version = version + 1,
                            updated_at = CURRENT_TIMESTAMP
                        WHERE id = {self._placeholder(5)}
                          AND content_hash IS NOT DISTINCT FROM {self._placeholder(6)}
                        ''',
                    metadata,
                    dedup_content_type,
                    summary,
                    content_hash,
                    existing_id,
                    observed_hash,
                )
                rows_affected = int(result.split()[-1]) if result else 0
                if rows_affected > 0:
                    logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                    # The pool's uuid->str codec (registered in _init_connection)
                    # already yields the canonical 32-char hex the MCP API contract
                    # requires (the SQLite branch's TEXT id is likewise hex);
                    # normalize_id(str(...)) is idempotent defense-in-depth for
                    # codec-less connections.
                    return normalize_id(str(existing_id)), True
                # A concurrent writer invalidated the dedup decision; fall through to
                # the INSERT below (the caller's reconcile machinery regenerates any
                # generation legs its pre-check skipped expecting an UPDATE).
                logger.info(
                    'Deduplication UPDATE for context %s in thread %s matched 0 rows '
                    '(concurrent modification); inserting a new entry instead',
                    existing_id, thread_id,
                )

            # No duplicate - insert new entry with pre-generated UUIDv7 hex id.
            new_id = generate_id()
            await conn.execute(
                f'''
                    INSERT INTO context_entries
                    (id, thread_id, source, content_type, text_content, metadata, summary, content_hash,
                     owner_id, visibility)
                    VALUES ({self._placeholders(10)})
                    ''',
                new_id,
                thread_id,
                source,
                content_type,
                text_content,
                metadata,
                summary,
                content_hash,
                owner_id,
                visibility,
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
    ) -> DuplicateCandidate | None:
        """Check if the latest entry matches the given content (read-only pre-check).

        This is a performance optimization for the embedding-first pattern.
        It allows skipping expensive embedding generation when the content
        is identical to the latest entry. The in-transaction deduplication
        in store_with_deduplication remains as the authoritative safety net.

        The candidate's stored ``summary`` is returned from the SAME statement
        that matched the content hash (see :class:`DuplicateCandidate`), so a
        caller reusing it never pairs a summary with text it does not
        describe: a separate later summary read could observe a row version a
        concurrent update committed in between, and the dedup UPDATE's
        content-hash predicate cannot tell a revision-consistent row from a
        restored one, so the mismatched summary would persist via COALESCE.

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
            A :class:`DuplicateCandidate` snapshot (context_id + stored
            summary) if a duplicate is found, None if no match or if
            interleaving entries suppress deduplication.
        """
        content_hash = compute_content_hash(text_content)

        if self.backend.backend_type == 'sqlite':

            def _check_sqlite(conn: sqlite3.Connection) -> DuplicateCandidate | None:
                cursor = conn.cursor()
                cursor.execute(
                    f'''
                    SELECT id, content_hash, text_content, summary FROM context_entries
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
                return DuplicateCandidate(
                    context_id=cast(str, candidate_id),
                    summary=cast(str | None, row['summary']),
                )

            return await self.backend.execute_read(_check_sqlite)

        # PostgreSQL
        async def _check_postgresql(conn: 'asyncpg.Connection') -> DuplicateCandidate | None:
            row = await conn.fetchrow(
                f'''
                    SELECT id, content_hash, text_content, summary FROM context_entries
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
            return DuplicateCandidate(
                context_id=cast(str, candidate_id),
                summary=cast(str | None, row['summary']),
            )

        return await self.backend.execute_read(_check_postgresql)
