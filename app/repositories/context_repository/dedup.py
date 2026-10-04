"""Deduplicating store for context entries.

A store whose text matches its dedup candidate updates the candidate instead of
inserting a duplicate. The candidate is the latest entry of the thread and
source that the caller may read; the store merges into it only when the caller
owns it, and an opposite-source entry the caller may read after the candidate
(a new conversational turn rather than a retransmitted request) suppresses the
merge. Every other store inserts a new entry owned by the caller.

The transactional store and its read-only pre-check build the candidate and
turn statements through the same helpers, so the pre-check reads exactly the
rows the store will read.
"""

import logging
import sqlite3
from typing import TYPE_CHECKING
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.ids import generate_id
from app.ids import normalize_id
from app.repositories.base import BaseRepository
from app.repositories.context_repository.helpers import compute_content_hash
from app.repositories.context_repository.records import DuplicateCandidate

if TYPE_CHECKING:
    import asyncpg

    from app.backends.base import TransactionContext


logger = logging.getLogger(__name__)

type Statement = tuple[str, list[object]]


def _param(backend_type: str, position: int) -> str:
    """Return the placeholder of the 1-based parameter ``position`` on ``backend_type``."""
    return '?' if backend_type == 'sqlite' else f'${position}'


def _candidate_sql(backend_type: str, thread_id: str, source: str, *, scope: AccessScope) -> Statement:
    """Build the statement selecting the dedup candidate of a store.

    The candidate is the latest entry of the thread and source that ``scope``
    may read; entries it may not read are skipped as if absent. The statement
    also returns the candidate's owner, because a store merges only into an
    entry its caller owns, and its summary, so the pre-check reuses a summary
    from the same row read that matched the content hash.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        thread_id: Thread of the store.
        source: Source of the store.
        scope: The caller's scope.

    Returns:
        The statement and its parameters.
    """
    read = build_access_predicate(
        scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries', start=3,
    )
    sql = (
        'SELECT id, content_hash, text_content, summary, owner_id FROM context_entries '
        f'WHERE thread_id = {_param(backend_type, 1)} AND source = {_param(backend_type, 2)}{read.and_clause()} '
        'ORDER BY id DESC LIMIT 1'
    )
    return sql, [thread_id, source, *read.params]


def _interleave_sql(
    backend_type: str, thread_id: str, source: str, candidate_id: object, *, scope: AccessScope,
) -> Statement:
    """Build the statement looking for a new conversational turn after the candidate.

    A turn is an entry of the opposite source in the same thread that follows
    the candidate. Only entries ``scope`` may read count, so an entry hidden
    from the caller never turns a retransmit into a new turn.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        thread_id: Thread of the store.
        source: Source of the store; the statement looks for the other one.
        candidate_id: ID of the dedup candidate.
        scope: The caller's scope.

    Returns:
        The statement and its parameters.
    """
    opposite_source = 'agent' if source == 'user' else 'user'
    read = build_access_predicate(
        scope, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries', start=4,
    )
    sql = (
        'SELECT 1 FROM context_entries '
        f'WHERE thread_id = {_param(backend_type, 1)} AND source = {_param(backend_type, 2)} '
        f'AND id > {_param(backend_type, 3)}{read.and_clause()} '
        'LIMIT 1'
    )
    return sql, [thread_id, opposite_source, candidate_id, *read.params]


def _dedup_update_sql(
    backend_type: str,
    *,
    scope: AccessScope,
    metadata: str | None,
    content_type: str | None,
    summary: str | None,
    content_hash: str,
    candidate_id: object,
    observed_hash: str | None,
) -> Statement:
    """Build the deduplication UPDATE of the candidate.

    ``metadata``, ``content_type`` and ``summary`` replace the stored values
    through COALESCE, so None keeps them. The WHERE clause re-asserts the dedup
    decision at write time: the null-safe content-hash term misses when a
    concurrent writer changed or deleted the candidate after it was read (under
    PostgreSQL READ COMMITTED a bare id predicate would re-evaluate against the
    newer row version and overwrite it), and the owner term limits the UPDATE
    to an entry the caller owns. A miss updates no row and the store inserts a
    new entry instead.

    Args:
        backend_type: ``'sqlite'`` or ``'postgresql'``.
        scope: The caller's scope; its principal must own the candidate.
        metadata: JSON metadata, or None to keep the stored metadata.
        content_type: Content type, or None to keep the stored one.
        summary: Summary, or None to keep the stored one.
        content_hash: Hash of the stored text, written back.
        candidate_id: ID of the dedup candidate.
        observed_hash: The candidate's hash the dedup decision read (None for
            rows stored before content hashes existed).

    Returns:
        The statement and its parameters.
    """
    null_safe_equal = 'IS' if backend_type == 'sqlite' else 'IS NOT DISTINCT FROM'
    owner = build_access_predicate(
        scope, mode=AccessMode.OWNER, backend_type=backend_type, outer='context_entries', start=7,
    )
    sql = (
        'UPDATE context_entries '
        f'SET metadata = COALESCE({_param(backend_type, 1)}, metadata), '
        f'content_type = COALESCE({_param(backend_type, 2)}, content_type), '
        f'summary = COALESCE({_param(backend_type, 3)}, summary), '
        f'content_hash = {_param(backend_type, 4)}, '
        'version = version + 1, '
        'updated_at = CURRENT_TIMESTAMP '
        f'WHERE id = {_param(backend_type, 5)} '
        f'AND content_hash {null_safe_equal} {_param(backend_type, 6)}{owner.and_clause()}'
    )
    return sql, [metadata, content_type, summary, content_hash, candidate_id, observed_hash, *owner.params]


def _is_owned_duplicate(
    owner_id: object,
    stored_hash: object,
    stored_text: object,
    *,
    scope: AccessScope,
    content_hash: str,
    text_content: str,
) -> bool:
    """Return whether a dedup candidate is a duplicate the caller may merge into.

    The caller must own the candidate. The content hash decides the match; rows
    stored before content hashes existed carry NULL and are compared by text.

    Args:
        owner_id: The candidate's owner.
        stored_hash: The candidate's content hash, None for a pre-hash row.
        stored_text: The candidate's text.
        scope: The caller's scope.
        content_hash: Hash of the text being stored.
        text_content: The text being stored.

    Returns:
        True when the caller owns the candidate and its text matches.
    """
    if owner_id != scope.principal_id:
        return False
    if stored_hash is not None:
        return stored_hash == content_hash
    return stored_text == text_content


class ContextDedupMixin(BaseRepository):
    """Deduplicating store over ``context_entries``.

    ``store_with_deduplication`` updates the dedup candidate when the caller owns
    it, its text is identical and no opposite-source entry the caller may read
    follows it, and inserts a new entry otherwise; ``check_latest_is_duplicate``
    runs the same match read-only so callers can skip generation for a
    retransmitted store.
    """

    async def store_with_deduplication(
        self,
        thread_id: str,
        source: str,
        content_type: str,
        text_content: str,
        *,
        scope: AccessScope,
        visibility: str,
        metadata: str | None = None,
        summary: str | None = None,
        preserve_content_type_on_dedup: bool = False,
        txn: 'TransactionContext | None' = None,
    ) -> tuple[str, bool]:
        """Store a context entry, updating the dedup candidate when this store re-sends its text.

        The candidate is the latest entry of the thread and source that ``scope``
        may read. When the scope's principal owns it, its text is identical and no
        opposite-source entry the scope may read follows it, the candidate's
        metadata and summary are updated (via COALESCE) together with
        content_type, content_hash, version (bumped by one) and updated_at.
        Otherwise a new entry is inserted.

        The access-control columns are stamped on INSERT only: a deduplication
        UPDATE deliberately never touches ``owner_id`` or ``visibility`` (a
        retransmit must not re-publish the existing row).

        Args:
            thread_id: Thread identifier
            source: 'user' or 'agent'
            content_type: 'text' or 'multimodal'
            text_content: The actual text content
            scope: The caller's scope, resolved by the server and never
                caller-supplied at the tool boundary. Deduplication considers
                only entries it may read and merges only into an entry its
                principal owns; a fresh INSERT is owned by its principal.
            visibility: 'private' or 'public'; stamped on a fresh INSERT.
            metadata: JSON metadata string or None
            summary: LLM-generated summary text or None
            preserve_content_type_on_dedup: When True, a deduplication UPDATE keeps the
                existing content_type instead of overwriting it. The store path sets this
                when images are PRESERVED (none provided this call) so a multimodal entry
                does not flip to 'text' while its image rows remain (which would make the
                images unretrievable). The INSERT path always uses the concrete
                content_type. Defaults to False (overwrite).
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

        # The candidate is matched by content hash; its text is compared only for
        # rows stored before content hashes existed (NULL hash).
        content_hash = compute_content_hash(text_content)
        candidate_sql, candidate_params = _candidate_sql(backend_type, thread_id, source, scope=scope)

        def _update_statement(candidate_id: object, observed_hash: str | None) -> Statement:
            return _dedup_update_sql(
                backend_type, scope=scope,
                metadata=metadata, content_type=dedup_content_type, summary=summary,
                content_hash=content_hash, candidate_id=candidate_id, observed_hash=observed_hash,
            )

        insert_values = (
            thread_id, source, content_type, text_content, metadata, summary, content_hash,
            scope.principal_id, visibility,
        )
        insert_sql = f'''
            INSERT INTO context_entries
            (id, thread_id, source, content_type, text_content, metadata, summary, content_hash,
             owner_id, visibility)
            VALUES ({self._placeholders(10)})
            '''

        if backend_type == 'sqlite':

            def _store_sqlite(conn: sqlite3.Connection) -> tuple[str, bool]:
                cursor = conn.cursor()
                cursor.execute(candidate_sql, candidate_params)
                candidate = cursor.fetchone()

                is_duplicate = candidate is not None and _is_owned_duplicate(
                    candidate['owner_id'], candidate['content_hash'], candidate['text_content'],
                    scope=scope, content_hash=content_hash, text_content=text_content,
                )
                if is_duplicate:
                    # A readable opposite-source entry after the candidate means this is a
                    # new conversational turn, not a retransmit: keep the chronological
                    # order by inserting instead.
                    cursor.execute(*_interleave_sql(backend_type, thread_id, source, candidate['id'], scope=scope))
                    is_duplicate = cursor.fetchone() is None

                if is_duplicate:
                    # The UPDATE's WHERE re-asserts the decision (see _dedup_update_sql).
                    # SQLite's serialized single writer makes a concurrent change between
                    # the read above and this UPDATE unreachable, but the same statement
                    # keeps the two backends behaviorally identical.
                    existing_id = cast(str, candidate['id'])
                    cursor.execute(*_update_statement(existing_id, candidate['content_hash']))
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
                cursor.execute(insert_sql, (new_id, *insert_values))
                logger.debug(f'Inserted new context entry {new_id} for thread {thread_id}')
                return new_id, False

            if txn:
                return await self._run_sqlite_txn(_store_sqlite, cast(sqlite3.Connection, txn.connection))
            return await self.backend.execute_write(_store_sqlite)

        # PostgreSQL
        async def _store_postgresql(conn: 'asyncpg.Connection') -> tuple[str, bool]:
            candidate = await conn.fetchrow(candidate_sql, *candidate_params)

            is_duplicate = candidate is not None and _is_owned_duplicate(
                candidate['owner_id'], candidate['content_hash'], candidate['text_content'],
                scope=scope, content_hash=content_hash, text_content=text_content,
            )
            if is_duplicate and candidate is not None:
                # A readable opposite-source entry after the candidate means this is a
                # new conversational turn, not a retransmit: keep the chronological
                # order by inserting instead.
                turn_sql, turn_params = _interleave_sql(backend_type, thread_id, source, candidate['id'], scope=scope)
                is_duplicate = await conn.fetchrow(turn_sql, *turn_params) is None

            if is_duplicate and candidate is not None:
                # The UPDATE's WHERE re-asserts the decision (see _dedup_update_sql):
                # under READ COMMITTED a concurrent writer can commit a text change or a
                # delete between the read above and this UPDATE.
                existing_id = candidate['id']
                update_sql, update_params = _update_statement(existing_id, candidate['content_hash'])
                result = await conn.execute(update_sql, *update_params)
                rows_affected = int(result.split()[-1]) if result else 0
                if rows_affected > 0:
                    logger.debug(f'Updated existing context entry {existing_id} for thread {thread_id}')
                    # The pool's uuid->str codec (registered in init_pool_connection)
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
            await conn.execute(insert_sql, new_id, *insert_values)
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
        *,
        scope: AccessScope,
    ) -> DuplicateCandidate | None:
        """Check whether a store would merge into its dedup candidate (read-only pre-check).

        This is a performance optimization for the embedding-first pattern.
        It allows skipping expensive embedding generation when the content
        is identical to the candidate. The in-transaction deduplication
        in store_with_deduplication remains as the authoritative safety net.

        The candidate and the turn check are the same statements the store runs:
        the candidate is the latest entry of the thread and source that ``scope``
        may read, and it is reported only when the scope's principal owns it and
        its text matches. A candidate owned by another principal is never
        reported, so the caller neither reuses its summary nor probes its
        embeddings.

        The candidate's stored ``summary`` is returned from the SAME statement
        that matched the content hash (see :class:`DuplicateCandidate`), so a
        caller reusing it never pairs a summary with text it does not
        describe: a separate later summary read could observe a row version a
        concurrent update committed in between, and the dedup UPDATE's
        content-hash predicate cannot tell a revision-consistent row from a
        restored one, so the mismatched summary would persist via COALESCE.

        An opposite-source entry (agent for user source, user for agent source)
        that ``scope`` may read after the candidate suppresses deduplication and
        returns None: identical text sent as a new conversational turn keeps its
        chronological place instead of merging.

        Args:
            thread_id: Thread identifier
            source: 'user' or 'agent'
            text_content: Text content to check for duplicates
            scope: The caller's scope.

        Returns:
            A :class:`DuplicateCandidate` snapshot (context_id + stored
            summary) if a duplicate is found, None if no match, if another
            principal owns the candidate, or if a readable new turn follows it.
        """
        content_hash = compute_content_hash(text_content)
        backend_type = self.backend.backend_type
        candidate_sql, candidate_params = _candidate_sql(backend_type, thread_id, source, scope=scope)

        if backend_type == 'sqlite':

            def _check_sqlite(conn: sqlite3.Connection) -> DuplicateCandidate | None:
                cursor = conn.cursor()
                cursor.execute(candidate_sql, candidate_params)
                row = cursor.fetchone()
                if row is None or not _is_owned_duplicate(
                    row['owner_id'], row['content_hash'], row['text_content'],
                    scope=scope, content_hash=content_hash, text_content=text_content,
                ):
                    return None
                cursor.execute(*_interleave_sql(backend_type, thread_id, source, row['id'], scope=scope))
                if cursor.fetchone() is not None:
                    return None
                return DuplicateCandidate(
                    context_id=cast(str, row['id']),
                    summary=cast(str | None, row['summary']),
                )

            return await self.backend.execute_read(_check_sqlite)

        # PostgreSQL
        async def _check_postgresql(conn: 'asyncpg.Connection') -> DuplicateCandidate | None:
            row = await conn.fetchrow(candidate_sql, *candidate_params)
            if row is None or not _is_owned_duplicate(
                row['owner_id'], row['content_hash'], row['text_content'],
                scope=scope, content_hash=content_hash, text_content=text_content,
            ):
                return None
            turn_sql, turn_params = _interleave_sql(backend_type, thread_id, source, row['id'], scope=scope)
            if await conn.fetchrow(turn_sql, *turn_params) is not None:
                return None
            return DuplicateCandidate(
                context_id=cast(str, row['id']),
                summary=cast(str | None, row['summary']),
            )

        return await self.backend.execute_read(_check_postgresql)
