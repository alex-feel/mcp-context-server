"""Corpus sync tools: plan the changes that bring a corpus thread in line with its source, and list its stored state.

A sync client manages the entries of a corpus thread whose reserved ``metadata.ingest``
block names that thread as its scope. Both tools only read:

- ``plan_context_sync`` compares a manifest of the corpus's source documents with
  the caller's managed entries of a source-id range and answers which items to
  create, which entries to update and which to delete; the client applies the plan
  through the batch tools.
- ``list_ingest_state`` pages through the caller's managed entries of a thread in
  ``(source_id, context_id)`` order.

Both manage only entries the caller owns; another principal's readable members are
counted by the plan and never listed.
"""

import base64
import json
import logging
from typing import Annotated
from typing import cast

from fastmcp.exceptions import ToolError
from pydantic import Field

from app.auth import resolve_access_scope
from app.errors import format_exception_message
from app.ids import normalize_id
from app.ingest import DEFAULT_INGEST_STATE_PAGE
from app.ingest import MAX_INGEST_STATE_PAGE
from app.ingest import MAX_MANIFEST_ITEMS
from app.ingest import MAX_PLAN_DELETES
from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import IngestStateEntryDict
from app.ingest import ListIngestStateDict
from app.ingest import PlanContextSyncDict
from app.ingest import SyncManifestItem
from app.models import MAX_THREAD_ID_LENGTH
from app.repositories.context_repository.records import IngestRow
from app.startup import ensure_repositories
from app.tools._sync_plan import compute_sync_plan
from app.tools._sync_plan import manifest_error
from app.tools._sync_plan import stored_member
from app.tools._validation import reject_unstorable_input

logger = logging.getLogger(__name__)

_INVALID_CURSOR = 'invalid cursor'


async def plan_context_sync(
    thread_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_THREAD_ID_LENGTH,
            description='The corpus thread to plan, such as kb:<corpus>',
        ),
    ],
    manifest: Annotated[
        list[SyncManifestItem],
        Field(
            max_length=MAX_MANIFEST_ITEMS,
            description='The source documents of the planned range: {source_id, text_digest, meta_digest} items, '
                        'or {source_id, keep: true} for a source the client could not read; each source_id at most '
                        f'once (at most {MAX_MANIFEST_ITEMS} items)',
        ),
    ],
    source_id_after: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=MAX_SOURCE_ID_LENGTH,
            description='Exclusive lower bound of the planned source ids; omit for no lower bound',
        ),
    ] = None,
    source_id_through: Annotated[
        str | None,
        Field(
            min_length=1,
            max_length=MAX_SOURCE_ID_LENGTH,
            description='Inclusive upper bound of the planned source ids; omit for no upper bound',
        ),
    ] = None,
    allow_empty_manifest: Annotated[
        bool,
        Field(description='Confirm an empty manifest, which plans the deletion of every managed entry in the range'),
    ] = False,
) -> PlanContextSyncDict:
    """Plan the changes that bring a corpus thread in line with a manifest of its source documents.

    Compares the manifest with the caller's own managed entries of the thread, the
    entries whose metadata.ingest block names the thread as its scope, inside the
    range (source_id_after, source_id_through]; an omitted bound leaves that side
    open, and source ids compare by code point. An ordinary item carries the SHA-256
    digest of the text it would store after surrounding whitespace is removed, and
    the digest of its metadata; a keep item {source_id, keep: true} marks a source
    the client could not read in this run, so nothing stored for it changes.

    Creates and updates are named by manifest index: creates lists the items that
    have no stored entry, and updates names the stored entry each changed item
    replaces and whether its text, its metadata or both differ. Deletes are named
    by context id with a reason: absent when the manifest does not name the source,
    duplicate when another entry of the same source survives. At most 100 deletes
    are listed per call and deletes_total counts them all, so delete the listed
    entries and plan again until deletes_total is 0. unchanged_count counts the
    items already in line, kept_count the keep items, and foreign_count the
    readable managed entries of the range another principal owns, which a sync
    never changes.

    An empty manifest plans the deletion of every managed entry in the range, so it
    needs allow_empty_manifest=true. The tool only reads; apply the plan with the
    batch tools. See docs/knowledge-base.md, 'Sync Planning'.

    Returns:
        PlanContextSyncDict with thread_id, creates, updates, deletes, deletes_total,
        unchanged_count, kept_count and foreign_count.

    Raises:
        ToolError: If the arguments are invalid or reading the stored entries fails.
    """
    try:
        thread_id = thread_id.strip()
        if not thread_id:
            raise ToolError('thread_id cannot be empty or whitespace')
        reject_unstorable_input(
            thread_id=thread_id,
            source_id_after=source_id_after,
            source_id_through=source_id_through,
            manifest_source_ids=[item.source_id for item in manifest],
        )
        problem = manifest_error(
            manifest,
            source_id_after=source_id_after,
            source_id_through=source_id_through,
            allow_empty=allow_empty_manifest,
        )
        if problem is not None:
            raise ToolError(problem)

        scope = resolve_access_scope()
        repos = await ensure_repositories()
        rows = await repos.context.ingest_rows_in_range(
            scope=scope,
            thread_id=thread_id,
            source_id_after=source_id_after,
            source_id_through=source_id_through,
        )
        members = [
            member
            for row in rows
            if (
                member := stored_member(
                    row.block, context_id=row.context_id, content_hash=row.content_hash, is_owner=row.is_owner,
                )
            )
            is not None
        ]
        plan = compute_sync_plan(manifest, members)

        return {
            'thread_id': thread_id,
            'creates': plan.creates,
            'updates': plan.updates,
            'deletes': plan.deletes[:MAX_PLAN_DELETES],
            'deletes_total': len(plan.deletes),
            'unchanged_count': plan.unchanged_count,
            'kept_count': plan.kept_count,
            'foreign_count': plan.foreign_count,
        }
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error planning context sync: {exc}')
        raise ToolError(f'Failed to plan context sync: {format_exception_message(exc)}') from exc


async def list_ingest_state(
    thread_id: Annotated[
        str,
        Field(
            min_length=1,
            max_length=MAX_THREAD_ID_LENGTH,
            description='The corpus thread to list, such as kb:<corpus>',
        ),
    ],
    limit: Annotated[
        int,
        Field(
            ge=1,
            le=MAX_INGEST_STATE_PAGE,
            description=f'Maximum entries per page (1-{MAX_INGEST_STATE_PAGE}, default {DEFAULT_INGEST_STATE_PAGE})',
        ),
    ] = DEFAULT_INGEST_STATE_PAGE,
    cursor: Annotated[
        str | None,
        Field(min_length=1, description='The next_cursor of the previous page; omit for the first page'),
    ] = None,
) -> ListIngestStateDict:
    """List one page of the caller's own managed entries of a corpus thread.

    Lists the entries the caller owns whose metadata.ingest block names the thread
    as its scope, ordered by source id and then context id in code-point order, the
    order the ranges of plan_context_sync use. Entries owned by another principal are
    never listed. Each entry carries its context_id, source_id, text_digest (the
    stored SHA-256 digest of its text), meta_digest and synced_at. Pass a page's
    next_cursor to read the following page; next_cursor is null on the last page.
    See docs/knowledge-base.md, 'Listing Ingest State'.

    Returns:
        ListIngestStateDict with thread_id, entries, count and next_cursor.

    Raises:
        ToolError: If the cursor is invalid or reading the stored entries fails.
    """
    try:
        thread_id = thread_id.strip()
        if not thread_id:
            raise ToolError('thread_id cannot be empty or whitespace')
        reject_unstorable_input(thread_id=thread_id)
        after = _decode_cursor(cursor) if cursor is not None else None

        scope = resolve_access_scope()
        repos = await ensure_repositories()
        # One row past the page tells whether another page follows.
        rows = await repos.context.owned_ingest_page(scope=scope, thread_id=thread_id, after=after, limit=limit + 1)
        window = rows[:limit]
        entries = [entry for row in window if (entry := _state_entry(row)) is not None]
        # The cursor names the last row of the window even when its block is invalid,
        # so a skipped row never stops the paging.
        next_cursor = _encode_cursor(window[-1].source_id, window[-1].context_id) if len(rows) > limit else None

        return {
            'thread_id': thread_id,
            'entries': entries,
            'count': len(entries),
            'next_cursor': next_cursor,
        }
    except ToolError:
        raise
    except Exception as exc:
        logger.error(f'Error listing ingest state: {exc}')
        raise ToolError(f'Failed to list ingest state: {format_exception_message(exc)}') from exc


def _state_entry(row: IngestRow) -> IngestStateEntryDict | None:
    """Return the listed form of a managed row, or None when its block is invalid."""
    member = stored_member(row.block, context_id=row.context_id, content_hash=row.content_hash, is_owner=row.is_owner)
    if member is None:
        return None
    # A valid block is an object of exactly four string fields.
    block = cast('dict[str, str]', row.block)
    return {
        'context_id': member.context_id,
        'source_id': member.source_id,
        'text_digest': member.content_hash,
        'meta_digest': member.meta_digest,
        'synced_at': block['synced_at'],
    }


def _encode_cursor(source_id: str, context_id: str) -> str:
    """Return the cursor of the page that follows the row ``(source_id, context_id)``.

    Args:
        source_id: The source id of the last row of the page.
        context_id: The context id of the last row of the page.

    Returns:
        The unpadded base64url form of the compact UTF-8 JSON array ``[source_id, context_id]``.
    """
    payload = json.dumps([source_id, context_id], separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return base64.urlsafe_b64encode(payload).decode('ascii').rstrip('=')


def _decode_cursor(cursor: str) -> tuple[str, str]:
    """Return the ``(source_id, context_id)`` a cursor names.

    Decoding is strict: the cursor must be unpadded base64url of a UTF-8 JSON array
    of exactly two strings, a source id of 1 to ``MAX_SOURCE_ID_LENGTH`` code points
    and a valid context id. Only that shape is checked, so a well-formed cursor this
    tool never returned is accepted; the page read binds it as parameters under the
    caller's owner predicate, so it moves the window only within the caller's own
    entries.

    Args:
        cursor: The cursor a previous page returned.

    Returns:
        The source id and the normalized context id of the last row of that page.

    Raises:
        ToolError: If the cursor does not have that shape.
    """
    try:
        payload = base64.b64decode(cursor + '=' * (-len(cursor) % 4), altchars=b'-_', validate=True)
        value: object = json.loads(payload.decode('utf-8'))
    except ValueError:
        # binascii.Error, UnicodeDecodeError and json.JSONDecodeError are all ValueErrors.
        raise ToolError(_INVALID_CURSOR) from None
    if not isinstance(value, list):
        raise ToolError(_INVALID_CURSOR)
    parts = cast('list[object]', value)
    if len(parts) != 2:
        raise ToolError(_INVALID_CURSOR)
    source_id, context_id = parts
    if not isinstance(source_id, str) or not isinstance(context_id, str):
        raise ToolError(_INVALID_CURSOR)
    if not 1 <= len(source_id) <= MAX_SOURCE_ID_LENGTH:
        raise ToolError(_INVALID_CURSOR)
    try:
        return source_id, normalize_id(context_id)
    except ValueError:
        raise ToolError(_INVALID_CURSOR) from None
