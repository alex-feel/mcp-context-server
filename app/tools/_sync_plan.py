"""The corpus sync diff: from a manifest and the stored members of a range to a plan.

``plan_context_sync`` checks its arguments with :func:`manifest_error`, turns
each managed row it read into a :class:`StoredMember` with :func:`stored_member`
and computes the plan with :func:`compute_sync_plan`. Nothing here performs I/O,
so a plan is decided by its inputs alone.

A plan manages only the caller's own members: another principal's members of
the range are counted and never planned. A keep item marks a source the client
could not read in this run, so nothing stored for it changes, and a client-side
read failure never deletes server state. When one source has several of the
caller's members, a total order picks the survivor, so every plan over the same
rows names the same survivor whatever order the rows arrive in.
"""

from collections import Counter
from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Sequence
from dataclasses import dataclass
from operator import itemgetter
from typing import cast

from app.ingest import SyncChange
from app.ingest import SyncDeleteDict
from app.ingest import SyncDeleteReason
from app.ingest import SyncManifestItem
from app.ingest import SyncUpdateDict
from app.ingest import ingest_block_error

# How many repeated source ids an argument error names before it counts the rest.
_MAX_NAMED_REPEATS = 10


@dataclass(frozen=True, slots=True)
class StoredMember:
    """A managed entry of the planned range, read back from the database.

    Attributes:
        context_id: The entry id.
        source_id: The source id the entry's block carries.
        content_hash: The stored hash of the entry's text; None for an entry
            stored before content hashes existed, which counts as changed text.
        meta_digest: The block's metadata digest.
        is_owner: Whether the caller owns the entry.
    """

    context_id: str
    source_id: str
    content_hash: str | None
    meta_digest: str
    is_owner: bool


@dataclass(frozen=True, slots=True)
class SyncPlan:
    """What brings the caller's managed entries of a range in line with a manifest.

    Attributes:
        creates: Manifest indices of the items to store as new entries.
        updates: The stored entries to update, in manifest order.
        deletes: Every planned delete, ordered by source id and then context id.
        unchanged_count: Manifest items whose surviving entry already matches.
        kept_count: Keep items in the manifest.
        foreign_count: Members owned by another principal.
    """

    creates: list[int]
    updates: list[SyncUpdateDict]
    deletes: list[SyncDeleteDict]
    unchanged_count: int
    kept_count: int
    foreign_count: int


def stored_member(block: object, *, context_id: str, content_hash: str | None, is_owner: bool) -> StoredMember | None:
    """Return the member a stored row represents, or None when its block is invalid.

    The write tools refuse a malformed block, so an invalid block survives only
    in rows written before that check; such a row is skipped, the same way in
    every plan. Whether the block's scope names the planned thread is decided by
    the read that produced the row.

    Args:
        block: The row's parsed ``metadata.ingest`` block.
        context_id: The row's entry id.
        content_hash: The row's stored text hash.
        is_owner: Whether the caller owns the row.

    Returns:
        The member, or None when the block fails validation.
    """
    if ingest_block_error(block) is not None:
        return None
    # A valid block is an object of exactly four string fields.
    fields = cast('dict[str, str]', block)
    return StoredMember(
        context_id=context_id,
        source_id=fields['source_id'],
        content_hash=content_hash,
        meta_digest=fields['meta_digest'],
        is_owner=is_owner,
    )


def manifest_error(
    manifest: Sequence[SyncManifestItem],
    *,
    source_id_after: str | None,
    source_id_through: str | None,
    allow_empty: bool,
) -> str | None:
    """Return the first problem with a plan request's arguments, or None.

    The checks run in a fixed order: the range bounds, an empty manifest the
    caller did not confirm, repeated source ids, then items outside the range
    ``(source_id_after, source_id_through]``. Source ids compare by code point.

    Args:
        manifest: The manifest items.
        source_id_after: The exclusive lower bound of the planned range, or None.
        source_id_through: The inclusive upper bound of the planned range, or None.
        allow_empty: Whether the caller confirmed that an empty manifest is meant,
            which plans the deletion of every managed entry in the range.

    Returns:
        The error message, or None when the arguments are valid.
    """
    if source_id_after is not None and source_id_through is not None and source_id_after >= source_id_through:
        return 'source_id_after must sort before source_id_through'
    if not manifest and not allow_empty:
        return (
            'manifest is empty: planning an empty manifest deletes every managed entry in the range; '
            'pass allow_empty_manifest=true to confirm'
        )
    repeated = [source_id for source_id, count in Counter(item.source_id for item in manifest).items() if count > 1]
    if repeated:
        named = ', '.join(repr(source_id) for source_id in repeated[:_MAX_NAMED_REPEATS])
        unnamed = len(repeated) - _MAX_NAMED_REPEATS
        remainder = f' and {unnamed} more' if unnamed > 0 else ''
        return f'manifest repeats source_id values (each may appear once): {named}{remainder}'
    for index, item in enumerate(manifest):
        below = source_id_after is not None and item.source_id <= source_id_after
        above = source_id_through is not None and item.source_id > source_id_through
        if below or above:
            return (
                f'manifest item {index} has source_id {item.source_id!r}, '
                'outside the planned range (source_id_after, source_id_through]'
            )
    return None


def compute_sync_plan(manifest: Sequence[SyncManifestItem], members: Iterable[StoredMember]) -> SyncPlan:
    """Compare a manifest with the stored members of its range and plan the difference.

    Each ordinary item whose source has none of the caller's members is created.
    Otherwise its members are ranked, those matching the item's text digest
    first, then those matching its metadata digest, then the newest id; the
    first survives and the others are deleted as duplicates. The survivor is
    updated in whatever differs, or counted unchanged. A keep item is counted
    and leaves its source's members as they are. The caller's members of
    sources the manifest does not name are deleted as absent. Members owned by
    another principal are only counted.

    Args:
        manifest: Items whose source ids are distinct, as :func:`manifest_error`
            requires.
        members: The stored members of the planned range, in any order.

    Returns:
        The plan.
    """
    owned: defaultdict[str, list[StoredMember]] = defaultdict(list)
    foreign_count = 0
    for member in members:
        if member.is_owner:
            owned[member.source_id].append(member)
        else:
            foreign_count += 1

    creates: list[int] = []
    updates: list[SyncUpdateDict] = []
    deletes: list[SyncDeleteDict] = []
    unchanged_count = 0
    kept_count = 0
    for index, item in enumerate(manifest):
        # Taking the item's members out leaves only sources the manifest does not name.
        candidates = owned.pop(item.source_id, [])
        if item.keep:
            kept_count += 1
            continue
        if not candidates:
            creates.append(index)
            continue
        survivor, *duplicates = _ranked(candidates, item)
        deletes.extend(_delete(member, 'duplicate') for member in duplicates)
        change = _change(survivor, item)
        if change is None:
            unchanged_count += 1
        else:
            updates.append({'index': index, 'context_id': survivor.context_id, 'change': change})
    for absent in owned.values():
        deletes.extend(_delete(member, 'absent') for member in absent)
    deletes.sort(key=itemgetter('source_id', 'context_id'))

    return SyncPlan(
        creates=creates,
        updates=updates,
        deletes=deletes,
        unchanged_count=unchanged_count,
        kept_count=kept_count,
        foreign_count=foreign_count,
    )


def _ranked(candidates: list[StoredMember], item: SyncManifestItem) -> list[StoredMember]:
    """Rank one source's members best first: text match, then metadata match, then the newest id.

    Context ids are unique, so the order is total and never depends on the
    order the members arrived in.

    Args:
        candidates: The caller's members of the item's source.
        item: The manifest item they are compared with.

    Returns:
        The members, the survivor first.
    """

    def rank(member: StoredMember) -> tuple[bool, bool, str]:
        return member.content_hash == item.text_digest, member.meta_digest == item.meta_digest, member.context_id

    return sorted(candidates, key=rank, reverse=True)


def _change(survivor: StoredMember, item: SyncManifestItem) -> SyncChange | None:
    """Return what the survivor needs from the item, or None when it already matches."""
    text_matches = survivor.content_hash == item.text_digest
    meta_matches = survivor.meta_digest == item.meta_digest
    if text_matches and meta_matches:
        return None
    if meta_matches:
        return 'text'
    if text_matches:
        return 'metadata'
    return 'both'


def _delete(member: StoredMember, reason: SyncDeleteReason) -> SyncDeleteDict:
    """Return the planned delete of a member."""
    return {'context_id': member.context_id, 'source_id': member.source_id, 'reason': reason}
