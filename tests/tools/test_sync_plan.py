"""Tests for the pure corpus sync diff: stored members, argument errors and the plan.

The diff compares a manifest with the caller's stored members of one thread's
range. It plans creates and updates by manifest index and deletes by id, keeps
whatever a keep item names, never touches another principal's entries, and picks
the surviving copy of a duplicated source by a total order, so any two plans over
the same rows name the same survivor.
"""

import hashlib
import itertools
from typing import Literal

import pytest

from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import SyncDeleteDict
from app.ingest import SyncManifestItem
from app.ingest import SyncUpdateDict
from app.tools._sync_plan import StoredMember
from app.tools._sync_plan import SyncPlan
from app.tools._sync_plan import compute_sync_plan
from app.tools._sync_plan import manifest_error
from app.tools._sync_plan import stored_member

type Change = Literal['text', 'metadata', 'both']
type Reason = Literal['absent', 'duplicate']


def _digest(seed: str) -> str:
    """Return a distinct 64-character lowercase hex digest for a seed."""
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()


TEXT = _digest('text')
OTHER_TEXT = _digest('other text')
META = _digest('meta')
OTHER_META = _digest('other meta')

_EMPTY_ERROR = (
    'manifest is empty: planning an empty manifest deletes every managed entry in the range; '
    'pass allow_empty_manifest=true to confirm'
)
_REPEAT_PREFIX = 'manifest repeats source_id values (each may appear once): '


def _id(n: int) -> str:
    """Return a 32-character hex context id; a larger n sorts later, as a newer UUIDv7 does."""
    return f'0190{n:028x}'


def _item(source_id: str, *, text: str = TEXT, meta: str = META) -> SyncManifestItem:
    """Return an ordinary manifest item."""
    return SyncManifestItem(source_id=source_id, text_digest=text, meta_digest=meta)


def _keep(source_id: str) -> SyncManifestItem:
    """Return a keep item."""
    return SyncManifestItem(source_id=source_id, keep=True)


def _member(
    n: int,
    source_id: str,
    *,
    text: str | None = TEXT,
    meta: str = META,
    is_owner: bool = True,
) -> StoredMember:
    """Return a stored member whose context id is ``_id(n)``."""
    return StoredMember(context_id=_id(n), source_id=source_id, content_hash=text, meta_digest=meta, is_owner=is_owner)


def _update(index: int, n: int, change: Change) -> SyncUpdateDict:
    """Return the planned update of manifest item ``index`` onto member ``n``."""
    return {'index': index, 'context_id': _id(n), 'change': change}


def _delete(n: int, source_id: str, reason: Reason) -> SyncDeleteDict:
    """Return the planned delete of member ``n``."""
    return {'context_id': _id(n), 'source_id': source_id, 'reason': reason}


def _plan(
    *,
    creates: list[int] | None = None,
    updates: list[SyncUpdateDict] | None = None,
    deletes: list[SyncDeleteDict] | None = None,
    unchanged: int = 0,
    kept: int = 0,
    foreign: int = 0,
) -> SyncPlan:
    """Return an expected plan, every omitted part empty or zero."""
    return SyncPlan(
        creates=creates or [],
        updates=updates or [],
        deletes=deletes or [],
        unchanged_count=unchanged,
        kept_count=kept,
        foreign_count=foreign,
    )


def _block(**overrides: object) -> dict[str, object]:
    """Return a valid stored ingest block with the given fields replaced."""
    block: dict[str, object] = {
        'scope': 'kb:handbook',
        'source_id': 'a.md',
        'meta_digest': META,
        'synced_at': '2026-10-18T12:00:00Z',
    }
    block.update(overrides)
    return block


class TestStoredMember:
    """A stored row becomes a member only when its block is valid."""

    def test_valid_block(self) -> None:
        """The member carries the block's source id and meta digest and the row's id, hash and ownership."""
        member = stored_member(_block(source_id='guides/setup.md'), context_id=_id(1), content_hash=TEXT, is_owner=True)

        assert member == StoredMember(
            context_id=_id(1),
            source_id='guides/setup.md',
            content_hash=TEXT,
            meta_digest=META,
            is_owner=True,
        )

    def test_null_hash_and_foreign_row_kept_as_given(self) -> None:
        """A row without a stored hash, owned by someone else, is still a member."""
        member = stored_member(_block(), context_id=_id(2), content_hash=None, is_owner=False)

        assert member is not None
        assert member.content_hash is None
        assert member.is_owner is False

    def test_scope_is_not_judged_here(self) -> None:
        """Membership by scope is decided by the read that produced the row, not by the diff."""
        member = stored_member(_block(scope='kb:elsewhere'), context_id=_id(1), content_hash=TEXT, is_owner=True)

        assert member is not None

    @pytest.mark.parametrize(
        'block',
        [
            None,
            'a.md',
            ['a.md'],
            {},
            {'scope': 'kb:handbook', 'source_id': 'a.md', 'meta_digest': META},
            _block(extra='value'),
            _block(source_id=''),
            _block(source_id='x' * (MAX_SOURCE_ID_LENGTH + 1)),
            _block(source_id=7),
            _block(meta_digest=META.upper()),
            _block(synced_at='2026-02-30T12:00:00Z'),
        ],
        ids=[
            'null',
            'string',
            'list',
            'empty-object',
            'missing-key',
            'extra-key',
            'empty-source-id',
            'source-id-too-long',
            'non-string-source-id',
            'uppercase-digest',
            'impossible-date',
        ],
    )
    def test_invalid_block_skipped(self, block: object) -> None:
        """A row whose block fails validation is skipped rather than planned."""
        assert stored_member(block, context_id=_id(1), content_hash=TEXT, is_owner=True) is None


class TestManifestError:
    """Argument errors are reported before any query, in a fixed order."""

    def test_valid_manifest_without_bounds(self) -> None:
        """Distinct source ids and no range: nothing to report."""
        manifest = [_item('a.md'), _keep('b.md')]

        assert manifest_error(manifest, source_id_after=None, source_id_through=None, allow_empty=False) is None

    def test_valid_manifest_inside_bounds(self) -> None:
        """Items strictly after the lower bound and up to the upper bound are inside."""
        manifest = [_item('b.md'), _item('c.md')]

        assert manifest_error(manifest, source_id_after='a.md', source_id_through='c.md', allow_empty=False) is None

    @pytest.mark.parametrize(('after', 'through'), [('b', 'b'), ('c', 'b')], ids=['equal', 'reversed'])
    def test_bounds_out_of_order(self, after: str, through: str) -> None:
        """The lower bound must sort strictly before the upper bound."""
        error = manifest_error([], source_id_after=after, source_id_through=through, allow_empty=False)

        assert error == 'source_id_after must sort before source_id_through'

    def test_bounds_compare_by_code_point(self) -> None:
        """'Z' sorts before 'a' by code point, so the range is ordered."""
        manifest = [_item('a')]

        assert manifest_error(manifest, source_id_after='Z', source_id_through='a', allow_empty=False) is None

    def test_lower_bound_alone(self) -> None:
        """The last slice has only a lower bound; items after it are inside."""
        assert manifest_error([_item('c')], source_id_after='b', source_id_through=None, allow_empty=False) is None

    def test_upper_bound_alone(self) -> None:
        """The first slice has only an upper bound; items up to it are inside."""
        assert manifest_error([_item('a'), _item('b')], source_id_after=None, source_id_through='b', allow_empty=False) is None

    def test_empty_manifest_refused_without_opt_in(self) -> None:
        """An empty manifest would delete every managed entry of the range."""
        assert manifest_error([], source_id_after=None, source_id_through=None, allow_empty=False) == _EMPTY_ERROR

    def test_empty_manifest_accepted_with_opt_in(self) -> None:
        """The caller confirms the empty manifest explicitly."""
        assert manifest_error([], source_id_after=None, source_id_through=None, allow_empty=True) is None

    def test_bounds_error_precedes_empty_error(self) -> None:
        """The bounds are checked first."""
        error = manifest_error([], source_id_after='b', source_id_through='a', allow_empty=False)

        assert error == 'source_id_after must sort before source_id_through'

    def test_repeated_source_ids_named_in_manifest_order(self) -> None:
        """Each repeated value is named once, in the order it first appears."""
        manifest = [_item('b'), _item('a'), _keep('a'), _item('b'), _item('c'), _item('a')]

        error = manifest_error(manifest, source_id_after=None, source_id_through=None, allow_empty=False)

        assert error == _REPEAT_PREFIX + "'b', 'a'"

    def test_ten_repeated_values_named_without_remainder(self) -> None:
        """Ten repeated values are all named."""
        names = [f's{n:02d}' for n in range(10)]
        manifest = [_item(name) for name in names] + [_item(name) for name in names]

        error = manifest_error(manifest, source_id_after=None, source_id_through=None, allow_empty=False)

        assert error == _REPEAT_PREFIX + ', '.join(repr(name) for name in names)

    def test_more_than_ten_repeated_values_counted(self) -> None:
        """The first ten repeated values are named and the rest counted."""
        names = [f's{n:02d}' for n in range(13)]
        manifest = [_item(name) for name in names] + [_item(name) for name in reversed(names)]

        error = manifest_error(manifest, source_id_after=None, source_id_through=None, allow_empty=False)

        assert error == _REPEAT_PREFIX + ', '.join(repr(name) for name in names[:10]) + ' and 3 more'

    def test_item_at_the_exclusive_lower_bound_refused(self) -> None:
        """The lower bound itself belongs to the previous range."""
        manifest = [_item('b'), _item('a')]

        error = manifest_error(manifest, source_id_after='a', source_id_through=None, allow_empty=False)

        assert error == "manifest item 1 has source_id 'a', outside the planned range (source_id_after, source_id_through]"

    def test_item_past_the_inclusive_upper_bound_refused(self) -> None:
        """An item after the upper bound belongs to the next range."""
        manifest = [_item('a'), _item('c'), _item('d')]

        error = manifest_error(manifest, source_id_after=None, source_id_through='b', allow_empty=False)

        assert error == "manifest item 1 has source_id 'c', outside the planned range (source_id_after, source_id_through]"

    def test_keep_item_outside_the_range_refused(self) -> None:
        """A keep item is a member of the manifest like any other item."""
        manifest = [_keep('z')]

        error = manifest_error(manifest, source_id_after='a', source_id_through='m', allow_empty=False)

        assert error == "manifest item 0 has source_id 'z', outside the planned range (source_id_after, source_id_through]"

    def test_repeat_error_precedes_range_error(self) -> None:
        """Repeated values are reported before items outside the range."""
        manifest = [_item('z'), _item('z')]

        error = manifest_error(manifest, source_id_after=None, source_id_through='m', allow_empty=False)

        assert error == _REPEAT_PREFIX + "'z'"


class TestCreatesAndChanges:
    """One source per item: create, leave unchanged, or update what differs."""

    def test_new_source_created(self) -> None:
        """An item with no stored member is created by its manifest index."""
        plan = compute_sync_plan([_item('a.md'), _item('b.md')], [_member(1, 'a.md')])

        assert plan == _plan(creates=[1], unchanged=1)

    def test_unknown_thread_creates_every_item(self) -> None:
        """Without stored members every ordinary item is created, keep items only counted."""
        plan = compute_sync_plan([_item('a.md'), _keep('b.md'), _item('c.md')], [])

        assert plan == _plan(creates=[0, 2], kept=1)

    def test_unchanged(self) -> None:
        """Equal text and metadata digests plan nothing."""
        plan = compute_sync_plan([_item('a.md')], [_member(1, 'a.md')])

        assert plan == _plan(unchanged=1)

    def test_text_changed(self) -> None:
        """A different text digest with equal metadata updates the text."""
        plan = compute_sync_plan([_item('a.md', text=OTHER_TEXT)], [_member(1, 'a.md')])

        assert plan == _plan(updates=[_update(0, 1, 'text')])

    def test_null_hash_counts_as_changed_text(self) -> None:
        """A row without a stored hash cannot prove its text matches."""
        plan = compute_sync_plan([_item('a.md')], [_member(1, 'a.md', text=None)])

        assert plan == _plan(updates=[_update(0, 1, 'text')])

    def test_null_hash_with_changed_metadata(self) -> None:
        """No stored hash and different metadata update both."""
        plan = compute_sync_plan([_item('a.md', meta=OTHER_META)], [_member(1, 'a.md', text=None)])

        assert plan == _plan(updates=[_update(0, 1, 'both')])

    def test_metadata_changed(self) -> None:
        """A different metadata digest with equal text updates the metadata."""
        plan = compute_sync_plan([_item('a.md', meta=OTHER_META)], [_member(1, 'a.md')])

        assert plan == _plan(updates=[_update(0, 1, 'metadata')])

    def test_both_changed(self) -> None:
        """Both digests different update both."""
        plan = compute_sync_plan([_item('a.md', text=OTHER_TEXT, meta=OTHER_META)], [_member(1, 'a.md')])

        assert plan == _plan(updates=[_update(0, 1, 'both')])

    def test_updates_follow_manifest_order(self) -> None:
        """Updates are listed by manifest index, whatever order the members arrive in."""
        manifest = [_item('b.md', text=OTHER_TEXT), _item('a.md', meta=OTHER_META), _item('c.md')]
        members = [_member(3, 'c.md'), _member(1, 'a.md'), _member(2, 'b.md')]

        plan = compute_sync_plan(manifest, members)

        assert plan == _plan(updates=[_update(0, 2, 'text'), _update(1, 1, 'metadata')], unchanged=1)

    def test_members_may_be_any_iterable(self) -> None:
        """The members are read once, so a generator works."""
        plan = compute_sync_plan([_item('a.md')], (member for member in [_member(1, 'a.md')]))

        assert plan == _plan(unchanged=1)


class TestSurvivorOrder:
    """Duplicates of one source keep one survivor, chosen by a total order."""

    def test_newest_of_two_exact_matches_survives(self) -> None:
        """Two identical copies keep the newest and delete the other as a duplicate."""
        plan = compute_sync_plan([_item('a.md')], [_member(1, 'a.md'), _member(2, 'a.md')])

        assert plan == _plan(deletes=[_delete(1, 'a.md', 'duplicate')], unchanged=1)

    def test_text_match_beats_a_newer_non_match(self) -> None:
        """A copy whose text matches needs only a metadata update, so it outranks a newer copy matching nothing."""
        members = [_member(1, 'a.md', meta=OTHER_META), _member(2, 'a.md', text=OTHER_TEXT, meta=OTHER_META)]

        plan = compute_sync_plan([_item('a.md')], members)

        assert plan == _plan(updates=[_update(0, 1, 'metadata')], deletes=[_delete(2, 'a.md', 'duplicate')])

    def test_text_match_beats_a_newer_metadata_match(self) -> None:
        """Matching text outranks matching metadata."""
        members = [_member(1, 'a.md', meta=OTHER_META), _member(2, 'a.md', text=OTHER_TEXT)]

        plan = compute_sync_plan([_item('a.md')], members)

        assert plan == _plan(updates=[_update(0, 1, 'metadata')], deletes=[_delete(2, 'a.md', 'duplicate')])

    def test_metadata_match_beats_a_newer_non_match(self) -> None:
        """Without a text match, matching metadata decides."""
        members = [_member(1, 'a.md', text=OTHER_TEXT), _member(2, 'a.md', text=OTHER_TEXT, meta=OTHER_META)]

        plan = compute_sync_plan([_item('a.md')], members)

        assert plan == _plan(updates=[_update(0, 1, 'text')], deletes=[_delete(2, 'a.md', 'duplicate')])

    def test_exact_match_beats_a_newer_text_match(self) -> None:
        """A copy matching both digests outranks a newer copy matching the text only."""
        members = [_member(1, 'a.md'), _member(2, 'a.md', meta=OTHER_META)]

        plan = compute_sync_plan([_item('a.md')], members)

        assert plan == _plan(deletes=[_delete(2, 'a.md', 'duplicate')], unchanged=1)

    def test_newest_of_non_matches_survives(self) -> None:
        """When no copy matches anything, the newest survives and takes both changes."""
        members = [_member(2, 'a.md', text=None), _member(1, 'a.md', text=None)]

        plan = compute_sync_plan([_item('a.md', meta=OTHER_META)], members)

        assert plan == _plan(updates=[_update(0, 2, 'both')], deletes=[_delete(1, 'a.md', 'duplicate')])

    def test_every_extra_copy_deleted(self) -> None:
        """Three copies keep one and delete two, in context id order."""
        members = [_member(3, 'a.md'), _member(5, 'a.md'), _member(4, 'a.md')]

        plan = compute_sync_plan([_item('a.md')], members)

        assert plan == _plan(deletes=[_delete(3, 'a.md', 'duplicate'), _delete(4, 'a.md', 'duplicate')], unchanged=1)

    def test_every_member_order_gives_the_same_plan(self) -> None:
        """Any database row order yields the same survivors, updates, deletes and counts."""
        manifest = [_item('a.md'), _item('b.md', meta=OTHER_META), _keep('c.md'), _item('e.md')]
        members = [
            _member(1, 'a.md', meta=OTHER_META),
            _member(2, 'a.md', text=OTHER_TEXT, meta=OTHER_META),
            _member(3, 'b.md', text=None),
            _member(4, 'b.md'),
            _member(5, 'c.md'),
            _member(6, 'd.md'),
            _member(7, 'e.md', is_owner=False),
        ]
        expected = _plan(
            creates=[3],
            updates=[_update(0, 1, 'metadata'), _update(1, 4, 'metadata')],
            deletes=[_delete(2, 'a.md', 'duplicate'), _delete(3, 'b.md', 'duplicate'), _delete(6, 'd.md', 'absent')],
            kept=1,
            foreign=1,
        )

        for order in itertools.permutations(members):
            assert compute_sync_plan(manifest, order) == expected


class TestAbsentSources:
    """Owned sources the manifest does not name are deleted."""

    def test_absent_source_deleted(self) -> None:
        """A stored source missing from the manifest is deleted as absent."""
        plan = compute_sync_plan([_item('a.md')], [_member(1, 'a.md'), _member(2, 'gone.md')])

        assert plan == _plan(deletes=[_delete(2, 'gone.md', 'absent')], unchanged=1)

    def test_every_copy_of_an_absent_source_deleted(self) -> None:
        """Duplicated copies of an absent source all go."""
        plan = compute_sync_plan([], [_member(2, 'gone.md'), _member(1, 'gone.md')])

        assert plan == _plan(deletes=[_delete(1, 'gone.md', 'absent'), _delete(2, 'gone.md', 'absent')])


class TestKeepItems:
    """A keep item marks a source the client could not read: nothing about it changes."""

    def test_kept_source_neither_updated_nor_deleted(self) -> None:
        """A keep item over a stored member plans nothing for it."""
        plan = compute_sync_plan([_keep('a.md')], [_member(1, 'a.md', text=OTHER_TEXT, meta=OTHER_META)])

        assert plan == _plan(kept=1)

    def test_kept_source_duplicates_untouched(self) -> None:
        """Without the content the client cannot pick a survivor, so no copy is deleted."""
        plan = compute_sync_plan([_keep('a.md')], [_member(1, 'a.md'), _member(2, 'a.md', text=None)])

        assert plan == _plan(kept=1)

    def test_kept_unknown_source_creates_nothing(self) -> None:
        """A keep item for a source with no stored member is only counted."""
        plan = compute_sync_plan([_keep('new.md')], [])

        assert plan == _plan(kept=1)

    def test_kept_count_counts_items(self) -> None:
        """kept_count counts keep items, however many members they cover."""
        manifest = [_keep('a.md'), _keep('b.md'), _item('c.md')]
        members = [_member(1, 'a.md'), _member(2, 'a.md'), _member(3, 'c.md')]

        plan = compute_sync_plan(manifest, members)

        assert plan == _plan(unchanged=1, kept=2)


class TestForeignMembers:
    """Readable members owned by another principal are counted, never planned."""

    def test_foreign_member_is_no_survivor(self) -> None:
        """An item whose only member is foreign is created for the caller."""
        plan = compute_sync_plan([_item('a.md')], [_member(1, 'a.md', is_owner=False)])

        assert plan == _plan(creates=[0], foreign=1)

    def test_foreign_absent_source_not_deleted(self) -> None:
        """Another principal's source missing from the manifest stays."""
        plan = compute_sync_plan([_item('a.md')], [_member(1, 'a.md'), _member(2, 'theirs.md', is_owner=False)])

        assert plan == _plan(unchanged=1, foreign=1)

    def test_foreign_copy_is_no_duplicate(self) -> None:
        """A foreign copy beside the caller's own copy is neither the survivor nor deleted."""
        members = [_member(1, 'a.md'), _member(2, 'a.md', is_owner=False), _member(3, 'a.md', is_owner=False)]

        plan = compute_sync_plan([_item('a.md', text=OTHER_TEXT)], members)

        assert plan == _plan(updates=[_update(0, 1, 'text')], foreign=2)

    def test_every_foreign_member_counted(self) -> None:
        """foreign_count counts every foreign member, whatever the manifest says about its source."""
        manifest = [_keep('a.md'), _item('b.md')]
        members = [
            _member(1, 'a.md', is_owner=False),
            _member(2, 'b.md', is_owner=False),
            _member(3, 'c.md', is_owner=False),
            _member(4, 'b.md'),
        ]

        plan = compute_sync_plan(manifest, members)

        assert plan == _plan(unchanged=1, kept=1, foreign=3)


class TestDeleteOrdering:
    """Every planned delete is listed, sorted by source id and then context id."""

    def test_deletes_sorted_by_source_then_id(self) -> None:
        """Absent and duplicate deletes interleave in one code-point order."""
        manifest = [_item('b.md'), _item('a.md')]
        members = [
            _member(9, 'zz.md'),
            _member(2, 'b.md'),
            _member(8, 'b.md'),
            _member(7, 'B.md'),
            _member(4, 'a.md'),
            _member(6, 'a.md'),
            _member(5, 'a.md'),
            _member(1, 'é.md'),
        ]

        plan = compute_sync_plan(manifest, members)

        assert plan.deletes == [
            _delete(7, 'B.md', 'absent'),
            _delete(4, 'a.md', 'duplicate'),
            _delete(5, 'a.md', 'duplicate'),
            _delete(2, 'b.md', 'duplicate'),
            _delete(9, 'zz.md', 'absent'),
            _delete(1, 'é.md', 'absent'),
        ]

    def test_plan_lists_every_delete(self) -> None:
        """The diff itself is uncapped; the tool slices the listed deletes."""
        members = [_member(n, f'gone-{n:03d}.md') for n in range(1, 151)]

        plan = compute_sync_plan([], members)

        assert len(plan.deletes) == 150
        assert [delete['context_id'] for delete in plan.deletes] == [_id(n) for n in range(1, 151)]
