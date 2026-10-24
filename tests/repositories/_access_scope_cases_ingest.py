"""Two-principal cases of the reads of ingest-managed entries (S8, X3).

An entry is a member of a corpus thread when its ``metadata.ingest`` block carries a source
id and names that thread as its scope. The block never takes part in an access decision: a
member is read, flagged as owned and paged exactly as the access model decides for its row.

Every case seeds a corpus that mirrors the seed: one member per seed row, with the row's
owner, visibility and grants, under a source id that sorts in seed order. Public decoys of
alice sit beside it: an unmanaged row of the corpus thread, a row of the corpus thread whose
block names another scope, and a row of another thread whose block names the corpus.

- S8: ``ingest_rows_in_range`` returns exactly the members the caller may read (variant
  ``members``, read-decided, so the database alone must reproduce it), inside the requested
  range when bounds are given, and flags the ones the caller owns (variant ``owner-flag``,
  which the application computes in the select list). A block that names another principal
  or claims an owner, a visibility or groups changes nothing anyone reads or owns.
- X3: ``owned_ingest_page`` lists the members the caller owns in source-id order (variant
  ``members``) and pages through them with a one-row lookahead (variant ``page-fill``):
  members of other principals, readable or hidden, sit between and after the caller's own
  and never take a page slot or extend a page.
"""

import operator

from app.access_scope import Scope
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SEED_ROWS
from tests.repositories._access_scope_cases import TEAM_GROUP
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Grant
from tests.repositories._access_scope_cases import Invoker
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import Setup
from tests.repositories._access_scope_cases import Visibility
from tests.repositories._access_scope_cases import per_scope

type MemberObservable = tuple[str, ...]
type OwnerFlagObservable = tuple[tuple[str, bool], ...]
type PagesObservable = tuple[tuple[str, ...], ...]

# One corpus row: label, owner, visibility, source id, grants and the extra keys of its block.
type CorpusRow = tuple[str, str, Visibility, str, tuple[Grant, ...], dict[str, object]]

CORPUS = 'kb:access-seams'
OTHER_CORPUS = 'kb:access-seams-other'
_DIGEST = '0' * 64
_SYNCED_AT = '2026-10-18T12:00:00Z'

# Rows of the owned page walk per call; the walk reads one row more to tell whether a next page exists.
PAGE_SIZE = 2


def _member_label(seed_label: str) -> str:
    """The label of the corpus member that mirrors a seed row."""
    return f'member_{seed_label}'


def _block(source_id: str, *, scope: str = CORPUS, claims: dict[str, object] | None = None) -> dict[str, object]:
    """Return an ingest block for ``source_id``, with any extra keys a client wrote into it."""
    return {'scope': scope, 'source_id': source_id, 'meta_digest': _DIGEST, 'synced_at': _SYNCED_AT, **(claims or {})}


# The members mirroring the seed, under source ids that sort in seed order.
_MIRROR: tuple[CorpusRow, ...] = tuple(
    (_member_label(row.label), row.owner, row.visibility, f'{position:02d}-{row.label}.md', row.grants, {})
    for position, row in enumerate(SEED_ROWS)
)

# Range bounds of the filtered variant: after the member at position 1, through the member at position 5.
_RANGE_AFTER = _MIRROR[1][3]
_RANGE_THROUGH = _MIRROR[5][3]
_IN_RANGE = frozenset(row[0] for row in _MIRROR[2:6])

# Members whose blocks claim access they do not have: another owner, public visibility, groups.
_ADVERSARIAL: tuple[CorpusRow, ...] = (
    (
        'adversarial_alice_public_claims_bob', 'alice', 'public', '08-adversarial-alice-public.md', (),
        {'owner_id': 'bob', 'principal_id': 'bob'},
    ),
    (
        'adversarial_alice_private_claims_public', 'alice', 'private', '09-adversarial-alice-private.md', (),
        {'visibility': 'public', 'owner_id': 'carol'},
    ),
    (
        'adversarial_bob_private_claims_alice', 'bob', 'private', '10-adversarial-bob-private.md', (),
        {'owner_id': 'alice', 'groups': [TEAM_GROUP]},
    ),
)

# What each scope reads of the adversarial members: the rows' real owners and visibilities decide.
_ADVERSARIAL_READABLE: dict[str, frozenset[str]] = {
    'alice': frozenset({'adversarial_alice_public_claims_bob', 'adversarial_alice_private_claims_public'}),
    'bob': frozenset({'adversarial_alice_public_claims_bob', 'adversarial_bob_private_claims_alice'}),
    'carol': frozenset({'adversarial_alice_public_claims_bob'}),
    'dave': frozenset({'adversarial_alice_public_claims_bob'}),
    'anonymous': frozenset({'adversarial_alice_public_claims_bob'}),
    'system': frozenset(row[0] for row in _ADVERSARIAL),
}

# Members of the page-fill variant placed between and after the seed mirror: bob's rows split
# alice's run, and a hidden row of alice follows bob's last one.
_FILL: tuple[CorpusRow, ...] = (
    ('fill_bob_private', 'bob', 'private', '02z-fill-bob-private.md', (), {}),
    ('fill_bob_public', 'bob', 'public', '05z-fill-bob-public.md', (), {}),
    ('fill_alice_tail', 'alice', 'private', '11-fill-alice-tail.md', (), {}),
)


async def _add_rows(db: ScopedDb, rows: tuple[CorpusRow, ...]) -> None:
    """Store corpus members in the corpus thread with their grants and blocks."""
    for label, owner, visibility, source_id, grants, claims in rows:
        await db.add_entry(
            label, owner=owner, visibility=visibility, thread_id=CORPUS,
            metadata={'label': label, 'ingest': _block(source_id, claims=claims)}, grants=grants,
        )


async def _add_decoys(db: ScopedDb) -> None:
    """Store alice's public decoys, which every scope may read and none may count as a member."""
    await db.add_entry('decoy_unmanaged', owner='alice', visibility='public', thread_id=CORPUS)
    await db.add_entry(
        'decoy_other_scope', owner='alice', visibility='public', thread_id=CORPUS,
        metadata={'ingest': _block('03-decoy-other-scope.md', scope=OTHER_CORPUS)},
    )
    await db.add_entry(
        'decoy_other_thread', owner='alice', visibility='public', thread_id=OTHER_CORPUS,
        metadata={'ingest': _block('03-decoy-other-thread.md')},
    )


def _setup(*extra: tuple[CorpusRow, ...]) -> Setup:
    """Build a setup storing the seed mirror, the ``extra`` members and the decoys."""

    async def setup(db: ScopedDb) -> None:
        await _add_rows(db, _MIRROR)
        for rows in extra:
            await _add_rows(db, rows)
        await _add_decoys(db)

    return setup


async def _range_rows(
    db: ScopedDb, scope: Scope, *, after: str | None = None, through: str | None = None,
) -> list[tuple[str, bool]]:
    """Return ``(label, is_owner)`` of every member the range read returns."""
    rows = await db.repos.context.ingest_rows_in_range(
        scope=scope, thread_id=CORPUS, source_id_after=after, source_id_through=through,
    )
    return list(zip(db.labels_of(row.context_id for row in rows), (row.is_owner for row in rows), strict=True))


def _read_members(*, after: str | None = None, through: str | None = None) -> Invoker:
    """Build an invoker returning the sorted labels of the members the range read returns."""

    async def invoke(db: ScopedDb, scope: Scope) -> MemberObservable:
        return tuple(sorted(label for label, _is_owner in await _range_rows(db, scope, after=after, through=through)))

    return invoke


async def _read_owner_flags(db: ScopedDb, scope: Scope) -> OwnerFlagObservable:
    """Return the sorted ``(label, is_owner)`` pairs of every member the range read returns."""
    return tuple(sorted(await _range_rows(db, scope)))


def _readable_members(scope_name: str) -> MemberObservable:
    """The mirror members of the seed rows the scope reads, sorted."""
    return tuple(sorted(_member_label(label) for label in READABLE[scope_name]))


def _readable_in_range(scope_name: str) -> MemberObservable:
    """The mirror members the scope reads inside the range of the filtered variant, sorted."""
    return tuple(label for label in _readable_members(scope_name) if label in _IN_RANGE)


def _readable_with_adversarial(scope_name: str) -> MemberObservable:
    """The mirror members and the adversarial members the scope reads, sorted."""
    return tuple(sorted({*_readable_members(scope_name), *_ADVERSARIAL_READABLE[scope_name]}))


def _owns(scope_name: str, owner: str) -> bool:
    """Whether the scope owns a row of ``owner``; the system scope's empty predicate admits every row."""
    return scope_name in ('system', owner)


def _owner_flags_of(scope_name: str, rows: tuple[CorpusRow, ...]) -> OwnerFlagObservable:
    """The sorted ``(label, is_owner)`` pairs of the ``rows`` the scope reads."""
    readable = {*_readable_members(scope_name), *_ADVERSARIAL_READABLE[scope_name]}
    return tuple(sorted((row[0], _owns(scope_name, row[1])) for row in rows if row[0] in readable))


def _mirror_owner_flags(scope_name: str) -> OwnerFlagObservable:
    """The owner flag of every mirror member the scope reads."""
    return _owner_flags_of(scope_name, _MIRROR)


def _adversarial_owner_flags(scope_name: str) -> OwnerFlagObservable:
    """The owner flag of every mirror and adversarial member the scope reads."""
    return _owner_flags_of(scope_name, (*_MIRROR, *_ADVERSARIAL))


async def _owned_members(db: ScopedDb, scope: Scope) -> MemberObservable:
    """Return the labels of one owned page large enough to hold every member, in the order it reads them."""
    rows = await db.repos.context.owned_ingest_page(scope=scope, thread_id=CORPUS, after=None, limit=50)
    return tuple(db.labels_of(row.context_id for row in rows))


async def _owned_pages(db: ScopedDb, scope: Scope) -> PagesObservable:
    """Walk the owned pages of :data:`PAGE_SIZE` rows with a one-row lookahead, as the listing tool pages."""
    pages: list[tuple[str, ...]] = []
    after: tuple[str, str] | None = None
    while True:
        rows = await db.repos.context.owned_ingest_page(scope=scope, thread_id=CORPUS, after=after, limit=PAGE_SIZE + 1)
        window = rows[:PAGE_SIZE]
        pages.append(tuple(db.labels_of(row.context_id for row in window)))
        if len(rows) <= PAGE_SIZE:
            return tuple(pages)
        after = (window[-1].source_id, window[-1].context_id)


def _owned_in_order(scope_name: str, rows: tuple[CorpusRow, ...]) -> tuple[str, ...]:
    """The labels of ``rows`` the scope owns, in source-id order."""
    return tuple(row[0] for row in sorted(rows, key=operator.itemgetter(3)) if _owns(scope_name, row[1]))


def _owned_mirror(scope_name: str) -> MemberObservable:
    """The mirror members the scope owns, in source-id order."""
    return _owned_in_order(scope_name, _MIRROR)


def _expected_pages(scope_name: str) -> PagesObservable:
    """The owned members of the mirror and the fill, cut into pages; a scope owning nothing reads one empty page."""
    owned = _owned_in_order(scope_name, (*_MIRROR, *_FILL))
    if not owned:
        return ((),)
    return tuple(owned[start:start + PAGE_SIZE] for start in range(0, len(owned), PAGE_SIZE))


def _members_case(layout: Layout, suffix: str) -> AccessCase:
    """The unfiltered S8 membership case on one layout."""
    variant = '-'.join(part for part in ('members', suffix) if part)
    return AccessCase('S8', _read_members(), per_scope(_readable_members), layout=layout, variant=variant, setup=_setup())


def _owned_members_case(layout: Layout, suffix: str) -> AccessCase:
    """The unpaged X3 listing case on one layout."""
    return AccessCase('X3', _owned_members, per_scope(_owned_mirror), layout=layout, variant=suffix, setup=_setup())


# The S8 cases: membership in every arrangement (read-decided) and the owner flag (computed by the application).
INGEST_CASES: tuple[AccessCase, ...] = (
    _members_case('compressed', ''),
    _members_case('fp32', 'fp32'),
    AccessCase(
        'S8', _read_members(after=_RANGE_AFTER, through=_RANGE_THROUGH), per_scope(_readable_in_range),
        filtered=True, variant='members', setup=_setup(),
    ),
    AccessCase(
        'S8', _read_members(), per_scope(_readable_with_adversarial), variant='members-adversarial',
        setup=_setup(_ADVERSARIAL),
    ),
    AccessCase(
        'S8', _read_owner_flags, per_scope(_mirror_owner_flags), variant='owner-flag', setup=_setup(),
        reports_write_access=True,
    ),
    AccessCase(
        'S8', _read_owner_flags, per_scope(_adversarial_owner_flags), variant='owner-flag-adversarial',
        setup=_setup(_ADVERSARIAL), reports_write_access=True,
    ),
)

# The X3 cases: the owned listing and its page walk.
INGEST_OWNER_CASES: tuple[AccessCase, ...] = (
    _owned_members_case('compressed', ''),
    _owned_members_case('fp32', 'fp32'),
    AccessCase('X3', _owned_pages, per_scope(_expected_pages), variant='page-fill', setup=_setup(_FILL)),
)
