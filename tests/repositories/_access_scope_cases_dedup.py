"""Two-principal cases of the deduplicating store and its read-only pre-check (K1-K4).

The dedup candidate is the latest entry of the thread and source that the caller may read.
A store merges into it only when the caller owns it and the text matches, and an
opposite-source entry the caller may read after the candidate marks a new conversational
turn, which suppresses the merge. Each case adds its own thread in its setup, so the seed
takes no part. Store cases observe the whole thread afterwards through raw SQL: every row's
label, owner and version in id order, with ``new`` for the row the store inserted.

- K1: the candidate's owner decides the merge. A readable latest row owned by another
  principal, even one write-granted to the caller, is never merged into; a hidden latest
  row is skipped and the caller's own older row is merged into.
- K2: the opposite-source turn suppresses the merge only when the caller may read it.
- K3: the pre-check reports a candidate (and its summary for reuse) only when the caller
  owns it, under the same candidate and turn rules as the store.
- K4: two principals writing the same source into one thread: a re-sent text after another
  principal's readable row lands as a new row after it, keeping the chronological order.
"""

from app.access_scope import AccessScope
from app.access_scope import Scope
from tests.repositories._access_scope_cases import SEED_SOURCE
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Grant
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import Source

type ThreadRow = tuple[str, str, int]
type StoreObservable = tuple[bool, tuple[ThreadRow, ...]]
type CandidateObservable = tuple[str, str | None] | None

# The source of a conversational turn that answers SEED_SOURCE entries.
TURN_SOURCE: Source = 'user'

K1_THREAD = 'dedup-k1'
K1_TEXT = 'K1 retransmitted text'
K2_THREAD = 'dedup-k2'
K2_TEXT = 'K2 repeated agent text'
K3_THREAD = 'dedup-k3'
K3_TEXT = 'K3 checked text'
K4_THREAD = 'dedup-k4'
K4_TEXT = 'K4 text alice sends twice'


def _request_scope(scope: Scope) -> AccessScope:
    """Return the scope of the request a store runs under; a store never runs as the system scope."""
    assert isinstance(scope, AccessScope)
    return scope


async def _thread_rows(db: ScopedDb, thread_id: str) -> tuple[ThreadRow, ...]:
    """Return every row of the thread in id order as ``(label, owner, version)``, ``new`` for unlabeled rows."""
    rows = await db.fetch_all(
        'SELECT id, owner_id, version FROM context_entries WHERE thread_id = ? ORDER BY id', (thread_id,),
    )
    labeled = set(db.ids.values())
    return tuple(
        (db.labels_of([str(row[0])])[0] if str(row[0]) in labeled else 'new', str(row[1]), int(str(row[2])))
        for row in rows
    )


async def _store_as(db: ScopedDb, scope: Scope, thread_id: str, text: str) -> StoreObservable:
    """Store ``text`` in ``thread_id`` as the scope and return whether it merged plus the thread afterwards."""
    _, merged = await db.repos.context.store_with_deduplication(
        thread_id=thread_id,
        source=SEED_SOURCE,
        content_type='text',
        text_content=text,
        scope=_request_scope(scope),
        visibility='private',
    )
    return merged, await _thread_rows(db, thread_id)


async def _resend_k1(db: ScopedDb, scope: Scope) -> StoreObservable:
    """Re-send the K1 text as the scope."""
    return await _store_as(db, scope, K1_THREAD, K1_TEXT)


async def _k1_alice_public_latest(db: ScopedDb) -> None:
    """Alice's public row carrying the K1 text is the latest of the thread."""
    await db.add_entry('k1_alice', owner='alice', visibility='public', thread_id=K1_THREAD, text=K1_TEXT)


async def _k1_alice_write_granted_latest(db: ScopedDb) -> None:
    """Alice's private row carrying the K1 text, write-granted to bob, is the latest of the thread."""
    await db.add_entry(
        'k1_alice', owner='alice', visibility='private', thread_id=K1_THREAD, text=K1_TEXT,
        grants=(Grant('user', 'bob', 'write'),),
    )


async def _k1_hidden_row_after_bob(db: ScopedDb) -> None:
    """Bob's private row carrying the K1 text, followed by a private alice row with other text."""
    await db.add_entry('k1_bob', owner='bob', visibility='private', thread_id=K1_THREAD, text=K1_TEXT)
    await db.add_entry('k1_alice_draft', owner='alice', visibility='private', thread_id=K1_THREAD, text='K1 alice draft')


def _k1_foreign_latest_expected() -> dict[str, StoreObservable]:
    """Alice merges into her row; everyone else inserts a row of their own and leaves hers unchanged."""
    expected: dict[str, StoreObservable] = {'alice': (True, (('k1_alice', 'alice', 1),))}
    for name in ('bob', 'carol', 'dave'):
        expected[name] = (False, (('k1_alice', 'alice', 0), ('new', name, 0)))
    return expected


async def _resend_k2(db: ScopedDb, scope: Scope) -> StoreObservable:
    """Re-send the K2 text as the scope."""
    return await _store_as(db, scope, K2_THREAD, K2_TEXT)


async def _add_k2_alice(db: ScopedDb) -> None:
    """Alice's public row carrying the K2 text."""
    await db.add_entry('k2_alice', owner='alice', visibility='public', thread_id=K2_THREAD, text=K2_TEXT)


async def _k2_public_turn(db: ScopedDb) -> None:
    """Alice's K2 row followed by bob's public opposite-source turn."""
    await _add_k2_alice(db)
    await db.add_entry(
        'k2_bob_turn', owner='bob', visibility='public', thread_id=K2_THREAD, source=TURN_SOURCE, text='K2 bob turn',
    )


async def _k2_granted_turn(db: ScopedDb) -> None:
    """Alice's K2 row followed by bob's private opposite-source turn, read-granted to alice."""
    await _add_k2_alice(db)
    await db.add_entry(
        'k2_bob_turn', owner='bob', visibility='private', thread_id=K2_THREAD, source=TURN_SOURCE, text='K2 bob turn',
        grants=(Grant('user', 'alice', 'read'),),
    )


async def _k2_hidden_turn(db: ScopedDb) -> None:
    """Alice's K2 row followed by bob's private opposite-source turn, hidden from alice."""
    await _add_k2_alice(db)
    await db.add_entry(
        'k2_bob_turn', owner='bob', visibility='private', thread_id=K2_THREAD, source=TURN_SOURCE, text='K2 bob turn',
    )


# Alice may read the turn: a new turn, so her re-send lands as a new row after it.
_K2_ALICE_NEW_TURN: StoreObservable = (False, (('k2_alice', 'alice', 0), ('k2_bob_turn', 'bob', 0), ('new', 'alice', 0)))
# Bob never owns the candidate, so his re-send always inserts.
_K2_BOB_INSERTS: StoreObservable = (False, (('k2_alice', 'alice', 0), ('k2_bob_turn', 'bob', 0), ('new', 'bob', 0)))


async def _check_k3(db: ScopedDb, scope: Scope) -> CandidateObservable:
    """Run the pre-check for the K3 text as the scope and return the candidate's label and summary."""
    candidate = await db.repos.context.check_latest_is_duplicate(
        thread_id=K3_THREAD, source=SEED_SOURCE, text_content=K3_TEXT, scope=_request_scope(scope),
    )
    if candidate is None:
        return None
    return db.labels_of([candidate.context_id])[0], candidate.summary


async def _add_k3_alice(db: ScopedDb) -> None:
    """Alice's public row carrying the K3 text and a summary."""
    await db.add_entry(
        'k3_alice', owner='alice', visibility='public', thread_id=K3_THREAD, text=K3_TEXT, summary='Summary of k3_alice',
    )


async def _k3_hidden_row_after_bob(db: ScopedDb) -> None:
    """Bob's private row carrying the K3 text, followed by a private alice row with other text."""
    await db.add_entry(
        'k3_bob', owner='bob', visibility='private', thread_id=K3_THREAD, text=K3_TEXT, summary='Summary of k3_bob',
    )
    await db.add_entry('k3_alice_draft', owner='alice', visibility='private', thread_id=K3_THREAD, text='K3 alice draft')


async def _k3_public_turn(db: ScopedDb) -> None:
    """Alice's K3 row followed by bob's public opposite-source turn."""
    await _add_k3_alice(db)
    await db.add_entry(
        'k3_bob_turn', owner='bob', visibility='public', thread_id=K3_THREAD, source=TURN_SOURCE, text='K3 bob turn',
    )


async def _k3_hidden_turn(db: ScopedDb) -> None:
    """Alice's K3 row followed by bob's private opposite-source turn, hidden from alice."""
    await _add_k3_alice(db)
    await db.add_entry(
        'k3_bob_turn', owner='bob', visibility='private', thread_id=K3_THREAD, source=TURN_SOURCE, text='K3 bob turn',
    )


async def _send_k4(db: ScopedDb, scope: Scope) -> StoreObservable:
    """Send the K4 text as the scope."""
    return await _store_as(db, scope, K4_THREAD, K4_TEXT)


async def _k4_bob_public_after_alice(db: ScopedDb) -> None:
    """Alice sends the K4 text, then bob sends other text into the same thread and source, both public."""
    await db.add_entry('k4_alice', owner='alice', visibility='public', thread_id=K4_THREAD, text=K4_TEXT)
    await db.add_entry('k4_bob', owner='bob', visibility='public', thread_id=K4_THREAD, text='K4 bob text')


async def _k4_bob_private_after_alice(db: ScopedDb) -> None:
    """Alice sends the K4 text, then bob sends other text into the same thread and source, privately."""
    await db.add_entry('k4_alice', owner='alice', visibility='public', thread_id=K4_THREAD, text=K4_TEXT)
    await db.add_entry('k4_bob', owner='bob', visibility='private', thread_id=K4_THREAD, text='K4 bob text')


# Bob's latest readable row is his own with other text, so he always inserts.
_K4_BOB_INSERTS: StoreObservable = (False, (('k4_alice', 'alice', 0), ('k4_bob', 'bob', 0), ('new', 'bob', 0)))

DEDUP_CASES: tuple[AccessCase, ...] = (
    AccessCase('K1', _resend_k1, _k1_foreign_latest_expected(), variant='readable-foreign', setup=_k1_alice_public_latest),
    AccessCase(
        'K1', _resend_k1,
        {
            'alice': (True, (('k1_alice', 'alice', 1),)),
            'bob': (False, (('k1_alice', 'alice', 0), ('new', 'bob', 0))),
        },
        variant='write-granted-foreign', setup=_k1_alice_write_granted_latest,
    ),
    AccessCase(
        'K1', _resend_k1,
        {
            'bob': (True, (('k1_bob', 'bob', 1), ('k1_alice_draft', 'alice', 0))),
            'alice': (False, (('k1_bob', 'bob', 0), ('k1_alice_draft', 'alice', 0), ('new', 'alice', 0))),
            'carol': (False, (('k1_bob', 'bob', 0), ('k1_alice_draft', 'alice', 0), ('new', 'carol', 0))),
        },
        variant='hidden-latest', setup=_k1_hidden_row_after_bob,
    ),
    AccessCase(
        'K2', _resend_k2, {'alice': _K2_ALICE_NEW_TURN, 'bob': _K2_BOB_INSERTS},
        variant='readable-turn', setup=_k2_public_turn,
    ),
    AccessCase(
        'K2', _resend_k2, {'alice': _K2_ALICE_NEW_TURN, 'bob': _K2_BOB_INSERTS},
        variant='granted-turn', setup=_k2_granted_turn,
    ),
    AccessCase(
        'K2', _resend_k2,
        {'alice': (True, (('k2_alice', 'alice', 1), ('k2_bob_turn', 'bob', 0))), 'bob': _K2_BOB_INSERTS},
        variant='hidden-turn', setup=_k2_hidden_turn,
    ),
    AccessCase(
        'K3', _check_k3,
        {'alice': ('k3_alice', 'Summary of k3_alice'), 'bob': None, 'carol': None, 'dave': None},
        variant='readable-foreign', setup=_add_k3_alice,
    ),
    AccessCase(
        'K3', _check_k3,
        {'bob': ('k3_bob', 'Summary of k3_bob'), 'alice': None, 'carol': None},
        variant='hidden-latest', setup=_k3_hidden_row_after_bob,
    ),
    AccessCase('K3', _check_k3, {'alice': None}, variant='readable-turn', setup=_k3_public_turn),
    AccessCase(
        'K3', _check_k3, {'alice': ('k3_alice', 'Summary of k3_alice')}, variant='hidden-turn', setup=_k3_hidden_turn,
    ),
    AccessCase(
        'K4', _send_k4,
        {
            'alice': (False, (('k4_alice', 'alice', 0), ('k4_bob', 'bob', 0), ('new', 'alice', 0))),
            'bob': _K4_BOB_INSERTS,
        },
        variant='readable-other', setup=_k4_bob_public_after_alice,
    ),
    AccessCase(
        'K4', _send_k4,
        {'alice': (True, (('k4_alice', 'alice', 1), ('k4_bob', 'bob', 0))), 'bob': _K4_BOB_INSERTS},
        variant='hidden-other', setup=_k4_bob_private_after_alice,
    ),
)
