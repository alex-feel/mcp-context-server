"""Two-principal cases of the row delete and the batch-delete id snapshot (X1, X2).

Deleting an entry and changing its visibility are owner-only, so the delete statement
removes only the rows the caller owns, whatever grants or visibility another row carries.
The batch-delete snapshot selects the ids its criteria match among the rows the caller may
read (READ mode, for a delete that names ids and must see the visible rows it would refuse)
or among the rows the caller owns (OWNER mode, for thread and criteria deletes, which skip
every other row).

- X1: ``delete_by_ids`` deletes exactly the requested rows the caller owns, outside and
  inside a transaction and across statement chunks; a read grant, a write grant and a
  public row delete nothing.
- X2: ``get_ids_matching_batch_criteria`` returns the readable matches in READ mode, granted
  and public rows of other principals included, and only the caller's own matches in OWNER
  mode, by thread and by named ids; ``older_than_days`` selects only rows older than the
  bound number of days; a call without criteria matches nothing.
"""

from collections.abc import Callable
from collections.abc import Sequence
from typing import Literal

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.ids import generate_id
from tests.repositories._access_scope_cases import OWNED
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import SEED_THREAD
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Invoker
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import per_scope
from tests.repositories._access_scope_cases_reads import ABSENT_ID
from tests.repositories._access_scope_cases_reads import ABSENT_LABEL

type DeleteObservable = tuple[int, tuple[str, ...]]
type SnapshotMode = Literal[AccessMode.READ, AccessMode.OWNER]
type PerRowObservable = tuple[tuple[tuple[str, int], ...], tuple[str, ...]]

# Requested ids of the chunked variant: enough to span two statement chunks, with the seed
# ids placed on both sides of the first chunk boundary.
_CHUNKED_ID_COUNT = 1200
_CHUNKED_SEED_POSITIONS = (0, 450, 898, 899, 900, 901, 1100, 1199)

# Age of the rows the older_than_days variant backdates, against a bound of OLDER_THAN_DAYS:
# two seed rows just past the bound and one just inside it.
OLDER_THAN_DAYS = 5
_BACKDATED_DAYS = {'alice_public': 6, 'bob_private': 6, 'alice_private': 4}
_OLDER_LABELS = tuple(label for label, days in _BACKDATED_DAYS.items() if days > OLDER_THAN_DAYS)


def _seed_ids_with_absent(db: ScopedDb) -> list[str]:
    """Return the ids of every seed row followed by the absent id."""
    return [*(db.ids[label] for label in SEED_LABELS), ABSENT_ID]


def _chunked_ids(db: ScopedDb) -> list[str]:
    """Return absent ids spanning two statement chunks, with the seed ids around the first boundary."""
    ids = [generate_id() for _ in range(_CHUNKED_ID_COUNT)]
    for position, label in zip(_CHUNKED_SEED_POSITIONS, SEED_LABELS, strict=True):
        ids[position] = db.ids[label]
    return ids


async def _surviving_labels(db: ScopedDb) -> tuple[str, ...]:
    """Return the labels of the seed rows still stored, read through raw SQL, in seed order."""
    rows = await db.fetch_all('SELECT id FROM context_entries')
    stored = {str(row[0]) for row in rows}
    return tuple(label for label in SEED_LABELS if db.ids[label] in stored)


def _not_owned(scope_name: str) -> tuple[str, ...]:
    """Return the seed rows the scope does not own, in seed order."""
    return tuple(label for label in SEED_LABELS if label not in OWNED[scope_name])


async def _delete_all(db: ScopedDb, scope: Scope) -> DeleteObservable:
    """Delete the seed ids and the absent id in one call as the scope."""
    deleted = await db.repos.context.delete_by_ids(_seed_ids_with_absent(db), scope=scope)
    return deleted, await _surviving_labels(db)


async def _delete_all_in_transaction(db: ScopedDb, scope: Scope) -> DeleteObservable:
    """Delete the seed ids and the absent id on a transaction connection as the scope."""
    async with db.backend.begin_transaction() as txn:
        deleted = await db.repos.context.delete_by_ids(_seed_ids_with_absent(db), scope=scope, txn=txn)
    return deleted, await _surviving_labels(db)


async def _delete_chunked(db: ScopedDb, scope: Scope) -> DeleteObservable:
    """Delete the seed rows through an id list that spans two statement chunks."""
    deleted = await db.repos.context.delete_by_ids(_chunked_ids(db), scope=scope)
    return deleted, await _surviving_labels(db)


def _expected_delete(scope_name: str) -> DeleteObservable:
    """Every owned seed row is deleted; every other row survives."""
    return len(OWNED[scope_name]), _not_owned(scope_name)


async def _delete_each(db: ScopedDb, scope: Scope) -> PerRowObservable:
    """Delete every seed row and the absent id one call at a time, reporting each count."""
    outcomes: list[tuple[str, int]] = []
    for label in (*SEED_LABELS, ABSENT_LABEL):
        context_id = db.ids.get(label, ABSENT_ID)
        outcomes.append((label, await db.repos.context.delete_by_ids([context_id], scope=scope)))
    return tuple(outcomes), await _surviving_labels(db)


def _expected_delete_each(scope_name: str) -> PerRowObservable:
    """Only an owned row reports a deleted row; a granted, public or absent row reports none."""
    outcomes = tuple((label, int(label in OWNED[scope_name])) for label in (*SEED_LABELS, ABSENT_LABEL))
    return outcomes, _not_owned(scope_name)


async def _snapshot_labels(
    db: ScopedDb,
    scope: Scope,
    mode: SnapshotMode,
    *,
    context_ids: Sequence[str] | None = None,
    thread_ids: Sequence[str] | None = None,
    older_than_days: int | None = None,
) -> tuple[str, ...]:
    """Return the labels of the snapshot ids, in seed order."""
    matched = await db.repos.context.get_ids_matching_batch_criteria(
        context_ids=list(context_ids) if context_ids is not None else None,
        thread_ids=list(thread_ids) if thread_ids is not None else None,
        older_than_days=older_than_days,
        scope=scope,
        mode=mode,
    )
    return in_seed_order(db.labels_of(matched))


def _snapshot_by_thread(mode: SnapshotMode) -> Invoker:
    """Build an invoker selecting the seed thread in ``mode``."""

    async def invoke(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
        return await _snapshot_labels(db, scope, mode, thread_ids=[SEED_THREAD])

    return invoke


def _snapshot_by_ids(mode: SnapshotMode) -> Invoker:
    """Build an invoker naming every seed id and the absent id in ``mode``."""

    async def invoke(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
        return await _snapshot_labels(db, scope, mode, context_ids=_seed_ids_with_absent(db))

    return invoke


def _snapshot_older(mode: SnapshotMode) -> Invoker:
    """Build an invoker selecting the seed thread's rows older than the bound in ``mode``."""

    async def invoke(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
        return await _snapshot_labels(db, scope, mode, thread_ids=[SEED_THREAD], older_than_days=OLDER_THAN_DAYS)

    return invoke


def _snapshot_without_criteria(mode: SnapshotMode) -> Invoker:
    """Build an invoker passing no criteria at all in ``mode``."""

    async def invoke(db: ScopedDb, scope: Scope) -> tuple[str, ...]:
        return await _snapshot_labels(db, scope, mode)

    return invoke


async def _backdate_rows(db: ScopedDb) -> None:
    """Move the creation time of the backdated rows the given number of days into the past."""
    if db.backend.backend_type == 'sqlite':
        statement = "UPDATE context_entries SET created_at = datetime('now', ?) WHERE id = ?"
        for label, days in _BACKDATED_DAYS.items():
            await db.execute(statement, (f'-{days} days', db.ids[label]))
        return
    statement = "UPDATE context_entries SET created_at = NOW() - (?::integer * INTERVAL '1 day') WHERE id = ?"
    for label, days in _BACKDATED_DAYS.items():
        await db.execute(statement, (days, db.ids[label]))


def _expected_older(table: dict[str, tuple[str, ...]]) -> Callable[[str], tuple[str, ...]]:
    """Build the expectation: the rows past the bound that ``table`` admits for the scope."""

    def build(scope_name: str) -> tuple[str, ...]:
        return in_seed_order(label for label in _OLDER_LABELS if label in table[scope_name])

    return build


def _nothing(scope_name: str) -> tuple[str, ...]:
    """No scope matches anything without criteria."""
    del scope_name
    return ()


DELETE_CASES: tuple[AccessCase, ...] = (
    AccessCase('X1', _delete_all, per_scope(_expected_delete)),
    AccessCase('X1', _delete_all_in_transaction, per_scope(_expected_delete), variant='in-transaction'),
    AccessCase('X1', _delete_chunked, per_scope(_expected_delete), variant='chunked'),
    AccessCase('X1', _delete_each, per_scope(_expected_delete_each), variant='per-row'),
    AccessCase('X2', _snapshot_by_thread(AccessMode.READ), per_scope(READABLE.__getitem__), variant='read-mode'),
    AccessCase('X2', _snapshot_by_thread(AccessMode.OWNER), per_scope(OWNED.__getitem__), variant='owner-mode'),
    AccessCase('X2', _snapshot_by_ids(AccessMode.READ), per_scope(READABLE.__getitem__), variant='read-mode-ids'),
    AccessCase('X2', _snapshot_by_ids(AccessMode.OWNER), per_scope(OWNED.__getitem__), variant='owner-mode-ids'),
    AccessCase(
        'X2', _snapshot_older(AccessMode.READ), per_scope(_expected_older(READABLE)),
        variant='read-mode-older', setup=_backdate_rows,
    ),
    AccessCase(
        'X2', _snapshot_older(AccessMode.OWNER), per_scope(_expected_older(OWNED)),
        variant='owner-mode-older', setup=_backdate_rows,
    ),
    AccessCase('X2', _snapshot_without_criteria(AccessMode.READ), per_scope(_nothing), variant='read-mode-no-criteria'),
    AccessCase(
        'X2', _snapshot_without_criteria(AccessMode.OWNER), per_scope(_nothing), variant='owner-mode-no-criteria',
    ),
)
