"""Two-principal cases of the browse search and the grep pre-filter scan (S1, S2).

Both seams build their WHERE clause through the shared filter builder, which appends the
READ predicate after every client filter, so the caller's scope and its filters compose:
the rows returned, their order, the page boundaries and the scan figures are computed over
the readable rows alone, and ``filters_applied`` counts the client filters only.

- S1: ``search_contexts`` returns the readable rows newest first. Variants: unfiltered;
  filtered by thread, source, tag, a simple metadata key and an ``in`` filter, with decoy
  rows each failing one filter; an adversarial metadata filter whose value is another
  principal's id; ``explain_query``; and page-fill, where more hidden rows than the page
  size are newer than every readable row, yet each page is full of readable rows in order.
- S2: ``grep_scan_text_contents`` returns the readable rows newest first with ``scanned``
  and ``truncated`` counting readable rows only. Variants: unfiltered; filtered (the same
  filters plus the ASCII pre-narrow, with a decoy lacking the keyword); adversarial; and
  page-fill, where hidden rows newer than every readable row never take a slot under the
  entry cap and hidden rows beyond the cap never flag the scan as truncated.
"""

from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from app.access_scope import Scope
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SCOPES
from tests.repositories._access_scope_cases import SEED_KEYWORD
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import SEED_SOURCE
from tests.repositories._access_scope_cases import SEED_THREAD
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import ScopedDb

type BrowseObservable = tuple[tuple[str, ...], int]
type BrowsePages = tuple[tuple[str, ...], tuple[str, ...], int]
type ExplainObservable = tuple[tuple[str, ...], bool]
type ScanObservable = tuple[tuple[str, ...], int, bool]

# Private alice rows newer than every seed row: hidden from every scope but alice and the
# system scope, and more of them than one page or the scan's entry cap holds.
FILL_LABELS: tuple[str, ...] = tuple(f'fill_{index}' for index in range(5))

# A row carrying every filtered attribute except the keyword the scan's ASCII pre-narrow
# looks for, so only the pre-narrow excludes it.
KEYWORD_DECOY = 'decoy_keyword'

PAGE_SIZE = 3
SCAN_CAP = 3
SCAN_PAGE_SIZE = 2

# The filtered variants' client filters: every seed and fill row passes them all, and each
# decoy fails exactly one. The simple metadata key and the ``in`` filter make two metadata
# conditions, so five filters apply in total.
_FILTER_LABELS: tuple[str, ...] = (*SEED_LABELS, *FILL_LABELS, KEYWORD_DECOY)
_FILTERS: Mapping[str, Any] = {
    'thread_id': SEED_THREAD,
    'source': SEED_SOURCE,
    'tags': ['seed'],
    'metadata': {'corpus': 'seed'},
    'metadata_filters': [{'key': 'label', 'operator': 'in', 'value': list(_FILTER_LABELS)}],
}
_FILTER_COUNT = 5

# The adversarial filter names alice as the author: a placeholder-numbering collision that
# bound this value into the owner arm of the access predicate would expose alice's private
# rows to every other scope.
_ADVERSARIAL_METADATA: Mapping[str, Any] = {'author': 'alice'}
_ADVERSARIAL_METADATA_FILTERS: list[dict[str, Any]] = [{'key': 'author', 'operator': 'eq', 'value': 'alice'}]


def _per_scope[T](build: Callable[[str], T]) -> dict[str, T]:
    """Return ``build(scope_name)`` for every scope name, the system scope included."""
    return {scope_name: build(scope_name) for scope_name in SCOPES}


def _newest_first(scope_name: str, *, with_fill: bool = False, author: str | None = None) -> tuple[str, ...]:
    """Return the rows the scope reads, newest first.

    Rows are stored in seed order and the fill rows after them, so newest first is the
    reverse of that order. Every fill row and every seed row but ``bob_private`` is alice's.

    Args:
        scope_name: The scope, a key of :data:`READABLE`.
        with_fill: Include the fill rows the scope reads.
        author: Keep only this author's rows (``alice`` is the only author filtered on).

    Returns:
        The labels newest first.
    """
    readable = list(READABLE[scope_name])
    if with_fill and scope_name in ('alice', 'system'):
        readable.extend(FILL_LABELS)
    if author == 'alice':
        readable = [label for label in readable if label != 'bob_private']
    return tuple(reversed(readable))


async def _add_fill_rows(db: ScopedDb) -> None:
    """Store the fill rows: private alice rows, newer than the seed, passing every filter."""
    for label in FILL_LABELS:
        await db.add_entry(label, owner='alice', visibility='private')


async def _add_filter_decoys(db: ScopedDb) -> None:
    """Store public alice rows each failing exactly one client filter."""
    await db.add_entry('decoy_thread', owner='alice', visibility='public', thread_id=f'{SEED_THREAD}-decoy')
    await db.add_entry('decoy_source', owner='alice', visibility='public', source='user')
    await db.add_entry('decoy_tag', owner='alice', visibility='public', tags=('decoy',))
    await db.add_entry(
        'decoy_corpus', owner='alice', visibility='public',
        metadata={'author': 'alice', 'label': 'decoy_corpus', 'corpus': 'decoy'},
    )
    await db.add_entry('decoy_label', owner='alice', visibility='public')


async def _add_scan_decoys(db: ScopedDb) -> None:
    """Store the filter decoys plus a public row that passes every filter but lacks the keyword."""
    await _add_filter_decoys(db)
    await db.add_entry(
        KEYWORD_DECOY, owner='alice', visibility='public', text=f'# {KEYWORD_DECOY}\n\nAn entry without the scanned word.\n',
    )


async def _add_fill_and_filter_decoys(db: ScopedDb) -> None:
    """Store the filter decoys, then the fill rows."""
    await _add_filter_decoys(db)
    await _add_fill_rows(db)


async def _add_fill_and_scan_decoys(db: ScopedDb) -> None:
    """Store the scan decoys, then the fill rows."""
    await _add_scan_decoys(db)
    await _add_fill_rows(db)


async def _browse(db: ScopedDb, scope: Scope, **arguments: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Run ``search_contexts`` as the scope and return the row labels in result order with the stats."""
    rows, stats = await db.repos.context.search_contexts(scope=scope, **arguments)
    assert 'error' not in stats, stats
    return tuple(db.labels_of(str(row['id']) for row in rows)), stats


async def _browse_unfiltered(db: ScopedDb, scope: Scope) -> BrowseObservable:
    """Browse every row as the scope."""
    labels, stats = await _browse(db, scope, limit=50)
    return labels, stats['filters_applied']


async def _browse_filtered(db: ScopedDb, scope: Scope) -> BrowseObservable:
    """Browse with the five client filters as the scope."""
    labels, stats = await _browse(db, scope, limit=50, **_FILTERS)
    return labels, stats['filters_applied']


async def _browse_adversarial(db: ScopedDb, scope: Scope) -> BrowseObservable:
    """Browse alice's rows by the author metadata key as the scope."""
    labels, stats = await _browse(db, scope, limit=50, metadata=dict(_ADVERSARIAL_METADATA))
    return labels, stats['filters_applied']


async def _browse_explained(db: ScopedDb, scope: Scope) -> ExplainObservable:
    """Browse with ``explain_query`` as the scope and report whether a query plan came back."""
    labels, stats = await _browse(db, scope, limit=50, explain_query=True)
    plan = stats.get('query_plan')
    return labels, isinstance(plan, str) and bool(plan)


async def _browse_two_pages(db: ScopedDb, scope: Scope, **filters: Any) -> BrowsePages:
    """Browse the first two pages of :data:`PAGE_SIZE` rows as the scope."""
    first, stats = await _browse(db, scope, limit=PAGE_SIZE, offset=0, **filters)
    second, _ = await _browse(db, scope, limit=PAGE_SIZE, offset=PAGE_SIZE, **filters)
    return first, second, stats['filters_applied']


async def _browse_pages_unfiltered(db: ScopedDb, scope: Scope) -> BrowsePages:
    """Browse two unfiltered pages as the scope."""
    return await _browse_two_pages(db, scope)


async def _browse_pages_filtered(db: ScopedDb, scope: Scope) -> BrowsePages:
    """Browse two filtered pages as the scope."""
    return await _browse_two_pages(db, scope, **_FILTERS)


def _expected_pages(scope_name: str, filter_count: int) -> BrowsePages:
    """The first two pages of the readable rows, the fill rows included, newest first."""
    ordered = _newest_first(scope_name, with_fill=True)
    return ordered[:PAGE_SIZE], ordered[PAGE_SIZE:2 * PAGE_SIZE], filter_count


async def _scan(db: ScopedDb, scope: Scope, **arguments: Any) -> ScanObservable:
    """Run ``grep_scan_text_contents`` as the scope and return the labels in scan order, ``scanned`` and ``truncated``."""
    rows, stats = await db.repos.context.grep_scan_text_contents(scope=scope, **arguments)
    assert 'validation_errors' not in stats, stats
    return tuple(db.labels_of(context_id for context_id, _text in rows)), stats['scanned'], stats['truncated']


async def _scan_unfiltered(db: ScopedDb, scope: Scope) -> ScanObservable:
    """Scan every row as the scope."""
    return await _scan(db, scope)


async def _scan_filtered(db: ScopedDb, scope: Scope) -> ScanObservable:
    """Scan with the five client filters and the keyword pre-narrow as the scope."""
    return await _scan(db, scope, ascii_literal=SEED_KEYWORD, **_FILTERS)


async def _scan_adversarial(db: ScopedDb, scope: Scope) -> ScanObservable:
    """Scan alice's rows by an advanced author filter as the scope."""
    return await _scan(db, scope, metadata_filters=_ADVERSARIAL_METADATA_FILTERS)


async def _scan_capped(db: ScopedDb, scope: Scope) -> ScanObservable:
    """Scan under the entry cap in pages of :data:`SCAN_PAGE_SIZE` as the scope."""
    return await _scan(db, scope, max_entries_scanned=SCAN_CAP, page_size=SCAN_PAGE_SIZE)


async def _scan_capped_filtered(db: ScopedDb, scope: Scope) -> ScanObservable:
    """Scan with the filters and the pre-narrow under the entry cap in small pages as the scope."""
    return await _scan(
        db, scope, ascii_literal=SEED_KEYWORD, max_entries_scanned=SCAN_CAP, page_size=SCAN_PAGE_SIZE, **_FILTERS,
    )


def _expected_scan(labels: tuple[str, ...]) -> ScanObservable:
    """A complete scan of the given rows: all of them, none left over."""
    return labels, len(labels), False


def _expected_capped_scan(scope_name: str) -> ScanObservable:
    """The newest :data:`SCAN_CAP` readable rows, truncated only when a further readable row exists."""
    ordered = _newest_first(scope_name, with_fill=True)
    kept = ordered[:SCAN_CAP]
    return kept, len(kept), len(ordered) > SCAN_CAP


SEARCH_CASES: tuple[AccessCase, ...] = (
    AccessCase('S1', _browse_unfiltered, _per_scope(lambda name: (_newest_first(name), 0))),
    AccessCase(
        'S1', _browse_filtered, _per_scope(lambda name: (_newest_first(name), _FILTER_COUNT)),
        filtered=True, setup=_add_filter_decoys,
    ),
    AccessCase(
        'S1', _browse_adversarial, _per_scope(lambda name: (_newest_first(name, author='alice'), 1)),
        filtered=True, variant='adversarial',
    ),
    AccessCase('S1', _browse_explained, _per_scope(lambda name: (_newest_first(name), True)), variant='explain'),
    AccessCase(
        'S1', _browse_pages_unfiltered, _per_scope(lambda name: _expected_pages(name, 0)),
        variant='page-fill', setup=_add_fill_rows,
    ),
    AccessCase(
        'S1', _browse_pages_filtered, _per_scope(lambda name: _expected_pages(name, _FILTER_COUNT)),
        filtered=True, variant='page-fill', setup=_add_fill_and_filter_decoys,
    ),
    AccessCase('S2', _scan_unfiltered, _per_scope(lambda name: _expected_scan(_newest_first(name)))),
    AccessCase(
        'S2', _scan_filtered, _per_scope(lambda name: _expected_scan(_newest_first(name))),
        filtered=True, setup=_add_scan_decoys,
    ),
    AccessCase(
        'S2', _scan_adversarial, _per_scope(lambda name: _expected_scan(_newest_first(name, author='alice'))),
        filtered=True, variant='adversarial',
    ),
    AccessCase('S2', _scan_capped, _per_scope(_expected_capped_scan), variant='page-fill', setup=_add_fill_rows),
    AccessCase(
        'S2', _scan_capped_filtered, _per_scope(_expected_capped_scan),
        filtered=True, variant='page-fill', setup=_add_fill_and_scan_decoys,
    ),
)
