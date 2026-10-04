"""Two-principal cases of the ranked searches: vector search on both layouts and full-text search (R1-R4).

Every ranked statement carries the caller's READ predicate before its rank depth, LIMIT and
OFFSET, so a row the caller may not read never takes a rank position: each page is a full
window into the readable rows alone, in rank order, and ``filters_applied`` counts the
client filters only.

- R1: ``EmbeddingRepository.search`` on the fp32 layout returns the readable rows nearest
  first. Variants: unfiltered; filtered by thread, source, tag, a simple metadata key and
  an ``in`` filter, with public decoys nearer than every seed row each failing one filter;
  an adversarial metadata filter whose value is another principal's id; and page-fill,
  unfiltered and filtered, where private rows nearer the query than every seed row are
  hidden from every scope but their owner, yet every other scope's pages are full of its
  readable rows in rank order.
- R2: the same variants on the compressed layout, where the READ predicate narrows the
  candidate set whose payloads are scored.
- R3: the compressed search re-applies the READ predicate when it hydrates the ranked
  page, so a row that stops being readable between candidate selection and hydration is
  dropped from the result.
- R4: ``FtsRepository.search`` returns the readable matches ranked by score, with the
  variants of R1; its page-fill rows repeat the keyword more often than any seed row, so
  they out-rank the seed on both backends. Two variants pin how the backends score: on
  SQLite the FTS5 ``bm25()`` score draws on statistics of the whole index, so a row the
  caller cannot read changes the scores of the rows it can while their membership and page
  stay the same; on PostgreSQL ``ts_rank_cd`` scores each document alone, so the scores
  stay identical.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any
from unittest.mock import patch

from app.access_scope import Scope
from tests.repositories._access_scope_cases import QUERY_VECTOR
from tests.repositories._access_scope_cases import READABLE
from tests.repositories._access_scope_cases import SEED_KEYWORD
from tests.repositories._access_scope_cases import SEED_LABELS
from tests.repositories._access_scope_cases import SEED_SOURCE
from tests.repositories._access_scope_cases import SEED_THREAD
from tests.repositories._access_scope_cases import AccessCase
from tests.repositories._access_scope_cases import Layout
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import Source
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import per_scope

type RankedObservable = tuple[tuple[str, ...], int]
type PagesObservable = tuple[tuple[str, ...], tuple[str, ...], int]
type FtsObservable = tuple[tuple[str, ...], bool, int]
type RevocationObservable = tuple[tuple[str, ...], bool]
type ScorePinObservable = tuple[tuple[str, ...], tuple[str, ...], bool, int, bool]

PAGE_SIZE = 2

# Scopes that read every row their owner alice adds, private ones included.
_OWNER_SCOPES = frozenset({'alice', 'system'})

# Private alice rows nearer the query vector than every seed row, nearest first. Their
# angles are far enough apart for the compressed inner-product estimate, which errs by about
# 0.02 per vector, to rank them in the order of the true angle as the fp32 distance does.
VECTOR_FILL: tuple[tuple[str, float], ...] = (('vector_fill_0', 0.0), ('vector_fill_1', 0.4), ('vector_fill_2', 0.5))

# Private alice rows (hidden) and public alice rows (visible to every scope) that repeat the
# keyword more often than any seed row, best first. A row repeating the keyword more scores
# higher on both backends, so the hidden rows out-rank the visible ones, which out-rank the seed.
FTS_FILL: tuple[tuple[str, int], ...] = (('fts_fill_0', 40), ('fts_fill_1', 30), ('fts_fill_2', 20))
FTS_VISIBLE: tuple[tuple[str, int], ...] = (
    ('fts_visible_0', 12), ('fts_visible_1', 10), ('fts_visible_2', 8), ('fts_visible_3', 6),
)

# Public alice rows nearer the query vector and repeating the keyword more often than every
# other row, each failing exactly one client filter of the filtered variants.
_DECOY_ANGLE = 0.0
_DECOY_KEYWORD_COUNT = 50
_DECOY_LABELS = ('decoy_thread', 'decoy_source', 'decoy_tag', 'decoy_corpus', 'decoy_label')

# The filtered variants' client filters: every seed and fill row passes them all, and each
# decoy fails exactly one. The simple metadata key and the ``in`` filter make two metadata
# conditions, so five filters apply in total.
_FILTER_LABELS: tuple[str, ...] = (
    *SEED_LABELS,
    *(label for label, _angle in VECTOR_FILL),
    *(label for label, _count in (*FTS_FILL, *FTS_VISIBLE)),
)
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

# The row the hydration case makes private between candidate selection and hydration: alice
# owns it and grants nothing on it, so only alice and the system scope still read it.
_REVOKED_LABEL = 'alice_public'
_HYDRATE_CLOSURES = frozenset({'_hydrate_sqlite', '_hydrate_pg'})

# Rows that do not mention the keyword, keeping its inverse document frequency positive in
# the score-pin variants, and the hidden row those variants add.
_UNMATCHED_LABELS = tuple(f'unmatched_{index}' for index in range(10))
_HIDDEN_MATCH_LABEL = 'fts_hidden_match'
_HIDDEN_MATCH_KEYWORD_COUNT = 25

# A seed row every scope of the score-pin variants reads, whose score they compare.
_SCORE_REFERENCE = 'alice_public'
_SCORE_PIN_SCOPES = ('bob', 'carol', 'dave')


def _keyword_text(label: str, count: int) -> str:
    """Return Markdown text for ``label`` repeating :data:`SEED_KEYWORD` ``count`` times."""
    return f'# {label}\n\n{" ".join([SEED_KEYWORD] * count)}\n'


def _authored_by_alice(labels: tuple[str, ...]) -> tuple[str, ...]:
    """Keep the seed rows alice authored: every seed row but bob's."""
    return tuple(label for label in labels if label != 'bob_private')


async def _add_decoy(
    db: ScopedDb,
    label: str,
    *,
    thread_id: str = SEED_THREAD,
    source: Source = SEED_SOURCE,
    tags: tuple[str, ...] | None = None,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Store one public decoy nearer the query vector and repeating the keyword more than every other row."""
    await db.add_entry(
        label, owner='alice', visibility='public', thread_id=thread_id, source=source, tags=tags,
        metadata=metadata, angle=_DECOY_ANGLE, text=_keyword_text(label, _DECOY_KEYWORD_COUNT),
    )


async def _add_decoys(db: ScopedDb) -> None:
    """Store the public decoys, each failing exactly one client filter."""
    await _add_decoy(db, 'decoy_thread', thread_id=f'{SEED_THREAD}-decoy')
    await _add_decoy(db, 'decoy_source', source='user')
    await _add_decoy(db, 'decoy_tag', tags=('decoy',))
    await _add_decoy(db, 'decoy_corpus', metadata={'author': 'alice', 'label': 'decoy_corpus', 'corpus': 'decoy'})
    await _add_decoy(db, 'decoy_label')


async def _add_vector_fill(db: ScopedDb) -> None:
    """Store the hidden rows nearer the query vector than every seed row."""
    for label, angle in VECTOR_FILL:
        await db.add_entry(label, owner='alice', visibility='private', angle=angle)


async def _add_decoys_and_vector_fill(db: ScopedDb) -> None:
    """Store the decoys, then the hidden vector rows."""
    await _add_decoys(db)
    await _add_vector_fill(db)


async def _add_fts_fill(db: ScopedDb) -> None:
    """Store the hidden and the visible rows that out-score every seed row."""
    for label, count in FTS_FILL:
        await db.add_entry(label, owner='alice', visibility='private', text=_keyword_text(label, count))
    for label, count in FTS_VISIBLE:
        await db.add_entry(label, owner='alice', visibility='public', text=_keyword_text(label, count))


async def _add_decoys_and_fts_fill(db: ScopedDb) -> None:
    """Store the decoys, then the hidden and visible full-text rows."""
    await _add_decoys(db)
    await _add_fts_fill(db)


async def _add_unmatched_rows(db: ScopedDb) -> None:
    """Store public rows that never mention the keyword."""
    for label in _UNMATCHED_LABELS:
        await db.add_entry(label, owner='alice', visibility='public', text=f'# {label}\n\nAn entry about something else.\n')


async def _vector_search(db: ScopedDb, scope: Scope, **arguments: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Run ``EmbeddingRepository.search`` for :data:`QUERY_VECTOR` as the scope; return labels in rank order and the stats."""
    rows, stats = await db.repos.embeddings.search(query_embedding=QUERY_VECTOR, scope=scope, **arguments)
    return tuple(db.labels_of(str(row['id']) for row in rows)), stats


async def _nearest_unfiltered(db: ScopedDb, scope: Scope) -> RankedObservable:
    """Rank every row as the scope."""
    labels, stats = await _vector_search(db, scope, limit=50)
    return labels, stats['filters_applied']


async def _nearest_filtered(db: ScopedDb, scope: Scope) -> RankedObservable:
    """Rank with the five client filters as the scope."""
    labels, stats = await _vector_search(db, scope, limit=50, **_FILTERS)
    return labels, stats['filters_applied']


async def _nearest_adversarial(db: ScopedDb, scope: Scope) -> RankedObservable:
    """Rank alice's rows by the author metadata key as the scope."""
    labels, stats = await _vector_search(db, scope, limit=50, metadata=dict(_ADVERSARIAL_METADATA))
    return labels, stats['filters_applied']


async def _two_pages(
    search: Callable[..., Awaitable[tuple[tuple[str, ...], dict[str, Any]]]],
    db: ScopedDb,
    scope: Scope,
    **filters: Any,
) -> PagesObservable:
    """Run a ranked search for its first two pages of :data:`PAGE_SIZE` rows as the scope."""
    first, stats = await search(db, scope, limit=PAGE_SIZE, offset=0, **filters)
    second, _ = await search(db, scope, limit=PAGE_SIZE, offset=PAGE_SIZE, **filters)
    return first, second, stats['filters_applied']


async def _vector_pages_unfiltered(db: ScopedDb, scope: Scope) -> PagesObservable:
    """Rank two unfiltered pages as the scope."""
    return await _two_pages(_vector_search, db, scope)


async def _vector_pages_filtered(db: ScopedDb, scope: Scope) -> PagesObservable:
    """Rank two filtered pages as the scope."""
    return await _two_pages(_vector_search, db, scope, **_FILTERS)


def _pages_of(ordered: tuple[str, ...], filter_count: int) -> PagesObservable:
    """The first two pages of an ordering, with the filter count."""
    return ordered[:PAGE_SIZE], ordered[PAGE_SIZE:2 * PAGE_SIZE], filter_count


def _expected_vector_pages(scope_name: str, filter_count: int) -> PagesObservable:
    """The readable rows nearest first: the hidden rows lead for their owner, the seed follows."""
    hidden = tuple(label for label, _angle in VECTOR_FILL) if scope_name in _OWNER_SCOPES else ()
    return _pages_of((*hidden, *READABLE[scope_name]), filter_count)


async def _search_revoking_before_hydration(db: ScopedDb, scope: Scope) -> RevocationObservable:
    """Rank every row as the scope, making :data:`_REVOKED_LABEL` private just before the page is hydrated.

    Returns:
        The labels in rank order and whether the row was made private during the search.
    """
    execute_read = db.backend.execute_read
    revoked: list[str] = []

    async def _revoke_before_hydration(operation: Callable[..., Any], *args: Any, **kwargs: Any) -> object:
        if getattr(operation, '__name__', '') in _HYDRATE_CLOSURES:
            await db.execute(
                'UPDATE context_entries SET visibility = ? WHERE id = ?', ('private', db.ids[_REVOKED_LABEL]),
            )
            revoked.append(_REVOKED_LABEL)
        return await execute_read(operation, *args, **kwargs)

    with patch.object(db.backend, 'execute_read', _revoke_before_hydration):
        labels, _stats = await _vector_search(db, scope, limit=50)
    return labels, revoked == [_REVOKED_LABEL]


def _expected_after_revocation(scope_name: str) -> RevocationObservable:
    """The readable rows nearest first, without the revoked row unless the scope still reads it."""
    keep_revoked = scope_name in _OWNER_SCOPES
    return tuple(label for label in READABLE[scope_name] if keep_revoked or label != _REVOKED_LABEL), True


async def _fts_search(db: ScopedDb, scope: Scope, **arguments: Any) -> tuple[tuple[tuple[str, float], ...], dict[str, Any]]:
    """Run ``FtsRepository.search`` for :data:`SEED_KEYWORD` as the scope; return ``(label, score)`` pairs and the stats."""
    rows, stats = await db.repos.fts.search(query=SEED_KEYWORD, scope=scope, **arguments)
    labels = db.labels_of(str(row['id']) for row in rows)
    return tuple((label, float(row['score'])) for label, row in zip(labels, rows, strict=True)), stats


async def _fts_labels(db: ScopedDb, scope: Scope, **arguments: Any) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Run the full-text search as the scope; return the labels in rank order and the stats."""
    scored, stats = await _fts_search(db, scope, **arguments)
    return tuple(label for label, _score in scored), stats


def _membership(scored: tuple[tuple[str, float], ...], stats: dict[str, Any]) -> FtsObservable:
    """The matched labels in seed order, whether their scores never rise down the ranking, and the filter count."""
    scores = [score for _label, score in scored]
    ranked = all(earlier >= later for earlier, later in zip(scores, scores[1:], strict=False))
    return in_seed_order(label for label, _score in scored), ranked, stats['filters_applied']


async def _matches_unfiltered(db: ScopedDb, scope: Scope) -> FtsObservable:
    """Match every row as the scope."""
    return _membership(*await _fts_search(db, scope, limit=50))


async def _matches_filtered(db: ScopedDb, scope: Scope) -> FtsObservable:
    """Match with the five client filters as the scope."""
    return _membership(*await _fts_search(db, scope, limit=50, **_FILTERS))


async def _matches_adversarial(db: ScopedDb, scope: Scope) -> FtsObservable:
    """Match alice's rows by the author metadata key as the scope."""
    return _membership(*await _fts_search(db, scope, limit=50, metadata=dict(_ADVERSARIAL_METADATA)))


async def _fts_pages_unfiltered(db: ScopedDb, scope: Scope) -> PagesObservable:
    """Match two unfiltered pages as the scope."""
    return await _two_pages(_fts_labels, db, scope)


async def _fts_pages_filtered(db: ScopedDb, scope: Scope) -> PagesObservable:
    """Match two filtered pages as the scope."""
    return await _two_pages(_fts_labels, db, scope, **_FILTERS)


def _expected_fts_pages(scope_name: str, filter_count: int) -> PagesObservable:
    """The readable full-text rows best first: the hidden rows lead for their owner, the visible ones follow."""
    hidden = tuple(label for label, _count in FTS_FILL) if scope_name in _OWNER_SCOPES else ()
    return _pages_of((*hidden, *(label for label, _count in FTS_VISIBLE)), filter_count)


async def _scores_around_a_hidden_match(db: ScopedDb, scope: Scope) -> ScorePinObservable:
    """Search as the scope before and after alice adds a private row matching the keyword.

    Returns:
        The matched labels in seed order before and after, whether the first page is
        unchanged, the size of that page after, and whether the score of
        :data:`_SCORE_REFERENCE` changed.
    """
    before, _ = await _fts_search(db, scope, limit=50)
    page_before, _ = await _fts_labels(db, scope, limit=PAGE_SIZE)

    await db.add_entry(
        _HIDDEN_MATCH_LABEL, owner='alice', visibility='private',
        text=_keyword_text(_HIDDEN_MATCH_LABEL, _HIDDEN_MATCH_KEYWORD_COUNT),
    )

    after, _ = await _fts_search(db, scope, limit=50)
    page_after, _ = await _fts_labels(db, scope, limit=PAGE_SIZE)
    before_scores, after_scores = dict(before), dict(after)
    return (
        in_seed_order(before_scores),
        in_seed_order(after_scores),
        page_before == page_after,
        len(page_after),
        before_scores[_SCORE_REFERENCE] != after_scores[_SCORE_REFERENCE],
    )


def _expected_score_pin(score_changes: bool) -> dict[str, ScorePinObservable]:
    """Membership and page unchanged for every scope that cannot read the added row; the score as the backend scores."""
    return {
        scope_name: (READABLE[scope_name], READABLE[scope_name], True, PAGE_SIZE, score_changes)
        for scope_name in _SCORE_PIN_SCOPES
    }


def _vector_cases(case_id: str, layout: Layout) -> tuple[AccessCase, ...]:
    """The vector-search variants of one layout."""
    return (
        AccessCase(case_id, _nearest_unfiltered, per_scope(lambda name: (READABLE[name], 0)), layout=layout),
        AccessCase(
            case_id, _nearest_filtered, per_scope(lambda name: (READABLE[name], _FILTER_COUNT)),
            filtered=True, layout=layout, setup=_add_decoys,
        ),
        AccessCase(
            case_id, _nearest_adversarial, per_scope(lambda name: (_authored_by_alice(READABLE[name]), 1)),
            filtered=True, layout=layout, variant='adversarial',
        ),
        AccessCase(
            case_id, _vector_pages_unfiltered, per_scope(lambda name: _expected_vector_pages(name, 0)),
            layout=layout, variant='page-fill', setup=_add_vector_fill,
        ),
        AccessCase(
            case_id, _vector_pages_filtered, per_scope(lambda name: _expected_vector_pages(name, _FILTER_COUNT)),
            filtered=True, layout=layout, variant='page-fill', setup=_add_decoys_and_vector_fill,
        ),
    )


RANKED_CASES: tuple[AccessCase, ...] = (
    *_vector_cases('R1', 'fp32'),
    *_vector_cases('R2', 'compressed'),
    AccessCase('R3', _search_revoking_before_hydration, per_scope(_expected_after_revocation)),
    AccessCase('R4', _matches_unfiltered, per_scope(lambda name: (READABLE[name], True, 0))),
    AccessCase(
        'R4', _matches_filtered, per_scope(lambda name: (READABLE[name], True, _FILTER_COUNT)),
        filtered=True, setup=_add_decoys,
    ),
    AccessCase(
        'R4', _matches_adversarial, per_scope(lambda name: (_authored_by_alice(READABLE[name]), True, 1)),
        filtered=True, variant='adversarial',
    ),
    AccessCase(
        'R4', _fts_pages_unfiltered, per_scope(lambda name: _expected_fts_pages(name, 0)),
        variant='page-fill', setup=_add_fts_fill,
    ),
    AccessCase(
        'R4', _fts_pages_filtered, per_scope(lambda name: _expected_fts_pages(name, _FILTER_COUNT)),
        filtered=True, variant='page-fill', setup=_add_decoys_and_fts_fill,
    ),
    AccessCase(
        'R4', _scores_around_a_hidden_match, _expected_score_pin(score_changes=True),
        variant='score-pin', setup=_add_unmatched_rows, backend='sqlite',
    ),
    AccessCase(
        'R4', _scores_around_a_hidden_match, _expected_score_pin(score_changes=False),
        variant='score-identity', setup=_add_unmatched_rows, backend='postgresql',
    ),
)
