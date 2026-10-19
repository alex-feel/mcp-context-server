"""The manifest a corpus sync sends, and the byte measure every request of a sync is held to.

A manifest lists every document of a corpus by its source id with the two digests
the server compares with what it stores: the text digest, which equals the content
hash the server records for the stored text, and the metadata digest over
everything else the client sends. A source the client could not read in this run
is listed as a keep item instead, so the plan leaves its stored entry alone.

The server plans at most ``MAX_MANIFEST_ITEMS`` items per call and an HTTP request
body is capped at 4 MiB, a limit the transport reports only as an opaque error. The
client therefore sends the sorted manifest in consecutive source-id ranges and
packs batch calls by entry count and by serialized size, measuring each request
before it is sent.
"""

import json
from collections.abc import Mapping
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import cast

from app.digests import canonical_json_digest
from app.digests import content_hash
from app.digests import normalize_stored_text
from app.ingest import MAX_MANIFEST_ITEMS
from app.models import MAX_THREAD_ID_LENGTH
from app.models import normalize_tag_list

if TYPE_CHECKING:
    # corpus builds its documents with the digests and the measure defined here, so the
    # corpus type is resolved for type checkers only.
    from app.cli.ingest.corpus import Corpus

# The framing of one tool call, and the framing each batch entry adds beside its own
# compact JSON, as measured on the wire for calls a fastmcp client sends.
CALL_ENVELOPE_BYTES = 391
ENTRY_FRAMING_BYTES = 66
# A plan call names its thread, which is at most MAX_THREAD_ID_LENGTH code points of at
# most six bytes each once JSON-escaped; measuring that width keeps the plan measure
# an upper bound for every thread id the server accepts.
_THREAD_ID_ALLOWANCE = 6 * MAX_THREAD_ID_LENGTH


@dataclass(frozen=True)
class ManifestSlice:
    """One plan call's share of the sorted manifest and the source-id range it covers.

    Attributes:
        items: The manifest items of the range, sorted by source id.
        source_id_after: The exclusive lower bound, or None for the first range.
        source_id_through: The inclusive upper bound, or None for the last range.
    """

    items: list[dict[str, object]]
    source_id_after: str | None
    source_id_through: str | None


def text_digest(text: str) -> str:
    """Return the digest the server records for a text once it is stored.

    Args:
        text: The text the client sends.

    Returns:
        The content hash of the text as the server normalizes it.
    """
    return content_hash(normalize_stored_text(text))


def meta_digest(metadata: Mapping[str, object], tags: Sequence[str], visibility: str) -> str:
    """Return the digest of everything the client sends besides the text.

    Tags are hashed as the server stores them, a set of normalized labels, so their
    order and case do not matter; the visibility is the value the client sends.

    Args:
        metadata: The entry metadata without the ingest block.
        tags: The tags the client sends.
        visibility: The visibility the client sends with every create and update.

    Returns:
        64 lowercase hexadecimal characters.
    """
    return canonical_json_digest({
        'metadata': dict(metadata),
        'tags': sorted(normalize_tag_list(list(tags))),
        'visibility': visibility,
    })


def manifest_items(corpus: 'Corpus') -> list[dict[str, object]]:
    """Return the manifest of a corpus, sorted by source id in code-point order.

    Every document becomes an item with its digests, and every kept file a keep
    item. A file excluded from the manifest contributes nothing.

    Args:
        corpus: The corpus a run built.

    Returns:
        The manifest items.
    """
    items: list[dict[str, object]] = [
        {'source_id': document.source_id, 'text_digest': document.text_digest, 'meta_digest': document.meta_digest}
        for document in corpus.documents
    ]
    items.extend({'source_id': failure.source_id, 'keep': True} for failure in corpus.file_failures if failure.keep)
    return sorted(items, key=_source_id)


def request_bytes(entries: Sequence[dict[str, object]]) -> int:
    """Return the size of a batch request carrying the given entries.

    Args:
        entries: The entries, as the client serializes them.

    Returns:
        The call envelope plus, per entry, its framing and its compact UTF-8 JSON.
    """
    return CALL_ENVELOPE_BYTES + sum(_entry_bytes(entry) for entry in entries)


def plan_request_bytes(
    items: Sequence[dict[str, object]],
    *,
    source_id_after: str | None,
    source_id_through: str | None,
) -> int:
    """Return an upper bound of the size of a plan request for a slice of the manifest.

    Args:
        items: The manifest items of the slice.
        source_id_after: The slice's exclusive lower bound.
        source_id_through: The slice's inclusive upper bound.

    Returns:
        The request size, measured with the widest thread id the server accepts.
    """
    return _plan_bytes(
        sum(_entry_bytes(item) for item in items),
        source_id_after=source_id_after,
        source_id_through=source_id_through,
    )


def slice_manifest(
    items: Sequence[dict[str, object]],
    *,
    max_items: int = MAX_MANIFEST_ITEMS,
    max_request_bytes: int,
) -> list[ManifestSlice]:
    """Split a manifest into consecutive ranges that together cover every source id.

    The items are sorted by source id and taken greedily: a slice ends when the
    next item would pass ``max_items`` or make the plan request larger than
    ``max_request_bytes``. Slice k covers ``(last source id of slice k-1, last
    source id of slice k]``; the first range has no lower bound and the last no
    upper bound, so every stored entry falls in exactly one range, and an empty
    manifest is one range covering everything.

    Args:
        items: The manifest items.
        max_items: The most items one plan call carries.
        max_request_bytes: The request budget of one call.

    Returns:
        The slices in source-id order.

    Raises:
        ValueError: When the budget cannot hold a plan request for even one item.
    """
    slices: list[ManifestSlice] = []
    current: list[dict[str, object]] = []
    current_bytes = 0
    after: str | None = None
    for item in sorted(items, key=_source_id):
        source_id = _source_id(item)
        item_bytes = _entry_bytes(item)
        if current and (
            len(current) >= max_items
            or _plan_bytes(current_bytes + item_bytes, source_id_after=after, source_id_through=source_id) > max_request_bytes
        ):
            through = _source_id(current[-1])
            slices.append(ManifestSlice(items=current, source_id_after=after, source_id_through=through))
            current, current_bytes, after = [], 0, through
        alone = _plan_bytes(item_bytes, source_id_after=after, source_id_through=source_id)
        if alone > max_request_bytes:
            raise ValueError(
                f'max_request_bytes ({max_request_bytes:,}) cannot hold the plan request for {source_id!r}, '
                f'which needs {alone:,} bytes',
            )
        current.append(item)
        current_bytes += item_bytes
    slices.append(ManifestSlice(items=current, source_id_after=after, source_id_through=None))
    return slices


def pack(
    entries: Sequence[dict[str, object]],
    *,
    batch_size: int,
    max_request_bytes: int,
) -> list[list[dict[str, object]]]:
    """Group entries into batch requests that respect the entry count and the byte budget.

    Entries keep their order; a batch ends when the next entry would pass
    ``batch_size`` entries or make the request larger than ``max_request_bytes``.

    Args:
        entries: The create or update entries to send.
        batch_size: The most entries one batch call carries.
        max_request_bytes: The request budget of one call.

    Returns:
        The batches in order.

    Raises:
        ValueError: When one entry alone needs more than the budget.
    """
    batches: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    current_bytes = CALL_ENVELOPE_BYTES
    for entry in entries:
        entry_bytes = _entry_bytes(entry)
        if CALL_ENVELOPE_BYTES + entry_bytes > max_request_bytes:
            raise ValueError(
                f'max_request_bytes ({max_request_bytes:,}) cannot hold one entry that needs '
                f'{CALL_ENVELOPE_BYTES + entry_bytes:,} bytes',
            )
        if current and (len(current) >= batch_size or current_bytes + entry_bytes > max_request_bytes):
            batches.append(current)
            current, current_bytes = [], CALL_ENVELOPE_BYTES
        current.append(entry)
        current_bytes += entry_bytes
    if current:
        batches.append(current)
    return batches


def _source_id(item: Mapping[str, object]) -> str:
    """Return the source id of a manifest item."""
    return cast('str', item['source_id'])


def _compact_size(value: object) -> int:
    """Return the UTF-8 length of a value's compact JSON form, as the client sends it."""
    return len(json.dumps(value, separators=(',', ':'), ensure_ascii=False).encode('utf-8'))


def _entry_bytes(entry: Mapping[str, object]) -> int:
    """Return what one entry or manifest item adds to a request."""
    return ENTRY_FRAMING_BYTES + _compact_size(entry)


def _plan_bytes(items_bytes: int, *, source_id_after: str | None, source_id_through: str | None) -> int:
    """Return the plan request size for items of the given total size and the given bounds."""
    arguments: dict[str, object] = {
        'thread_id': '',
        'manifest': [],
        'source_id_after': source_id_after,
        'source_id_through': source_id_through,
        'allow_empty_manifest': False,
    }
    return CALL_ENVELOPE_BYTES + items_bytes + _compact_size(arguments) + _THREAD_ID_ALLOWANCE
