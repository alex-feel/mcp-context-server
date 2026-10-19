"""The corpus sync protocol shared by the server and the sync client.

An entry a sync client manages carries the reserved metadata block
``metadata.ingest = {scope, source_id, meta_digest, synced_at}``. ``scope`` names
the corpus, which is the thread the entry belongs to; ``source_id`` identifies the
source document inside the corpus; ``meta_digest`` is the client's digest of what
it controls besides the text; ``synced_at`` records when a sync last wrote the
entry. The server validates the block's shape on every write and decides
membership when it reads the block back: an entry is a member of thread ``T``
exactly when its block is valid and its ``scope`` equals ``T``. The block is
descriptive only and never takes part in deciding who may read or write an
entry.

A client describes its corpus as a manifest of :class:`SyncManifestItem` values,
one per source document, and the sync tools answer with the plan and state
shapes defined here, which the client also uses to read their responses. The
protocol constants bound what one call carries, because the transports a client
uses limit request and response sizes. This module imports the standard library,
pydantic and the thread-id bound of :mod:`app.models`, never the server's
settings or backends, so the client can import it.
"""

import re
from datetime import datetime
from typing import Annotated
from typing import Literal
from typing import Self
from typing import TypedDict
from typing import TypeGuard
from typing import cast

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

from app.models import MAX_THREAD_ID_LENGTH

INGEST_KEY = 'ingest'
INGEST_FIELDS = frozenset({'scope', 'source_id', 'meta_digest', 'synced_at'})
# A PostgreSQL btree index key holds thread_id and source_id together, so both stay
# far enough below the index-tuple ceiling that a valid store never aborts inside
# its transaction.
MAX_SOURCE_ID_LENGTH = 256
MAX_MANIFEST_ITEMS = 1000
MAX_PLAN_DELETES = 100
MAX_INGEST_STATE_PAGE = 200
DEFAULT_INGEST_STATE_PAGE = 100
# Anchored for pydantic's Field(pattern=...); matched here with re.fullmatch, so a
# trailing newline, which '$' alone would admit, is refused.
DIGEST_PATTERN = '^[0-9a-f]{64}$'
INGEST_BLOCK_SHAPE = 'an object with exactly the keys ' + ', '.join(sorted(INGEST_FIELDS))

_DIGEST = re.compile(DIGEST_PATTERN)
# ASCII digits only: in a str pattern, [0-9] never matches other Unicode digits.
_SYNCED_AT = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(?:\.[0-9]{1,6})?Z')
_BOUNDED_STRING_FIELDS = (('scope', MAX_THREAD_ID_LENGTH), ('source_id', MAX_SOURCE_ID_LENGTH))


def is_ingest_block_shaped(value: object) -> TypeGuard[dict[str, object]]:
    """Return whether a value is an object with exactly the block's four keys.

    Args:
        value: The value stored under the reserved key.

    Returns:
        True when the value is a dict whose key set is exactly ``INGEST_FIELDS``.
    """
    return isinstance(value, dict) and cast('dict[str, object]', value).keys() == INGEST_FIELDS


def ingest_block_error(value: object, *, path: str = 'metadata.ingest') -> str | None:
    """Return the first problem with an ingest block, or None when it is valid.

    The checks run in a fixed order, so a block with several problems always
    reports the same one: the shape, then the string type of ``scope`` and
    ``source_id``, then their emptiness, then their lengths in code points, then
    ``meta_digest``, then ``synced_at``. The check covers shape only; whether the
    block's scope matches the entry's thread is decided when the block is read.

    Args:
        value: The value stored under the reserved key.
        path: The name the messages give the block, such as ``metadata.ingest``
            or ``metadata_patch.ingest``.

    Returns:
        A message naming the field and the rule it breaks, or None.
    """
    if not is_ingest_block_shaped(value):
        return f'{path} must be {INGEST_BLOCK_SHAPE}'
    for field, _ in _BOUNDED_STRING_FIELDS:
        if not isinstance(value[field], str):
            return f'{path}.{field} must be a string'
    strings = {field: cast('str', value[field]) for field, _ in _BOUNDED_STRING_FIELDS}
    for field, _ in _BOUNDED_STRING_FIELDS:
        if not strings[field]:
            return f'{path}.{field} cannot be empty'
    for field, bound in _BOUNDED_STRING_FIELDS:
        if len(strings[field]) > bound:
            return f'{path}.{field} is too long: {len(strings[field])} characters, maximum is {bound}'
    meta_digest = value['meta_digest']
    if not isinstance(meta_digest, str) or _DIGEST.fullmatch(meta_digest) is None:
        return f'{path}.meta_digest must be 64 lowercase hexadecimal characters'
    synced_at = value['synced_at']
    if not isinstance(synced_at, str) or _SYNCED_AT.fullmatch(synced_at) is None or not _is_real_instant(synced_at):
        return f'{path}.synced_at must be a UTC timestamp such as 2026-10-18T12:00:00Z'
    return None


def _is_real_instant(timestamp: str) -> bool:
    """Return whether a timestamp of the expected form names a date and time that exist."""
    try:
        datetime.fromisoformat(timestamp)
    except ValueError:
        return False
    return True


class SyncManifestItem(BaseModel):
    """One source document of a corpus as a sync client describes it.

    An ordinary item carries both digests of the document; a keep item says the
    source exists but the client could not read it in this run, so a plan
    changes nothing stored for it. The source id is bounded like the stored
    block's on every item, keep items included, and an unknown key is refused,
    so a misspelled ``keep`` never turns a keep item into an ordinary one.

    Attributes:
        source_id: The document's identity inside the corpus, compared by code
            point.
        text_digest: ``app.digests.content_hash`` of the normalized text the
            client would send; None on a keep item.
        meta_digest: The client's digest of everything it sends besides the text;
            None on a keep item.
        keep: Whether the client could not read the source in this run.
    """

    model_config = ConfigDict(extra='forbid')

    source_id: Annotated[str, Field(min_length=1, max_length=MAX_SOURCE_ID_LENGTH)]
    text_digest: Annotated[str | None, Field(pattern=DIGEST_PATTERN)] = None
    meta_digest: Annotated[str | None, Field(pattern=DIGEST_PATTERN)] = None
    keep: bool = False

    @model_validator(mode='after')
    def _digests_match_keep(self) -> Self:
        """Require both digests on an ordinary item and refuse them on a keep item.

        Returns:
            The validated item.

        Raises:
            ValueError: When a keep item carries a digest or an ordinary item lacks one.
        """
        if self.keep:
            if self.text_digest is not None or self.meta_digest is not None:
                raise ValueError('a keep item carries no text_digest or meta_digest')
        elif self.text_digest is None or self.meta_digest is None:
            raise ValueError('an item without keep requires both text_digest and meta_digest')
        return self


# Plain aliases rather than ``type`` statements: a ``type`` alias becomes a named
# definition in the tools' output schemas, while these inline as the enums they name.
SyncChange = Literal['text', 'metadata', 'both']
SyncDeleteReason = Literal['absent', 'duplicate']


class SyncUpdateDict(TypedDict):
    """A planned update of a stored entry from a manifest item.

    Attributes:
        index: The position of the manifest item in the request's manifest.
        context_id: The stored entry to update.
        change: What differs: the text, the metadata, or both.
    """

    index: int
    context_id: str
    change: SyncChange


class SyncDeleteDict(TypedDict):
    """A planned delete of a stored entry.

    Attributes:
        context_id: The stored entry to delete.
        source_id: The source id the entry's block carries.
        reason: ``absent`` when the manifest does not name the source,
            ``duplicate`` when another entry of the same source survives.
    """

    context_id: str
    source_id: str
    reason: SyncDeleteReason


class PlanContextSyncDict(TypedDict):
    """The plan that brings the caller's managed entries of a range in line with a manifest.

    Attributes:
        thread_id: The planned thread.
        creates: Manifest indices of the items to store as new entries.
        updates: The stored entries to update, by manifest index.
        deletes: The first ``MAX_PLAN_DELETES`` planned deletes, ordered by
            source id and then context id.
        deletes_total: How many deletes the plan holds in all.
        unchanged_count: Manifest items whose stored entry already matches.
        kept_count: Keep items, whose stored entries the plan leaves as they are.
        foreign_count: Readable managed entries of the range owned by another
            principal, which a sync never changes.
    """

    thread_id: str
    creates: list[int]
    updates: list[SyncUpdateDict]
    deletes: list[SyncDeleteDict]
    deletes_total: int
    unchanged_count: int
    kept_count: int
    foreign_count: int


class IngestStateEntryDict(TypedDict):
    """One managed entry as the ingest state listing reports it.

    Attributes:
        context_id: The entry id.
        source_id: The source id the entry's block carries.
        text_digest: The stored ``content_hash`` of the entry's text; None for an
            entry stored before content hashes existed.
        meta_digest: The block's metadata digest.
        synced_at: When a sync last wrote the entry.
    """

    context_id: str
    source_id: str
    text_digest: str | None
    meta_digest: str
    synced_at: str


class ListIngestStateDict(TypedDict):
    """One page of the caller's managed entries of a thread.

    Attributes:
        thread_id: The listed thread.
        entries: The page's entries, ordered by source id and then context id.
        count: The number of entries on the page.
        next_cursor: The cursor of the next page; None on the last page.
    """

    thread_id: str
    entries: list[IngestStateEntryDict]
    count: int
    next_cursor: str | None
