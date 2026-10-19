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

The protocol constants bound what one call carries, because the transports a
client uses limit request and response sizes. This module imports neither the
server's settings nor its backends, so the client can import it.
"""

import re
from datetime import datetime
from typing import TypeGuard
from typing import cast

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
