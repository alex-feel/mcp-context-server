"""Centralized identifier handling for context entries.

Public identifier format
    Context-entry IDs are UUIDv7 values displayed as a 32-character lowercase
    hex string without hyphens (matching the regex ``^[0-9a-f]{32}$``). All
    callers exchange identifiers in this canonical form.

Accepted input formats
    Tool boundaries accept either the canonical 32-char hex form or the
    36-char hyphenated UUID form. Whitespace is stripped and case is folded
    to lowercase before validation. Prefix lookups require 8 to 31 hex
    characters.

Lowercase output is required
    ``normalize_id`` always emits lowercase hex. Under SQLite TEXT BINARY
    collation, uppercase ``A-F`` sorts before lowercase ``a-f``; mixing
    cases would corrupt any ``id > ?`` ordering comparison performed by the
    storage layer.

Module contents
    - :data:`ContextId` -- type alias for the canonical hex string.
    - :func:`generate_id` -- produce a fresh UUIDv7 in canonical form.
    - :func:`generate_id_with_timestamp` -- produce a UUIDv7 anchored to a
      specific :class:`datetime.datetime`.
    - :func:`normalize_id` -- validate and canonicalize any accepted input.
    - :func:`is_id_prefix` -- predicate for partial-hex prefix strings.
    - :func:`resolve_prefix` -- resolve a prefix to a unique full ID via a
      repository implementing the :class:`_PrefixResolverRepo` protocol.
"""

from datetime import datetime
from typing import Protocol
from typing import runtime_checkable

import uuid_utils

type ContextId = str  # PEP 695 type alias


@runtime_checkable
class _PrefixResolverRepo(Protocol):
    """Repository capability required by :func:`resolve_prefix`.

    Defined here as a Protocol so that :mod:`app.ids` has no hard
    dependency on the repository layer.
    """

    async def find_ids_by_prefix(self, prefix: ContextId, limit: int = 2) -> list[ContextId]:
        """Return up to ``limit`` context-entry IDs starting with ``prefix``.

        Returning more than one indicates ambiguity. Implementations should
        cap the result at ``limit`` (default 2) since the caller only needs
        to distinguish unique from ambiguous matches.
        """
        ...


def generate_id() -> ContextId:
    """Generate a fresh UUIDv7 as a 32-character lowercase hex string.

    Returns:
        A string matching ``^[0-9a-f]{32}$``.
    """
    return uuid_utils.uuid7().hex


def generate_id_with_timestamp(created_at: datetime) -> ContextId:
    """Generate a UUIDv7 whose embedded timestamp matches ``created_at``.

    Produces deterministic ordering for callers that need to assign IDs
    to records with predetermined creation times.

    Args:
        created_at: A timezone-aware datetime (UTC recommended). The
            timezone offset is honored by :meth:`datetime.timestamp`.

    Returns:
        A 32-character lowercase hex string matching ``^[0-9a-f]{32}$``.
        The embedded UUIDv7 timestamp decodes to the same calendar
        millisecond as ``created_at``.
    """
    # uuid_utils.uuid7 expects ``timestamp`` in UNIX SECONDS (an integer)
    # and ``nanos`` as the nanosecond fraction within that second. Passing
    # milliseconds here would place the embedded timestamp roughly 1000
    # times further into the future, producing UUIDs whose year decodes to
    # ~50,000 AD instead of the input year.
    # Upstream tracker on the timestamp parameter's units: https://github.com/aminalaee/uuid-utils/issues/73
    seconds = int(created_at.timestamp())
    nanos = created_at.microsecond * 1_000  # microseconds -> nanoseconds
    return uuid_utils.uuid7(timestamp=seconds, nanos=nanos).hex


def normalize_id(value: str) -> ContextId:
    """Validate ``value`` and return it in canonical 32-char lowercase hex form.

    Accepts either the 32-character hex form or the 36-character hyphenated
    UUID form. Whitespace is stripped and case is folded to lowercase
    before validation. The function is idempotent on already-normalized
    input.

    The lowercase output is required: under SQLite TEXT BINARY collation,
    uppercase ``A-F`` sorts before lowercase ``a-f``, so any ``id > ?``
    ordering comparison would behave incorrectly if uppercase IDs were
    accepted.

    Args:
        value: A UUID in 32-char hex form (e.g.
            ``'0190abcdef1234567890abcdef123456'``) or 36-char hyphenated
            form (e.g. ``'0190abcd-ef12-3456-7890-abcdef123456'``).

    Returns:
        A 32-character lowercase hex string matching ``^[0-9a-f]{32}$``.

    Raises:
        ValueError: If ``value`` is not a valid 32-char hex or 36-char
            hyphenated UUID after normalization.
    """
    s = value.strip().lower()
    if len(s) == 36 and s.count('-') == 4:
        s = s.replace('-', '')
    if len(s) != 32:
        raise ValueError(f'Invalid UUID identifier length (expected 32 hex chars after normalization): {value!r}')
    if any(c not in '0123456789abcdef' for c in s):
        raise ValueError(f'Invalid UUID identifier (non-hex character): {value!r}')
    return s


def is_id_prefix(value: str) -> bool:
    """Return ``True`` if ``value`` is a UUID hex prefix of length 8 to 31.

    Used at tool boundaries to decide whether to call :func:`normalize_id`
    (full UUID) or :func:`resolve_prefix` (partial hex prefix). Whitespace
    is stripped and case is folded internally; callers do not need to
    pre-normalize.

    Args:
        value: Candidate prefix string.

    Returns:
        ``True`` iff ``value`` has length 8 through 31 inclusive and
        contains only hex characters (after lowercasing). ``False`` for
        full 32-char UUIDs, 36-char hyphenated UUIDs, prefixes shorter
        than 8 characters, and any string containing non-hex characters.
    """
    s = value.strip().lower()
    if not (8 <= len(s) < 32):
        return False
    return all(c in '0123456789abcdef' for c in s)


async def resolve_prefix(prefix: str, repo: _PrefixResolverRepo) -> ContextId:
    """Resolve an 8-31 char hex prefix to a unique full UUID via ``repo``.

    Args:
        prefix: A hex string 8 through 31 characters long (validated via
            :func:`is_id_prefix`). Whitespace is stripped and case is
            folded internally.
        repo: Object implementing the :class:`_PrefixResolverRepo`
            protocol; provides :meth:`find_ids_by_prefix`.

    Returns:
        A 32-character lowercase hex string identifying the unique context
        entry whose ``id`` starts with ``prefix``.

    Raises:
        ValueError: If ``prefix`` is not a valid 8-31 char hex prefix
            (length below 8, length 32 or above, contains non-hex
            characters), if no entry matches, or if more than one entry
            matches the prefix.
    """
    s = prefix.strip().lower()
    if not is_id_prefix(s):
        raise ValueError(f'Invalid UUID prefix (must be 8-31 hex chars): {prefix!r}')
    matches = await repo.find_ids_by_prefix(s, limit=2)
    if not matches:
        raise ValueError(f'No context entry matches prefix {prefix!r}')
    if len(matches) > 1:
        raise ValueError(f'Ambiguous prefix {prefix!r} matches multiple entries')
    return matches[0]
