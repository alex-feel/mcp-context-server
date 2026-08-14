"""Metadata filtering types and operators for advanced search functionality.

The ``references.context_ids`` metadata field stores UUIDv7 hex strings
(32 lowercase hex characters). Filters that target this field use string
equality or the ``array_contains`` operator with a UUIDv7 hex value, for
example::

    MetadataFilter(
        key='references.context_ids',
        operator='array_contains',
        value='0190abcdef1234567890abcdef123456',
    )

The :class:`MetadataFilter` ``value`` field accepts string values, so no
type extension is required to support UUIDv7 identifiers.
"""

import json
import math
import re
from enum import StrEnum
from typing import cast

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator

_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)

# Maximum member count for an IN / NOT_IN metadata-filter value list. Each member
# expands into one SQL bind placeholder in a single statement on both backends, so an
# unbounded client-supplied list could overflow the backend's bind limit (999
# variables on conservative SQLite builds; asyncpg's wire-protocol argument cap)
# with a driver error that is not a ControlFlowError and therefore charges the
# circuit breaker for purely invalid client input. Rejecting past the cap keeps the
# failure inside the structured validation channels (MetadataFilterValidationError
# and the search_contexts validation_errors stats), which are breaker-exempt on both
# backends. Aligned with the 100-item bounds on the sibling client-facing lists (the
# search tools' MAX_FILTER_TAGS tags cap and the batch tools' 100-entry cap).
MAX_IN_LIST_MEMBERS = 100

# Aggregate bind-parameter budget for ONE built metadata WHERE clause, enforced
# incrementally by MetadataQueryBuilder as filters are added. The per-dimension caps
# (MAX_IN_LIST_MEMBERS here; the tool-boundary MAX_FILTER_TAGS / MAX_METADATA_FILTERS
# / MAX_METADATA_KEYS caps) bound each input list individually, but capped dimensions
# MULTIPLY (filters times members), so this budget is the defense-in-depth backstop
# guaranteeing that no future cap change or new filter dimension can multiply the
# clause past the backend's per-statement bind limit (32,766 variables on a
# python.org SQLite build; asyncpg's 32,767-argument wire cap). The value sits well
# above the largest legal combination under the current caps (100 filters times 100
# members plus 100 simple keys ~= 10,100 binds) and safely below the driver limits,
# leaving headroom for the enclosing statement's non-metadata binds (tags, dates,
# LIMIT/OFFSET). Exceeding it raises ValueError from the builder, which every
# construction site routes into the structured, breaker-exempt validation-error
# channel on both backends.
MAX_METADATA_BIND_PARAMS = 30_000

# Aggregate budget for the SQL TEXT one built metadata WHERE clause may occupy,
# enforced incrementally by MetadataQueryBuilder alongside MAX_METADATA_BIND_PARAMS.
# Bind COUNT and statement SIZE are independent dimensions: on PostgreSQL a single
# bind can carry a multi-kilobyte comparison expression (the exact/double numeric
# discriminator inlines two long decimal literals), and a metadata key path is
# repeated ~10 times inside one numeric comparison, so a request that satisfies every
# per-dimension cap AND the bind budget can still assemble a statement orders of
# magnitude larger than its own payload -- a cross-backend amplification that costs
# event-loop time to build and full parse+plan time on every call, because a statement
# this large also exceeds asyncpg's cacheable-statement size. The largest legal
# combination under the current caps (100 filters times 100 float IN members, with
# ordinary short keys) builds roughly 0.6 MB, so this budget leaves meaningful headroom
# for longer key paths while bounding the pathological cases (very long key paths, or a
# future expression change that grows the per-filter SQL). Exceeding it raises
# ValueError from the builder, which every construction site routes into the
# structured, breaker-exempt validation-error channel on both backends.
MAX_METADATA_CLAUSE_CHARS = 1_000_000


def reject_out_of_int64(
    value: str | float | bool | list[str | int | float | bool] | None,
) -> None:
    """Reject integer metadata-filter values outside the signed 64-bit range on BOTH backends.

    SQLite binds a Python int as a 64-bit column value and raises OverflowError
    ('Python int too large to convert to SQLite INTEGER') for anything outside
    [-2**63, 2**63-1], aborting the whole search, while PostgreSQL binds the same
    int into an arbitrary-precision NUMERIC (or jsonb) context and matches
    normally -- a hard cross-backend divergence across every numeric operator
    (eq/ne/gt/gte/lt/lte/in/not_in/array_contains) and the simple metadata={}
    equality path. SQLite cannot store or compare a >64-bit integer exactly, so
    PostgreSQL's arbitrary-precision match has no SQLite counterpart; the only
    parity-correct contract is to reject such a value uniformly (matching the
    dotted-integer-segment rejection in validate_key). bool is an int subclass but
    is always in range and is never coerced numerically here, so it is unaffected.

    Args:
        value: A scalar metadata-filter value, or a list of them (IN/NOT_IN).

    Raises:
        ValueError: If any integer member is outside [-2**63, 2**63-1].
    """
    candidates: list[object] = list(value) if isinstance(value, list) else [value]
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, int) and not (_INT64_MIN <= candidate <= _INT64_MAX):
            raise ValueError(
                f'Integer metadata-filter value {candidate} is outside the supported signed '
                f'64-bit range [-2**63, 2**63-1]. SQLite cannot store or compare an integer this '
                f'large, so it is rejected on both backends for cross-backend parity.',
            )


def reject_non_finite(
    value: str | float | bool | list[str | int | float | bool] | None,
) -> None:
    """Reject NaN/Infinity float metadata-filter values on BOTH backends.

    SQLite binds a non-finite float as SQL NULL, so every numeric comparison
    returns no rows, while PostgreSQL treats NaN as equal to itself and greater
    than all numbers (and +/-Infinity as ordered extremes), so ``lt``/``lte``/
    ``ne`` with a NaN param matches EVERY number row -- an all-vs-none
    cross-backend divergence on the same path :func:`reject_out_of_int64` guards
    for the int64 case. Non-finite floats are not valid JSON either; rejecting
    them uniformly turns the filter into a clean ``ValueError`` instead of silent
    wrong results. ``bool`` is an int subclass, never a float, so it is
    unaffected.

    Args:
        value: A scalar metadata-filter value, or a list of them (IN/NOT_IN).

    Raises:
        ValueError: If any float member is not finite (NaN or +/-Infinity).
    """
    candidates: list[object] = list(value) if isinstance(value, list) else [value]
    for candidate in candidates:
        if isinstance(candidate, bool):
            continue
        if isinstance(candidate, float) and not math.isfinite(candidate):
            raise ValueError(
                f'Non-finite metadata-filter value {candidate!r} (NaN or Infinity) is not '
                f'supported: SQLite binds it as NULL (matching no rows) while PostgreSQL orders '
                f'NaN above all numbers, so the same filter diverges across backends.',
            )


def non_finite_metadata_error(metadata: object) -> str | None:
    """Return a message if a stored metadata value contains a non-finite float, else None.

    A store must not accept a non-finite float in metadata: ``json.dumps`` emits
    the invalid-JSON tokens ``NaN``/``Infinity`` (its ``allow_nan`` default),
    which PostgreSQL's jsonb parser REJECTS -- so the same document stores on
    SQLite but hard-fails on PostgreSQL, and the PostgreSQL failure only surfaces
    AFTER embedding/summary generation already ran (wasted model calls, an opaque
    driver error). Called in the input-validation phase BEFORE generation so the
    store fails fast with a clear message and burns no generation pass. Walks
    nested dicts and lists so a non-finite float at any depth is caught. Returns
    a message (the tool raises ``ToolError``/records a per-entry failure) rather
    than raising, mirroring the guard-message idiom and keeping ``str(e)`` out of
    the tool boundary.

    Args:
        metadata: The metadata value to validate (any JSON-compatible structure).

    Returns:
        An operator-facing message on the first non-finite float found, else None.
    """
    if isinstance(metadata, bool):
        return None
    if isinstance(metadata, float):
        if not math.isfinite(metadata):
            return (
                f'Non-finite float {metadata!r} (NaN or Infinity) in metadata is not '
                f'supported: it serializes to invalid JSON that PostgreSQL rejects, so the '
                f'same entry would store on SQLite but fail on PostgreSQL.'
            )
        return None
    if isinstance(metadata, dict):
        for item in cast('dict[object, object]', metadata).values():
            message = non_finite_metadata_error(item)
            if message is not None:
                return message
    elif isinstance(metadata, (list, tuple)):
        for item in cast('list[object]', metadata):
            message = non_finite_metadata_error(item)
            if message is not None:
                return message
    return None


# PostgreSQL's INTEGER is 32-bit, so a JSON number outside this range renders as
# text the ``::INTEGER`` expression-index cast rejects as out of range.
_PG_INT32_MIN = -(2 ** 31)
_PG_INT32_MAX = 2 ** 31 - 1

# The whitespace PostgreSQL's scalar input functions trim, which is ASCII ONLY: their
# scanners run under the C locale and test isspace() on single bytes, so U+00A0, U+3000,
# U+2028, U+0085, U+205F and the other Unicode space characters are ordinary content that
# makes the cast fail. Python's argument-less str.strip() removes all of those as well,
# so trimming with it would accept values PostgreSQL rejects -- exactly the divergence
# this validation exists to close. Verified against PostgreSQL 18: chr(9..13) and chr(32)
# are trimmed on integer, boolean and numeric input; chr(160), chr(133) and chr(8232) are
# not.
_PG_ASCII_WHITESPACE = ' \t\n\v\f\r'

# The widest significant digit run each accepted integer base can hold inside a 32-bit
# INTEGER. A longer run is out of range by inspection, which also keeps the int()
# conversion below clear of the interpreter's string-to-int digit ceiling for an absurdly
# long decimal run.
_PG_INT32_MAX_SIGNIFICANT_DIGITS = {2: 32, 8: 11, 10: 10, 16: 8}

# The literals PostgreSQL's boolean input accepts, case-insensitively and with
# surrounding whitespace trimmed -- every unambiguous prefix of "true"/"false"/"yes"/"no",
# the two-character-and-longer prefixes of "on"/"off" (a lone 'o' is ambiguous between
# them and is rejected), and the single characters '1'/'0'.
_PG_BOOLEAN_LITERALS = frozenset({
    't', 'tr', 'tru', 'true', 'y', 'ye', 'yes', 'on', '1',
    'f', 'fa', 'fal', 'fals', 'false', 'n', 'no', 'of', 'off', '0',
})

# The non-finite values PostgreSQL's NUMERIC input accepts by name.
_PG_NUMERIC_SPECIALS = frozenset({
    'nan', 'inf', '+inf', '-inf', 'infinity', '+infinity', '-infinity',
})

# PostgreSQL 16 extended the INTEGER and NUMERIC input functions with non-decimal
# literals (0x hexadecimal, 0o octal, 0b binary) and '_' digit separators, so a value
# like '0x10' or '1_000' casts successfully and must not be refused here -- refusing it
# would block a store on BOTH backends that previously succeeded on both. A separator
# may sit directly after a base prefix but never leads, trails, or doubles.
_PG_NON_DECIMAL_INTEGER_RES = (
    (16, re.compile(r'^(?P<sign>[+-]?)0[xX](?P<digits>(?:_?[0-9a-fA-F])+)$')),
    (8, re.compile(r'^(?P<sign>[+-]?)0[oO](?P<digits>(?:_?[0-7])+)$')),
    (2, re.compile(r'^(?P<sign>[+-]?)0[bB](?P<digits>(?:_?[01])+)$')),
)
_PG_DECIMAL_INTEGER_RE = re.compile(r'^(?P<sign>[+-]?)(?P<digits>[0-9](?:_?[0-9])*)$')
_PG_DECIMAL_DIGITS = r'[0-9](?:_?[0-9])*'
_PG_NUMERIC_RE = re.compile(
    rf'^[+-]?(?:{_PG_DECIMAL_DIGITS}(?:\.(?:{_PG_DECIMAL_DIGITS})?)?|\.{_PG_DECIMAL_DIGITS})'
    rf'(?:[eE][+-]?{_PG_DECIMAL_DIGITS})?$',
)

# The METADATA_INDEXED_FIELDS type hints app/migrations/metadata.py turns into a hard
# SQL cast inside the PostgreSQL expression index. 'string' adds no cast, and
# 'array'/'object' build no expression index at all (the always-present GIN index
# serves them), so neither can fail a cast.
_PG_CAST_HINTS = frozenset({'integer', 'boolean', 'float'})


def _truncate_for_message(text: str, limit: int = 60) -> str:
    """Shorten a client value so it can be quoted inside an error message.

    Args:
        text: The value's text rendering.
        limit: Maximum characters to keep before eliding.

    Returns:
        The text, elided with a trailing ellipsis when longer than ``limit``.
    """
    return text if len(text) <= limit else f'{text[:limit]}...'


def _pg_integer_input(candidate: str) -> tuple[bool, int | None]:
    """Parse a text value the way PostgreSQL's INTEGER input function would.

    Accepts everything PostgreSQL 16+ accepts: an optional sign, then either a decimal
    run or a ``0x``/``0o``/``0b`` non-decimal run, with ``_`` digit separators permitted
    between digits and directly after a base prefix but never leading, trailing, or
    doubled. Surrounding whitespace is the caller's responsibility.

    Args:
        candidate: The whitespace-trimmed text rendering of the value.

    Returns:
        ``(matches_grammar, value)``. ``matches_grammar`` is False when PostgreSQL would
        report invalid input syntax. ``value`` is the parsed integer, or None when the
        literal is well-formed but carries more significant digits than a 32-bit INTEGER
        can hold -- which is decided by inspecting the digit run, so an absurdly long
        decimal literal never reaches the interpreter's string-to-int digit ceiling.
    """
    for base, pattern in _PG_NON_DECIMAL_INTEGER_RES:
        match = pattern.match(candidate)
        if match is not None:
            digits = match['digits'].replace('_', '').lstrip('0')
            if len(digits) > _PG_INT32_MAX_SIGNIFICANT_DIGITS[base]:
                return True, None
            magnitude = int(digits, base) if digits else 0
            return True, -magnitude if match['sign'] == '-' else magnitude
    decimal = _PG_DECIMAL_INTEGER_RE.match(candidate)
    if decimal is None:
        return False, None
    digits = decimal['digits'].replace('_', '').lstrip('0')
    if len(digits) > _PG_INT32_MAX_SIGNIFICANT_DIGITS[10]:
        return True, None
    magnitude = int(digits) if digits else 0
    return True, -magnitude if decimal['sign'] == '-' else magnitude


def pg_indexed_cast_error(field: str, value: object, type_hint: str) -> str | None:
    """Return an error message when a value cannot survive its index's SQL cast.

    A ``METADATA_INDEXED_FIELDS`` entry may carry a type hint, and on PostgreSQL
    ``integer``/``boolean``/``float`` become a hard cast inside the expression index
    (``((metadata->>'<field>')::INTEGER)`` and siblings). PostgreSQL evaluates an
    expression index ON INSERT, so a value that satisfies the index's
    ``IS NOT NULL`` predicate but does not survive the cast aborts the write with a
    raw driver error -- after a full generation pass, inside the transaction, and
    charging the circuit breaker -- while SQLite's uncast ``json_extract`` index
    stores the identical value happily. Validating here at the write boundary makes
    both backends accept and reject identically, before any generation runs.

    The checks mirror PostgreSQL's own input parsers applied to the TEXT rendering
    ``->>`` produces, so a value PostgreSQL would accept is not refused: a JSON
    string ``"5"`` renders as ``5`` and casts fine, whereas a JSON number ``5.5``
    renders as ``5.5`` and does not cast to INTEGER.

    Args:
        field: The metadata key, used in the message.
        value: The client-supplied value stored under that key.
        type_hint: The configured type hint for the field.

    Returns:
        An error message when the value cannot be cast, or None when it can (or when
        the hint produces no cast).
    """
    if type_hint not in _PG_CAST_HINTS:
        return None
    # A JSON null renders as SQL NULL, which the index's own
    # ``WHERE metadata->>'<field>' IS NOT NULL`` predicate excludes, so no cast runs.
    if value is None:
        return None

    # Booleans are checked before the int branch: bool is an int subclass in Python,
    # but JSON true/false render as the text 'true'/'false', not as 1/0.
    if isinstance(value, bool):
        rendered = 'true' if value else 'false'
    elif isinstance(value, str):
        rendered = value
    elif isinstance(value, (int, float)):
        rendered = json.dumps(value)
    else:
        # A JSON array or object renders as its JSON text, which casts to no scalar
        # type; any other Python type could not have arrived as JSON at all.
        return (
            f'metadata field {field!r} is indexed as {type_hint} and a '
            f'{type(value).__name__} value cannot be stored in it'
        )

    candidate = rendered.strip(_PG_ASCII_WHITESPACE)
    invalid = (
        f'metadata field {field!r} is indexed as {type_hint} and its value is not a '
        f'valid {type_hint}: {_truncate_for_message(rendered)!r}'
    )

    if type_hint == 'integer':
        matches_grammar, parsed = _pg_integer_input(candidate)
        if not matches_grammar:
            return invalid
        if parsed is None or not (_PG_INT32_MIN <= parsed <= _PG_INT32_MAX):
            return (
                f'metadata field {field!r} is indexed as integer and its value is out of '
                f'range: {_truncate_for_message(rendered)} (the supported range is '
                f'{_PG_INT32_MIN} to {_PG_INT32_MAX})'
            )
        return None

    if type_hint == 'boolean':
        return None if candidate.lower() in _PG_BOOLEAN_LITERALS else invalid

    # 'float' -> ::NUMERIC. NUMERIC accepts every INTEGER literal form as well, including
    # the non-decimal bases, so the integer grammar is consulted before declaring the
    # value invalid ('0x10'::numeric is 16 on PostgreSQL 16+).
    if _PG_NUMERIC_RE.match(candidate) or candidate.lower() in _PG_NUMERIC_SPECIALS:
        return None
    return None if _pg_integer_input(candidate)[0] else invalid


def pg_indexed_metadata_text(value: object) -> str | None:
    """Return the text ``metadata->>'<field>'`` yields for a decoded JSON value.

    The single source of truth for "what does the PostgreSQL expression index on
    ``context_entries((metadata->>'<field>'))`` actually store". PostgreSQL's ``->>``
    operator returns a JSON string UNQUOTED and every other JSON value in its
    serialized form, so a list or object stored under an indexed key is indexed as
    its whole serialized text -- which a check that inspects only ``str`` values
    misses entirely, letting an oversized container reach the INSERT and abort it
    with a raw btree index-tuple error that SQLite's uncast ``json_extract`` index
    never produces. A JSON ``null`` yields SQL NULL, which the expression index
    excludes through its ``WHERE ... IS NOT NULL`` predicate, so it has no indexed
    text at all.

    Both the write-boundary guard and the migration CLI's pre-check measure the value
    through this function, so the two can never disagree about what the index holds.

    Args:
        value: A decoded top-level metadata value.

    Returns:
        The text the expression index would hold, or None when the value indexes as
        SQL NULL.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    # jsonb renders container members with ', ' and ': ' separators, so encoding with
    # the same separators measures what the index actually stores.
    return json.dumps(value, ensure_ascii=False, separators=(', ', ': '))


def pg_bind_reject_reason(text: str) -> str | None:
    """Return why a string cannot be stored or bound on PostgreSQL, else None.

    Two byte sequences that Python strings, SQLite, and JSON all accept are fatal
    on PostgreSQL: an embedded NUL (U+0000), which PostgreSQL TEXT and jsonb both
    reject (asyncpg raises for any NUL-carrying text bind, and jsonb rejects the
    ``\\u0000`` escape ``json.dumps`` emits), and an unpaired UTF-16 surrogate,
    which is not encodable as UTF-8 at all (asyncpg and SQLite both raise a
    ``UnicodeEncodeError`` at the driver boundary). SQLite stores and matches the
    NUL case without complaint, so a value carrying either sequence is a
    cross-backend divergence AND -- because the driver error is not a
    :class:`~app.errors.ControlFlowError` -- charges the circuit breaker on
    PostgreSQL. The NUL check runs first because a NUL is itself UTF-8-encodable,
    so ``str.encode`` would not catch it.

    Args:
        text: The string to inspect.

    Returns:
        A short reason phrase on the first offending sequence found, else None.
    """
    if '\x00' in text:
        return 'an embedded NUL (U+0000) character'
    try:
        text.encode('utf-8')
    except UnicodeEncodeError:
        return 'an unpaired UTF-16 surrogate (a code point that cannot be encoded as UTF-8)'
    return None


def reject_nul(
    value: str | float | bool | list[str | int | float | bool] | None,
) -> None:
    """Reject NUL/non-UTF-8-encodable string metadata-filter values on BOTH backends.

    A NUL (U+0000) or unpaired UTF-16 surrogate in a filter value binds and
    matches cleanly on SQLite but aborts the query on PostgreSQL (asyncpg raises
    ``CharacterNotInRepertoireError`` for the raw bind, or a jsonb parse error for
    the ``array_contains`` cast), and because that driver error is not a
    :class:`~app.errors.ControlFlowError` it is charged to the process-wide
    circuit breaker -- a client-input value that opens the breaker into an outage.
    Rejecting it uniformly turns the filter into a clean ``ValueError`` on both
    backends, the same cross-backend-parity contract :func:`reject_out_of_int64`
    and :func:`reject_non_finite` enforce for the int64 and non-finite cases on
    this exact path. Non-string members (int/float/bool) carry no such byte and
    are skipped.

    Args:
        value: A scalar metadata-filter value, or a list of them (IN/NOT_IN).

    Raises:
        ValueError: If any string member contains a NUL or an unpaired surrogate.
    """
    candidates: list[object] = list(value) if isinstance(value, list) else [value]
    for candidate in candidates:
        if isinstance(candidate, str):
            reason = pg_bind_reject_reason(candidate)
            if reason is not None:
                raise ValueError(
                    f'String metadata-filter value contains {reason}, which PostgreSQL cannot bind '
                    f'as a query parameter: SQLite would match it while PostgreSQL aborts the query, '
                    f'so it is rejected on both backends for cross-backend parity.',
                )


def unstorable_string_error(value: object) -> str | None:
    """Return a message if a stored value contains a PostgreSQL-unstorable string, else None.

    The store/update write path must not accept a string carrying an embedded NUL
    (U+0000) or an unpaired UTF-16 surrogate: PostgreSQL rejects it (TEXT bind or
    jsonb parse), so the same entry stores on SQLite but hard-fails on PostgreSQL,
    and with generation enabled the failure only surfaces AFTER a wasted
    embedding/summary pass, inside the transaction where a non-ControlFlowError
    charges the circuit breaker. This is the exact ``json.dumps`` accepts /
    PostgreSQL rejects divergence class :func:`non_finite_metadata_error` guards
    for NaN/Infinity, so it is called at the same pre-generation call sites and
    returns a message (the tool raises ``ToolError``) rather than raising. Walks
    dict KEYS and values and list items so an offending string at any nesting
    depth is caught; a scalar string (``text_content``, ``thread_id``, a tag) is
    validated directly.

    Args:
        value: The stored value to validate (a string, tag list, or metadata
            structure of arbitrary depth).

    Returns:
        An operator-facing message on the first offending string found, else None.
    """
    if isinstance(value, str):
        reason = pg_bind_reject_reason(value)
        if reason is not None:
            return (
                f'A string value contains {reason}: PostgreSQL cannot store it, so the same entry '
                f'would store on SQLite but fail on PostgreSQL. Remove it before storing.'
            )
        return None
    if isinstance(value, dict):
        for key, item in cast('dict[object, object]', value).items():
            if isinstance(key, str):
                reason = pg_bind_reject_reason(key)
                if reason is not None:
                    return (
                        f'A metadata key contains {reason}: PostgreSQL cannot store it, so the same '
                        f'entry would store on SQLite but fail on PostgreSQL. Remove it before storing.'
                    )
            message = unstorable_string_error(item)
            if message is not None:
                return message
    elif isinstance(value, (list, tuple)):
        for item in cast('list[object]', value):
            message = unstorable_string_error(item)
            if message is not None:
                return message
    return None


def sanitize_pg_unstorable_text(value: str) -> str:
    """Repair a PostgreSQL-unstorable string so it binds identically on both backends.

    The counterpart to :func:`unstorable_string_error` for a DIFFERENT source of the
    same divergence. ``unstorable_string_error`` REJECTS a client-supplied string,
    because the client can fix its own input. This function REPAIRS a value the
    server itself GENERATED -- a model-produced summary or per-node summary -- where
    rejection would be wrong: the client's own text is valid, so an abort-mandatory
    store must not be refused because the summary model happened to emit a stray NUL
    (U+0000) or an unpaired UTF-16 surrogate. Left unrepaired, such a byte stores on
    SQLite yet aborts the PostgreSQL bind inside the transaction, charging the
    circuit breaker for a provider quirk the client did not cause.

    The repair is idempotent and a no-op for a clean string (the common case): the
    :func:`pg_bind_reject_reason` probe returns fast when there is nothing to fix.

    Args:
        value: A server-generated string to make storable on both backends.

    Returns:
        ``value`` unchanged when it is already storable, otherwise a copy with
        embedded NULs stripped and any unpaired surrogate replaced by a placeholder,
        which is guaranteed NUL-free and UTF-8-encodable.
    """
    if pg_bind_reject_reason(value) is None:
        return value
    # Drop embedded NULs, then round-trip through UTF-8 with the ``replace`` error
    # handler so any unpaired UTF-16 surrogate becomes a placeholder; the result is
    # guaranteed NUL-free and UTF-8-encodable, so both backends store it identically.
    repaired = value.replace('\x00', '')
    return repaired.encode('utf-8', 'replace').decode('utf-8')


class MetadataOperator(StrEnum):
    """Comprehensive metadata comparison operators.

    Supports 16 different operators for flexible metadata filtering.
    Note: REGEX operator removed due to SQLite limitations.
    """

    EQ = 'eq'  # Equals (default)
    NE = 'ne'  # Not equals
    GT = 'gt'  # Greater than
    GTE = 'gte'  # Greater than or equal
    LT = 'lt'  # Less than
    LTE = 'lte'  # Less than or equal
    IN = 'in'  # Value in list
    NOT_IN = 'not_in'  # Value not in list
    EXISTS = 'exists'  # Key exists
    NOT_EXISTS = 'not_exists'  # Key doesn't exist
    CONTAINS = 'contains'  # String contains
    STARTS_WITH = 'starts_with'  # String starts with
    ENDS_WITH = 'ends_with'  # String ends with
    IS_NULL = 'is_null'  # Value is null
    IS_NOT_NULL = 'is_not_null'  # Value is not null
    ARRAY_CONTAINS = 'array_contains'  # Array contains element


class MetadataFilter(BaseModel):
    """Advanced metadata filter specification.

    Supports complex filtering with specific operators and nested JSON paths.

    Unknown keys are REJECTED (``extra='forbid'``): with Pydantic's default
    ``extra='ignore'`` a misspelled key -- ``'op'`` for ``'operator'``, or
    ``'case-sensitive'`` for ``'case_sensitive'`` -- would be silently dropped and
    the field would keep its default, silently converting the filter (e.g. a
    ``gt`` filter runs as ``eq``) and returning a wrong result set with no error.
    That is the same silent-filter-alteration class the value validators below
    exist to prevent; a typo instead raises ``ValidationError``, which every
    construction site routes into the structured validation-error channel.
    """

    model_config = ConfigDict(extra='forbid')

    key: str = Field(
        ...,
        description='JSON path to metadata field (e.g., "status" or "user.preferences.theme")',
    )
    operator: MetadataOperator = Field(default=MetadataOperator.EQ, description='Comparison operator')
    value: str | int | float | bool | list[str | int | float | bool] | None = Field(
        default=None,
        validate_default=True,
        description='Value to compare against (not needed for EXISTS, IS_NULL, etc.)',
    )
    case_sensitive: bool = Field(default=False, description='Case sensitivity for string operations')

    @field_validator('key')
    @classmethod
    def validate_key(cls, v: str) -> str:
        """Validate JSON path key for safety."""
        # Validate required key field: must contain non-whitespace characters
        # Since v is typed as str (not str | None) by Pydantic, it cannot be None
        # We only need to check if it's empty or contains only whitespace
        if not v.strip():
            raise ValueError('Metadata key cannot be empty')

        # Basic validation to prevent obvious SQL injection attempts
        # Allow alphanumeric, dots, underscores, and hyphens for JSON paths.
        # fullmatch (not match) so a trailing newline is rejected: `$` also
        # matches immediately before a single trailing '\n', which would let a
        # key like 'a.status\n' through and diverge across backends (twin of the
        # MetadataQueryBuilder._is_safe_key guard on the simple-filter path).
        import re

        if not re.fullmatch(r'[a-zA-Z0-9_.-]+', v):
            raise ValueError(
                f'Invalid metadata key: {v}. Only alphanumeric characters, dots, underscores, and hyphens are allowed.',
            )

        # Reject integer path segments after the first: such a segment (e.g. 'items.0',
        # 'a.-1') array-indexes on PostgreSQL but resolves to a literal object key on
        # SQLite, a silent cross-backend divergence. Dotted array indexing was never a
        # documented capability, so forbid it on both backends (parity by construction).
        if any(re.fullmatch(r'-?\d+', seg) for seg in v.split('.')[1:]):
            raise ValueError(
                f'Invalid metadata key: {v}. Numeric path segments after the first '
                f'(e.g. "items.0") are not allowed: they array-index on PostgreSQL but '
                f'resolve to a literal object key on SQLite.',
            )

        # Reject empty path segments (a leading '.x', trailing 'x.', consecutive 'a..b',
        # or the degenerate '.'/'..'). On PostgreSQL the metadata #>> accessor builds an
        # array literal like '{a,,b}' that the array-literal parser rejects (a raw error
        # surfaced to the client), while SQLite either silently treats a trailing empty
        # segment as an absent path or raises a different JSON-path error -- a silent
        # cross-backend divergence. Forbid on both backends (parity by construction),
        # mirroring the numeric-segment rejection above.
        if '' in v.split('.'):
            raise ValueError(
                f'Invalid metadata key: {v}. Empty path segments (leading, trailing, or '
                f'consecutive dots) are not allowed.',
            )

        return v.strip()

    @field_validator('value')
    @classmethod
    def validate_value_for_operator(
        cls,
        v: str | float | bool | list[str | int | float | bool] | None,
        info: ValidationInfo,
    ) -> str | int | float | bool | list[str | int | float | bool] | None:
        """Validate value based on operator requirements."""
        operator = info.data.get('operator', MetadataOperator.EQ)

        # Operators that don't require a value
        if operator in (
            MetadataOperator.EXISTS,
            MetadataOperator.NOT_EXISTS,
            MetadataOperator.IS_NULL,
            MetadataOperator.IS_NOT_NULL,
        ):
            return None  # Value is ignored for these operators

        # IN and NOT_IN require list values
        if operator in (MetadataOperator.IN, MetadataOperator.NOT_IN) and not isinstance(v, list):
            raise ValueError(f'Operator {operator} requires a list value')
        if operator in (MetadataOperator.IN, MetadataOperator.NOT_IN) and isinstance(v, list) and not v:
            raise ValueError(f'Operator {operator} requires a non-empty list')

        # Cap the IN / NOT_IN membership list so it can never expand into an
        # oversized single-statement placeholder run -- see MAX_IN_LIST_MEMBERS.
        if (
            operator in (MetadataOperator.IN, MetadataOperator.NOT_IN)
            and isinstance(v, list)
            and len(v) > MAX_IN_LIST_MEMBERS
        ):
            raise ValueError(
                f'Operator {operator} accepts at most {MAX_IN_LIST_MEMBERS} list members, '
                f'got {len(v)}; narrow the membership list',
            )

        # String operators require string values. None is rejected too: a
        # missing value would otherwise produce no SQL condition and silently
        # drop the filter, leaving the query unrestricted (over-broad results).
        if (
            operator in (MetadataOperator.CONTAINS, MetadataOperator.STARTS_WITH, MetadataOperator.ENDS_WITH)
            and not isinstance(v, str)
        ):
            raise ValueError(f'Operator {operator} requires a string value')

        # Equality and comparison operators require a scalar value. A list here
        # matches no dispatch branch and silently drops the filter; callers that
        # want membership must use IN / NOT_IN instead.
        if (
            operator in (
                MetadataOperator.EQ,
                MetadataOperator.NE,
                MetadataOperator.GT,
                MetadataOperator.GTE,
                MetadataOperator.LT,
                MetadataOperator.LTE,
            )
            and isinstance(v, list)
        ):
            raise ValueError(f'Operator {operator} requires a scalar value, not a list')

        # Equality and comparison operators require a non-null value. For
        # GT/GTE/LT/LTE None would be str()-coerced to the literal 'None'; for
        # EQ/NE None binds SQL NULL, and `<expr> = NULL` / `<expr> != NULL` are
        # never TRUE under SQL three-valued logic, so the filter always returns
        # zero rows. Use IS_NULL / IS_NOT_NULL for null checks instead.
        if (
            operator in (
                MetadataOperator.EQ,
                MetadataOperator.NE,
                MetadataOperator.GT,
                MetadataOperator.GTE,
                MetadataOperator.LT,
                MetadataOperator.LTE,
            )
            and v is None
        ):
            raise ValueError(f'Operator {operator} requires a non-null scalar value')

        # Ordered comparison operators reject a boolean. bool is a subclass of int,
        # so without this guard True/False would silently coerce to 1/0 and be
        # ordered against stored JSON numbers -- a meaningless comparison. EQ/NE
        # intentionally still accept a boolean (boolean-typed equality via the
        # bool-first builder guard); use those, or IS_NULL / IS_NOT_NULL, instead.
        if (
            operator in (
                MetadataOperator.GT,
                MetadataOperator.GTE,
                MetadataOperator.LT,
                MetadataOperator.LTE,
            )
            and isinstance(v, bool)
        ):
            raise ValueError(f'Operator {operator} requires a numeric or string value, not a boolean')

        # ARRAY_CONTAINS requires a single scalar value (not a list)
        if operator == MetadataOperator.ARRAY_CONTAINS:
            if isinstance(v, list):
                raise ValueError('Operator array_contains requires a single value, not a list')
            if v is None:
                raise ValueError('Operator array_contains requires a non-null value')

        # Reject integer values outside the signed 64-bit range, non-finite
        # floats (NaN/Infinity), and strings carrying a NUL or unpaired UTF-16
        # surrogate, on BOTH backends. The identical guards run at the simple
        # metadata={} equality builder (add_simple_filter), so the advanced
        # metadata_filters path and the simple path reject the same divergent
        # inputs -- see reject_out_of_int64 / reject_non_finite / reject_nul.
        reject_out_of_int64(v)
        reject_non_finite(v)
        reject_nul(v)

        return v
