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

import math
from enum import StrEnum
from typing import cast

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator

_INT64_MAX = (1 << 63) - 1
_INT64_MIN = -(1 << 63)


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
    """

    key: str = Field(
        ...,
        description='JSON path to metadata field (e.g., "status" or "user.preferences.theme")',
    )
    operator: MetadataOperator = Field(default=MetadataOperator.EQ, description='Comparison operator')
    value: str | int | float | bool | list[str | int | float | bool] | None = Field(
        default=None,
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
