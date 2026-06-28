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

from enum import StrEnum

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
        # Allow alphanumeric, dots, underscores, and hyphens for JSON paths
        import re

        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
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

        # ARRAY_CONTAINS requires a single scalar value (not a list)
        if operator == MetadataOperator.ARRAY_CONTAINS:
            if isinstance(v, list):
                raise ValueError('Operator array_contains requires a single value, not a list')
            if v is None:
                raise ValueError('Operator array_contains requires a non-null value')

        # Reject integer values outside the signed 64-bit range on BOTH backends.
        # The identical guard runs at the simple metadata={} equality builder
        # (MetadataQueryBuilder.add_simple_filter), so the advanced metadata_filters
        # path and the simple path reject an out-of-range integer the same way --
        # see reject_out_of_int64 for the full cross-backend rationale.
        reject_out_of_int64(v)

        return v
