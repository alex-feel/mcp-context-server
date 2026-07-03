"""Operator/value type validation for ``MetadataFilter``.

A value whose type does not match its operator must be rejected at construction
(a loud ``ValidationError``) rather than passing validation and then producing no
SQL condition, which silently drops the filter and returns over-broad results.
"""

import pytest
from pydantic import ValidationError

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator


@pytest.mark.parametrize(
    'operator',
    [
        MetadataOperator.EQ,
        MetadataOperator.NE,
        MetadataOperator.GT,
        MetadataOperator.GTE,
        MetadataOperator.LT,
        MetadataOperator.LTE,
    ],
)
def test_scalar_operator_rejects_list_value(operator: MetadataOperator) -> None:
    """Equality/comparison operators reject a list value (use IN / NOT_IN)."""
    with pytest.raises(ValidationError):
        MetadataFilter(key='status', operator=operator, value=['a', 'b'])


@pytest.mark.parametrize(
    'operator',
    [
        MetadataOperator.CONTAINS,
        MetadataOperator.STARTS_WITH,
        MetadataOperator.ENDS_WITH,
    ],
)
def test_string_operator_rejects_none_value(operator: MetadataOperator) -> None:
    """String operators reject a None value (it would silently drop the filter)."""
    with pytest.raises(ValidationError):
        MetadataFilter(key='status', operator=operator, value=None)


def test_scalar_operator_accepts_scalar_value() -> None:
    """A scalar value remains valid for EQ (control)."""
    metadata_filter = MetadataFilter(key='status', operator=MetadataOperator.EQ, value='done')
    assert metadata_filter.value == 'done'


def test_in_operator_still_accepts_list() -> None:
    """IN continues to accept a list value (unchanged)."""
    metadata_filter = MetadataFilter(key='status', operator=MetadataOperator.IN, value=['a', 'b'])
    assert metadata_filter.value == ['a', 'b']


def test_contains_still_accepts_string_value() -> None:
    """CONTAINS continues to accept a string value (control)."""
    metadata_filter = MetadataFilter(key='note', operator=MetadataOperator.CONTAINS, value='hello')
    assert metadata_filter.value == 'hello'


@pytest.mark.parametrize(
    'operator',
    [
        MetadataOperator.GT,
        MetadataOperator.GTE,
        MetadataOperator.LT,
        MetadataOperator.LTE,
    ],
)
def test_comparison_operator_rejects_none_value(operator: MetadataOperator) -> None:
    """Comparison operators reject None: it would be str()-coerced to the literal
    'None' and compared as text, returning wrong rows (use IS_NULL/IS_NOT_NULL)."""
    with pytest.raises(ValidationError):
        MetadataFilter(key='priority', operator=operator, value=None)


def test_comparison_operator_accepts_scalar_value() -> None:
    """A numeric scalar remains valid for GT (control)."""
    metadata_filter = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
    assert metadata_filter.value == 5


@pytest.mark.parametrize('operator', [MetadataOperator.EQ, MetadataOperator.NE])
def test_equality_operator_rejects_none_value(operator: MetadataOperator) -> None:
    """EQ/NE reject None: it binds SQL NULL and `= NULL` / `!= NULL` are never TRUE
    under three-valued logic (always-empty results). Use IS_NULL / IS_NOT_NULL."""
    with pytest.raises(ValidationError):
        MetadataFilter(key='s', operator=operator, value=None)
