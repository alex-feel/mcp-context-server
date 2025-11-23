"""Metadata filtering types and operators for advanced search functionality."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel
from pydantic import Field
from pydantic import ValidationInfo
from pydantic import field_validator


class MetadataOperator(StrEnum):
    """Comprehensive metadata comparison operators.

    Supports 15 different operators for flexible metadata filtering.
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

        # String operators require string values
        if (
            operator in (MetadataOperator.CONTAINS, MetadataOperator.STARTS_WITH, MetadataOperator.ENDS_WITH)
            and v is not None
            and not isinstance(v, str)
        ):
            raise ValueError(f'Operator {operator} requires a string value')

        return v
