"""SQL query builder for metadata filtering with security validation."""

from __future__ import annotations

import re
from typing import Any

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator


class MetadataQueryBuilder:
    """Build SQL WHERE clauses for metadata filtering with security validation.

    Provides safe SQL generation for JSON metadata filtering with support for
    15 different operators and nested JSON paths.
    """

    def __init__(self) -> None:
        """Initialize the query builder."""
        self.conditions: list[str] = []
        self.parameters: list[Any] = []
        self._filter_count = 0

    def add_simple_filter(self, key: str, value: str | float | bool | None) -> None:
        """Add a simple key=value metadata filter.

        Args:
            key: JSON path to metadata field
            value: Value to match (exact equality)

        Raises:
            ValueError: If key is invalid or contains unsafe characters
        """
        if not self._is_safe_key(key):
            raise ValueError(f'Invalid metadata key: {key}')

        json_path = self._build_json_path(key)
        self.conditions.append(f"json_extract(metadata, '{json_path}') = ?")
        self.parameters.append(self._normalize_value(value))
        self._filter_count += 1

    def add_advanced_filter(self, filter_spec: MetadataFilter) -> None:
        """Add an advanced metadata filter with operator support.

        Args:
            filter_spec: MetadataFilter with key, operator, value, and options

        Raises:
            ValueError: If key is invalid or contains unsafe characters
        """
        if not self._is_safe_key(filter_spec.key):
            raise ValueError(f'Invalid metadata key: {filter_spec.key}')

        json_path = self._build_json_path(filter_spec.key)
        operator = filter_spec.operator
        value = filter_spec.value
        case_sensitive = filter_spec.case_sensitive

        # Build condition based on operator
        if operator == MetadataOperator.EQ:
            if not isinstance(value, list):
                self._add_equality_condition(json_path, value, case_sensitive)
        elif operator == MetadataOperator.NE:
            if not isinstance(value, list):
                self._add_not_equal_condition(json_path, value, case_sensitive)
        elif operator in (MetadataOperator.GT, MetadataOperator.GTE, MetadataOperator.LT, MetadataOperator.LTE):
            if not isinstance(value, list):
                self._add_comparison_condition(json_path, operator, value)
        elif operator == MetadataOperator.IN:
            if isinstance(value, list):
                self._add_in_condition(json_path, value, case_sensitive)
        elif operator == MetadataOperator.NOT_IN:
            if isinstance(value, list):
                self._add_not_in_condition(json_path, value, case_sensitive)
        elif operator == MetadataOperator.EXISTS:
            self._add_exists_condition(json_path)
        elif operator == MetadataOperator.NOT_EXISTS:
            self._add_not_exists_condition(json_path)
        elif operator == MetadataOperator.CONTAINS:
            if isinstance(value, str) or value is None:
                self._add_contains_condition(json_path, value, case_sensitive)
        elif operator == MetadataOperator.STARTS_WITH:
            if isinstance(value, str) or value is None:
                self._add_starts_with_condition(json_path, value, case_sensitive)
        elif operator == MetadataOperator.ENDS_WITH:
            if isinstance(value, str) or value is None:
                self._add_ends_with_condition(json_path, value, case_sensitive)
        elif operator == MetadataOperator.REGEX:
            if isinstance(value, str) or value is None:
                self._add_regex_condition(json_path, value, case_sensitive)
        elif operator == MetadataOperator.IS_NULL:
            self._add_is_null_condition(json_path)
        elif operator == MetadataOperator.IS_NOT_NULL:
            self._add_is_not_null_condition(json_path)

        self._filter_count += 1

    def build_where_clause(self, use_and: bool = True) -> tuple[str, list[Any]]:
        """Build the complete WHERE clause with parameter bindings.

        Args:
            use_and: If True, combine conditions with AND; else use OR

        Returns:
            Tuple of (WHERE clause SQL, parameter values)
        """
        if not self.conditions:
            return ('', [])

        operator = ' AND ' if use_and else ' OR '
        where_clause = f'({operator.join(self.conditions)})'
        return (where_clause, self.parameters)

    def get_filter_count(self) -> int:
        """Get the number of filters applied."""
        return self._filter_count

    # Private helper methods

    @staticmethod
    def _is_safe_key(key: str) -> bool:
        """Validate key for SQL injection prevention.

        Args:
            key: Metadata key to validate

        Returns:
            True if key is safe, False otherwise
        """
        if not key or not key.strip():
            return False

        # Only allow alphanumeric, dots, underscores, and hyphens
        return bool(re.match(r'^[a-zA-Z0-9_\-\.]+$', key))

    @staticmethod
    def _build_json_path(key: str) -> str:
        """Convert key to JSONPath format with nested support.

        Args:
            key: Dot-separated path (e.g., 'user.preferences.theme')

        Returns:
            JSONPath string (e.g., '$.user.preferences.theme')
        """
        # Ensure path starts with $
        if not key.startswith('$'):
            key = f'$.{key}'
        return key

    @staticmethod
    def _normalize_value(value: str | float | bool | None) -> str | int | float | None:
        """Normalize value for SQL comparison.

        Args:
            value: Value to normalize

        Returns:
            Normalized value for SQL parameter binding
        """
        # Convert Python booleans to SQLite integers (0/1)
        if isinstance(value, bool):
            return 1 if value else 0
        # Handle None/null
        if value is None:
            return None
        # Keep strings, numbers as-is
        return value

    def _add_equality_condition(
        self, json_path: str, value: str | float | bool | None, case_sensitive: bool,
    ) -> None:
        """Add an equality condition."""
        if isinstance(value, str) and not case_sensitive:
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) = LOWER(?)")
        else:
            self.conditions.append(f"json_extract(metadata, '{json_path}') = ?")
        self.parameters.append(self._normalize_value(value))

    def _add_not_equal_condition(
        self, json_path: str, value: str | float | bool | None, case_sensitive: bool,
    ) -> None:
        """Add a not-equal condition."""
        if isinstance(value, str) and not case_sensitive:
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) != LOWER(?)")
        else:
            self.conditions.append(f"json_extract(metadata, '{json_path}') != ?")
        self.parameters.append(self._normalize_value(value))

    def _add_comparison_condition(
        self, json_path: str, operator: MetadataOperator, value: str | float | bool | None,
    ) -> None:
        """Add numeric comparison conditions (GT, GTE, LT, LTE)."""
        sql_operators = {
            MetadataOperator.GT: '>',
            MetadataOperator.GTE: '>=',
            MetadataOperator.LT: '<',
            MetadataOperator.LTE: '<=',
        }
        sql_op = sql_operators[operator]

        # Cast to appropriate type for numeric comparisons
        if isinstance(value, (int, float)):
            self.conditions.append(f"CAST(json_extract(metadata, '{json_path}') AS NUMERIC) {sql_op} ?")
            self.parameters.append(value)
        else:
            # String comparison
            self.conditions.append(f"json_extract(metadata, '{json_path}') {sql_op} ?")
            self.parameters.append(str(value))

    def _add_in_condition(
        self, json_path: str, values: list[str | int | float | bool], case_sensitive: bool,
    ) -> None:
        """Add an IN condition for list membership."""
        if not values:
            # Empty list - nothing matches
            self.conditions.append('0 = 1')
            return

        placeholders = ', '.join(['?' for _ in values])
        if not case_sensitive and any(isinstance(v, str) for v in values):
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) IN ({placeholders})")
            self.parameters.extend([str(v).lower() if isinstance(v, str) else self._normalize_value(v) for v in values])
        else:
            self.conditions.append(f"json_extract(metadata, '{json_path}') IN ({placeholders})")
            self.parameters.extend([self._normalize_value(v) for v in values])

    def _add_not_in_condition(
        self, json_path: str, values: list[str | int | float | bool], case_sensitive: bool,
    ) -> None:
        """Add a NOT IN condition."""
        if not values:
            # Empty list - everything matches
            self.conditions.append('1 = 1')
            return

        placeholders = ', '.join(['?' for _ in values])
        if not case_sensitive and any(isinstance(v, str) for v in values):
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) NOT IN ({placeholders})")
            self.parameters.extend([str(v).lower() if isinstance(v, str) else self._normalize_value(v) for v in values])
        else:
            self.conditions.append(f"json_extract(metadata, '{json_path}') NOT IN ({placeholders})")
            self.parameters.extend([self._normalize_value(v) for v in values])

    def _add_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key exists."""
        self.conditions.append(f"json_extract(metadata, '{json_path}') IS NOT NULL")

    def _add_not_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key does not exist."""
        self.conditions.append(f"json_extract(metadata, '{json_path}') IS NULL")

    def _add_contains_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string contains condition."""
        if value is None:
            return

        if case_sensitive:
            self.conditions.append(f"json_extract(metadata, '{json_path}') LIKE '%' || ? || '%'")
            self.parameters.append(value)
        else:
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) LIKE '%' || LOWER(?) || '%'")
            self.parameters.append(value)

    def _add_starts_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string starts-with condition."""
        if value is None:
            return

        if case_sensitive:
            self.conditions.append(f"json_extract(metadata, '{json_path}') LIKE ? || '%'")
            self.parameters.append(value)
        else:
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) LIKE LOWER(?) || '%'")
            self.parameters.append(value)

    def _add_ends_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string ends-with condition."""
        if value is None:
            return

        if case_sensitive:
            self.conditions.append(f"json_extract(metadata, '{json_path}') LIKE '%' || ?")
            self.parameters.append(value)
        else:
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) LIKE '%' || LOWER(?)")
            self.parameters.append(value)

    def _add_regex_condition(self, json_path: str, pattern: str | None, case_sensitive: bool) -> None:
        """Add a regex match condition using SQLite's REGEXP operator.

        Note: Requires SQLite to be compiled with REGEXP support or a custom function.
        """
        if pattern is None:
            return

        # SQLite doesn't have built-in REGEXP by default, but we can use GLOB as fallback
        # or assume a REGEXP extension is available
        if case_sensitive:
            # Try to use REGEXP if available
            self.conditions.append(f"json_extract(metadata, '{json_path}') REGEXP ?")
            self.parameters.append(pattern)
        else:
            # Case-insensitive regex is more complex in SQLite
            # Fallback to LIKE for basic patterns or use REGEXP with (?i) flag
            self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) REGEXP LOWER(?)")
            self.parameters.append(pattern)

    def _add_is_null_condition(self, json_path: str) -> None:
        """Add a condition to check if value is JSON null."""
        # In SQLite JSON, null values are stored as JSON null, not SQL NULL
        self.conditions.append(f"json_type(metadata, '{json_path}') = 'null'")

    def _add_is_not_null_condition(self, json_path: str) -> None:
        """Add a condition to check if value is not JSON null."""
        self.conditions.append(f"json_type(metadata, '{json_path}') != 'null'")
