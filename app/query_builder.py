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

    def __init__(
        self,
        backend_type: str = 'sqlite',
        json_extract_fn: str | None = None,
        param_offset: int = 0,
    ) -> None:
        """Initialize the query builder.

        Args:
            backend_type: Backend type ('sqlite' or 'postgresql') for placeholder generation
            json_extract_fn: Optional JSON extraction function name override
            param_offset: Starting position for PostgreSQL placeholders (for combining queries)
        """
        self.conditions: list[str] = []
        self.parameters: list[Any] = []
        self._filter_count = 0
        self.backend_type = backend_type
        self.json_extract_fn = json_extract_fn or ('json_extract' if backend_type == 'sqlite' else 'jsonb_extract_path_text')
        self.param_offset = param_offset

    def _placeholder(self) -> str:
        """Generate placeholder for current parameter position.

        Returns:
            Placeholder string ('?' for SQLite, '$N' for PostgreSQL)
        """
        if self.backend_type == 'sqlite':
            return '?'
        # PostgreSQL uses $1, $2, $3... with offset
        return f'${self.param_offset + len(self.parameters) + 1}'

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
        placeholder = self._placeholder()
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_extract(metadata, '{json_path}') = {placeholder}")
        else:  # postgresql
            # For nested paths, use #>> with array notation
            key_path = json_path[2:]  # Remove $. prefix
            if '.' in key_path:
                # Nested path: convert 'user.preferences.theme' to array notation '{user,preferences,theme}'
                path_parts = key_path.split('.')
                array_path = '{' + ','.join(path_parts) + '}'
                # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
                if isinstance(value, bool):
                    # JSONB ->> returns 'true' or 'false' as TEXT for booleans
                    self.conditions.append(f"metadata#>>'{array_path}' = {placeholder}::TEXT")
                elif isinstance(value, (int, float)):
                    # Numeric comparison
                    self.conditions.append(f"(metadata#>>'{array_path}')::NUMERIC = {placeholder}")
                else:
                    # Text comparison
                    self.conditions.append(f"metadata#>>'{array_path}' = {placeholder}::TEXT")
            else:
                # Single-level path
                # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
                if isinstance(value, bool):
                    # JSONB ->> returns 'true' or 'false' as TEXT for booleans
                    self.conditions.append(f"metadata->>'{key_path}' = {placeholder}::TEXT")
                elif isinstance(value, (int, float)):
                    # Numeric comparison: cast JSON field to numeric
                    self.conditions.append(f"(metadata->>'{key_path}')::NUMERIC = {placeholder}")
                else:
                    # Text comparison
                    self.conditions.append(f"metadata->>'{key_path}' = {placeholder}::TEXT")
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
        # Validate required key parameter: must contain non-whitespace characters
        # Since key is typed as str (not str | None), it cannot be None at this point
        # We only need to check if it's empty or contains only whitespace
        if not key.strip():
            return False

        # Only allow alphanumeric, dots, underscores, and hyphens
        return bool(re.match(r'^[a-zA-Z0-9_.-]+$', key))

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

    def _normalize_value(self, value: str | float | bool | None) -> str | int | float | None:
        """Normalize value for SQL comparison based on backend type.

        Args:
            value: Value to normalize

        Returns:
            Normalized value for SQL parameter binding

        Note:
            Boolean handling differs by backend:
            - SQLite: Booleans stored as integers (0/1) in JSON
            - PostgreSQL: JSONB ->> extracts booleans as TEXT ('true'/'false')
        """
        if isinstance(value, bool):
            if self.backend_type == 'postgresql':
                # PostgreSQL JSONB ->> returns 'true' or 'false' as TEXT for booleans
                return 'true' if value else 'false'
            # SQLite stores JSON booleans as integers (0/1)
            return 1 if value else 0
        # Handle None/null
        if value is None:
            return None
        # Keep strings, numbers as-is
        return value

    def _add_equality_condition(
        self,
        json_path: str,
        value: str | float | bool | None,
        case_sensitive: bool,
    ) -> None:
        """Add an equality condition."""
        placeholder = self._placeholder()
        key_path = json_path[2:]  # Remove $. prefix

        if self.backend_type == 'sqlite':
            if isinstance(value, str) and not case_sensitive:
                self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) = LOWER({placeholder})")
            else:
                self.conditions.append(f"json_extract(metadata, '{json_path}') = {placeholder}")
        else:  # postgresql
            # For nested paths, use #>> with array notation
            if '.' in key_path:
                path_parts = key_path.split('.')
                array_path = '{' + ','.join(path_parts) + '}'
                # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
                if isinstance(value, bool):
                    # JSONB ->> returns 'true' or 'false' as TEXT for booleans
                    self.conditions.append(f"metadata#>>'{array_path}' = {placeholder}::TEXT")
                elif isinstance(value, (int, float)):
                    self.conditions.append(f"(metadata#>>'{array_path}')::NUMERIC = {placeholder}")
                elif isinstance(value, str) and not case_sensitive:
                    self.conditions.append(f"LOWER(metadata#>>'{array_path}') = LOWER({placeholder}::TEXT)")
                else:
                    self.conditions.append(f"metadata#>>'{array_path}' = {placeholder}::TEXT")
            else:
                # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
                if isinstance(value, bool):
                    # JSONB ->> returns 'true' or 'false' as TEXT for booleans
                    self.conditions.append(f"metadata->>'{key_path}' = {placeholder}::TEXT")
                elif isinstance(value, (int, float)):
                    # Numeric comparison: cast JSON field to numeric
                    self.conditions.append(f"(metadata->>'{key_path}')::NUMERIC = {placeholder}")
                elif isinstance(value, str) and not case_sensitive:
                    self.conditions.append(f"LOWER(metadata->>'{key_path}') = LOWER({placeholder}::TEXT)")
                else:
                    # String comparison
                    self.conditions.append(f"metadata->>'{key_path}' = {placeholder}::TEXT")
        self.parameters.append(self._normalize_value(value))

    def _add_not_equal_condition(
        self,
        json_path: str,
        value: str | float | bool | None,
        case_sensitive: bool,
    ) -> None:
        """Add a not-equal condition."""
        placeholder = self._placeholder()
        key_path = json_path[2:]

        if self.backend_type == 'sqlite':
            if isinstance(value, str) and not case_sensitive:
                self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) != LOWER({placeholder})")
            else:
                self.conditions.append(f"json_extract(metadata, '{json_path}') != {placeholder}")
        else:  # postgresql
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # JSONB ->> returns 'true' or 'false' as TEXT for booleans
                self.conditions.append(f"metadata->>'{key_path}' != {placeholder}::TEXT")
            elif isinstance(value, (int, float)):
                # Numeric comparison
                self.conditions.append(f"(metadata->>'{key_path}')::NUMERIC != {placeholder}")
            elif isinstance(value, str) and not case_sensitive:
                self.conditions.append(f"LOWER(metadata->>'{key_path}') != LOWER({placeholder}::TEXT)")
            else:
                self.conditions.append(f"metadata->>'{key_path}' != {placeholder}::TEXT")
        self.parameters.append(self._normalize_value(value))

    def _add_comparison_condition(
        self,
        json_path: str,
        operator: MetadataOperator,
        value: str | float | bool | None,
    ) -> None:
        """Add numeric comparison conditions (GT, GTE, LT, LTE)."""
        sql_operators = {
            MetadataOperator.GT: '>',
            MetadataOperator.GTE: '>=',
            MetadataOperator.LT: '<',
            MetadataOperator.LTE: '<=',
        }
        sql_op = sql_operators[operator]
        placeholder = self._placeholder()
        key_path = json_path[2:]

        if isinstance(value, (int, float)):
            if self.backend_type == 'sqlite':
                self.conditions.append(f"CAST(json_extract(metadata, '{json_path}') AS NUMERIC) {sql_op} {placeholder}")
            else:  # postgresql - use ->> and cast
                self.conditions.append(f"(metadata->>'{key_path}')::NUMERIC {sql_op} {placeholder}")
            self.parameters.append(value)
        else:
            if self.backend_type == 'sqlite':
                self.conditions.append(f"json_extract(metadata, '{json_path}') {sql_op} {placeholder}")
            else:  # postgresql
                self.conditions.append(f"metadata->>'{key_path}' {sql_op} {placeholder}::TEXT")
            self.parameters.append(str(value))

    def _add_in_condition(
        self,
        json_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> None:
        """Add an IN condition for list membership.

        All values are converted to strings and compared as TEXT because:
        - SQLite's json_extract() returns typed values (INTEGER for numbers)
        - SQLite's IN clause doesn't auto-convert types like equality does
        - PostgreSQL's ->> operator returns TEXT values
        - Type mismatch causes silent failures on SQLite and explicit errors on PostgreSQL

        Solution: Cast json_extract result to TEXT on SQLite and convert all parameters to strings.
        """
        if not values:
            self.conditions.append('0 = 1')
            return

        key_path = json_path[2:]

        if self.backend_type == 'sqlite':
            # Generate placeholders BEFORE extending parameters
            placeholders = ', '.join(['?' for _ in values])
            # Cast json_extract to TEXT to ensure consistent comparison with string parameters
            # json_extract returns INTEGER for JSON numbers, which doesn't match TEXT in IN clause
            if not case_sensitive and any(isinstance(v, str) for v in values):
                self.conditions.append(f"LOWER(CAST(json_extract(metadata, '{json_path}') AS TEXT)) IN ({placeholders})")
                self.parameters.extend([str(v).lower() for v in values])
            else:
                self.conditions.append(f"CAST(json_extract(metadata, '{json_path}') AS TEXT) IN ({placeholders})")
                # Convert all values to strings for consistent TEXT comparison
                self.parameters.extend([str(v) for v in values])
        else:  # postgresql
            # Generate placeholders with proper numbering BEFORE extending parameters
            start_pos = self.param_offset + len(self.parameters) + 1
            cast_placeholders = ', '.join([f'${start_pos + i}::TEXT' for i in range(len(values))])
            if not case_sensitive and any(isinstance(v, str) for v in values):
                self.conditions.append(f"LOWER(metadata->>'{key_path}') IN ({cast_placeholders})")
                self.parameters.extend([str(v).lower() for v in values])
            else:
                self.conditions.append(f"metadata->>'{key_path}' IN ({cast_placeholders})")
                # Convert all values to strings for asyncpg TEXT parameter binding
                self.parameters.extend([str(v) for v in values])

    def _add_not_in_condition(
        self,
        json_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> None:
        """Add a NOT IN condition.

        All values are converted to strings and compared as TEXT because:
        - SQLite's json_extract() returns typed values (INTEGER for numbers)
        - SQLite's IN clause doesn't auto-convert types like equality does
        - PostgreSQL's ->> operator returns TEXT values
        - Type mismatch causes silent failures on SQLite and explicit errors on PostgreSQL

        Solution: Cast json_extract result to TEXT on SQLite and convert all parameters to strings.
        """
        if not values:
            self.conditions.append('1 = 1')
            return

        key_path = json_path[2:]

        if self.backend_type == 'sqlite':
            placeholders = ', '.join(['?' for _ in values])
            # Cast json_extract to TEXT to ensure consistent comparison with string parameters
            # json_extract returns INTEGER for JSON numbers, which doesn't match TEXT in NOT IN clause
            if not case_sensitive and any(isinstance(v, str) for v in values):
                self.conditions.append(f"LOWER(CAST(json_extract(metadata, '{json_path}') AS TEXT)) NOT IN ({placeholders})")
                self.parameters.extend([str(v).lower() for v in values])
            else:
                self.conditions.append(f"CAST(json_extract(metadata, '{json_path}') AS TEXT) NOT IN ({placeholders})")
                # Convert all values to strings for consistent TEXT comparison
                self.parameters.extend([str(v) for v in values])
        else:  # postgresql
            start_pos = self.param_offset + len(self.parameters) + 1
            cast_placeholders = ', '.join([f'${start_pos + i}::TEXT' for i in range(len(values))])
            if not case_sensitive and any(isinstance(v, str) for v in values):
                self.conditions.append(f"LOWER(metadata->>'{key_path}') NOT IN ({cast_placeholders})")
                self.parameters.extend([str(v).lower() for v in values])
            else:
                self.conditions.append(f"metadata->>'{key_path}' NOT IN ({cast_placeholders})")
                # Convert all values to strings for asyncpg TEXT parameter binding
                self.parameters.extend([str(v) for v in values])

    def _add_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key exists."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_extract(metadata, '{json_path}') IS NOT NULL")
        else:  # postgresql
            self.conditions.append(f"metadata->>'{key_path}' IS NOT NULL")

    def _add_not_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key does not exist."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_extract(metadata, '{json_path}') IS NULL")
        else:  # postgresql
            self.conditions.append(f"metadata->>'{key_path}' IS NULL")

    def _add_contains_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string contains condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        if self.backend_type == 'sqlite':
            if case_sensitive:
                self.conditions.append(f"INSTR(json_extract(metadata, '{json_path}'), {placeholder}) > 0")
            else:
                self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) LIKE '%' || LOWER({placeholder}) || '%'")
            self.parameters.append(value)
        else:  # postgresql
            if case_sensitive:
                self.conditions.append(f"metadata->>'{key_path}' LIKE '%' || {placeholder}::TEXT || '%'")
            else:
                self.conditions.append(f"LOWER(metadata->>'{key_path}') LIKE '%' || LOWER({placeholder}::TEXT) || '%'")
            self.parameters.append(value)

    def _add_starts_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string starts-with condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        if self.backend_type == 'sqlite':
            if case_sensitive:
                escaped_value = self._escape_glob_pattern(value)
                self.conditions.append(f"json_extract(metadata, '{json_path}') GLOB {placeholder} || '*'")
                self.parameters.append(escaped_value)
            else:
                self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) LIKE LOWER({placeholder}) || '%'")
                self.parameters.append(value)
        else:  # postgresql
            if case_sensitive:
                self.conditions.append(f"metadata->>'{key_path}' LIKE {placeholder}::TEXT || '%'")
            else:
                self.conditions.append(f"LOWER(metadata->>'{key_path}') LIKE LOWER({placeholder}::TEXT) || '%'")
            self.parameters.append(value)

    def _add_ends_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string ends-with condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        if self.backend_type == 'sqlite':
            if case_sensitive:
                escaped_value = self._escape_glob_pattern(value)
                self.conditions.append(f"json_extract(metadata, '{json_path}') GLOB '*' || {placeholder}")
                self.parameters.append(escaped_value)
            else:
                self.conditions.append(f"LOWER(json_extract(metadata, '{json_path}')) LIKE '%' || LOWER({placeholder})")
                self.parameters.append(value)
        else:  # postgresql
            if case_sensitive:
                self.conditions.append(f"metadata->>'{key_path}' LIKE '%' || {placeholder}::TEXT")
            else:
                self.conditions.append(f"LOWER(metadata->>'{key_path}') LIKE '%' || LOWER({placeholder}::TEXT)")
            self.parameters.append(value)

    def _add_regex_condition(self, json_path: str, pattern: str | None, case_sensitive: bool) -> None:
        """Add a regex match condition (not supported).

        Args:
            json_path: JSON path to metadata field (unused)
            pattern: Regex pattern (unused)
            case_sensitive: Whether to match case-sensitively (unused)

        Raises:
            ValueError: Always raised as REGEX is not supported in SQLite
        """
        # Use parameters to avoid linting warnings
        _ = (json_path, pattern, case_sensitive)

        # SQLite doesn't have built-in REGEXP function
        # Raise a clear error instead of generating SQL that will fail
        raise ValueError(
            'REGEX operator is not supported in the current SQLite implementation. '
            'Please use CONTAINS, STARTS_WITH, or ENDS_WITH operators instead.',
        )

    def _add_is_null_condition(self, json_path: str) -> None:
        """Add a condition to check if value is JSON null."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            # In SQLite JSON, null values are stored as JSON null, not SQL NULL
            self.conditions.append(f"json_type(metadata, '{json_path}') = 'null'")
        else:  # postgresql
            # In PostgreSQL, check if the value is NULL or the JSON value is null
            self.conditions.append(f"metadata->>'{key_path}' IS NULL OR metadata->'{key_path}' = 'null'::jsonb")

    def _add_is_not_null_condition(self, json_path: str) -> None:
        """Add a condition to check if value is not JSON null."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_type(metadata, '{json_path}') != 'null'")
        else:  # postgresql
            self.conditions.append(f"metadata->>'{key_path}' IS NOT NULL AND metadata->'{key_path}' != 'null'::jsonb")

    @staticmethod
    def _escape_glob_pattern(value: str) -> str:
        """Escape special characters in GLOB patterns.

        GLOB special characters are: * ? [ ]
        We need to escape them with backslash.

        Args:
            value: String value to escape

        Returns:
            Escaped string safe for GLOB patterns
        """
        # Escape special GLOB characters
        escaped = value.replace('\\', '\\\\')
        escaped = escaped.replace('*', '\\*')
        escaped = escaped.replace('?', '\\?')
        escaped = escaped.replace('[', '\\[')
        return escaped.replace(']', '\\]')
