"""SQL query builder for metadata filtering with security validation."""

import json
import re
from typing import Any

from app.metadata_membership import MembershipConditionsMixin
from app.metadata_sql import build_json_path
from app.metadata_sql import escape_glob_pattern
from app.metadata_sql import escape_like_value
from app.metadata_sql import is_safe_key
from app.metadata_sql import normalize_value
from app.metadata_sql import pg_ci
from app.metadata_sql import pg_json_accessor
from app.metadata_sql import pg_number_guard
from app.metadata_sql import pg_numeric_body
from app.metadata_sql import pg_numeric_compare
from app.metadata_sql import pg_text_accessor
from app.metadata_sql import pg_text_guard
from app.metadata_sql import sqlite_bool_guard
from app.metadata_sql import sqlite_number_guard
from app.metadata_sql import sqlite_text_guard
from app.metadata_types import MAX_METADATA_BIND_PARAMS
from app.metadata_types import MAX_METADATA_CLAUSE_CHARS
from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.metadata_types import reject_non_finite
from app.metadata_types import reject_nul
from app.metadata_types import reject_out_of_int64

# Match the metadata COLUMN token only, so a table alias can qualify the column
# WITHOUT ever rewriting a user-supplied JSON key (a global str.replace corrupted
# keys containing the substring 'metadata', e.g. 'metadata_version'). The column
# position is backend-specific: PostgreSQL accesses it via ``->``/``#>`` (a JSON
# key is [A-Za-z0-9_.-] only, so it never contains those), while SQLite passes it
# as the first ``json_extract``/``json_type``/``json_each`` argument, i.e.
# immediately followed by a comma (a SQLite JSON path never contains a comma). The
# PG pattern deliberately omits the comma so a nested array-path segment like
# ``{metadata,x}`` is NOT mistaken for the column.
_PG_METADATA_COLUMN_RE = re.compile(r"(?<![\w.'])metadata(?=->|#>)")
_SQLITE_METADATA_COLUMN_RE = re.compile(r"(?<![\w.'])metadata(?=\s*,)")


class MetadataQueryBuilder(MembershipConditionsMixin):
    """Build SQL WHERE clauses for metadata filtering with security validation.

    Provides safe SQL generation for JSON metadata filtering with support for
    16 different operators and nested JSON paths.
    """

    def __init__(
        self,
        backend_type: str = 'sqlite',
        param_offset: int = 0,
        table_alias: str | None = None,
    ) -> None:
        """Initialize the query builder.

        Args:
            backend_type: Backend type ('sqlite' or 'postgresql') for placeholder generation
            param_offset: Starting position for PostgreSQL placeholders (for combining queries)
            table_alias: Optional table alias for the ``metadata`` column. When set
                (e.g. ``'ce'`` for a query that JOINs ``context_entries ce``), the
                built clause qualifies the column as ``<alias>.metadata`` so callers
                never rewrite the built SQL. Only column positions are qualified;
                JSON keys (even ones containing 'metadata') are left untouched.
        """
        self.conditions: list[str] = []
        self.parameters: list[Any] = []
        self._filter_count = 0
        self.backend_type = backend_type
        self.param_offset = param_offset
        self.table_alias = table_alias

    def _placeholder(self) -> str:
        """Generate placeholder for current parameter position.

        Returns:
            Placeholder string ('?' for SQLite, '$N' for PostgreSQL)
        """
        if self.backend_type == 'sqlite':
            return '?'
        # PostgreSQL uses $1, $2, $3... with offset
        return f'${self.param_offset + len(self.parameters) + 1}'

    def _enforce_clause_budgets(self) -> None:
        """Reject the clause once its binds OR its generated SQL text exceed the shared budgets.

        Called after each filter is added. The per-dimension caps bound every input
        list individually, but capped dimensions MULTIPLY (filters times IN-list
        members), so these aggregate backstops guarantee the built clause can never
        grow past the backend's limits regardless of how future filter dimensions
        combine.

        BIND COUNT (``MAX_METADATA_BIND_PARAMS``) tracks the backend's per-statement
        bind limit and includes ``param_offset`` (the enclosing statement's earlier
        placeholders on PostgreSQL) so it reflects the true statement position.

        SQL TEXT (``MAX_METADATA_CLAUSE_CHARS``) is a SEPARATE dimension: the generated
        statement size is NOT proportional to the bind count. One PostgreSQL numeric
        comparison inlines a multi-kilobyte exact/double discriminator, and the metadata
        key path is repeated inside it, so bind-legal input could otherwise assemble an
        arbitrarily large statement -- costly to build on the event loop and to
        parse+plan on every call, since it also exceeds asyncpg's cacheable-statement
        size. The measured length sums the accumulated conditions; the small constant
        joiner/alias overhead added by :meth:`build_where_clause` is immaterial at this
        scale.

        The raised ``ValueError`` is routed by every construction site into the
        structured, breaker-exempt validation-error channel; the query is then never
        executed, so the builder's partially-accumulated state is harmless.

        Raises:
            ValueError: If the accumulated bind-parameter count exceeds
                MAX_METADATA_BIND_PARAMS, or the generated clause text exceeds
                MAX_METADATA_CLAUSE_CHARS.
        """
        total = self.param_offset + len(self.parameters)
        if total > MAX_METADATA_BIND_PARAMS:
            raise ValueError(
                f'Metadata filters expand into too many SQL bind parameters: {total} exceeds '
                f'the budget of {MAX_METADATA_BIND_PARAMS}. Narrow the filters (fewer filters, '
                f'or shorter in/not_in value lists).',
            )
        clause_chars = sum(len(condition) for condition in self.conditions)
        if clause_chars > MAX_METADATA_CLAUSE_CHARS:
            raise ValueError(
                f'Metadata filters expand into too much SQL text: {clause_chars} characters '
                f'exceeds the budget of {MAX_METADATA_CLAUSE_CHARS}. Narrow the filters (fewer '
                f'filters, shorter in/not_in value lists, or shorter metadata key paths).',
            )

    def add_simple_filter(self, key: str, value: str | float | bool | None) -> None:
        """Add a simple key=value metadata filter.

        Args:
            key: JSON path to metadata field
            value: Value to match (exact equality)

        Raises:
            ValueError: If key is invalid or contains unsafe characters, or the
                accumulated binds or generated SQL text exceed the clause budgets
        """
        if not is_safe_key(key):
            raise ValueError(f'Invalid metadata key: {key}')

        # The simple metadata={} equality path bypasses MetadataFilter validation,
        # so reject out-of-int64 integers, non-finite floats, AND strings carrying
        # a NUL or unpaired UTF-16 surrogate here too: without these guards an
        # out-of-range int aborts the search on SQLite while PostgreSQL matches it,
        # a NaN param matches nothing on SQLite but every number row on PostgreSQL,
        # and a NUL-bearing string matches on SQLite but aborts the query and
        # charges the circuit breaker on PostgreSQL -- the cross-backend
        # divergences the advanced metadata_filters path rejects via the same helpers.
        reject_out_of_int64(value)
        reject_non_finite(value)
        reject_nul(value)

        json_path = build_json_path(key)
        placeholder = self._placeholder()
        if self.backend_type == 'sqlite':
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean equality matches JSON booleans only (parity with PostgreSQL)
                guard = sqlite_bool_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (parity with PostgreSQL)
                guard = sqlite_number_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            else:
                # String (or None): the text guard restricts the match to a JSON-string-typed
                # stored value, so a stored NUMBER/boolean/null is excluded (a number's text form
                # diverges across backends). The CAST is a harmless text->text identity here.
                guard = sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) = {placeholder})",
                )
        else:  # postgresql
            # Route through the shared accessor so a nested key traverses via
            # #>> array notation instead of being read as a literal top-level key.
            acc = pg_text_accessor(json_path[2:])  # strip $. prefix
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean equality matches JSON booleans only (->> returns 'true'/'false' text)
                bguard = f"jsonb_typeof({pg_json_accessor(json_path[2:])}) = 'boolean'"
                self.conditions.append(f'({bguard} AND {acc} = {placeholder}::TEXT)')
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (guard), compared with SQLite's exact
                # integer/double semantics (pg_numeric_body).
                guard = pg_number_guard(json_path[2:])
                body = pg_numeric_body(json_path[2:], '=', placeholder, value)
                self.conditions.append(f'({guard} AND ({body}))')
            else:
                # Text comparison: the text guard restricts the match to a JSON-string-typed
                # stored value, so a stored number/boolean/null is excluded (parity with SQLite).
                guard = pg_text_guard(json_path[2:])
                self.conditions.append(f'({guard} AND {acc} = {placeholder}::TEXT)')
        self.parameters.append(normalize_value(value, backend_type=self.backend_type))
        self._filter_count += 1
        self._enforce_clause_budgets()

    def add_advanced_filter(self, filter_spec: MetadataFilter) -> None:
        """Add an advanced metadata filter with operator support.

        Args:
            filter_spec: MetadataFilter with key, operator, value, and options

        Raises:
            ValueError: If key is invalid or contains unsafe characters, or the
                accumulated binds or generated SQL text exceed the clause budgets
        """
        if not is_safe_key(filter_spec.key):
            raise ValueError(f'Invalid metadata key: {filter_spec.key}')

        json_path = build_json_path(filter_spec.key)
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
        elif operator == MetadataOperator.ARRAY_CONTAINS and not isinstance(value, list) and value is not None:
            self._add_array_contains_condition(json_path, value, case_sensitive)

        self._filter_count += 1
        self._enforce_clause_budgets()

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
        if self.table_alias:
            # Qualify the metadata COLUMN with the caller's table alias so the
            # clause matches a JOINed target (e.g. 'ce.metadata'). Only column
            # positions are rewritten -- a JSON key containing 'metadata' (e.g.
            # 'metadata_version', or a nested '{metadata,x}' segment) is never
            # touched (see the backend-specific column patterns above).
            column_re = _SQLITE_METADATA_COLUMN_RE if self.backend_type == 'sqlite' else _PG_METADATA_COLUMN_RE
            where_clause = column_re.sub(f'{self.table_alias}.metadata', where_clause)
        return (where_clause, self.parameters)

    def get_filter_count(self) -> int:
        """Get the number of filters applied."""
        return self._filter_count

    # Private helper methods

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
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean equality matches JSON booleans only (parity with PostgreSQL)
                guard = sqlite_bool_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (parity with PostgreSQL)
                guard = sqlite_number_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            elif isinstance(value, str) and not case_sensitive:
                guard = sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND LOWER(json_extract(metadata, '{json_path}')) = LOWER({placeholder}))",
                )
            else:
                # Case-sensitive string EQ (and value=None). The text guard restricts the match
                # to a JSON-string-typed stored value, so a stored NUMBER/boolean/null is
                # excluded (a number's text form diverges across backends). The CAST is a
                # harmless text->text identity under the guard.
                guard = sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) = {placeholder})",
                )
        else:  # postgresql
            # Route through the shared accessor so a nested key traverses via
            # #>> array notation rather than being read as a literal top-level key.
            acc = pg_text_accessor(key_path)
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean equality matches JSON booleans only (->> returns 'true'/'false' text)
                bguard = f"jsonb_typeof({pg_json_accessor(key_path)}) = 'boolean'"
                self.conditions.append(f'({bguard} AND {acc} = {placeholder}::TEXT)')
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (guard), compared with SQLite's exact
                # integer/double semantics (pg_numeric_body).
                guard = pg_number_guard(key_path)
                body = pg_numeric_body(key_path, '=', placeholder, value)
                self.conditions.append(f'({guard} AND ({body}))')
            elif isinstance(value, str) and not case_sensitive:
                # Case-insensitive string EQ. The text guard restricts the match to a
                # JSON-string-typed stored value (a stored number/boolean is excluded, parity
                # with SQLite). ASCII-only case fold (pg_ci) matches SQLite's ASCII-only LOWER().
                guard = pg_text_guard(key_path)
                ci_acc = pg_ci(acc)
                ci_val = pg_ci(f'{placeholder}::TEXT')
                self.conditions.append(f'({guard} AND {ci_acc} = {ci_val})')
            else:
                # Case-sensitive string EQ: text guard restricts to JSON-string stored values.
                guard = pg_text_guard(key_path)
                self.conditions.append(f'({guard} AND {acc} = {placeholder}::TEXT)')
        self.parameters.append(normalize_value(value, backend_type=self.backend_type))

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
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean NE matches JSON booleans that differ (parity with PostgreSQL):
                # a non-boolean value never satisfies a boolean operator on either backend.
                guard = sqlite_bool_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') != {placeholder})")
            elif isinstance(value, (int, float)):
                # Numeric NE matches JSON numbers that differ (parity with PostgreSQL):
                # a non-number value never satisfies a numeric operator on either backend.
                guard = sqlite_number_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') != {placeholder})")
            elif isinstance(value, str) and not case_sensitive:
                guard = sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND LOWER(json_extract(metadata, '{json_path}')) != LOWER({placeholder}))",
                )
            else:
                # Case-sensitive string NE (and value=None). The text guard restricts the match
                # to a JSON-string-typed stored value, so a stored NUMBER/boolean/null never
                # participates (a number's text form diverges across backends). The CAST is a
                # harmless text->text identity under the guard.
                guard = sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) != {placeholder})",
                )
        else:  # postgresql
            acc = pg_text_accessor(key_path)
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean NE matches JSON booleans that differ (->> returns 'true'/'false' text)
                bguard = f"jsonb_typeof({pg_json_accessor(key_path)}) = 'boolean'"
                self.conditions.append(f'({bguard} AND {acc} != {placeholder}::TEXT)')
            elif isinstance(value, (int, float)):
                # Numeric NE matches JSON numbers that differ (guard excludes non-numbers, parity
                # with SQLite), compared with SQLite's exact integer/double semantics.
                guard = pg_number_guard(key_path)
                body = pg_numeric_body(key_path, '!=', placeholder, value)
                self.conditions.append(f'({guard} AND ({body}))')
            elif isinstance(value, str) and not case_sensitive:
                # Case-insensitive string NE. The text guard restricts the match to a
                # JSON-string-typed stored value (a stored number/boolean is excluded, parity
                # with SQLite).
                guard = pg_text_guard(key_path)
                ci_acc = pg_ci(acc)
                ci_val = pg_ci(f'{placeholder}::TEXT')
                self.conditions.append(f'({guard} AND {ci_acc} != {ci_val})')
            else:
                guard = pg_text_guard(key_path)
                self.conditions.append(f'({guard} AND {acc} != {placeholder}::TEXT)')
        self.parameters.append(normalize_value(value, backend_type=self.backend_type))

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
            # Numeric comparison matches JSON numbers only on BOTH backends; a
            # non-number value (text/bool/json-null/absent) never matches, so the
            # backends agree without relying on SQLite's CAST(text AS NUMERIC) coercion.
            if self.backend_type == 'sqlite':
                guard = sqlite_number_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS NUMERIC) {sql_op} {placeholder})",
                )
            else:  # postgresql - JSON numbers only (guard); compared with SQLite's exact
                # integer/double semantics (pg_numeric_body) so the boundary row matches SQLite.
                guard = pg_number_guard(key_path)
                body = pg_numeric_body(key_path, sql_op, placeholder, value)
                self.conditions.append(f'({guard} AND ({body}))')
            self.parameters.append(value)
        else:
            # String-valued ordered comparison. Two parity requirements: (1) the text guard
            # restricts ordering to JSON-string-typed stored values, so a stored NUMBER/boolean
            # is never ordered as text (a number's text form, and SQLite's sort of a typed
            # number BELOW all text, both diverge from PostgreSQL's ->>); (2) force byte-wise
            # collation on PostgreSQL (COLLATE "C") to match SQLite's default BINARY text
            # ordering (else locale collation reorders case).
            if self.backend_type == 'sqlite':
                guard = sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) {sql_op} {placeholder})",
                )
            else:  # postgresql
                guard = pg_text_guard(key_path)
                acc = pg_text_accessor(key_path)
                self.conditions.append(f'({guard} AND {acc} COLLATE "C" {sql_op} {placeholder}::TEXT)')
            self.parameters.append(str(value))

    def _add_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key exists."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_extract(metadata, '{json_path}') IS NOT NULL")
        else:  # postgresql
            self.conditions.append(f'{pg_text_accessor(key_path)} IS NOT NULL')

    def _add_not_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key does not exist."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_extract(metadata, '{json_path}') IS NULL")
        else:  # postgresql
            self.conditions.append(f'{pg_text_accessor(key_path)} IS NULL')

    def _add_contains_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string contains condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        # The text guard restricts a STRING substring operator to a JSON-string-typed stored
        # value: a stored number or boolean must never CONTAINS-match (PostgreSQL ``->>`` renders
        # them as text, e.g. 'tru'/'1', diverging from SQLite) -- parity by construction.
        if self.backend_type == 'sqlite':
            guard = sqlite_text_guard(json_path)
            if case_sensitive:
                # INSTR matches a literal substring; no LIKE wildcards to escape.
                pred = f"INSTR(json_extract(metadata, '{json_path}'), {placeholder}) > 0"
                self.parameters.append(value)
            else:
                pred = (
                    f"LOWER(json_extract(metadata, '{json_path}')) "
                    f"LIKE '%' || LOWER({placeholder}) || '%' ESCAPE '\\'"
                )
                self.parameters.append(escape_like_value(value))
            self.conditions.append(f'({guard} AND {pred})')
        else:  # postgresql
            guard = pg_text_guard(key_path)
            acc = pg_text_accessor(key_path)
            if case_sensitive:
                pred = f"{acc} LIKE '%' || {placeholder}::TEXT || '%' ESCAPE '\\'"
            else:
                pred = f"{pg_ci(acc)} LIKE '%' || {pg_ci(f'{placeholder}::TEXT')} || '%' ESCAPE '\\'"
            self.conditions.append(f'({guard} AND {pred})')
            self.parameters.append(escape_like_value(value))

    def _add_starts_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string starts-with condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        # The text guard restricts a STRING starts-with to a JSON-string-typed stored value, so
        # a stored number/boolean is never matched as text (parity by construction).
        if self.backend_type == 'sqlite':
            guard = sqlite_text_guard(json_path)
            if case_sensitive:
                pred = f"json_extract(metadata, '{json_path}') GLOB {placeholder} || '*'"
                self.parameters.append(escape_glob_pattern(value))
            else:
                pred = (
                    f"LOWER(json_extract(metadata, '{json_path}')) "
                    f"LIKE LOWER({placeholder}) || '%' ESCAPE '\\'"
                )
                self.parameters.append(escape_like_value(value))
            self.conditions.append(f'({guard} AND {pred})')
        else:  # postgresql
            guard = pg_text_guard(key_path)
            acc = pg_text_accessor(key_path)
            if case_sensitive:
                pred = f"{acc} LIKE {placeholder}::TEXT || '%' ESCAPE '\\'"
            else:
                pred = f"{pg_ci(acc)} LIKE {pg_ci(f'{placeholder}::TEXT')} || '%' ESCAPE '\\'"
            self.conditions.append(f'({guard} AND {pred})')
            self.parameters.append(escape_like_value(value))

    def _add_ends_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string ends-with condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        # The text guard restricts a STRING ends-with to a JSON-string-typed stored value, so a
        # stored number/boolean is never matched as text (parity by construction).
        if self.backend_type == 'sqlite':
            guard = sqlite_text_guard(json_path)
            if case_sensitive:
                pred = f"json_extract(metadata, '{json_path}') GLOB '*' || {placeholder}"
                self.parameters.append(escape_glob_pattern(value))
            else:
                pred = (
                    f"LOWER(json_extract(metadata, '{json_path}')) "
                    f"LIKE '%' || LOWER({placeholder}) ESCAPE '\\'"
                )
                self.parameters.append(escape_like_value(value))
            self.conditions.append(f'({guard} AND {pred})')
        else:  # postgresql
            guard = pg_text_guard(key_path)
            acc = pg_text_accessor(key_path)
            if case_sensitive:
                pred = f"{acc} LIKE '%' || {placeholder}::TEXT ESCAPE '\\'"
            else:
                pred = f"{pg_ci(acc)} LIKE '%' || {pg_ci(f'{placeholder}::TEXT')} ESCAPE '\\'"
            self.conditions.append(f'({guard} AND {pred})')
            self.parameters.append(escape_like_value(value))

    def _add_is_null_condition(self, json_path: str) -> None:
        """Add a condition to check if value is JSON null."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            # In SQLite JSON, null values are stored as JSON null, not SQL NULL.
            # json_type returns 'null' for a present JSON null and SQL NULL for a
            # missing key, so a missing key does NOT match.
            self.conditions.append(f"json_type(metadata, '{json_path}') = 'null'")
        else:  # postgresql
            # Match a PRESENT JSON null only, mirroring the SQLite json_type='null'
            # semantics above. jsonb_typeof returns 'null' for a stored JSON null and
            # SQL NULL for a missing key, so a missing key does NOT match. The earlier
            # "->>key IS NULL OR ..." form conflated absent-key with JSON-null and, being
            # an unparenthesized OR, also risked AND/OR precedence bugs when combined.
            self.conditions.append(f"jsonb_typeof({pg_json_accessor(key_path)}) = 'null'")

    def _add_is_not_null_condition(self, json_path: str) -> None:
        """Add a condition to check if value is not JSON null."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_type(metadata, '{json_path}') != 'null'")
        else:  # postgresql
            self.conditions.append(
                f"{pg_text_accessor(key_path)} IS NOT NULL "
                f"AND {pg_json_accessor(key_path)} != 'null'::jsonb",
            )

    def _add_array_contains_condition(
        self,
        json_path: str,
        value: str | float | bool,
        case_sensitive: bool,
    ) -> None:
        """Add a condition to check if a JSON array contains a specific element.

        Uses EXISTS subquery with json_each() for SQLite, and @> containment operator
        for PostgreSQL.

        IMPORTANT: This method includes type checks to gracefully handle non-array fields.
        Without these checks, jsonb_array_elements_text() throws "cannot extract elements
        from a scalar" error on PostgreSQL when the field contains a scalar value.
        The documented behavior is to return empty results (not error) for non-array fields.

        Args:
            json_path: JSON path to the array field (e.g., '$.technologies')
            value: The element value to search for in the array
            case_sensitive: Whether string comparison should be case-sensitive
        """
        placeholder = self._placeholder()
        key_path = json_path[2:]  # Remove $. prefix

        if self.backend_type == 'sqlite':
            # SQLite: Use EXISTS with json_each() table-valued function
            # json_each expands the array into rows, each with a 'value' column
            # IMPORTANT: Add json_type check to gracefully handle non-array fields.
            # json_type returns 'array' for arrays, other values for scalars/objects.
            # If field is not an array, condition evaluates to FALSE (no match, no error).
            if isinstance(value, str) and not case_sensitive:
                # A string member matches ONLY a JSON-string array element, mirroring the
                # string-only contract sqlite_text_guard/pg_text_guard apply to the scalar
                # string operators. Without the json_each.type='text' guard a NUMERIC element
                # would be compared via SQLite's double->text rendering (an out-of-int64 /
                # high-precision / trailing-zero / scientific-notation number renders
                # differently than PostgreSQL's exact jsonb element text), so a string member
                # could match a numeric element on SQLite but not PostgreSQL. Restricting to
                # text elements also excludes boolean elements (json_each.value is 1/0),
                # consistent with the case-sensitive @> path and PostgreSQL's
                # jsonb_typeof(elem)='string' filter below.
                self.conditions.append(
                    f"(json_type(metadata, '{json_path}') = 'array' AND "
                    f"EXISTS (SELECT 1 FROM json_each(metadata, '{json_path}') "
                    f"WHERE json_each.type = 'text' AND LOWER(json_each.value) = LOWER({placeholder})))",
                )
            elif isinstance(value, bool):
                # SQLite JSON renders a boolean element's json_each.value as 1/0 -- the
                # SAME value an integer 1/0 element yields -- so guard on json_each.type
                # IN ('true','false') to match a JSON boolean element ONLY, never a
                # numeric 1/0. This keeps array_contains parity with PostgreSQL's
                # type-exact ``@> '<json bool>'::jsonb`` containment and consistent with
                # the EQ/NE/IN boolean contract.
                self.conditions.append(
                    f"(json_type(metadata, '{json_path}') = 'array' AND "
                    f"EXISTS (SELECT 1 FROM json_each(metadata, '{json_path}') "
                    f"WHERE json_each.type IN ('true', 'false') AND json_each.value = {placeholder}))",
                )
                self.parameters.append(1 if value else 0)
                return  # Early return since we already added the parameter
            else:
                # Numbers and case-sensitive strings. A numeric member matches only JSON
                # number array elements: guard json_each.type so it never matches a JSON
                # boolean element (whose json_each.value is 1/0, the same value an integer
                # 1/0 element yields), matching PostgreSQL's type-exact ``@> '<num>'::jsonb``.
                # A case-sensitive string member matches only JSON-string elements: without
                # the type='text' guard, json_each.value renders a NESTED array/object
                # element as its minified JSON text, which a string member CAN equal (e.g.
                # value '["x","y"]' against element ["x","y"]), while PostgreSQL's
                # type-exact ``@> '"<str>"'::jsonb`` containment never matches a string
                # against a container element. Mirrors the case-insensitive branch above.
                type_guard = (
                    "json_each.type IN ('integer', 'real') AND "
                    if isinstance(value, (int, float))
                    else "json_each.type = 'text' AND "
                )
                self.conditions.append(
                    f"(json_type(metadata, '{json_path}') = 'array' AND "
                    f"EXISTS (SELECT 1 FROM json_each(metadata, '{json_path}') "
                    f'WHERE {type_guard}json_each.value = {placeholder}))',
                )
            self.parameters.append(value)
        else:  # postgresql
            # PostgreSQL: Use @> containment operator for array containment check.
            # IMPORTANT: We use json.dumps() + ::jsonb cast instead of to_jsonb() because:
            # - to_jsonb() is polymorphic (anyelement) and requires type information
            # - asyncpg sends integers/floats/booleans as type "unknown" to PostgreSQL
            # - This causes "could not determine polymorphic type" error
            # - By using json.dumps() in Python and ::jsonb cast in SQL, we avoid this issue
            # This pattern is also used in app/repositories/context_repository/updates.py for metadata patching.
            # IMPORTANT: We wrap in CASE WHEN jsonb_typeof() = 'array' to gracefully handle
            # non-array fields. Without this check, jsonb_array_elements_text() throws
            # "cannot extract elements from a scalar" error on scalar fields.
            # ONE accessor for both the flat (``->'key'``) and nested (``#>'{"a","b"}'``)
            # shapes: pg_json_accessor is the single place a dotted key becomes an array
            # literal, so this path cannot drift from the scalar operators' quoting (an
            # unquoted ``null`` segment would parse as a SQL NULL element and make the
            # whole accessor NULL -- see _pg_path_literal).
            json_acc = pg_json_accessor(key_path)
            if isinstance(value, str) and not case_sensitive:
                # A string member matches ONLY a JSON-string array element (parity with
                # the SQLite json_each.type='text' branch and the string-only scalar
                # operators): iterate elements as jsonb, keep only jsonb_typeof(elem)=
                # 'string', and compare the unquoted text (elem #>> '{}'). Comparing a
                # NUMERIC element as text would diverge from SQLite's double-rendered
                # number text. Wrap in CASE to handle non-array fields gracefully.
                elem_text = pg_ci("elem #>> '{}'")
                self.conditions.append(
                    f"(CASE WHEN jsonb_typeof({json_acc}) = 'array' "
                    f'THEN EXISTS (SELECT 1 FROM jsonb_array_elements({json_acc}) AS elem '
                    f"WHERE jsonb_typeof(elem) = 'string' AND {elem_text} = {pg_ci(placeholder)}) "
                    f'ELSE FALSE END)',
                )
                self.parameters.append(value)
            elif isinstance(value, float):
                # A float member matches a numeric element with the SAME exact/double
                # semantics as the scalar operators: a bare @> containment matches only
                # the float's canonical decimal form, which diverges from SQLite's
                # exact int-vs-double element comparison above 2**53. Iterate the number
                # elements and reuse the shared discriminator so array_contains stays
                # consistent with the scalar path (only the documented int-origin /
                # canonical-double-form residual remains).
                elem_num = "(elem #>> '{}')::NUMERIC"
                compare = pg_numeric_compare(elem_num, '=', placeholder, value)
                self.conditions.append(
                    f"(CASE WHEN jsonb_typeof({json_acc}) = 'array' "
                    f'THEN EXISTS (SELECT 1 FROM jsonb_array_elements({json_acc}) AS elem '
                    f"WHERE jsonb_typeof(elem) = 'number' AND {compare}) ELSE FALSE END)",
                )
                self.parameters.append(value)
            else:
                # Case-sensitive string, int, or bool: use @> operator with json.dumps() +
                # ::jsonb (exact for ints/bools; a canonical string match for strings).
                # Wrap in CASE to handle non-array fields gracefully.
                self.conditions.append(
                    f"(CASE WHEN jsonb_typeof({json_acc}) = 'array' "
                    f'THEN {json_acc} @> {placeholder}::jsonb ELSE FALSE END)',
                )
                self.parameters.append(json.dumps(value))
