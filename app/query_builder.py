"""SQL query builder for metadata filtering with security validation."""


import json
import re
import string
from typing import Any

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
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

# ASCII-only lowercase table: folds A-Z to a-z and leaves every other character (including
# non-ASCII letters like 'É') untouched, matching SQLite's built-in ASCII-only LOWER() and the
# PostgreSQL _pg_ci() SQL fold. Used to lower IN/NOT_IN string members so the bound parameter
# folds the SAME way as the accessor expression on BOTH backends (case-insensitive matching is
# ASCII-fold only -- SQLite cannot Unicode-fold without an ICU extension).
_ASCII_LOWER_TABLE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


class MetadataQueryBuilder:
    """Build SQL WHERE clauses for metadata filtering with security validation.

    Provides safe SQL generation for JSON metadata filtering with support for
    16 different operators and nested JSON paths.
    """

    def __init__(
        self,
        backend_type: str = 'sqlite',
        json_extract_fn: str | None = None,
        param_offset: int = 0,
        table_alias: str | None = None,
    ) -> None:
        """Initialize the query builder.

        Args:
            backend_type: Backend type ('sqlite' or 'postgresql') for placeholder generation
            json_extract_fn: Optional JSON extraction function name override
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
        self.json_extract_fn = json_extract_fn or ('json_extract' if backend_type == 'sqlite' else 'jsonb_extract_path_text')
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

        # The simple metadata={} equality path bypasses MetadataFilter validation,
        # so reject out-of-int64 integers here too: without this guard an
        # out-of-range int aborts the search on SQLite (OverflowError) while
        # PostgreSQL matches it -- the cross-backend divergence the advanced
        # metadata_filters path already rejects via the same helper.
        reject_out_of_int64(value)

        json_path = self._build_json_path(key)
        placeholder = self._placeholder()
        if self.backend_type == 'sqlite':
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean equality matches JSON booleans only (parity with PostgreSQL)
                guard = self._sqlite_bool_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (parity with PostgreSQL)
                guard = self._sqlite_number_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            else:
                # String (or None): the text guard restricts the match to a JSON-string-typed
                # stored value, so a stored NUMBER/boolean/null is excluded (a number's text form
                # diverges across backends). The CAST is a harmless text->text identity here.
                guard = self._sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) = {placeholder})",
                )
        else:  # postgresql
            # Route through the shared accessor so a nested key traverses via
            # #>> array notation instead of being read as a literal top-level key.
            acc = self._pg_text_accessor(json_path[2:])  # strip $. prefix
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean equality matches JSON booleans only (->> returns 'true'/'false' text)
                bguard = f"jsonb_typeof({self._pg_json_accessor(json_path[2:])}) = 'boolean'"
                self.conditions.append(f'({bguard} AND {acc} = {placeholder}::TEXT)')
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (guard), compared with SQLite's exact
                # integer/double semantics (_pg_numeric_body).
                guard = self._pg_number_guard(json_path[2:])
                body = self._pg_numeric_body(json_path[2:], '=', placeholder, value)
                self.conditions.append(f'({guard} AND ({body}))')
            else:
                # Text comparison: the text guard restricts the match to a JSON-string-typed
                # stored value, so a stored number/boolean/null is excluded (parity with SQLite).
                guard = self._pg_text_guard(json_path[2:])
                self.conditions.append(f'({guard} AND {acc} = {placeholder}::TEXT)')
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
        elif operator == MetadataOperator.ARRAY_CONTAINS and not isinstance(value, list) and value is not None:
            self._add_array_contains_condition(json_path, value, case_sensitive)

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
        if not re.match(r'^[a-zA-Z0-9_.-]+$', key):
            return False

        # Reject empty path segments (leading/trailing/consecutive dots): they build a
        # malformed array literal like '{a,,b}' on PostgreSQL (a raw parser error) and a
        # silently-divergent JSON path on SQLite. Forbid on both backends, mirroring the
        # numeric-segment rejection below and MetadataFilter.validate_key.
        if '' in key.split('.'):
            return False

        # Reject integer path segments AFTER the first. A dotted segment that is an
        # integer (e.g. 'items.0', 'a.-1') array-indexes on PostgreSQL
        # (metadata#>>'{items,0}') but resolves to a literal object key on SQLite
        # ($.items.0, which is NOT $.items[0]) -- a silent backend divergence. The
        # first segment always indexes the metadata object itself (never an array),
        # so only later segments can land on an array parent.
        return not any(re.fullmatch(r'-?\d+', seg) for seg in key.split('.')[1:])

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
    def _pg_text_accessor(key_path: str) -> str:
        """PostgreSQL ``->>``/``#>>`` accessor extracting a key as TEXT.

        A dotted ``key_path`` is split into ``#>>'{a,b,c}'`` array notation so a
        nested path is TRAVERSED. PostgreSQL ``->>'a.b.c'`` would instead look up
        a single top-level key literally named ``a.b.c`` (never traversing), so
        every operator that accesses metadata as TEXT MUST route through this
        helper to stay consistent with the SQLite ``json_extract`` traversal and
        with one another. A flat key uses ``->>'key'``.

        Args:
            key_path: Dot-separated key WITHOUT the ``$.`` JSONPath prefix.

        Returns:
            A PostgreSQL JSON text accessor expression for the metadata column.
        """
        if '.' in key_path:
            array_path = '{' + ','.join(key_path.split('.')) + '}'
            return f"metadata#>>'{array_path}'"
        return f"metadata->>'{key_path}'"

    @staticmethod
    def _pg_json_accessor(key_path: str) -> str:
        """PostgreSQL ``->``/``#>`` accessor extracting a key as JSONB.

        Nested ``key_path`` becomes ``#>'{a,b,c}'`` array notation; a flat key
        uses ``->'key'``. Used where the JSONB value itself is needed rather than
        its text form (for example ``jsonb_typeof``).

        Args:
            key_path: Dot-separated key WITHOUT the ``$.`` JSONPath prefix.

        Returns:
            A PostgreSQL JSONB accessor expression for the metadata column.
        """
        if '.' in key_path:
            array_path = '{' + ','.join(key_path.split('.')) + '}'
            return f"metadata#>'{array_path}'"
        return f"metadata->'{key_path}'"

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

    def _in_list_text(self, value: str | float | bool, *, lower: bool) -> str:
        """TEXT form of an IN / NOT_IN list member matching the stored JSON text.

        Booleans must match the per-backend stored boolean text that EQ uses via
        :meth:`_normalize_value`: '1'/'0' on SQLite (CAST(json_extract ... AS TEXT)
        of a JSON boolean) and 'true'/'false' on PostgreSQL (->>). A blanket
        ``str(bool)`` would bind 'True'/'False', which matches neither backend, so
        an IN containing a boolean would silently match nothing (NOT_IN everything).
        Other values use ``str``; case-folding is applied to genuine strings only.

        Args:
            value: A list member from an IN / NOT_IN filter.
            lower: Whether the case-insensitive branch is active (fold strings).

        Returns:
            The TEXT parameter to bind for this member.
        """
        if isinstance(value, bool):
            if self.backend_type == 'postgresql':
                return 'true' if value else 'false'
            return '1' if value else '0'
        text = str(value)
        # ASCII-only fold (NOT str.lower(), which is full-Unicode) so the bound IN/NOT_IN
        # member folds the SAME way as the accessor's SQL fold (SQLite LOWER / _pg_ci) -- else a
        # non-ASCII member would diverge between backends or against its own accessor.
        return text.translate(_ASCII_LOWER_TABLE) if lower and isinstance(value, str) else text

    def _pg_numeric_body(self, key_path: str, sql_op: str, placeholder: str, value: float) -> str:
        """PostgreSQL numeric comparison body (without the JSON-number type guard) matching
        SQLite's exact integer/double comparison.

        The stored value is read as exact ``NUMERIC`` and NEVER down-cast: a stored-side ``DOUBLE
        PRECISION`` cast truncated a stored integer > 2**53, and a ``float8::numeric`` of the param
        rounds to ~15 significant digits -- both diverge from SQLite. SQLite parses a JSON INTEGER
        to an exact int64 but a JSON FLOAT to the nearest double, and asyncpg binds the param in a
        ``NUMERIC`` context as the param's EXACT double value, so:

        - INTEGER param: compare exact ``NUMERIC`` for every stored value (SQLite compares int64 /
          double-vs-int64 exactly).
        - FLOAT param: branch on the stored value's integrality to reproduce SQLite's per-type
          snapping -- an integral stored value compares exact ``NUMERIC`` against the param's exact
          double value; a fractional stored value compares double-vs-double (both snapped to
          ``float8``), so a stored ``0.3`` still equals a ``0.3`` param (same IEEE double) while a
          stored ``2**53+1`` is never collapsed onto a ``2**53`` param.

        ``$N`` stays a single ``numeric``-typed parameter (used bare in the integral branch, cast
        via ``(<ph>)::float8`` in the fractional branch), so asyncpg binds the float once as its
        exact double value.

        Args:
            key_path: The metadata key path (without the leading ``$.``).
            sql_op: The comparison operator (``=``, ``!=``, ``>``, ``>=``, ``<``, ``<=``).
            placeholder: The bound-parameter placeholder for this comparison.
            value: The numeric filter param (int or float; bool handled separately upstream).

        Returns:
            A boolean SQL expression comparing the stored number to the param.
        """
        num = f'({self._pg_text_accessor(key_path)})::NUMERIC'
        # ``isinstance(value, int)`` (not ``float``) distinguishes an int param from a float under
        # the numeric-tower ``value: float`` annotation; bool is excluded upstream.
        if isinstance(value, int):
            return f'{num} {sql_op} {placeholder}'
        return (
            f'CASE WHEN {num} = trunc({num}) '
            f'THEN {num} {sql_op} {placeholder} '
            f'ELSE {num}::float8 {sql_op} ({placeholder})::float8 END'
        )

    def _pg_number_guard(self, key_path: str) -> str:
        """PostgreSQL predicate restricting a numeric operator to JSON numbers.

        Parity counterpart of :meth:`_sqlite_number_guard`: ``jsonb_typeof(...) = 'number'`` is
        true only for a JSON number, so a non-number value -- text, boolean, JSON null, or an
        absent key -- never matches any numeric operator and the query never aborts on a
        ``(metadata->>'k')::NUMERIC`` of non-numeric text. Paired with :meth:`_pg_numeric_body`
        (which assumes a JSON number) as an explicit ``guard AND body``; the explicit boolean guard
        (rather than a value-or-NULL accessor) keeps NOT_IN's ``present AND NOT match``
        deterministic for a present non-number on BOTH backends.

        Args:
            key_path: The metadata key path (without the leading ``$.``).

        Returns:
            A boolean SQL predicate true only when the path holds a JSON number.
        """
        return f"jsonb_typeof({self._pg_json_accessor(key_path)}) = 'number'"

    def _sqlite_number_guard(self, json_path: str) -> str:
        """SQLite predicate restricting a numeric operator to JSON numbers.

        ``json_type(metadata, '$.k')`` returns 'integer'/'real' only for JSON
        numbers (booleans are 'true'/'false', JSON null is 'null', an absent path
        yields SQL NULL), so this guard makes numeric EQ/NE/GT/GTE/LT/LTE ignore
        every non-number value -- mirroring the PostgreSQL
        ``jsonb_typeof(...) = 'number'`` accessor for cross-backend parity. The
        ``json_path`` is a validated ``$.``-prefixed path (see ``_is_safe_key``),
        safe to inline.

        Args:
            json_path: The validated ``$.``-prefixed metadata path.

        Returns:
            A boolean SQL predicate true only when the path holds a JSON number.
        """
        return f"json_type(metadata, '{json_path}') IN ('integer', 'real')"

    def _sqlite_bool_guard(self, json_path: str) -> str:
        """SQLite predicate restricting a boolean operator to JSON booleans.

        ``json_type(metadata, '$.k')`` returns 'true'/'false' only for JSON booleans
        (a numeric 0/1 is 'integer', the string 'true' is 'text'), so this guard makes
        a boolean EQ/NE match ONLY a JSON boolean -- mirroring the PostgreSQL
        ``jsonb_typeof(...) = 'boolean'`` guard. Without it, SQLite (which compares the
        typed ``json_extract`` 0/1) would match a stored numeric 0/1 that PostgreSQL
        (comparing ``->>`` text 'true'/'false') would not, a silent cross-backend
        divergence on mixed-type metadata.

        Args:
            json_path: The validated ``$.``-prefixed metadata path.

        Returns:
            A boolean SQL predicate true only when the path holds a JSON boolean.
        """
        return f"json_type(metadata, '{json_path}') IN ('true', 'false')"

    def _sqlite_text_guard(self, json_path: str) -> str:
        """SQLite predicate restricting a STRING operator to JSON-string-typed values.

        String operators (eq/ne with a string value, the ordered comparisons with a
        string value, contains/starts_with/ends_with, and string IN/NOT_IN members) match
        a stored value ONLY when it is a JSON string. A stored JSON NUMBER is excluded
        because comparing it as text diverges across backends: SQLite parses a JSON number
        into a 64-bit int / IEEE double and renders THAT (losing the exact text for an
        out-of-int64 or high-precision value), while PostgreSQL ``->>`` returns the exact
        original JSON number text. ``json_type(...) = 'text'`` is the only value SQLite
        renders identically to PostgreSQL, so restricting string operators to it is the
        parity-by-construction contract -- symmetric with the number-only numeric
        operators and the boolean-only boolean operators. A JSON boolean and a JSON null
        are likewise excluded.

        Args:
            json_path: The validated ``$.``-prefixed metadata path.

        Returns:
            A boolean SQL predicate true only when the path holds a JSON string.
        """
        return f"json_type(metadata, '{json_path}') = 'text'"

    def _pg_text_guard(self, key_path: str) -> str:
        """PostgreSQL predicate restricting a STRING operator to JSON-string-typed values.

        Parity counterpart of :meth:`_sqlite_text_guard`: a string operator matches a
        stored value only when ``jsonb_typeof`` is 'string'. A stored JSON number is
        excluded so it is never compared as text -- PostgreSQL ``->>`` renders the exact
        arbitrary-precision JSON number text that SQLite (double-rendered) cannot
        reproduce, which would diverge for out-of-int64 / high-precision numbers.

        Args:
            key_path: The metadata key path (without the leading ``$.``).

        Returns:
            A boolean SQL predicate true only when the path holds a JSON string.
        """
        return f"jsonb_typeof({self._pg_json_accessor(key_path)}) = 'string'"

    @staticmethod
    def _pg_ci(expr: str) -> str:
        """ASCII-only case fold for PostgreSQL, matching SQLite's ASCII-only ``LOWER()``.

        PostgreSQL's ``LOWER()`` under a UTF-8 locale folds the FULL Unicode range (e.g.
        'CAFÉ' -> 'café'), but SQLite's built-in ``LOWER()`` folds ASCII A-Z ONLY ('É' left
        untouched). Using ``LOWER()`` on both backends therefore returns DIFFERENT result sets
        for non-ASCII text under the default case-insensitive matching. ``translate`` folds
        exactly the 26 ASCII letters, making PostgreSQL's case-insensitive comparison
        byte-for-byte identical to SQLite's. SQLite cannot do Unicode folding without an ICU
        extension, so ASCII-only folding is the portable, parity-by-construction contract for
        case-insensitive metadata matching on both backends.

        Args:
            expr: The SQL text expression to ASCII-lowercase.

        Returns:
            A ``translate(...)`` SQL expression folding only ASCII A-Z to a-z.
        """
        return f"translate({expr}, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ', 'abcdefghijklmnopqrstuvwxyz')"

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
                guard = self._sqlite_bool_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (parity with PostgreSQL)
                guard = self._sqlite_number_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') = {placeholder})")
            elif isinstance(value, str) and not case_sensitive:
                guard = self._sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND LOWER(json_extract(metadata, '{json_path}')) = LOWER({placeholder}))",
                )
            else:
                # Case-sensitive string EQ (and value=None). The text guard restricts the match
                # to a JSON-string-typed stored value, so a stored NUMBER/boolean/null is
                # excluded (a number's text form diverges across backends). The CAST is a
                # harmless text->text identity under the guard.
                guard = self._sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) = {placeholder})",
                )
        else:  # postgresql
            # Route through the shared accessor so a nested key traverses via
            # #>> array notation rather than being read as a literal top-level key.
            acc = self._pg_text_accessor(key_path)
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean equality matches JSON booleans only (->> returns 'true'/'false' text)
                bguard = f"jsonb_typeof({self._pg_json_accessor(key_path)}) = 'boolean'"
                self.conditions.append(f'({bguard} AND {acc} = {placeholder}::TEXT)')
            elif isinstance(value, (int, float)):
                # Numeric equality matches JSON numbers only (guard), compared with SQLite's exact
                # integer/double semantics (_pg_numeric_body).
                guard = self._pg_number_guard(key_path)
                body = self._pg_numeric_body(key_path, '=', placeholder, value)
                self.conditions.append(f'({guard} AND ({body}))')
            elif isinstance(value, str) and not case_sensitive:
                # Case-insensitive string EQ. The text guard restricts the match to a
                # JSON-string-typed stored value (a stored number/boolean is excluded, parity
                # with SQLite). ASCII-only case fold (_pg_ci) matches SQLite's ASCII-only LOWER().
                guard = self._pg_text_guard(key_path)
                ci_acc = self._pg_ci(acc)
                ci_val = self._pg_ci(f'{placeholder}::TEXT')
                self.conditions.append(f'({guard} AND {ci_acc} = {ci_val})')
            else:
                # Case-sensitive string EQ: text guard restricts to JSON-string stored values.
                guard = self._pg_text_guard(key_path)
                self.conditions.append(f'({guard} AND {acc} = {placeholder}::TEXT)')
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
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean NE matches JSON booleans that differ (parity with PostgreSQL):
                # a non-boolean value never satisfies a boolean operator on either backend.
                guard = self._sqlite_bool_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') != {placeholder})")
            elif isinstance(value, (int, float)):
                # Numeric NE matches JSON numbers that differ (parity with PostgreSQL):
                # a non-number value never satisfies a numeric operator on either backend.
                guard = self._sqlite_number_guard(json_path)
                self.conditions.append(f"({guard} AND json_extract(metadata, '{json_path}') != {placeholder})")
            elif isinstance(value, str) and not case_sensitive:
                guard = self._sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND LOWER(json_extract(metadata, '{json_path}')) != LOWER({placeholder}))",
                )
            else:
                # Case-sensitive string NE (and value=None). The text guard restricts the match
                # to a JSON-string-typed stored value, so a stored NUMBER/boolean/null never
                # participates (a number's text form diverges across backends). The CAST is a
                # harmless text->text identity under the guard.
                guard = self._sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) != {placeholder})",
                )
        else:  # postgresql
            acc = self._pg_text_accessor(key_path)
            # CRITICAL: Check bool BEFORE int/float (bool is subclass of int in Python)
            if isinstance(value, bool):
                # Boolean NE matches JSON booleans that differ (->> returns 'true'/'false' text)
                bguard = f"jsonb_typeof({self._pg_json_accessor(key_path)}) = 'boolean'"
                self.conditions.append(f'({bguard} AND {acc} != {placeholder}::TEXT)')
            elif isinstance(value, (int, float)):
                # Numeric NE matches JSON numbers that differ (guard excludes non-numbers, parity
                # with SQLite), compared with SQLite's exact integer/double semantics.
                guard = self._pg_number_guard(key_path)
                body = self._pg_numeric_body(key_path, '!=', placeholder, value)
                self.conditions.append(f'({guard} AND ({body}))')
            elif isinstance(value, str) and not case_sensitive:
                # Case-insensitive string NE. The text guard restricts the match to a
                # JSON-string-typed stored value (a stored number/boolean is excluded, parity
                # with SQLite).
                guard = self._pg_text_guard(key_path)
                ci_acc = self._pg_ci(acc)
                ci_val = self._pg_ci(f'{placeholder}::TEXT')
                self.conditions.append(f'({guard} AND {ci_acc} != {ci_val})')
            else:
                guard = self._pg_text_guard(key_path)
                self.conditions.append(f'({guard} AND {acc} != {placeholder}::TEXT)')
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
            # Numeric comparison matches JSON numbers only on BOTH backends; a
            # non-number value (text/bool/json-null/absent) never matches, so the
            # backends agree without relying on SQLite's CAST(text AS NUMERIC) coercion.
            if self.backend_type == 'sqlite':
                guard = self._sqlite_number_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS NUMERIC) {sql_op} {placeholder})",
                )
            else:  # postgresql - JSON numbers only (guard); compared with SQLite's exact
                # integer/double semantics (_pg_numeric_body) so the boundary row matches SQLite.
                guard = self._pg_number_guard(key_path)
                body = self._pg_numeric_body(key_path, sql_op, placeholder, value)
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
                guard = self._sqlite_text_guard(json_path)
                self.conditions.append(
                    f"({guard} AND CAST(json_extract(metadata, '{json_path}') AS TEXT) {sql_op} {placeholder})",
                )
            else:  # postgresql
                guard = self._pg_text_guard(key_path)
                acc = self._pg_text_accessor(key_path)
                self.conditions.append(f'({guard} AND {acc} COLLATE "C" {sql_op} {placeholder}::TEXT)')
            self.parameters.append(str(value))

    def _membership_match_sql(
        self,
        json_path: str,
        key_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> str:
        """Build a SQL predicate true when the stored value matches any list member.

        Members are matched TYPE-AWARE so the SQLite and PostgreSQL result sets are
        identical and no member is compared across JSON types:

        - STRING members match a JSON-STRING-typed stored value only (text guard), with
          optional ASCII case-fold. A stored JSON NUMBER is NOT compared as text -- its
          text form diverges across backends for out-of-int64 / high-precision values
          (SQLite double-rendered vs PostgreSQL exact ``->>``), mirroring the string-only
          EQ/NE/comparison contract.
        - NUMERIC members match a JSON-NUMBER-typed stored value NUMERICALLY (number
          guard / numeric accessor), so ``in [1, 2, 3]`` keeps matching stored numbers
          identically on both backends without any text comparison.
        - BOOLEAN members match a JSON-BOOLEAN-typed stored value only (bool guard), so a
          boolean member never collides with a same-text integer 1/0 or string
          'true'/'false'.

        Bound params are appended in placeholder order; used by IN (the predicate) and
        NOT_IN (its negation under an IS NOT NULL presence guard).

        Args:
            json_path: The validated ``$.``-prefixed metadata path (SQLite).
            key_path: The same path without the ``$.`` prefix (PostgreSQL accessor).
            values: The IN / NOT_IN member list.
            case_sensitive: When False, string members fold case (numbers/booleans never do).

        Returns:
            A parenthesized SQL predicate; appends its bound params to ``self.parameters``.
        """
        # bool is a subclass of int, so partition booleans out FIRST.
        bool_members = [v for v in values if isinstance(v, bool)]
        numeric_members = [v for v in values if not isinstance(v, bool) and isinstance(v, (int, float))]
        string_members = [v for v in values if isinstance(v, str)]
        fold = not case_sensitive and bool(string_members)
        parts: list[str] = []
        if self.backend_type == 'sqlite':
            if string_members:
                text_acc = f"json_extract(metadata, '{json_path}')"
                lhs = f'LOWER({text_acc})' if fold else text_acc
                ph = ', '.join('?' for _ in string_members)
                parts.append(f'({self._sqlite_text_guard(json_path)} AND {lhs} IN ({ph}))')
                self.parameters.extend(self._in_list_text(v, lower=fold) for v in string_members)
            if numeric_members:
                ph = ', '.join('?' for _ in numeric_members)
                parts.append(
                    f"({self._sqlite_number_guard(json_path)} AND "
                    f"json_extract(metadata, '{json_path}') IN ({ph}))",
                )
                self.parameters.extend(numeric_members)
            if bool_members:
                acc = f"CAST(json_extract(metadata, '{json_path}') AS TEXT)"
                bph = ', '.join('?' for _ in bool_members)
                parts.append(f'({self._sqlite_bool_guard(json_path)} AND {acc} IN ({bph}))')
                self.parameters.extend(self._in_list_text(v, lower=False) for v in bool_members)
        else:  # postgresql
            acc = self._pg_text_accessor(key_path)
            json_acc = self._pg_json_accessor(key_path)
            if string_members:
                start = self.param_offset + len(self.parameters) + 1
                ph = ', '.join(f'${start + i}::TEXT' for i in range(len(string_members)))
                lhs = self._pg_ci(acc) if fold else acc  # ASCII-only fold (parity with SQLite LOWER)
                parts.append(f'({self._pg_text_guard(key_path)} AND {lhs} IN ({ph}))')
                self.parameters.extend(self._in_list_text(v, lower=fold) for v in string_members)
            if numeric_members:
                # Per-member numeric equality with SQLite's exact integer/double semantics
                # (_pg_numeric_body). Wrap the whole OR-group in an explicit jsonb_typeof =
                # 'number' boolean guard (_pg_number_guard), mirroring SQLite's
                # _sqlite_number_guard AND-form: an explicit boolean guard (rather than a
                # value-or-NULL accessor) is required for NOT_IN, which is `present AND NOT
                # match` -- for a present NON-number stored value a NULL match would make NOT
                # NULL stay NULL and drop the row on PostgreSQL, while SQLite's explicit FALSE
                # guard yields NOT FALSE -> TRUE and KEEPS it. The guard forces a deterministic
                # FALSE for a non-number on BOTH backends, so IN and NOT_IN agree.
                num_guard = self._pg_number_guard(key_path)
                num_parts: list[str] = []
                for m in numeric_members:
                    pos = self.param_offset + len(self.parameters) + 1
                    member_ph = f'${pos}'
                    body = self._pg_numeric_body(key_path, '=', member_ph, m)
                    num_parts.append(f'({body})')
                    self.parameters.append(m)
                parts.append(f'({num_guard} AND (' + ' OR '.join(num_parts) + '))')
            if bool_members:
                bguard = f"jsonb_typeof({json_acc}) = 'boolean'"
                start = self.param_offset + len(self.parameters) + 1
                bph = ', '.join(f'${start + i}::TEXT' for i in range(len(bool_members)))
                parts.append(f'({bguard} AND {acc} IN ({bph}))')
                self.parameters.extend(self._in_list_text(v, lower=False) for v in bool_members)
        return '(' + ' OR '.join(parts) + ')'

    def _add_in_condition(
        self,
        json_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> None:
        """Add an IN condition for list membership.

        Delegates to :meth:`_membership_match_sql`, which compares members as TEXT but
        gates boolean members on the JSON type so a boolean member matches only a JSON
        boolean (never a same-text integer 0/1 on SQLite or string 'true'/'false' on
        PostgreSQL), keeping the backends identical.
        """
        if not values:
            self.conditions.append('0 = 1')
            return
        self.conditions.append(self._membership_match_sql(json_path, json_path[2:], values, case_sensitive))

    def _add_not_in_condition(
        self,
        json_path: str,
        values: list[str | int | float | bool],
        case_sensitive: bool,
    ) -> None:
        """Add a NOT IN condition.

        The match predicate is :meth:`_membership_match_sql` (type-aware for booleans);
        NOT_IN is its negation under an explicit presence guard so a missing key (or JSON
        null) is excluded -- the same three-valued-logic outcome the prior bare
        ``... NOT IN (...)`` produced via NULL propagation, now with boolean members
        matched JSON-boolean-only on both backends.
        """
        if not values:
            self.conditions.append('1 = 1')
            return

        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            present = f"json_extract(metadata, '{json_path}') IS NOT NULL"
        else:  # postgresql
            present = f'{self._pg_text_accessor(key_path)} IS NOT NULL'
        match = self._membership_match_sql(json_path, key_path, values, case_sensitive)
        self.conditions.append(f'({present} AND NOT {match})')

    def _add_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key exists."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_extract(metadata, '{json_path}') IS NOT NULL")
        else:  # postgresql
            self.conditions.append(f'{self._pg_text_accessor(key_path)} IS NOT NULL')

    def _add_not_exists_condition(self, json_path: str) -> None:
        """Add a condition to check if a key does not exist."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_extract(metadata, '{json_path}') IS NULL")
        else:  # postgresql
            self.conditions.append(f'{self._pg_text_accessor(key_path)} IS NULL')

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
            guard = self._sqlite_text_guard(json_path)
            if case_sensitive:
                # INSTR matches a literal substring; no LIKE wildcards to escape.
                pred = f"INSTR(json_extract(metadata, '{json_path}'), {placeholder}) > 0"
                self.parameters.append(value)
            else:
                pred = (
                    f"LOWER(json_extract(metadata, '{json_path}')) "
                    f"LIKE '%' || LOWER({placeholder}) || '%' ESCAPE '\\'"
                )
                self.parameters.append(self._escape_like_value(value))
            self.conditions.append(f'({guard} AND {pred})')
        else:  # postgresql
            guard = self._pg_text_guard(key_path)
            acc = self._pg_text_accessor(key_path)
            if case_sensitive:
                pred = f"{acc} LIKE '%' || {placeholder}::TEXT || '%' ESCAPE '\\'"
            else:
                pred = f"{self._pg_ci(acc)} LIKE '%' || {self._pg_ci(f'{placeholder}::TEXT')} || '%' ESCAPE '\\'"
            self.conditions.append(f'({guard} AND {pred})')
            self.parameters.append(self._escape_like_value(value))

    def _add_starts_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string starts-with condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        # The text guard restricts a STRING starts-with to a JSON-string-typed stored value, so
        # a stored number/boolean is never matched as text (parity by construction).
        if self.backend_type == 'sqlite':
            guard = self._sqlite_text_guard(json_path)
            if case_sensitive:
                pred = f"json_extract(metadata, '{json_path}') GLOB {placeholder} || '*'"
                self.parameters.append(self._escape_glob_pattern(value))
            else:
                pred = (
                    f"LOWER(json_extract(metadata, '{json_path}')) "
                    f"LIKE LOWER({placeholder}) || '%' ESCAPE '\\'"
                )
                self.parameters.append(self._escape_like_value(value))
            self.conditions.append(f'({guard} AND {pred})')
        else:  # postgresql
            guard = self._pg_text_guard(key_path)
            acc = self._pg_text_accessor(key_path)
            if case_sensitive:
                pred = f"{acc} LIKE {placeholder}::TEXT || '%' ESCAPE '\\'"
            else:
                pred = f"{self._pg_ci(acc)} LIKE {self._pg_ci(f'{placeholder}::TEXT')} || '%' ESCAPE '\\'"
            self.conditions.append(f'({guard} AND {pred})')
            self.parameters.append(self._escape_like_value(value))

    def _add_ends_with_condition(self, json_path: str, value: str | None, case_sensitive: bool) -> None:
        """Add a string ends-with condition."""
        if value is None:
            return

        placeholder = self._placeholder()
        key_path = json_path[2:]

        # The text guard restricts a STRING ends-with to a JSON-string-typed stored value, so a
        # stored number/boolean is never matched as text (parity by construction).
        if self.backend_type == 'sqlite':
            guard = self._sqlite_text_guard(json_path)
            if case_sensitive:
                pred = f"json_extract(metadata, '{json_path}') GLOB '*' || {placeholder}"
                self.parameters.append(self._escape_glob_pattern(value))
            else:
                pred = (
                    f"LOWER(json_extract(metadata, '{json_path}')) "
                    f"LIKE '%' || LOWER({placeholder}) ESCAPE '\\'"
                )
                self.parameters.append(self._escape_like_value(value))
            self.conditions.append(f'({guard} AND {pred})')
        else:  # postgresql
            guard = self._pg_text_guard(key_path)
            acc = self._pg_text_accessor(key_path)
            if case_sensitive:
                pred = f"{acc} LIKE '%' || {placeholder}::TEXT ESCAPE '\\'"
            else:
                pred = f"{self._pg_ci(acc)} LIKE '%' || {self._pg_ci(f'{placeholder}::TEXT')} ESCAPE '\\'"
            self.conditions.append(f'({guard} AND {pred})')
            self.parameters.append(self._escape_like_value(value))

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
            self.conditions.append(f"jsonb_typeof({self._pg_json_accessor(key_path)}) = 'null'")

    def _add_is_not_null_condition(self, json_path: str) -> None:
        """Add a condition to check if value is not JSON null."""
        key_path = json_path[2:]
        if self.backend_type == 'sqlite':
            self.conditions.append(f"json_type(metadata, '{json_path}') != 'null'")
        else:  # postgresql
            self.conditions.append(
                f"{self._pg_text_accessor(key_path)} IS NOT NULL "
                f"AND {self._pg_json_accessor(key_path)} != 'null'::jsonb",
            )

    @staticmethod
    def _escape_like_value(value: str) -> str:
        """Escape LIKE wildcards in a literal so it matches as a literal substring.

        Escapes the backslash escape character first, then the ``%`` (any run) and
        ``_`` (any single char) wildcards, for use with an explicit ``ESCAPE '\\'``
        clause on BOTH backends. Without this a filter value containing ``%`` or
        ``_`` (e.g. ``"50%"``) would be interpreted as a pattern and return
        over-broad/wrong results. Mirrors ``_escape_like`` in context_repository.py.

        Args:
            value: The literal substring to embed in a LIKE pattern.

        Returns:
            The escaped literal (the ``%`` sentinels are added in the SQL).
        """
        return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')

    @staticmethod
    def _escape_glob_pattern(value: str) -> str:
        """Neutralize SQLite GLOB metacharacters so a value matches literally.

        SQLite GLOB has NO ESCAPE clause and treats backslash as a LITERAL
        character, so backslash-escaping (the previous approach) produced patterns
        demanding a literal backslash absent from the data and silently mismatched.
        The only safe way to make the GLOB metacharacters ``*``, ``?`` and ``[``
        literal is to wrap each in a single-character bracket class (``[*]``,
        ``[?]``, ``[[]``). ``]`` is literal outside a class and backslash is
        literal, so neither needs escaping. GLOB stays case-sensitive (its intended
        behavior for case-sensitive STARTS_WITH/ENDS_WITH).

        Args:
            value: String value to escape.

        Returns:
            A GLOB pattern fragment that matches ``value`` literally.
        """
        out: list[str] = []
        for ch in value:
            if ch in '*?[':
                out.append(f'[{ch}]')
            else:
                out.append(ch)
        return ''.join(out)

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
                # string-only contract _sqlite_text_guard/_pg_text_guard apply to the scalar
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
                # A case-sensitive string member already cannot match a 1/0 boolean element
                # (text param != integer value), so it needs no extra guard.
                type_guard = "json_each.type IN ('integer', 'real') AND " if isinstance(value, (int, float)) else ''
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
            # This pattern is also used in context_repository.py for metadata patching.
            # IMPORTANT: We wrap in CASE WHEN jsonb_typeof() = 'array' to gracefully handle
            # non-array fields. Without this check, jsonb_array_elements_text() throws
            # "cannot extract elements from a scalar" error on scalar fields.
            if '.' in key_path:
                # Nested path: use #> with array notation
                path_parts = key_path.split('.')
                array_path = '{' + ','.join(path_parts) + '}'
                if isinstance(value, str) and not case_sensitive:
                    # A string member matches ONLY a JSON-string array element (parity with
                    # the SQLite json_each.type='text' branch and the string-only scalar
                    # operators): iterate elements as jsonb, keep only jsonb_typeof(elem)=
                    # 'string', and compare the unquoted text (elem #>> '{}'). Comparing a
                    # NUMERIC element as text would diverge from SQLite's double-rendered
                    # number text. Wrap in CASE to handle non-array fields gracefully.
                    elem_text = self._pg_ci("elem #>> '{}'")
                    self.conditions.append(
                        f"(CASE WHEN jsonb_typeof(metadata#>'{array_path}') = 'array' "
                        f"THEN EXISTS (SELECT 1 FROM jsonb_array_elements(metadata#>'{array_path}') AS elem "
                        f"WHERE jsonb_typeof(elem) = 'string' AND {elem_text} = {self._pg_ci(placeholder)}) "
                        f'ELSE FALSE END)',
                    )
                    self.parameters.append(value)
                else:
                    # Case-sensitive or non-string: use @> operator with json.dumps() + ::jsonb
                    # Wrap in CASE to handle non-array fields gracefully
                    self.conditions.append(
                        f"(CASE WHEN jsonb_typeof(metadata#>'{array_path}') = 'array' "
                        f"THEN metadata#>'{array_path}' @> {placeholder}::jsonb ELSE FALSE END)",
                    )
                    self.parameters.append(json.dumps(value))
            else:
                # Single-level path
                if isinstance(value, str) and not case_sensitive:
                    # A string member matches ONLY a JSON-string array element (see the nested
                    # branch above and the SQLite json_each.type='text' branch): keep only
                    # jsonb_typeof(elem)='string' and compare the unquoted text, so a numeric
                    # element is never text-compared (which would diverge from SQLite).
                    elem_text = self._pg_ci("elem #>> '{}'")
                    self.conditions.append(
                        f"(CASE WHEN jsonb_typeof(metadata->'{key_path}') = 'array' "
                        f"THEN EXISTS (SELECT 1 FROM jsonb_array_elements(metadata->'{key_path}') AS elem "
                        f"WHERE jsonb_typeof(elem) = 'string' AND {elem_text} = {self._pg_ci(placeholder)}) "
                        f'ELSE FALSE END)',
                    )
                    self.parameters.append(value)
                else:
                    # Case-sensitive or non-string: use @> operator with json.dumps() + ::jsonb
                    # Wrap in CASE to handle non-array fields gracefully
                    self.conditions.append(
                        f"(CASE WHEN jsonb_typeof(metadata->'{key_path}') = 'array' "
                        f"THEN metadata->'{key_path}' @> {placeholder}::jsonb ELSE FALSE END)",
                    )
                    self.parameters.append(json.dumps(value))
