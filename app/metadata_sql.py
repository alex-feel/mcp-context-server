"""Pure SQL-fragment builders for metadata filters on SQLite and PostgreSQL.

Validates metadata keys, renders JSON paths and PostgreSQL accessors, and emits
the per-backend JSON-type guards, the exact PostgreSQL numeric comparison that
mirrors SQLite's integer/double semantics, the bound-value normalization, and
the LIKE and GLOB escapes. Every function is stateless; ``MetadataQueryBuilder``
composes them into a WHERE clause.
"""

import re
import string

# ASCII-only lowercase table: folds A-Z to a-z and leaves every other character (including
# non-ASCII letters like 'É') untouched, matching SQLite's built-in ASCII-only LOWER() and the
# PostgreSQL pg_ci() SQL fold. Used to lower IN/NOT_IN string members so the bound parameter
# folds the SAME way as the accessor expression on BOTH backends (case-insensitive matching is
# ASCII-fold only -- SQLite cannot Unicode-fold without an ICU extension).
_ASCII_LOWER_TABLE = str.maketrans(string.ascii_uppercase, string.ascii_lowercase)


# int64 bounds: SQLite reads a JSON integer OUTSIDE this range as REAL (nearest
# double), so only within-range integral stored values can be int-origin; an
# out-of-int64 integer must take the double comparison to match SQLite.
_INT64_MIN = '-9223372036854775808'
_INT64_MAX = '9223372036854775807'
# float8 overflow threshold = 2**1024 - 2**970, the exact midpoint between DBL_MAX
# and 2**1024: a NUMERIC of magnitude >= this rounds to +/-Infinity on a ``::float8``
# cast (SQLSTATE 22003), so map it to +/-inf -- matching PostgreSQL's own float8
# overflow point -- rather than aborting the whole query. It MUST be the true overflow
# boundary, NOT DBL_MAX's shortest-repr decimal (1.7976931348623157e308, strictly
# smaller): a stored value in the finite band (DBL_MAX, midpoint) casts to a FINITE
# DBL_MAX on PostgreSQL, so mapping it to Infinity would flip every comparison against a
# DBL_MAX-range param on PostgreSQL only. This literal is authoritative for PostgreSQL
# and for SQLite builds whose strtod flips at the IEEE midpoint (<= 3.40.x, e.g. the
# shipped Docker image); newer SQLite builds (observed 3.47.x/3.49.x) round a stored
# integer in a narrow band at/above the midpoint to a FINITE DBL_MAX instead of inf, so
# on those runtimes the guard compares that value as Infinity while SQLite compares it
# finite -- a version-dependent, irreducible residual documented in pg_numeric_compare,
# since no single literal tracks SQLite's version-specific flip point.
_FLOAT8_OVERFLOW = str(2**1024 - 2**970)

# Exact decimal of 2**-1075, the IEEE round-half-to-even boundary at/below which a
# finite NUMERIC underflows to 0.0 when cast to float8. PostgreSQL raises 22003 on that
# underflow (aborting the whole query -- the symmetric low-magnitude twin of the
# overflow abort above), while SQLite reads the same stored value as 0.0, so safe_float8
# maps this band to 0 to restore both the no-abort contract and cross-backend parity.
# Built as an exact string: ``2**-1075`` underflows to 0.0 as a Python float, and
# ``Decimal.scaleb`` rounds to context precision, so neither yields the exact boundary --
# and the value just above it (the smallest denormal, 2**-1074) is a legal float8 both
# engines keep, so the threshold must be exact, not approximate.
_FLOAT8_TINY = '0.' + '0' * (1075 - len(str(5**1075))) + str(5**1075)


def is_safe_key(key: str) -> bool:
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

    # Only allow alphanumeric, dots, underscores, and hyphens.
    # fullmatch (not match) so a trailing newline is rejected: in Python
    # `$` also matches immediately before a single trailing '\n', so
    # re.match(r'^...$', 'status\n') would pass and diverge across backends
    # (SQLite json_extract('$.a.status\n') misses while PostgreSQL's #>>
    # array-literal parse trims the newline and matches).
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', key):
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


def build_json_path(key: str) -> str:
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


def _pg_path_literal(key_path: str) -> str:
    """Build the PostgreSQL ``text[]`` path literal for a dotted key path.

    Every segment is DOUBLE-QUOTED. PostgreSQL's array-literal parser reads an
    unquoted, case-insensitive bareword ``null`` as a genuine SQL NULL element, and
    ``#>>``/``#>`` return NULL as soon as ANY path element is NULL, so an object key
    legitimately spelled ``null``/``NULL``/``Null`` would silently collapse the whole
    accessor to NULL: ``a.null eq x`` then matches nothing on PostgreSQL while SQLite
    matches it, and ``a.null not_exists`` returns the very entry that DOES carry the
    key. Quoting makes every segment a literal string, so the two backends traverse
    the same path. No escaping is needed because ``is_safe_key`` and
    ``MetadataFilter.validate_key`` restrict segments to ``[A-Za-z0-9_-]``, so no
    quote, backslash, comma, brace or whitespace can ever reach the literal.

    Args:
        key_path: Dot-separated key WITHOUT the ``$.`` JSONPath prefix.

    Returns:
        A quoted PostgreSQL array-literal string such as ``{"a","b"}``.
    """
    return '{' + ','.join(f'"{segment}"' for segment in key_path.split('.')) + '}'


def pg_text_accessor(key_path: str) -> str:
    """PostgreSQL ``->>``/``#>>`` accessor extracting a key as TEXT.

    A dotted ``key_path`` is split into ``#>>'{"a","b","c"}'`` array notation so a
    nested path is TRAVERSED. PostgreSQL ``->>'a.b.c'`` would instead look up
    a single top-level key literally named ``a.b.c`` (never traversing), so
    every operator that accesses metadata as TEXT MUST route through this
    helper to stay consistent with the SQLite ``json_extract`` traversal and
    with one another. A flat key uses ``->>'key'`` -- a plain key name, not an
    array literal, so it needs no quoting (see :func:`_pg_path_literal`).

    Args:
        key_path: Dot-separated key WITHOUT the ``$.`` JSONPath prefix.

    Returns:
        A PostgreSQL JSON text accessor expression for the metadata column.
    """
    if '.' in key_path:
        return f"metadata#>>'{_pg_path_literal(key_path)}'"
    return f"metadata->>'{key_path}'"


def pg_json_accessor(key_path: str) -> str:
    """PostgreSQL ``->``/``#>`` accessor extracting a key as JSONB.

    Nested ``key_path`` becomes ``#>'{"a","b","c"}'`` array notation; a flat key
    uses ``->'key'``. Used where the JSONB value itself is needed rather than
    its text form (for example ``jsonb_typeof``). The nested form routes through
    :func:`_pg_path_literal` so a ``null`` segment stays a literal key.

    Args:
        key_path: Dot-separated key WITHOUT the ``$.`` JSONPath prefix.

    Returns:
        A PostgreSQL JSONB accessor expression for the metadata column.
    """
    if '.' in key_path:
        return f"metadata#>'{_pg_path_literal(key_path)}'"
    return f"metadata->'{key_path}'"


def normalize_value(value: str | float | bool | None, *, backend_type: str) -> str | int | float | None:
    """Normalize value for SQL comparison based on backend type.

    Args:
        value: Value to normalize
        backend_type: Backend type ('sqlite' or 'postgresql') whose boolean encoding applies

    Returns:
        Normalized value for SQL parameter binding

    Note:
        Boolean handling differs by backend:
        - SQLite: Booleans stored as integers (0/1) in JSON
        - PostgreSQL: JSONB ->> extracts booleans as TEXT ('true'/'false')
    """
    if isinstance(value, bool):
        if backend_type == 'postgresql':
            # PostgreSQL JSONB ->> returns 'true' or 'false' as TEXT for booleans
            return 'true' if value else 'false'
        # SQLite stores JSON booleans as integers (0/1)
        return 1 if value else 0
    # Handle None/null
    if value is None:
        return None
    # Keep strings, numbers as-is
    return value


def in_list_text(value: str | float | bool, *, backend_type: str, lower: bool) -> str:
    """TEXT form of an IN / NOT_IN list member matching the stored JSON text.

    Booleans must match the per-backend stored boolean text that EQ uses via
    :func:`normalize_value`: '1'/'0' on SQLite (CAST(json_extract ... AS TEXT)
    of a JSON boolean) and 'true'/'false' on PostgreSQL (->>). A blanket
    ``str(bool)`` would bind 'True'/'False', which matches neither backend, so
    an IN containing a boolean would silently match nothing (NOT_IN everything).
    Other values use ``str``; case-folding is applied to genuine strings only.

    Args:
        value: A list member from an IN / NOT_IN filter.
        backend_type: Backend type ('sqlite' or 'postgresql') whose boolean text applies.
        lower: Whether the case-insensitive branch is active (fold strings).

    Returns:
        The TEXT parameter to bind for this member.
    """
    if isinstance(value, bool):
        if backend_type == 'postgresql':
            return 'true' if value else 'false'
        return '1' if value else '0'
    text = str(value)
    # ASCII-only fold (NOT str.lower(), which is full-Unicode) so the bound IN/NOT_IN
    # member folds the SAME way as the accessor's SQL fold (SQLite LOWER / pg_ci) -- else a
    # non-ASCII member would diverge between backends or against its own accessor.
    return text.translate(_ASCII_LOWER_TABLE) if lower and isinstance(value, str) else text


def pg_numeric_compare(num: str, sql_op: str, placeholder: str, value: float) -> str:
    """Compare a NUMERIC stored expression to a numeric param, matching SQLite's semantics.

    The stored value is read as exact ``NUMERIC`` and NEVER down-cast: a stored-side ``DOUBLE
    PRECISION`` cast truncated a stored integer > 2**53, and a ``float8::numeric`` of the param
    rounds the PARAM to ~15 significant digits -- both diverge from SQLite. SQLite parses a
    JSON INTEGER within int64 to an exact int64 (integers OUTSIDE int64, and JSON floats, to
    the nearest double), and asyncpg binds the param in a ``NUMERIC`` context as the param's
    EXACT double value, so:

    - INTEGER param: compare exact ``NUMERIC`` for every stored value. Within int64 this is
      exact on both engines (SQLite reads the stored JSON integer as an exact int64). A stored
      integer OUTSIDE int64 -- which SQLite reads as its nearest double -- leaves a documented,
      irreducible residual on one narrow corner (see below): PostgreSQL cannot materialize a
      ``float8``'s exact decimal value, so there is no faithful SQL reconstruction of SQLite's
      double-vs-int64 comparison, and exact ``NUMERIC`` is the closest available (it diverges on
      the fewest inputs of any option).
    - FLOAT param: reproduce SQLite's per-type snapping via a ``CASE`` over the shared
      :func:`pg_int_origin_probe` discriminator, whose ELSE arm compares double-vs-double
      through :func:`pg_safe_float8`. Factoring both out keeps each expensive expression --
      the probe and the two long decimal literals inside ``safe_float8`` -- emitted ONCE per
      comparison instead of once per branch arm, which is what bounds the generated statement
      text (see ``MAX_METADATA_CLAUSE_CHARS``); the semantics are unchanged. The exact
      ``NUMERIC`` compare is kept ONLY for a stored value that is integral, WITHIN int64, AND
      provably int-origin -- NOT equal to ``(stored::float8)::text::NUMERIC``, the shortest
      round-trip decimal of its nearest double. Everything else (fractional, out-of-int64, or
      an integral value that IS a double's canonical decimal form) is compared double-vs-double
      via ``safe_float8`` on both sides. The probe MUST route through ``::text``: ``float8out``
      emits the shortest round-trip decimal (Ryu, PostgreSQL 12+; the app pins
      ``extra_float_digits`` in ``server_settings`` so a cluster override cannot revert it to
      ``%.15g`` and misclassify), while the direct ``float8::NUMERIC`` cast rounds to ~15
      digits. The int64 guard and ``safe_float8`` are load-bearing: the exact-form probe casts
      to ``float8`` and would raise 22003 on a stored value at/beyond the float8 overflow
      threshold (``_FLOAT8_OVERFLOW`` = 2**1024 - 2**970), so it runs ONLY in the within-int64
      nested branch (CASE guarantees only the matching branch is evaluated, unlike ``AND`` which
      PostgreSQL may not short-circuit), and the double branch maps a NUMERIC at/beyond that
      overflow threshold to +/-inf to mirror SQLite's REAL read of a huge JSON integer while
      leaving the finite band (DBL_MAX, threshold) to cast to a finite DBL_MAX on PostgreSQL and
      on SQLite builds that flip at the IEEE midpoint (<= 3.40.x; newer builds keep a narrow band
      above the midpoint finite -- an irreducible residual below). ``safe_float8`` ALSO clamps the
      symmetric low-magnitude band -- a nonzero NUMERIC of magnitude <= ``_FLOAT8_TINY`` (2**-1075,
      the round-to-zero boundary) -- to 0, because such a value underflows to 0.0 when cast and
      PostgreSQL raises 22003 while SQLite reads it as 0.0, so clamping restores both parity and
      the no-abort contract. A
      stored ``0.3`` still equals a ``0.3`` param, a stored integer ``2**53+1`` is never
      collapsed onto a ``2**53`` param, a stored ``float(2**55)`` matches its own value, an
      out-of-int64 integer compares by double (matching SQLite), and a 309-digit integer no
      longer aborts the query.

    Known irreducible residuals (no faithful SQL reconstruction exists):
    - FLOAT param: an in-int64 INT-origin stored value that happens to equal the canonical
      decimal form of some double is indistinguishable from a float-origin one after ``jsonb``
      normalization and takes the ``float8`` comparison, while SQLite compares it exactly; no
      provenance survives to separate the two.
    - INT param: SQLite reads a stored integer OUTSIDE int64 as its nearest double, then
      compares that double's EXACT value against the exact int64 param. PostgreSQL cannot
      materialize a ``float8``'s exact decimal (``float8::numeric`` rounds to ~15 digits and
      ``float8::text::numeric`` yields the shortest round-trip decimal, neither equal to the
      exact binary value), so the exact-``NUMERIC`` compare used here diverges from SQLite on a
      narrow corner -- e.g. a stored value in ``[-2**63-1024, -2**63-1]`` (SQLite rounds to
      ``-2**63``) against a ``-2**63`` param. A ``float8``-vs-``float8`` compare would not
      reduce this: it merely shifts the same-size window to the positive corner (a stored
      ``2**63`` against a ``2**63-1`` param) while also corrupting large in-range params, so
      exact ``NUMERIC`` is retained as the minimum-divergence option.
    - Overflow boundary (SQLite version): ``_FLOAT8_OVERFLOW`` = 2**1024 - 2**970 is PostgreSQL's
      exact float8 overflow point, so a stored integer >= it maps to +/-inf on both engines when
      SQLite's strtod also flips there (<= 3.40.x, e.g. the shipped Docker image). Newer SQLite
      builds (observed 3.47.x/3.49.x) round a stored integer in a narrow band at/above the midpoint
      to a FINITE DBL_MAX, so on those runtimes this guard compares it as Infinity while SQLite
      compares it finite. No single literal tracks SQLite's version-specific flip point, and the
      midpoint is authoritative for PostgreSQL (and correct for glibc/Python and older SQLite), so
      it is retained; the divergence is a narrow, host-dependent residual.

    Args:
        num: A SQL expression yielding the stored value as ``NUMERIC``.
        sql_op: The comparison operator (``=``, ``!=``, ``>``, ``>=``, ``<``, ``<=``).
        placeholder: The bound-parameter placeholder for this comparison.
        value: The numeric filter param (int or float; bool handled separately upstream).

    Returns:
        A boolean SQL expression comparing the stored number to the param.
    """
    # ``isinstance(value, int)`` (not ``float``) distinguishes an int param from a float under
    # the numeric-tower ``value: float`` annotation; bool is excluded upstream. An INTEGER
    # param compares exact ``NUMERIC`` for EVERY stored value: for a stored integer OUTSIDE
    # int64 (which SQLite reads as its nearest double) this leaves a documented, irreducible
    # residual on one narrow corner -- see the docstring -- because PostgreSQL cannot
    # materialize a ``float8``'s exact decimal value (both ``::numeric`` and ``::text::numeric``
    # are lossy), and a ``float8``-vs-``float8`` compare would only shift the same-size
    # divergence to the positive corner while corrupting large in-range params. Exact
    # ``NUMERIC`` is the closest faithful reconstruction available in SQL.
    if isinstance(value, int):
        return f'{num} {sql_op} {placeholder}'
    return (
        f'CASE WHEN {pg_int_origin_probe(num)} '
        f'THEN {num} {sql_op} {placeholder} '
        f'ELSE {pg_safe_float8(num)} {sql_op} ({placeholder})::float8 END'
    )


def pg_int_origin_probe(num: str) -> str:
    """Boolean SQL: is the stored NUMERIC provably an INT-origin value SQLite reads exactly?

    True only when the stored value is integral, WITHIN int64, and NOT equal to
    ``(stored::float8)::text::NUMERIC`` (the shortest round-trip decimal of its nearest
    double) -- exactly the condition under which :func:`pg_numeric_compare` keeps the
    exact ``NUMERIC`` comparison instead of comparing double-vs-double. The nested
    ``CASE`` is load-bearing, NOT cosmetic: the ``::float8`` probe would raise 22003 on a
    stored value at/beyond the float8 overflow threshold, so it may only be evaluated
    inside the within-int64 branch, and ``CASE`` (unlike ``AND``, which PostgreSQL may not
    short-circuit) guarantees that. Emitting the probe as ONE boolean expression lets the
    caller reference the expensive ``safe_float8`` fallback ONCE instead of once per
    comparison arm, which keeps the generated statement text small (see
    ``MAX_METADATA_CLAUSE_CHARS``). A NULL stored value yields FALSE here and takes the
    double branch, the same outcome the nested-CASE form produced.

    Args:
        num: A SQL expression yielding the stored value as ``NUMERIC``.

    Returns:
        A boolean SQL expression selecting the exact-``NUMERIC`` comparison.
    """
    return (
        f'CASE WHEN {num} = trunc({num}) AND {num} BETWEEN {_INT64_MIN} AND {_INT64_MAX} '
        f'THEN (({num}::float8)::text::NUMERIC <> {num}) ELSE FALSE END'
    )


def pg_safe_float8(num: str) -> str:
    """``float8`` form of a stored NUMERIC that can never raise 22003.

    Maps a magnitude at/beyond the float8 overflow threshold to +/-Infinity (mirroring
    SQLite's REAL read of a huge JSON integer) and the symmetric nonzero underflow band
    (magnitude <= 2**-1075) to 0 (mirroring SQLite's 0.0 read), so a legal stored value
    never aborts the whole query on the ``::float8`` cast. Shared by the scalar
    comparisons and the IN / NOT_IN membership groups so both emit the two long decimal
    literals ONCE per filter.

    Args:
        num: A SQL expression yielding the stored value as ``NUMERIC``.

    Returns:
        A ``float8``-typed SQL expression for the stored value.
    """
    return (
        f"CASE WHEN {num} >= {_FLOAT8_OVERFLOW} THEN 'infinity'::float8 "
        f"WHEN {num} <= -{_FLOAT8_OVERFLOW} THEN '-infinity'::float8 "
        f'WHEN {num} <> 0 AND {num} BETWEEN -{_FLOAT8_TINY} AND {_FLOAT8_TINY} '
        f'THEN (0)::float8 '
        f'ELSE {num}::float8 END'
    )


def pg_numeric_body(key_path: str, sql_op: str, placeholder: str, value: float) -> str:
    """PostgreSQL scalar numeric comparison body (without the JSON-number type guard).

    Thin wrapper: builds the ``NUMERIC`` accessor for ``key_path`` and delegates to
    :func:`pg_numeric_compare`, the shared exact/double discriminator also used by
    ``array_contains`` numeric members.

    Args:
        key_path: The metadata key path (without the leading ``$.``).
        sql_op: The comparison operator (``=``, ``!=``, ``>``, ``>=``, ``<``, ``<=``).
        placeholder: The bound-parameter placeholder for this comparison.
        value: The numeric filter param (int or float; bool handled separately upstream).

    Returns:
        A boolean SQL expression comparing the stored number to the param.
    """
    num = f'({pg_text_accessor(key_path)})::NUMERIC'
    return pg_numeric_compare(num, sql_op, placeholder, value)


def pg_number_guard(key_path: str) -> str:
    """PostgreSQL predicate restricting a numeric operator to JSON numbers.

    Parity counterpart of :func:`sqlite_number_guard`: ``jsonb_typeof(...) = 'number'`` is
    true only for a JSON number, so a non-number value -- text, boolean, JSON null, or an
    absent key -- never matches any numeric operator and the query never aborts on a
    ``(metadata->>'k')::NUMERIC`` of non-numeric text. Paired with :func:`pg_numeric_body`
    (which assumes a JSON number) as an explicit ``guard AND body``; the explicit boolean guard
    (rather than a value-or-NULL accessor) keeps NOT_IN's ``present AND NOT match``
    deterministic for a present non-number on BOTH backends.

    Args:
        key_path: The metadata key path (without the leading ``$.``).

    Returns:
        A boolean SQL predicate true only when the path holds a JSON number.
    """
    return f"jsonb_typeof({pg_json_accessor(key_path)}) = 'number'"


def sqlite_number_guard(json_path: str) -> str:
    """SQLite predicate restricting a numeric operator to JSON numbers.

    ``json_type(metadata, '$.k')`` returns 'integer'/'real' only for JSON
    numbers (booleans are 'true'/'false', JSON null is 'null', an absent path
    yields SQL NULL), so this guard makes numeric EQ/NE/GT/GTE/LT/LTE ignore
    every non-number value -- mirroring the PostgreSQL
    ``jsonb_typeof(...) = 'number'`` accessor for cross-backend parity. The
    ``json_path`` is a validated ``$.``-prefixed path (see ``is_safe_key``),
    safe to inline.

    Args:
        json_path: The validated ``$.``-prefixed metadata path.

    Returns:
        A boolean SQL predicate true only when the path holds a JSON number.
    """
    return f"json_type(metadata, '{json_path}') IN ('integer', 'real')"


def sqlite_bool_guard(json_path: str) -> str:
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


def sqlite_text_guard(json_path: str) -> str:
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


def pg_text_guard(key_path: str) -> str:
    """PostgreSQL predicate restricting a STRING operator to JSON-string-typed values.

    Parity counterpart of :func:`sqlite_text_guard`: a string operator matches a
    stored value only when ``jsonb_typeof`` is 'string'. A stored JSON number is
    excluded so it is never compared as text -- PostgreSQL ``->>`` renders the exact
    arbitrary-precision JSON number text that SQLite (double-rendered) cannot
    reproduce, which would diverge for out-of-int64 / high-precision numbers.

    Args:
        key_path: The metadata key path (without the leading ``$.``).

    Returns:
        A boolean SQL predicate true only when the path holds a JSON string.
    """
    return f"jsonb_typeof({pg_json_accessor(key_path)}) = 'string'"


def pg_ci(expr: str) -> str:
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


def escape_like_value(value: str) -> str:
    """Escape LIKE wildcards in a literal so it matches as a literal substring.

    Escapes the backslash escape character first, then the ``%`` (any run) and
    ``_`` (any single char) wildcards, for use with an explicit ``ESCAPE '\\'``
    clause on BOTH backends. Without this a filter value containing ``%`` or
    ``_`` (e.g. ``"50%"``) would be interpreted as a pattern and return
    over-broad/wrong results. Mirrors ``_escape_like`` in app/repositories/context_repository/search.py.

    Args:
        value: The literal substring to embed in a LIKE pattern.

    Returns:
        The escaped literal (the ``%`` sentinels are added in the SQL).
    """
    return value.replace('\\', '\\\\').replace('%', '\\%').replace('_', '\\_')


def escape_glob_pattern(value: str) -> str:
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
