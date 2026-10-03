"""Value validation for ``MetadataFilter`` and the shared value guards in ``app.metadata_types``.

A value whose type does not match its operator is rejected at construction (a loud
``ValidationError``) rather than passing validation and producing no SQL condition,
which would silently drop the filter and return over-broad results. The same guards
reject ARRAY_CONTAINS list and null values, integers outside the signed 64-bit range,
and NUL or unpaired-surrogate strings identically on both backends.
"""

import pytest
from pydantic import ValidationError

from app.metadata_types import MAX_IN_LIST_MEMBERS
from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


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
    """IN accepts a list value (control)."""
    metadata_filter = MetadataFilter(key='status', operator=MetadataOperator.IN, value=['a', 'b'])
    assert metadata_filter.value == ['a', 'b']


@pytest.mark.parametrize('operator', [MetadataOperator.IN, MetadataOperator.NOT_IN])
def test_membership_operator_rejects_list_over_member_cap(operator: MetadataOperator) -> None:
    """IN/NOT_IN reject a membership list larger than MAX_IN_LIST_MEMBERS.

    Each member binds one SQL placeholder in a single statement, so an unbounded
    client-supplied list could overflow the backend bind limit inside connection
    scope (a non-ControlFlowError that charges the circuit breaker); the cap keeps
    the failure a structured validation error on both backends.
    """
    oversized: list[str | int | float | bool] = [f'v{i}' for i in range(MAX_IN_LIST_MEMBERS + 1)]
    with pytest.raises(ValidationError, match='at most'):
        MetadataFilter(key='status', operator=operator, value=oversized)


@pytest.mark.parametrize('operator', [MetadataOperator.IN, MetadataOperator.NOT_IN])
def test_membership_operator_accepts_list_at_member_cap(operator: MetadataOperator) -> None:
    """A membership list of exactly MAX_IN_LIST_MEMBERS members remains valid (inclusive cap)."""
    at_cap: list[str | int | float | bool] = [f'v{i}' for i in range(MAX_IN_LIST_MEMBERS)]
    metadata_filter = MetadataFilter(key='status', operator=operator, value=at_cap)
    assert metadata_filter.value == at_cap


@pytest.mark.parametrize('typo_key', ['op', 'operators', 'case-sensitive', 'values'])
def test_unknown_key_rejected(typo_key: str) -> None:
    """An unknown construction key raises ValidationError instead of being ignored.

    Under Pydantic's default ``extra='ignore'`` a misspelled key (e.g. ``'op'`` for
    ``'operator'``) is silently dropped, the field keeps its default, and the filter
    silently runs as EQ -- returning a wrong result set with no error. With
    ``extra='forbid'`` the typo is a loud validation error routed through the
    structured channel.
    """
    filter_dict: dict[str, object] = {'key': 'priority', typo_key: 'gt', 'value': 5}
    with pytest.raises(ValidationError, match='[Ee]xtra'):
        MetadataFilter.model_validate(filter_dict)


def test_known_keys_still_accepted() -> None:
    """All four declared fields construct normally (control for extra='forbid')."""
    metadata_filter = MetadataFilter(
        key='priority',
        operator=MetadataOperator.GT,
        value=5,
        case_sensitive=True,
    )
    assert metadata_filter.operator is MetadataOperator.GT
    assert metadata_filter.case_sensitive is True


def test_contains_still_accepts_string_value() -> None:
    """CONTAINS accepts a string value (control)."""
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


class TestArrayContainsValidation:
    """Tests for ARRAY_CONTAINS operator validation."""

    def test_array_contains_rejects_list_value(self) -> None:
        """Test that array_contains rejects list values."""
        with pytest.raises(ValueError, match='requires a single value'):
            MetadataFilter(
                key='technologies',
                operator=MetadataOperator.ARRAY_CONTAINS,
                value=['python', 'fastapi'],
            )

    def test_array_contains_rejects_none_value(self) -> None:
        """Test that array_contains rejects None value."""
        with pytest.raises(ValueError, match='requires a non-null value'):
            MetadataFilter(
                key='technologies',
                operator=MetadataOperator.ARRAY_CONTAINS,
                value=None,
            )

    def test_array_contains_accepts_string_value(self) -> None:
        """Test that array_contains accepts string value."""
        f = MetadataFilter(
            key='technologies',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value='python',
        )
        assert f.value == 'python'

    def test_array_contains_accepts_integer_value(self) -> None:
        """Test that array_contains accepts integer value."""
        f = MetadataFilter(
            key='priority_levels',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=5,
        )
        assert f.value == 5

    def test_array_contains_accepts_float_value(self) -> None:
        """Test that array_contains accepts float value."""
        f = MetadataFilter(
            key='scores',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=85.5,
        )
        assert f.value == 85.5

    def test_array_contains_accepts_boolean_value(self) -> None:
        """Test that array_contains accepts boolean value."""
        f = MetadataFilter(
            key='flags',
            operator=MetadataOperator.ARRAY_CONTAINS,
            value=True,
        )
        assert f.value is True


class TestOutOfInt64FilterRejection:
    """Integer filter values outside the signed 64-bit range are rejected
    uniformly on BOTH backends.

    SQLite binds a Python int as a 64-bit column value and raises OverflowError for
    anything outside [-2**63, 2**63-1], aborting the whole search, while PostgreSQL
    binds it into an arbitrary-precision NUMERIC and matches -- a cross-backend
    divergence. The validator forbids the divergent case on both backends instead.
    """

    @pytest.mark.parametrize('value', [10**20, 2**63, -(2**63) - 1, 99999999999999999999999999])
    def test_scalar_out_of_int64_rejected(self, value: int) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.EQ, value=value)
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.GT, value=value)

    def test_list_member_out_of_int64_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.IN, value=[7, 10**20])
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.NOT_IN, value=[10**20])

    def test_array_contains_out_of_int64_rejected(self) -> None:
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match='64-bit'):
            MetadataFilter(key='v', operator=MetadataOperator.ARRAY_CONTAINS, value=2**63)

    @pytest.mark.parametrize('value', [2**63 - 1, -(2**63), 0, 9999, True, False])
    def test_in_range_and_bool_accepted(self, value: int | bool) -> None:
        # bool is an int subclass but always in range; in-range ints (incl. the
        # int64 boundaries) are accepted unchanged.
        MetadataFilter(key='v', operator=MetadataOperator.EQ, value=value)

    @pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
    @pytest.mark.parametrize('value', [10**20, 2**63, -(2**63) - 1])
    def test_simple_filter_out_of_int64_rejected(self, backend_type: str, value: int) -> None:
        # The simple metadata={} equality path goes through add_simple_filter, which
        # bypasses MetadataFilter validation; it must reject an out-of-int64 integer
        # the same way on BOTH backends (ValueError, before any backend-specific bind)
        # instead of aborting the search on SQLite (OverflowError) while PostgreSQL
        # matches. The guard fires before the backend branch, so a single raise covers
        # both backend_type values.
        builder = MetadataQueryBuilder(backend_type=backend_type)
        with pytest.raises(ValueError, match='64-bit'):
            builder.add_simple_filter('k', value)

    @pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
    @pytest.mark.parametrize('value', [2**63 - 1, -(2**63), 0, 9999, True, False, 'x'])
    def test_simple_filter_in_range_and_non_int_accepted(
        self, backend_type: str, value: str | int | bool,
    ) -> None:
        # In-range ints (incl. the int64 boundaries), bools, and strings pass the
        # guard and build a condition normally on both backends.
        builder = MetadataQueryBuilder(backend_type=backend_type)
        builder.add_simple_filter('k', value)
        assert builder.conditions  # a condition was built, no rejection


class TestNulAndSurrogateRejection:
    """A NUL (U+0000) or unpaired UTF-16 surrogate in a string is rejected uniformly.

    Both sequences store and match on SQLite but abort the query on PostgreSQL
    (asyncpg rejects the bind or the jsonb parser rejects the escape), and the
    driver error -- not a ControlFlowError -- charges the circuit breaker. Rejecting
    them at the filter-value guards (reject_nul) and the store/update walker
    (unstorable_string_error) makes both backends fail fast and identically,
    mirroring the reject_non_finite / reject_out_of_int64 parity guards.
    """

    def test_pg_bind_reject_reason_detects_nul_and_surrogate(self) -> None:
        """The low-level predicate flags a NUL and a lone surrogate, passes clean text."""
        from app.metadata_types import pg_bind_reject_reason

        assert pg_bind_reject_reason('clean text') is None
        assert pg_bind_reject_reason('') is None
        nul_reason = pg_bind_reject_reason('a\x00b')
        assert nul_reason is not None
        assert 'NUL' in nul_reason
        surrogate_reason = pg_bind_reject_reason('a\ud800b')
        assert surrogate_reason is not None
        assert 'surrogate' in surrogate_reason

    @pytest.mark.parametrize('bad', ['done\x00', 'x\ud800'])
    def test_reject_nul_filter_value_rejected_both_paths(self, bad: str) -> None:
        """A NUL/surrogate filter value is rejected on the advanced and simple paths.

        The value binds and matches on SQLite but aborts the query and charges the
        circuit breaker on PostgreSQL, so it is rejected on both for parity.
        """
        with pytest.raises(ValueError, match='NUL|surrogate'):
            MetadataFilter(key='status', operator=MetadataOperator.EQ, value=bad)

        with pytest.raises(ValueError, match='NUL|surrogate'):
            MetadataQueryBuilder(backend_type='sqlite').add_simple_filter('status', bad)

        # A NUL/surrogate member inside an IN list is rejected too.
        with pytest.raises(ValueError, match='NUL|surrogate'):
            MetadataFilter(key='status', operator=MetadataOperator.IN, value=['ok', bad])

    def test_reject_nul_allows_clean_values_and_non_strings(self) -> None:
        """reject_nul is a no-op for clean strings, numbers, booleans, None, and clean lists."""
        from app.metadata_types import reject_nul

        reject_nul('clean')
        reject_nul('unicode-é中')
        reject_nul(2.5)
        reject_nul(True)
        reject_nul(None)
        clean_list: list[str | int | float | bool] = ['a', 'b']
        reject_nul(clean_list)

    def test_unstorable_string_error_walks_keys_values_and_lists(self) -> None:
        """The store/update walker catches a NUL/surrogate in a scalar, a value, a KEY, or a list."""
        from app.metadata_types import unstorable_string_error

        assert unstorable_string_error('plain text') is None
        assert unstorable_string_error({'a': 1, 'b': ['ok', {'c': 'fine'}], 'd': True}) is None
        assert unstorable_string_error(['tag-one', 'tag-two']) is None

        assert unstorable_string_error('bad\x00text') is not None
        assert unstorable_string_error('bad\ud800text') is not None
        # NUL in a nested value.
        assert unstorable_string_error({'note': {'deep': 'x\x00y'}}) is not None
        # NUL in a metadata KEY (not only values) -- PostgreSQL jsonb rejects it too.
        key_message = unstorable_string_error({'k\x00ey': 'value'})
        assert key_message is not None
        assert 'key' in key_message
        # NUL in a list member (e.g. a tag list).
        assert unstorable_string_error(['ok', 'ta\x00g']) is not None

    def test_sqlite_binds_nul_string_documenting_the_divergence(self) -> None:
        """SQLite binds and round-trips a NUL-bearing string (the exact divergence guarded).

        PostgreSQL's asyncpg rejects the same bind, so without the guards the two
        backends diverge; this pins the SQLite half of the divergence the guards close.
        """
        import sqlite3

        db = sqlite3.connect(':memory:')
        db.execute('CREATE TABLE t (v TEXT)')
        db.execute('INSERT INTO t (v) VALUES (?)', ('a\x00b',))
        stored = db.execute('SELECT v FROM t').fetchone()[0]
        assert stored == 'a\x00b'
