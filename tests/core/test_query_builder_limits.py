"""MetadataQueryBuilder statement budgets: accumulated bind parameters and generated clause text."""

import pytest

from app.metadata_sql import _FLOAT8_OVERFLOW
from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


class TestBindParameterBudget:
    """The builder rejects a clause whose accumulated binds exceed the shared budget.

    The per-dimension caps bound each input list individually, but capped dimensions
    multiply (filters times IN-list members), so MAX_METADATA_BIND_PARAMS is the
    aggregate backstop guaranteeing no combination of filter dimensions can grow a
    single statement past the backend's per-statement bind limit. The raised
    ValueError rides the same structured validation channel as every other filter
    validation failure, so it never charges the circuit breaker.
    """

    def test_simple_filter_over_budget_rejected(self) -> None:
        """add_simple_filter raises once the accumulated binds exceed the budget.

        param_offset counts the enclosing statement's earlier placeholders, so
        seeding it just below the budget exercises the overflow without building
        tens of thousands of filters.
        """
        from app.metadata_types import MAX_METADATA_BIND_PARAMS

        builder = MetadataQueryBuilder(backend_type='postgresql', param_offset=MAX_METADATA_BIND_PARAMS)
        with pytest.raises(ValueError, match='bind parameters'):
            builder.add_simple_filter('status', 'active')

    def test_advanced_filter_over_budget_rejected(self) -> None:
        """add_advanced_filter raises once the accumulated binds exceed the budget."""
        from app.metadata_types import MAX_METADATA_BIND_PARAMS

        builder = MetadataQueryBuilder(backend_type='sqlite', param_offset=MAX_METADATA_BIND_PARAMS)
        spec = MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5)
        with pytest.raises(ValueError, match='bind parameters'):
            builder.add_advanced_filter(spec)

    def test_within_budget_accepted(self) -> None:
        """A clause within the budget builds normally (control)."""
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_simple_filter('status', 'active')
        builder.add_advanced_filter(MetadataFilter(key='priority', operator=MetadataOperator.GT, value=5))
        where_clause, params = builder.build_where_clause()
        assert where_clause
        assert len(params) == 2


class TestGeneratedClauseTextBudget:
    """The built clause is bounded in SQL TEXT, not only in bind count.

    Bind COUNT and statement SIZE are independent dimensions on PostgreSQL: one numeric
    comparison inlines a multi-kilobyte exact/double discriminator (two long decimal
    literals plus the repeated key accessor), so input that satisfies every per-dimension
    cap AND the bind budget could still assemble a statement orders of magnitude larger
    than the request payload -- built on the event loop, too large for asyncpg's statement
    cache, and re-parsed on every call. The IN/NOT_IN numeric group emits ONE
    discriminator per filter rather than one per member, and MAX_METADATA_CLAUSE_CHARS
    backstops any remaining growth through the same structured validation channel.
    """

    @staticmethod
    def _build(backend: str, filters: list[MetadataFilter]) -> str:
        builder = MetadataQueryBuilder(backend_type=backend)
        for spec in filters:
            builder.add_advanced_filter(spec)
        clause, _ = builder.build_where_clause()
        return clause

    def test_float_in_members_do_not_multiply_the_discriminator(self) -> None:
        """The float discriminator is emitted once per FILTER, not once per member."""
        one = self._build(
            'postgresql',
            [MetadataFilter(key='k', operator=MetadataOperator.IN, value=[0.5])],
        )
        many = self._build(
            'postgresql',
            [MetadataFilter(key='k', operator=MetadataOperator.IN, value=[i + 0.5 for i in range(100)])],
        )
        overflow = _FLOAT8_OVERFLOW
        # safe_float8 inlines the overflow literal twice; that count must not scale with
        # the member count.
        assert one.count(overflow) == 2
        assert many.count(overflow) == 2
        # 100 members may add only their placeholders, not another discriminator each.
        assert len(many) - len(one) < 3_000

    def test_float_in_group_keeps_both_comparison_arms(self) -> None:
        """Hoisting the discriminator preserves the exact-NUMERIC and double arms."""
        clause = self._build(
            'postgresql',
            [MetadataFilter(key='k', operator=MetadataOperator.IN, value=[1.5, 2.5])],
        )
        assert 'trunc(' in clause  # integral / int64 / canonical-double probe
        assert '::float8)::text::NUMERIC' in clause
        assert "(metadata->>'k')::NUMERIC IN ($1, $2)" in clause  # exact arm
        assert '($1)::float8, ($2)::float8' in clause  # double arm
        assert "jsonb_typeof(metadata->'k') = 'number'" in clause

    def test_mixed_int_and_float_members_keep_their_own_semantics(self) -> None:
        """An int member always compares exact NUMERIC; only float members take the probe."""
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_advanced_filter(MetadataFilter(key='k', operator=MetadataOperator.IN, value=[7, 1.5, 9]))
        clause, params = builder.build_where_clause()
        # Binding order still follows the caller's list; only the SQL grouping differs.
        assert params == [7, 1.5, 9]
        assert "(metadata->>'k')::NUMERIC IN ($1, $3)" in clause  # int members, no probe
        assert '($2)::float8' in clause  # the float member's double arm
        # An int-only list needs no discriminator at all.
        int_only = self._build(
            'postgresql',
            [MetadataFilter(key='k', operator=MetadataOperator.IN, value=[7, 9])],
        )
        assert 'trunc(' not in int_only

    def test_cap_legal_request_stays_within_the_text_budget(self) -> None:
        """The largest cap-legal filter set builds a bounded clause on BOTH backends."""
        from app.metadata_types import MAX_METADATA_CLAUSE_CHARS

        members: list[str | int | float | bool] = [i + 0.5 for i in range(100)]
        filters = [
            MetadataFilter(key=f'k{i}', operator=MetadataOperator.IN, value=list(members))
            for i in range(100)
        ]
        pg_clause = self._build('postgresql', filters)
        sqlite_clause = self._build('sqlite', filters)
        assert len(pg_clause) < MAX_METADATA_CLAUSE_CHARS
        assert len(sqlite_clause) < MAX_METADATA_CLAUSE_CHARS
        # The PostgreSQL clause may still be larger than the SQLite one (its numeric
        # semantics need real SQL), but not by the ~1,600x that per-member expansion cost.
        assert len(pg_clause) < 50 * len(sqlite_clause)

    def test_oversized_clause_text_rejected_through_the_validation_channel(self) -> None:
        """A clause past the text budget raises the same ValueError the bind budget uses."""
        from app.metadata_types import MAX_METADATA_CLAUSE_CHARS

        # A very long key path is the remaining amplifier: the accessor is repeated inside
        # every numeric comparison, so one filter can outgrow the budget on one bind.
        long_key = 'k' * (MAX_METADATA_CLAUSE_CHARS // 4)
        builder = MetadataQueryBuilder(backend_type='postgresql')
        with pytest.raises(ValueError, match='too much SQL text'):
            builder.add_advanced_filter(MetadataFilter(key=long_key, operator=MetadataOperator.GT, value=1.5))

    def test_ordinary_filters_stay_well_within_the_text_budget(self) -> None:
        """Control: a realistic filter set is nowhere near the budget."""
        from app.metadata_types import MAX_METADATA_CLAUSE_CHARS

        clause = self._build(
            'postgresql',
            [
                MetadataFilter(key='status', operator=MetadataOperator.EQ, value='active'),
                MetadataFilter(key='priority', operator=MetadataOperator.GT, value=3),
                MetadataFilter(key='tags', operator=MetadataOperator.ARRAY_CONTAINS, value='x'),
            ],
        )
        assert 0 < len(clause) < MAX_METADATA_CLAUSE_CHARS // 100
