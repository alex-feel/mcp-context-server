"""LIKE and GLOB metacharacter escaping in MetadataQueryBuilder string operators."""

import pytest

from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.query_builder import MetadataQueryBuilder


class TestMetadataQueryBuilderLikeWildcardEscaping:
    """contains/starts_with/ends_with escape LIKE wildcards (%, _) in the value.

    A filter value containing ``%`` (any run) or ``_`` (any single char) must match
    literally, so CONTAINS ``"50%"`` does not match every value containing ``"50"``
    followed by anything. The value is escaped and an explicit ``ESCAPE '\\'`` clause
    is added on every LIKE branch of both backends (the SQLite case-sensitive
    INSTR/GLOB branches need no escape and are excluded). This is about correctness,
    not injection: the value is always bound as a parameter.
    """

    @pytest.mark.parametrize('backend', ['sqlite', 'postgresql'])
    @pytest.mark.parametrize(
        'op',
        [MetadataOperator.CONTAINS, MetadataOperator.STARTS_WITH, MetadataOperator.ENDS_WITH],
        ids=lambda o: o.value,
    )
    def test_case_insensitive_like_escapes_wildcards(self, backend: str, op: MetadataOperator) -> None:
        builder = MetadataQueryBuilder(backend_type=backend)
        builder.add_advanced_filter(
            MetadataFilter(key='note', operator=op, value='50%_x', case_sensitive=False),
        )
        clause, params = builder.build_where_clause()
        assert 'LIKE' in clause
        assert "ESCAPE '\\'" in clause, f'{op.value}/{backend} missing ESCAPE clause: {clause}'
        # % and _ in the bound value are escaped so they match literally.
        assert params == ['50\\%\\_x']

    @pytest.mark.parametrize(
        'op',
        [MetadataOperator.CONTAINS, MetadataOperator.STARTS_WITH, MetadataOperator.ENDS_WITH],
        ids=lambda o: o.value,
    )
    def test_postgresql_case_sensitive_like_escapes_wildcards(self, op: MetadataOperator) -> None:
        # PostgreSQL has no GLOB/INSTR fallback, so even the case-sensitive
        # contains/starts/ends branches use LIKE and must escape.
        builder = MetadataQueryBuilder(backend_type='postgresql')
        builder.add_advanced_filter(
            MetadataFilter(key='note', operator=op, value='50%', case_sensitive=True),
        )
        clause, params = builder.build_where_clause()
        assert 'LIKE' in clause
        assert "ESCAPE '\\'" in clause, f'{op.value} missing ESCAPE clause: {clause}'
        assert params == ['50\\%']

    def test_sqlite_case_sensitive_contains_uses_instr_no_escape(self) -> None:
        # SQLite case-sensitive contains uses INSTR (literal), so the raw value is
        # bound unchanged and there is no LIKE/ESCAPE to add.
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(
            MetadataFilter(key='note', operator=MetadataOperator.CONTAINS, value='50%', case_sensitive=True),
        )
        clause, params = builder.build_where_clause()
        assert 'INSTR' in clause
        assert 'LIKE' not in clause
        assert params == ['50%']

    @pytest.mark.parametrize(
        ('op', 'expect_sql'),
        [
            (MetadataOperator.STARTS_WITH, "GLOB ? || '*'"),
            (MetadataOperator.ENDS_WITH, "GLOB '*' || ?"),
        ],
        ids=['starts_with', 'ends_with'],
    )
    def test_sqlite_case_sensitive_glob_bracket_escapes_metacharacters(
        self, op: MetadataOperator, expect_sql: str,
    ) -> None:
        """SQLite case-sensitive STARTS_WITH/ENDS_WITH bracket-escape GLOB specials.

        SQLite GLOB has NO ESCAPE clause and treats backslash as a literal, so
        backslash-escaping would make a value containing ``* ? [`` silently
        mismatch. The value's GLOB metacharacters are wrapped in single-char
        bracket classes (``[*]`` etc.) so it matches literally while GLOB stays
        case-sensitive.
        """
        builder = MetadataQueryBuilder(backend_type='sqlite')
        builder.add_advanced_filter(
            MetadataFilter(key='note', operator=op, value='a*b?[c', case_sensitive=True),
        )
        clause, params = builder.build_where_clause()
        assert 'GLOB' in clause
        assert expect_sql in clause
        # *, ?, [ each wrapped in a single-char bracket class; other chars unchanged.
        assert params == ['a[*]b[?][[]c']
        # No backslash escaping: GLOB reads a backslash as a literal character.
        assert '\\' not in params[0]
