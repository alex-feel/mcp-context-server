"""PostgreSQL metadata-filter SQL never casts stored data to a non-text type outside the total helpers.

A generated condition may run on stored values of any JSON type, including rows
the caller cannot read, so no condition may be able to raise on a stored value.
Casting stored text to a non-text type is the one construct in the builder that
can: PostgreSQL does not define the order in which it evaluates the operands of
AND and OR, so a type guard beside a bare cast does not keep the cast from
running. The only casts of stored data the builder may emit are the ones inside
the total helpers, whose CASE guards fix the evaluation order.

The guard renders the PostgreSQL SQL of every builder path, removes the spans the
total helpers emit (computed by calling those helpers with the same arguments),
and fails on any remaining cast whose operand reads stored data and whose target
is not text. Casts of bound parameters and literals are not flagged. The reads of
ingest-managed entries compare stored metadata in every statement too, so their
rendered statements pass the same detector with no allowlist at all.
"""

from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from unittest.mock import Mock

import pytest

from app.access_scope import AccessScope
from app.backends.base import BackendAccess
from app.backends.base import StorageBackend
from app.backends.base import base_relation
from app.ids import generate_id
from app.metadata_sql import pg_int_origin_probe
from app.metadata_sql import pg_numeric_accessor
from app.metadata_sql import pg_safe_float8
from app.metadata_sql import pg_total_numeric
from app.metadata_types import MetadataOperator
from app.repositories.context_repository import ContextRepository
from tests.core._metadata_paths import METADATA_PATHS
from tests.core._metadata_paths import MetadataPath
from tests.core._pg_sql_text import column_derived_casts

# The numeric reading of one array element inside an array_contains float member.
ELEMENT_NUMERIC = pg_total_numeric('elem', "elem #>> '{}'")

# What an allowlisted span is replaced with: an identifier that reads no stored data.
_ALLOWED = 'allowed_total_span'

_NUMERIC_OPERATORS = (
    MetadataOperator.EQ,
    MetadataOperator.NE,
    MetadataOperator.GT,
    MetadataOperator.GTE,
    MetadataOperator.LT,
    MetadataOperator.LTE,
    MetadataOperator.IN,
    MetadataOperator.NOT_IN,
)


def _allowlisted_spans(key: str) -> list[str]:
    """Return the total-helper spans a path on ``key`` may emit, longest first.

    The probe and the safe float8 form each contain the accessor they were built
    from, so they are removed before it.

    Args:
        key: The dotted metadata key path.

    Returns:
        The spans, longest first.
    """
    spans: list[str] = []
    for numeric in (pg_numeric_accessor(key), ELEMENT_NUMERIC):
        spans.extend((pg_int_origin_probe(numeric), pg_safe_float8(numeric), numeric))
    return sorted(spans, key=len, reverse=True)


def _without_allowlisted(sql: str, key: str) -> str:
    """Return ``sql`` with every allowlisted total-helper span replaced by a neutral token."""
    for span in _allowlisted_spans(key):
        sql = sql.replace(span, _ALLOWED)
    return sql


def _has_numeric_member(path: MetadataPath) -> bool:
    """Return True when the path compares stored data against a number."""
    members: list[object] = list(path.value) if isinstance(path.value, list) else [path.value]
    return any(isinstance(member, (int, float)) and not isinstance(member, bool) for member in members)


class TestCastGuard:
    """Every rendered PostgreSQL condition casts stored data only inside the total helpers."""

    def test_paths_cover_every_operator_and_the_simple_path(self) -> None:
        """The rendered set reaches every operator, both key shapes and the simple equality path."""
        assert {path.operator for path in METADATA_PATHS} == {*MetadataOperator, None}
        assert {path.key for path in METADATA_PATHS} == {'k', 'a.b'}

    @pytest.mark.parametrize('path', METADATA_PATHS, ids=lambda path: path.case_id)
    def test_no_cast_of_stored_data_outside_the_total_helpers(self, path: MetadataPath) -> None:
        """No cast of a column-derived operand to a non-text type survives the allowlist."""
        clause, _ = path.render('postgresql')
        offending = column_derived_casts(_without_allowlisted(clause, path.key))
        assert offending == [], f'{path.case_id} casts stored data outside the total helpers: {offending}'

    @pytest.mark.parametrize(
        'path',
        [
            path
            for path in METADATA_PATHS
            if (path.operator in _NUMERIC_OPERATORS or path.operator is None) and _has_numeric_member(path)
        ],
        ids=lambda path: path.case_id,
    )
    def test_numeric_comparisons_read_stored_data_through_the_total_accessor(self, path: MetadataPath) -> None:
        """Every scalar and membership comparison against a number reads the stored value totally."""
        clause, _ = path.render('postgresql')
        assert pg_numeric_accessor(path.key) in clause

    @pytest.mark.parametrize('key', ['k', 'a.b'])
    def test_float_array_member_reads_elements_through_the_total_accessor(self, key: str) -> None:
        """A float array_contains member reads each array element totally."""
        path = next(
            path
            for path in METADATA_PATHS
            if path.key == key and path.operator == MetadataOperator.ARRAY_CONTAINS and isinstance(path.value, float)
        )
        clause, _ = path.render('postgresql')
        assert ELEMENT_NUMERIC in clause


async def _rendered_ingest_statements() -> list[str]:
    """Render both ingest reads on PostgreSQL with every optional condition present.

    Returns:
        The range read with both bounds and the owned page read with a cursor, in that order.
    """
    statements: list[str] = []

    class _RecordingConnection:
        async def fetch(self, query: str, *args: object) -> list[object]:
            _ = args
            statements.append(query)
            return []

    pg_backend = Mock()
    pg_backend.backend_type = 'postgresql'
    pg_backend.readable_relation = base_relation

    async def _execute_read(closure: Callable[[object], Awaitable[object]], *, access_scope: BackendAccess) -> object:
        _ = access_scope
        return await closure(_RecordingConnection())

    pg_backend.execute_read = _execute_read
    repo = ContextRepository(cast(StorageBackend, pg_backend))
    scope = AccessScope('alice', frozenset({'team-x'}))
    await repo.ingest_rows_in_range(scope=scope, thread_id='kb:t', source_id_after='a', source_id_through='z')
    await repo.owned_ingest_page(scope=scope, thread_id='kb:t', after=('m', generate_id()), limit=101)
    return statements


class TestIngestStatements:
    """The reads of ingest-managed entries read stored metadata through text accessors alone."""

    @pytest.mark.asyncio
    async def test_no_cast_of_stored_data_in_the_ingest_reads(self) -> None:
        """Neither rendered ingest statement casts a column-derived operand to a non-text type."""
        statements = await _rendered_ingest_statements()

        assert len(statements) == 2
        for statement in statements:
            assert 'metadata' in statement
            assert column_derived_casts(statement) == [], statement


class TestCastDetector:
    """The detector flags casts of stored data and nothing else."""

    @pytest.mark.parametrize(
        'sql',
        [
            "(jsonb_typeof(metadata->'k') = 'number' AND ((metadata->>'k')::NUMERIC > $1))",
            "((metadata->>'k')::INTEGER = $1)",
            "((metadata->>'k')::BOOLEAN = $1)",
            "CAST(metadata->>'k' AS NUMERIC) > $1",
            "(elem #>> '{}')::NUMERIC = $1",
            '''(metadata#>>'{"a","b"}')::timestamptz < $1''',
            "(ce.metadata->>'k')::float8 >= $1",
            "(metadata->>'k')::text[] = $1",
            "(metadata->>'k')::text::NUMERIC > $1",
        ],
        ids=[
            'bare-numeric', 'integer', 'boolean', 'cast-function', 'element', 'nested-timestamptz', 'aliased', 'text-array',
            'chained',
        ],
    )
    def test_flags_casts_of_stored_data(self, sql: str) -> None:
        """A cast whose operand reads stored data and whose target is not text is flagged."""
        assert column_derived_casts(sql) != []

    @pytest.mark.parametrize(
        'sql',
        [
            "metadata->>'k' = $1::TEXT",
            'allowed_total_span > ($2)::float8',
            "metadata->'k' @> $3::jsonb",
            "'infinity'::float8",
            '(0)::float8',
            "(metadata->>'k')::text = $1::TEXT",
            "metadata->'k' != 'null'::jsonb",
            "translate(metadata->>'k', 'ABC', 'abc') = translate($1::TEXT, 'ABC', 'abc')",
            "metadata->>'k::NUMERIC' = $1::TEXT",
        ],
        ids=[
            'text-param', 'float8-param', 'jsonb-param', 'literal', 'number', 'text-target', 'null-literal', 'fold', 'quoted',
        ],
    )
    def test_ignores_casts_of_parameters_literals_and_text_targets(self, sql: str) -> None:
        """Casts of bound parameters and literals, and casts to text, are not flagged."""
        assert column_derived_casts(sql) == []
