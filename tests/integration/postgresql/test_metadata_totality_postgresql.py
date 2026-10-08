"""Totality of the PostgreSQL metadata-filter SQL over every kind of stored value.

No generated condition may raise on a stored value, because a condition can run
on rows the caller cannot read and PostgreSQL does not define the order in which
it evaluates the operands of AND and OR. The matrix renders every builder path
(tests/core/_metadata_paths.py) and evaluates it against stored values of every
JSON type in the target list of ``SELECT <expr> FROM (SELECT $1::jsonb AS
metadata) t``: the whole condition, and every maximal operand of every AND and
OR on its own, so no evaluation order a planner could choose escapes the check.
Each statement runs under a forced custom plan, where the planner folds the
stored value as a constant, and under a forced generic plan, where the executor
evaluates it at run time.

The whole condition must also decide the row exactly as SQLite decides it for
the same filter on the same stored JSON, the project's parity oracle, except in
the documented residual corners (a stored integer outside int64, and a float
filter value against an int-origin stored value), where only "no error" is
asserted.

The float-typed index pin asserts that the expression index a ``float``
METADATA_INDEXED_FIELDS entry builds serves the builder's numeric comparison
under ``enable_seqscan = off``. The reconciliation tests assert that startup
rebuilds an index whose definition differs from the current one (compared
through ``pg_get_indexdef``) in the modes that provision indexes, reports it in
the others, and keeps it when the new definition cannot be built.
"""

import contextlib
import json
import logging
import sqlite3
from collections.abc import AsyncIterator
from typing import cast
from unittest.mock import MagicMock
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

import asyncpg
import pytest
import pytest_asyncio

from app.backends import create_backend
from app.errors import ConfigurationError
from app.metadata_types import MetadataFilter
from app.metadata_types import MetadataOperator
from app.migrations import handle_metadata_indexes
from app.migrations.metadata import _generate_create_index_postgresql
from app.query_builder import MetadataQueryBuilder
from tests.core._metadata_paths import METADATA_PATHS
from tests.core._metadata_paths import MetadataPath
from tests.core._pg_sql_text import boolean_operands

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

# Stored values placed under the filtered key, as JSON text; None leaves the key absent.
STORED_VALUES: dict[str, str | None] = {
    'non_numeric_text': '"HIDDEN-secret-4711"',
    'numeric_text': '"42"',
    'true': 'true',
    'false': 'false',
    'json_null': 'null',
    'empty_object': '{}',
    'mixed_array': '[1, "a"]',
    'absent': None,
    'int': '42',
    'negative_int': '-7',
    'float': '3.5',
    'float8_max_band': '1e308',
    'beyond_float8': '1e400',
    'float8_underflow_band': '1e-320',
    'beyond_int64': str(2**63 + 1),
}

# Stored values whose SQLite reading cannot be reproduced exactly in PostgreSQL SQL.
_OUT_OF_INT64 = frozenset({'beyond_int64'})
_INT_ORIGIN = frozenset({'int', 'negative_int', 'beyond_int64'})

_PLAN_MODES = ('force_custom_plan', 'force_generic_plan')

_INDEX_DB = 'mcp_metadata_totality_index'
_FLOAT_FIELD = 'f'


def _document(key: str, json_value: str | None) -> str:
    """Return the stored metadata document holding ``json_value`` at ``key``.

    Args:
        key: The dotted metadata key path.
        json_value: The stored value as JSON text, or None to leave the key absent.

    Returns:
        The metadata document as JSON text.
    """
    *parents, leaf = key.split('.')
    body = '{}' if json_value is None else f'{{"{leaf}": {json_value}}}'
    for parent in reversed(parents):
        body = f'{{"{parent}": {body}}}'
    return body


def _has_float(value: object) -> bool:
    """Return True when a filter value is, or contains, a float."""
    members = value if isinstance(value, list) else [value]
    return any(isinstance(member, float) for member in members)


def _parity_applies(path: MetadataPath, stored_id: str) -> bool:
    """Return False in the documented residual corners, where only "no error" is asserted."""
    if stored_id in _OUT_OF_INT64:
        return False
    return not (_has_float(path.value) and stored_id in _INT_ORIGIN)


def _sqlite_matches(path: MetadataPath, document: str) -> bool:
    """Return whether SQLite's rendering of ``path`` matches a row holding ``document``.

    Args:
        path: The builder path.
        document: The stored metadata document as JSON text.

    Returns:
        True when the condition is TRUE for the row, as in a WHERE clause.
    """
    clause, params = path.render('sqlite')
    # SQLite numbers '?' placeholders in text order, so the clause's own parameters
    # come before the stored document bound in the FROM clause.
    with contextlib.closing(sqlite3.connect(':memory:')) as conn:
        row = conn.execute(f'SELECT ({clause}) FROM (SELECT ? AS metadata) t', [*params, document]).fetchone()
    return row[0] == 1


@pytest_asyncio.fixture
async def pg_conn(pg_test_url: str) -> AsyncIterator[asyncpg.Connection]:
    """Yield one connection to the throwaway PostgreSQL container.

    Yields:
        An open asyncpg connection, closed at teardown.
    """
    conn = await asyncpg.connect(pg_test_url)
    try:
        yield conn
    finally:
        await conn.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('path', METADATA_PATHS, ids=lambda path: path.case_id)
async def test_no_stored_value_makes_a_condition_raise(pg_conn: asyncpg.Connection, path: MetadataPath) -> None:
    """Neither the condition nor any of its AND/OR operands raises on any stored value.

    The whole condition also decides every row as SQLite does outside the
    documented residual corners.
    """
    clause, params = path.render('postgresql', param_offset=1)
    expressions = [clause, *boolean_operands(clause)]
    columns = ', '.join(f'({expression}) AS c{index}' for index, expression in enumerate(expressions))
    sql = f'SELECT {columns} FROM (SELECT $1::jsonb AS metadata) t'

    failures: list[str] = []
    for plan_mode in _PLAN_MODES:
        await pg_conn.execute(f'SET plan_cache_mode = {plan_mode}')
        for stored_id, json_value in STORED_VALUES.items():
            document = _document(path.key, json_value)
            try:
                row = await pg_conn.fetchrow(sql, document, *params)
            except asyncpg.PostgresError as exc:
                failures.append(f'{plan_mode} {stored_id}: SQLSTATE {exc.sqlstate}: {exc}')
                continue
            assert row is not None
            if _parity_applies(path, stored_id):
                matched = row['c0'] is True
                expected = _sqlite_matches(path, document)
                if matched != expected:
                    failures.append(f'{plan_mode} {stored_id}: PostgreSQL matches={matched}, SQLite matches={expected}')

    assert failures == [], f'{path.case_id} ({len(expressions) - 1} operands):\n' + '\n'.join(failures)


def _replace_db_name(pg_url: str, new_db: str) -> str:
    """Return ``pg_url`` with the database name replaced by ``new_db``."""
    parts = urlsplit(pg_url)
    return urlunsplit((parts.scheme, parts.netloc, f'/{new_db}', parts.query, parts.fragment))


@pytest_asyncio.fixture
async def index_db_url(pg_test_url: str) -> AsyncIterator[str]:
    """Yield an isolated database holding a bare ``context_entries`` table.

    Yields:
        The connection URL of the isolated database, dropped at teardown.
    """
    admin = await asyncpg.connect(pg_test_url)
    try:
        await admin.execute(f'DROP DATABASE IF EXISTS {_INDEX_DB}')
        await admin.execute(f'CREATE DATABASE {_INDEX_DB}')
    finally:
        await admin.close()
    url = _replace_db_name(pg_test_url, _INDEX_DB)
    setup = await asyncpg.connect(url)
    try:
        await setup.execute('CREATE TABLE context_entries (id BIGSERIAL PRIMARY KEY, metadata JSONB)')
    finally:
        await setup.close()
    try:
        yield url
    finally:
        admin = await asyncpg.connect(pg_test_url)
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS {_INDEX_DB} WITH (FORCE)')
        finally:
            await admin.close()


def _index_names(plan: object) -> set[str]:
    """Return every index name an EXPLAIN (FORMAT JSON) plan scans."""
    names: set[str] = set()
    if isinstance(plan, dict):
        node = cast('dict[str, object]', plan)
        name = node.get('Index Name')
        if isinstance(name, str):
            names.add(name)
        for child in node.values():
            names |= _index_names(child)
    elif isinstance(plan, list):
        for item in cast('list[object]', plan):
            names |= _index_names(item)
    return names


async def _scanned_indexes(conn: asyncpg.Connection, metadata_filter: MetadataFilter) -> set[str]:
    """Return the indexes the planner scans for one filter with sequential scans disabled.

    Args:
        conn: A connection to the isolated database.
        metadata_filter: The filter whose builder SQL is planned.

    Returns:
        The names of the indexes in the plan.
    """
    builder = MetadataQueryBuilder(backend_type='postgresql')
    builder.add_advanced_filter(metadata_filter)
    clause, params = builder.build_where_clause()
    async with conn.transaction():
        await conn.execute('SET LOCAL enable_seqscan = off')
        plan = await conn.fetchval(f'EXPLAIN (FORMAT JSON) SELECT id FROM context_entries WHERE {clause}', *params)
    return _index_names(json.loads(plan))


async def _seed_numbers(conn: asyncpg.Connection) -> None:
    """Store 2000 rows whose float field holds a JSON number, then refresh the statistics."""
    await conn.execute(
        f"INSERT INTO context_entries (metadata) "
        f"SELECT jsonb_build_object('{_FLOAT_FIELD}', g) FROM generate_series(1, 2000) AS g",
    )
    await conn.execute('ANALYZE context_entries')


_INTEGER_FILTER = MetadataFilter(key=_FLOAT_FIELD, operator=MetadataOperator.GT, value=42)


@pytest.mark.asyncio
async def test_float_typed_index_serves_an_integer_filter(index_db_url: str) -> None:
    """The index a float-typed indexed field builds serves an integer filter on that field.

    The index reads JSON numbers only, so rows holding any other value under the field
    are stored without error and stay outside it.
    """
    conn = await asyncpg.connect(index_db_url)
    try:
        await conn.execute(_generate_create_index_postgresql(_FLOAT_FIELD, 'float'))
        for json_value in ('"HIDDEN-secret-4711"', 'true', 'null', '{}', '[1, "a"]'):
            document = _document(_FLOAT_FIELD, json_value)
            await conn.execute('INSERT INTO context_entries (metadata) VALUES ($1::jsonb)', document)
        await _seed_numbers(conn)
        scanned = await _scanned_indexes(conn, _INTEGER_FILTER)
    finally:
        await conn.close()
    assert f'idx_metadata_{_FLOAT_FIELD}' in scanned


def _drifted_float_index(schema: str = 'public') -> str:
    """Return DDL for a float-field index whose definition differs from the current one.

    Args:
        schema: The schema holding ``context_entries``.

    Returns:
        CREATE INDEX for a cast of every non-NULL text value, which the builder's
        numeric comparisons do not match.
    """
    return (
        f'CREATE INDEX "idx_metadata_{_FLOAT_FIELD}" ON {schema}.context_entries '
        f"(((metadata->>'{_FLOAT_FIELD}')::NUMERIC)) WHERE metadata->>'{_FLOAT_FIELD}' IS NOT NULL"
    )


def _sync_settings(mode: str, fields: dict[str, str], schema: str) -> MagicMock:
    """Return a settings stand-in that configures the metadata index reconciliation."""
    stand_in = MagicMock()
    stand_in.storage.metadata_indexed_fields = fields
    stand_in.storage.metadata_index_sync_mode = mode
    stand_in.storage.postgresql_schema = schema
    stand_in.storage.postgresql_migration_timeout_s = 60.0
    return stand_in


async def _float_index(conn: asyncpg.Connection, schema: str = 'public') -> tuple[int, str] | None:
    """Return the OID and ``pg_get_indexdef`` of the float field's index, or None when absent."""
    row = await conn.fetchrow(
        'SELECT c.oid::bigint AS oid, pg_get_indexdef(c.oid) AS definition FROM pg_class c '
        'JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = $1 AND c.relname = $2',
        schema,
        f'idx_metadata_{_FLOAT_FIELD}',
    )
    return None if row is None else (row['oid'], row['definition'])


async def _reconcile(url: str, mode: str, fields: dict[str, str], schema: str = 'public') -> None:
    """Run the startup metadata index reconciliation against ``url`` in ``mode``."""
    backend = create_backend(backend_type='postgresql', connection_string=url)
    await backend.initialize()
    try:
        with patch('app.migrations.metadata.settings', _sync_settings(mode, fields, schema)):
            await handle_metadata_indexes(backend)
    finally:
        await backend.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize('mode', ['additive', 'auto'])
async def test_reconciliation_rebuilds_a_drifted_float_index(index_db_url: str, mode: str) -> None:
    """A float index whose definition differs from the current one is rebuilt once, then serves the builder."""
    conn = await asyncpg.connect(index_db_url)
    try:
        await conn.execute(_drifted_float_index())
        await _seed_numbers(conn)
        drifted = await _float_index(conn)
        assert drifted is not None
        assert f'idx_metadata_{_FLOAT_FIELD}' not in await _scanned_indexes(conn, _INTEGER_FILTER)

        await _reconcile(index_db_url, mode, {_FLOAT_FIELD: 'float'})
        rebuilt = await _float_index(conn)
        assert rebuilt is not None
        assert rebuilt[0] != drifted[0]
        assert 'jsonb_typeof' in rebuilt[1]
        assert f'idx_metadata_{_FLOAT_FIELD}' in await _scanned_indexes(conn, _INTEGER_FILTER)

        await _reconcile(index_db_url, mode, {_FLOAT_FIELD: 'float'})
        assert await _float_index(conn) == rebuilt
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_reconciliation_rebuilds_a_drifted_index_in_the_configured_schema(index_db_url: str) -> None:
    """The drift check and the rebuild address the index in the configured schema."""
    schema = 'mcp_metadata_schema'
    conn = await asyncpg.connect(index_db_url)
    try:
        await conn.execute(f'CREATE SCHEMA {schema}')
        await conn.execute(f'CREATE TABLE {schema}.context_entries (id BIGSERIAL PRIMARY KEY, metadata JSONB)')
        await conn.execute(_drifted_float_index(schema))
        drifted = await _float_index(conn, schema)
        assert drifted is not None

        await _reconcile(index_db_url, 'additive', {_FLOAT_FIELD: 'float'}, schema)
        rebuilt = await _float_index(conn, schema)
        assert rebuilt is not None
        assert rebuilt[0] != drifted[0]
        assert 'jsonb_typeof' in rebuilt[1]
        assert await _float_index(conn) is None
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_current_definitions_are_not_drifted(index_db_url: str) -> None:
    """Indexes built from the current definitions pass strict mode for every scalar type hint."""
    fields = {'s': 'string', 'i': 'integer', 'b': 'boolean', _FLOAT_FIELD: 'float', 'MixedCase': 'string'}
    conn = await asyncpg.connect(index_db_url)
    try:
        for field, type_hint in fields.items():
            await conn.execute(_generate_create_index_postgresql(field, type_hint))
        await _reconcile(index_db_url, 'strict', fields)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_strict_mode_refuses_a_drifted_index(index_db_url: str) -> None:
    """Strict mode reports a drifted definition as a mismatch and leaves the index alone."""
    conn = await asyncpg.connect(index_db_url)
    try:
        await conn.execute(_drifted_float_index())
        drifted = await _float_index(conn)
        with pytest.raises(ConfigurationError, match=f"Drifted: {{'{_FLOAT_FIELD}'}}"):
            await _reconcile(index_db_url, 'strict', {_FLOAT_FIELD: 'float'})
        assert await _float_index(conn) == drifted
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_warn_mode_reports_a_drifted_index(index_db_url: str, caplog: pytest.LogCaptureFixture) -> None:
    """Warn mode logs a drifted definition and leaves the index alone."""
    conn = await asyncpg.connect(index_db_url)
    try:
        await conn.execute(_drifted_float_index())
        drifted = await _float_index(conn)
        with caplog.at_level(logging.WARNING, logger='app.migrations.metadata'):
            await _reconcile(index_db_url, 'warn', {_FLOAT_FIELD: 'float'})
        assert await _float_index(conn) == drifted
        assert any(_FLOAT_FIELD in record.getMessage() and 'definition' in record.getMessage() for record in caplog.records)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_rebuild_that_cannot_be_built_keeps_the_existing_index(index_db_url: str) -> None:
    """A type-hint change whose new cast rejects stored rows refuses startup and keeps the old index.

    The field is indexed as text, then configured as an integer while a stored row holds
    a value the integer cast rejects; the new index is built before the old one is
    dropped, so the failed build leaves the text index in place.
    """
    conn = await asyncpg.connect(index_db_url)
    try:
        await conn.execute(_generate_create_index_postgresql(_FLOAT_FIELD, 'string'))
        await conn.execute('INSERT INTO context_entries (metadata) VALUES ($1::jsonb)', _document(_FLOAT_FIELD, '"high"'))
        existing = await _float_index(conn)
        with pytest.raises(ConfigurationError, match=f'Cannot index metadata field "{_FLOAT_FIELD}" as integer'):
            await _reconcile(index_db_url, 'additive', {_FLOAT_FIELD: 'integer'})
        assert await _float_index(conn) == existing
    finally:
        await conn.close()
