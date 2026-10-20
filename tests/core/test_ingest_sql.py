"""Tests for the SQL fragments that read the reserved ``ingest`` metadata block.

Covers app.ingest_sql: every fragment per backend with and without a table
alias, the ingest index DDL each backend renders from the unaliased fragments,
and the validity of the SQLite DDL against the base schema.
"""

import ast
import sqlite3
import sys
from pathlib import Path

import pytest

import app.ingest_sql
from app.ingest_sql import INGEST_INDEX_NAME
from app.ingest_sql import block_sql
from app.ingest_sql import ingest_index_ddl
from app.ingest_sql import scope_sql
from app.ingest_sql import source_id_key_sql
from app.ingest_sql import source_id_sql
from app.schemas import load_schema

_SQLITE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_context_ingest ON context_entries(thread_id, json_extract(metadata, "
    "'$.ingest.source_id'), id, json_extract(metadata, '$.ingest.scope'), json_extract(metadata, '$.ingest'), "
    "content_hash, owner_id, visibility) WHERE json_extract(metadata, '$.ingest.source_id') IS NOT NULL"
)
_POSTGRESQL_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_context_ingest ON context_entries (thread_id, ((metadata #>> "
    "'{ingest,source_id}') COLLATE \"C\"), id) WHERE (metadata #>> '{ingest,source_id}') IS NOT NULL"
)


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        pytest.param(None, "json_extract(metadata, '$.ingest.source_id')", id='no-alias'),
        pytest.param('ce', "json_extract(ce.metadata, '$.ingest.source_id')", id='alias'),
    ],
)
def test_sqlite_source_id(alias: str | None, expected: str) -> None:
    """SQLite reads the source id with json_extract, qualified by the alias when one is given."""
    assert source_id_sql('sqlite', alias) == expected


@pytest.mark.parametrize('alias', [None, 'ce'])
def test_sqlite_source_id_key_is_the_source_id(alias: str | None) -> None:
    """SQLite orders and bounds the source id under its default BINARY collation, so the key is the value."""
    assert source_id_key_sql('sqlite', alias) == source_id_sql('sqlite', alias)


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        pytest.param(None, "json_extract(metadata, '$.ingest.scope')", id='no-alias'),
        pytest.param('ce', "json_extract(ce.metadata, '$.ingest.scope')", id='alias'),
    ],
)
def test_sqlite_scope(alias: str | None, expected: str) -> None:
    """SQLite reads the block's scope with json_extract."""
    assert scope_sql('sqlite', alias) == expected


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        pytest.param(None, "json_extract(metadata, '$.ingest')", id='no-alias'),
        pytest.param('ce', "json_extract(ce.metadata, '$.ingest')", id='alias'),
    ],
)
def test_sqlite_block(alias: str | None, expected: str) -> None:
    """SQLite reads the whole block as JSON text."""
    assert block_sql('sqlite', alias) == expected


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        pytest.param(None, "(metadata #>> '{ingest,source_id}')", id='no-alias'),
        pytest.param('ce', "(ce.metadata #>> '{ingest,source_id}')", id='alias'),
    ],
)
def test_postgresql_source_id(alias: str | None, expected: str) -> None:
    """PostgreSQL reads the source id as text through the path accessor."""
    assert source_id_sql('postgresql', alias) == expected


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        pytest.param(None, "((metadata #>> '{ingest,source_id}') COLLATE \"C\")", id='no-alias'),
        pytest.param('ce', "((ce.metadata #>> '{ingest,source_id}') COLLATE \"C\")", id='alias'),
    ],
)
def test_postgresql_source_id_key(alias: str | None, expected: str) -> None:
    """PostgreSQL orders and bounds the source id in code-point order under the C collation."""
    assert source_id_key_sql('postgresql', alias) == expected


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        pytest.param(None, "(metadata #>> '{ingest,scope}')", id='no-alias'),
        pytest.param('ce', "(ce.metadata #>> '{ingest,scope}')", id='alias'),
    ],
)
def test_postgresql_scope(alias: str | None, expected: str) -> None:
    """PostgreSQL reads the block's scope as text."""
    assert scope_sql('postgresql', alias) == expected


@pytest.mark.parametrize(
    ('alias', 'expected'),
    [
        pytest.param(None, "metadata -> 'ingest'", id='no-alias'),
        pytest.param('ce', "ce.metadata -> 'ingest'", id='alias'),
    ],
)
def test_postgresql_block(alias: str | None, expected: str) -> None:
    """PostgreSQL reads the whole block as jsonb."""
    assert block_sql('postgresql', alias) == expected


def test_index_name() -> None:
    """The index name stays clear of the prefixes the metadata index reconciler parses as fields."""
    assert INGEST_INDEX_NAME == 'idx_context_ingest'
    assert not INGEST_INDEX_NAME.startswith(('idx_metadata_', 'idx_thread_metadata_'))


@pytest.mark.parametrize(
    ('backend_type', 'expected'),
    [
        pytest.param('sqlite', _SQLITE_DDL, id='sqlite'),
        pytest.param('postgresql', _POSTGRESQL_DDL, id='postgresql'),
    ],
)
def test_index_ddl(backend_type: str, expected: str) -> None:
    """Each backend renders its partial ingest index from the unaliased fragments."""
    assert ingest_index_ddl(backend_type) == expected


@pytest.mark.parametrize('backend_type', ['sqlite', 'postgresql'])
def test_index_ddl_is_built_from_the_query_fragments(backend_type: str) -> None:
    """The index key and predicate are the exact expressions the repository reads compare and order by."""
    ddl = ingest_index_ddl(backend_type)

    assert f'{source_id_key_sql(backend_type)}, id' in ddl
    assert ddl.endswith(f'WHERE {source_id_sql(backend_type)} IS NOT NULL')


def test_sqlite_index_ddl_runs_on_the_base_schema() -> None:
    """The SQLite statement creates the index on the base schema, and a second run is a no-op."""
    conn = sqlite3.connect(':memory:')
    try:
        conn.executescript(load_schema('sqlite'))
        conn.execute(ingest_index_ddl('sqlite'))
        conn.execute(ingest_index_ddl('sqlite'))
        names = [
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?", (INGEST_INDEX_NAME,))
        ]
    finally:
        conn.close()

    assert names == [INGEST_INDEX_NAME]


def test_module_imports_only_the_standard_library() -> None:
    """app.ingest_sql is a leaf the migrations and the repositories share."""
    module_file = app.ingest_sql.__file__
    assert module_file is not None
    tree = ast.parse(Path(module_file).read_text(encoding='utf-8'))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            imported.add(node.module.partition('.')[0])

    assert imported <= sys.stdlib_module_names
