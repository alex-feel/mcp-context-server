"""Tests for the unique-tags statement of the database statistics on both backends.

PostgreSQL counts the distinct readable tags through a DISTINCT subquery over exactly the
rows the top-tags aggregate groups, so the planner can read the tags in tag order from
their index instead of sorting every readable tag row; SQLite keeps ``COUNT(DISTINCT)``.
Both statements count the distinct tags of the entries the caller may read.
"""

import sqlite3
from typing import cast

import pytest

from app.access_scope import AccessMode
from app.access_scope import build_access_predicate
from app.access_scope import build_readable_parent_predicate
from app.backends.base import MAINTENANCE
from app.backends.base import StorageBackend
from app.backends.base import base_relation
from app.repositories.statistics_repository import StatisticsRepository
from tests.helpers import LOCAL_SCOPE
from tests.repositories._statistics_recording import RecordingPgBackend
from tests.repositories._statistics_recording import RecordingSqliteBackend

TOP_TAGS_ORDER = 'GROUP BY t.tag ORDER BY count DESC'


def _postgresql_readable_tags() -> str:
    """Return the FROM, join and filter the PostgreSQL tag aggregates read the caller's tags through."""
    read = build_access_predicate(LOCAL_SCOPE, mode=AccessMode.READ, backend_type='postgresql', outer='ce')
    tags = base_relation('tags', alias='t')
    entries = base_relation('context_entries', alias='ce')
    return f'{tags} JOIN {entries} ON ce.id = t.context_entry_id{read.where_clause()}'


@pytest.mark.asyncio
async def test_postgresql_counts_distinct_tags_through_a_distinct_subquery() -> None:
    """The unique-tags count wraps a DISTINCT of the readable tags and no longer aggregates COUNT(DISTINCT)."""
    backend = RecordingPgBackend()

    await StatisticsRepository(cast(StorageBackend, backend)).get_database_statistics(scope=LOCAL_SCOPE)

    queries = backend.connection.queries
    readable_tags = _postgresql_readable_tags()
    assert f'SELECT COUNT(*) AS count FROM (SELECT DISTINCT t.tag FROM {readable_tags}) AS d' in queries, queries
    assert not any('COUNT(DISTINCT t.tag)' in query for query in queries), queries


@pytest.mark.asyncio
async def test_postgresql_unique_tags_read_the_rows_the_top_tags_group() -> None:
    """The unique-tags subquery and the top-tags aggregate share their FROM, join and filter."""
    backend = RecordingPgBackend()

    await StatisticsRepository(cast(StorageBackend, backend)).get_database_statistics(scope=LOCAL_SCOPE)

    readable_tags = _postgresql_readable_tags()
    [top_tags] = [query for query in backend.connection.queries if TOP_TAGS_ORDER in query]
    assert top_tags.startswith(f'SELECT t.tag AS tag, COUNT(*) AS count FROM {readable_tags} {TOP_TAGS_ORDER}'), top_tags


def _insert_tagged_entries(conn: sqlite3.Connection) -> None:
    """Seed two readable entries sharing a tag and a hidden entry whose tag is its own."""
    cursor = conn.cursor()
    rows = (
        ('0190abcdef1234567890abcf00000001', 'local', ('shared', 'first')),
        ('0190abcdef1234567890abcf00000002', 'local', ('shared', 'second')),
        ('0190abcdef1234567890abcf00000003', 'someone-else', ('hidden',)),
    )
    for context_id, owner, tags in rows:
        cursor.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id) '
            "VALUES (?, 'tag-count', 'user', 'text', 'tag count entry', ?)",
            (context_id, owner),
        )
        for tag in tags:
            cursor.execute('INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)', (context_id, tag))


@pytest.mark.asyncio
async def test_sqlite_counts_distinct_readable_tags_with_count_distinct(stats_test_db: StorageBackend) -> None:
    """SQLite keeps its COUNT(DISTINCT) statement and counts each readable tag once."""
    await stats_test_db.execute_write(_insert_tagged_entries, access_scope=MAINTENANCE)
    backend = RecordingSqliteBackend(stats_test_db)

    stats = await StatisticsRepository(cast(StorageBackend, backend)).get_database_statistics(scope=LOCAL_SCOPE)

    parents = build_readable_parent_predicate(
        LOCAL_SCOPE, child_key='t.context_entry_id', parent_key='id', backend_type='sqlite',
    )
    readable_tags = f'{base_relation("tags", alias="t")}{parents.where_clause()}'
    assert f'SELECT COUNT(DISTINCT t.tag) AS count FROM {readable_tags}' in backend.queries, backend.queries
    assert stats['unique_tags'] == 3
