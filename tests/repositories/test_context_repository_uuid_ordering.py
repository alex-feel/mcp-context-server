"""Tests that the dedup interleaving check honors UUIDv7 lex-string ordering.

UUIDv7 lex-string ordering matches creation-time at millisecond granularity
per RFC 9562 section 5.7. The ``id > ?`` interleaving check used in
``store_with_deduplication`` must therefore correctly detect that a later
opposite-source entry exists when comparing string IDs under SQLite's
TEXT BINARY collation.
"""

import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.ids import generate_id_with_timestamp
from app.repositories import RepositoryContainer


@pytest_asyncio.fixture
async def repos(tmp_path: Path) -> AsyncGenerator[RepositoryContainer, None]:
    """Build a SQLite backend with the full schema and a RepositoryContainer."""
    db_path = tmp_path / 'test.db'
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    yield RepositoryContainer(backend)
    await backend.shutdown()


class TestUuidIdOrdering:
    """UUIDv7 lex-string ordering preserves chronological ordering across ms boundaries."""

    def test_generate_id_with_timestamp_orders_by_input_datetime(self) -> None:
        """Two UUIDs from 2-ms apart timestamps sort by their input order under str.compare."""
        t0 = datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
        t1 = t0 + timedelta(milliseconds=2)
        id0 = generate_id_with_timestamp(t0)
        id1 = generate_id_with_timestamp(t1)
        assert id0 < id1, f'Expected {id0} < {id1} for timestamps {t0} < {t1}'


class TestDedupInterleavingCheck:
    """`store_with_deduplication` correctly detects interleaved opposite-source entries."""

    @pytest.mark.asyncio
    async def test_interleaving_check_with_uuid_id_ordering(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """When an opposite-source entry is created after a candidate duplicate,
        deduplication is suppressed and a new row is inserted.

        Sequence:
          1. User stores 'message X' -> entry A (user/X).
          2. Agent stores 'reply Y'  -> entry B (agent/Y), id > A's id (lex order).
          3. User stores 'message X' again -> MUST insert entry C (user/X),
             NOT deduplicate against A, because B exists with id > A's id.
        """
        id_a_before, was_updated_a = await repos.context.store_with_deduplication(
            thread_id='t', source='user', content_type='text',
            text_content='message X', metadata=None,
        )
        assert was_updated_a is False
        id_b, was_updated_b = await repos.context.store_with_deduplication(
            thread_id='t', source='agent', content_type='text',
            text_content='reply Y', metadata=None,
        )
        assert was_updated_b is False
        assert id_b > id_a_before

        id_c, was_updated_c = await repos.context.store_with_deduplication(
            thread_id='t', source='user', content_type='text',
            text_content='message X', metadata=None,
        )
        assert was_updated_c is False, (
            'Expected new insertion (not dedup) because an opposite-source '
            'entry B was created after the original user entry A.'
        )
        assert id_c != id_a_before
