"""Tests for ContextDedupMixin.insert_entry, the fresh owner-stamped insert of the store path.

The insert never looks for a dedup candidate: it writes exactly one new row owned by the
caller, whatever the thread already holds, and it runs either on its own write or inside
the caller's transaction.
"""

import re
import sqlite3
from collections.abc import Callable
from typing import Any

import pytest

from app import digests
from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.repositories.context_repository import ContextRepository

ALICE = AccessScope('alice', frozenset({'team-x'}))

# Transaction control the sqlite3 module issues around a write on its own.
_TRANSACTION_CONTROL = ('BEGIN', 'COMMIT', 'ROLLBACK')

_STORED_COLUMNS = (
    'thread_id, source, content_type, text_content, metadata, summary, content_hash, owner_id, visibility, version'
)


async def _stored_row(backend: StorageBackend, context_id: str) -> dict[str, object] | None:
    """Return the stored columns of one entry, None when it does not exist."""

    def _read(conn: sqlite3.Connection) -> dict[str, object] | None:
        row = conn.execute(f'SELECT {_STORED_COLUMNS} FROM context_entries WHERE id = ?', (context_id,)).fetchone()
        return dict(row) if row is not None else None

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


async def _thread_ids(backend: StorageBackend, thread_id: str) -> list[str]:
    """Return the ids of a thread's entries in id order."""

    def _read(conn: sqlite3.Connection) -> list[str]:
        rows = conn.execute('SELECT id FROM context_entries WHERE thread_id = ? ORDER BY id', (thread_id,)).fetchall()
        return [str(row[0]) for row in rows]

    return await backend.execute_read(_read, access_scope=MAINTENANCE)


class TestInsertEntry:
    """insert_entry writes one new row owned by the caller."""

    @pytest.mark.asyncio
    async def test_insert_stamps_owner_visibility_and_content_hash(self, async_db_initialized: StorageBackend) -> None:
        """The row carries the scope's principal, the given visibility and the hash of its text."""
        repo = ContextRepository(async_db_initialized)

        context_id = await repo.insert_entry(
            'insert-thread', 'agent', 'text', 'inserted body',
            scope=ALICE, visibility='public', metadata='{"k": 1}', summary='A summary',
        )

        assert re.fullmatch(r'[0-9a-f]{32}', context_id)
        assert await _stored_row(async_db_initialized, context_id) == {
            'thread_id': 'insert-thread',
            'source': 'agent',
            'content_type': 'text',
            'text_content': 'inserted body',
            'metadata': '{"k": 1}',
            'summary': 'A summary',
            'content_hash': digests.content_hash('inserted body'),
            'owner_id': 'alice',
            'visibility': 'public',
            'version': 0,
        }

    @pytest.mark.parametrize('summary', ['', '   \n'])
    @pytest.mark.asyncio
    async def test_blank_summary_is_stored_as_null(self, async_db_initialized: StorageBackend, summary: str) -> None:
        """A blank summary is stored as NULL, as the deduplicating store stores it."""
        repo = ContextRepository(async_db_initialized)

        context_id = await repo.insert_entry(
            'insert-thread', 'agent', 'text', 'body without summary', scope=ALICE, visibility='private', summary=summary,
        )

        row = await _stored_row(async_db_initialized, context_id)
        assert row is not None
        assert row['summary'] is None

    @pytest.mark.asyncio
    async def test_never_merges_into_an_identical_latest_row(self, async_db_initialized: StorageBackend) -> None:
        """An identical latest row of the caller's own stays untouched beside the new row."""
        repo = ContextRepository(async_db_initialized)
        existing_id, _ = await repo.store_with_deduplication(
            thread_id='insert-thread', source='agent', content_type='text', text_content='same body',
            scope=ALICE, visibility='private', metadata='{"first": true}',
        )

        new_id = await repo.insert_entry(
            'insert-thread', 'agent', 'text', 'same body', scope=ALICE, visibility='private', metadata='{"second": true}',
        )

        assert new_id != existing_id
        assert await _thread_ids(async_db_initialized, 'insert-thread') == [existing_id, new_id]
        existing = await _stored_row(async_db_initialized, existing_id)
        assert existing is not None
        assert (existing['metadata'], existing['version']) == ('{"first": true}', 0)

    @pytest.mark.asyncio
    async def test_runs_inside_the_given_transaction(self, async_db_initialized: StorageBackend) -> None:
        """Given a transaction, the insert commits and rolls back with it."""
        repo = ContextRepository(async_db_initialized)
        inserted: list[str] = []

        async def _insert_then_abort() -> None:
            async with async_db_initialized.begin_transaction(access_scope=ALICE) as txn:
                inserted.append(await repo.insert_entry(
                    'txn-thread', 'agent', 'text', 'rolled back body', scope=ALICE, visibility='private', txn=txn,
                ))
                raise RuntimeError('abort the transaction')

        with pytest.raises(RuntimeError, match='abort the transaction'):
            await _insert_then_abort()
        async with async_db_initialized.begin_transaction(access_scope=ALICE) as txn:
            inserted.append(await repo.insert_entry(
                'txn-thread', 'agent', 'text', 'committed body', scope=ALICE, visibility='private', txn=txn,
            ))

        assert await _thread_ids(async_db_initialized, 'txn-thread') == [inserted[1]]

    @pytest.mark.asyncio
    async def test_runs_only_the_insert(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The executed statements are the one INSERT: no candidate read, no turn check."""
        repo = ContextRepository(async_db_initialized)
        statements: list[str] = []
        execute_write = async_db_initialized.execute_write

        async def _tracing_execute_write(operation: Callable[..., object], *args: Any, **kwargs: Any) -> object:
            def _traced(conn: sqlite3.Connection, *inner_args: Any, **inner_kwargs: Any) -> object:
                conn.set_trace_callback(statements.append)
                try:
                    return operation(conn, *inner_args, **inner_kwargs)
                finally:
                    conn.set_trace_callback(None)

            return await execute_write(_traced, *args, **kwargs)

        monkeypatch.setattr(async_db_initialized, 'execute_write', _tracing_execute_write)

        await repo.insert_entry('trace-thread', 'agent', 'text', 'traced body', scope=ALICE, visibility='private')

        executed = [statement for statement in statements if not statement.startswith(_TRANSACTION_CONTROL)]
        assert len(executed) == 1
        assert executed[0].startswith('INSERT INTO context_entries (id, thread_id, source,')
