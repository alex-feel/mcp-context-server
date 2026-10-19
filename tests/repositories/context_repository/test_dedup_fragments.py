"""Tests for the SQL the deduplicating store, its read-only pre-check and the fresh insert share.

Both dedup methods select the dedup candidate and run the opposite-source turn check through
the same statement helpers, so a pre-check and the store it predicts read exactly the same
rows. The dedup UPDATE re-asserts in its WHERE clause that the caller owns the row; that
guard is not observable through behavior, because ownership cannot change between the
candidate read and the UPDATE of one transaction, so its text is asserted here. Every fresh
row, inserted by the deduplicating store or by ``insert_entry``, comes from one INSERT
builder that binds the owner from the caller's scope.
"""

import sqlite3
from collections.abc import Awaitable
from collections.abc import Callable
from typing import TYPE_CHECKING
from typing import cast

import pytest

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.backends import StorageBackend
from app.repositories.context_repository import ContextRepository
from app.repositories.context_repository import dedup

if TYPE_CHECKING:
    from app.backends.base import TransactionContext

SCOPE = AccessScope('alice', frozenset({'team-x'}))

_INSERT_COLUMNS = 'id, thread_id, source, content_type, text_content, metadata, summary, content_hash, owner_id, visibility'

# The id both insert paths draw, so their executed statements can be compared byte for byte.
_FIXED_ID = '0190abcdef1234567890abcdef123456'

# A relation a backend that reads through an access view names for ``context_entries``.
VIEW_RELATION = 'context_entries_readable AS context_entries'

type Statement = tuple[str, list[object]]


def _read_predicate(backend_type: str, start: int) -> str:
    """Return the READ predicate text for SCOPE on ``context_entries`` from ``start``."""
    return build_access_predicate(
        SCOPE, mode=AccessMode.READ, backend_type=backend_type, outer='context_entries', start=start,
    ).sql


def _words(sql: str) -> list[str]:
    """Split a statement into words, so layout whitespace does not matter."""
    return sql.split()


class TestCandidateSql:
    """The candidate is the latest row of the thread and source the scope may read."""

    @pytest.mark.parametrize(
        ('backend_type', 'thread_source', 'groups_param'),
        [
            ('sqlite', 'thread_id = ? AND source = ?', '["team-x"]'),
            ('postgresql', 'thread_id = $1 AND source = $2', ['team-x']),
        ],
    )
    def test_candidate_is_the_latest_readable_row(
        self, backend_type: str, thread_source: str, groups_param: object,
    ) -> None:
        """The READ predicate follows the thread and source terms, before ORDER BY and LIMIT."""
        sql, params = dedup._candidate_sql(
            backend_type, 'thread-1', 'agent', scope=SCOPE, relation='context_entries',
        )

        assert _words(sql) == _words(
            'SELECT id, content_hash, text_content, summary, owner_id FROM context_entries '
            f'WHERE {thread_source} AND {_read_predicate(backend_type, 3)} ORDER BY id DESC LIMIT 1',
        )
        assert params == ['thread-1', 'agent', 'alice', 'alice', groups_param]

    def test_candidate_reads_the_given_relation(self) -> None:
        """The statement selects from the relation the repository named for the backend."""
        sql, _ = dedup._candidate_sql(
            'postgresql', 'thread-1', 'agent', scope=SCOPE, relation=VIEW_RELATION,
        )

        assert f'FROM {VIEW_RELATION} WHERE' in sql


class TestInterleaveSql:
    """The turn check counts only opposite-source rows after the candidate that the scope may read."""

    @pytest.mark.parametrize(
        ('backend_type', 'leading_terms'),
        [
            ('sqlite', 'thread_id = ? AND source = ? AND id > ?'),
            ('postgresql', 'thread_id = $1 AND source = $2 AND id > $3'),
        ],
    )
    def test_turn_check_is_read_scoped(self, backend_type: str, leading_terms: str) -> None:
        """The READ predicate follows the candidate-id term."""
        sql, params = dedup._interleave_sql(
            backend_type, 'thread-1', 'agent', 'candidate-id', scope=SCOPE, relation='context_entries',
        )

        assert _words(sql) == _words(
            f'SELECT 1 FROM context_entries WHERE {leading_terms} AND {_read_predicate(backend_type, 4)} LIMIT 1',
        )
        assert params[:3] == ['thread-1', 'user', 'candidate-id']
        assert params[3:5] == ['alice', 'alice']

    @pytest.mark.parametrize(('source', 'opposite'), [('agent', 'user'), ('user', 'agent')])
    def test_turn_check_looks_for_the_opposite_source(self, source: str, opposite: str) -> None:
        """A user store looks for agent turns and an agent store for user turns."""
        _, params = dedup._interleave_sql(
            'sqlite', 'thread-1', source, 'candidate-id', scope=SCOPE, relation='context_entries',
        )

        assert params[1] == opposite

    def test_turn_check_reads_the_given_relation(self) -> None:
        """The statement selects from the relation the repository named for the backend."""
        sql, _ = dedup._interleave_sql(
            'postgresql', 'thread-1', 'agent', 'candidate-id', scope=SCOPE, relation=VIEW_RELATION,
        )

        assert f'FROM {VIEW_RELATION} WHERE' in sql


class TestDedupUpdateSql:
    """The dedup UPDATE writes only a row the scope owns whose hash the decision observed."""

    @pytest.mark.parametrize(
        ('backend_type', 'where'),
        [
            ('sqlite', 'id = ? AND content_hash IS ? AND context_entries.owner_id = ?'),
            ('postgresql', 'id = $5 AND content_hash IS NOT DISTINCT FROM $6 AND context_entries.owner_id = $7'),
        ],
    )
    def test_update_where_carries_the_owner_arm(self, backend_type: str, where: str) -> None:
        """The owner arm follows the id and hash terms and binds the scope's principal last."""
        sql, params = dedup._dedup_update_sql(
            backend_type, scope=SCOPE,
            metadata='{"k": 1}', content_type='text', summary=None,
            content_hash='new-hash', candidate_id='candidate-id', observed_hash='old-hash',
        )

        assert _words(sql.split('WHERE', 1)[1]) == _words(where)
        assert params == ['{"k": 1}', 'text', None, 'new-hash', 'candidate-id', 'old-hash', 'alice']


class TestInsertEntrySql:
    """Every fresh row comes from one INSERT whose owner is the scope's principal."""

    @pytest.mark.parametrize(
        ('backend_type', 'placeholders'),
        [
            ('sqlite', '?, ?, ?, ?, ?, ?, ?, ?, ?, ?'),
            ('postgresql', '$1, $2, $3, $4, $5, $6, $7, $8, $9, $10'),
        ],
    )
    def test_insert_is_one_line_with_the_owner_from_the_scope(self, backend_type: str, placeholders: str) -> None:
        """The statement names every stamped column and binds owner_id from the scope."""
        sql, params = dedup._insert_entry_sql(
            backend_type, scope=SCOPE,
            context_id='new-id', thread_id='thread-1', source='agent', content_type='text',
            text_content='body', metadata='{"k": 1}', summary='a summary', content_hash='hash',
            visibility='private',
        )

        assert sql == f'INSERT INTO context_entries ({_INSERT_COLUMNS}) VALUES ({placeholders})'
        assert params == ['new-id', 'thread-1', 'agent', 'text', 'body', '{"k": 1}', 'a summary', 'hash', 'alice', 'private']

    def test_owner_follows_the_scope(self) -> None:
        """Another scope stamps another owner; no argument names the owner directly."""
        _, params = dedup._insert_entry_sql(
            'sqlite', scope=AccessScope('bob', frozenset()),
            context_id='new-id', thread_id='thread-1', source='agent', content_type='text',
            text_content='body', metadata=None, summary=None, content_hash='hash', visibility='public',
        )

        assert params[8:] == ['bob', 'public']


def _record(calls: list[Statement], helper: Callable[..., Statement]) -> Callable[..., Statement]:
    """Wrap a statement helper so every statement it builds is appended to ``calls``."""

    def _recording(*args: object, **kwargs: object) -> Statement:
        statement = helper(*args, **kwargs)
        calls.append(statement)
        return statement

    return _recording


class TestSharedStatements:
    """The pre-check and the store run the very same candidate and turn statements."""

    @pytest.mark.asyncio
    async def test_precheck_and_store_run_identical_statements(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """For one re-sent text both methods build identical statements with identical parameters."""
        repo = ContextRepository(async_db_initialized)
        stored_id, _ = await repo.store_with_deduplication(
            thread_id='shared-thread', source='agent', content_type='text', text_content='re-sent text',
            scope=SCOPE, visibility='private',
        )
        calls: list[Statement] = []
        monkeypatch.setattr(dedup, '_candidate_sql', _record(calls, dedup._candidate_sql))
        monkeypatch.setattr(dedup, '_interleave_sql', _record(calls, dedup._interleave_sql))

        candidate = await repo.check_latest_is_duplicate(
            thread_id='shared-thread', source='agent', text_content='re-sent text', scope=SCOPE,
        )
        precheck_calls = list(calls)
        calls.clear()
        merged_id, merged = await repo.store_with_deduplication(
            thread_id='shared-thread', source='agent', content_type='text', text_content='re-sent text',
            scope=SCOPE, visibility='private',
        )

        assert candidate is not None
        assert candidate.context_id == stored_id
        assert (merged_id, merged) == (stored_id, True)
        assert len(precheck_calls) == 2
        assert calls == precheck_calls


class _RollBackError(Exception):
    """Raised to roll a traced transaction back, so the next path can insert the same id."""


async def _traced_inserts(
    backend: StorageBackend, insert: Callable[['TransactionContext'], Awaitable[object]],
) -> list[str]:
    """Run ``insert`` in a transaction that is rolled back and return the INSERT statements it executed."""
    statements: list[str] = []

    async def _run() -> None:
        async with backend.begin_transaction(access_scope=SCOPE) as txn:
            connection = cast(sqlite3.Connection, txn.connection)
            connection.set_trace_callback(statements.append)
            try:
                await insert(txn)
            finally:
                connection.set_trace_callback(None)
            raise _RollBackError

    with pytest.raises(_RollBackError):
        await _run()
    return [statement for statement in statements if statement.startswith('INSERT')]


class TestSharedInsertStatement:
    """The deduplicating store's fresh-insert branch and ``insert_entry`` execute the same INSERT."""

    @pytest.mark.asyncio
    async def test_both_insert_paths_execute_identical_text(
        self, async_db_initialized: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the same id and values, both paths execute byte-identical INSERT statements."""
        repo = ContextRepository(async_db_initialized)
        monkeypatch.setattr(dedup, 'generate_id', lambda: _FIXED_ID)

        dedup_inserts = await _traced_inserts(async_db_initialized, lambda txn: repo.store_with_deduplication(
            thread_id='insert-thread', source='agent', content_type='text', text_content='fresh body',
            scope=SCOPE, visibility='private', metadata='{"k": 1}', summary='a summary', txn=txn,
        ))
        fresh_inserts = await _traced_inserts(async_db_initialized, lambda txn: repo.insert_entry(
            'insert-thread', 'agent', 'text', 'fresh body',
            scope=SCOPE, visibility='private', metadata='{"k": 1}', summary='a summary', txn=txn,
        ))

        assert len(dedup_inserts) == 1
        assert dedup_inserts[0].startswith(f'INSERT INTO context_entries ({_INSERT_COLUMNS}) VALUES (')
        assert fresh_inserts == dedup_inserts
