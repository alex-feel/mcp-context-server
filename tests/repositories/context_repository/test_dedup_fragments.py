"""Tests for the SQL the deduplicating store and its read-only pre-check share.

Both methods select the dedup candidate and run the opposite-source turn check through the
same statement helpers, so a pre-check and the store it predicts read exactly the same
rows. The dedup UPDATE re-asserts in its WHERE clause that the caller owns the row; that
guard is not observable through behavior, because ownership cannot change between the
candidate read and the UPDATE of one transaction, so its text is asserted here.
"""

from collections.abc import Callable

import pytest

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.access_scope import build_access_predicate
from app.backends import StorageBackend
from app.repositories.context_repository import ContextRepository
from app.repositories.context_repository import dedup

SCOPE = AccessScope('alice', frozenset({'team-x'}))

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
        sql, params = dedup._candidate_sql(backend_type, SCOPE, 'thread-1', 'agent')

        assert _words(sql) == _words(
            'SELECT id, content_hash, text_content, summary, owner_id FROM context_entries '
            f'WHERE {thread_source} AND {_read_predicate(backend_type, 3)} ORDER BY id DESC LIMIT 1',
        )
        assert params == ['thread-1', 'agent', 'alice', 'alice', groups_param]


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
        sql, params = dedup._interleave_sql(backend_type, SCOPE, 'thread-1', 'agent', 'candidate-id')

        assert _words(sql) == _words(
            f'SELECT 1 FROM context_entries WHERE {leading_terms} AND {_read_predicate(backend_type, 4)} LIMIT 1',
        )
        assert params[:3] == ['thread-1', 'user', 'candidate-id']
        assert params[3:5] == ['alice', 'alice']

    @pytest.mark.parametrize(('source', 'opposite'), [('agent', 'user'), ('user', 'agent')])
    def test_turn_check_looks_for_the_opposite_source(self, source: str, opposite: str) -> None:
        """A user store looks for agent turns and an agent store for user turns."""
        _, params = dedup._interleave_sql('sqlite', SCOPE, 'thread-1', source, 'candidate-id')

        assert params[1] == opposite


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
            backend_type, SCOPE,
            metadata='{"k": 1}', content_type='text', summary=None,
            content_hash='new-hash', candidate_id='candidate-id', observed_hash='old-hash',
        )

        assert _words(sql.split('WHERE', 1)[1]) == _words(where)
        assert params == ['{"k": 1}', 'text', None, 'new-hash', 'candidate-id', 'old-hash', 'alice']


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
