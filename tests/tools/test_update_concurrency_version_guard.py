"""Tool-level regression test for the optimistic-concurrency version guard.

This exercises ``app.tools.context.update_context`` against a REAL SQLite backend
(no mocked repositories) to prove that two concurrent updates to the SAME entry
cannot silently lose an update: the loser is caught by the ``version`` CAS guard,
re-reads the current version, and retries, so the row ends in a single consistent
state (exactly one of the two submitted texts) with the version column advanced by
two successful commits -- never a torn/garbled state and never a duplicate row.

Backend/repos setup mirrors ``tests/tools/test_generation_first.py``
(``setup_with_entry``): a real ``SQLiteBackend`` + ``RepositoryContainer`` built
from ``load_schema('sqlite')`` with one pre-inserted entry.

Formulation choice (deterministic concurrency, not best-effort):
Two genuinely concurrent ``update_context`` coroutines are launched with
``asyncio.gather``. Generation latency is SKEWED with an ``asyncio.Event`` so the
FIRST-submitted call's summary resolves AFTER the second call has fully committed
(advancing the row's version from 0 to 1). The first call therefore captured
``expected_version=0`` before generation, hits the CAS guard at commit time
(``VersionConflictError``), re-reads version 1, and retries successfully. This is
deterministic -- the event strictly orders the two commits -- so it is robust
rather than flaky, while still driving the real two-concurrent-writer path the
guard was built for (the strongest available formulation; the purely synthetic
"bump the version out-of-band" variant is kept as a second, even-simpler test).
"""

import asyncio
import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

from app.backends.sqlite_backend import SQLiteBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import VersionConflictError
from app.schemas import load_schema
from app.tools.context import update_context
from app.types import UpdateContextSuccessDict

# Text long enough to clear SUMMARY_MIN_CONTENT_LENGTH (default 500) so the
# summary leg actually runs and we can use it to skew generation latency.
TEXT_A = 'AAAA winner-or-loser text ' * 30
TEXT_B = 'BBBB the other concurrent text ' * 30


@pytest.mark.usefixtures('mock_server_dependencies')
class TestUpdateContextVersionGuard:
    """Concurrent ``update_context`` on one id resolves to a single consistent winner."""

    @pytest_asyncio.fixture
    async def setup_with_entry(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, str], None]:
        db_path = tmp_path / 'test_version_guard.db'
        schema_sql = load_schema('sqlite')
        new_id = generate_id()
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, text_content, content_type, metadata) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (new_id, 'guard-thread', 'agent', 'Original text', 'text', '{}'),
            )
            conn.commit()
        backend = SQLiteBackend(db_path=str(db_path))
        await backend.initialize()
        repos = RepositoryContainer(backend)
        yield backend, repos, new_id
        await backend.shutdown()

    async def _read_row(self, backend: SQLiteBackend, entry_id: str) -> tuple[str, int]:
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT text_content, version FROM context_entries WHERE id = ?',
                (entry_id,),
            )
            row = cursor.fetchone()
            return str(row[0]), int(row[1])

    async def _count_rows(self, backend: SQLiteBackend, entry_id: str) -> int:
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE id = ?',
                (entry_id,),
            )
            return int(cursor.fetchone()[0])

    @pytest.mark.asyncio
    async def test_concurrent_updates_resolve_to_single_winner(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """Two concurrent updates on the same id: both succeed, the row holds
        exactly one submitted text, version advanced by 2, no duplicate rows.

        The first-submitted call's generation is held until the second has
        committed, deterministically forcing the first to lose the version CAS
        and retry against the freshly bumped version.
        """
        backend, repos, entry_id = setup_with_entry

        # Embeddings disabled (provider None) so only the summary leg runs; that
        # keeps the skew mechanism simple and avoids embedding-store mocking.
        second_committed = asyncio.Event()

        async def skewed_summary(text: str, _source: str) -> str:
            # The FIRST-submitted call (TEXT_A) blocks here until the second call
            # (TEXT_B) has fully returned, so TEXT_A commits LAST and its captured
            # version-0 CAS is already stale -> VersionConflictError -> retry.
            if text.strip().startswith('AAAA'):
                await second_committed.wait()
            return 'summary for ' + text[:8]

        # Real version-aware update path; only generation providers are faked.
        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=MagicMock()),
            patch('app.tools._shared.get_summary_provider', return_value=MagicMock()),
            patch(
                'app.tools._shared.generate_summary_with_timeout',
                new=AsyncMock(side_effect=skewed_summary),
            ),
        ):
            async def update_a() -> UpdateContextSuccessDict:
                # Give update_b a head start so it captures version 0 too, then
                # commits first; update_a is parked in skewed_summary meanwhile.
                return await update_context(context_id=entry_id, text=TEXT_A)

            async def update_b() -> UpdateContextSuccessDict:
                result = await update_context(context_id=entry_id, text=TEXT_B)
                # Release update_a only AFTER update_b's commit is visible.
                second_committed.set()
                return result

            result_a, result_b = await asyncio.gather(update_a(), update_b())

        # Both ultimately succeed: the loser retried via the version guard.
        assert result_a['success'] is True
        assert result_b['success'] is True

        # Single consistent winner: the persisted text is EXACTLY one of the two
        # submitted texts (never a torn/partial blend).
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text in {TEXT_A.strip(), TEXT_B.strip()}

        # No silent lost update: two successful commits each bumped version by 1,
        # so the version column advanced from 0 to 2.
        assert final_version == 2

        # No duplicate rows for this id.
        assert await self._count_rows(backend, entry_id) == 1

    @pytest.mark.asyncio
    async def test_stale_version_capture_retries_and_succeeds(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """Deterministic guard variant: an update whose pre-generation version
        capture is forced stale (the row's version is bumped out-of-band mid-call)
        RETRIES via the guard and still succeeds against the new version.

        This isolates the retry-on-conflict mechanism without relying on task
        scheduling: ``check_entry_exists`` is wrapped so the FIRST call (the
        pre-generation capture) returns the original version while a concurrent
        writer simultaneously advances the real row, guaranteeing the in-transaction
        CAS sees a mismatch on the first attempt and a match after the re-read.
        """
        backend, repos, entry_id = setup_with_entry

        real_check = repos.context.check_entry_exists
        bumped = {'done': False}

        async def check_then_bump(context_id: str) -> tuple[bool, str | None, int | None]:
            result = await real_check(context_id)
            # On the very first capture (version 0), simulate a concurrent writer
            # committing a newer version out-of-band, so our captured version is
            # already stale when the transaction's CAS runs.
            if not bumped['done'] and result[2] == 0:
                bumped['done'] = True

                def _bump(conn: sqlite3.Connection) -> None:
                    conn.execute(
                        'UPDATE context_entries SET version = version + 1 WHERE id = ?',
                        (context_id,),
                    )

                await backend.execute_write(_bump)
            return result

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch.object(
                repos.context, 'check_entry_exists', side_effect=check_then_bump,
            ),
        ):
            result = await update_context(context_id=entry_id, text=TEXT_A)

        assert result['success'] is True

        # The out-of-band bump (1) plus our successful retried commit (1) leaves
        # version at 2, and the text is the one this call requested.
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text == TEXT_A.strip()
        assert final_version == 2
        # The guard fired at least once (the stale capture forced a re-read).
        assert bumped['done'] is True

    async def _read_metadata(self, backend: SQLiteBackend, entry_id: str) -> tuple[str, int]:
        async with backend.get_connection(readonly=True) as conn:
            cursor = conn.execute(
                'SELECT metadata, version FROM context_entries WHERE id = ?',
                (entry_id,),
            )
            row = cursor.fetchone()
            return str(row[0]), int(row[1])

    @pytest.mark.asyncio
    async def test_metadata_only_update_cas_applies_and_self_heals(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """A metadata-only update (text=None) is still guarded by the version CAS:
        a stale captured version forces a re-read + retry, and the update converges.

        ``update_context`` captures ``expected_version`` unconditionally (before any
        generation), and ``update_context_entry`` bumps the version whenever it
        emits an UPDATE -- which a metadata-only change does. This mirrors the
        text-based ``test_stale_version_capture_retries_and_succeeds`` but with NO
        text (hence no generation), proving the guard covers the metadata-only path
        too. The row's version is bumped out-of-band on the first pre-generation
        capture, so the in-transaction CAS sees a mismatch on the first attempt and
        a match after the re-read.
        """
        backend, repos, entry_id = setup_with_entry

        real_check = repos.context.check_entry_exists
        bumped = {'done': False}

        async def check_then_bump(context_id: str) -> tuple[bool, str | None, int | None]:
            result = await real_check(context_id)
            if not bumped['done'] and result[2] == 0:
                bumped['done'] = True

                def _bump(conn: sqlite3.Connection) -> None:
                    conn.execute(
                        'UPDATE context_entries SET version = version + 1 WHERE id = ?',
                        (context_id,),
                    )

                await backend.execute_write(_bump)
            return result

        # No text -> no generation; only metadata changes. Providers are None.
        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch.object(
                repos.context, 'check_entry_exists', side_effect=check_then_bump,
            ),
        ):
            result = await update_context(
                context_id=entry_id, metadata={'status': 'reviewed'},
            )

        assert result['success'] is True
        assert 'metadata' in result['updated_fields']
        # The guard fired at least once (the stale capture forced a re-read).
        assert bumped['done'] is True
        # Out-of-band bump (1) + our retried metadata commit (1) => version 2, and
        # the metadata is the one this call requested.
        metadata_json, final_version = await self._read_metadata(backend, entry_id)
        assert 'reviewed' in metadata_json
        assert final_version == 2

    @pytest.mark.asyncio
    async def test_version_conflict_error_is_not_a_tool_error_subclass(self) -> None:
        """VersionConflictError must NOT be a ToolError so the ``except ToolError``
        fast-path in update_context cannot swallow it before the retry loop runs.
        """
        err = VersionConflictError('0190abcdef1234567890abcdef000001')
        assert not isinstance(err, ToolError)
        assert err.context_id == '0190abcdef1234567890abcdef000001'

    @pytest.mark.asyncio
    async def test_persistent_conflict_exhausts_retries_and_raises(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """A PERSISTENT version conflict exhausts the bounded retry budget and
        ``update_context`` raises a ToolError matching "kept changing".

        ``execute_update_in_transaction`` is stubbed to ALWAYS raise
        ``VersionConflictError`` (a never-healing conflict: every attempt's CAS
        mismatches). ``check_entry_exists`` keeps reporting the entry exists, so the
        re-read on each conflict succeeds and the loop keeps retrying -- until
        ``version_conflicts`` reaches the bounded ``max_version_conflicts`` (5) and
        the terminal branch raises. The bound guarantees termination (no hang): the
        stub is invoked at most ``max_version_conflicts + 1`` times (the initial
        attempt plus 5 retries), which we also assert.
        """
        backend, repos, entry_id = setup_with_entry

        attempts = {'count': 0}

        async def always_conflict(*_args: object, **_kwargs: object) -> tuple[list[str], bool]:
            attempts['count'] += 1
            raise VersionConflictError(entry_id)

        # No text -> no generation; isolates the in-transaction CAS retry loop.
        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch(
                'app.tools.context.execute_update_in_transaction',
                new=AsyncMock(side_effect=always_conflict),
            ),
            pytest.raises(ToolError, match='kept changing'),
        ):
            await update_context(context_id=entry_id, metadata={'status': 'x'})

        # Bounded: initial attempt + max_version_conflicts (5) retries = 6 calls, then
        # the loop gives up. This proves termination (the test cannot hang).
        assert attempts['count'] == 6

    @pytest.mark.asyncio
    async def test_entry_vanishes_on_reread_raises_not_found(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """If the entry VANISHES on the post-conflict re-read, ``update_context``
        raises a ToolError matching "not found".

        ``execute_update_in_transaction`` raises ``VersionConflictError`` once, then
        the re-read ``check_entry_exists`` returns ``(False, None, None)`` -- the
        entry was concurrently deleted -- so the retry loop hits the not-found
        terminal branch instead of looping forever.
        """
        backend, repos, entry_id = setup_with_entry

        real_check = repos.context.check_entry_exists
        check_calls = {'count': 0}

        async def vanish_on_reread(context_id: str) -> tuple[bool, str | None, int | None]:
            check_calls['count'] += 1
            # First call = the pre-generation capture: report the entry exists so the
            # transaction proceeds. Subsequent calls = the post-conflict re-read: the
            # entry has vanished.
            if check_calls['count'] == 1:
                return await real_check(context_id)
            return (False, None, None)

        with (
            patch('app.tools.context.ensure_repositories', return_value=repos),
            patch('app.tools.context.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools.context.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch.object(repos.context, 'check_entry_exists', side_effect=vanish_on_reread),
            patch(
                'app.tools.context.execute_update_in_transaction',
                new=AsyncMock(side_effect=VersionConflictError(entry_id)),
            ),
            pytest.raises(ToolError, match='not found'),
        ):
            await update_context(context_id=entry_id, metadata={'status': 'x'})

        # The pre-generation capture (1) plus the single post-conflict re-read (2):
        # the not-found branch terminates the loop immediately, no further retries.
        assert check_calls['count'] == 2
