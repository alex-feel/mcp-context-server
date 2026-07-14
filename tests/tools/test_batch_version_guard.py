"""Regression tests for the optimistic-concurrency version guard in batch paths.

These exercise the version compare-and-swap (CAS) added to ``context_entries``
against a REAL ``SQLiteBackend`` + ``RepositoryContainer`` (NOT mocked repos), so
the CAS predicate runs in actual SQL the way it does in production. Three gaps an
adversarial review flagged in the just-landed version guard are closed here:

1. ``test_duplicate_context_id_*`` -- the regression that was fixed: two updates in
   one ``update_context_batch`` targeting the SAME context_id (different text) must
   BOTH apply (last wins) instead of the second colliding on a stale captured
   version. The pre-fix code captured ``entry_versions`` ONCE per context_id and
   reused it for the repeated id, so the second same-id update presented a stale
   ``expected_version``, matched 0 rows, and raised ``VersionConflictError``
   (atomic: whole-batch abort with a misleading "concurrent modification" message;
   non-atomic: a spurious per-entry failure). These tests therefore FAIL against
   the pre-fix single-captured-version code and PASS now that the batch advances a
   running ``live_versions[context_id]`` after each committed same-id update.
2. ``test_external_bump_*`` -- the guard fires against a genuine EXTERNAL concurrent
   writer: the row's version is advanced out-of-band BEFORE the batch's PHASE 4
   transaction, so the batch's pre-generation captured version is stale. Atomic
   mode aborts the whole batch with a ToolError mentioning concurrent modification
   and leaves the row unmodified; non-atomic mode self-heals (re-reads the version
   and retries) into a success.
3. ``test_dedup_store_bumps_version`` -- a dedup-store UPDATE advances the optimistic
   token, pinning that ANY same-id write moves ``version`` forward (closing the
   dedup-vs-update lost-update window).

Backend/repos setup mirrors ``tests/tools/test_update_concurrency_version_guard.py``
and ``tests/tools/test_generation_first.py``: a real ``SQLiteBackend`` +
``RepositoryContainer`` built from ``load_schema('sqlite')``. Embedding/summary
providers are patched to ``None`` so generation does not run (no Ollama needed) and
the updates apply as pure text/metadata writes through the real CAS path.
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

from app.backends.base import TransactionContext
from app.backends.sqlite_backend import SQLiteBackend
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import VersionConflictError
from app.schemas import load_schema
from app.tools.batch import update_context_batch


@pytest.mark.usefixtures('mock_server_dependencies')
class TestBatchVersionGuard:
    """update_context_batch under the version CAS guard (real SQLite backend)."""

    @pytest_asyncio.fixture
    async def setup_with_entry(
        self, tmp_path: Path,
    ) -> AsyncGenerator[tuple[SQLiteBackend, RepositoryContainer, str], None]:
        db_path = tmp_path / 'test_batch_version_guard.db'
        schema_sql = load_schema('sqlite')
        new_id = generate_id()
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)
            conn.execute(
                'INSERT INTO context_entries (id, thread_id, source, text_content, content_type, metadata) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (new_id, 'batch-guard-thread', 'agent', 'Original text', 'text', '{}'),
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

    # ---- Test 1: duplicate context_id in one batch (the fixed regression) ----

    @pytest.mark.asyncio
    async def test_duplicate_context_id_atomic_both_apply_second_wins(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """atomic=True: two updates for the SAME context_id (different text) BOTH
        apply; the batch succeeds and the persisted text is the SECOND update's.

        Pre-fix this FAILED: the second same-id update reused the version captured
        once for that id, matched 0 rows on the CAS, and raised
        ``VersionConflictError`` -> the whole atomic batch aborted with a
        "Concurrent modification" ToolError. The fix advances
        ``live_versions[context_id]`` after each committed same-id update.
        """
        backend, repos, entry_id = setup_with_entry

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
        ):
            result = await update_context_batch(
                updates=[
                    {'context_id': entry_id, 'text': 'First same-id update'},
                    {'context_id': entry_id, 'text': 'Second same-id update'},
                ],
                atomic=True,
            )

        assert result['success'] is True
        assert result['succeeded'] == 2
        assert result['failed'] == 0
        # Last write wins: the second update's text is persisted.
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text == 'Second same-id update'
        # Two committed same-id text updates each bumped the version (0 -> 2).
        assert final_version == 2

    @pytest.mark.asyncio
    async def test_duplicate_context_id_nonatomic_both_apply_second_wins(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """atomic=False: two updates for the SAME context_id (different text) BOTH
        succeed (no spurious version-conflict failure); final text is the SECOND.

        Pre-fix this FAILED: the second same-id update hit a spurious
        ``VersionConflictError`` recorded as a per-entry failure. The fix persists a
        running ``live_versions[context_id]`` across the sequential non-atomic loop.

        DISCRIMINATION (the running-version increment, not merely the self-heal):
        wrap ``check_entry_exists`` to count its invocations. For a CLEAN duplicate-id
        run the non-atomic path calls it EXACTLY ONCE PER VALIDATED UPDATE up front in
        PHASE 2 (two same-id updates -> 2 calls) and NEVER again, because the running
        ``live_versions[context_id] += 1`` after the first committed update feeds the
        SECOND update the post-bump expected_version, so its CAS matches on the first
        try and raises NO ``VersionConflictError``. The ONLY other call site for
        ``check_entry_exists`` is the conflict self-heal re-read
        (``app/tools/batch.py`` non-atomic ``except VersionConflictError`` branch). So
        if ``live_versions[context_id] += 1`` were dropped, the second update would
        present the STALE captured version, hit a spurious ``VersionConflictError``,
        and trigger one extra self-heal re-read -- pushing the count to 3. Asserting
        the count is EXACTLY 2 (the up-front baseline, ZERO conflict re-reads) fails if
        the increment is dropped, so the self-heal can no longer mask a dropped bump.
        """
        backend, repos, entry_id = setup_with_entry

        real_check = repos.context.check_entry_exists
        check_calls = {'count': 0}

        async def counting_check(context_id: str) -> tuple[bool, str | None, int | None]:
            check_calls['count'] += 1
            return await real_check(context_id)

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch.object(repos.context, 'check_entry_exists', side_effect=counting_check),
        ):
            result = await update_context_batch(
                updates=[
                    {'context_id': entry_id, 'text': 'First same-id update'},
                    {'context_id': entry_id, 'text': 'Second same-id update'},
                ],
                atomic=False,
            )

        assert result['succeeded'] == 2
        assert result['failed'] == 0
        by_index = {r['index']: r for r in result['results']}
        assert by_index[0]['success'] is True
        assert by_index[1]['success'] is True
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text == 'Second same-id update'
        assert final_version == 2
        # Exactly the two PHASE 2 up-front existence checks, ZERO conflict re-reads:
        # no VersionConflictError was self-healed for the duplicate-id case. A dropped
        # ``live_versions[context_id] += 1`` would add a self-heal re-read (count 3).
        assert check_calls['count'] == 2

    # ---- Test 2: batch guard against an EXTERNAL concurrent bump ----

    @pytest.mark.asyncio
    async def test_external_bump_atomic_aborts_whole_batch_no_modification(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """atomic=True: an external writer advances the row's version AFTER the
        batch captures it but BEFORE the PHASE 4 transaction. The stale captured
        version makes the CAS match 0 rows -> the whole atomic batch aborts with a
        ToolError mentioning concurrent modification, and the entry is NOT modified.

        The bump is injected by wrapping ``check_entry_exists`` (the pre-generation
        capture): the first capture returns the original version while a concurrent
        writer simultaneously advances the real row out-of-band, so the captured
        ``expected_version`` is already stale when the transaction's CAS runs.
        """
        backend, repos, entry_id = setup_with_entry

        real_check = repos.context.check_entry_exists
        bumped = {'done': False}

        async def check_then_bump(context_id: str) -> tuple[bool, str | None, int | None]:
            result = await real_check(context_id)
            if not bumped['done'] and result[0]:
                bumped['done'] = True

                def _bump(conn: sqlite3.Connection) -> None:
                    conn.execute(
                        'UPDATE context_entries SET version = version + 1 WHERE id = ?',
                        (context_id,),
                    )

                await backend.execute_write(_bump)
            return result

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch.object(repos.context, 'check_entry_exists', side_effect=check_then_bump),
            pytest.raises(ToolError, match='[Cc]oncurrent modification'),
        ):
            await update_context_batch(
                updates=[{'context_id': entry_id, 'text': 'Update racing an external writer'}],
                atomic=True,
            )

        assert bumped['done'] is True
        # The atomic batch aborted: the entry's text is unchanged. Only the external
        # bump touched the row, so version is 1 and text is still the original.
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text == 'Original text'
        assert final_version == 1

    @pytest.mark.asyncio
    async def test_external_bump_nonatomic_self_heals_to_success(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """atomic=False: the same external bump makes the captured version stale,
        but the non-atomic path re-reads the current version and retries, so the
        entry self-heals into a SUCCESS rather than a spurious failure.
        """
        backend, repos, entry_id = setup_with_entry

        real_check = repos.context.check_entry_exists
        bumped = {'done': False}

        async def check_then_bump(context_id: str) -> tuple[bool, str | None, int | None]:
            result = await real_check(context_id)
            # Only bump on the very FIRST pre-generation capture (version 0), so the
            # retry's re-read sees the post-bump version and the CAS then matches.
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
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch.object(repos.context, 'check_entry_exists', side_effect=check_then_bump),
        ):
            result = await update_context_batch(
                updates=[{'context_id': entry_id, 'text': 'Self-healing non-atomic update'}],
                atomic=False,
            )

        assert bumped['done'] is True
        assert result['succeeded'] == 1
        assert result['failed'] == 0
        assert result['results'][0]['success'] is True
        # External bump (1) + our retried commit (1) => version 2, our text persisted.
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text == 'Self-healing non-atomic update'
        assert final_version == 2

    # ---- Test 3: dedup-store UPDATE bumps version ----

    @pytest.mark.asyncio
    async def test_dedup_store_bumps_version(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """A dedup-store UPDATE advances the optimistic ``version`` token.

        Insert leaves version 0; storing the SAME thread_id/source/text (a dedup
        UPDATE, was_updated=True) must advance the row's version to 1. This pins
        that ANY same-id write moves the token forward, closing the
        dedup-vs-update lost-update window.
        """
        backend, repos, entry_id = setup_with_entry

        # Precondition: the inserted row starts at version 0.
        _text0, version0 = await self._read_row(backend, entry_id)
        assert version0 == 0

        # Same thread_id/source/text as the inserted row -> dedup UPDATE path.
        context_id, was_updated = await repos.context.store_with_deduplication(
            thread_id='batch-guard-thread',
            source='agent',
            content_type='text',
            text_content='Original text',
        )

        assert was_updated is True
        assert context_id == entry_id
        # The dedup UPDATE bumped version 0 -> 1.
        _text1, version1 = await self._read_row(backend, entry_id)
        assert version1 == 1

    # ---- Test 4: non-atomic exhaustion + not-found-on-reread terminal branches ----

    @pytest.mark.asyncio
    async def test_nonatomic_persistent_conflict_exhausts_to_one_failure(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """atomic=False: a PERSISTENT version conflict exhausts the bounded retry
        budget and records EXACTLY ONE per-entry failure (not a hang, not a success).

        ``execute_update_in_transaction`` is stubbed to ALWAYS raise
        ``VersionConflictError`` (a never-healing conflict). ``check_entry_exists``
        keeps reporting the entry exists, so each conflict's re-read succeeds and the
        loop retries -- until ``version_conflicts`` reaches the bounded
        ``max_version_conflicts`` (5) and the non-atomic terminal branch records the
        entry as failed. The bound guarantees termination: the stub is invoked at most
        ``max_version_conflicts + 1`` (the initial attempt plus 5 retries), asserted
        below so the test provably cannot hang.
        """
        backend, repos, entry_id = setup_with_entry

        attempts = {'count': 0}

        async def always_conflict(*_args: object, **_kwargs: object) -> tuple[list[str], bool]:
            attempts['count'] += 1
            raise VersionConflictError(entry_id)

        # No text -> no generation; isolates the in-transaction CAS retry loop.
        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch(
                'app.tools.batch.execute_update_in_transaction',
                new=AsyncMock(side_effect=always_conflict),
            ),
        ):
            result = await update_context_batch(
                updates=[{'context_id': entry_id, 'metadata': {'status': 'x'}}],
                atomic=False,
            )

        assert result['succeeded'] == 0
        assert result['failed'] == 1
        assert result['results'][0]['success'] is False
        # Bounded: initial attempt + max_version_conflicts (5) retries = 6 calls, then
        # the loop gives up. This proves termination (the test cannot hang).
        assert attempts['count'] == 6
        # The entry is untouched (every attempt rolled back).
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text == 'Original text'
        assert final_version == 0

    @pytest.mark.asyncio
    async def test_nonatomic_entry_vanishes_on_reread_records_not_found(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """atomic=False: if the entry VANISHES on the post-conflict re-read, the
        non-atomic path records a per-entry "not found" failure (not a hang).

        ``execute_update_in_transaction`` raises ``VersionConflictError`` once, then
        the re-read ``check_entry_exists`` returns ``(False, None, None)`` -- the row
        was concurrently deleted -- so the loop hits the not-found terminal branch.
        The PHASE 2 up-front existence check (call 1) reports the entry exists so the
        batch proceeds; the post-conflict re-read (call 2) sees it gone.
        """
        backend, repos, entry_id = setup_with_entry

        real_check = repos.context.check_entry_exists
        check_calls = {'count': 0}

        async def vanish_on_reread(context_id: str) -> tuple[bool, str | None, int | None]:
            check_calls['count'] += 1
            if check_calls['count'] == 1:
                return await real_check(context_id)
            return (False, None, None)

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch.object(repos.context, 'check_entry_exists', side_effect=vanish_on_reread),
            patch(
                'app.tools.batch.execute_update_in_transaction',
                new=AsyncMock(side_effect=VersionConflictError(entry_id)),
            ),
        ):
            result = await update_context_batch(
                updates=[{'context_id': entry_id, 'metadata': {'status': 'x'}}],
                atomic=False,
            )

        assert result['succeeded'] == 0
        assert result['failed'] == 1
        assert result['results'][0]['success'] is False
        assert result['results'][0]['error'] is not None
        assert 'not found' in result['results'][0]['error']
        # PHASE 2 up-front check (1) + the single post-conflict re-read (2): the
        # not-found branch terminates the loop immediately, no further retries.
        assert check_calls['count'] == 2

    # ---- Test 5: atomic delete-mid-CAS disambiguates to not-found ----

    @pytest.mark.asyncio
    async def test_atomic_row_deleted_mid_cas_aborts_not_found(
        self, setup_with_entry: tuple[SQLiteBackend, RepositoryContainer, str],
    ) -> None:
        """atomic=True: a row deleted between its version capture and the CAS UPDATE
        aborts the batch with the standard not-found error, not retry advice.

        A CAS matching zero rows is ambiguous -- version bumped by a concurrent
        writer (retryable) or row deleted (permanent) -- so the atomic handler
        re-probes existence on the OPEN transaction connection to disambiguate.
        ``execute_update_in_transaction`` is stubbed to delete the row on the
        transaction connection and then raise ``VersionConflictError``,
        deterministically reproducing the post-capture/pre-CAS delete
        interleaving. Without the disambiguation the batch aborted with
        'Concurrent modification ... Retry the request.' for a permanently
        deleted row -- advice no retry could ever satisfy.
        """
        backend, repos, entry_id = setup_with_entry

        async def delete_then_conflict(
            repos_arg: RepositoryContainer,
            txn: TransactionContext,
            **_kwargs: object,
        ) -> tuple[list[str], bool]:
            await repos_arg.context.delete_by_ids([entry_id], txn=txn)
            raise VersionConflictError(entry_id)

        with (
            patch('app.tools.batch.ensure_repositories', return_value=repos),
            patch('app.tools.batch.get_embedding_provider', return_value=None),
            patch('app.tools.batch.get_summary_provider', return_value=None),
            patch('app.tools._shared.get_embedding_provider', return_value=None),
            patch('app.tools._shared.get_summary_provider', return_value=None),
            patch(
                'app.tools.batch.execute_update_in_transaction',
                new=AsyncMock(side_effect=delete_then_conflict),
            ),
            pytest.raises(ToolError, match='not found') as exc_info,
        ):
            await update_context_batch(
                updates=[{'context_id': entry_id, 'metadata': {'status': 'x'}}],
                atomic=True,
            )

        # The standard not-found abort every other update path emits, without
        # concurrent-modification retry advice.
        assert 'Retry the request' not in str(exc_info.value)
        assert 'No entries were updated (atomic batch)' in str(exc_info.value)
        # The atomic batch rolled back, so the stub's in-transaction delete was
        # discarded and the row is still present with its original text.
        final_text, final_version = await self._read_row(backend, entry_id)
        assert final_text == 'Original text'
        assert final_version == 0
