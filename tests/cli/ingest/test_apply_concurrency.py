"""Tests for a sync run that keeps several create and update batches in flight at once.

Each run talks to the in-memory server, which records how many calls were in flight
when each one arrived and can hold calls so that they finish in a chosen order.
Every connection is a session that carries one call at a time, so a run that sent a
second call over a busy session fails the test.
"""

from pathlib import Path
from typing import cast

import httpx2
import pytest

from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import EXIT_TRANSIENT
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import TokenFileAuth
from app.cli.ingest.client import ToolFailure
from app.cli.ingest.report import RunReport
from app.ingest import INGEST_KEY
from tests.cli.ingest._fake_server import FakeServer
from tests.cli.ingest._fake_server import Json
from tests.cli.ingest._sync_run import DELETE
from tests.cli.ingest._sync_run import PLAN
from tests.cli.ingest._sync_run import STORE
from tests.cli.ingest._sync_run import THREAD_ID
from tests.cli.ingest._sync_run import UPDATE
from tests.cli.ingest._sync_run import sources_with
from tests.cli.ingest._sync_run import sync_directory
from tests.cli.ingest._sync_run import write_document

_TOO_LARGE = 'the request was too large for the server (HTTP 413)'
_GENERATION_FAILED = 'Generation failed after exhausting configured retries'


def _write_documents(root: Path, count: int) -> list[Path]:
    """Write numbered corpus documents, which sort in the order they are numbered."""
    return [write_document(root, f'{number:02}.md', f'document {number}') for number in range(count)]


def _names(count: int) -> list[str]:
    """Return the source ids of the numbered documents."""
    return [f'{number:02}.md' for number in range(count)]


def _peak(server: FakeServer, tool: str) -> int:
    """Return the most calls that were in flight when a call of a tool arrived."""
    return max(call.in_flight for call in server.calls_to(tool))


def _stored_sources(server: FakeServer) -> list[str]:
    """Return the source ids of every entry sent through the batch store, in the order the calls arrived."""
    return [
        cast('str', cast('Json', cast('Json', entry['metadata'])[INGEST_KEY])['source_id'])
        for call in server.calls_to(STORE)
        for entry in cast('list[Json]', call.arguments['entries'])
    ]


def _texts(server: FakeServer) -> dict[str, list[str]]:
    """Return the stored texts of each source id of the corpus thread."""
    return {source: [entry.text for entry in entries] for source, entries in server.by_source(THREAD_ID).items()}


def _outcomes(report: RunReport) -> list[tuple[str, str, str | None]]:
    """Return every document outcome without its context id, in the order the report holds them."""
    return [(item.source_id, item.outcome, item.error) for item in report.outcomes]


def _failures(report: RunReport) -> list[tuple[str, str | None]]:
    """Return the failed documents with their errors."""
    return [(item.source_id, item.error) for item in report.outcomes if item.outcome == 'failed']


def _assert_created_where_reported(server: FakeServer, report: RunReport) -> None:
    """Assert that the entry each created outcome names holds that outcome's document."""
    for item in report.outcomes:
        if item.outcome == 'created':
            assert item.context_id is not None
            assert server.entries[item.context_id].source_id == item.source_id


@pytest.fixture
def server() -> FakeServer:
    """Return an empty in-memory server."""
    return FakeServer()


class TestOverlap:
    """Up to --concurrency batch calls are in flight at once, each on a session of its own."""

    @pytest.mark.asyncio
    async def test_up_to_the_concurrency_batches_are_in_flight_at_once(self, server: FakeServer, tmp_path: Path) -> None:
        """Creates and then updates overlap three at a time, are sent in order and never share a session."""
        paths = _write_documents(tmp_path, 7)

        created = await sync_directory(server, tmp_path, batch_size=1, concurrency=3)

        assert created.exit_code == EXIT_CONVERGED
        assert _peak(server, STORE) == 3
        assert len(server.connections) == 3
        assert _stored_sources(server) == _names(7)
        assert sources_with(created, 'created') == _names(7)
        assert sorted(server.by_source(THREAD_ID)) == _names(7)

        for path in paths:
            path.write_text(path.read_text(encoding='utf-8') + ', revised', encoding='utf-8')
        server.connections.clear()

        updated = await sync_directory(server, tmp_path, batch_size=1, concurrency=3)

        assert updated.exit_code == EXIT_CONVERGED
        assert _peak(server, UPDATE) == 3
        assert len(server.connections) == 3
        assert sources_with(updated, 'updated') == _names(7)

    @pytest.mark.asyncio
    async def test_the_default_sends_one_batch_at_a_time(self, server: FakeServer, tmp_path: Path) -> None:
        """Without --concurrency every batch call waits for the one before it, over the run's only session."""
        _write_documents(tmp_path, 4)

        report = await sync_directory(server, tmp_path, batch_size=1)

        assert report.exit_code == EXIT_CONVERGED
        assert _peak(server, STORE) == 1
        assert len(server.connections) == 1

    @pytest.mark.asyncio
    async def test_a_run_opens_only_the_sessions_its_batches_use(self, server: FakeServer, tmp_path: Path) -> None:
        """Two batches open two sessions however high the concurrency; a run with nothing to write opens one."""
        _write_documents(tmp_path, 2)

        first = await sync_directory(server, tmp_path, batch_size=1, concurrency=5)

        assert first.exit_code == EXIT_CONVERGED
        assert (len(server.connections), _peak(server, STORE)) == (2, 2)
        server.connections.clear()

        rerun = await sync_directory(server, tmp_path, batch_size=1, concurrency=5)

        assert rerun.exit_code == EXIT_CONVERGED
        assert len(server.connections) == 1


class TestSequentialParity:
    """Batches that finish out of order leave the same report a sequential run leaves."""

    @pytest.mark.asyncio
    async def test_the_report_matches_a_sequential_run(self, tmp_path: Path) -> None:
        """Outcomes in batch order, the counts, the errors, the exit code and the stored documents all match."""
        _write_documents(tmp_path, 9)
        sequential_server, concurrent_server = FakeServer(), FakeServer()
        for each in (sequential_server, concurrent_server):
            each.fail_next(STORE, ToolFailure(_TOO_LARGE))
            each.failing_sources['05.md'] = _GENERATION_FAILED
        concurrent_server.hold_next(STORE, 6, 1, 4, 2, 1)

        sequential = await sync_directory(sequential_server, tmp_path, batch_size=2)
        concurrent = await sync_directory(concurrent_server, tmp_path, batch_size=2, concurrency=3)

        assert _peak(concurrent_server, STORE) == 3
        assert concurrent.exit_code == sequential.exit_code == EXIT_PARTIAL
        assert _outcomes(concurrent) == _outcomes(sequential)
        assert concurrent.counts == sequential.counts
        assert concurrent.errors == sequential.errors
        assert _failures(concurrent) == [('00.md', _TOO_LARGE), ('01.md', _TOO_LARGE), ('05.md', _GENERATION_FAILED)]
        assert _texts(concurrent_server) == _texts(sequential_server)
        _assert_created_where_reported(concurrent_server, concurrent)

    @pytest.mark.asyncio
    async def test_per_entry_results_are_matched_by_index_in_any_order(self, server: FakeServer, tmp_path: Path) -> None:
        """A server that lists a batch's results in another order still has each result land on its own document."""
        _write_documents(tmp_path, 3)
        server.reverse_results = True
        server.failing_sources['01.md'] = _GENERATION_FAILED

        report = await sync_directory(server, tmp_path, batch_size=3)

        assert report.exit_code == EXIT_PARTIAL
        assert _outcomes(report)[:3] == [
            ('00.md', 'created', None), ('01.md', 'failed', _GENERATION_FAILED), ('02.md', 'created', None),
        ]
        _assert_created_where_reported(server, report)


class TestFailedBatches:
    """A failed batch never cancels the batches in flight beside it."""

    @pytest.mark.asyncio
    async def test_a_refused_batch_leaves_the_batch_beside_it_running(self, server: FakeServer, tmp_path: Path) -> None:
        """The refused batch fails its own entry while its sibling finishes, and the later batches are still sent."""
        _write_documents(tmp_path, 5)
        server.fail_next(STORE, ToolFailure(_TOO_LARGE))
        server.hold_next(STORE, 1, 3)

        report = await sync_directory(server, tmp_path, batch_size=1, concurrency=2)

        assert report.exit_code == EXIT_PARTIAL
        assert [call.in_flight for call in server.calls_to(STORE)][:2] == [1, 2]
        assert len(server.calls_to(STORE)) == 5
        assert _failures(report) == [('00.md', _TOO_LARGE)]
        assert sources_with(report, 'created') == _names(5)[1:]

    @pytest.mark.asyncio
    async def test_a_call_that_ends_the_run_lets_the_running_batches_finish(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """No batch starts after the failure, the ones in flight finish and are reported, and a rerun converges."""
        _write_documents(tmp_path, 6)
        timeout = IngestExitError(EXIT_TRANSIENT, 'the call to store_context_batch got no answer within --call-timeout')
        server.fail_next(STORE, timeout, commit=True)
        server.hold_next(STORE, 1, 3, 3)

        report = await sync_directory(server, tmp_path, batch_size=1, concurrency=3)

        assert report.exit_code == EXIT_TRANSIENT
        assert report.errors == [timeout.message]
        assert [call.tool for call in server.calls] == ['preflight', PLAN, STORE, STORE, STORE]
        assert _outcomes(report) == [('01.md', 'created', None), ('02.md', 'created', None)]
        server.calls.clear()

        rerun = await sync_directory(server, tmp_path, batch_size=1, concurrency=3)

        assert rerun.exit_code == EXIT_CONVERGED
        assert sources_with(rerun, 'created') == ['03.md', '04.md', '05.md']
        assert {source: len(entries) for source, entries in server.by_source(THREAD_ID).items()} == dict.fromkeys(_names(6), 1)

    @pytest.mark.asyncio
    async def test_every_distinct_failure_that_ends_the_run_is_reported(self, server: FakeServer, tmp_path: Path) -> None:
        """Batches failing together report each distinct failure once, and the most severe one sets the exit code."""
        _write_documents(tmp_path, 4)
        lost = 'the call to store_context_batch lost its connection to the server (ReadError)'
        expired = 'the server refused the bearer token (HTTP 401) after accepting it earlier in this run'
        server.fail_next(STORE, IngestExitError(EXIT_TRANSIENT, lost))
        server.fail_next(STORE, IngestExitError(EXIT_TRANSIENT, lost))
        server.fail_next(STORE, IngestExitError(EXIT_CONFIG, expired))

        report = await sync_directory(server, tmp_path, batch_size=1, concurrency=3)

        assert report.exit_code == EXIT_CONFIG
        assert report.errors == [lost, expired]
        assert len(server.calls_to(STORE)) == 3
        assert report.outcomes == []


class TestSequentialLegs:
    """The deletes run alone after every create and update, and the final plan runs last."""

    @pytest.mark.asyncio
    async def test_deletes_and_the_final_plan_run_alone_after_the_batches(self, server: FakeServer, tmp_path: Path) -> None:
        """Creates and updates overlap; every plan and delete call is the only call in flight, over the first session."""
        paths = _write_documents(tmp_path, 6)
        await sync_directory(server, tmp_path)
        for path in paths[:3]:
            path.write_text(path.read_text(encoding='utf-8') + ', revised', encoding='utf-8')
        for path in paths[3:5]:
            path.unlink()
        write_document(tmp_path, '10.md', 'document 10')
        write_document(tmp_path, '11.md', 'document 11')
        server.calls.clear()

        report = await sync_directory(server, tmp_path, batch_size=1, concurrency=3)

        assert report.exit_code == EXIT_CONVERGED
        assert (report.counts['created'], report.counts['updated'], report.counts['deleted']) == (2, 3, 2)
        assert (_peak(server, STORE), _peak(server, UPDATE)) == (2, 3)
        assert all(call.in_flight == 1 for call in server.calls if call.tool in {PLAN, DELETE})
        (first,) = (call.session for call in server.calls if call.tool == 'preflight')
        assert {call.session for call in server.calls if call.tool in {PLAN, DELETE}} == {first}
        assert len({call.session for call in server.calls_to(UPDATE)}) == 3
        tools = [call.tool for call in server.calls]
        assert max(index for index, tool in enumerate(tools) if tool in {STORE, UPDATE}) < tools.index(DELETE)
        assert tools[-1] == PLAN


class TestSessions:
    """Each further session gets the token the way the first one does."""

    @pytest.mark.asyncio
    async def test_every_session_reads_the_token_file_itself(self, server: FakeServer, tmp_path: Path) -> None:
        """A token file is read by each session's own auth, which tracks the token that session sent."""
        corpus_root = tmp_path / 'corpus'
        _write_documents(corpus_root, 2)
        token = tmp_path / 'token'
        token.write_text('ingest-token\n', encoding='utf-8')

        report = await sync_directory(server, corpus_root, token_file=token, batch_size=1, concurrency=2)

        assert report.exit_code == EXIT_CONVERGED
        first, second = server.connections
        assert isinstance(first, TokenFileAuth)
        assert isinstance(second, TokenFileAuth)
        assert first is not second
        assert (first.path, second.path) == (token, token)

    @pytest.mark.asyncio
    async def test_sessions_share_the_token_read_once_from_the_environment(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A token from --token-env is read once and every session sends it."""
        _write_documents(tmp_path, 2)
        monkeypatch.setenv('INGEST_TEST_TOKEN', 'ingest-token')

        report = await sync_directory(server, tmp_path, token_env='INGEST_TEST_TOKEN', batch_size=1, concurrency=2)

        assert report.exit_code == EXIT_CONVERGED
        first, second = server.connections
        assert isinstance(first, httpx2.Auth)
        assert first is second
