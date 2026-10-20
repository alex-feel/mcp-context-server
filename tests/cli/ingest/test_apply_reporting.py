"""Tests for what a sync run prints and counts: batch progress, the dry run and foreign members.

Each run talks to the in-memory server that plans with the server's own diff, and
each test reads the lines the run printed and the counts its report holds.
"""

import re
from pathlib import Path

import pytest

from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import ToolFailure
from tests.cli.ingest._fake_server import OTHER
from tests.cli.ingest._fake_server import FakeServer
from tests.cli.ingest._fake_server import Json
from tests.cli.ingest._sync_run import PLAN
from tests.cli.ingest._sync_run import STORE
from tests.cli.ingest._sync_run import THREAD_ID
from tests.cli.ingest._sync_run import managed_metadata
from tests.cli.ingest._sync_run import sources_with
from tests.cli.ingest._sync_run import sync_directory
from tests.cli.ingest._sync_run import write_document

# The line a run prints after every batch call: the leg, the documents it applied so
# far, the leg's documents, the batch's position, its failed entries and the seconds
# since the leg began.
_BATCH_LINE = re.compile(r'(created|updated) (\d+) of (\d+) documents \(batch (\d+) of (\d+), (\d+) failed, (\d+) s\)')

type BatchLine = tuple[str, int, int, int, int, int]


def _batch_lines(printed: str) -> list[BatchLine]:
    """Return the batch lines a run printed, in order, after checking that each leg's elapsed seconds never decrease."""
    lines: list[BatchLine] = []
    elapsed: dict[str, int] = {}
    for line in printed.splitlines():
        match = _BATCH_LINE.fullmatch(line)
        if match is None:
            continue
        leg, applied, total, position, batches, failed, seconds = match.groups()
        assert int(seconds) >= elapsed.get(leg, 0)
        elapsed[leg] = int(seconds)
        lines.append((leg, int(applied), int(total), int(position), int(batches), int(failed)))
    return lines


def _write_documents(root: Path, count: int) -> list[Path]:
    """Write numbered corpus documents, which sort in the order they are numbered."""
    return [write_document(root, f'{number:02}.md', f'document {number}') for number in range(count)]


class TestBatchProgress:
    """Every batch call prints a line, so a long create or update leg shows how far it got."""

    @pytest.mark.asyncio
    async def test_every_batch_prints_the_documents_applied_so_far(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Each create and update batch prints the running count, its position and failures, then the leg its total."""
        server = FakeServer()
        paths = _write_documents(tmp_path, 25)

        created = await sync_directory(server, tmp_path, batch_size=10)

        printed = capsys.readouterr().out
        assert created.exit_code == EXIT_CONVERGED
        assert _batch_lines(printed) == [
            ('created', 10, 25, 1, 3, 0),
            ('created', 20, 25, 2, 3, 0),
            ('created', 25, 25, 3, 3, 0),
        ]
        assert 'created 25 of 25 documents' in printed.splitlines()

        for path in paths:
            path.write_text(path.read_text(encoding='utf-8') + ', revised', encoding='utf-8')

        updated = await sync_directory(server, tmp_path, batch_size=10)

        printed = capsys.readouterr().out
        assert updated.exit_code == EXIT_CONVERGED
        assert _batch_lines(printed) == [
            ('updated', 10, 25, 1, 3, 0),
            ('updated', 20, 25, 2, 3, 0),
            ('updated', 25, 25, 3, 3, 0),
        ]
        assert 'updated 25 of 25 documents' in printed.splitlines()

    @pytest.mark.asyncio
    async def test_a_batch_prints_its_line_before_the_next_batch_is_sent(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The line of a batch reaches standard output while the run still waits on the next call."""
        server = FakeServer()
        _write_documents(tmp_path, 25)
        before_each_store: list[list[BatchLine]] = []
        printed: list[str] = []
        answer = server.call_tool

        async def call_tool(name: str, arguments: Json) -> Json | ToolFailure:
            if name == STORE:
                printed.append(capsys.readouterr().out)
                before_each_store.append(_batch_lines(''.join(printed)))
            return await answer(name, arguments)

        monkeypatch.setattr(server, 'call_tool', call_tool)

        report = await sync_directory(server, tmp_path, batch_size=10)

        assert report.exit_code == EXIT_CONVERGED
        assert before_each_store == [
            [],
            [('created', 10, 25, 1, 3, 0)],
            [('created', 10, 25, 1, 3, 0), ('created', 20, 25, 2, 3, 0)],
        ]

    @pytest.mark.asyncio
    async def test_a_failed_batch_still_prints_its_line(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """A refused batch call fails all its entries and a refused entry fails alone; each batch still prints its line."""
        server = FakeServer()
        _write_documents(tmp_path, 25)
        server.fail_next(STORE, ToolFailure('the request was too large for the server (HTTP 413)'))
        server.failing_sources['13.md'] = 'Generation failed after exhausting configured retries'

        report = await sync_directory(server, tmp_path, batch_size=10)

        printed = capsys.readouterr().out
        assert report.exit_code == EXIT_PARTIAL
        assert _batch_lines(printed) == [
            ('created', 0, 25, 1, 3, 10),
            ('created', 9, 25, 2, 3, 1),
            ('created', 14, 25, 3, 3, 0),
        ]
        assert 'created 14 of 25 documents' in printed.splitlines()


class TestForeignEntries:
    """Another principal's members are counted and warned about, never changed."""

    @pytest.mark.asyncio
    async def test_foreign_members_warn_and_stay_untouched(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        """The run builds the caller's own copy and leaves the other principal's public entry alone."""
        server = FakeServer()
        write_document(tmp_path, 'a.md', 'alpha')
        foreign = server.seed(
            thread_id=THREAD_ID, text='theirs', metadata=managed_metadata('a.md'), owner=OTHER, visibility='public',
        )

        report = await sync_directory(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert report.counts['foreign'] == 1
        assert server.entries[foreign].text == 'theirs'
        assert len(server.by_source(THREAD_ID)['a.md']) == 2
        assert 'another principal owns' in capsys.readouterr().err


class TestDryRun:
    """A dry run plans, prints the plan and writes nothing."""

    @pytest.mark.asyncio
    async def test_a_dry_run_prints_the_plan_and_writes_nothing(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Creates, updates and deletes are printed with their ids, and only the plan tool is called."""
        server = FakeServer()
        edited = write_document(tmp_path, 'a.md', 'alpha')
        removed = write_document(tmp_path, 'b.md', 'beta')
        await sync_directory(server, tmp_path)
        ids = {source: entries[0].context_id for source, entries in server.by_source(THREAD_ID).items()}
        edited.write_text('alpha, revised', encoding='utf-8')
        removed.unlink()
        write_document(tmp_path, 'c.md', 'gamma')
        server.calls.clear()
        capsys.readouterr()

        report = await sync_directory(server, tmp_path, dry_run=True)

        assert report.exit_code == EXIT_CONVERGED
        assert {call.tool for call in server.calls} == {'preflight', PLAN}
        printed = capsys.readouterr().out.splitlines()
        assert 'would create c.md' in printed
        assert f'would update a.md {ids["a.md"]} (text)' in printed
        assert f'would delete b.md {ids["b.md"]} (absent)' in printed
        assert (report.counts['created'], report.counts['updated'], report.counts['deleted']) == (1, 1, 1)
        assert server.entries[ids['a.md']].text == 'alpha'

    @pytest.mark.asyncio
    async def test_a_dry_run_names_the_planned_deletes_past_the_listed_ones(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A plan lists 100 deletes; the dry run prints them, says how many more the plan holds and counts them all."""
        server = FakeServer()
        write_document(tmp_path, 'a.md', 'alpha')
        for number in range(105):
            server.seed(thread_id=THREAD_ID, text=f'gone {number}', metadata=managed_metadata(f'gone/{number:03}.md'))

        report = await sync_directory(server, tmp_path, dry_run=True)

        printed = capsys.readouterr().out.splitlines()
        assert report.exit_code == EXIT_CONVERGED
        assert 'plan: 1 to create, 0 to update, 105 to delete, 0 unchanged, 0 kept, 0 foreign' in printed
        assert '... and 5 more planned deletes in the corpus' in printed
        assert sum(line.startswith('would delete gone/') for line in printed) == 100
        assert report.counts['deleted'] == 105
        assert len(sources_with(report, 'deleted')) == 100
        assert len(server.entries) == 105

    @pytest.mark.asyncio
    async def test_a_dry_run_with_a_kept_file_exits_1(self, tmp_path: Path) -> None:
        """A keep item makes even a dry run partial."""
        server = FakeServer()
        write_document(tmp_path, 'a.md', 'alpha')
        (tmp_path / 'b.md').write_bytes(b'\xff not utf-8')

        report = await sync_directory(server, tmp_path, dry_run=True)

        assert report.exit_code == EXIT_PARTIAL
        assert sources_with(report, 'kept') == ['b.md']
        assert server.entries == {}
