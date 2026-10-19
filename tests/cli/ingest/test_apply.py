"""Tests for the legs of a sync run: plan, creates, updates, deletes and the final verification.

Each run talks to an in-memory server that plans with the server's own diff, so a
sequence of runs over a changing directory shows what every leg sends and that the
thread converges on the corpus.
"""

import os
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TypedDict
from typing import Unpack
from typing import cast

import pytest

from app.cli.ingest import corpus
from app.cli.ingest.apply import RunConfig
from app.cli.ingest.apply import run_sync
from app.cli.ingest.client import EXIT_CONFIG
from app.cli.ingest.client import EXIT_CONVERGED
from app.cli.ingest.client import EXIT_PARTIAL
from app.cli.ingest.client import EXIT_TRANSIENT
from app.cli.ingest.client import IngestExitError
from app.cli.ingest.client import TokenFileAuth
from app.cli.ingest.client import ToolFailure
from app.cli.ingest.corpus import DirectoryFailure
from app.cli.ingest.corpus import Visibility
from app.cli.ingest.manifest import meta_digest
from app.cli.ingest.manifest import request_bytes
from app.cli.ingest.report import RunReport
from app.ingest import INGEST_KEY
from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import ingest_block_error
from tests.cli.ingest._fake_server import OTHER
from tests.cli.ingest._fake_server import FakeServer
from tests.cli.ingest._fake_server import Json

THREAD_ID = 'kb:handbook'
SERVER = 'http://127.0.0.1:8765/mcp'
STORE = 'store_context_batch'
UPDATE = 'update_context_batch'
DELETE = 'delete_context_batch'
PLAN = 'plan_context_sync'
SYNCED_AT = '2026-10-18T12:00:00Z'


def _write(root: Path, relative: str, text: str) -> Path:
    """Write a corpus file, creating its directories."""
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


class _Settings(TypedDict, total=False):
    """The run settings a test changes from their defaults."""

    tags: tuple[str, ...]
    metadata: Mapping[str, object]
    visibility: Visibility
    token_file: Path
    dry_run: bool
    allow_empty_corpus: bool
    batch_size: int
    max_request_bytes: int


async def _sync(server: FakeServer, root: Path, **settings: Unpack[_Settings]) -> RunReport:
    """Run a sync of a directory against the fake server."""
    config = RunConfig(server=SERVER, corpus='handbook', source_dir=root, **settings)
    return await run_sync(config, connect=server.connect)


def _managed(source_id: str) -> Json:
    """Return the metadata of an entry an earlier run stored for a source."""
    return {INGEST_KEY: {'scope': THREAD_ID, 'source_id': source_id, 'meta_digest': '0' * 64, 'synced_at': SYNCED_AT}}


def _sent(server: FakeServer, tool: str, key: str) -> list[Json]:
    """Return every entry the run sent through a batch tool, in order."""
    return [entry for call in server.calls_to(tool) for entry in cast('list[Json]', call.arguments[key])]


def _block(entry: Json) -> Json:
    """Return the ingest block of a sent entry."""
    return cast('Json', cast('Json', entry['metadata'])[INGEST_KEY])


def _outcomes(report: RunReport, outcome: str) -> list[str]:
    """Return the source ids of the documents with one outcome."""
    return [item.source_id for item in report.outcomes if item.outcome == outcome]


def _failures(report: RunReport) -> list[tuple[str, str | None]]:
    """Return the failed documents with their errors."""
    return [(item.source_id, item.error) for item in report.outcomes if item.outcome == 'failed']


def _refuse_listing(monkeypatch: pytest.MonkeyPatch, refused: Path) -> None:
    """Make os.scandir fail for one directory, as a directory without read permission does."""
    real_scandir = os.scandir

    def scandir(path: str) -> Iterator[os.DirEntry[str]]:
        if Path(path) == refused:
            raise PermissionError(13, 'Permission denied', path)
        return real_scandir(path)

    monkeypatch.setattr(os, 'scandir', scandir)


@pytest.fixture
def server() -> FakeServer:
    """Return an empty in-memory server."""
    return FakeServer()


class TestCreates:
    """New documents are stored as fresh entries, never merged, with the visibility the digest covers."""

    @pytest.mark.asyncio
    async def test_bootstrap_stores_every_document_without_deduplication(self, server: FakeServer, tmp_path: Path) -> None:
        """Every create goes through a non-atomic batch with deduplicate false and an explicit visibility."""
        _write(tmp_path, 'a.md', '# Alpha\n\nfirst')
        _write(tmp_path, 'guides/b.md', 'second')

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [call.arguments['atomic'] for call in server.calls_to(STORE)] == [False]
        entries = _sent(server, STORE, 'entries')
        assert [_block(entry)['source_id'] for entry in entries] == ['a.md', 'guides/b.md']
        for entry in entries:
            assert entry['deduplicate'] is False
            assert entry['visibility'] == 'private'
            assert entry['thread_id'] == THREAD_ID
            assert entry['source'] == 'user'
            assert ingest_block_error(_block(entry)) is None
            assert _block(entry)['scope'] == THREAD_ID
        assert sorted(server.by_source(THREAD_ID)) == ['a.md', 'guides/b.md']
        assert _outcomes(report, 'created') == ['a.md', 'guides/b.md']
        assert {item.context_id for item in report.outcomes} == set(server.entries)
        assert server.calls_to(DELETE) == []
        assert report.errors == []

    @pytest.mark.asyncio
    async def test_sent_visibility_is_the_hashed_visibility(self, server: FakeServer, tmp_path: Path) -> None:
        """The digest each entry carries covers exactly the metadata, tags and visibility it is sent with."""
        _write(tmp_path, 'a.md', 'alpha')

        report = await _sync(server, tmp_path, visibility='public', tags=('Guide', 'ops'), metadata={'team': 'docs'})

        assert report.exit_code == EXIT_CONVERGED
        (entry,) = _sent(server, STORE, 'entries')
        metadata = {key: value for key, value in cast('Json', entry['metadata']).items() if key != INGEST_KEY}
        assert entry['visibility'] == 'public'
        assert metadata == {'title': 'a', 'team': 'docs'}
        assert _block(entry)['meta_digest'] == meta_digest(metadata, cast('list[str]', entry['tags']), 'public')

    @pytest.mark.asyncio
    async def test_creates_are_packed_by_count_and_by_bytes(self, server: FakeServer, tmp_path: Path) -> None:
        """A batch holds at most --batch-size entries and never more bytes than the request budget."""
        for name in 'abcde':
            _write(tmp_path, f'{name}.md', name * 2000)

        by_count = await _sync(server, tmp_path, batch_size=2)

        assert by_count.exit_code == EXIT_CONVERGED
        assert [len(cast('list[Json]', call.arguments['entries'])) for call in server.calls_to(STORE)] == [2, 2, 1]

        bytes_server = FakeServer()
        by_bytes = await _sync(bytes_server, tmp_path, max_request_bytes=6000)

        assert by_bytes.exit_code == EXIT_CONVERGED
        batches = [cast('list[Json]', call.arguments['entries']) for call in bytes_server.calls_to(STORE)]
        assert len(batches) == 3
        assert all(request_bytes(batch) <= 6000 for batch in batches)
        assert len(bytes_server.by_source(THREAD_ID)) == 5

    @pytest.mark.asyncio
    async def test_a_refused_entry_fails_alone_and_leaves_the_run_unconverged(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """A per-entry failure is that document's failure; the rest are stored and the final plan still shows it."""
        _write(tmp_path, 'a.md', 'alpha')
        _write(tmp_path, 'b.md', 'beta')
        server.failing_sources['b.md'] = 'Generation failed after exhausting configured retries'

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert _outcomes(report, 'created') == ['a.md']
        assert _failures(report) == [('b.md', 'Generation failed after exhausting configured retries')]
        assert any('did not converge' in error for error in report.errors)

    @pytest.mark.asyncio
    async def test_a_refused_batch_call_fails_its_entries_and_the_run_goes_on(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """A batch the server or a proxy refused fails its own entries; the next batch is still sent."""
        _write(tmp_path, 'a.md', 'alpha')
        _write(tmp_path, 'b.md', 'beta')
        server.fail_next(STORE, ToolFailure('the request was too large for the server (HTTP 413)'))

        report = await _sync(server, tmp_path, batch_size=1)

        assert report.exit_code == EXIT_PARTIAL
        assert len(server.calls_to(STORE)) == 2
        assert _failures(report) == [('a.md', 'the request was too large for the server (HTTP 413)')]
        assert _outcomes(report, 'created') == ['b.md']

    @pytest.mark.asyncio
    async def test_an_entry_the_response_does_not_answer_fails(self, server: FakeServer, tmp_path: Path) -> None:
        """An entry without a per-entry result is a failure of that document."""
        _write(tmp_path, 'a.md', 'alpha')
        _write(tmp_path, 'b.md', 'beta')
        server.answer_next(STORE, {
            'success': True, 'total': 2, 'succeeded': 1, 'failed': 0, 'message': 'Stored 1',
            'results': [{'index': 0, 'success': True, 'context_id': '0190' + '0' * 28, 'error': None}],
        })

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert _outcomes(report, 'created') == ['a.md']
        assert _failures(report) == [('b.md', 'store_context_batch returned no result for this entry')]

    @pytest.mark.asyncio
    async def test_a_batch_response_of_another_shape_exits_2(self, server: FakeServer, tmp_path: Path) -> None:
        """A batch answer without the batch response's fields ends the run as a configuration error."""
        _write(tmp_path, 'a.md', 'alpha')
        server.answer_next(STORE, {'stored': 1})

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any('store_context_batch with an unexpected result' in error for error in report.errors)


class TestUpdates:
    """A changed document updates its stored entry, replacing the metadata in full."""

    @pytest.mark.asyncio
    async def test_a_rerun_without_changes_writes_nothing(self, server: FakeServer, tmp_path: Path) -> None:
        """Every document is unchanged, so no batch tool is called."""
        _write(tmp_path, 'a.md', 'alpha')
        _write(tmp_path, 'b.md', 'beta')
        await _sync(server, tmp_path)
        server.calls.clear()

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert {call.tool for call in server.calls} == {'preflight', PLAN}
        assert _outcomes(report, 'unchanged') == ['a.md', 'b.md']

    @pytest.mark.asyncio
    async def test_a_text_change_sends_the_text_and_the_full_metadata(self, server: FakeServer, tmp_path: Path) -> None:
        """The update carries the text, the whole metadata with a fresh block, the tags and the visibility."""
        path = _write(tmp_path, 'a.md', 'alpha')
        await _sync(server, tmp_path)
        (context_id,) = server.entries
        path.write_text('alpha, revised', encoding='utf-8')

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [call.arguments['atomic'] for call in server.calls_to(UPDATE)] == [False]
        (update,) = _sent(server, UPDATE, 'updates')
        assert set(update) == {'context_id', 'text', 'metadata', 'tags', 'visibility'}
        assert (update['context_id'], update['text'], update['visibility']) == (context_id, 'alpha, revised', 'private')
        assert ingest_block_error(_block(update)) is None
        assert server.entries[context_id].text == 'alpha, revised'
        assert [(item.source_id, item.context_id) for item in report.outcomes if item.outcome == 'updated'] == [
            ('a.md', context_id),
        ]

    @pytest.mark.asyncio
    async def test_a_metadata_change_sends_no_text(self, server: FakeServer, tmp_path: Path) -> None:
        """A tag change replaces the metadata, tags and visibility without regenerating the text."""
        _write(tmp_path, 'a.md', 'alpha')
        await _sync(server, tmp_path)

        report = await _sync(server, tmp_path, tags=('guide',))

        assert report.exit_code == EXIT_CONVERGED
        (update,) = _sent(server, UPDATE, 'updates')
        assert set(update) == {'context_id', 'metadata', 'tags', 'visibility'}
        assert update['tags'] == ['guide']

    @pytest.mark.asyncio
    async def test_a_text_and_metadata_change_sends_both(self, server: FakeServer, tmp_path: Path) -> None:
        """A change of both sends the text together with the full metadata."""
        path = _write(tmp_path, 'a.md', 'alpha')
        await _sync(server, tmp_path)
        path.write_text('alpha, revised', encoding='utf-8')

        report = await _sync(server, tmp_path, tags=('guide',))

        assert report.exit_code == EXIT_CONVERGED
        (update,) = _sent(server, UPDATE, 'updates')
        assert (update['text'], update['tags']) == ('alpha, revised', ['guide'])

    @pytest.mark.asyncio
    async def test_a_refused_update_is_a_failure(self, server: FakeServer, tmp_path: Path) -> None:
        """A per-entry not-authorized answer fails that document."""
        path = _write(tmp_path, 'a.md', 'alpha')
        await _sync(server, tmp_path)
        (context_id,) = server.entries
        server.failing_updates[context_id] = f'Not authorized to modify context entry {context_id}'
        path.write_text('alpha, revised', encoding='utf-8')

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert _failures(report) == [('a.md', f'Not authorized to modify context entry {context_id}')]


class TestDeletes:
    """Entries of removed documents and duplicates are deleted by id, after every create and update."""

    @pytest.mark.asyncio
    async def test_a_removed_document_is_deleted_by_id_after_the_updates(self, server: FakeServer, tmp_path: Path) -> None:
        """The delete names the entry's id alone and follows the update of the remaining document."""
        path = _write(tmp_path, 'a.md', 'alpha')
        removed = _write(tmp_path, 'b.md', 'beta')
        await _sync(server, tmp_path)
        (removed_id,) = [entry.context_id for entry in server.by_source(THREAD_ID)['b.md']]
        removed.unlink()
        path.write_text('alpha, revised', encoding='utf-8')
        server.calls.clear()

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        tools = [call.tool for call in server.calls]
        assert tools.index(UPDATE) < tools.index(DELETE)
        assert [call.arguments for call in server.calls_to(DELETE)] == [{'context_ids': [removed_id]}]
        assert [(item.source_id, item.context_id) for item in report.outcomes if item.outcome == 'deleted'] == [
            ('b.md', removed_id),
        ]
        assert sorted(server.by_source(THREAD_ID)) == ['a.md']

    @pytest.mark.asyncio
    async def test_deletes_repeat_until_the_plan_lists_none(self, server: FakeServer, tmp_path: Path) -> None:
        """A plan lists at most 100 deletes, so the leg deletes, plans again and deletes the rest."""
        _write(tmp_path, 'a.md', 'alpha')
        for number in range(150):
            server.seed(thread_id=THREAD_ID, text=f'gone {number}', metadata=_managed(f'gone/{number:03}.md'))

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [len(cast('list[str]', call.arguments['context_ids'])) for call in server.calls_to(DELETE)] == [100, 50]
        assert report.counts['deleted'] == 150
        assert sorted(server.by_source(THREAD_ID)) == ['a.md']

    @pytest.mark.asyncio
    async def test_each_range_is_planned_with_its_bounds_and_deletes_only_its_own_entries(
        self, server: FakeServer, tmp_path: Path,
    ) -> None:
        """A small request budget splits the manifest into ranges, and a removal is deleted once, by its range."""
        for name in 'abcdef':
            _write(tmp_path, f'{name}.md', name)
        await _sync(server, tmp_path, max_request_bytes=2600)
        (removed_id,) = [entry.context_id for entry in server.by_source(THREAD_ID)['c.md']]
        (tmp_path / 'c.md').unlink()
        server.calls.clear()

        report = await _sync(server, tmp_path, max_request_bytes=2600)

        assert report.exit_code == EXIT_CONVERGED
        bounds = [
            (call.arguments.get('source_id_after'), call.arguments.get('source_id_through')) for call in server.calls_to(PLAN)
        ]
        assert bounds[0][0] is None
        assert bounds[0][1] is not None
        assert any(after is not None and through is None for after, through in bounds)
        assert [call.arguments for call in server.calls_to(DELETE)] == [{'context_ids': [removed_id]}]
        assert sorted(server.by_source(THREAD_ID)) == ['a.md', 'b.md', 'd.md', 'e.md', 'f.md']

    @pytest.mark.asyncio
    async def test_the_delete_loop_stops_when_a_round_makes_no_progress(self, server: FakeServer, tmp_path: Path) -> None:
        """A delete that removes nothing is not repeated, and the run reports it as not converged."""
        _write(tmp_path, 'a.md', 'alpha')
        stuck = server.seed(thread_id=THREAD_ID, text='gone', metadata=_managed('gone.md'))
        server.undeletable.add(stuck)

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert [call.arguments for call in server.calls_to(DELETE)] == [{'context_ids': [stuck]}]
        assert any('no progress' in error for error in report.errors)
        assert any('did not converge' in error for error in report.errors)
        assert _outcomes(report, 'deleted') == []

    @pytest.mark.asyncio
    async def test_a_duplicate_is_deleted_and_the_newest_copy_survives(self, server: FakeServer, tmp_path: Path) -> None:
        """Two stored copies of one document converge on one: the plan's survivor."""
        _write(tmp_path, 'a.md', 'alpha')
        await _sync(server, tmp_path)
        (original,) = server.entries
        newer = server.duplicate(original)

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert [entry.context_id for entry in server.by_source(THREAD_ID)['a.md']] == [newer]
        assert [(item.source_id, item.context_id) for item in report.outcomes if item.outcome == 'deleted'] == [
            ('a.md', original),
        ]

    @pytest.mark.asyncio
    async def test_a_refused_delete_call_fails_its_entries(self, server: FakeServer, tmp_path: Path) -> None:
        """A delete the server refuses fails the entries it named and leaves them stored."""
        _write(tmp_path, 'a.md', 'alpha')
        removed = _write(tmp_path, 'b.md', 'beta')
        await _sync(server, tmp_path)
        removed.unlink()
        server.fail_next(DELETE, ToolFailure('Not authorized to delete context entries'))

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert _failures(report) == [('b.md', 'Not authorized to delete context entries')]
        assert sorted(server.by_source(THREAD_ID)) == ['a.md', 'b.md']


class TestUnusableFiles:
    """A file the run cannot use never turns into a deletion, and an over-long path is never sent."""

    @pytest.mark.asyncio
    async def test_an_unreadable_stored_document_is_kept(self, server: FakeServer, tmp_path: Path) -> None:
        """An undecodable file is a keep item: its stored entry is neither updated nor deleted."""
        _write(tmp_path, 'a.md', 'alpha')
        broken = _write(tmp_path, 'b.md', 'beta')
        await _sync(server, tmp_path)
        broken.write_bytes(b'\xff\xfe not utf-8 \x80')
        server.calls.clear()

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        manifests = [cast('list[Json]', call.arguments['manifest']) for call in server.calls_to(PLAN)]
        assert {'source_id': 'b.md', 'keep': True} in manifests[0]
        assert server.calls_to(UPDATE) == []
        assert server.calls_to(DELETE) == []
        assert [entry.text for entry in server.by_source(THREAD_ID)['b.md']] == ['beta']
        assert _outcomes(report, 'kept') == ['b.md']
        assert not any('did not converge' in error for error in report.errors)

    @pytest.mark.asyncio
    async def test_an_overlong_path_is_never_sent(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A source id above the bound is in no manifest and no batch, and the run names it."""
        path = _write(tmp_path, 'a.md', 'alpha')
        too_long = 'x' * (MAX_SOURCE_ID_LENGTH + 1)

        def walk(
            root: Path, *, includes: Sequence[str], excludes: Sequence[str],
        ) -> tuple[list[tuple[str, Path]], list[DirectoryFailure]]:
            del root, includes, excludes
            return [('a.md', path), (too_long, path)], []

        monkeypatch.setattr(corpus, 'walk', walk)

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        manifests = [cast('list[Json]', call.arguments['manifest']) for call in server.calls_to(PLAN)]
        assert {item['source_id'] for manifest in manifests for item in manifest} == {'a.md'}
        assert [_block(entry)['source_id'] for entry in _sent(server, STORE, 'entries')] == ['a.md']
        assert _outcomes(report, 'failed') == [too_long]
        assert any(too_long in error for error in report.errors)


class TestDirectoryFailures:
    """A directory the walk cannot list withholds every delete of the run, and nothing else."""

    @pytest.mark.asyncio
    async def test_creates_and_updates_run_and_every_delete_is_withheld(
        self, server: FakeServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run creates and updates, never calls the delete tool, counts the withheld deletes and names the directory."""
        top = _write(tmp_path, 'top.md', 'top')
        _write(tmp_path, 'locked/inner.md', 'inner')
        gone = _write(tmp_path, 'gone.md', 'gone')
        await _sync(server, tmp_path)
        gone.unlink()
        top.write_text('top, revised', encoding='utf-8')
        _write(tmp_path, 'new.md', 'new')
        _refuse_listing(monkeypatch, tmp_path / 'locked')

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_PARTIAL
        assert server.calls_to(DELETE) == []
        assert (report.counts['withheld'], report.counts['updated'], report.counts['created']) == (2, 1, 1)
        assert 'directory not listed: locked: Permission denied' in report.errors
        assert not any('did not converge' in error for error in report.errors)
        assert sorted(server.by_source(THREAD_ID)) == ['gone.md', 'locked/inner.md', 'new.md', 'top.md']

    @pytest.mark.asyncio
    async def test_a_dry_run_reports_the_deletes_as_withheld(
        self,
        server: FakeServer,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A dry run under a directory failure prints the planned deletes as withheld and writes nothing."""
        _write(tmp_path, 'top.md', 'top')
        _write(tmp_path, 'locked/inner.md', 'inner')
        await _sync(server, tmp_path)
        (inner_id,) = [entry.context_id for entry in server.by_source(THREAD_ID)['locked/inner.md']]
        _refuse_listing(monkeypatch, tmp_path / 'locked')
        capsys.readouterr()

        report = await _sync(server, tmp_path, dry_run=True)

        assert report.exit_code == EXIT_PARTIAL
        assert report.counts['withheld'] == 1
        assert _outcomes(report, 'deleted') == []
        assert f'withheld delete locked/inner.md {inner_id} (absent)' in capsys.readouterr().out.splitlines()
        assert inner_id in server.entries


class TestRunEnding:
    """A failure that ends the run stops it at once, and nothing it may have done is sent again."""

    @pytest.mark.asyncio
    async def test_a_timed_out_call_is_never_resent(self, server: FakeServer, tmp_path: Path) -> None:
        """A store whose answer never came ends the run; the next run finds what it committed and converges."""
        _write(tmp_path, 'a.md', 'alpha')
        _write(tmp_path, 'b.md', 'beta')
        timeout = IngestExitError(EXIT_TRANSIENT, 'the call to store_context_batch got no answer within --call-timeout')
        server.fail_next(STORE, timeout, commit=True)

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_TRANSIENT
        assert report.errors == [timeout.message]
        assert len(server.calls_to(STORE)) == 1
        assert server.calls[-1].tool == STORE
        server.calls.clear()

        rerun = await _sync(server, tmp_path)

        assert rerun.exit_code == EXIT_CONVERGED
        assert server.calls_to(STORE) == []
        assert {source: len(entries) for source, entries in server.by_source(THREAD_ID).items()} == {'a.md': 1, 'b.md': 1}

    @pytest.mark.asyncio
    async def test_a_refused_plan_exits_2_before_any_write(self, server: FakeServer, tmp_path: Path) -> None:
        """The plan tool's error is a plan error, reported, and nothing is written."""
        _write(tmp_path, 'a.md', 'alpha')
        refusal = "manifest repeats source_id values (each may appear once): 'a.md'"
        server.fail_next(PLAN, ToolFailure(refusal))

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any(refusal in error for error in report.errors)
        assert server.calls_to(STORE) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('answer', 'message'),
        [
            ({'thread_id': THREAD_ID, 'creates': []}, 'with an unexpected result'),
            (
                {
                    'thread_id': THREAD_ID, 'creates': [5], 'updates': [], 'deletes': [], 'deletes_total': 0,
                    'unchanged_count': 0, 'kept_count': 0, 'foreign_count': 0,
                },
                'a manifest index that names no document',
            ),
        ],
    )
    async def test_a_plan_of_another_shape_exits_2(
        self, server: FakeServer, tmp_path: Path, answer: Json, message: str,
    ) -> None:
        """A plan without the plan's fields, or naming an index past the manifest, ends the run before any write."""
        _write(tmp_path, 'a.md', 'alpha')
        server.answer_next(PLAN, answer)

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any(message in error for error in report.errors)
        assert server.calls_to(STORE) == []

    @pytest.mark.asyncio
    async def test_a_failed_preflight_ends_the_run_with_its_code(self, server: FakeServer, tmp_path: Path) -> None:
        """A server without the sync tools is refused before the first plan."""
        _write(tmp_path, 'a.md', 'alpha')
        server.preflight_exit = IngestExitError(EXIT_CONFIG, 'the server does not offer plan_context_sync')

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert report.errors == ['the server does not offer plan_context_sync']
        assert server.calls_to(PLAN) == []

    @pytest.mark.asyncio
    async def test_the_token_file_is_read_before_the_server_is_contacted(self, server: FakeServer, tmp_path: Path) -> None:
        """An empty token file exits 2 without a connection; a usable one becomes the session's auth."""
        corpus_root = tmp_path / 'corpus'
        _write(corpus_root, 'a.md', 'alpha')
        token = tmp_path / 'token'
        token.write_text('\n', encoding='utf-8')

        refused = await _sync(server, corpus_root, token_file=token)

        assert refused.exit_code == EXIT_CONFIG
        assert refused.errors == [f'the token file is empty: {token}']
        assert server.connections == []

        token.write_text('ingest-token\n', encoding='utf-8')
        accepted = await _sync(server, corpus_root, token_file=token)

        assert accepted.exit_code == EXIT_CONVERGED
        (auth,) = server.connections
        assert isinstance(auth, TokenFileAuth)
        assert auth.path == token

    @pytest.mark.asyncio
    async def test_a_request_budget_below_one_plan_call_exits_2(self, server: FakeServer, tmp_path: Path) -> None:
        """A --max-request-bytes too small for a plan request is a configuration error found before contact."""
        _write(tmp_path, 'a.md', 'alpha')

        report = await _sync(server, tmp_path, max_request_bytes=500)

        assert report.exit_code == EXIT_CONFIG
        assert any('--max-request-bytes' in error for error in report.errors)
        assert server.connections == []


class TestForeignEntries:
    """Another principal's members are counted and warned about, never changed."""

    @pytest.mark.asyncio
    async def test_foreign_members_warn_and_stay_untouched(
        self, server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The run builds the caller's own copy and leaves the other principal's public entry alone."""
        _write(tmp_path, 'a.md', 'alpha')
        foreign = server.seed(thread_id=THREAD_ID, text='theirs', metadata=_managed('a.md'), owner=OTHER, visibility='public')

        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONVERGED
        assert report.counts['foreign'] == 1
        assert server.entries[foreign].text == 'theirs'
        assert len(server.by_source(THREAD_ID)['a.md']) == 2
        assert 'another principal owns' in capsys.readouterr().err


class TestDryRun:
    """A dry run plans, prints the plan and writes nothing."""

    @pytest.mark.asyncio
    async def test_a_dry_run_prints_the_plan_and_writes_nothing(
        self, server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Creates, updates and deletes are printed with their ids, and only the plan tool is called."""
        edited = _write(tmp_path, 'a.md', 'alpha')
        removed = _write(tmp_path, 'b.md', 'beta')
        await _sync(server, tmp_path)
        ids = {source: entries[0].context_id for source, entries in server.by_source(THREAD_ID).items()}
        edited.write_text('alpha, revised', encoding='utf-8')
        removed.unlink()
        _write(tmp_path, 'c.md', 'gamma')
        server.calls.clear()
        capsys.readouterr()

        report = await _sync(server, tmp_path, dry_run=True)

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
        self, server: FakeServer, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A plan lists 100 deletes; the dry run prints them and how many more the plan holds."""
        _write(tmp_path, 'a.md', 'alpha')
        for number in range(105):
            server.seed(thread_id=THREAD_ID, text=f'gone {number}', metadata=_managed(f'gone/{number:03}.md'))

        report = await _sync(server, tmp_path, dry_run=True)

        printed = capsys.readouterr().out.splitlines()
        assert report.exit_code == EXIT_CONVERGED
        assert 'plan: 1 to create, 0 to update, 105 to delete, 0 unchanged, 0 kept, 0 foreign' in printed
        assert '... and 5 more planned deletes in the corpus' in printed
        assert sum(line.startswith('would delete gone/') for line in printed) == 100
        assert len(server.entries) == 105

    @pytest.mark.asyncio
    async def test_a_dry_run_with_a_kept_file_exits_1(self, server: FakeServer, tmp_path: Path) -> None:
        """A keep item makes even a dry run partial."""
        _write(tmp_path, 'a.md', 'alpha')
        (tmp_path / 'b.md').write_bytes(b'\xff not utf-8')

        report = await _sync(server, tmp_path, dry_run=True)

        assert report.exit_code == EXIT_PARTIAL
        assert _outcomes(report, 'kept') == ['b.md']
        assert server.entries == {}


class TestEmptyCorpus:
    """Deleting a whole corpus takes an explicit flag."""

    @pytest.mark.asyncio
    async def test_an_empty_walk_exits_2_without_contact(self, server: FakeServer, tmp_path: Path) -> None:
        """A directory without documents is refused before the server is contacted."""
        report = await _sync(server, tmp_path)

        assert report.exit_code == EXIT_CONFIG
        assert any('--allow-empty-corpus' in error for error in report.errors)
        assert server.connections == []

    @pytest.mark.asyncio
    async def test_the_flag_deletes_every_managed_entry(self, server: FakeServer, tmp_path: Path) -> None:
        """With the flag the empty manifest is confirmed and every entry of the corpus is deleted."""
        first = _write(tmp_path, 'a.md', 'alpha')
        second = _write(tmp_path, 'b.md', 'beta')
        await _sync(server, tmp_path)
        first.unlink()
        second.unlink()
        server.calls.clear()

        report = await _sync(server, tmp_path, allow_empty_corpus=True)

        assert report.exit_code == EXIT_CONVERGED
        plans = server.calls_to(PLAN)
        assert all(call.arguments['manifest'] == [] for call in plans)
        assert all(call.arguments['allow_empty_manifest'] is True for call in plans)
        assert server.by_source(THREAD_ID) == {}
        assert report.counts['deleted'] == 2
