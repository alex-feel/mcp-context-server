"""Tool-level tests for grep_context and read_context_range (SQLite backend).

Exercises the public tool surface: grep output modes, Unicode-aware
case-insensitivity, thread scoping, bounded output, regex handling, the
mandatory clamp+echo of read_context_range, and the locate->extract composition
(grep match offsets feeding read_context_range).
"""

import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import cast

import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

import app.startup
from app.backends import StorageBackend
from app.backends import create_backend
from app.ids import generate_id_with_timestamp
from app.migrations.index_tree import apply_index_tree_migration
from app.repositories import RepositoryContainer
from app.repositories.index_node_repository import IndexNodeRow
from app.services.grep_service import GrepEntryResult
from app.startup import ensure_repositories
from app.tools.navigation import grep_context
from app.tools.navigation import navigate_context
from app.tools.navigation import read_context_range


@pytest_asyncio.fixture
async def nav_backend(tmp_path: Path) -> AsyncGenerator[StorageBackend, None]:
    """SQLite backend wired into app.startup so the tools resolve repositories."""
    from app.schemas import load_schema

    db_path = tmp_path / 'nav.db'
    conn = sqlite3.connect(str(db_path))
    conn.executescript(load_schema('sqlite'))
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    repos = RepositoryContainer(backend)
    app.startup.set_backend(backend)
    app.startup.set_repositories(repos)
    try:
        yield backend
    finally:
        await backend.shutdown()
        app.startup.set_backend(None)
        app.startup.set_repositories(None)


_SEQ = datetime(2024, 1, 1, tzinfo=UTC)


async def _store(backend: StorageBackend, text: str, *, thread_id: str = 't', offset_seconds: int = 0) -> str:
    """Insert one entry and return its canonical id."""
    cid = generate_id_with_timestamp(_SEQ + timedelta(seconds=offset_seconds))

    def _write(conn: sqlite3.Connection) -> None:
        conn.execute(
            'INSERT INTO context_entries (id, thread_id, source, content_type, text_content) VALUES (?, ?, ?, ?, ?)',
            (cid, thread_id, 'agent', 'text', text),
        )

    await backend.execute_write(_write)
    return cid


async def _grep(**kwargs: Any) -> dict[str, Any]:
    """Call grep_context, returning a plain dict for ergonomic test assertions."""
    return cast(dict[str, Any], await grep_context(**kwargs))


async def _navigate(**kwargs: Any) -> dict[str, Any]:
    """Call navigate_context, returning a plain dict for ergonomic test assertions."""
    return cast(dict[str, Any], await navigate_context(**kwargs))


class TestGrepContextModes:
    """Output modes and their row shapes."""

    @pytest.mark.asyncio
    async def test_files_with_matches_default(self, nav_backend: StorageBackend) -> None:
        hit = await _store(nav_backend, 'the needle is here', offset_seconds=1)
        await _store(nav_backend, 'only hay', offset_seconds=2)
        result = await _grep(pattern='needle', thread_id='t')
        assert result['mode'] == 'files_with_matches'
        ids = {row['context_id'] for row in result['results']}
        assert ids == {hit}

    @pytest.mark.asyncio
    async def test_content_mode_returns_offsets_and_context(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'first line\nsecond needle line\nthird line')
        result = await _grep(pattern='needle', thread_id='t', output_mode='content', context_lines=1)
        assert result['total_matches'] == 1
        match = result['results'][0]
        assert match['line_number'] == 2
        assert match['line'] == 'second needle line'
        assert match['before'] == ['first line']
        assert match['after'] == ['third line']
        assert match['match_end'] > match['match_start']

    @pytest.mark.asyncio
    async def test_count_mode(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'x x x on one line\nand x again')
        result = await _grep(pattern='x', thread_id='t', output_mode='count', case_sensitive=True)
        assert result['results'][0]['count'] == 4


class TestGrepContextMatching:
    """Matching semantics: case, literal vs regex, Unicode, scoping."""

    @pytest.mark.asyncio
    async def test_case_insensitive_default(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'Contains ERROR token')
        result = await _grep(pattern='error', thread_id='t')
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_case_sensitive_excludes(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'Contains ERROR token')
        result = await _grep(pattern='error', thread_id='t', case_sensitive=True)
        assert result['results'] == []

    @pytest.mark.asyncio
    async def test_cyrillic_case_insensitive(self, nav_backend: StorageBackend) -> None:
        # Stored lower-case Cyrillic; query upper-case. Requires Python re
        # IGNORECASE (the ASCII-only SQL pre-narrow is skipped for non-ASCII).
        lower = ''.join(chr(c) for c in (0x043F, 0x0440, 0x0438, 0x0432, 0x0435, 0x0442))
        upper = lower.upper()
        await _store(nav_backend, f'message: {lower}')
        result = await _grep(pattern=upper, thread_id='t')
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_literal_does_not_treat_dot_as_wildcard(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'value axb here')
        result = await _grep(pattern='a.b', thread_id='t')
        assert result['results'] == []

    @pytest.mark.asyncio
    async def test_regex_mode(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'value axb here')
        result = await _grep(pattern='a.b', thread_id='t', is_regex=True)
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_invalid_regex_raises(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'anything')
        with pytest.raises(ToolError):
            await _grep(pattern='a(b', thread_id='t', is_regex=True)

    @pytest.mark.asyncio
    async def test_thread_scoping(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'needle in t', thread_id='t')
        await _store(nav_backend, 'needle in other', thread_id='other')
        result = await _grep(pattern='needle', thread_id='t')
        assert len(result['results']) == 1

    @pytest.mark.asyncio
    async def test_max_matches_truncates(self, nav_backend: StorageBackend) -> None:
        await _store(nav_backend, 'm\nm\nm\nm\nm')
        result = await _grep(pattern='m', thread_id='t', output_mode='content', max_matches=2, case_sensitive=True)
        assert result['total_matches'] == 2
        assert result['truncated'] is True

    @pytest.mark.asyncio
    async def test_no_spurious_truncation_at_exact_cap(self, nav_backend: StorageBackend) -> None:
        # An entry with exactly max_matches matches, then a trailing (older,
        # visited-after) candidate with ZERO matches, must NOT report truncated --
        # a zero-match candidate is not a dropped match (regression for the
        # +1-overflow loop; the old top-of-loop guard flipped truncated here).
        await _store(nav_backend, 'no hits here', thread_id='t', offset_seconds=0)  # 0 matches, visited last
        await _store(nav_backend, 'm\nm', thread_id='t', offset_seconds=1)          # 2 matches, visited first
        result = await _grep(pattern='m', thread_id='t', output_mode='content', max_matches=2, case_sensitive=True)
        assert result['total_matches'] == 2
        assert result['truncated'] is False

    @pytest.mark.asyncio
    async def test_truncation_when_real_extra_match_exists(self, nav_backend: StorageBackend) -> None:
        # A genuine extra match beyond the cap (in a later entry) -> truncated True,
        # output trimmed back to exactly max_matches.
        await _store(nav_backend, 'm', thread_id='t', offset_seconds=0)     # 1 match, visited last
        await _store(nav_backend, 'm\nm', thread_id='t', offset_seconds=1)  # 2 matches, visited first
        result = await _grep(pattern='m', thread_id='t', output_mode='content', max_matches=2, case_sensitive=True)
        assert result['total_matches'] == 2
        assert result['truncated'] is True

    @pytest.mark.asyncio
    async def test_regex_timeout_surfaced_in_response(
        self, nav_backend: StorageBackend, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A regex-timed-out entry is skipped (match_count 0) but surfaced via
        # timed_out_context_ids so the caller knows the result is incomplete for it.
        cid = await _store(nav_backend, 'aaaa', thread_id='t')

        async def _timeout(context_id: str, *_args: Any, **_kwargs: Any) -> GrepEntryResult:
            return GrepEntryResult(context_id=context_id, matches=(), match_count=0, timed_out=True)

        monkeypatch.setattr('app.tools.navigation.match_entry', _timeout)
        result = await _grep(pattern='a+', thread_id='t', is_regex=True)
        assert result.get('timed_out_context_ids') == [cid]


class TestReadContextRange:
    """Partial reads by char/line range with mandatory clamp + echo."""

    @pytest.mark.asyncio
    async def test_char_range(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'hello world')
        result = await read_context_range(context_id=cid, start_char=6, end_char=11)
        assert result['text'] == 'world'
        assert result['start_char'] == 6
        assert result['end_char'] == 11

    @pytest.mark.asyncio
    async def test_line_range(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'line1\nline2\nline3')
        result = await read_context_range(context_id=cid, start_line=2, end_line=2)
        assert result['text'] == 'line2'
        assert result['start_line'] == 2
        assert result['end_line'] == 2

    @pytest.mark.asyncio
    async def test_out_of_range_is_clamped_and_echoed(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'short')
        result = await read_context_range(context_id=cid, start_char=2, end_char=9999)
        assert result['text'] == 'ort'
        assert result['end_char'] == 5  # clamped to len('short')

    @pytest.mark.asyncio
    async def test_both_modes_rejected(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'data')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid, start_char=0, start_line=1)

    @pytest.mark.asyncio
    async def test_no_mode_rejected(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'data')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid)

    @pytest.mark.asyncio
    async def test_missing_entry_raises(self, nav_backend: StorageBackend) -> None:
        assert nav_backend is not None  # fixture provides the wired repositories
        with pytest.raises(ToolError):
            await read_context_range(context_id='0' * 32, start_char=0, end_char=1)


class TestLocateThenExtract:
    """grep content-mode offsets feed read_context_range to extract the hit."""

    @pytest.mark.asyncio
    async def test_grep_offsets_read_back_the_match(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'prefix text then TARGET then suffix')
        grep_result = await _grep(pattern='TARGET', thread_id='t', output_mode='content', case_sensitive=True)
        match = grep_result['results'][0]
        assert match['context_id'] == cid
        read_result = await read_context_range(
            context_id=match['context_id'],
            start_char=match['match_start'],
            end_char=match['match_end'],
        )
        assert read_result['text'] == 'TARGET'


class TestNavigateContext:
    """navigate_context builds the on-demand Markdown outline."""

    @pytest.mark.asyncio
    async def test_outline_tree_and_summary_root(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, '# Intro\nhello\n## Details\nmore\n')
        result = await _navigate(context_id=cid)
        assert result['context_id'] == cid
        assert result['node_count'] == 2
        root = result['root']
        assert root['node_id'] == 'root'
        # No stored summary on this short entry -> root summary mirrors None.
        assert root['summary'] is None
        intro = root['children'][0]
        assert intro['title'] == 'Intro'
        assert intro['children'][0]['title'] == 'Details'

    @pytest.mark.asyncio
    async def test_headingless_entry_yields_childless_root(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'plain text with no headings at all')
        result = await _navigate(context_id=cid)
        assert result['node_count'] == 0
        assert result['root']['children'] == []

    @pytest.mark.asyncio
    async def test_missing_entry_raises(self, nav_backend: StorageBackend) -> None:
        assert nav_backend is not None
        with pytest.raises(ToolError):
            await _navigate(context_id='0' * 32)


class TestNavigateNodeSummaries:
    """include_node_summaries surfaces stored per-node summaries onto descendants."""

    @pytest.mark.asyncio
    async def test_stored_node_summaries_surface(self, nav_backend: StorageBackend) -> None:
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        cid = await _store(nav_backend, '# Intro\nhello\n## Details\nmore body here\n')

        baseline = await _navigate(context_id=cid)
        details = baseline['root']['children'][0]['children'][0]
        assert details['node_id'] == 'intro/details'

        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id='intro/details', level=2, ordinal=1, title='Details',
                node_summary='Covers the details.',
                char_start=details['char_start'], char_end=details['char_end'],
            ),
        ])

        enriched = await _navigate(context_id=cid, include_node_summaries=True)
        assert enriched['root']['children'][0]['children'][0]['summary'] == 'Covers the details.'

    @pytest.mark.asyncio
    async def test_summaries_omitted_by_default(self, nav_backend: StorageBackend) -> None:
        await apply_index_tree_migration(nav_backend, force=True)
        repos = await ensure_repositories()
        cid = await _store(nav_backend, '# Intro\nhello\n## Details\nmore body here\n')
        await repos.index_nodes.replace_nodes_for_context(cid, [
            IndexNodeRow(
                node_id='intro/details', level=2, ordinal=1, title='Details',
                node_summary='Covers the details.', char_start=0, char_end=10,
            ),
        ])
        # Default include_node_summaries=False: descendant summaries stay None.
        result = await _navigate(context_id=cid)
        assert result['root']['children'][0]['children'][0]['summary'] is None


class TestReadByNodeId:
    """read_context_range resolves a navigate_context node_id to its section."""

    @pytest.mark.asyncio
    async def test_navigate_then_read_node(self, nav_backend: StorageBackend) -> None:
        text = '# Intro\nintro body\n## Details\ndetail body here\n'
        cid = await _store(nav_backend, text)
        nav = await _navigate(context_id=cid)
        details = nav['root']['children'][0]['children'][0]
        assert details['node_id'] == 'intro/details'
        read_result = await read_context_range(context_id=cid, node_id='intro/details')
        # The Details section spans from its heading to end of document.
        assert read_result['text'].startswith('## Details')
        assert 'detail body here' in read_result['text']

    @pytest.mark.asyncio
    async def test_unknown_node_id_raises(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, '# Only\nbody')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid, node_id='nope/missing')

    @pytest.mark.asyncio
    async def test_node_id_and_char_range_mutually_exclusive(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, '# Only\nbody')
        with pytest.raises(ToolError):
            await read_context_range(context_id=cid, node_id='only', start_char=0)


class TestReadLineRangeEcho:
    """read_context_range echoes the resolved line range faithfully."""

    @pytest.mark.asyncio
    async def test_line_range_through_trailing_empty_line(self, nav_backend: StorageBackend) -> None:
        # Text ends with a newline, so the split yields a trailing empty line 3.
        # Requesting lines 1..3 must echo end_line=3 (not 2): recomputing the echo
        # from the exclusive end offset previously mapped it back to the prior line.
        cid = await _store(nav_backend, 'a\nb\n')
        result = await read_context_range(context_id=cid, start_line=1, end_line=3)
        assert result['start_line'] == 1
        assert result['end_line'] == 3
        assert result['text'] == 'a\nb\n'

    @pytest.mark.asyncio
    async def test_line_range_clamps_beyond_eof(self, nav_backend: StorageBackend) -> None:
        cid = await _store(nav_backend, 'one\ntwo\nthree')
        result = await read_context_range(context_id=cid, start_line=2, end_line=99)
        assert result['start_line'] == 2
        assert result['end_line'] == 3
        assert result['text'] == 'two\nthree'


class TestGrepCaseFoldParity:
    """The ASCII pre-narrow must not drop non-ASCII case-folds on the default
    case-insensitive path."""

    @pytest.mark.asyncio
    async def test_case_insensitive_ascii_letter_matches_unicode_fold(self, nav_backend: StorageBackend) -> None:
        # The stored text's only 's'-like code point is U+017F (LONG S), which
        # Python re.IGNORECASE folds to 's'. A case-insensitive grep for 's' must
        # still find it: the ASCII LIKE pre-narrow is skipped for ASCII letters
        # under case-insensitive matching, falling back to a full Python scan.
        # Before the fix the SQL LIKE '%s%' pre-narrow excluded this row (0 hits).
        await _store(nav_backend, 'meaſure')
        result = await _grep(pattern='s', thread_id='t', output_mode='content')
        assert result['total_matches'] == 1
