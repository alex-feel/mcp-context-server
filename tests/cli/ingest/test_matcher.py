"""Tests for the include and exclude matcher of the corpus walk.

The matcher compares glob patterns with root-relative POSIX paths case-sensitively
on every host, so a Windows host and the Linux image select the same files.
"""

from pathlib import Path

import pytest

from app.cli.ingest.corpus import matches
from app.cli.ingest.corpus import walk


class TestDoubleStar:
    """A whole-segment ``**`` matches any number of path segments."""

    @pytest.mark.parametrize('path', ['a.md', 'x/a.md', 'x/y/z/a.md'])
    def test_leading_double_star_matches_zero_or_more_segments(self, path: str) -> None:
        """``**/`` also matches zero segments, so a root-level file is selected."""
        assert matches('**/*.md', path)

    @pytest.mark.parametrize('path', ['docs/a.md', 'docs/x/a.md', 'docs/x/y/a.md'])
    def test_inner_double_star_matches_zero_or_more_segments(self, path: str) -> None:
        """``docs/**/*.md`` selects Markdown files at every depth under docs."""
        assert matches('docs/**/*.md', path)

    @pytest.mark.parametrize('path', ['docs/a', 'docs/x/y/z.txt'])
    def test_trailing_double_star_matches_everything_below(self, path: str) -> None:
        """``docs/**`` matches every file below the docs directory."""
        assert matches('docs/**', path)

    @pytest.mark.parametrize('path', ['a.md', 'other/docs/a.md', 'docsx/a.md'])
    def test_trailing_double_star_stays_under_its_directory(self, path: str) -> None:
        """``docs/**`` matches nothing outside the docs directory."""
        assert not matches('docs/**', path)

    def test_double_star_inside_a_segment_is_a_single_star(self) -> None:
        """Consecutive stars that do not form a whole segment never cross a slash."""
        assert matches('a**b.md', 'axyb.md')
        assert not matches('a**b.md', 'ax/yb.md')


class TestSegmentWildcards:
    """``*``, ``?`` and character classes never cross a slash."""

    def test_star_stays_inside_one_segment(self) -> None:
        """``*.md`` selects root-level files only."""
        assert matches('*.md', 'a.md')
        assert not matches('*.md', 'x/a.md')

    def test_star_matches_an_empty_run(self) -> None:
        """``*`` matches zero characters."""
        assert matches('a*.md', 'a.md')

    def test_question_mark_matches_one_character(self) -> None:
        """``?`` matches exactly one character other than a slash."""
        assert matches('?.md', 'a.md')
        assert not matches('?.md', 'ab.md')
        assert not matches('a?b', 'a/b')

    @pytest.mark.parametrize(
        ('pattern', 'path', 'expected'),
        [
            ('[abc].md', 'b.md', True),
            ('[abc].md', 'd.md', False),
            ('[a-c].md', 'c.md', True),
            ('[a-c].md', 'C.md', False),
            ('[!a].md', 'b.md', True),
            ('[!a].md', 'a.md', False),
            ('[]].md', '].md', True),
            ('[!]].md', 'x.md', True),
            ('[!]].md', '].md', False),
            ('a[!b]c', 'a/c', False),
            ('[^a].md', '^.md', True),
            ('[^a].md', 'b.md', False),
        ],
    )
    def test_character_classes(self, pattern: str, path: str, expected: bool) -> None:
        """Classes follow fnmatch syntax: ranges, ``!`` negation, a leading ``]`` taken literally."""
        assert matches(pattern, path) is expected

    def test_unterminated_class_is_literal(self) -> None:
        """An opening bracket without a closing one matches itself."""
        assert matches('a[b', 'a[b')
        assert not matches('a[b', 'ab')

    def test_invalid_class_range_is_refused(self) -> None:
        """A class whose range runs backwards cannot be compiled."""
        with pytest.raises(ValueError, match=r'z-a'):
            matches('[z-a].md', 'a.md')


class TestLiteralCharacters:
    """Everything that is not a wildcard matches itself exactly."""

    def test_regex_metacharacters_are_literal(self) -> None:
        """Dots, plus signs and parentheses carry no regular-expression meaning."""
        assert matches('a+(b).md', 'a+(b).md')
        assert not matches('a.md', 'axmd')

    def test_whole_path_must_match(self) -> None:
        """A pattern matching only a prefix or a suffix of the path does not select it."""
        assert not matches('docs', 'docs/a.md')
        assert not matches('a.md', 'x/a.md')

    @pytest.mark.parametrize(
        ('pattern', 'path'),
        [('*.md', 'README.MD'), ('README.md', 'readme.md'), ('Docs/*.md', 'docs/a.md'), ('docs/**', 'Docs/a.md')],
    )
    def test_matching_is_case_sensitive(self, pattern: str, path: str) -> None:
        """Case is significant on every host, Windows included."""
        assert not matches(pattern, path)


class TestWalkSelection:
    """The walk selects a file when an include matches and no exclude matches."""

    @staticmethod
    def _tree(root: Path) -> None:
        """Create a tree whose names differ from the patterns only by case.

        Only one spelling of each name exists per directory, so the tree can be
        created on a case-insensitive file system too.
        """
        for relative in (
            'README.MD',
            'sub/readme.md',
            'Guides/Setup.md',
            'a/drafts/hidden.md',
            'b/Drafts/shown.md',
            'notes.txt',
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('# Title\n\nbody\n', encoding='utf-8')

    def test_same_selection_on_every_host(self, tmp_path: Path) -> None:
        """An upper-case extension is not ``*.md``, and a lower-case exclude spares ``Drafts``."""
        self._tree(tmp_path)

        files, failures = walk(tmp_path, includes=['**/*.md'], excludes=['*/drafts/**'])

        assert failures == []
        assert [source_id for source_id, _ in files] == ['Guides/Setup.md', 'b/Drafts/shown.md', 'sub/readme.md']

    def test_upper_case_include_selects_only_its_spelling(self, tmp_path: Path) -> None:
        """``**/*.MD`` selects the upper-case file alone, wherever the host folds case."""
        self._tree(tmp_path)

        files, _ = walk(tmp_path, includes=['**/*.MD'], excludes=[])

        assert [source_id for source_id, _ in files] == ['README.MD']

    def test_any_include_selects(self, tmp_path: Path) -> None:
        """A file is selected when any of several includes matches it."""
        self._tree(tmp_path)

        files, _ = walk(tmp_path, includes=['*.txt', 'Guides/*.md'], excludes=[])

        assert [source_id for source_id, _ in files] == ['Guides/Setup.md', 'notes.txt']

    def test_any_exclude_removes(self, tmp_path: Path) -> None:
        """A file matched by any exclude is dropped even when an include selects it."""
        self._tree(tmp_path)

        files, _ = walk(tmp_path, includes=['**/*.md'], excludes=['sub/**', 'Guides/*'])

        assert [source_id for source_id, _ in files] == ['a/drafts/hidden.md', 'b/Drafts/shown.md']

    def test_paths_point_at_the_files(self, tmp_path: Path) -> None:
        """Every selected source id is paired with the file it names."""
        self._tree(tmp_path)

        files, _ = walk(tmp_path, includes=['**/*.md'], excludes=[])

        for source_id, path in files:
            assert path == tmp_path.joinpath(*source_id.split('/'))
