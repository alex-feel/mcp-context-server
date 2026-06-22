"""Unit tests for the code-derived Markdown outline parser.

Covers ATX/setext heading detection, fenced-code awareness (including the
unclosed-fence-at-EOF and #-inside-fence regressions), the synthetic root for
heading-light documents, sibling-ordinal node-id disambiguation, max_depth
folding, code-point offsets, and node-span resolution.
"""

from app.services.outline_service import ROOT_NODE_ID
from app.services.outline_service import OutlineNode
from app.services.outline_service import count_nodes
from app.services.outline_service import parse_outline
from app.services.outline_service import resolve_node_span
from app.services.outline_service import slugify


class TestSlugify:
    def test_basic(self) -> None:
        assert slugify('Set Up The Thing') == 'set-up-the-thing'

    def test_punctuation_collapses(self) -> None:
        assert slugify('  Hello, World!!  ') == 'hello-world'

    def test_empty_falls_back(self) -> None:
        assert slugify('***') == 'section'


class TestParseOutlineStructure:
    def test_nested_atx_headings(self) -> None:
        text = '# A\n\nalpha\n\n## B\n\nbeta\n'
        root = parse_outline(text)
        assert root.node_id == ROOT_NODE_ID
        assert root.char_start == 0
        assert root.char_end == len(text)
        assert len(root.children) == 1
        node_a = root.children[0]
        assert node_a.title == 'A'
        assert node_a.node_id == 'a'
        assert node_a.level == 1
        assert text[node_a.char_start:node_a.char_start + 3] == '# A'
        assert len(node_a.children) == 1
        node_b = node_a.children[0]
        assert node_b.title == 'B'
        assert node_b.node_id == 'a/b'
        assert node_b.level == 2

    def test_section_end_is_next_same_or_shallower(self) -> None:
        text = '# One\nbody one\n# Two\nbody two\n'
        root = parse_outline(text)
        one = root.children[0]
        two = root.children[1]
        # One ends where Two begins.
        assert one.char_end == two.char_start
        assert text[two.char_start:two.char_start + 5] == '# Two'

    def test_no_headings_yields_childless_root(self) -> None:
        text = 'just a paragraph\nwith two lines'
        root = parse_outline(text)
        assert root.children == ()
        assert root.char_end == len(text)
        assert count_nodes(root) == 0

    def test_single_heading(self) -> None:
        text = '# Only\nbody'
        root = parse_outline(text)
        assert count_nodes(root) == 1
        assert root.children[0].node_id == 'only'


class TestDuplicateHeadings:
    def test_sibling_ordinal_disambiguation(self) -> None:
        text = '# Top\n## Notes\nx\n## Notes\ny\n'
        root = parse_outline(text)
        top = root.children[0]
        assert [c.node_id for c in top.children] == ['top/notes', 'top/notes-2']
        assert [c.ordinal for c in top.children] == [1, 2]

    def test_generated_suffix_does_not_collide_with_literal_slug(self) -> None:
        # Regression: the 2nd "Notes" generates 'notes-2'; a literal "Notes 2"
        # heading ALSO slugifies to 'notes-2'. node_ids MUST stay unique among
        # siblings so resolve_node_span / read_context_range maps each id to its
        # OWN section instead of silently reading the wrong one.
        text = '# Top\n## Notes\na\n## Notes\nb\n## Notes 2\nc\n'
        root = parse_outline(text)
        top = root.children[0]
        ids = [c.node_id for c in top.children]
        assert len(ids) == len(set(ids)), f'node_ids not unique: {ids}'
        assert ids == ['top/notes', 'top/notes-2', 'top/notes-2-2']
        # Each id resolves to a distinct span, and the literal "Notes 2" section
        # keeps its own body.
        spans = {nid: resolve_node_span(text, nid) for nid in ids}
        assert len(set(spans.values())) == 3
        third = top.children[2]
        assert third.title == 'Notes 2'
        assert 'c' in text[third.char_start:third.char_end]


class TestAtxClosingSequence:
    """ATX headings whose entire content is a closing '#' run are EMPTY (CommonMark)."""

    def test_closing_only_heading_has_empty_title(self) -> None:
        # '## ###' is an empty heading: '###' is a closing run directly following
        # the opening marker, so _ATX_TRAILING_RE (which requires leading
        # whitespace) cannot strip it -- the parser must still emit an empty title.
        assert parse_outline('## ###').children[0].title == ''

    def test_single_closing_hash_heading_is_empty(self) -> None:
        assert parse_outline('### #').children[0].title == ''

    def test_trailing_space_closing_run_is_empty(self) -> None:
        assert parse_outline('# ## ').children[0].title == ''

    def test_inner_hash_content_is_preserved(self) -> None:
        # '## # #' is NOT closing-only: only the final ' #' is a closing run, so
        # the content '#' survives (CommonMark) and must NOT be over-emptied.
        assert parse_outline('## # #').children[0].title == '#'

    def test_ordinary_trailing_hashes_still_stripped(self) -> None:
        # The decorative trailing-'#' strip (leading whitespace + run) is unchanged.
        assert parse_outline('# Title ###').children[0].title == 'Title'


class TestFencedCode:
    def test_hash_inside_closed_fence_ignored(self) -> None:
        text = '# Real\n\n```\n# not a heading\n```\n\n## After\n'
        root = parse_outline(text)
        titles = [c.title for c in root.children]
        # 'Real' plus its child 'After'; the fenced '# not a heading' is skipped.
        assert titles == ['Real']
        assert root.children[0].children[0].title == 'After'

    def test_unclosed_fence_does_not_swallow_following_headings(self) -> None:
        # The fence never closes; the lenient parser must still surface '# After'.
        text = '# Before\n\n```python\nsome code\n# After\nmore code\n'
        root = parse_outline(text)
        titles = [c.title for c in root.children]
        assert 'Before' in titles
        assert 'After' in titles

    def test_tilde_fence(self) -> None:
        text = '# Real\n~~~\n# fake\n~~~\n## After\n'
        root = parse_outline(text)
        assert root.children[0].children[0].title == 'After'


class TestSetext:
    def test_setext_h1_and_h2(self) -> None:
        text = 'Title\n=====\nbody\n\nSub\n---\nmore\n'
        root = parse_outline(text)
        assert root.children[0].title == 'Title'
        assert root.children[0].level == 1
        assert root.children[0].children[0].title == 'Sub'
        assert root.children[0].children[0].level == 2


class TestMaxDepth:
    def test_deeper_headings_folded(self) -> None:
        text = '# A\n## B\n### C\n'
        root = parse_outline(text, max_depth=2)
        node_a = root.children[0]
        node_b = node_a.children[0]
        assert node_b.title == 'B'
        # '### C' exceeds max_depth=2 and is not a node.
        assert node_b.children == ()


class TestNodeIdDepthInvariance:
    """Node ids are assigned over the full canonical heading set, so they are
    stable across ``max_depth`` -- a folded (lower-``max_depth``) view reuses the
    same ids the canonical depth produces. This guards the contract that lets
    per-node summaries (generated at the canonical depth) attach to the right
    sections and a node_id from ``navigate_context(max_depth<6)`` round-trip
    through ``read_context_range`` (which resolves at the canonical depth).
    """

    # Deeper duplicate-slug heading BEFORE the shallower same-slug siblings: the
    # regression structure where folding '### Notes' out used to shift the
    # surviving '## Notes' ids from notes-2/notes-3 down to notes/notes-2.
    _TEXT = '# Top\n### Notes\n## Notes\n## Notes'

    @staticmethod
    def _ids_by_span(root: OutlineNode) -> dict[tuple[int, int], str]:
        spans: dict[tuple[int, int], str] = {}
        stack = list(root.children)
        while stack:
            node = stack.pop()
            spans[(node.char_start, node.char_end)] = node.node_id
            stack.extend(node.children)
        return spans

    def test_node_ids_stable_across_max_depth(self) -> None:
        deep = self._ids_by_span(parse_outline(self._TEXT, max_depth=6))
        shallow = self._ids_by_span(parse_outline(self._TEXT, max_depth=2))
        # Every section present at BOTH depths keeps the same node_id.
        shared = set(deep) & set(shallow)
        assert shared  # the two L2 "## Notes" sections survive the fold
        for span in shared:
            assert deep[span] == shallow[span]
        # Concretely, the [16:25] section is notes-2 at BOTH depths -- before the
        # fix the depth-2 parse shifted it to 'top/notes'.
        assert shallow[(16, 25)] == 'top/notes-2'

    def test_folded_node_id_round_trips_through_resolve(self) -> None:
        # A node_id surfaced by a max_depth=2 parse resolves (at the canonical
        # default depth, exactly as read_context_range does) to the SAME span it
        # showed -- not a different section.
        shallow_root = parse_outline(self._TEXT, max_depth=2)
        stack = list(shallow_root.children)
        checked = 0
        while stack:
            node = stack.pop()
            assert resolve_node_span(self._TEXT, node.node_id) == (node.char_start, node.char_end)
            checked += 1
            stack.extend(node.children)
        assert checked >= 2  # the surviving L2 sections were actually exercised


class TestCodePointOffsets:
    def test_offset_after_multibyte_prefix(self) -> None:
        prefix = chr(0x0451) * 4  # four two-byte Cyrillic letters, one line
        text = f'{prefix}\n# Heading\nbody'
        root = parse_outline(text)
        heading = root.children[0]
        # char_start counts code points: 4 (prefix) + 1 (newline) = 5.
        assert heading.char_start == 5
        assert text[heading.char_start:heading.char_start + 9] == '# Heading'


class TestResolveNodeSpan:
    def test_resolves_known_node(self) -> None:
        text = '# A\nbody\n## B\nmore\n'
        root = parse_outline(text)
        node_b = root.children[0].children[0]
        span = resolve_node_span(text, 'a/b')
        assert span == (node_b.char_start, node_b.char_end)

    def test_root_resolves_to_whole_document(self) -> None:
        text = '# A\nbody'
        assert resolve_node_span(text, ROOT_NODE_ID) == (0, len(text))

    def test_unknown_node_returns_none(self) -> None:
        text = '# A\nbody'
        assert resolve_node_span(text, 'does/not/exist') is None
