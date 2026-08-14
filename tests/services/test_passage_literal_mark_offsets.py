"""Highlight positions stay correct when the document itself contains markup.

Neither SQLite highlight() nor PostgreSQL ts_headline() escapes markup already
present in the stored text, so a literal '<mark>' in a document reaches the
highlight output looking exactly like an inserted marker. Counting it as one
subtracts a phantom tag offset and shifts every later region, which pointed the
reranker's passage at unrelated text and silently degraded result order.
"""

from app.services.passage_extraction_service import HighlightRegion
from app.services.passage_extraction_service import extract_rerank_passage
from app.services.passage_extraction_service import parse_highlight_positions

LITERAL_MARK_TEXT = (
    'Use the HTML tag <mark>highlighted</mark> to emphasize text. '
    'The rendering engine processes documents efficiently and quickly.'
)


def _highlight(text: str, *terms: str) -> str:
    """Wrap each term in the text with the markers an FTS engine would insert.

    Args:
        text: The original document text.
        terms: The matched terms to wrap.

    Returns:
        The document with markers inserted around every occurrence of each term.
    """
    marked = text
    for term in terms:
        marked = marked.replace(term, f'<mark>{term}</mark>')
    return marked


class TestLiteralMarkupInSource:
    """Regions must locate the matched terms, not markup already in the text."""

    def test_regions_point_at_the_matched_terms(self) -> None:
        highlighted = _highlight(LITERAL_MARK_TEXT, 'rendering', 'engine')

        regions = parse_highlight_positions(highlighted, LITERAL_MARK_TEXT)

        assert [LITERAL_MARK_TEXT[region.start:region.end] for region in regions] == [
            'rendering',
            'engine',
        ]

    def test_extracted_passage_contains_the_matched_terms(self) -> None:
        """Literal markup ahead of the match must not drag the passage window away.

        Every literal tag pair the source carries is worth thirteen characters of
        drift, so a document with a dozen of them ahead of the real match moves the
        extracted window into unrelated text entirely.
        """
        legend = 'Legend: ' + ' '.join(f'<mark>item{index}</mark>' for index in range(12)) + '. '
        filler = 'Filler sentence about unrelated storage topics. ' * 6
        text = f'{legend}{filler}Finally the rendering engine processes documents.'
        highlighted = _highlight(text, 'rendering engine')

        passage = extract_rerank_passage(
            text_content=text,
            highlighted=highlighted,
            window_size=40,
            max_passage_size=300,
            gap_merge_threshold=20,
        )

        assert 'rendering engine' in passage

    def test_literal_markup_before_and_after_a_match(self) -> None:
        text = 'Opening <mark>tag, the rendering engine, closing </mark>tag.'
        highlighted = _highlight(text, 'rendering')

        regions = parse_highlight_positions(highlighted, text)

        assert [text[region.start:region.end] for region in regions] == ['rendering']

    def test_several_literal_tags_do_not_accumulate_drift(self) -> None:
        text = (
            '<mark>one</mark> <mark>two</mark> <mark>three</mark> '
            'and then the searchable keyword appears here.'
        )
        highlighted = _highlight(text, 'keyword')

        regions = parse_highlight_positions(highlighted, text)

        assert [text[region.start:region.end] for region in regions] == ['keyword']

    def test_unrelated_highlight_yields_no_positions(self) -> None:
        """A highlight that cannot be reconciled with the text produces nothing.

        Guessing positions from an unreconcilable highlight would point the
        passage at arbitrary text, which is worse than falling back to the
        document beginning.
        """
        text = 'A document carrying a literal <mark>tag</mark> inside it.'

        assert parse_highlight_positions('<mark>totally</mark> different', text) == []


class TestPlainSourceUnchanged:
    """Documents without literal markup keep the cheap, exact mapping."""

    def test_positions_match_with_and_without_the_source_text(self) -> None:
        text = 'Alpha beta gamma. The rendering engine is fast and small.'
        highlighted = _highlight(text, 'rendering', 'engine')

        with_source = parse_highlight_positions(highlighted, text)
        without_source = parse_highlight_positions(highlighted)

        assert with_source == without_source
        assert with_source == [
            HighlightRegion(start=text.index('rendering'), end=text.index('rendering') + len('rendering')),
            HighlightRegion(start=text.index('engine'), end=text.index('engine') + len('engine')),
        ]

    def test_no_marks_yields_no_regions(self) -> None:
        text = 'Nothing matched in this document at all.'

        assert parse_highlight_positions(text, text) == []
