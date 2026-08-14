"""
Passage extraction for FTS reranking.

This module extracts coherent text passages from FTS highlight results
for use in cross-encoder reranking. It handles:
- Multiple disjoint matches in a document
- Expansion to sentence/paragraph boundaries
- Merging of nearby match regions
- Consistent behavior across SQLite and PostgreSQL

The algorithm uses the same separators as RecursiveCharacterTextSplitter
for natural language boundary detection.
"""


import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Separators in priority order (from RecursiveCharacterTextSplitter)
# Prioritizes: paragraph > line > sentence > word
DEFAULT_SEPARATORS: list[str] = ['\n\n', '\n', '. ', ' ']

# Regex for parsing <mark> tags
MARK_PATTERN = re.compile(r'<mark>(.*?)</mark>', re.DOTALL)
MARK_OPEN = '<mark>'
MARK_CLOSE = '</mark>'
MARK_OPEN_LEN = len(MARK_OPEN)
MARK_CLOSE_LEN = len(MARK_CLOSE)


@dataclass(frozen=True, slots=True)
class HighlightRegion:
    """A highlighted region in original text coordinates.

    Attributes:
        start: Character offset where the highlighted term starts in original text.
        end: Character offset where the highlighted term ends in original text.
    """

    start: int
    end: int


def _tag_offset_regions(highlighted: str) -> list[HighlightRegion]:
    """Map every <mark> pair to original coordinates by cumulative tag offset.

    Correct exactly when every tag in ``highlighted`` was inserted by the search
    engine, which holds whenever the source text carries no literal markup of its
    own -- both SQLite ``highlight()`` and PostgreSQL ``ts_headline()`` return the
    document verbatim with markers added.

    Args:
        highlighted: Text with <mark>term</mark> tags from FTS highlight.

    Returns:
        List of HighlightRegion with start/end positions in ORIGINAL text
        (without tags). Sorted by start position.
    """
    regions: list[HighlightRegion] = []
    tag_offset = 0  # Cumulative offset from tags seen so far

    for match in MARK_PATTERN.finditer(highlighted):
        # Position in highlighted string
        highlighted_start = match.start()

        # Calculate position in original text
        # Before this match, we've seen `tag_offset` chars of tags
        original_start = highlighted_start - tag_offset

        # After opening tag
        tag_offset += MARK_OPEN_LEN

        # The matched content length
        content_len = len(match.group(1))
        original_end = original_start + content_len

        # After closing tag
        tag_offset += MARK_CLOSE_LEN

        regions.append(HighlightRegion(start=original_start, end=original_end))

    return regions


def _aligned_regions(highlighted: str, text_content: str) -> list[HighlightRegion] | None:
    """Map <mark> pairs to original coordinates by aligning against the source text.

    Walks the highlighted string and the original text together. A tag occurrence
    that the original does NOT also carry at the same point is one the search engine
    inserted, and only those shift coordinates; a tag the original does carry is
    ordinary document text and is matched through like any other characters. The
    engine only ever opens a marker at the first character of a token, so an opening
    marker can never be confused with a literal '<mark>' in the source.

    Args:
        highlighted: Text with <mark>term</mark> tags from FTS highlight.
        text_content: The original document text the highlight was produced from.

    Returns:
        The highlighted regions in original coordinates, or None when the two
        strings cannot be reconciled (the highlight does not correspond to this
        text), in which case no trustworthy positions exist.
    """
    regions: list[HighlightRegion] = []
    open_starts: list[int] = []
    highlighted_pos = 0
    original_pos = 0
    highlighted_len = len(highlighted)
    original_len = len(text_content)

    while highlighted_pos < highlighted_len:
        opens_here = highlighted.startswith(MARK_OPEN, highlighted_pos)
        closes_here = highlighted.startswith(MARK_CLOSE, highlighted_pos)
        if opens_here and not text_content.startswith(MARK_OPEN, original_pos):
            open_starts.append(original_pos)
            highlighted_pos += MARK_OPEN_LEN
        elif (
            closes_here
            and open_starts
            and not text_content.startswith(MARK_CLOSE, original_pos)
        ):
            regions.append(HighlightRegion(start=open_starts.pop(), end=original_pos))
            highlighted_pos += MARK_CLOSE_LEN
        elif original_pos < original_len and highlighted[highlighted_pos] == text_content[original_pos]:
            highlighted_pos += 1
            original_pos += 1
        else:
            return None

    if original_pos != original_len or open_starts:
        return None

    regions.sort(key=lambda region: (region.start, region.end))
    return regions


def parse_highlight_positions(highlighted: str, text_content: str | None = None) -> list[HighlightRegion]:
    """Parse <mark> tags and return positions in original text coordinates.

    The highlighted text contains <mark> tags that shift positions, so recovering
    original coordinates means deciding which tags the search engine inserted.
    Neither SQLite ``highlight()`` nor PostgreSQL ``ts_headline()`` escapes markup
    already present in the document, so a literal '<mark>' stored in the text is
    indistinguishable from an inserted marker by shape alone -- counting every tag
    as inserted subtracts a phantom offset and shifts every later region, pointing
    the extracted passage at unrelated text. Passing ``text_content`` resolves that:
    the two strings are aligned, and only the tags the source does not itself carry
    are treated as markers.

    Args:
        highlighted: Text with <mark>term</mark> tags from FTS highlight.
        text_content: The original document text, when available. Without it the
            positions can only be derived by assuming every tag was inserted.

    Returns:
        List of HighlightRegion with start/end positions in ORIGINAL text
        (without tags). Sorted by start position. Empty when the highlight cannot
        be reconciled with the supplied text, since no position derived from it
        would be trustworthy.

    Example:
        >>> parse_highlight_positions('Hello <mark>world</mark>!')
        [HighlightRegion(start=6, end=11)]
    """
    if text_content is None or (MARK_OPEN not in text_content and MARK_CLOSE not in text_content):
        # No literal markup in the source: every tag in the highlight was inserted.
        return _tag_offset_regions(highlighted)

    aligned = _aligned_regions(highlighted, text_content)
    if aligned is None:
        logger.debug('Highlight could not be aligned with the source text; no positions derived')
        return []
    return aligned


def expand_to_boundary(
    text: str,
    pos: int,
    direction: str,
    separators: list[str],
    max_search: int = 200,
) -> int:
    """Expand position to nearest sentence/paragraph boundary.

    Uses RecursiveCharacterTextSplitter separator priority:
    1. '\\n\\n' - Paragraph break (highest priority)
    2. '\\n' - Line break
    3. '. ' - Sentence end
    4. ' ' - Word boundary (fallback)

    Args:
        text: The full document text.
        pos: Starting position to expand from.
        direction: 'left' to search backwards, 'right' to search forwards.
        separators: List of separators in priority order.
        max_search: Maximum characters to search (default: 200).

    Returns:
        Expanded position (moved to boundary if found, otherwise unchanged).

    Example:
        >>> expand_to_boundary('Hello world. This is text.', 14, 'left', ['. '])
        13  # Position after '. '
    """
    if direction == 'left':
        # Search backwards from pos
        search_start = max(0, pos - max_search)
        search_text = text[search_start:pos]

        for separator in separators:
            if not separator:
                continue  # Skip empty separator
            idx = search_text.rfind(separator)
            if idx != -1:
                # Found separator, return position AFTER it
                return search_start + idx + len(separator)

        # No separator found, return original
        return pos

    # direction == 'right'
    # Search forwards from pos
    search_end = min(len(text), pos + max_search)
    search_text = text[pos:search_end]

    for separator in separators:
        if not separator:
            continue
        idx = search_text.find(separator)
        if idx != -1:
            # Found separator, return position AT separator end
            return pos + idx + len(separator)

    # No separator found, return original
    return pos


def merge_windows(
    windows: list[tuple[int, int]],
    gap_threshold: int,
) -> list[tuple[int, int]]:
    """Merge overlapping or nearby windows.

    Two windows are merged if:
    - They overlap (start2 <= end1), OR
    - Gap between them is <= gap_threshold

    This prevents creating fragmented passages when matches are close together.

    Args:
        windows: List of (start, end) tuples, may be unsorted.
        gap_threshold: Merge windows within this character distance.

    Returns:
        List of merged (start, end) tuples, sorted by start position.

    Example:
        >>> merge_windows([(0, 100), (150, 250), (90, 120)], gap_threshold=100)
        [(0, 250)]  # All three merged due to overlap/proximity
    """
    if not windows:
        return []

    # Sort by start position
    sorted_windows = sorted(windows)
    merged = [sorted_windows[0]]

    for start, end in sorted_windows[1:]:
        prev_start, prev_end = merged[-1]

        # Check if should merge
        if start <= prev_end + gap_threshold:
            # Merge: extend the previous window
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            # Gap too large, keep separate
            merged.append((start, end))

    return merged


def truncate_at_boundary(text: str, max_length: int) -> str:
    """Truncate text at word boundary.

    Args:
        text: Text to truncate.
        max_length: Maximum length.

    Returns:
        Truncated text with '...' suffix if truncated.
    """
    if len(text) <= max_length:
        return text

    truncated = text[:max_length]
    last_space = truncated.rfind(' ')

    # Only use word boundary if it's not too far back
    if last_space > max_length * 0.7:
        return truncated[:last_space] + '...'

    return truncated + '...'


def strip_tags(text: str) -> str:
    """Remove <mark> tags from text.

    Args:
        text: Text potentially containing <mark></mark> tags.

    Returns:
        Text with all <mark> and </mark> tags removed.
    """
    return text.replace(MARK_OPEN, '').replace(MARK_CLOSE, '')


def extract_rerank_passage(
    text_content: str,
    highlighted: str | None,
    window_size: int = 750,
    max_passage_size: int = 2000,
    gap_merge_threshold: int = 100,
    separators: list[str] | None = None,
) -> str:
    """Extract coherent passage from FTS results for reranking.

    Algorithm:
    1. Parse <mark> positions from highlighted text
    2. Map positions to original text (aligning against it, so markup already
       present in the document is not mistaken for an inserted marker)
    3. Apply window around each match
    4. Expand windows to sentence boundaries (using separators)
    5. Merge overlapping/nearby windows
    6. Extract and join passages
    7. Truncate if exceeds max_passage_size
    8. Strip <mark> tags from final output

    Args:
        text_content: Original full document text (no marks).
        highlighted: Text with <mark> tags from FTS (full document with ALL marks).
            If None or empty, returns beginning of document.
        window_size: Characters to include around each match (default: 750).
        max_passage_size: Maximum final passage length (default: 2000).
        gap_merge_threshold: Merge regions closer than this (default: 100).
        separators: Boundary markers in priority order (default: paragraph/line/sentence/word).

    Returns:
        Clean text passage suitable for reranker (no <mark> tags).
        Returns beginning of document if no highlights found.

    Example:
        >>> text = 'Start. ' + 'x' * 5000 + ' Middle match here. ' + 'y' * 5000 + ' End.'
        >>> highlighted = text  # Assume same with marks
        >>> passage = extract_rerank_passage(text.replace('<mark>', '').replace('</mark>', ''), highlighted)
        >>> 'match' in passage
        True
        >>> len(passage) <= 2000
        True
    """
    if separators is None:
        separators = DEFAULT_SEPARATORS

    # If no highlight available, return beginning of document
    if not highlighted:
        logger.debug('No highlighted text, returning document beginning')
        return strip_tags(text_content[:max_passage_size])

    # Step 1: Parse highlight positions against the original text, so a literal
    # <mark> stored in the document cannot be mistaken for an inserted marker.
    regions = parse_highlight_positions(highlighted, text_content)

    if not regions:
        # No usable match positions, return beginning
        logger.debug('No usable <mark> positions, returning document beginning')
        return strip_tags(text_content[:max_passage_size])

    logger.debug(f'Found {len(regions)} highlighted regions')

    # Step 2: Apply windows around each region
    text_length = len(text_content)
    windows: list[tuple[int, int]] = []
    for region in regions:
        window_start = max(0, region.start - window_size)
        window_end = min(text_length, region.end + window_size)
        windows.append((window_start, window_end))

    # Step 3: Expand to boundaries
    expanded_windows: list[tuple[int, int]] = []
    for start, end in windows:
        new_start = expand_to_boundary(text_content, start, 'left', separators)
        new_end = expand_to_boundary(text_content, end, 'right', separators)
        expanded_windows.append((new_start, new_end))

    # Step 4: Merge overlapping/nearby windows
    merged = merge_windows(expanded_windows, gap_merge_threshold)

    logger.debug(f'After merging: {len(merged)} passage segments')

    # Step 5: Extract and combine passages
    passages: list[str] = []
    for start, end in merged:
        passages.append(text_content[start:end])

    # Join with ellipsis for discontinuous passages
    combined = passages[0] if len(passages) == 1 else ' ... '.join(passages)

    # Step 6: Truncate if still too large
    if len(combined) > max_passage_size:
        combined = truncate_at_boundary(combined, max_passage_size)
        logger.debug(f'Truncated to {len(combined)} chars')

    # Step 7: Strip any remaining tags (safety)
    result = strip_tags(combined)

    logger.debug(f'Final passage: {len(result)} chars from {len(text_content)} char document')

    return result
