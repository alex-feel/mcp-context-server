"""Navigation tool settings: ``grep_context``, ``read_context_range``, and ``navigate_context``."""

from typing import Literal

from pydantic import Field

from app.settings.base import FeatureToggleSettings


class GrepContextSettings(FeatureToggleSettings):
    """Server-side grep tool configuration.

    Controls registration of the grep_context tool plus the safety bounds that
    keep an unranked, line-oriented scan from flooding an agent's context window
    or exhausting memory. Matching itself runs in Python (re), so there are no
    extra dependencies and 'auto' registers the tool by default.
    """

    mode: Literal['auto', 'true', 'false'] = Field(
        default='auto',
        alias='ENABLE_GREP_CONTEXT',
        description='grep_context tool registration: auto (register; matching is '
                    'pure-Python, no extra dependencies), true (force on), false '
                    '(force off).',
    )

    max_matches_cap: int = Field(
        default=1000,
        alias='GREP_MAX_MATCHES_CAP',
        ge=1,
        description='Hard ceiling applied to grep_context max_matches per request.',
    )

    max_context_lines: int = Field(
        default=20,
        alias='GREP_MAX_CONTEXT_LINES',
        ge=0,
        description='Hard ceiling applied to grep_context context_lines per request.',
    )

    max_entries_scanned: int = Field(
        default=1000,
        alias='GREP_MAX_ENTRIES_SCANNED',
        ge=1,
        description='Hard ceiling on the number of entries the grep scan visits.',
    )

    aggregate_bytes_budget: int = Field(
        default=67108864,
        alias='GREP_AGGREGATE_BYTES_BUDGET',
        ge=1,
        description='Approximate resident-memory cap (summed code-point length of '
                    'fetched text) before a grep scan stops; the first entry that '
                    'crosses the budget is still scanned.',
    )

    regex_timeout_s: float = Field(
        default=5.0,
        alias='GREP_REGEX_TIMEOUT_S',
        gt=0,
        description='Per-entry timeout for is_regex=True matching (ReDoS guard); a '
                    'timeout skips that entry, never aborts the read.',
    )

    regex_total_timeout_s: float = Field(
        default=30.0,
        alias='GREP_REGEX_TOTAL_TIMEOUT_S',
        gt=0,
        description='Aggregate wall-clock budget across an entire is_regex=True scan. '
                    'GREP_REGEX_TIMEOUT_S bounds ONE entry; this bounds the cumulative '
                    'scan so a pathological pattern over many entries cannot hold one '
                    'request open for max_entries_scanned * the per-entry timeout. When '
                    'exceeded the scan stops and returns the matches collected so far '
                    'with truncated=True, never aborting the read.',
    )

    max_pattern_chars: int = Field(
        default=32768,
        alias='GREP_MAX_PATTERN_CHARS',
        ge=1,
        description='Maximum grep_context pattern length in characters, advertised in '
                    'the wire schema and re-checked in the tool body as a structured '
                    'validation error. Pattern COMPILATION runs before any matching '
                    'timeout applies and its cost grows with pattern size, so an '
                    'unbounded multi-megabyte pattern could stall the request for '
                    'seconds just to compile. The 32768 (32 KiB) default accommodates '
                    'any realistic literal or generated alternation (hundreds of '
                    'OR-joined terms) while keeping a worst-case compile in the low '
                    'milliseconds.',
    )


class ContextRangeSettings(FeatureToggleSettings):
    """Partial-read tool configuration.

    Controls registration of the read_context_range tool, which slices full
    text_content by character or line range. Backend-agnostic and dependency-free,
    so 'auto' registers it by default.
    """

    mode: Literal['auto', 'true', 'false'] = Field(
        default='auto',
        alias='ENABLE_CONTEXT_RANGE',
        description='read_context_range tool registration: auto (register; slices '
                    'stored text, no extra dependencies), true (force on), false '
                    '(force off).',
    )


class ContextNavigationSettings(FeatureToggleSettings):
    """Record-navigation tool configuration.

    Controls registration of the navigate_context tool, which builds a
    code-derived Markdown outline (index_tree) on demand. Pure-Python and
    dependency-free, so 'auto' registers it by default.
    """

    mode: Literal['auto', 'true', 'false'] = Field(
        default='auto',
        alias='ENABLE_CONTEXT_NAVIGATION',
        description='navigate_context tool registration: auto (register; builds a '
                    'code-derived Markdown outline, no extra dependencies), true '
                    '(force on), false (force off).',
    )
