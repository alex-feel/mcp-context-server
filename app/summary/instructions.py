"""Default summarization prompt for LLM-based context summarization.

Provides the DEFAULT_SUMMARY_PROMPT constant used when SUMMARY_PROMPT
environment variable is not set. The prompt is optimized for small
models (qwen3:1.7b) and follows best practices for LLM summarization.

The prompt uses a system message format:
- Contains /no_think instruction for Qwen3 models
- Role definition + formatting rules + constraints
- Designed for zero-shot (no examples) to maximize input token budget

The content to summarize is passed separately as a user/human message
at runtime (not part of this prompt constant).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.settings import SummarySettings

# Default system message for summary generation.
# Override at runtime via SUMMARY_PROMPT environment variable.
#
# Design decisions:
# 1. /no_think prefix disables Qwen3 reasoning mode (saves tokens + time)
# 2. Zero-shot (no examples) - maximizes input token budget for small models
# 3. Output token limit controlled via SUMMARY_MAX_TOKENS at LLM API level
# 4. "Output ONLY the summary" prevents preamble/labels from small models
# 5. Focuses on entities/topics/decisions - what agents need for relevance judgment
# 6. "Do not add" negation constraints prevent common small-model failure modes
# 7. Single paragraph format - easiest for small models to follow
# 8. Temperature=0 set at model level for deterministic output (not in prompt)
#
# Prompt engineering best practices applied:
# - Structured, goal-oriented phrasing (not vague "summarize this")
# - Explicit output format constraints (single paragraph)
# - Role assignment ("expert summarizer") for consistent behavior
# - Negative constraints to prevent common failure modes
# - Information density focus over narrative fluency
DEFAULT_SUMMARY_PROMPT: str = ('''
/no_think
You are an expert summarizer for a context storage system used by AI agents.
Your task is to produce a single, dense paragraph that captures the essential meaning of the input text.

Requirements:
- Include key topics, named entities, decisions, conclusions, and action items
- Prioritize information that helps determine relevance without reading the full text
- Use specific terms from the original text (do not generalize or abstract away details)
- Write exactly one paragraph with no line breaks
- Do not add any labels, prefixes, headers, or explanations
- Do not start with "This text" or "The author" or similar meta-references
- Output ONLY the summary text, nothing else
''').strip()


def resolve_summary_prompt(summary_settings: SummarySettings) -> str:
    """Resolve the summary prompt to use for generation.

    Both empty string and None fall back to the default prompt.
    Unlike MCP_SERVER_INSTRUCTIONS (which can be disabled via empty string),
    summary generation REQUIRES instructions to produce meaningful output.
    Disabling the summary prompt is intentionally not supported.

    Args:
        summary_settings: The SummarySettings instance.

    Returns:
        The resolved summary prompt text.
    """
    prompt = summary_settings.prompt
    if prompt is None or prompt.strip() == '':
        return DEFAULT_SUMMARY_PROMPT
    return prompt
