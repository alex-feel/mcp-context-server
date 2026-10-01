"""Summary generation settings: the flat document summary and the index_tree per-node summaries."""

import os
from typing import Literal

from pydantic import Field
from pydantic import SecretStr
from pydantic import field_validator

from app.settings.base import CommonSettings


class SummarySettings(CommonSettings):
    """Summary generation settings for LLM-based context summarization.

    Controls automatic summary generation for stored context entries.
    Summaries are stored alongside full text and returned in search results,
    providing dense context for LLM agents to assess relevance quickly.
    """

    generation_enabled: bool = Field(
        default=True,
        alias='ENABLE_SUMMARY_GENERATION',
        description='Enable summary generation for stored context entries. '
                    'If true and dependencies are not met, server will NOT start. '
                    'Set to false to disable summaries entirely.',
    )

    provider: Literal['ollama', 'openai', 'anthropic'] = Field(
        default='ollama',
        alias='SUMMARY_PROVIDER',
        description='Summary provider: ollama (default, local/free), openai, anthropic',
    )

    model: str = Field(
        default='qwen3:0.6b',
        alias='SUMMARY_MODEL',
        description='Summary generation model name. Default: qwen3:0.6b. '
                    'Alternatives: qwen3:1.7b (higher quality), qwen3:4b (high quality), '
                    'qwen3:8b (highest quality)',
    )

    max_tokens: int = Field(
        default=4000,
        alias='SUMMARY_MAX_TOKENS',
        ge=50,
        le=16384,
        description='Maximum output tokens for summary generation. '
                    'Acts as a safety ceiling passed to the LLM API '
                    '(max_tokens for OpenAI/Anthropic, num_predict for Ollama). '
                    'The LLM typically generates far fewer tokens than this limit. '
                    'Reasoning models may consume a portion of this budget on '
                    'internal reasoning tokens -- increase if summaries are truncated.',
    )

    timeout_s: float = Field(
        default=240.0,
        alias='SUMMARY_TIMEOUT_S',
        gt=0,
        le=600,
        description='Timeout in seconds for summary generation API calls',
    )

    retry_max_attempts: int = Field(
        default=5,
        alias='SUMMARY_RETRY_MAX_ATTEMPTS',
        ge=1,
        le=10,
        description='Maximum number of retry attempts for summary generation',
    )

    retry_base_delay_s: float = Field(
        default=3.0,
        alias='SUMMARY_RETRY_BASE_DELAY_S',
        gt=0,
        le=30,
        description='Base delay in seconds between retry attempts (with exponential backoff)',
    )

    max_concurrent: int = Field(
        default=2,
        alias='SUMMARY_MAX_CONCURRENT',
        ge=1,
        le=20,
        description='Maximum concurrent calls against the summary model; one shared '
                    'budget covering both the flat document summary and every '
                    'index_tree per-node summary.',
    )

    prompt: str | None = Field(
        default=None,
        alias='SUMMARY_PROMPT',
        description='Custom summarization prompt. Overrides the built-in default prompt. '
                    'The prompt is used as the system message when calling the LLM. '
                    'Should instruct the model to output only the summary text. '
                    'For Qwen3 models, include /no_think prefix to disable reasoning mode.',
    )

    min_content_length: int = Field(
        default=500,
        ge=0,
        le=10000,
        alias='SUMMARY_MIN_CONTENT_LENGTH',
        description='Minimum text content length (characters) to trigger summary generation. '
                    'Content shorter than this threshold is not summarized (the truncated preview '
                    'at 300 characters adequately represents the content, so a separate '
                    'LLM-generated summary adds minimal value for small models). '
                    'Set to 0 to always generate summaries regardless of content length.',
    )

    # Ollama-specific context and truncation settings
    ollama_num_ctx: int = Field(
        default=32768,
        alias='SUMMARY_OLLAMA_NUM_CTX',
        ge=512,
        le=2097152,
        description='Ollama summary context length in tokens. Default 32768 (matches qwen3 native context). '
                    'Must match or exceed model capabilities for summary generation.',
    )
    ollama_truncate: bool = Field(
        default=False,
        alias='SUMMARY_OLLAMA_TRUNCATE',
        description='Control text truncation when exceeding summary context length. '
                    'False (default): Returns error on exceeded context. '
                    'True: Silently truncates input (summary generated from incomplete text).',
    )

    # OpenAI-specific authentication (shared env var with EmbeddingSettings)
    openai_api_key: SecretStr | None = Field(
        default=None,
        alias='OPENAI_API_KEY',
        description='OpenAI API key for summary generation',
    )
    openai_api_base: str | None = Field(
        default=None,
        alias='OPENAI_API_BASE',
        description='Custom base URL for OpenAI-compatible APIs',
    )

    # Anthropic-specific authentication
    anthropic_api_key: SecretStr | None = Field(
        default=None,
        alias='ANTHROPIC_API_KEY',
        description='Anthropic API key for summary generation',
    )

    # Cross-provider reasoning/effort control
    openai_reasoning_effort: str | None = Field(
        default='low',
        alias='SUMMARY_OPENAI_REASONING_EFFORT',
        description='Reasoning effort level for OpenAI reasoning models (e.g., gpt-5.4-nano). '
                    'Controls how many tokens the model spends on internal reasoning. '
                    'Valid values vary by model generation: '
                    'gpt-5 (original): low, medium, high. '
                    'gpt-5.1/5.2/5.4 (including nano/mini): none, low, medium, high, xhigh. '
                    'Default: low (universally valid across all generations). '
                    'Set to None to omit the parameter entirely.',
    )
    anthropic_effort: Literal['max', 'high', 'medium', 'low'] | None = Field(
        default=None,
        alias='SUMMARY_ANTHROPIC_EFFORT',
        description='Effort level for Anthropic Claude models. '
                    'Controls inference effort (adaptive thinking). '
                    'Valid values: max, high, medium, low. '
                    'Default: None (do not send -- required for models that do not '
                    'support the effort parameter, such as Haiku 4.5). '
                    'When set, passed directly as effort= to ChatAnthropic constructor.',
    )

    @field_validator('openai_reasoning_effort', 'anthropic_effort', mode='before')
    @classmethod
    def _empty_effort_to_none(cls, value: object) -> object:
        """Fold an empty/whitespace-only effort string to None.

        Environment variables cannot express Python ``None``, so the documented
        way to omit the effort parameter is an empty value (e.g.
        ``SUMMARY_OPENAI_REASONING_EFFORT=``). Without this coercion the empty
        string reaches the provider verbatim: ``ChatOpenAI`` would be handed
        ``reasoning_effort=''`` (rejected by the OpenAI API), and the Anthropic
        ``Literal`` would reject ``''`` at startup. Folding the empty string to
        ``None`` makes the documented "set to empty to omit" idiom work for both
        fields while leaving every non-empty value to its own validation.

        Returns:
            ``None`` when ``value`` is an empty or whitespace-only string,
            otherwise ``value`` unchanged.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value


class IndexTreeNodeSummarySettings(CommonSettings):
    """Optional per-node LLM summaries for the navigate_context index_tree.

    The code-derived heading outline is always free; this OPTIONAL layer enriches
    each section with an LLM-written abstract. It reuses the existing summary
    provider instance (no second client) with a dedicated SHORT prompt, runs in a
    fenced never-raise pass that can never abort a store, and is the ONLY thing
    that provisions the context_index_nodes table. Default ON per the chosen
    design.
    """

    node_summaries_enabled: bool = Field(
        default=True,
        alias='ENABLE_INDEX_TREE_NODE_SUMMARIES',
        description='Generate per-node LLM summaries for the index_tree and provision '
                    'the context_index_nodes table. Additive/never-raise: a node-summary '
                    'failure never aborts a store. Set false to keep navigation purely '
                    'code-derived with no table and no per-store LLM cost.',
    )

    prompt: str | None = Field(
        default=None,
        alias='INDEX_TREE_NODE_SUMMARY_PROMPT',
        description='Override the per-node summary system prompt. None/empty resolves to '
                    'a dedicated SHORT prompt (one-sentence section abstract), distinct '
                    'from the 100-250-word entry-summary prompt.',
    )

    min_content_length: int = Field(
        default=500,
        alias='INDEX_TREE_NODE_SUMMARY_MIN_CONTENT_LENGTH',
        ge=0,
        le=100000,
        description='Heading sections shorter than this (characters) skip node-summary '
                    'generation (0 = always summarize).',
    )

    timeout_s: float = Field(
        default=240.0,
        alias='INDEX_TREE_NODE_SUMMARY_TIMEOUT_S',
        gt=0,
        le=600,
        description='Per-node summary timeout in seconds; a timeout omits that node, '
                    'never aborts the store.',
    )

    max_nodes: int = Field(
        default=200,
        alias='INDEX_TREE_NODE_SUMMARY_MAX_NODES',
        ge=1,
        le=10000,
        description='Maximum number of heading sections summarized for one entry. Bounds TOTAL '
                    'work per store (the concurrency caps bound only how much runs at once): a '
                    'heading-dense document would otherwise issue one model call per section. '
                    'When more sections qualify, the shallowest and longest are summarized first '
                    'so the outline degrades gracefully.',
    )

    total_timeout_s: float = Field(
        default=600.0,
        alias='INDEX_TREE_NODE_SUMMARY_TOTAL_TIMEOUT_S',
        gt=0,
        le=3600,
        description='Aggregate wall-clock budget in seconds for the whole per-node summary pass '
                    'of one entry. When it expires the pass stops and keeps the summaries produced '
                    'so far; like every other node-summary limit it never aborts the store.',
    )

    max_concurrent: int = Field(
        default_factory=lambda: min(os.cpu_count() or 4, 4),
        alias='INDEX_TREE_NODE_SUMMARY_MAX_CONCURRENT',
        ge=1,
        le=32,
        description='Node-task fan-out cap: the maximum number of per-node index_tree summary '
                    'coroutines in flight at once. This is NOT a second model budget -- the actual '
                    'summary-model concurrency is governed by SUMMARY_MAX_CONCURRENT (one shared '
                    'budget acquired by both the flat document summary and every per-node summary). '
                    'Default min(cpu_count, 4).',
    )
