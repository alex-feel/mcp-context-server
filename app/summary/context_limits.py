"""Context window limits for summary models by provider.

Mirrors the embedding context_limits.py pattern for summary generation.
Used for validation and startup logging.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SummaryModelSpec:
    """Specification for a summary model."""

    provider: str
    model: str
    max_input_tokens: int
    truncation_behavior: Literal['error', 'silent', 'configurable']
    notes: str = ''


# Known summary model specifications
SUMMARY_MODEL_SPECS: dict[str, SummaryModelSpec] = {
    # Ollama models -- truncation_behavior is 'configurable' via SUMMARY_OLLAMA_TRUNCATE
    'qwen3:0.6b': SummaryModelSpec(
        provider='ollama',
        model='qwen3:0.6b',
        max_input_tokens=32768,
        truncation_behavior='configurable',
        notes='Context controlled by SUMMARY_OLLAMA_NUM_CTX. Truncation controlled by SUMMARY_OLLAMA_TRUNCATE.',
    ),
    'qwen3:1.7b': SummaryModelSpec(
        provider='ollama',
        model='qwen3:1.7b',
        max_input_tokens=32768,
        truncation_behavior='configurable',
    ),
    'qwen3:4b': SummaryModelSpec(
        provider='ollama',
        model='qwen3:4b',
        max_input_tokens=131072,
        truncation_behavior='configurable',
        notes='YaRN enabled by default on Ollama, extending context to 131K.',
    ),
    'qwen3:8b': SummaryModelSpec(
        provider='ollama',
        model='qwen3:8b',
        max_input_tokens=32768,
        truncation_behavior='configurable',
    ),
    'qwen3:14b': SummaryModelSpec(
        provider='ollama',
        model='qwen3:14b',
        max_input_tokens=32768,
        truncation_behavior='configurable',
    ),
    'qwen3:32b': SummaryModelSpec(
        provider='ollama',
        model='qwen3:32b',
        max_input_tokens=32768,
        truncation_behavior='configurable',
    ),
    # OpenAI models -- always return HTTP error on context exceed
    'gpt-5.4-nano': SummaryModelSpec(
        provider='openai',
        model='gpt-5.4-nano',
        max_input_tokens=400000,
        truncation_behavior='error',
        notes='Returns HTTP 400 when context exceeded.',
    ),
    'gpt-5.4-mini': SummaryModelSpec(
        provider='openai',
        model='gpt-5.4-mini',
        max_input_tokens=400000,
        truncation_behavior='error',
    ),
    'gpt-5.4': SummaryModelSpec(
        provider='openai',
        model='gpt-5.4',
        max_input_tokens=400000,
        truncation_behavior='error',
    ),
    'gpt-5-nano': SummaryModelSpec(
        provider='openai',
        model='gpt-5-nano',
        max_input_tokens=400000,
        truncation_behavior='error',
        notes='Original GPT-5 nano. Returns HTTP 400 when context exceeded.',
    ),
    # Anthropic models -- always return HTTP error on context exceed
    'claude-haiku-4-5-20251001': SummaryModelSpec(
        provider='anthropic',
        model='claude-haiku-4-5-20251001',
        max_input_tokens=200000,
        truncation_behavior='error',
        notes='Returns HTTP 400 when context exceeded.',
    ),
    'claude-sonnet-4-20250514': SummaryModelSpec(
        provider='anthropic',
        model='claude-sonnet-4-20250514',
        max_input_tokens=200000,
        truncation_behavior='error',
    ),
    'claude-opus-4-6': SummaryModelSpec(
        provider='anthropic',
        model='claude-opus-4-6',
        max_input_tokens=200000,
        truncation_behavior='error',
    ),
    'claude-sonnet-4-6': SummaryModelSpec(
        provider='anthropic',
        model='claude-sonnet-4-6',
        max_input_tokens=200000,
        truncation_behavior='error',
    ),
}


def get_summary_model_spec(model: str) -> SummaryModelSpec | None:
    """Get specification for a summary model by name."""
    return SUMMARY_MODEL_SPECS.get(model)


def get_summary_provider_default_context(provider: str) -> int:
    """Get conservative default context limit for a summary provider."""
    defaults = {
        'ollama': 32768,      # Qwen3 native context
        'openai': 400000,     # GPT-5.4 family
        'anthropic': 200000,  # Claude models (standard tier)
    }
    return defaults.get(provider, 32768)
