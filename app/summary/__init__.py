"""Summary generation for context entries.

Provides LLM-based summarization to generate concise summaries
stored alongside full text for search result context.

Architecture mirrors app/embeddings/ for consistency:
- base.py: SummaryProvider protocol definition
- factory.py: Provider factory with auto-import
- retry.py: Retry wrapper for summary operations
- instructions.py: Default prompt and resolution
- providers/: Concrete provider implementations

Example Usage:
    from app.summary import create_summary_provider, SummaryProvider

    provider = create_summary_provider()  # Uses SUMMARY_PROVIDER setting
    await provider.initialize()

    summary = await provider.summarize("Long text content here...")

    await provider.shutdown()
"""

from app.summary.base import SummaryProvider
from app.summary.factory import create_summary_provider
from app.summary.retry import SummaryRetryExhaustedError
from app.summary.retry import SummaryTimeoutError
from app.summary.retry import with_summary_retry_and_timeout

__all__ = [
    'SummaryProvider',
    'create_summary_provider',
    'SummaryRetryExhaustedError',
    'SummaryTimeoutError',
    'with_summary_retry_and_timeout',
]
