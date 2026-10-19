"""Corpus sync tool settings: ``plan_context_sync`` and ``list_ingest_state``."""

from typing import Literal

from pydantic import Field

from app.settings.base import FeatureToggleSettings


class ContextSyncSettings(FeatureToggleSettings):
    """Corpus sync tool configuration.

    Controls registration of plan_context_sync and list_ingest_state, the two
    read-only tools a sync client uses to compare a corpus with its stored
    entries. Both read stored rows and need no extra dependencies, so 'auto'
    registers them by default. The per-call caps are protocol constants in
    ``app.ingest``, because the transport limits they protect are fixed.
    """

    mode: Literal['auto', 'true', 'false'] = Field(
        default='auto',
        alias='ENABLE_CONTEXT_SYNC',
        description='plan_context_sync and list_ingest_state registration: auto (register; '
                    'no extra dependencies), true (force on), false (force off)',
    )
