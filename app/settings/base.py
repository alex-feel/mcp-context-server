"""Settings base classes: the shared environment and ``.env`` configuration, and the tri-state feature toggle."""

from typing import Literal

from dotenv import find_dotenv
from pydantic import field_validator
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class CommonSettings(BaseSettings):
    """Base for the settings classes: values come from the environment and the nearest ``.env`` file, and are frozen."""

    model_config = SettingsConfigDict(
        frozen=True,
        env_file=find_dotenv(),
        env_file_encoding='utf-8',
        case_sensitive=True,
        extra='ignore',
        populate_by_name=True,
    )


def _normalize_feature_toggle(value: object) -> object:
    """Normalize a search feature-toggle value to 'auto' | 'true' | 'false'.

    Accepts the tri-state strings plus the conventional boolean spellings, so an
    existing ENABLE_*=true/false/1/0/yes/no configuration keeps working. 'auto'
    (the default) defers the decision to runtime capability detection. Unknown
    values fall through to Literal validation, which rejects them with a clear
    error.

    Returns:
        The normalized token; 'auto', 'true', or 'false' for recognized inputs,
        otherwise the lowercased text unchanged for Literal validation to reject.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    text = str(value).strip().lower()
    if text in {'true', '1', 'yes', 'on', 'enabled'}:
        return 'true'
    if text in {'false', '0', 'no', 'off', 'disabled'}:
        return 'false'
    if text in {'auto', ''}:
        return 'auto'
    return text


class FeatureToggleSettings(CommonSettings):
    """Base for the search feature toggles, which are tri-state by design.

    The three search tools default to 'auto' rather than being opt-in: a
    deployment whose prerequisites are already present (an embedding provider for
    semantic search; nothing extra for full-text search, which uses built-in
    database capabilities) exposes the corresponding tool with no configuration,
    while a deployment that lacks the prerequisites silently skips it. The
    explicit 'true' registers the tool exactly when its prerequisites are present
    -- for a prerequisite-gated tool a missing prerequisite logs a warning and the
    tool is still NOT registered (a prerequisite-free tool simply registers) --
    and 'false' forces the tool off (minimal tool surface). Embedding storage is
    provisioned from ENABLE_EMBEDDING_GENERATION independently of these toggles,
    so enabling a search tool later never requires re-embedding.

    Subclasses declare ``mode`` with the feature-specific env alias.
    """

    mode: Literal['auto', 'true', 'false'] = 'auto'

    @field_validator('mode', mode='before')
    @classmethod
    def _coerce_mode(cls, value: object) -> object:
        return _normalize_feature_toggle(value)

    @property
    def enabled(self) -> bool:
        """True unless the feature is force-disabled (``mode == 'false'``).

        'auto' and 'true' both report enabled; the runtime decides whether the
        prerequisites are actually present before exposing the tool.
        """
        return self.mode != 'false'
