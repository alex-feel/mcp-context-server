"""Tests for AccessControlSettings.

Covers the defaults, the env-var aliases, the enum constraints on visibility
and group-grant policy, and the DDL-safe character-set validation on the
default principal.
"""

import pytest
from pydantic import ValidationError

from app.settings import AccessControlSettings
from app.settings import get_settings


class TestAccessControlSettings:
    """AccessControlSettings defaults and validation."""

    def test_defaults(self) -> None:
        """The documented defaults apply without any env configuration."""
        settings = AccessControlSettings()
        assert settings.default_principal == 'local'
        assert settings.default_visibility == 'private'
        assert settings.default_group_grants == 'none'
        assert settings.publish_role is None

    def test_env_aliases(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Each field binds to its ACCESS_CONTROL_* env var."""
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_PRINCIPAL', 'kb-service')
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_VISIBILITY', 'shared')
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_GROUP_GRANTS', 'author_groups')
        monkeypatch.setenv('ACCESS_CONTROL_PUBLISH_ROLE', 'publisher')
        settings = AccessControlSettings()
        assert settings.default_principal == 'kb-service'
        assert settings.default_visibility == 'shared'
        assert settings.default_group_grants == 'author_groups'
        assert settings.publish_role == 'publisher'

    def test_invalid_visibility_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """default_visibility only accepts the three-value enum."""
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_VISIBILITY', 'everyone')
        with pytest.raises(ValidationError):
            AccessControlSettings()

    def test_invalid_group_grants_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """default_group_grants only accepts none or author_groups."""
        monkeypatch.setenv('ACCESS_CONTROL_DEFAULT_GROUP_GRANTS', 'everyone')
        with pytest.raises(ValidationError):
            AccessControlSettings()

    @pytest.mark.parametrize('principal', [
        'local',
        'kb.service@example.com',
        'user:role+tag-1_x',
        'A' * 128,
    ])
    def test_safe_principals_accepted(self, principal: str) -> None:
        """Principals within the DDL-safe character set validate."""
        assert AccessControlSettings(ACCESS_CONTROL_DEFAULT_PRINCIPAL=principal).default_principal == principal

    @pytest.mark.parametrize('principal', [
        "bad'quote",
        'has space',
        'back\\slash',
        '',
        'A' * 129,
        'semi;colon',
    ])
    def test_unsafe_principals_rejected(self, principal: str) -> None:
        """Principals that could break the DDL literal are refused at settings build."""
        with pytest.raises(ValidationError):
            AccessControlSettings(ACCESS_CONTROL_DEFAULT_PRINCIPAL=principal)

    def test_composed_into_app_settings(self) -> None:
        """AppSettings exposes the nested access_control settings."""
        get_settings.cache_clear()
        try:
            assert get_settings().access_control.default_visibility in ('private', 'shared', 'public')
        finally:
            get_settings.cache_clear()
