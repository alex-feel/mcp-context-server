"""Tests for access-control policy helpers.

Covers app.auth.access: resolve_effective_principal (verified-token passthrough
and the no-token fallback to the configured default principal) and
visibility_denied_reason (the ACCESS_CONTROL_PUBLISH_ROLE publish gate).
"""

import os
from unittest.mock import patch

import pytest

from app.auth.access import resolve_effective_principal
from app.auth.access import visibility_denied_reason
from app.auth.principal import RequestPrincipal
from app.settings import get_settings


class TestResolveEffectivePrincipal:
    """Tests for resolve_effective_principal."""

    def setup_method(self) -> None:
        """Reset the settings singleton so per-test env vars take effect."""
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        """Drop settings built from per-test env vars."""
        get_settings.cache_clear()

    def test_verified_principal_passes_through(self) -> None:
        """A resolved request principal is returned unchanged."""
        verified = RequestPrincipal(
            principal_id='alice',
            groups=frozenset({'team-a'}),
            roles=frozenset({'publisher'}),
        )
        with patch('app.auth.access.resolve_request_principal', return_value=verified):
            assert resolve_effective_principal() is verified

    def test_no_token_falls_back_to_default_principal(self) -> None:
        """Without a verified token the configured default principal applies."""
        with patch('app.auth.access.resolve_request_principal', return_value=None):
            principal = resolve_effective_principal()
        assert principal == RequestPrincipal(
            principal_id='local',
            groups=frozenset(),
            roles=frozenset(),
        )

    def test_configured_default_principal_is_honored(self) -> None:
        """ACCESS_CONTROL_DEFAULT_PRINCIPAL selects the fallback identity."""
        env = {'ACCESS_CONTROL_DEFAULT_PRINCIPAL': 'ci-runner'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            with patch('app.auth.access.resolve_request_principal', return_value=None):
                principal = resolve_effective_principal()
        assert principal.principal_id == 'ci-runner'
        assert principal.groups == frozenset()
        assert principal.roles == frozenset()


class TestVisibilityDeniedReason:
    """Tests for the publish gate."""

    def setup_method(self) -> None:
        """Reset the settings singleton so per-test env vars take effect."""
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        """Drop settings built from per-test env vars."""
        get_settings.cache_clear()

    @staticmethod
    def _principal(roles: frozenset[str] = frozenset()) -> RequestPrincipal:
        return RequestPrincipal(principal_id='p', groups=frozenset(), roles=roles)

    @pytest.mark.parametrize('visibility', ['private', 'shared'])
    def test_non_public_is_never_gated(self, visibility: str) -> None:
        """private and shared are always allowed, publish role or not."""
        env = {'ACCESS_CONTROL_PUBLISH_ROLE': 'publisher'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            assert visibility_denied_reason(visibility, self._principal()) is None

    def test_public_allowed_when_role_unset(self) -> None:
        """With no configured publish role, any owner may publish."""
        assert visibility_denied_reason('public', self._principal()) is None

    def test_public_denied_without_required_role(self) -> None:
        """A configured publish role denies callers that lack it, naming the role."""
        env = {'ACCESS_CONTROL_PUBLISH_ROLE': 'publisher'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            denial = visibility_denied_reason('public', self._principal())
        assert denial is not None
        assert 'publisher' in denial

    def test_public_allowed_with_required_role(self) -> None:
        """A caller whose roles carry the configured publish role may publish."""
        env = {'ACCESS_CONTROL_PUBLISH_ROLE': 'publisher'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            principal = self._principal(roles=frozenset({'publisher', 'other'}))
            assert visibility_denied_reason('public', principal) is None
