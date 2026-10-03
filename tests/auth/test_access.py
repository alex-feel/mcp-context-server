"""Tests for access-control policy helpers.

Covers app.auth.access: resolve_effective_principal (verified-token passthrough
and the fallback to the configured default principal for requests without a jwt
identity, simple_token included), resolve_access_scope (the same identity as
the scope every read and write runs under, roles excluded) and
visibility_denied_reason (the ACCESS_CONTROL_PUBLISH_ROLE publish gate).
"""

import os
from unittest.mock import patch

from fastmcp.server.auth import AccessToken

import app.auth
from app.access_scope import AccessScope
from app.auth.access import resolve_access_scope
from app.auth.access import resolve_effective_principal
from app.auth.access import visibility_denied_reason
from app.auth.principal import RequestPrincipal
from app.settings import get_settings
from tests.helpers import as_principal


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

    def test_simple_token_request_maps_to_default_principal(self) -> None:
        """A simple_token request owns rows as the default principal, not its client id."""
        token = AccessToken(
            token='opaque-token-value',
            client_id='mcp-client',
            scopes=[],
            expires_at=None,
            claims={},
        )
        env = {'MCP_AUTH_PROVIDER': 'simple_token', 'MCP_AUTH_TOKEN': 'test-token'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            with patch('app.auth.principal.get_access_token', return_value=token):
                principal = resolve_effective_principal()
        assert principal == RequestPrincipal(
            principal_id='local',
            groups=frozenset(),
            roles=frozenset(),
        )


class TestResolveAccessScope:
    """Tests for resolve_access_scope."""

    def setup_method(self) -> None:
        """Reset the settings singleton so per-test env vars take effect."""
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        """Drop settings built from per-test env vars."""
        get_settings.cache_clear()

    def test_no_token_gives_default_principal_scope(self) -> None:
        """Without a verified token the scope is the default principal with no groups."""
        with patch('app.auth.access.resolve_request_principal', return_value=None):
            assert resolve_access_scope() == AccessScope('local', frozenset())

    def test_configured_default_principal_scopes_the_request(self) -> None:
        """ACCESS_CONTROL_DEFAULT_PRINCIPAL is the scope of a request without a jwt identity."""
        env = {'ACCESS_CONTROL_DEFAULT_PRINCIPAL': 'ci-runner'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            with patch('app.auth.access.resolve_request_principal', return_value=None):
                assert resolve_access_scope() == AccessScope('ci-runner', frozenset())

    def test_jwt_token_gives_principal_scope(self) -> None:
        """A verified jwt token scopes the request to its subject and groups, without its roles."""
        token = AccessToken(
            token='opaque-token-value',
            client_id='mcp-client',
            scopes=[],
            expires_at=None,
            claims={'sub': 'alice', 'groups': ['team-a', 'team-b'], 'roles': ['publisher']},
        )
        with patch.dict(os.environ, {'MCP_AUTH_PROVIDER': 'jwt'}, clear=False):
            get_settings.cache_clear()
            with patch('app.auth.principal.get_access_token', return_value=token):
                scope = resolve_access_scope()
        assert scope == AccessScope('alice', frozenset({'team-a', 'team-b'}))

    def test_simple_token_request_gives_default_principal_scope(self) -> None:
        """A simple_token request reads and writes as the default principal, not as its client id."""
        token = AccessToken(
            token='opaque-token-value',
            client_id='mcp-client',
            scopes=[],
            expires_at=None,
            claims={},
        )
        env = {'MCP_AUTH_PROVIDER': 'simple_token', 'MCP_AUTH_TOKEN': 'test-token'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            with patch('app.auth.principal.get_access_token', return_value=token):
                assert resolve_access_scope() == AccessScope('local', frozenset())

    def test_as_principal_scopes_read_and_write_resolution(self) -> None:
        """The as_principal test helper sets the identity both resolvers return."""
        with as_principal('bob', groups=['team-x'], roles=['publisher']):
            scope = resolve_access_scope()
            principal = resolve_effective_principal()
        assert scope == AccessScope('bob', frozenset({'team-x'}))
        assert principal == RequestPrincipal(
            principal_id='bob',
            groups=frozenset({'team-x'}),
            roles=frozenset({'publisher'}),
        )

    def test_exported_from_auth_package_without_scope_types(self) -> None:
        """app.auth exports the resolver; the scope types stay in their standard-library-only module."""
        assert app.auth.resolve_access_scope is resolve_access_scope
        assert 'resolve_access_scope' in app.auth.__all__
        assert {'AccessScope', 'SystemScope', 'SYSTEM_SCOPE', 'Scope'}.isdisjoint(app.auth.__all__)


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

    def test_private_is_never_gated(self) -> None:
        """private is always allowed, publish role or not."""
        env = {'ACCESS_CONTROL_PUBLISH_ROLE': 'publisher'}
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            assert visibility_denied_reason('private', self._principal()) is None

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
