"""Tests for access-control policy helpers.

Covers app.auth.access: resolve_effective_principal (verified-token passthrough
and the fallback to the configured default principal for requests without a jwt
identity, simple_token included), resolve_access_scope (the same identity as
the scope every read runs under, roles excluded), may_publish and
visibility_denied_reason (the ACCESS_CONTROL_PUBLISH_ROLE publish gate, one rule
for both), resolve_write_scope (the write tools' scope, which carries the
caller's publish permission) and the read-only principal named by
ACCESS_CONTROL_ANONYMOUS_PRINCIPAL (is_read_only, the groups-and-roles discard,
the refusal of its token without an expiry, reject_read_only_principal and the
startup log line).
"""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import patch

import pytest
from fastmcp.exceptions import ToolError
from fastmcp.server.auth import AccessToken

import app.auth
from app.access_scope import AccessScope
from app.auth.access import is_read_only
from app.auth.access import log_read_only_principal
from app.auth.access import may_publish
from app.auth.access import reject_read_only_principal
from app.auth.access import resolve_access_scope
from app.auth.access import resolve_effective_principal
from app.auth.access import resolve_write_scope
from app.auth.access import visibility_denied_reason
from app.auth.principal import RequestPrincipal
from app.settings import get_settings
from app.settings.auth import AccessControlSettings
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


_PUBLISH_CASES = [
    pytest.param(None, frozenset[str](), True, id='role-unset'),
    pytest.param('publisher', frozenset({'publisher', 'other'}), True, id='role-held'),
    pytest.param('publisher', frozenset({'other'}), False, id='role-missing'),
]


class TestPublishPermission:
    """Tests for may_publish and resolve_write_scope."""

    def setup_method(self) -> None:
        """Reset the settings singleton so per-test env vars take effect."""
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        """Drop settings built from per-test env vars."""
        get_settings.cache_clear()

    @staticmethod
    def _env(publish_role: str | None) -> dict[str, str]:
        return {} if publish_role is None else {'ACCESS_CONTROL_PUBLISH_ROLE': publish_role}

    @pytest.mark.parametrize(('publish_role', 'roles', 'expected'), _PUBLISH_CASES)
    def test_may_publish_follows_the_publish_role(
        self, publish_role: str | None, roles: frozenset[str], expected: bool,
    ) -> None:
        """Any caller may publish when no role is configured; otherwise only a caller holding it."""
        principal = RequestPrincipal(principal_id='p', groups=frozenset(), roles=roles)
        with patch.dict(os.environ, self._env(publish_role), clear=False):
            get_settings.cache_clear()
            assert may_publish(principal) is expected
            assert (visibility_denied_reason('public', principal) is None) is expected

    @pytest.mark.parametrize(('publish_role', 'roles', 'expected'), _PUBLISH_CASES)
    def test_write_scope_carries_the_publish_permission(
        self, publish_role: str | None, roles: frozenset[str], expected: bool,
    ) -> None:
        """The write scope is the principal's id and groups plus whether it may publish."""
        principal = RequestPrincipal(principal_id='alice', groups=frozenset({'team-a'}), roles=roles)
        with patch.dict(os.environ, self._env(publish_role), clear=False):
            get_settings.cache_clear()
            scope = resolve_write_scope(principal)
        assert scope == AccessScope('alice', frozenset({'team-a'}), may_publish=expected)

    def test_read_scope_never_asserts_the_publish_permission(self) -> None:
        """resolve_access_scope serves reads, which publish nothing, so it leaves the permission unset."""
        with as_principal('alice', roles=['publisher']):
            scope = resolve_access_scope()
            principal = resolve_effective_principal()
        assert may_publish(principal)
        assert scope.may_publish is False

    def test_write_scope_is_exported_from_auth_package(self) -> None:
        """The write tools import resolve_write_scope from app.auth."""
        assert app.auth.resolve_write_scope is resolve_write_scope
        assert 'resolve_write_scope' in app.auth.__all__


_READ_ONLY_REFUSAL = 'The anonymous principal is read-only: {tool} is not permitted'
_EXPIRY_REFUSAL = "The anonymous principal's token carries no expiry (exp); mint it with a bounded exp"
_WRITE_TOOLS = (
    'store_context',
    'store_context_batch',
    'update_context',
    'update_context_batch',
    'delete_context',
    'delete_context_batch',
)


@contextmanager
def _settings(**env: str) -> Generator[None, None, None]:
    """Run the block with the given environment in freshly built settings."""
    with patch.dict(os.environ, env, clear=False):
        get_settings.cache_clear()
        try:
            yield
        finally:
            get_settings.cache_clear()


def _jwt_token(subject: str, *, expires_at: int | None) -> AccessToken:
    """Build a verified jwt access token for one subject, with or without an expiry."""
    return AccessToken(
        token='opaque-token-value',
        client_id=subject,
        scopes=[],
        expires_at=expires_at,
        claims={'sub': subject, 'groups': ['team-x'], 'roles': ['publisher']},
    )


@contextmanager
def _verified_jwt(token: AccessToken) -> Generator[None, None, None]:
    """Serve one token to both lookups of the current request's access token."""
    with (
        patch('app.auth.principal.get_access_token', return_value=token),
        patch('app.auth.access.get_access_token', return_value=token),
    ):
        yield


class TestReadOnlyPrincipal:
    """Tests for the read-only principal named by ACCESS_CONTROL_ANONYMOUS_PRINCIPAL."""

    def setup_method(self) -> None:
        """Reset the settings singleton so per-test env vars take effect."""
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        """Drop settings built from per-test env vars."""
        get_settings.cache_clear()

    @staticmethod
    def _principal(principal_id: str = 'anonymous') -> RequestPrincipal:
        return RequestPrincipal(
            principal_id=principal_id,
            groups=frozenset({'team-x'}),
            roles=frozenset({'publisher'}),
        )

    def test_unset_setting_makes_no_principal_read_only(self) -> None:
        """Unset, the setting binds nobody: the anonymous subject is an ordinary principal."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ACCESS_CONTROL_ANONYMOUS_PRINCIPAL', None)
            get_settings.cache_clear()
            assert is_read_only(self._principal()) is False
            with as_principal('anonymous', groups=['team-x'], roles=['publisher']):
                assert resolve_effective_principal() == self._principal()

    def test_matching_principal_is_read_only(self) -> None:
        """The principal whose id equals the setting verbatim is read-only."""
        with _settings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'):
            assert is_read_only(self._principal()) is True

    @pytest.mark.parametrize('principal_id', ['alice', 'Anonymous', 'anonymous ', 'local'])
    def test_other_principals_are_not_read_only(self, principal_id: str) -> None:
        """Only a verbatim match binds: case, padding and the default principal never do."""
        with _settings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'):
            assert is_read_only(self._principal(principal_id)) is False

    def test_default_principal_is_unaffected(self) -> None:
        """A request without a jwt identity keeps resolving to the writable default principal."""
        with (
            _settings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'),
            patch('app.auth.access.resolve_request_principal', return_value=None),
        ):
            principal = resolve_effective_principal()
            assert principal == RequestPrincipal(principal_id='local', groups=frozenset(), roles=frozenset())
            assert is_read_only(principal) is False
            reject_read_only_principal(principal, tool='store_context')

    def test_read_only_principal_loses_its_groups_and_roles(self) -> None:
        """A mis-minted token carries neither group access nor the publish role for this principal."""
        with (
            _settings(MCP_AUTH_PROVIDER='jwt', ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'),
            _verified_jwt(_jwt_token('anonymous', expires_at=4102444800)),
        ):
            principal = resolve_effective_principal()
            scope = resolve_access_scope()
        assert principal == RequestPrincipal(principal_id='anonymous', groups=frozenset(), roles=frozenset())
        assert scope == AccessScope('anonymous', frozenset())

    def test_read_only_principal_without_a_readable_token_is_refused(self) -> None:
        """Without a token to prove an expiry, the read-only principal is refused rather than trusted."""
        with (
            _settings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'),
            as_principal('anonymous'),
            patch('app.auth.access.get_access_token', return_value=None),
            pytest.raises(ToolError) as refusal,
        ):
            resolve_effective_principal()
        assert str(refusal.value) == _EXPIRY_REFUSAL

    def test_other_principals_keep_their_groups_and_roles(self) -> None:
        """The discard applies to the read-only principal alone."""
        with (
            _settings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'),
            as_principal('alice', groups=['team-x'], roles=['publisher']),
        ):
            principal = resolve_effective_principal()
        assert principal == self._principal('alice')

    @pytest.mark.parametrize('publish_role', [None, 'publisher'])
    def test_read_only_principal_may_never_publish(self, publish_role: str | None) -> None:
        """Neither an unset publish role nor a roles claim carrying it lets the read-only principal publish."""
        env = {'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL': 'anonymous'}
        if publish_role is not None:
            env['ACCESS_CONTROL_PUBLISH_ROLE'] = publish_role
        with _settings(**env):
            principal = self._principal()
            assert may_publish(principal) is False
            assert resolve_write_scope(principal).may_publish is False
            assert may_publish(self._principal('alice')) is True

    def test_read_only_token_without_expiry_is_refused(self) -> None:
        """The read-only principal's token must carry exp, on every tool, reads included."""
        with (
            _settings(MCP_AUTH_PROVIDER='jwt', ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'),
            _verified_jwt(_jwt_token('anonymous', expires_at=None)),
        ):
            with pytest.raises(ToolError) as effective:
                resolve_effective_principal()
            with pytest.raises(ToolError) as scoped:
                resolve_access_scope()
        assert str(effective.value) == _EXPIRY_REFUSAL
        assert str(scoped.value) == _EXPIRY_REFUSAL

    def test_read_only_token_with_expiry_is_accepted(self) -> None:
        """A bounded read-only token resolves to the principal with no groups and no roles."""
        with (
            _settings(MCP_AUTH_PROVIDER='jwt', ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'),
            _verified_jwt(_jwt_token('anonymous', expires_at=4102444800)),
        ):
            principal = resolve_effective_principal()
        assert principal == RequestPrincipal(principal_id='anonymous', groups=frozenset(), roles=frozenset())

    def test_other_principal_token_without_expiry_is_accepted(self) -> None:
        """Every other principal keeps fastmcp's behavior: a token without exp stays valid."""
        with (
            _settings(MCP_AUTH_PROVIDER='jwt', ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'),
            _verified_jwt(_jwt_token('alice', expires_at=None)),
        ):
            principal = resolve_effective_principal()
        assert principal == self._principal('alice')

    @pytest.mark.parametrize('tool', _WRITE_TOOLS)
    def test_reject_names_the_refused_tool(self, tool: str) -> None:
        """The refusal text names the tool and depends on nothing else."""
        with _settings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'), pytest.raises(ToolError) as refusal:
            reject_read_only_principal(self._principal(), tool=tool)
        assert str(refusal.value) == _READ_ONLY_REFUSAL.format(tool=tool)

    def test_reject_lets_every_other_principal_through(self) -> None:
        """A writable principal passes the gate."""
        with _settings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous'):
            reject_read_only_principal(self._principal('alice'), tool='store_context')

    def test_reject_lets_everyone_through_when_unset(self) -> None:
        """Unset, the gate refuses nobody, the anonymous subject included (the fail-open property)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('ACCESS_CONTROL_ANONYMOUS_PRINCIPAL', None)
            get_settings.cache_clear()
            reject_read_only_principal(self._principal(), tool='store_context')

    @pytest.mark.parametrize(
        ('configured', 'expected'),
        [('anonymous', 'Read-only principal: anonymous'), (None, 'Read-only principal: none')],
        ids=['set', 'unset'],
    )
    def test_startup_line_names_the_read_only_principal(
        self, configured: str | None, expected: str, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The startup INFO line keeps the fail-open setting visible either way."""
        access = AccessControlSettings(ACCESS_CONTROL_ANONYMOUS_PRINCIPAL=configured)
        with caplog.at_level(logging.INFO, logger='app.auth.access'):
            log_read_only_principal(access)
        records = [record for record in caplog.records if record.name == 'app.auth.access']
        assert [(record.levelno, record.getMessage()) for record in records] == [(logging.INFO, expected)]
