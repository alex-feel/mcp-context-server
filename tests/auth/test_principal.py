"""Tests for per-request principal resolution.

Covers app.auth.principal.resolve_request_principal against mocked FastMCP
access tokens: the no-token case, JWT subject and claim mapping, the
client-id fallback for tokens without a sub claim, and configured claim keys.
"""

import dataclasses
import os
from unittest.mock import patch

import pytest
from fastmcp.server.auth import AccessToken

from app.auth.principal import RequestPrincipal
from app.auth.principal import resolve_request_principal
from app.settings import get_settings


def _token(claims: dict[str, object], client_id: str = 'mcp-client') -> AccessToken:
    """Build a FastMCP AccessToken carrying the given verified claims."""
    return AccessToken(
        token='opaque-token-value',
        client_id=client_id,
        scopes=[],
        expires_at=None,
        claims=claims,
    )


class TestResolveRequestPrincipal:
    """Tests for resolve_request_principal."""

    def setup_method(self) -> None:
        """Reset the settings singleton so per-test env vars take effect."""
        get_settings.cache_clear()

    def teardown_method(self) -> None:
        """Drop settings built from per-test env vars."""
        get_settings.cache_clear()

    def test_no_token_returns_none(self) -> None:
        """Without a verified access token there is no principal."""
        with patch('app.auth.principal.get_access_token', return_value=None):
            assert resolve_request_principal() is None

    def test_jwt_claims_resolve_to_principal(self) -> None:
        """A JWT-shaped token maps sub, groups, and roles into the principal."""
        token = _token({'sub': 'alice', 'groups': ['team-a', 'team-b'], 'roles': ['publisher']})
        with patch('app.auth.principal.get_access_token', return_value=token):
            principal = resolve_request_principal()
        assert principal == RequestPrincipal(
            principal_id='alice',
            groups=frozenset({'team-a', 'team-b'}),
            roles=frozenset({'publisher'}),
        )

    def test_empty_claims_fall_back_to_client_id(self) -> None:
        """A token with empty claims (simple_token) uses the client id as principal."""
        token = _token({}, client_id='ci-service')
        with patch('app.auth.principal.get_access_token', return_value=token):
            principal = resolve_request_principal()
        assert principal is not None
        assert principal.principal_id == 'ci-service'
        assert principal.groups == frozenset()
        assert principal.roles == frozenset()

    def test_non_string_sub_falls_back_to_client_id(self) -> None:
        """A non-string or empty sub claim falls back to the client id."""
        for bad_sub in (12345, ''):
            token = _token({'sub': bad_sub})
            with patch('app.auth.principal.get_access_token', return_value=token):
                principal = resolve_request_principal()
            assert principal is not None
            assert principal.principal_id == 'mcp-client'

    def test_configured_claim_keys_are_honored(self) -> None:
        """MCP_AUTH_GROUPS_CLAIM / MCP_AUTH_ROLES_CLAIM select the claim keys."""
        env = {
            'MCP_AUTH_GROUPS_CLAIM': 'https://example.com/groups',
            'MCP_AUTH_ROLES_CLAIM': 'realm_access.roles',
        }
        token = _token({
            'sub': 'bob',
            'https://example.com/groups': ['kb-readers'],
            'realm_access': {'roles': ['admin']},
            'groups': ['ignored-default-claim'],
        })
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            with patch('app.auth.principal.get_access_token', return_value=token):
                principal = resolve_request_principal()
        assert principal is not None
        assert principal.groups == frozenset({'kb-readers'})
        assert principal.roles == frozenset({'admin'})

    def test_entra_overage_yields_empty_groups(self) -> None:
        """Overage markers on a token resolve to an empty group set, fail-closed."""
        token = _token({
            'sub': 'carol',
            '_claim_names': {'groups': 'src1'},
            '_claim_sources': {'src1': {'endpoint': 'https://graph.microsoft.com/v1.0/me/getMemberObjects'}},
        })
        with patch('app.auth.principal.get_access_token', return_value=token):
            principal = resolve_request_principal()
        assert principal is not None
        assert principal.principal_id == 'carol'
        assert principal.groups == frozenset()

    def test_principal_is_immutable(self) -> None:
        """RequestPrincipal is a frozen value object."""
        principal = RequestPrincipal(principal_id='p', groups=frozenset(), roles=frozenset())
        field_name = 'principal_id'
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(principal, field_name, 'other')
