"""Tests for the anonymous principal's static-key verifier and its composition.

On a JWKS deployment the auth factory composes app.auth.anonymous's
AnonymousSubjectVerifier beside the JWKS verifier through fastmcp's MultiAuth:
the order of the two verifiers, the configuration it refuses, the subject it
accepts, and the issuer and audience it pins.
"""

import json
import logging
import os
import socket
import time
from collections.abc import Iterator
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from fastmcp.server.auth import MultiAuth
from fastmcp.server.auth.providers.jwt import JWTVerifier
from fastmcp.server.auth.providers.jwt import RSAKeyPair
from joserfc import jwt
from joserfc.jwk import OctKey

from app.settings import get_settings


@pytest.fixture(scope='module')
def rsa_key_pair() -> RSAKeyPair:
    """Generate the identity provider's key pair, which the anonymous verifier must not trust."""
    return RSAKeyPair.generate()


@pytest.fixture(scope='module')
def anonymous_key_pair() -> RSAKeyPair:
    """Generate the operator-held key pair the anonymous verifier trusts."""
    return RSAKeyPair.generate()


@pytest.fixture
def closed_jwks_uri() -> str:
    """Return a JWKS URI on a local port nothing listens on, so a JWKS lookup can only fail.

    Returns:
        An http URI on 127.0.0.1 whose port was free a moment ago.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(('127.0.0.1', 0))
        port = probe.getsockname()[1]
    return f'http://127.0.0.1:{port}/jwks.json'


class TestAnonymousVerifierComposition:
    """The operator-held anonymous verifier composed beside a JWKS verifier.

    The JWKS URI points at a closed local port throughout, so any token these
    tests see verified was verified by the anonymous static-key verifier alone.
    """

    @pytest.fixture(autouse=True)
    def clear_settings_cache(self) -> Iterator[None]:
        """Build fresh settings for each test and drop them afterwards."""
        get_settings.cache_clear()
        yield
        get_settings.cache_clear()

    @staticmethod
    def _env(**overrides: str) -> dict[str, str]:
        """Build a jwt environment free of inherited auth and access-control variables."""
        env = {k: v for k, v in os.environ.items() if not k.startswith(('MCP_AUTH', 'ACCESS_CONTROL'))}
        env['MCP_AUTH_PROVIDER'] = 'jwt'
        env.update(overrides)
        return env

    def _composed_env(self, closed_jwks_uri: str, anonymous_key_pair: RSAKeyPair, **overrides: str) -> dict[str, str]:
        """Build the environment of a JWKS deployment that also accepts the anonymous token."""
        return self._env(
            MCP_AUTH_JWT_JWKS_URI=closed_jwks_uri,
            MCP_AUTH_JWT_ISSUER='https://issuer.test',
            MCP_AUTH_JWT_AUDIENCE='ctx-server',
            MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY=anonymous_key_pair.public_key,
            ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous',
            **overrides,
        )

    @staticmethod
    def _anonymous_token(anonymous_key_pair: RSAKeyPair, subject: str = 'anonymous') -> str:
        """Mint a token with the anonymous key for the pinned issuer and audience."""
        return anonymous_key_pair.create_token(subject=subject, issuer='https://issuer.test', audience='ctx-server')

    def test_factory_composes_the_anonymous_verifier_before_the_jwks_verifier(
        self, closed_jwks_uri: str, anonymous_key_pair: RSAKeyPair,
    ) -> None:
        """The key with a JWKS URI yields MultiAuth over the anonymous verifier, then the JWKS verifier."""
        with patch.dict(os.environ, self._composed_env(closed_jwks_uri, anonymous_key_pair), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.auth.anonymous import AnonymousSubjectVerifier

            provider = create_auth_provider()

        assert isinstance(provider, MultiAuth)
        assert provider.server is None
        anonymous, primary = provider.verifiers
        assert isinstance(anonymous, AnonymousSubjectVerifier)
        assert type(primary) is JWTVerifier
        assert primary.jwks_uri == closed_jwks_uri
        assert anonymous.subject == 'anonymous'
        assert anonymous.public_key == anonymous_key_pair.public_key.strip()
        assert anonymous.jwks_uri is None
        assert anonymous.algorithm == 'RS256'
        assert (anonymous.issuer, anonymous.audience) == (primary.issuer, primary.audience) == (
            'https://issuer.test', 'ctx-server',
        )

    def test_factory_applies_the_anonymous_algorithm(self, closed_jwks_uri: str) -> None:
        """MCP_AUTH_ANONYMOUS_JWT_ALGORITHM selects the anonymous verifier's algorithm, not the JWKS one."""
        env = self._env(
            MCP_AUTH_JWT_JWKS_URI=closed_jwks_uri,
            MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY='anonymous-shared-secret',
            MCP_AUTH_ANONYMOUS_JWT_ALGORITHM='HS512',
            ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()

        assert isinstance(provider, MultiAuth)
        anonymous, primary = provider.verifiers
        assert isinstance(anonymous, JWTVerifier)
        assert anonymous.algorithm == 'HS512'
        assert isinstance(primary, JWTVerifier)
        assert primary.algorithm == 'RS256'

    def test_factory_logs_the_composition(
        self, closed_jwks_uri: str, anonymous_key_pair: RSAKeyPair, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The startup log names both verifiers, their algorithms and the anonymous subject."""
        env = self._composed_env(closed_jwks_uri, anonymous_key_pair, MCP_AUTH_JWT_ALGORITHM='ES256')
        with patch.dict(os.environ, env, clear=True), caplog.at_level(logging.INFO, logger='app.auth'):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            create_auth_provider()

        assert (
            'Authentication: JWTVerifier (JWKS endpoint, algorithm ES256) composed with the anonymous '
            'static-key verifier (algorithm RS256, subject anonymous)'
        ) in caplog.messages

    @pytest.mark.parametrize('key', ['', '   '])
    def test_blank_anonymous_key_keeps_the_single_verifier(self, key: str, closed_jwks_uri: str) -> None:
        """An empty or whitespace-only anonymous key counts as unset: the JWKS verifier stands alone."""
        env = self._env(
            MCP_AUTH_JWT_JWKS_URI=closed_jwks_uri,
            MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY=key,
            ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()

        assert type(provider) is JWTVerifier

    def test_anonymous_key_with_a_static_primary_key_is_refused(
        self, rsa_key_pair: RSAKeyPair, anonymous_key_pair: RSAKeyPair,
    ) -> None:
        """A static-key deployment signs the anonymous token with its own key, so a second key is a misconfiguration."""
        env = self._env(
            MCP_AUTH_JWT_PUBLIC_KEY=rsa_key_pair.public_key,
            MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY=anonymous_key_pair.public_key,
            ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError) as excinfo:
                create_auth_provider()

        assert str(excinfo.value) == (
            'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY applies to MCP_AUTH_JWT_JWKS_URI deployments; '
            'with MCP_AUTH_JWT_PUBLIC_KEY, sign the anonymous token with that key'
        )

    def test_anonymous_key_without_the_anonymous_principal_is_refused(
        self, closed_jwks_uri: str, anonymous_key_pair: RSAKeyPair,
    ) -> None:
        """The anonymous verifier accepts only ACCESS_CONTROL_ANONYMOUS_PRINCIPAL, so the key needs it set."""
        env = self._env(
            MCP_AUTH_JWT_JWKS_URI=closed_jwks_uri,
            MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY=anonymous_key_pair.public_key,
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError) as excinfo:
                create_auth_provider()

        assert str(excinfo.value) == (
            'MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY requires ACCESS_CONTROL_ANONYMOUS_PRINCIPAL, '
            'the only subject the anonymous verifier accepts'
        )

    @pytest.mark.parametrize(
        ('algorithm', 'message'),
        [('none', 'Unsupported algorithm'), ('HS256', 'shared secret')],
        ids=['unsupported-algorithm', 'symmetric-algorithm-with-pem-key'],
    )
    def test_anonymous_verifier_configuration_rejected_by_the_verifier_is_refused(
        self, algorithm: str, message: str, closed_jwks_uri: str, anonymous_key_pair: RSAKeyPair,
    ) -> None:
        """A PEM key with an algorithm the JWT verifier rejects for it is a startup misconfiguration (exit 78)."""
        env = self._composed_env(closed_jwks_uri, anonymous_key_pair, MCP_AUTH_ANONYMOUS_JWT_ALGORITHM=algorithm)
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider
            from app.errors import ConfigurationError

            with pytest.raises(ConfigurationError, match=message):
                create_auth_provider()

    @pytest.mark.asyncio
    async def test_anonymous_token_is_verified_without_a_jwks_lookup(
        self, closed_jwks_uri: str, anonymous_key_pair: RSAKeyPair,
    ) -> None:
        """The anonymous verifier answers first, so the JWKS endpoint is never contacted for its token."""
        with patch.dict(os.environ, self._composed_env(closed_jwks_uri, anonymous_key_pair), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()
        assert isinstance(provider, MultiAuth)
        primary = next(v for v in provider.verifiers if isinstance(v, JWTVerifier) and v.jwks_uri)

        with patch.object(primary, '_fetch_jwks', new=AsyncMock(side_effect=AssertionError('JWKS fetched'))) as fetch:
            access_token = await provider.verify_token(self._anonymous_token(anonymous_key_pair))

        fetch.assert_not_awaited()
        assert access_token is not None
        assert access_token.subject == 'anonymous'
        assert access_token.claims['sub'] == 'anonymous'
        assert access_token.expires_at is not None
        assert json.loads(access_token.client_id)[0] == 'anonymous'
        assert access_token.original_client_id == 'anonymous'

    @pytest.mark.asyncio
    async def test_symmetric_anonymous_token_is_verified(self, closed_jwks_uri: str) -> None:
        """An HS* anonymous verifier checks the shared secret the operator mints with."""
        env = self._env(
            MCP_AUTH_JWT_JWKS_URI=closed_jwks_uri,
            MCP_AUTH_JWT_AUDIENCE='ctx-server',
            MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY='anonymous-shared-secret-of-sufficient-length',
            MCP_AUTH_ANONYMOUS_JWT_ALGORITHM='HS256',
            ACCESS_CONTROL_ANONYMOUS_PRINCIPAL='anonymous',
        )
        with patch.dict(os.environ, env, clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()
        assert provider is not None

        now = int(time.time())
        claims = {'sub': 'anonymous', 'aud': 'ctx-server', 'iat': now, 'exp': now + 600}
        secret = OctKey.import_key('anonymous-shared-secret-of-sufficient-length')
        access_token = await provider.verify_token(jwt.encode({'alg': 'HS256'}, claims, secret))

        assert access_token is not None
        assert access_token.subject == 'anonymous'

    @pytest.mark.asyncio
    async def test_static_key_token_for_another_subject_is_rejected(
        self, closed_jwks_uri: str, anonymous_key_pair: RSAKeyPair, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """The anonymous key mints no other principal: a token for any other subject verifies nowhere."""
        with patch.dict(os.environ, self._composed_env(closed_jwks_uri, anonymous_key_pair), clear=True):
            get_settings.cache_clear()
            from app.auth import create_auth_provider

            provider = create_auth_provider()
        assert provider is not None

        with caplog.at_level(logging.INFO, logger='app.auth.anonymous'):
            access_token = await provider.verify_token(self._anonymous_token(anonymous_key_pair, subject='alice'))

        assert access_token is None
        assert 'Bearer token rejected: the anonymous verifier accepts only its configured subject' in caplog.messages

    @pytest.mark.asyncio
    async def test_token_signed_by_another_key_is_rejected_without_a_subject_log(
        self, rsa_key_pair: RSAKeyPair, anonymous_key_pair: RSAKeyPair, caplog: pytest.LogCaptureFixture,
    ) -> None:
        """An identity-provider token fails the anonymous signature check and logs nothing at INFO."""
        from app.auth.anonymous import AnonymousSubjectVerifier

        verifier = AnonymousSubjectVerifier(
            subject='anonymous',
            public_key=anonymous_key_pair.public_key,
            algorithm='RS256',
            issuer='https://issuer.test',
            audience='ctx-server',
        )
        idp_token = rsa_key_pair.create_token(subject='anonymous', issuer='https://issuer.test', audience='ctx-server')

        with caplog.at_level(logging.INFO, logger='app.auth.anonymous'):
            assert await verifier.load_access_token(idp_token) is None

        assert not [record for record in caplog.records if record.name == 'app.auth.anonymous']

    @pytest.mark.asyncio
    async def test_anonymous_verifier_enforces_the_pinned_audience(self, anonymous_key_pair: RSAKeyPair) -> None:
        """The anonymous verifier pins the same issuer and audience as the primary verifier."""
        from app.auth.anonymous import AnonymousSubjectVerifier

        verifier = AnonymousSubjectVerifier(
            subject='anonymous',
            public_key=anonymous_key_pair.public_key,
            algorithm='RS256',
            issuer='https://issuer.test',
            audience='ctx-server',
        )
        token = anonymous_key_pair.create_token(subject='anonymous', issuer='https://issuer.test', audience='other')

        assert await verifier.load_access_token(token) is None
