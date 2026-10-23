"""Static-key verifier for the operator-minted anonymous token on a JWKS deployment.

A JWKS deployment trusts only tokens its identity provider signs, so the
operator cannot mint the shared read-only token for
ACCESS_CONTROL_ANONYMOUS_PRINCIPAL there. The auth factory composes this
verifier beside the JWKS verifier through fastmcp's ``MultiAuth``: it checks the
operator-held key configured in MCP_AUTH_ANONYMOUS_JWT_PUBLIC_KEY, pins the same
issuer and audience as the JWKS verifier, and accepts exactly one subject, so the
operator's key can mint the anonymous principal and nobody else.

The factory imports this module only inside its jwt arm, because the JWT
verifier pulls in joserfc and cryptography, which no other startup needs.
"""

import logging
from typing import override

from fastmcp.server.auth import AccessToken
from fastmcp.server.auth.providers.jwt import JWTVerifier

logger = logging.getLogger(__name__)


class AnonymousSubjectVerifier(JWTVerifier):
    """JWT verifier that accepts only tokens for one configured subject.

    Signature, expiry, issuer and audience are checked exactly as the parent
    verifier checks them; a valid token for any other subject is rejected.
    """

    def __init__(
        self,
        *,
        subject: str,
        public_key: str,
        algorithm: str,
        issuer: str | None,
        audience: str | None,
    ) -> None:
        """Configure the verifier for one subject and one static key.

        The parent verifier raises ValueError for a key and algorithm it
        rejects (an unsupported algorithm, or a PEM key with an HS* algorithm).

        Args:
            subject: The only ``sub`` claim value the verifier accepts.
            public_key: PEM-encoded public key, or the shared secret of an HS* algorithm.
            algorithm: The signing algorithm to accept.
            issuer: Expected ``iss`` claim value, or None to skip the issuer check.
            audience: Expected ``aud`` claim value, or None to skip the audience check.
        """
        super().__init__(public_key=public_key, issuer=issuer, audience=audience, algorithm=algorithm)
        self._subject = subject

    @property
    def subject(self) -> str:
        """The only subject this verifier accepts."""
        return self._subject

    @override
    async def load_access_token(self, token: str) -> AccessToken | None:
        """Verify the token, then accept it only for the configured subject.

        Args:
            token: The bearer token to verify.

        Returns:
            The verified access token when it is valid and its ``sub`` equals the
            configured subject, else None.
        """
        result = await super().load_access_token(token)
        if result is None:
            return None
        if result.subject != self._subject:
            logger.info('Bearer token rejected: the anonymous verifier accepts only its configured subject')
            return None
        return result
