"""Fixtures shared by the SQLite and PostgreSQL integration tests."""

import pytest
from fastmcp.server.auth.providers.jwt import RSAKeyPair


@pytest.fixture(scope='module')
def jwt_key_pair() -> RSAKeyPair:
    """Generate one RSA key pair shared by every JWT test in a module.

    Returns:
        The key pair; servers verify its public key, tests mint tokens with it.
    """
    return RSAKeyPair.generate()
