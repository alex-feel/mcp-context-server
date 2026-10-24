"""The read-only caller shared by the write tool tests."""

import os
from collections.abc import Generator
from contextlib import contextmanager
from unittest.mock import patch

from fastmcp.server.auth import AccessToken

from app.settings import get_settings
from tests.helpers import as_principal


@contextmanager
def read_only_caller() -> Generator[None, None, None]:
    """Run the block as the read-only principal ``anonymous``, holding a group and the publish role.

    ``ACCESS_CONTROL_ANONYMOUS_PRINCIPAL`` names ``anonymous`` in freshly built
    settings inside the block, and the request's token carries an expiry, as the
    read-only principal's token must.

    Yields:
        None.
    """
    token = AccessToken(token='anonymous-token', client_id='anonymous', scopes=[], expires_at=4102444800)
    with patch.dict(os.environ, {'ACCESS_CONTROL_ANONYMOUS_PRINCIPAL': 'anonymous'}):
        get_settings.cache_clear()
        try:
            with (
                as_principal('anonymous', groups=['team-x'], roles=['publisher']),
                patch('app.auth.access.get_access_token', return_value=token),
            ):
                yield
        finally:
            get_settings.cache_clear()
