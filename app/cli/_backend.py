"""Backend construction and bounded shutdown shared by the maintenance CLI modes."""

import asyncio
import contextlib

from app.backends import StorageBackend
from app.backends import create_backend
from app.cli._database_url import parse_backend_url


def make_backend(
    source_url: str, *, provision_vector: bool | None = None,
) -> StorageBackend:
    """Build a backend pointed at ``source_url``.

    Args:
        source_url: URL passed to ``--source-url``.
        provision_vector: Explicit pgvector-provisioning decision forwarded to
            the PostgreSQL backend (ignored for SQLite, whose sqlite-vec load is
            unconditional). ``None`` lets the backend resolve it from settings
            and a probe at ``initialize()`` time; ``False`` skips ``CREATE
            EXTENSION vector`` so the zero-data reverse path runs on a
            pgvector-less host; ``True`` provisions the extension for the fp32
            rebuild. See
            :func:`app.cli.migrate_compression.decompress._decompress_needs_vector`.

    Returns:
        Constructed (not yet initialized) :class:`StorageBackend` matching the
        URL scheme. The URL classification is performed by
        :func:`parse_backend_url` which raises ``ValueError`` for unrecognized
        schemes; the exception propagates to the caller.
    """
    backend_kind, address = parse_backend_url(source_url)
    if backend_kind == 'sqlite':
        return create_backend(backend_type='sqlite', db_path=address)
    return create_backend(
        backend_type='postgresql',
        connection_string=address,
        provision_vector=provision_vector,
    )


async def shutdown_backend(backend: StorageBackend) -> None:
    """Suppress timeouts during backend shutdown.

    A long-running CLI invocation may keep PG sessions open until the
    process exits; bound the wait so a stuck connection does not block
    teardown.
    """
    with contextlib.suppress(TimeoutError):
        await asyncio.wait_for(backend.shutdown(), timeout=10.0)
