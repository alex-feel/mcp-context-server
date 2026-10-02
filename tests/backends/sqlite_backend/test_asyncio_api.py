"""Regression test for the asyncio API used inside the ``app/backends/sqlite_backend/`` package.

The production backend modules MUST use :func:`asyncio.get_running_loop`
exclusively, because every call site executes inside an ``async def``
method (an event loop is guaranteed to be running). The deprecated
:func:`asyncio.get_event_loop` would emit ``DeprecationWarning`` on
Python 3.12+ and silently spin up a new loop in environments where no
loop is bound, which is exactly the wrong behavior for the backend.
"""

from pathlib import Path

_SQLITE_BACKEND_PACKAGE = (
    Path(__file__).resolve().parent.parent.parent.parent
    / 'app'
    / 'backends'
    / 'sqlite_backend'
)


def _package_sources() -> dict[str, str]:
    """Read every module of the SQLite backend package.

    Returns:
        Module source text keyed by file name.
    """
    sources = {path.name: path.read_text(encoding='utf-8') for path in sorted(_SQLITE_BACKEND_PACKAGE.glob('*.py'))}
    assert sources, f'no modules found under {_SQLITE_BACKEND_PACKAGE}'
    return sources


def test_no_get_event_loop_callsites_in_sqlite_backend() -> None:
    """``asyncio.get_event_loop()`` must not appear in any ``sqlite_backend`` module."""
    offenders = [name for name, source in _package_sources().items() if 'asyncio.get_event_loop()' in source]
    assert not offenders, (
        f'app/backends/sqlite_backend/ modules {offenders} contain a deprecated '
        'asyncio.get_event_loop() call. Use asyncio.get_running_loop() '
        'instead -- every callsite executes inside an async def method.'
    )


def test_get_running_loop_used_in_sqlite_backend() -> None:
    """Confirm ``asyncio.get_running_loop()`` is the only loop accessor."""
    sources = _package_sources()
    assert any('asyncio.get_running_loop()' in source for source in sources.values()), (
        'app/backends/sqlite_backend/ is expected to use '
        'asyncio.get_running_loop() for executor scheduling.'
    )
