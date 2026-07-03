"""Regression test for the asyncio API used inside ``app/backends/sqlite_backend.py``.

The production backend module MUST use :func:`asyncio.get_running_loop`
exclusively, because every call site executes inside an ``async def``
method (an event loop is guaranteed to be running). The deprecated
:func:`asyncio.get_event_loop` would emit ``DeprecationWarning`` on
Python 3.12+ and silently spin up a new loop in environments where no
loop is bound, which is exactly the wrong behavior for the backend.
"""

from pathlib import Path

_SQLITE_BACKEND_SOURCE = (
    Path(__file__).resolve().parent.parent.parent
    / 'app'
    / 'backends'
    / 'sqlite_backend.py'
)


def test_no_get_event_loop_callsites_in_sqlite_backend() -> None:
    """``asyncio.get_event_loop()`` must not appear in ``sqlite_backend.py``."""
    source = _SQLITE_BACKEND_SOURCE.read_text(encoding='utf-8')
    assert 'asyncio.get_event_loop()' not in source, (
        'app/backends/sqlite_backend.py contains a deprecated '
        'asyncio.get_event_loop() call. Use asyncio.get_running_loop() '
        'instead -- every callsite executes inside an async def method.'
    )


def test_get_running_loop_used_in_sqlite_backend() -> None:
    """Confirm ``asyncio.get_running_loop()`` is the only loop accessor."""
    source = _SQLITE_BACKEND_SOURCE.read_text(encoding='utf-8')
    assert 'asyncio.get_running_loop()' in source, (
        'app/backends/sqlite_backend.py is expected to use '
        'asyncio.get_running_loop() for executor scheduling.'
    )
