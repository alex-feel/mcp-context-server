"""Uninterruptible executor offload for blocking backend calls.

Shared by the SQLite backend's transaction-boundary hops (commit, rollback,
the direct write path) and the repository transaction-body runner. Offloading
a blocking ``sqlite3`` call to the executor keeps the event loop responsive,
but a task cancellation landing on the ``await`` cancels only the ``asyncio``
wrapper future -- the callable already dispatched to a worker thread keeps
running (a still-queued callable can also be cancelled before it runs at all).

For a commit or rollback on the shared pooled writer connection that is a
data-corruption hazard: the cancelled ``await`` lets the writer lock release
while the callable is still in flight, or the rollback never runs because it
was cancelled while queued, so a failed transaction's partial writes stay open
on the ``DEFERRED``-isolation writer and the NEXT write silently commits them.
This helper shields the future and, on any unwind, drains it to completion
before the exception propagates, so the callable has provably finished and the
connection is quiescent before control leaves the offload.
"""

import asyncio
from collections.abc import Callable


async def run_in_executor_uninterruptible[T](
    loop: asyncio.AbstractEventLoop,
    func: Callable[..., T],
    *args: object,
) -> T:
    """Run ``func(*args)`` on the default executor, draining it on cancellation.

    The wrapper future is shielded so a cancellation does not tear down the
    ``await`` early; on any ``BaseException`` unwind the already-dispatched
    callable is drained to completion (re-delivered ``CancelledError`` is
    swallowed so the join is never cut short) before the exception re-raises.
    The drain deliberately NEVER cancels the wrapper future: cancelling an
    ``asyncio`` future succeeds immediately regardless of the running callable,
    which would end the join early and resurrect the very race this guards.

    Args:
        loop: The running event loop.
        func: A blocking callable to run on the executor thread.
        *args: Positional arguments forwarded to ``func``.

    Returns:
        Whatever ``func`` returns.
    """
    future = loop.run_in_executor(None, func, *args)
    try:
        return await asyncio.shield(future)
    except BaseException:
        while not future.done():
            try:
                await asyncio.wait([future])
            except asyncio.CancelledError:
                # Re-cancellation during the drain: keep waiting -- the
                # callable must finish before the unwind may continue.
                continue
        if not future.cancelled():
            # Retrieve (and discard) the callable's own outcome so an
            # exception set during a cancellation unwind does not log a
            # "Future exception was never retrieved" warning.
            future.exception()
        raise
