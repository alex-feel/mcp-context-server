"""Classification of SQLite write contention.

The SQLITE_BUSY / SQLITE_LOCKED predicate shared by the backend, the repositories and the
tool layer, which treat that family as self-clearing contention rather than a database fault.
"""

import sqlite3

# Primary SQLite result codes of the self-clearing write-contention family.
# Extended result codes (e.g. SQLITE_BUSY_SNAPSHOT = 517) carry the primary
# code in their low byte.
_SQLITE_CONTENTION_PRIMARY_CODES = frozenset({sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED})


def is_sqlite_locked_error(exc: BaseException) -> bool:
    """Classify the SQLite locked/busy write-contention family.

    SQLITE_BUSY ('database is locked', including extended forms such as
    SQLITE_BUSY_SNAPSHOT) and SQLITE_LOCKED ('database table is locked')
    signal that another connection -- typically another process sharing the
    same database file -- holds a conflicting lock. The condition self-clears
    once the competing writer finishes, so it is routine contention to retry,
    NOT a database fault. This is the single shared predicate for that
    classification: ``_execute_write_with_retry`` retries the family inside
    the backend, ``begin_transaction`` re-raises it without charging the
    circuit breaker, and the tool layer's ``is_connection_error`` treats it as
    transient so the store/update retry loops re-run the transaction with
    backoff.

    Args:
        exc: The exception to classify.

    Returns:
        True when the exception is an ``sqlite3.OperationalError`` in the
        SQLITE_BUSY / SQLITE_LOCKED family.
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    errorcode = getattr(exc, 'sqlite_errorcode', None)
    if isinstance(errorcode, int):
        return (errorcode & 0xFF) in _SQLITE_CONTENTION_PRIMARY_CODES
    # Instances constructed in Python (tests, wrappers) carry no
    # sqlite_errorcode attribute; fall back to the canonical messages the
    # sqlite3 module emits for this family.
    message = str(exc)
    return 'database is locked' in message or 'database table is locked' in message
