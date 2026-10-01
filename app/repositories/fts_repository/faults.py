"""Failure classification for full-text search.

Holds the client-input ``FtsValidationError``, the single client-facing detail
for a query the search engine could not process, and the SQLite and PostgreSQL
classifiers that decide whether an execution failure comes from the client's
query text or from a genuine database fault.
"""

import sqlite3

from app.backends.sqlite_backend import is_sqlite_locked_error
from app.errors import ControlFlowError

# Primary SQLite result codes of the DATABASE-FAULT families: a problem with the database file,
# its environment, or the permissions guarding it -- never a defect in the query text. Extended
# result codes (e.g. SQLITE_IOERR_READ = 266) carry the primary code in their low byte, so the
# comparison masks with 0xFF. The SQLITE_BUSY / SQLITE_LOCKED contention family is deliberately
# absent here: it has ONE definition, the shared ``is_sqlite_locked_error`` predicate the backend
# retry loops and the tool layer already classify with, and this module reuses that rather than
# restating it.
_SQLITE_FAULT_PRIMARY_CODES = frozenset({
    sqlite3.SQLITE_PERM,
    sqlite3.SQLITE_NOMEM,
    sqlite3.SQLITE_READONLY,
    sqlite3.SQLITE_INTERRUPT,
    sqlite3.SQLITE_IOERR,
    sqlite3.SQLITE_CORRUPT,
    sqlite3.SQLITE_FULL,
    sqlite3.SQLITE_CANTOPEN,
    sqlite3.SQLITE_PROTOCOL,
    sqlite3.SQLITE_AUTH,
    sqlite3.SQLITE_NOTADB,
})


# Canonical sqlite3 messages for the fault families above, consulted ONLY for an exception that
# carries no ``sqlite_errorcode`` -- an instance constructed in Python by a wrapper or a test.
# Every exception the sqlite3 module itself raises on Python 3.12+ carries the code, so this is
# a compatibility fallback, not the primary classifier.
_SQLITE_FAULT_MESSAGE_FRAGMENTS = (
    'disk i/o error',
    'database disk image is malformed',
    'database or disk is full',
    'unable to open database file',
    'attempt to write a readonly database',
    'file is not a database',
    'out of memory',
    'access permission denied',
    'locking protocol',
    'authorization denied',
)


# The relations the FTS search statement reads. A missing relation reports SQLITE_ERROR -- the
# same primary code FTS5 uses for a MATCH grammar error -- so the code cannot separate the two
# and the message must not be trusted either: FTS5 echoes client tokens back into its messages,
# so matching a phrase like 'no such table' against the text lets a crafted query steer its own
# classification. The catalog is consulted instead (see fts_relations_present), which is
# provenance the client cannot influence.
_FTS_SEARCH_RELATIONS = frozenset({'context_entries', 'context_entries_fts'})


def fts_relations_present(conn: sqlite3.Connection) -> bool:
    """Return True when both relations the FTS search statement reads exist.

    Runs only on the error path, where one catalog lookup is free. A probe that itself
    fails cannot establish provenance, so it reports "absent", which routes the original
    error to the database-fault branch -- the conservative reading when the database is
    demonstrably not answering.

    Args:
        conn: The connection the failed statement ran on.

    Returns:
        True when both relations exist and the failure therefore cannot be a missing index.
    """
    try:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name IN (?, ?)",
            tuple(sorted(_FTS_SEARCH_RELATIONS)),
        )
        present = {str(row[0]) for row in cursor.fetchall()}
    except sqlite3.Error:
        return False
    return present >= _FTS_SEARCH_RELATIONS


def is_fts5_grammar_error(exc: sqlite3.OperationalError, *, relations_present: bool) -> bool:
    """Return True if exc is attributable to the client's MATCH expression, not a database fault.

    The FTS search statement mixes server-authored SQL with exactly ONE client-controlled
    fragment: the MATCH expression, forwarded verbatim in boolean mode and sanitized in the
    other modes. Whatever the engine rejects in that fragment is a CLIENT-INPUT failure, and the
    set of messages FTS5 can produce for it is open-ended -- 'fts5: parser stack overflow' from
    deeply nested parentheses is one such message. Enumerating the accepted grammar messages
    therefore misclassifies the ones nobody enumerated: they look like server faults, propagate
    as hard errors, and charge the process-global circuit breaker, so a client repeating one
    malformed query opens the breaker and rejects every other caller's reads and writes.

    Classification is inverted for that reason: the DATABASE-FAULT families are the closed set
    (the SQLITE_BUSY / SQLITE_LOCKED contention family via the shared predicate, the
    disk / permission / durability result codes, and an un-provisioned FTS index), and everything
    else raised out of this statement is attributed to the MATCH expression. The caller then
    degrades a malformed boolean query to the sanitized term match or reports a structured
    validation error, neither of which charges the breaker.

    Every input to the decision is provenance the client cannot reach: a result code the engine
    assigns, and the catalog state of the relations the statement reads. No part of the error
    TEXT participates except as a compatibility fallback for an exception carrying no result
    code at all, so no wording a client can push through the MATCH expression -- FTS5 echoes
    client tokens into its messages -- can steer a query error onto the breaker.

    Args:
        exc: The OperationalError raised while executing a MATCH query.
        relations_present: Whether the relations the statement reads exist, from
            :func:`fts_relations_present`. False means the FTS index is not provisioned,
            which is a server-side schema state that must reach the operator.

    Returns:
        True if the error is a client query-grammar failure rather than a database fault.
    """
    if is_sqlite_locked_error(exc):
        return False
    errorcode = getattr(exc, 'sqlite_errorcode', None)
    if isinstance(errorcode, int):
        if (errorcode & 0xFF) in _SQLITE_FAULT_PRIMARY_CODES:
            return False
    # No result code to classify with (a Python-constructed instance): fall back to the
    # canonical fault messages.
    elif any(fragment in str(exc).lower() for fragment in _SQLITE_FAULT_MESSAGE_FRAGMENTS):
        return False
    return relations_present


# The SQLSTATE classes PostgreSQL reserves for a failure of the DATABASE or its environment
# rather than of the statement text: 08 connection_exception, 53 insufficient_resources
# (disk full, out of memory, connection slots exhausted), 57 operator_intervention (admin
# shutdown, backend termination, statement timeout), 58 system_error (I/O error, missing
# file). A failure in one of these classes says something about the server's health, which is
# exactly what the circuit breaker exists to track.
_PG_FAULT_SQLSTATE_CLASSES = frozenset({'08', '53', '57', '58'})


# Individual SQLSTATEs outside those classes that are still database faults: on-disk
# corruption, and the provisioning states that mean the schema this server expects is not
# there (a missing table or text-search configuration, an unreachable schema, a revoked
# privilege). These mirror the SQLite side's un-provisioned-index branch.
_PG_FAULT_SQLSTATES = frozenset({
    'XX001',  # data_corrupted
    'XX002',  # index_corrupted
    '42P01',  # undefined_table
    '42704',  # undefined_object (a missing text search configuration)
    '42501',  # insufficient_privilege
    '3D000',  # invalid_catalog_name
    '3F000',  # invalid_schema_name
})


# The one client-facing detail for a query the search engine could not process, shared by both
# backends so the same input reports the same thing regardless of which one answers. It replaces
# the engine's own wording on purpose: that wording is an internal detail, differs between FTS5
# and PostgreSQL for identical input, and echoes fragments of the query back at the caller.
FTS_UNPARSEABLE_QUERY_DETAIL = 'The full-text search engine could not process this query'


def is_postgresql_query_failure(exc: BaseException) -> bool:
    """Return True if exc is attributable to the client's query text, not a database fault.

    The PostgreSQL counterpart of :func:`is_fts5_grammar_error`, and it exists for the same
    reason: the FTS statement's tsquery argument is client-controlled, PostgreSQL can reject it
    at EXECUTION rather than at parse time (a query of many thousands of terms raises 'invalid
    memory alloc request size' while building the tsquery), and an unclassified failure charges
    the process-global PostgreSQL circuit breaker. Ten such calls open it for every caller, which
    is the same client-triggered denial of service the SQLite path closes -- so the two backends
    classify the same way rather than diverging on which inputs are survivable.

    The fault set is closed and consists of SQLSTATEs that describe the SERVER's condition; a
    statement-level rejection describes the STATEMENT, is perfectly reproducible, and says
    nothing about database health. An exception with no SQLSTATE at all never came from the
    server (a driver, transport, or cancellation failure) and is treated as a fault so the pool
    and breaker still see it.

    Args:
        exc: The exception raised while executing the FTS statement.

    Returns:
        True if the failure is attributable to the client-supplied query text.
    """
    sqlstate = getattr(exc, 'sqlstate', None)
    if not isinstance(sqlstate, str) or len(sqlstate) != 5:
        return False
    if sqlstate in _PG_FAULT_SQLSTATES:
        return False
    return sqlstate[:2] not in _PG_FAULT_SQLSTATE_CLASSES


class FtsValidationError(ControlFlowError):
    """Exception raised when FTS query or filters fail validation.

    This exception enables unified error handling between fts_search_context
    and other search tools.

    Subclasses ``ControlFlowError`` because a client-input validation failure is
    normal control flow, not a database fault: the backend connection wrappers
    exempt ``ControlFlowError`` from circuit-breaker failure accounting, so a
    client repeatedly sending an invalid query or filter cannot open the breaker
    and reject every other caller's healthy requests.
    """

    def __init__(self, message: str, validation_errors: list[str]) -> None:
        """Initialize the exception.

        Args:
            message: Error message
            validation_errors: List of validation error messages
        """
        super().__init__(message)
        self.message = message
        self.validation_errors = validation_errors
