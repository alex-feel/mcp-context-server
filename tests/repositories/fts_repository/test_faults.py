"""Unit tests for the FTS failure classification in app.repositories.fts_repository.faults.

Covers FtsValidationError and the SQLite and PostgreSQL classifiers that attribute a failed
FTS statement either to the client's query text or to a genuine database fault. The SQLite
cases run against a throwaway database so they observe the real result codes and messages.
"""

import sqlite3

import pytest

from app.repositories.fts_repository.faults import FtsValidationError
from app.repositories.fts_repository.faults import fts_relations_present
from app.repositories.fts_repository.faults import is_fts5_grammar_error
from app.repositories.fts_repository.faults import is_postgresql_query_failure


class TestFtsValidationError:
    """Test FtsValidationError exception."""

    def test_exception_creation(self) -> None:
        """Test exception can be created with message and errors."""
        errors = ['Error 1', 'Error 2']
        exc = FtsValidationError('Validation failed', errors)
        assert exc.message == 'Validation failed'
        assert exc.validation_errors == errors

    def test_exception_string_representation(self) -> None:
        """Test exception string representation."""
        errors = ['Error 1']
        exc = FtsValidationError('Validation failed', errors)
        assert str(exc) == 'Validation failed'

    def test_exception_empty_errors(self) -> None:
        """Test exception with empty errors list."""
        exc = FtsValidationError('No specific errors', [])
        assert exc.message == 'No specific errors'
        assert exc.validation_errors == []


class TestFtsMatchFailureClassification:
    """A failed MATCH is classified by database-fault family, not by known grammar wordings.

    The FTS statement mixes server-authored SQL with exactly one client-controlled fragment --
    the MATCH expression -- and the set of messages FTS5 can produce for a bad expression is
    open-ended: deeply nested parentheses raise 'fts5: parser stack overflow', a wording no
    enumeration of known grammar messages contained. Treating an unenumerated wording as a
    server fault re-raised it and charged the process-global SQLite circuit breaker, so a
    client repeating one malformed query could open the breaker and reject every other
    caller's reads and writes for the recovery timeout. Classification therefore enumerates
    the fault families and attributes everything else to the query.
    """

    @staticmethod
    def _fts5_error(query: str) -> sqlite3.OperationalError:
        """Capture the real error SQLite raises for a MATCH expression.

        Args:
            query: The MATCH expression to bind.

        Returns:
            The OperationalError SQLite raised, carrying its real sqlite_errorcode.
        """
        conn = sqlite3.connect(':memory:')
        try:
            conn.execute('CREATE VIRTUAL TABLE d USING fts5(b)')
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                conn.execute('SELECT rowid FROM d WHERE d MATCH ?', (query,)).fetchall()
        finally:
            conn.close()
        return exc_info.value

    def test_parser_stack_overflow_is_attributed_to_the_query(self) -> None:
        """Deeply nested parentheses are malformed client input, not a database fault."""
        exc = self._fts5_error('(' * 100 + 'term' + ')' * 100)

        assert 'parser stack overflow' in str(exc)
        assert is_fts5_grammar_error(exc, relations_present=True) is True

    @pytest.mark.parametrize(
        'query',
        [
            'term AND',  # dangling trailing operator
            '(term',  # unbalanced parenthesis
            '*term',  # leading '*' read as an unknown special query
            'term : other',  # stray column filter
            'NEAR(a b, x)',  # non-integer NEAR distance
            'term "unterminated',  # unterminated string literal
        ],
    )
    def test_malformed_expressions_are_attributed_to_the_query(self, query: str) -> None:
        """Every shape of malformed MATCH expression classifies as client input."""
        assert is_fts5_grammar_error(self._fts5_error(query), relations_present=True) is True

    def test_fault_wording_echoed_from_the_query_stays_a_query_error(self) -> None:
        """A client cannot dress its malformed query up as a database fault.

        FTS5 echoes the offending token back in its message, so the client controls part of
        the text. Classification reads the result code first, which keeps an echoed word that
        happens to appear in a fault message from steering the failure onto the breaker.
        """
        exc = self._fts5_error('*interrupted')

        assert 'interrupted' in str(exc)
        assert is_fts5_grammar_error(exc, relations_present=True) is True

    def test_missing_fts_table_stays_a_database_fault(self) -> None:
        """A missing FTS index must still propagate, not be reported as a bad query.

        A missing relation reports SQLITE_ERROR -- the same result code FTS5 uses for a
        grammar error -- so the code alone cannot separate the two; an un-provisioned
        index is a server-side schema state that has to reach the operator. The two are
        told apart by the CATALOG rather than by the message, because FTS5 echoes client
        tokens into its messages and a message match would let a crafted query decide
        its own classification.
        """
        conn = sqlite3.connect(':memory:')
        try:
            with pytest.raises(sqlite3.OperationalError) as exc_info:
                conn.execute('SELECT rowid FROM absent_fts WHERE absent_fts MATCH ?', ('term',)).fetchall()
            assert fts_relations_present(conn) is False
        finally:
            conn.close()

        assert is_fts5_grammar_error(exc_info.value, relations_present=False) is False

    def test_relation_probe_sees_a_provisioned_index(self) -> None:
        """With both relations present the probe reports so, and a bad query stays a query error."""
        conn = sqlite3.connect(':memory:')
        try:
            conn.execute('CREATE TABLE context_entries (id TEXT)')
            conn.execute('CREATE VIRTUAL TABLE context_entries_fts USING fts5(text_content)')

            assert fts_relations_present(conn) is True
        finally:
            conn.close()

    def test_classification_follows_the_catalog_not_the_message(self) -> None:
        """The SAME error classifies differently only because the CATALOG differs.

        A missing relation and a bad MATCH expression share a result code, so something
        else has to separate them. Reading the message would hand that decision to the
        client, which controls part of the text FTS5 echoes back; reading whether the
        relations exist is provenance no query can influence.
        """
        exc = self._fts5_error('*"no such table":foo')

        assert is_fts5_grammar_error(exc, relations_present=True) is True
        assert is_fts5_grammar_error(exc, relations_present=False) is False


class TestPostgresqlFtsFailureClassification:
    """A failed PostgreSQL FTS statement is classified by SQLSTATE, not by wording.

    The tsquery argument is client-controlled and PostgreSQL rejects an oversized one at
    EXECUTION ('invalid memory alloc request size' while the tsquery is assembled), which
    unclassified charges the process-global circuit breaker -- ten such calls open it for
    every caller, the same denial of service the SQLite path closes. Only SQLSTATEs that
    describe the SERVER's condition count as faults.
    """

    class _PgError(Exception):
        """An exception shaped like the ones asyncpg raises, carrying a SQLSTATE."""

        def __init__(self, sqlstate: str) -> None:
            """Initialize the exception.

            Args:
                sqlstate: The five-character SQLSTATE the server reported.
            """
            super().__init__('postgres said no')
            self.sqlstate = sqlstate

    @pytest.mark.parametrize(
        'sqlstate',
        [
            'XX000',  # internal_error: 'invalid memory alloc request size' on a huge tsquery
            '42601',  # syntax_error in tsquery
            '54000',  # program_limit_exceeded: tsquery/word too large
            '22P02',  # invalid_text_representation
        ],
    )
    def test_statement_level_rejections_are_attributed_to_the_query(self, sqlstate: str) -> None:
        """A rejection of the statement says nothing about database health.

        Args:
            sqlstate: The SQLSTATE PostgreSQL reports for one such rejection.
        """
        assert is_postgresql_query_failure(self._PgError(sqlstate)) is True

    @pytest.mark.parametrize(
        'sqlstate',
        [
            '08006',  # connection_failure
            '08003',  # connection_does_not_exist
            '53100',  # disk_full
            '53200',  # out_of_memory
            '57P01',  # admin_shutdown
            '58030',  # io_error
            'XX001',  # data_corrupted
            'XX002',  # index_corrupted
            '42P01',  # undefined_table: the FTS index is not provisioned
            '42704',  # undefined_object: the configured text search configuration is missing
            '42501',  # insufficient_privilege
            '3F000',  # invalid_schema_name
        ],
    )
    def test_database_faults_still_propagate(self, sqlstate: str) -> None:
        """A fault of the server or its environment keeps charging the breaker.

        Args:
            sqlstate: The SQLSTATE of one such fault.
        """
        assert is_postgresql_query_failure(self._PgError(sqlstate)) is False

    def test_an_error_without_a_sqlstate_is_a_fault(self) -> None:
        """A driver or transport failure never reached the server, so it is not the query's."""
        assert is_postgresql_query_failure(Exception('connection reset')) is False

    @pytest.mark.parametrize(
        'message',
        [
            'database is locked',
            'database table is locked',
            'disk I/O error',
            'database disk image is malformed',
            'database or disk is full',
            'unable to open database file',
            'attempt to write a readonly database',
        ],
    )
    def test_database_fault_families_propagate(self, message: str) -> None:
        """Contention and disk/permission faults stay database faults and keep charging.

        These instances carry no result code (they are constructed in Python, as a wrapper or
        a test does), which exercises the canonical-message fallback.

        Args:
            message: The canonical sqlite3 message for one fault family.
        """
        assert is_fts5_grammar_error(sqlite3.OperationalError(message), relations_present=True) is False
