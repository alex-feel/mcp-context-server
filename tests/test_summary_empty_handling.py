"""Tests for empty summary normalization chain (Issue 6).

Tests that empty/whitespace-only summaries are properly normalized to None
at multiple defense-in-depth layers:
- generate_summary_with_timeout: provider -> None normalization
- store_context: summary_generated flag uses bool() not is not None
- store_with_deduplication: empty summary -> None before COALESCE
- get_summary: empty string -> None normalization on read
- search_context: empty summary -> None in response
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from app.repositories.context_repository import ContextRepository

_CREATE_TABLE_SQL = '''
    CREATE TABLE context_entries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        thread_id TEXT NOT NULL,
        source TEXT NOT NULL,
        content_type TEXT NOT NULL DEFAULT 'text',
        text_content TEXT NOT NULL,
        metadata TEXT,
        summary TEXT,
        content_hash TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
'''


def _make_sqlite_write_backend(db_path: str) -> MagicMock:
    """Create a mock SQLite backend with a working execute_write."""
    mock_backend = MagicMock()
    mock_backend.backend_type = 'sqlite'

    async def mock_execute_write(
        func: Callable[[sqlite3.Connection], tuple[int, bool]],
    ) -> tuple[int, bool]:
        test_conn = sqlite3.connect(db_path)
        test_conn.row_factory = sqlite3.Row
        try:
            result = func(test_conn)
            test_conn.commit()
            return result
        finally:
            test_conn.close()

    mock_backend.execute_write = mock_execute_write
    return mock_backend


def _make_sqlite_read_backend(db_path: str) -> MagicMock:
    """Create a mock SQLite backend with a working execute_read."""
    mock_backend = MagicMock()
    mock_backend.backend_type = 'sqlite'

    async def mock_execute_read(
        func: Callable[[sqlite3.Connection], str | None],
    ) -> str | None:
        test_conn = sqlite3.connect(db_path)
        test_conn.row_factory = sqlite3.Row
        try:
            return func(test_conn)
        finally:
            test_conn.close()

    mock_backend.execute_read = mock_execute_read
    return mock_backend


class TestGenerateSummaryWithTimeout:
    """Tests for generate_summary_with_timeout empty normalization."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_none(self) -> None:
        """Mock provider returning empty string -> generate_summary_with_timeout returns None."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='')

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.compute_summary_total_timeout', return_value=120.0),
        ):
            from app.tools.context import generate_summary_with_timeout

            result = await generate_summary_with_timeout('Some text to summarize')

        assert result is None

    @pytest.mark.asyncio
    async def test_whitespace_only_returns_none(self) -> None:
        """Mock provider returning whitespace -> generate_summary_with_timeout returns None."""
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value='   \n\t  ')

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.compute_summary_total_timeout', return_value=120.0),
        ):
            from app.tools.context import generate_summary_with_timeout

            result = await generate_summary_with_timeout('Some text to summarize')

        assert result is None

    @pytest.mark.asyncio
    async def test_valid_summary_passes_through(self) -> None:
        """Mock provider returning valid summary -> passes through unchanged."""
        expected = 'This is a valid summary of the input text.'
        mock_provider = MagicMock()
        mock_provider.summarize = AsyncMock(return_value=expected)

        with (
            patch('app.tools.context.get_summary_provider', return_value=mock_provider),
            patch('app.tools.context.compute_summary_total_timeout', return_value=120.0),
        ):
            from app.tools.context import generate_summary_with_timeout

            result = await generate_summary_with_timeout('Some text to summarize')

        assert result == expected

    @pytest.mark.asyncio
    async def test_no_provider_returns_none(self) -> None:
        """No summary provider configured -> returns None without error."""
        with patch('app.tools.context.get_summary_provider', return_value=None):
            from app.tools.context import generate_summary_with_timeout

            result = await generate_summary_with_timeout('Some text to summarize')

        assert result is None


class TestStoreWithDeduplicationEmptySummary:
    """Tests for store_with_deduplication empty summary normalization."""

    @pytest.mark.asyncio
    async def test_empty_summary_normalized_to_none(self, tmp_path: Path) -> None:
        """Empty summary should be normalized to None before COALESCE.

        Verifies that an empty string summary does not overwrite a valid
        existing summary during deduplication.
        """
        db_path = str(tmp_path / 'test.db')

        # Create database and table with initial entry
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(
            '''INSERT INTO context_entries
            (thread_id, source, content_type, text_content, summary)
            VALUES (?, ?, ?, ?, ?)''',
            ('test-thread', 'agent', 'text', 'Some text content', 'Valid existing summary'),
        )
        conn.commit()
        conn.close()

        mock_backend = _make_sqlite_write_backend(db_path)
        repo = ContextRepository(mock_backend)

        # Call store_with_deduplication with empty summary (duplicate text)
        context_id, was_updated = await repo.store_with_deduplication(
            thread_id='test-thread',
            source='agent',
            content_type='text',
            text_content='Some text content',
            summary='',  # Empty string should be normalized to None
        )

        assert was_updated is True

        # Verify the valid summary was preserved (not overwritten by empty string)
        verify_conn = sqlite3.connect(db_path)
        verify_conn.row_factory = sqlite3.Row
        row = verify_conn.execute(
            'SELECT summary FROM context_entries WHERE id = ?', (context_id,),
        ).fetchone()
        verify_conn.close()

        assert row['summary'] == 'Valid existing summary'


class TestGetSummaryEmptyNormalization:
    """Tests for get_summary empty string normalization."""

    @pytest.mark.asyncio
    async def test_empty_string_returns_none(self, tmp_path: Path) -> None:
        """get_summary should normalize empty string to None."""
        db_path = str(tmp_path / 'test.db')

        # Create database with an entry that has empty summary
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(
            '''INSERT INTO context_entries
            (thread_id, source, content_type, text_content, summary)
            VALUES (?, ?, ?, ?, ?)''',
            ('test-thread', 'agent', 'text', 'Some text', ''),
        )
        conn.commit()
        conn.close()

        mock_backend = _make_sqlite_read_backend(db_path)
        repo = ContextRepository(mock_backend)
        result = await repo.get_summary(1)

        assert result is None

    @pytest.mark.asyncio
    async def test_valid_summary_passes_through(self, tmp_path: Path) -> None:
        """get_summary should return valid summary unchanged."""
        db_path = str(tmp_path / 'test.db')

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(
            '''INSERT INTO context_entries
            (thread_id, source, content_type, text_content, summary)
            VALUES (?, ?, ?, ?, ?)''',
            ('test-thread', 'agent', 'text', 'Some text', 'A valid summary'),
        )
        conn.commit()
        conn.close()

        mock_backend = _make_sqlite_read_backend(db_path)
        repo = ContextRepository(mock_backend)
        result = await repo.get_summary(1)

        assert result == 'A valid summary'
