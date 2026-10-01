"""Unit tests for the FTS maintenance mixin in app.repositories.fts_repository.maintenance.

Covers the availability probes, the tokenizer selection and inspection, and the PostgreSQL
language detection.
"""

import sqlite3
from unittest.mock import MagicMock

import pytest

from app.repositories.fts_repository import FtsRepository


class TestFtsIsAvailablePropagatesOperationalFaults:
    """is_available() reports feature ABSENCE, never an operational fault.

    Every search request runs this probe (it backs fts_search_context and the hybrid FTS
    leg), so swallowing exceptions turned a transient lock held by an external VACUUM or
    backup -- and a corrupt database image -- into the misleading "FTS index not found,
    restart the server to apply migrations" error, while hybrid search silently returned
    semantic-only results. Swallowing inside the callable also consumed the fault before the
    backend's bounded locked-retry loop and circuit-breaker accounting could see it. Neither
    probe needs a handler for its stated purpose: sqlite_master yields zero rows for a
    missing table and to_regclass yields NULL for a missing relation.
    """

    @pytest.mark.asyncio
    async def test_sqlite_probe_error_propagates(self) -> None:
        """A locked database surfaces as an error, not as 'FTS unavailable'."""
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _LockedConnection:
            def execute(self, _sql: str, _params: object = None) -> object:
                raise sqlite3.OperationalError('database is locked')

        class _Backend:
            backend_type = 'sqlite'

            async def execute_read(self, operation: Callable[[object], bool]) -> bool:
                return operation(_LockedConnection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        with pytest.raises(sqlite3.OperationalError, match='database is locked'):
            await repo.is_available()

    @pytest.mark.asyncio
    async def test_sqlite_probe_reports_absent_table_as_false(self) -> None:
        """A missing FTS table is still reported as unavailable (zero catalog rows)."""
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _EmptyCursor:
            def fetchone(self) -> object | None:
                return None

        class _Connection:
            def execute(self, _sql: str, _params: object = None) -> _EmptyCursor:
                return _EmptyCursor()

        class _Backend:
            backend_type = 'sqlite'

            async def execute_read(self, operation: Callable[[object], bool]) -> bool:
                return operation(_Connection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        assert await repo.is_available() is False

    @pytest.mark.asyncio
    async def test_postgresql_probe_error_propagates(self) -> None:
        """A dropped connection surfaces as an error, not as 'FTS unavailable'."""
        from collections.abc import Awaitable
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _BrokenConnection:
            async def fetchval(self, _sql: str) -> object:
                raise OSError('connection reset by peer')

        class _Backend:
            backend_type = 'postgresql'

            async def execute_read(self, operation: Callable[[object], Awaitable[bool]]) -> bool:
                return await operation(_BrokenConnection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        with pytest.raises(OSError, match='connection reset by peer'):
            await repo.is_available()

    @pytest.mark.asyncio
    async def test_postgresql_probe_reports_absent_column_as_false(self) -> None:
        """A missing relation/column is still reported as unavailable (EXISTS false)."""
        from collections.abc import Awaitable
        from collections.abc import Callable
        from typing import cast

        from app.backends.base import StorageBackend

        class _Connection:
            async def fetchval(self, _sql: str) -> bool:
                return False

        class _Backend:
            backend_type = 'postgresql'

            async def execute_read(self, operation: Callable[[object], Awaitable[bool]]) -> bool:
                return await operation(_Connection())

        repo = FtsRepository(cast(StorageBackend, _Backend()))

        assert await repo.is_available() is False

    def test_is_available_pg_probe_is_schema_aware(self) -> None:
        """The PG availability probe resolves context_entries via search_path.

        A schema-blind information_schema.columns query (no table_schema filter)
        reported the FTS column present when it existed in ANY visible schema
        (e.g. a colliding public.context_entries), defeating the schema-aware
        FTS backstop on a non-default POSTGRESQL_SCHEMA. The probe must resolve
        the relation via to_regclass (search_path) like the FTS reads/writes.
        """
        import inspect

        src = inspect.getsource(FtsRepository.is_available)
        assert 'to_regclass' in src
        assert 'pg_attribute' in src
        # The old schema-blind lookup query (FROM information_schema.columns with
        # no table_schema filter) must be gone from the SQL itself.
        assert 'FROM information_schema.columns' not in src
        assert 'FROM\n                        information_schema.columns' not in src


class TestFtsTokenizerSelection:
    """Test language-aware tokenizer selection for SQLite FTS5."""

    @pytest.fixture
    def mock_backend(self) -> MagicMock:
        """Create a mock backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        return backend

    @pytest.fixture
    def repo(self, mock_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock backend."""
        return FtsRepository(mock_backend)

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_english(self, repo: FtsRepository) -> None:
        """Test that English language uses Porter stemmer."""
        tokenizer = await repo.get_desired_tokenizer('english')
        assert tokenizer == 'porter unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_english_uppercase(self, repo: FtsRepository) -> None:
        """Test that English (uppercase) uses Porter stemmer."""
        tokenizer = await repo.get_desired_tokenizer('ENGLISH')
        assert tokenizer == 'porter unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_german(self, repo: FtsRepository) -> None:
        """Test that German language uses unicode61 only (no stemming)."""
        tokenizer = await repo.get_desired_tokenizer('german')
        assert tokenizer == 'unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_french(self, repo: FtsRepository) -> None:
        """Test that French language uses unicode61 only (no stemming)."""
        tokenizer = await repo.get_desired_tokenizer('french')
        assert tokenizer == 'unicode61'

    @pytest.mark.asyncio
    async def test_desired_tokenizer_for_spanish(self, repo: FtsRepository) -> None:
        """Test that Spanish language uses unicode61 only (no stemming)."""
        tokenizer = await repo.get_desired_tokenizer('spanish')
        assert tokenizer == 'unicode61'

    @pytest.mark.asyncio
    async def test_get_current_tokenizer_no_fts_table(self) -> None:
        """Test get_current_tokenizer returns None when FTS table doesn't exist."""
        from unittest.mock import AsyncMock

        # Create backend with execute_read configured BEFORE creating repo
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        backend.execute_read = AsyncMock(return_value=None)
        repo = FtsRepository(backend)

        tokenizer = await repo.get_current_tokenizer()
        assert tokenizer is None

    @pytest.mark.asyncio
    async def test_get_current_tokenizer_postgresql_returns_none(self) -> None:
        """Test get_current_tokenizer returns None for PostgreSQL backend."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        repo = FtsRepository(backend)

        tokenizer = await repo.get_current_tokenizer()
        assert tokenizer is None


class TestFtsLanguageDetection:
    """Test PostgreSQL FTS language detection."""

    @pytest.fixture
    def mock_pg_backend(self) -> MagicMock:
        """Create a mock PostgreSQL backend for testing."""
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        return backend

    @pytest.fixture
    def repo(self, mock_pg_backend: MagicMock) -> FtsRepository:
        """Create a repository with mock PostgreSQL backend."""
        return FtsRepository(mock_pg_backend)

    @pytest.mark.asyncio
    async def test_get_current_language_sqlite_returns_none(self) -> None:
        """Test get_current_language returns None for SQLite backend."""
        backend = MagicMock()
        backend.backend_type = 'sqlite'
        repo = FtsRepository(backend)

        language = await repo.get_current_language()
        assert language is None

    @pytest.mark.asyncio
    async def test_get_current_language_no_tsvector_column(self) -> None:
        """Test get_current_language returns None when tsvector column doesn't exist."""
        from unittest.mock import AsyncMock

        # Create backend with execute_read configured BEFORE creating repo
        backend = MagicMock()
        backend.backend_type = 'postgresql'
        backend.execute_read = AsyncMock(return_value=None)
        repo = FtsRepository(backend)

        language = await repo.get_current_language()
        assert language is None
