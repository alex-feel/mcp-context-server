"""Tests for ID prefix lookup behavior at the MCP tool boundary.

Verifies that UUIDv7 prefix lookups are accepted uniformly across every
ID-accepting tool -- ``update_context``, ``get_context_by_ids``,
``delete_context`` and ``delete_context_batch`` -- require at least 8 hex
characters, resolve a unique prefix to its full ID, and return clear errors
for ambiguous prefixes. The uniformity guards against the regression where
only the update tools resolved prefixes while the get/delete tools rejected
them.
"""

import sqlite3
from collections.abc import AsyncGenerator
from datetime import UTC
from datetime import datetime
from pathlib import Path

import pytest
import pytest_asyncio
from fastmcp.exceptions import ToolError

import app.startup
from app.backends import StorageBackend
from app.backends import create_backend
from app.ids import generate_id_with_timestamp
from app.repositories import RepositoryContainer
from app.tools.batch import delete_context_batch
from app.tools.context import delete_context
from app.tools.context import get_context_by_ids
from app.tools.context import update_context


@pytest_asyncio.fixture
async def backend_with_repos(
    tmp_path: Path,
) -> AsyncGenerator[tuple[StorageBackend, RepositoryContainer], None]:
    """Create a SQLite backend, schema, and RepositoryContainer for testing."""
    db_path = tmp_path / 'test.db'
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    repos = RepositoryContainer(backend)
    app.startup.set_backend(backend)
    app.startup.set_repositories(repos)
    try:
        yield backend, repos
    finally:
        await backend.shutdown()
        app.startup.set_backend(None)
        app.startup.set_repositories(None)


class TestPrefixLookupMinimumLength:
    """The MCP tool boundary requires 8 or more hex chars for prefix-based lookup."""

    @pytest.mark.asyncio
    async def test_prefix_below_minimum_rejected_in_update(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A 7-character prefix raises an error in update_context."""
        _backend, _repos = backend_with_repos
        with pytest.raises((ToolError, ValueError)):
            await update_context(context_id='0190abc', text='ignored')

    @pytest.mark.asyncio
    async def test_full_uuid_hex_accepted_in_update(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A 32-char canonical hex is accepted by update_context."""
        backend, _repos = backend_with_repos
        canonical = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, content_type, text_content)
                   VALUES (?, ?, ?, ?, ?)''',
                (canonical, 't', 'user', 'text', 'original'),
            )

        await backend.execute_write(_insert)

        result = await update_context(context_id=canonical, text='updated')
        assert result['success'] is True
        assert result['context_id'] == canonical


class TestPrefixLookupAmbiguity:
    """Ambiguous prefixes raise a clear error message."""

    @pytest.mark.asyncio
    async def test_ambiguous_prefix_raises_error(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """Two stored IDs share a prefix; resolution must fail with a descriptive error."""
        backend, _repos = backend_with_repos
        base = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        sibling = base[:8] + 'b' * 24

        def _insert(conn: sqlite3.Connection) -> None:
            for entry_id in (base, sibling):
                conn.execute(
                    '''INSERT INTO context_entries
                       (id, thread_id, source, content_type, text_content)
                       VALUES (?, ?, ?, ?, ?)''',
                    (entry_id, 't', 'user', 'text', 'x'),
                )

        await backend.execute_write(_insert)

        shared_prefix = base[:8]
        with pytest.raises((ToolError, ValueError), match='[Aa]mbiguous'):
            await update_context(context_id=shared_prefix, text='ignored')

    @pytest.mark.asyncio
    async def test_unique_prefix_resolves_in_update(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A unique 8-char prefix successfully resolves to the full UUID."""
        backend, _repos = backend_with_repos
        full_id = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, content_type, text_content)
                   VALUES (?, ?, ?, ?, ?)''',
                (full_id, 't', 'user', 'text', 'original'),
            )

        await backend.execute_write(_insert)

        prefix = full_id[:8]
        result = await update_context(context_id=prefix, text='updated')
        assert result['success'] is True
        assert result['context_id'] == full_id


def _insert_entry(conn: sqlite3.Connection, entry_id: str, text: str = 'x') -> None:
    """Insert a single context entry directly for prefix-resolution tests."""
    conn.execute(
        '''INSERT INTO context_entries
           (id, thread_id, source, content_type, text_content)
           VALUES (?, ?, ?, ?, ?)''',
        (entry_id, 't', 'user', 'text', text),
    )


class TestPrefixLookupUniformAcrossTools:
    """get_context_by_ids and the delete tools resolve prefixes like update_context.

    Regression coverage for the defect where get_context_by_ids, delete_context,
    and delete_context_batch normalized IDs with a bare ``normalize_id`` and so
    rejected the 8-31 char prefixes that update_context already accepted.
    """

    @pytest.mark.asyncio
    async def test_unique_prefix_resolves_in_get_context_by_ids(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A unique 8-char prefix resolves to the full entry in get_context_by_ids."""
        backend, _repos = backend_with_repos
        full_id = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        await backend.execute_write(lambda conn: _insert_entry(conn, full_id, 'hello'))

        entries = await get_context_by_ids(context_ids=[full_id[:8]])
        assert len(entries) == 1
        assert entries[0].get('id') == full_id

    @pytest.mark.asyncio
    async def test_prefix_below_minimum_rejected_in_get_context_by_ids(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A 7-character prefix is rejected by get_context_by_ids."""
        _backend, _repos = backend_with_repos
        with pytest.raises((ToolError, ValueError)):
            await get_context_by_ids(context_ids=['0190abc'])

    @pytest.mark.asyncio
    async def test_ambiguous_prefix_raises_in_get_context_by_ids(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """An ambiguous prefix raises a descriptive error in get_context_by_ids."""
        backend, _repos = backend_with_repos
        base = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        sibling = base[:8] + 'b' * 24
        await backend.execute_write(lambda conn: _insert_entry(conn, base))
        await backend.execute_write(lambda conn: _insert_entry(conn, sibling))

        with pytest.raises((ToolError, ValueError), match='[Aa]mbiguous'):
            await get_context_by_ids(context_ids=[base[:8]])

    @pytest.mark.asyncio
    async def test_unique_prefix_resolves_in_delete_context(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A unique prefix deletes the matching entry via delete_context."""
        backend, _repos = backend_with_repos
        full_id = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        await backend.execute_write(lambda conn: _insert_entry(conn, full_id, 'gone soon'))

        result = await delete_context(context_ids=[full_id[:8]])
        assert result['success'] is True
        assert result['deleted_count'] == 1
        assert await get_context_by_ids(context_ids=[full_id]) == []

    @pytest.mark.asyncio
    async def test_prefix_below_minimum_rejected_in_delete_context(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A 7-character prefix is rejected by delete_context."""
        _backend, _repos = backend_with_repos
        with pytest.raises((ToolError, ValueError)):
            await delete_context(context_ids=['0190abc'])

    @pytest.mark.asyncio
    async def test_unique_prefix_resolves_in_delete_context_batch(
        self,
        backend_with_repos: tuple[StorageBackend, RepositoryContainer],
    ) -> None:
        """A unique prefix deletes the matching entry via delete_context_batch."""
        backend, _repos = backend_with_repos
        full_id = generate_id_with_timestamp(datetime(2024, 1, 1, tzinfo=UTC))
        await backend.execute_write(lambda conn: _insert_entry(conn, full_id, 'batch gone'))

        result = await delete_context_batch(context_ids=[full_id[:8]])
        assert result['success'] is True
        assert result['deleted_count'] == 1
        assert await get_context_by_ids(context_ids=[full_id]) == []
