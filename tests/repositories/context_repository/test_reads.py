"""Tests for ContextRepository by-id reads and the existence and content-type probes."""

from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import Mock

import pytest

from app.backends.base import StorageBackend
from app.backends.base import TransactionContext
from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


class TestContextRepositoryGetById:
    """Test get_by_ids operations."""

    @pytest.mark.asyncio
    async def test_get_by_ids_single(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting single entry by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='get_thread',
            source='user',
            content_type='text',
            text_content='Test entry',
        )

        rows = await repos.context.get_by_ids([ctx_id], scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id

    @pytest.mark.asyncio
    async def test_get_by_ids_multiple(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting multiple entries by IDs."""
        ids = []
        for i in range(3):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='multi_get',
                source='user',
                content_type='text',
                text_content=f'Entry {i}',
            )
            ids.append(ctx_id)

        rows = await repos.context.get_by_ids(ids, scope=LOCAL_SCOPE)

        assert len(rows) == 3
        returned_ids = {r['id'] for r in rows}
        assert returned_ids == set(ids)

    @pytest.mark.asyncio
    async def test_get_by_ids_empty_list(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting entries with empty ID list."""
        rows = await repos.context.get_by_ids([], scope=LOCAL_SCOPE)

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting nonexistent IDs returns empty."""
        rows = await repos.context.get_by_ids([generate_id(), generate_id()], scope=LOCAL_SCOPE)

        assert rows == []

    @pytest.mark.asyncio
    async def test_get_by_ids_partial_match(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting mix of existing and nonexistent IDs."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='partial_get',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        rows = await repos.context.get_by_ids([ctx_id, generate_id()], scope=LOCAL_SCOPE)

        assert len(rows) == 1
        assert rows[0]['id'] == ctx_id

    @pytest.mark.asyncio
    async def test_get_by_ids_spans_multiple_chunks_preserves_order(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """get_by_ids chunks an oversized id list and preserves the ordering contract.

        A 33,000-id list bound as one IN clause exceeds SQLite's default 32,766
        per-statement variable ceiling ("too many SQL variables"), so the fetch is
        issued in bounded chunks like delete_by_ids. Real ids are planted on both
        sides of the 900-id chunk boundary with the OLDEST rows in the FIRST chunk
        and the NEWEST row in the LAST chunk, so plain per-chunk concatenation
        would invert the order -- the assertion therefore also proves the
        accumulated rows are re-sorted into the single-statement
        ORDER BY created_at DESC, id DESC contract.
        """
        real_ids: list[str] = []
        for i in range(5):
            ctx_id, _ = await repos.context.store_with_deduplication(
                scope=LOCAL_SCOPE,
                visibility='private',
                thread_id='chunk_get_thread',
                source='user',
                content_type='text',
                text_content=f'Chunk get entry {i}',
            )
            real_ids.append(ctx_id)

        # Expected order = the documented contract, computed from per-row reads
        # (each a single-chunk fetch): created_at DESC, id DESC.
        keyed: list[tuple[str, str]] = []
        for ctx_id in real_ids:
            row = (await repos.context.get_by_ids([ctx_id], scope=LOCAL_SCOPE))[0]
            keyed.append((row['created_at'], row['id']))
        expected_ids = [entry_id for _created_at, entry_id in sorted(keyed, reverse=True)]

        # Non-existent ids pad the list past the statement variable ceiling; the
        # oldest real rows go into the first chunk and the newest into the last.
        mixed = [generate_id() for _ in range(33000)]
        oldest_first = list(reversed(expected_ids))
        for pos, real_id in zip((0, 450, 899, 900, 32999), oldest_first, strict=True):
            mixed[pos] = real_id

        rows = await repos.context.get_by_ids(mixed, scope=LOCAL_SCOPE)

        assert [row['id'] for row in rows] == expected_ids


async def _store_local_entries(repos: RepositoryContainer, count: int, thread_id: str) -> list[str]:
    """Store ``count`` entries owned by the default principal and return their ids."""
    stored: list[str] = []
    for i in range(count):
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id=thread_id,
            source='user',
            content_type='text',
            text_content=f'{thread_id} entry {i}',
        )
        stored.append(ctx_id)
    return stored


def _pad_to_1000_ids(real_ids: list[str]) -> list[str]:
    """Return 1,000 ids with the real ids at the first, last-of-chunk and last positions."""
    ids = [generate_id() for _ in range(1000)]
    for position, real_id in zip((0, 899, 999), real_ids, strict=True):
        ids[position] = real_id
    return ids


@pytest.mark.usefixtures('sqlite_999_variables')
class TestScopedReadsUnderVariableCap:
    """The scoped id reads fit each 900-id chunk plus its access binds within 999 variables."""

    @pytest.mark.asyncio
    async def test_get_by_ids_binds_1000_ids_with_scope(self, repos: RepositoryContainer) -> None:
        """A 1,000-id read under a scope binds without error and returns every stored row."""
        real_ids = await _store_local_entries(repos, 3, 'capped_get')

        rows = await repos.context.get_by_ids(_pad_to_1000_ids(real_ids), scope=LOCAL_SCOPE)

        assert {row['id'] for row in rows} == set(real_ids)

    @pytest.mark.asyncio
    async def test_probe_ids_binds_1000_ids_with_scope(self, repos: RepositoryContainer) -> None:
        """A 1,000-id access probe binds without error and reports every stored row."""
        real_ids = await _store_local_entries(repos, 3, 'capped_probe')

        access = await repos.context.probe_ids(_pad_to_1000_ids(real_ids), scope=LOCAL_SCOPE)

        assert set(access) == set(real_ids)
        assert all(entry.can_write and entry.is_owner for entry in access.values())

    @pytest.mark.asyncio
    async def test_probe_ids_of_an_empty_list_is_empty(self, repos: RepositoryContainer) -> None:
        """No ids means no statement and an empty result."""
        assert await repos.context.probe_ids([], scope=LOCAL_SCOPE) == {}


class TestContextRepositoryUpdate:
    """Test update operations in ContextRepository."""

    @pytest.mark.asyncio
    async def test_check_entry_exists(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test checking if entry exists."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='exists_thread',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        probe = await repos.context.check_entry_exists(ctx_id, scope=LOCAL_SCOPE)
        assert probe.exists is True
        assert probe.source == 'user'
        assert isinstance(probe.version, int)
        assert probe.version == 0
        assert probe.owner_id == 'local'
        assert probe.can_write is True

        missing = await repos.context.check_entry_exists(generate_id(), scope=LOCAL_SCOPE)
        assert missing.exists is False
        assert missing.source is None
        assert missing.version is None
        assert missing.owner_id is None
        assert missing.can_write is False

    @pytest.mark.asyncio
    async def test_entry_exists(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """entry_exists returns True for a stored id and False for an absent one."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='entry_exists_thread',
            source='user',
            content_type='text',
            text_content='Exists',
        )

        assert await repos.context.entry_exists(ctx_id) is True
        assert await repos.context.entry_exists(generate_id()) is False

    @pytest.mark.asyncio
    async def test_entry_exists_locks_parent_row_on_postgresql_transaction(self) -> None:
        """On PostgreSQL the in-transaction presence check locks the parent row.

        The tags-only / images-only update guard runs entry_exists on the open
        transaction connection; it must emit FOR KEY SHARE so a concurrent DELETE
        blocks until commit and cannot leave the child tag/image writes violating
        the foreign key. Outside a transaction the lock would release at statement
        end, so it must NOT be emitted there. Both cases are asserted against a
        recording connection without needing a live PostgreSQL.
        """
        txn_conn = AsyncMock()
        txn_conn.fetchrow = AsyncMock(return_value={'?column?': 1})
        txn_backend = Mock()
        txn_backend.backend_type = 'postgresql'
        txn = Mock()
        txn.backend_type = 'postgresql'
        txn.connection = txn_conn

        repo_txn = ContextRepository(cast(StorageBackend, txn_backend))
        assert await repo_txn.entry_exists('abc123', txn=cast(TransactionContext, txn)) is True
        assert 'FOR KEY SHARE' in txn_conn.fetchrow.call_args.args[0]

        pool_conn = AsyncMock()
        pool_conn.fetchrow = AsyncMock(return_value={'?column?': 1})

        async def _execute_read(closure: Callable[[object], Awaitable[bool]]) -> bool:
            return await closure(pool_conn)

        pool_backend = Mock()
        pool_backend.backend_type = 'postgresql'
        pool_backend.execute_read = _execute_read

        repo_pool = ContextRepository(cast(StorageBackend, pool_backend))
        assert await repo_pool.entry_exists('abc123') is True
        assert 'FOR KEY SHARE' not in pool_conn.fetchrow.call_args.args[0]

    @pytest.mark.asyncio
    async def test_get_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type by ID."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='type_thread',
            source='user',
            content_type='text',
            text_content='Text content',
        )

        content_type = await repos.context.get_content_type(ctx_id)

        assert content_type == 'text'

    @pytest.mark.asyncio
    async def test_get_content_type_nonexistent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test getting content type for nonexistent entry."""
        content_type = await repos.context.get_content_type(generate_id())

        assert content_type is None

    @pytest.mark.asyncio
    async def test_update_content_type(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Test updating content type."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='update_type',
            source='user',
            content_type='text',
            text_content='Content',
        )

        await repos.context.update_content_type(ctx_id, 'multimodal')

        new_type = await repos.context.get_content_type(ctx_id)
        assert new_type == 'multimodal'
