"""Tests for the grant repository.

Covers app.repositories.grant_repository against a real temp SQLite database
built from the full base schema: idempotent group read-grant insertion, the
deterministic read order, the empty-groups no-op, and cascade deletion with the
parent entry.
"""

import sqlite3

import pytest

from app.backends import StorageBackend
from app.repositories.context_repository import ContextRepository
from app.repositories.grant_repository import GrantRepository
from app.repositories.grant_repository import GrantRow


async def _store_entry(backend: StorageBackend, thread_id: str = 'grant-thread') -> str:
    repo = ContextRepository(backend)
    context_id, _ = await repo.store_with_deduplication(
        thread_id=thread_id,
        source='agent',
        content_type='text',
        text_content=f'grant test entry for {thread_id}',
        owner_id='alice',
        visibility='shared',
    )
    return context_id


class TestStoreGroupReadGrants:
    """Tests for store_group_read_grants."""

    @pytest.mark.asyncio
    async def test_grants_stored_sorted_with_grantor(self, async_db_initialized: StorageBackend) -> None:
        """Each group receives one read grant recording the grantor."""
        context_id = await _store_entry(async_db_initialized)
        grants_repo = GrantRepository(async_db_initialized)

        await grants_repo.store_group_read_grants(context_id, {'team-b', 'team-a'}, granted_by='alice')

        assert await grants_repo.get_grants_for_context(context_id) == [
            GrantRow('group', 'team-a', 'read', 'alice'),
            GrantRow('group', 'team-b', 'read', 'alice'),
        ]

    @pytest.mark.asyncio
    async def test_repeated_grants_are_idempotent(self, async_db_initialized: StorageBackend) -> None:
        """Re-granting the same groups inserts no duplicate rows."""
        context_id = await _store_entry(async_db_initialized)
        grants_repo = GrantRepository(async_db_initialized)

        await grants_repo.store_group_read_grants(context_id, ['team-a'], granted_by='alice')
        await grants_repo.store_group_read_grants(context_id, ['team-a'], granted_by='alice')

        assert await grants_repo.get_grants_for_context(context_id) == [
            GrantRow('group', 'team-a', 'read', 'alice'),
        ]

    @pytest.mark.asyncio
    async def test_empty_groups_is_a_noop(self, async_db_initialized: StorageBackend) -> None:
        """An empty group collection writes nothing."""
        context_id = await _store_entry(async_db_initialized)
        grants_repo = GrantRepository(async_db_initialized)

        await grants_repo.store_group_read_grants(context_id, [], granted_by='alice')

        assert await grants_repo.get_grants_for_context(context_id) == []

    @pytest.mark.asyncio
    async def test_grants_cascade_with_entry_delete(self, async_db_initialized: StorageBackend) -> None:
        """Deleting the parent entry removes its grant rows (FK ON DELETE CASCADE)."""
        context_id = await _store_entry(async_db_initialized)
        grants_repo = GrantRepository(async_db_initialized)
        await grants_repo.store_group_read_grants(context_id, ['team-a'], granted_by='alice')

        await ContextRepository(async_db_initialized).delete_by_ids([context_id])

        def _count(conn: sqlite3.Connection) -> int:
            cursor = conn.execute(
                'SELECT COUNT(*) FROM context_entry_grants WHERE context_entry_id = ?',
                (context_id,),
            )
            return int(cursor.fetchone()[0])

        assert await async_db_initialized.execute_read(_count) == 0
