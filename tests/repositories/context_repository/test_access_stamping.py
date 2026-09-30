"""Tests for access-control stamping at the repository write paths.

Covers store_with_deduplication (owner_id/visibility stamped on INSERT, never
touched by a deduplication UPDATE), update_context_entry (visibility rides the
dynamic SET list and the compare-and-set), and the check_entry_exists probe's
owner_id field -- all against a real temp SQLite database built from the full
base schema.
"""

import sqlite3

import pytest

from app.backends import StorageBackend
from app.repositories.context_repository import ContextRepository


async def _read_access_columns(backend: StorageBackend, context_id: str) -> tuple[str, str, int]:
    def _read(conn: sqlite3.Connection) -> tuple[str, str, int]:
        cursor = conn.execute(
            'SELECT owner_id, visibility, version FROM context_entries WHERE id = ?',
            (context_id,),
        )
        row = cursor.fetchone()
        assert row is not None
        return row[0], row[1], row[2]

    return await backend.execute_read(_read)


class TestStoreStamping:
    """store_with_deduplication stamps the access columns on INSERT only."""

    @pytest.mark.asyncio
    async def test_insert_stamps_owner_and_visibility(self, async_db_initialized: StorageBackend) -> None:
        """A fresh INSERT carries the caller-resolved owner and visibility."""
        repo = ContextRepository(async_db_initialized)
        context_id, was_updated = await repo.store_with_deduplication(
            thread_id='stamp-thread',
            source='agent',
            content_type='text',
            text_content='stamped entry',
            owner_id='alice',
            visibility='shared',
        )
        assert was_updated is False
        owner, visibility, _ = await _read_access_columns(async_db_initialized, context_id)
        assert (owner, visibility) == ('alice', 'shared')

    @pytest.mark.asyncio
    async def test_dedup_update_never_changes_owner_or_visibility(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """A deduplication UPDATE leaves the stored owner and visibility untouched,
        whatever values the retransmit carries."""
        repo = ContextRepository(async_db_initialized)
        context_id, _ = await repo.store_with_deduplication(
            thread_id='stamp-thread',
            source='agent',
            content_type='text',
            text_content='dedup-preserved entry',
            owner_id='alice',
            visibility='private',
        )

        dedup_id, was_updated = await repo.store_with_deduplication(
            thread_id='stamp-thread',
            source='agent',
            content_type='text',
            text_content='dedup-preserved entry',
            owner_id='mallory',
            visibility='public',
        )
        assert was_updated is True
        assert dedup_id == context_id
        owner, visibility, _ = await _read_access_columns(async_db_initialized, context_id)
        assert (owner, visibility) == ('alice', 'private')


class TestUpdateVisibility:
    """update_context_entry applies a provided visibility value."""

    @pytest.mark.asyncio
    async def test_visibility_only_update(self, async_db_initialized: StorageBackend) -> None:
        """A visibility-only update writes the new value and reports the field."""
        repo = ContextRepository(async_db_initialized)
        context_id, _ = await repo.store_with_deduplication(
            thread_id='vis-thread',
            source='agent',
            content_type='text',
            text_content='visibility update target',
            owner_id='alice',
            visibility='private',
        )

        success, fields = await repo.update_context_entry(context_id, visibility='public')
        assert success is True
        assert fields == ['visibility']
        _, visibility, _ = await _read_access_columns(async_db_initialized, context_id)
        assert visibility == 'public'

    @pytest.mark.asyncio
    async def test_visibility_rides_the_compare_and_set(
        self, async_db_initialized: StorageBackend,
    ) -> None:
        """With expected_version, a visibility update bumps version like text/metadata."""
        repo = ContextRepository(async_db_initialized)
        context_id, _ = await repo.store_with_deduplication(
            thread_id='vis-thread',
            source='agent',
            content_type='text',
            text_content='cas visibility target',
            owner_id='alice',
            visibility='private',
        )
        probe = await repo.check_entry_exists(context_id)
        assert probe.version is not None

        success, fields = await repo.update_context_entry(
            context_id,
            visibility='shared',
            expected_version=probe.version,
        )
        assert success is True
        assert fields == ['visibility']
        _, visibility, version = await _read_access_columns(async_db_initialized, context_id)
        assert visibility == 'shared'
        assert version == probe.version + 1

    @pytest.mark.asyncio
    async def test_omitted_visibility_is_untouched(self, async_db_initialized: StorageBackend) -> None:
        """A text-only update leaves the stored visibility unchanged."""
        repo = ContextRepository(async_db_initialized)
        context_id, _ = await repo.store_with_deduplication(
            thread_id='vis-thread',
            source='agent',
            content_type='text',
            text_content='text-only update target',
            owner_id='alice',
            visibility='shared',
        )

        success, fields = await repo.update_context_entry(context_id, text_content='new text')
        assert success is True
        assert 'visibility' not in fields
        owner, visibility, _ = await _read_access_columns(async_db_initialized, context_id)
        assert (owner, visibility) == ('alice', 'shared')


class TestEntryProbeOwner:
    """check_entry_exists surfaces the stamped owner."""

    @pytest.mark.asyncio
    async def test_probe_returns_owner_id(self, async_db_initialized: StorageBackend) -> None:
        """The probe carries the owner backing the owner-only visibility check."""
        repo = ContextRepository(async_db_initialized)
        context_id, _ = await repo.store_with_deduplication(
            thread_id='probe-thread',
            source='user',
            content_type='text',
            text_content='probe target',
            owner_id='alice',
            visibility='private',
        )

        probe = await repo.check_entry_exists(context_id)
        assert probe.exists is True
        assert probe.source == 'user'
        assert probe.owner_id == 'alice'

    @pytest.mark.asyncio
    async def test_probe_missing_entry_is_all_none(self, async_db_initialized: StorageBackend) -> None:
        """A missing entry probes as (False, None, None, None)."""
        repo = ContextRepository(async_db_initialized)
        probe = await repo.check_entry_exists('0' * 32)
        assert probe == (False, None, None, None)
