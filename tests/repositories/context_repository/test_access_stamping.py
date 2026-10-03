"""Tests for access-control stamping at the repository write paths.

Covers store_with_deduplication (the caller's principal and the visibility
stamped on INSERT, never touched by a deduplication UPDATE, which only the
owner's retransmit reaches), update_context_entry (visibility rides the dynamic
SET list and the compare-and-set), and the check_entry_exists probe's owner_id
field -- all against a real temp SQLite database built from the full base
schema.
"""

import sqlite3

import pytest

from app.access_scope import AccessScope
from app.backends import StorageBackend
from app.repositories.context_repository import ContextRepository

ALICE = AccessScope('alice', frozenset())
MALLORY = AccessScope('mallory', frozenset())


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
            scope=ALICE,
            visibility='public',
        )
        assert was_updated is False
        owner, visibility, _ = await _read_access_columns(async_db_initialized, context_id)
        assert (owner, visibility) == ('alice', 'public')

    @pytest.mark.asyncio
    async def test_owner_retransmit_keeps_owner_and_visibility(self, async_db_initialized: StorageBackend) -> None:
        """The owner's retransmit updates the row and leaves its owner and visibility
        untouched, whatever visibility the retransmit carries."""
        repo = ContextRepository(async_db_initialized)
        context_id, _ = await repo.store_with_deduplication(
            thread_id='stamp-thread',
            source='agent',
            content_type='text',
            text_content='dedup-preserved entry',
            scope=ALICE,
            visibility='private',
        )

        dedup_id, was_updated = await repo.store_with_deduplication(
            thread_id='stamp-thread',
            source='agent',
            content_type='text',
            text_content='dedup-preserved entry',
            scope=ALICE,
            visibility='public',
        )
        assert was_updated is True
        assert dedup_id == context_id
        owner, visibility, version = await _read_access_columns(async_db_initialized, context_id)
        assert (owner, visibility, version) == ('alice', 'private', 1)

    @pytest.mark.asyncio
    async def test_foreign_identical_text_inserts_a_row_of_its_own(self, async_db_initialized: StorageBackend) -> None:
        """Another principal's identical text never merges into a row it can read but does
        not own: it lands as a new row owned by that principal, and the owner's row stays
        unchanged."""
        repo = ContextRepository(async_db_initialized)
        context_id, _ = await repo.store_with_deduplication(
            thread_id='stamp-thread',
            source='agent',
            content_type='text',
            text_content='dedup-preserved entry',
            scope=ALICE,
            visibility='public',
        )

        mallory_id, was_updated = await repo.store_with_deduplication(
            thread_id='stamp-thread',
            source='agent',
            content_type='text',
            text_content='dedup-preserved entry',
            scope=MALLORY,
            visibility='private',
        )
        assert was_updated is False
        assert mallory_id != context_id
        assert await _read_access_columns(async_db_initialized, mallory_id) == ('mallory', 'private', 0)
        assert await _read_access_columns(async_db_initialized, context_id) == ('alice', 'public', 0)


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
            scope=ALICE,
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
            scope=ALICE,
            visibility='private',
        )
        probe = await repo.check_entry_exists(context_id)
        assert probe.version is not None

        success, fields = await repo.update_context_entry(
            context_id,
            visibility='public',
            expected_version=probe.version,
        )
        assert success is True
        assert fields == ['visibility']
        _, visibility, version = await _read_access_columns(async_db_initialized, context_id)
        assert visibility == 'public'
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
            scope=ALICE,
            visibility='public',
        )

        success, fields = await repo.update_context_entry(context_id, text_content='new text')
        assert success is True
        assert 'visibility' not in fields
        owner, visibility, _ = await _read_access_columns(async_db_initialized, context_id)
        assert (owner, visibility) == ('alice', 'public')


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
            scope=ALICE,
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
