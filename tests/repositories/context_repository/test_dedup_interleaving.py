"""Tests for the interleaving check in `ContextRepository.store_with_deduplication`.

An opposite-source entry after the candidate turns an identical store into a new entry.
"""

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.repositories import RepositoryContainer


@pytest_asyncio.fixture
async def repos(backend: StorageBackend) -> RepositoryContainer:
    """Create a RepositoryContainer with the test database manager."""
    return RepositoryContainer(backend)


@pytest.mark.asyncio
class TestDeduplicationInterleaving:
    """Tests for the interleaving check in deduplication logic.

    Validates that deduplication is suppressed when opposite-source entries
    exist between the candidate duplicate and the new entry, preserving
    chronological ordering for repeated identical messages in conversations.
    """

    async def test_dedup_blocked_when_opposite_source_interleaves(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Dedup is suppressed when an opposite-source entry exists after the candidate."""
        # User says "Proceed"
        id_a, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Proceed', metadata=None,
        )
        # Agent responds
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='agent', content_type='text',
            text_content='Working on it', metadata=None,
        )
        # User says "Proceed" again (new conversational turn)
        id_c, was_updated = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Proceed', metadata=None,
        )
        assert id_c != id_a, 'Should create a new entry, not update the old one'
        assert id_c > id_a, 'New entry should have a higher id'
        assert was_updated is False, 'Should be an insertion, not an update'
        # Verify original entry still exists unchanged
        entries = await repos.context.get_by_ids([id_a])
        assert len(entries) == 1
        assert entries[0]['text_content'] == 'Proceed'

    async def test_dedup_allowed_when_no_interleaving(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Dedup proceeds normally for genuine rapid duplicates (retry protection)."""
        id_a, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Hello', metadata=None,
        )
        id_b, was_updated = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Hello', metadata=None,
        )
        assert id_b == id_a, 'Should update the existing entry'
        assert was_updated is True

    async def test_dedup_blocked_symmetrically_for_agent(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Interleaving check works for agent source with user interleaving."""
        id_a, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='agent', content_type='text',
            text_content='status update', metadata=None,
        )
        # User interleaves
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='acknowledged', metadata=None,
        )
        # Agent sends identical text again
        id_c, was_updated = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='agent', content_type='text',
            text_content='status update', metadata=None,
        )
        assert id_c != id_a
        assert was_updated is False

    async def test_dedup_precheck_respects_interleaving(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Pre-check method returns None when interleaving is detected."""
        id_a, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Proceed', metadata=None,
        )
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='agent', content_type='text',
            text_content='Done', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='t1', source='user', text_content='Proceed',
        )
        assert result is None, (
            f'Expected None (interleaving suppresses dedup), got {result}'
        )

    async def test_dedup_single_source_thread(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Dedup works normally in a single-source thread (no opposite-source entries)."""
        id_a, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Only users here', metadata=None,
        )
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Different message', metadata=None,
        )
        # Store duplicate of first when latest is different -- should insert (not latest)
        id_c, was_updated_c = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='t1', source='user', content_type='text',
            text_content='Different message', metadata=None,
        )
        # Latest matches, no opposite-source entries at all -- dedup should work
        assert was_updated_c is True

    async def test_chronological_ordering_preserved_after_fix(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """End-to-end ordering: alternating identical messages create distinct entries."""
        ids: list[str] = []
        sources = ['user', 'agent', 'user', 'agent', 'user']
        texts = ['Go', 'Done', 'Go', 'Done again', 'Go']
        for source, text in zip(sources, texts, strict=True):
            ctx_id, _ = await repos.context.store_with_deduplication(
                owner_id='local',
                visibility='private',
                thread_id='t1', source=source, content_type='text',
                text_content=text, metadata=None,
            )
            ids.append(ctx_id)
        # All 5 should be distinct entries
        assert len(set(ids)) == 5, f'Expected 5 distinct ids, got {ids}'
        # Monotonically increasing
        for i in range(1, len(ids)):
            assert ids[i] > ids[i - 1], f'ids not monotonically increasing: {ids}'
        # Verify all entries exist
        entries, _ = await repos.context.search_contexts(thread_id='t1', limit=1000)
        assert len(entries) == 5
