"""Tests for `ContextRepository.check_latest_is_duplicate`, the read-only pre-check the store tools run before generation."""

import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.repositories import RepositoryContainer


@pytest_asyncio.fixture
async def repos(backend: StorageBackend) -> RepositoryContainer:
    """Create a RepositoryContainer with the test database manager."""
    return RepositoryContainer(backend)


@pytest.mark.asyncio
class TestDuplicatePreCheck:
    """Tests for check_latest_is_duplicate pre-check method."""

    async def test_check_latest_is_duplicate_found(self, repos: RepositoryContainer) -> None:
        """Pre-check detects duplicate content."""
        context_id, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Duplicate text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='user', text_content='Duplicate text',
        )
        assert result is not None
        assert result.context_id == context_id

    async def test_check_latest_is_duplicate_not_found(self, repos: RepositoryContainer) -> None:
        """Pre-check returns None for different content."""
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Original text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='user', text_content='Different text',
        )
        assert result is None

    async def test_check_latest_is_duplicate_empty_thread(self, repos: RepositoryContainer) -> None:
        """Pre-check returns None for empty thread."""
        result = await repos.context.check_latest_is_duplicate(
            thread_id='nonexistent', source='user', text_content='Any text',
        )
        assert result is None

    async def test_check_latest_different_source(self, repos: RepositoryContainer) -> None:
        """Pre-check returns None when source differs."""
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Same text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='agent', text_content='Same text',
        )
        assert result is None

    async def test_check_latest_only_checks_latest(self, repos: RepositoryContainer) -> None:
        """Pre-check only checks the latest entry, not older ones."""
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='Old matching text', metadata=None,
        )
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='test-thread', source='user', content_type='text',
            text_content='New different text', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='test-thread', source='user', text_content='Old matching text',
        )
        assert result is None  # Latest is "New different text", not matching

    async def test_precheck_returns_summary_from_same_snapshot(
        self, repos: RepositoryContainer,
    ) -> None:
        """The candidate's stored summary rides the SAME statement as the hash match.

        Reading the summary in a separate later statement could observe a row
        version a concurrent update committed in between, pairing a reused
        summary with text it does not describe -- and the dedup UPDATE's
        content-hash predicate cannot tell a revision-consistent row from a
        restored one, so the mismatched summary would persist via COALESCE.
        """
        ctx_id, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='snap-t1', source='user', content_type='text',
            text_content='Snapshot text', metadata=None, summary='Snapshot summary',
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='snap-t1', source='user', text_content='Snapshot text',
        )
        assert result is not None
        assert result.context_id == ctx_id
        assert result.summary == 'Snapshot summary'

    async def test_precheck_summary_none_when_entry_has_no_summary(
        self, repos: RepositoryContainer,
    ) -> None:
        """A candidate without a stored summary yields summary=None in the snapshot."""
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='snap-t2', source='user', content_type='text',
            text_content='No summary here', metadata=None,
        )
        result = await repos.context.check_latest_is_duplicate(
            thread_id='snap-t2', source='user', text_content='No summary here',
        )
        assert result is not None
        assert result.summary is None


@pytest.mark.asyncio
class TestBatchPreCheckInterleaving:
    """Tests for the batch pre-check optimization and its interaction with interleaving.

    The batch pre-check calls check_latest_is_duplicate before generating
    embeddings/summaries. These tests verify the pre-check returns correct
    results in batch-relevant scenarios.
    """

    async def test_batch_precheck_returns_id_for_duplicate(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Pre-check returns existing ID when entry is a genuine duplicate (no interleaving).

        In batch context, this would cause the tool layer to skip embedding/summary
        generation for this entry, saving LLM API calls.
        """
        ctx_id, _ = await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='batch-t1', source='user', content_type='text',
            text_content='Batch duplicate', metadata=None,
        )
        # Pre-check should identify this as a duplicate
        result = await repos.context.check_latest_is_duplicate(
            thread_id='batch-t1', source='user', text_content='Batch duplicate',
        )
        assert result is not None, 'Pre-check should find the genuine duplicate'
        assert result.context_id == ctx_id, (
            'Pre-check should return existing ID for genuine duplicate'
        )

    async def test_batch_precheck_returns_none_when_interleaving(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Pre-check returns None when interleaving is detected.

        In batch context, this would cause the tool layer to generate new
        embeddings/summaries for this entry (new conversational turn).
        """
        # Store user entry, then agent entry (creates interleaving)
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='batch-t1', source='user', content_type='text',
            text_content='Batch entry', metadata=None,
        )
        await repos.context.store_with_deduplication(
            owner_id='local',
            visibility='private',
            thread_id='batch-t1', source='agent', content_type='text',
            text_content='Agent response', metadata=None,
        )
        # Pre-check for same user text should return None (interleaving detected)
        result = await repos.context.check_latest_is_duplicate(
            thread_id='batch-t1', source='user', text_content='Batch entry',
        )
        assert result is None, (
            'Pre-check should return None when interleaving detected '
            '(agent entry exists after user candidate)'
        )
