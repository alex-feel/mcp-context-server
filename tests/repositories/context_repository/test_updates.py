"""Tests for ContextRepository metadata patches and the version compare-and-set on updates."""

import json
import sqlite3

import pytest

from app.ids import generate_id
from app.repositories import RepositoryContainer
from app.repositories.context_repository import ContextRepository
from tests.helpers import LOCAL_SCOPE


class TestContextRepositoryPatchMetadata:
    """Test patch_metadata method of ContextRepository (RFC 7396)."""

    @pytest.mark.asyncio
    async def test_patch_metadata_adds_new_key(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching adds a new key to existing metadata."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='patch-add-thread',
            source='user',
            content_type='text',
            text_content='Patch test entry',
            metadata=json.dumps({'existing': 'value'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {'new_key': 'new_value'})
        assert success is True
        assert 'metadata' in fields

        rows, _ = await context_repo.search_contexts(thread_id='patch-add-thread')
        assert len(rows) == 1
        meta = json.loads(rows[0]['metadata'])
        assert meta['existing'] == 'value'
        assert meta['new_key'] == 'new_value'

    @pytest.mark.asyncio
    async def test_patch_metadata_updates_existing_key(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching updates an existing key's value."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='patch-update-thread',
            source='user',
            content_type='text',
            text_content='Patch update test',
            metadata=json.dumps({'status': 'pending'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {'status': 'done'})
        assert success is True

        rows, _ = await context_repo.search_contexts(thread_id='patch-update-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['status'] == 'done'

    @pytest.mark.asyncio
    async def test_patch_metadata_deletes_key_with_null(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Patching with null value deletes the key (RFC 7396)."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='patch-delete-thread',
            source='user',
            content_type='text',
            text_content='Patch delete test',
            metadata=json.dumps({'keep': 'yes', 'remove': 'me'}),
        )
        success, _ = await context_repo.patch_metadata(ctx_id, {'remove': None})
        assert success is True

        rows, _ = await context_repo.search_contexts(thread_id='patch-delete-thread')
        meta = json.loads(rows[0]['metadata'])
        assert 'keep' in meta
        assert 'remove' not in meta

    @pytest.mark.asyncio
    async def test_patch_metadata_nonexistent_entry(
        self, context_repo: ContextRepository,
    ) -> None:
        """Patching nonexistent entry returns (False, [])."""
        success, fields = await context_repo.patch_metadata(generate_id(), {'key': 'value'})
        assert success is False
        assert fields == []

    @pytest.mark.asyncio
    async def test_patch_metadata_empty_patch(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Empty patch is a no-op for data but updates timestamp."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='patch-empty-thread',
            source='user',
            content_type='text',
            text_content='Patch empty test',
            metadata=json.dumps({'unchanged': 'value'}),
        )
        success, fields = await context_repo.patch_metadata(ctx_id, {})
        assert success is True
        assert 'metadata' in fields

        rows, _ = await context_repo.search_contexts(thread_id='patch-empty-thread')
        meta = json.loads(rows[0]['metadata'])
        assert meta['unchanged'] == 'value'


class TestContextRepositoryVersionCAS:
    """Tests for the optimistic-concurrency version guard on ``update_context_entry``.

    Covers the ``version`` column and the ``expected_version`` compare-and-set.
    """

    async def _read_row(
        self, context_repo: ContextRepository, context_id: str,
    ) -> tuple[str, int]:
        """Read (text_content, version) for an entry via a direct SELECT.

        check_entry_exists returns version but NOT text_content, so a direct
        SELECT is the most precise way to assert the row was (or was not) mutated.

        Returns:
            Tuple of (text_content, version) for the row.
        """

        def _select(conn: sqlite3.Connection) -> tuple[str, int]:
            cursor = conn.cursor()
            cursor.execute(
                'SELECT text_content, version FROM context_entries WHERE id = ?',
                (context_id,),
            )
            row = cursor.fetchone()
            assert row is not None
            return str(row['text_content']), int(row['version'])

        return await context_repo.backend.execute_read(_select)

    @pytest.mark.asyncio
    async def test_initial_version_is_zero(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """A freshly inserted entry starts at version 0 (schema DEFAULT 0)."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='cas-init-thread',
            source='user',
            content_type='text',
            text_content='Initial version content',
        )

        probe = await repos.context.check_entry_exists(ctx_id, scope=LOCAL_SCOPE)
        assert probe.exists is True
        assert probe.version == 0

        _text, row_version = await self._read_row(context_repo, ctx_id)
        assert row_version == 0

    @pytest.mark.asyncio
    async def test_cas_success_bumps_version(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """update with expected_version=0 succeeds and bumps version to 1."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='cas-success-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        success, fields = await repos.context.update_context_entry(
            ctx_id, text_content='v1', expected_version=0,
        )
        assert success is True
        assert 'text_content' in fields

        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'v1'
        assert version == 1

    @pytest.mark.asyncio
    async def test_stale_expected_version_raises_and_leaves_row_unchanged(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """A second CAS with a now-stale expected_version=0 raises
        VersionConflictError and does NOT mutate the row.
        """
        from app.repositories.context_repository.records import VersionConflictError

        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='cas-stale-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        # First update advances version 0 -> 1.
        await repos.context.update_context_entry(ctx_id, text_content='v1', expected_version=0)

        # Second update reuses the now-stale captured version 0.
        with pytest.raises(VersionConflictError) as exc_info:
            await repos.context.update_context_entry(ctx_id, text_content='v2', expected_version=0)
        assert exc_info.value.context_id == ctx_id

        # Row is untouched: text stays 'v1', version stays 1 (no spurious bump).
        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'v1'
        assert version == 1

    @pytest.mark.asyncio
    async def test_cas_with_current_version_succeeds(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """Re-reading the current version (1) and retrying CAS succeeds, bumping to 2."""
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='cas-current-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        await repos.context.update_context_entry(ctx_id, text_content='v1', expected_version=0)

        success, _fields = await repos.context.update_context_entry(
            ctx_id, text_content='v3', expected_version=1,
        )
        assert success is True

        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'v3'
        assert version == 2

    @pytest.mark.asyncio
    async def test_unguarded_path_no_cas_no_version_bump(
        self, context_repo: ContextRepository, repos: RepositoryContainer,
    ) -> None:
        """expected_version=None runs the unguarded update: it succeeds with NO
        CAS predicate and does NOT bump the version column.
        """
        ctx_id, _ = await repos.context.store_with_deduplication(
            scope=LOCAL_SCOPE,
            visibility='private',
            thread_id='cas-legacy-thread',
            source='user',
            content_type='text',
            text_content='v0',
        )

        success, fields = await repos.context.update_context_entry(
            ctx_id, text_content='legacy', expected_version=None,
        )
        assert success is True
        assert 'text_content' in fields

        text, version = await self._read_row(context_repo, ctx_id)
        assert text == 'legacy'
        # The unguarded path leaves the version untouched (no SET version = version + 1).
        assert version == 0

    @pytest.mark.asyncio
    async def test_cas_against_nonexistent_id_returns_false(
        self, repos: RepositoryContainer,
    ) -> None:
        """A CAS against a non-existent id returns (False, []), NOT a
        VersionConflictError -- the existence pre-check distinguishes
        "no such row" from "row exists but version moved".
        """
        success, fields = await repos.context.update_context_entry(
            generate_id(), text_content='ghost', expected_version=0,
        )
        assert success is False
        assert fields == []
