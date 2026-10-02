"""Tests for apply_chunking_migration() on PostgreSQL with a mocked backend.

Covers the id BIGSERIAL column on vec_context_embeddings, idempotence, and the
applied-probe under the fp32 and compressed layouts.
"""

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest


class TestApplyChunkingMigrationPostgreSQL:
    """Tests for apply_chunking_migration() with PostgreSQL backend."""

    @pytest.mark.asyncio
    async def test_migration_postgresql_adds_id_column(self) -> None:
        """Verify id BIGSERIAL column added for PostgreSQL."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
            'ENABLE_SEMANTIC_SEARCH': 'true',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            # First call: check if id column exists (returns False)
            mock_backend.execute_read = AsyncMock(return_value=False)
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_chunking_migration

            await apply_chunking_migration(backend=mock_backend)

            # Should have called execute_read (to check id column) and execute_write (to apply)
            mock_backend.execute_read.assert_called_once()
            mock_backend.execute_write.assert_called_once()

    @pytest.mark.asyncio
    async def test_migration_postgresql_idempotent(self) -> None:
        """Verify PostgreSQL migration skips SQL execution when already applied."""
        env = {
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'postgresql',
            'POSTGRESQL_SCHEMA': 'public',
            'ENABLE_SEMANTIC_SEARCH': 'true',
        }

        import os

        with patch.dict(os.environ, env, clear=False):
            mock_backend = MagicMock()
            mock_backend.backend_type = 'postgresql'
            # id column already exists (migration already applied)
            mock_backend.execute_read = AsyncMock(return_value=True)
            mock_backend.execute_write = AsyncMock()

            from app.migrations import apply_chunking_migration

            # Should not raise
            await apply_chunking_migration(backend=mock_backend)

            # execute_read called to check if migration already applied
            mock_backend.execute_read.assert_called_once()
            # execute_write should NOT be called (early return when already applied)
            mock_backend.execute_write.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_chunk_index_column_postgresql_sql(self) -> None:
        """CRITICAL: Verify PostgreSQL SQL has NO chunk_index column (user decision)."""
        migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_chunking_postgresql.sql'

        assert migration_path.exists(), 'PostgreSQL migration file should exist'

        sql_content = migration_path.read_text(encoding='utf-8')

        # Verify NO chunk_index, chunk_start, chunk_end in SQL
        assert 'chunk_index' not in sql_content.lower(), 'chunk_index should NOT be in PostgreSQL migration'
        assert 'chunk_start' not in sql_content.lower(), 'chunk_start should NOT be in PostgreSQL migration'
        assert 'chunk_end' not in sql_content.lower(), 'chunk_end should NOT be in PostgreSQL migration'

        # Verify expected content is present
        assert 'id BIGSERIAL' in sql_content or 'id bigserial' in sql_content.lower(), (
            'id BIGSERIAL column should be in PostgreSQL migration'
        )
        assert 'chunk_count' in sql_content.lower(), 'chunk_count column should be in PostgreSQL migration'


class TestChunkingAppliedProbePostgreSQL:
    """Tests for the PostgreSQL applied-probe under fp32 and compressed layouts.

    The probe must short-circuit correctly in BOTH layouts. It targets the
    ``chunk_count`` column the migration adds to ``embedding_metadata`` (its final
    step, run in both layouts) rather than ``vec_context_embeddings.id``, which
    never exists under compression -- the compressed table replaces the fp32 one
    and ``skip_fp32_vec`` strips every ``vec_context_embeddings`` statement.
    """

    @pytest.mark.asyncio
    async def test_probe_returns_true_under_compressed_layout(self) -> None:
        """Applied=True when embedding_metadata has chunk_count and no fp32 vec table.

        Under compression the fp32 ``vec_context_embeddings`` table does not exist,
        so a ``vec_context_embeddings.id`` marker would always read False and re-run
        the migration body on every boot. The probe checks
        ``embedding_metadata.chunk_count`` (present in the compressed layout), so it
        reports the migration as applied.
        """
        from app.migrations.chunking import _check_chunking_migration_applied_postgresql

        captured: dict[str, object] = {}

        async def _fetchval(query: str, schema: str) -> bool:
            captured['query'] = query
            captured['schema'] = schema
            # Simulate the compressed layout: embedding_metadata.chunk_count exists,
            # while the fp32 vec_context_embeddings table (and its id column) does not.
            return True

        conn = MagicMock()
        conn.fetchval = AsyncMock(side_effect=_fetchval)

        with patch('app.migrations.chunking.settings') as mock_settings:
            mock_settings.storage.postgresql_schema = 'public'
            result = await _check_chunking_migration_applied_postgresql(conn)

        assert result is True
        # The probe must target the dual-layout marker, never the fp32-only column.
        query_text = str(captured['query'])
        assert "table_name = 'embedding_metadata'" in query_text
        assert "column_name = 'chunk_count'" in query_text
        assert 'vec_context_embeddings' not in query_text
        assert captured['schema'] == 'public'

    @pytest.mark.asyncio
    async def test_probe_returns_false_on_pre_chunking_layout(self) -> None:
        """Applied=False when embedding_metadata lacks chunk_count (pre-chunking).

        A database that has embedding storage but has not yet run the chunking
        migration carries no ``chunk_count`` column, so the probe must report the
        migration as not applied and let the body run.
        """
        from app.migrations.chunking import _check_chunking_migration_applied_postgresql

        conn = MagicMock()
        # No chunk_count column yet: the information_schema EXISTS check is False.
        conn.fetchval = AsyncMock(return_value=False)

        with patch('app.migrations.chunking.settings') as mock_settings:
            mock_settings.storage.postgresql_schema = 'public'
            result = await _check_chunking_migration_applied_postgresql(conn)

        assert result is False

    @pytest.mark.asyncio
    async def test_migration_body_skipped_when_probe_reports_applied_postgresql(self) -> None:
        """Compressed-PG boot: probe True short-circuits before any write.

        End-to-end at the ``apply_chunking_migration`` level under the DEFAULT
        compressed configuration (compression on, generation on, ``force=False``):
        with the probe reporting applied (the compressed-layout marker present),
        the migration must early-return and never open a write transaction, so a
        compressed database stops re-running the migration body every boot.
        """
        from app.migrations import apply_chunking_migration

        mock_backend = MagicMock()
        mock_backend.backend_type = 'postgresql'
        # Probe reports the migration already applied (chunk_count present).
        mock_backend.execute_read = AsyncMock(return_value=True)
        mock_backend.execute_write = AsyncMock()

        mock_settings = MagicMock()
        mock_settings.embedding.generation_enabled = True
        mock_settings.compression.enabled = True
        mock_settings.storage.postgresql_schema = 'public'
        mock_settings.storage.postgresql_migration_timeout_s = 300.0

        with patch('app.migrations.chunking.settings', mock_settings):
            await apply_chunking_migration(backend=mock_backend)

            mock_backend.execute_read.assert_called_once()
            mock_backend.execute_write.assert_not_called()
