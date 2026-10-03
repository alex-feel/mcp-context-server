"""Tests that embedding storage is provisioned from embedding generation, independent of the semantic-search tool
toggle (app/migrations/semantic.py, app/migrations/chunking.py).
"""

import os
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest

from tests.conftest import requires_sqlite_vec


class TestEmbeddingStorageDecoupledFromSearchTool:
    """Embedding storage is provisioned by GENERATION, not the search TOOL.

    The vec0 storage migrations gate on ``settings.embedding.generation_enabled``,
    so storage exists regardless of whether the search tool is exposed. With the
    search tool OFF but embedding generation ON, the fp32 vector table and chunk
    columns must still be created, or embedding writes fail for a missing-table
    reason.
    """

    @requires_sqlite_vec
    @pytest.mark.asyncio
    async def test_fp32_storage_created_with_generation_on_and_search_off(self, tmp_path: Path) -> None:
        """vec_context_embeddings + chunk columns are created and writable.

        Configuration under test (the fp32 edge):
        - embedding generation ON (settings.embedding.generation_enabled = True)
        - semantic-search TOOL forced OFF (mode='false', so the .enabled property is False)
        - embedding compression OFF (fp32 vec0 layout, not the compressed table)
        """
        from app.settings import get_settings

        db_path = tmp_path / 'test_storage_decoupled.db'

        # Create the base schema first (mirrors the other SQLite migration tests).
        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        with sqlite3.connect(str(db_path)) as conn:
            conn.executescript(schema_sql)

        env = {
            'DB_PATH': str(db_path),
            'MCP_TEST_MODE': '1',
            'STORAGE_BACKEND': 'sqlite',
            # Generation ON: provisions embedding storage.
            'ENABLE_EMBEDDING_GENERATION': 'true',
            # Search TOOL OFF: storage provisioning does not depend on it. Storage MUST still be built.
            'ENABLE_SEMANTIC_SEARCH': 'false',
            # Compression OFF: assert the fp32 vec0 layout, not the compressed table.
            'ENABLE_EMBEDDING_COMPRESSION': 'false',
            'EMBEDDING_DIM': '4',
        }

        # Build a fresh settings singleton under the env above and route both
        # migrations' module-level ``settings`` bindings to it, so the migrations run
        # (generation ON) even though the search tool is off. Restore the cache after.
        with patch.dict(os.environ, env, clear=False):
            get_settings.cache_clear()
            fresh_settings = get_settings()

            # Sanity-check the edge configuration is exactly what we intend to exercise.
            assert fresh_settings.embedding.generation_enabled is True
            assert fresh_settings.semantic_search.mode == 'false'
            assert fresh_settings.semantic_search.enabled is False
            assert fresh_settings.compression.enabled is False

            try:
                import app.backends.sqlite_backend.connections as sqlite_connections_module
                import app.backends.sqlite_backend.core as sqlite_core_module
                import app.backends.sqlite_backend.lifecycle as sqlite_lifecycle_module
                import app.backends.sqlite_backend.write_queue as sqlite_write_queue_module
                from app.migrations import chunking as chunking_module
                from app.migrations import semantic as semantic_module

                # The SQLite backend reads only ``settings.storage``, takes its database
                # path from its constructor and reports a fixed backend type, so its
                # import-time bindings already hold every value it reads under the env above.
                unread_by_backend = {'db_path', 'backend_type'}
                expected_storage = fresh_settings.storage.model_dump(exclude=unread_by_backend)
                for backend_module in (
                    sqlite_core_module,
                    sqlite_connections_module,
                    sqlite_write_queue_module,
                    sqlite_lifecycle_module,
                ):
                    assert backend_module.settings.storage.model_dump(exclude=unread_by_backend) == expected_storage

                with (
                    patch.object(semantic_module, 'settings', fresh_settings),
                    patch.object(chunking_module, 'settings', fresh_settings),
                ):
                    from app.backends.sqlite_backend import SQLiteBackend

                    backend = SQLiteBackend(db_path=str(db_path))
                    await backend.initialize()

                    try:
                        from app.migrations import apply_chunking_migration
                        from app.migrations import apply_semantic_search_migration

                        # Storage is provisioned by GENERATION, so both migrations run
                        # to completion even though the search TOOL is off.
                        await apply_semantic_search_migration(backend=backend)
                        await apply_chunking_migration(backend=backend)

                        def _inspect(conn: sqlite3.Connection) -> tuple[bool, bool, list[str], list[str]]:
                            cursor = conn.execute(
                                "SELECT name FROM sqlite_master "
                                "WHERE type='table' AND name='vec_context_embeddings'",
                            )
                            vec_table = cursor.fetchone() is not None

                            cursor = conn.execute(
                                "SELECT name FROM sqlite_master "
                                "WHERE type='table' AND name='embedding_chunks'",
                            )
                            chunks_table = cursor.fetchone() is not None

                            cursor = conn.execute('PRAGMA table_info(embedding_chunks)')
                            chunk_columns = [row[1] for row in cursor.fetchall()]

                            cursor = conn.execute('PRAGMA table_info(embedding_metadata)')
                            metadata_columns = [row[1] for row in cursor.fetchall()]

                            return vec_table, chunks_table, chunk_columns, metadata_columns

                        vec_table, chunks_table, chunk_columns, metadata_columns = await backend.execute_read(_inspect)

                        # The fp32 vector table exists despite the search tool being off.
                        assert vec_table, 'vec_context_embeddings must exist when embedding generation is on'
                        # The chunking layer (1:N bridge) is provisioned too.
                        assert chunks_table, 'embedding_chunks must exist when embedding generation is on'
                        assert 'context_id' in chunk_columns
                        assert 'vec_rowid' in chunk_columns
                        assert 'start_index' in chunk_columns
                        assert 'end_index' in chunk_columns
                        assert 'chunk_count' in metadata_columns

                        # A write into the vec table must NOT fail for a missing-table reason.
                        def _store_embedding(conn: sqlite3.Connection) -> int:
                            embedding = bytes([1, 2, 3, 4] * 4)  # 16 bytes = 4 float32 values
                            conn.execute(
                                'INSERT INTO vec_context_embeddings(rowid, embedding) VALUES (1, ?)',
                                (embedding,),
                            )
                            cursor = conn.execute('SELECT COUNT(*) FROM vec_context_embeddings')
                            return int(cursor.fetchone()[0])

                        stored_count = await backend.execute_write(_store_embedding)
                        assert stored_count == 1, 'store into vec_context_embeddings should succeed'
                    finally:
                        await backend.shutdown()
            finally:
                # Restore the process-wide settings singleton to the cached default so
                # this test does not leak its env-driven configuration into others.
                get_settings.cache_clear()
