"""Tests for the order of the database preparation phase in app.startup.database_setup.

Starting with ``ENABLE_POSTGRESQL_RLS`` off over the row-level security objects is
refused right after the compression guard, before any provisioning migration runs, and
the row-level security migration runs last, once every table and column this boot
provisions exists, so the read view lists them all.
"""

from types import SimpleNamespace
from typing import cast

import pytest

import app.startup.database_setup as database_setup
from app.backends import StorageBackend
from app.settings import AppSettings
from app.startup.database_setup import prepare_database

_ASYNC_STEPS = (
    'init_database',
    'handle_metadata_indexes',
    'guard_compression_disable_over_populated',
    'guard_rls_disable_over_enabled',
    'apply_semantic_search_migration',
    'apply_jsonb_merge_patch_migration',
    'apply_function_search_path_migration',
    'apply_fts_migration',
    'apply_chunking_migration',
    'apply_summary_migration',
    'apply_content_hash_migration',
    'apply_version_migration',
    'apply_access_control_migration',
    'apply_tag_uniqueness_migration',
    'apply_compression_migration',
    'apply_index_tree_migration',
    'validate_compression_provenance',
    '_announce_compression',
    'apply_rls',
)


@pytest.mark.asyncio
async def test_the_rls_guard_runs_early_and_the_rls_migration_runs_last(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, tuple[object, ...]]] = []

    def _recorder(name: str) -> object:
        async def _step(*args: object, **kwargs: object) -> None:
            calls.append((name, (*args, *kwargs.values())))

        return _step

    for name in _ASYNC_STEPS:
        monkeypatch.setattr(database_setup, name, _recorder(name))

    def _validate_pool_acquire_timeout() -> None:
        calls.append(('validate_pool_acquire_timeout', ()))

    monkeypatch.setattr(database_setup, 'validate_pool_acquire_timeout', _validate_pool_acquire_timeout)
    backend = cast(StorageBackend, SimpleNamespace(backend_type='postgresql'))
    settings = cast(AppSettings, SimpleNamespace())

    await prepare_database(backend, settings)

    names = [name for name, _ in calls]
    assert names.index('guard_rls_disable_over_enabled') == names.index('guard_compression_disable_over_populated') + 1
    assert names.index('guard_rls_disable_over_enabled') < names.index('apply_semantic_search_migration')
    assert names[-2:] == ['_announce_compression', 'apply_rls']
    arguments = dict(calls)
    assert arguments['guard_rls_disable_over_enabled'] == (backend, settings)
    assert arguments['apply_rls'] == (backend, settings)
