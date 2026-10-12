"""Row-level security objects under a non-default POSTGRESQL_SCHEMA.

The migration names every object by the configured schema, so the helpers, the guard
trigger, the policies and the read view land in ``mcp_test`` and none in ``public``; a
second run over the current objects changes nothing; a start with the flag off is
refused while they exist; and removal takes every one of them out of ``mcp_test``
again.
"""

from __future__ import annotations

import asyncio
import contextlib

import asyncpg
import pytest

from app.backends import create_backend
from app.errors import ConfigurationError
from app.migrations import apply_rls
from app.migrations import guard_rls_disable_over_enabled
from app.migrations import remove_rls
from app.migrations.rls_ddl import policy_names
from app.settings import AppSettings
from app.settings import get_settings
from app.startup import init_database
from tests.integration.postgresql._rls_catalog import object_oids
from tests.integration.postgresql._rls_catalog import policies
from tests.integration.postgresql.conftest import NON_DEFAULT_SCHEMA
from tests.integration.postgresql.non_default_schema._schema import configure_non_default_env
from tests.integration.postgresql.non_default_schema._schema import install_search_path_patch

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]


def _with_rls(settings: AppSettings, *, enabled: bool) -> AppSettings:
    access_control = settings.access_control.model_copy(update={'postgresql_rls_enabled': enabled})
    return settings.model_copy(update={'access_control': access_control})


async def _snapshot(url: str) -> tuple[dict[str, int], dict[str, int], dict[tuple[str, str], object]]:
    """Return the object OIDs of ``mcp_test`` and ``public`` and the policies of ``mcp_test``."""
    conn = await asyncpg.connect(url)
    try:
        return (
            await object_oids(conn, NON_DEFAULT_SCHEMA),
            await object_oids(conn, 'public'),
            dict(await policies(conn, NON_DEFAULT_SCHEMA)),
        )
    finally:
        await conn.close()


def test_rls_objects_live_in_the_configured_schema_and_come_out_again(
    pg_non_default_schema_db: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_search_path_patch(monkeypatch, NON_DEFAULT_SCHEMA)
    configure_non_default_env(monkeypatch, pg_non_default_schema_db)
    flag_off = get_settings()
    flag_on = _with_rls(flag_off, enabled=True)

    async def _scenario() -> None:
        backend = create_backend(backend_type='postgresql', connection_string=pg_non_default_schema_db)
        await backend.initialize()
        try:
            await init_database(backend=backend)
            await apply_rls(backend, flag_on)
            first, public_objects, installed = await _snapshot(pg_non_default_schema_db)

            assert ('context_entries', 'context_entries_read') in installed
            assert {name for table, name in installed if table == 'context_entry_grants'} == set(
                policy_names('context_entry_grants'),
            )
            assert 'view context_entries_readable' in first
            assert {'function mcp_context_principal_id', 'function mcp_context_guard_access_columns'} <= set(first)
            assert not any(key.startswith(('policy ', 'view ', 'function mcp_context_')) for key in public_objects)

            await apply_rls(backend, flag_on)
            assert (await _snapshot(pg_non_default_schema_db))[0] == first

            with pytest.raises(ConfigurationError, match=r'\(tables: context_entries, context_entry_grants'):
                await guard_rls_disable_over_enabled(backend, flag_off)

            report = await remove_rls(backend, flag_off, dry_run=False, force=False)
            assert 'context_entries' in report.disabled
            remaining, _, left = await _snapshot(pg_non_default_schema_db)
            assert left == {}
            assert not any(key.startswith(('policy ', 'view ', 'function mcp_context_')) for key in remaining)
            await guard_rls_disable_over_enabled(backend, flag_off)
        finally:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(backend.shutdown(), timeout=10.0)

    asyncio.run(_scenario())
