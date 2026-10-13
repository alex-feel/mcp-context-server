"""The row-level security migration across boots, against a real PostgreSQL server.

The database is ``pg_rls_db``, owned by the non-superuser ``mcp_owner``, which every boot
here prepares through ``prepare_database`` on a maintenance backend, as the server does
with ``ENABLE_POSTGRESQL_RLS``. These tests prove that a boot over a current database
issues no DDL and takes no lock, while a difference makes the migration wait for the
schema-init lock and repair only that; that the fp32 and compressed layouts, a database
without embedding storage and a node-summary table switched on later each end with
policies on exactly the tables that exist; that a new policy role re-creates every
policy and both read views while a new member login role changes nothing; that an object
altered outside the server, a column added to ``context_entries`` (which re-creates
``tags_readable`` with the read view it reads through), a ``tags_readable`` replaced
outside the server and an extra grantee on either view are repaired and named; that the
FTS language rebuild re-creates both views in its own transaction; and that a boot with
the flag off over installed objects is refused.
"""

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from typing import Final

import asyncpg
import pytest
import pytest_asyncio

from app.backends import StorageBackend
from app.backends.base import MAINTENANCE
from app.errors import ConfigurationError
from app.migrations import apply_request_role_grants
from app.migrations import apply_rls
from app.migrations import guard_rls_disable_over_enabled
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import parse_fingerprint
from app.migrations.rls_ddl import policy_names
from app.migrations.rls_view_ddl import read_view
from app.repositories import RepositoryContainer
from app.settings import get_settings
from app.startup.database_setup import prepare_database
from tests.integration.postgresql._rls_catalog import ADVISORY_WAITERS_SQL
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_LOCK_SQL
from tests.integration.postgresql._rls_catalog import SCHEMA_INIT_UNLOCK_SQL
from tests.integration.postgresql._rls_catalog import configure_rls_layout
from tests.integration.postgresql._rls_catalog import ddl_log
from tests.integration.postgresql._rls_catalog import install_ddl_log
from tests.integration.postgresql._rls_catalog import maintenance_backend
from tests.integration.postgresql._rls_catalog import object_oids
from tests.integration.postgresql._rls_catalog import policies
from tests.integration.postgresql._rls_catalog import row_security
from tests.integration.postgresql._rls_catalog import view_acl
from tests.integration.postgresql._rls_catalog import view_columns
from tests.integration.postgresql._rls_matrix import POLICY_ROLE
from tests.integration.postgresql._rls_matrix import REQUEST_ROLE
from tests.integration.postgresql._rls_matrix import rls_settings
from tests.integration.postgresql.conftest import RlsUrls
from tests.repositories._access_scope_cases import Layout

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

SCHEMA_NAME: Final = 'public'
SCHEMA: Final = '"public"'
VIEW: Final = f'{SCHEMA}.context_entries_readable'
TAGS_VIEW: Final = f'{SCHEMA}.tags_readable'
VIEW_OID: Final = 'SELECT to_regclass($1)::oid'
ALT_POLICY_ROLE: Final = 'mcp_rls_alt_policy'
ALT_LOGIN_ROLE: Final = 'mcp_rls_alt_login'
INSERT_PUBLIC_ENTRY: Final = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
    "VALUES ($1, 'rls-migration', 'agent', 'text', 'migration probe', 'alice', 'public')"
)


@pytest_asyncio.fixture
async def admin(pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A superuser connection to the database.

    Yields:
        The open connection, closed afterwards.
    """
    conn = await asyncpg.connect(pg_rls_db.admin)
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def owner(pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A connection of the table owner ``mcp_owner``, for changes made outside the server.

    Yields:
        The open connection, closed afterwards.
    """
    conn = await asyncpg.connect(pg_rls_db.maintenance)
    try:
        yield conn
    finally:
        await conn.close()


@pytest_asyncio.fixture
async def booted(pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch) -> AsyncGenerator[StorageBackend]:
    """The fp32 layout booted once with row-level security on, the policy role granted what requests need.

    Yields:
        The maintenance backend, logged in as ``mcp_owner``.
    """
    settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides={}, rls=True)
    async with maintenance_backend(pg_rls_db.maintenance) as backend:
        await prepare_database(backend, settings, request_role=None)
        await apply_request_role_grants(backend, settings, request_role=REQUEST_ROLE)
        yield backend


async def _managed_tables(conn: asyncpg.Connection) -> set[str]:
    return set(await row_security(conn, SCHEMA_NAME)) & set(MANAGED_TABLES)


async def _assert_installed(conn: asyncpg.Connection, *, policy_role: str = POLICY_ROLE) -> None:
    """Assert every managed table that exists carries exactly our policies, for the policy role, enabled without FORCE."""
    tables = await _managed_tables(conn)
    installed = await policies(conn, SCHEMA_NAME)
    assert set(installed) == {(table, name) for table in tables for name in policy_names(table)}
    assert {policy.roles for policy in installed.values()} == {(policy_role,)}
    assert all(parse_fingerprint(policy.comment) is not None for policy in installed.values())
    enabled = await row_security(conn, SCHEMA_NAME)
    assert {table: enabled[table] for table in tables} == dict.fromkeys(tables, (True, False))
    for view in (VIEW, TAGS_VIEW):
        view_owner = await conn.fetchval('SELECT pg_get_userbyid(relowner) FROM pg_class WHERE oid = to_regclass($1)', view)
        assert view_owner == 'mcp_owner', view
        assert await view_acl(conn, view) == [(policy_role, 'SELECT')], view


async def _apply_holding_the_lock(
    backend: StorageBackend, admin: asyncpg.Connection, urls: RlsUrls, *, policy_role: str = POLICY_ROLE,
) -> int:
    """Apply the migration while another session holds the schema-init lock.

    A migration that asked for the lock would wait for it, so the run must finish while
    the lock is still held.

    Returns:
        The most sessions seen waiting for an advisory lock during the run.
    """
    holder = await asyncpg.connect(urls.admin)
    await holder.execute(SCHEMA_INIT_LOCK_SQL)
    waiters = 0
    task = asyncio.create_task(apply_rls(backend, rls_settings(policy_role=policy_role)))
    try:
        for _ in range(3000):
            if task.done():
                break
            waiters = max(waiters, int(await admin.fetchval(ADVISORY_WAITERS_SQL)))
            await asyncio.sleep(0.01)
        assert task.done(), 'the migration waited for the schema-init lock'
    finally:
        await holder.execute(SCHEMA_INIT_UNLOCK_SQL)
        await holder.close()
        await asyncio.wait_for(task, timeout=30)
    return waiters


class TestNoDdlBoot:
    """A boot over a current database issues no DDL and takes no lock; a difference waits for the lock."""

    @pytest.mark.parametrize('layout', ['fp32', 'compressed'])
    @pytest.mark.asyncio
    async def test_a_second_boot_issues_no_ddl_and_takes_no_lock(
        self, layout: Layout, pg_rls_db: RlsUrls, admin: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        settings = configure_rls_layout(pg_rls_db, monkeypatch, layout=layout, overrides={}, rls=True)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            await prepare_database(backend, settings, request_role=None)
            await _assert_installed(admin)
            vector_table = 'vec_context_embeddings' if layout == 'fp32' else 'vec_context_embeddings_compressed'
            assert vector_table in await _managed_tables(admin)
            before = await object_oids(admin, SCHEMA_NAME)
            assert {'view context_entries_readable', 'view tags_readable'} <= set(before)
            assert {'trigger context_entries_readable.reject_writes', 'trigger tags_readable.reject_writes'} <= set(before)
            await install_ddl_log(admin)

            waiters = await _apply_holding_the_lock(backend, admin, pg_rls_db)

            assert waiters == 0
            assert await ddl_log(admin) == []
            assert await object_oids(admin, SCHEMA_NAME) == before

    @pytest.mark.asyncio
    async def test_a_difference_waits_for_the_lock_and_repairs_only_itself(
        self, booted: StorageBackend, pg_rls_db: RlsUrls, admin: asyncpg.Connection, owner: asyncpg.Connection,
    ) -> None:
        await owner.execute('DROP POLICY tags_insert ON tags')
        before = await object_oids(admin, SCHEMA_NAME)
        holder = await asyncpg.connect(pg_rls_db.admin)
        await holder.execute(SCHEMA_INIT_LOCK_SQL)
        try:
            task = asyncio.create_task(apply_rls(booted, rls_settings()))
            for _ in range(1500):
                if int(await admin.fetchval(ADVISORY_WAITERS_SQL)) == 1:
                    break
                await asyncio.sleep(0.01)
            assert int(await admin.fetchval(ADVISORY_WAITERS_SQL)) == 1
            assert not task.done()
        finally:
            await holder.execute(SCHEMA_INIT_UNLOCK_SQL)
            await holder.close()
        await asyncio.wait_for(task, timeout=30)

        after = await object_oids(admin, SCHEMA_NAME)
        assert set(after) - set(before) == {'policy tags.tags_insert'}
        assert {key: oid for key, oid in after.items() if key in before} == before
        await _assert_installed(admin)


class TestTablesThatExist:
    """Each boot covers the managed tables that exist, and a table created later is covered by the next boot."""

    @pytest.mark.asyncio
    async def test_a_database_without_embedding_storage_gets_policies_on_its_tables_alone(
        self, pg_rls_db: RlsUrls, admin: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        overrides = {'ENABLE_EMBEDDING_GENERATION': 'false'}
        settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides=overrides, rls=True)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            await prepare_database(backend, settings, request_role=None)

        assert await _managed_tables(admin) == {
            'context_entries', 'context_entry_grants', 'tags', 'image_attachments', 'context_index_nodes',
        }
        await _assert_installed(admin)

    @pytest.mark.asyncio
    async def test_the_node_summary_table_switched_on_later_gets_its_policies_at_the_next_boot(
        self, pg_rls_db: RlsUrls, admin: asyncpg.Connection, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        off = {'ENABLE_INDEX_TREE_NODE_SUMMARIES': 'false'}
        settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides=off, rls=True)
        async with maintenance_backend(pg_rls_db.maintenance) as backend:
            await prepare_database(backend, settings, request_role=None)
            assert 'context_index_nodes' not in await _managed_tables(admin)
            await _assert_installed(admin)
            before = await object_oids(admin, SCHEMA_NAME)

            on = {'ENABLE_INDEX_TREE_NODE_SUMMARIES': 'true'}
            settings = configure_rls_layout(pg_rls_db, monkeypatch, layout='fp32', overrides=on, rls=True)
            await prepare_database(backend, settings, request_role=None)

        await _assert_installed(admin)
        after = await object_oids(admin, SCHEMA_NAME)
        assert {key: oid for key, oid in after.items() if key in before} == before
        late = {f'policy context_index_nodes.{name}' for name in policy_names('context_index_nodes')}
        assert set(after) - set(before) == late
        read = (await policies(admin, SCHEMA_NAME))[('context_index_nodes', 'context_index_nodes_read')]
        fingerprint = parse_fingerprint(read.comment)
        assert fingerprint is not None
        assert fingerprint.preenabled is False


class TestRoles:
    """The policy role is part of every policy and of the view; a login role is part of nothing."""

    @pytest_asyncio.fixture
    async def alt_policy_role(self, admin: asyncpg.Connection) -> AsyncGenerator[str]:
        """A second NOLOGIN policy role, dropped with whatever names it afterwards.

        Yields:
            Its name.
        """
        await admin.execute(f'CREATE ROLE {ALT_POLICY_ROLE} NOLOGIN NOSUPERUSER NOBYPASSRLS')
        try:
            yield ALT_POLICY_ROLE
        finally:
            await admin.execute(f'DROP OWNED BY {ALT_POLICY_ROLE}')
            await admin.execute(f'DROP ROLE {ALT_POLICY_ROLE}')

    @pytest_asyncio.fixture
    async def alt_login_role(self, admin: asyncpg.Connection) -> AsyncGenerator[str]:
        """A second request login role, an inheriting member of the policy role.

        Yields:
            Its name, which is also its password.
        """
        await admin.execute(
            f"CREATE ROLE {ALT_LOGIN_ROLE} LOGIN PASSWORD '{ALT_LOGIN_ROLE}' NOSUPERUSER NOBYPASSRLS INHERIT",
        )
        await admin.execute(f'GRANT {POLICY_ROLE} TO {ALT_LOGIN_ROLE}')
        try:
            yield ALT_LOGIN_ROLE
        finally:
            await admin.execute(f'DROP ROLE {ALT_LOGIN_ROLE}')

    @pytest.mark.asyncio
    async def test_a_new_policy_role_re_creates_every_policy_and_the_view(
        self, booted: StorageBackend, admin: asyncpg.Connection, alt_policy_role: str,
    ) -> None:
        before = await policies(admin, SCHEMA_NAME)
        view_oids = [await admin.fetchval(VIEW_OID, view) for view in (VIEW, TAGS_VIEW)]

        await apply_rls(booted, rls_settings(policy_role=alt_policy_role))

        after = await policies(admin, SCHEMA_NAME)
        assert set(after) == set(before)
        assert all(after[key].oid != before[key].oid for key in before)
        await _assert_installed(admin, policy_role=alt_policy_role)
        after_oids = [await admin.fetchval(VIEW_OID, view) for view in (VIEW, TAGS_VIEW)]
        assert all(current != earlier for current, earlier in zip(after_oids, view_oids, strict=True))
        assert f"'{alt_policy_role}'" in await admin.fetchval('SELECT pg_get_viewdef(to_regclass($1), true)', VIEW)

        await install_ddl_log(admin)
        await apply_rls(booted, rls_settings(policy_role=alt_policy_role))
        assert await ddl_log(admin) == []

    @pytest.mark.asyncio
    async def test_a_new_member_login_role_changes_nothing_and_reads_through_the_view(
        self, booted: StorageBackend, pg_rls_db: RlsUrls, admin: asyncpg.Connection, owner: asyncpg.Connection,
        alt_login_role: str,
    ) -> None:
        before = await object_oids(admin, SCHEMA_NAME)
        await install_ddl_log(admin)

        await apply_rls(booted, rls_settings())
        report = await apply_request_role_grants(booted, rls_settings(), request_role=alt_login_role)

        assert report.statements == []
        assert not report.locked
        assert await ddl_log(admin) == []
        assert await object_oids(admin, SCHEMA_NAME) == before
        await owner.execute(INSERT_PUBLIC_ENTRY, uuid.uuid4())
        login_url = pg_rls_db.app.replace(f'{REQUEST_ROLE}:{REQUEST_ROLE}@', f'{alt_login_role}:{alt_login_role}@')
        login = await asyncpg.connect(login_url)
        try:
            assert await login.fetchval('SELECT current_user') == alt_login_role
            assert await login.fetchval(f'SELECT count(*) FROM {VIEW}') == 1
            assert await login.fetchval(f'SELECT count(*) FROM {TAGS_VIEW}') == 0
        finally:
            await login.close()


class TestRepairs:
    """What changed outside the server is put back at the next boot and named in a WARNING."""

    @pytest.mark.asyncio
    async def test_an_altered_policy_is_re_created(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        before = await object_oids(admin, SCHEMA_NAME)
        await owner.execute('ALTER POLICY tags_read ON tags USING (true)')

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        after = await object_oids(admin, SCHEMA_NAME)
        assert after['policy tags.tags_read'] != before['policy tags.tags_read']
        assert {key: oid for key, oid in after.items() if key != 'policy tags.tags_read'} == {
            key: oid for key, oid in before.items() if key != 'policy tags.tags_read'
        }
        assert await admin.fetchval(
            "SELECT pg_get_expr(polqual, polrelid) FROM pg_policy WHERE polname = 'tags_read'",
        ) != 'true'
        fingerprint = parse_fingerprint((await policies(admin, SCHEMA_NAME))[('tags', 'tags_read')].comment)
        assert fingerprint is not None
        assert fingerprint.preenabled is False
        assert any(f'POLICY tags_read ON {SCHEMA}.tags' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_a_column_added_re_creates_both_views_with_their_acl_and_trigger_naming_the_grants_it_drops(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute('ALTER TABLE context_entries ADD COLUMN extra_probe text')
        await owner.execute(f'GRANT SELECT ON {VIEW} TO mcp_outsider')
        policies_before = await policies(admin, SCHEMA_NAME)
        tags_view_oid = await admin.fetchval(VIEW_OID, TAGS_VIEW)

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        columns = await view_columns(admin, VIEW)
        assert columns == await view_columns(admin, f'{SCHEMA}.context_entries')
        assert columns[-1] == 'extra_probe'
        assert await admin.fetchval(VIEW_OID, TAGS_VIEW) != tags_view_oid
        for relation in (VIEW, TAGS_VIEW):
            assert await view_acl(admin, relation) == [(POLICY_ROLE, 'SELECT')], relation
            assert await admin.fetchval(
                "SELECT count(*) FROM pg_trigger WHERE tgrelid = to_regclass($1) AND tgname = 'reject_writes' "
                "AND tgenabled <> 'D'",
                relation,
            ) == 1, relation
        view = read_view([f'"{column}"' for column in columns], schema=SCHEMA, role_name=POLICY_ROLE)
        fingerprint = parse_fingerprint(await admin.fetchval("SELECT obj_description(to_regclass($1), 'pg_class')", VIEW))
        assert fingerprint is not None
        assert fingerprint.sha == view.sha
        assert fingerprint.definition == digest(await read_definition(admin, view))
        assert {key: policy.oid for key, policy in (await policies(admin, SCHEMA_NAME)).items()} == {
            key: policy.oid for key, policy in policies_before.items()
        }
        assert any('mcp_outsider (SELECT)' in record.getMessage() for record in caplog.records)

    @pytest.mark.asyncio
    async def test_an_extra_grantee_on_the_unchanged_view_is_revoked_and_named(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute(f'GRANT SELECT ON {VIEW} TO mcp_outsider')
        await owner.execute(f'GRANT INSERT ON {VIEW} TO {POLICY_ROLE}')
        before = await object_oids(admin, SCHEMA_NAME)

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        assert await object_oids(admin, SCHEMA_NAME) == before
        assert await view_acl(admin, VIEW) == [(POLICY_ROLE, 'SELECT')]
        [warning] = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert 'mcp_outsider (SELECT)' in warning
        assert f'{POLICY_ROLE} (INSERT)' in warning

    @pytest.mark.asyncio
    async def test_a_stray_grantee_on_the_unchanged_tags_view_is_revoked_with_a_warning_naming_it(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute(f'GRANT SELECT ON {TAGS_VIEW} TO mcp_outsider')
        before = await object_oids(admin, SCHEMA_NAME)

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        assert await object_oids(admin, SCHEMA_NAME) == before
        assert await view_acl(admin, TAGS_VIEW) == [(POLICY_ROLE, 'SELECT')]
        [warning] = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert TAGS_VIEW in warning
        assert 'mcp_outsider (SELECT)' in warning

    @pytest.mark.asyncio
    async def test_a_tags_view_replaced_outside_the_server_is_re_created_alone_and_named(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection, caplog: pytest.LogCaptureFixture,
    ) -> None:
        await owner.execute(
            f'CREATE OR REPLACE VIEW {TAGS_VIEW} AS SELECT t.id, t.context_entry_id, t.tag FROM {SCHEMA}.tags AS t',
        )
        before = await object_oids(admin, SCHEMA_NAME)

        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await apply_rls(booted, rls_settings())

        after = await object_oids(admin, SCHEMA_NAME)
        changed = {key for key in before if after.get(key) != before[key]}
        assert changed == {'view tags_readable', 'trigger tags_readable.reject_writes'}
        assert 'context_entries_readable' in await admin.fetchval('SELECT pg_get_viewdef(to_regclass($1), true)', TAGS_VIEW)
        assert await view_acl(admin, TAGS_VIEW) == [(POLICY_ROLE, 'SELECT')]
        assert any(
            f'VIEW {TAGS_VIEW}' in record.getMessage() and 'outside the server' in record.getMessage()
            for record in caplog.records
        )
        await install_ddl_log(admin)
        await apply_rls(booted, rls_settings())
        assert await ddl_log(admin) == []


class TestFtsLanguageRebuild:
    """The FTS language rebuild re-creates both read views in its own transaction, as the migration would render them.

    Like the migration, it names every grant on a view its drop takes away beyond the
    policy role's SELECT.
    """

    @pytest.mark.asyncio
    async def test_the_view_survives_a_failed_rebuild_and_is_re_created_by_a_successful_one(
        self, booted: StorageBackend, admin: asyncpg.Connection, owner: asyncpg.Connection,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        fts = RepositoryContainer(booted).fts
        view_oids = [await admin.fetchval(VIEW_OID, view) for view in (VIEW, TAGS_VIEW)]

        with pytest.raises(Exception, match='no_such_language'):
            await fts.migrate_language('no_such_language', scope=MAINTENANCE)
        assert [await admin.fetchval(VIEW_OID, view) for view in (VIEW, TAGS_VIEW)] == view_oids

        await owner.execute(f'GRANT SELECT ON {VIEW} TO mcp_outsider')
        await owner.execute(f'GRANT INSERT ON {TAGS_VIEW} TO mcp_outsider')
        with caplog.at_level(logging.WARNING, logger='app.migrations.rls'):
            await fts.migrate_language('german', scope=MAINTENANCE)

        warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
        assert any(TAGS_VIEW in message and 'mcp_outsider (INSERT)' in message for message in warnings)
        assert any(VIEW in message and 'mcp_outsider (SELECT)' in message for message in warnings)
        after_oids = [await admin.fetchval(VIEW_OID, view) for view in (VIEW, TAGS_VIEW)]
        assert all(current != earlier for current, earlier in zip(after_oids, view_oids, strict=True))
        columns = await view_columns(admin, VIEW)
        assert columns == await view_columns(admin, f'{SCHEMA}.context_entries')
        assert columns[-1] == 'text_search_vector'
        assert await fts.get_current_language(scope=MAINTENANCE) == 'german'
        for view in (VIEW, TAGS_VIEW):
            assert await view_acl(admin, view) == [(POLICY_ROLE, 'SELECT')], view
        await install_ddl_log(admin)
        await apply_rls(booted, rls_settings())
        assert await ddl_log(admin) == []


class TestFlagOffGuard:
    """A boot with ENABLE_POSTGRESQL_RLS off over the installed objects is refused before any provisioning."""

    @pytest.mark.asyncio
    async def test_starting_with_the_flag_off_is_refused_naming_every_table(self, booted: StorageBackend) -> None:
        flag_off = get_settings()
        assert not flag_off.access_control.postgresql_rls_enabled

        with pytest.raises(ConfigurationError) as raised:
            await guard_rls_disable_over_enabled(booted, flag_off)

        tables = [table for table in MANAGED_TABLES if table != 'vec_context_embeddings_compressed']
        assert f'(tables: {", ".join(tables)}, context_entries_readable, tags_readable)' in str(raised.value)
        assert 'mcp-context-server-migrate --disable-rls' in str(raised.value)
        with pytest.raises(ConfigurationError, match='ENABLE_POSTGRESQL_RLS=false'):
            await prepare_database(booted, flag_off, request_role=None)
