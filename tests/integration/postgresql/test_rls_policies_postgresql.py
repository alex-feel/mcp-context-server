"""The row-level security policies against a real PostgreSQL server.

Every check connects as the request login role ``mcp_app`` over a raw asyncpg connection,
binds an identity with the request profile's bind statement inside a transaction that is
rolled back, and issues SQL without any application predicate, so the rows it reaches are
the policies' answer alone. The matrix of :mod:`tests.integration.postgresql._rls_matrix`
covers the access-scope principals, a principal with only an unrelated group, no bind at
all, the empty principal and group ids holding a comma, a quote, braces and a backslash.
Every refusal is asserted as SQLSTATE 42501, so a recursive policy (42P17) fails the test.
"""

import uuid
from collections.abc import AsyncGenerator

import asyncpg
import pytest
import pytest_asyncio

from app.access_scope import AccessMode
from app.access_scope import AccessScope
from app.migrations.rls import read_definition
from app.migrations.rls_ddl import CHILD_PARENT_KEYS
from app.migrations.rls_ddl import MANAGED_TABLES
from app.migrations.rls_ddl import RlsObject
from app.migrations.rls_ddl import digest
from app.migrations.rls_ddl import fingerprint
from app.migrations.rls_ddl import guard_function
from app.migrations.rls_ddl import guard_trigger
from app.migrations.rls_ddl import helper_functions
from app.migrations.rls_ddl import records_preenabled
from app.migrations.rls_ddl import table_policies
from tests.integration.postgresql._rls_matrix import MATRIX_SCOPES
from tests.integration.postgresql._rls_matrix import MATRIX_VISIBILITY
from tests.integration.postgresql._rls_matrix import ODD_GROUPS
from tests.integration.postgresql._rls_matrix import POLICY_ROLE
from tests.integration.postgresql._rls_matrix import bind
from tests.integration.postgresql._rls_matrix import entry_uuid
from tests.integration.postgresql._rls_matrix import labels
from tests.integration.postgresql._rls_matrix import policy_visibility
from tests.integration.postgresql._rls_matrix import rls_matrix_db
from tests.integration.postgresql._rls_matrix import scoped_transaction
from tests.integration.postgresql.conftest import RlsUrls
from tests.repositories._access_scope_cases import SCOPES
from tests.repositories._access_scope_cases import ScopedDb
from tests.repositories._access_scope_cases import in_seed_order
from tests.repositories._access_scope_cases import observe_predicate_visibility

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

ALICE = AccessScope('alice', frozenset())
ALICE_PUBLISHER = AccessScope('alice', frozenset(), may_publish=True)
BOB = AccessScope('bob', frozenset())
CAROL = AccessScope('carol', frozenset({'team-x'}))
DAVE = AccessScope('dave', frozenset())
REFUSED = asyncpg.exceptions.InsufficientPrivilegeError

INSERT_ENTRY = (
    'INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id, visibility) '
    "VALUES ($1, 'rls-policies', 'agent', 'text', 'policy probe', $2, $3) RETURNING id"
)
INSERT_GRANT = (
    'INSERT INTO context_entry_grants (context_entry_id, principal_type, principal_id, permission, granted_by) '
    'VALUES ($1, $2, $3, $4, $5)'
)
# The grant repository's idempotent form; the conflict target and RETURNING also apply the read policy to the new row.
UPSERT_GRANT = (
    f'{INSERT_GRANT} ON CONFLICT (context_entry_id, principal_type, principal_id, permission) DO NOTHING RETURNING id'
)

# The comment on one installed object, by its anchor ($1) and name ($2).
_COMMENT_SQL = {
    'FUNCTION': (
        "SELECT obj_description(p.oid, 'pg_proc') FROM pg_proc p WHERE p.oid = to_regprocedure($1) AND p.proname = $2"
    ),
    'TRIGGER': (
        "SELECT obj_description(t.oid, 'pg_trigger') FROM pg_trigger t WHERE t.tgrelid = to_regclass($1) AND t.tgname = $2"
    ),
    'POLICY': (
        "SELECT obj_description(p.oid, 'pg_policy') FROM pg_policy p WHERE p.polrelid = to_regclass($1) AND p.polname = $2"
    ),
}
# One valid row per child table, keyed to the parent entry bound as $1.
CHILD_INSERTS = {
    'tags': "INSERT INTO tags (context_entry_id, tag) VALUES ($1, 'policy-probe')",
    'image_attachments': (
        "INSERT INTO image_attachments (context_entry_id, image_data, mime_type) VALUES ($1, '\\x00'::bytea, 'image/png')"
    ),
    'embedding_metadata': (
        "INSERT INTO embedding_metadata (context_id, model_name, dimensions) VALUES ($1, 'policy-probe', 128)"
    ),
    'vec_context_embeddings': 'INSERT INTO vec_context_embeddings (context_id, embedding) VALUES ($1, NULL)',
    'vec_context_embeddings_compressed': (
        "INSERT INTO vec_context_embeddings_compressed (context_id, chunk_index, payload) VALUES ($1, 9, '\\x00'::bytea)"
    ),
    'context_index_nodes': (
        'INSERT INTO context_index_nodes (context_id, node_id, level, ordinal, title, char_start, char_end) '
        "VALUES ($1, 'policy-probe', 1, 9, 'probe', 0, 0)"
    ),
}


@pytest_asyncio.fixture(params=['fp32'])
async def matrix_db(
    request: pytest.FixtureRequest, pg_rls_db: RlsUrls, monkeypatch: pytest.MonkeyPatch,
) -> AsyncGenerator[ScopedDb]:
    """The policy matrix database in the requested layout (fp32 unless a test parametrizes it).

    Yields:
        The seeded database with the policies installed.
    """
    async with rls_matrix_db(pg_rls_db, layout=request.param, monkeypatch=monkeypatch) as db:
        yield db


@pytest_asyncio.fixture
async def app_conn(matrix_db: ScopedDb, pg_rls_db: RlsUrls) -> AsyncGenerator[asyncpg.Connection]:
    """A raw connection of the request login role ``mcp_app`` to the matrix database.

    Yields:
        The open connection, closed afterwards.
    """
    _ = matrix_db
    conn = await asyncpg.connect(pg_rls_db.app)
    try:
        yield conn
    finally:
        await conn.close()


async def _child_tables(db: ScopedDb) -> list[tuple[str, str]]:
    """Return the child tables of the layout with their parent key, in managed order."""
    rows = await db.fetch_all(
        'SELECT tablename FROM pg_tables WHERE schemaname = current_schema() AND tablename = ANY(?::text[])',
        (list(CHILD_PARENT_KEYS),),
    )
    present = {str(row[0]) for row in rows}
    return [(table, key) for table, key in CHILD_PARENT_KEYS.items() if table in present]


async def _child_counts(db: ScopedDb, table: str, key: str) -> dict[str, int]:
    """Return the number of ``table`` rows per parent label, read as the owner."""
    rows = await db.fetch_all(f'SELECT {key}, COUNT(*) FROM {table} GROUP BY {key}')
    return {db.labels_of([str(row[0])])[0]: int(str(row[1])) for row in rows}


class TestContextEntries:
    """The four context_entries policies give each command exactly the rows of its access mode."""

    @pytest.mark.asyncio
    async def test_each_command_reaches_exactly_the_rows_of_its_mode(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """SELECT reaches the readable rows, UPDATE the writable rows and DELETE the owned rows."""
        assert await policy_visibility(app_conn, matrix_db) == MATRIX_VISIBILITY

    @pytest.mark.asyncio
    async def test_policies_agree_with_the_application_predicate(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """For every principal a request can resolve to, the database and the application predicate admit the same rows.

        The empty principal is left out: no request resolves to it, and the application
        predicate would compare it literally with stored owners, while the database turns
        it into no principal and admits only the public rows.
        """
        observed = await policy_visibility(app_conn, matrix_db)
        oracle = await observe_predicate_visibility(matrix_db)

        for mode in AccessMode:
            for name, scope in MATRIX_SCOPES.items():
                if scope is None or not scope.principal_id:
                    continue
                assert observed[mode][name] == await matrix_db.labels_selected_by(scope, mode), (mode, name)
                if name in SCOPES:
                    assert observed[mode][name] == oracle[(mode, name)], (mode, name)

    @pytest.mark.asyncio
    async def test_an_own_row_inserts_and_any_other_owner_is_refused(self, app_conn: asyncpg.Connection) -> None:
        """The INSERT check stamps the bound principal; an unbound or empty principal owns nothing."""
        async with scoped_transaction(app_conn, BOB):
            assert await app_conn.fetchval(INSERT_ENTRY, uuid.uuid4(), 'bob', 'private') is not None

        for scope, owner in [(BOB, 'alice'), (None, ''), (None, 'bob'), (MATRIX_SCOPES['blank'], '')]:
            async with scoped_transaction(app_conn, scope):
                with pytest.raises(REFUSED, match='row-level security'):
                    await app_conn.fetchval(INSERT_ENTRY, uuid.uuid4(), owner, 'private')

    @pytest.mark.asyncio
    async def test_access_column_changes_are_refused_while_content_edits_pass(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """A write grantee edits text but never visibility, and nobody changes an owner."""
        granted = entry_uuid(matrix_db, 'alice_private_bob_write')
        owned = entry_uuid(matrix_db, 'alice_private')

        async with scoped_transaction(app_conn, BOB):
            assert await app_conn.execute(
                "UPDATE context_entries SET text_content = 'edited by bob' WHERE id = $1", granted,
            ) == 'UPDATE 1'
        refusals = [
            (BOB, granted, "visibility = 'public'", 'only the owner may change the visibility'),
            (BOB, granted, "owner_id = 'bob'", 'is immutable'),
            (ALICE_PUBLISHER, owned, "owner_id = 'bob'", 'is immutable'),
        ]
        for scope, entry, assignment, message in refusals:
            async with scoped_transaction(app_conn, scope):
                with pytest.raises(REFUSED, match=message):
                    await app_conn.execute(f'UPDATE context_entries SET {assignment} WHERE id = $1', entry)
        async with scoped_transaction(app_conn, ALICE_PUBLISHER):
            # Assigning the stored owner changes nothing, so the guard lets the row through.
            assert await app_conn.execute("UPDATE context_entries SET owner_id = 'alice' WHERE id = $1", owned) == 'UPDATE 1'

    @pytest.mark.asyncio
    async def test_the_table_owner_changes_access_columns_without_a_bound_principal(self, matrix_db: ScopedDb) -> None:
        """Maintenance transactions, such as the owner reassignment, bind no principal and pass the guard."""
        entry = matrix_db.ids['alice_private']

        await matrix_db.execute("UPDATE context_entries SET owner_id = 'bob', visibility = 'public' WHERE id = ?", (entry,))

        assert await matrix_db.fetch_all('SELECT owner_id, visibility FROM context_entries WHERE id = ?', (entry,)) == [
            ('bob', 'public'),
        ]

    @pytest.mark.asyncio
    async def test_publishing_needs_the_publish_right(self, app_conn: asyncpg.Connection, matrix_db: ScopedDb) -> None:
        """INSERT and UPDATE to 'public' pass only with may_publish on; the owner may always unpublish."""
        owned_private = entry_uuid(matrix_db, 'alice_private')
        owned_public = entry_uuid(matrix_db, 'alice_public')

        async with scoped_transaction(app_conn, ALICE):
            with pytest.raises(REFUSED, match='row-level security'):
                await app_conn.fetchval(INSERT_ENTRY, uuid.uuid4(), 'alice', 'public')
        async with scoped_transaction(app_conn, ALICE):
            with pytest.raises(REFUSED, match='requires the publish role'):
                await app_conn.execute("UPDATE context_entries SET visibility = 'public' WHERE id = $1", owned_private)
        async with scoped_transaction(app_conn, ALICE):
            assert await app_conn.execute(
                "UPDATE context_entries SET visibility = 'private' WHERE id = $1", owned_public,
            ) == 'UPDATE 1'
        async with scoped_transaction(app_conn, ALICE_PUBLISHER):
            assert await app_conn.fetchval(INSERT_ENTRY, uuid.uuid4(), 'alice', 'public') is not None
            assert await app_conn.execute(
                "UPDATE context_entries SET visibility = 'public' WHERE id = $1", owned_private,
            ) == 'UPDATE 1'

    @pytest.mark.asyncio
    async def test_locking_reads_return_only_writable_rows(self, app_conn: asyncpg.Connection, matrix_db: ScopedDb) -> None:
        """FOR KEY SHARE and FOR UPDATE apply the update policy: a read-only grantee locks nothing and gets no error."""
        cases = [
            (BOB, 'alice_private_bob_read', ()),
            (BOB, 'alice_private_bob_write', ('alice_private_bob_write',)),
            (CAROL, 'alice_private_team_read', ()),
            (DAVE, 'alice_public_dave_read', ()),
            (CAROL, 'alice_public_carol_write', ('alice_public_carol_write',)),
        ]
        for scope, label, expected in cases:
            for clause in ('FOR KEY SHARE', 'FOR UPDATE'):
                async with scoped_transaction(app_conn, scope):
                    rows = await app_conn.fetch(
                        f'SELECT id FROM context_entries WHERE id = $1 {clause}', entry_uuid(matrix_db, label),
                    )
                    assert labels(matrix_db, rows) == expected, (scope.principal_id, label, clause)


class TestChildTables:
    """Child rows follow their parent: reads need READ, inserts and deletes need WRITE, nothing updates them."""

    @pytest.mark.parametrize('matrix_db', ['fp32', 'compressed'], indirect=True)
    @pytest.mark.asyncio
    async def test_child_rows_are_read_exactly_where_the_parent_is_readable(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """Every child table shows each scope the children of the entries it may read, and no others."""
        tables = await _child_tables(matrix_db)
        assert {table for table, _ in tables} >= {'tags', 'image_attachments', 'embedding_metadata', 'context_index_nodes'}
        for table, key in tables:
            parents = set(await _child_counts(matrix_db, table, key))
            assert parents, table
            for name, scope in MATRIX_SCOPES.items():
                async with scoped_transaction(app_conn, scope):
                    rows = await app_conn.fetch(f'SELECT DISTINCT {key} FROM {table}')
                expected = in_seed_order(parents & set(MATRIX_VISIBILITY[AccessMode.READ][name]))
                assert labels(matrix_db, rows) == expected, (table, name)

    @pytest.mark.parametrize('matrix_db', ['fp32', 'compressed'], indirect=True)
    @pytest.mark.asyncio
    async def test_child_inserts_and_deletes_need_write_on_the_parent(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """A write grantee replaces children; a read-only, an invisible and a missing parent refuse alike."""
        writable = entry_uuid(matrix_db, 'alice_private_bob_write')
        read_only = entry_uuid(matrix_db, 'alice_private_bob_read')
        invisible = entry_uuid(matrix_db, 'alice_private')
        for table, key in await _child_tables(matrix_db):
            counts = await _child_counts(matrix_db, table, key)
            insert = CHILD_INSERTS[table]
            delete = f'DELETE FROM {table} WHERE {key} = $1 RETURNING {key}'

            async with scoped_transaction(app_conn, BOB):
                assert len(await app_conn.fetch(delete, writable)) == counts.get('alice_private_bob_write', 0), table
                await app_conn.execute(insert, writable)
            async with scoped_transaction(app_conn, BOB):
                assert await app_conn.fetch(delete, read_only) == [], table

            messages = []
            for parent in (read_only, invisible, uuid.uuid4()):
                async with scoped_transaction(app_conn, BOB):
                    with pytest.raises(REFUSED) as refusal:
                        await app_conn.execute(insert, parent)
                messages.append(str(refusal.value))
            assert messages == [f'new row violates row-level security policy for table "{table}"'] * 3

    @pytest.mark.asyncio
    async def test_a_repeated_tag_on_a_read_only_parent_is_refused(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """ON CONFLICT DO NOTHING still checks the insert policy, so the repeat raises instead of passing silently."""
        async with scoped_transaction(app_conn, BOB):
            with pytest.raises(REFUSED, match='"tags"'):
                await app_conn.execute(
                    "INSERT INTO tags (context_entry_id, tag) VALUES ($1, 'seed') "
                    'ON CONFLICT (context_entry_id, tag) DO NOTHING',
                    entry_uuid(matrix_db, 'alice_private_bob_read'),
                )

    @pytest.mark.parametrize('matrix_db', ['fp32', 'compressed'], indirect=True)
    @pytest.mark.asyncio
    async def test_no_child_row_is_ever_updated(self, app_conn: asyncpg.Connection, matrix_db: ScopedDb) -> None:
        """No child table has an update policy, so even the owner's UPDATE reaches zero rows."""
        owned = entry_uuid(matrix_db, 'alice_private')
        for table, key in await _child_tables(matrix_db):
            assert (await _child_counts(matrix_db, table, key)).get('alice_private', 0) > 0, table
            async with scoped_transaction(app_conn, ALICE):
                rows = await app_conn.fetch(f'UPDATE {table} SET {key} = {key} WHERE {key} = $1 RETURNING {key}', owned)
            assert rows == [], table

    @pytest.mark.parametrize('matrix_db', ['fp32', 'compressed'], indirect=True)
    @pytest.mark.asyncio
    async def test_deleting_an_entry_cascades_to_every_child(self, app_conn: asyncpg.Connection, matrix_db: ScopedDb) -> None:
        """The cascade is a referential action, which no child policy filters."""
        tables = await _child_tables(matrix_db)
        before = {table: (await _child_counts(matrix_db, table, key)).get('alice_private', 0) for table, key in tables}
        assert all(count > 0 for count in before.values()), before

        async with app_conn.transaction():
            await bind(app_conn, ALICE)
            assert await app_conn.execute(
                'DELETE FROM context_entries WHERE id = $1', entry_uuid(matrix_db, 'alice_private'),
            ) == 'DELETE 1'

        after = {table: (await _child_counts(matrix_db, table, key)).get('alice_private', 0) for table, key in tables}
        assert after == dict.fromkeys(before, 0)


class TestGrants:
    """Grant rows are visible to their grantees and written only by the entry's owner as the grantor."""

    @pytest.mark.asyncio
    async def test_only_the_owner_grants_and_only_as_the_grantor(
        self, app_conn: asyncpg.Connection, matrix_db: ScopedDb,
    ) -> None:
        """The owner inserts a grant under its own name; a write grantee, a forged grantor and no bind are refused.

        A grant to another principal is a row its grantor cannot read, so the idempotent form,
        whose conflict target and RETURNING apply the read policy to the new row, is refused
        for it; the only grants the server writes name the author's own groups.
        """
        owned = entry_uuid(matrix_db, 'alice_private')
        granted = entry_uuid(matrix_db, 'alice_private_bob_write')

        async with scoped_transaction(app_conn, ALICE):
            assert await app_conn.execute(INSERT_GRANT, owned, 'user', 'dave', 'read', 'alice') == 'INSERT 0 1'
        for scope, entry, grantor in [(BOB, granted, 'bob'), (ALICE, owned, 'bob'), (None, owned, 'alice')]:
            async with scoped_transaction(app_conn, scope):
                with pytest.raises(REFUSED, match='context_entry_grants'):
                    await app_conn.execute(INSERT_GRANT, entry, 'user', 'dave', 'read', grantor)
        async with scoped_transaction(app_conn, ALICE):
            with pytest.raises(REFUSED, match='context_entry_grants'):
                await app_conn.fetchval(UPSERT_GRANT, owned, 'user', 'dave', 'read', 'alice')

    @pytest.mark.asyncio
    async def test_grant_rows_are_visible_to_their_grantees_only(self, app_conn: asyncpg.Connection) -> None:
        """A grant names the caller or one of its groups exactly; the empty principal and group match nothing."""
        expected = {
            'alice': [],
            'bob': [('user', 'bob'), ('user', 'bob')],
            'carol': [('group', 'team-x'), ('user', 'carol')],
            'dave': [('user', 'dave')],
            'mallory': [],
            'unset': [],
            'blank': [],
            'oscar': sorted(('group', group) for group in ODD_GROUPS),
        }
        observed = {}
        for name, scope in MATRIX_SCOPES.items():
            async with scoped_transaction(app_conn, scope):
                rows = await app_conn.fetch('SELECT principal_type, principal_id FROM context_entry_grants')
            observed[name] = sorted((str(row[0]), str(row[1])) for row in rows)
        assert observed == expected

    @pytest.mark.asyncio
    async def test_author_group_grants_are_stored_with_the_entry_in_one_transaction(
        self, app_conn: asyncpg.Connection,
    ) -> None:
        """The author_groups store shape: the entry, then a read grant per author group, both under the policies."""
        entry = uuid.uuid4()
        async with scoped_transaction(app_conn, CAROL):
            assert await app_conn.fetchval(INSERT_ENTRY, entry, 'carol', 'private') == entry
            assert await app_conn.fetchval(UPSERT_GRANT, entry, 'group', 'team-x', 'read', 'carol') is not None
            assert await app_conn.fetchval(UPSERT_GRANT, entry, 'group', 'team-x', 'read', 'carol') is None
            rows = await app_conn.fetch(
                'SELECT principal_type, principal_id, permission FROM context_entry_grants WHERE context_entry_id = $1',
                entry,
            )
            assert [tuple(row) for row in rows] == [('group', 'team-x', 'read')]


class TestInstalledObjects:
    """What the migration leaves in the catalog: fingerprints, enablement without FORCE and the role the policies name."""

    @pytest.mark.parametrize('matrix_db', ['fp32', 'compressed'], indirect=True)
    @pytest.mark.asyncio
    async def test_every_object_carries_the_fingerprint_of_its_statement_and_its_deparse(
        self, matrix_db: ScopedDb, pg_rls_db: RlsUrls,
    ) -> None:
        """Each comment holds the sha of the rendered statement and the digest of the server's own definition."""
        schema, role = '"public"', f'"{POLICY_ROLE}"'
        objects: list[RlsObject] = [*helper_functions(schema), guard_function(schema), guard_trigger(schema)]
        for table, _ in [('context_entries', ''), ('context_entry_grants', ''), *await _child_tables(matrix_db)]:
            objects.extend(table_policies(table, schema=schema, role=role))
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            for obj in objects:
                comment = await conn.fetchval(_COMMENT_SQL[obj.kind], obj.anchor, obj.name)
                # The migration enabled every table, so each read policy records it was off before.
                preenabled = False if records_preenabled(obj) else None
                definition = digest(await read_definition(conn, obj))
                expected = fingerprint(sha=obj.sha, definition=definition, preenabled=preenabled)
                assert comment == expected, obj.target
        finally:
            await conn.close()

    @pytest.mark.asyncio
    async def test_tables_are_enabled_without_force_and_policies_name_only_the_policy_role(
        self, matrix_db: ScopedDb, pg_rls_db: RlsUrls,
    ) -> None:
        """The owner keeps its bypass, and no policy applies to PUBLIC or to any other role."""
        _ = matrix_db
        conn = await asyncpg.connect(pg_rls_db.admin)
        try:
            tables = await conn.fetch(
                'SELECT c.relname, c.relrowsecurity, c.relforcerowsecurity FROM pg_class c '
                "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = 'public' AND c.relname = ANY($1::text[])",
                list(MANAGED_TABLES),
            )
            roles = await conn.fetch(
                'SELECT DISTINCT pg_get_userbyid(r) FROM pg_policy p CROSS JOIN LATERAL unnest(p.polroles) AS r',
            )
            helper_acl = await conn.fetchval(
                "SELECT proacl::text FROM pg_proc WHERE oid = to_regprocedure('public.mcp_context_groups()')",
            )
        finally:
            await conn.close()

        assert {str(row[0]) for row in tables} >= {'context_entries', 'context_entry_grants', 'tags', 'image_attachments'}
        assert all(row[1] and not row[2] for row in tables), [tuple(row) for row in tables]
        assert [str(row[0]) for row in roles] == [POLICY_ROLE]
        assert f'{POLICY_ROLE}=X/' in helper_acl
