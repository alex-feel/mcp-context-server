"""Tests for the rendering of the row-level security read view of ``context_entries``.

Covers app.migrations.rls_ddl: the owner-owned view request roles read the context
table through, its policy-role gate and READ conjunct, its column list, the
function and INSTEAD OF trigger that keep it read-only, and the statements that
grant the policy role SELECT on it and revoke what anyone else holds. The READ
conjunct comes from the one predicate composer in app.access_scope, so it is
asserted against the context_entries read policy's own text.
"""

import hashlib
import re

import pytest

from app.access_scope import READ_VIEW
from app.access_scope import AccessMode
from app.access_scope import build_policy_predicate
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import read_view
from app.migrations.rls_ddl import read_view_drop
from app.migrations.rls_ddl import read_view_grant
from app.migrations.rls_ddl import read_view_revoke
from app.migrations.rls_ddl import reject_function
from app.migrations.rls_ddl import reject_trigger
from app.migrations.rls_ddl import table_policies

SCHEMA = '"public"'
ROLE = '"mcp_context_request"'
ROLE_NAME = 'mcp_context_request'
COLUMNS = ('"id"', '"thread_id"', '"text_content"', '"owner_id"', '"visibility"')
GATE = "pg_catalog.pg_has_role(CURRENT_USER, 'mcp_context_request', 'USAGE')"
VIEW = f'{SCHEMA}.context_entries_readable'


def _read_policy_using() -> str:
    """Return the USING expression of the context_entries read policy."""
    read_policy = table_policies('context_entries', schema=SCHEMA, role=ROLE)[0]
    match = re.fullmatch(r'CREATE POLICY context_entries_read ON .+ FOR SELECT TO .+ USING \((.+)\)', read_policy.statement)
    assert match is not None
    return match.group(1)


# ============================================================================
# The view
# ============================================================================


def test_read_view_selects_every_column_through_the_gate_and_the_read_predicate() -> None:
    """The view lists the given columns in order and keeps a row only for policy-role members that may read it."""
    view = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME)
    read = build_policy_predicate(AccessMode.READ, outer='ce', schema=SCHEMA)

    assert view.statement == (
        f'CREATE VIEW {VIEW} AS SELECT ce."id", ce."thread_id", ce."text_content", ce."owner_id", ce."visibility" '
        f'FROM {SCHEMA}.context_entries AS ce WHERE {GATE} AND {read}'
    )
    assert view.kind is RlsObjectKind.VIEW
    assert view.name == 'context_entries_readable'
    assert view.anchor == VIEW


def test_read_view_conjunct_is_the_read_policy_text_with_the_view_alias() -> None:
    """The view and the SELECT policy read through one predicate, differing only in what qualifies the columns."""
    view = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME)
    conjunct = view.statement.partition(f' WHERE {GATE} AND ')[2]

    assert conjunct == _read_policy_using().replace('context_entries.', 'ce.')


def test_read_view_carries_no_option_that_would_cost_the_indexes() -> None:
    """Neither security_barrier nor security_invoker is set: the view is pulled up and reads as its owner."""
    statement = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME).statement

    assert 'security_barrier' not in statement
    assert 'security_invoker' not in statement
    assert not statement.startswith(f'CREATE VIEW {VIEW} WITH')


def test_read_view_gate_names_the_policy_role_as_a_literal() -> None:
    """The gate compares role membership by name, so the role is a string literal with quotes doubled."""
    statement = read_view(COLUMNS, schema=SCHEMA, role_name="o'brien").statement

    assert "pg_catalog.pg_has_role(CURRENT_USER, 'o''brien', 'USAGE')" in statement


def test_read_view_sha_follows_the_columns_the_schema_and_the_role() -> None:
    """A column added to context_entries, another schema or another policy role changes the view's sha."""
    base = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME)

    assert base.sha == hashlib.sha256(base.statement.encode('utf-8')).hexdigest()
    assert base.sha == read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME).sha
    assert base.sha != read_view((*COLUMNS, '"text_search_vector"'), schema=SCHEMA, role_name=ROLE_NAME).sha
    assert base.sha != read_view(COLUMNS, schema='"tenant one"', role_name=ROLE_NAME).sha
    assert base.sha != read_view(COLUMNS, schema=SCHEMA, role_name='other_request').sha


def test_read_view_needs_a_column() -> None:
    """A view over no column means context_entries was not found."""
    with pytest.raises(ValueError, match='column'):
        read_view((), schema=SCHEMA, role_name=ROLE_NAME)


def test_read_view_is_the_relation_the_request_profile_reads_context_entries_through() -> None:
    """The migration creates the view the request backend's plain reads name."""
    assert READ_VIEW == 'context_entries_readable'
    assert RLS_READ_VIEWS == {'context_entries': READ_VIEW}


def test_read_view_comment_and_drop_name_the_view() -> None:
    """COMMENT ON VIEW attaches the fingerprint, and DROP VIEW IF EXISTS removes the view with its trigger."""
    view = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME)
    definition = hashlib.sha256(b'deparse').hexdigest()

    assert view.target == f'VIEW {VIEW}'
    assert comment_statement(view, definition=definition) == (
        f"COMMENT ON VIEW {VIEW} IS 'mcp-context-server rls v1 sha={view.sha} def={definition}'"
    )
    assert view.drop_statement == f'DROP VIEW IF EXISTS {VIEW}'
    assert read_view_drop(SCHEMA) == view.drop_statement


# ============================================================================
# Write rejection
# ============================================================================


def test_reject_function_refuses_every_write_as_insufficient_privilege() -> None:
    """A write through the view raises 42501 instead of running as the view's owner."""
    function = reject_function(SCHEMA)

    assert function.kind is RlsObjectKind.FUNCTION
    assert function.target == f'FUNCTION {SCHEMA}.mcp_context_reject_view_write()'
    assert function.statement.startswith(
        f'CREATE OR REPLACE FUNCTION {SCHEMA}.mcp_context_reject_view_write() RETURNS trigger\n'
        '    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$\n',
    )
    assert "RAISE EXCEPTION 'context_entries_readable is read-only' USING ERRCODE = '42501';" in function.statement
    assert 'RETURN ' not in function.statement


def test_reject_trigger_fires_instead_of_every_write_on_the_view() -> None:
    """INSERT, UPDATE and DELETE on the view all run the rejecting function instead."""
    trigger = reject_trigger(SCHEMA)

    assert trigger.kind is RlsObjectKind.TRIGGER
    assert trigger.target == f'TRIGGER reject_writes ON {VIEW}'
    assert trigger.statement == (
        f'CREATE TRIGGER reject_writes INSTEAD OF INSERT OR UPDATE OR DELETE ON {VIEW} '
        f'FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.mcp_context_reject_view_write()'
    )


def test_reject_objects_follow_the_schema() -> None:
    """The function and the trigger are re-created in another schema with another sha."""
    assert reject_function(SCHEMA).sha != reject_function('"tenant one"').sha
    assert reject_trigger(SCHEMA).sha != reject_trigger('"tenant one"').sha


# ============================================================================
# Privileges
# ============================================================================


def test_read_view_grant_gives_the_policy_role_select_only() -> None:
    """The policy role reads through the view and holds no write privilege on it."""
    assert read_view_grant(SCHEMA, ROLE) == f'GRANT SELECT ON {VIEW} TO {ROLE}'


@pytest.mark.parametrize(
    ('grantee', 'privileges', 'statement'),
    [
        pytest.param('PUBLIC', ('ALL',), f'REVOKE ALL ON {VIEW} FROM PUBLIC', id='public'),
        pytest.param('"mcp_outsider"', ('ALL',), f'REVOKE ALL ON {VIEW} FROM "mcp_outsider"', id='other-role'),
        pytest.param(ROLE, ('INSERT', 'UPDATE'), f'REVOKE INSERT, UPDATE ON {VIEW} FROM {ROLE}', id='policy-role-extra'),
    ],
)
def test_read_view_revoke_names_the_grantee_and_the_privileges(
    grantee: str, privileges: tuple[str, ...], statement: str,
) -> None:
    """A revocation removes everything from another grantee, or the policy role's privileges beyond SELECT."""
    assert read_view_revoke(SCHEMA, grantee, privileges) == statement


@pytest.mark.parametrize('privileges', [(), ('SELECT; DROP TABLE context_entries',), ('select',)])
def test_read_view_revoke_accepts_only_table_privilege_keywords(privileges: tuple[str, ...]) -> None:
    """Privileges are interpolated, so only the keywords of a table's ACL are accepted."""
    with pytest.raises(ValueError, match='privilege'):
        read_view_revoke(SCHEMA, ROLE, privileges)
