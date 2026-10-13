"""Tests for the rendering of the row-level security read views.

Covers app.migrations.rls_view_ddl: the owner-owned views request roles read through,
``context_entries_readable`` with its READ condition and the policy-role gate folded into
the identity operands and the public arm, and ``tags_readable``, which lists the tags of
the entries that view returns; their column lists; the order the views are created in and
which of them the existing tables call for; the function and INSTEAD OF triggers that keep
them read-only; and the statements that grant the policy role SELECT on each and revoke
what anyone else holds. The condition comes from the one predicate composer in
app.access_scope, so it is asserted against the context_entries read policy's own text.
"""

import hashlib
import re

import pytest

from app.access_scope import READ_VIEW
from app.access_scope import TAGS_READ_VIEW
from app.access_scope import build_gated_read_predicate
from app.backends.postgresql_backend.session_profiles import RLS_READ_VIEWS
from app.migrations.rls_ddl import RlsObjectKind
from app.migrations.rls_ddl import comment_statement
from app.migrations.rls_ddl import table_policies
from app.migrations.rls_view_ddl import READ_VIEW_TABLES
from app.migrations.rls_view_ddl import read_view
from app.migrations.rls_view_ddl import read_view_drop
from app.migrations.rls_view_ddl import read_view_grant
from app.migrations.rls_view_ddl import read_view_revoke
from app.migrations.rls_view_ddl import read_views
from app.migrations.rls_view_ddl import reject_function
from app.migrations.rls_view_ddl import reject_trigger
from app.migrations.rls_view_ddl import tags_read_view
from app.migrations.rls_view_ddl import write_rejection

SCHEMA = '"public"'
ROLE = '"mcp_context_request"'
ROLE_NAME = 'mcp_context_request'
COLUMNS = ('"id"', '"thread_id"', '"text_content"', '"owner_id"', '"visibility"')
TAG_COLUMNS = ('"id"', '"context_entry_id"', '"tag"')
GATE = "pg_catalog.pg_has_role(CURRENT_USER, 'mcp_context_request', 'USAGE')"
VIEW = f'{SCHEMA}.context_entries_readable'
TAGS_VIEW = f'{SCHEMA}.tags_readable'
POLICY_PRINCIPAL = f'(SELECT {SCHEMA}.mcp_context_principal_id())'
POLICY_GROUPS = f'(SELECT {SCHEMA}.mcp_context_groups())::text[]'
GATED_PRINCIPAL = f'(SELECT CASE WHEN {GATE} THEN {SCHEMA}.mcp_context_principal_id() END)'
GATED_GROUPS = f"(SELECT CASE WHEN {GATE} THEN {SCHEMA}.mcp_context_groups() ELSE '{{}}'::text[] END)::text[]"


def _read_policy_using() -> str:
    """Return the USING expression of the context_entries read policy."""
    read_policy = table_policies('context_entries', schema=SCHEMA, role=ROLE)[0]
    match = re.fullmatch(r'CREATE POLICY context_entries_read ON .+ FOR SELECT TO .+ USING \((.+)\)', read_policy.statement)
    assert match is not None
    return match.group(1)


# ============================================================================
# The read view of context_entries
# ============================================================================


def test_read_view_selects_every_column_through_the_gated_read_predicate() -> None:
    """The view lists the given columns in order and keeps a row only for policy-role members that may read it."""
    view = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME)
    read = build_gated_read_predicate(outer='ce', schema=SCHEMA, gate=GATE)

    assert view.statement == (
        f'CREATE VIEW {VIEW} AS SELECT ce."id", ce."thread_id", ce."text_content", ce."owner_id", ce."visibility" '
        f'FROM {SCHEMA}.context_entries AS ce WHERE {read}'
    )
    assert view.kind is RlsObjectKind.VIEW
    assert view.name == 'context_entries_readable'
    assert view.anchor == VIEW


def test_read_view_condition_is_the_read_policy_text_with_the_gate_in_its_operands() -> None:
    """The view and the SELECT policy read through one predicate; the view gates its identity operands and public arm."""
    view = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME)
    condition = view.statement.partition(' FROM "public".context_entries AS ce WHERE ')[2]

    assert condition == (
        _read_policy_using()
        .replace('context_entries.', 'ce.')
        .replace(POLICY_PRINCIPAL, GATED_PRINCIPAL)
        .replace(POLICY_GROUPS, GATED_GROUPS)
        .replace("ce.visibility = 'public'", f"(ce.visibility = 'public' AND (SELECT {GATE}))")
    )


def test_read_view_has_no_conjunct_of_the_gate_alone() -> None:
    """The gate is never a condition of its own, which the planner would hoist over every join with the view."""
    statement = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME).statement

    assert f'WHERE {GATE}' not in statement
    assert f'{GATE} AND' not in statement
    assert statement.count(GATE) == statement.count(f'(SELECT CASE WHEN {GATE} THEN ') + statement.count(f'(SELECT {GATE})')
    assert f'g.principal_id = ANY({GATED_GROUPS})' in statement


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


def test_read_view_comment_and_drop_name_the_view() -> None:
    """COMMENT ON VIEW attaches the fingerprint, and DROP VIEW IF EXISTS removes the view with its trigger."""
    view = read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME)
    definition = hashlib.sha256(b'deparse').hexdigest()

    assert view.target == f'VIEW {VIEW}'
    assert comment_statement(view, definition=definition, preenabled=None) == (
        f"COMMENT ON VIEW {VIEW} IS 'mcp-context-server rls v1 sha={view.sha} def={definition}'"
    )
    assert view.drop_statement == f'DROP VIEW IF EXISTS {VIEW}'
    assert read_view_drop(SCHEMA, view=READ_VIEW) == view.drop_statement


# ============================================================================
# The read view of tags
# ============================================================================


def test_tags_read_view_lists_the_tags_of_the_readable_entries() -> None:
    """The tags view joins every tag to the read view of context_entries, so it lists the tags of readable entries alone."""
    view = tags_read_view(TAG_COLUMNS, schema=SCHEMA)

    assert view.statement == (
        f'CREATE VIEW {TAGS_VIEW} AS SELECT t."id", t."context_entry_id", t."tag" FROM {SCHEMA}.tags AS t '
        f'JOIN {VIEW} AS ce ON ce.id = t.context_entry_id'
    )
    assert view.kind is RlsObjectKind.VIEW
    assert view.name == 'tags_readable'
    assert view.anchor == TAGS_VIEW
    assert view.target == f'VIEW {TAGS_VIEW}'
    assert view.drop_statement == f'DROP VIEW IF EXISTS {TAGS_VIEW}'
    assert read_view_drop(SCHEMA, view=TAGS_READ_VIEW) == view.drop_statement


def test_tags_read_view_carries_no_option_and_no_gate_of_its_own() -> None:
    """The tags view reads as its owner and leaves the identity and the gate to the read view it joins."""
    statement = tags_read_view(TAG_COLUMNS, schema=SCHEMA).statement

    assert 'security_barrier' not in statement
    assert 'security_invoker' not in statement
    assert not statement.startswith(f'CREATE VIEW {TAGS_VIEW} WITH')
    assert 'pg_has_role' not in statement
    assert 'WHERE' not in statement


def test_tags_read_view_sha_follows_the_columns_and_the_schema() -> None:
    """A column added to tags or another schema changes the tags view's sha."""
    base = tags_read_view(TAG_COLUMNS, schema=SCHEMA)

    assert base.sha == tags_read_view(TAG_COLUMNS, schema=SCHEMA).sha
    assert base.sha != tags_read_view((*TAG_COLUMNS, '"extra"'), schema=SCHEMA).sha
    assert base.sha != tags_read_view(TAG_COLUMNS, schema='"tenant one"').sha


def test_tags_read_view_needs_a_column() -> None:
    """A view over no column means tags was not found."""
    with pytest.raises(ValueError, match='column'):
        tags_read_view((), schema=SCHEMA)


# ============================================================================
# The views the tables call for
# ============================================================================


def test_the_views_are_created_in_dependency_order_and_read_by_the_request_profile() -> None:
    """The read view of context_entries comes first, tags_readable reads through it, and the request profile maps both."""
    assert READ_VIEW == 'context_entries_readable'
    assert TAGS_READ_VIEW == 'tags_readable'
    assert list(READ_VIEW_TABLES.items()) == [(READ_VIEW, 'context_entries'), (TAGS_READ_VIEW, 'tags')]
    assert {table: view for view, table in READ_VIEW_TABLES.items()} == RLS_READ_VIEWS


def test_read_views_renders_both_views_when_both_tables_exist() -> None:
    """Both tables exist: the read view of context_entries, then tags_readable."""
    views = read_views({'context_entries': COLUMNS, 'tags': TAG_COLUMNS}, schema=SCHEMA, role_name=ROLE_NAME)

    assert views == (read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME), tags_read_view(TAG_COLUMNS, schema=SCHEMA))


def test_read_views_needs_context_entries_for_every_view() -> None:
    """Without tags only the read view of context_entries exists; without context_entries no view does."""
    assert read_views({'context_entries': COLUMNS}, schema=SCHEMA, role_name=ROLE_NAME) == (
        read_view(COLUMNS, schema=SCHEMA, role_name=ROLE_NAME),
    )
    assert read_views({'tags': TAG_COLUMNS}, schema=SCHEMA, role_name=ROLE_NAME) == ()
    assert read_views({}, schema=SCHEMA, role_name=ROLE_NAME) == ()


# ============================================================================
# Write rejection
# ============================================================================


def test_reject_function_refuses_every_write_naming_the_view_written_to() -> None:
    """A write through either view raises 42501 naming that view, instead of running as the view's owner."""
    function = reject_function(SCHEMA)

    assert function.kind is RlsObjectKind.FUNCTION
    assert function.target == f'FUNCTION {SCHEMA}.mcp_context_reject_view_write()'
    assert function.statement.startswith(
        f'CREATE OR REPLACE FUNCTION {SCHEMA}.mcp_context_reject_view_write() RETURNS trigger\n'
        '    LANGUAGE plpgsql SET search_path = pg_catalog, pg_temp AS $$\n',
    )
    assert "RAISE EXCEPTION '% is read-only', TG_TABLE_NAME USING ERRCODE = '42501';" in function.statement
    assert 'context_entries_readable' not in function.statement
    assert 'RETURN ' not in function.statement


@pytest.mark.parametrize(('view', 'anchor'), [(READ_VIEW, VIEW), (TAGS_READ_VIEW, TAGS_VIEW)])
def test_reject_trigger_fires_instead_of_every_write_on_its_view(view: str, anchor: str) -> None:
    """INSERT, UPDATE and DELETE on either view all run the one rejecting function instead."""
    trigger = reject_trigger(SCHEMA, view=view)

    assert trigger.kind is RlsObjectKind.TRIGGER
    assert trigger.target == f'TRIGGER reject_writes ON {anchor}'
    assert trigger.statement == (
        f'CREATE TRIGGER reject_writes INSTEAD OF INSERT OR UPDATE OR DELETE ON {anchor} '
        f'FOR EACH ROW EXECUTE FUNCTION {SCHEMA}.mcp_context_reject_view_write()'
    )


def test_the_shared_function_is_created_with_the_first_view_and_a_trigger_with_each() -> None:
    """The read view of context_entries brings the function and its trigger; tags_readable brings its trigger alone."""
    assert write_rejection(SCHEMA, view=READ_VIEW) == (reject_function(SCHEMA), reject_trigger(SCHEMA, view=READ_VIEW))
    assert write_rejection(SCHEMA, view=TAGS_READ_VIEW) == (reject_trigger(SCHEMA, view=TAGS_READ_VIEW),)


def test_reject_objects_follow_the_schema_and_the_view() -> None:
    """The function and the triggers are re-created in another schema with another sha, and each view has its own trigger."""
    assert reject_function(SCHEMA).sha != reject_function('"tenant one"').sha
    assert reject_trigger(SCHEMA, view=READ_VIEW).sha != reject_trigger('"tenant one"', view=READ_VIEW).sha
    assert reject_trigger(SCHEMA, view=READ_VIEW).key != reject_trigger(SCHEMA, view=TAGS_READ_VIEW).key


# ============================================================================
# Privileges
# ============================================================================


@pytest.mark.parametrize(('view', 'anchor'), [(READ_VIEW, VIEW), (TAGS_READ_VIEW, TAGS_VIEW)])
def test_read_view_grant_gives_the_policy_role_select_only(view: str, anchor: str) -> None:
    """The policy role reads through each view and holds no write privilege on it."""
    assert read_view_grant(SCHEMA, ROLE, view=view) == f'GRANT SELECT ON {anchor} TO {ROLE}'


@pytest.mark.parametrize(('view', 'anchor'), [(READ_VIEW, VIEW), (TAGS_READ_VIEW, TAGS_VIEW)])
@pytest.mark.parametrize(
    ('grantee', 'privileges', 'statement'),
    [
        pytest.param('PUBLIC', ('ALL',), 'REVOKE ALL ON {anchor} FROM PUBLIC', id='public'),
        pytest.param('"mcp_outsider"', ('ALL',), 'REVOKE ALL ON {anchor} FROM "mcp_outsider"', id='other-role'),
        pytest.param(ROLE, ('INSERT', 'UPDATE'), 'REVOKE INSERT, UPDATE ON {anchor} FROM ' + ROLE, id='policy-role-extra'),
    ],
)
def test_read_view_revoke_names_the_view_the_grantee_and_the_privileges(
    view: str, anchor: str, grantee: str, privileges: tuple[str, ...], statement: str,
) -> None:
    """A revocation removes everything from another grantee, or the policy role's privileges beyond SELECT."""
    assert read_view_revoke(SCHEMA, grantee, privileges, view=view) == statement.format(anchor=anchor)


@pytest.mark.parametrize('privileges', [(), ('SELECT; DROP TABLE context_entries',), ('select',)])
def test_read_view_revoke_accepts_only_table_privilege_keywords(privileges: tuple[str, ...]) -> None:
    """Privileges are interpolated, so only the keywords of a table's ACL are accepted."""
    with pytest.raises(ValueError, match='privilege'):
        read_view_revoke(SCHEMA, ROLE, privileges, view=READ_VIEW)
