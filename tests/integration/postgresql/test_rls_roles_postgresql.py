"""The PostgreSQL test container carries the roles the row-level security tests connect as.

``initdb/02-roles.sql`` creates them on every container start: the NOLOGIN policy role,
the request login role that is its member, a non-superuser table owner and an outsider
with no membership. The bootstrap ``mcp_test`` superuser stays the administrative role.
"""

from __future__ import annotations

import asyncpg
import pytest

from tests.integration.postgresql.conftest import login_url

pytestmark = [pytest.mark.requires_docker_postgres, pytest.mark.integration]

# rolname -> (rolcanlogin, rolsuper, rolbypassrls)
EXPECTED_ATTRIBUTES = {
    'mcp_context_request': (False, False, False),
    'mcp_app': (True, False, False),
    'mcp_owner': (True, False, False),
    'mcp_outsider': (True, False, False),
    'mcp_test': (True, True, True),
}
LOGIN_ROLES = ['mcp_app', 'mcp_owner', 'mcp_outsider']


@pytest.mark.asyncio
async def test_roles_carry_the_expected_attributes(pg_test_url: str) -> None:
    conn = await asyncpg.connect(pg_test_url)
    try:
        rows = await conn.fetch(
            'SELECT rolname, rolcanlogin, rolsuper, rolbypassrls FROM pg_catalog.pg_roles '
            'WHERE rolname = ANY($1::text[])',
            list(EXPECTED_ATTRIBUTES),
        )
    finally:
        await conn.close()

    assert {row['rolname']: (row['rolcanlogin'], row['rolsuper'], row['rolbypassrls']) for row in rows} == EXPECTED_ATTRIBUTES


@pytest.mark.asyncio
async def test_only_the_request_role_is_a_member_of_the_policy_role(pg_test_url: str) -> None:
    conn = await asyncpg.connect(pg_test_url)
    try:
        members = await conn.fetch(
            'SELECT m.rolname, a.inherit_option FROM pg_catalog.pg_auth_members a '
            'JOIN pg_catalog.pg_roles m ON m.oid = a.member '
            "WHERE a.roleid = 'mcp_context_request'::regrole",
        )
        privileges = {
            role: await conn.fetchval("SELECT pg_catalog.pg_has_role($1, 'mcp_context_request', 'USAGE')", role)
            for role in LOGIN_ROLES
        }
    finally:
        await conn.close()

    assert [(row['rolname'], row['inherit_option']) for row in members] == [('mcp_app', True)]
    assert privileges == {'mcp_app': True, 'mcp_owner': False, 'mcp_outsider': False}


@pytest.mark.asyncio
@pytest.mark.parametrize('role', LOGIN_ROLES)
async def test_login_role_connects_with_its_password(pg_test_url: str, role: str) -> None:
    conn = await asyncpg.connect(login_url(pg_test_url, role))
    try:
        assert await conn.fetchval('SELECT current_user') == role
    finally:
        await conn.close()
