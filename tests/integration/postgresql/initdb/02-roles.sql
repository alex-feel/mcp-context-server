-- Roles for the row-level security tests, created on every container start
-- (the data directory is tmpfs, so initdb always runs).
--
-- mcp_context_request: the NOLOGIN role the policies name.
-- mcp_app: the request login role, an inheriting member of the policy role.
-- mcp_owner: a login role that owns the tables of an RLS test database; it is
--   not a superuser, so those tests prove the model needs no superuser.
-- mcp_outsider: a login role with no membership, for the refusal tests.
-- mcp_test, the bootstrap superuser, stays the administrative role.
CREATE ROLE mcp_context_request NOLOGIN NOSUPERUSER NOBYPASSRLS;
CREATE ROLE mcp_app LOGIN PASSWORD 'mcp_app' NOSUPERUSER NOBYPASSRLS INHERIT;
GRANT mcp_context_request TO mcp_app;
CREATE ROLE mcp_owner LOGIN PASSWORD 'mcp_owner' NOSUPERUSER NOBYPASSRLS;
CREATE ROLE mcp_outsider LOGIN PASSWORD 'mcp_outsider' NOSUPERUSER NOBYPASSRLS;
