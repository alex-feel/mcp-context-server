-- Bootstrap script run on first PostgreSQL container start.
-- The MCP server lazily detects the pgvector extension's schema via
-- pg_extension; the extension must exist before the first pool
-- connection registers its codec.
CREATE EXTENSION IF NOT EXISTS vector;
