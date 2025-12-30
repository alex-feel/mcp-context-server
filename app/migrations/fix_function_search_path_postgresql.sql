-- Fix function search_path for security (CVE-2018-1058 mitigation)
-- This migration sets search_path for all functions to prevent
-- potential search_path hijacking attacks.
--
-- This migration is IDEMPOTENT - safe to run multiple times.
-- ALTER FUNCTION SET works on existing functions and simply
-- updates the configuration parameter.
--
-- Applied to:
-- 1. update_updated_at_column() - main schema trigger
-- 2. update_embedding_metadata_timestamp() - semantic search trigger
-- 3. jsonb_merge_patch(jsonb, jsonb) - RFC 7396 implementation
--
-- Reference: CVE-2018-1058, PostgreSQL Security Best Practices

-- Fix update_updated_at_column (from postgresql_schema.sql)
-- This function always exists after schema initialization
ALTER FUNCTION public.update_updated_at_column()
SET search_path = pg_catalog, pg_temp;

-- Fix update_embedding_metadata_timestamp (from add_semantic_search_postgresql.sql)
-- This function only exists if semantic search was enabled
-- DO block handles conditional execution
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
        AND p.proname = 'update_embedding_metadata_timestamp'
    ) THEN
        ALTER FUNCTION public.update_embedding_metadata_timestamp()
        SET search_path = pg_catalog, pg_temp;
    END IF;
END $$;

-- Fix jsonb_merge_patch (from add_jsonb_merge_patch_postgresql.sql)
-- This function only exists for PostgreSQL backends
-- DO block handles conditional execution
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc p
        JOIN pg_namespace n ON p.pronamespace = n.oid
        WHERE n.nspname = 'public'
        AND p.proname = 'jsonb_merge_patch'
    ) THEN
        ALTER FUNCTION public.jsonb_merge_patch(jsonb, jsonb)
        SET search_path = pg_catalog, pg_temp;
    END IF;
END $$;
