-- RFC 7396 JSON Merge Patch Implementation for PostgreSQL
-- This migration adds a production-grade jsonb_merge_patch() function that implements
-- TRUE recursive deep merge semantics as specified in RFC 7396.
--
-- RFC 7396 Specification: https://datatracker.ietf.org/doc/html/rfc7396
--
-- Key Semantics:
-- 1. If patch is not an object, return patch directly (replaces target entirely)
-- 2. If target is not an object, start with empty object for merging
-- 3. For each key in patch:
--    - If value is null, DELETE that key from target
--    - Otherwise, RECURSIVELY merge the value into target's corresponding key
-- 4. Keys in target but not in patch are PRESERVED (unchanged)
--
-- This function replaces the shallow || - pattern which only handles top-level keys.
-- The new implementation correctly handles deeply nested structures.

CREATE OR REPLACE FUNCTION jsonb_merge_patch(
    target jsonb,
    patch jsonb
)
RETURNS jsonb
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
AS $$
BEGIN
    -- RFC 7396 Section 2, Step 1:
    -- "If the provided Merge Patch is not a JSON object, then the result is to
    --  replace the entire target with the entire patch."
    -- This covers: null, arrays, strings, numbers, booleans - all non-object types
    IF patch IS NULL OR jsonb_typeof(patch) != 'object' THEN
        RETURN patch;
    END IF;

    -- RFC 7396 Section 2, Step 2:
    -- "If the original document is not an object, its value is replaced
    --  entirely by the object provided in the patch document."
    -- This means: start with empty object if target is null or non-object
    IF target IS NULL OR jsonb_typeof(target) != 'object' THEN
        target := '{}'::jsonb;
    END IF;

    -- RFC 7396 Section 2, Step 3 (recursive merge):
    -- Use FULL OUTER JOIN to iterate over all keys from both target and patch.
    -- This ensures we handle:
    --   - Keys only in target (preserved)
    --   - Keys only in patch (added)
    --   - Keys in both (merged/updated/deleted)
    --
    -- The WHERE clause implements RFC 7396 null-deletion semantics:
    --   - Keys with null values in patch are DELETED (filtered out)
    --   - Keys only in target (patch_key IS NULL) are PRESERVED
    RETURN COALESCE(
        (
            SELECT jsonb_object_agg(
                COALESCE(target_key, patch_key),
                CASE
                    -- Key only in target (not in patch): preserve target value unchanged
                    WHEN patch_key IS NULL THEN target_value
                    -- Key in both or only in patch: recursively merge
                    -- This handles nested objects correctly via recursive call
                    ELSE jsonb_merge_patch(target_value, patch_value)
                END
            )
            FROM jsonb_each(target) AS t(target_key, target_value)
            FULL OUTER JOIN jsonb_each(patch) AS p(patch_key, patch_value)
                ON t.target_key = p.patch_key
            -- RFC 7396 null-deletion: Filter out keys where patch has null value
            -- Condition: (target-only keys) OR (patch value is not JSON null)
            -- This correctly handles RFC 7396 test case #13: {"e":null} + {"a":1} = {"e":null,"a":1}
            -- The existing null in target is preserved because it's not being patched with null
            WHERE patch_key IS NULL OR jsonb_typeof(patch_value) != 'null'
        ),
        '{}'::jsonb
    );
END;
$$;

-- ============================================================================
-- RFC 7396 Appendix A: Test Cases Verification
-- ============================================================================
-- All 15 test cases from RFC 7396 are documented below with expected results.
-- These can be used to verify the function works correctly:
--
-- Test Case 1: Simple value replacement
-- SELECT jsonb_merge_patch('{"a":"b"}'::jsonb, '{"a":"c"}'::jsonb);
-- Expected: {"a":"c"}
--
-- Test Case 2: Add new key
-- SELECT jsonb_merge_patch('{"a":"b"}'::jsonb, '{"b":"c"}'::jsonb);
-- Expected: {"a":"b","b":"c"}
--
-- Test Case 3: Delete key with null (RFC 7396 core semantic)
-- SELECT jsonb_merge_patch('{"a":"b"}'::jsonb, '{"a":null}'::jsonb);
-- Expected: {}
--
-- Test Case 4: Delete one key, preserve others
-- SELECT jsonb_merge_patch('{"a":"b","b":"c"}'::jsonb, '{"a":null}'::jsonb);
-- Expected: {"b":"c"}
--
-- Test Case 5: Array replacement (arrays are NOT merged)
-- SELECT jsonb_merge_patch('{"a":["b"]}'::jsonb, '{"a":"c"}'::jsonb);
-- Expected: {"a":"c"}
--
-- Test Case 6: Replace value with array
-- SELECT jsonb_merge_patch('{"a":"c"}'::jsonb, '{"a":["b"]}'::jsonb);
-- Expected: {"a":["b"]}
--
-- Test Case 7: CRITICAL - Nested object merge with deletion
-- SELECT jsonb_merge_patch('{"a":{"b":"c"}}'::jsonb, '{"a":{"b":"d","c":null}}'::jsonb);
-- Expected: {"a":{"b":"d"}}
-- Note: This is where the old || - pattern failed (shallow merge replaced entire nested object)
--
-- Test Case 8: Array of objects replacement
-- SELECT jsonb_merge_patch('{"a":[{"b":"c"}]}'::jsonb, '{"a":[1]}'::jsonb);
-- Expected: {"a":[1]}
--
-- Test Case 9: Array replacement (top-level arrays)
-- SELECT jsonb_merge_patch('["a","b"]'::jsonb, '["c","d"]'::jsonb);
-- Expected: ["c","d"]
--
-- Test Case 10: Object replaced by array
-- SELECT jsonb_merge_patch('{"a":"b"}'::jsonb, '["c"]'::jsonb);
-- Expected: ["c"]
--
-- Test Case 11: Null patch replaces everything
-- SELECT jsonb_merge_patch('{"a":"foo"}'::jsonb, 'null'::jsonb);
-- Expected: null
--
-- Test Case 12: String patch replaces everything
-- SELECT jsonb_merge_patch('{"a":"foo"}'::jsonb, '"bar"'::jsonb);
-- Expected: "bar"
--
-- Test Case 13: CRITICAL - Existing null value preserved (NOT deleted)
-- SELECT jsonb_merge_patch('{"e":null}'::jsonb, '{"a":1}'::jsonb);
-- Expected: {"a":1,"e":null}
-- Note: The null value in TARGET is preserved because patch doesn't modify it
--
-- Test Case 14: Array becomes object after patch
-- SELECT jsonb_merge_patch('[1,2]'::jsonb, '{"a":"b","c":null}'::jsonb);
-- Expected: {"a":"b"}
--
-- Test Case 15: CRITICAL - Deeply nested null deletion
-- SELECT jsonb_merge_patch('{}'::jsonb, '{"a":{"bb":{"ccc":null}}}'::jsonb);
-- Expected: {"a":{"bb":{}}}
-- Note: This is where recursive merge is essential - the deeply nested null causes
-- deletion at the deepest level, but the containing objects are preserved
--
-- ============================================================================
-- Verification Query (run all test cases at once):
-- ============================================================================
-- DO $$
-- DECLARE
--     test_results boolean[];
-- BEGIN
--     test_results := ARRAY[
--         jsonb_merge_patch('{"a":"b"}', '{"a":"c"}') = '{"a":"c"}',
--         jsonb_merge_patch('{"a":"b"}', '{"b":"c"}') = '{"a":"b","b":"c"}',
--         jsonb_merge_patch('{"a":"b"}', '{"a":null}') = '{}',
--         jsonb_merge_patch('{"a":"b","b":"c"}', '{"a":null}') = '{"b":"c"}',
--         jsonb_merge_patch('{"a":["b"]}', '{"a":"c"}') = '{"a":"c"}',
--         jsonb_merge_patch('{"a":"c"}', '{"a":["b"]}') = '{"a":["b"]}',
--         jsonb_merge_patch('{"a":{"b":"c"}}', '{"a":{"b":"d","c":null}}') = '{"a":{"b":"d"}}',
--         jsonb_merge_patch('{"a":[{"b":"c"}]}', '{"a":[1]}') = '{"a":[1]}',
--         jsonb_merge_patch('["a","b"]', '["c","d"]') = '["c","d"]',
--         jsonb_merge_patch('{"a":"b"}', '["c"]') = '["c"]',
--         jsonb_merge_patch('{"a":"foo"}', 'null') IS NULL,
--         jsonb_merge_patch('{"a":"foo"}', '"bar"') = '"bar"',
--         jsonb_merge_patch('{"e":null}', '{"a":1}') = '{"a":1,"e":null}',
--         jsonb_merge_patch('[1,2]', '{"a":"b","c":null}') = '{"a":"b"}',
--         jsonb_merge_patch('{}', '{"a":{"bb":{"ccc":null}}}') = '{"a":{"bb":{}}}'
--     ];
--
--     FOR i IN 1..array_length(test_results, 1) LOOP
--         IF NOT test_results[i] THEN
--             RAISE EXCEPTION 'RFC 7396 Test Case % failed', i;
--         END IF;
--     END LOOP;
--
--     RAISE NOTICE 'All 15 RFC 7396 test cases passed!';
-- END;
-- $$;
