"""Unit tests for the JSON string deserializer middleware."""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import mcp.types as mt
import pytest
from fastmcp.server.middleware.middleware import MiddlewareContext
from fastmcp.tools.tool import Tool
from fastmcp.tools.tool import ToolResult

from app.middleware.json_string_deserializer import JsonStringDeserializerMiddleware
from app.middleware.json_string_deserializer import _is_complex_type
from app.middleware.json_string_deserializer import _resolve_ref
from app.middleware.json_string_deserializer import build_schema_map

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tool(name: str, parameters: dict[str, Any]) -> Tool:
    """Create a minimal Tool for schema-map tests."""
    return Tool(name=name, parameters=parameters)


def _make_context(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
) -> MiddlewareContext[mt.CallToolRequestParams]:
    params = mt.CallToolRequestParams(name=tool_name, arguments=arguments)
    return MiddlewareContext(message=params)


def _make_call_next() -> AsyncMock:
    return AsyncMock(return_value=ToolResult(content=[]))


# ===========================================================================
# TestResolveRef (5 tests)
# ===========================================================================


class TestResolveRef:
    """Tests for _resolve_ref helper."""

    def test_resolves_valid_local_ref(self):
        defs = {'MetadataDict': {'type': 'object', 'additionalProperties': True}}
        result = _resolve_ref('#/$defs/MetadataDict', defs)
        assert result == {'type': 'object', 'additionalProperties': True}

    def test_returns_empty_for_missing_def(self):
        defs = {'MetadataDict': {'type': 'object'}}
        result = _resolve_ref('#/$defs/Missing', defs)
        assert result == {}

    def test_returns_empty_for_non_local_ref(self):
        defs = {'MetadataDict': {'type': 'object'}}
        result = _resolve_ref('https://example.com/schema#/MetadataDict', defs)
        assert result == {}

    def test_returns_empty_for_empty_defs(self):
        result = _resolve_ref('#/$defs/MetadataDict', {})
        assert result == {}

    def test_handles_ref_with_complex_name(self):
        defs = {'My_Complex-Name': {'type': 'array', 'items': {'type': 'string'}}}
        result = _resolve_ref('#/$defs/My_Complex-Name', defs)
        assert result == {'type': 'array', 'items': {'type': 'string'}}


# ===========================================================================
# TestIsComplexType (16 tests)
# ===========================================================================


class TestIsComplexType:
    """Tests for _is_complex_type helper."""

    def test_direct_array_type(self):
        assert _is_complex_type({'type': 'array'}, {}) is True

    def test_direct_object_type(self):
        assert _is_complex_type({'type': 'object'}, {}) is True

    def test_string_type_excluded(self):
        assert _is_complex_type({'type': 'string'}, {}) is False

    def test_integer_type_excluded(self):
        assert _is_complex_type({'type': 'integer'}, {}) is False

    def test_boolean_type_excluded(self):
        assert _is_complex_type({'type': 'boolean'}, {}) is False

    def test_number_type_excluded(self):
        assert _is_complex_type({'type': 'number'}, {}) is False

    def test_null_type_excluded(self):
        assert _is_complex_type({'type': 'null'}, {}) is False

    def test_ref_to_object_in_defs(self):
        defs = {'MetadataDict': {'type': 'object'}}
        schema = {'$ref': '#/$defs/MetadataDict'}
        assert _is_complex_type(schema, defs) is True

    def test_ref_to_array_in_defs(self):
        defs = {'TagList': {'type': 'array', 'items': {'type': 'string'}}}
        schema = {'$ref': '#/$defs/TagList'}
        assert _is_complex_type(schema, defs) is True

    def test_ref_to_string_in_defs(self):
        defs = {'NameDef': {'type': 'string'}}
        schema = {'$ref': '#/$defs/NameDef'}
        assert _is_complex_type(schema, defs) is False

    def test_ref_to_missing_def(self):
        schema = {'$ref': '#/$defs/Missing'}
        assert _is_complex_type(schema, {}) is False

    def test_anyof_with_array_variant(self):
        schema = {'anyOf': [{'type': 'array', 'items': {'type': 'string'}}, {'type': 'null'}]}
        assert _is_complex_type(schema, {}) is True

    def test_anyof_with_object_variant(self):
        schema = {'anyOf': [{'type': 'object'}, {'type': 'null'}]}
        assert _is_complex_type(schema, {}) is True

    def test_anyof_with_ref_to_object(self):
        defs = {'MetadataDict': {'type': 'object'}}
        schema = {'anyOf': [{'$ref': '#/$defs/MetadataDict'}, {'type': 'null'}]}
        assert _is_complex_type(schema, defs) is True

    def test_anyof_with_only_scalars(self):
        schema = {'anyOf': [{'type': 'string'}, {'type': 'null'}]}
        assert _is_complex_type(schema, {}) is False

    def test_empty_schema(self):
        assert _is_complex_type({}, {}) is False


# ===========================================================================
# TestBuildSchemaMap (12 tests)
# ===========================================================================


class TestBuildSchemaMap:
    """Tests for build_schema_map function."""

    def test_detects_array_params(self):
        tool = _tool('store_context', {
            'properties': {'tags': {'type': 'array', 'items': {'type': 'string'}}},
        })
        result = build_schema_map([tool])
        assert result == {'store_context': {'tags'}}

    def test_detects_object_params(self):
        tool = _tool('store_context', {
            'properties': {'metadata': {'type': 'object'}},
        })
        result = build_schema_map([tool])
        assert result == {'store_context': {'metadata'}}

    def test_detects_optional_array_via_anyof(self):
        tool = _tool('store_context', {
            'properties': {
                'tags': {'anyOf': [{'type': 'array', 'items': {'type': 'string'}}, {'type': 'null'}]},
            },
        })
        result = build_schema_map([tool])
        assert result == {'store_context': {'tags'}}

    def test_detects_optional_object_via_anyof(self):
        tool = _tool('store_context', {
            'properties': {
                'metadata': {'anyOf': [{'type': 'object'}, {'type': 'null'}]},
            },
        })
        result = build_schema_map([tool])
        assert result == {'store_context': {'metadata'}}

    def test_detects_ref_to_object_in_defs(self):
        tool = _tool('store_context', {
            '$defs': {'MetadataDict': {'type': 'object'}},
            'properties': {
                'metadata': {'anyOf': [{'$ref': '#/$defs/MetadataDict'}, {'type': 'null'}]},
            },
        })
        result = build_schema_map([tool])
        assert result == {'store_context': {'metadata'}}

    def test_excludes_string_params(self):
        tool = _tool('store_context', {
            'properties': {
                'text': {'type': 'string'},
                'tags': {'type': 'array'},
            },
        })
        result = build_schema_map([tool])
        assert 'text' not in result.get('store_context', set())
        assert 'tags' in result['store_context']

    def test_excludes_integer_params(self):
        tool = _tool('my_tool', {
            'properties': {'count': {'type': 'integer'}},
        })
        result = build_schema_map([tool])
        assert 'my_tool' not in result

    def test_tool_with_no_complex_params_excluded(self):
        tool = _tool('simple_tool', {
            'properties': {
                'name': {'type': 'string'},
                'age': {'type': 'integer'},
            },
        })
        result = build_schema_map([tool])
        assert 'simple_tool' not in result

    def test_multiple_tools(self):
        tool_a = _tool('tool_a', {'properties': {'items': {'type': 'array'}}})
        tool_b = _tool('tool_b', {'properties': {'config': {'type': 'object'}}})
        result = build_schema_map([tool_a, tool_b])
        assert result == {'tool_a': {'items'}, 'tool_b': {'config'}}

    def test_empty_tools_list(self):
        result = build_schema_map([])
        assert result == {}

    def test_mixed_complex_and_scalar_params(self):
        tool = _tool('mixed_tool', {
            'properties': {
                'name': {'type': 'string'},
                'tags': {'type': 'array'},
                'count': {'type': 'integer'},
                'metadata': {'type': 'object'},
            },
        })
        result = build_schema_map([tool])
        assert result == {'mixed_tool': {'tags', 'metadata'}}

    def test_no_properties_key(self):
        tool = _tool('no_props', {'type': 'object'})
        result = build_schema_map([tool])
        assert 'no_props' not in result


# ===========================================================================
# TestJsonStringDeserializerMiddleware (22 tests)
# ===========================================================================


class TestJsonStringDeserializerMiddleware:
    """Tests for the JsonStringDeserializerMiddleware class."""

    # ---- Deserialization ----

    @pytest.mark.asyncio
    async def test_stringified_list_deserialized(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'tags': '["tag1","tag2"]'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] == ['tag1', 'tag2']
        call_next.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_stringified_dict_deserialized(self):
        schema_map = {'store_context': {'metadata'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'metadata': '{"key":"value"}'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['metadata'] == {'key': 'value'}

    @pytest.mark.asyncio
    async def test_stringified_list_of_dicts(self):
        schema_map = {'store_context': {'images'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'images': '[{"data":"abc"}]'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['images'] == [{'data': 'abc'}]

    @pytest.mark.asyncio
    async def test_stringified_list_of_integers(self):
        schema_map = {'get_context_by_ids': {'context_ids'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('get_context_by_ids', {'context_ids': '[1,2,3]'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['context_ids'] == [1, 2, 3]

    # ---- Over-deserialization guard ----

    @pytest.mark.asyncio
    async def test_string_param_not_deserialized(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'text': '["not","a","list"]', 'tags': ['real']})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        # 'text' is not in the schema_map, so stays as string
        assert ctx.message.arguments['text'] == '["not","a","list"]'

    @pytest.mark.asyncio
    async def test_query_param_not_deserialized(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'query': '{"k":"v"}', 'tags': ['x']})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['query'] == '{"k":"v"}'

    # ---- Double-encoding ----

    @pytest.mark.asyncio
    async def test_double_encoded_list(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        # Double-encoded: json.dumps(json.dumps(["a","b"])) produces '"[\\"a\\",\\"b\\"]"'
        double_encoded = json.dumps(json.dumps(['a', 'b']))
        ctx = _make_context('store_context', {'tags': double_encoded})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] == ['a', 'b']

    @pytest.mark.asyncio
    async def test_double_encoded_dict(self):
        schema_map = {'store_context': {'metadata'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        double_encoded = json.dumps(json.dumps({'key': 'value'}))
        ctx = _make_context('store_context', {'metadata': double_encoded})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['metadata'] == {'key': 'value'}

    # ---- Passthrough ----

    @pytest.mark.asyncio
    async def test_native_list_passthrough(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        native_list = ['tag1', 'tag2']
        ctx = _make_context('store_context', {'tags': native_list})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] is native_list

    @pytest.mark.asyncio
    async def test_native_dict_passthrough(self):
        schema_map = {'store_context': {'metadata'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        native_dict = {'key': 'value'}
        ctx = _make_context('store_context', {'metadata': native_dict})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['metadata'] is native_dict

    @pytest.mark.asyncio
    async def test_none_value_passthrough(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'tags': None})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] is None

    @pytest.mark.asyncio
    async def test_unknown_tool_passthrough(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('unknown_tool', {'tags': '["a","b"]'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        # Tool not in schema_map, value stays as string
        assert ctx.message.arguments['tags'] == '["a","b"]'

    @pytest.mark.asyncio
    async def test_no_arguments_passthrough(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', None)
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments is None
        call_next.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_empty_arguments_passthrough(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments == {}
        call_next.assert_awaited_once_with(ctx)

    @pytest.mark.asyncio
    async def test_param_not_in_arguments_passthrough(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'text': 'hello'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments == {'text': 'hello'}

    # ---- Edge cases ----

    @pytest.mark.asyncio
    async def test_invalid_json_string_unchanged(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'tags': 'not json at all'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] == 'not json at all'

    @pytest.mark.asyncio
    async def test_numeric_json_string_unchanged(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'tags': '42'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        # json.loads("42") = 42 (int), not list/dict, so stays as string
        assert ctx.message.arguments['tags'] == '42'

    @pytest.mark.asyncio
    async def test_boolean_json_string_unchanged(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'tags': 'true'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] == 'true'

    @pytest.mark.asyncio
    async def test_null_json_string_unchanged(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'tags': 'null'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] == 'null'

    @pytest.mark.asyncio
    async def test_empty_schema_map(self):
        mw = JsonStringDeserializerMiddleware({})
        ctx = _make_context('store_context', {'tags': '["a","b"]'})
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        # Empty schema_map means no tool has complex params
        assert ctx.message.arguments['tags'] == '["a","b"]'
        call_next.assert_awaited_once_with(ctx)

    # ---- Multi-param and propagation ----

    @pytest.mark.asyncio
    async def test_multiple_complex_params_deserialized(self):
        schema_map = {'store_context': {'tags', 'metadata', 'images'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {
            'tags': '["tag1","tag2"]',
            'metadata': '{"agent_name":"test"}',
            'images': '[{"data":"abc","mime_type":"image/png"}]',
            'text': 'some text',
        })
        call_next = _make_call_next()

        await mw.on_call_tool(ctx, call_next)

        assert ctx.message.arguments['tags'] == ['tag1', 'tag2']
        assert ctx.message.arguments['metadata'] == {'agent_name': 'test'}
        assert ctx.message.arguments['images'] == [{'data': 'abc', 'mime_type': 'image/png'}]
        # text is not in schema_map, stays as string
        assert ctx.message.arguments['text'] == 'some text'

    @pytest.mark.asyncio
    async def test_call_next_return_value_propagated(self):
        schema_map = {'store_context': {'tags'}}
        mw = JsonStringDeserializerMiddleware(schema_map)
        ctx = _make_context('store_context', {'tags': '["a"]'})
        expected_result = ToolResult(content=[])
        call_next = AsyncMock(return_value=expected_result)

        result = await mw.on_call_tool(ctx, call_next)

        assert result is expected_result
