"""Tests for app.tools._validation: image validation and normalization, the image-count
tool schema bound, and the indexed-metadata value checks.
"""

import base64
from collections.abc import Awaitable
from collections.abc import Callable
from typing import cast
from typing import get_type_hints

import pytest
from fastmcp.exceptions import ToolError
from pydantic import TypeAdapter

import app.tools._validation as validation_module
from app.models import MAX_IMAGES_PER_ENTRY
from app.settings import get_settings
from app.tools._validation import entry_boundary_error
from app.tools._validation import indexed_value_error
from app.tools._validation import reject_invalid_indexed_values
from app.tools._validation import validate_and_normalize_images
from app.tools.context import store_context
from app.tools.context import update_context

# Valid base64 PNG (1x1 transparent pixel)
VALID_BASE64_PNG = base64.b64encode(b'\x89PNG\r\n\x1a\n' + b'\x00' * 50).decode()


class TestValidateAndNormalizeImages:
    """Test validate_and_normalize_images shared function."""

    def test_none_images_returns_text(self) -> None:
        """None images returns empty list, 'text' content type, no errors."""
        images, content_type, errors = validate_and_normalize_images(None)
        assert images == []
        assert content_type == 'text'
        assert errors == []

    def test_empty_list_returns_text(self) -> None:
        """Empty list returns empty list, 'text' content type, no errors."""
        images, content_type, errors = validate_and_normalize_images([])
        assert images == []
        assert content_type == 'text'
        assert errors == []

    def test_valid_image_returns_multimodal(self) -> None:
        """Valid base64 image returns multimodal content type."""
        img = {'data': VALID_BASE64_PNG, 'mime_type': 'image/png'}
        images, content_type, errors = validate_and_normalize_images([img])
        assert content_type == 'multimodal'
        assert errors == []
        assert len(images) == 1

    def test_defaults_mime_type(self) -> None:
        """Image without mime_type gets 'image/png' default."""
        img = {'data': VALID_BASE64_PNG}
        images, content_type, errors = validate_and_normalize_images([img])
        assert images[0]['mime_type'] == 'image/png'
        assert content_type == 'multimodal'
        assert errors == []

    def test_json_string_image_metadata_passes(self) -> None:
        """A JSON-encoded-string metadata value (the string-valued tool contract) passes.

        Per-image dicts are dict[str, str]: callers pass structured image metadata as a
        JSON-encoded string, which carries no bare float and is not rejected.
        """
        img = {'data': VALID_BASE64_PNG, 'mime_type': 'image/png', 'metadata': '{"position": 1}'}
        _, content_type, errors = validate_and_normalize_images([img])
        assert content_type == 'multimodal'
        assert errors == []

    def test_raise_mode_rejects_dict_image_metadata(self) -> None:
        """A dict-valued image metadata (untyped batch path only) is rejected.

        Per-image metadata crosses the boundary as a JSON-ENCODED STRING: the typed
        single-entry tools declare images as list[dict[str, str]], so a dict there is
        a Pydantic error. The untyped batch path bypassed that and stored the dict,
        which the strict get_context_by_ids output schema then refused to serialize --
        making the entry permanently unreadable. Rejecting the shape here also
        subsumes the NaN/Infinity parity hazard: only a bare float serializes to the
        invalid-JSON tokens PostgreSQL's jsonb column rejects, and a JSON-encoded
        string carries none.
        """
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': {'score': float('nan')}}])
        with pytest.raises(ToolError, match='Image 0 metadata must be a JSON-encoded string'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_collect_mode_rejects_non_string_image_metadata(self) -> None:
        """collect mode reports the same shape rejection as a per-entry error."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': {'deep': {'x': 1}}}])
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'Image 0 metadata must be a JSON-encoded string' in errors[0]

    def test_null_image_metadata_is_treated_as_absent(self) -> None:
        """An explicit null metadata means "no metadata" and is accepted.

        It stores as SQL NULL exactly like an omitted key, so refusing it would add
        friction without preventing anything.
        """
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'metadata': None}])
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'multimodal'
        assert errors == []

    def test_raise_mode_missing_data(self) -> None:
        """error_mode='raise' raises ToolError for missing data field."""
        with pytest.raises(ToolError, match='Image 0 is missing required "data" field'):
            validate_and_normalize_images([{'mime_type': 'image/png'}], error_mode='raise')

    def test_raise_mode_empty_data(self) -> None:
        """error_mode='raise' raises ToolError for empty data field."""
        with pytest.raises(ToolError, match='Image 0 has empty "data" field'):
            validate_and_normalize_images([{'data': '   '}], error_mode='raise')

    def test_raise_mode_invalid_base64(self) -> None:
        """error_mode='raise' raises ToolError for invalid base64 encoding."""
        with pytest.raises(ToolError, match='invalid base64 encoding'):
            validate_and_normalize_images([{'data': '!!!not-base64!!!'}], error_mode='raise')

    def test_raise_mode_oversized(self) -> None:
        """error_mode='raise' raises ToolError for oversized image."""
        # Create data that exceeds MAX_IMAGE_SIZE_MB (10MB default)
        large_data = base64.b64encode(b'\x00' * (11 * 1024 * 1024)).decode()
        with pytest.raises(ToolError, match='exceeds.*MB limit'):
            validate_and_normalize_images([{'data': large_data}], error_mode='raise')

    def test_collect_mode_missing_data(self) -> None:
        """error_mode='collect' returns errors list for missing data."""
        images, content_type, errors = validate_and_normalize_images(
            [{'mime_type': 'image/png'}], error_mode='collect',
        )
        assert len(errors) == 1
        assert 'Image 0 is missing required "data" field' in errors[0]

    def test_collect_mode_empty_data(self) -> None:
        """error_mode='collect' returns errors list, does not raise."""
        images, content_type, errors = validate_and_normalize_images(
            [{'data': ''}], error_mode='collect',
        )
        assert len(errors) == 1
        assert 'Image 0 has empty "data" field' in errors[0]

    def test_collect_mode_multiple_errors(self) -> None:
        """First error returned in collect mode (early return)."""
        imgs = [
            {'mime_type': 'image/png'},  # missing data
            {'data': ''},  # empty data
        ]
        _, _, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert len(errors) == 1
        assert 'Image 0' in errors[0]

    def test_enumerate_index_in_errors(self) -> None:
        """Error messages include correct image index."""
        imgs = [
            {'data': VALID_BASE64_PNG, 'mime_type': 'image/png'},
            {'mime_type': 'image/png'},  # missing data at index 1
        ]
        _, _, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert len(errors) == 1
        assert 'Image 1' in errors[0]

    def test_raise_mode_non_string_mime_type(self) -> None:
        """A present non-string mime_type (untyped batch input) is rejected in raise mode."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'mime_type': 123}])
        with pytest.raises(ToolError, match='Image 0 has a non-string "mime_type" field'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_raise_mode_null_mime_type(self) -> None:
        """A present null mime_type is rejected, not bound into the NOT NULL column."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'mime_type': None}])
        with pytest.raises(ToolError, match='Image 0 has a non-string "mime_type" field'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_collect_mode_non_string_mime_type(self) -> None:
        """collect mode records a non-string mime_type error instead of raising."""
        imgs = cast('list[dict[str, str]]', [{'data': VALID_BASE64_PNG, 'mime_type': None}])
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'Image 0 has a non-string "mime_type" field' in errors[0]

    def test_raise_mode_non_string_data(self) -> None:
        """A present non-string data value is rejected before .strip()/base64 decode."""
        imgs = cast('list[dict[str, str]]', [{'data': 123, 'mime_type': 'image/png'}])
        with pytest.raises(ToolError, match='Image 0 has a non-string "data" field'):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_collect_mode_non_string_data(self) -> None:
        """collect mode records a non-string data error instead of crashing with AttributeError."""
        imgs = cast('list[dict[str, str]]', [{'data': 123, 'mime_type': 'image/png'}])
        _, _, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert len(errors) == 1
        assert 'Image 0 has a non-string "data" field' in errors[0]

    def test_raise_mode_garbage_rejected_by_strict_decode(self) -> None:
        """Garbage input (all non-alphabet characters) fails loudly under the strict decode.

        The lenient decode used previously silently discarded every character outside
        the alphabet, so a value like '!!!!' decoded to b'' and was caught only by the
        zero-byte guard. The strict decode (validate=True) rejects it directly with a
        clear, deterministic per-image error.
        """
        with pytest.raises(ToolError, match='Image 0 has invalid base64 encoding: Invalid base64 data'):
            validate_and_normalize_images([{'data': '!!!!'}], error_mode='raise')

    def test_collect_mode_garbage_rejected_by_strict_decode(self) -> None:
        """collect mode records the strict-decode failure instead of raising."""
        _, content_type, errors = validate_and_normalize_images(
            [{'data': '@#$%'}], error_mode='collect',
        )
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'invalid base64 encoding' in errors[0]

    def test_raise_mode_empty_data_uri_payload_decodes_to_zero_bytes(self) -> None:
        """A data-URI prefix with an empty payload normalizes to '' and is rejected as zero bytes."""
        with pytest.raises(ToolError, match='Image 0 "data" decodes to zero bytes'):
            validate_and_normalize_images([{'data': 'data:image/png;base64,'}], error_mode='raise')

    def test_whitespace_wrapped_base64_still_accepted(self) -> None:
        """Newline-wrapped base64 is accepted: whitespace is removed by normalization before the strict decode."""
        wrapped = VALID_BASE64_PNG[:4] + '\n' + VALID_BASE64_PNG[4:]
        images, content_type, errors = validate_and_normalize_images(
            [{'data': wrapped, 'mime_type': 'image/png'}], error_mode='collect',
        )
        assert content_type == 'multimodal'
        assert errors == []
        assert images[0]['data'] == VALID_BASE64_PNG

    def test_data_uri_prefix_stripped_decodes_to_same_bytes(self) -> None:
        """A data-URI-prefixed payload decodes to the same bytes as the bare payload.

        Under the lenient decode, a prefix whose base64-alphabet character count is a
        multiple of 4 (e.g. some jpeg data-URIs) decoded as garbage bytes silently
        prepended to the image. Normalization strips the prefix so the stored bytes
        are exactly the intended image.
        """
        prefixed = {'data': 'data:image/jpeg;base64,' + VALID_BASE64_PNG}
        images, content_type, errors = validate_and_normalize_images([prefixed])
        assert errors == []
        assert content_type == 'multimodal'
        assert images[0]['data'] == VALID_BASE64_PNG
        assert base64.b64decode(images[0]['data'], validate=True) == base64.b64decode(VALID_BASE64_PNG, validate=True)

    def test_url_safe_alphabet_normalized_to_standard(self) -> None:
        """A URL-safe-alphabet payload is translated to the standard alphabet and decodes correctly."""
        raw = bytes(range(251, 256)) * 6
        standard = base64.b64encode(raw).decode()
        url_safe = base64.urlsafe_b64encode(raw).decode()
        assert url_safe != standard  # fixture sanity: the translation is actually exercised
        images, content_type, errors = validate_and_normalize_images([{'data': url_safe}])
        assert errors == []
        assert content_type == 'multimodal'
        assert images[0]['data'] == standard
        assert base64.b64decode(images[0]['data'], validate=True) == raw

    def test_url_safe_payload_without_padding_restored(self) -> None:
        """A URL-safe payload with stripped '=' padding is repadded and decodes to the original bytes."""
        raw = b'\xfb\xef\x01\x02'
        stripped = base64.urlsafe_b64encode(raw).decode().rstrip('=')
        images, _, errors = validate_and_normalize_images([{'data': stripped}])
        assert errors == []
        assert images[0]['data'] == base64.b64encode(raw).decode()
        assert base64.b64decode(images[0]['data'], validate=True) == raw

    def test_normalization_mutates_input_dict_in_place(self) -> None:
        """The canonical payload is written back into the caller's dict.

        The batch tools rely on this: they discard the returned list and later pass
        the same dict objects to the transaction helpers, so the repository re-decode
        must see the normalized payload through the original dict.
        """
        img = {'data': 'data:image/png;base64,' + VALID_BASE64_PNG}
        validate_and_normalize_images([img])
        assert img['data'] == VALID_BASE64_PNG

    def test_count_over_limit_rejected_raise_mode(self) -> None:
        """More than MAX_IMAGES_PER_ENTRY images is rejected at the shared chokepoint."""
        imgs = [{'data': VALID_BASE64_PNG} for _ in range(MAX_IMAGES_PER_ENTRY + 1)]
        expected = (
            f'Too many images: {MAX_IMAGES_PER_ENTRY + 1} provided, '
            f'maximum is {MAX_IMAGES_PER_ENTRY} per entry'
        )
        with pytest.raises(ToolError, match=expected):
            validate_and_normalize_images(imgs, error_mode='raise')

    def test_count_over_limit_rejected_collect_mode(self) -> None:
        """collect mode records the count-limit error (covers the batch tools' untyped path)."""
        imgs = [{'data': VALID_BASE64_PNG} for _ in range(MAX_IMAGES_PER_ENTRY + 1)]
        _, content_type, errors = validate_and_normalize_images(imgs, error_mode='collect')
        assert content_type == 'text'
        assert len(errors) == 1
        assert 'Too many images' in errors[0]

    def test_count_at_limit_accepted(self) -> None:
        """Exactly MAX_IMAGES_PER_ENTRY images passes validation."""
        imgs = [{'data': VALID_BASE64_PNG} for _ in range(MAX_IMAGES_PER_ENTRY)]
        images, content_type, errors = validate_and_normalize_images(imgs)
        assert errors == []
        assert content_type == 'multimodal'
        assert len(images) == MAX_IMAGES_PER_ENTRY


class TestImageCountLimitToolSchema:
    """The MCP wire schema advertises the image-count bound on the live tool params."""

    @pytest.mark.parametrize('tool_fn', [store_context, update_context])
    def test_images_param_advertises_max_items(self, tool_fn: Callable[..., Awaitable[object]]) -> None:
        """The images parameter declares maxItems=MAX_IMAGES_PER_ENTRY in its JSON schema."""
        hints = get_type_hints(tool_fn, include_extras=True)
        schema = TypeAdapter(hints['images']).json_schema()
        branches = schema.get('anyOf', [schema])
        array_branches = [b for b in branches if isinstance(b, dict) and b.get('type') == 'array']
        assert array_branches, f'no array branch in images schema: {schema}'
        assert array_branches[0].get('maxItems') == MAX_IMAGES_PER_ENTRY


class TestIndexedValueCastCompatibility:
    """A typed METADATA_INDEXED_FIELDS entry is validated at the write boundary.

    An ``integer``/``boolean``/``float`` type hint becomes a hard SQL cast inside
    the PostgreSQL expression index, which PostgreSQL evaluates on INSERT. Without a
    boundary check a cast-incompatible value passed every guard, paid the full
    generation pass, and then aborted the transaction with a raw driver error --
    while storing happily on SQLite, whose json_extract index applies no cast.
    """

    @pytest.fixture
    def typed_indexed_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configure one field per castable type hint for the module under test."""
        monkeypatch.setenv(
            'METADATA_INDEXED_FIELDS',
            'status,priority:integer,completed:boolean,score:float',
        )
        get_settings.cache_clear()
        monkeypatch.setattr(validation_module, 'settings', get_settings())

    @pytest.mark.usefixtures('typed_indexed_fields')
    @pytest.mark.parametrize(
        ('metadata', 'expected'),
        [
            ({'priority': 5}, None),
            ({'priority': '5'}, None),
            ({'priority': ' -12 '}, None),
            # PostgreSQL 16 added non-decimal literals and '_' digit separators to the
            # integer and numeric input functions, so these cast fine and refusing them
            # would block a store both backends accept.
            ({'priority': '0x10'}, None),
            ({'priority': '0o17'}, None),
            ({'priority': '0b101'}, None),
            ({'priority': '1_000'}, None),
            ({'priority': '0x_10'}, None),
            ({'priority': '1__0'}, 'not a valid integer'),
            ({'priority': '_1'}, 'not a valid integer'),
            ({'priority': '1_'}, 'not a valid integer'),
            # PostgreSQL trims ASCII whitespace only: its scanners run under the C locale
            # and test single bytes, so a Unicode space is ordinary content the cast
            # chokes on. Python's argument-less strip() would remove it and accept these.
            ({'priority': '\xa05'}, 'not a valid integer'),
            ({'priority': '5　'}, 'not a valid integer'),
            ({'priority': ' 5'}, 'not a valid integer'),
            ({'completed': '\xa0true'}, 'not a valid boolean'),
            ({'score': '\x1c1.5'}, 'not a valid float'),
            ({'priority': 'high'}, 'not a valid integer'),
            ({'priority': 5.5}, 'not a valid integer'),
            ({'priority': True}, 'not a valid integer'),
            ({'priority': [1, 2]}, 'cannot be stored'),
            ({'priority': 99999999999}, 'out of range'),
            ({'priority': -99999999999}, 'out of range'),
            ({'priority': None}, None),
            ({'completed': True}, None),
            ({'completed': 'yes'}, None),
            # 'of' is an accepted prefix of 'off'; a lone 'o' is ambiguous and rejected.
            ({'completed': 'of'}, None),
            ({'completed': 'o'}, 'not a valid boolean'),
            ({'completed': 'maybe'}, 'not a valid boolean'),
            ({'completed': 7}, 'not a valid boolean'),
            ({'score': 1.5}, None),
            ({'score': '2e3'}, None),
            ({'score': 'NaN'}, None),
            ({'score': 'high'}, 'not a valid float'),
            ({'status': 'anything at all'}, None),
            ({'unindexed': 'anything at all'}, None),
        ],
    )
    def test_cast_compatibility_matches_postgresql(
        self, metadata: dict[str, object], expected: str | None,
    ) -> None:
        """Values PostgreSQL would accept pass; values it would reject are refused."""
        result = indexed_value_error(metadata=metadata)
        if expected is None:
            assert result is None
        else:
            assert result is not None
            assert expected in result

    @pytest.mark.usefixtures('typed_indexed_fields')
    def test_raising_wrapper_reports_the_field(self) -> None:
        """The single-entry wrapper names the offending field, as the length cap does."""
        with pytest.raises(ToolError, match="metadata field 'priority'"):
            reject_invalid_indexed_values(metadata={'priority': 'high'})

    @pytest.mark.usefixtures('typed_indexed_fields')
    def test_batch_chokepoint_covers_metadata_and_patch(self) -> None:
        """The untyped batch chokepoint checks both the replacement and the patch form."""
        assert entry_boundary_error(metadata={'priority': 'high'}) is not None
        assert entry_boundary_error(metadata_patch={'completed': 'maybe'}) is not None

    def test_default_configuration_has_no_castable_hints(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The shipped defaults declare only string/array/object, so nothing casts.

        Array and object fields build no expression index at all (the always-present
        GIN index serves them), so their values must not be type-checked here.
        """
        get_settings.cache_clear()
        monkeypatch.setattr(validation_module, 'settings', get_settings())
        assert indexed_value_error(metadata={'status': 'anything', 'technologies': ['a', 'b']}) is None


class TestIndexedValueLengthMeasuresTheIndexedText:
    """The length cap measures the text ``metadata->>'<field>'`` actually yields.

    ``->>`` renders a JSON string unquoted and every other JSON value as its serialized
    form, so a list or object stored under a string-typed indexed field is indexed at
    its full serialized width. A cap that inspected only ``str`` values let such a
    container through to abort the PostgreSQL INSERT inside the store transaction --
    after a full generation pass, while charging the circuit breaker -- for a value
    SQLite stored happily.
    """

    @pytest.fixture
    def string_indexed_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Configure one string-typed and one array-typed indexed field."""
        monkeypatch.setenv('METADATA_INDEXED_FIELDS', 'project,technologies:array')
        get_settings.cache_clear()
        monkeypatch.setattr(validation_module, 'settings', get_settings())

    @pytest.mark.usefixtures('string_indexed_field')
    def test_oversized_container_under_a_string_field_is_refused(self) -> None:
        """A list whose serialized text exceeds the cap is refused, like a long string."""
        oversized = ['x' * 100] * 40

        message = indexed_value_error(metadata={'project': oversized})

        assert message is not None
        assert 'too long' in message

    @pytest.mark.usefixtures('string_indexed_field')
    def test_small_container_under_a_string_field_is_accepted(self) -> None:
        """A container whose serialized text fits is stored, not blanket-refused."""
        assert indexed_value_error(metadata={'project': ['alpha', 'beta']}) is None

    @pytest.mark.usefixtures('string_indexed_field')
    def test_array_typed_field_is_exempt_from_the_cap(self) -> None:
        """An array-typed field builds no expression index, so its width is unbounded.

        It is served by the always-present GIN index, which hashes its entries.
        """
        assert indexed_value_error(metadata={'technologies': ['x' * 100] * 40}) is None

    @pytest.mark.usefixtures('string_indexed_field')
    def test_batch_chokepoint_sees_the_same_breach(self) -> None:
        """The untyped batch path refuses it too, so both write paths agree."""
        assert entry_boundary_error(metadata={'project': ['x' * 100] * 40}) is not None
