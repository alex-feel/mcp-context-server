"""Tests for the reserved ingest metadata block: the protocol constants and the block validator.

The block validator checks shape only and reports the first problem with a fixed
text; ``ingest_metadata_error`` applies it to the two write forms, full metadata
and a merge patch, where a patch may also remove the block with ``null``.
"""

import ast
import sys
from pathlib import Path

import pytest

import app.ingest
from app.ingest import DEFAULT_INGEST_STATE_PAGE
from app.ingest import DIGEST_PATTERN
from app.ingest import INGEST_FIELDS
from app.ingest import INGEST_KEY
from app.ingest import MAX_INGEST_STATE_PAGE
from app.ingest import MAX_MANIFEST_ITEMS
from app.ingest import MAX_PLAN_DELETES
from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import ingest_block_error
from app.models import MAX_THREAD_ID_LENGTH
from app.tools._validation import ingest_metadata_error

_DIGEST = '0123456789abcdef' * 4
_SHAPE_ERROR = 'metadata.ingest must be an object with exactly the keys meta_digest, scope, source_id, synced_at'
_PATCH_SHAPE_ERROR = (
    'metadata_patch.ingest must be null or an object with exactly the keys meta_digest, scope, source_id, synced_at'
)
_SYNCED_AT_ERROR = 'metadata.ingest.synced_at must be a UTC timestamp such as 2026-10-18T12:00:00Z'
_DIGEST_ERROR = 'metadata.ingest.meta_digest must be 64 lowercase hexadecimal characters'
# U+1D11E MUSICAL SYMBOL G CLEF: one code point, two UTF-16 units, four UTF-8 bytes.
_ASTRAL = '\U0001d11e'


def _block(**overrides: object) -> dict[str, object]:
    """Return a valid block with the given fields replaced."""
    block: dict[str, object] = {
        'scope': 'kb:handbook',
        'source_id': 'guides/setup.md',
        'meta_digest': _DIGEST,
        'synced_at': '2026-10-18T12:00:00Z',
    }
    block.update(overrides)
    return block


class TestProtocolConstants:
    """The protocol constants the server and the sync client share."""

    def test_values(self) -> None:
        """Each constant carries the agreed value."""
        assert INGEST_KEY == 'ingest'
        assert frozenset({'scope', 'source_id', 'meta_digest', 'synced_at'}) == INGEST_FIELDS
        assert MAX_SOURCE_ID_LENGTH == 256
        assert MAX_MANIFEST_ITEMS == 1000
        assert MAX_PLAN_DELETES == 100
        assert MAX_INGEST_STATE_PAGE == 200
        assert DEFAULT_INGEST_STATE_PAGE == 100
        assert DIGEST_PATTERN == '^[0-9a-f]{64}$'

    def test_scope_bound_is_the_thread_id_bound(self) -> None:
        """A scope names a thread, so it is bounded like a thread_id."""
        assert MAX_THREAD_ID_LENGTH == 256


class TestValidBlocks:
    """Well-formed blocks pass."""

    def test_minimal_valid_block(self) -> None:
        """A block with the four fields in range passes."""
        assert ingest_block_error(_block()) is None

    @pytest.mark.parametrize('synced_at', ['2026-10-18T12:00:00.5Z', '2026-10-18T12:00:00.123456Z', '2024-02-29T23:59:59Z'])
    def test_fractional_seconds_and_leap_day(self, synced_at: str) -> None:
        """One to six fraction digits and a real leap day are accepted."""
        assert ingest_block_error(_block(synced_at=synced_at)) is None

    def test_bounds_are_inclusive(self) -> None:
        """Values of exactly 256 code points pass, astral characters counting once."""
        assert ingest_block_error(_block(scope=_ASTRAL * 256, source_id=_ASTRAL * 256)) is None

    def test_scope_need_not_name_the_entry_thread(self) -> None:
        """The validator checks shape only; membership is decided when the block is read."""
        assert ingest_block_error(_block(scope='kb:another-corpus')) is None


class TestShape:
    """The block is an object with exactly the four keys."""

    @pytest.mark.parametrize('value', [None, 'kb:handbook', ['scope'], 42, True])
    def test_non_objects_refused(self, value: object) -> None:
        """Anything but an object is refused with the shape text."""
        assert ingest_block_error(value) == _SHAPE_ERROR

    @pytest.mark.parametrize('missing', sorted(INGEST_FIELDS))
    def test_missing_key_refused(self, missing: str) -> None:
        """A block lacking any one key is refused."""
        block = _block()
        del block[missing]
        assert ingest_block_error(block) == _SHAPE_ERROR

    @pytest.mark.parametrize('extra', ['text_digest', 'owner_id', 'visibility'])
    def test_extra_key_refused(self, extra: str) -> None:
        """A block carrying any other key is refused, access-looking keys included."""
        assert ingest_block_error(_block(**{extra: 'x'})) == _SHAPE_ERROR

    def test_empty_object_refused(self) -> None:
        """An empty object is refused."""
        assert ingest_block_error({}) == _SHAPE_ERROR


class TestStringFields:
    """scope and source_id are non-empty strings of at most 256 code points."""

    @pytest.mark.parametrize('field', ['scope', 'source_id'])
    @pytest.mark.parametrize('value', [None, 7, ['a'], {'a': 1}, False])
    def test_non_string_refused(self, field: str, value: object) -> None:
        """A non-string value is refused with the field's type text."""
        assert ingest_block_error(_block(**{field: value})) == f'metadata.ingest.{field} must be a string'

    @pytest.mark.parametrize('field', ['scope', 'source_id'])
    def test_empty_refused(self, field: str) -> None:
        """An empty string is refused."""
        assert ingest_block_error(_block(**{field: ''})) == f'metadata.ingest.{field} cannot be empty'

    @pytest.mark.parametrize('field', ['scope', 'source_id'])
    @pytest.mark.parametrize('value', ['a' * 257, _ASTRAL * 257])
    def test_too_long_refused(self, field: str, value: str) -> None:
        """257 code points are refused, counted as code points."""
        expected = f'metadata.ingest.{field} is too long: 257 characters, maximum is 256'
        assert ingest_block_error(_block(**{field: value})) == expected


class TestMetaDigest:
    """meta_digest is exactly 64 lowercase hexadecimal characters."""

    @pytest.mark.parametrize(
        'value',
        [
            _DIGEST.upper(),
            _DIGEST[:63],
            _DIGEST + '0',
            _DIGEST + '\n',
            '\n' + _DIGEST,
            _DIGEST[:63] + 'g',
            '',
            None,
            12,
        ],
        ids=['uppercase', 'short', 'long', 'trailing-newline', 'leading-newline', 'non-hex', 'empty', 'null', 'number'],
    )
    def test_refused(self, value: object) -> None:
        """Anything but 64 lowercase hex characters is refused."""
        assert ingest_block_error(_block(meta_digest=value)) == _DIGEST_ERROR


class TestSyncedAt:
    """synced_at is a UTC timestamp with ASCII digits that parses as a real instant."""

    @pytest.mark.parametrize(
        'value',
        [
            '2026-10-18T12:00:00',
            '2026-10-18T12:00:00+00:00',
            '2026-10-18 12:00:00Z',
            '2026-10-18T12:00Z',
            '2026-10-18T12:00:00.1234567Z',
            '2026-10-18T12:00:00.Z',
            '2026-10-18T12:00:00Z\n',
            '2026-10-18t12:00:00z',
            '٢٠٢٦-10-18T12:00:00Z',
            '２０２６-10-18T12:00:00Z',
            '2026-02-30T12:00:00Z',
            '2025-02-29T12:00:00Z',
            '2026-13-01T12:00:00Z',
            '2026-10-18T24:00:00Z',
            '2026-10-18T23:59:60Z',
            '',
            None,
            1760788800,
        ],
        ids=[
            'no-zone',
            'offset',
            'space-separator',
            'no-seconds',
            'seven-fraction-digits',
            'empty-fraction',
            'trailing-newline',
            'lowercase-designators',
            'arabic-indic-digits',
            'fullwidth-digits',
            'february-30',
            'february-29-common-year',
            'month-13',
            'hour-24',
            'second-60',
            'empty',
            'null',
            'number',
        ],
    )
    def test_refused(self, value: object) -> None:
        """Every other form, an impossible date, and non-ASCII digits are refused."""
        assert ingest_block_error(_block(synced_at=value)) == _SYNCED_AT_ERROR


class TestErrorOrder:
    """The first problem in a fixed order is reported."""

    def test_type_checks_precede_empty_checks(self) -> None:
        """A non-string source_id is reported before an empty scope."""
        assert ingest_block_error(_block(scope='', source_id=3)) == 'metadata.ingest.source_id must be a string'

    def test_empty_checks_precede_length_checks(self) -> None:
        """An empty source_id is reported before an over-long scope."""
        assert ingest_block_error(_block(scope='a' * 300, source_id='')) == 'metadata.ingest.source_id cannot be empty'

    def test_scope_precedes_source_id(self) -> None:
        """When both strings are over-long, scope is reported."""
        expected = 'metadata.ingest.scope is too long: 300 characters, maximum is 256'
        assert ingest_block_error(_block(scope='a' * 300, source_id='b' * 300)) == expected

    def test_strings_precede_the_digest_and_the_timestamp(self) -> None:
        """The string fields are checked before meta_digest, and meta_digest before synced_at."""
        assert ingest_block_error(_block(scope=1, meta_digest='x', synced_at='x')) == 'metadata.ingest.scope must be a string'
        assert ingest_block_error(_block(meta_digest='x', synced_at='x')) == _DIGEST_ERROR

    def test_path_prefixes_every_message(self) -> None:
        """The path argument names the field in every message."""
        assert ingest_block_error(_block(scope=''), path='metadata_patch.ingest') == (
            'metadata_patch.ingest.scope cannot be empty'
        )
        assert ingest_block_error([], path='metadata_patch.ingest') == (
            'metadata_patch.ingest must be an object with exactly the keys meta_digest, scope, source_id, synced_at'
        )


class TestIngestMetadataError:
    """The write-boundary check applies the validator to full metadata and to a merge patch."""

    def test_metadata_without_the_block_passes(self) -> None:
        """Metadata that does not use the reserved key is not inspected."""
        assert ingest_metadata_error(metadata={'status': 'done', 'nested': {'ingest': 1}}) is None

    def test_absent_and_non_object_values_pass(self) -> None:
        """Absent values and non-object values are left to the other boundary checks."""
        assert ingest_metadata_error() is None
        assert ingest_metadata_error(metadata=['ingest'], metadata_patch='ingest') is None

    def test_valid_block_in_metadata_passes(self) -> None:
        """A complete valid block in full metadata passes."""
        assert ingest_metadata_error(metadata={'title': 'Setup', INGEST_KEY: _block()}) is None

    def test_invalid_block_in_metadata_refused(self) -> None:
        """A malformed block in full metadata is refused under metadata.ingest."""
        assert ingest_metadata_error(metadata={INGEST_KEY: _block(meta_digest='ABC')}) == _DIGEST_ERROR

    def test_null_block_in_metadata_refused(self) -> None:
        """Full metadata cannot carry a null block; omitting the key is how it holds none."""
        assert ingest_metadata_error(metadata={INGEST_KEY: None}) == _SHAPE_ERROR

    def test_null_block_in_patch_passes(self) -> None:
        """A patch may remove the block with null."""
        assert ingest_metadata_error(metadata_patch={INGEST_KEY: None}) is None

    def test_complete_block_in_patch_passes(self) -> None:
        """A patch may replace the block with a complete valid one."""
        assert ingest_metadata_error(metadata_patch={INGEST_KEY: _block(), 'title': 'New'}) is None

    @pytest.mark.parametrize(
        'value',
        [{'synced_at': '2026-10-19T08:00:00Z'}, {'scope': 'kb:handbook', 'source_id': 'a.md'}, 'kb:handbook', []],
        ids=['one-field', 'two-fields', 'string', 'list'],
    )
    def test_partial_or_non_object_block_in_patch_refused(self, value: object) -> None:
        """A partial block in a patch is refused, because the merge would leave a block of mixed origin."""
        assert ingest_metadata_error(metadata_patch={INGEST_KEY: value}) == _PATCH_SHAPE_ERROR

    def test_invalid_field_in_patch_block_refused(self) -> None:
        """A complete patch block with an invalid field is refused under metadata_patch.ingest."""
        assert ingest_metadata_error(metadata_patch={INGEST_KEY: _block(source_id='')}) == (
            'metadata_patch.ingest.source_id cannot be empty'
        )


def test_module_imports_only_the_standard_library_pydantic_and_models() -> None:
    """app.ingest is shared with the ingest client, so it never imports settings, backends or tools."""
    module_file = app.ingest.__file__
    assert module_file is not None
    tree = ast.parse(Path(module_file).read_text(encoding='utf-8'))
    third_party: set[str] = set()
    app_modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert node.level == 0
            names = [node.module]
        else:
            continue
        for name in names:
            if name.partition('.')[0] == 'app':
                app_modules.add(name)
            elif name.partition('.')[0] not in sys.stdlib_module_names:
                third_party.add(name.partition('.')[0])

    assert third_party <= {'pydantic'}
    assert app_modules <= {'app.models'}
