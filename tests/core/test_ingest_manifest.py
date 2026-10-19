"""Tests for the sync manifest item and the response types of the corpus sync protocol.

A manifest item either describes a source document by its two digests or, as a
keep item, says the client could not read the source in this run. The model
refuses every other combination, misspelled keys, malformed digests and source
ids longer than the bound the stored block carries, whatever the item's form.
"""

import typing
from typing import Literal

import pytest
from pydantic import ValidationError

from app.ingest import MAX_SOURCE_ID_LENGTH
from app.ingest import IngestStateEntryDict
from app.ingest import ListIngestStateDict
from app.ingest import PlanContextSyncDict
from app.ingest import SyncDeleteDict
from app.ingest import SyncManifestItem
from app.ingest import SyncUpdateDict

_TEXT = '0123456789abcdef' * 4
_META = 'fedcba9876543210' * 4
_KEEP_WITH_DIGEST = 'a keep item carries no text_digest or meta_digest'
_MISSING_DIGEST = 'an item without keep requires both text_digest and meta_digest'
# U+1D11E MUSICAL SYMBOL G CLEF: one code point, two UTF-16 units, four UTF-8 bytes.
_ASTRAL = '\U0001d11e'


def _single_error(exc_info: pytest.ExceptionInfo[ValidationError]) -> tuple[str, tuple[str | int, ...], str]:
    """Return the type, location and message of the only validation error raised."""
    errors = exc_info.value.errors()
    assert len(errors) == 1
    error = errors[0]
    return error['type'], tuple(error['loc']), error['msg']


class TestOrdinaryItems:
    """An item without keep carries the source id and both digests."""

    def test_valid_item(self) -> None:
        """Both digests given: an ordinary item that is not a keep item."""
        item = SyncManifestItem(source_id='guides/setup.md', text_digest=_TEXT, meta_digest=_META)

        assert item.source_id == 'guides/setup.md'
        assert item.text_digest == _TEXT
        assert item.meta_digest == _META
        assert item.keep is False

    def test_explicit_keep_false(self) -> None:
        """keep=false with both digests is the same ordinary item."""
        item = SyncManifestItem.model_validate(
            {'source_id': 'a.md', 'text_digest': _TEXT, 'meta_digest': _META, 'keep': False},
        )

        assert item.keep is False

    def test_parsed_from_json(self) -> None:
        """The item parses from the JSON a client sends."""
        item = SyncManifestItem.model_validate_json(
            f'{{"source_id": "a.md", "text_digest": "{_TEXT}", "meta_digest": "{_META}"}}',
        )

        assert item == SyncManifestItem(source_id='a.md', text_digest=_TEXT, meta_digest=_META)

    @pytest.mark.parametrize(
        'fields',
        [
            {'meta_digest': _META},
            {'text_digest': _TEXT},
            {},
            {'text_digest': None, 'meta_digest': _META},
        ],
        ids=['text-digest-missing', 'meta-digest-missing', 'both-missing', 'text-digest-null'],
    )
    def test_missing_digest_refused(self, fields: dict[str, object]) -> None:
        """An item that is not a keep item must carry both digests."""
        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate({'source_id': 'a.md', **fields})

        error_type, location, message = _single_error(exc_info)
        assert error_type == 'value_error'
        assert location == ()
        assert message == f'Value error, {_MISSING_DIGEST}'


class TestKeepItems:
    """A keep item names a source the client could not read and carries no digest."""

    def test_valid_keep_item(self) -> None:
        """source_id and keep=true alone form a keep item."""
        item = SyncManifestItem.model_validate({'source_id': 'unreadable.md', 'keep': True})

        assert item.keep is True
        assert item.text_digest is None
        assert item.meta_digest is None

    @pytest.mark.parametrize(
        'fields',
        [
            {'text_digest': _TEXT},
            {'meta_digest': _META},
            {'text_digest': _TEXT, 'meta_digest': _META},
        ],
        ids=['text-digest', 'meta-digest', 'both-digests'],
    )
    def test_keep_item_with_digest_refused(self, fields: dict[str, object]) -> None:
        """A keep item says the source could not be read, so it cannot describe its content."""
        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate({'source_id': 'a.md', 'keep': True, **fields})

        error_type, location, message = _single_error(exc_info)
        assert error_type == 'value_error'
        assert location == ()
        assert message == f'Value error, {_KEEP_WITH_DIGEST}'


class TestUnknownKeys:
    """A misspelled key is an error, never a silently ordinary item."""

    def test_misspelled_keep_refused(self) -> None:
        """'kep' instead of 'keep' is refused rather than read as an item without keep."""
        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate({'source_id': 'a.md', 'kep': True})

        errors = exc_info.value.errors()
        assert [(error['type'], tuple(error['loc'])) for error in errors if error['type'] == 'extra_forbidden'] == [
            ('extra_forbidden', ('kep',)),
        ]

    def test_extra_key_on_an_ordinary_item_refused(self) -> None:
        """A complete ordinary item with one more key is refused."""
        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate(
                {'source_id': 'a.md', 'text_digest': _TEXT, 'meta_digest': _META, 'synced_at': '2026-10-18T12:00:00Z'},
            )

        error_type, location, _ = _single_error(exc_info)
        assert error_type == 'extra_forbidden'
        assert location == ('synced_at',)


class TestDigestPattern:
    """Both digests are exactly 64 lowercase hexadecimal characters."""

    @pytest.mark.parametrize(
        'value',
        [
            _TEXT.upper(),
            _TEXT[:63],
            _TEXT + '0',
            _TEXT + '\n',
            'g' * 64,
            '',
        ],
        ids=['uppercase', 'short', 'long', 'trailing-newline', 'non-hex', 'empty'],
    )
    @pytest.mark.parametrize('field', ['text_digest', 'meta_digest'])
    def test_malformed_digest_refused(self, field: str, value: str) -> None:
        """A digest that is not 64 lowercase hex characters fails the pattern."""
        fields: dict[str, object] = {'text_digest': _TEXT, 'meta_digest': _META, field: value}

        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate({'source_id': 'a.md', **fields})

        error_type, location, _ = _single_error(exc_info)
        assert error_type == 'string_pattern_mismatch'
        assert location == (field,)

    @pytest.mark.parametrize('field', ['text_digest', 'meta_digest'])
    def test_non_string_digest_refused(self, field: str) -> None:
        """A number is not a digest."""
        fields: dict[str, object] = {'text_digest': _TEXT, 'meta_digest': _META, field: 12345}

        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate({'source_id': 'a.md', **fields})

        error_type, location, _ = _single_error(exc_info)
        assert error_type == 'string_type'
        assert location == (field,)


class TestSourceIdBound:
    """source_id is 1 to 256 code points on every item, keep items included."""

    @pytest.mark.parametrize(
        'value',
        ['x' * MAX_SOURCE_ID_LENGTH, _ASTRAL * MAX_SOURCE_ID_LENGTH, 'a'],
        ids=['ascii-at-bound', 'astral-at-bound', 'one-character'],
    )
    def test_within_bound_accepted(self, value: str) -> None:
        """The bound counts code points and is inclusive."""
        ordinary = SyncManifestItem(source_id=value, text_digest=_TEXT, meta_digest=_META)
        keep = SyncManifestItem(source_id=value, keep=True)

        assert ordinary.source_id == value
        assert keep.source_id == value

    @pytest.mark.parametrize(
        'value',
        ['x' * (MAX_SOURCE_ID_LENGTH + 1), _ASTRAL * (MAX_SOURCE_ID_LENGTH + 1)],
        ids=['ascii', 'astral'],
    )
    @pytest.mark.parametrize(
        'fields',
        [{'text_digest': _TEXT, 'meta_digest': _META}, {'keep': True}],
        ids=['ordinary-item', 'keep-item'],
    )
    def test_over_bound_refused(self, fields: dict[str, object], value: str) -> None:
        """257 code points are refused whatever the item's form."""
        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate({'source_id': value, **fields})

        error_type, location, _ = _single_error(exc_info)
        assert error_type == 'string_too_long'
        assert location == ('source_id',)

    def test_empty_refused(self) -> None:
        """An empty source id names no document."""
        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem(source_id='', keep=True)

        error_type, location, _ = _single_error(exc_info)
        assert error_type == 'string_too_short'
        assert location == ('source_id',)

    def test_missing_refused(self) -> None:
        """Every item names its source."""
        with pytest.raises(ValidationError) as exc_info:
            SyncManifestItem.model_validate({'keep': True})

        error_type, location, _ = _single_error(exc_info)
        assert error_type == 'missing'
        assert location == ('source_id',)


class TestResponseTypes:
    """The response shapes the sync tools return and the client reads."""

    @pytest.mark.parametrize(
        ('typed_dict', 'expected'),
        [
            (SyncUpdateDict, {'index': int, 'context_id': str, 'change': Literal['text', 'metadata', 'both']}),
            (SyncDeleteDict, {'context_id': str, 'source_id': str, 'reason': Literal['absent', 'duplicate']}),
            (
                PlanContextSyncDict,
                {
                    'thread_id': str,
                    'creates': list[int],
                    'updates': list[SyncUpdateDict],
                    'deletes': list[SyncDeleteDict],
                    'deletes_total': int,
                    'unchanged_count': int,
                    'kept_count': int,
                    'foreign_count': int,
                },
            ),
            (
                IngestStateEntryDict,
                {'context_id': str, 'source_id': str, 'text_digest': str | None, 'meta_digest': str, 'synced_at': str},
            ),
            (
                ListIngestStateDict,
                {'thread_id': str, 'entries': list[IngestStateEntryDict], 'count': int, 'next_cursor': str | None},
            ),
        ],
        ids=['SyncUpdateDict', 'SyncDeleteDict', 'PlanContextSyncDict', 'IngestStateEntryDict', 'ListIngestStateDict'],
    )
    def test_shape(self, typed_dict: type, expected: dict[str, object]) -> None:
        """Each type has exactly these keys and types, every key required."""
        assert typing.get_type_hints(typed_dict) == expected
        assert typing.is_typeddict(typed_dict)
        assert vars(typed_dict)['__required_keys__'] == frozenset(expected)
        assert vars(typed_dict)['__optional_keys__'] == frozenset()

    def test_no_type_reports_owner_or_visibility(self) -> None:
        """Responses never say who owns an entry or how visible it is."""
        for typed_dict in (SyncUpdateDict, SyncDeleteDict, PlanContextSyncDict, IngestStateEntryDict, ListIngestStateDict):
            keys = set(typing.get_type_hints(typed_dict))
            assert not keys & {'owner_id', 'owner', 'visibility', 'is_owner', 'owned'}
