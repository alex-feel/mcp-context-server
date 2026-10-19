"""Tests for app/digests.py: stored-text normalization, the content hash and the canonical JSON digest."""

import ast
import hashlib
import json
import sys
from pathlib import Path

import pytest

import app.digests
from app.digests import canonical_json_digest
from app.digests import content_hash
from app.digests import normalize_stored_text


class TestNormalizeStoredText:
    """normalize_stored_text applies the strip the server stores and hashes."""

    def test_strips_surrounding_whitespace(self) -> None:
        """Leading and trailing whitespace, newlines included, is removed."""
        assert normalize_stored_text('  \n\tbody text\r\n  ') == 'body text'

    def test_keeps_inner_whitespace(self) -> None:
        """Whitespace between words and inner line breaks stay as they are."""
        assert normalize_stored_text(' line one\n\nline  two ') == 'line one\n\nline  two'

    def test_whitespace_only_text_becomes_empty(self) -> None:
        """A text of whitespace alone normalizes to the empty string."""
        assert normalize_stored_text(' \n\t ') == ''


class TestContentHash:
    """content_hash is the SHA-256 hex digest of the UTF-8 text."""

    def test_consistent_results(self) -> None:
        """Same input always produces same hash."""
        text = 'Hello, World!'
        assert content_hash(text) == content_hash(text)

    def test_different_inputs_produce_different_hashes(self) -> None:
        """Different inputs produce different hashes."""
        assert content_hash('text A') != content_hash('text B')

    def test_returns_hex_string(self) -> None:
        """Hash is a 64-character lowercase hex string (SHA-256)."""
        result = content_hash('test')
        assert len(result) == 64
        assert all(c in '0123456789abcdef' for c in result)

    def test_empty_string(self) -> None:
        """Empty string produces the well-known SHA-256 of no bytes."""
        assert content_hash('') == 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'

    def test_unicode_content(self) -> None:
        """Non-ASCII text hashes its UTF-8 encoding."""
        text = 'Привет мир 🌍'
        assert content_hash(text) == hashlib.sha256(text.encode('utf-8')).hexdigest()

    def test_long_content(self) -> None:
        """Long content produces same-length hash."""
        assert len(content_hash('A' * 100_000)) == 64

    def test_hash_of_normalized_text_matches_the_stored_hash_form(self) -> None:
        """A manifest text digest equals the hash of the text as the server stores it."""
        sent = '\n# Title\n\nBody\n'
        assert content_hash(normalize_stored_text(sent)) == content_hash('# Title\n\nBody')


class TestCanonicalJsonDigest:
    """canonical_json_digest hashes sorted, compact, non-ASCII-preserving JSON."""

    def test_formula(self) -> None:
        """The digest is the SHA-256 of the canonical serialization."""
        value = {'b': 1, 'a': [True, None, 2.5]}
        expected = hashlib.sha256(b'{"a":[true,null,2.5],"b":1}').hexdigest()
        assert canonical_json_digest(value) == expected

    def test_key_order_does_not_matter_at_any_depth(self) -> None:
        """Nested objects with the same members in another order give the same digest."""
        first = {'metadata': {'title': 'Setup', 'meta': {'x': 1, 'y': 2}}, 'tags': ['a'], 'visibility': 'private'}
        second = {'visibility': 'private', 'tags': ['a'], 'metadata': {'meta': {'y': 2, 'x': 1}, 'title': 'Setup'}}
        assert canonical_json_digest(first) == canonical_json_digest(second)

    def test_non_ascii_is_serialized_as_utf8(self) -> None:
        """Non-ASCII characters are encoded as UTF-8, not as escape sequences."""
        value = {'title': 'Привет', 'emoji': '🌍'}
        serialized = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
        assert '\\u' not in serialized
        assert canonical_json_digest(value) == hashlib.sha256(serialized.encode('utf-8')).hexdigest()

    def test_list_order_matters(self) -> None:
        """Arrays keep their order, so a reordered array is a different value."""
        assert canonical_json_digest({'tags': ['a', 'b']}) != canonical_json_digest({'tags': ['b', 'a']})

    @pytest.mark.parametrize('number', [float('nan'), float('inf'), float('-inf')])
    def test_non_finite_numbers_are_refused(self, number: float) -> None:
        """A value JSON cannot represent raises instead of producing a digest."""
        with pytest.raises(ValueError, match='JSON compliant'):
            canonical_json_digest({'score': number})

    def test_returns_lowercase_hex(self) -> None:
        """The digest is 64 lowercase hexadecimal characters."""
        digest = canonical_json_digest({})
        assert len(digest) == 64
        assert all(c in '0123456789abcdef' for c in digest)


def test_module_imports_only_the_standard_library() -> None:
    """app.digests is shared by the server and the ingest client, so it imports nothing outside the standard library."""
    module_file = app.digests.__file__
    assert module_file is not None
    tree = ast.parse(Path(module_file).read_text(encoding='utf-8'))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.partition('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.module is not None
            assert node.level == 0
            imported.add(node.module.partition('.')[0])

    assert imported
    assert imported <= sys.stdlib_module_names
