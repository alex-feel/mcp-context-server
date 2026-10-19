"""Text normalization and the digests the server and the corpus sync client compare.

The server stores a text after :func:`normalize_stored_text` and records
:func:`content_hash` of the stored text; a sync client that hashes its text the
same way can tell whether a stored entry still carries it without fetching the
text. :func:`canonical_json_digest` hashes a JSON value independently of key
order, for digests a client computes over structured data such as metadata.

The module imports only the standard library, so the server and the client
share it without pulling in either side's dependencies.
"""

import hashlib
import json


def normalize_stored_text(text: str) -> str:
    """Return the text as the server stores and hashes it.

    Args:
        text: The text as sent by a client.

    Returns:
        The text without leading and trailing whitespace.
    """
    return text.strip()


def content_hash(text: str) -> str:
    """Return the SHA-256 hex digest of a text's UTF-8 encoding.

    The deduplicating store compares this digest instead of the full text, and a
    sync plan compares it with the client's digest of the text it would send.

    Args:
        text: The text to hash, already normalized by :func:`normalize_stored_text`
            when it is a stored text.

    Returns:
        64 lowercase hexadecimal characters.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def canonical_json_digest(value: object) -> str:
    """Return the SHA-256 hex digest of a JSON value in canonical form.

    The canonical form sorts object keys at every depth, omits insignificant
    whitespace and keeps non-ASCII characters as UTF-8, so two values with the
    same members in any key order give the same digest. A value holding a NaN or
    an infinity has no JSON form, so serializing it raises ``ValueError`` instead
    of producing a digest.

    Args:
        value: A JSON-serializable value.

    Returns:
        64 lowercase hexadecimal characters.
    """
    serialized = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
