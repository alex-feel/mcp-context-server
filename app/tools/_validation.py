"""Input validation shared by the write, read, and search tools.

Rejects PostgreSQL-unstorable text, oversized or too many tags, and indexed
metadata values the backend cannot store or index, and validates and normalizes
image attachments before any generation or database work starts.
"""

import base64
import logging
from typing import Literal
from typing import cast

from fastmcp.exceptions import ToolError

from app.errors import format_exception_message
from app.metadata_types import pg_indexed_cast_error
from app.metadata_types import pg_indexed_metadata_text
from app.metadata_types import unstorable_string_error
from app.models import MAX_IMAGES_PER_ENTRY
from app.models import MAX_INDEXED_METADATA_VALUE_LENGTH
from app.models import MAX_TAG_LENGTH
from app.models import MAX_TAGS_PER_ENTRY
from app.models import MAX_THREAD_ID_LENGTH
from app.models import normalize_base64_image_data
from app.settings import get_settings
from app.startup import MAX_IMAGE_SIZE_MB
from app.startup import MAX_TOTAL_SIZE_MB

logger = logging.getLogger(__name__)
settings = get_settings()


def reject_unstorable_input(**fields: object) -> None:
    """Raise ``ToolError`` if any user-supplied field carries a PostgreSQL-unstorable string.

    A ``thread_id``, ``text``, tag, or metadata string carrying an embedded NUL
    (U+0000) or an unpaired UTF-16 surrogate stores on SQLite but is rejected by
    PostgreSQL (TEXT bind or jsonb parse), so the same request diverges across
    backends and -- on the store/update write path -- the failure surfaces only
    after a wasted generation pass, inside the transaction where a
    non-ControlFlowError charges the circuit breaker. Rejecting at the tool
    boundary in the input-validation phase (before any generation or connection
    scope) fails both backends fast and identically with a clean client error and
    zero breaker charge, mirroring the ``non_finite_metadata_error`` guard.
    ``unstorable_string_error`` walks metadata dict keys/values and tag lists, so
    a scalar string and a nested structure are both validated.

    Args:
        **fields: Named user-supplied values to validate; the name is surfaced in
            the error message so the caller learns which field is at fault.

    Raises:
        ToolError: On the first field containing an unstorable string.
    """
    for name, value in fields.items():
        if value is None:
            continue
        message = unstorable_string_error(value)
        if message is not None:
            raise ToolError(f'{name}: {message}')


def tag_limits_error(tags: list[str] | None) -> str | None:
    """Return an error message when a tag list breaches a per-entry write cap.

    Two dimensions are checked, both of which the write path previously left
    unbounded:

    * COUNT -- every tag is a separate INSERT issued inside the open store /
      update transaction (on SQLite that is the single writer), and every later
      search response re-hydrates the entry's full tag list, so one oversized
      list stalls the write path once and inflates every subsequent read
      forever.
    * PER-TAG LENGTH -- ``idx_tags_tag`` is a PostgreSQL btree index whose
      index-tuple ceiling (~2704 bytes) rejects an oversized tag inside the
      transaction, AFTER a full generation pass and while charging the circuit
      breaker, while SQLite stores the same value happily. Capping at the tool
      boundary makes both backends accept and reject identically, and keeps the
      value migratable between them.

    This is the single source of truth shared by the typed single-entry tools
    (whose ``Field`` declarations advertise the same bounds in the MCP wire
    schema) and the untyped batch tools (whose per-entry lists never pass
    through a Pydantic model), mirroring how ``MAX_IMAGES_PER_ENTRY`` is
    enforced by :func:`validate_and_normalize_images`.

    Args:
        tags: The client-supplied tag list, or ``None`` when absent.

    Returns:
        An error message describing the first breach, or ``None`` when the list
        is within both caps.
    """
    if tags is None:
        return None
    if len(tags) > MAX_TAGS_PER_ENTRY:
        return f'Too many tags: {len(tags)} provided, maximum is {MAX_TAGS_PER_ENTRY} per entry'
    for idx, tag in enumerate(tags):
        if len(tag) > MAX_TAG_LENGTH:
            return (
                f'Tag {idx} is too long: {len(tag)} characters, maximum is '
                f'{MAX_TAG_LENGTH} characters per tag'
            )
    return None


def reject_oversized_tags(tags: list[str] | None) -> None:
    """Raise ``ToolError`` when a tag list breaches a per-entry write cap.

    The raising wrapper used by the single-entry tools; the batch tools call
    :func:`tag_limits_error` directly so they can record a per-entry failure
    instead of aborting the whole request.

    Args:
        tags: The client-supplied tag list, or ``None`` when absent.

    Raises:
        ToolError: When the list exceeds the count or per-tag length cap.
    """
    message = tag_limits_error(tags)
    if message is not None:
        raise ToolError(message)


def indexed_value_error(
    thread_id: str | None = None,
    metadata: object = None,
) -> str | None:
    """Return an error message when a client value cannot survive its index.

    Two dimensions, both of which make PostgreSQL reject INSIDE the store
    transaction -- after a full generation pass, and while charging the circuit
    breaker -- what SQLite stores happily:

    * LENGTH. ``tags`` is not the only write-path value PostgreSQL indexes with a
      btree whose index-tuple ceiling (~2704 bytes) rejects an oversized entry.
      ``thread_id`` feeds ``idx_thread_id``, ``idx_thread_source``,
      ``idx_context_entries_dedup_hash`` and ``idx_thread_created``; every value
      stored under a ``string``-typed ``METADATA_INDEXED_FIELDS`` key feeds that
      field's TEXT expression index ``idx_metadata_<field>`` on
      ``metadata->>'<field>'``. The length is measured on the text that expression
      YIELDS (:func:`pg_indexed_metadata_text`), not on ``str`` values alone: ``->>``
      renders a list or object as its whole serialized JSON, so a container under an
      indexed key is indexed at full width and an inspection restricted to strings
      lets it through to abort the INSERT.
    * CAST COMPATIBILITY. A ``METADATA_INDEXED_FIELDS`` entry configured with an
      ``integer``/``boolean``/``float`` type hint puts a hard SQL cast in that
      expression index, evaluated on every INSERT (see :func:`~app.metadata_types.pg_indexed_cast_error`).

    Values under NON-indexed metadata keys are deliberately not capped or type-checked:
    jsonb imposes no such limit and the always-present GIN index uses
    ``jsonb_path_ops``, which hashes its entries. The same reasoning exempts an
    ``array``/``object``-typed indexed field, which builds no expression index at all
    and is served by that GIN index, and an ``integer``/``boolean``/``float`` field,
    whose index datum is the fixed-width cast result rather than the source text.
    Only the top level of ``metadata`` is inspected, because ``metadata->>'<field>'``
    addresses top-level keys only.

    Args:
        thread_id: The client-supplied thread identifier, or None when absent.
        metadata: The client-supplied metadata mapping, or None when absent.

    Returns:
        An error message describing the first breach, or None when every indexed
        value is storable.
    """
    if thread_id is not None and len(thread_id) > MAX_THREAD_ID_LENGTH:
        return (
            f'thread_id is too long: {len(thread_id)} characters, maximum is '
            f'{MAX_THREAD_ID_LENGTH} characters'
        )
    if isinstance(metadata, dict):
        indexed_fields = settings.storage.metadata_indexed_fields
        for key, value in cast('dict[object, object]', metadata).items():
            if not isinstance(key, str) or key not in indexed_fields:
                continue
            if indexed_fields[key] == 'string':
                indexed_text = pg_indexed_metadata_text(value)
                if indexed_text is not None and len(indexed_text) > MAX_INDEXED_METADATA_VALUE_LENGTH:
                    return (
                        f'metadata field {key!r} is indexed and its value is too long: '
                        f'{len(indexed_text)} characters, maximum is '
                        f'{MAX_INDEXED_METADATA_VALUE_LENGTH} characters'
                    )
            cast_error = pg_indexed_cast_error(key, value, indexed_fields[key])
            if cast_error is not None:
                return cast_error
    return None


def entry_boundary_error(
    *,
    thread_id: str | None = None,
    text: str | None = None,
    tags: object = None,
    metadata: object = None,
    metadata_patch: object = None,
) -> str | None:
    """Return the first cross-backend boundary error for one client-supplied entry.

    The single chokepoint the UNTYPED batch paths use where the typed single-entry
    tools rely on their wire schema plus the individual raising guards. It bundles
    both families of "SQLite accepts it, PostgreSQL rejects it" input so the batch
    loops carry one call instead of a long boolean chain: PostgreSQL-unstorable
    strings (embedded NUL, unpaired UTF-16 surrogate) and the length caps plus cast
    compatibility of the values that land in a PostgreSQL btree index.

    Every argument is optional so a call site passes only the fields that shape
    exists (``update`` has no ``thread_id``; ``store`` has no ``metadata_patch``).
    Absent values are skipped, and a non-string / non-container value is ignored by
    the underlying checks rather than raising.

    Args:
        thread_id: The client-supplied thread identifier, when the shape has one.
        text: The client-supplied text content, when provided.
        tags: The client-supplied tag list, when provided.
        metadata: The client-supplied metadata mapping (full replacement).
        metadata_patch: The client-supplied merge-patch mapping, when provided.

    Returns:
        The first error message found, or None when the entry is acceptable on both
        backends.
    """
    return (
        unstorable_string_error(thread_id)
        or unstorable_string_error(text)
        or unstorable_string_error(tags)
        or unstorable_string_error(metadata)
        or unstorable_string_error(metadata_patch)
        or indexed_value_error(thread_id=thread_id, metadata=metadata)
        or indexed_value_error(metadata=metadata_patch)
    )


def reject_invalid_indexed_values(
    thread_id: str | None = None,
    metadata: object = None,
) -> None:
    """Raise ``ToolError`` when an indexed client value cannot survive its index.

    The raising wrapper used by the single-entry tools; the batch tools call
    :func:`indexed_value_error` directly so they can record a per-entry failure
    instead of aborting the whole request.

    Args:
        thread_id: The client-supplied thread identifier, or None when absent.
        metadata: The client-supplied metadata mapping, or None when absent.

    Raises:
        ToolError: When an indexed value exceeds its length cap or cannot be cast
            to its configured index type.
    """
    message = indexed_value_error(thread_id=thread_id, metadata=metadata)
    if message is not None:
        raise ToolError(message)


def validate_and_normalize_images(
    images: list[dict[str, str]] | None,
    *,
    error_mode: Literal['raise', 'collect'] = 'raise',
) -> tuple[list[dict[str, str]], Literal['text', 'multimodal'], list[str]]:
    """Validate and normalize image attachments.

    Performs all image validation steps:
    - Enforces the per-entry image count limit (MAX_IMAGES_PER_ENTRY)
    - Checks required 'data' field presence
    - Rejects a non-string 'data' or 'mime_type' value (the batch path is untyped)
    - Rejects empty/whitespace data (prevents silent 0-byte storage)
    - Defaults mime_type to 'image/png' when the key is absent
    - Normalizes each base64 payload to canonical standard-alphabet form
      (data-URI prefix stripped, ASCII whitespace removed, URL-safe alphabet
      translated, '=' padding restored) and REWRITES img['data'] in place to
      that canonical string, so every later re-decode (ImageRepository write
      paths) operates on the exact payload validated here
    - Decodes STRICTLY (base64.b64decode with validate=True), so genuinely
      non-base64 input fails loudly instead of being silently mangled
    - Enforces per-image size limit (MAX_IMAGE_SIZE_MB)
    - Enforces total size limit (MAX_TOTAL_SIZE_MB)
    - Rejects a per-image 'metadata' value that is neither absent/null nor a
      JSON-encoded string (the canonical per-image metadata wire shape). This
      subsumes the non-finite-float check entry metadata needs: only a bare
      float can serialize to the invalid-JSON NaN/Infinity tokens PostgreSQL's
      jsonb parser rejects, and a JSON-encoded string carries no bare float
    - Uses enumerate() for indexed error messages

    Args:
        images: List of image dicts with 'data' and optional 'mime_type' keys.
            None or empty list means no images.
        error_mode: 'raise' raises ToolError on first validation failure
            (for non-batch single-entry operations).
            'collect' accumulates errors and returns them
            (for batch operations where per-entry error reporting is needed).

    Returns:
        Tuple of (validated_images, content_type, errors):
        - validated_images: The validated image list (may have mime_type added)
        - content_type: 'multimodal' if images present, 'text' otherwise
        - errors: Empty list in 'raise' mode; list of error strings in 'collect' mode

    Raises:
        ToolError: In 'raise' mode, on the first validation failure.
    """
    if not images:
        return [], 'text', []

    errors: list[str] = []
    total_size: float = 0.0

    # Enforce the documented per-entry count limit here at the single shared
    # chokepoint so it covers store_context, update_context, AND both batch
    # tools (whose per-entry image lists never pass through the Pydantic
    # models). The tool-boundary Field declarations in app/tools/context/store.py
    # and app/tools/context/update.py advertise the same bound as maxItems in the
    # MCP wire schema.
    if len(images) > MAX_IMAGES_PER_ENTRY:
        msg = f'Too many images: {len(images)} provided, maximum is {MAX_IMAGES_PER_ENTRY} per entry'
        if error_mode == 'raise':
            raise ToolError(msg)
        errors.append(msg)
        return images, 'text', errors

    for idx, img in enumerate(images):
        # Validate required data field
        if 'data' not in img:
            msg = f'Image {idx} is missing required "data" field'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # The batch tools accept untyped list[dict[str, Any]] entries, so a value
        # that would be rejected by the single-entry Pydantic list[dict[str, str]]
        # schema (a JSON null or number) can reach here. Reject a non-string "data"
        # before .strip()/base64 decode rather than crashing with an opaque
        # AttributeError, keeping the two paths consistent.
        data_val = cast(object, img['data'])
        if not isinstance(data_val, str):
            msg = f'Image {idx} has a non-string "data" field'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors
        img_data_str = data_val
        if not img_data_str or not img_data_str.strip():
            msg = f'Image {idx} has empty "data" field'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # mime_type is optional and defaults to 'image/png' only when the key is
        # ABSENT. A PRESENT but non-string value (a JSON null/number from the
        # untyped batch path) must be rejected, not bound into the mime_type
        # TEXT NOT NULL column: SQLite would silently coerce a number to text
        # while PostgreSQL raises a DataError, and a null trips the NOT NULL
        # constraint and aborts an atomic batch. This mirrors the single-entry
        # Pydantic list[dict[str, str]] contract, which already rejects a
        # non-string mime_type at the tool boundary.
        if 'mime_type' not in img:
            img['mime_type'] = 'image/png'
        else:
            mime_val = cast(object, img['mime_type'])
            if not isinstance(mime_val, str):
                msg = f'Image {idx} has a non-string "mime_type" field'
                if error_mode == 'raise':
                    raise ToolError(msg)
                errors.append(msg)
                return images, 'text', errors
            # A mime_type STRING carrying an embedded NUL (U+0000) or an unpaired
            # UTF-16 surrogate binds into the image_attachments.mime_type TEXT NOT
            # NULL column inside the transaction -- SQLite stores it while
            # PostgreSQL raises a DataError AFTER a full generation pass, charging
            # the circuit breaker. The Pydantic list[dict[str, str]] contract on the
            # single-entry path enforces str TYPE but not this byte content, so the
            # guard must live here (shared by both paths), mirroring the per-image
            # metadata check below.
            mime_unstorable = unstorable_string_error(mime_val)
            if mime_unstorable is not None:
                msg = f'Image {idx} mime_type: {mime_unstorable}'
                if error_mode == 'raise':
                    raise ToolError(msg)
                errors.append(msg)
                return images, 'text', errors

        # Per-image 'metadata' crosses the boundary as a JSON-ENCODED STRING: the
        # typed single-entry tools declare images as list[dict[str, str]], the write
        # path json.dumps that already-stringified value, and the read path json.loads
        # it back to the same string. The untyped batch path (list[dict[str, Any]])
        # bypasses that contract, so a dict/number/list slips through and is stored in
        # a shape the single-entry tool would have refused -- and which then fails the
        # strict get_context_by_ids output schema, making the entry permanently
        # unreadable. Enforce the same shape here, at the chokepoint both paths share.
        # A JSON null is treated as "no metadata" (it stores as SQL NULL either way).
        metadata_value = cast(object, img.get('metadata'))
        if metadata_value is not None and not isinstance(metadata_value, str):
            msg = (
                f'Image {idx} metadata must be a JSON-encoded string '
                f'(got {type(metadata_value).__name__})'
            )
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # A per-image 'metadata' key or value carrying an embedded NUL (U+0000) or an
        # unpaired UTF-16 surrogate stores on SQLite but is rejected by PostgreSQL's jsonb
        # image_metadata column -- the same cross-backend divergence and breaker-charging
        # failure unstorable_string_error guards for entry metadata, applied here for the
        # untyped batch path's per-image metadata.
        unstorable_error = unstorable_string_error(cast(object, img.get('metadata')))
        if unstorable_error is not None:
            msg = f'Image {idx} metadata: {unstorable_error}'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # Normalize the payload to canonical standard-alphabet base64 BEFORE
        # decoding: strip one RFC 2397 data-URI prefix, remove ASCII whitespace,
        # translate the URL-safe alphabet, restore '=' padding. A lenient
        # b64decode previously accepted these shapes and silently corrupted the
        # bytes (a data-URI prefix whose base64-alphabet length is a multiple of
        # 4 decodes as garbage prepended to the image; '-'/'_' were discarded,
        # shifting every following byte). The STRICT decode (validate=True) then
        # either yields exactly the intended bytes or fails loudly per image.
        normalized_data = normalize_base64_image_data(img_data_str)
        try:
            image_binary = base64.b64decode(normalized_data, validate=True)
        except Exception as e:
            if error_mode == 'raise':
                raise ToolError(
                    f'Image {idx} has invalid base64 encoding: Invalid base64 data ({format_exception_message(e)})',
                ) from None
            errors.append(f'Image {idx} has invalid base64 encoding')
            return images, 'text', errors

        # Rewrite the payload in place to the canonical string so the write-path
        # re-decode in ImageRepository (strict as well) operates on the exact
        # payload validated here. The batch tools rely on this in-place mutation:
        # they discard the returned list and later pass the same dict objects to
        # the transaction helpers.
        img['data'] = normalized_data

        # A payload that normalizes to the empty string (e.g. a bare data-URI
        # prefix with nothing after the comma) decodes to zero bytes and is not
        # a real image; reject it rather than storing a 0-byte attachment.
        # Non-alphabet garbage no longer reaches this guard -- the strict decode
        # above rejects it loudly.
        if not image_binary:
            msg = f'Image {idx} "data" decodes to zero bytes (not valid base64 image content)'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        # Validate image size
        image_size_mb = len(image_binary) / (1024 * 1024)

        if image_size_mb > MAX_IMAGE_SIZE_MB:
            msg = f'Image {idx} exceeds {MAX_IMAGE_SIZE_MB}MB limit'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

        total_size += image_size_mb
        if total_size > MAX_TOTAL_SIZE_MB:
            msg = f'Total image size exceeds {MAX_TOTAL_SIZE_MB}MB limit'
            if error_mode == 'raise':
                raise ToolError(msg)
            errors.append(msg)
            return images, 'text', errors

    logger.debug(f'Pre-validation passed for {len(images)} images, total size: {total_size:.2f}MB')
    return images, 'multimodal', []
