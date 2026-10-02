"""Rewrite of ``metadata.references.context_ids`` through the integer-to-UUIDv7 mapping."""

import json
from collections.abc import Mapping
from typing import cast

from app.cli.migrate_uuid.records import MigrationStats


def _rewrite_context_ids_list(
    items: list[object],
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    row_pk: int,
) -> list[object]:
    """Rewrite a single ``context_ids`` list.

    Integer entries are remapped to UUIDv7 hex strings via
    ``id_mapping``. Strings are preserved unchanged. Booleans and other
    types are flagged as malformed but preserved.

    Args:
        items: The list pulled from ``references.context_ids``.
        id_mapping: Integer-to-UUIDv7 mapping.
        stats: Mutated to count rewrites, orphans, and malformed entries.
        row_pk: The source row's integer ID, used for log context.

    Returns:
        A new list with integers remapped where possible.
    """
    out: list[object] = []
    for element in items:
        if isinstance(element, bool):
            stats.malformed_references += 1
            stats.errors.append(
                f'row {row_pk}: references.context_ids contains a boolean entry; preserved unchanged',
            )
            out.append(element)
            continue
        if isinstance(element, int):
            mapped = id_mapping.get(element)
            if mapped is not None:
                stats.references_rewritten += 1
                out.append(mapped)
            else:
                stats.orphan_references += 1
                stats.warnings.append(
                    f'row {row_pk}: references.context_ids contains orphan integer {element}; preserved',
                )
                out.append(element)
            continue
        if isinstance(element, str):
            out.append(element)
            continue
        stats.malformed_references += 1
        stats.errors.append(
            f'row {row_pk}: references.context_ids contains non-int/non-str entry '
            f'{type(element).__name__}; preserved',
        )
        out.append(element)
    return out


def _walk_and_rewrite(
    node: object,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    row_pk: int,
    seen: set[int],
) -> None:
    """Recursively walk ``node`` and rewrite any references.context_ids.

    Dictionaries and lists are mutated in place. A ``seen`` set of object
    ids prevents infinite recursion on self-referential structures.
    """
    obj_id = id(node)
    if obj_id in seen:
        return
    if isinstance(node, dict):
        seen.add(obj_id)
        typed_node = cast(dict[str, object], node)
        references = typed_node.get('references')
        if isinstance(references, dict):
            typed_refs = cast(dict[str, object], references)
            context_ids_value = typed_refs.get('context_ids')
            if isinstance(context_ids_value, list):
                typed_refs['context_ids'] = _rewrite_context_ids_list(
                    cast(list[object], context_ids_value),
                    id_mapping,
                    stats,
                    row_pk,
                )
            elif context_ids_value is not None:
                stats.malformed_references += 1
                stats.errors.append(
                    f'row {row_pk}: metadata.references.context_ids is not a list '
                    f'({type(context_ids_value).__name__}); preserved',
                )
        for value in typed_node.values():
            _walk_and_rewrite(value, id_mapping, stats, row_pk, seen)
    elif isinstance(node, list):
        seen.add(obj_id)
        for element in cast(list[object], node):
            _walk_and_rewrite(element, id_mapping, stats, row_pk, seen)


def rewrite_metadata_references(
    metadata_json: str | None,
    id_mapping: Mapping[int, str],
    stats: MigrationStats,
    row_pk: int,
) -> str | None:
    """Rewrite integer ``context_ids`` arrays inside the JSON metadata.

    Walks the parsed metadata structure looking for every
    ``references.context_ids`` list. Each integer entry is replaced with
    its mapped UUIDv7 hex string. Non-integer entries are preserved
    unchanged. Unmapped integers (orphans) are preserved as integers and
    counted in ``stats.orphan_references``; a warning is recorded.
    Malformed structures are preserved unchanged and counted in
    ``stats.malformed_references``.

    Re-encoding is done with ``allow_nan=False``. This function runs on ALL FOUR
    migration directions, and it is where an invalid token would be MANUFACTURED:
    ``json.loads`` accepts the non-standard ``NaN``/``Infinity``/``-Infinity`` AND
    silently turns a standard-but-overflowing literal such as ``1e400`` into
    ``inf``, and a default ``json.dumps`` then writes those back as tokens no
    RFC 8259 parser accepts. That converts VALID source metadata into INVALID
    target metadata on the SQLite paths (``json_valid`` flips to 0) and aborts the
    whole transaction on the PostgreSQL paths (the jsonb bind rejects it). With
    ``allow_nan=False`` the encoder raises instead, and the row's ORIGINAL metadata
    is preserved verbatim -- which is valid JSON on every target, since only the
    Python float round trip could not represent it. Reference rewriting is skipped
    for that row and the omission is recorded in ``stats.errors`` (a non-empty
    error list makes the CLI exit non-zero), so the operator can repair the value
    rather than discover a stale reference later.

    Args:
        metadata_json: Raw metadata JSON string from the source row, or
            ``None`` if no metadata was stored.
        id_mapping: Integer-to-UUIDv7 hex mapping.
        stats: Mutated to record rewrite counts and warning/error
            messages.
        row_pk: Source row's integer ID, used for log/error context.

    Returns:
        Re-encoded JSON string with rewritten references, or ``None``
        when the input was ``None``.
    """
    if metadata_json is None:
        return None
    try:
        parsed: object = json.loads(metadata_json)
    except json.JSONDecodeError as exc:
        stats.errors.append(f'row {row_pk}: metadata JSON parse failed ({exc}); preserved verbatim')
        return metadata_json
    # _walk_and_rewrite counts each remapping as it mutates the parsed structure, but
    # the encoder below can still reject that structure -- in which case the mutations
    # are discarded and the ORIGINAL metadata is returned. Snapshot the counter so the
    # discarded remappings are not reported as remappings that reached the target.
    references_rewritten_before = stats.references_rewritten
    _walk_and_rewrite(parsed, id_mapping, stats, row_pk, seen=set())
    try:
        return json.dumps(parsed, ensure_ascii=False, allow_nan=False)
    except ValueError as exc:
        stats.references_rewritten = references_rewritten_before
        stats.errors.append(
            f'row {row_pk}: metadata contains a number Python cannot round-trip as JSON '
            f'({exc}); the original metadata is preserved verbatim and any integer '
            f'context_ids references inside it were NOT rewritten',
        )
        return metadata_json
