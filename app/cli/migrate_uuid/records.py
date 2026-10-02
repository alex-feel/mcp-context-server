"""Statistics and options records of the integer-to-UUIDv7 migration."""

from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from typing import cast


@dataclass
class MigrationStats:
    """Counters and warning/error log for a migration run.

    Attributes:
        rows_migrated: Number of ``context_entries`` rows copied to the
            target.
        references_rewritten: Number of integer entries inside
            ``metadata.references.context_ids`` arrays that were
            successfully remapped to UUIDv7 hex strings AND reached the
            target. A remapping the run then discards -- because the
            re-encode was rejected and the original metadata was preserved,
            or because the whole row was skipped -- is not counted.
        orphan_references: Number of integer entries inside
            ``metadata.references.context_ids`` arrays that did not match
            any source ``context_entries.id``; these are preserved as
            integers and a warning is logged for each.
        malformed_references: Number of rows whose ``metadata`` contained
            a ``references`` block with an unexpected shape (for example,
            a non-array ``context_ids`` value). The row's metadata is
            preserved unchanged and a warning is logged.
        tags_migrated: Number of tag rows copied.
        images_migrated: Number of ``image_attachments`` rows copied.
        embedding_metadata_migrated: Number of ``embedding_metadata`` rows
            copied.
        embedding_chunks_migrated: Number of ``embedding_chunks`` rows
            copied (SQLite).
        vec_rows_migrated: Number of ``vec_context_embeddings`` rows
            copied.
        fts_rebuilt: Whether the FTS5 index rebuild succeeded on the
            target.
        warnings: Free-form warning messages.
        errors: Free-form error messages.
    """

    rows_migrated: int = 0
    references_rewritten: int = 0
    orphan_references: int = 0
    malformed_references: int = 0
    tags_migrated: int = 0
    images_migrated: int = 0
    embedding_metadata_migrated: int = 0
    embedding_chunks_migrated: int = 0
    vec_rows_migrated: int = 0
    fts_rebuilt: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a plain-dict view suitable for :func:`json.dump`.

        Returns:
            Dictionary with the same key ordering as the dataclass field
            declaration order.
        """
        return cast(dict[str, object], asdict(self))


@dataclass
class MigrationOptions:
    """Parsed CLI arguments.

    Attributes:
        source_url: URL or path identifying the source database. Accepted
            forms: ``sqlite:///abs/path/file.db``, ``/abs/path/file.db``,
            ``postgresql://user:pass@host/db``.
        target_url: URL or path identifying the target database. Same
            forms as ``source_url``.
        dry_run: When True, run the full migration logic in memory but
            issue no INSERT statements against the target.
        report_path: Optional path. When set, write the migration
            statistics as JSON to this file at end of run.
    """

    source_url: str
    target_url: str
    dry_run: bool = False
    report_path: Path | None = None
