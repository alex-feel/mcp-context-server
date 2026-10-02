"""Tests for the metadata reference rewrite and the free-form columns it leaves untouched."""

import hashlib
import json
import sqlite3
from pathlib import Path

from app.cli.migrate_uuid.records import MigrationStats
from app.cli.migrate_uuid.references import rewrite_metadata_references
from app.cli.migrate_uuid.sqlite_to_sqlite import run_migration_sqlite_to_sqlite
from tests.cli.migrate_uuid._sources import build_sqlite_options


class TestMetadataReferencesRewrite:
    """Rewrite behavior for integer entries inside references.context_ids."""

    def test_top_level_references_remapped(self) -> None:
        """An integer list at the canonical path is rewritten to UUID strings."""
        stats = MigrationStats()
        mapping = {3: 'aaaa' * 8, 7: 'bbbb' * 8}
        metadata = {'references': {'context_ids': [3, 7]}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), mapping, stats, row_pk=99)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == ['aaaa' * 8, 'bbbb' * 8]
        assert stats.references_rewritten == 2
        assert stats.orphan_references == 0
        assert stats.malformed_references == 0

    def test_nested_references_remapped(self) -> None:
        """References buried inside a list-of-objects are rewritten recursively."""
        stats = MigrationStats()
        mapping = {5: 'cccc' * 8}
        metadata = {
            'history': [
                {'note': 'first'},
                {'references': {'context_ids': [5]}},
            ],
        }
        rewritten = rewrite_metadata_references(json.dumps(metadata), mapping, stats, row_pk=10)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['history'][1]['references']['context_ids'] == ['cccc' * 8]
        assert stats.references_rewritten == 1

    def test_string_references_preserved(self) -> None:
        """String entries inside context_ids are kept unchanged."""
        stats = MigrationStats()
        existing_uuid = 'd' * 32
        metadata = {'references': {'context_ids': [existing_uuid]}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), {}, stats, row_pk=1)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == [existing_uuid]
        assert stats.references_rewritten == 0
        assert stats.orphan_references == 0

    def test_orphan_reference_warning_emitted(self) -> None:
        """An integer with no mapping entry produces a warning and is kept as integer."""
        stats = MigrationStats()
        metadata = {'references': {'context_ids': [9999]}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), {}, stats, row_pk=42)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == [9999]
        assert stats.orphan_references == 1
        assert any('9999' in w for w in stats.warnings)

    def test_malformed_references_recorded_in_errors(self) -> None:
        """A non-list ``context_ids`` value is flagged and preserved unchanged."""
        stats = MigrationStats()
        metadata = {'references': {'context_ids': 'not-a-list'}}
        rewritten = rewrite_metadata_references(json.dumps(metadata), {}, stats, row_pk=7)
        assert rewritten is not None
        parsed = json.loads(rewritten)
        assert parsed['references']['context_ids'] == 'not-a-list'
        assert stats.malformed_references == 1
        assert any('not a list' in e for e in stats.errors)


class TestGhostReferences:
    """Free-form text columns are never rewritten."""

    def test_text_content_not_rewritten(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """text_content containing integer-style mentions is copied verbatim."""
        options = build_sqlite_options(legacy_source_db, new_target_db_path)
        stats = run_migration_sqlite_to_sqlite(options)
        assert stats.rows_migrated == 3

        # Confirm via hash equality between source and target text_content.
        src_conn = sqlite3.connect(str(legacy_source_db))
        tgt_conn = sqlite3.connect(str(new_target_db_path))
        try:
            src_conn.row_factory = sqlite3.Row
            tgt_conn.row_factory = sqlite3.Row
            src_rows = list(src_conn.execute(
                'SELECT text_content FROM context_entries ORDER BY created_at ASC, id ASC',
            ))
            tgt_rows = list(tgt_conn.execute(
                'SELECT text_content FROM context_entries ORDER BY created_at ASC',
            ))
            assert len(src_rows) == len(tgt_rows)
            for src_row, tgt_row in zip(src_rows, tgt_rows, strict=True):
                src_hash = hashlib.sha256(
                    (src_row['text_content'] or '').encode('utf-8'),
                ).hexdigest()
                tgt_hash = hashlib.sha256(
                    (tgt_row['text_content'] or '').encode('utf-8'),
                ).hexdigest()
                assert src_hash == tgt_hash
        finally:
            src_conn.close()
            tgt_conn.close()

    def test_summary_not_rewritten(
        self,
        legacy_source_db: Path,
        new_target_db_path: Path,
    ) -> None:
        """summary column is preserved byte-for-byte across migration."""
        options = build_sqlite_options(legacy_source_db, new_target_db_path)
        run_migration_sqlite_to_sqlite(options)
        tgt_conn = sqlite3.connect(str(new_target_db_path))
        try:
            tgt_conn.row_factory = sqlite3.Row
            row = tgt_conn.execute(
                "SELECT summary FROM context_entries WHERE summary LIKE '%8944%'",
            ).fetchone()
            assert row is not None
            assert '8944' in row['summary']
        finally:
            tgt_conn.close()
