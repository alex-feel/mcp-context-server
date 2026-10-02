"""Tests for the integer-to-UUIDv7 id mapping and its created_at tolerance."""

import sqlite3
from datetime import UTC
from datetime import datetime
from pathlib import Path

from app.cli.migrate_uuid.id_mapping import build_id_mapping
from app.cli.migrate_uuid.sqlite_to_sqlite import run_migration_sqlite_to_sqlite
from tests.cli.migrate_uuid._sources import HEX_32_RE
from tests.cli.migrate_uuid._sources import build_sqlite_options
from tests.cli.migrate_uuid._sources import seed_source_db


def _decode_unix_ts_ms(uuid_hex: str) -> int:
    """Extract the 48-bit unix_ts_ms field embedded in a UUIDv7 hex value."""
    return int(uuid_hex[:12], 16)


class TestUuidMappingDeterminism:
    """Determinism guarantees for the integer-to-UUIDv7 mapping table."""

    def test_mapping_repeatable_at_ms_granularity(self) -> None:
        """Two runs against the same rows produce UUIDs whose unix_ts_ms field matches row-by-row."""
        created_a = datetime(2025, 3, 4, 12, 0, 0, tzinfo=UTC)
        created_b = datetime(2025, 3, 4, 12, 0, 1, tzinfo=UTC)
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, created_a.isoformat()))
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, created_b.isoformat()))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        first = build_id_mapping(rows)
        second = build_id_mapping(rows)
        for row_id in first:
            assert _decode_unix_ts_ms(first[row_id]) == _decode_unix_ts_ms(second[row_id])


class TestUuidMappingFormat:
    """Format invariants for produced UUIDv7 hex strings."""

    def test_mapping_emits_32char_lowercase_hex(self) -> None:
        """Each value in the mapping matches ``^[0-9a-f]{32}$``."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        for i, year in enumerate((2024, 2025, 2026), start=1):
            conn.execute(
                'INSERT INTO r VALUES (?, ?)',
                (i, datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()),
            )
        rows = list(conn.execute('SELECT * FROM r'))
        mapping = build_id_mapping(rows)
        for value in mapping.values():
            assert HEX_32_RE.match(value), f'{value!r} is not 32-char lowercase hex'


class TestUuidYearAnchor:
    """Decoded year-from-UUIDv7 matches the source created_at year."""

    def test_uuid_year_matches_source_created_at_year(self) -> None:
        """Decoded UUIDv7 timestamp lands in the source's year, NOT year ~50,000."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        years = (2024, 2025, 2026)
        for i, year in enumerate(years, start=1):
            conn.execute(
                'INSERT INTO r VALUES (?, ?)',
                (i, datetime(year, 6, 15, 12, 0, 0, tzinfo=UTC).isoformat()),
            )
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        for row_id, expected_year in zip(sorted(mapping.keys()), years, strict=True):
            unix_ts_ms = _decode_unix_ts_ms(mapping[row_id])
            decoded = datetime.fromtimestamp(unix_ts_ms / 1000.0, tz=UTC)
            assert decoded.year == expected_year, (
                f'row {row_id}: decoded year {decoded.year}, expected {expected_year}'
            )


class TestNullCreatedAtTolerance:
    """A schema-legal NULL created_at must not abort the migration."""

    def test_build_id_mapping_anchors_null_created_at(self) -> None:
        """build_id_mapping anchors a NULL created_at row instead of raising."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, '2025-06-24T12:00:00+00:00'))
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, None))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1, 2}
        assert all(HEX_32_RE.match(value) for value in mapping.values())

    def test_build_id_mapping_anchors_pre_epoch_created_at(self) -> None:
        """A pre-1970 created_at is anchored (no uuid7 OverflowError on negative epoch)."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at TIMESTAMP)')
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, '1969-06-15T12:00:00+00:00'))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1}
        assert HEX_32_RE.match(mapping[1])

    def test_build_id_mapping_handles_numeric_epoch_created_at(self) -> None:
        """A numeric Unix-epoch created_at (some non-app sources store it) is coerced via
        epoch+timedelta, not aborted; a negative (pre-1970) epoch is anchored like NULL."""
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at)')  # typeless: keep the raw int
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, 1700000000))  # 2023 epoch seconds
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, -100))  # pre-1970 epoch -> anchored
        conn.execute('INSERT INTO r VALUES (?, ?)', (3, 1700000000.5))  # float epoch
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1, 2, 3}
        assert all(HEX_32_RE.match(value) for value in mapping.values())

    def test_migration_succeeds_with_null_created_at_row(
        self,
        tmp_path: Path,
        new_target_db_path: Path,
    ) -> None:
        """A source row with NULL created_at migrates without aborting the run."""
        source = tmp_path / 'null_created_at_source.db'
        seed_source_db(source, [
            {
                'id': 1, 'thread_id': 't', 'source': 'user', 'content_type': 'text',
                'text_content': 'has timestamp', 'metadata': None,
                'created_at': '2025-06-24 12:00:00',
            },
            {
                'id': 2, 'thread_id': 't', 'source': 'agent', 'content_type': 'text',
                'text_content': 'null timestamp', 'metadata': None,
                'created_at': None,
            },
        ])
        stats = run_migration_sqlite_to_sqlite(build_sqlite_options(source, new_target_db_path))
        assert not stats.errors
        assert stats.rows_migrated == 2
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            total = conn.execute('SELECT COUNT(*) FROM context_entries').fetchone()[0]
            null_ts = conn.execute(
                'SELECT COUNT(*) FROM context_entries WHERE created_at IS NULL',
            ).fetchone()[0]
        finally:
            conn.close()
        assert total == 2
        assert null_ts == 1

    def test_build_id_mapping_anchors_malformed_string_created_at(self) -> None:
        """A malformed non-ISO string created_at is anchored (not aborted) for id derivation.

        Mirrors the NULL / pre-epoch / numeric-epoch tolerance: an arbitrary non-app source
        database may store created_at as a non-ISO string, and one such row must not abort the
        whole migration via an uncaught _coerce_datetime ValueError.
        """
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        conn.execute('CREATE TABLE r (id INTEGER, created_at)')  # typeless: keep the raw text
        conn.execute('INSERT INTO r VALUES (?, ?)', (1, '2024/01/01 12:00:00'))  # non-ISO slashes
        conn.execute('INSERT INTO r VALUES (?, ?)', (2, '15-06-2024 10:00:00'))  # day-first
        conn.execute('INSERT INTO r VALUES (?, ?)', (3, 'not-a-date'))
        rows = list(conn.execute('SELECT * FROM r ORDER BY id'))
        mapping = build_id_mapping(rows)
        assert set(mapping) == {1, 2, 3}
        assert all(HEX_32_RE.match(value) for value in mapping.values())

    def test_stored_datetime_or_none_tolerates_malformed_and_null(self) -> None:
        """The verbatim-bind helper yields None for NULL/malformed input and a datetime for valid."""
        from app.cli.migrate_uuid.id_mapping import stored_datetime_or_none

        assert stored_datetime_or_none(None) is None
        assert stored_datetime_or_none('2024/01/01 12:00:00') is None  # non-ISO -> None, not raise
        assert stored_datetime_or_none('not-a-date') is None
        valid = stored_datetime_or_none('2025-06-24T12:00:00+00:00')
        assert valid is not None
        assert valid.year == 2025

    def test_migration_succeeds_with_malformed_string_created_at_row(
        self,
        tmp_path: Path,
        new_target_db_path: Path,
    ) -> None:
        """A source row with a malformed string created_at migrates without aborting the run."""
        source = tmp_path / 'malformed_created_at_source.db'
        seed_source_db(source, [
            {
                'id': 1, 'thread_id': 't', 'source': 'user', 'content_type': 'text',
                'text_content': 'well-formed timestamp', 'metadata': None,
                'created_at': '2025-06-24 12:00:00',
            },
            {
                'id': 2, 'thread_id': 't', 'source': 'agent', 'content_type': 'text',
                'text_content': 'malformed timestamp', 'metadata': None,
                'created_at': '2024/01/01 12:00:00',  # non-ISO -> id anchored, value preserved
            },
        ])
        stats = run_migration_sqlite_to_sqlite(build_sqlite_options(source, new_target_db_path))
        assert not stats.errors
        assert stats.rows_migrated == 2
        conn = sqlite3.connect(str(new_target_db_path))
        try:
            total = conn.execute('SELECT COUNT(*) FROM context_entries').fetchone()[0]
        finally:
            conn.close()
        assert total == 2
