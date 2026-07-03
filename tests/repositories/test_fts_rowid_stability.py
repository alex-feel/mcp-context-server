"""Tests that the SQLite FTS5 ``rowid_int`` surrogate is stable across VACUUM.

The schema defines ``rowid_int INTEGER PRIMARY KEY AUTOINCREMENT`` so that
VACUUM does not reassign rowids. The FTS5 external-content table uses this
surrogate as its ``content_rowid``, and the FTS triggers reference
``new.rowid_int`` / ``old.rowid_int``. This test asserts those properties hold
end-to-end through a real VACUUM cycle.
"""

import sqlite3
from pathlib import Path

import pytest

from app.ids import generate_id


@pytest.fixture
def db_with_fts(tmp_path: Path) -> Path:
    """Build a SQLite database with the base schema plus FTS migration applied."""
    db_path = tmp_path / 'fts.db'
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    fts_sql_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
    fts_sql = fts_sql_path.read_text(encoding='utf-8').replace('{TOKENIZER}', 'unicode61')

    with sqlite3.connect(str(db_path)) as conn:
        conn.executescript(schema_sql)
        conn.executescript(fts_sql)
    return db_path


class TestRowidIntStability:
    """The private ``rowid_int`` column is stable across VACUUM."""

    def test_rowid_int_persists_through_vacuum(self, db_with_fts: Path) -> None:
        """Inserted rowid_int values are unchanged after VACUUM."""
        with sqlite3.connect(str(db_with_fts)) as conn:
            conn.row_factory = sqlite3.Row
            id_a = generate_id()
            id_b = generate_id()
            for entry_id, text in ((id_a, 'first entry'), (id_b, 'second entry')):
                conn.execute(
                    '''INSERT INTO context_entries
                       (id, thread_id, source, content_type, text_content)
                       VALUES (?, ?, ?, ?, ?)''',
                    (entry_id, 't', 'user', 'text', text),
                )
            conn.commit()

            cursor = conn.execute(
                'SELECT id, rowid_int FROM context_entries ORDER BY rowid_int',
            )
            before = {row['id']: row['rowid_int'] for row in cursor.fetchall()}
            assert len(before) == 2

        with sqlite3.connect(str(db_with_fts)) as conn:
            conn.isolation_level = None
            conn.execute('VACUUM')

        with sqlite3.connect(str(db_with_fts)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                'SELECT id, rowid_int FROM context_entries ORDER BY rowid_int',
            )
            after = {row['id']: row['rowid_int'] for row in cursor.fetchall()}

        assert before == after, (
            'rowid_int values changed across VACUUM; FTS5 external content '
            'mapping would be corrupted. Verify the schema uses '
            'INTEGER PRIMARY KEY AUTOINCREMENT for rowid_int.'
        )

    def test_fts_match_after_vacuum_returns_uuid_id(self, db_with_fts: Path) -> None:
        """FTS5 search joining on rowid_int returns the correct UUID hex id post-VACUUM."""
        with sqlite3.connect(str(db_with_fts)) as conn:
            id_a = generate_id()
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, content_type, text_content)
                   VALUES (?, ?, ?, ?, ?)''',
                (id_a, 't', 'user', 'text', 'searchable token unique12345'),
            )
            conn.commit()

        with sqlite3.connect(str(db_with_fts)) as conn:
            conn.isolation_level = None
            conn.execute('VACUUM')

        with sqlite3.connect(str(db_with_fts)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                """SELECT ce.id
                   FROM context_entries ce
                   JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                   WHERE fts.text_content MATCH 'unique12345'""",
            )
            rows = cursor.fetchall()

        assert len(rows) == 1
        assert rows[0]['id'] == id_a
