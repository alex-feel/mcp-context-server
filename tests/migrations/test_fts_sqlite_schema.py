"""SQLite FTS5 table and triggers of the FTS migration, queried with raw SQL: query modes, sync and filters."""

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def fts_enabled_db(tmp_path: Path) -> Path:
    """Create a database with FTS enabled.

    Returns:
        Path to the test database with FTS5 table.
    """
    db_path = tmp_path / 'test_fts.db'

    # Load main schema
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')

    # Load FTS migration template and apply tokenizer replacement
    # Use 'unicode61' (no stemming) to test multilingual support behavior
    migration_path = Path(__file__).parent.parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
    fts_sql = migration_path.read_text()
    fts_sql = fts_sql.replace('{TOKENIZER}', 'unicode61')

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(schema_sql)
        conn.executescript(fts_sql)

        # Insert test data
        conn.execute(
            '''
            INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
            VALUES ('0190abcdef1234567890abcd00000001', 'test-thread', 'agent', 'text', 'Python programming language tutorial',
                    'local')
        ''',
        )
        conn.execute(
            '''
            INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
            VALUES ('0190abcdef1234567890abcd00000002', 'test-thread', 'user', 'text', 'How to learn JavaScript quickly',
                    'local')
        ''',
        )
        conn.execute(
            '''
            INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
            VALUES ('0190abcdef1234567890abcd00000003', 'test-thread', 'agent', 'text', 'Running Python scripts on Linux',
                    'local')
        ''',
        )
        conn.execute(
            '''
            INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
            VALUES ('0190abcdef1234567890abcd00000004', 'other-thread', 'user', 'text', 'Database indexing strategies',
                    'local')
        ''',
        )
        conn.commit()

    return db_path


class TestFtsSQLiteIntegration:
    """Test FTS with SQLite backend."""

    def test_fts_match_mode(self, fts_enabled_db: Path) -> None:
        """Test basic FTS match mode search."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'python'
                ORDER BY score DESC
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 2  # Both Python entries
            assert 'Python' in results[0]['text_content']

    def test_fts_phrase_mode(self, fts_enabled_db: Path) -> None:
        """Test FTS phrase mode search."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.*
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH '"programming language"'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'programming language' in results[0]['text_content']

    def test_fts_no_stemming_with_unicode61(self, fts_enabled_db: Path) -> None:
        """Test that stemming does NOT work with unicode61 tokenizer.

        With unicode61 tokenizer (multilingual support), there is no stemming.
        This means "run" will NOT match "running" - this is the expected
        trade-off for multilingual support. Use PostgreSQL for stemming.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.*
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'run'
            ''',
            )
            results = cursor.fetchall()

            # "Running" should NOT match "run" with unicode61 (no stemming)
            # This is the trade-off for multilingual support
            assert len(results) == 0

    def test_fts_exact_word_match(self, fts_enabled_db: Path) -> None:
        """Test that exact word matches still work with unicode61."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.*
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'Running'
            ''',
            )
            results = cursor.fetchall()

            # Exact word "Running" should match (case-insensitive with unicode61)
            assert len(results) == 1
            assert 'Running' in results[0]['text_content']

    def test_fts_prefix_mode(self, fts_enabled_db: Path) -> None:
        """Test FTS prefix mode with wildcard."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.*
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'prog*'
            ''',
            )
            results = cursor.fetchall()

            # Should match "programming"
            assert len(results) >= 1
            assert any('programming' in r['text_content'].lower() for r in results)

    def test_fts_highlight(self, fts_enabled_db: Path) -> None:
        """Test FTS highlight function."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT highlight(context_entries_fts, 0, '<b>', '</b>') as highlighted
                FROM context_entries_fts
                WHERE text_content MATCH 'python'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 2
            assert '<b>' in results[0]['highlighted']

    def test_fts_no_results(self, fts_enabled_db: Path) -> None:
        """Test FTS returns empty results for non-matching query."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.*
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'nonexistentword123456'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 0

    def test_fts_score_ordering(self, fts_enabled_db: Path) -> None:
        """Test that results are ordered by relevance score."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'python'
                ORDER BY score DESC
            ''',
            )
            results = cursor.fetchall()

            # Verify we have results with scores
            assert len(results) >= 1
            scores = [r['score'] for r in results]
            # Scores should be in descending order
            assert scores == sorted(scores, reverse=True)


class TestFtsTriggerSync:
    """Test that FTS index stays in sync with main table."""

    def test_insert_sync(self, fts_enabled_db: Path) -> None:
        """Test FTS index is updated on INSERT."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Insert new entry
            conn.execute(
                '''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                VALUES ('0190abcdef1234567890abcd00000005', 'test-thread', 'agent', 'text', 'Unique searchable content XYZ123',
                        'local')
            ''',
            )
            conn.commit()

            # Search for it
            cursor = conn.execute(
                '''
                SELECT COUNT(*) as count FROM context_entries_fts WHERE text_content MATCH 'XYZ123'
            ''',
            )
            result = cursor.fetchone()
            assert result['count'] == 1

    def test_delete_sync(self, fts_enabled_db: Path) -> None:
        """Test FTS index is updated on DELETE."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            # Get count before delete
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM context_entries_fts WHERE text_content MATCH 'python'",
            )
            count_before = cursor.fetchone()[0]
            assert count_before == 2

            # Get ID of one Python entry
            cursor = conn.execute("SELECT id FROM context_entries WHERE text_content LIKE '%Python%' LIMIT 1")
            entry_id = cursor.fetchone()[0]

            # Delete it
            conn.execute('DELETE FROM context_entries WHERE id = ?', (entry_id,))
            conn.commit()

            # Verify FTS count decreased
            cursor = conn.execute(
                "SELECT COUNT(*) as count FROM context_entries_fts WHERE text_content MATCH 'python'",
            )
            count_after = cursor.fetchone()[0]
            # Should be 1 now (was 2 before)
            assert count_after == 1

    def test_update_sync(self, fts_enabled_db: Path) -> None:
        """Test FTS index is updated on UPDATE."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            # Update text content
            conn.execute(
                '''
                UPDATE context_entries SET text_content = 'Rust programming language'
                WHERE text_content LIKE '%Python programming%'
            ''',
            )
            conn.commit()

            # Verify old term not found with phrase search
            cursor = conn.execute(
                "SELECT COUNT(*) FROM context_entries_fts WHERE text_content MATCH '\"Python programming\"'",
            )
            assert cursor.fetchone()[0] == 0

            # Verify new term found
            cursor = conn.execute(
                "SELECT COUNT(*) FROM context_entries_fts WHERE text_content MATCH 'rust'",
            )
            assert cursor.fetchone()[0] == 1


class TestFtsWithFilters:
    """Test FTS with additional filters."""

    def test_fts_with_thread_filter(self, fts_enabled_db: Path) -> None:
        """Test FTS combined with thread_id filter."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search for 'language' - should be in both threads but filter to test-thread
            cursor = conn.execute(
                '''
                SELECT ce.*
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'language'
                AND ce.thread_id = 'test-thread'
            ''',
            )
            results = cursor.fetchall()

            # Only the Python tutorial entry should match
            assert len(results) == 1
            assert results[0]['thread_id'] == 'test-thread'

    def test_fts_with_source_filter(self, fts_enabled_db: Path) -> None:
        """Test FTS combined with source filter."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search for entries from agents only
            cursor = conn.execute(
                '''
                SELECT ce.*
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'python'
                AND ce.source = 'agent'
            ''',
            )
            results = cursor.fetchall()

            # Both Python entries are from agents
            assert len(results) == 2
            for r in results:
                assert r['source'] == 'agent'

    def test_fts_index_rebuild(self, fts_enabled_db: Path) -> None:
        """Test FTS index rebuild functionality."""
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            # Verify entries exist before rebuild
            cursor = conn.execute('SELECT COUNT(*) FROM context_entries')
            assert cursor.fetchone()[0] > 0

            # Rebuild index
            conn.execute("INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')")
            conn.commit()

            # Verify entries are still searchable
            cursor = conn.execute(
                '''
                SELECT COUNT(*)
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'python'
            ''',
            )
            result = cursor.fetchone()[0]
            assert result == 2  # Both Python entries still found

    def test_fts_with_tag_filter(self, fts_enabled_db: Path) -> None:
        """Test FTS search with tag filtering.

        Covers the tag filtering logic of the FTS search mixin.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # First, insert an entry with tags
            entry_id = '0190abcdef1234567890abcd00000006'
            conn.execute(
                '''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                VALUES (?, 'tag-thread', 'agent', 'text', 'Python programming with tags', 'local')
                ''',
                (entry_id,),
            )

            # Add tags
            conn.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (entry_id, 'python'),
            )
            conn.execute(
                'INSERT INTO tags (context_entry_id, tag) VALUES (?, ?)',
                (entry_id, 'programming'),
            )
            conn.commit()

            # Search with tag filter using a join
            cursor = conn.execute(
                '''
                SELECT DISTINCT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                JOIN tags t ON t.context_entry_id = ce.id
                WHERE fts.text_content MATCH 'python'
                AND t.tag IN ('python', 'programming')
                ORDER BY score DESC
                ''',
            )
            results = cursor.fetchall()

            assert len(results) >= 1
            assert 'Python' in results[0]['text_content']

    def test_fts_with_content_type_filter(self, fts_enabled_db: Path) -> None:
        """Test FTS search with content_type filter.

        Covers the content_type filtering of the FTS search mixin.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Insert multimodal entry
            conn.execute(
                '''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, owner_id)
                VALUES ('0190abcdef1234567890abcd00000007', 'type-thread', 'agent', 'multimodal', 'Python with image', 'local')
                ''',
            )
            conn.commit()

            # Search for text content_type only
            cursor = conn.execute(
                '''
                SELECT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'python'
                AND ce.content_type = 'text'
                ORDER BY score DESC
                ''',
            )
            results = cursor.fetchall()

            # All results should be 'text' type
            for result in results:
                assert result['content_type'] == 'text'

    def test_fts_with_metadata_filter(self, fts_enabled_db: Path) -> None:
        """Test FTS search with metadata filtering.

        Covers the metadata filtering logic of the FTS search mixin.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Insert entry with metadata
            conn.execute(
                '''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id)
                VALUES (
                    '0190abcdef1234567890abcd00000008',
                    'meta-thread',
                    'agent',
                    'text',
                    'Python data processing',
                    '{"priority": 5}'
                , 'local')
                ''',
            )
            conn.execute(
                '''
                INSERT INTO context_entries (id, thread_id, source, content_type, text_content, metadata, owner_id)
                VALUES (
                    '0190abcdef1234567890abcd00000009',
                    'meta-thread',
                    'agent',
                    'text',
                    'Python web development',
                    '{"priority": 3}'
                , 'local')
                ''',
            )
            conn.commit()

            # Search with metadata filter using json_extract
            cursor = conn.execute(
                '''
                SELECT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH 'python'
                AND json_extract(ce.metadata, '$.priority') > 4
                ORDER BY score DESC
                ''',
            )
            results = cursor.fetchall()

            # Should only return the high priority entry
            assert len(results) == 1
            assert 'data processing' in results[0]['text_content']

    def test_fts_explain_query_returns_plan(self, fts_enabled_db: Path) -> None:
        """Test that EXPLAIN QUERY PLAN works with FTS queries.

        Verifies FTS explain_query functionality.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            sql_query = '''
                SELECT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.rowid_int = fts.rowid
                WHERE fts.text_content MATCH ?
                ORDER BY score DESC
                LIMIT ? OFFSET ?
            '''

            cursor = conn.execute(f'EXPLAIN QUERY PLAN {sql_query}', ('python', 10, 0))
            plan_rows = cursor.fetchall()

            assert len(plan_rows) > 0
            # Plan should contain details about FTS5 usage
            plan_text = ' '.join(dict(row).get('detail', '') for row in plan_rows)
            assert len(plan_text) > 0  # Should have some plan output
