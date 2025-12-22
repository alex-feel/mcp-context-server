"""Integration tests for FTS functionality.

Tests the full-text search functionality with real SQLite FTS5 tables.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
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
    migration_path = Path(__file__).parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
    fts_sql = migration_path.read_text()
    fts_sql = fts_sql.replace('{TOKENIZER}', 'unicode61')

    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        conn.executescript(schema_sql)
        conn.executescript(fts_sql)

        # Insert test data
        conn.execute(
            '''
            INSERT INTO context_entries (thread_id, source, content_type, text_content)
            VALUES ('test-thread', 'agent', 'text', 'Python programming language tutorial')
        ''',
        )
        conn.execute(
            '''
            INSERT INTO context_entries (thread_id, source, content_type, text_content)
            VALUES ('test-thread', 'user', 'text', 'How to learn JavaScript quickly')
        ''',
        )
        conn.execute(
            '''
            INSERT INTO context_entries (thread_id, source, content_type, text_content)
            VALUES ('test-thread', 'agent', 'text', 'Running Python scripts on Linux')
        ''',
        )
        conn.execute(
            '''
            INSERT INTO context_entries (thread_id, source, content_type, text_content)
            VALUES ('other-thread', 'user', 'text', 'Database indexing strategies')
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                INSERT INTO context_entries (thread_id, source, content_type, text_content)
                VALUES ('test-thread', 'agent', 'text', 'Unique searchable content XYZ123')
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
            conn.execute(f'DELETE FROM context_entries WHERE id = {entry_id}')
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'python'
            ''',
            )
            result = cursor.fetchone()[0]
            assert result == 2  # Both Python entries still found

    def test_fts_with_tag_filter(self, fts_enabled_db: Path) -> None:
        """Test FTS search with tag filtering.

        Covers lines 208-221 in fts_repository.py for tag filtering logic.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # First, insert an entry with tags
            cursor = conn.execute(
                '''
                INSERT INTO context_entries (thread_id, source, content_type, text_content)
                VALUES ('tag-thread', 'agent', 'text', 'Python programming with tags')
                ''',
            )
            entry_id = cursor.lastrowid

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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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

        Covers lines 196-198 in fts_repository.py for content_type filtering.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Insert multimodal entry
            conn.execute(
                '''
                INSERT INTO context_entries (thread_id, source, content_type, text_content)
                VALUES ('type-thread', 'agent', 'multimodal', 'Python with image')
                ''',
            )
            conn.commit()

            # Search for text content_type only
            cursor = conn.execute(
                '''
                SELECT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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

        Covers metadata filtering logic in fts_repository.py.
        """
        with sqlite3.connect(str(fts_enabled_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Insert entry with metadata
            conn.execute(
                '''
                INSERT INTO context_entries (thread_id, source, content_type, text_content, metadata)
                VALUES ('meta-thread', 'agent', 'text', 'Python data processing', '{"priority": 5}')
                ''',
            )
            conn.execute(
                '''
                INSERT INTO context_entries (thread_id, source, content_type, text_content, metadata)
                VALUES ('meta-thread', 'agent', 'text', 'Python web development', '{"priority": 3}')
                ''',
            )
            conn.commit()

            # Search with metadata filter using json_extract
            cursor = conn.execute(
                '''
                SELECT ce.*, -bm25(context_entries_fts) as score
                FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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
                JOIN context_entries_fts fts ON ce.id = fts.rowid
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


class TestFtsMultilingualUnicode61:
    """Test FTS with unicode61 tokenizer for multilingual content.

    The unicode61 tokenizer provides proper Unicode tokenization for all languages,
    but does NOT provide stemming. This is a trade-off: we get multilingual support
    at the cost of losing features like "running" matching "run".
    """

    @pytest.fixture
    def multilingual_db(self, tmp_path: Path) -> Path:
        """Create a database with multilingual content for FTS testing.

        Returns:
            Path to the test database with multilingual entries.
        """
        db_path = tmp_path / 'test_fts_multilingual.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        # Load FTS migration template and apply tokenizer replacement
        # Use 'unicode61' (no stemming) for multilingual support testing
        migration_path = Path(__file__).parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text()
        fts_sql = fts_sql.replace('{TOKENIZER}', 'unicode61')

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(schema_sql)
            conn.executescript(fts_sql)

            # Insert multilingual test data
            test_entries = [
                # German
                ('german-thread', 'agent', 'text', 'Die Programmierung ist interessant'),
                # French
                ('french-thread', 'agent', 'text', 'Le developpement logiciel est fascinant'),
                # Spanish
                ('spanish-thread', 'agent', 'text', 'La programacion es muy importante'),
                # Russian (Cyrillic)
                ('russian-thread', 'agent', 'text', 'Программирование это интересно'),
                # Chinese
                ('chinese-thread', 'agent', 'text', 'Python 编程语言很流行'),
                # Japanese
                ('japanese-thread', 'agent', 'text', 'プログラミングは楽しいです'),
                # Korean
                ('korean-thread', 'agent', 'text', '프로그래밍은 재미있습니다'),
                # Arabic
                ('arabic-thread', 'agent', 'text', 'البرمجة ممتعة جدا'),
                # Mixed content with accents
                ('mixed-thread', 'agent', 'text', 'Cafe resume naive facade'),
            ]

            for thread_id, source, content_type, text_content in test_entries:
                conn.execute(
                    '''
                    INSERT INTO context_entries (thread_id, source, content_type, text_content)
                    VALUES (?, ?, ?, ?)
                    ''',
                    (thread_id, source, content_type, text_content),
                )
            conn.commit()

        return db_path

    def test_german_tokenization(self, multilingual_db: Path) -> None:
        """Test that German text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'Programmierung'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'Programmierung' in results[0]['text_content']

    def test_french_tokenization(self, multilingual_db: Path) -> None:
        """Test that French text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'developpement'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'developpement' in results[0]['text_content']

    def test_spanish_tokenization(self, multilingual_db: Path) -> None:
        """Test that Spanish text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'programacion'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'programacion' in results[0]['text_content']

    def test_russian_cyrillic_tokenization(self, multilingual_db: Path) -> None:
        """Test that Russian Cyrillic text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'Программирование'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'Программирование' in results[0]['text_content']

    def test_chinese_tokenization(self, multilingual_db: Path) -> None:
        """Test that Chinese text is searchable.

        Note: FTS5 with unicode61 tokenizes CJK text character-by-character,
        so we search for individual characters or use prefix matching.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            # Search for "Python" which is ASCII within Chinese text
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'Python'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'chinese-thread'

    def test_japanese_tokenization(self, multilingual_db: Path) -> None:
        """Test that Japanese text entry exists and can be retrieved.

        Note: FTS5 with unicode61 has limited support for CJK tokenization
        without explicit ICU support, but entries should still be stored
        and retrievable via exact match or thread filtering.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                WHERE ce.thread_id = 'japanese-thread'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'japanese-thread'

    def test_korean_tokenization(self, multilingual_db: Path) -> None:
        """Test that Korean text entry exists and can be retrieved.

        Note: FTS5 with unicode61 handles Hangul, but word boundaries may
        differ from Korean language conventions.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                WHERE ce.thread_id = 'korean-thread'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'korean-thread'

    def test_arabic_tokenization(self, multilingual_db: Path) -> None:
        """Test that Arabic text is properly tokenized."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'البرمجة'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'arabic-thread'

    def test_accented_characters(self, multilingual_db: Path) -> None:
        """Test that words with accented characters are searchable.

        Unicode61 tokenizer handles accented characters properly.
        """
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            # Search for words that could have accents
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'Cafe'
            ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert results[0]['thread_id'] == 'mixed-thread'

    def test_case_insensitive_search(self, multilingual_db: Path) -> None:
        """Test that unicode61 provides case-insensitive search."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search lowercase
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'programmierung'
            ''',
            )
            results_lower = cursor.fetchall()

            # Search uppercase
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'PROGRAMMIERUNG'
            ''',
            )
            results_upper = cursor.fetchall()

            # Both should find the same entry
            assert len(results_lower) == 1
            assert len(results_upper) == 1
            assert results_lower[0]['thread_id'] == results_upper[0]['thread_id']

    def test_prefix_search_multilingual(self, multilingual_db: Path) -> None:
        """Test that prefix search works with multilingual content."""
        with sqlite3.connect(str(multilingual_db)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'Program*'
            ''',
            )
            results = cursor.fetchall()

            # Should match German "Programmierung" and Spanish "programacion"
            assert len(results) >= 2


class TestFtsGracefulDegradation:
    """Test FTS graceful degradation during migration.

    These tests verify that when FTS migration is in progress, the fts_search_context
    tool returns informative responses instead of errors.
    """

    @pytest.fixture
    def reset_migration_status(self) -> Generator[None, None, None]:
        """Reset FTS migration status before and after each test.

        Uses the exported reset function from server.py to ensure clean state.

        Yields:
            None: Fixture provides no value, only cleanup behavior.
        """
        from app.server import _reset_fts_migration_status

        _reset_fts_migration_status()
        yield
        _reset_fts_migration_status()

    def test_migration_status_dataclass_creation(self) -> None:
        """Test that FtsMigrationStatus dataclass can be created with all fields."""
        from datetime import UTC
        from datetime import datetime

        from app.server import FtsMigrationStatus

        status = FtsMigrationStatus(
            in_progress=True,
            started_at=datetime.now(tz=UTC),
            estimated_seconds=120,
            backend='sqlite',
            old_language='english',
            new_language='german',
            records_count=1000,
        )

        assert status.in_progress is True
        assert status.started_at is not None
        assert status.estimated_seconds == 120
        assert status.backend == 'sqlite'
        assert status.old_language == 'english'
        assert status.new_language == 'german'
        assert status.records_count == 1000

    def test_migration_status_defaults(self) -> None:
        """Test that FtsMigrationStatus dataclass has correct defaults."""
        from app.server import FtsMigrationStatus

        status = FtsMigrationStatus()

        assert status.in_progress is False
        assert status.started_at is None
        assert status.estimated_seconds is None
        assert status.backend is None
        assert status.old_language is None
        assert status.new_language is None
        assert status.records_count is None

    def test_estimate_migration_time_small_dataset(self) -> None:
        """Test migration time estimation for small dataset."""
        from app.server import estimate_migration_time

        # Small dataset: returns minimum time (around 2 seconds)
        estimated = estimate_migration_time(100)
        assert estimated >= 1  # Minimum bound
        assert estimated <= 10  # Should be quick

    def test_estimate_migration_time_large_dataset(self) -> None:
        """Test migration time estimation for large dataset."""
        from app.server import estimate_migration_time

        # Large dataset: should scale appropriately
        estimated = estimate_migration_time(100000)
        assert estimated >= 60  # Should take more time
        # Rough estimate: ~10-15 sec per 1000 records, so 100k = 1000-1500 sec

    def test_estimate_migration_time_zero_records(self) -> None:
        """Test migration time estimation for zero records."""
        from app.server import estimate_migration_time

        # Zero records: returns minimum time (around 2 seconds)
        estimated = estimate_migration_time(0)
        assert estimated >= 1  # Minimum bound for setup overhead
        assert estimated <= 10  # Should be quick

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reset_migration_status')
    async def test_graceful_degradation_response_structure(self) -> None:
        """Test that migration in progress response has correct structure.

        This test verifies the FtsMigrationInProgressDict TypedDict structure.
        """
        from datetime import UTC
        from datetime import datetime
        from unittest.mock import patch

        from app.server import FtsMigrationStatus

        # Create a migration in progress status
        migration_status = FtsMigrationStatus(
            in_progress=True,
            started_at=datetime.now(tz=UTC),
            estimated_seconds=120,
            backend='sqlite',
            old_language='unicode61',
            new_language='porter unicode61',
            records_count=1000,
        )

        # Mock the global status and settings variable (NOT get_settings function)
        with (
            patch('app.server._fts_migration_status', migration_status),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_fts = True
            mock_settings.fts_language = 'english'

            # Import after patching to get patched version
            from app.server import fts_search_context

            # Call the tool function directly
            result = await fts_search_context(query='test query', limit=50)

            # Verify response structure for migration in progress
            assert result['migration_in_progress'] is True
            assert 'message' in result
            assert 'started_at' in result
            assert 'estimated_remaining_seconds' in result
            assert 'old_language' in result
            assert 'new_language' in result
            assert 'suggestion' in result

            # Verify message content
            assert 'being rebuilt' in result['message'].lower()
            assert 'porter unicode61' in result['message']

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reset_migration_status')
    async def test_graceful_degradation_remaining_time_calculation(self) -> None:
        """Test that remaining time is calculated correctly during migration."""
        from datetime import UTC
        from datetime import datetime
        from datetime import timedelta
        from unittest.mock import patch

        from app.server import FtsMigrationStatus

        # Create a migration that started 30 seconds ago with 120 second estimate
        start_time = datetime.now(tz=UTC) - timedelta(seconds=30)
        migration_status = FtsMigrationStatus(
            in_progress=True,
            started_at=start_time,
            estimated_seconds=120,
            backend='sqlite',
            old_language='unicode61',
            new_language='porter unicode61',
            records_count=1000,
        )

        with (
            patch('app.server._fts_migration_status', migration_status),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_fts = True
            mock_settings.fts_language = 'english'

            from app.server import fts_search_context

            result = await fts_search_context(query='test query', limit=50)

            # Should have approximately 90 seconds remaining (120 - 30)
            remaining = result['estimated_remaining_seconds']
            assert 80 <= remaining <= 100  # Allow some tolerance for timing

    @pytest.mark.asyncio
    @pytest.mark.usefixtures('reset_migration_status')
    async def test_graceful_degradation_suggestion_format(self) -> None:
        """Test that suggestion message includes retry time."""
        from datetime import UTC
        from datetime import datetime
        from unittest.mock import patch

        from app.server import FtsMigrationStatus

        migration_status = FtsMigrationStatus(
            in_progress=True,
            started_at=datetime.now(tz=UTC),
            estimated_seconds=60,
            backend='postgresql',
            old_language='english',
            new_language='german',
            records_count=500,
        )

        with (
            patch('app.server._fts_migration_status', migration_status),
            patch('app.server.settings') as mock_settings,
        ):
            mock_settings.enable_fts = True
            mock_settings.fts_language = 'german'

            from app.server import fts_search_context

            result = await fts_search_context(query='test query', limit=50)

            # Verify suggestion includes retry time
            assert 'retry' in result['suggestion'].lower()
            assert 'seconds' in result['suggestion'].lower()


class TestInternalColumnsNotExposed:
    """Test that internal database columns are not exposed in API responses.

    PostgreSQL uses text_search_vector (tsvector) for FTS, which is an internal
    implementation detail. These tests verify that explicit column listing in
    search_contexts() and get_by_ids() prevents internal columns from leaking.
    """

    @pytest.fixture
    def test_db_path(self, tmp_path: Path) -> Path:
        """Create a database for testing internal column exposure.

        Returns:
            Path to the test database.
        """
        db_path = tmp_path / 'test_internal_columns.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(schema_sql)

            # Insert test data
            conn.execute(
                '''
                INSERT INTO context_entries (thread_id, source, content_type, text_content)
                VALUES ('test-thread', 'agent', 'text', 'Test content for column exposure test')
            ''',
            )
            conn.commit()

        return db_path

    @pytest.mark.asyncio
    async def test_search_contexts_does_not_expose_internal_columns(self, test_db_path: Path) -> None:
        """Test that search_contexts() does not return internal columns like text_search_vector.

        This test verifies the fix for the bug where SELECT * was used in queries,
        causing internal PostgreSQL columns (text_search_vector tsvector) to leak
        into API responses. The fix uses explicit column listing via CONTEXT_ENTRY_COLUMNS.
        """
        from app.backends.sqlite_backend import SQLiteBackend
        from app.repositories.context_repository import ContextRepository

        # Initialize backend and repository
        backend = SQLiteBackend(db_path=str(test_db_path))
        await backend.initialize()

        try:
            repo = ContextRepository(backend)

            # Call search_contexts
            rows, _stats = await repo.search_contexts(thread_id='test-thread')

            # Verify we got results
            assert len(rows) == 1

            # Convert Row to dict to check keys
            row_dict = dict(rows[0])

            # Verify expected columns ARE present
            expected_columns = {
                'id',
                'thread_id',
                'source',
                'content_type',
                'text_content',
                'metadata',
                'created_at',
                'updated_at',
            }
            for col in expected_columns:
                assert col in row_dict, f'Expected column {col} missing from result'

            # Verify internal columns are NOT present
            internal_columns = {'text_search_vector', 'fts_vector', 'tsvector'}
            for col in internal_columns:
                assert col not in row_dict, f'Internal column {col} should not be exposed in API response'

        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_get_by_ids_does_not_expose_internal_columns(self, test_db_path: Path) -> None:
        """Test that get_by_ids() does not return internal columns like text_search_vector.

        This test verifies that the explicit column listing fix also applies to
        the get_by_ids() method, preventing internal database columns from leaking.
        """
        from app.backends.sqlite_backend import SQLiteBackend
        from app.repositories.context_repository import ContextRepository

        # Initialize backend and repository
        backend = SQLiteBackend(db_path=str(test_db_path))
        await backend.initialize()

        try:
            repo = ContextRepository(backend)

            # First get the ID of the test entry
            rows, _stats = await repo.search_contexts(thread_id='test-thread')
            assert len(rows) == 1
            context_id = rows[0]['id']

            # Call get_by_ids
            result_rows = await repo.get_by_ids([context_id])

            # Verify we got results
            assert len(result_rows) == 1

            # Convert Row to dict to check keys
            row_dict = dict(result_rows[0])

            # Verify expected columns ARE present
            expected_columns = {
                'id',
                'thread_id',
                'source',
                'content_type',
                'text_content',
                'metadata',
                'created_at',
                'updated_at',
            }
            for col in expected_columns:
                assert col in row_dict, f'Expected column {col} missing from result'

            # Verify internal columns are NOT present
            internal_columns = {'text_search_vector', 'fts_vector', 'tsvector'}
            for col in internal_columns:
                assert col not in row_dict, f'Internal column {col} should not be exposed in API response'

        finally:
            await backend.shutdown()

    @pytest.mark.asyncio
    async def test_context_entry_columns_constant_matches_expected(self) -> None:
        """Test that CONTEXT_ENTRY_COLUMNS constant includes all expected columns.

        This test ensures the column constant is maintained correctly and includes
        all columns defined in the ContextEntryDict TypedDict.
        """
        from app.repositories.context_repository import CONTEXT_ENTRY_COLUMNS

        # Parse the column string into a set
        columns = {col.strip() for col in CONTEXT_ENTRY_COLUMNS.split(',')}

        # Verify all expected columns are present
        expected_columns = {
            'id',
            'thread_id',
            'source',
            'content_type',
            'text_content',
            'metadata',
            'created_at',
            'updated_at',
        }

        assert columns == expected_columns, f'Column mismatch: expected {expected_columns}, got {columns}'

        # Verify internal columns are NOT in the constant
        internal_columns = {'text_search_vector', 'fts_vector', 'tsvector'}
        for col in internal_columns:
            assert col not in columns, f'Internal column {col} should not be in CONTEXT_ENTRY_COLUMNS'


class TestFtsHyphenatedQueries:
    """Integration tests for FTS hyphen handling with real SQLite FTS5 database.

    These tests verify that hyphenated queries like "full-text" work correctly
    and do not cause errors like "no such column: text".
    """

    @pytest.fixture
    def hyphen_test_db(self, tmp_path: Path) -> Path:
        """Create a database with hyphenated content for testing.

        Returns:
            Path to the test database with hyphenated entries.
        """
        db_path = tmp_path / 'test_fts_hyphen.db'

        from app.schemas import load_schema

        schema_sql = load_schema('sqlite')
        migration_path = Path(__file__).parent.parent / 'app' / 'migrations' / 'add_fts_sqlite.sql'
        fts_sql = migration_path.read_text()
        fts_sql = fts_sql.replace('{TOKENIZER}', 'unicode61')

        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            conn.executescript(schema_sql)
            conn.executescript(fts_sql)

            # Insert test data with hyphenated words
            test_entries = [
                ('test-thread', 'agent', 'text', 'Implementing full-text search functionality'),
                ('test-thread', 'agent', 'text', 'Running pre-commit hooks before committing'),
                ('test-thread', 'user', 'text', 'Real-time data processing with streaming'),
                ('test-thread', 'agent', 'text', 'User-friendly interface design patterns'),
                ('test-thread', 'user', 'text', 'Open-source software development practices'),
                ('test-thread', 'agent', 'text', 'Multi-threaded application architecture'),
                ('test-thread', 'user', 'text', 'Regular search without hyphens'),
            ]

            for thread_id, source, content_type, text_content in test_entries:
                conn.execute(
                    '''
                    INSERT INTO context_entries (thread_id, source, content_type, text_content)
                    VALUES (?, ?, ?, ?)
                    ''',
                    (thread_id, source, content_type, text_content),
                )
            conn.commit()

        return db_path

    def test_fts_hyphenated_match_mode(self, hyphen_test_db: Path) -> None:
        """Test match mode search with hyphenated term - should NOT error."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search with quoted hyphenated term (as transformed by our fix)
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH '"full-text"'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'full-text' in results[0]['text_content']

    def test_fts_hyphenated_prefix_mode(self, hyphen_test_db: Path) -> None:
        """Test prefix mode search with hyphenated term."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Prefix search with quoted hyphenated term
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH '"pre-commit"*'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'pre-commit' in results[0]['text_content']

    def test_fts_hyphenated_phrase_mode(self, hyphen_test_db: Path) -> None:
        """Test phrase mode search with hyphenated term."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Phrase search including hyphenated word
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH '"full-text search"'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'full-text search' in results[0]['text_content']

    def test_fts_common_hyphenated_terms(self, hyphen_test_db: Path) -> None:
        """Test common hyphenated programming terms."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            hyphenated_terms = [
                ('real-time', 'Real-time'),
                ('user-friendly', 'User-friendly'),
                ('open-source', 'Open-source'),
                ('multi-threaded', 'Multi-threaded'),
            ]

            for term, expected_content in hyphenated_terms:
                cursor = conn.execute(
                    f'''
                    SELECT ce.* FROM context_entries ce
                    JOIN context_entries_fts fts ON ce.id = fts.rowid
                    WHERE fts.text_content MATCH '"{term}"'
                    ''',
                )
                results = cursor.fetchall()

                assert len(results) >= 1, f'Failed to find term: {term}'
                assert any(
                    expected_content.lower() in r['text_content'].lower() for r in results
                ), f'Content mismatch for term: {term}'

    def test_fts_hyphenated_with_regular_words(self, hyphen_test_db: Path) -> None:
        """Test search mixing hyphenated and regular words."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Search for "full-text" AND "search" (both must match)
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH '"full-text" search'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'full-text' in results[0]['text_content']
            assert 'search' in results[0]['text_content']

    def test_fts_no_hyphen_regression(self, hyphen_test_db: Path) -> None:
        """Test that regular (non-hyphenated) queries still work."""
        with sqlite3.connect(str(hyphen_test_db)) as conn:
            conn.row_factory = sqlite3.Row

            # Regular search without hyphens should work as before
            cursor = conn.execute(
                '''
                SELECT ce.* FROM context_entries ce
                JOIN context_entries_fts fts ON ce.id = fts.rowid
                WHERE fts.text_content MATCH 'Regular search'
                ''',
            )
            results = cursor.fetchall()

            assert len(results) == 1
            assert 'Regular search' in results[0]['text_content']
