"""Positive assertion: free-form text mentioning ID references is NEVER auto-rewritten.

Free-form text content in ``text_content`` and ``summary`` columns is treated
as opaque user content and is never rewritten by any production code path --
even when the text contains substrings that look like ID references. This
test pins that behavior to prevent future regressions where an automatic
ID rewriter would corrupt user content.
"""

import sqlite3
from collections.abc import AsyncGenerator
from pathlib import Path

import pytest
import pytest_asyncio

from app.backends import create_backend
from app.ids import generate_id
from app.repositories import RepositoryContainer

GHOST_TEXT = (
    'See entries 8944 and 9044, plus 14226 in the broader context. '
    'The numeric tokens above are inert ghost references in free-form text '
    'and must remain unchanged in storage.'
)


@pytest_asyncio.fixture
async def repos(tmp_path: Path) -> AsyncGenerator[RepositoryContainer, None]:
    db_path = tmp_path / 'test.db'
    from app.schemas import load_schema

    schema_sql = load_schema('sqlite')
    conn = sqlite3.connect(str(db_path))
    conn.executescript(schema_sql)
    conn.close()

    backend = create_backend(backend_type='sqlite', db_path=str(db_path))
    await backend.initialize()
    yield RepositoryContainer(backend)
    await backend.shutdown()


class TestGhostReferencesNotRewritten:
    """Free-form text containing integer-style ID mentions is stored verbatim."""

    @pytest.mark.asyncio
    async def test_store_preserves_integer_mentions_in_text_content(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """``store_with_deduplication`` does not rewrite any substring in text_content."""
        context_id, _ = await repos.context.store_with_deduplication(
            thread_id='t', source='user', content_type='text',
            text_content=GHOST_TEXT, metadata=None,
        )
        entries = await repos.context.get_by_ids([context_id])
        assert len(entries) == 1
        assert entries[0]['text_content'] == GHOST_TEXT

    @pytest.mark.asyncio
    async def test_update_preserves_integer_mentions_in_text_content(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """``update_context_entry`` writes text_content verbatim without rewriting."""
        original_id, _ = await repos.context.store_with_deduplication(
            thread_id='t', source='user', content_type='text',
            text_content='Old content', metadata=None,
        )
        async with repos.context.backend.begin_transaction() as txn:
            success, _ = await repos.context.update_context_entry(
                context_id=original_id,
                text_content=GHOST_TEXT,
                txn=txn,
            )
        assert success
        entries = await repos.context.get_by_ids([original_id])
        assert entries[0]['text_content'] == GHOST_TEXT

    @pytest.mark.asyncio
    async def test_summary_column_preserves_integer_mentions(
        self,
        repos: RepositoryContainer,
    ) -> None:
        """Summaries stored via the repository layer are not rewritten either."""
        new_id = generate_id()

        def _insert(conn: sqlite3.Connection) -> None:
            conn.execute(
                '''INSERT INTO context_entries
                   (id, thread_id, source, content_type, text_content, summary)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (new_id, 't', 'user', 'text', 'irrelevant', GHOST_TEXT),
            )

        await repos.context.backend.execute_write(_insert)

        entries = await repos.context.get_by_ids([new_id])
        assert len(entries) == 1
        assert entries[0]['summary'] == GHOST_TEXT
