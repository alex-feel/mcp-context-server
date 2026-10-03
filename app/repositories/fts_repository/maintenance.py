"""Full-text search availability, statistics, and tokenizer and language migrations."""

import sqlite3
from typing import TYPE_CHECKING
from typing import Any
from typing import cast

from app.access_scope import AccessMode
from app.access_scope import Scope
from app.access_scope import build_access_predicate
from app.repositories.base import BaseRepository
from app.repositories.fts_repository.query import desired_sqlite_fts_tokenizer

if TYPE_CHECKING:
    import asyncpg


class FtsMaintenanceMixin(BaseRepository):
    """Availability, statistics and index maintenance for full-text search.

    Reports whether the FTS index exists and how many of the entries a scope may
    read it covers, inspects the tokenizer or language the index was built with,
    and migrates the SQLite FTS5 tokenizer or the PostgreSQL tsvector language
    when the configured ``FTS_LANGUAGE`` changes.
    """

    async def get_statistics(self, *, scope: Scope) -> dict[str, Any]:
        """Get FTS index statistics over the entries the scope may read.

        Both counts apply the READ predicate. The system scope counts every entry; the
        FTS migration uses it to size its rebuild estimate outside any request.

        Args:
            scope: The caller's scope, or the system scope.

        Returns:
            Dictionary with statistics (entry count, index info)
        """
        predicate = build_access_predicate(
            scope, mode=AccessMode.READ, backend_type=self.backend.backend_type, outer='context_entries',
        )
        total_sql = f'SELECT COUNT(*) FROM context_entries{predicate.where_clause()}'

        def _figures(total_count: int, indexed_count: int, backend: str, engine: str) -> dict[str, Any]:
            return {
                'total_entries': total_count,
                'indexed_entries': indexed_count,
                'coverage_percentage': round((indexed_count / total_count * 100) if total_count > 0 else 0.0, 2),
                'backend': backend,
                'engine': engine,
            }

        if self.backend.backend_type == 'sqlite':
            parents = build_access_predicate(scope, mode=AccessMode.READ, backend_type='sqlite', outer='ce')
            indexed_sql = (
                'SELECT COUNT(*) FROM context_entries_fts fts '
                f'JOIN context_entries ce ON ce.rowid_int = fts.rowid{parents.where_clause()}'
            )

            def _get_stats_sqlite(conn: sqlite3.Connection) -> dict[str, Any]:
                indexed_count = conn.execute(indexed_sql, parents.params).fetchone()[0]
                total_count = conn.execute(total_sql, predicate.params).fetchone()[0]
                return _figures(total_count, indexed_count, 'sqlite', 'fts5')

            return await self.backend.execute_read(_get_stats_sqlite)

        # postgresql: an entry is indexed once its tsvector is populated
        indexed_sql = (
            f'SELECT COUNT(*) FROM context_entries WHERE text_search_vector IS NOT NULL{predicate.and_clause()}'
        )

        async def _get_stats_postgresql(conn: 'asyncpg.Connection') -> dict[str, Any]:
            indexed_count = await conn.fetchval(indexed_sql, *predicate.params)
            total_count = await conn.fetchval(total_sql, *predicate.params)
            return _figures(total_count, indexed_count, 'postgresql', 'tsvector')

        return await self.backend.execute_read(cast(Any, _get_stats_postgresql))

    async def is_available(self) -> bool:
        """Check if FTS functionality is available.

        Both probes test the ABSENCE CONDITION directly and let every operational fault
        propagate. A blanket ``except Exception: return False`` here would report lock
        contention (SQLITE_BUSY held by an external VACUUM/backup), a malformed database
        image, or a permission failure as "FTS not migrated" -- consumed before
        ``execute_read``'s bounded locked-retry loop could retry it, stripped of circuit-
        breaker accounting, and surfaced to the operator as a "restart the server to apply
        migrations" instruction for a self-clearing transient. Neither handler was needed for
        its stated purpose either: ``sqlite_master`` always exists and yields ZERO ROWS for a
        missing table, and ``to_regclass`` yields NULL (never an error) for a missing relation.

        Returns:
            True if FTS is properly configured and available
        """
        if self.backend.backend_type == 'sqlite':

            def _check_sqlite(conn: sqlite3.Connection) -> bool:
                # Check if FTS5 table exists. sqlite_master is a catalog table present in
                # every database, so a missing context_entries_fts is simply an empty result.
                cursor = conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='context_entries_fts'",
                )
                return cursor.fetchone() is not None

            return await self.backend.execute_read(_check_sqlite)

        # postgresql
        async def _check_postgresql(conn: 'asyncpg.Connection') -> bool:
            # Resolve context_entries through the connection search_path
            # (to_regclass), then check THAT relation for the column via
            # pg_attribute. A schema-blind information_schema.columns query
            # (no table_schema filter) reported the column present when it
            # existed in ANY visible schema -- e.g. a colliding
            # public.context_entries -- so the FTS backstop believed the
            # configured target schema already had FTS and no-oped, leaving
            # migrated rows without full-text search under a non-default
            # POSTGRESQL_SCHEMA. Resolving by regclass matches exactly where
            # the FTS reads/writes and get_current_language() resolve.
            # to_regclass returns NULL for a missing/invisible relation, so the
            # EXISTS is false without raising -- the absence case needs no handler.
            result = await conn.fetchval(
                '''
                SELECT EXISTS (
                    SELECT 1 FROM pg_attribute
                    WHERE attrelid = to_regclass('context_entries')
                      AND attname = 'text_search_vector'
                      AND NOT attisdropped
                )
                ''',
            )
            return bool(result)

        return await self.backend.execute_read(cast(Any, _check_postgresql))

    async def get_current_tokenizer(self) -> str | None:
        """Get the current FTS5 tokenizer from SQLite (SQLite only).

        Parses the sqlite_master table to extract the tokenizer definition
        from the FTS5 virtual table creation SQL.

        Returns:
            The tokenizer string (e.g., 'unicode61' or 'porter unicode61'),
            or None if FTS5 table doesn't exist or backend is not SQLite.
        """
        if self.backend.backend_type != 'sqlite':
            return None

        def _get_tokenizer(conn: sqlite3.Connection) -> str | None:
            cursor = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='context_entries_fts'",
            )
            row = cursor.fetchone()
            if not row:
                return None

            # Parse the SQL to extract tokenizer
            # Example SQL: "CREATE VIRTUAL TABLE context_entries_fts USING fts5(..., tokenize='porter unicode61')"
            sql = row[0]
            if 'tokenize=' not in sql.lower():
                return 'unicode61'  # Default if not specified

            # Extract tokenizer value using string parsing
            # Find tokenize= and extract the quoted value
            import re

            # Pattern matches tokenize='...' or tokenize="..."
            pattern = r"tokenize\s*=\s*['\"]([^'\"]+)['\"]"
            match = re.search(pattern, sql, re.IGNORECASE)
            if match:
                return match.group(1)

            return 'unicode61'  # Default fallback

        return await self.backend.execute_read(_get_tokenizer)

    async def get_current_language(self) -> str | None:
        """Get the current FTS language from PostgreSQL tsvector column (PostgreSQL only).

        Queries pg_attrdef to decompile the GENERATED ALWAYS AS expression
        and extracts the language parameter from to_tsvector().

        Returns:
            The language string (e.g., 'english', 'german'),
            or None if tsvector column doesn't exist or backend is not PostgreSQL.
        """
        if self.backend.backend_type != 'postgresql':
            return None

        async def _get_language(conn: 'asyncpg.Connection') -> str | None:
            # Query to get the generation expression for text_search_vector column
            result = await conn.fetchval(
                '''
                SELECT pg_get_expr(ad.adbin, ad.adrelid) AS generation_expression
                FROM pg_attribute a
                JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
                WHERE a.attrelid = 'context_entries'::regclass
                  AND a.attname = 'text_search_vector'
                  AND a.attgenerated = 's'
                ''',
            )
            if not result:
                return None

            # Parse the expression to extract language
            # Example: "to_tsvector('english'::regconfig, COALESCE(text_content, ''::text))"
            import re

            # Pattern matches to_tsvector('language'::regconfig, ...) or to_tsvector('language', ...)
            pattern = r"to_tsvector\s*\(\s*'([^']+)'"
            match = re.search(pattern, result, re.IGNORECASE)
            if match:
                return match.group(1)

            return 'english'  # Default fallback

        return await self.backend.execute_read(cast(Any, _get_language))

    async def get_desired_tokenizer(self, language: str) -> str:
        """Determine the desired SQLite FTS5 tokenizer based on language setting.

        Based on the research: English benefits from Porter stemmer, other languages
        should use unicode61 for proper multilingual tokenization.

        Args:
            language: The FTS_LANGUAGE setting value

        Returns:
            The tokenizer string to use ('porter unicode61' or 'unicode61')
        """
        return desired_sqlite_fts_tokenizer(language)

    async def migrate_tokenizer(self, new_tokenizer: str) -> dict[str, Any]:
        """Migrate SQLite FTS5 to a new tokenizer (SQLite only).

        This operation drops the existing FTS5 virtual table and recreates it
        with the new tokenizer. The data is NOT lost because FTS5 uses external
        content mode (content='context_entries').

        Args:
            new_tokenizer: The new tokenizer to use (e.g., 'porter unicode61' or 'unicode61')

        Returns:
            Dictionary with migration results

        Raises:
            RuntimeError: If migration fails or backend is not SQLite
        """
        if self.backend.backend_type != 'sqlite':
            raise RuntimeError('migrate_tokenizer is only supported for SQLite backend')

        old_tokenizer = await self.get_current_tokenizer()

        def _migrate_tokenizer(conn: sqlite3.Connection) -> dict[str, Any]:
            # Count entries for statistics
            cursor = conn.execute('SELECT COUNT(*) FROM context_entries')
            entry_count = cursor.fetchone()[0]

            # Drop existing FTS5 table and triggers
            conn.execute('DROP TRIGGER IF EXISTS context_fts_insert')
            conn.execute('DROP TRIGGER IF EXISTS context_fts_delete')
            conn.execute('DROP TRIGGER IF EXISTS context_fts_update')
            conn.execute('DROP TABLE IF EXISTS context_entries_fts')

            # Recreate FTS5 table with new tokenizer.
            # The FTS5 ``content_rowid`` MUST point at an INTEGER PRIMARY KEY
            # column; ``context_entries.rowid_int`` is the SQLite private
            # surrogate used for this purpose, while ``context_entries.id`` is
            # the public UUIDv7 hex value exchanged across the MCP boundary.
            create_sql = f'''
                CREATE VIRTUAL TABLE context_entries_fts USING fts5(
                    text_content,
                    content='context_entries',
                    content_rowid='rowid_int',
                    tokenize='{new_tokenizer}'
                )
            '''
            conn.execute(create_sql)

            # Recreate triggers using ``rowid_int`` as the FTS5 rowid alias.
            conn.execute('''
                CREATE TRIGGER context_fts_insert AFTER INSERT ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(rowid, text_content)
                    VALUES (new.rowid_int, new.text_content);
                END
            ''')

            conn.execute('''
                CREATE TRIGGER context_fts_delete AFTER DELETE ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
                    VALUES('delete', old.rowid_int, old.text_content);
                END
            ''')

            conn.execute('''
                CREATE TRIGGER context_fts_update AFTER UPDATE OF text_content ON context_entries
                BEGIN
                    INSERT INTO context_entries_fts(context_entries_fts, rowid, text_content)
                    VALUES('delete', old.rowid_int, old.text_content);
                    INSERT INTO context_entries_fts(rowid, text_content)
                    VALUES (new.rowid_int, new.text_content);
                END
            ''')

            # Rebuild the FTS index from existing data
            conn.execute("INSERT INTO context_entries_fts(context_entries_fts) VALUES('rebuild')")

            return {
                'success': True,
                'backend': 'sqlite',
                'old_tokenizer': old_tokenizer,
                'new_tokenizer': new_tokenizer,
                'entries_migrated': entry_count,
            }

        return await self.backend.execute_write(_migrate_tokenizer)

    async def migrate_language(self, new_language: str) -> dict[str, Any]:
        """Migrate PostgreSQL tsvector column to a new language (PostgreSQL only).

        This operation drops the existing text_search_vector column (and its GIN index)
        and recreates it with the new language. The GENERATED ALWAYS AS column is
        automatically populated from text_content on recreation.

        Args:
            new_language: The new language for tsvector (e.g., 'english', 'german')

        Returns:
            Dictionary with migration results

        Raises:
            RuntimeError: If migration fails or backend is not PostgreSQL
        """
        if self.backend.backend_type != 'postgresql':
            raise RuntimeError('migrate_language is only supported for PostgreSQL backend')

        # Import locally to avoid a repository->migrations import at module load.
        from app.migrations._pg_ddl import begin_migration
        from app.migrations._pg_ddl import execute_migration_ddl
        from app.migrations._pg_ddl import fetchval_migration
        from app.settings import get_settings

        migration_timeout_s = get_settings().storage.postgresql_migration_timeout_s
        old_language = await self.get_current_language()

        async def _migrate_language(conn: 'asyncpg.Connection') -> dict[str, Any]:
            # Raise the transaction-scoped statement_timeout to the migration budget and
            # take the shared advisory lock under it BEFORE the rewrite. Recreating the
            # GENERATED ALWAYS AS (...) STORED tsvector column is the heaviest DDL of all
            # -- a full table rewrite plus a GIN index build -- so on a large existing
            # table it (and a lock wait on a peer pod's migration) must not be cancelled
            # at the pool's shorter command_timeout. SET LOCAL auto-reverts on
            # COMMIT/ROLLBACK, so no finally-restore (which would raise 25P02 in an
            # aborted transaction and mask the real DDL error) is used.
            await begin_migration(conn, migration_timeout_s)

            # Count entries for statistics
            entry_count = await fetchval_migration(conn, 'SELECT COUNT(*) FROM context_entries', migration_timeout_s)

            # Drop existing column (also drops dependent GIN index)
            await execute_migration_ddl(
                conn,
                'ALTER TABLE context_entries DROP COLUMN IF EXISTS text_search_vector',
                migration_timeout_s,
            )

            # Recreate column with new language
            await execute_migration_ddl(
                conn,
                f'''
                ALTER TABLE context_entries
                ADD COLUMN text_search_vector tsvector
                GENERATED ALWAYS AS (to_tsvector('{new_language}', COALESCE(text_content, ''))) STORED
            ''',
                migration_timeout_s,
            )

            # Recreate GIN index
            await execute_migration_ddl(
                conn,
                '''
                CREATE INDEX IF NOT EXISTS idx_text_search_gin
                ON context_entries USING GIN(text_search_vector)
            ''',
                migration_timeout_s,
            )

            return {
                'success': True,
                'backend': 'postgresql',
                'old_language': old_language,
                'new_language': new_language,
                'entries_migrated': entry_count,
            }

        return await self.backend.execute_write(cast(Any, _migrate_language))
