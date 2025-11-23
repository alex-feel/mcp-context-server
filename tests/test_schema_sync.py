"""
Test suite for schema synchronization between legacy and new schema files.

This test ensures that the legacy app/schema.sql remains synchronized with
the new app/schemas/sqlite_schema.sql. The test uses database introspection
to compare the resulting database structures, which is resilient to formatting
differences while catching semantic changes.
"""

import re
import sqlite3
from pathlib import Path

import pytest


def extract_schema_structure(conn: sqlite3.Connection) -> dict[str, list[tuple[str, str | None]]]:
    """
    Extract normalized schema structure from a SQLite database.

    Uses sqlite_master system table to query the database structure. This approach
    compares the actual database objects created by the schemas, not the SQL text,
    making it resilient to formatting differences while detecting semantic changes.

    Args:
        conn: SQLite database connection to extract schema from

    Returns:
        Dictionary containing:
            - 'tables': List of (name, sql) tuples for table definitions
            - 'indexes': List of (name, sql) tuples for index definitions
    """
    cursor = conn.cursor()
    cursor.execute('''
        SELECT type, name, sql
        FROM sqlite_master
        WHERE type IN ('table', 'index')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type, name
    ''')

    rows = cursor.fetchall()
    structure: dict[str, list[tuple[str, str | None]]] = {
        'tables': [],
        'indexes': [],
    }

    for row in rows:
        type_, name, sql = row
        if type_ == 'table':
            structure['tables'].append((name, sql))
        elif type_ == 'index':
            structure['indexes'].append((name, sql))

    return structure


def normalize_sql(sql: str | None) -> str:
    """
    Normalize SQL for comparison by removing extra whitespace.

    This function makes the comparison resilient to formatting differences
    while still catching semantic changes in the schema definitions.

    Args:
        sql: SQL statement to normalize

    Returns:
        Normalized SQL string with consistent whitespace
    """
    if sql is None:
        return ''

    # Remove comments (-- style)
    sql = re.sub(r'--[^\n]*', '', sql)

    # Collapse multiple whitespace into single space
    sql = re.sub(r'\s+', ' ', sql)

    # Remove leading/trailing whitespace and return
    return sql.strip()


class TestSchemaSynchronization:
    """Test schema synchronization between legacy and new schema files."""

    def test_schema_file_matches_sqlite_schema(self) -> None:
        """
        Test that the legacy app/schema.sql matches app/schemas/sqlite_schema.sql.

        This test creates two in-memory SQLite databases:
        1. One initialized with the schema from app/schema.sql (legacy)
        2. One initialized with the schema from app/schemas/sqlite_schema.sql (new)

        It then compares the resulting database structures using sqlite_master
        introspection. This approach ensures semantic equivalence while being
        resilient to formatting differences like whitespace and comments.

        The test catches:
        - Missing or extra tables
        - Different table definitions (columns, types, constraints)
        - Missing or extra indexes
        - Different index definitions

        The test ignores:
        - Whitespace differences
        - Comment differences
        - Statement ordering (if semantically equivalent)
        """
        # Read legacy schema
        legacy_schema_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
        if not legacy_schema_path.exists():
            pytest.fail(f'Legacy schema file not found: {legacy_schema_path}')

        legacy_schema_sql = legacy_schema_path.read_text(encoding='utf-8')

        # Read new SQLite schema
        new_schema_path = Path(__file__).parent.parent / 'app' / 'schemas' / 'sqlite_schema.sql'
        if not new_schema_path.exists():
            pytest.fail(f'New schema file not found: {new_schema_path}')

        new_schema_sql = new_schema_path.read_text(encoding='utf-8')

        # Create in-memory database with legacy schema
        conn_legacy = sqlite3.connect(':memory:')
        try:
            conn_legacy.executescript(legacy_schema_sql)
            legacy_structure = extract_schema_structure(conn_legacy)
        finally:
            conn_legacy.close()

        # Create in-memory database with new schema
        conn_new = sqlite3.connect(':memory:')
        try:
            conn_new.executescript(new_schema_sql)
            new_structure = extract_schema_structure(conn_new)
        finally:
            conn_new.close()

        # Compare table names
        legacy_table_names = {name for name, _ in legacy_structure['tables']}
        new_table_names = {name for name, _ in new_structure['tables']}

        if legacy_table_names != new_table_names:
            missing_in_new = legacy_table_names - new_table_names
            extra_in_new = new_table_names - legacy_table_names
            error_msg = 'Schema synchronization failed - Table names mismatch!\n\n'
            if missing_in_new:
                error_msg += f'Tables in legacy schema but missing in new: {sorted(missing_in_new)}\n'
            if extra_in_new:
                error_msg += f'Tables in new schema but not in legacy: {sorted(extra_in_new)}\n'
            pytest.fail(error_msg)

        # Compare table definitions
        legacy_tables_dict = dict(legacy_structure['tables'])
        new_tables_dict = dict(new_structure['tables'])

        table_diffs = []
        for table_name in sorted(legacy_table_names):
            legacy_sql = normalize_sql(legacy_tables_dict[table_name])
            new_sql = normalize_sql(new_tables_dict[table_name])

            if legacy_sql != new_sql:
                table_diffs.append(
                    f'\nTable: {table_name}\n  Legacy schema SQL:\n    {legacy_sql}\n  New schema SQL:\n    {new_sql}\n',
                )

        if table_diffs:
            error_msg = 'Schema synchronization failed - Table definitions differ!\n'
            error_msg += '\n'.join(table_diffs)
            pytest.fail(error_msg)

        # Compare index names
        legacy_index_names = {name for name, _ in legacy_structure['indexes']}
        new_index_names = {name for name, _ in new_structure['indexes']}

        if legacy_index_names != new_index_names:
            missing_in_new = legacy_index_names - new_index_names
            extra_in_new = new_index_names - legacy_index_names
            error_msg = 'Schema synchronization failed - Index names mismatch!\n\n'
            if missing_in_new:
                error_msg += f'Indexes in legacy schema but missing in new: {sorted(missing_in_new)}\n'
            if extra_in_new:
                error_msg += f'Indexes in new schema but not in legacy: {sorted(extra_in_new)}\n'
            pytest.fail(error_msg)

        # Compare index definitions
        legacy_indexes_dict = dict(legacy_structure['indexes'])
        new_indexes_dict = dict(new_structure['indexes'])

        index_diffs = []
        for index_name in sorted(legacy_index_names):
            legacy_sql = normalize_sql(legacy_indexes_dict[index_name])
            new_sql = normalize_sql(new_indexes_dict[index_name])

            if legacy_sql != new_sql:
                index_diffs.append(
                    f'\nIndex: {index_name}\n  Legacy schema SQL:\n    {legacy_sql}\n  New schema SQL:\n    {new_sql}\n',
                )

        if index_diffs:
            error_msg = 'Schema synchronization failed - Index definitions differ!\n'
            error_msg += '\n'.join(index_diffs)
            pytest.fail(error_msg)

        # If we got here, schemas are synchronized
        print('\nSchema synchronization validated successfully!')
        print(f'  Tables: {len(legacy_table_names)}')
        print(f'  Indexes: {len(legacy_index_names)}')
