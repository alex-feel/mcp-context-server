"""
Test suite for schema synchronization between file and embedded schemas.

This test ensures that the embedded fallback schema in app/server.py remains
synchronized with the primary schema file app/schema.sql. The test uses database
introspection to compare the resulting database structures, which is resilient
to formatting differences while catching semantic changes.
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


def extract_embedded_schema_from_server() -> str:
    """
    Extract the embedded fallback schema from app/server.py.

    Parses the server.py file to extract the schema SQL defined in the
    init_database() function's fallback branch (lines 370-445).

    Returns:
        Embedded schema SQL as a string

    Raises:
        RuntimeError: If schema cannot be extracted from server.py
    """
    server_path = Path(__file__).parent.parent / 'app' / 'server.py'

    if not server_path.exists():
        raise RuntimeError(f'Server file not found: {server_path}')

    server_content = server_path.read_text(encoding='utf-8')

    # Find the embedded schema using regex
    # Looking for: schema_sql = '''...'''
    pattern = r"schema_sql\s*=\s*'''(.*?)'''"
    match = re.search(pattern, server_content, re.DOTALL)

    if not match:
        raise RuntimeError('Could not find embedded schema in server.py')

    embedded_schema = match.group(1)

    # Remove any leading/trailing whitespace but preserve the schema structure
    return embedded_schema.strip()


class TestSchemaSynchronization:
    """Test schema synchronization between file and embedded schemas."""

    def test_schema_file_matches_embedded_schema(self) -> None:
        """
        Test that the embedded fallback schema matches the file schema.

        This test creates two in-memory SQLite databases:
        1. One initialized with the schema from app/schema.sql
        2. One initialized with the embedded schema from app/server.py

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
        # Read schema from file
        schema_file_path = Path(__file__).parent.parent / 'app' / 'schema.sql'
        if not schema_file_path.exists():
            pytest.fail(f'Schema file not found: {schema_file_path}')

        file_schema_sql = schema_file_path.read_text(encoding='utf-8')

        # Extract embedded schema from server.py
        embedded_schema_sql = extract_embedded_schema_from_server()

        # Create in-memory database with file schema
        conn_file = sqlite3.connect(':memory:')
        try:
            conn_file.executescript(file_schema_sql)
            file_structure = extract_schema_structure(conn_file)
        finally:
            conn_file.close()

        # Create in-memory database with embedded schema
        conn_embedded = sqlite3.connect(':memory:')
        try:
            conn_embedded.executescript(embedded_schema_sql)
            embedded_structure = extract_schema_structure(conn_embedded)
        finally:
            conn_embedded.close()

        # Compare table names
        file_table_names = {name for name, _ in file_structure['tables']}
        embedded_table_names = {name for name, _ in embedded_structure['tables']}

        if file_table_names != embedded_table_names:
            missing_in_embedded = file_table_names - embedded_table_names
            extra_in_embedded = embedded_table_names - file_table_names
            error_msg = 'Schema synchronization failed - Table names mismatch!\n\n'
            if missing_in_embedded:
                error_msg += f'Tables in file schema but missing in embedded: {sorted(missing_in_embedded)}\n'
            if extra_in_embedded:
                error_msg += f'Tables in embedded schema but not in file: {sorted(extra_in_embedded)}\n'
            pytest.fail(error_msg)

        # Compare table definitions
        file_tables_dict = dict(file_structure['tables'])
        embedded_tables_dict = dict(embedded_structure['tables'])

        table_diffs = []
        for table_name in sorted(file_table_names):
            file_sql = normalize_sql(file_tables_dict[table_name])
            embedded_sql = normalize_sql(embedded_tables_dict[table_name])

            if file_sql != embedded_sql:
                table_diffs.append(
                    f'\nTable: {table_name}\n'
                    f'  File schema SQL:\n    {file_sql}\n'
                    f'  Embedded schema SQL:\n    {embedded_sql}\n',
                )

        if table_diffs:
            error_msg = 'Schema synchronization failed - Table definitions differ!\n'
            error_msg += '\n'.join(table_diffs)
            pytest.fail(error_msg)

        # Compare index names
        file_index_names = {name for name, _ in file_structure['indexes']}
        embedded_index_names = {name for name, _ in embedded_structure['indexes']}

        if file_index_names != embedded_index_names:
            missing_in_embedded = file_index_names - embedded_index_names
            extra_in_embedded = embedded_index_names - file_index_names
            error_msg = 'Schema synchronization failed - Index names mismatch!\n\n'
            if missing_in_embedded:
                error_msg += f'Indexes in file schema but missing in embedded: {sorted(missing_in_embedded)}\n'
            if extra_in_embedded:
                error_msg += f'Indexes in embedded schema but not in file: {sorted(extra_in_embedded)}\n'
            pytest.fail(error_msg)

        # Compare index definitions
        file_indexes_dict = dict(file_structure['indexes'])
        embedded_indexes_dict = dict(embedded_structure['indexes'])

        index_diffs = []
        for index_name in sorted(file_index_names):
            file_sql = normalize_sql(file_indexes_dict[index_name])
            embedded_sql = normalize_sql(embedded_indexes_dict[index_name])

            if file_sql != embedded_sql:
                index_diffs.append(
                    f'\nIndex: {index_name}\n'
                    f'  File schema SQL:\n    {file_sql}\n'
                    f'  Embedded schema SQL:\n    {embedded_sql}\n',
                )

        if index_diffs:
            error_msg = 'Schema synchronization failed - Index definitions differ!\n'
            error_msg += '\n'.join(index_diffs)
            pytest.fail(error_msg)

        # If we got here, schemas are synchronized
        print('\nSchema synchronization validated successfully!')
        print(f'  Tables: {len(file_table_names)}')
        print(f'  Indexes: {len(file_index_names)}')
