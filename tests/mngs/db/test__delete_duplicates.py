#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-02 13:10:00 (Claude)"
# File: /tests/mngs/db/test__delete_duplicates.py

import os
import sys
import tempfile
import shutil
import pytest
import sqlite3
import pandas as pd
from unittest.mock import patch, MagicMock

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from mngs.db._delete_duplicates import (
    delete_duplicates,
    _sort_db,
    _determine_columns,
    _fetch_as_df,
    _find_duplicated,
    verify_duplicated_index,
    _delete_entry
)


class TestDeleteDuplicates:
    """Test cases for delete_duplicates functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test databases."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)

    @pytest.fixture
    def db_path(self, temp_dir):
        """Get a temporary database path."""
        return os.path.join(temp_dir, "test.db")

    @pytest.fixture
    def db_with_duplicates(self, db_path):
        """Create a database with duplicate entries."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Create table
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL,
                data BLOB
            )
        """)
        
        # Insert data with duplicates
        data = [
            (1, "Alice", 100.0, b"data1"),
            (2, "Bob", 200.0, b"data2"),
            (3, "Alice", 100.0, b"data1"),  # Duplicate of row 1
            (4, "Charlie", 300.0, b"data3"),
            (5, "Bob", 200.0, b"data2"),  # Duplicate of row 2
            (6, "Alice", 100.0, b"data1"),  # Another duplicate of row 1
        ]
        
        cursor.executemany(
            "INSERT INTO test_table VALUES (?, ?, ?, ?)",
            data
        )
        conn.commit()
        conn.close()
        
        return db_path

    def test_delete_duplicates_basic(self, db_with_duplicates):
        """Test basic duplicate deletion functionality."""
        # Act
        total_processed, total_duplicates = delete_duplicates(
            db_with_duplicates,
            "test_table",
            columns=["name", "value"],
            dry_run=False
        )
        
        # Assert
        assert total_processed is not None
        assert total_duplicates is not None
        assert total_duplicates == 3  # We have 3 duplicate rows
        
        # Verify database state
        conn = sqlite3.connect(db_with_duplicates)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_table")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 3  # Should have only 3 unique entries

    def test_delete_duplicates_dry_run(self, db_with_duplicates):
        """Test dry run mode doesn't modify database."""
        # Get initial count
        conn = sqlite3.connect(db_with_duplicates)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_table")
        initial_count = cursor.fetchone()[0]
        conn.close()
        
        # Act
        delete_duplicates(
            db_with_duplicates,
            "test_table",
            columns=["name", "value"],
            dry_run=True
        )
        
        # Assert - count should remain the same
        conn = sqlite3.connect(db_with_duplicates)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_table")
        final_count = cursor.fetchone()[0]
        conn.close()
        
        assert final_count == initial_count

    def test_delete_duplicates_all_columns(self, db_with_duplicates):
        """Test deleting duplicates considering all columns."""
        # Act
        total_processed, total_duplicates = delete_duplicates(
            db_with_duplicates,
            "test_table",
            columns="all",
            include_blob=True,
            dry_run=False
        )
        
        # Assert
        assert total_duplicates == 3  # Same duplicates when considering all columns

    def test_delete_duplicates_exclude_blob(self, db_with_duplicates):
        """Test deleting duplicates excluding BLOB columns."""
        # Act
        total_processed, total_duplicates = delete_duplicates(
            db_with_duplicates,
            "test_table",
            columns="all",
            include_blob=False,  # Exclude BLOB columns
            dry_run=False
        )
        
        # Assert
        assert total_duplicates == 3  # Should still find duplicates in non-BLOB columns

    def test_sort_db(self, db_path):
        """Test database sorting functionality."""
        # Setup
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        
        # Insert unordered data
        data = [
            (1, "Charlie", 300.0),
            (2, "Alice", 100.0),
            (3, "Bob", 200.0),
        ]
        cursor.executemany("INSERT INTO test_table VALUES (?, ?, ?)", data)
        conn.commit()
        
        # Act
        _sort_db(cursor, "test_table", ["name"])
        conn.commit()
        
        # Assert - check order
        cursor.execute("SELECT name FROM test_table")
        names = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        assert names == ["Alice", "Bob", "Charlie"]

    def test_determine_columns(self, db_path):
        """Test column determination logic."""
        # Setup
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL,
                data BLOB
            )
        """)
        conn.commit()
        
        # Test "all" with include_blob=True
        columns = _determine_columns(cursor, "test_table", "all", include_blob=True)
        assert "id" in columns
        assert "name" in columns
        assert "value" in columns
        assert "data" in columns
        
        # Test "all" with include_blob=False
        columns = _determine_columns(cursor, "test_table", "all", include_blob=False)
        assert "data" not in columns  # BLOB column should be excluded
        
        # Test specific columns
        columns = _determine_columns(cursor, "test_table", ["name", "value"], include_blob=False)
        assert columns == ["name", "value"]
        
        # Test single column as string
        columns = _determine_columns(cursor, "test_table", "name", include_blob=False)
        assert columns == ["name"]
        
        conn.close()

    def test_fetch_as_df(self, db_with_duplicates):
        """Test fetching data as DataFrame."""
        # Setup
        conn = sqlite3.connect(db_with_duplicates)
        cursor = conn.cursor()
        
        # Act
        df = _fetch_as_df(cursor, ["name", "value"], "test_table")
        
        # Assert
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # Total rows including duplicates
        assert list(df.columns) == ["name", "value"]
        
        conn.close()

    def test_find_duplicated(self):
        """Test finding duplicated entries in DataFrame."""
        # Setup
        data = {
            "name": ["Alice", "Bob", "Alice", "Charlie", "Bob"],
            "value": [100, 200, 100, 300, 200]
        }
        df = pd.DataFrame(data)
        
        # Act
        duplicates = _find_duplicated(df)
        
        # Assert
        assert len(duplicates) == 2  # Two duplicate rows
        assert duplicates.iloc[0]["name"] == "Alice"
        assert duplicates.iloc[1]["name"] == "Bob"

    def test_verify_duplicated_index(self, db_with_duplicates):
        """Test verification of duplicated entries."""
        # Setup
        conn = sqlite3.connect(db_with_duplicates)
        cursor = conn.cursor()
        
        duplicated_row = pd.Series({"name": "Alice", "value": 100.0})
        
        # Act
        query, is_verified = verify_duplicated_index(
            cursor, duplicated_row, "test_table", dry_run=False
        )
        
        # Assert
        assert is_verified is True
        assert "SELECT" in query
        assert "WHERE" in query
        
        conn.close()

    def test_delete_entry(self, db_with_duplicates):
        """Test deleting a single entry."""
        # Setup
        conn = sqlite3.connect(db_with_duplicates)
        cursor = conn.cursor()
        
        duplicated_row = pd.Series({"name": "Alice", "value": 100.0})
        
        # Get initial count
        cursor.execute("SELECT COUNT(*) FROM test_table WHERE name='Alice' AND value=100.0")
        initial_count = cursor.fetchone()[0]
        
        # Act - delete one entry
        _delete_entry(cursor, duplicated_row, "test_table", dry_run=False)
        conn.commit()
        
        # Assert
        cursor.execute("SELECT COUNT(*) FROM test_table WHERE name='Alice' AND value=100.0")
        final_count = cursor.fetchone()[0]
        
        assert final_count == initial_count - 1
        
        conn.close()

    def test_delete_duplicates_with_chunks(self, db_path):
        """Test deleting duplicates with chunked processing."""
        # Create larger dataset
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        
        # Insert many rows with duplicates
        data = []
        for i in range(100):
            # Create pattern with duplicates
            name = f"Person_{i % 20}"  # Will create duplicates
            value = float(i % 10)  # Will create more duplicates
            data.append((i, name, value))
        
        cursor.executemany("INSERT INTO test_table VALUES (?, ?, ?)", data)
        conn.commit()
        conn.close()
        
        # Act
        total_processed, total_duplicates = delete_duplicates(
            db_path,
            "test_table",
            columns=["name", "value"],
            chunk_size=10,  # Small chunk size for testing
            dry_run=False
        )
        
        # Assert
        assert total_processed >= 100
        assert total_duplicates > 0
        
        # Verify unique entries
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT name || value) FROM test_table")
        unique_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM test_table")
        total_count = cursor.fetchone()[0]
        conn.close()
        
        assert total_count == unique_count  # All remaining should be unique

    def test_delete_duplicates_error_handling(self, temp_dir):
        """Test error handling in delete_duplicates."""
        # Test with non-existent database
        result = delete_duplicates(
            os.path.join(temp_dir, "non_existent.db"),
            "test_table",
            dry_run=False
        )
        
        assert result == (None, None)  # Should return None on error

    def test_delete_duplicates_empty_table(self, db_path):
        """Test deleting duplicates from empty table."""
        # Create empty table
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        conn.commit()
        conn.close()
        
        # Act
        total_processed, total_duplicates = delete_duplicates(
            db_path,
            "test_table",
            dry_run=False
        )
        
        # Assert
        assert total_processed == 0
        assert total_duplicates == 0

    def test_delete_duplicates_no_duplicates(self, db_path):
        """Test deleting duplicates when there are none."""
        # Create table with unique entries
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE test_table (
                id INTEGER PRIMARY KEY,
                name TEXT,
                value REAL
            )
        """)
        
        # Insert unique data
        data = [
            (1, "Alice", 100.0),
            (2, "Bob", 200.0),
            (3, "Charlie", 300.0),
        ]
        cursor.executemany("INSERT INTO test_table VALUES (?, ?, ?)", data)
        conn.commit()
        conn.close()
        
        # Act
        total_processed, total_duplicates = delete_duplicates(
            db_path,
            "test_table",
            columns=["name", "value"],
            dry_run=False
        )
        
        # Assert
        assert total_duplicates == 0
        
        # Verify no data was lost
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM test_table")
        count = cursor.fetchone()[0]
        conn.close()
        
        assert count == 3  # All original rows should remain


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])
<<<<<<< HEAD
=======

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_delete_duplicates.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-11 14:16:58 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/db/_delete_duplicates.py
# 
# import sqlite3
# from typing import List, Optional, Tuple, Union
# import pandas as pd
# 
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-20 02:17:10 (ywatanabe)"
# # /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_delete_duplicates_clean.py
# 
# 
# 
# """
# Functionality:
#     - Deletes duplicate entries from an SQLite database table
# Input:
#     - SQLite database file path, table name, columns to consider for duplicates
# Output:
#     - Updated SQLite database with duplicates removed
# Prerequisites:
#     - sqlite3, pandas, tqdm, mngs
# """
# 
# 
# 
# 
# def _sort_db(
#     cursor: sqlite3.Cursor,
#     table_name: str,
#     columns: List[str]
# ) -> None:
#     """
#     Sorts the database table based on the specified columns.
# 
#     Parameters
#     ----------
#     cursor : sqlite3.Cursor
#         The cursor object for executing SQL commands.
#     table_name : str
#         The name of the table to be sorted.
#     columns : List[str]
#         The list of column names to sort by, in order of priority.
# 
#     Example
#     -------
#     >>> conn = sqlite3.connect('example.db')
#     >>> cursor = conn.cursor()
#     >>> _sort_db(cursor, 'my_table', ['column1', 'column2'])
#     >>> conn.commit()
#     >>> conn.close()
#     """
#     columns_str = ', '.join(columns)
#     temp_table = f"{table_name}_temp"
# 
#     cursor.execute(f"CREATE TABLE {temp_table} AS SELECT * FROM {table_name} ORDER BY {columns_str}")
#     cursor.execute(f"DROP TABLE {table_name}")
#     cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
# 
# 
# def _determine_columns(
#     cursor: sqlite3.Cursor,
#     table_name: str,
#     columns: Union[str, List[str]],
#     include_blob: bool,
# ) -> List[str]:
#     cursor.execute(f"PRAGMA table_info({table_name})")
#     table_info = cursor.fetchall()
#     all_columns = [col[1] for col in table_info]
#     column_types = {col[1]: col[2] for col in table_info}
# 
#     if columns == "all":
#         columns = (
#             all_columns
#             if include_blob
#             else [
#                 col
#                 for col in all_columns
#                 if column_types[col].lower() != "blob"
#             ]
#         )
#     elif isinstance(columns, str):
#         columns = [columns]
# 
#     columns_str = ", ".join(columns)
#     print(f"Columns considered for duplicates: {columns_str}")
# 
#     return columns
# 
# def _fetch_as_df(
#     cursor: sqlite3.Cursor, columns: List[str], table_name: str
# ) -> pd.DataFrame:
#     print("\nFetching all database entries...")
#     columns_str = ", ".join(columns)
#     query = f"SELECT {columns_str} FROM {table_name}"
#     cursor.execute(query)
#     df_entries = cursor.fetchall()
#     return pd.DataFrame(df_entries, columns=columns)
# 
# def _find_duplicated(df: pd.DataFrame) -> pd.DataFrame:
#     df_duplicated = df[df.duplicated(keep="first")].copy()
#     duplication_rate = len(df_duplicated) / (len(df) - len(df_duplicated))
#     print(
#         f"\n{100*duplication_rate:.2f}% of data was duplicated. Cleaning up..."
#     )
#     print(f"\nOriginal entries:\n{df.head()}")
#     print(f"\nDuplicated entries:\n{df_duplicated.head()}")
#     return df_duplicated
# 
# def verify_duplicated_index(
#     cursor: sqlite3.Cursor, duplicated_row: pd.Series, table_name: str, dry_run: bool
# ) -> Tuple[str, bool]:
#     """Check if entry to delete is the one intended"""
#     columns = list(duplicated_row.index)
#     columns_str = ", ".join(columns)
# 
#     where_conditions = " AND ".join([f"{col} = ?" for col in columns])
#     select_query = f"""
#         SELECT {columns_str}
#         FROM {table_name}
#         WHERE {where_conditions}
#     """
#     cursor.execute(select_query, tuple(duplicated_row))
#     entries = cursor.fetchall()
# 
#     is_verified = len(entries) >= 1
# 
#     if dry_run:
#         print(f"Expected duplicate entry: {tuple(duplicated_row)}")
#         print(f"Found entries: {entries}")
#         print(f"Verification {'succeeded' if is_verified else 'failed'}")
# 
#     return select_query, is_verified
# 
# def _delete_entry(
#     cursor: sqlite3.Cursor,
#     duplicated_row: pd.Series,
#     table_name: str,
#     dry_run: bool = True,
# ) -> None:
#     select_query, is_verified = verify_duplicated_index(cursor, duplicated_row, table_name, dry_run)
#     if is_verified:
#         delete_query = select_query.replace("SELECT", "DELETE")
#         if dry_run:
#             print(f"[DRY RUN] Would delete entry:\n{duplicated_row}")
#         else:
#             cursor.execute(delete_query, tuple(duplicated_row))
#             print(f"Deleted entry:\n{duplicated_row}")
#     else:
#         print(f"Skipping entry (not found or already deleted):\n{duplicated_row}")
# 
# 
# # def delete_duplicates(
# #     lpath_db: str,
# #     table_name: str,
# #     columns: Union[str, List[str]] = "all",
# #     include_blob: bool = False,
# #     batch_size: int = 1000,
# #     reindex: bool = False,
# #     sort: bool = False,
# #     dry_run: bool = True,
# # ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
# #     """
# #     Delete duplicate entries from an SQLite database table.
# 
# #     Parameters
# #     ----------
# #     lpath_db : str
# #         Path to the SQLite database file.
# #     table_name : str
# #         Name of the table to remove duplicates from.
# #     columns : Union[str, List[str]], optional
# #         Columns to consider when identifying duplicates. Default is "all".
# #     include_blob : bool, optional
# #         Whether to include BLOB columns when considering duplicates. Default is False.
# #     batch_size : int, optional
# #         Number of rows to process in each batch. Default is 1000.
# #     reindex : bool, optional
# #         Whether to reindex the table after deletion. Default is False.
# #     dry_run : bool, optional
# #         If True, simulates the deletion without actually modifying the database. Default is True.
# 
# #     Returns
# #     -------
# #     Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]
# #         A tuple containing:
# #         - DataFrame of all entries after deletion process
# #         - DataFrame of remaining duplicates if any, None otherwise
# #     """
# #     try:
# #         conn = sqlite3.connect(lpath_db)
# #         cursor = conn.cursor()
# 
# #         columns = _determine_columns(cursor, table_name, columns, include_blob)
# 
# #         if sort:
# #             _sort_db(cursor, table_name, columns)
# 
# #         df_orig = _fetch_as_df(cursor, columns, table_name)
# #         duplicates = _find_duplicated(df_orig)
# 
# #         if duplicates.empty:
# #             print("Congratulations. Database is clean.")
# #             return df_orig, None
# 
# #         columns_str = ", ".join(columns)
# #         where_conditions = " AND ".join([f"{col} = ?" for col in columns])
# #         delete_query = f"""
# #             DELETE FROM {table_name}
# #             WHERE {where_conditions}
# #         """
# 
# #         for start in tqdm(range(0, len(duplicates), batch_size)):
# #             batch = duplicates.iloc[start:start+batch_size]
# #             batch_values = batch.values.tolist()
# 
# #             if dry_run:
# #                 print(f"[DRY RUN] Would delete {len(batch)} entries")
# #             else:
# #                 cursor.executemany(delete_query, batch_values)
# #                 conn.commit()
# 
# #         if not dry_run:
# #             conn.commit()
# 
# #             if reindex:
# #                 print("Reindexing the table...")
# #                 cursor.execute(f"REINDEX {table_name}")
# #                 conn.commit()
# 
# #         df_after = _fetch_as_df(cursor, columns, table_name)
# #         remaining_duplicates = _find_duplicated(df_after)
# 
# #         if remaining_duplicates.empty:
# #             print("All duplicates successfully removed.")
# #             return df_after, None
# #         else:
# #             print(f"Warning: {len(remaining_duplicates)} duplicates still remain.\n{remaining_duplicates}")
# #             return df_after, remaining_duplicates
# 
# #     except Exception as error:
# #         print(f"An error occurred: {error}")
# #         return None, None
# 
# #     finally:
# #         conn.close()
# 
# # def delete_duplicates(
# #     lpath_db: str,
# #     table_name: str,
# #     columns: Union[str, List[str]] = "all",
# #     include_blob: bool = False,
# #     batch_size: int = 1000,
# #     chunk_size: int = 100_000,
# #     reindex: bool = False,
# #     sort: bool = False,
# #     dry_run: bool = True,
# # ) -> Tuple[Optional[int], Optional[int]]:
# #     try:
# #         conn = sqlite3.connect(lpath_db)
# #         cursor = conn.cursor()
# 
# #         columns = _determine_columns(cursor, table_name, columns, include_blob)
# 
# #         if sort:
# #             _sort_db(cursor, table_name, columns)
# 
# #         columns_str = ", ".join(columns)
# #         where_conditions = " AND ".join([f"{col} = ?" for col in columns])
# #         delete_query = f"""
# #             DELETE FROM {table_name}
# #             WHERE {where_conditions}
# #         """
# 
# #         total_rows = cursor.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
# #         total_deleted = 0
# #         total_duplicates = 0
# 
# #         for offset in tqdm(range(0, total_rows, chunk_size)):
# #             chunk_query = f"""
# #                 SELECT {columns_str}
# #                 FROM {table_name}
# #                 LIMIT {chunk_size} OFFSET {offset}
# #             """
# #             df_chunk = pd.read_sql_query(chunk_query, conn)
# #             duplicates = _find_duplicated(df_chunk)
# #             total_duplicates += len(duplicates)
# 
# #             if duplicates.empty:
# #                 continue
# 
# #             for start in range(0, len(duplicates), batch_size):
# #                 batch = duplicates.iloc[start:start+batch_size]
# #                 batch_values = batch.values.tolist()
# 
# #                 if dry_run:
# #                     print(f"[DRY RUN] Would delete {len(batch)} entries")
# #                 else:
# #                     cursor.executemany(delete_query, batch_values)
# #                     conn.commit()
# #                     total_deleted += len(batch)
# 
# #         if not dry_run:
# #             if reindex:
# #                 print("Reindexing the table...")
# #                 cursor.execute(f"REINDEX {table_name}")
# #                 conn.commit()
# 
# #         print(f"Total duplicates found: {total_duplicates}")
# #         print(f"Total entries deleted: {total_deleted}")
# 
# #         return total_duplicates, total_deleted
# 
# #     except Exception as error:
# #         print(f"An error occurred: {error}")
# #         return None, None
# 
# #     finally:
# #         conn.close()
# 
# 
# def delete_duplicates(
#     lpath_db: str,
#     table_name: str,
#     columns: Union[str, List[str]] = "all",
#     include_blob: bool = False,
#     chunk_size: int = 10_000,
#     dry_run: bool = True,
# ) -> Tuple[Optional[int], Optional[int]]:
#     try:
#         conn = sqlite3.connect(lpath_db)
#         cursor = conn.cursor()
# 
#         # Vacuum the database to free up space
#         if not dry_run:
#             cursor.execute("VACUUM")
#             conn.commit()
# 
#         columns = _determine_columns(cursor, table_name, columns, include_blob)
#         columns_str = ", ".join(columns)
# 
#         # Create a temporary table to store unique rows
#         temp_table = f"{table_name}_temp"
#         cursor.execute(f"CREATE TABLE {temp_table} AS SELECT DISTINCT {columns_str} FROM {table_name} LIMIT 0")
# 
#         # Process in small chunks
#         offset = 0
#         total_processed = 0
#         total_unique = 0
# 
#         while True:
#             chunk_query = f"""
#                 INSERT OR IGNORE INTO {temp_table}
#                 SELECT DISTINCT {columns_str}
#                 FROM {table_name}
#                 LIMIT {chunk_size} OFFSET {offset}
#             """
# 
#             if dry_run:
#                 print(f"[DRY RUN] Would execute: {chunk_query}")
#             else:
#                 cursor.execute(chunk_query)
#                 conn.commit()
# 
#             rows_affected = cursor.rowcount
#             if rows_affected == 0:
#                 break
# 
#             total_processed += chunk_size
#             total_unique += rows_affected
#             offset += chunk_size
# 
#             print(f"Processed {total_processed} rows, {total_unique} unique")
# 
#         total_duplicates = total_processed - total_unique
# 
#         if not dry_run:
#             # Replace original table with the deduplicated one
#             cursor.execute(f"DROP TABLE {table_name}")
#             cursor.execute(f"ALTER TABLE {temp_table} RENAME TO {table_name}")
#             conn.commit()
# 
#         print(f"Total rows processed: {total_processed}")
#         print(f"Total unique rows: {total_unique}")
#         print(f"Total duplicates removed: {total_duplicates}")
# 
#         return total_processed, total_duplicates
# 
#     except Exception as error:
#         print(f"An error occurred: {error}")
#         return None, None
# 
#     finally:
#         conn.close()
# 
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/db/_delete_duplicates.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
