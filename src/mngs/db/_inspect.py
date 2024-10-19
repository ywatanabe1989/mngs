#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-19 17:11:21 (ywatanabe)"
# /mnt/ssd/mngs_repo/src/mngs/db/_inspect.py

import sqlite3
from typing import List, Tuple, Any, Optional
from tabulate import tabulate

class Inspector:
    def __init__(self, db_path: str):
        self.db_path = db_path

    def get_table_names(self) -> List[str]:
        """Retrieves all table names from the database.

        Returns:
            List[str]: List of table names
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            return [table[0] for table in cursor.fetchall()]

    def get_table_info(self, table_name: str) -> List[Tuple[int, str, str, int, Any, int, str]]:
        """Retrieves table structure information.

        Args:
            table_name (str): Name of the table

        Returns:
            List[Tuple[int, str, str, int, Any, int, str]]: List of column information tuples
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()

            cursor.execute(f"PRAGMA index_list({table_name})")
            indexes = cursor.fetchall()
            pk_columns = []
            for idx in indexes:
                if idx[2] == 1:  # Is primary key
                    cursor.execute(f"PRAGMA index_info({idx[1]})")
                    pk_columns.extend([info[2] for info in cursor.fetchall()])

            enhanced_columns = []
            for col in columns:
                constraints = []
                if col[1] in pk_columns:
                    constraints.append("PRIMARY KEY")
                if col[3] == 1:
                    constraints.append("NOT NULL")
                enhanced_columns.append(col + (" ".join(constraints),))

            return enhanced_columns

    def get_sample_data(self, table_name: str, limit: int = 5) -> Tuple[List[str], List[Tuple], int]:
        """Retrieves sample data from the specified table.

        Args:
            table_name (str): Name of the table
            limit (int, optional): Number of rows to retrieve. Defaults to 5.

        Returns:
            Tuple[List[str], List[Tuple], int]: Column names, sample data rows, and total row count
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
            columns = [description[0] for description in cursor.description]
            sample_data = cursor.fetchall()

            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            total_rows = cursor.fetchone()[0]

            return columns, sample_data, total_rows

    def inspect(self, table_names: Optional[List[str]] = None) -> None:
        """Inspects and prints database structure and sample data.

        Args:
            table_names (Optional[List[str]], optional): List of table names to inspect.
                If None, inspects all tables. Defaults to None.
        """
        if table_names is None:
            table_names = self.get_table_names()
            print("Tables in the database:")
            for name in table_names:
                print(f"  {name}")
        else:
            for table_name in table_names:
                columns = self.get_table_info(table_name)
                column_names, rows, total_rows = self.get_sample_data(table_name)

                print(f"\nTable: {table_name}")
                print(f"Total rows: {total_rows:,}")
                print("\nColumns:")
                for col in columns:
                    constraints = f"({col[6]})" if col[6] else ""
                    print(f"  {col[1]} ({col[2]}) {constraints}")

                print("\nSample data:")
                table_data = []
                for row in rows:
                    table_data.append([str(value) if not isinstance(value, bytes) else "<BLOB>" for value in row])
                print(tabulate(table_data, headers=column_names, tablefmt="grid"))

def inspect(lpath_db: str, table_names: Optional[List[str]] = None) -> None:
    """
    Inspects the specified SQLite database.

    Example:
    >>> inspect('path/to/database.db')
    >>> inspect('path/to/database.db', ['table1', 'table2'])

    Args:
        lpath_db (str): Path to the SQLite database file
        table_names (Optional[List[str]], optional): List of table names to inspect.
            If None, inspects all tables. Defaults to None.
    """
    inspector = Inspector(lpath_db)
    inspector.inspect(table_names)

# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-10-15 00:41:56 (ywatanabe)"
# # /mnt/ssd/mngs_repo/src/mngs/db/_inspect.py

# import sqlite3
# from typing import List, Tuple, Any, Optional

# class Inspector:
#     def __init__(self, db_path: str):
#         self.db_path = db_path

#     def get_table_names(self) -> List[str]:
#         """Retrieves all table names from the database.

#         Returns:
#             List[str]: List of table names
#         """
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#             return [table[0] for table in cursor.fetchall()]

#     def get_table_info(self, table_name: str) -> List[Tuple[int, str, str, int, Any, int]]:
#         """Retrieves table structure information.

#         Args:
#             table_name (str): Name of the table

#         Returns:
#             List[Tuple[int, str, str, int, Any, int]]: List of column information tuples
#         """
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute(f"PRAGMA table_info({table_name})")
#             return cursor.fetchall()

#     def get_sample_data(self, table_name: str, limit: int = 5) -> Tuple[List[str], List[Tuple]]:
#         """Retrieves sample data from the specified table.

#         Args:
#             table_name (str): Name of the table
#             limit (int, optional): Number of rows to retrieve. Defaults to 5.

#         Returns:
#             Tuple[List[str], List[Tuple]]: Column names and sample data rows
#         """
#         with sqlite3.connect(self.db_path) as conn:
#             cursor = conn.cursor()
#             cursor.execute(f"SELECT * FROM {table_name} LIMIT {limit}")
#             columns = [description[0] for description in cursor.description]
#             return columns, cursor.fetchall()

#     def inspect(self, table_names: Optional[List[str]] = None) -> None:
#         """Inspects and prints database structure and sample data.

#         Args:
#             table_names (Optional[List[str]], optional): List of table names to inspect.
#                 If None, inspects all tables. Defaults to None.
#         """
#         if table_names is None:
#             table_names = self.get_table_names()
#             print("Tables in the database:")
#             for name in table_names:
#                 print(f"  {name}")
#         else:
#             for table_name in table_names:
#                 columns = self.get_table_info(table_name)
#                 column_names, rows = self.get_sample_data(table_name)

#                 print(f"\nTable: {table_name}")
#                 print("Columns:")
#                 for col in columns:
#                     print(f"  {col[1]} ({col[2]})")

#                 print("\nSample data:")
#                 for row in rows:
#                     safe_row = []
#                     for col, value in zip(column_names, row):
#                         if isinstance(value, bytes):
#                             safe_row.append(f"{col}: <BLOB>")
#                         else:
#                             safe_row.append(f"{col}: {value}")
#                     print(", ".join(safe_row))

# def inspect(lpath_db: str, table_names: Optional[List[str]] = None) -> None:
#     """
#     Inspects the specified SQLite database.

#     Example:
#     >>> inspect('path/to/database.db')
#     >>> inspect('path/to/database.db', ['table1', 'table2'])

#     Args:
#         lpath_db (str): Path to the SQLite database file
#         table_names (Optional[List[str]], optional): List of table names to inspect.
#             If None, inspects all tables. Defaults to None.
#     """

#     inspector = Inspector(lpath_db)
#     inspector.inspect(table_names)
