#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-19 17:15:52 (ywatanabe)"
# /home/yusukew/proj/mngs_repo/src/mngs/db/_delete_duplicates.py

"""
Functionality:
    - Deletes duplicate entries from an SQLite database table
Input:
    - SQLite database file path, table name, columns to consider for duplicates
Output:
    - Updated SQLite database with duplicates removed
Prerequisites:
    - sqlite3, pandas, tqdm
"""

import sqlite3
from tqdm import tqdm
import pandas as pd
from typing import Union, List, Optional

def delete_duplicates(
    lpath_db: str,
    table_name: str,
    columns: Union[str, List[str]] = "all",
    include_blob: bool = False,
) -> Optional[pd.DataFrame]:
    """
    Delete duplicate entries from an SQLite database table.

    Example:
    --------
    >>> lpath_db = "path/to/database.db"
    >>> table_name = "my_table"
    >>> remaining_duplicates = delete_duplicates(lpath_db, table_name)
    >>> if remaining_duplicates is not None:
    ...     print(f"Remaining duplicates: {len(remaining_duplicates)}")

    Parameters:
    -----------
    lpath_db : str
        Path to the SQLite database file.
    table_name : str
        Name of the table to remove duplicates from.
    columns : str or List[str], optional
        Columns to consider when identifying duplicates. Default is "all".
    include_blob : bool, optional
        Whether to include BLOB columns when considering duplicates. Default is False.

    Returns:
    --------
    Optional[pd.DataFrame]
        DataFrame of remaining duplicates if any, None otherwise.
    """
    try:
        conn = sqlite3.connect(lpath_db)
        cursor = conn.cursor()

        cursor.execute(f"PRAGMA table_info({table_name})")
        table_info = cursor.fetchall()
        all_columns = [col[1] for col in table_info]
        column_types = {col[1]: col[2] for col in table_info}

        if columns == "all":
            columns = all_columns if include_blob else [col for col in all_columns if column_types[col].lower() != 'blob']
        elif isinstance(columns, str):
            columns = [columns]

        columns_str = ", ".join(columns)
        query = f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {columns_str} ORDER BY rowid) as rn FROM {table_name}"

        df = pd.read_sql_query(query, conn)
        df_duplicated = df[df["rn"] > 1]

        if df_duplicated.empty:
            print("No duplicates found.")
            return None

        print(f"Found {len(df_duplicated):,} duplicates.")

        delete_query = f"""
            DELETE FROM {table_name}
            WHERE rowid IN (
                SELECT rowid
                FROM (
                    SELECT rowid, ROW_NUMBER() OVER (PARTITION BY {columns_str} ORDER BY rowid) as rn
                    FROM {table_name}
                ) sub
                WHERE rn > 1
            )
        """

        cursor.execute(delete_query)
        conn.commit()

        print(f"Removed {cursor.rowcount:,} duplicate entries from the database.")

        df_after = pd.read_sql_query(query, conn)
        remaining_duplicates = df_after[df_after["rn"] > 1]

        if remaining_duplicates.empty:
            print("All duplicates successfully removed.")
            return None
        else:
            print(f"Warning: {len(remaining_duplicates):,} duplicates still remain.")
            return remaining_duplicates

    except sqlite3.Error as err:
        print(f"SQLite error: {err}")
        return None
    finally:
        if conn:
            conn.close()
