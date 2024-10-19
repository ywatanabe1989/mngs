#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-19 17:25:09 (ywatanabe)"
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
import pandas as _pd
from typing import Union, List, Optional


def delete_duplicates(
    lpath_db: str,
    table_name: str,
    columns: Union[str, List[str]] = "all",
    include_blob: bool = False,
    batch_size: int = 1000,
) -> Optional[_pd.DataFrame]:
    """
    Delete duplicate entries from an SQLite database table using batch processing.

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
    batch_size : int, optional
        Number of rows to process in each batch. Default is 1000.

    Returns:
    --------
    Optional[_pd.DataFrame]
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
            columns = (
                all_columns
                if include_blob
                else [
                    col
                    for col in all_columns
                    if column_types[col].lower() != "blob"
                ]
            )
        elif isinstance(columns, str):
            columns = [columns]

        columns_str = ", ".join(columns)

        # Count total rows
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        total_rows = cursor.fetchone()[0]

        # Process in batches
        offset = 0
        total_duplicates = 0
        total_removed = 0

        while offset < total_rows:
            query = f"""
                SELECT *, ROW_NUMBER() OVER (PARTITION BY {columns_str} ORDER BY rowid) as rn
                FROM {table_name}
                LIMIT {batch_size} OFFSET {offset}
            """

            df = _pd.read_sql_query(query, conn)
            df_duplicated = df[df["rn"] > 1]

            if not df_duplicated.empty:
                total_duplicates += len(df_duplicated)

                delete_query = f"""
                    DELETE FROM {table_name}
                    WHERE rowid IN (
                        SELECT rowid
                        FROM (
                            SELECT rowid, ROW_NUMBER() OVER (PARTITION BY {columns_str} ORDER BY rowid) as rn
                            FROM {table_name}
                            LIMIT {batch_size} OFFSET {offset}
                        ) sub
                        WHERE rn > 1
                    )
                """

                cursor.execute(delete_query)
                total_removed += cursor.rowcount

            offset += batch_size
            print(
                f"Processed {min(offset, total_rows):,} / {total_rows:,} rows"
            )

        conn.commit()

        print(f"Found {total_duplicates:,} duplicates.")
        print(
            f"Removed {total_removed:,} duplicate entries from the database."
        )

        # Verify removal
        verify_query = f"SELECT *, ROW_NUMBER() OVER (PARTITION BY {columns_str} ORDER BY rowid) as rn FROM {table_name}"
        df_after = _pd.read_sql_query(verify_query, conn)
        remaining_duplicates = df_after[df_after["rn"] > 1]

        if remaining_duplicates.empty:
            print("All duplicates successfully removed.")
            return None
        else:
            print(
                f"Warning: {len(remaining_duplicates):,} duplicates still remain."
            )
            return remaining_duplicates

    except sqlite3.Error as err:
        print(f"SQLite error: {err}")
        return None
    finally:
        if conn:
            conn.close()
