#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-10-19 23:46:57 (ywatanabe)"
# /home/yusukew/proj/mngs_repo/src/mngs/db/_delete_duplicates.py

"""
Functionality:
    - Deletes duplicate entries from an SQLite database table
Input:
    - SQLite database file path, table name, columns to consider for duplicates
Output:
    - Updated SQLite database with duplicates removed
Prerequisites:
    - sqlite3, pandas, tqdm, mngs
"""

import sqlite3
import pandas as pd
from typing import Union, List, Optional, Tuple
from tqdm import tqdm
import mngs
from time import sleep

def _determine_columns(
    cursor: sqlite3.Cursor,
    table_name: str,
    columns: Union[str, List[str]],
    include_blob: bool,
) -> List[str]:
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
    print(f"Columns considered for duplicates: {columns_str}")

    return columns


def _fetch_as_df(
    cursor: sqlite3.Cursor, columns: List[str], table_name: str
) -> pd.DataFrame:
    print("\nFetching all database entries...")
    columns_str = ", ".join(columns)
    query = f"SELECT {columns_str} FROM {table_name}"
    cursor.execute(query)
    df_entries = cursor.fetchall()
    return pd.DataFrame(df_entries, columns=columns)


def _find_duplicated(df: pd.DataFrame) -> pd.DataFrame:
    df_duplicated = df[df.duplicated(keep="first")].copy()
    duplication_rate = len(df_duplicated) / (len(df) - len(df_duplicated))
    print(
        f"\n{100*duplication_rate:.2f}% of data was duplicated. Cleaning up..."
    )
    print(f"\nOriginal entries:\n{df.head()}")
    print(f"\nDuplicated entries:\n{df_duplicated.head()}")
    return df_duplicated

# def verify_duplicated_index(
#     cursor: sqlite3.Cursor, duplicated_row: pd.Series, table_name: str
# ) -> bool:
#     """Check if entry to delete is the one intended"""
#     columns = list(duplicated_row.index)
#     columns_str = ", ".join(columns)

#     # Select by values for safe (but slow)
#     where_conditions = " AND ".join([f"{col} = ?" for col in columns])
#     select_query_values = f"""
#         SELECT {columns_str}
#         FROM {table_name}
#         WHERE {where_conditions}
#     """
#     cursor.execute(select_query_values, tuple(duplicated_row))
#     entry_by_values = cursor.fetchone()

#     # Select by id for speed (but I am not confident)
#     select_query_id = f"""
#         SELECT {columns_str}
#         FROM {table_name}
#         WHERE rowid = ?
#     """
#     cursor.execute(select_query_id, (duplicated_row.name,))
#     entry_by_id = cursor.fetchone()

#     # Check
#     is_verified = entry_by_values == entry_by_id

#     return select_query_id, is_verified

# def verify_duplicated_index(
#         cursor: sqlite3.Cursor, duplicated_row: pd.Series, table_name: str, dry_run: bool
# ) -> bool:
#     """Check if entry to delete is the one intended"""

#     columns = list(duplicated_row.index)
#     columns_str = ", ".join(columns)

#     select_query_id = f"""
#         SELECT {columns_str}
#         FROM {table_name}
#         WHERE rowid = ?
#     """

#     if dry_run:
#         # Select by values for safe (but slow)
#         where_conditions = " AND ".join([f"{col} = ?" for col in columns])
#         select_query_values = f"""
#             SELECT {columns_str}
#             FROM {table_name}
#             WHERE {where_conditions}
#         """
#         cursor.execute(select_query_values, tuple(duplicated_row))
#         entry_by_values = cursor.fetchone()

#         # Select by id for speed (but I am not confident)
#         cursor.execute(select_query_id, (duplicated_row.name,))
#         entry_by_id = cursor.fetchone()

#         # Check
#         is_verified = entry_by_values == entry_by_id

#         if not is_verified:
#             print(f"Verification failed:")
#             print(f"Entry by values: {entry_by_values}")
#             print(f"Entry by ID: {entry_by_id}")

#         return select_query_id, is_verified  # Return values-based query for safety

#     else:
#         return select_query_id, True

def verify_duplicated_index(
    cursor: sqlite3.Cursor, duplicated_row: pd.Series, table_name: str, dry_run: bool
) -> Tuple[str, bool]:
    """Check if entry to delete is the one intended"""
    columns = list(duplicated_row.index)
    columns_str = ", ".join(columns)

    where_conditions = " AND ".join([f"{col} = ?" for col in columns])
    select_query = f"""
        SELECT {columns_str}
        FROM {table_name}
        WHERE {where_conditions}
    """
    cursor.execute(select_query, tuple(duplicated_row))
    entries = cursor.fetchall()

    is_verified = len(entries) >= 1  # At least one entry found

    if dry_run:
        print(f"Expected duplicate entry: {tuple(duplicated_row)}")
        print(f"Found entries: {entries}")
        print(f"Verification {'succeeded' if is_verified else 'failed'}")

    return select_query, is_verified

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
#         raise ValueError("Not Verified")

def _delete_entry(
    cursor: sqlite3.Cursor,
    duplicated_row: pd.Series,
    table_name: str,
    dry_run: bool = True,
) -> None:
    select_query, is_verified = verify_duplicated_index(cursor, duplicated_row, table_name, dry_run)
    if is_verified:
        delete_query = select_query.replace("SELECT", "DELETE")
        if dry_run:
            print(f"[DRY RUN] Would delete entry:\n{duplicated_row}")
        else:
            cursor.execute(delete_query, tuple(duplicated_row))
            print(f"Deleted entry:\n{duplicated_row}")
    else:
        print(f"Skipping entry (not found or already deleted):\n{duplicated_row}")


# def delete_duplicates(
#     lpath_db: str,
#     table_name: str,
#     columns: Union[str, List[str]] = "all",
#     include_blob: bool = False,
#     batch_size: int = 1000,
#     dry_run: bool = True,
# ) -> Optional[pd.DataFrame]:
#     """
#     Delete duplicate entries from an SQLite database table.

#     Parameters
#     ----------
#     lpath_db : str
#         Path to the SQLite database file.
#     table_name : str
#         Name of the table to remove duplicates from.
#     columns : Union[str, List[str]], optional
#         Columns to consider when identifying duplicates. Default is "all".
#     include_blob : bool, optional
#         Whether to include BLOB columns when considering duplicates. Default is False.
#     batch_size : int, optional
#         Number of rows to process in each batch. Default is 1000.
#     dry_run : bool, optional
#         If True, simulates the deletion without actually modifying the database. Default is True.

#     Returns
#     -------
#     Optional[pd.DataFrame]
#         DataFrame of remaining duplicates if any, None otherwise.
#     """
#     assert (
#         dry_run
#     ), "Dry run is enforced for safety. Set to False only when you're sure about the operation."

#     try:
#         conn = sqlite3.connect(lpath_db)
#         cursor = conn.cursor()

#         columns = _determine_columns(cursor, table_name, columns, include_blob)

#         df_orig = mngs.io.load(lpath_db.replace(".db", ".csv"))
#         duplicates = _find_duplicated(df_orig)

#         if duplicates.empty:
#             print("Congratulations. Database is clean.")
#             return None

#         for ii, (_, row) in enumerate(
#             tqdm(duplicates.iterrows(), total=len(duplicates))
#         ):
#             _delete_entry(cursor, row, table_name, dry_run=dry_run)
#             # if dry_run:
#             #     sleep(0.1)

#             if ii % batch_size == 0 and not dry_run:
#                 conn.commit()

#         if not dry_run:
#             conn.commit()

#         df_after = _fetch_as_df(cursor, columns, table_name)
#         remaining_duplicates = _find_duplicated(df_after)

#         if remaining_duplicates.empty:
#             print("All duplicates successfully removed.")
#             return None
#         else:
#             print(
#                 f"Warning: {len(remaining_duplicates)} duplicates still remain."
#             )
#             return remaining_duplicates

#     except Exception as error:
#         print(f"An error occurred: {error}")
#         return None

#     finally:
#         conn.close()

def delete_duplicates(
    lpath_db: str,
    table_name: str,
    columns: Union[str, List[str]] = "all",
    include_blob: bool = False,
    batch_size: int = 1000,
    dry_run: bool = True,
) -> Optional[pd.DataFrame]:
    try:
        conn = sqlite3.connect(lpath_db)
        cursor = conn.cursor()

        columns = _determine_columns(cursor, table_name, columns, include_blob)

        df_orig = _fetch_as_df(cursor, columns, table_name)
        # df_orig = mngs.io.load(lpath_db.replace(".db", ".csv"))
        duplicates = _find_duplicated(df_orig)

        if duplicates.empty:
            print("Congratulations. Database is clean.")
            return None

        columns_str = ", ".join(columns)
        where_conditions = " AND ".join([f"{col} = ?" for col in columns])
        delete_query = f"""
            DELETE FROM {table_name}
            WHERE {where_conditions}
        """

        for start in tqdm(range(0, len(duplicates), batch_size)):
            batch = duplicates.iloc[start:start+batch_size]
            batch_values = batch.values.tolist()

            if dry_run:
                print(f"[DRY RUN] Would delete {len(batch)} entries")
            else:
                cursor.executemany(delete_query, batch_values)
                conn.commit()

        if not dry_run:
            conn.commit()

        df_after = _fetch_as_df(cursor, columns, table_name)
        remaining_duplicates = _find_duplicated(df_after)

        if remaining_duplicates.empty:
            print("All duplicates successfully removed.")
            return df_after, None
        else:
            print(f"Warning: {len(remaining_duplicates)} duplicates still remain.\n{remaining_duplicates}")
            return df_after, remaining_duplicates

    except Exception as error:
        print(f"An error occurred: {error}")
        return None, None

    finally:
        conn.close()
