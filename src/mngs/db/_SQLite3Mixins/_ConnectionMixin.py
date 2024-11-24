#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2024-11-25 01:35:39 (ywatanabe)"
# File: ./mngs_repo/src/mngs/db/_SQLite3Mixins/_ConnectionMixin.py

__file__ = "/home/ywatanabe/proj/mngs_repo/src/mngs/db/_SQLite3Mixins/_ConnectionMixin.py"

"""
1. Functionality:
   - Manages SQLite database connections with thread-safe operations
   - Handles database journal files and transaction states
2. Input:
   - Database file path
3. Output:
   - Managed SQLite connection and cursor objects
4. Prerequisites:
   - sqlite3
   - threading
"""

import sqlite3
import threading
from typing import Optional
import os
import shutil
import tempfile
from .._BaseMixins._BaseConnectionMixin import _BaseConnectionMixin


class _ConnectionMixin(_BaseConnectionMixin):
    """Connection management functionality"""

    def __init__(self, db_path: str, use_temp_db: bool = False):
        self.lock = threading.Lock()
        self._maintenance_lock = threading.Lock()
        self.conn: Optional[sqlite3.Connection] = None
        self.cursor: Optional[sqlite3.Cursor] = None
        self.db_path = db_path
        self.temp_path = None
        if db_path:
            self.connect(db_path, use_temp_db)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _create_temp_copy(self, db_path: str) -> str:
        """Creates temporary copy of database."""
        temp_dir = tempfile.gettempdir()
        self.temp_path = os.path.join(
            temp_dir, f"temp_{os.path.basename(db_path)}"
        )
        shutil.copy2(db_path, self.temp_path)
        return self.temp_path

    def connect(self, db_path: str, use_temp_db: bool = False) -> None:
        """Establishes database connection"""
        if self.conn:
            self.close()

        path_to_connect = (
            self._create_temp_copy(db_path) if use_temp_db else db_path
        )

        self.conn = sqlite3.connect(
            path_to_connect, timeout=60.0, isolation_level="EXCLUSIVE"
        )
        self.cursor = self.conn.cursor()

        with self.lock:
            self.cursor.execute("PRAGMA journal_mode = DELETE")
            self.cursor.execute("PRAGMA synchronous = OFF")
            self.cursor.execute("PRAGMA locking_mode = EXCLUSIVE")
            self.conn.rollback()
            self.conn.commit()

    def close(self) -> None:
        if self.cursor:
            self.cursor.close()
        if self.conn:
            try:
                self.conn.rollback()
                self.conn.close()
            except sqlite3.Error:
                pass
        self.cursor = None
        self.conn = None

        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.remove(self.temp_path)
                self.temp_path = None
            except OSError:
                pass

    def reconnect(self, use_temp_db: bool = False) -> None:
        if self.db_path:
            self.connect(self.db_path, use_temp_db)
        else:
            raise ValueError("No database path specified for reconnection")


# class _ConnectionMixin:
#     """Connection management functionality"""

#     def __init__(self, db_path: str):
#         self.lock = threading.Lock()
#         self._maintenance_lock = threading.Lock()
#         self.conn: Optional[sqlite3.Connection] = None
#         self.cursor: Optional[sqlite3.Cursor] = None
#         self.db_path = db_path
#         if db_path:
#             self.connect(db_path)

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.close()

#     def _create_temp_copy(self, db_path: str) -> str:
#         """Creates temporary copy of database."""
#         temp_dir = tempfile.gettempdir()
#         self.temp_path = os.path.join(temp_dir, f"temp_{os.path.basename(db_path)}")
#         shutil.copy2(db_path, self.temp_path)
#         return self.temp_path

#     def connect(self, db_path: str) -> None:
#         """Establishes database connection"""
#         if self.conn:
#             self.close()

#         self.conn = sqlite3.connect(
#             db_path, timeout=60.0, isolation_level="EXCLUSIVE"
#         )
#         self.cursor = self.conn.cursor()

#         with self.lock:
#             self.cursor.execute("PRAGMA journal_mode = DELETE")
#             self.cursor.execute("PRAGMA synchronous = OFF")
#             self.cursor.execute("PRAGMA locking_mode = EXCLUSIVE")
#             self.conn.rollback()
#             self.conn.commit()

#     def close(self) -> None:
#         if self.cursor:
#             self.cursor.close()
#         if self.conn:
#             try:
#                 self.conn.rollback()
#                 self.conn.close()
#             except sqlite3.Error:
#                 pass
#         self.cursor = None
#         self.conn = None

#     def reconnect(self) -> None:
#         if self.db_path:
#             self.connect(self.db_path)
#         else:
#             raise ValueError("No database path specified for reconnection")


# # EOF

# # #!/usr/bin/env python3
# # # -*- coding: utf-8 -*-
# # # Time-stamp: "2024-11-11 14:38:01 (ywatanabe)"
# # # File: ./mngs_repo/src/mngs/db/_BaseSQLiteDB_modules/_ConnectionMixin.py

# # import sqlite3
# # import threading
# # from typing import Optional

# # """
# # 1. Functionality:
# #    - (e.g., Executes XYZ operation)
# # 2. Input:
# #    - (e.g., Required data for XYZ)
# # 3. Output:
# #    - (e.g., Results of XYZ operation)
# # 4. Prerequisites:
# #    - (e.g., Necessary dependencies for XYZ)

# # (Remove me: Please fill docstrings above, while keeping the bulette point style, and remove this instruction line)
# # """

# # # class _ConnectionMixin:
# # #     """Connection management functionality"""
# # #     def __init__(self, db_path: str):
# # #         self.lock = threading.Lock()
# # #         self._maintenance_lock = threading.Lock()
# # #         self.conn: Optional[sqlite3.Connection] = None
# # #         self.cursor: Optional[sqlite3.Cursor] = None
# # #         self.db_path = db_path
# # #         if db_path:
# # #             self.connect(db_path)

# # #     def __enter__(self):
# # #         return self

# # #     def __exit__(self, exc_type, exc_val, exc_tb):
# # #         self.close()

# # #     def connect(self, db_path: str) -> None:
# # #         if self.conn:
# # #             self.close()
# # #         self.conn = sqlite3.connect(db_path)
# # #         self.cursor = self.conn.cursor()

# # #     def close(self) -> None:
# # #         if self.cursor:
# # #             self.cursor.close()
# # #         if self.conn:
# # #             self.conn.close()
# # #         self.cursor = None
# # #         self.conn = None

# # #     def reconnect(self) -> None:
# # #         if self.db_path:
# # #             self.connect(self.db_path)
# # #         else:
# # #             raise ValueError("No database path specified for reconnection")

# # class _ConnectionMixin:
# #     """Connection management functionality"""
# #     def __init__(self, db_path: str):
# #         self.lock = threading.Lock()
# #         self._maintenance_lock = threading.Lock()
# #         self.conn: Optional[sqlite3.Connection] = None
# #         self.cursor: Optional[sqlite3.Cursor] = None
# #         self.db_path = db_path
# #         if db_path:
# #             self.connect(db_path)

# #     def __enter__(self):
# #         return self

# #     def __exit__(self, exc_type, exc_val, exc_tb):
# #         self.close()

# #     def connect(self, db_path: str) -> None:
# #         if self.conn:
# #             self.close()
# #         self.conn = sqlite3.connect(db_path)
# #         self.cursor = self.conn.cursor()

# #         # Handle journal and set pragmas
# #         with self.lock:
# #             self.cursor.execute("PRAGMA journal_mode = DELETE")
# #             self.cursor.execute("PRAGMA busy_timeout = 5000")
# #             self.conn.rollback()  # Clear any pending transactions
# #             self.conn.commit()

# #     def close(self) -> None:
# #         if self.cursor:
# #             self.cursor.close()
# #         if self.conn:
# #             try:
# #                 self.conn.rollback()  # Ensure no pending transactions
# #                 self.conn.close()
# #             except sqlite3.Error:
# #                 pass
# #         self.cursor = None
# #         self.conn = None

# #     def reconnect(self) -> None:
# #         if self.db_path:
# #             self.connect(self.db_path)
# #         else:
# #             raise ValueError("No database path specified for reconnection")

# #

# EOF
