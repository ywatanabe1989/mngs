#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-02 13:05:00 (Claude)"
# File: /tests/mngs/db/test__SQLite3.py

import os
import sys
import tempfile
import shutil
import pytest
import sqlite3
from unittest.mock import patch, MagicMock, call
import pandas as pd
import numpy as np

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../src"))

from mngs.db._SQLite3 import SQLite3


class TestSQLite3:
    """Test cases for SQLite3 database class."""

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
    def db(self, db_path):
        """Create a SQLite3 instance for testing."""
        db = SQLite3(db_path)
        yield db
        # Cleanup
        if hasattr(db, 'close'):
            db.close()

    @pytest.fixture
    def db_with_table(self, db):
        """Create a database with a test table."""
        db.create_table(
            "test_table",
            {
                "id": "INTEGER PRIMARY KEY",
                "name": "TEXT",
                "value": "REAL",
                "data": "BLOB"
            }
        )
        return db

    def test_init_creates_database(self, db_path):
        """Test that initialization creates a database file."""
        # Act
        db = SQLite3(db_path)
        
        # Assert
        assert os.path.exists(db_path)
        assert db.db_path == db_path
        assert db.conn is not None
        assert db.cursor is not None
        
        # Cleanup
        db.close()

    def test_init_with_temp_database(self, temp_dir):
        """Test initialization with temporary database option."""
        # Arrange
        original_path = os.path.join(temp_dir, "original.db")
        
        # Create original database
        db1 = SQLite3(original_path)
        db1.create_table("test", {"id": "INTEGER"})
        db1.close()
        
        # Act
        db2 = SQLite3(original_path, use_temp=True)
        
        # Assert
        assert db2.db_path != original_path
        assert "tmp" in db2.db_path
        assert os.path.exists(db2.db_path)
        
        # Cleanup
        db2.close()

    def test_context_manager(self, db_path):
        """Test that SQLite3 works as a context manager."""
        # Act & Assert
        with SQLite3(db_path) as db:
            assert db.conn is not None
            assert db.cursor is not None
        
        # After context, connection should be closed
        assert db.conn is None
        assert db.cursor is None

    def test_call_method_returns_summary(self, db_with_table):
        """Test the __call__ method returns table summaries."""
        # Arrange
        db_with_table.execute(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ("test1", 1.5)
        )
        
        # Act
        summary = db_with_table(return_summary=True, print_summary=False)
        
        # Assert
        assert isinstance(summary, dict)
        assert "test_table" in summary
        assert isinstance(summary["test_table"], pd.DataFrame)

    def test_summary_property(self, db_with_table):
        """Test the summary property."""
        # Arrange
        with patch.object(db_with_table, '__call__') as mock_call:
            # Act
            db_with_table.summary
            
            # Assert
            mock_call.assert_called_once()

    def test_create_table_basic(self, db):
        """Test basic table creation."""
        # Act
        db.create_table(
            "users",
            {
                "id": "INTEGER PRIMARY KEY",
                "name": "TEXT NOT NULL",
                "email": "TEXT UNIQUE"
            }
        )
        
        # Assert
        tables = db.get_table_names()
        assert "users" in tables
        
        schema = db.get_table_schema("users")
        assert len(schema) == 3
        assert "id" in schema["name"].values
        assert "name" in schema["name"].values
        assert "email" in schema["name"].values

    def test_create_table_with_blob(self, db):
        """Test table creation with BLOB columns adds metadata columns."""
        # Act
        db.create_table(
            "images",
            {
                "id": "INTEGER PRIMARY KEY",
                "image_data": "BLOB"
            }
        )
        
        # Assert
        schema = db.get_table_schema("images")
        column_names = schema["name"].values
        assert "image_data" in column_names
        assert "image_data_dtype" in column_names
        assert "image_data_shape" in column_names

    def test_create_table_if_not_exists(self, db):
        """Test that creating existing table with if_not_exists=True doesn't fail."""
        # Arrange
        db.create_table("test", {"id": "INTEGER"})
        
        # Act & Assert - should not raise
        db.create_table("test", {"id": "INTEGER"}, if_not_exists=True)

    def test_drop_table(self, db_with_table):
        """Test dropping a table."""
        # Act
        db_with_table.drop_table("test_table")
        
        # Assert
        tables = db_with_table.get_table_names()
        assert "test_table" not in tables

    def test_basic_crud_operations(self, db_with_table):
        """Test basic Create, Read, Update, Delete operations."""
        # Create (Insert)
        db_with_table.execute(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ("test_item", 42.0)
        )
        
        # Read
        result = db_with_table.get_rows("test_table", where="name='test_item'")
        assert len(result) == 1
        assert result.iloc[0]["name"] == "test_item"
        assert result.iloc[0]["value"] == 42.0
        
        # Update
        db_with_table.execute(
            "UPDATE test_table SET value = ? WHERE name = ?",
            (99.0, "test_item")
        )
        
        result = db_with_table.get_rows("test_table", where="name='test_item'")
        assert result.iloc[0]["value"] == 99.0
        
        # Delete
        db_with_table.execute("DELETE FROM test_table WHERE name = ?", ("test_item",))
        result = db_with_table.get_rows("test_table", where="name='test_item'")
        assert len(result) == 0

    def test_transaction_commit(self, db_with_table):
        """Test transaction commit."""
        # Act
        with db_with_table.transaction():
            db_with_table.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("transaction_test", 1.0)
            )
        
        # Assert - data should be committed
        result = db_with_table.get_rows("test_table", where="name='transaction_test'")
        assert len(result) == 1

    def test_transaction_rollback(self, db_with_table):
        """Test transaction rollback on error."""
        # Act & Assert
        with pytest.raises(Exception):
            with db_with_table.transaction():
                db_with_table.execute(
                    "INSERT INTO test_table (name, value) VALUES (?, ?)",
                    ("rollback_test", 1.0)
                )
                # Force an error
                raise Exception("Test error")
        
        # Data should not be committed
        result = db_with_table.get_rows("test_table", where="name='rollback_test'")
        assert len(result) == 0

    def test_batch_insert(self, db_with_table):
        """Test batch insert operations."""
        # Arrange
        rows = [
            {"name": f"item_{i}", "value": float(i)}
            for i in range(100)
        ]
        
        # Act
        db_with_table.insert_many("test_table", rows, batch_size=10)
        
        # Assert
        count = db_with_table.get_row_count("test_table")
        assert count == 100

    def test_batch_update(self, db_with_table):
        """Test batch update operations."""
        # Arrange - insert test data
        rows = [
            {"name": f"item_{i}", "value": 0.0}
            for i in range(10)
        ]
        db_with_table.insert_many("test_table", rows)
        
        # Prepare updates
        updates = [
            {"name": f"item_{i}", "value": float(i * 10)}
            for i in range(10)
        ]
        
        # Act
        db_with_table.update_many("test_table", updates, where="name = ?")
        
        # Assert
        result = db_with_table.get_rows("test_table", order_by="name")
        assert len(result) == 10

    def test_save_and_load_numpy_array(self, db_with_table):
        """Test saving and loading numpy arrays as BLOBs."""
        # Arrange
        test_array = np.random.rand(10, 20).astype(np.float32)
        
        # Insert a row first
        db_with_table.execute(
            "INSERT INTO test_table (id, name) VALUES (?, ?)",
            (1, "array_test")
        )
        
        # Act - Save array
        db_with_table.save_array(
            "test_table",
            test_array,
            column="data",
            ids=1
        )
        
        # Act - Load array
        loaded_array = db_with_table.load_array(
            "test_table",
            column="data",
            ids=1
        )
        
        # Assert
        assert loaded_array is not None
        assert loaded_array.shape == (1, 10, 20)  # Extra dimension from stacking
        assert loaded_array.dtype == test_array.dtype
        np.testing.assert_array_almost_equal(loaded_array[0], test_array)

    def test_create_index(self, db_with_table):
        """Test index creation."""
        # Act
        db_with_table.create_index("test_table", ["name"], unique=True)
        
        # Assert - try to insert duplicate name
        db_with_table.execute(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ("unique_name", 1.0)
        )
        
        with pytest.raises(sqlite3.IntegrityError):
            db_with_table.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                ("unique_name", 2.0)
            )

    def test_get_table_schema(self, db_with_table):
        """Test retrieving table schema."""
        # Act
        schema = db_with_table.get_table_schema("test_table")
        
        # Assert
        assert isinstance(schema, pd.DataFrame)
        assert "name" in schema.columns
        assert "type" in schema.columns
        assert "pk" in schema.columns
        
        # Check primary key
        pk_rows = schema[schema["pk"] == 1]
        assert len(pk_rows) == 1
        assert pk_rows.iloc[0]["name"] == "id"

    def test_get_rows_with_filters(self, db_with_table):
        """Test getting rows with various filters."""
        # Arrange - insert test data
        for i in range(20):
            db_with_table.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                (f"item_{i}", float(i))
            )
        
        # Test with WHERE clause
        result = db_with_table.get_rows("test_table", where="value > 10")
        assert len(result) == 9  # 11-19
        
        # Test with ORDER BY
        result = db_with_table.get_rows("test_table", order_by="value DESC", limit=5)
        assert len(result) == 5
        assert result.iloc[0]["value"] == 19.0
        
        # Test with LIMIT and OFFSET
        result = db_with_table.get_rows("test_table", limit=5, offset=10)
        assert len(result) == 5

    def test_foreign_key_constraints(self, db):
        """Test foreign key constraint functionality."""
        # Create parent table
        db.create_table(
            "departments",
            {"id": "INTEGER PRIMARY KEY", "name": "TEXT"}
        )
        
        # Create child table with foreign key
        db.create_table(
            "employees",
            {
                "id": "INTEGER PRIMARY KEY",
                "name": "TEXT",
                "dept_id": "INTEGER"
            },
            foreign_keys=[{
                "tgt_column": "dept_id",
                "src_table": "departments",
                "src_column": "id"
            }]
        )
        
        # Enable foreign keys
        db.enable_foreign_keys()
        
        # Insert parent record
        db.execute("INSERT INTO departments (id, name) VALUES (1, 'Engineering')")
        
        # Valid insert
        db.execute("INSERT INTO employees (name, dept_id) VALUES ('John', 1)")
        
        # Invalid insert should fail
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("INSERT INTO employees (name, dept_id) VALUES ('Jane', 999)")

    def test_csv_export_import(self, db_with_table, temp_dir):
        """Test CSV export and import functionality."""
        # Arrange
        csv_path = os.path.join(temp_dir, "export.csv")
        
        # Insert test data
        for i in range(10):
            db_with_table.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                (f"item_{i}", float(i))
            )
        
        # Act - Export
        db_with_table.save_to_csv("test_table", csv_path)
        
        # Assert export
        assert os.path.exists(csv_path)
        df = pd.read_csv(csv_path)
        assert len(df) == 10
        
        # Act - Import to new table
        db_with_table.create_table(
            "imported_table",
            {"name": "TEXT", "value": "REAL"}
        )
        db_with_table.load_from_csv("imported_table", csv_path)
        
        # Assert import
        imported_count = db_with_table.get_row_count("imported_table")
        assert imported_count == 10

    def test_database_maintenance(self, db_with_table):
        """Test database maintenance operations."""
        # Insert some data
        for i in range(100):
            db_with_table.execute(
                "INSERT INTO test_table (name, value) VALUES (?, ?)",
                (f"item_{i}", float(i))
            )
        
        # Test vacuum
        db_with_table.vacuum()
        
        # Test optimize
        db_with_table.optimize()
        
        # Database should still be functional
        count = db_with_table.get_row_count("test_table")
        assert count == 100

    def test_backup_database(self, db_with_table, temp_dir):
        """Test database backup functionality."""
        # Arrange
        backup_path = os.path.join(temp_dir, "backup.db")
        
        # Insert test data
        db_with_table.execute(
            "INSERT INTO test_table (name, value) VALUES (?, ?)",
            ("backup_test", 123.45)
        )
        
        # Act
        db_with_table.backup(backup_path)
        
        # Assert
        assert os.path.exists(backup_path)
        
        # Open backup and verify data
        backup_db = SQLite3(backup_path)
        result = backup_db.get_rows("test_table", where="name='backup_test'")
        assert len(result) == 1
        assert result.iloc[0]["value"] == 123.45
        backup_db.close()

    def test_error_handling(self, db):
        """Test error handling for various operations."""
        # Test querying non-existent table
        with pytest.raises(Exception):
            db.get_rows("non_existent_table")
        
        # Test invalid SQL
        with pytest.raises(Exception):
            db.execute("INVALID SQL STATEMENT")
        
        # Test creating table with invalid column definition
        with pytest.raises(Exception):
            db.create_table("bad_table", {"id": "INVALID_TYPE"})

    def test_mixins_integration(self, db):
        """Test that all mixins are properly integrated."""
        # ConnectionMixin
        assert hasattr(db, 'connect')
        assert hasattr(db, 'close')
        assert hasattr(db, 'reconnect')
        
        # QueryMixin
        assert hasattr(db, 'execute')
        assert hasattr(db, 'executemany')
        
        # TransactionMixin
        assert hasattr(db, 'transaction')
        assert hasattr(db, 'begin')
        assert hasattr(db, 'commit')
        assert hasattr(db, 'rollback')
        
        # TableMixin
        assert hasattr(db, 'create_table')
        assert hasattr(db, 'drop_table')
        assert hasattr(db, 'get_table_names')
        assert hasattr(db, 'get_table_schema')
        
        # IndexMixin
        assert hasattr(db, 'create_index')
        assert hasattr(db, 'drop_index')
        
        # RowMixin
        assert hasattr(db, 'get_rows')
        assert hasattr(db, 'get_row_count')
        
        # BatchMixin
        assert hasattr(db, 'insert_many')
        assert hasattr(db, 'update_many')
        assert hasattr(db, 'delete_where')
        
        # BlobMixin
        assert hasattr(db, 'save_array')
        assert hasattr(db, 'load_array')
        
        # ImportExportMixin
        assert hasattr(db, 'load_from_csv')
        assert hasattr(db, 'save_to_csv')
        
        # MaintenanceMixin
        assert hasattr(db, 'vacuum')
        assert hasattr(db, 'optimize')
        assert hasattr(db, 'backup')


if __name__ == "__main__":
    pytest.main([os.path.abspath(__file__)])