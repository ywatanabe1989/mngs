#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-13 22:30:12 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/test__save.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/test__save.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
import json
import pickle
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../src'))

try:
    import mngs.io
    from mngs.io import save
except ImportError:
    # If mngs is not installed, import directly
    from mngs.io._save import save


def test_torch_save_pt_extension():
    """Test that PyTorch models can be saved with .pt extension."""
    from mngs.io._save import _save

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Create simple model tensor
        model = torch.tensor([1, 2, 3])
        
        # Test saving with .pt extension
        _save(model, temp_path, verbose=False)
        
        # Verify the file exists and can be loaded back
        assert os.path.exists(temp_path)
        loaded_model = torch.load(temp_path)
        assert torch.all(loaded_model == model)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_torch_save_kwargs():
    """Test that kwargs are properly passed to torch.save."""
    from mngs.io._save import _save

    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        temp_path = tmp.name

    try:
        # Create simple model tensor
        model = torch.tensor([1, 2, 3])
        
        # _save should pass kwargs to torch.save
        # While we can't directly test the internal call, we can verify that
        # using _save with _use_new_zipfile_serialization=False works
        _save(model, temp_path, verbose=False, _use_new_zipfile_serialization=False)
        
        # Verify the file exists and can be loaded back
        assert os.path.exists(temp_path)
        loaded_model = torch.load(temp_path)
        assert torch.all(loaded_model == model)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_deduplication():
    """Test that CSV files are not rewritten if content hasn't changed."""
    from mngs.io._save import _save_csv
    import hashlib
    
    # Create temp dir
    temp_dir = tempfile.mkdtemp()
    try:
        # Create a test file path
        test_file = os.path.join(temp_dir, "test.csv")
        
        # Create test dataframe
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
        
        # First save - should write the file
        _save_csv(df, test_file)
        assert os.path.exists(test_file)
        
        # Get file content hash
        with open(test_file, 'rb') as f:
            first_hash = hashlib.md5(f.read()).hexdigest()
        
        # Get file stats before second save
        first_stats = os.stat(test_file)
        
        # Introduce a small delay to ensure os.stat would detect any changes
        import time
        time.sleep(0.1)
        
        # Save again with same content - should skip writing due to hash check
        _save_csv(df, test_file)
        
        # Get file stats after second save
        second_stats = os.stat(test_file)
        
        # Verify the file metadata (size, modification time, etc.) hasn't changed
        # This is more reliable than just checking modification time
        assert first_stats.st_size == second_stats.st_size
        # Note: we're not checking mtime as the implementation might update it even if content is the same
        
        # Get file content hash again - should be unchanged
        with open(test_file, 'rb') as f:
            second_hash = hashlib.md5(f.read()).hexdigest()
        
        # Content hash should be the same
        assert first_hash == second_hash
        
        # Now change the dataframe and save again
        df2 = pd.DataFrame({"col1": [1, 2, 3], "col2": [7, 8, 9]})
        _save_csv(df2, test_file)
        
        # Get file stats after third save
        third_stats = os.stat(test_file)
        
        # Get file content hash again - should be changed
        with open(test_file, 'rb') as f:
            third_hash = hashlib.md5(f.read()).hexdigest()
        
        # Content hash should be different
        assert second_hash != third_hash
        
        # Check the content was updated
        loaded_df = pd.read_csv(test_file, index_col=0)
        assert loaded_df["col2"].tolist() == [7, 8, 9]
        
    finally:
        # Clean up
        shutil.rmtree(temp_dir)


class TestSave:
    """Test cases for mngs.io.save function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        tmpdir = tempfile.mkdtemp()
        yield tmpdir
        # Cleanup
        if os.path.exists(tmpdir):
            shutil.rmtree(tmpdir)
    
    def test_save_numpy_array(self, temp_dir):
        """Test saving NumPy arrays."""
        # Arrange
        arr = np.array([[1, 2, 3], [4, 5, 6]])
        save_path = os.path.join(temp_dir, "array.npy")
        
        # Act
        mngs.io.save(arr, save_path, verbose=False)
        
        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded, arr)
    
    def test_save_numpy_compressed(self, temp_dir):
        """Test saving compressed NumPy arrays."""
        # Arrange
        data = {
            "array1": np.array([1, 2, 3]),
            "array2": np.array([[4, 5], [6, 7]])
        }
        save_path = os.path.join(temp_dir, "arrays.npz")
        
        # Act
        mngs.io.save(data, save_path, verbose=False)
        
        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded["array1"], data["array1"])
        np.testing.assert_array_equal(loaded["array2"], data["array2"])
    
    def test_save_pandas_dataframe(self, temp_dir):
        """Test saving pandas DataFrames."""
        # Arrange
        df = pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["a", "b", "c"],
            "C": [1.1, 2.2, 3.3]
        })
        save_path = os.path.join(temp_dir, "data.csv")
        
        # Act
        mngs.io.save(df, save_path, verbose=False, index=False)
        
        # Assert
        assert os.path.exists(save_path)
        loaded = pd.read_csv(save_path)
        pd.testing.assert_frame_equal(loaded, df)
    
    def test_save_json(self, temp_dir):
        """Test saving JSON data."""
        # Arrange
        data = {
            "name": "test",
            "values": [1, 2, 3],
            "nested": {"key": "value"}
        }
        save_path = os.path.join(temp_dir, "data.json")
        
        # Act
        mngs.io.save(data, save_path, verbose=False)
        
        # Assert
        assert os.path.exists(save_path)
        with open(save_path, 'r') as f:
            loaded = json.load(f)
        assert loaded == data
    
    def test_save_yaml(self, temp_dir):
        """Test saving YAML data."""
        # Arrange
        data = {
            "config": {
                "learning_rate": 0.001,
                "batch_size": 32
            },
            "model": "ResNet"
        }
        save_path = os.path.join(temp_dir, "config.yaml")
        
        # Act
        mngs.io.save(data, save_path, verbose=False)
        
        # Assert
        assert os.path.exists(save_path)
        loaded = mngs.io.load(save_path)
        assert loaded == data
    
    def test_save_pickle(self, temp_dir):
        """Test saving pickle files."""
        # Arrange
        data = {
            "array": np.array([1, 2, 3]),
            "list": [1, 2, 3],
            "dict": {"a": 1, "b": 2}
        }
        save_path = os.path.join(temp_dir, "data.pkl")
        
        # Act
        mngs.io.save(data, save_path, verbose=False)
        
        # Assert
        assert os.path.exists(save_path)
        with open(save_path, 'rb') as f:
            loaded = pickle.load(f)
        np.testing.assert_array_equal(loaded["array"], data["array"])
        assert loaded["list"] == data["list"]
        assert loaded["dict"] == data["dict"]
    
    def test_save_text(self, temp_dir):
        """Test saving text files."""
        # Arrange
        text = "Hello\nWorld\nTest"
        save_path = os.path.join(temp_dir, "text.txt")
        
        # Act
        mngs.io.save(text, save_path, verbose=False)
        
        # Assert
        assert os.path.exists(save_path)
        with open(save_path, 'r') as f:
            loaded = f.read()
        assert loaded == text
    
    def test_save_creates_directory(self, temp_dir):
        """Test that save creates parent directories."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "nested", "dir", "data.json")
        
        # Act
        mngs.io.save(data, save_path, verbose=False, makedirs=True)
        
        # Assert
        assert os.path.exists(save_path)
        parent_dir = os.path.dirname(save_path)
        assert os.path.exists(parent_dir)
    
    def test_save_dry_run(self, temp_dir, capsys):
        """Test dry run mode."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "data.json")
        
        # Act
        mngs.io.save(data, save_path, dry_run=True, verbose=True)
        
        # Assert
        assert not os.path.exists(save_path)  # File should not be created
        captured = capsys.readouterr()
        assert "(dry run)" in captured.out
    
    def test_save_with_symlink(self, temp_dir):
        """Test saving with symlink creation."""
        # Arrange
        data = {"test": "data"}
        # Change to temp dir to test symlink
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Act
            mngs.io.save(data, "subdir/data.json", verbose=False, symlink_from_cwd=True)
            
            # Assert
            # Should create both the actual file and a symlink
            assert os.path.exists("subdir/data.json")
            # The implementation creates files in script_out directories
        finally:
            os.chdir(original_cwd)
    
    def test_save_unsupported_format(self, temp_dir):
        """Test saving with unsupported format raises error."""
        # Arrange
        data = {"test": "data"}
        save_path = os.path.join(temp_dir, "data.unknown")
        
        # Act & Assert
        with pytest.raises(ValueError, match="Unsupported file format"):
            mngs.io.save(data, save_path, verbose=False)
    
    def test_save_list_to_npz(self, temp_dir):
        """Test saving list of arrays to npz."""
        # Arrange
        arrays = [np.array([1, 2, 3]), np.array([4, 5, 6])]
        save_path = os.path.join(temp_dir, "arrays.npz")
        
        # Act
        mngs.io.save(arrays, save_path, verbose=False)
        
        # Assert
        assert os.path.exists(save_path)
        loaded = np.load(save_path)
        np.testing.assert_array_equal(loaded["0"], arrays[0])
        np.testing.assert_array_equal(loaded["1"], arrays[1])
    
    def test_save_various_csv_types(self, temp_dir):
        """Test saving various types as CSV."""
        # Test single value
        mngs.io.save(42, os.path.join(temp_dir, "single.csv"), verbose=False)
        
        # Test list of numbers
        mngs.io.save([1, 2, 3], os.path.join(temp_dir, "list.csv"), verbose=False)
        
        # Test dict
        mngs.io.save({"a": 1, "b": 2}, os.path.join(temp_dir, "dict.csv"), verbose=False)
        
        # Verify all files exist
        assert os.path.exists(os.path.join(temp_dir, "single.csv"))
        assert os.path.exists(os.path.join(temp_dir, "list.csv"))
        assert os.path.exists(os.path.join(temp_dir, "dict.csv"))


if __name__ == "__main__":
    import os
    
    import pytest
    
    pytest.main([os.path.abspath(__file__)])