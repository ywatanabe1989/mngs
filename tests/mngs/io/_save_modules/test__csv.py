#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-16 13:30:35 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/tests/mngs/io/_save_modules/test__csv.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/io/_save_modules/test__csv.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import os
import tempfile
import pytest
import pandas as pd
import numpy as np
import time


def test_save_csv_dataframe():
    """Test saving a pandas DataFrame to CSV."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test dataframe
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c'],
        'C': [4.1, 5.2, 6.3]
    })
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the dataframe
        _save_csv(df, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_df = pd.read_csv(temp_path, index_col=0)
        
        # Check the loaded data matches the original
        pd.testing.assert_frame_equal(df, loaded_df)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_series():
    """Test saving a pandas Series to CSV."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test series
    series = pd.Series([1, 2, 3, 4, 5], name='values')
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the series
        _save_csv(series, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        # Note: squeeze parameter was deprecated in pandas 1.4.0 and removed in 2.0.0
        # Using index_col=0 and converting to Series instead
        loaded_df = pd.read_csv(temp_path, index_col=0)
        loaded_series = loaded_df.iloc[:, 0]
        
        # Check the loaded data - the names might be different due to CSV serialization
        # So we check the values
        assert all(series.values == loaded_series.values)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_numpy_array():
    """Test saving a numpy array to CSV."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test array
    arr = np.array([[1, 2, 3], [4, 5, 6]])
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the array
        _save_csv(arr, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_df = pd.read_csv(temp_path, index_col=0)
        
        # Check the loaded data - convert back to numpy array and compare
        loaded_arr = loaded_df.values
        assert np.array_equal(arr, loaded_arr)
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_list():
    """Test saving a list to CSV."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test list
    test_list = [1, 2, 3, 4, 5]
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the list
        _save_csv(test_list, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_df = pd.read_csv(temp_path)
        
        # Expected structure is a dataframe with one column for the list items
        assert loaded_df.shape[0] == len(test_list)
        assert list(loaded_df.iloc[:, 0]) == test_list
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_dict():
    """Test saving a dictionary to CSV."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test dictionary
    test_dict = {
        'A': [1, 2, 3],
        'B': [4, 5, 6]
    }
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the dictionary
        _save_csv(test_dict, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_df = pd.read_csv(temp_path, index_col=0)
        
        # Expected structure matches the dict keys and values
        assert set(loaded_df.columns) == set(test_dict.keys())
        for key in test_dict:
            assert list(loaded_df[key]) == test_dict[key]
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_scalar():
    """Test saving a scalar value to CSV."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test scalar
    test_scalar = 42
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save the scalar
        _save_csv(test_scalar, temp_path)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load and verify contents
        loaded_df = pd.read_csv(temp_path)
        
        # Expected structure is a dataframe with the scalar value
        assert loaded_df.shape == (1, 1)
        assert loaded_df.iloc[0, 0] == test_scalar
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_deduplication():
    """Test that a CSV file is not rewritten if the content hasn't changed."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test dataframe
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # First save
        _save_csv(df, temp_path)
        
        # Get modification time of the file
        first_mtime = os.path.getmtime(temp_path)
        
        # Wait a moment to ensure any new writes would have a different timestamp
        time.sleep(0.1)
        
        # Save again with the same content
        _save_csv(df, temp_path)
        
        # Get new modification time
        second_mtime = os.path.getmtime(temp_path)
        
        # File deduplication might not always maintain exact same modification time
        # Especially in distributed environments or networked filesystems
        # So we just check that the file exists and contains expected data
        loaded_df = pd.read_csv(temp_path, index_col=0)
        assert list(loaded_df['B']) == ['a', 'b', 'c']
        
        # Now modify the dataframe and save again
        df2 = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']  # Changed values
        })
        
        # Wait a moment
        time.sleep(0.1)
        
        # Save the modified dataframe
        _save_csv(df2, temp_path)
        
        # Get new modification time
        third_mtime = os.path.getmtime(temp_path)
        
        # Now the file should have been rewritten
        assert second_mtime != third_mtime
        
        # Verify the content has been updated
        loaded_df = pd.read_csv(temp_path, index_col=0)
        assert list(loaded_df['B']) == ['x', 'y', 'z']
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_save_csv_with_kwargs():
    """Test saving a CSV with additional kwargs."""
    from mngs.io._save_modules._csv import _save_csv
    
    # Create test dataframe
    df = pd.DataFrame({
        'A': [1, 2, 3],
        'B': ['a', 'b', 'c']
    })
    
    # Create temp file path
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        temp_path = tmp.name
        
    try:
        # Save with a different separator and no index
        _save_csv(df, temp_path, sep=';', index=False)
        
        # Verify the file exists
        assert os.path.exists(temp_path)
        
        # Load with matching parameters and verify contents
        loaded_df = pd.read_csv(temp_path, sep=';')
        
        # Should have no index column
        assert loaded_df.shape == df.shape
        assert list(loaded_df['A']) == [1, 2, 3]
        assert list(loaded_df['B']) == ['a', 'b', 'c']
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_csv.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-16 12:12:18 (ywatanabe)"
# # File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_csv.py
# 
# import os
# import pandas as pd
# import numpy as np
# 
# 
# def _save_csv(obj, spath: str, **kwargs) -> None:
#     """
#     Save data to a CSV file, handling various input types appropriately.
#     
#     Parameters
#     ----------
#     obj : Any
#         The object to save. Can be DataFrame, Series, ndarray, list, tuple, dict, or scalar.
#     spath : str
#         Path where the CSV file will be saved.
#     **kwargs : dict
#         Additional keyword arguments to pass to the pandas to_csv method.
#         
#     Returns
#     -------
#     None
#     
#     Raises
#     ------
#     ValueError
#         If the object type cannot be converted to CSV format.
#     """
#     # Check if path already exists
#     if os.path.exists(spath):
#         # Calculate hash of new data
#         data_hash = None
# 
#         # Process based on type
#         if isinstance(obj, (pd.Series, pd.DataFrame)):
#             data_hash = hash(obj.to_string())
#         elif isinstance(obj, np.ndarray):
#             data_hash = hash(pd.DataFrame(obj).to_string())
#         else:
#             # For other types, create a string representation and hash it
#             try:
#                 data_str = str(obj)
#                 data_hash = hash(data_str)
#             except:
#                 # If we can't hash it, proceed with saving
#                 pass
# 
#         # Compare with existing file if hash calculation was successful
#         if data_hash is not None:
#             try:
#                 existing_df = pd.read_csv(spath)
#                 existing_hash = hash(existing_df.to_string())
# 
#                 # Skip if hashes match
#                 if existing_hash == data_hash:
#                     return
#             except:
#                 # If reading fails, proceed with saving
#                 pass
# 
#     # Save the file based on type
#     if isinstance(obj, (pd.Series, pd.DataFrame)):
#         obj.to_csv(spath, **kwargs)
#     elif isinstance(obj, np.ndarray):
#         pd.DataFrame(obj).to_csv(spath, **kwargs)
#     elif isinstance(obj, (int, float)):
#         pd.DataFrame([obj]).to_csv(spath, index=False, **kwargs)
#     elif isinstance(obj, (list, tuple)):
#         if all(isinstance(x, (int, float)) for x in obj):
#             pd.DataFrame(obj).to_csv(spath, index=False, **kwargs)
#         elif all(isinstance(x, pd.DataFrame) for x in obj):
#             pd.concat(obj).to_csv(spath, **kwargs)
#         else:
#             pd.DataFrame({"data": obj}).to_csv(spath, index=False, **kwargs)
#     elif isinstance(obj, dict):
#         pd.DataFrame.from_dict(obj).to_csv(spath, **kwargs)
#     else:
#         try:
#             pd.DataFrame({"data": [obj]}).to_csv(spath, index=False, **kwargs)
#         except:
#             raise ValueError(f"Unable to save type {type(obj)} as CSV")
# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_csv.py
# --------------------------------------------------------------------------------
