#!/usr/bin/env python3
# -*- coding: utf-8 -*-
<<<<<<< HEAD
# Timestamp: "2025-06-12 12:58:00 (ywatanabe)"
# File: /ssh:sp:/home/ywatanabe/proj/.claude-worktree/mngs_repo/src/mngs/io/_save_modules/_csv.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/io/_save_modules/_csv.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

"""
CSV saving functionality for mngs.io.save
"""
=======
# Timestamp: "2025-05-16 12:12:18 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/io/_save_modules/_csv.py
>>>>>>> origin/main

import os
import pandas as pd
import numpy as np


<<<<<<< HEAD
def save_csv(obj, spath: str, **kwargs) -> None:
    """Handle various input types for CSV saving with caching support.
    
    Parameters
    ----------
    obj : various types
        Object to save as CSV. Can be DataFrame, Series, ndarray, list, tuple, dict, or scalar
    spath : str
        Path where CSV file will be saved
    **kwargs
        Additional keyword arguments passed to pandas.DataFrame.to_csv()
        
    Notes
    -----
    - Implements caching by checking if file content would be identical
    - Automatically handles index parameter for different data types
    - Converts various data types to DataFrame before saving
=======
def _save_csv(obj, spath: str, **kwargs) -> None:
    """
    Save data to a CSV file, handling various input types appropriately.
    
    Parameters
    ----------
    obj : Any
        The object to save. Can be DataFrame, Series, ndarray, list, tuple, dict, or scalar.
    spath : str
        Path where the CSV file will be saved.
    **kwargs : dict
        Additional keyword arguments to pass to the pandas to_csv method.
        
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        If the object type cannot be converted to CSV format.
>>>>>>> origin/main
    """
    # Check if path already exists
    if os.path.exists(spath):
        # Calculate hash of new data
        data_hash = None

        # Process based on type
        if isinstance(obj, (pd.Series, pd.DataFrame)):
<<<<<<< HEAD
            # Hash DataFrames considering whether index will be saved
            if isinstance(obj, pd.DataFrame) and 'index' in kwargs and not kwargs['index']:
                data_hash = hash(obj.to_string(index=False))
            else:
                data_hash = hash(obj.to_string())
        elif isinstance(obj, np.ndarray):
            # Hash without index to match how it will be saved
            df_for_hash = pd.DataFrame(obj)
            data_hash = hash(df_for_hash.to_string(index=False))
        elif isinstance(obj, (list, tuple)):
            # Hash lists the same way they'll be saved
            if all(isinstance(x, (int, float)) for x in obj):
                df_for_hash = pd.DataFrame(obj)
                data_hash = hash(df_for_hash.to_string(index=False))
            elif all(isinstance(x, pd.DataFrame) for x in obj):
                df_for_hash = pd.concat(obj)
                data_hash = hash(df_for_hash.to_string())
            else:
                df_for_hash = pd.DataFrame({"data": obj})
                data_hash = hash(df_for_hash.to_string(index=False))
        elif isinstance(obj, dict):
            # Hash dicts the same way they'll be saved
            df_for_hash = pd.DataFrame.from_dict(obj)
            # Dicts are saved without index by default unless specified
            if 'index' not in kwargs or not kwargs['index']:
                data_hash = hash(df_for_hash.to_string(index=False))
            else:
                data_hash = hash(df_for_hash.to_string())
        elif isinstance(obj, (int, float)):
            # Hash single values the same way they'll be saved
            df_for_hash = pd.DataFrame([obj])
            data_hash = hash(df_for_hash.to_string(index=False))
        else:
            # For other types, create a DataFrame representation and hash it
            try:
                df_for_hash = pd.DataFrame({"data": [obj]})
                data_hash = hash(df_for_hash.to_string(index=False))
=======
            data_hash = hash(obj.to_string())
        elif isinstance(obj, np.ndarray):
            data_hash = hash(pd.DataFrame(obj).to_string())
        else:
            # For other types, create a string representation and hash it
            try:
                data_str = str(obj)
                data_hash = hash(data_str)
>>>>>>> origin/main
            except:
                # If we can't hash it, proceed with saving
                pass

        # Compare with existing file if hash calculation was successful
        if data_hash is not None:
            try:
                existing_df = pd.read_csv(spath)
<<<<<<< HEAD
                
                # Calculate hash based on how the data was saved
                if isinstance(obj, pd.DataFrame):
                    # DataFrames might have been saved with or without index
                    if 'index' in kwargs and kwargs['index']:
                        existing_hash = hash(existing_df.to_string())
                    else:
                        existing_hash = hash(existing_df.to_string(index=False))
                elif isinstance(obj, pd.Series):
                    existing_hash = hash(existing_df.to_string())
                elif isinstance(obj, (np.ndarray, list, tuple, int, float)):
                    # These types are saved without index
                    existing_hash = hash(existing_df.to_string(index=False))
                elif isinstance(obj, dict):
                    # Dicts are saved without index by default unless specified
                    if 'index' not in kwargs or not kwargs['index']:
                        existing_hash = hash(existing_df.to_string(index=False))
                    else:
                        existing_hash = hash(existing_df.to_string())
                else:
                    # Other types saved as {"data": [obj]}
                    existing_hash = hash(existing_df.to_string(index=False))
=======
                existing_hash = hash(existing_df.to_string())
>>>>>>> origin/main

                # Skip if hashes match
                if existing_hash == data_hash:
                    return
            except:
                # If reading fails, proceed with saving
                pass

<<<<<<< HEAD
=======
    # Set default index=False if not explicitly specified in kwargs
    if 'index' not in kwargs:
        kwargs['index'] = False
        
>>>>>>> origin/main
    # Save the file based on type
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        obj.to_csv(spath, **kwargs)
    elif isinstance(obj, np.ndarray):
<<<<<<< HEAD
        # Don't save index for numpy arrays unless explicitly requested
        if 'index' not in kwargs:
            kwargs['index'] = False
        pd.DataFrame(obj).to_csv(spath, **kwargs)
    elif isinstance(obj, (int, float)):
        pd.DataFrame([obj]).to_csv(spath, index=False, **kwargs)
    elif isinstance(obj, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in obj):
            pd.DataFrame(obj).to_csv(spath, index=False, **kwargs)
        elif all(isinstance(x, pd.DataFrame) for x in obj):
            pd.concat(obj).to_csv(spath, **kwargs)
        else:
            pd.DataFrame({"data": obj}).to_csv(spath, index=False, **kwargs)
    elif isinstance(obj, dict):
        # Don't save index for dicts unless explicitly requested
        if 'index' not in kwargs:
            kwargs['index'] = False
        pd.DataFrame.from_dict(obj).to_csv(spath, **kwargs)
    else:
        try:
            pd.DataFrame({"data": [obj]}).to_csv(spath, index=False, **kwargs)
        except:
            raise ValueError(f"Unable to save type {type(obj)} as CSV")


# EOF
=======
        pd.DataFrame(obj).to_csv(spath, **kwargs)
    elif isinstance(obj, (int, float)):
        pd.DataFrame([obj]).to_csv(spath, **kwargs)
    elif isinstance(obj, (list, tuple)):
        if all(isinstance(x, (int, float)) for x in obj):
            pd.DataFrame(obj).to_csv(spath, **kwargs)
        elif all(isinstance(x, pd.DataFrame) for x in obj):
            pd.concat(obj).to_csv(spath, **kwargs)
        else:
            pd.DataFrame({"data": obj}).to_csv(spath, **kwargs)
    elif isinstance(obj, dict):
        pd.DataFrame.from_dict(obj).to_csv(spath, **kwargs)
    else:
        try:
            pd.DataFrame({"data": [obj]}).to_csv(spath, **kwargs)
        except:
            raise ValueError(f"Unable to save type {type(obj)} as CSV")
>>>>>>> origin/main
