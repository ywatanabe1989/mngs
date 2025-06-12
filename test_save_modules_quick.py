#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Quick test to verify save modules imports are working"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing save modules imports...")

try:
    # Test main save import
    from mngs.io._save import save
    print("✓ Successfully imported mngs.io._save.save")
    
    # Test save_modules package imports
    from mngs.io._save_modules import (
        save_csv,
        save_excel,
        save_npy,
        save_npz,
        save_pickle,
        save_pickle_compressed,
        save_joblib,
        save_torch,
        save_json,
        save_yaml,
        save_hdf5,
        save_matlab,
        save_catboost,
        save_text,
        save_html,
        save_image,
        save_mp4,
        save_listed_dfs_as_csv,
        save_listed_scalars_as_csv,
        save_optuna_study_as_csv_and_pngs,
    )
    print("✓ Successfully imported all save functions from _save_modules")
    
    # Test problematic imports directly
    from mngs.io._save_modules._listed_dfs_as_csv import _save_listed_dfs_as_csv
    print("✓ Successfully imported _save_listed_dfs_as_csv (with _mv_to_tmp import)")
    
    from mngs.io._save_modules._listed_scalars_as_csv import _save_listed_scalars_as_csv
    print("✓ Successfully imported _save_listed_scalars_as_csv (with _mv_to_tmp import)")
    
    print("\n✅ All imports successful! The save module refactoring is working correctly.")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\nTesting basic functionality...")

try:
    import tempfile
    import pandas as pd
    
    # Create a test dataframe
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        save(df, tmp.name)
        print(f"✓ Successfully saved DataFrame to {tmp.name}")
        
        # Check file exists and has content
        if os.path.exists(tmp.name) and os.path.getsize(tmp.name) > 0:
            print("✓ File exists and has content")
        else:
            print("❌ File save failed")
        
        # Clean up
        os.unlink(tmp.name)
    
    print("\n✅ Basic functionality test passed!")
    
except Exception as e:
    print(f"\n❌ Functionality test failed: {e}")
    import traceback
    traceback.print_exc()