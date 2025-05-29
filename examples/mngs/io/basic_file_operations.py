#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-30 02:00:00 (Claude)"
# File: examples/mngs/io/basic_file_operations.py

"""
Basic file I/O operations with mngs.io

This example demonstrates:
- Loading various file formats
- Saving data with automatic directory creation
- Working with different data types
"""

import sys
import numpy as np
import pandas as pd
import mngs

def main():
    """Demonstrate basic file I/O operations."""
    
    print("=== MNGS I/O Examples ===\n")
    
    # 1. Working with NumPy arrays
    print("1. NumPy Arrays")
    arr = np.random.randn(100, 50)
    mngs.io.save(arr, "./output/arrays/random_data.npy")
    loaded_arr = mngs.io.load("./output/arrays/random_data.npy")
    print(f"   Saved and loaded array shape: {loaded_arr.shape}")
    
    # 2. Working with DataFrames
    print("\n2. Pandas DataFrames")
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100),
        'value': np.random.randn(100),
        'category': np.random.choice(['A', 'B', 'C'], 100)
    })
    mngs.io.save(df, "./output/dataframes/timeseries.csv", index=False)
    loaded_df = mngs.io.load("./output/dataframes/timeseries.csv")
    print(f"   Saved and loaded DataFrame shape: {loaded_df.shape}")
    
    # 3. Working with dictionaries/JSON
    print("\n3. Dictionaries and JSON")
    config = {
        'experiment': {
            'name': 'test_run',
            'seed': 42,
            'parameters': {
                'learning_rate': 0.001,
                'batch_size': 32
            }
        },
        'data': {
            'train_size': 0.8,
            'val_size': 0.1,
            'test_size': 0.1
        }
    }
    mngs.io.save(config, "./output/configs/experiment_config.json")
    mngs.io.save(config, "./output/configs/experiment_config.yaml")
    loaded_json = mngs.io.load("./output/configs/experiment_config.json")
    print(f"   Saved config with {len(config)} top-level keys")
    
    # 4. Working with compressed numpy arrays
    print("\n4. Compressed NumPy Arrays")
    data_dict = {
        'features': np.random.randn(1000, 128),
        'labels': np.random.randint(0, 10, 1000),
        'metadata': np.array(['sample_' + str(i) for i in range(1000)])
    }
    mngs.io.save(data_dict, "./output/arrays/compressed_data.npz")
    loaded_npz = mngs.io.load("./output/arrays/compressed_data.npz")
    print(f"   Saved {len(data_dict)} arrays in compressed format")
    
    # 5. Working with text files
    print("\n5. Text Files")
    report = """Experiment Report
==================

Date: 2025-05-30
Model: MNGS Example
Results: Success

This is a sample report demonstrating text file handling.
"""
    mngs.io.save(report, "./output/reports/experiment_report.txt")
    mngs.io.save(report, "./output/reports/experiment_report.md")
    loaded_text = mngs.io.load("./output/reports/experiment_report.txt")
    print(f"   Saved text report with {len(loaded_text.splitlines())} lines")
    
    # 6. Demonstrating automatic directory creation
    print("\n6. Automatic Directory Creation")
    nested_data = {"deep": {"nested": {"structure": "works"}}}
    mngs.io.save(
        nested_data, 
        "./output/very/deep/nested/directory/structure/data.json"
    )
    print("   Created nested directory structure automatically")
    
    # 7. Working with lists of dataframes
    print("\n7. Lists and Collections")
    df_list = [
        pd.DataFrame({'x': np.random.randn(10), 'y': np.random.randn(10)})
        for _ in range(5)
    ]
    # Save as concatenated CSV
    mngs.io.save(df_list, "./output/collections/all_dataframes.csv")
    print(f"   Saved {len(df_list)} DataFrames as single CSV")
    
    print("\n✅ All examples completed successfully!")
    print(f"📁 Check the './output/' directory for all saved files")


if __name__ == "__main__":
    # Basic usage without mngs.gen.start
    main()
    
    # Alternative: Full mngs workflow
    # CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(sys)
    # main()
    # mngs.gen.close(CONFIG)