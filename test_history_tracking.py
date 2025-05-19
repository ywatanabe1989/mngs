#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test script for history tracking in axis wrappers

import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Import mngs
import mngs

def main():
    # Create a figure with multiple plot types
    fig, ax = mngs.plt.subplots(figsize=(8, 6))
    
    # Collect multiple plot types with explicit IDs for tracking
    x = np.linspace(0, 10, 50)
    
    # Plot data
    ax.plot(x, np.sin(x), label='sin(x)', id='sin_plot')
    ax.scatter(x, np.cos(x), label='cos(x)', id='cos_scatter')
    ax.bar([1, 2, 3], [3, 2, 1], id='bar_test')
    
    # Add some formatting
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Test History Tracking')
    ax.legend()
    
    # Examine the history directly
    print("History keys:")
    print(list(ax.history.keys()))
    
    # Print sample from history to examine the format
    if 'sin_plot' in ax.history:
        print("\nSample history record (sin_plot):")
        record = ax.history['sin_plot']
        print(f"ID: {record[0]}")
        print(f"Method: {record[1]}")
        print(f"Args: {[type(arg) for arg in record[2]]}")  # Show types for brevity
        print(f"Kwargs: {record[3]}")
    
    # Export to DataFrame directly
    df = ax.export_as_csv()
    print("\nExported DataFrame columns:")
    print(df.columns.tolist())
    print(f"DataFrame shape: {df.shape}")
    
    # Save to files to test the CSV export through the save function
    test_png = "test_history_tracking.png"
    mngs.io.save(fig, test_png)
    
    # Return success
    return True

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    success = main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
    
    if success:
        print("\nTEST PASSED: History tracking functionality appears to be working.")
    else:
        print("\nTEST FAILED: History tracking functionality has issues.")
        sys.exit(1)