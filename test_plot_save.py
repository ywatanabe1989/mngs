#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Test script that plots data and saves figures using mngs.plt

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Import mngs
import mngs

def create_sample_data():
    """Create some sample data for plotting."""
    x = np.linspace(0, 10, 100)
    sin_y = np.sin(x)
    cos_y = np.cos(x)
    tan_y = np.tan(x)
    return x, sin_y, cos_y, tan_y

def main():
    """Main function to create and save plots using mngs."""
    print("Creating sample data...")
    x, sin_y, cos_y, tan_y = create_sample_data()
    
    # Create a figure with multiple subplots using mngs.plt.subplots
    print("Creating plots with mngs.plt.subplots...")
    fig, axs = mngs.plt.subplots(nrows=2, ncols=2, figsize=(10, 8))
    
    # Plot 1: Simple line plot
    axs[0, 0].plot(x, sin_y, color='blue', id='sin_plot')
    axs[0, 0].set_title('Sine Wave')
    axs[0, 0].set_xlabel('x')
    axs[0, 0].set_ylabel('sin(x)')
    
    # Plot 2: Multiple lines with legend
    axs[0, 1].plot(x, sin_y, label='sin(x)', id='multi_sin')
    axs[0, 1].plot(x, cos_y, label='cos(x)', id='multi_cos')
    axs[0, 1].legend()
    axs[0, 1].set_title('Sine and Cosine')
    axs[0, 1].set_xlabel('x')
    axs[0, 1].set_ylabel('y')
    
    # Plot 3: Scatter plot
    axs[1, 0].scatter(x[::5], sin_y[::5], color='red', id='sin_scatter')
    axs[1, 0].set_title('Sine Scatter')
    axs[1, 0].set_xlabel('x')
    axs[1, 0].set_ylabel('sin(x)')
    
    # Plot 4: Bar chart
    categories = ['A', 'B', 'C', 'D', 'E']
    values = [5, 7, 3, 8, 2]
    axs[1, 1].bar(categories, values, id='bar_chart')
    axs[1, 1].set_title('Sample Bar Chart')
    axs[1, 1].set_xlabel('Category')
    axs[1, 1].set_ylabel('Value')
    
    # Adjust layout
    fig.tight_layout()
    
    # Save the figure with various formats
    output_base = "sample_plots"
    print(f"Saving plots to {output_base}.png and {output_base}.csv...")
    
    # Save with mngs.io.save (should create both PNG and CSV)
    mngs.io.save(fig, f"{output_base}.png", symlink_from_cwd=True)
    
    # Wait for user confirmation to close
    print("\nPlots have been saved. Checking for CSV export...")
    
    # Check if CSV file was created
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = f"{script_dir}/test_plot_save_out"
    csv_path = f"{output_dir}/{output_base}.csv"
    
    if os.path.exists(csv_path):
        # Load and examine the CSV
        df = pd.read_csv(csv_path)
        print(f"CSV file created successfully at: {csv_path}")
        print(f"CSV columns: {df.columns.tolist()}")
        print(f"CSV shape: {df.shape}")
        return True
    else:
        print(f"ERROR: CSV file was not created at: {csv_path}")
        return False

if __name__ == "__main__":
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    success = main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)
    
    if success:
        print("\nSUCCESS: Plot creation and saving with CSV export worked!")
    else:
        print("\nFAILURE: There were issues with plot creation or CSV export.")
        sys.exit(1)