#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-30 02:10:00 (Claude)"
# File: examples/mngs/plt/enhanced_plotting.py

"""
Enhanced plotting with mngs.plt

This example demonstrates:
- Using mngs.plt.subplots for automatic data tracking
- Setting labels with ax.set_xyt
- Automatic CSV export alongside plots
- Integration with mngs.io.save
"""

import sys
import numpy as np
import pandas as pd
import mngs

def example_basic_plotting():
    """Basic plotting with data tracking."""
    print("\n1. Basic Line Plot with Data Tracking")
    
    # Generate sample data
    x = np.linspace(0, 2*np.pi, 100)
    y1 = np.sin(x)
    y2 = np.cos(x)
    
    # Create figure with mngs
    fig, ax = mngs.plt.subplots(figsize=(8, 6))
    
    # Plot data
    ax.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
    ax.plot(x, y2, label='cos(x)', color='red', linewidth=2)
    
    # Use mngs enhanced labeling
    ax.set_xyt(
        xlabel="x (radians)",
        ylabel="y value",
        title="Trigonometric Functions"
    )
    
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save - this creates both .png and .csv files!
    mngs.io.save(fig, "./output/plots/trig_functions.png")
    print("   ✅ Saved: trig_functions.png AND trig_functions.png.csv")

def example_multi_panel():
    """Multi-panel figure with different plot types."""
    print("\n2. Multi-panel Figure")
    
    # Create 2x2 subplot
    fig, axes = mngs.plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: Scatter plot
    n_points = 200
    x_scatter = np.random.randn(n_points)
    y_scatter = 2 * x_scatter + np.random.randn(n_points) * 0.5
    
    axes[0, 0].scatter(x_scatter, y_scatter, alpha=0.5, s=30)
    axes[0, 0].set_xyt("X values", "Y values", "Scatter Plot")
    
    # Panel 2: Histogram
    data_hist = np.random.normal(100, 15, 1000)
    axes[0, 1].hist(data_hist, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xyt("Value", "Frequency", "Normal Distribution")
    
    # Panel 3: Bar plot
    categories = ['A', 'B', 'C', 'D', 'E']
    values = np.random.randint(10, 100, len(categories))
    axes[1, 0].bar(categories, values, color='orange', alpha=0.8)
    axes[1, 0].set_xyt("Category", "Count", "Bar Chart")
    
    # Panel 4: Time series
    time = pd.date_range('2024-01-01', periods=100, freq='D')
    signal = np.cumsum(np.random.randn(100)) + 50
    axes[1, 1].plot(time, signal, color='purple', linewidth=1.5)
    axes[1, 1].set_xyt("Date", "Value", "Time Series")
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    mngs.io.save(fig, "./output/plots/multi_panel_demo.png")
    print("   ✅ Saved: multi_panel_demo.png with all plotted data")

def example_statistical_plot():
    """Statistical visualization with error bars."""
    print("\n3. Statistical Plot with Error Bars")
    
    # Generate grouped data
    groups = ['Control', 'Treatment A', 'Treatment B', 'Treatment C']
    n_samples = 30
    
    # Simulate experimental data
    data = {
        'Control': np.random.normal(100, 10, n_samples),
        'Treatment A': np.random.normal(110, 12, n_samples),
        'Treatment B': np.random.normal(105, 8, n_samples),
        'Treatment C': np.random.normal(115, 15, n_samples)
    }
    
    # Calculate statistics
    means = [np.mean(data[g]) for g in groups]
    stds = [np.std(data[g]) for g in groups]
    sems = [s/np.sqrt(n_samples) for s in stds]
    
    # Create figure
    fig, ax = mngs.plt.subplots(figsize=(10, 6))
    
    # Bar plot with error bars
    x_pos = np.arange(len(groups))
    bars = ax.bar(x_pos, means, yerr=sems, capsize=10, 
                   color=['gray', 'blue', 'green', 'red'],
                   alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Customize
    ax.set_xticks(x_pos)
    ax.set_xticklabels(groups)
    ax.set_xyt("Group", "Mean Value ± SEM", "Treatment Comparison")
    
    # Add significance stars (mock)
    ax.text(1, means[1] + sems[1] + 2, '*', ha='center', fontsize=16)
    ax.text(3, means[3] + sems[3] + 2, '***', ha='center', fontsize=16)
    
    # Add grid
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(means) * 1.3)
    
    mngs.io.save(fig, "./output/plots/statistical_comparison.png")
    print("   ✅ Saved: statistical_comparison.png with statistics")

def example_custom_styling():
    """Demonstrate custom styling and color usage."""
    print("\n4. Custom Styling Example")
    
    # Generate data for multiple series
    x = np.linspace(0, 10, 50)
    
    # Create figure with custom size
    fig, ax = mngs.plt.subplots(figsize=(10, 6))
    
    # Plot multiple series with different styles
    styles = ['-', '--', '-.', ':']
    markers = ['o', 's', '^', 'D']
    
    for i in range(4):
        y = np.sin(x + i * np.pi/4) * np.exp(-x/10) * (i + 1)
        ax.plot(x, y, 
                linestyle=styles[i],
                marker=markers[i],
                markevery=5,
                markersize=8,
                linewidth=2,
                label=f'Series {i+1}',
                alpha=0.8)
    
    # Styling
    ax.set_xyt(
        xlabel="Time (s)",
        ylabel="Amplitude",
        title="Damped Oscillations"
    )
    
    ax.legend(loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set axis limits
    ax.set_xlim(0, 10)
    ax.set_ylim(-4, 4)
    
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
    
    mngs.io.save(fig, "./output/plots/custom_styling.png")
    print("   ✅ Saved: custom_styling.png with style demo")

def example_data_export():
    """Demonstrate the CSV export functionality."""
    print("\n5. Data Export Demonstration")
    
    # Create simple plot
    fig, ax = mngs.plt.subplots()
    
    x = np.array([1, 2, 3, 4, 5])
    y1 = np.array([1, 4, 9, 16, 25])
    y2 = np.array([1, 8, 27, 64, 125])
    
    ax.plot(x, y1, 'o-', label='x²')
    ax.plot(x, y2, 's-', label='x³')
    ax.set_xyt("x", "y", "Power Functions")
    ax.legend()
    
    # Save the plot
    output_path = "./output/plots/power_functions.png"
    mngs.io.save(fig, output_path)
    
    # Load and display the automatically exported CSV
    csv_path = output_path + ".csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"\n   📊 Exported data shape: {df.shape}")
        print(f"   📊 Columns: {list(df.columns)}")
        print(f"\n   First few rows of exported data:")
        print(df.head())

def main():
    """Run all examples."""
    print("=== MNGS Enhanced Plotting Examples ===")
    
    # Create output directory
    os.makedirs("./output/plots", exist_ok=True)
    
    # Run examples
    example_basic_plotting()
    example_multi_panel()
    example_statistical_plot()
    example_custom_styling()
    example_data_export()
    
    print("\n✅ All plotting examples completed!")
    print("📁 Check ./output/plots/ for generated files")
    print("📊 Each .png file has a corresponding .csv with the plotted data!")

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    
    # Option 1: Simple usage
    main()
    plt.close('all')
    
    # Option 2: With full mngs workflow
    # CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
    #     sys=sys,
    #     plt=plt,
    #     fig_size_mm=(160, 100),
    #     dpi_save=300
    # )
    # main()
    # mngs.gen.close(CONFIG)