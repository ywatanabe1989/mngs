#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import directly from the package
from mngs.plt._subplots._export_as_csv import (
    export_as_csv, format_record, 
    format_plot, format_scatter, format_bar, format_barh,
    format_hist, format_boxplot, format_contour, format_errorbar,
    format_eventplot, format_fill, format_fill_between, format_imshow,
    format_violin, format_violinplot
)

def test_basic_formatters():
    """Test basic matplotlib formatters."""
    print("\nTesting basic matplotlib formatters...")
    
    # Test plot formatter
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    print("Testing plot formatter...")
    result = format_plot("plot1", (x, y), {})
    assert isinstance(result, pd.DataFrame)
    assert "plot1_plot_x" in result.columns
    assert "plot1_plot_y" in result.columns
    print("Plot formatter test passed!")
    
    print("Testing scatter formatter...")
    result = format_scatter("scatter1", (x, y), {})
    assert isinstance(result, pd.DataFrame)
    assert "scatter1_scatter_x" in result.columns
    assert "scatter1_scatter_y" in result.columns
    print("Scatter formatter test passed!")
    
    print("Testing bar formatter...")
    categories = ["A", "B", "C"]
    values = [4, 5, 6]
    result = format_bar("bar1", (categories, values), {})
    assert isinstance(result, pd.DataFrame)
    assert "bar1_bar_x" in result.columns
    assert "bar1_bar_y" in result.columns
    print("Bar formatter test passed!")
    
    print("Testing barh formatter...")
    result = format_barh("barh1", (categories, values), {})
    assert isinstance(result, pd.DataFrame)
    assert "barh1_barh_x" in result.columns
    assert "barh1_barh_y" in result.columns
    print("Barh formatter test passed!")
    
    print("Testing hist formatter...")
    data = [1, 2, 2, 3, 3, 3, 4, 4, 5]
    result = format_hist("hist1", data, {})
    assert isinstance(result, pd.DataFrame)
    assert "hist1_hist_x" in result.columns
    print("Hist formatter test passed!")
    
    print("All basic matplotlib formatter tests passed!")

def test_complex_formatters():
    """Test more complex matplotlib formatters."""
    print("\nTesting complex matplotlib formatters...")
    
    print("Testing contour formatter...")
    x = np.linspace(-3, 3, 5)
    y = np.linspace(-3, 3, 5)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) + np.cos(Y)
    result = format_contour("contour1", (X, Y, Z), {})
    assert isinstance(result, pd.DataFrame)
    assert "contour1_contour_x" in result.columns
    assert "contour1_contour_y" in result.columns
    assert "contour1_contour_z" in result.columns
    print("Contour formatter test passed!")
    
    print("Testing errorbar formatter...")
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    yerr = np.array([0.1, 0.2, 0.3])
    # Just check that it doesn't throw an error
    result = format_errorbar("errorbar1", (x, y), {"yerr": yerr})
    assert isinstance(result, pd.DataFrame)
    print("Errorbar formatter test passed!")
    
    print("Testing fill formatter...")
    x = np.array([1, 2, 3, 4])
    y1 = np.array([1, 4, 2, 3])
    y2 = np.array([2, 3, 4, 5])
    result = format_fill("fill1", (x, y1, y2), {})
    assert isinstance(result, pd.DataFrame)
    print("Fill formatter test passed!")
    
    print("Testing fill_between formatter...")
    result = format_fill_between("fillbw1", (x, y1, y2), {})
    assert isinstance(result, pd.DataFrame)
    print("Fill_between formatter test passed!")
    
    print("Testing imshow formatter...")
    img = np.random.rand(5, 5)
    result = format_imshow("im1", (img,), {})
    assert isinstance(result, pd.DataFrame)
    print("Imshow formatter test passed!")
    
    print("All complex matplotlib formatter tests passed!")

def test_export_as_csv_integration():
    """Test integration with export_as_csv function."""
    print("\nTesting integration with export_as_csv...")
    
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    
    # Create a records dictionary with multiple plot types
    history = {
        "plot1": ("plot1", "plot", (x, y), {}),
        "scatter1": ("scatter1", "scatter", (x, y), {})
    }
    
    # Export to CSV
    result = export_as_csv(history)
    assert isinstance(result, pd.DataFrame)
    assert "plot1_plot_x" in result.columns
    assert "plot1_plot_y" in result.columns
    assert "scatter1_scatter_x" in result.columns
    assert "scatter1_scatter_y" in result.columns
    
    print("Integration test passed!")

if __name__ == "__main__":
    print("Running individual formatter tests...\n")
    
    try:
        test_basic_formatters()
        test_complex_formatters()
        test_export_as_csv_integration()
        print("\nAll tests completed successfully!")
    except AssertionError as e:
        print(f"\nTest failed: {e}")
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()