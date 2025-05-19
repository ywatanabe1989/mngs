#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv_formatters/_format_plot_line.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_export_as_csv_formatters/_format_plot_line.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd

def _format_plot_line(id, tracked_dict, kwargs):
    """Format data from a plot_line call.
    
    Processes plot_line data for CSV export.
    
    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing 'plot_df' key with plot data
        kwargs (dict): Keyword arguments passed to plot_line
        
    Returns:
        pd.DataFrame: Formatted line plot data
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()
    
    # Get the plot_df from tracked_dict
    plot_df = tracked_dict.get('plot_df')
    
    if plot_df is None or not isinstance(plot_df, pd.DataFrame):
        return pd.DataFrame()
    
    # Create a copy to avoid modifying the original
    result = plot_df.copy()
    
    # Add prefix to column names if ID is provided
    if id is not None:
        # Rename columns with ID prefix
        result.columns = [f"{id}_line_{col}" for col in result.columns]
    
    return result