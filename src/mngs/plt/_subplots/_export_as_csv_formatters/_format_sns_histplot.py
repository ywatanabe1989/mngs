#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv_formatters/_format_sns_histplot.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_export_as_csv_formatters/_format_sns_histplot.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd

def _format_sns_histplot(id, tracked_dict, kwargs):
    """Format data from a sns_histplot call.
    
    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to sns_histplot
        
    Returns:
        pd.DataFrame: Formatted data for the plot
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()
    
    # If 'data' key is in tracked_dict, use it
    if 'data' in tracked_dict:
        df = tracked_dict['data']
        if isinstance(df, pd.DataFrame):
            # Add the id prefix to all columns
            return df.add_prefix(f"{id}_")
    
    # Legacy handling for args
    if 'args' in tracked_dict:
        df = tracked_dict['args']
        if isinstance(df, pd.DataFrame):
            return df.add_prefix(f"{id}_")
    
    # Default empty DataFrame if we can't process the input
    return pd.DataFrame()