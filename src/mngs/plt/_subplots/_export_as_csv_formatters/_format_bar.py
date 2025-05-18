#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv_formatters/_format_bar.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_export_as_csv_formatters/_format_bar.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd

def _format_bar(id, tracked_dict, kwargs):
    """Format data from a bar call."""
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()
    
    # Get the args from tracked_dict
    args = tracked_dict.get('args', [])
    
    # Extract x and y data if available
    if len(args) >= 2:
        x, y = args[0], args[1]
        
        # Get yerr from kwargs
        yerr = kwargs.get("yerr")
        
        # Convert single values to Series
        if isinstance(x, (int, float)):
            x = pd.Series(x, name="x")
        if isinstance(y, (int, float)):
            y = pd.Series(y, name="y")
    else:
        # Not enough arguments
        return pd.DataFrame()

    df = pd.DataFrame({f"{id}_bar_x": x, f"{id}_bar_y": y})

    if yerr is not None:
        if isinstance(yerr, (int, float)):
            yerr = pd.Series(yerr, name="yerr")
        df[f"{id}_bar_yerr"] = yerr
    return df