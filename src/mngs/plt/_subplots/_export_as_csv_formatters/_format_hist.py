#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv_formatters/_format_hist.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_export_as_csv_formatters/_format_hist.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd

def _format_hist(id, tracked_dict, kwargs):
    """Format data from a hist call."""
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()
    
    # Get the args from tracked_dict
    args = tracked_dict.get('args', [])
    
    # Extract data if available
    if len(args) >= 1:
        x = args[0]
        df = pd.DataFrame({f"{id}_hist_x": x})
        return df
    
    return pd.DataFrame()