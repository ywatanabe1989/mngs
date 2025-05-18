#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv_formatters/_format_violin.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_export_as_csv_formatters/_format_violin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd

def _format_violin(id, tracked_dict, kwargs):
    """Format data from a violin call.
    
    Formats data in a long-format for better compatibility.
    
    Args:
        id (str): Identifier for the plot
        tracked_dict (dict): Dictionary containing tracked data
        kwargs (dict): Keyword arguments passed to violin plot
        
    Returns:
        pd.DataFrame: Formatted violin data in long format
    """
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()
    
    # Get the args from tracked_dict
    args = tracked_dict.get('args', [])
    
    # Similar to boxplot but shows probability density
    if len(args) >= 1:
        data = args[0]
        
        # Handle case when data is a simple array or list
        if isinstance(data, (list, np.ndarray)) and not isinstance(data[0], (list, np.ndarray, dict)):
            # Convert to long format with group and value columns
            rows = [{'group': '0', 'value': val} for val in data]
            df = pd.DataFrame(rows)
            # Prefix columns with id
            df.columns = [f"{id}_violin_{col}" for col in df.columns]
            return df
            
        # Handle case when data is a dictionary
        elif isinstance(data, dict):
            # Convert to long format with group and value columns
            rows = []
            for group, values in data.items():
                for val in values:
                    rows.append({'group': str(group), 'value': val})
            
            if rows:
                df = pd.DataFrame(rows)
                # Prefix columns with id
                df.columns = [f"{id}_violin_{col}" for col in df.columns]
                return df
    
    # If we get here, either no data or unsupported format
    return pd.DataFrame()