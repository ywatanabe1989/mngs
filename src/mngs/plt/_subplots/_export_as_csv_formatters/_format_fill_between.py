#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-18 18:14:26 (ywatanabe)"
# File: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/plt/_subplots/_export_as_csv_formatters/_format_fill_between.py
# ----------------------------------------
import os
__FILE__ = (
    "./src/mngs/plt/_subplots/_export_as_csv_formatters/_format_fill_between.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import pandas as pd

def _format_fill_between(id, tracked_dict, kwargs):
    """Format data from a fill_between call."""
    # Check if tracked_dict is empty or not a dictionary
    if not tracked_dict or not isinstance(tracked_dict, dict):
        return pd.DataFrame()
    
    # Get the args from tracked_dict
    args = tracked_dict.get('args', [])
    
    # Typical args: x, y1, y2
    if len(args) >= 3:
        x, y1, y2 = args[:3]

        df = pd.DataFrame(
            {
                f"{id}_fill_between_x": x,
                f"{id}_fill_between_y1": y1,
                f"{id}_fill_between_y2": y2,
            }
        )
        return df
    return pd.DataFrame()