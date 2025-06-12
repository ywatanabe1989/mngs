#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 17:01:32 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_export_as_csv.py
# ----------------------------------------
import os

__FILE__ = "./src/mngs/plt/_subplots/_export_as_csv.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import sys
import warnings

import matplotlib.pyplot as plt
import mngs
import numpy as np
import pandas as pd
import xarray as xr

from mngs.pd import to_xyz


def _to_numpy(data):
    """Convert various data types to numpy array.
    
    Handles torch tensors, pandas Series/DataFrame, and other array-like objects.
    
    Parameters
    ----------
    data : array-like
        Data to convert to numpy array
        
    Returns
    -------
    numpy.ndarray
        Data as numpy array
    """
    if hasattr(data, 'numpy'):  # torch tensor
        return data.detach().numpy() if hasattr(data, 'detach') else data.numpy()
    elif hasattr(data, 'values'):  # pandas series/dataframe
        return data.values
    else:
        return np.asarray(data)


def export_as_csv(history_records):
    """Export plotting history records as a pandas DataFrame.

    Converts the plotting history records maintained by MNGS plotting
    functions into a pandas DataFrame suitable for CSV export. This allows
    you to save the exact data that was plotted for reproducibility and
    further analysis.

    Parameters
    ----------
    history_records : dict
        Dictionary of plotting records, typically from FigWrapper or
        AxesWrapper history. Each record contains information about
        plotted data.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the plotted data from all records.
        Returns empty DataFrame if no records found or concatenation fails.

    Warnings
    --------
    UserWarning
        If no plotting records are found or if concatenation fails.

    Examples
    --------
    >>> fig, ax = mngs.plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6], label='data1')
    >>> ax.plot([1, 2, 3], [7, 8, 9], label='data2')
    >>> df = export_as_csv(ax.history)
    >>> df.to_csv('plotted_data.csv')

    >>> # Access specific plot data
    >>> fig, axes = mngs.plt.subplots(2, 2)
    >>> # ... plotting code ...
    >>> df = export_as_csv(fig.history)
    >>> print(df.columns)  # Shows all plotted series

    See Also
    --------
    format_record : Formats individual plotting records
    mngs.plt.subplots : Creates figures with history tracking

    Notes
    -----
    The history tracking feature is unique to MNGS plotting functions
    and provides automatic data export capabilities.
    """
    if len(history_records) <= 0:
        warnings.warn("Plotting records not found. Empty dataframe returned. "
                     "Ensure plots are created with tracking enabled (track=True) "
                     "and that plotting methods like plot(), scatter(), bar() etc. are used.")
        return pd.DataFrame()
    else:
        dfs = []
        failed_records = []
        
        for record_id, record in history_records.items():
            try:
                if record and len(record) >= 2:
                    plot_id, method_name = record[0], record[1]
                    formatted = format_record(record)
                    if formatted is not None and not formatted.empty:
                        dfs.append(formatted)
                    else:
                        failed_records.append(f"id={plot_id}, method={method_name} (returned empty)")
                else:
                    failed_records.append(f"id={record_id} (invalid record structure)")
            except Exception as e:
                method_name = record[1] if record and len(record) >= 2 else "unknown"
                failed_records.append(f"id={record_id}, method={method_name} (error: {str(e)[:50]})")
        
        if not dfs:
            warnings.warn(
                f"No valid plotting records could be formatted from {len(history_records)} records. "
                f"Failed records: [{', '.join(failed_records)}]. "
                f"Check that supported plot types (plot, scatter, bar, hist, boxplot, etc.) are being used."
            )
            return pd.DataFrame()
            
        try:
            df = pd.concat(dfs, axis=1)
            if failed_records:
                warnings.warn(
                    f"Successfully exported {len(dfs)} records, but {len(failed_records)} failed: "
                    f"[{', '.join(failed_records)}]"
                )
            return df
        except Exception as e:
            warnings.warn(
                f"Failed to combine {len(dfs)} valid DataFrames. "
                f"Error: {str(e)}. "
                f"Records: {[f'id={r[0]}, method={r[1]}' for r in history_records.values() if r and len(r) >= 2]}"
            )
            return pd.DataFrame()


def _format_imshow2d(record):
    id, method, args, kwargs = record
    df = args
    # df.columns = [f"{id}_{method}_{col}" for col in df.columns]
    # df.index = [f"{id}_{method}_{idx}" for idx in df.index]
    return df


def format_record(record):
    """Format a single plotting record for CSV export.
    
    Returns None if the record cannot be formatted.
    """
    try:
        id, method, args, kwargs = record
    except ValueError as e:
        warnings.warn(f"Invalid record format: {record}. Error: {e}")
        return None

    if method == "imshow2d":
        return _format_imshow2d(record)

    elif method in ["plot"]:
        # Convert torch tensors to numpy arrays if needed
        # Using module-level _to_numpy function
        # Extract x, y data from matplotlib plot arguments
        # Matplotlib plot accepts: plot(y), plot(x,y), plot(x,y,fmt), plot(x,y,fmt,x2,y2,fmt2,...)
        plot_data = []
        i = 0
        while i < len(args):
            if i == 0 and len(args) == 1:
                # Single argument: y values with implicit x
                y = _to_numpy(args[0])
                x = np.arange(len(y))
                plot_data.append((x, y))
                break
            elif i + 1 < len(args):
                # Check if we have x, y pair
                potential_x = args[i]
                potential_y = args[i + 1]
                
                # Skip if either is a string (format specifier)
                if isinstance(potential_x, str) or isinstance(potential_y, str):
                    i += 1
                    continue
                    
                try:
                    x = _to_numpy(potential_x)
                    y = _to_numpy(potential_y)
                    plot_data.append((x, y))
                    i += 2
                    
                    # Skip format string if present
                    if i < len(args) and isinstance(args[i], str):
                        i += 1
                except:
                    i += 1
            else:
                i += 1
        
        if not plot_data:
            return None
            
        # Format the extracted data
        if len(plot_data) == 1:
            x, y = plot_data[0]
            if hasattr(y, 'ndim') and y.ndim == 2:
                from collections import OrderedDict
                out = OrderedDict()
                for ii in range(y.shape[1]):
                    out[f"{id}_{method}_x{ii:02d}"] = x
                    out[f"{id}_{method}_y{ii:02d}"] = y[:, ii]
                df = pd.DataFrame(out)
                return df
            else:
                # 1D y data
                try:
                    df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
                    return df
                except Exception:
                    return None
        else:
            # Multiple line plots
            from collections import OrderedDict
            out = OrderedDict()
            for plot_idx, (x, y) in enumerate(plot_data):
                try:
                    out[f"{id}_{method}_x{plot_idx:02d}"] = x
                    out[f"{id}_{method}_y{plot_idx:02d}"] = y
                except:
                    continue
            if out:
                try:
                    df = pd.DataFrame(out)
                    return df
                except:
                    return None
            return None

    elif method == "scatter":
        # Using module-level _to_numpy function
        x, y = args[:2]  # Handle additional args like 's', 'c', etc.
        df_data = {f"{id}_{method}_x": _to_numpy(x), f"{id}_{method}_y": _to_numpy(y)}
        
        # Handle size and color data if present
        if len(args) > 2:
            s = args[2]  # sizes
            df_data[f"{id}_{method}_s"] = _to_numpy(s)
        if len(args) > 3:
            c = args[3]  # colors
            df_data[f"{id}_{method}_c"] = _to_numpy(c)
            
        df = pd.DataFrame(df_data)
        return df

    elif method in ["bar", "barh"]:
        # Using module-level _to_numpy function
        x, y = args
        yerr = kwargs.get("yerr")

        if isinstance(x, (int, float)):
            x = pd.Series(x, name="x")
        if isinstance(y, (int, float)):
            y = pd.Series(y, name="y")

        df = pd.DataFrame({f"{id}_{method}_x": _to_numpy(x), f"{id}_{method}_y": _to_numpy(y)})

        if yerr is not None:
            if isinstance(yerr, (int, float)):
                yerr = pd.Series(yerr, name="yerr")
            df[f"{id}_{method}_yerr"] = _to_numpy(yerr)
        return df

    elif method == "hist":
        # Using module-level _to_numpy function
        x = args[0]
        df = pd.DataFrame({f"{id}_{method}_x": _to_numpy(x)})
        return df

    elif method == "plot_line":
        # Handle plot_line method which stores data as plot_df
        # The args should contain a tracked_dict with plot_df
        if args and isinstance(args[0], dict) and 'plot_df' in args[0]:
            plot_df = args[0]['plot_df']
            # Rename columns to include id and method
            if isinstance(plot_df, pd.DataFrame):
                renamed_cols = {}
                for col in plot_df.columns:
                    renamed_cols[col] = f"{id}_{method}_{col}"
                df = plot_df.rename(columns=renamed_cols)
                return df
        # Fallback to standard handling if plot_df not found
        elif len(args) >= 1:
            # Using module-level _to_numpy function
            # Extract data and optional xx
            data = _to_numpy(args[0])
            xx = _to_numpy(args[1]) if len(args) > 1 else np.arange(len(data))
            
            df = pd.DataFrame({f"{id}_{method}_x": xx, f"{id}_{method}_y": data})
            return df
        else:
            return None

    elif method in ["boxplot", "violinplot"]:
        # Using module-level _to_numpy function
        # Define is_listed_X function
        try:
            from mngs.types import is_listed_X as mngs_types_is_listed_X
            is_listed_X = mngs_types_is_listed_X
        except ImportError:
            def is_listed_X(x, types):
                try:
                    return all(isinstance(item, tuple(types)) for item in x)
                except (TypeError, ValueError):
                    return False
                    
        x = args[0]

        # One box/violin plot
        if isinstance(x, np.ndarray) or is_listed_X(x, [float, int]):
            df = pd.DataFrame({f"{id}_{method}_0_x": _to_numpy(x)})

        else:
            # Multiple boxes/violins
            try:
                import mngs.pd
                df = mngs.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})
                df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
                df = df.apply(lambda col: col.dropna().reset_index(drop=True))
            except (ImportError, AttributeError):
                # Fallback implementation
                max_len = max(len(arr) for arr in x)
                data_dict = {}
                for i_x, _x in enumerate(x):
                    arr = _to_numpy(_x)
                    # Pad shorter arrays with NaN
                    if len(arr) < max_len:
                        arr = np.concatenate([arr, np.full(max_len - len(arr), np.nan)])
                    data_dict[f"{id}_{method}_{i_x}_x"] = arr
                df = pd.DataFrame(data_dict)
            
        return df

    # elif method == "boxplot_":
    #     __import__("ipdb").set_trace()
    #     x = args[0]
    #     df =
    #     df.columns = [f"{id}_{method}_{col}" for col in df.columns]

    #     return df

    # elif method == "plot_":
    #     df = args
    #     df.columns = [f"{id}_{method}_{col}" for col in df.columns]
    #     return df

    elif method == "plot_fillv":
        starts, ends = args
        df = pd.DataFrame(
            {
                f"{id}_{method}_starts": starts,
                f"{id}_{method}_ends": ends,
            }
        )
        return df

    elif method == "plot_raster":
        df = args
        return df

    elif method == "plot_ecdf":
        df = args
        return df

    elif method == "plot_kde":
        df = args
        if id is not None:
            df.columns = [f"{id}_{method}_{col}" for col in df.columns]
        return df

    elif method == "sns_barplot":
        # Similar handling as sns_lineplot
        if isinstance(args, pd.DataFrame):
            df = args.copy()
            
            # Try to get original column names from kwargs
            x = kwargs.get('x')
            y = kwargs.get('y') 
            hue = kwargs.get('hue')
            
            # For barplot, if it's a pivot table with diagonal values, extract them
            if hasattr(df, 'values') and df.values.ndim == 2 and df.shape[0] == df.shape[1]:
                # Extract diagonal for aggregated data
                df = pd.DataFrame(pd.Series(np.array(df).diagonal(), index=df.columns)).T
            
            # Create meaningful column names
            if x and y:
                renamed_cols = {}
                for col in df.columns:
                    if hue and '-' in str(col):
                        hue_val = str(col).split('-', 1)[-1]
                        renamed_cols[col] = f"{id}_{method}_{y}_{hue}_{hue_val}"
                    else:
                        if str(col) == x:
                            renamed_cols[col] = f"{id}_{method}_{x}"
                        elif str(col) == y:
                            renamed_cols[col] = f"{id}_{method}_{y}"
                        elif str(col) == hue:
                            renamed_cols[col] = f"{id}_{method}_{hue}"
                        else:
                            renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            else:
                renamed_cols = {}
                for col in df.columns:
                    renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            
            return df
        return None

    elif method == "sns_boxplot":
        # Similar handling as other seaborn plots
        if isinstance(args, pd.DataFrame):
            df = args.copy()
            
            # Try to get original column names from kwargs
            x = kwargs.get('x')
            y = kwargs.get('y') 
            hue = kwargs.get('hue')
            
            # Create meaningful column names
            if x and y:
                renamed_cols = {}
                for col in df.columns:
                    if hue and '-' in str(col):
                        hue_val = str(col).split('-', 1)[-1]
                        renamed_cols[col] = f"{id}_{method}_{y}_{hue}_{hue_val}"
                    else:
                        if str(col) == x:
                            renamed_cols[col] = f"{id}_{method}_{x}"
                        elif str(col) == y:
                            renamed_cols[col] = f"{id}_{method}_{y}"
                        elif str(col) == hue:
                            renamed_cols[col] = f"{id}_{method}_{hue}"
                        else:
                            renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            else:
                # Fallback to generic naming
                renamed_cols = {}
                for col in df.columns:
                    renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            
            return df
        return None

    elif method == "sns_heatmap":
        df = args
        return df

    elif method == "sns_histplot":
        df = args
        return df

    elif method == "sns_kdeplot":
        # KDE plot data is not easily exportable as it's a density estimation
        warnings.warn(
            f"Export for sns_kdeplot (id='{id}') is not yet implemented. "
            "KDE plots show density estimations which are computed internally by seaborn."
        )
        return None

    elif method == "sns_lineplot":
        # Extract the tracked data from args
        # For seaborn plots with x, y, hue format, args contains the prepared DataFrame
        if isinstance(args, pd.DataFrame):
            df = args.copy()
            
            # Try to get original column names from kwargs
            x = kwargs.get('x')
            y = kwargs.get('y') 
            hue = kwargs.get('hue')
            
            # If we have x, y, hue info and the DataFrame was pivoted, 
            # create more meaningful column names
            if x and y:
                renamed_cols = {}
                for col in df.columns:
                    # Check if this is a pivoted column with format "value-hue"
                    if hue and '-' in str(col):
                        # Extract the hue value from pivoted column name
                        hue_val = str(col).split('-', 1)[-1]
                        renamed_cols[col] = f"{id}_{method}_{y}_{hue}_{hue_val}"
                    else:
                        # For non-pivoted data, use original column names
                        if col == x:
                            renamed_cols[col] = f"{id}_{method}_{x}"
                        elif col == y:
                            renamed_cols[col] = f"{id}_{method}_{y}"
                        elif col == hue:
                            renamed_cols[col] = f"{id}_{method}_{hue}"
                        else:
                            renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            else:
                # Fallback to generic naming
                renamed_cols = {}
                for col in df.columns:
                    renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            
            return df
        
        # If args is not a DataFrame, try to extract from kwargs
        elif kwargs:
            data = kwargs.get('data')
            x = kwargs.get('x')
            y = kwargs.get('y')
            hue = kwargs.get('hue')
            
            if data is not None and x and y:
                # Create a subset DataFrame with the plotted columns
                cols_to_export = [col for col in [x, y, hue] if col is not None]
                if all(col in data.columns for col in cols_to_export):
                    df = data[cols_to_export].copy()
                    
                    # Rename columns with id, method, and original names
                    renamed_cols = {}
                    for col in cols_to_export:
                        renamed_cols[col] = f"{id}_{method}_{col}"
                    df = df.rename(columns=renamed_cols)
                    return df
        
        # If we can't extract the data, return None
        return None

    elif method == "sns_pairplot":
        warnings.warn(
            f"Export for sns_pairplot (id='{id}') is not yet implemented. "
            "Pairplot creates multiple subplots which require special handling."
        )
        return None

    elif method == "sns_scatterplot":
        # Similar handling as sns_lineplot
        if isinstance(args, pd.DataFrame):
            df = args.copy()
            
            # Try to get original column names from kwargs
            x = kwargs.get('x')
            y = kwargs.get('y') 
            hue = kwargs.get('hue')
            
            # Create meaningful column names
            if x and y:
                renamed_cols = {}
                for col in df.columns:
                    if hue and '-' in str(col):
                        hue_val = str(col).split('-', 1)[-1]
                        renamed_cols[col] = f"{id}_{method}_{y}_{hue}_{hue_val}"
                    else:
                        if col == x:
                            renamed_cols[col] = f"{id}_{method}_{x}"
                        elif col == y:
                            renamed_cols[col] = f"{id}_{method}_{y}"
                        elif col == hue:
                            renamed_cols[col] = f"{id}_{method}_{hue}"
                        else:
                            renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            else:
                renamed_cols = {}
                for col in df.columns:
                    renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            
            return df
        return None

    elif method == "sns_violinplot":
        # Similar handling as other seaborn plots
        if isinstance(args, pd.DataFrame):
            df = args.copy()
            
            # Try to get original column names from kwargs
            x = kwargs.get('x')
            y = kwargs.get('y') 
            hue = kwargs.get('hue')
            
            # Create meaningful column names
            if x and y:
                renamed_cols = {}
                for col in df.columns:
                    if hue and '-' in str(col):
                        hue_val = str(col).split('-', 1)[-1]
                        renamed_cols[col] = f"{id}_{method}_{y}_{hue}_{hue_val}"
                    else:
                        if str(col) == x:
                            renamed_cols[col] = f"{id}_{method}_{x}"
                        elif str(col) == y:
                            renamed_cols[col] = f"{id}_{method}_{y}"
                        elif str(col) == hue:
                            renamed_cols[col] = f"{id}_{method}_{hue}"
                        else:
                            renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            else:
                # Fallback to generic naming
                renamed_cols = {}
                for col in df.columns:
                    renamed_cols[col] = f"{id}_{method}_{col}"
                df = df.rename(columns=renamed_cols)
            
            return df
        return None

    elif method == "sns_jointplot":
        warnings.warn(
            f"Export for sns_jointplot (id='{id}') is not yet implemented. "
            "Jointplot creates multiple axes (main plot + marginal plots) which require special handling."
        )
        return None

    elif method in ["fill_between", "fill_betweenx"]:
        # Using module-level _to_numpy function
        if method == "fill_between":
            x, y1 = args[:2]
            y2 = args[2] if len(args) > 2 else np.zeros_like(y1)
            df = pd.DataFrame({
                f"{id}_{method}_x": _to_numpy(x),
                f"{id}_{method}_y1": _to_numpy(y1),
                f"{id}_{method}_y2": _to_numpy(y2)
            })
        else:  # fill_betweenx
            y, x1 = args[:2]
            x2 = args[2] if len(args) > 2 else np.zeros_like(x1)
            df = pd.DataFrame({
                f"{id}_{method}_y": _to_numpy(y),
                f"{id}_{method}_x1": _to_numpy(x1),
                f"{id}_{method}_x2": _to_numpy(x2)
            })
        return df
        
    elif method == "errorbar":
        # Using module-level _to_numpy function
        x, y = args[:2]
        df_data = {f"{id}_{method}_x": _to_numpy(x), f"{id}_{method}_y": _to_numpy(y)}
        
        # Handle error bars
        yerr = kwargs.get('yerr')
        xerr = kwargs.get('xerr')
        if yerr is not None:
            df_data[f"{id}_{method}_yerr"] = _to_numpy(yerr)
        if xerr is not None:
            df_data[f"{id}_{method}_xerr"] = _to_numpy(xerr)
            
        df = pd.DataFrame(df_data)
        return df
        
    elif method in ["step", "stem"]:
        # Using module-level _to_numpy function
        x, y = args[:2]
        df = pd.DataFrame({f"{id}_{method}_x": _to_numpy(x), f"{id}_{method}_y": _to_numpy(y)})
        return df
        
    elif method in ["hist2d", "hexbin"]:
        # Using module-level _to_numpy function
        x, y = args[:2]
        df = pd.DataFrame({f"{id}_{method}_x": _to_numpy(x), f"{id}_{method}_y": _to_numpy(y)})
        return df
        
    elif method == "pie":
        # Using module-level _to_numpy function
        x = args[0]  # pie sizes
        labels = kwargs.get('labels', np.arange(len(x)))
        df = pd.DataFrame({
            f"{id}_{method}_values": _to_numpy(x),
            f"{id}_{method}_labels": labels
        })
        return df
        
    elif method in ["contour", "contourf", "tricontour", "tricontourf"]:
        # Using module-level _to_numpy function
        if len(args) >= 3:
            X, Y, Z = args[:3]
            # Flatten 2D arrays for CSV export
            X_flat = _to_numpy(X).flatten()
            Y_flat = _to_numpy(Y).flatten()
            Z_flat = _to_numpy(Z).flatten()
            
            df = pd.DataFrame({
                f"{id}_{method}_X": X_flat,
                f"{id}_{method}_Y": Y_flat,
                f"{id}_{method}_Z": Z_flat
            })
            return df
        return None
        
    elif method in ["imshow", "matshow", "spy"]:
        # Using module-level _to_numpy function
        X = args[0]  # 2D array
        X_array = _to_numpy(X)
        if X_array.ndim == 2:
            # Convert 2D image to long format for CSV
            rows, cols = X_array.shape
            row_indices = np.repeat(np.arange(rows), cols)
            col_indices = np.tile(np.arange(cols), rows)
            values = X_array.flatten()
            
            df = pd.DataFrame({
                f"{id}_{method}_row": row_indices,
                f"{id}_{method}_col": col_indices,
                f"{id}_{method}_value": values
            })
            return df
        return None
        
    elif method in ["quiver", "streamplot"]:
        # Using module-level _to_numpy function
        if len(args) >= 4:
            X, Y, U, V = args[:4]
            X_flat = _to_numpy(X).flatten()
            Y_flat = _to_numpy(Y).flatten()
            U_flat = _to_numpy(U).flatten()
            V_flat = _to_numpy(V).flatten()
            
            df = pd.DataFrame({
                f"{id}_{method}_X": X_flat,
                f"{id}_{method}_Y": Y_flat,
                f"{id}_{method}_U": U_flat,
                f"{id}_{method}_V": V_flat
            })
            return df
        return None
        
    elif method in ["plot3D", "scatter3D"]:
        # Using module-level _to_numpy function
        if len(args) >= 3:
            x, y, z = args[:3]
            df = pd.DataFrame({
                f"{id}_{method}_x": _to_numpy(x),
                f"{id}_{method}_y": _to_numpy(y),
                f"{id}_{method}_z": _to_numpy(z)
            })
            return df
        return None
        
    elif method == "bar3d":
        # Using module-level _to_numpy function
        if len(args) >= 6:
            xpos, ypos, zpos, dx, dy, dz = args[:6]
            df = pd.DataFrame({
                f"{id}_{method}_xpos": _to_numpy(xpos),
                f"{id}_{method}_ypos": _to_numpy(ypos),
                f"{id}_{method}_zpos": _to_numpy(zpos),
                f"{id}_{method}_dx": _to_numpy(dx),
                f"{id}_{method}_dy": _to_numpy(dy),
                f"{id}_{method}_dz": _to_numpy(dz)
            })
            return df
        return None
        
    elif method in ["plot_surface", "plot_wireframe"]:
        # Using module-level _to_numpy function
        if len(args) >= 3:
            X, Y, Z = args[:3]
            X_flat = _to_numpy(X).flatten()
            Y_flat = _to_numpy(Y).flatten()
            Z_flat = _to_numpy(Z).flatten()
            
            df = pd.DataFrame({
                f"{id}_{method}_X": X_flat,
                f"{id}_{method}_Y": Y_flat,
                f"{id}_{method}_Z": Z_flat
            })
            return df
        return None
        
    elif method in ["annotate", "text"]:
        # For text annotations, we can export the text content and position
        if method == "annotate":
            text = args[0] if args else kwargs.get('s', '')
            xy = args[1] if len(args) > 1 else kwargs.get('xy', (0, 0))
        else:  # text
            x = args[0] if args else 0
            y = args[1] if len(args) > 1 else 0
            text = args[2] if len(args) > 2 else kwargs.get('s', '')
            xy = (x, y)
            
        df = pd.DataFrame({
            f"{id}_{method}_x": [xy[0]],
            f"{id}_{method}_y": [xy[1]],
            f"{id}_{method}_text": [str(text)]
        })
        return df

    elif method == "plot_image":
        # Handle image data export
        # For plot_image, the tracked_dict is passed as args to _track
        if args and isinstance(args, dict) and 'image_df' in args:
            # args is the tracked_dict
            image_df = args['image_df']
            
            # If it's already in XYZ format, use it directly
            if isinstance(image_df, pd.DataFrame) and 'X' in image_df.columns and 'Y' in image_df.columns and 'Z' in image_df.columns:
                df = image_df.copy()
            else:
                # Convert 2D array representation to XYZ format
                df = to_xyz(image_df)
            
            # Add prefix to avoid column name conflicts
            df = df.rename(columns={
                'X': f'{id}_{method}_X',
                'Y': f'{id}_{method}_Y', 
                'Z': f'{id}_{method}_Z'
            })
            
            return df
        else:
            warnings.warn(
                f"Method '{method}' data not found in expected format. "
                f"Record id: {id}, args type: {type(args)}"
            )
            return None
            
    else:
        # Return None for unhandled methods instead of passing silently
        if not method.startswith("set_") and not method.startswith("get_"):
            warnings.warn(
                f"Method '{method}' is not implemented in export_as_csv. "
                f"Record id: {id}, args length: {len(args) if args else 0}"
            )
        return None


def export_as_csv_for_sigmaplot(history_records, include_visual_params=True):
    """Export plotting history records as a pandas DataFrame in SigmaPlot format.

    Converts the plotting history records maintained by MNGS plotting
    functions into a pandas DataFrame formatted specifically for SigmaPlot.
    This format includes visual parameters, graph wizard parameters, and 
    properly padded data columns.

    Parameters
    ----------
    history_records : dict
        Dictionary of plotting records, typically from FigWrapper or
        AxesWrapper history. Each record contains information about
        plotted data.
    include_visual_params : bool, optional
        Whether to include visual parameters (xlabel, ylabel, scales, etc.)
        at the top of the CSV. Default is True.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing the plotted data formatted for SigmaPlot.
        Returns empty DataFrame if no records found or concatenation fails.

    Examples
    --------
    >>> fig, ax = mngs.plt.subplots()
    >>> ax.plot([1, 2, 3], [4, 5, 6], label='data1')
    >>> ax.scatter([1, 2, 3], [7, 8, 9], label='data2')
    >>> df = export_as_csv_for_sigmaplot(ax.history)
    >>> df.to_csv('for_sigmaplot.csv', index=False)

    Notes
    -----
    The SigmaPlot format expects:
    - Visual parameters in the first rows (if included)
    - Graph wizard parameters for each plot
    - Data columns with consistent lengths (padded with NaN)
    - Specific column naming conventions for different plot types
    """
    if len(history_records) <= 0:
        warnings.warn("Plotting records not found. Empty dataframe returned.")
        return pd.DataFrame()
    
    # First, get the regular export data
    data_df = export_as_csv(history_records)
    if data_df.empty:
        return pd.DataFrame()
    
    # Detect plot types from the records
    plot_types = []
    for record_id, record in history_records.items():
        if record and len(record) >= 2:
            _, method_name = record[0], record[1]
            plot_types.append(method_name)
    
    if not include_visual_params:
        return data_df
    
    # Generate visual parameters based on the first plot type
    visual_params_dict = _generate_visual_params(plot_types[0] if plot_types else 'plot')
    
    # Create visual params dataframe
    visual_params_data = {
        "visual parameter label": list(visual_params_dict.keys()),
        "visual parameter value": list(visual_params_dict.values())
    }
    visual_params_df = pd.DataFrame(visual_params_data)
    
    # Add xticks and yticks columns
    xticks = visual_params_dict.get("xticks", ["auto"])
    yticks = visual_params_dict.get("yticks", ["auto"])
    max_ticks = max(len(xticks), len(yticks))
    
    # Pad the visual params dataframe to match tick length
    current_len = len(visual_params_df)
    if current_len < max_ticks:
        padding_rows = max_ticks - current_len
        padding_df = pd.DataFrame({
            "visual parameter label": [np.nan] * padding_rows,
            "visual parameter value": [np.nan] * padding_rows
        })
        visual_params_df = pd.concat([visual_params_df, padding_df], ignore_index=True)
    
    # Add ticks columns
    visual_params_df["xticks"] = xticks + [np.nan] * (len(visual_params_df) - len(xticks))
    visual_params_df["yticks"] = yticks + [np.nan] * (len(visual_params_df) - len(yticks))
    
    # Ensure all columns have the same length by padding with NaN
    max_rows = max(len(visual_params_df), len(data_df))
    
    # Pad visual params if needed
    if len(visual_params_df) < max_rows:
        padding_rows = max_rows - len(visual_params_df)
        padding_dict = {col: [np.nan] * padding_rows for col in visual_params_df.columns}
        padding_df = pd.DataFrame(padding_dict)
        visual_params_df = pd.concat([visual_params_df, padding_df], ignore_index=True)
    
    # Pad data if needed
    if len(data_df) < max_rows:
        padding_rows = max_rows - len(data_df)
        padding_dict = {col: [np.nan] * padding_rows for col in data_df.columns}
        padding_df = pd.DataFrame(padding_dict)
        data_df = pd.concat([data_df, padding_df], ignore_index=True)
    
    # Combine visual params and data
    result_df = pd.concat([visual_params_df, data_df], axis=1)
    
    # Add preserved columns for future expansion (SigmaPlot convention)
    n_cols_preserve = 8 - (len(visual_params_df.columns) + len(data_df.columns))
    if n_cols_preserve > 0:
        for i in range(n_cols_preserve):
            result_df[f"preserved {i}"] = "NONE_STR"
    
    return result_df


def _generate_visual_params(plot_type):
    """Generate visual parameters for SigmaPlot based on plot type."""
    # Default parameters
    default_params = {
        "xlabel": "X-Axis Label",
        "xrot": 0,
        "xmm": 40,
        "xscale": "linear",
        "xmin": "auto",
        "xmax": "auto",
        "xticks": ["auto"],
        "ylabel": "Y-Axis Label", 
        "yrot": 0,
        "ymm": 28,  # 40 * 0.7
        "yscale": "linear",
        "ymin": "auto",
        "ymax": "auto",
        "yticks": ["auto"],
    }
    
    # Plot-specific overrides
    plot_specific = {
        "bar": {
            "xrot": 45,
            "xscale": "category",
            "ymin": 0,
        },
        "barh": {
            "xmin": 0,
            "yscale": "category",
        },
        "hist": {
            "xrot": 45,
            "ymin": 0,
        },
        "boxplot": {
            "xrot": 90,
            "xscale": "category",
            "ymin": 0,
        },
        "violinplot": {
            "xscale": "category",
        },
    }
    
    # Apply plot-specific overrides
    params = default_params.copy()
    if plot_type in plot_specific:
        params.update(plot_specific[plot_type])
    
    return params


def main():
    # Line
    fig, ax = mngs.plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
    mngs.io.save(fig, "./plots.png")
    mngs.io.save(ax.export_as_csv(), "./plots.csv")

    # No tracking
    fig, ax = mngs.plt.subplots(track=False)
    ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
    ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
    mngs.io.save(fig, "./plots_wo_tracking.png")
    mngs.io.save(ax.export_as_csv(), "./plots_wo_tracking.csv")

    # Scatter
    fig, ax = mngs.plt.subplots()
    ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
    ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
    mngs.io.save(fig, "./scatters.png")
    mngs.io.save(ax.export_as_csv(), "./scatters.csv")

    # Box
    fig, ax = mngs.plt.subplots()
    ax.boxplot([1, 2, 3], id="boxplot1")
    mngs.io.save(fig, "./boxplot1.png")
    mngs.io.save(ax.export_as_csv(), "./boxplot1.csv")

    # Bar
    fig, ax = mngs.plt.subplots()
    ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
    mngs.io.save(fig, "./bar1.png")
    mngs.io.save(ax.export_as_csv(), "./bar1.csv")

    # # Bar
    # # fig, ax = mngs.plt.subplots()
    # fig, ax = plt.subplots()
    # ax.bar(["A", "B", "C"], [4, 5, 6], id="bar2")
    # mngs.io.save(fig, "./bar2.png")
    # mngs.io.save(ax.export_as_csv(), "./bar2.csv")

    # print(ax.export_as_csv())
    # #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
    # # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
    # # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
    # # 2           3.0           6.0           6.0  ...                 3.0           C         6.0

    # print(ax.export_as_csv().keys())  # plot3 and plot 4 are not tracked
    # # [3 rows x 11 columns]
    # # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
    # #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
    # #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
    # #       dtype='object')

    # # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
    # ax.export_as_csv("../for_sigmaplot.csv")
    # # Saved to: ../for_sigmaplot.csv
    
    # Test SigmaPlot export
    print("\n--- Testing SigmaPlot Export ---")
    fig, ax = mngs.plt.subplots()
    ax.plot([1, 2, 3], [4, 5, 6], id="line1")
    ax.scatter([1, 2, 3], [7, 8, 9], id="scatter1")
    df_sigmaplot = export_as_csv_for_sigmaplot(ax.history)
    print("SigmaPlot format DataFrame:")
    print(df_sigmaplot)
    mngs.io.save(df_sigmaplot, "./for_sigmaplot.csv")


if __name__ == "__main__":
    # # Argument Parser
    # import argparse
    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--var', '-v', type=int, default=1, help='')
    # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
    # args = parser.parse_args()

    # Main
    CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
        sys, plt, verbose=False, agg=True
    )
    main()
    mngs.gen.close(CONFIG, verbose=False, notify=False)

# EOF
