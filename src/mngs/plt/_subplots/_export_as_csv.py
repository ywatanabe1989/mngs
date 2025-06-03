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
        def to_numpy(data):
            if hasattr(data, 'numpy'):  # torch tensor
                return data.detach().numpy() if hasattr(data, 'detach') else data.numpy()
            elif hasattr(data, 'values'):  # pandas series/dataframe
                return data.values
            else:
                return np.asarray(data)
        
        # Extract x, y data from matplotlib plot arguments
        # Matplotlib plot accepts: plot(y), plot(x,y), plot(x,y,fmt), plot(x,y,fmt,x2,y2,fmt2,...)
        plot_data = []
        i = 0
        while i < len(args):
            if i == 0 and len(args) == 1:
                # Single argument: y values with implicit x
                y = to_numpy(args[0])
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
                    x = to_numpy(potential_x)
                    y = to_numpy(potential_y)
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
        x, y = args
        df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
        return df

    elif method == "bar":
        x, y = args
        yerr = kwargs.get("yerr")

        if isinstance(x, (int, float)):
            x = pd.Series(x, name="x")
        if isinstance(y, (int, float)):
            y = pd.Series(y, name="y")

        df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})

        if yerr is not None:
            if isinstance(yerr, (int, float)):
                yerr = pd.Series(yerr, name="yerr")
            df[f"{id}_{method}_yerr"] = yerr
        return df

    elif method == "hist":
        x = args
        df = pd.DataFrame({f"{id}_{method}_x": x})
        return df

    elif method == "boxplot":
        x = args[0]

        # One box plot
        from mngs.types import is_listed_X as mngs_types_is_listed_X

        if isinstance(x, np.ndarray) or mngs_types_is_listed_X(x, [float, int]):
            df = pd.DataFrame(x)

        else:
            # Multiple boxes
            import mngs.pd.force_df as mngs_pd_force_df

            df = mngs.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})
        df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
        df = df.apply(lambda col: col.dropna().reset_index(drop=True))
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
        df = args

        # When xyhue, without errorbar
        df = pd.DataFrame(pd.Series(np.array(df).diagonal(), index=df.columns)).T
        return df

    elif method == "sns_boxplot":
        df = args
        if id is not None:
            df.columns = [f"{id}_{method}_{col}" for col in df.columns]
        return df

    elif method == "sns_heatmap":
        df = args
        return df

    elif method == "sns_histplot":
        df = args
        return df

    elif method == "sns_kdeplot":
        pass
        # df = args
        # __import__("ipdb").set_trace()
        # return df

    elif method == "sns_lineplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_pairplot":
        __import__("ipdb").set_trace()
        return df

    elif method == "sns_scatterplot":
        return df

    elif method == "sns_violinplot":
        df = args
        return df

    elif method == "sns_jointplot":
        __import__("ipdb").set_trace()
        return df

    else:
        # Return None for unhandled methods instead of passing silently
        if not method.startswith("set_") and not method.startswith("get_"):
            warnings.warn(
                f"Method '{method}' is not implemented in export_as_csv. "
                f"Record id: {id}, args length: {len(args) if args else 0}"
            )
        return None


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
