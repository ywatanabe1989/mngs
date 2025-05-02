#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-01 16:57:14 (ywatanabe)"
# File: /home/ywatanabe/proj/_mngs_repo/tests/mngs/plt/_subplots/test__export_as_csv.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/test__export_as_csv.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import numpy as np
import pandas as pd
from mngs.plt._subplots._export_as_csv import export_as_csv, format_record


class TestExportAsCSV:
    def test_empty_history(self):
        history = {}
        result = export_as_csv(history)
        assert result is None or isinstance(result, pd.DataFrame)

    def test_simple_plot(self):
        history = {"plot1": ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {})}
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert "plot1_plot_y" in result.columns
        assert result["plot1_plot_x"].tolist() == [1, 2, 3]
        assert result["plot1_plot_y"].tolist() == [4, 5, 6]

    def test_multiple_plots(self):
        history = {
            "plot1": ("plot1", "plot", ([1, 2, 3], [4, 5, 6]), {}),
            "plot2": ("plot2", "plot", ([4, 5, 6], [1, 2, 3]), {}),
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == [
            "plot1_plot_x",
            "plot1_plot_y",
            "plot2_plot_x",
            "plot2_plot_y",
        ]

    def test_scatter_plot(self):
        history = {
            "scatter1": ("scatter1", "scatter", ([1, 2, 3], [4, 5, 6]), {})
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "scatter1_scatter_x" in result.columns
        assert "scatter1_scatter_y" in result.columns

    def test_bar_plot(self):
        history = {"bar1": ("bar1", "bar", (["A", "B", "C"], [4, 5, 6]), {})}
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "bar1_bar_x" in result.columns
        assert "bar1_bar_y" in result.columns

    def test_bar_plot_with_yerr(self):
        history = {
            "bar1": (
                "bar1",
                "bar",
                (["A", "B", "C"], [4, 5, 6]),
                {"yerr": [0.1, 0.2, 0.3]},
            )
        }
        result = export_as_csv(history)
        assert isinstance(result, pd.DataFrame)
        assert "bar1_bar_yerr" in result.columns


class TestFormatPlottingArgs:
    def test_plot_with_single_array(self):
        record = ("plot1", "plot", [np.array([[1, 4], [2, 5], [3, 6]])], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns

    def test_plot_with_separate_arrays(self):
        record = (
            "plot1",
            "plot",
            [np.array([1, 2, 3]), np.array([4, 5, 6])],
            {},
        )
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert "plot1_plot_y" in result.columns

    def test_plot_with_2d_y_array(self):
        record = (
            "plot1",
            "plot",
            [np.array([1, 2, 3]), np.array([[4, 7], [5, 8], [6, 9]])],
            {},
        )
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x00" in result.columns
        assert "plot1_plot_y00" in result.columns
        assert "plot1_plot_y01" in result.columns

    def test_plot_with_dataframe_y(self):
        y_df = pd.DataFrame({"col1": [4, 5, 6], "col2": [7, 8, 9]})
        record = ("plot1", "plot", [np.array([1, 2, 3]), y_df], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "plot1_plot_x" in result.columns
        assert "plot1_plot_y00" in result.columns
        assert "plot1_plot_y01" in result.columns

    def test_boxplot(self):
        record = ("box1", "boxplot", [[1, 2, 3]], {})
        result = format_record(record)
        assert isinstance(result, pd.DataFrame)
        assert "box1_boxplot_0_x" in result.columns

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_export_as_csv.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-01 17:01:32 (ywatanabe)"
# # File: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_export_as_csv.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_export_as_csv.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
# 
# import sys
# import warnings
# 
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# import pandas as pd
# import xarray as xr
# 
# 
# def export_as_csv(history_records):
#     if len(history_records) <= 0:
#         warnings.warn("Plotting records not found. Empty dataframe returned.")
#         return pd.DataFrame()
#     else:
#         dfs = [
#             format_record(record) for record in list(history_records.values())
#         ]
#         try:
#             df = pd.concat(dfs, axis=1)
#             return df
#         except Exception as e:
#             warnings.warn(
#                 f"Plotting records not combined. Empty dataframe returned {e}"
#             )
#             return pd.DataFrame()
# 
# 
# def _format_imshow2d(record):
#     id, method, args, kwargs = record
#     df = args
#     # df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#     # df.index = [f"{id}_{method}_{idx}" for idx in df.index]
#     return df
# 
# 
# def format_record(record):
#     id, method, args, kwargs = record
# 
#     if method == "imshow2d":
#         return _format_imshow2d(record)
# 
#     elif method in ["plot"]:
#         if len(args) == 1:
#             args = args[0]
#             if args.ndim == 2:
#                 x, y = args[:, 0], args[:, 1]
#                 df = pd.DataFrame({f"{id}_{method}_x": x})
#                 return df
# 
#         elif len(args) == 2:
#             x, y = args
# 
#             if isinstance(y, (np.ndarray, xr.DataArray)):
#                 if y.ndim == 2:
#                     from collections import OrderedDict
# 
#                     out = OrderedDict()
# 
#                     for ii in range(y.shape[1]):
#                         out[f"{id}_{method}_x{ii:02d}"] = x
#                         out[f"{id}_{method}_y{ii:02d}"] = y[:, ii]
#                     df = pd.DataFrame(out)
# 
#                     return df
# 
#             if isinstance(y, pd.DataFrame):
#                 df = pd.DataFrame(
#                     {
#                         f"{id}_{method}_x": x,
#                         **{
#                             f"{id}_{method}_y{ii:02d}": np.array(y[col])
#                             for ii, col in enumerate(y.columns)
#                         },
#                     }
#                 )
#                 return df
# 
#             else:
#                 if isinstance(y, (np.ndarray, xr.DataArray, list)):
#                     df = pd.DataFrame(
#                         {f"{id}_{method}_x": x, f"{id}_{method}_y": y}
#                     )
#                     return df
# 
#     elif method == "scatter":
#         x, y = args
#         df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
#         return df
# 
#     elif method == "bar":
#         x, y = args
#         yerr = kwargs.get("yerr")
# 
#         if isinstance(x, (int, float)):
#             x = pd.Series(x, name="x")
#         if isinstance(y, (int, float)):
#             y = pd.Series(y, name="y")
# 
#         df = pd.DataFrame({f"{id}_{method}_x": x, f"{id}_{method}_y": y})
# 
#         if yerr is not None:
#             if isinstance(yerr, (int, float)):
#                 yerr = pd.Series(yerr, name="yerr")
#             df[f"{id}_{method}_yerr"] = yerr
#         return df
# 
#     elif method == "hist":
#         x = args
#         df = pd.DataFrame({f"{id}_{method}_x": x})
#         return df
# 
#     elif method == "boxplot":
#         x = args[0]
# 
#         # One box plot
#         from mngs.types import is_listed_X as mngs_types_is_listed_X
# 
#         if isinstance(x, np.ndarray) or mngs_types_is_listed_X(
#             x, [float, int]
#         ):
#             df = pd.DataFrame(x)
# 
#         else:
#             # Multiple boxes
#             import mngs.pd.force_df as mngs_pd_force_df
# 
#             df = mngs.pd.force_df({i_x: _x for i_x, _x in enumerate(x)})
#         df.columns = [f"{id}_{method}_{col}_x" for col in df.columns]
#         df = df.apply(lambda col: col.dropna().reset_index(drop=True))
#         return df
# 
#     # elif method == "boxplot_":
#     #     __import__("ipdb").set_trace()
#     #     x = args[0]
#     #     df =
#     #     df.columns = [f"{id}_{method}_{col}" for col in df.columns]
# 
#     #     return df
# 
#     # elif method == "plot_":
#     #     df = args
#     #     df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#     #     return df
# 
#     elif method == "plot_fillv":
#         starts, ends = args
#         df = pd.DataFrame(
#             {
#                 f"{id}_{method}_starts": starts,
#                 f"{id}_{method}_ends": ends,
#             }
#         )
#         return df
# 
#     elif method == "plot_raster":
#         df = args
#         return df
# 
#     elif method == "plot_ecdf":
#         df = args
#         return df
# 
#     elif method == "plot_kde":
#         df = args
#         if id is not None:
#             df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#         return df
# 
#     elif method == "sns_barplot":
#         df = args
# 
#         # When xyhue, without errorbar
#         df = pd.DataFrame(
#             pd.Series(np.array(df).diagonal(), index=df.columns)
#         ).T
#         return df
# 
#     elif method == "sns_boxplot":
#         df = args
#         if id is not None:
#             df.columns = [f"{id}_{method}_{col}" for col in df.columns]
#         return df
# 
#     elif method == "sns_heatmap":
#         df = args
#         return df
# 
#     elif method == "sns_histplot":
#         df = args
#         return df
# 
#     elif method == "sns_kdeplot":
#         pass
#         # df = args
#         # __import__("ipdb").set_trace()
#         # return df
# 
#     elif method == "sns_lineplot":
#         __import__("ipdb").set_trace()
#         return df
# 
#     elif method == "sns_pairplot":
#         __import__("ipdb").set_trace()
#         return df
# 
#     elif method == "sns_scatterplot":
#         return df
# 
#     elif method == "sns_violinplot":
#         df = args
#         return df
# 
#     elif method == "sns_jointplot":
#         __import__("ipdb").set_trace()
#         return df
# 
#     else:
#         pass
#         # if not method.startswith("set_"):
#         #     logging.warn(
#         #         f"{method} is not implemented in _export_as_csv method of the mngs.plt module."
#         #     )
# 
# 
# def main():
#     # Line
#     fig, ax = mngs.plt.subplots()
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot1")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot2")
#     mngs.io.save(fig, "./plots.png")
#     mngs.io.save(ax.export_as_csv(), "./plots.csv")
# 
#     # No tracking
#     fig, ax = mngs.plt.subplots(track=False)
#     ax.plot([1, 2, 3], [4, 5, 6], id="plot3")
#     ax.plot([4, 5, 6], [1, 2, 3], id="plot4")
#     mngs.io.save(fig, "./plots_wo_tracking.png")
#     mngs.io.save(ax.export_as_csv(), "./plots_wo_tracking.csv")
# 
#     # Scatter
#     fig, ax = mngs.plt.subplots()
#     ax.scatter([1, 2, 3], [4, 5, 6], id="scatter1")
#     ax.scatter([4, 5, 6], [1, 2, 3], id="scatter2")
#     mngs.io.save(fig, "./scatters.png")
#     mngs.io.save(ax.export_as_csv(), "./scatters.csv")
# 
#     # Box
#     fig, ax = mngs.plt.subplots()
#     ax.boxplot([1, 2, 3], id="boxplot1")
#     mngs.io.save(fig, "./boxplot1.png")
#     mngs.io.save(ax.export_as_csv(), "./boxplot1.csv")
# 
#     # Bar
#     fig, ax = mngs.plt.subplots()
#     ax.bar(["A", "B", "C"], [4, 5, 6], id="bar1")
#     mngs.io.save(fig, "./bar1.png")
#     mngs.io.save(ax.export_as_csv(), "./bar1.csv")
# 
#     # # Bar
#     # # fig, ax = mngs.plt.subplots()
#     # fig, ax = plt.subplots()
#     # ax.bar(["A", "B", "C"], [4, 5, 6], id="bar2")
#     # mngs.io.save(fig, "./bar2.png")
#     # mngs.io.save(ax.export_as_csv(), "./bar2.csv")
# 
#     # print(ax.export_as_csv())
#     # #    plot1_plot_x  plot1_plot_y  plot2_plot_x  ...  boxplot1_boxplot_x  bar1_bar_x  bar1_bar_y
#     # # 0           1.0           4.0           4.0  ...                 1.0           A         4.0
#     # # 1           2.0           5.0           5.0  ...                 2.0           B         5.0
#     # # 2           3.0           6.0           6.0  ...                 3.0           C         6.0
# 
#     # print(ax.export_as_csv().keys())  # plot3 and plot 4 are not tracked
#     # # [3 rows x 11 columns]
#     # # Index(['plot1_plot_x', 'plot1_plot_y', 'plot2_plot_x', 'plot2_plot_y',
#     # #        'scatter1_scatter_x', 'scatter1_scatter_y', 'scatter2_scatter_x',
#     # #        'scatter2_scatter_y', 'boxplot1_boxplot_x', 'bar1_bar_x', 'bar1_bar_y'],
#     # #       dtype='object')
# 
#     # # If a path is passed, the sigmaplot-friendly dataframe is saved as a csv file.
#     # ax.export_as_csv("../for_sigmaplot.csv")
#     # # Saved to: ../for_sigmaplot.csv
# 
# 
# if __name__ == "__main__":
#     # # Argument Parser
#     # import argparse
#     # parser = argparse.ArgumentParser(description='')
#     # parser.add_argument('--var', '-v', type=int, default=1, help='')
#     # parser.add_argument('--flag', '-f', action='store_true', default=False, help='')
#     # args = parser.parse_args()
# 
#     # Main
#     CONFIG, sys.stdout, sys.stderr, plt, CC = mngs.gen.start(
#         sys, plt, verbose=False, agg=True
#     )
#     main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_export_as_csv.py
# --------------------------------------------------------------------------------
