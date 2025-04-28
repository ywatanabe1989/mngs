# --------------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import pytest
    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot_.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-11-17 11:59:15 (ywatanabe)"
# # File: ./mngs_repo/src/mngs/plt/ax/_plot_.py
# 
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot_.py"
# 
# from typing import Optional, Tuple, Union
# 
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from matplotlib.axes import Axes
# from numpy.typing import ArrayLike
# 
# from ...decorators import numpy_fn
# 
# 
# class StatisticalPlotter:
#     """Class for creating statistical plots with confidence intervals.
# 
#     Examples
#     --------
#     Simple line plot:
#     >>> data = np.random.rand(100)
#     >>> fig, ax = plt.subplots()
#     >>> plotter = StatisticalPlotter(ax)
#     >>> plotter.plot(data)
# 
#     Plot with mean and standard deviation:
#     >>> data = np.random.rand(10, 100)  # 10 samples, 100 timepoints
#     >>> plotter.plot(data, line='mean', fill='std', n=2)
# 
#     Plot with median and IQR:
#     >>> plotter.plot(data, line='median', fill='iqr')
# 
#     Multiple lines with confidence intervals:
#     >>> data = np.random.rand(10, 100, 3)  # 10 samples, 100 timepoints, 3 variables
#     >>> plotter.plot(data, line='mean', fill='ci', n=0.95)
#     """
# 
#     def __init__(self, axis: Axes):
#         """Initialize the plotter with a matplotlib axis.
# 
#         Parameters
#         ----------
#         axis : Axes
#             Matplotlib axes to plot on
#         """
#         self.axis = axis
#         self.stats_df = pd.DataFrame()
# 
#     def _validate_arguments(self, data, yy, line, fill, n, alpha):
#         """Validate input arguments for plotting.
# 
#         Parameters
#         ----------
#         data : ArrayLike
#             Input data array
#         yy : Optional[ArrayLike]
#             Y-axis data for simple plotting
#         line : Optional[str]
#             Type of central tendency ('mean' or 'median')
#         fill : Optional[str]
#             Type of interval ('std', 'ci', or 'iqr')
#         n : Optional[Union[int, float]]
#             Number of standard deviations or confidence level
#         alpha : float
#             Transparency of fill area
# 
#         Raises
#         ------
#         TypeError
#             If data types are incorrect
#         ValueError
#             If values are invalid
#         """
#         # Validate data
#         if not isinstance(data, (np.ndarray, list)):
#             raise TypeError("data must be numpy array or list")
# 
#         # Validate yy
#         if yy is not None and not isinstance(yy, (np.ndarray, list)):
#             raise TypeError("yy must be numpy array or list")
# 
#         # Validate line
#         if line is not None and line not in ["mean", "median"]:
#             raise ValueError("line must be 'mean' or 'median'")
# 
#         # Validate fill
#         if fill is not None and fill not in ["std", "ci", "iqr"]:
#             raise ValueError("fill must be 'std', 'ci', or 'iqr'")
# 
#         # Validate n
#         if n is not None:
#             if fill == "std" and not isinstance(n, (int, float)):
#                 raise TypeError("n must be int or float for std")
#             elif fill == "ci":
#                 if not isinstance(n, float) or n <= 0 or n >= 1:
#                     raise ValueError("n must be float between 0 and 1 for ci")
# 
#         # Validate alpha
#         if not isinstance(alpha, float) or alpha < 0 or alpha > 1:
#             raise ValueError("alpha must be float between 0 and 1")
# 
#     def _validate_data_size(self, data: ArrayLike) -> None:
#         """Validate input data dimensions and size.
# 
#         Parameters
#         ----------
#         data : ArrayLike
#             Input data array to validate
# 
#         Raises
#         ------
#         ValueError
#             If data dimensions or size are invalid
#         """
#         if data.ndim not in [1, 2, 3]:
#             raise ValueError(f"Data must be 1D, 2D, or 3D, got {data.ndim}D")
# 
#         if data.ndim == 2 and data.shape[0] < 2:
#             raise ValueError(
#                 f"2D data requires at least 2 samples, got {data.shape[0]}"
#             )
# 
#         if data.ndim == 3 and data.shape[0] < 2:
#             raise ValueError(
#                 f"3D data requires at least 2 samples, got {data.shape[0]}"
#             )
# 
#     def _simple_plot(
#         self, data: ArrayLike, yy: Optional[ArrayLike], kwargs
#     ) -> None:
#         """Create a simple line plot.
# 
#         Parameters
#         ----------
#         data : ArrayLike
#             X-axis data for simple plotting
#         yy : Optional[ArrayLike]
#             Y-axis data for simple plotting
#         kwargs : dict
#             Additional plotting parameters
#         """
#         if yy is None:
#             assert data.ndim == 1
#             self.axis.plot(data, **kwargs)
#         else:
#             assert data.ndim == 1
#             assert len(data) == len(yy)
#             self.axis.plot(data, yy, **kwargs)
# 
#     def _prepare_data(self, data: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
#         """Prepare data for statistical calculations.
# 
#         Parameters
#         ----------
#         data : ArrayLike
#             Input data array (2D or 3D)
# 
#         Returns
#         -------
#         Tuple[np.ndarray, np.ndarray]
#             X-axis values and reshaped 2D data
#         """
#         if data.ndim == 2:
#             xx = np.arange(data.shape[1])
#             data_2d = data
#         elif data.ndim == 3:
#             xx = np.arange(data.shape[1])
#             data_2d = data.reshape(data.shape[0], -1)
#         else:
#             raise ValueError("Data must be 1D, 2D, or 3D")
#         return xx, data_2d
# 
#     def _calculate_central(self, data_2d: np.ndarray, line: str) -> np.ndarray:
#         """Calculate central tendency (mean or median).
# 
#         Parameters
#         ----------
#         data_2d : np.ndarray
#             2D input data
#         line : str
#             Type of central tendency ('mean' or 'median')
# 
#         Returns
#         -------
#         np.ndarray
#             Central tendency values
#         """
#         sample_sizes = np.sum(~np.isnan(data_2d), axis=0)
#         self.stats_df["n"] = sample_sizes
# 
#         if line == "mean":
#             central = np.nanmean(data_2d, axis=0)
#             self.stats_df["mean"] = central
#         else:
#             central = np.nanmedian(data_2d, axis=0)
#             self.stats_df["median"] = central
#         return central
# 
#     def _calculate_intervals(
#         self,
#         data_2d: np.ndarray,
#         central: np.ndarray,
#         fill: str,
#         n: Optional[Union[int, float]],
#     ) -> Tuple[np.ndarray, np.ndarray]:
#         """Calculate interval bounds (std, ci, or iqr).
# 
#         Parameters
#         ----------
#         data_2d : np.ndarray
#             2D input data
#         central : np.ndarray
#             Central tendency values
#         fill : str
#             Type of interval ('std', 'ci', or 'iqr')
#         n : Optional[Union[int, float]]
#             Number of standard deviations or confidence level
# 
#         Returns
#         -------
#         Tuple[np.ndarray, np.ndarray]
#             Lower and upper bounds
#         """
#         if "n" not in self.stats_df:
#             self.stats_df["n"] = np.sum(~np.isnan(data_2d), axis=0)
# 
#         original_shape = central.shape
#         flat_central = central.ravel() if central.ndim > 1 else central
# 
#         if fill == "std":
#             std = np.where(
#                 self.stats_df["n"] >= 2,
#                 np.nanstd(data_2d, axis=0, ddof=1),
#                 np.nan,
#             )
#             lower = flat_central - (n if n else 1) * std
#             upper = flat_central + (n if n else 1) * std
#             self.stats_df["std"] = std
# 
#         elif fill == "ci":
#             confidence = n if n else 0.95
#             valid_mask = self.stats_df["n"] >= 3
#             lower = np.where(
#                 valid_mask,
#                 np.nanpercentile(data_2d, (1 - confidence) * 50, axis=0),
#                 np.nan,
#             )
#             upper = np.where(
#                 valid_mask,
#                 np.nanpercentile(data_2d, 100 - (1 - confidence) * 50, axis=0),
#                 np.nan,
#             )
#             self.stats_df["ci_lower"] = lower
#             self.stats_df["ci_upper"] = upper
# 
#         else:
#             valid_mask = self.stats_df["n"] >= 3
#             lower = np.where(
#                 valid_mask, np.nanpercentile(data_2d, 25, axis=0), np.nan
#             )
#             upper = np.where(
#                 valid_mask, np.nanpercentile(data_2d, 75, axis=0), np.nan
#             )
#             self.stats_df["q1"] = lower
#             self.stats_df["q3"] = upper
# 
#         # Reshape back to original dimensions if needed
#         if central.ndim > 1:
#             lower = lower.reshape(original_shape)
#             upper = upper.reshape(original_shape)
# 
#         return lower, upper
# 
#     def _plot_results(
#         self,
#         xx: np.ndarray,
#         central: np.ndarray,
#         lower: Optional[np.ndarray],
#         upper: Optional[np.ndarray],
#         alpha: float,
#         kwargs,
#     ) -> None:
#         color = kwargs.pop("color", None)
#         colors = kwargs.pop("colors", None)
#         fill_alpha = kwargs.pop("fill_alpha", alpha)
#         label = kwargs.pop("label", "")
# 
#         # Generate statistical label
#         if not label:
#             ns = self.stats_df["n"]
#             n_str = (
#                 f"ns={int(min(ns))}"
#                 if min(ns) == max(ns)
#                 else f"ns={int(min(ns))}-{int(max(ns))}"
#             )
# 
#             if "mean" in self.stats_df:
#                 base_label = "mean"
#                 if "std" in self.stats_df:
#                     label = f"{base_label} ± std ({n_str})"
#                 elif "ci_lower" in self.stats_df:
#                     label = f"{base_label} with 95% CI ({n_str})"
#             elif "median" in self.stats_df:
#                 label = f"median ± IQR ({n_str})"
# 
#         if central.ndim == 1:
#             self.axis.plot(xx, central, color=color, label=label, **kwargs)
#             if lower is not None and upper is not None:
#                 self.axis.fill_between(
#                     xx,
#                     lower,
#                     upper,
#                     alpha=fill_alpha,
#                     color=color,
#                     label=None,
#                     **{
#                         k: v
#                         for k, v in kwargs.items()
#                         if k not in ["marker", "linestyle"]
#                     },
#                 )
#         else:
#             if colors is None and color is None:
#                 colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#             elif color is not None:
#                 colors = [color] * central.shape[1]
# 
#             for yy_idx in range(central.shape[1]):
#                 current_color = colors[yy_idx % len(colors)]
#                 current_label = f"{label} ({yy_idx+1})" if label else ""
# 
#                 self.axis.plot(
#                     xx,
#                     central[:, yy_idx],
#                     color=current_color,
#                     label=current_label,
#                     **kwargs,
#                 )
# 
#                 if lower is not None and upper is not None:
#                     self.axis.fill_between(
#                         xx,
#                         lower[:, yy_idx],
#                         upper[:, yy_idx],
#                         alpha=fill_alpha,
#                         color=current_color,
#                         label=None,
#                         **{
#                             k: v
#                             for k, v in kwargs.items()
#                             if k not in ["marker", "linestyle"]
#                         },
#                     )
# 
#     def plot(
#         self,
#         data: ArrayLike,
#         xx: Optional[ArrayLike] = None,
#         yy: Optional[ArrayLike] = None,
#         line: Optional[str] = None,
#         fill: Optional[str] = None,
#         n: Optional[Union[int, float, ArrayLike]] = None,
#         alpha: float = 0.3,
#         **kwargs,
#     ) -> Tuple[Axes, pd.DataFrame]:
#         """Create statistical plot with optional intervals.
# 
#         Parameters
#         ----------
#         data : ArrayLike
#             Input data array (1D, 2D, or 3D)
#         xx : Optional[ArrayLike]
#             X-axis values
#         yy : Optional[ArrayLike]
#             Y-axis data for simple plotting
#         line : Optional[str]
#             Type of central tendency ('mean' or 'median')
#         fill : Optional[str]
#             Type of interval ('std', 'ci', or 'iqr')
#         n : Optional[Union[int, float, ArrayLike]]
#             Number of standard deviations or confidence level
#         alpha : float
#             Transparency of fill area
#         **kwargs : dict
#             Additional plotting parameters
# 
#         Returns
#         -------
#         Tuple[Axes, pd.DataFrame]
#             Matplotlib axes and statistics DataFrame
#         """
#         self._validate_arguments(data, yy, line, fill, n, alpha)
#         self._validate_data_size(data)
# 
#         if (yy is None and line is None) or yy is not None:
#             self._simple_plot(data, yy, kwargs)
#             return self.axis, self.stats_df
# 
#         if line is not None:
#             x_values, data_2d = self._prepare_data(data)
#             x_values = xx if xx is not None else x_values
#             central = self._calculate_central(data_2d, line)
# 
#             if data.ndim == 3:
#                 central = central.reshape(data.shape[1:])
# 
#             if fill is not None:
#                 lower, upper = self._calculate_intervals(data_2d, central, fill, n)
# 
#                 if data.ndim == 3:
#                     lower = lower.reshape(data.shape[1:])
#                     upper = upper.reshape(data.shape[1:])
#                     self.stats_df["lower"] = lower.reshape(-1)
#                     self.stats_df["upper"] = upper.reshape(-1)
#                 else:
#                     self.stats_df["lower"] = lower
#                     self.stats_df["upper"] = upper
# 
#             self._plot_results(x_values, central, lower, upper, alpha, kwargs)
# 
#         return self.axis, self.stats_df
# 
# 
# @numpy_fn
# def plot_(
#     axis: Axes,
#     data: ArrayLike,
#     xx: Optional[ArrayLike] = None,
#     yy: Optional[ArrayLike] = None,
#     line: Optional[str] = None,
#     fill: Optional[str] = None,
#     n: Optional[Union[int, float, ArrayLike]] = None,
#     alpha: float = 0.3,
#     **kwargs,
# ) -> Tuple[Axes, pd.DataFrame]:
#     """Create statistical plot using StatisticalPlotter class.
# 
#     Parameters
#     ----------
#     axis : Axes
#         Matplotlib axes to plot on
#     data : ArrayLike
#         Input data array (1D, 2D, or 3D)
#     xx : Optional[ArrayLike]
#         X-axis values
#     yy : Optional[ArrayLike]
#         Y-axis data for simple plotting
#     line : Optional[str]
#         Type of central tendency ('mean' or 'median')
#     fill : Optional[str]
#         Type of interval ('std', 'ci', or 'iqr')
#     n : Optional[Union[int, float, ArrayLike]]
#         Number of standard deviations or confidence level
#     alpha : float
#         Transparency of fill area
#     **kwargs : dict
#         Additional plotting parameters
# 
#     Returns
#     -------
#     Tuple[Axes, pd.DataFrame]
#         Matplotlib axes and statistics DataFrame
# 
# 
#     Examples
#     --------
#     Simple line plot:
#     >>> fig, ax = plt.subplots()
#     >>> data = np.random.rand(100)
#     >>> plot_(ax, data)
# 
#     X-Y plot:
#     >>> x = np.linspace(0, 10, 100)
#     >>> y = np.sin(x)
#     >>> plot_(ax, x, y)
# 
#     Mean with standard deviation:
#     >>> data = np.random.rand(10, 100)
#     >>> ax, stats = plot_(ax, data, line='mean', fill='std', n=2)
#     >>> print(stats['mean'], stats['std'])
# 
#     Median with 95% confidence interval:
#     >>> plot_(ax, data, line='median', fill='ci', n=0.95)
# 
#     Multiple variables with IQR:
#     >>> data = np.random.rand(10, 100, 3)
#     >>> plot_(ax, data, line='mean', fill='iqr')
#     """
# 
#     plotter = StatisticalPlotter(axis)
#     return plotter.plot(data, xx, yy, line, fill, n, alpha, **kwargs)
# 
# 
# # @deprecated("Use plot_() instead.")
# # def fill_between(
# #     axis: Axes,
# #     xx: Optional[ArrayLike] = None,
# #     mean: Optional[ArrayLike] = None,
# #     median: Optional[ArrayLike] = None,
# #     std: Optional[ArrayLike] = None,
# #     ci: Optional[ArrayLike] = None,
# #     iqr: Optional[ArrayLike] = None,
# #     n: Optional[Union[int, float, ArrayLike]] = None,
# #     alpha: float = 0.3,
# #     **kwargs,
# # ) -> Tuple[Axes, pd.DataFrame]:
# 
# #     return plot_with_ci(
# #         axis,
# #         xx=xx,
# #         mean=mean,
# #         median=median,
# #         std=std,
# #         ci=ci,
# #         iqr=iqr,
# #         n=n,
# #         alpha=alpha,
# #         **kwargs,
# #     )
# 
# # def plot_(
# #     axis: Axes,
# #     xx: Optional[ArrayLike] = None,
# #     yy: Optional[ArrayLike] = None,
# #     mean: Optional[ArrayLike] = None,
# #     median: Optional[ArrayLike] = None,
# #     std: Optional[ArrayLike] = None,
# #     ci: Optional[ArrayLike] = None,
# #     iqr: Optional[ArrayLike] = None,
# #     n: Optional[Union[int, float, ArrayLike]] = None,
# #     alpha: float = 0.3,
# #     **kwargs,
# # ) -> Tuple[Axes, pd.DataFrame]:
# #     """
# #      Unified plotting function for both simple plots and those with confidence intervals.
# 
# #      This function handles multiple plotting scenarios:
# #      1. Simple line plot when only yy is provided
# #      2. Mean with standard deviation bands
# #      3. Mean with confidence interval bands
# #      4. Median with interquartile range bands
# #      Each plot can include sample size annotations.
# 
# #      Input Combinations:
# #      -----------------
# #      1. Simple Plot: (xx, yy)
# #         - Basic line plot without confidence bands
# 
# #      2. Mean Plot:
# #         a. (mean, std) - Shows mean ± standard deviation
# #         b. (mean, ci)  - Shows mean with 95% confidence intervals
# 
# #      3. Median Plot:
# #         - (median, iqr) - Shows median ± interquartile range
# 
# #      Sample Size Annotation Options:
# #      -----------------------------
# #      - Single n: Adds "(n=100)" to label
# #      - Array n: Shows range "(ns=100-140)"
# #      - Different length n: Uses mean, shows "(n (mean)=120.5)"
# 
# #      Parameters
# #      ----------
# #      axis : matplotlib.axes.Axes
# #          Target axes for plotting
# #      xx : array-like, optional
# #          X-axis values
# #      yy : array-like, optional
# #          Y-axis values for simple plotting
# #      mean : array-like, optional
# #          Mean values
# #      median : array-like, optional
# #          Median values
# #     std : array-like, optional
# #          Standard deviation values
# #      ci : array-like, optional
# #          Confidence interval values
# #      iqr : array-like, optional
# #          Interquartile range values
# #      n : int, float, or array-like, optional
# #          Sample size for label annotation
# #      alpha : float, default=0.3
# #          Transparency for confidence bands
# #      **kwargs : dict
# #          Additional arguments for plotting
# 
# #      Examples
# #      --------
# #      1. Simple Plot:
# #          >>> axis.plot_(xx=[1,2,3], yy=[4,5,6])
# #          # Basic line plot with no confidence intervals
# 
# #      2. Mean with Standard Deviation:
# #          >>> axis.plot_(
# #          ...     xx=[1,2,3],
# #          ...     mean=[4,5,6],
# #          ...     std=[0.5,0.5,0.5]
# #          ... )
# #          # Line plot with shaded std bands
# #          # Label: "mean ± std"
# 
# #      3. Mean with Confidence Intervals:
# #          >>> axis.plot_(
# #          ...     mean=[4,5,6],
# #          ...     ci=[1,1,1]
# #          ... )
# #          # Line plot with shaded CI bands
# #          # Label: "mean with 95% CI"
# 
# #      4. Median with IQR:
# #          >>> axis.plot_(
# #          ...     median=[4,5,6],
# #          ...     iqr=[1,1,1]
# #          ... )
# #          # Line plot with shaded IQR bands
# #          # Label: "median ± IQR"
# 
# #      5. Sample Size Annotations:
# #          >>> axis.plot_(mean=[4,5,6], std=[1,1,1], n=100)
# #          # Label: "... (n=100)"
# #          >>> axis.plot_(mean=[4,5,6], std=[1,1,1], n=[100,120,140])
# #          # Label: "... (ns=100–140)"
# 
# #      Returns
# #      -------
# #      Tuple[matplotlib.axes.Axes, pd.DataFrame]
# #          Modified axes and plotting data
# #      ----------
# #     """
# #     # Validate argument combinations
# #     if yy is not None:
# #         if any(x is not None for x in [mean, median, std, ci, iqr]):
# #             raise ValueError(
# #                 "When yy is provided, statistical measures (mean, median, std, ci, iqr) should not be used"
# #             )
# #     else:
# #         if mean is not None and median is not None:
# #             raise ValueError("Cannot specify both mean and median")
# #         if mean is not None:
# #             if std is not None and ci is not None:
# #                 raise ValueError("Cannot specify both std and ci with mean")
# #             if iqr is not None:
# #                 raise ValueError("IQR can only be used with median")
# #         if median is not None:
# #             if std is not None or ci is not None:
# #                 raise ValueError("std and ci can only be used with mean")
# 
# #     # Length validation for arrays
# #     arrays = [
# #         (xx, "xx"),
# #         (yy, "yy"),
# #         (mean, "mean"),
# #         (median, "median"),
# #         (std, "std"),
# #         (ci, "ci"),
# #         (iqr, "iqr"),
# #     ]
# #     lengths = [len(arr) for arr, name in arrays if arr is not None]
# 
# #     if lengths and not all(l == lengths[0] for l in lengths):
# #         raise ValueError("All provided arrays must have the same length")
# 
# #     if yy is not None:
# #         if xx is None:
# #             xx = np.arange(len(yy))
# #         axis.plot(xx, yy, **kwargs)
# #         return axis, pd.DataFrame({"xx": xx, "yy": yy})
# 
# #     return plot_with_ci(
# #         axis=axis,
# #         xx=xx,
# #         mean=mean,
# #         median=median,
# #         std=std,
# #         ci=ci,
# #         iqr=iqr,
# #         n=n,
# #         alpha=alpha,
# #         **kwargs,
# #     )
# 
# 
# # def plot_with_ci(
# #     axis: Axes,
# #     xx: Optional[ArrayLike] = None,
# #     mean: Optional[ArrayLike] = None,
# #     median: Optional[ArrayLike] = None,
# #     std: Optional[ArrayLike] = None,
# #     ci: Optional[ArrayLike] = None,
# #     iqr: Optional[ArrayLike] = None,
# #     n: Optional[Union[int, float, ArrayLike]] = None,
# #     alpha: float = 0.3,
# #     **kwargs,
# # ) -> Tuple[Axes, pd.DataFrame]:
# #     """
# #     Plots central tendency with confidence intervals or ranges.
# 
# #     Parameters
# #     ----------
# #     axis : matplotlib.axes.Axes
# #         Target axes for plotting
# #     xx : array-like, optional
# #         X-axis values
# #     yy : array-like, optional
# #         Y-axis values for simple plotting
# #     mean : array-like, optional
# #         Mean values
# #     median : array-like, optional
# #         Median values
# #     std : array-like, optional
# #         Standard deviation values
# #     ci : array-like, optional
# #         Confidence interval values
# #     iqr : array-like, optional
# #         Interquartile range values
# #     n : int, float, or array-like, optional
# #         Sample size for label annotation
# #     alpha : float, default=0.3
# #         Transparency for confidence bands
# #     **kwargs : dict
# #         Additional arguments for plotting
# 
# #     Returns
# #     -------
# #     Tuple[matplotlib.axes.Axes, pd.DataFrame]
# #         Modified axes and plotting data
# #     ----------
# #     """
# #     label = kwargs.pop("label", "")
# 
# #     if median is not None:
# #         central_tendency = median
# #         label += " (median"
# #         if iqr is not None:
# #             label += " ± IQR)"
# #             lower_bound = median - iqr / 2
# #             upper_bound = median + iqr / 2
# #         else:
# #             raise ValueError("IQR required when plotting median")
# #     elif mean is not None:
# #         central_tendency = mean
# #         label += " (mean"
# #         if std is not None:
# #             label += " ± std)"
# #             lower_bound = mean - std
# #             upper_bound = mean + std
# #         elif ci is not None:
# #             label += " with 95% CI)"
# #             lower_bound = mean - ci / 2
# #             upper_bound = mean + ci / 2
# #         else:
# #             raise ValueError("Either std or ci required when plotting mean")
# #     else:
# #         raise ValueError("Either mean or median required")
# 
# #     sample_size = None
# #     if n is not None:
# #         if isinstance(n, (int, float)):
# #             sample_size = n * np.ones_like(xx)
# #             label += f" (n={n:,})" if isinstance(n, int) else f" (n={n:.1f})"
# #         elif len(n) == len(xx):
# #             sample_size = n
# #             if min(n) == max(n):
# #                 label += f" (ns={min(n):,})"
# #             else:
# #                 label += f" (ns={min(n):,}–{max(n):,})"
# #         else:
# #             mean_n = np.mean(n)
# #             sample_size = mean_n * np.ones_like(xx)
# #             label += f" (n (mean) ={mean_n:.1f})"
# 
# #     if xx is None:
# #         xx = np.arange(len(central_tendency))
# 
# #     axis.plot(xx, central_tendency, label=label, **kwargs)
# #     axis.fill_between(xx, lower_bound, upper_bound, alpha=alpha, **kwargs)
# 
# #     return axis, pd.DataFrame(
# #         {
# #             "label": [label] * len(xx),
# #             "xx": xx,
# #             "lower_bound": lower_bound,
# #             "central_tendency": central_tendency,
# #             "upper_bound": upper_bound,
# #             "sample_size": sample_size,
# #         }
# #     )
# 
# 
# #
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot_.py
# --------------------------------------------------------------------------------
