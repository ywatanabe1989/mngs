#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-30 11:49:22 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/_AxisWrapperMixins/test__BasicPlotMixin.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/_AxisWrapperMixins/test__BasicPlotMixin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import mngs
import numpy as np

matplotlib.use("agg")


def test_imshow2d():
    """Test imshow2d function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    data = np.random.randn(20, 20)
    # Plot
    ax.imshow2d(data, cmap="viridis", interpolation="nearest")
    # Visualization
    ax.set_xyt("X", "Y", "Imshow2D Test")
    # Saving
    spath = f"./imshow2d_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_imshow2d_xyz():
    """Test imshow2d with xyz=True"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    data = np.random.randn(20, 20)
    # Plot
    ax.imshow2d(data, xyz=True, cmap="plasma")
    # Visualization
    ax.set_xyt("X", "Y", "Imshow2D with XYZ Test")
    # Saving
    spath = f"./imshow2d_xyz_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_plot_():
    """Test plot_ function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    # Plot
    ax.plot_(data=x, yy=y, line="-", label="Sine Function")
    # Visualization
    ax.set_xyt("X axis", "Y axis", "Plot_ Test")
    ax.legend()
    # Saving
    spath = f"./plot_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


# def test_plot_with_fill():
#     """Test plot_ function with fill"""
#     # Figure
#     fig, ax = mngs.plt.subplots()
#     # Data
#     x = np.linspace(0, 10, 100)
#     y = np.sin(x)
#     # Plot
#     ax.plot_(
#         xx=x, yy=y, line="-", fill="below", label="Sine Function with Fill"
#     )
#     # Visualization
#     ax.set_xyt("X axis", "Y axis", "Plot_ with Fill Test")
#     ax.legend()
#     # Saving
#     spath = f"./plot_with_fill_test.png"
#     mngs.io.save(fig, spath, symlink_from_cwd=False)
#     # Closing
#     mngs.plt.close(fig)
#     # Assertion
#     assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_kde():
    """Test kde function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    data = np.concatenate(
        [np.random.normal(0, 1, 500), np.random.normal(5, 1, 300)]
    )
    # Plot
    ax.kde(data, label="Bimodal Distribution", fill=True)
    # Visualization
    ax.set_xyt("Value", "Density", "KDE Test")
    ax.legend()
    # Saving
    spath = f"./kde_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_kde_cumulative():
    """Test kde function with cumulative=True"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    data = np.random.normal(0, 1, 1000)
    # Plot
    ax.kde(data, label="Normal Distribution", cumulative=True)
    # Visualization
    ax.set_xyt("Value", "Cumulative Density", "Cumulative KDE Test")
    ax.legend()
    # Saving
    spath = f"./kde_cumulative_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_conf_mat():
    """Test conf_mat function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    conf_matrix = np.array([[85, 10, 5], [15, 70, 15], [10, 20, 70]])
    class_labels = ["Class A", "Class B", "Class C"]
    # Plot
    ax.conf_mat(
        conf_matrix,
        x_labels=class_labels,
        y_labels=class_labels,
        title="Confusion Matrix Test",
        bacc=True,
    )
    # Saving
    spath = f"./conf_mat_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_rectangle():
    """Test rectangle function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)

    # Add rectangles
    ax.rectangle(
        2, 0, 2, 0.5, color="red", alpha=0.3, label="Highlight Region 1"
    )
    ax.rectangle(
        6, -0.5, 2, 0.5, color="blue", alpha=0.3, label="Highlight Region 2"
    )
    # Visualization
    ax.set_xyt("X axis", "Y axis", "Rectangle Test")
    ax.legend()
    # Saving
    spath = f"./rectangle_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_fillv():
    """Test fillv function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    # Add vertical shaded regions
    ax.fillv([2, 6], [4, 8], color="green", alpha=0.3)
    # Visualization
    ax.set_xyt("X axis", "Y axis", "FillV Test")
    # Saving
    spath = f"./fillv_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_boxplot_():
    """Test boxplot_ function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    data = [
        np.random.normal(0, 1, 100),
        np.random.normal(2, 1.5, 100),
        np.random.normal(5, 0.8, 100),
    ]
    # Plot
    ax.boxplot_(data, labels=["Group A", "Group B", "Group C"])
    # Visualization
    ax.set_xyt("Groups", "Values", "Boxplot_ Test")
    # Saving
    spath = f"./boxplot_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_raster():
    """Test raster function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data - sample spike train data
    n_neurons = 5
    t_max = 100
    np.random.seed(42)  # For reproducibility
    positions = [
        np.sort(np.random.uniform(0, t_max, np.random.randint(20, 50)))
        for _ in range(n_neurons)
    ]
    labels = [f"Neuron {ii+1}" for ii in range(n_neurons)]
    # Plot
    ax.raster(positions, labels=labels)
    # Visualization
    ax.set_xyt("Time (ms)", "Neuron", "Raster Plot Test")
    # Saving
    spath = f"./raster_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_ecdf():
    """Test ecdf function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data
    data = np.random.normal(0, 1, 1000)
    # Plot
    ax.ecdf(data, label="Normal Distribution ECDF")
    # Visualization
    ax.set_xyt("Value", "Cumulative Probability", "ECDF Test")
    ax.legend()
    # Saving
    spath = f"./ecdf_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


def test_joyplot():
    """Test joyplot function"""
    # Figure
    fig, ax = mngs.plt.subplots()
    # Data for multiple distributions
    data = {
        "Group A": np.random.normal(0, 1, 500),
        "Group B": np.random.normal(2, 1.5, 500),
        "Group C": np.random.normal(5, 0.8, 500),
        "Group D": np.random.normal(8, 2, 500),
    }
    # Plot
    ax.joyplot(data)
    # Visualization
    ax.set_xyt("Value", "", "Joyplot Test")
    # Saving
    spath = f"./joyplot_test.png"
    mngs.io.save(fig, spath, symlink_from_cwd=False)
    # Closing
    mngs.plt.close(fig)
    # Assertion
    assert os.path.exists(spath), f"Failed to save figure to {spath}"


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-04-27 12:21:40 (ywatanabe)"
# # File: /ssh:sp:/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# THIS_FILE = "/home/ywatanabe/proj/mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py"
#
# from functools import wraps
# from typing import Any, Dict, List, Optional, Tuple, Union
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# from scipy.stats import gaussian_kde
#
# from ....pd import to_xyz
# from ....plt import ax as ax_module
# from ....types import ArrayLike
#
#
# class BasicPlotMixin:
#     """Mixin class for basic plotting operations."""
#
#     @wraps(ax_module.imshow2d)
#     def imshow2d(self, arr_2d: ArrayLike, **kwargs) -> None:
#         method_name = "imshow2d"
#         with self._no_tracking():
#             self.axis = ax_module.imshow2d(self.axis, arr_2d, **kwargs)
#         out = pd.DataFrame(arr_2d)
#         if kwargs.get("xyz", False):
#             out = to_xyz(out)
#         self._track(
#             kwargs.get("track"), kwargs.get("id"), method_name, out, None
#         )
#
#     @wraps(ax_module.plot_)
#     def plot_(
#         self,
#         data: ArrayLike,
#         xx: Optional[ArrayLike] = None,
#         yy: Optional[ArrayLike] = None,
#         line: Optional[str] = None,
#         fill: Optional[str] = None,
#         n: Optional[Union[int, float, ArrayLike]] = None,
#         alpha: float = 0.3,
#         **kwargs,
#     ) -> None:
#         method_name = "plot_"
#         with self._no_tracking():
#             self.axis, df = ax_module.plot_(
#                 self.axis,
#                 data=data,
#                 xx=xx,
#                 yy=yy,
#                 line=line,
#                 fill=fill,
#                 n=n,
#                 alpha=alpha,
#                 **kwargs,
#             )
#         self._track(
#             kwargs.get("track"), kwargs.get("id"), method_name, df, None
#         )
#
#     # @wraps(ax_module.kde)
#     def kde(self, data: ArrayLike, **kwargs) -> None:
#         method_name = "kde"
#         n_samples = (~np.isnan(data)).sum()
#         if kwargs.get("label"):
#             kwargs["label"] = f"{kwargs['label']} (n={n_samples})"
#         xlim = kwargs.get("xlim", (data.min(), data.max()))
#         xs = np.linspace(*xlim, int(1e3))
#         density = gaussian_kde(data)(xs)
#         density /= density.sum()
#         if kwargs.get("cumulative"):
#             density = np.cumsum(density)
#
#         with self._no_tracking():
#             if kwargs.get("fill"):
#                 self.axis.fill_between(xs, density, **kwargs)
#                 # self.axis.plot_(xs, density, **kwargs)
#             else:
#                 self.plot_(xx=xs, yy=density, label=kwargs.get("label"))
#
#         out = pd.DataFrame(
#             {"x": xs, "kde": density, "n": [len(data) for _ in range(len(xs))]}
#         )
#         self._track(
#             kwargs.get("track"), kwargs.get("id"), method_name, out, None
#         )
#
#     # @wraps(ax_module.conf_mat)
#     def conf_mat(
#         self,
#         data: ArrayLike,
#         x_labels: Optional[List[str]] = None,
#         y_labels: Optional[List[str]] = None,
#         title: str = "Confusion Matrix",
#         cmap: str = "Blues",
#         cbar: bool = True,
#         cbar_kw: Dict[str, Any] = {},
#         label_rotation_xy: Tuple[float, float] = (15, 15),
#         x_extend_ratio: float = 1.0,
#         y_extend_ratio: float = 1.0,
#         bacc: bool = False,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#         method_name = "conf_mat"
#         with self._no_tracking():
#             out = ax_module.conf_mat(
#                 self.axis,
#                 data,
#                 x_labels=x_labels,
#                 y_labels=y_labels,
#                 title=title,
#                 cmap=cmap,
#                 cbar=cbar,
#                 cbar_kw=cbar_kw,
#                 label_rotation_xy=label_rotation_xy,
#                 x_extend_ratio=x_extend_ratio,
#                 y_extend_ratio=y_extend_ratio,
#                 bacc=bacc,
#                 track=track,
#                 id=id,
#                 **kwargs,
#             )
#             bacc_val = None
#             if bacc:
#                 self.axis, bacc_val = out
#             else:
#                 self.axis = out
#         out = data, bacc_val
#         self._track(track, id, method_name, out, None)
#
#     @wraps(ax_module.rectangle)
#     def rectangle(
#         self,
#         xx: float,
#         yy: float,
#         width: float,
#         height: float,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#         method_name = "rectangle"
#         with self._no_tracking():
#             self.axis = ax_module.rectangle(
#                 self.axis, xx, yy, width, height, **kwargs
#             )
#         self._track(track, id, method_name, None, None)
#
#     @wraps(ax_module.fillv)
#     def fillv(
#         self,
#         starts: ArrayLike,
#         ends: ArrayLike,
#         color: str = "red",
#         alpha: float = 0.2,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#
#         method_name = "fillv"
#         self.axis = ax_module.fillv(
#             self.axis, starts, ends, color=color, alpha=alpha
#         )
#         out = (starts, ends)
#         self._track(track, id, method_name, out, None)
#
#     # @wraps(ax_module.boxplot_)
#     def boxplot_(
#         self,
#         data: ArrayLike,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#         method_name = "boxplot_"
#         _data = data.copy()
#         n = len(data)
#
#         if kwargs.get("label"):
#             kwargs["label"] = kwargs["label"] + f" (n={n})"
#
#         with self._no_tracking():
#             self.axis.boxplot(data, **kwargs)
#
#         out = pd.DataFrame(
#             {
#                 "data": _data,
#                 "n": [n for _ in range(len(data))],
#             }
#         )
#         self._track(track, id, method_name, out, None)
#
#     # @wraps(ax_module.raster)
#     def raster(
#         self,
#         positions: List[ArrayLike],
#         time: Optional[ArrayLike] = None,
#         labels: Optional[List[str]] = None,
#         colors: Optional[List[str]] = None,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#
#         method_name = "raster"
#         if colors is None:
#             colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
#         if len(colors) < len(positions):
#             colors = colors * (len(positions) // len(colors) + 1)
#
#         with self._no_tracking():
#             for i, (pos, color) in enumerate(zip(positions, colors)):
#                 label = (
#                     labels[i]
#                     if labels is not None and i < len(labels)
#                     else None
#                 )
#                 self.axis.eventplot(pos, colors=color, label=label, **kwargs)
#
#             if labels is not None:
#                 self.axis.legend()
#
#             df = ax_module.raster_plot(self.axis, positions, time=time)[1]
#
#         if id is not None:
#             df.columns = [f"{id}_{method_name}_{col}" for col in df.columns]
#         out = df
#         self._track(track, id, method_name, out, None)
#
#     @wraps(ax_module.ecdf)
#     def ecdf(
#         self,
#         data: ArrayLike,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#         method_name = "ecdf"
#         with self._no_tracking():
#             self.axis, df = ax_module.ecdf(self.axis, data, **kwargs)
#         out = df
#         self._track(track, id, method_name, out, None)
#
#     @wraps(ax_module.joyplot)
#     def joyplot(
#         self,
#         data: ArrayLike,
#         track: bool = True,
#         id: Optional[str] = None,
#         **kwargs,
#     ) -> None:
#         method_name = "joyplot"
#         with self._no_tracking():
#             self.axis = ax_module.joyplot(self.axis, data, **kwargs)
#         out = data
#         self._track(track, id, method_name, out, None)
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/_subplots/_AxisWrapperMixins/_BasicPlotMixin.py
# --------------------------------------------------------------------------------

# EOF