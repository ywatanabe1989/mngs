#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-05-02 10:28:39 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/_plot/test__plot_conf_mat.py
# ----------------------------------------
import os

__FILE__ = "./tests/mngs/plt/ax/_plot/test__plot_conf_mat.py"
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from mngs.plt.ax._plot._plot_conf_mat import plot_conf_mat
from mngs.plt.utils import calc_bacc_from_conf_mat


class TestPlotConfMat:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)
        # Create a sample confusion matrix
        self.cm_data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
        # Create output directory if it doesn't exist
        self.out_dir = __file__.replace(".py", "_out")
        os.makedirs(self.out_dir, exist_ok=True)

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def save_test_figure(self, method_name):
        """Helper method to save figure using method name"""
        from mngs.io import save

        spath = f"./{os.path.basename(__file__).replace('.py', '')}_{method_name}.jpg"
        save(self.fig, spath)
        # Check saved file
        actual_spath = os.path.join(self.out_dir, spath)
        assert os.path.exists(actual_spath), f"Failed to save figure to {spath}"

    def test_basic_functionality(self):
        # Test basic confusion matrix plotting
        ax = plot_conf_mat(self.ax, self.cm_data)

        # Save figure
        self.save_test_figure("test_basic_functionality")

        # Should create a heatmap
        assert len(self.ax.collections) > 0
        # Should set labels
        assert self.ax.get_xlabel() == "Predicted label"
        assert self.ax.get_ylabel() == "True label"
        # Should include bACC in title
        assert "bACC" in self.ax.get_title()

    def test_return_bacc(self):
        # Test returning balanced accuracy
        ax, bacc = plot_conf_mat(self.ax, self.cm_data, calc_bacc=True)

        # Save figure
        self.save_test_figure("test_return_bacc")

        # Calculate expected bACC
        expected_bacc = calc_bacc_from_conf_mat(self.cm_data)
        # Check the returned bACC value
        assert np.isclose(bacc, expected_bacc)

    def test_custom_cmap(self):
        # Test with custom colormap
        custom_cmap = "hot"
        ax = plot_conf_mat(self.ax, self.cm_data, cmap=custom_cmap)

        # Save figure
        self.save_test_figure("test_custom_cmap")

        # Check that the colormap was applied
        assert self.ax.collections[0].cmap.name == custom_cmap

    def test_custom_title(self):
        # Test with custom title
        custom_title = "My Confusion Matrix"
        ax = plot_conf_mat(self.ax, self.cm_data, title=custom_title)

        # Save figure
        self.save_test_figure("test_custom_title")

        # Title should include custom title and bACC
        assert custom_title in self.ax.get_title()
        assert "bACC" in self.ax.get_title()

    def test_calc_bacc_function(self):
        # Test balanced accuracy calculation
        bacc = calc_bacc_from_conf_mat(self.cm_data)

        # Calculate expected bACC
        per_class = np.diag(self.cm_data) / np.sum(self.cm_data, axis=1)
        expected_bacc = np.mean(per_class)
        expected_bacc = round(expected_bacc, 3)

        # Check the calculated bACC value
        assert np.isclose(bacc, expected_bacc)

    def test_with_dataframe(self):
        # Test with pandas DataFrame input
        df = pd.DataFrame(self.cm_data)
        ax = plot_conf_mat(self.ax, df)

        # Save figure
        self.save_test_figure("test_with_dataframe")

        # Should work with DataFrame same as array
        assert len(self.ax.collections) > 0
        assert "bACC" in self.ax.get_title()

    def test_with_labels(self):
        # Test with custom labels
        x_labels = ["A", "B", "C"]
        y_labels = ["X", "Y", "Z"]
        ax, _ = plot_conf_mat(
            self.ax, self.cm_data, x_labels=x_labels, y_labels=y_labels
        )

        # Save figure
        self.save_test_figure("test_with_labels")

    def test_plot_conf_mat_savefig(self):
        ax, _ = plot_conf_mat(
            self.ax,
            self.cm_data,
            x_labels=["A", "B", "C"],
            y_labels=["X", "Y", "Z"],
        )

        # Saving
        from mngs.io import save

        spath = f"./{os.path.basename(__file__)}.jpg"
        save(self.fig, spath)

        # Check saved file
        ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
        actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
        assert os.path.exists(actual_spath), f"Failed to save figure to {spath}"


# class TestMainFunctionality:
#     def setup_method(self):
#         # Setup test fixtures
#         self.fig = plt.figure()
#         self.ax = self.fig.add_subplot(111)

#         # Create a sample confusion matrix
#         self.cm_data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])

#     def teardown_method(self):
#         # Clean up after tests
#         plt.close(self.fig)

#     def test_basic_functionality(self):
#         # Test basic confusion matrix plotting
#         ax = plot_conf_mat(self.ax, self.cm_data)

#         # Should create a heatmap
#         assert len(self.ax.collections) > 0

#         # Should set labels
#         assert self.ax.get_xlabel() == "Predicted label"
#         assert self.ax.get_ylabel() == "True label"

#         # Should include bACC in title
#         assert "bACC" in self.ax.get_title()

#     # def test_with_labels(self):
#     #     # Test with custom x and y labels
#     #     x_labels = ["A", "B", "C"]
#     #     y_labels = ["X", "Y", "Z"]
#     #     ax = plot_conf_mat(
#     #         self.ax, self.cm_data, x_labels=x_labels, y_labels=y_labels
#     #     )

#     #     # Force draw to ensure tick labels are updated
#     #     self.fig.canvas.draw()

#     #     # Check tick labels
#     #     x_ticklabels = [label.get_text() for label in ax.get_xticklabels()]
#     #     y_ticklabels = [label.get_text() for label in ax.get_yticklabels()]

#     #     assert x_ticklabels == x_labels
#     #     assert y_ticklabels == y_labels

#     def test_return_bacc(self):
#         # Test returning balanced accuracy
#         ax, bacc = plot_conf_mat(self.ax, self.cm_data, calc_bacc=True)

#         # Calculate expected bACC
#         expected_bacc = calc_bacc_from_conf_mat(self.cm_data)

#         # Check the returned bACC value
#         assert np.isclose(bacc, expected_bacc)

#     def test_custom_cmap(self):
#         # Test with custom colormap
#         custom_cmap = "hot"
#         ax = plot_conf_mat(self.ax, self.cm_data, cmap=custom_cmap)

#         # Check that the colormap was applied
#         assert self.ax.collections[0].cmap.name == custom_cmap

#     def test_custom_title(self):
#         # Test with custom title
#         custom_title = "My Confusion Matrix"
#         ax = plot_conf_mat(self.ax, self.cm_data, title=custom_title)

#         # Title should include custom title and bACC
#         assert custom_title in self.ax.get_title()
#         assert "bACC" in self.ax.get_title()

#     def test_calc_bacc_function(self):
#         # Test balanced accuracy calculation
#         bacc = calc_bacc_from_conf_mat(self.cm_data)

#         # Calculate expected bACC
#         per_class = np.diag(self.cm_data) / np.sum(self.cm_data, axis=1)
#         expected_bacc = np.mean(per_class)
#         expected_bacc = round(expected_bacc, 3)

#         # Check the calculated bACC value
#         assert np.isclose(bacc, expected_bacc)

#     # def test_with_rotation(self):
#     #     # Test custom label rotation
#     #     rotation = (30, 45)
#     #     ax = plot_conf_mat(self.ax, self.cm_data, label_rotation_xy=rotation)

#     #     # Force draw to ensure tick labels are updated
#     #     self.fig.canvas.draw()

#     #     # Check rotation of tick labels
#     #     x_rotations = [label.get_rotation() for label in ax.get_xticklabels()]
#     #     y_rotations = [label.get_rotation() for label in ax.get_yticklabels()]

#     #     assert all(rot == rotation[0] for rot in x_rotations)
#     #     assert all(rot == rotation[1] for rot in y_rotations)

#     def test_with_dataframe(self):
#         # Test with pandas DataFrame input
#         df = pd.DataFrame(self.cm_data)
#         ax = plot_conf_mat(self.ax, df)

#         # Should work with DataFrame same as array
#         assert len(self.ax.collections) > 0
#         assert "bACC" in self.ax.get_title()

#     def test_plot_conf_mat_savefig(self):

#         ax, _ = plot_conf_mat(
#             self.ax,
#             self.cm_data,
#             x_labels=["A", "B", "C"],
#             y_labels=["X", "Y", "Z"],
#         )

#         # Saving
#         from mngs.io import save

#         spath = f"./{os.path.basename(__file__)}.jpg"
#         save(self.fig, spath)

#         # Check saved file
#         ACTUAL_SAVE_DIR = __file__.replace(".py", "_out")
#         actual_spath = os.path.join(ACTUAL_SAVE_DIR, spath)
#         assert os.path.exists(
#             actual_spath
#         ), f"Failed to save figure to {spath}"

if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_conf_mat.py
# --------------------------------------------------------------------------------
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-
# # Timestamp: "2025-05-02 09:20:23 (ywatanabe)"
# # File: /home/ywatanabe/proj/mngs_repo/src/mngs/plt/ax/_plot/_plot_conf_mat.py
# # ----------------------------------------
# import os
# __FILE__ = (
#     "./src/mngs/plt/ax/_plot/_plot_conf_mat.py"
# )
# __DIR__ = os.path.dirname(__FILE__)
# # ----------------------------------------
#
# from typing import List, Optional, Tuple, Union
#
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import seaborn as sns
#
# from ...utils._calc_bacc_from_conf_mat import calc_bacc_from_conf_mat
# from .._style._extend import extend as mngs_plt_extend
#
#
# def plot_conf_mat(
#     axis: plt.Axes,
#     data: Union[np.ndarray, pd.DataFrame],
#     x_labels: Optional[List[str]] = None,
#     y_labels: Optional[List[str]] = None,
#     title: str = "Confusion Matrix",
#     cmap: str = "Blues",
#     cbar: bool = True,
#     cbar_kw: dict = {},
#     label_rotation_xy: Tuple[float, float] = (15, 15),
#     x_extend_ratio: float = 1.0,
#     y_extend_ratio: float = 1.0,
#     calc_bacc: bool = False,
#     **kwargs,
# ) -> Union[plt.Axes, Tuple[plt.Axes, float]]:
#     """Creates a confusion matrix heatmap with optional balanced accuracy.
#
#     Parameters
#     ----------
#     axis : plt.Axes
#         Matplotlib axes to plot on
#     data : Union[np.ndarray, pd.DataFrame]
#         Confusion matrix data
#     x_labels : Optional[List[str]], optional
#         Labels for predicted classes
#     y_labels : Optional[List[str]], optional
#         Labels for true classes
#     title : str, optional
#         Plot title
#     cmap : str, optional
#         Colormap name
#     cbar : bool, optional
#         Whether to show colorbar
#     cbar_kw : dict, optional
#         Colorbar parameters
#     label_rotation_xy : Tuple[float, float], optional
#         (x,y) label rotation angles
#     x_extend_ratio : float, optional
#         X-axis extension ratio
#     y_extend_ratio : float, optional
#         Y-axis extension ratio
#     calc_bacc : bool, optional
#         Calculate Balanced Accuracy from Confusion Matrix
#
#     Returns
#     -------
#     Union[plt.Axes, Tuple[plt.Axes, float]]
#         Axes object and optionally balanced accuracy
#
#     Example
#     -------
#     >>> data = np.array([[10, 2, 0], [1, 15, 3], [0, 2, 20]])
#     >>> fig, ax = plt.subplots()
#     >>> ax, bacc = plot_conf_mat(ax, data, x_labels=['A','B','C'],
#     ...                     y_labels=['X','Y','Z'], calc_bacc=True)
#     >>> print(f"Balanced Accuracy: {bacc:.3f}")
#     Balanced Accuracy: 0.889
#     """
#
#     assert isinstance(
#         axis, matplotlib.axes._axes.Axes
#     ), "First argument must be a matplotlib axis"
#
#     if not isinstance(data, pd.DataFrame):
#         data = pd.DataFrame(data)
#
#     bacc_val = calc_bacc_from_conf_mat(data.values)
#     title = f"{title} (bACC = {bacc_val:.3f})"
#
#     res = sns.heatmap(
#         data,
#         ax=axis,
#         cmap=cmap,
#         annot=True,
#         fmt=",d",
#         cbar=False,
#         vmin=0,
#         **kwargs,
#     )
#
#     res.invert_yaxis()
#
#     for _, spine in res.spines.items():
#         spine.set_visible(False)
#
#     axis.set_xlabel("Predicted label")
#     axis.set_ylabel("True label")
#     axis.set_title(title)
#
#     if x_labels is not None:
#         axis.set_xticklabels(x_labels)
#     if y_labels is not None:
#         axis.set_yticklabels(y_labels)
#
#     axis = mngs_plt_extend(axis, x_extend_ratio, y_extend_ratio)
#     if data.shape[0] == data.shape[1]:
#         axis.set_box_aspect(1)
#         axis.set_xticklabels(
#             axis.get_xticklabels(),
#             rotation=label_rotation_xy[0],
#             fontdict={"verticalalignment": "top"},
#         )
#         axis.set_yticklabels(
#             axis.get_yticklabels(),
#             rotation=label_rotation_xy[1],
#             fontdict={"horizontalalignment": "right"},
#         )
#
#     if calc_bacc:
#         return axis, bacc_val
#     else:
#         return axis, None
#
# # EOF
# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_plot/_plot_conf_mat.py
# --------------------------------------------------------------------------------
