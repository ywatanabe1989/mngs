#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:32:32 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/ax/test__force_aspect.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/ax/test__force_aspect.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mngs.plt.ax._force_aspect import force_aspect

matplotlib.use("Agg")  # Use non-GUI backend for testing


class TestMainFunctionality:
    def setup_method(self):
        # Setup test fixtures
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111)

        # Create an image with known dimensions
        data = np.random.rand(10, 20)  # Height x Width
        self.im = self.ax.imshow(data, extent=[0, 20, 0, 10])  # Width x Height

    def teardown_method(self):
        # Clean up after tests
        plt.close(self.fig)

    def test_basic_functionality(self):
        # Test with default aspect (aspect=1)
        ax = force_aspect(self.ax)

        # Get the current aspect ratio
        current_aspect = self.ax.get_aspect()

        # With aspect=1, it should set aspect to ratio of width/height (20/10 = 2) divided by 1
        # So aspect should be 2
        assert np.isclose(current_aspect, 2.0, rtol=1e-2)

    def test_custom_aspect(self):
        # Test with custom aspect = 2
        ax = force_aspect(self.ax, aspect=2)

        # Get the current aspect ratio
        current_aspect = self.ax.get_aspect()

        # With aspect=2, it should set aspect to ratio of width/height (20/10 = 2) divided by 2
        # So aspect should be 1
        assert np.isclose(current_aspect, 1.0, rtol=1e-2)

    def test_no_images(self):
        # Test with no images on the axes
        empty_ax = self.fig.add_subplot(122)

        # Should raise IndexError as the function tries to access im[0]
        with pytest.raises(IndexError):
            force_aspect(empty_ax)

    def test_with_multiple_images(self):
        # Add another image with different dimensions
        second_data = np.random.rand(5, 10)  # Height x Width
        second_im = self.ax.imshow(
            second_data, extent=[0, 10, 0, 5]
        )  # Width x Height

        # The function should use the first image from get_images()
        ax = force_aspect(self.ax)

        # Get the current aspect ratio
        current_aspect = self.ax.get_aspect()

        # Should still be using the first image (20/10 = 2)
        assert np.isclose(current_aspect, 2.0, rtol=1e-2)

    def test_with_negative_extent(self):
        # Create an image with negative extent
        neg_data = np.random.rand(10, 20)  # Height x Width
        neg_ax = self.fig.add_subplot(133)
        neg_im = neg_ax.imshow(
            neg_data, extent=[-20, 0, -10, 0]
        )  # Width x Height

        # Test force_aspect
        neg_ax = force_aspect(neg_ax)

        # Should handle negative extent correctly, absolute value is used
        current_aspect = neg_ax.get_aspect()
        assert np.isclose(current_aspect, 2.0, rtol=1e-2)


if __name__ == "__main__":
    import os

    import matplotlib
    import pytest

    pytest.main([os.path.abspath(__file__)])
    import matplotlib

# --------------------------------------------------------------------------------
# Start of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_force_aspect.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: 2024-05-14 00:03:52 (2)
# # /ssh:ywatanabe@444:/home/ywatanabe/proj/mngs/src/mngs/plt/ax/_force_aspect.py
#
# import matplotlib.pyplot as plt
# import numpy as np
#
#
# def force_aspect(ax, aspect=1):
#     im = ax.get_images()
#
#     extent = im[0].get_extent()
#
#     ax.set_aspect(
#         abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect
#     )
#     return ax
#
#
# # Traceback (most recent call last):
# #   File "/home/ywatanabe/proj/entrance/neurovista/./scripts/ml/clustering_vit.py", line 199, in <module>
# #     main(args.model, args.clf_model)
# #   File "/home/ywatanabe/proj/entrance/neurovista/./scripts/ml/clustering_vit.py", line 152, in main
# #     fig, _legend_figs, _model = clustering_fn(
# #   File "/home/ywatanabe/proj/mngs/src/mngs/ml/clustering/_pca.py", line 64, in pca
# #     ax = mngs.plt.ax.force_aspect(ax)
# #   File "/home/ywatanabe/proj/mngs/src/mngs/plt/ax/_force_aspect.py", line 13, in force_aspect
# #     extent = im[0].get_extent()
# # IndexError: list index out of range

# --------------------------------------------------------------------------------
# End of Source Code from: /home/ywatanabe/proj/_mngs_repo/src/mngs/plt/ax/_force_aspect.py
# --------------------------------------------------------------------------------

# EOF