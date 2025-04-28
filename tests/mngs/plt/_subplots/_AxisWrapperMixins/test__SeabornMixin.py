#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-04-28 16:53:40 (ywatanabe)"
# File: /home/ywatanabe/proj/mngs_repo/tests/mngs/plt/_subplots/_AxisWrapperMixins/test__SeabornMixin.py
# ----------------------------------------
import os
__FILE__ = (
    "./tests/mngs/plt/_subplots/_AxisWrapperMixins/test__SeabornMixin.py"
)
__DIR__ = os.path.dirname(__FILE__)
# ----------------------------------------
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest


# Test data for seaborn mixin functionality
@pytest.fixture
def sample_data():
    """Fixture that creates sample data for testing."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "category": ["A", "B", "C", "A", "B", "C"] * 5,
            "value": np.random.normal(0, 1, 30),
            "group": ["X", "Y"] * 15,
        }
    )
    return data


def test_sns_copy_doc_decorator():
    """Test that sns_copy_doc decorator properly copies docstrings."""
    # Import the decorator and seaborn
    import seaborn as sns
    from mngs.plt._subplots._AxisWrapperMixins._SeabornMixin import (
        sns_copy_doc,
    )

    # Create a test function with the decorator
    @sns_copy_doc
    def sns_boxplot(self, *args, **kwargs):
        """This docstring should be replaced."""
        pass

    # Check if the docstring was copied from seaborn
    assert sns_boxplot.__doc__ == sns.boxplot.__doc__
    assert "This docstring should be replaced" not in sns_boxplot.__doc__


def test_sns_barplot(sample_data):
    """Test that sns_barplot method correctly wraps seaborn's barplot."""
    # Create a mock figure and axis
    fig, ax = plt.subplots()

    # Import SeabornMixin
    from mngs.plt._subplots._AxisWrapperMixins._SeabornMixin import (
        SeabornMixin,
    )

    # Create a minimal class that inherits from SeabornMixin
    class TestWrapper(SeabornMixin):
        def __init__(self, axis):
            self.axis = axis
            self.tracked_data = {}

        def _track(self, track, id, method, obj, kwargs):
            if track:
                self.tracked_data[id or method] = {
                    "obj": obj,
                    "kwargs": kwargs,
                }

        def _no_tracking(self):
            class NoTrackingContextManager:
                def __enter__(self_cm):
                    return None

                def __exit__(self_cm, *args):
                    return False

            return NoTrackingContextManager()

    # Create an instance of the test wrapper
    wrapper = TestWrapper(ax)

    # Test sns_barplot with basic parameters
    with patch("seaborn.barplot") as mock_barplot:
        mock_barplot.return_value = ax
        wrapper.sns_barplot(
            data=sample_data, x="category", y="value", id="test_plot"
        )

        # Verify seaborn.barplot was called with correct parameters
        mock_barplot.assert_called_once()
        call_kwargs = mock_barplot.call_args[1]
        assert call_kwargs["data"] is sample_data
        assert call_kwargs["x"] == "category"
        assert call_kwargs["y"] == "value"
        assert call_kwargs["ax"] == ax

        # Verify tracking was called
        assert "test_plot" in wrapper.tracked_data
        # The track_obj should be a DataFrame with pivoted data
        assert isinstance(
            wrapper.tracked_data["test_plot"]["obj"], pd.DataFrame
        )


def test_sns_prepare_xyhue(sample_data):
    """Test the _sns_prepare_xyhue method handles different data structures correctly."""
    # Import SeabornMixin
    from mngs.plt._subplots._AxisWrapperMixins._SeabornMixin import (
        SeabornMixin,
    )

    # Create a minimal class that inherits from SeabornMixin
    class TestWrapper(SeabornMixin):
        def __init__(self):
            self.tracked_data = {}

        def _track(self, track, id, method, obj, kwargs):
            pass

        def _no_tracking(self):
            class NoTrackingContextManager:
                def __enter__(self_cm):
                    return None

                def __exit__(self_cm, *args):
                    return False

            return NoTrackingContextManager()

    # Create an instance of the test wrapper
    wrapper = TestWrapper()

    # Test with x, y, and no hue
    result = wrapper._sns_prepare_xyhue(sample_data, "category", "value")
    assert isinstance(result, pd.DataFrame)
    assert result.shape[1] == len(sample_data["category"].unique())

    # Test with x, y, and hue
    result = wrapper._sns_prepare_xyhue(
        sample_data, "category", "value", "group"
    )
    assert isinstance(result, pd.DataFrame)

    # Test with only x
    result = wrapper._sns_prepare_xyhue(sample_data, x="category")
    assert isinstance(result, pd.DataFrame)
    assert "category" in result.columns

    # Test with only y
    result = wrapper._sns_prepare_xyhue(sample_data, y="value")
    assert isinstance(result, pd.DataFrame)
    assert "value" in result.columns

    # Test with no x, y but with hue
    result = wrapper._sns_prepare_xyhue(sample_data, hue="group")
    assert result is sample_data.reset_index()


if __name__ == "__main__":
    import os

    import pytest

    pytest.main([os.path.abspath(__file__)])

# EOF