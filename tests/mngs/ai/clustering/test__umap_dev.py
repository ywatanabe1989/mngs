#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Timestamp: "2025-06-02 17:05:00 (ywatanabe)"
# File: ./tests/mngs/ai/clustering/test__umap_dev.py

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
import matplotlib.pyplot as plt


def test_umap_basic_functionality():
    """Test basic UMAP functionality with minimal parameters."""
    from mngs.ai.clustering._umap import umap
    
    # Generate sample data
    np.random.seed(42)
    data = [np.random.randn(100, 10)]
    labels = [np.random.randint(0, 3, 100)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(100, 2)
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.flat = [mock_ax]  # Add flat attribute to single axis for legend processing
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = umap(data, labels)
            
            assert isinstance(result, tuple)
            assert len(result) == 3  # fig, legend_figs, umap_model
            mock_umap_class.assert_called_once_with(random_state=42)


def test_umap_supervised_mode():
    """Test supervised UMAP clustering."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(50, 5)]
    labels = [np.random.randint(0, 2, 50)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(50, 2)
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.flat = [mock_ax]  # Add flat attribute to single axis for legend processing
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = umap(data, labels, supervised=True)
            
            # Verify supervised fit was called with labels
            mock_umap.fit.assert_called_once()
            call_args = mock_umap.fit.call_args
            assert call_args[1]['y'] is not None  # y parameter should be provided


def test_umap_unsupervised_mode():
    """Test unsupervised UMAP clustering."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(50, 5)]
    labels = [np.random.randint(0, 2, 50)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(50, 2)
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.flat = [mock_ax]  # Add flat attribute to single axis for legend processing
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = umap(data, labels, supervised=False)
            
            # Verify unsupervised fit was called without labels
            mock_umap.fit.assert_called_once()
            call_args = mock_umap.fit.call_args
            assert call_args[1]['y'] is None  # y parameter should be None


def test_umap_with_hues():
    """Test UMAP with custom hue coloring."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(30, 8)]
    labels = [np.random.randint(0, 3, 30)]
    hues = [np.random.choice(['A', 'B', 'C'], 30)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(30, 2)
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.flat = [mock_ax]  # Add flat attribute to single axis for legend processing
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = umap(data, labels, hues=hues)
            
            assert isinstance(result, tuple)
            assert len(result) == 3


def test_umap_with_colors():
    """Test UMAP with custom color mapping."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(30, 8)]
    labels = [np.random.randint(0, 3, 30)]
    colors = [np.random.rand(30, 3).tolist()]  # RGB colors as list
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(30, 2)
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.flat = [mock_ax]  # Add flat attribute to single axis for legend processing
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = umap(data, labels, hues_colors=colors)
            
            assert isinstance(result, tuple)


def test_umap_with_existing_axes():
    """Test UMAP plotting on existing axes."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(50, 10)]
    labels = [np.random.randint(0, 2, 50)]
    
    # For existing axes test, use a single axis wrapped to handle both axis and array behavior
    mock_ax = Mock()
    mock_fig = Mock()
    mock_ax.get_figure.return_value = mock_fig
    mock_ax.flat = [mock_ax]
    # Add methods that make it work like both single axis and array
    mock_ax.__len__ = lambda self: 1
    mock_ax.__iter__ = lambda self: iter([mock_ax])
    mock_ax.__getitem__ = lambda self, key: mock_ax  # For indexing
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(50, 2)
        mock_umap_class.return_value = mock_umap
        
        result = umap(data, labels, axes=mock_ax)
        
        assert isinstance(result, tuple)
        # Should use existing axes instead of creating new ones


def test_umap_with_pretrained_model():
    """Test UMAP with pre-fitted model."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(40, 6)]
    labels = [np.random.randint(0, 4, 40)]
    
    # Mock pre-trained model
    mock_pretrained = Mock()
    mock_pretrained.transform.return_value = np.random.randn(40, 2)
    
    with patch('mngs.plt.subplots') as mock_subplots:
        mock_fig = Mock()
        mock_ax = Mock()
        mock_axes = Mock()
        mock_axes.flat = [mock_ax]
        mock_subplots.return_value = (mock_fig, mock_axes)
        
        result = umap(data, labels, umap_model=mock_pretrained)
        
        # Should use pre-trained model instead of creating new one
        mock_pretrained.transform.assert_called()
        assert result[2] == mock_pretrained


def test_umap_multiple_datasets():
    """Test UMAP with multiple datasets."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(30, 5), np.random.randn(40, 5)]
    labels = [np.random.randint(0, 2, 30), np.random.randint(0, 2, 40)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.side_effect = [
            np.random.randn(30, 2), 
            np.random.randn(40, 2)
        ]
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax1, mock_ax2 = Mock(), Mock()
            # Configure axis mock methods for sharing functionality
            mock_ax1.get_xlim.return_value = (0, 1)
            mock_ax1.get_ylim.return_value = (0, 1)
            mock_ax2.get_xlim.return_value = (0, 1)
            mock_ax2.get_ylim.return_value = (0, 1)
            mock_axes = np.array([mock_ax1, mock_ax2])  # Use numpy array for multiple axes
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = umap(data, labels)
            
            # Should call transform for each dataset
            assert mock_umap.transform.call_count == 2


def test_umap_superimposed_plot():
    """Test UMAP with superimposed plotting."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(30, 5), np.random.randn(40, 5)]
    labels = [np.random.randint(0, 2, 30), np.random.randint(0, 2, 40)]
    colors = [np.random.rand(30, 3).tolist(), np.random.rand(40, 3).tolist()]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.side_effect = [
            np.random.randn(30, 2), 
            np.random.randn(40, 2)
        ]
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax1, mock_ax2, mock_ax3 = Mock(), Mock(), Mock()
            # Configure axis mock methods for sharing functionality
            for ax in [mock_ax1, mock_ax2, mock_ax3]:
                ax.get_xlim.return_value = (0, 1)
                ax.get_ylim.return_value = (0, 1)
            mock_axes = np.array([mock_ax1, mock_ax2, mock_ax3])  # Use numpy array for multiple axes
            mock_subplots.return_value = (mock_fig, mock_axes)
            
            result = umap(data, labels, hues_colors=colors, add_super_imposed=True)
            
            assert isinstance(result, tuple)


def test_umap_independent_legend():
    """Test UMAP with independent legend creation."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(25, 4)]
    labels = [np.random.randint(0, 3, 25)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(25, 2)
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots, \
             patch('matplotlib.pyplot.figure') as mock_figure:
            
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.flat = [mock_ax]  # Add flat for legend processing
            mock_ax.__iter__ = lambda self: iter([mock_ax])  # Make single axis iterable for legend processing
            mock_legend = Mock()
            mock_legend.get_lines.return_value = []
            mock_legend.texts = []
            mock_ax.get_legend.return_value = mock_legend
            mock_subplots.return_value = (mock_fig, mock_ax)
            mock_figure.return_value = Mock()
            
            result = umap(data, labels, use_independent_legend=True)
            
            assert isinstance(result, tuple)
            assert len(result) == 3


def test_umap_visualization_parameters():
    """Test UMAP with custom visualization parameters."""
    from mngs.ai.clustering._umap import umap
    
    np.random.seed(42)
    data = [np.random.randn(20, 3)]
    labels = [np.random.randint(0, 2, 20)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap.transform.return_value = np.random.randn(20, 2)
        mock_umap_class.return_value = mock_umap
        
        with patch('mngs.plt.subplots') as mock_subplots:
            mock_fig = Mock()
            mock_ax = Mock()
            mock_ax.flat = [mock_ax]  # Add flat attribute to single axis for legend processing
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            result = umap(
                data, 
                labels, 
                title="Custom Title",
                alpha=0.5,
                s=10,
                axes_titles=["Custom Axis Title"]
            )
            
            assert isinstance(result, tuple)


def test_check_input_vars():
    """Test input validation function."""
    from mngs.ai.clustering._umap import _check_input_vars
    
    data_all = [np.random.randn(10, 5)]
    labels_all = [np.random.randint(0, 2, 10)]
    
    # Test with None values
    result = _check_input_vars(data_all, labels_all, None, None)
    assert len(result) == 4
    assert result[2] == [None]  # hues_all
    assert result[3] == [None]  # hues_colors_all
    
    # Test with provided values
    hues_all = [np.random.choice(['A', 'B'], 10)]
    hues_colors_all = [np.random.rand(10, 3)]
    
    result = _check_input_vars(data_all, labels_all, hues_all, hues_colors_all)
    assert len(result) == 4
    assert result[2] == hues_all
    assert result[3] == hues_colors_all


def test_check_input_vars_validation():
    """Test input validation with mismatched lengths."""
    from mngs.ai.clustering._umap import _check_input_vars
    
    data_all = [np.random.randn(10, 5)]
    labels_all = [np.random.randint(0, 2, 10)]
    hues_all = [None, None]  # Wrong length
    hues_colors_all = [None]
    
    with pytest.raises(AssertionError):
        _check_input_vars(data_all, labels_all, hues_all, hues_colors_all)


def test_check_input_vars_type_validation():
    """Test input validation with wrong types."""
    from mngs.ai.clustering._umap import _check_input_vars
    
    data_all = np.random.randn(10, 5)  # Not a list
    labels_all = [np.random.randint(0, 2, 10)]
    
    with pytest.raises(AssertionError):
        _check_input_vars(data_all, labels_all, None, None)


def test_run_umap_new_model():
    """Test _run_umap with new model creation."""
    from mngs.ai.clustering._umap import _run_umap
    
    data_all = [np.random.randn(30, 5)]
    labels_all = [np.random.randint(0, 3, 30)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap_class.return_value = mock_umap
        
        result = _run_umap(None, data_all, labels_all, False, "Test")
        
        assert result == mock_umap
        mock_umap_class.assert_called_once_with(random_state=42)
        mock_umap.fit.assert_called_once_with(data_all[0], y=None)


def test_run_umap_supervised():
    """Test _run_umap with supervised learning."""
    from mngs.ai.clustering._umap import _run_umap
    
    data_all = [np.random.randn(30, 5)]
    labels_all = [np.random.randint(0, 3, 30)]
    
    with patch('mngs.ai.clustering._umap.umap_orig.UMAP') as mock_umap_class:
        mock_umap = Mock()
        mock_umap.fit.return_value = mock_umap
        mock_umap_class.return_value = mock_umap
        
        result = _run_umap(None, data_all, labels_all, True, "Test")
        
        mock_umap.fit.assert_called_once_with(data_all[0], y=labels_all[0])


def test_run_umap_existing_model():
    """Test _run_umap with existing model."""
    from mngs.ai.clustering._umap import _run_umap
    
    data_all = [np.random.randn(30, 5)]
    labels_all = [np.random.randint(0, 3, 30)]
    existing_model = Mock()
    
    result = _run_umap(existing_model, data_all, labels_all, False, "Test")
    
    assert result == existing_model


def test_test_function_iris():
    """Test the _test function with iris dataset."""
    from mngs.ai.clustering._umap import _test
    
    with patch('mngs.ai.clustering._umap.umap') as mock_umap, \
         patch('sklearn.datasets.load_iris') as mock_load_iris, \
         patch('mngs.io.save') as mock_save:
        
        # Mock iris dataset
        mock_dataset = Mock()
        mock_dataset.data = np.random.randn(150, 4)
        mock_dataset.target = np.random.randint(0, 3, 150)
        mock_load_iris.return_value = mock_dataset
        
        # Mock umap return values
        mock_fig = Mock()
        mock_legend_figs = [Mock()]
        mock_model = Mock()
        mock_umap.return_value = (mock_fig, mock_legend_figs, mock_model)
        
        _test("iris")
        
        mock_umap.assert_called_once()
        mock_save.assert_called()  # Should save the figure


def test_test_function_mnist():
    """Test the _test function with MNIST dataset."""
    from mngs.ai.clustering._umap import _test
    
    with patch('mngs.ai.clustering._umap.umap') as mock_umap, \
         patch('sklearn.datasets.load_digits') as mock_load_digits, \
         patch('mngs.io.save') as mock_save:
        
        # Mock MNIST dataset
        mock_dataset = Mock()
        mock_dataset.data = np.random.randn(1797, 64)
        mock_dataset.target = np.random.randint(0, 10, 1797)
        mock_load_digits.return_value = mock_dataset
        
        # Mock umap return values
        mock_fig = Mock()
        mock_legend_figs = [Mock(), Mock()]
        mock_model = Mock()
        mock_umap.return_value = (mock_fig, mock_legend_figs, mock_model)
        
        _test("mnist")
        
        mock_umap.assert_called_once()
        # Should save main figure and legend figures
        assert mock_save.call_count >= 2


if __name__ == "__main__":
    import os
    import pytest
<<<<<<< HEAD
    pytest.main([os.path.abspath(__file__)])
=======

    pytest.main([os.path.abspath(__file__)])

# --------------------------------------------------------------------------------
# Start of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/clustering/_umap_dev.py
# --------------------------------------------------------------------------------
# #!./env/bin/python3
# # -*- coding: utf-8 -*-
# # Time-stamp: "2024-09-12 05:37:55 (ywatanabe)"
# # _umap_dev.py
# 
# 
# """
# This script does XYZ.
# """
# 
# 
# """
# Imports
# """
# import sys
# 
# import matplotlib.pyplot as plt
# import mngs
# import numpy as np
# import umap.umap_ as umap_orig
# from natsort import natsorted
# from sklearn.preprocessing import LabelEncoder
# 
# # sys.path = ["."] + sys.path
# # from scripts import utils, load
# 
# """
# Warnings
# """
# # warnings.simplefilter("ignore", UserWarning)
# 
# 
# """
# Config
# """
# # CONFIG = mngs.gen.load_configs()
# 
# 
# """
# Functions & Classes
# """
# 
# 
# def umap(
#     data,
#     labels,
#     hues=None,
#     hues_colors=None,
#     axes=None,
#     axes_titles=None,
#     supervised=False,
#     title="UMAP Clustering",
#     alpha=1.0,
#     s=3,
#     use_independent_legend=False,
#     add_super_imposed=False,
#     umap_model=None,
# ):
#     """
#     Perform UMAP clustering and visualization.
# 
#     Parameters
#     ----------
#     data_all : list
#         List of data arrays to cluster
#     labels_all : list
#         List of label arrays corresponding to data_all
#     hues_all : list, optional
#         List of hue arrays for coloring points
#     hues_colors_all : list, optional
#         List of color mappings for hues
#     axes : matplotlib.axes.Axes, optional
#         Existing axes to plot on
#     axes_titles : list, optional
#         Titles for each subplot
#     supervised : bool, optional
#         Whether to use supervised UMAP
#     title : str, optional
#         Main title for the plot
#     alpha : float, optional
#         Transparency of points
#     s : int, optional
#         Size of points
#     use_independent_legend : bool, optional
#         Whether to create separate legend figures
#     add_super_imposed : bool, optional
#         Whether to add a superimposed plot
#     umap_model : umap.UMAP, optional
#         Pre-fitted UMAP model
# 
#     Returns
#     -------
#     tuple
#         Figure, legend figures (if applicable), and UMAP model
#     """
# 
#     # Renaming
#     data_all = data
#     labels_all = labels
#     hues_all = hues
#     hues_colors_all = hues_colors
# 
#     data_all, labels_all, hues_all, hues_colors_all = _check_input_vars(
#         data_all, labels_all, hues_all, hues_colors_all
#     )
# 
#     # Label Encoding
#     le = LabelEncoder()
#     le.fit(natsorted(np.hstack(labels_all)))
#     labels_all = [le.transform(labels) for labels in labels_all]
# 
#     # Running UMAP Clustering
#     _umap = _run_umap(umap_model, data_all, labels_all, supervised, title)
# 
#     # Plotting
#     fig, legend_figs = _plot(
#         _umap,
#         le,
#         data_all,
#         labels_all,
#         hues_all,
#         hues_colors_all,
#         add_super_imposed,
#         axes,
#         title,
#         axes_titles,
#         use_independent_legend,
#         s,
#         alpha,
#     )
# 
#     return fig, legend_figs, _umap
# 
# 
# def _plot(
#     _umap,
#     le,
#     data_all,
#     labels_all,
#     hues_all,
#     hues_colors_all,
#     add_super_imposed,
#     axes,
#     title,
#     axes_titles,
#     use_independent_legend,
#     s,
#     alpha,
# ):
#     # Plotting
#     ncols = len(data_all) + 1 if add_super_imposed else len(data_all)
#     share = True if ncols > 1 else False
# 
#     if axes is None:
#         fig, axes = mngs.plt.subplots(ncols=ncols, sharex=share, sharey=share)
#     else:
#         assert len(axes) == ncols
#         fig = (
#             axes[0].get_figure()
#             # axes
#             if isinstance(
#                 axes, (np.ndarray, mngs.plt._subplots._AxesWrapper.AxesWrapper)
#             )
#             # axis
#             else axes.get_figure()
#         )
# 
#     fig.supxyt("UMAP 1", "UMAP 2", title)
# 
#     for ii, (data, labels, hues, hues_colors) in enumerate(
#         zip(data_all, labels_all, hues_all, hues_colors_all)
#     ):
#         embedding = _umap.transform(data)
# 
#         # ax
#         if ncols == 1:
#             ax = axes
#         else:
#             ax = axes[ii + 1] if add_super_imposed else axes[ii]
# 
#         _hues = le.inverse_transform(labels) if hues is None else hues
#         for hue in np.unique(_hues):
#             indi = hue == np.array(_hues)
# 
#             if hues_colors:
#                 colors = np.vstack(hues_colors)[indi]
#                 colors = [colors[ii] for ii in range(len(colors))]
#             else:
#                 colors = None
#             ax.scatter(
#                 x=embedding[:, 0][indi],
#                 y=embedding[:, 1][indi],
#                 label=hue,
#                 c=colors,
#                 s=s,
#                 alpha=alpha,
#             )
# 
#         ax.set_box_aspect(1)
# 
#         if axes_titles is not None:
#             ax.set_title(axes_titles[ii])
# 
#         # Merged axis
#         if add_super_imposed:
#             ax = axes[0]
#             _hues = le.inverse_transform(labels) if hues is None else hues
#             for hue in np.unique(_hues):
#                 indi = hue == np.array(_hues)
#                 ax.scatter(
#                     x=embedding[:, 0][indi],
#                     y=embedding[:, 1][indi],
#                     label=hue,
#                     c=np.vstack(hues_colors)[indi][0],
#                     s=s,
#                     alpha=alpha,
#                 )
# 
#             ax.set_title("Superimposed")
#             ax.set_box_aspect(1)
#             # ax.sns_scatterplot(
#             #     x=embedding[:, 0],
#             #     y=embedding[:, 1],
#             #     hue=le.inverse_transform(labels) if hues is None else hues,
#             #     palette=hues_colors,
#             #     legend="full" if ii == 0 else False,
#             #     s=s,
#             #     alpha=alpha,
#             # )
# 
#     if share:
#         mngs.plt.ax.sharex(axes)
#         mngs.plt.ax.sharey(axes)
# 
#     if not use_independent_legend:
#         for ax in axes.flat:
#             ax.legend(loc="upper left")
#         return fig, None
# 
#     elif use_independent_legend:
#         legend_figs = []
#         for i, ax in enumerate(axes):
#             legend = ax.get_legend()
#             if legend:
#                 legend_fig = plt.figure(figsize=(3, 2))
# 
#                 new_legend = legend_fig.gca().legend(
#                     handles=legend.get_lines(),
#                     labels=[t.get_text() for t in legend.texts],
#                     loc="center",
#                 )
# 
#                 # new_legend = legend_fig.gca().legend(
#                 #     handles=legend.legendHandles,
#                 #     labels=legend.texts,
#                 #     loc="center",
#                 # )
# 
#                 # legend_fig.canvas.draw()
#                 legend_figs.append(legend_fig)
#                 ax.get_legend().remove()
# 
#         for ax in axes:
#             ax.legend_ = None
# 
#         # elif use_independent_legend:
#         #     legend_figs = []
#         #     for i, ax in enumerate(axes):
#         #         legend = ax.get_legend()
#         #         if legend:
#         #             legend_fig = plt.figure(figsize=(3, 2))
#         #             new_legend = legend_fig.gca().legend(
#         #                 handles=legend.legendHandles,
#         #                 labels=legend.texts,
#         #                 loc="center",
#         #             )
#         #             legend_fig.canvas.draw()
#         #             legend_filename = f"legend_{i}.png"
#         #             legend_fig.savefig(legend_filename, bbox_inches="tight")
#         #             legend_figs.append(legend_fig)
#         #             plt.close(legend_fig)
# 
#         #     for ax in axes:
#         #         ax.legend_ = None
# 
#     return fig, legend_figs
# 
# 
# def _run_umap(umap_model, data_all, labels_all, supervised, title):
#     # UMAP Clustering
#     if not umap_model:
#         umap_model = umap_orig.UMAP(random_state=42)
#         supervised_label_or_none = labels_all[0] if supervised else None
#         title = (
#             f"(Supervised) {title}"
#             if supervised
#             else f"(Unsupervised) {title}"
#         )
#         _umap = umap_model.fit(data_all[0], y=supervised_label_or_none)
#     else:
#         _umap = umap_model
# 
#     return _umap
# 
# 
# def _check_input_vars(data_all, labels_all, hues_all, hues_colors_all):
#     # Ensures input formats
#     if hues_all is None:
#         hues_all = [None for _ in range(len(data_all))]
# 
#     if hues_colors_all is None:
#         hues_colors_all = [None for _ in range(len(data_all))]
# 
#     assert (
#         len(data_all)
#         == len(labels_all)
#         == len(hues_all)
#         == len(hues_colors_all)
#     )
# 
#     assert (
#         isinstance(data_all, list)
#         and isinstance(labels_all, list)
#         and isinstance(hues_all, list)
#         and isinstance(hues_colors_all, list)
#     )
#     return data_all, labels_all, hues_all, hues_colors_all
# 
# 
# def _test(dataset_str="iris"):
#     import matplotlib.pyplot as plt
#     import numpy as np
#     from sklearn.datasets import load_digits, load_iris
#     from sklearn.model_selection import train_test_split
# 
#     # Load iris dataset
#     load_dataset = {"iris": load_iris, "mnist": load_digits}[dataset_str]
# 
#     dataset = load_dataset()
#     X = dataset.data
#     y = dataset.target
# 
#     # Split data into two parts
#     X1, X2, y1, y2 = train_test_split(X, y, test_size=0.5, random_state=42)
# 
#     # Call umap function
#     fig, legend_figs, umap_model = umap(
#         data=[X1, X2],
#         labels=[y1, y2],
#         # axes=axes,
#         axes_titles=[f"{dataset_str} Set 1", f"{dataset_str} Set 2"],
#         supervised=True,
#         title=dataset_str,
#         use_independent_legend=True,
#         s=10,
#     )
# 
#     # plt.tight_layout()
#     mngs.io.save(fig, f"/tmp/mngs/umap/{dataset_str}.jpg")
# 
#     # Save legend figures if any
#     if legend_figs:
#         for i, leg_fig in enumerate(legend_figs):
#             mngs.io.save(
#                 leg_fig, f"/tmp/mngs/umap/{dataset_str}_legend_{i}.jpg"
#             )
# 
# 
# main = umap
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
#     _test(dataset_str="mnist")
#     # main()
#     mngs.gen.close(CONFIG, verbose=False, notify=False)
# 
# # EOF

# --------------------------------------------------------------------------------
# End of Source Code from: /data/gpfs/projects/punim2354/ywatanabe/mngs_repo/src/mngs/ai/clustering/_umap_dev.py
# --------------------------------------------------------------------------------
>>>>>>> origin/main
