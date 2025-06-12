#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Time-stamp: "2025-06-05 07:45:00 (ywatanabe)"
# File: ./tests/mngs/plt/ax/_style/test__show_spines.py

"""
Functionality:
    Comprehensive tests for _show_spines module
Input:
    Various matplotlib axes configurations and spine parameters
Output:
    Test results validating spine visibility control functionality
Prerequisites:
    pytest, matplotlib, mngs
"""

import pytest
import matplotlib.pyplot as plt
import matplotlib.axes
import numpy as np
from unittest.mock import Mock, patch

# Import the functions to test
import mngs


class TestShowSpines:
    """Test the main show_spines function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
        # Start with all spines hidden (common mngs default)
        for spine in self.ax.spines.values():
            spine.set_visible(False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_show_all_spines_default(self):
        """Test showing all spines with default parameters."""
        result = mngs.plt.ax.show_spines(self.ax)
        
        assert result is self.ax
        assert self.ax.spines['top'].get_visible()
        assert self.ax.spines['bottom'].get_visible()
        assert self.ax.spines['left'].get_visible()
        assert self.ax.spines['right'].get_visible()
    
    def test_show_selective_spines(self):
        """Test showing only specific spines."""
        mngs.plt.ax.show_spines(self.ax, top=False, right=False, bottom=True, left=True)
        
        assert not self.ax.spines['top'].get_visible()
        assert self.ax.spines['bottom'].get_visible()
        assert self.ax.spines['left'].get_visible()
        assert not self.ax.spines['right'].get_visible()
    
    def test_spine_width_setting(self):
        """Test setting custom spine width."""
        width = 2.5
        mngs.plt.ax.show_spines(self.ax, spine_width=width)
        
        for spine in self.ax.spines.values():
            assert spine.get_linewidth() == width
    
    def test_spine_color_setting(self):
        """Test setting custom spine color."""
        color = 'red'
        mngs.plt.ax.show_spines(self.ax, spine_color=color)
        
        # matplotlib converts colors to RGBA tuples
        expected_rgba = (1.0, 0.0, 0.0, 1.0)  # red in RGBA
        for spine in self.ax.spines.values():
            assert spine.get_edgecolor() == expected_rgba
    
    def test_combined_styling(self):
        """Test combining width and color settings."""
        width, color = 1.8, 'blue'
        mngs.plt.ax.show_spines(self.ax, spine_width=width, spine_color=color)
        
        expected_rgba = (0.0, 0.0, 1.0, 1.0)  # blue in RGBA
        for spine in self.ax.spines.values():
            assert spine.get_linewidth() == width
            assert spine.get_edgecolor() == expected_rgba
    
    def test_tick_positioning_bottom_only(self):
        """Test tick positioning when only bottom spine is shown."""
        mngs.plt.ax.show_spines(self.ax, top=False, bottom=True, left=False, right=False)
        
        # Should position ticks on bottom only
        assert self.ax.xaxis.get_ticks_position() == 'bottom'
    
    def test_tick_positioning_top_only(self):
        """Test tick positioning when only top spine is shown."""
        mngs.plt.ax.show_spines(self.ax, top=True, bottom=False, left=False, right=False)
        
        assert self.ax.xaxis.get_ticks_position() == 'top'
    
    def test_tick_positioning_both_horizontal(self):
        """Test tick positioning when both horizontal spines are shown."""
        mngs.plt.ax.show_spines(self.ax, top=True, bottom=True, left=False, right=False)
        
        # When both spines are shown, matplotlib might use 'default' instead of 'both'
        tick_pos = self.ax.xaxis.get_ticks_position()
        assert tick_pos in ['both', 'default']
    
    def test_tick_positioning_left_only(self):
        """Test tick positioning when only left spine is shown."""
        mngs.plt.ax.show_spines(self.ax, top=False, bottom=False, left=True, right=False)
        
        assert self.ax.yaxis.get_ticks_position() == 'left'
    
    def test_tick_positioning_right_only(self):
        """Test tick positioning when only right spine is shown."""
        mngs.plt.ax.show_spines(self.ax, top=False, bottom=False, left=False, right=True)
        
        assert self.ax.yaxis.get_ticks_position() == 'right'
    
    def test_tick_positioning_both_vertical(self):
        """Test tick positioning when both vertical spines are shown."""
        mngs.plt.ax.show_spines(self.ax, top=False, bottom=False, left=True, right=True)
        
        # When both spines are shown, matplotlib might use 'default' instead of 'both'
        tick_pos = self.ax.yaxis.get_ticks_position()
        assert tick_pos in ['both', 'default']
    
    def test_ticks_disabled(self):
        """Test behavior when ticks are disabled."""
        original_x_pos = self.ax.xaxis.get_ticks_position()
        original_y_pos = self.ax.yaxis.get_ticks_position()
        
        mngs.plt.ax.show_spines(self.ax, ticks=False)
        
        # Tick positions should not be modified when ticks=False
        assert self.ax.xaxis.get_ticks_position() == original_x_pos
        assert self.ax.yaxis.get_ticks_position() == original_y_pos
    
    def test_restore_defaults_disabled(self):
        """Test behavior when restore_defaults is disabled."""
        mngs.plt.ax.show_spines(self.ax, restore_defaults=False)
        
        # Should still show spines but not modify tick settings
        assert all(spine.get_visible() for spine in self.ax.spines.values())
    
    def test_labels_functionality(self):
        """Test label restoration functionality."""
        # Set some data to generate ticks
        self.ax.plot([1, 2, 3], [1, 4, 2])
        
        mngs.plt.ax.show_spines(self.ax, labels=True)
        
        # Should have tick labels
        xticks = self.ax.get_xticks()
        yticks = self.ax.get_yticks()
        assert len(xticks) > 0
        assert len(yticks) > 0


class TestMngsAxisWrapperCompatibility:
    """Test compatibility with mngs AxisWrapper objects."""
    
    def setup_method(self):
        """Set up test fixtures with mock AxisWrapper."""
        self.fig, self.ax = plt.subplots()
        
        # Create a mock AxisWrapper that has _axis_mpl attribute
        self.mock_wrapper = Mock()
        self.mock_wrapper._axis_mpl = self.ax
        self.mock_wrapper.__class__.__name__ = 'AxisWrapper'
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_axis_wrapper_handling(self):
        """Test that function works with mngs AxisWrapper objects."""
        result = mngs.plt.ax.show_spines(self.mock_wrapper)
        
        # Should return the underlying matplotlib axis
        assert result is self.ax
        # All spines should be visible
        assert all(spine.get_visible() for spine in self.ax.spines.values())
    
    def test_invalid_axis_type(self):
        """Test error handling for invalid axis types."""
        with pytest.raises(AssertionError, match="First argument must be a matplotlib axis"):
            mngs.plt.ax.show_spines("not_an_axis")
    
    def test_none_axis(self):
        """Test error handling for None axis."""
        with pytest.raises(AssertionError, match="First argument must be a matplotlib axis"):
            mngs.plt.ax.show_spines(None)


class TestShowAllSpines:
    """Test the show_all_spines convenience function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_show_all_spines_basic(self):
        """Test basic show_all_spines functionality."""
        result = mngs.plt.ax.show_all_spines(self.ax)
        
        assert result is self.ax
        assert all(spine.get_visible() for spine in self.ax.spines.values())
    
    def test_show_all_spines_with_styling(self):
        """Test show_all_spines with styling parameters."""
        width, color = 2.0, 'green'
        mngs.plt.ax.show_all_spines(self.ax, spine_width=width, spine_color=color)
        
        expected_rgba = (0.0, 0.5019607843137255, 0.0, 1.0)  # green in RGBA
        for spine in self.ax.spines.values():
            assert spine.get_visible()
            assert spine.get_linewidth() == width
            assert spine.get_edgecolor() == expected_rgba
    
    def test_show_all_spines_no_ticks(self):
        """Test show_all_spines without ticks."""
        mngs.plt.ax.show_all_spines(self.ax, ticks=False)
        
        assert all(spine.get_visible() for spine in self.ax.spines.values())
    
    def test_show_all_spines_no_labels(self):
        """Test show_all_spines without labels."""
        mngs.plt.ax.show_all_spines(self.ax, labels=False)
        
        assert all(spine.get_visible() for spine in self.ax.spines.values())


class TestShowClassicSpines:
    """Test the show_classic_spines function (scientific plot style)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_classic_spines_pattern(self):
        """Test that classic spines shows only bottom and left."""
        mngs.plt.ax.show_classic_spines(self.ax)
        
        assert not self.ax.spines['top'].get_visible()
        assert self.ax.spines['bottom'].get_visible()
        assert self.ax.spines['left'].get_visible()
        assert not self.ax.spines['right'].get_visible()
    
    def test_classic_spines_with_styling(self):
        """Test classic spines with custom styling."""
        width, color = 1.5, 'black'
        mngs.plt.ax.show_classic_spines(self.ax, spine_width=width, spine_color=color)
        
        expected_rgba = (0.0, 0.0, 0.0, 1.0)  # black in RGBA
        # Only bottom and left should be styled and visible
        assert self.ax.spines['bottom'].get_visible()
        assert self.ax.spines['left'].get_visible()
        assert self.ax.spines['bottom'].get_linewidth() == width
        assert self.ax.spines['left'].get_linewidth() == width
        assert self.ax.spines['bottom'].get_edgecolor() == expected_rgba
        assert self.ax.spines['left'].get_edgecolor() == expected_rgba
    
    def test_scientific_spines_alias(self):
        """Test that scientific_spines is an alias for show_classic_spines."""
        mngs.plt.ax.scientific_spines(self.ax)
        
        assert not self.ax.spines['top'].get_visible()
        assert self.ax.spines['bottom'].get_visible()
        assert self.ax.spines['left'].get_visible()
        assert not self.ax.spines['right'].get_visible()


class TestShowBoxSpines:
    """Test the show_box_spines function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_box_spines_all_visible(self):
        """Test that box spines shows all four spines."""
        mngs.plt.ax.show_box_spines(self.ax)
        
        assert all(spine.get_visible() for spine in self.ax.spines.values())
    
    def test_box_spines_with_styling(self):
        """Test box spines with styling."""
        width, color = 1.0, 'purple'
        mngs.plt.ax.show_box_spines(self.ax, spine_width=width, spine_color=color)
        
        expected_rgba = (0.5019607843137255, 0.0, 0.5019607843137255, 1.0)  # purple in RGBA
        for spine in self.ax.spines.values():
            assert spine.get_visible()
            assert spine.get_linewidth() == width
            assert spine.get_edgecolor() == expected_rgba


class TestToggleSpines:
    """Test the toggle_spines function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
        # Set initial known state
        self.ax.spines['top'].set_visible(True)
        self.ax.spines['bottom'].set_visible(False)
        self.ax.spines['left'].set_visible(True)
        self.ax.spines['right'].set_visible(False)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_toggle_all_spines(self):
        """Test toggling all spines (None parameters)."""
        initial_states = {name: spine.get_visible() for name, spine in self.ax.spines.items()}
        
        mngs.plt.ax.toggle_spines(self.ax)
        
        for name, spine in self.ax.spines.items():
            assert spine.get_visible() == (not initial_states[name])
    
    def test_toggle_specific_spines(self):
        """Test setting specific spine states."""
        mngs.plt.ax.toggle_spines(self.ax, top=False, bottom=True)
        
        assert not self.ax.spines['top'].get_visible()
        assert self.ax.spines['bottom'].get_visible()
        # Left and right should be toggled from initial state
        assert not self.ax.spines['left'].get_visible()  # was True, now False
        assert self.ax.spines['right'].get_visible()    # was False, now True
    
    def test_toggle_mixed_parameters(self):
        """Test mixing explicit and toggle parameters."""
        mngs.plt.ax.toggle_spines(self.ax, top=True, right=False)
        
        assert self.ax.spines['top'].get_visible()      # explicitly set to True
        assert not self.ax.spines['right'].get_visible()  # explicitly set to False
        # Bottom and left should be toggled
        assert self.ax.spines['bottom'].get_visible()   # was False, now True
        assert not self.ax.spines['left'].get_visible()   # was True, now False


class TestCleanSpines:
    """Test the clean_spines function (no spines shown)."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
        # Start with all spines visible
        for spine in self.ax.spines.values():
            spine.set_visible(True)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_clean_spines_hides_all(self):
        """Test that clean_spines hides all spines."""
        mngs.plt.ax.clean_spines(self.ax)
        
        assert all(not spine.get_visible() for spine in self.ax.spines.values())
    
    def test_clean_spines_with_ticks_labels(self):
        """Test clean_spines with tick and label options."""
        mngs.plt.ax.clean_spines(self.ax, ticks=True, labels=True)
        
        # All spines should be hidden regardless of tick/label settings
        assert all(not spine.get_visible() for spine in self.ax.spines.values())


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_empty_axis_data(self):
        """Test behavior with axis that has no data."""
        # Should work without errors even with empty axis
        result = mngs.plt.ax.show_spines(self.ax)
        assert result is self.ax
    
    def test_axis_with_data(self):
        """Test behavior with axis containing data."""
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        self.ax.plot(x, y)
        
        result = mngs.plt.ax.show_spines(self.ax)
        assert result is self.ax
        assert all(spine.get_visible() for spine in self.ax.spines.values())
    
    def test_negative_spine_width(self):
        """Test behavior with negative spine width."""
        # Matplotlib should handle this gracefully
        mngs.plt.ax.show_spines(self.ax, spine_width=-1.0)
        
        for spine in self.ax.spines.values():
            assert spine.get_linewidth() == -1.0  # matplotlib allows negative widths
    
    def test_zero_spine_width(self):
        """Test behavior with zero spine width."""
        mngs.plt.ax.show_spines(self.ax, spine_width=0.0)
        
        for spine in self.ax.spines.values():
            assert spine.get_linewidth() == 0.0
    
    def test_invalid_color_format(self):
        """Test behavior with invalid color format."""
        # This should raise a matplotlib error
        with pytest.raises((ValueError, TypeError)):
            mngs.plt.ax.show_spines(self.ax, spine_color='invalid_color_name')
    
    def test_none_width_and_color(self):
        """Test that None values don't change existing properties."""
        # Set initial properties
        initial_width = self.ax.spines['bottom'].get_linewidth()
        initial_color = self.ax.spines['bottom'].get_edgecolor()
        
        mngs.plt.ax.show_spines(self.ax, spine_width=None, spine_color=None)
        
        # Properties should remain unchanged
        assert self.ax.spines['bottom'].get_linewidth() == initial_width
        assert self.ax.spines['bottom'].get_edgecolor() == initial_color


class TestIntegration:
    """Integration tests with realistic usage patterns."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.fig, self.ax = plt.subplots()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        plt.close(self.fig)
    
    def test_scientific_plot_workflow(self):
        """Test typical scientific plotting workflow."""
        # Generate sample data
        x = np.linspace(0, 2*np.pi, 100)
        y = np.sin(x)
        self.ax.plot(x, y)
        
        # Apply scientific styling
        mngs.plt.ax.show_classic_spines(
            self.ax, 
            spine_width=1.2, 
            spine_color='black'
        )
        
        # Verify the result
        assert not self.ax.spines['top'].get_visible()
        assert self.ax.spines['bottom'].get_visible()
        assert self.ax.spines['left'].get_visible()
        assert not self.ax.spines['right'].get_visible()
        
        # Check styling
        expected_rgba = (0.0, 0.0, 0.0, 1.0)  # black in RGBA
        assert self.ax.spines['bottom'].get_linewidth() == 1.2
        assert self.ax.spines['left'].get_linewidth() == 1.2
        assert self.ax.spines['bottom'].get_edgecolor() == expected_rgba
        assert self.ax.spines['left'].get_edgecolor() == expected_rgba
    
    def test_overlay_plot_workflow(self):
        """Test workflow for overlay plots with clean spines."""
        # Create base plot
        x = np.linspace(0, 10, 50)
        y = np.exp(-x/3)
        self.ax.plot(x, y)
        
        # Apply clean styling for overlay
        mngs.plt.ax.clean_spines(self.ax, ticks=False, labels=False)
        
        # Verify clean appearance
        assert all(not spine.get_visible() for spine in self.ax.spines.values())
    
    def test_publication_ready_workflow(self):
        """Test workflow for publication-ready figures."""
        # Create sample data
        categories = ['A', 'B', 'C', 'D']
        values = [23, 45, 56, 78]
        self.ax.bar(categories, values)
        
        # Apply publication styling
        mngs.plt.ax.show_box_spines(
            self.ax,
            spine_width=0.8,
            spine_color='#333333',
            ticks=True,
            labels=True
        )
        
        # Verify box appearance
        expected_rgba = (0.2, 0.2, 0.2, 1.0)  # #333333 in RGBA
        assert all(spine.get_visible() for spine in self.ax.spines.values())
        for spine in self.ax.spines.values():
            assert spine.get_linewidth() == 0.8
            assert spine.get_edgecolor() == expected_rgba
    
    def test_toggle_workflow(self):
        """Test interactive toggle workflow."""
        # Start with default state
        initial_states = {name: spine.get_visible() for name, spine in self.ax.spines.items()}
        
        # Toggle spines multiple times
        mngs.plt.ax.toggle_spines(self.ax)
        first_toggle = {name: spine.get_visible() for name, spine in self.ax.spines.items()}
        
        mngs.plt.ax.toggle_spines(self.ax)
        second_toggle = {name: spine.get_visible() for name, spine in self.ax.spines.items()}
        
        # Should return to initial state after double toggle
        assert initial_states == second_toggle
        
        # First toggle should be opposite of initial
        for name in initial_states:
            assert first_toggle[name] == (not initial_states[name])


if __name__ == "__main__":
    pytest.main([__file__])