#!/usr/bin/env python3
# Basic test script for axis wrapper compatibility

import sys
import os
import inspect
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# Define utility functions
def is_valid_axis(axis):
    """Check if the provided object is a valid axis (matplotlib Axes or mngs AxisWrapper)."""
    # Check if it's a matplotlib Axes directly
    if isinstance(axis, matplotlib.axes._axes.Axes):
        return True
    
    # Check if it's an AxisWrapper from mngs
    for cls in inspect.getmro(type(axis)):
        if cls.__name__ == 'AxisWrapper':
            return True
            
    # Check if it has common axis methods (fallback check)
    axis_methods = ['plot', 'scatter', 'set_xlabel', 'set_ylabel', 'get_figure']
    has_methods = all(hasattr(axis, method) for method in axis_methods)
    
    return has_methods

def assert_valid_axis(axis, error_message=None):
    """Assert that the provided object is a valid axis."""
    if error_message is None:
        error_message = "First argument must be a matplotlib axis or mngs axis wrapper"
        
    assert is_valid_axis(axis), error_message

def hide_spines(
    axis,
    top=True,
    bottom=True,
    left=True,
    right=True,
    ticks=True,
    labels=True,
):
    """
    Hides the specified spines of a matplotlib Axes object or mngs axis wrapper.
    """
    assert_valid_axis(axis, "First argument must be a matplotlib axis or mngs axis wrapper")

    tgts = []
    if top:
        tgts.append("top")
    if bottom:
        tgts.append("bottom")
    if left:
        tgts.append("left")
    if right:
        tgts.append("right")

    for tgt in tgts:
        # Spines
        axis.spines[tgt].set_visible(False)

        # Ticks
        if ticks:
            if tgt == "bottom":
                axis.xaxis.set_ticks_position("none")
            elif tgt == "left":
                axis.yaxis.set_ticks_position("none")

        # Labels
        if labels:
            if tgt == "bottom":
                axis.set_xticklabels([])
            elif tgt == "left":
                axis.set_yticklabels([])

    return axis

# Test with matplotlib axis
print("Testing with matplotlib axis...")
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [1, 2, 3])
hide_spines(ax, top=True, right=True, bottom=False, left=False)
plt.savefig('test_mpl_axis.png')
plt.close(fig)
print("✓ Test completed with matplotlib axis")

try:
    # Try with mngs axis wrapper if available
    print("\nTesting with mngs axis wrapper...")
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Create a minimal wrapper
    class FakeAxisWrapper:
        def __init__(self):
            self.fig, self.ax = plt.subplots()
            self.ax.plot([1, 2, 3], [1, 2, 3])
            
        def __getattr__(self, attr):
            return getattr(self.ax, attr)
            
        @property
        def spines(self):
            return self.ax.spines
            
        @property
        def xaxis(self):
            return self.ax.xaxis
            
        @property
        def yaxis(self):
            return self.ax.yaxis

    # Create a fake wrapper for testing
    wrapper = FakeAxisWrapper()
    print(f"Is valid axis: {is_valid_axis(wrapper)}")
    hide_spines(wrapper, top=True, right=True, bottom=False, left=False)
    wrapper.fig.savefig('test_wrapper_axis.png')
    plt.close(wrapper.fig)
    print("✓ Test completed with axis wrapper")
    
except Exception as e:
    print(f"⨯ Error with mngs axis wrapper: {e}")

print("\nTests completed!")